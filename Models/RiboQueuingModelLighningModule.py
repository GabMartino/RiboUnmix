from __future__ import annotations

from typing import Any

import math

import lightning as pl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from Models.utils.PCGrad_utils import (
    per_dataset_losses,
    pcgrad_combine,
    assign_flat_grads,
)
from Models.utils.dirichlet_multinomial_profile_loss import DirichletMultinomialProfileLoss
from Models.utils.multinomial_profile_loss import MultinomialProfileLoss
from Models.utils.ribo_lightning_helpers import flatten_current_grads


class RiboQueuingModelLightningModule(pl.LightningModule):
    """
    Lightning wrapper for the conditional ribo-seq profile model.

    The forward model is interpreted as a profile-allocation model:

        q_i  = L_queue_i * b_i + additive_noise_i
        pi_i = q_i / sum_j q_j
        mu_i = total_mass_observed * pi_i

    Therefore the primary loss should be a profile loss, not a Tweedie/NB raw-count
    likelihood. This class supports:

        - multinomial profile loss
        - Dirichlet-multinomial profile loss

    Legacy naming note:
        The underlying torch model may still return `(mu, p, phi, extras)` for
        compatibility. In this Lightning module:

            p   -> profile_aux, currently unused by the profile loss
            phi -> kappa_input or dispersion_input used to derive DM kappa

        The old Tweedie terminology is intentionally removed from logs/predictions.
    """

    def __init__(
        self,
        torch_model: nn.Module,
        *,
        config: Any,
        dataset_encoding: dict | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["torch_model", "config", "dataset_encoding"])

        self.model = torch_model
        self.config = config

        self.loss_type = str(self._cfg("loss.type", "dirichlet_multinomial_profile")).lower()
        self.loss_fn = self._build_profile_loss()

        enc = dataset_encoding or {}
        self.dataset_id_to_name = {int(v): str(k) for k, v in enc.items()}

        self.use_pcgrad = bool(self._cfg("optim.use_pcgrad", False))
        self.automatic_optimization = not self.use_pcgrad

        self.log_sync_dist = bool(self._cfg("trainer.sync_dist_logs", False))

        if self.use_pcgrad:
            print("Training is using PCGrad.")

        self._val_plot_logged_this_epoch = False

    # ============================================================
    # Config / initialization helpers
    # ============================================================

    def _cfg(self, path: str, default: Any = None) -> Any:
        cur = self.config

        for key in path.split("."):
            if cur is None:
                return default

            if isinstance(cur, dict):
                if key not in cur:
                    return default
                cur = cur[key]
            else:
                if not hasattr(cur, key):
                    return default
                cur = getattr(cur, key)

        return cur

    def _dataset_name(self, ds_id: int) -> str:
        return self.dataset_id_to_name.get(int(ds_id), f"dataset_{int(ds_id)}")

    def _build_profile_loss(self) -> nn.Module:
        eps = float(self._cfg("loss.eps", 1e-8))
        normalize_by_total = bool(self._cfg("loss.normalize_by_total", True))

        if self.loss_type in {"multinomial", "multinomial_profile", "profile_ce"}:
            return MultinomialProfileLoss(
                eps=eps,
                normalize_by_total=normalize_by_total,
            )

        if self.loss_type in {
            "dirichlet_multinomial",
            "dirichlet_multinomial_profile",
            "dm",
            "dm_profile",
        }:
            fixed_kappa = self._cfg("loss.kappa", None)
            if fixed_kappa is not None:
                fixed_kappa = float(fixed_kappa)

            return DirichletMultinomialProfileLoss(
                eps=eps,
                kappa=fixed_kappa,
                kappa_min=float(self._cfg("loss.kappa_min", 1e-2)),
                kappa_max=float(self._cfg("loss.kappa_max", 1e5)),
                normalize_by_total=normalize_by_total,
                include_multinomial_constant=bool(
                    self._cfg("loss.include_multinomial_constant", False)
                ),
                pool_position_kappa=str(self._cfg("loss.pool_position_kappa", "mean")),
            )

        raise ValueError(
            f"Unsupported loss.type={self.loss_type!r}. Use 'multinomial_profile' "
            "or 'dirichlet_multinomial_profile'."
        )

    # ============================================================
    # Loss helpers
    # ============================================================

    def _compute_kappa_from_model_output(
        self,
        *,
        kappa_input: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor | float | None:
        """
        Converts the model's uncertainty output into Dirichlet-multinomial kappa.

        Kappa interpretation:
            high kappa -> low overdispersion, close to multinomial
            low kappa  -> high overdispersion

        Supported modes:
            loss.kappa_source = "fixed"
                Use the fixed kappa configured inside DirichletMultinomialProfileLoss.

            loss.kappa_source = "model_direct"
                Treat model output as kappa directly.

            loss.kappa_source = "model_inverse"  [default]
                Treat model output as dispersion-like phi and set:
                    kappa = kappa_scale / phi
        """
        if self.loss_type in {"multinomial", "multinomial_profile", "profile_ce"}:
            return None

        source = str(self._cfg("loss.kappa_source", "model_inverse")).lower()

        if source == "fixed":
            return None

        if kappa_input is None:
            return None

        x = kappa_input.float()

        if source in {"model_direct", "direct", "kappa"}:
            return x

        if source in {"model_inverse", "inverse", "phi_inverse", "dispersion_inverse"}:
            scale = float(self._cfg("loss.kappa_scale", 100.0))
            min_disp = float(self._cfg("loss.kappa_input_min", 1e-4))
            return scale / x.clamp_min(min_disp)

        raise ValueError(
            f"Unsupported loss.kappa_source={source!r}. Use fixed, model_direct, "
            "or model_inverse."
        )

    def _compute_raw_nll_per_sample(
        self,
        *,
        mu: torch.Tensor,
        kappa_input: torch.Tensor | None,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=mu.device.type, enabled=False):
            if self.loss_type in {"multinomial", "multinomial_profile", "profile_ce"}:
                return self.loss_fn(
                    mu=mu.float(),
                    y_true=target.float(),
                    mask=mask.bool(),
                    return_per_sample=True,
                )

            kappa = self._compute_kappa_from_model_output(
                kappa_input=kappa_input,
                mask=mask,
            )

            return self.loss_fn(
                mu=mu.float(),
                kappa=kappa,
                y_true=target.float(),
                mask=mask.bool(),
                return_per_sample=True,
            )

    def _soft_cap_loss_per_sample(
        self,
        loss_per_sample: torch.Tensor,
    ) -> torch.Tensor:
        cap = self._cfg("loss.nll_soft_cap", None)

        if cap is None:
            cap = self._cfg("loss.nll_soft_cap_value", None)

        if cap is None:
            return loss_per_sample

        cap = float(cap)

        if cap <= 0.0:
            return loss_per_sample

        return cap * torch.log1p(loss_per_sample / cap)

    def _dataset_balanced_scalar_loss(
        self,
        *,
        loss_per_sample: torch.Tensor,
        dataset_ids: torch.Tensor,
    ) -> torch.Tensor:
        dataset_losses = per_dataset_losses(
            loss_per_sample=loss_per_sample,
            dataset_ids=dataset_ids,
        )

        if len(dataset_losses) > 0:
            return torch.stack(dataset_losses).mean()

        return loss_per_sample.mean()

    def bias_regularization(
        self,
        *,
        extras: dict,
        mask: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Regularizes observation-bias branches.

        Preferred keys:
            control / b_control:
                pre-activation control signal for b. This is safer than log(b)
                when b can become exactly zero.

            lambda_bg:
                additive-background amplitude.
        """
        mask_b = mask.bool()
        regs = []

        control = self._safe_component(
            extras,
            "control",
            "b_control",
            "log_b_control",
        )

        if control is not None:
            regs.append((control.float()[mask_b] ** 2).mean())
        else:
            b = self._get_b_multiplier(extras=extras, mask=mask_b)
            if b is not None:
                log_b = torch.log(b.float().clamp_min(eps))
                regs.append((log_b[mask_b] ** 2).mean())

        lambda_bg = extras.get("lambda_bg")
        if lambda_bg is not None:
            # L1-like penalty is more directly interpretable as "use less additive mass".
            regs.append(lambda_bg.float().abs().mean())

        if len(regs) == 0:
            return torch.zeros((), device=mask.device)

        return torch.stack(regs).sum()

    def background_fraction_regularization(
        self,
        *,
        extras: dict,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        bio_q = extras.get("bio_q")
        additive_noise = extras.get("additive_noise")

        if bio_q is None or additive_noise is None:
            return torch.zeros((), device=mask.device)

        mask_f = mask.bool().float()
        eps = float(self._cfg("loss.eps", 1e-8))

        bio_mass = (bio_q.float().clamp_min(0.0) * mask_f).sum(dim=1)
        bg_mass = (additive_noise.float().clamp_min(0.0) * mask_f).sum(dim=1)
        bg_frac = bg_mass / (bio_mass + bg_mass).clamp_min(eps)

        return bg_frac.pow(2).mean()

    def _build_loss_terms(
        self,
        *,
        mu: torch.Tensor,
        kappa_input: torch.Tensor | None,
        target: torch.Tensor,
        mask: torch.Tensor,
        extras: dict,
    ) -> dict[str, torch.Tensor]:
        raw_nll_per_sample = self._compute_raw_nll_per_sample(
            mu=mu,
            kappa_input=kappa_input,
            target=target,
            mask=mask,
        )

        nll_per_sample = self._soft_cap_loss_per_sample(raw_nll_per_sample)

        eps = float(self._cfg("loss.eps", 1e-8))
        extra_loss = torch.zeros((), device=mu.device)

        bias_weight = float(self._cfg("loss.lambda_bias_regularization", 0.0))
        # Backward-compatible name.
        bias_weight = float(self._cfg("loss.lambda_bias_log_l2", bias_weight))

        if bias_weight > 0.0:
            extra_loss = extra_loss + bias_weight * self.bias_regularization(
                extras=extras,
                mask=mask,
                eps=eps,
            )

        bg_frac_weight = float(self._cfg("loss.lambda_bg_frac_l2", 0.0))
        if bg_frac_weight > 0.0:
            extra_loss = extra_loss + bg_frac_weight * self.background_fraction_regularization(
                extras=extras,
                mask=mask,
            )

        return {
            "raw_profile_nll_per_sample": raw_nll_per_sample,
            "profile_nll_per_sample": nll_per_sample,
            "loss_per_sample": nll_per_sample,
            "extra_loss": extra_loss,
        }

    # ============================================================
    # Generic logging helpers
    # ============================================================

    def _finite_mean(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None

        x = x.detach().float().reshape(-1)
        finite = torch.isfinite(x)

        if not finite.any():
            return None

        return x[finite].mean()

    def _log_scalar(
        self,
        name: str,
        value: torch.Tensor | float | None,
        *,
        batch_size: int,
        prog_bar: bool = False,
        on_step: bool = False,
        on_epoch: bool = True,
    ) -> None:
        if value is None:
            return

        if not torch.is_tensor(value):
            value = torch.tensor(float(value), device=self.device)

        value = value.detach().float()

        if value.numel() != 1:
            value = value.reshape(-1)
            finite = torch.isfinite(value)

            if not finite.any():
                return

            value = value[finite].mean()

        if not torch.isfinite(value):
            return

        self.log(
            name,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            logger=True,
            batch_size=batch_size,
            sync_dist=self.log_sync_dist,
        )

    def _masked_pcc_per_sample(
        self,
        *,
        pred: torch.Tensor | None,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if pred is None:
            return None

        if pred.shape != target.shape:
            return None

        eps = float(self._cfg("loss.eps", 1e-8))

        mask_b = mask.bool()
        mask_f = mask_b.float()

        x = pred.detach().float()
        y = target.detach().float()

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        valid_count = mask_f.sum(dim=1).clamp_min(1.0)

        x_mean = (x * mask_f).sum(dim=1) / valid_count
        y_mean = (y * mask_f).sum(dim=1) / valid_count

        x_centered = (x - x_mean.unsqueeze(1)) * mask_f
        y_centered = (y - y_mean.unsqueeze(1)) * mask_f

        cov = (x_centered * y_centered).sum(dim=1)

        x_var = (x_centered ** 2).sum(dim=1)
        y_var = (y_centered ** 2).sum(dim=1)

        denom = torch.sqrt(x_var * y_var).clamp_min(eps)
        pcc = cov / denom

        valid = (mask_f.sum(dim=1) >= 2) & torch.isfinite(pcc)
        return torch.where(valid, pcc, torch.nan)

    def _log_vector_global_and_by_dataset(
        self,
        *,
        stage: str,
        name: str,
        values: torch.Tensor,
        dataset_ids: torch.Tensor,
        batch_size: int,
        log_global: bool = True,
        log_by_dataset: bool = True,
        prog_bar: bool = False,
    ) -> None:
        values = values.detach().float()

        if log_global:
            mean_value = self._finite_mean(values)

            if mean_value is not None:
                self._log_scalar(
                    f"{stage}_{name}",
                    mean_value,
                    batch_size=batch_size,
                    prog_bar=prog_bar,
                )

        if not log_by_dataset:
            return

        for ds_id in torch.unique(dataset_ids).detach().cpu().tolist():
            ds_id = int(ds_id)
            ds_mask = dataset_ids == ds_id
            ds_count = int(ds_mask.sum().detach().cpu().item())

            if ds_count == 0:
                continue

            ds_mean = self._finite_mean(values[ds_mask])

            if ds_mean is None:
                continue

            self._log_scalar(
                f"{stage}_{name}_by_dataset/{self._dataset_name(ds_id)}",
                ds_mean,
                batch_size=ds_count,
            )

    def _log_pcc_by_dataset(
        self,
        *,
        stage: str,
        name: str,
        pred: torch.Tensor | None,
        target: torch.Tensor,
        mask: torch.Tensor,
        dataset_ids: torch.Tensor,
        batch_size: int,
        log_global: bool = True,
        prog_bar: bool = False,
    ) -> None:
        pcc_per_sample = self._masked_pcc_per_sample(
            pred=pred,
            target=target,
            mask=mask,
        )

        if pcc_per_sample is None:
            return

        self._log_vector_global_and_by_dataset(
            stage=stage,
            name=f"{name}_pcc",
            values=pcc_per_sample,
            dataset_ids=dataset_ids,
            batch_size=batch_size,
            log_global=log_global,
            log_by_dataset=(stage == "val"),
            prog_bar=prog_bar,
        )

    def _per_sample_valid_mean(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_b = mask.bool()
        mask_f = mask_b.float()

        x = x.detach().float()

        if x.ndim == 0:
            return x.reshape(1).expand(mask.shape[0])

        if x.ndim == 1:
            if x.shape[0] == mask.shape[0]:
                return x
            return x.reshape(1).expand(mask.shape[0])

        if x.ndim == 2 and x.shape == mask.shape:
            return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

        if x.ndim == 2 and x.shape[1] == 1:
            return x.squeeze(1)

        return x.reshape(x.shape[0], -1).mean(dim=1)

    def _log_kappa_diagnostics(
        self,
        *,
        stage: str,
        kappa: torch.Tensor | float | None,
        mask: torch.Tensor,
        dataset_ids: torch.Tensor,
        batch_size: int,
    ) -> None:
        if kappa is None:
            fixed_kappa = self._cfg("loss.kappa", None)
            if fixed_kappa is None:
                return
            kappa_t = torch.full(
                (mask.shape[0],),
                float(fixed_kappa),
                device=mask.device,
            )
        elif not torch.is_tensor(kappa):
            kappa_t = torch.full(
                (mask.shape[0],),
                float(kappa),
                device=mask.device,
            )
        else:
            kappa_t = self._per_sample_valid_mean(kappa, mask)

        self._log_vector_global_and_by_dataset(
            stage=stage,
            name="profile_kappa",
            values=kappa_t,
            dataset_ids=dataset_ids,
            batch_size=batch_size,
            log_global=True,
            log_by_dataset=(stage == "val"),
        )

        profile_dispersion = 1.0 / kappa_t.clamp_min(1e-8)

        self._log_vector_global_and_by_dataset(
            stage=stage,
            name="profile_overdispersion_proxy",
            values=profile_dispersion,
            dataset_ids=dataset_ids,
            batch_size=batch_size,
            log_global=True,
            log_by_dataset=(stage == "val"),
        )

        kappa_min = float(self._cfg("loss.kappa_min", 1e-2))
        kappa_max = float(self._cfg("loss.kappa_max", 1e5))

        at_min = (kappa_t <= 1.05 * kappa_min).float()
        at_max = (kappa_t >= 0.95 * kappa_max).float()

        self._log_vector_global_and_by_dataset(
            stage=stage,
            name="profile_kappa_at_min_frac",
            values=at_min,
            dataset_ids=dataset_ids,
            batch_size=batch_size,
            log_global=True,
            log_by_dataset=(stage == "val"),
        )

        self._log_vector_global_and_by_dataset(
            stage=stage,
            name="profile_kappa_at_max_frac",
            values=at_max,
            dataset_ids=dataset_ids,
            batch_size=batch_size,
            log_global=True,
            log_by_dataset=(stage == "val"),
        )

    # ============================================================
    # Component helpers
    # ============================================================

    def _safe_component(self, extras: dict, *keys: str) -> torch.Tensor | None:
        for key in keys:
            value = extras.get(key)

            if value is not None:
                return value

        return None

    def _get_b_multiplier(
        self,
        *,
        extras: dict,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        b = self._safe_component(extras, "exp_b", "b_multiplier")

        if b is not None:
            return b

        b_raw = extras.get("b")

        if b_raw is not None:
            mask_b = mask.bool()

            with torch.no_grad():
                valid = b_raw.detach().float()[mask_b]
                finite = valid[torch.isfinite(valid)]

                if finite.numel() == 0:
                    return None

                looks_positive_multiplier = bool((finite >= 0).all().item())

            if looks_positive_multiplier:
                return b_raw

        log_b = extras.get("log_b")
        if log_b is not None:
            return torch.exp(log_b)

        if b_raw is not None:
            return torch.exp(b_raw)

        return None

    def _get_b_control(self, extras: dict) -> torch.Tensor | None:
        return self._safe_component(extras, "control", "b_control", "log_b_control")

    def _get_total_mass(
        self,
        *,
        extras: dict,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        total_mass = extras.get("total_mass")
        mask_f = mask.bool().float()

        if total_mass is None:
            return (
                target.float().clamp_min(0.0) * mask_f
            ).sum(dim=1, keepdim=True).detach()

        total_mass = total_mass.detach().float()

        if total_mass.ndim == 1:
            total_mass = total_mass.reshape(-1, 1)

        return total_mass

    def _make_L_only_mu(
        self,
        *,
        extras: dict,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        L_queue = extras.get("L_queue")

        if L_queue is None:
            return None

        mask_b = mask.bool()
        mask_f = mask_b.float()
        eps = float(self._cfg("loss.eps", 1e-8))

        total_mass = self._get_total_mass(extras=extras, target=target, mask=mask_b)

        L_q = L_queue.detach().float().clamp_min(0.0) * mask_f
        L_mass = L_q.sum(dim=1, keepdim=True).clamp_min(eps)

        mu_L_only = total_mass * L_q / L_mass
        return mu_L_only * mask_f

    def _make_bio_only_mu(
        self,
        *,
        extras: dict,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        mask_b = mask.bool()
        mask_f = mask_b.float()

        bio_q = extras.get("bio_q")

        if bio_q is None:
            L_queue = extras.get("L_queue")
            b = self._get_b_multiplier(extras=extras, mask=mask_b)

            if L_queue is None or b is None:
                return None

            bio_q = L_queue.detach().float().clamp_min(0.0) * b.detach().float().clamp_min(0.0)

        total_mass = self._get_total_mass(extras=extras, target=target, mask=mask_b)
        eps = float(self._cfg("loss.eps", 1e-8))

        bio_q = bio_q.detach().float().clamp_min(0.0) * mask_f
        bio_mass = bio_q.sum(dim=1, keepdim=True).clamp_min(eps)

        mu_bio_only = total_mass * bio_q / bio_mass
        return mu_bio_only * mask_f

    # ============================================================
    # Validation diagnostics
    # ============================================================

    def _log_core_validation_metrics(
        self,
        *,
        target: torch.Tensor,
        mask: torch.Tensor,
        dataset_ids: torch.Tensor,
        mu: torch.Tensor,
        extras: dict,
        batch_size: int,
    ) -> None:
        L_queue = extras.get("L_queue")

        if L_queue is not None:
            self._log_pcc_by_dataset(
                stage="val",
                name="L_queue",
                pred=L_queue,
                target=target,
                mask=mask,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
                prog_bar=True,
            )

        self._log_pcc_by_dataset(
            stage="val",
            name="mu",
            pred=mu,
            target=target,
            mask=mask,
            dataset_ids=dataset_ids,
            batch_size=batch_size,
            log_global=True,
            prog_bar=True,
        )

    def _log_bias_and_responsibility_diagnostics(
        self,
        *,
        target: torch.Tensor,
        mask: torch.Tensor,
        dataset_ids: torch.Tensor,
        mu: torch.Tensor,
        extras: dict,
        batch_size: int,
    ) -> None:
        mask_b = mask.bool()
        mask_f = mask_b.float()
        eps = float(self._cfg("loss.eps", 1e-8))

        b = self._get_b_multiplier(extras=extras, mask=mask_b)
        b_control = self._get_b_control(extras)
        log_b = extras.get("log_b")

        if log_b is None and b is not None:
            log_b = torch.log(b.detach().float().clamp_min(eps))

        R_shape = extras.get("R_shape")
        additive_noise = extras.get("additive_noise")
        L_queue = extras.get("L_queue")
        bio_q = extras.get("bio_q")
        q = extras.get("q")
        lambda_bg = extras.get("lambda_bg")

        # Component-to-target correlations.
        if b is not None:
            self._log_pcc_by_dataset(
                stage="val",
                name="b_vs_y",
                pred=b,
                target=target,
                mask=mask_b,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
            )

        if R_shape is not None:
            self._log_pcc_by_dataset(
                stage="val",
                name="R_shape_vs_y",
                pred=R_shape,
                target=target,
                mask=mask_b,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
            )

        if additive_noise is not None:
            self._log_pcc_by_dataset(
                stage="val",
                name="additive_noise_vs_y",
                pred=additive_noise,
                target=target,
                mask=mask_b,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
            )

        # L-only -> L*b -> full decomposition.
        mu_L_only = self._make_L_only_mu(extras=extras, target=target, mask=mask_b)
        mu_bio_only = self._make_bio_only_mu(extras=extras, target=target, mask=mask_b)

        if mu_L_only is not None:
            self._log_pcc_by_dataset(
                stage="val",
                name="mu_L_only",
                pred=mu_L_only,
                target=target,
                mask=mask_b,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
            )

        if mu_bio_only is not None:
            self._log_pcc_by_dataset(
                stage="val",
                name="mu_bio_only",
                pred=mu_bio_only,
                target=target,
                mask=mask_b,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
            )

        mu_pcc = self._masked_pcc_per_sample(pred=mu, target=target, mask=mask_b)
        L_pcc = self._masked_pcc_per_sample(pred=mu_L_only, target=target, mask=mask_b)
        bio_pcc = self._masked_pcc_per_sample(pred=mu_bio_only, target=target, mask=mask_b)

        if L_pcc is not None and bio_pcc is not None:
            b_gain = bio_pcc - L_pcc
            self._log_vector_global_and_by_dataset(
                stage="val",
                name="b_gain_pcc",
                values=b_gain,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
                log_by_dataset=True,
            )

        if mu_pcc is not None and bio_pcc is not None:
            R_gain = mu_pcc - bio_pcc
            self._log_vector_global_and_by_dataset(
                stage="val",
                name="R_gain_pcc",
                values=R_gain,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
                log_by_dataset=True,
            )

        # Alignment with biological support.
        if L_queue is not None and b is not None:
            self._log_pcc_by_dataset(
                stage="val",
                name="b_vs_L_queue",
                pred=b,
                target=L_queue,
                mask=mask_b,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
            )

        if L_queue is not None and R_shape is not None:
            self._log_pcc_by_dataset(
                stage="val",
                name="R_shape_vs_L_queue",
                pred=R_shape,
                target=L_queue,
                mask=mask_b,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
            )

        # Branch mass fractions / responsibilities.
        if bio_q is None and L_queue is not None and b is not None:
            bio_q = L_queue.detach().float().clamp_min(0.0) * b.detach().float().clamp_min(0.0)

        if bio_q is not None and additive_noise is not None:
            bio_mass = (bio_q.detach().float().clamp_min(0.0) * mask_f).sum(dim=1)
            bg_mass = (additive_noise.detach().float().clamp_min(0.0) * mask_f).sum(dim=1)
            bg_frac = bg_mass / (bio_mass + bg_mass).clamp_min(eps)

            self._log_vector_global_and_by_dataset(
                stage="val",
                name="bg_frac",
                values=bg_frac,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
                log_by_dataset=True,
            )

            if q is None:
                q = bio_q + additive_noise

            q_safe = q.detach().float().clamp_min(eps)
            bio_resp = (bio_q.detach().float().clamp_min(0.0) / q_safe) * mask_f
            bg_resp = (additive_noise.detach().float().clamp_min(0.0) / q_safe) * mask_f
            valid_lengths = mask_f.sum(dim=1).clamp_min(1.0)

            bio_resp_mean = bio_resp.sum(dim=1) / valid_lengths
            bg_resp_mean = bg_resp.sum(dim=1) / valid_lengths

            self._log_vector_global_and_by_dataset(
                stage="val",
                name="bio_responsibility",
                values=bio_resp_mean,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
                log_by_dataset=True,
            )

            self._log_vector_global_and_by_dataset(
                stage="val",
                name="bg_responsibility",
                values=bg_resp_mean,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
                log_by_dataset=True,
            )

        if R_shape is not None:
            eps_support = float(self._cfg("metrics.R_support_eps", 1e-8))
            R_support_frac = (
                ((R_shape.detach().float() > eps_support) & mask_b).float().sum(dim=1)
                / mask_f.sum(dim=1).clamp_min(1.0)
            )

            self._log_vector_global_and_by_dataset(
                stage="val",
                name="R_support_frac",
                values=R_support_frac,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
                log_by_dataset=True,
            )

        if b is not None:
            b_valid = b.detach().float()
            b_zero_eps = float(self._cfg("metrics.b_zero_eps", 1e-8))
            b_high_threshold = float(self._cfg("metrics.b_high_threshold", 2.0))

            b_zero_frac = ((b_valid <= b_zero_eps) & mask_b).float().sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            b_high_frac = ((b_valid >= b_high_threshold) & mask_b).float().sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

            self._log_vector_global_and_by_dataset(
                stage="val",
                name="b_zero_frac",
                values=b_zero_frac,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
                log_by_dataset=True,
            )

            self._log_vector_global_and_by_dataset(
                stage="val",
                name="b_high_frac",
                values=b_high_frac,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
                log_by_dataset=True,
            )

        if log_b is not None:
            b_abs_log_mean = (
                log_b.detach().float().abs() * mask_f
            ).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

            self._log_vector_global_and_by_dataset(
                stage="val",
                name="b_abs_log_mean",
                values=b_abs_log_mean,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
                log_by_dataset=True,
            )

        if b_control is not None:
            control_abs_mean = (
                b_control.detach().float().abs() * mask_f
            ).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

            self._log_vector_global_and_by_dataset(
                stage="val",
                name="b_control_abs_mean",
                values=control_abs_mean,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
                log_by_dataset=True,
            )

        if lambda_bg is not None:
            lambda_bg = lambda_bg.detach().float()

            if lambda_bg.ndim > 1:
                lambda_per_sample = lambda_bg.reshape(lambda_bg.shape[0], -1).mean(dim=1)
            else:
                lambda_per_sample = lambda_bg.reshape(-1)

            self._log_vector_global_and_by_dataset(
                stage="val",
                name="lambda_bg",
                values=lambda_per_sample,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                log_global=True,
                log_by_dataset=True,
            )

    # ============================================================
    # CSS recall helpers
    # ============================================================

    def _normalize_css_positions(self, css_i: Any, L: int) -> list[int]:
        if css_i is None:
            return []

        try:
            if torch.is_tensor(css_i):
                arr = css_i.detach().cpu().reshape(-1)
            else:
                arr = torch.as_tensor(css_i).detach().cpu().reshape(-1)
        except Exception:
            return []

        if arr.numel() == 0:
            return []

        if arr.dtype == torch.bool:
            arr_b = arr[:L]
            return [int(i) for i in torch.nonzero(arr_b, as_tuple=False).reshape(-1).tolist()]

        arr_f = arr.float()
        arr_f = arr_f[torch.isfinite(arr_f)]

        if arr_f.numel() == 0:
            return []

        if arr_f.numel() >= L:
            first_L = arr_f[:L]
            is_binary_mask = torch.all((first_L == 0.0) | (first_L == 1.0)).item()

            if is_binary_mask:
                return [
                    int(i)
                    for i in torch.nonzero(first_L.bool(), as_tuple=False).reshape(-1).tolist()
                ]

        pos = arr_f.long()
        pos = pos[(pos >= 0) & (pos < L)]
        return sorted(set(int(x) for x in pos.tolist()))

    def _peak_budget(self, L: int) -> int:
        top_frac = float(self._cfg("metrics.css_peak_top_frac", 0.02))
        min_peaks = int(self._cfg("metrics.css_peak_min_peaks", 1))
        max_peaks = int(self._cfg("metrics.css_peak_max_peaks", 50))

        k = int(math.ceil(float(L) * top_frac))
        k = max(k, min_peaks)
        k = min(k, max_peaks)
        k = min(k, L)
        return max(k, 0)

    def _topk_peak_positions_1d(
        self,
        signal_1d: torch.Tensor,
        *,
        k: int,
        nms_radius: int,
    ) -> list[int]:
        if signal_1d.numel() == 0 or k <= 0:
            return []

        x = signal_1d.detach().float().cpu()
        finite = torch.isfinite(x)

        if not finite.any():
            return []

        x = torch.where(finite, x, torch.full_like(x, -float("inf")))
        ordered = torch.argsort(x, descending=True).tolist()
        selected: list[int] = []

        for idx in ordered:
            if not torch.isfinite(x[idx]):
                continue

            if all(abs(idx - prev) > nms_radius for prev in selected):
                selected.append(int(idx))

            if len(selected) >= k:
                break

        return selected

    def _positions_hit_count(
        self,
        *,
        reference_positions: list[int],
        predicted_positions: list[int],
        tolerance: int,
    ) -> int:
        if len(reference_positions) == 0 or len(predicted_positions) == 0:
            return 0

        hits = 0
        for ref in reference_positions:
            if any(abs(ref - pred) <= tolerance for pred in predicted_positions):
                hits += 1

        return hits

    def _log_css_recall_diagnostics(
        self,
        *,
        target: torch.Tensor,
        mask: torch.Tensor,
        css: list,
        mu: torch.Tensor,
        L_queue: torch.Tensor | None,
        batch_size: int,
    ) -> None:
        if css is None or L_queue is None:
            return

        tolerance = int(self._cfg("metrics.css_recall_tolerance", 1))
        nms_radius = int(self._cfg("metrics.css_peak_nms_radius", tolerance))
        device = target.device
        mask_b = mask.bool()

        recalls = {"mu": [], "L_queue": []}
        components = {"mu": mu, "L_queue": L_queue}

        with torch.no_grad():
            B = int(target.shape[0])

            for i in range(B):
                mask_i = mask_b[i].cpu()
                L = int(mask_i.sum().item())

                if L < 2:
                    continue

                css_i = css[i] if i < len(css) else None
                css_positions = self._normalize_css_positions(css_i, L)

                if not css_positions:
                    continue

                k = self._peak_budget(L)
                y_valid = target[i].detach().float().cpu()[mask_i]

                y_peaks = self._topk_peak_positions_1d(
                    y_valid,
                    k=k,
                    nms_radius=nms_radius,
                )

                y_supported_css = [
                    c for c in css_positions
                    if any(abs(c - yp) <= tolerance for yp in y_peaks)
                ]

                if not y_supported_css:
                    continue

                for name, comp in components.items():
                    comp_valid = comp[i].detach().float().cpu()[mask_i]
                    pred_peaks = self._topk_peak_positions_1d(
                        comp_valid,
                        k=k,
                        nms_radius=nms_radius,
                    )

                    hits = self._positions_hit_count(
                        reference_positions=y_supported_css,
                        predicted_positions=pred_peaks,
                        tolerance=tolerance,
                    )

                    recalls[name].append(float(hits) / max(len(y_supported_css), 1))

            for name, vals in recalls.items():
                if vals:
                    self._log_scalar(
                        f"val_{name}_css_recall",
                        torch.tensor(vals, device=device).mean(),
                        batch_size=batch_size,
                        prog_bar=(name == "L_queue"),
                    )

    # ============================================================
    # Gradient diagnostics
    # ============================================================

    def _grad_stats_for_module(self, module: torch.nn.Module) -> dict[str, torch.Tensor] | None:
        grad_sq_sum = None
        param_sq_sum = None
        max_abs_grad = None

        n_params = 0
        n_with_grad = 0
        n_finite_grad = 0

        for p in module.parameters():
            if not p.requires_grad:
                continue

            n_params += p.numel()
            p_sq = p.detach().float().pow(2).sum()
            param_sq_sum = p_sq if param_sq_sum is None else param_sq_sum + p_sq

            if p.grad is None:
                continue

            g = p.grad.detach().float()
            finite = torch.isfinite(g)
            n_with_grad += g.numel()
            n_finite_grad += int(finite.sum().detach().cpu().item())
            g_safe = torch.where(finite, g, torch.zeros_like(g))

            g_sq = g_safe.pow(2).sum()
            g_max = g_safe.abs().max()

            grad_sq_sum = g_sq if grad_sq_sum is None else grad_sq_sum + g_sq
            max_abs_grad = g_max if max_abs_grad is None else torch.maximum(max_abs_grad, g_max)

        if n_params == 0:
            return None

        device = self.device
        grad_norm = torch.zeros((), device=device) if grad_sq_sum is None else grad_sq_sum.sqrt()
        param_norm = torch.zeros((), device=device) if param_sq_sum is None else param_sq_sum.sqrt()
        max_abs_grad = torch.zeros((), device=device) if max_abs_grad is None else max_abs_grad

        return {
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "max_abs_grad": max_abs_grad,
            "grad_coverage": torch.tensor(n_with_grad / max(n_params, 1), device=device),
            "finite_grad_frac": torch.tensor(n_finite_grad / max(n_with_grad, 1), device=device),
        }

    def _branch_modules_for_grad_logging(self) -> dict[str, torch.nn.Module]:
        branches: dict[str, torch.nn.Module] = {}

        if hasattr(self.model, "biological_model"):
            branches["biological"] = self.model.biological_model

        db = getattr(self.model, "dataset_bias_model", None)

        if db is not None:
            for name in [
                "dataset_embedding",
                "codon_embedding",
                "local_context_cnn",
                "dataset_multiplicative_bias_head",
                "additive_bias_head",
                "dispersion_head",
                "kappa_head",
                "dataset_kappa_head",
                "tweedie_power_head",  # harmless if still present as legacy aux head
            ]:
                if hasattr(db, name):
                    branches[f"bias/{name}"] = getattr(db, name)

        return branches

    def log_gradient_diagnostics(self) -> None:
        every_n = int(self._cfg("metrics.grad_log_every_n_steps", 0))

        if every_n <= 0:
            return

        if int(self.global_step) % every_n != 0:
            return

        if len(self.trainer.optimizers) == 0:
            return

        default_lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
        grad_norms = {}

        for name, module in self._branch_modules_for_grad_logging().items():
            stats = self._grad_stats_for_module(module)

            if stats is None:
                continue

            grad_norm = stats["grad_norm"]
            param_norm = stats["param_norm"]
            update_ratio = default_lr * grad_norm / param_norm.clamp_min(1e-12)

            self.log(f"grad/{name}/norm", grad_norm, on_step=True, on_epoch=False, logger=True)
            self.log(f"grad/{name}/param_norm", param_norm, on_step=True, on_epoch=False, logger=True)
            self.log(f"grad/{name}/update_ratio", update_ratio, on_step=True, on_epoch=False, logger=True)
            self.log(f"grad/{name}/max_abs", stats["max_abs_grad"], on_step=True, on_epoch=False, logger=True)
            self.log(f"grad/{name}/coverage", stats["grad_coverage"], on_step=True, on_epoch=False, logger=True)
            self.log(f"grad/{name}/finite_frac", stats["finite_grad_frac"], on_step=True, on_epoch=False, logger=True)

            grad_norms[name] = grad_norm.detach()

        if len(grad_norms) > 1:
            total = torch.stack(list(grad_norms.values())).sum().clamp_min(1e-12)

            for name, norm in grad_norms.items():
                self.log(f"grad_share/{name}", norm / total, on_step=True, on_epoch=False, logger=True)

    def on_after_backward(self) -> None:
        if not self.use_pcgrad:
            self.log_gradient_diagnostics()

    # ============================================================
    # Example plot
    # ============================================================

    def on_validation_epoch_start(self):
        self._val_plot_logged_this_epoch = False

    def _seq_to_np_for_plot(
        self,
        x: torch.Tensor | None,
        *,
        sample_idx: int,
        mask_i: torch.Tensor,
        L: int,
    ):
        if x is None or not torch.is_tensor(x):
            return None

        x = x.detach().float().cpu()

        if x.ndim == 0:
            return torch.full((L,), float(x.item())).numpy()

        if x.ndim == 1:
            if x.shape[0] == L:
                return x[:L].numpy()
            if x.shape[0] > sample_idx:
                return torch.full((L,), float(x[sample_idx].item())).numpy()
            return None

        if x.ndim == 2:
            if x.shape[0] <= sample_idx:
                return None
            if x.shape[1] == 1:
                return torch.full((L,), float(x[sample_idx, 0].item())).numpy()
            return x[sample_idx][mask_i].numpy()

        return None

    def _log_example_profile_plot(
        self,
        *,
        ids: list,
        dataset_ids: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        kappa_input: torch.Tensor | None,
        kappa: torch.Tensor | float | None,
        mask: torch.Tensor,
        extras: dict,
        batch_idx: int,
    ) -> None:
        if self._val_plot_logged_this_epoch:
            return
        if batch_idx != 0:
            return
        if not getattr(self.trainer, "is_global_zero", True):
            return
        if self.logger is None:
            return

        experiment = getattr(self.logger, "experiment", None)
        if experiment is None:
            return

        B = int(target.shape[0])
        if B < 2:
            return

        idx1 = 0
        idx2 = 1
        found_perfect_match = False

        for i in range(B):
            for j in range(i + 1, B):
                if ids[i] == ids[j] and dataset_ids[i] != dataset_ids[j]:
                    idx1 = i
                    idx2 = j
                    found_perfect_match = True
                    break
            if found_perfect_match:
                break

        if not found_perfect_match:
            for i in range(1, B):
                if dataset_ids[i] != dataset_ids[0]:
                    idx2 = i
                    break

        indices_to_plot = [idx1, idx2]

        b_multiplier = self._get_b_multiplier(extras=extras, mask=mask)
        mu_L_only = self._make_L_only_mu(extras=extras, target=target, mask=mask)
        mu_bio_only = self._make_bio_only_mu(extras=extras, target=target, mask=mask)

        if torch.is_tensor(kappa):
            kappa_plot = kappa.detach().float()
        elif kappa is None:
            kappa_plot = None
        else:
            kappa_plot = torch.tensor(float(kappa), device=target.device)

        fig, axes = plt.subplots(
            4,
            2,
            figsize=(24, 11),
            sharex="col",
            gridspec_kw={"height_ratios": [1.4, 1.0, 1.0, 1.0]},
        )

        for col, sample_idx in enumerate(indices_to_plot):
            mask_i = mask[sample_idx].detach().bool().cpu()
            L = int(mask_i.sum().item())

            if L < 2:
                continue

            y_np = self._seq_to_np_for_plot(target, sample_idx=sample_idx, mask_i=mask_i, L=L)
            mu_np = self._seq_to_np_for_plot(mu, sample_idx=sample_idx, mask_i=mask_i, L=L)
            mu_L_only_np = self._seq_to_np_for_plot(mu_L_only, sample_idx=sample_idx, mask_i=mask_i, L=L)
            mu_bio_only_np = self._seq_to_np_for_plot(mu_bio_only, sample_idx=sample_idx, mask_i=mask_i, L=L)

            L_queue_np = self._seq_to_np_for_plot(extras.get("L_queue"), sample_idx=sample_idx, mask_i=mask_i, L=L)
            bio_q_np = self._seq_to_np_for_plot(extras.get("bio_q"), sample_idx=sample_idx, mask_i=mask_i, L=L)

            additive_noise_np = self._seq_to_np_for_plot(extras.get("additive_noise"), sample_idx=sample_idx, mask_i=mask_i, L=L)
            R_shape_np = self._seq_to_np_for_plot(extras.get("R_shape"), sample_idx=sample_idx, mask_i=mask_i, L=L)

            exp_b_np = self._seq_to_np_for_plot(b_multiplier, sample_idx=sample_idx, mask_i=mask_i, L=L)
            log_b_np = self._seq_to_np_for_plot(extras.get("log_b"), sample_idx=sample_idx, mask_i=mask_i, L=L)
            control_np = self._seq_to_np_for_plot(self._get_b_control(extras), sample_idx=sample_idx, mask_i=mask_i, L=L)
            kappa_np = self._seq_to_np_for_plot(kappa_plot, sample_idx=sample_idx, mask_i=mask_i, L=L)
            kappa_input_np = self._seq_to_np_for_plot(kappa_input, sample_idx=sample_idx, mask_i=mask_i, L=L)

            x_axis = torch.arange(L).cpu().numpy()

            dataset_id = int(dataset_ids[sample_idx].detach().cpu().item())
            dataset_name = self._dataset_name(dataset_id)
            transcript_id = ids[sample_idx]
            match_status = "PERFECT MATCH" if found_perfect_match else "MISMATCHED TRANSCRIPT"

            kappa_value = None
            if kappa_np is not None:
                kappa_value = float(torch.as_tensor(kappa_np).float().mean().item())

            title = f"[{match_status}]\nDataset: {dataset_name} | ID: {transcript_id} | L={L}"
            if kappa_value is not None:
                title += f" | kappa={kappa_value:.3g}"
            axes[0, col].set_title(title)

            if y_np is not None:
                axes[0, col].plot(x_axis, y_np, label="target y", linewidth=1.0)
            if mu_np is not None:
                axes[0, col].plot(x_axis, mu_np, label="mu full", linewidth=1.0)
            if mu_bio_only_np is not None:
                axes[0, col].plot(x_axis, mu_bio_only_np, label="mu L*b", linewidth=1.0)
            if mu_L_only_np is not None:
                axes[0, col].plot(x_axis, mu_L_only_np, label="mu L-only", linewidth=0.8, alpha=0.7)

            if col == 0:
                axes[0, col].set_ylabel("profile")
            axes[0, col].grid(True, alpha=0.3)
            axes[0, col].legend(loc="upper right")

            if L_queue_np is not None:
                axes[1, col].plot(x_axis, L_queue_np, label="L_queue", linewidth=1.0)
            if bio_q_np is not None:
                axes[1, col].plot(x_axis, bio_q_np, label="bio_q = L*b", linewidth=1.0)

            if col == 0:
                axes[1, col].set_ylabel("biology")
            axes[1, col].grid(True, alpha=0.3)
            axes[1, col].legend(loc="upper right")

            if R_shape_np is not None:
                axes[2, col].plot(x_axis, R_shape_np, label="R_shape", linewidth=1.0)
            if additive_noise_np is not None:
                axes[2, col].plot(x_axis, additive_noise_np, label="additive_noise", linewidth=1.0)

            if exp_b_np is not None:
                ax2 = axes[2, col].twinx()
                ax2.plot(x_axis, exp_b_np, label="b multiplier", linewidth=0.8, linestyle="--", alpha=0.7)
                if col == 1:
                    ax2.set_ylabel("b multiplier")
                ax2.legend(loc="upper left")

            if col == 0:
                axes[2, col].set_ylabel("residual / b")
            axes[2, col].grid(True, alpha=0.3)
            axes[2, col].legend(loc="upper right")

            if kappa_np is not None:
                axes[3, col].plot(x_axis, kappa_np, label="profile kappa", linewidth=0.8)
            if kappa_input_np is not None:
                axes[3, col].plot(x_axis, kappa_input_np, label="kappa input", linewidth=0.8, alpha=0.6)
            if log_b_np is not None:
                axes[3, col].plot(x_axis, log_b_np, label="log_b", linewidth=0.8)
            if control_np is not None:
                axes[3, col].plot(x_axis, control_np, label="b control", linewidth=0.8, linestyle="--")

            if col == 0:
                axes[3, col].set_ylabel("diagnostic")
            axes[3, col].set_xlabel("codon position")
            axes[3, col].grid(True, alpha=0.3)
            axes[3, col].legend(loc="upper right")

        fig.tight_layout()

        tag = "val/example_profile_comparison"
        if hasattr(experiment, "add_figure"):
            experiment.add_figure(tag, fig, global_step=self.global_step)
        elif hasattr(experiment, "log_figure"):
            experiment.log_figure(figure_name=tag, figure=fig, step=self.global_step)

        plt.close(fig)
        self._val_plot_logged_this_epoch = True

    # ============================================================
    # PCGrad
    # ============================================================

    def pcgrad_optimize(
        self,
        *,
        loss_per_sample: torch.Tensor,
        dataset_ids: torch.Tensor,
        extra_loss: torch.Tensor | None = None,
    ) -> torch.Tensor:
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        biological_model = self.model.biological_model
        biological_model_params = [
            p for p in biological_model.parameters() if p.requires_grad
        ]

        dataset_losses = per_dataset_losses(
            loss_per_sample=loss_per_sample,
            dataset_ids=dataset_ids,
        )

        if len(dataset_losses) > 0:
            scalar_loss = torch.stack(dataset_losses).mean()
        else:
            scalar_loss = loss_per_sample.mean()

        full_loss = scalar_loss + extra_loss if extra_loss is not None else scalar_loss

        if len(dataset_losses) <= 1 or len(biological_model_params) == 0:
            self.manual_backward(full_loss)
            self.log_gradient_diagnostics()
            return scalar_loss.detach()

        max_datasets = int(
            self._cfg("optim.pcgrad_max_datasets_per_step", len(dataset_losses))
        )

        if len(dataset_losses) > max_datasets:
            perm = torch.randperm(len(dataset_losses), device=scalar_loss.device)
            selected = perm[:max_datasets].detach().cpu().tolist()
            dataset_losses_for_pcgrad = [dataset_losses[j] for j in selected]
        else:
            dataset_losses_for_pcgrad = dataset_losses

        raw_flat_dataset_grads = []
        unit_flat_dataset_grads = []

        for ds_loss in dataset_losses_for_pcgrad:
            opt.zero_grad(set_to_none=True)
            self.manual_backward(ds_loss, retain_graph=True)

            g_raw = flatten_current_grads(biological_model_params).detach()
            g_norm = g_raw.norm().clamp_min(1e-12)

            raw_flat_dataset_grads.append(g_raw)
            unit_flat_dataset_grads.append((g_raw / g_norm).detach())

        raw_grads = torch.stack(raw_flat_dataset_grads, dim=0)
        unit_grads = torch.stack(unit_flat_dataset_grads, dim=0)

        pcgrad_flat = pcgrad_combine(unit_flat_dataset_grads).detach()
        mean_unit_grad = unit_grads.mean(dim=0)

        alpha = float(self._cfg("optim.pcgrad_alpha", 1.0))
        pcgrad_flat = alpha * pcgrad_flat + (1.0 - alpha) * mean_unit_grad

        if bool(self._cfg("optim.pcgrad_rescale_to_mean_norm", True)):
            target_norm = raw_grads.norm(dim=1).mean().clamp_min(1e-12)
            pcgrad_norm = pcgrad_flat.norm().clamp_min(1e-12)
            pcgrad_flat = pcgrad_flat * (target_norm / pcgrad_norm)

        opt.zero_grad(set_to_none=True)
        self.manual_backward(full_loss)

        assign_flat_grads(
            params=biological_model_params,
            flat_grad=pcgrad_flat.detach(),
        )

        self.log_gradient_diagnostics()

        del raw_flat_dataset_grads
        del unit_flat_dataset_grads
        del raw_grads
        del unit_grads
        del pcgrad_flat
        del mean_unit_grad

        return scalar_loss.detach()

    # ============================================================
    # Training / validation
    # ============================================================

    def _forward_batch(self, batch):
        (
            ids_datasets_sorted,
            ids_sorted,
            seq_packed,
            prof_pad,
            lengths_sorted,
            mask_pad,
            codon_ids_pad,
            css_sorted,
        ) = batch

        # Backward-compatible model output:
        #     profile_aux is usually the old p output; it is not used by the profile loss.
        #     kappa_input is usually the old phi/dispersion output or a new kappa head output.
        mu, kappa_input, extras = self.model(
            seq_packed,
            codon_ids_pad,
            ids_datasets_sorted,
            prof_pad,
        )

        return {
            "dataset_ids": ids_datasets_sorted,
            "ids": ids_sorted,
            "target": prof_pad,
            "lengths": lengths_sorted,
            "mask": mask_pad,
            "codon_ids": codon_ids_pad,
            "css": css_sorted,
            "mu": mu,
            "kappa_input": kappa_input,
            "extras": extras,
        }

    def training_step(self, batch, batch_idx):
        out = self._forward_batch(batch)
        batch_size = int(out["target"].shape[0])

        loss_terms = self._build_loss_terms(
            mu=out["mu"],
            kappa_input=out["kappa_input"],
            target=out["target"],
            mask=out["mask"],
            extras=out["extras"],
        )

        scalar_loss = self._dataset_balanced_scalar_loss(
            loss_per_sample=loss_terms["loss_per_sample"],
            dataset_ids=out["dataset_ids"],
        )

        loss = scalar_loss + loss_terms["extra_loss"]

        self._log_scalar(
            "train_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
        )

        kappa = self._compute_kappa_from_model_output(
            kappa_input=out["kappa_input"],
            mask=out["mask"],
        )

        self._log_kappa_diagnostics(
            stage="train",
            kappa=kappa,
            mask=out["mask"],
            dataset_ids=out["dataset_ids"],
            batch_size=batch_size,
        )

        if not self.use_pcgrad:
            return loss

        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        pcgrad_every_n_steps = int(self._cfg("optim.pcgrad_every_n_steps", 1))
        use_pcgrad_this_step = (
            pcgrad_every_n_steps <= 1
            or self.global_step % pcgrad_every_n_steps == 0
        )

        if use_pcgrad_this_step:
            step_loss = self.pcgrad_optimize(
                loss_per_sample=loss_terms["loss_per_sample"],
                dataset_ids=out["dataset_ids"],
                extra_loss=loss_terms["extra_loss"],
            )
        else:
            self.manual_backward(loss)
            self.log_gradient_diagnostics()
            step_loss = scalar_loss.detach()

        grad_clip_val = float(self._cfg("trainer.gradient_clip_val", 0.0))

        if grad_clip_val > 0.0:
            self.clip_gradients(
                opt,
                gradient_clip_val=grad_clip_val,
                gradient_clip_algorithm=str(self._cfg("trainer.gradient_clip_algorithm", "norm")),
            )

        opt.step()
        opt.zero_grad(set_to_none=True)

        return step_loss.detach()

    def validation_step(self, batch, batch_idx):
        out = self._forward_batch(batch)
        batch_size = int(out["target"].shape[0])

        loss_terms = self._build_loss_terms(
            mu=out["mu"],
            kappa_input=out["kappa_input"],
            target=out["target"],
            mask=out["mask"],
            extras=out["extras"],
        )

        val_loss = self._dataset_balanced_scalar_loss(
            loss_per_sample=loss_terms["loss_per_sample"],
            dataset_ids=out["dataset_ids"],
        )
        val_loss = val_loss + loss_terms["extra_loss"]

        self._log_scalar(
            "val_loss",
            val_loss,
            batch_size=batch_size,
            prog_bar=True,
        )

        kappa = self._compute_kappa_from_model_output(
            kappa_input=out["kappa_input"],
            mask=out["mask"],
        )

        self._log_kappa_diagnostics(
            stage="val",
            kappa=kappa,
            mask=out["mask"],
            dataset_ids=out["dataset_ids"],
            batch_size=batch_size,
        )

        self._log_core_validation_metrics(
            target=out["target"],
            mask=out["mask"],
            dataset_ids=out["dataset_ids"],
            mu=out["mu"],
            extras=out["extras"],
            batch_size=batch_size,
        )

        self._log_css_recall_diagnostics(
            target=out["target"],
            mask=out["mask"],
            css=out["css"],
            mu=out["mu"],
            L_queue=out["extras"].get("L_queue"),
            batch_size=batch_size,
        )

        self._log_bias_and_responsibility_diagnostics(
            target=out["target"],
            mask=out["mask"],
            dataset_ids=out["dataset_ids"],
            mu=out["mu"],
            extras=out["extras"],
            batch_size=batch_size,
        )

        self._log_example_profile_plot(
            ids=out["ids"],
            dataset_ids=out["dataset_ids"],
            target=out["target"],
            mu=out["mu"],
            kappa_input=out["kappa_input"],
            kappa=kappa,
            mask=out["mask"],
            extras=out["extras"],
            batch_idx=batch_idx,
        )

        return val_loss

    def on_validation_epoch_end(self):
        if not self.use_pcgrad:
            return

        sched = self.lr_schedulers()
        monitor = str(self._cfg("optim.scheduler.monitor", "val_loss"))

        if monitor not in self.trainer.callback_metrics:
            return

        metric = self.trainer.callback_metrics[monitor]
        sched.step(metric)

    # ============================================================
    # Prediction
    # ============================================================

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        out = self._forward_batch(batch)

        def to_cpu(x):
            if torch.is_tensor(x):
                return x.detach().cpu()
            if isinstance(x, (list, tuple)):
                return [to_cpu(v) for v in x]
            return x

        kappa = self._compute_kappa_from_model_output(
            kappa_input=out["kappa_input"],
            mask=out["mask"],
        )

        output = {
            "ids": to_cpu(out["ids"]),
            "dataset_id": to_cpu(out["dataset_ids"]),
            "lengths": to_cpu(out["lengths"]),
            "mask": to_cpu(out["mask"]),
            "css": to_cpu(out["css"]),
            "y": to_cpu(out["target"]),
            "mu_obs": to_cpu(out["mu"]),
            "profile_aux": to_cpu(out["profile_aux"]),
            "kappa_input": to_cpu(out["kappa_input"]),
            "profile_kappa": to_cpu(kappa),
        }

        extras = out["extras"]
        if isinstance(extras, dict):
            for key, val in extras.items():
                if val is not None:
                    output[key] = to_cpu(val)

        return output

    # ============================================================
    # Optimizer
    # ============================================================

    def configure_optimizers(self):
        base_lr = float(self._cfg("optim.lr", 1e-3))
        bio_lr = float(self._cfg("optim.lr_biological", base_lr))
        rest_lr = float(self._cfg("optim.lr_rest", base_lr))
        weight_decay = float(self._cfg("optim.weight_decay", 1e-8))

        biological_params = []
        rest_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if name.startswith("biological_model."):
                biological_params.append(param)
            else:
                rest_params.append(param)

        param_groups = []

        if biological_params:
            param_groups.append(
                {
                    "params": biological_params,
                    "lr": bio_lr,
                    "weight_decay": weight_decay,
                }
            )

        if rest_params:
            param_groups.append(
                {
                    "params": rest_params,
                    "lr": rest_lr,
                    "weight_decay": weight_decay,
                }
            )

        if not param_groups:
            raise RuntimeError("No trainable parameters found.")

        opt = torch.optim.AdamW(param_groups)

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=str(self._cfg("optim.scheduler.mode", "min")),
            factor=float(self._cfg("optim.scheduler.factor", 0.9)),
            patience=int(self._cfg("optim.scheduler.patience", 10)),
            min_lr=float(self._cfg("optim.scheduler.min_lr", 1e-6)),
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": str(self._cfg("optim.scheduler.monitor", "val_loss")),
                "interval": "epoch",
                "frequency": 1,
            },
        }
