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
from Models.utils.ribo_lightning_helpers import flatten_current_grads
from Models.utils.tweedie_deviance_loss import TweedieDevianceLoss


class RiboQueuingModelLightningModule(pl.LightningModule):
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

        self.loss_fn = TweedieDevianceLoss(include_log_phi=True)

        enc = dataset_encoding or {}
        self.dataset_id_to_name = {int(v): str(k) for k, v in enc.items()}

        self.use_pcgrad = bool(self._cfg("optim.use_pcgrad", False))
        self.automatic_optimization = not self.use_pcgrad

        self.log_sync_dist = bool(self._cfg("trainer.sync_dist_logs", False))

        if self.use_pcgrad:
            print("Training is using PCGrad.")

        self._val_plot_logged_this_epoch = False

    # ============================================================
    # Config helper
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

    # ============================================================
    # Loss helpers
    # ============================================================

    def _compute_raw_nll_per_sample(
        self,
        *,
        mu: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=mu.device.type, enabled=False):
            return self.loss_fn(
                mu_phys=mu.float(),
                power=p.float(),
                phi=phi.float(),
                y_true=target.float(),
                mask=mask.bool(),
                return_per_sample=True,
                phi_reg_alpha=float(self._cfg("loss.phi_reg_alpha", 1.0)),
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
            stacked_losses = torch.stack(dataset_losses)

            # 1. Calculate the target magnitude (detached so it doesn't absorb gradients)
            # This is the average loss scale across the current batch.
            target_magnitude = stacked_losses.detach().mean().clamp_min(1e-8)

            # 2. Calculate a scale factor for each dataset
            # If Kutay = 500 and Grimson = 10, target = 255.
            # Kutay scale = 255/500 (0.51). Grimson scale = 255/10 (25.5).
            scale_factors = target_magnitude / stacked_losses.detach().clamp_min(1e-8)

            # 3. Apply the scale factors
            # Now both datasets report a loss magnitude of 255 to the optimizer,
            # meaning a 1% relative improvement in Grimson is valued exactly the same
            # as a 1% improvement in Kutay.
            balanced_losses = stacked_losses * scale_factors

            return balanced_losses.mean()

        return loss_per_sample.mean()

    def mass_matching_loss(
        self,
        *,
        mu: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        mask_f = mask.bool().to(dtype=mu.dtype)

        pred_mass = (mu.float() * mask_f.float()).sum(dim=1)
        target_mass = (target.float() * mask_f.float()).sum(dim=1).detach()

        pred_mass = pred_mass.clamp_min(eps)
        target_mass = target_mass.clamp_min(eps)

        return (torch.log(pred_mass) - torch.log(target_mass)).pow(2)

    def bias_log_regularization(
            self,
            *,
            extras: dict,
            mask: torch.Tensor,
            eps: float,
    ) -> torch.Tensor:
        mask_b = mask.bool()
        regs = []

        # 1. Multiplicative Bias (b) Penalty
        # PREVENT THE TRAP: We use the 'control' signal (stored in log_b)
        # so that exact zeros don't trigger a log(1e-8) explosion.
        control = extras.get("control")

        if control is not None:
            regs.append((control[mask_b] ** 2).mean())
        else:
            # Fallback only for older heads that don't output the control signal
            exp_b = self._get_b_multiplier(extras=extras, mask=mask)
            if exp_b is not None:
                log_exp_b = torch.log(exp_b.float().clamp_min(eps))
                regs.append((log_exp_b[mask_b] ** 2).mean())

        # 2. Additive Background (lambda_bg) Penalty
        lambda_bg = extras.get("lambda_bg")
        if lambda_bg is not None:
            #regs.append((lambda_bg.float() ** 2).mean())
            regs.append((lambda_bg.float().abs()).mean())

        # Backward compatibility for old heads (if still used)
        beta = extras.get("beta_per_position")
        if beta is not None:
            log_beta = torch.log(beta.float().clamp_min(eps))
            regs.append((log_beta[mask_b] ** 2).mean())

        if len(regs) == 0:
            return torch.zeros((), device=mask.device)

        return torch.stack(regs).sum()

    def _build_loss_terms(
            self,
            *,
            mu: torch.Tensor,
            p: torch.Tensor,
            phi: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor,
            extras: dict,
    ) -> dict[str, torch.Tensor]:
        raw_nll_per_sample = self._compute_raw_nll_per_sample(
            mu=mu,
            p=p,
            phi=phi,
            target=target,
            mask=mask,
        )

        nll_per_sample = self._soft_cap_loss_per_sample(raw_nll_per_sample)

        # Apply Bias Regularization
        reg_weight = float(self._cfg("loss.lambda_bias_log_l2", 0.0))
        extra_loss = torch.zeros((), device=mu.device)

        if reg_weight > 0.0:
            extra_loss = reg_weight * self.bias_log_regularization(
                extras=extras,
                mask=mask,
                eps=float(self._cfg("loss.eps", 1e-8))
            )

        return {
            "raw_nll_per_sample": raw_nll_per_sample,
            "nll_per_sample": nll_per_sample,
            "loss_per_sample": nll_per_sample,
            "extra_loss": extra_loss,
        }

    # ============================================================
    # Logging helpers
    # ============================================================

    def _finite_mean(self, x: torch.Tensor) -> torch.Tensor | None:
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
        value: torch.Tensor | float,
        *,
        batch_size: int,
        prog_bar: bool = False,
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
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            logger=True,
            batch_size=batch_size,
            sync_dist=self.log_sync_dist,
        )

    def _masked_pcc_per_sample(
        self,
        *,
        pred: torch.Tensor,
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
        pcc = torch.where(valid, pcc, torch.nan)

        return pcc

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
        pred: torch.Tensor,
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

    def _log_p_diagnostics(
        self,
        *,
        stage: str,
        p: torch.Tensor,
        mask: torch.Tensor,
        dataset_ids: torch.Tensor,
        batch_size: int,
    ) -> None:
        mask_b = mask.bool()
        mask_f = mask_b.float()

        with torch.no_grad():
            if p.ndim == 2 and p.shape == mask.shape:
                p_valid = p.detach().float()[mask_b]

                p_per_sample = (
                    (p.detach().float() * mask_f).sum(dim=1)
                    / mask_f.sum(dim=1).clamp_min(1.0)
                )
            else:
                p_flat = p.detach().float().reshape(p.shape[0], -1)
                p_per_sample = p_flat.mean(dim=1)
                p_valid = p_per_sample

            p_mean = self._finite_mean(p_valid)

            if p_mean is not None:
                self._log_scalar(
                    f"{stage}_tweedie_p",
                    p_mean,
                    batch_size=batch_size,
                )

            if stage == "val":
                self._log_vector_global_and_by_dataset(
                    stage=stage,
                    name="tweedie_p",
                    values=p_per_sample,
                    dataset_ids=dataset_ids,
                    batch_size=batch_size,
                    log_global=False,
                    log_by_dataset=True,
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

        log_b = extras.get("log_b")

        if log_b is not None:
            return torch.exp(log_b)

        # Ambiguous fallback: some older versions used "b".
        # If it is strictly positive, treat as multiplier.
        # Otherwise, treat as log-bias.
        b_raw = extras.get("b")

        if b_raw is None:
            return None

        mask_b = mask.bool()

        with torch.no_grad():
            valid = b_raw.detach().float()[mask_b]
            finite = valid[torch.isfinite(valid)]

            if finite.numel() == 0:
                return None

            looks_positive_multiplier = bool((finite > 0).all().item())

        if looks_positive_multiplier:
            return b_raw

        return torch.exp(b_raw)

    def _make_bio_only_mu(
        self,
        *,
        extras: dict,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=target.dtype)

        bio_q = extras.get("bio_q")

        if bio_q is None:
            L_queue = extras.get("L_queue")
            b = self._get_b_multiplier(extras=extras, mask=mask_b)

            if L_queue is None or b is None:
                return None

            bio_q = L_queue.detach().float().clamp_min(0.0) * b.detach().float().clamp_min(0.0)

        total_mass = extras.get("total_mass")

        if total_mass is None:
            total_mass = (
                target.float().clamp_min(0.0) * mask_f.float()
            ).sum(dim=1, keepdim=True).detach()
        else:
            total_mass = total_mass.detach().float()

            if total_mass.ndim == 1:
                total_mass = total_mass.reshape(-1, 1)

        eps = float(self._cfg("loss.eps", 1e-8))

        bio_q = bio_q.detach().float().clamp_min(0.0) * mask_f.float()
        bio_mass = bio_q.sum(dim=1, keepdim=True).clamp_min(eps)

        mu_bio_only = total_mass * bio_q / bio_mass
        mu_bio_only = mu_bio_only * mask_f.float()

        return mu_bio_only

    # ============================================================
    # Main validation diagnostics
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

    def _log_bias_leakage_diagnostics(
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

        b = self._get_b_multiplier(extras=extras, mask=mask_b)
        R_shape = extras.get("R_shape")
        additive_noise = extras.get("additive_noise")
        L_queue = extras.get("L_queue")
        bio_q = extras.get("bio_q")
        lambda_bg = extras.get("lambda_bg")

        # ------------------------------------------------------------
        # Leakage: does b itself track y?
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Leakage: does R itself track y?
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Bio-only mu vs full mu
        # ------------------------------------------------------------
        mu_bio_only = self._make_bio_only_mu(
            extras=extras,
            target=target,
            mask=mask_b,
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

            mu_pcc = self._masked_pcc_per_sample(
                pred=mu,
                target=target,
                mask=mask_b,
            )

            bio_pcc = self._masked_pcc_per_sample(
                pred=mu_bio_only,
                target=target,
                mask=mask_b,
            )

            if mu_pcc is not None and bio_pcc is not None:
                delta = mu_pcc - bio_pcc

                self._log_vector_global_and_by_dataset(
                    stage="val",
                    name="mu_full_minus_bio_only_pcc",
                    values=delta,
                    dataset_ids=dataset_ids,
                    batch_size=batch_size,
                    log_global=True,
                    log_by_dataset=True,
                )

        # ------------------------------------------------------------
        # b/R alignment with L_queue
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Background fraction
        # ------------------------------------------------------------
        if bio_q is None and L_queue is not None and b is not None:
            bio_q = L_queue.detach().float().clamp_min(0.0) * b.detach().float().clamp_min(0.0)

        if bio_q is not None and additive_noise is not None:
            bio_mass = (
                bio_q.detach().float().clamp_min(0.0) * mask_f
            ).sum(dim=1)

            bg_mass = (
                additive_noise.detach().float().clamp_min(0.0) * mask_f
            ).sum(dim=1)

            eps = float(self._cfg("loss.eps", 1e-8))
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

        # ------------------------------------------------------------
        # R support fraction. Useful especially if R uses entmax/sparsemax.
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # b strength
        # ------------------------------------------------------------
        log_b = extras.get("log_b")

        if log_b is None and b is not None:
            log_b = torch.log(b.detach().float().clamp_min(float(self._cfg("loss.eps", 1e-8))))

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

        # ------------------------------------------------------------
        # Lambda
        # ------------------------------------------------------------
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

    def _normalize_css_positions(
        self,
        css_i: Any,
        L: int,
    ) -> list[int]:
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
            return [
                int(i)
                for i in torch.nonzero(arr_b, as_tuple=False).reshape(-1).tolist()
            ]

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

    def _peak_budget(
        self,
        L: int,
    ) -> int:
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
        components = {
            "mu": mu,
            "L_queue": L_queue,
        }

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
                    c
                    for c in css_positions
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
        if x is None:
            return None

        if not torch.is_tensor(x):
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

        if x.ndim >= 3:
            return None

        return None

    def _log_example_profile_plot(
            self,
            *,
            ids: list,
            dataset_ids: torch.Tensor,
            target: torch.Tensor,
            mu: torch.Tensor,
            phi: torch.Tensor,
            p: torch.Tensor,
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

        # -------------------------------------------------------------------
        # INTELLECTUAL SPARRING: THE "APPLES TO APPLES" SCANNER
        # To truly see dataset bias adaptation, we must find the exact SAME
        # transcript appearing in TWO DIFFERENT datasets within this batch.
        # -------------------------------------------------------------------
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

        # Fallback: If no perfect match exists, just grab two different datasets
        if not found_perfect_match:
            for i in range(1, B):
                if dataset_ids[i] != dataset_ids[0]:
                    idx2 = i
                    break

        indices_to_plot = [idx1, idx2]

        b_multiplier = self._get_b_multiplier(extras=extras, mask=mask)
        mu_bio_only = self._make_bio_only_mu(extras=extras, target=target, mask=mask)

        # Create a 4x2 grid. Share the X-axis per column so zooming remains locked per transcript.
        fig, axes = plt.subplots(
            4, 2, figsize=(24, 11), sharex="col", gridspec_kw={"height_ratios": [1.4, 1.0, 1.0, 1.0]}
        )

        for col, sample_idx in enumerate(indices_to_plot):
            mask_i = mask[sample_idx].detach().bool().cpu()
            L = int(mask_i.sum().item())

            if L < 2:
                continue

            y_np = self._seq_to_np_for_plot(target, sample_idx=sample_idx, mask_i=mask_i, L=L)
            mu_np = self._seq_to_np_for_plot(mu, sample_idx=sample_idx, mask_i=mask_i, L=L)
            mu_bio_only_np = self._seq_to_np_for_plot(mu_bio_only, sample_idx=sample_idx, mask_i=mask_i, L=L)

            L_queue_np = self._seq_to_np_for_plot(extras.get("L_queue"), sample_idx=sample_idx, mask_i=mask_i, L=L)
            bio_q_np = self._seq_to_np_for_plot(extras.get("bio_q"), sample_idx=sample_idx, mask_i=mask_i, L=L)

            additive_noise_np = self._seq_to_np_for_plot(extras.get("additive_noise"), sample_idx=sample_idx,
                                                         mask_i=mask_i, L=L)
            R_shape_np = self._seq_to_np_for_plot(extras.get("R_shape"), sample_idx=sample_idx, mask_i=mask_i, L=L)

            exp_b_np = self._seq_to_np_for_plot(b_multiplier, sample_idx=sample_idx, mask_i=mask_i, L=L)
            log_b_np = self._seq_to_np_for_plot(extras.get("log_b"), sample_idx=sample_idx, mask_i=mask_i, L=L)

            phi_np = self._seq_to_np_for_plot(phi, sample_idx=sample_idx, mask_i=mask_i, L=L)
            p_np = self._seq_to_np_for_plot(p, sample_idx=sample_idx, mask_i=mask_i, L=L)

            x_axis = torch.arange(L).cpu().numpy()

            dataset_id = int(dataset_ids[sample_idx].detach().cpu().item())
            dataset_name = self._dataset_name(dataset_id)
            transcript_id = ids[sample_idx]

            # Safe mean calculation regardless of Tweedie tensor shape
            p_val_tensor = p[sample_idx] if p.ndim > 1 else p
            p_value = float(p_val_tensor.detach().float().mean().cpu())

            # Title flags if this is an apples-to-apples comparison
            match_status = "PERFECT MATCH" if found_perfect_match else "MISMATCHED TRANSCRIPT"
            axes[0, col].set_title(
                f"[{match_status}]\nDataset: {dataset_name} | ID: {transcript_id} | L={L} | p={p_value:.4f}")

            if y_np is not None:
                axes[0, col].plot(x_axis, y_np, label="target y", linewidth=1.0)
            if mu_np is not None:
                axes[0, col].plot(x_axis, mu_np, label="mu full", linewidth=1.0)
            if mu_bio_only_np is not None:
                axes[0, col].plot(x_axis, mu_bio_only_np, label="mu bio-only", linewidth=1.0)

            if col == 0: axes[0, col].set_ylabel("profile")
            axes[0, col].grid(True, alpha=0.3)
            axes[0, col].legend(loc="upper right")

            if L_queue_np is not None:
                axes[1, col].plot(x_axis, L_queue_np, label="L_queue", linewidth=1.0)
            if bio_q_np is not None:
                axes[1, col].plot(x_axis, bio_q_np, label="bio_q = L*b", linewidth=1.0)

            if col == 0: axes[1, col].set_ylabel("biology")
            axes[1, col].grid(True, alpha=0.3)
            axes[1, col].legend(loc="upper right")

            if R_shape_np is not None:
                axes[2, col].plot(x_axis, R_shape_np, label="R_shape", linewidth=1.0)
            if additive_noise_np is not None:
                axes[2, col].plot(x_axis, additive_noise_np, label="additive_noise", linewidth=1.0)

            if exp_b_np is not None:
                ax2 = axes[2, col].twinx()
                ax2.plot(x_axis, exp_b_np, label="exp_b", linewidth=0.8, linestyle="--", color='purple', alpha=0.7)
                if col == 1: ax2.set_ylabel("exp_b")
                ax2.legend(loc="upper left")

            if col == 0: axes[2, col].set_ylabel("residual")
            axes[2, col].grid(True, alpha=0.3)
            axes[2, col].legend(loc="upper right")

            if phi_np is not None:
                axes[3, col].plot(x_axis, phi_np, label="phi", linewidth=0.8)
            if p_np is not None:
                ax3 = axes[3, col].twinx()
                ax3.plot(x_axis, p_np, label="Tweedie p", linewidth=0.8, linestyle="--", color='purple', alpha=0.7)
                if col == 1: ax3.set_ylabel("p")
                ax3.legend(loc="upper left")
            if log_b_np is not None:
                axes[3, col].plot(x_axis, log_b_np, label="log_b", linewidth=0.8)

            if col == 0: axes[3, col].set_ylabel("value")
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
            p
            for p in biological_model.parameters()
            if p.requires_grad
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
            return scalar_loss.detach()

        max_datasets = int(
            self._cfg(
                "optim.pcgrad_max_datasets_per_step",
                len(dataset_losses),
            )
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

        mu, p, phi, extras = self.model(
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
            "p": p,
            "phi": phi,
            "extras": extras,
        }

    def training_step(self, batch, batch_idx):
        out = self._forward_batch(batch)
        batch_size = int(out["target"].shape[0])

        loss_terms = self._build_loss_terms(
            mu=out["mu"],
            p=out["p"],
            phi=out["phi"],
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

        self._log_p_diagnostics(
            stage="train",
            p=out["p"],
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
            p=out["p"],
            phi=out["phi"],
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

        self._log_p_diagnostics(
            stage="val",
            p=out["p"],
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

        self._log_bias_leakage_diagnostics(
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
            phi=out["phi"],
            p=out["p"],
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

        output = {
            "ids": to_cpu(out["ids"]),
            "dataset_id": to_cpu(out["dataset_ids"]),
            "lengths": to_cpu(out["lengths"]),
            "mask": to_cpu(out["mask"]),
            "css": to_cpu(out["css"]),
            "y": to_cpu(out["target"]),
            "mu_obs": to_cpu(out["mu"]),
            "phi": to_cpu(out["phi"]),
            "tweedie_p": to_cpu(out["p"]),
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