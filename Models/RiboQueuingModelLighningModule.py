from __future__ import annotations

import math
from typing import Any

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from Models.utils.PCGrad_utils import (
    assign_flat_grads,
    pcgrad_combine,
    per_dataset_losses,
)
from Models.utils.dirichlet_multinomial_profile_loss import DirichletMultinomialProfileLoss
from Models.utils.multinomial_profile_loss import MultinomialProfileLoss
from Models.utils.ribo_lightning_helpers import flatten_current_grads


class RiboQueuingModelLightningModule(pl.LightningModule):
    """
    Lightning wrapper for the multiplicative allocation-bias queueing model.

    Expected torch model output:
        - (mu, kappa_input, extras)
        - or (mu, profile_aux, kappa_input, extras)

    Current model semantics:
        Biological branch:
            w_bio_i = biological allocation
            h_bio_i = J_t * T_t * w_bio_i
            L_bio_i = exp(h_bio_i) - 1

        Dataset/protocol observation branch:
            b_raw_i = keep_gate_i * amplitude_i
            b_eff_i = b_raw_i / sum_j w_bio_j b_raw_j
            w_obs_i = w_bio_i * b_eff_i

        Observed support:
            h_obs_i = J_t * T_t * w_obs_i
            L_obs_i = exp(h_obs_i) - 1

        Profile mean:
            mu_i = total_mass * L_obs_i / sum_j L_obs_j

    Active regularizers:
        1. obs_bias_log_l2:
            mixed-position penalty on log(b_eff)^2.

        2. obs_gate_mean_floor:
            preferred gate regularizer for sparse zeros. It prevents global gate
            collapse but allows individual positions to close:

                ReLU(g_min - sum_i p_mix_i keep_prob_i)^2

        3. obs_gate_open:
            optional legacy per-position open penalty on -log(keep_prob).
            Keep this at 0.0 if you want sparse gate closures.

        4. optional kappa_log_l2.
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

        self.loss_type = str(
            self._cfg("loss.type", "dirichlet_multinomial_profile")
        ).lower()
        self.loss_fn = self._build_profile_loss()

        enc = dataset_encoding or {}
        self.dataset_id_to_name = {int(v): str(k) for k, v in enc.items()}

        self.dataset_balanced_loss = bool(
            self._cfg("loss.dataset_balanced_loss", False)
        )

        self.use_pcgrad = bool(self._cfg("optim.use_pcgrad", False))
        self.automatic_optimization = not self.use_pcgrad

        self.log_sync_dist = bool(self._cfg("trainer.sync_dist_logs", False))
        self._val_plot_logged_this_epoch = False

        if self.dataset_balanced_loss:
            print("Loss aggregation: dataset-balanced mean of per-dataset losses.")
        else:
            print("Loss aggregation: ordinary per-sample mean.")

        if self.use_pcgrad:
            print("Training is using PCGrad.")

    # ============================================================
    # Config / names
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

    @staticmethod
    def _component(extras: dict, *keys: str):
        for key in keys:
            value = extras.get(key)
            if value is not None:
                return value
        return None

    # ============================================================
    # Loss
    # ============================================================

    def _build_profile_loss(self) -> nn.Module:
        eps = float(self._cfg("loss.eps", 1.0e-8))
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
            fixed_kappa = None if fixed_kappa is None else float(fixed_kappa)

            return DirichletMultinomialProfileLoss(
                eps=eps,
                kappa=fixed_kappa,
                kappa_min=float(self._cfg("loss.kappa_min", 1.0e-2)),
                kappa_max=float(self._cfg("loss.kappa_max", 1.0e5)),
                normalize_by_total=normalize_by_total,
                include_multinomial_constant=bool(
                    self._cfg("loss.include_multinomial_constant", False)
                ),
                pool_position_kappa=str(self._cfg("loss.pool_position_kappa", "mean")),
            )

        raise ValueError(
            f"Unsupported loss.type={self.loss_type!r}. Use "
            "'multinomial_profile' or 'dirichlet_multinomial_profile'."
        )

    def _compute_kappa(
        self,
        *,
        kappa_input: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor | float | None:
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

        if source in {
            "model_inverse",
            "inverse",
            "phi_inverse",
            "dispersion_inverse",
        }:
            scale = float(self._cfg("loss.kappa_scale", 100.0))
            min_disp = float(self._cfg("loss.kappa_input_min", 1.0e-4))
            return scale / x.clamp_min(min_disp)

        raise ValueError(
            f"Unsupported loss.kappa_source={source!r}. Use 'fixed', "
            "'model_direct', or 'model_inverse'."
        )

    def _profile_nll_per_sample(
        self,
        *,
        mu: torch.Tensor,
        kappa_input: torch.Tensor | None,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=mu.device.type, enabled=False):
            if self.loss_type in {"multinomial", "multinomial_profile", "profile_ce"}:
                loss = self.loss_fn(
                    mu=mu.float(),
                    y_true=target.float(),
                    mask=mask.bool(),
                    return_per_sample=True,
                )
            else:
                loss = self.loss_fn(
                    mu=mu.float(),
                    kappa=self._compute_kappa(
                        kappa_input=kappa_input,
                        mask=mask,
                    ),
                    y_true=target.float(),
                    mask=mask.bool(),
                    return_per_sample=True,
                )

        cap = float(self._cfg("loss.nll_soft_cap", -1.0))
        if cap > 0.0:
            loss = cap * torch.log1p(loss / cap)

        return loss

    def _scalar_profile_loss(
        self,
        *,
        loss_per_sample: torch.Tensor,
        dataset_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self.dataset_balanced_loss:
            return loss_per_sample.mean()

        dataset_losses = per_dataset_losses(
            loss_per_sample=loss_per_sample,
            dataset_ids=dataset_ids,
        )

        if len(dataset_losses) == 0:
            return loss_per_sample.mean()

        return torch.stack(dataset_losses).mean()

    # ============================================================
    # Regularization
    # ============================================================

    def _mixed_position_weights_from_w_bio(
        self,
        *,
        extras: dict,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mixed regularization weights:

            p_mix = (1 - alpha) * w_bio + alpha * uniform

        This prevents unregularized behavior at positions where w_bio is tiny.
        """
        eps = float(self._cfg("loss.eps", 1.0e-8))
        alpha = float(self._cfg("loss.obs_bias_uniform_weight", 0.05))
        alpha = min(max(alpha, 0.0), 1.0)

        mask_b = mask.bool()
        mask_f = mask_b.float()

        uniform = mask_f / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        w_bio = self._component(extras, "w_bio", "w_prob")

        if torch.is_tensor(w_bio) and w_bio.shape == mask.shape:
            w = w_bio.detach().float().clamp_min(0.0) * mask_f
            w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
        else:
            w = uniform

        weights = (1.0 - alpha) * w + alpha * uniform
        weights = weights * mask_f
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)

        return weights

    def _obs_bias_regularization(
        self,
        *,
        extras: dict,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalizes effective multiplicative bias away from neutral:

            sum_i p_mix_i * log(b_eff_i)^2

        Neutral state:

            b_eff_i = 1
        """
        device = mask.device
        eps = float(self._cfg("loss.eps", 1.0e-8))

        b_eff = self._component(extras, "obs_bias_effective", "b_effective", "b_shape")

        if not torch.is_tensor(b_eff) or b_eff.shape != mask.shape:
            return torch.zeros((), device=device)

        mask_f = mask.bool().float()
        weights = self._mixed_position_weights_from_w_bio(
            extras=extras,
            mask=mask,
        ).detach()

        log_b = torch.log(b_eff.float().clamp_min(eps))
        reg_per_sample = (weights * log_b.pow(2) * mask_f).sum(dim=1)

        return reg_per_sample.mean()

    def _obs_gate_open_regularization(
        self,
        *,
        extras: dict,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optional legacy per-position gate-open penalty:

            sum_i p_mix_i * [-log(keep_prob_i)]

        This discourages every individual closure. If you want sparse gate zeros,
        keep loss.lambda_obs_gate_open = 0.0 and use
        loss.lambda_obs_gate_mean_floor instead.
        """
        device = mask.device
        eps = float(self._cfg("loss.eps", 1.0e-8))

        keep_prob = self._component(extras, "obs_bias_keep_prob", "keep_prob")

        if not torch.is_tensor(keep_prob) or keep_prob.shape != mask.shape:
            return torch.zeros((), device=device)

        mask_f = mask.bool().float()
        weights = self._mixed_position_weights_from_w_bio(
            extras=extras,
            mask=mask,
        ).detach()

        g = keep_prob.float().clamp(min=eps, max=1.0)
        reg_per_sample = (weights * (-torch.log(g)) * mask_f).sum(dim=1)

        return reg_per_sample.mean()

    def _obs_gate_mean_floor_regularization(
        self,
        *,
        extras: dict,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Preferred gate regularizer for sparse zeros.

        It penalizes only transcripts whose weighted mean keep probability falls
        below a configured floor:

            mean_t ReLU(g_min - sum_i p_mix_i keep_prob_i)^2

        This prevents global gate collapse while allowing individual positions
        to close.
        """
        device = mask.device

        keep_prob = self._component(extras, "obs_bias_keep_prob", "keep_prob")

        if not torch.is_tensor(keep_prob) or keep_prob.shape != mask.shape:
            return torch.zeros((), device=device)

        weights = self._mixed_position_weights_from_w_bio(
            extras=extras,
            mask=mask,
        ).detach()

        keep_prob = keep_prob.float().clamp(0.0, 1.0)
        mean_open = (weights * keep_prob).sum(dim=1)

        min_open = float(self._cfg("loss.obs_gate_min_open", 0.95))
        min_open = min(max(min_open, 0.0), 1.0)

        return torch.relu(min_open - mean_open).pow(2).mean()

    def _per_sample_mean(self, x: torch.Tensor | float | None, mask: torch.Tensor):
        if x is None:
            return None

        if not torch.is_tensor(x):
            return torch.full((mask.shape[0],), float(x), device=mask.device)

        x = x.float()
        mask_f = mask.bool().float()

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

    def _regularization_terms(
        self,
        *,
        extras: dict,
        kappa_input: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = mask.device

        terms: dict[str, torch.Tensor] = {}

        lambda_obs_bias = float(self._cfg("loss.lambda_obs_bias_log_l2", 0.0))
        obs_bias_reg_raw = self._obs_bias_regularization(
            extras=extras,
            mask=mask,
        )
        terms["obs_bias_log_l2"] = lambda_obs_bias * obs_bias_reg_raw

        # Legacy per-position open penalty. Set to 0.0 for sparse zeros.
        lambda_obs_gate = float(self._cfg("loss.lambda_obs_gate_open", 0.0))
        obs_gate_reg_raw = self._obs_gate_open_regularization(
            extras=extras,
            mask=mask,
        )
        terms["obs_gate_open"] = lambda_obs_gate * obs_gate_reg_raw

        # New mean-open floor penalty. This is the recommended gate regularizer.
        lambda_gate_floor = float(
            self._cfg("loss.lambda_obs_gate_mean_floor", 0.0)
        )
        gate_floor_reg_raw = self._obs_gate_mean_floor_regularization(
            extras=extras,
            mask=mask,
        )
        terms["obs_gate_mean_floor"] = lambda_gate_floor * gate_floor_reg_raw

        lambda_kappa = float(self._cfg("loss.lambda_kappa_log_l2", 0.0))
        kappa_reg = torch.zeros((), device=device)

        if lambda_kappa > 0.0:
            kappa = self._compute_kappa(kappa_input=kappa_input, mask=mask)

            if kappa is not None:
                if not torch.is_tensor(kappa):
                    kappa_t = torch.full((mask.shape[0],), float(kappa), device=device)
                else:
                    kappa_t = self._per_sample_mean(kappa, mask)

                kappa_ref = float(
                    self._cfg("loss.kappa_reference", self._cfg("loss.kappa", 500.0))
                )

                log_ref = torch.log(
                    torch.tensor(kappa_ref, device=device, dtype=torch.float32)
                )

                kappa_reg = (
                    torch.log(kappa_t.float().clamp_min(1.0e-8)) - log_ref
                ).pow(2).mean()

        terms["kappa_log_l2"] = lambda_kappa * kappa_reg

        total = torch.zeros((), device=device)
        for value in terms.values():
            total = total + value

        terms["total"] = total

        return terms

    def _loss_terms(self, out: dict) -> dict[str, torch.Tensor]:
        nll = self._profile_nll_per_sample(
            mu=out["mu"],
            kappa_input=out["kappa_input"],
            target=out["target"],
            mask=out["mask"],
        )

        profile_loss = self._scalar_profile_loss(
            loss_per_sample=nll,
            dataset_ids=out["dataset_ids"],
        )

        reg_terms = self._regularization_terms(
            extras=out["extras"],
            kappa_input=out["kappa_input"],
            mask=out["mask"],
        )

        extra_loss = reg_terms["total"]
        total_loss = profile_loss + extra_loss

        return {
            "nll_per_sample": nll,
            "profile_loss": profile_loss,
            "extra_loss": extra_loss,
            "total_loss": total_loss,
            "reg_obs_bias_log_l2": reg_terms["obs_bias_log_l2"],
            "reg_obs_gate_open": reg_terms["obs_gate_open"],
            "reg_obs_gate_mean_floor": reg_terms["obs_gate_mean_floor"],
            "reg_kappa_log_l2": reg_terms["kappa_log_l2"],
        }

    # ============================================================
    # Logging helpers
    # ============================================================

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
            batch_size=max(int(batch_size), 1),
            sync_dist=self.log_sync_dist,
        )

    def _finite_mean(self, x: torch.Tensor | None):
        if x is None:
            return None

        x = x.detach().float().reshape(-1)
        finite = torch.isfinite(x)

        if not finite.any():
            return None

        return x[finite].mean()

    def _masked_quantile_per_sample(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        q: float,
    ) -> torch.Tensor:
        if x.shape != mask.shape:
            raise ValueError(
                f"x and mask must have same shape, got "
                f"x={tuple(x.shape)}, mask={tuple(mask.shape)}."
            )

        q = float(min(max(q, 0.0), 1.0))
        mask_b = mask.bool()
        values = []

        for i in range(x.shape[0]):
            xi = x[i][mask_b[i]].detach().float()
            if xi.numel() == 0:
                values.append(torch.full((), float("nan"), device=x.device))
            else:
                values.append(torch.quantile(xi, q))

        return torch.stack(values, dim=0)

    def _log_vector(
        self,
        *,
        stage: str,
        name: str,
        values: torch.Tensor | None,
        dataset_ids: torch.Tensor,
        batch_size: int,
        by_dataset: bool = True,
        prog_bar: bool = False,
    ) -> None:
        if values is None:
            return

        values = values.detach().float()

        self._log_scalar(
            f"{stage}_{name}",
            self._finite_mean(values),
            batch_size=batch_size,
            prog_bar=prog_bar,
        )

        if not by_dataset:
            return

        for ds_id in torch.unique(dataset_ids).detach().cpu().tolist():
            ds_id = int(ds_id)
            ds_mask = dataset_ids == ds_id
            ds_mean = self._finite_mean(values[ds_mask])

            if ds_mean is not None:
                self._log_scalar(
                    f"{stage}_{name}_by_dataset/{self._dataset_name(ds_id)}",
                    ds_mean,
                    batch_size=int(ds_mask.sum().item()),
                )

    # ============================================================
    # PCC diagnostics
    # ============================================================

    def _masked_pcc(
        self,
        pred: torch.Tensor | None,
        target: torch.Tensor,
        mask: torch.Tensor,
    ):
        if pred is None or pred.shape != target.shape:
            return None

        eps = float(self._cfg("loss.eps", 1.0e-8))

        mask_f = mask.bool().float()
        valid_count = mask_f.sum(dim=1).clamp_min(1.0)

        x = torch.nan_to_num(
            pred.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y = torch.nan_to_num(
            target.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        x_mean = (x * mask_f).sum(dim=1) / valid_count
        y_mean = (y * mask_f).sum(dim=1) / valid_count

        xc = (x - x_mean.unsqueeze(1)) * mask_f
        yc = (y - y_mean.unsqueeze(1)) * mask_f

        cov = (xc * yc).sum(dim=1)
        x_var = xc.pow(2).sum(dim=1)
        y_var = yc.pow(2).sum(dim=1)

        pcc = cov / torch.sqrt(x_var * y_var).clamp_min(eps)
        valid = (mask_f.sum(dim=1) >= 2) & torch.isfinite(pcc)

        return torch.where(valid, pcc, torch.nan)

    def _log_pcc(
        self,
        *,
        stage: str,
        name: str,
        pred: torch.Tensor | None,
        target: torch.Tensor,
        mask: torch.Tensor,
        dataset_ids: torch.Tensor,
        batch_size: int,
        prog_bar: bool = False,
    ):
        pcc = self._masked_pcc(pred, target, mask)

        self._log_vector(
            stage=stage,
            name=f"{name}_pcc",
            values=pcc,
            dataset_ids=dataset_ids,
            batch_size=batch_size,
            by_dataset=(stage == "val"),
            prog_bar=prog_bar,
        )

        return pcc

    def _total_mass(
        self,
        extras: dict,
        target: torch.Tensor,
        mask: torch.Tensor,
    ):
        total_mass = extras.get("total_mass")

        if total_mass is None:
            return (
                target.float().clamp_min(0.0) * mask.bool().float()
            ).sum(dim=1, keepdim=True).detach()

        total_mass = total_mass.detach().float()
        return total_mass.reshape(-1, 1) if total_mass.ndim == 1 else total_mass

    def _mu_from_support(self, support: torch.Tensor | None, out: dict):
        if support is None or support.shape != out["target"].shape:
            return None

        eps = float(self._cfg("loss.eps", 1.0e-8))
        mask_f = out["mask"].bool().float()
        q = support.detach().float().clamp_min(0.0) * mask_f
        total_mass = self._total_mass(out["extras"], out["target"], out["mask"])

        return total_mass * q / q.sum(dim=1, keepdim=True).clamp_min(eps) * mask_f

    # ============================================================
    # CSS / peak-recall diagnostics
    # ============================================================

    def _flatten_css_values(self, x: Any) -> list[Any]:
        if x is None:
            return []

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()

        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return self._flatten_css_values(x.item())

            out: list[Any] = []
            for v in x.reshape(-1).tolist():
                out.extend(self._flatten_css_values(v))
            return out

        if isinstance(x, (list, tuple)):
            out: list[Any] = []
            for v in x:
                out.extend(self._flatten_css_values(v))
            return out

        return [x]

    def _css_positions_from_item(self, css_item: Any, valid_len: int) -> list[int]:
        if css_item is None or valid_len <= 0:
            return []

        if torch.is_tensor(css_item):
            css_item = css_item.detach().cpu().numpy()

        try:
            arr = np.asarray(css_item)
        except Exception:
            arr = None

        if arr is not None and arr.size > 0:
            if arr.ndim == 1 and arr.shape[0] == valid_len and arr.dtype == bool:
                return np.flatnonzero(arr).astype(int).tolist()

            if arr.ndim == 1 and arr.shape[0] == valid_len:
                try:
                    arr_float = arr.astype(float)
                    finite = np.isfinite(arr_float)
                    unique = set(np.unique(arr_float[finite]).tolist())

                    if unique.issubset({0.0, 1.0}):
                        return np.flatnonzero(arr_float > 0.5).astype(int).tolist()
                except Exception:
                    pass

        raw_values = self._flatten_css_values(css_item)
        positions: list[int] = []

        for v in raw_values:
            try:
                if v is None:
                    continue

                if isinstance(v, str):
                    s = v.strip()

                    if s in {"", "nan", "None", "null", "[]"}:
                        continue

                    if s.startswith("[") and s.endswith("]"):
                        s = s.strip("[]")
                        for piece in s.split(","):
                            piece = piece.strip()
                            if piece:
                                raw_values.append(piece)
                        continue

                    v = s

                fv = float(v)

                if math.isnan(fv) or math.isinf(fv):
                    continue

                p = int(round(fv))

            except Exception:
                continue

            if 0 <= p < valid_len:
                positions.append(p)

        return sorted(set(positions))

    def _css_z_thresholds(self) -> list[float]:
        raw = self._cfg("metrics.css_z_thresholds", [1.0, 2.0, 3.0, 4.0, 5.0])

        if isinstance(raw, (int, float)):
            return [float(raw)]

        return [float(x) for x in list(raw)]

    @staticmethod
    def _z_label(tau: float) -> str:
        tau = float(tau)
        return f"z{int(tau)}" if tau.is_integer() else "z" + str(tau).replace(".", "p")

    def _zscore_1d(
        self,
        score: torch.Tensor,
        valid_len: int,
    ) -> torch.Tensor | None:
        if valid_len <= 1:
            return None

        eps = float(self._cfg("metrics.css_z_eps", 1.0e-6))
        x = score.detach().float()[:valid_len].cpu()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        std = x.std(unbiased=False)

        if not torch.isfinite(std) or float(std.item()) < eps:
            return None

        return (x - x.mean()) / std.clamp_min(eps)

    @staticmethod
    def _peak_indices_from_z(
        z: torch.Tensor,
        z_threshold: float,
    ) -> list[int]:
        L = int(z.numel())
        if L <= 0:
            return []

        z = torch.nan_to_num(
            z.float(),
            nan=-float("inf"),
            posinf=float("inf"),
            neginf=-float("inf"),
        )

        left = torch.empty_like(z)
        right = torch.empty_like(z)

        left[0] = -float("inf")
        left[1:] = z[:-1]

        right[-1] = -float("inf")
        right[:-1] = z[1:]

        peak_mask = (z >= float(z_threshold)) & (z >= left) & (z >= right)

        return (
            torch.nonzero(peak_mask, as_tuple=False)
            .reshape(-1)
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )

    def _css_metrics_for_scores(
        self,
        *,
        scores: torch.Tensor | None,
        css_items: list[Any],
        mask: torch.Tensor,
    ) -> dict[float, dict[str, torch.Tensor]] | None:
        if scores is None or not torch.is_tensor(scores) or scores.shape != mask.shape:
            return None

        device = mask.device
        B = mask.shape[0]
        tol = int(self._cfg("metrics.css_recall_tolerance", 1))
        thresholds = self._css_z_thresholds()

        out: dict[float, dict[str, torch.Tensor]] = {}

        for tau in thresholds:
            out[float(tau)] = {
                "recall": torch.full((B,), float("nan"), device=device),
                "precision": torch.full((B,), float("nan"), device=device),
                "f1": torch.full((B,), float("nan"), device=device),
                "css_site_count": torch.zeros((B,), dtype=torch.float32, device=device),
                "css_hit_count": torch.zeros((B,), dtype=torch.float32, device=device),
                "peak_count": torch.zeros((B,), dtype=torch.float32, device=device),
                "peak_hit_count": torch.zeros((B,), dtype=torch.float32, device=device),
            }

        for i in range(B):
            valid_len = int(mask[i].detach().bool().sum().item())
            if valid_len <= 1:
                continue

            css_pos = self._css_positions_from_item(css_items[i], valid_len)
            css_n = len(css_pos)
            z = self._zscore_1d(scores[i], valid_len)

            if z is None:
                continue

            for tau in thresholds:
                tau = float(tau)
                metrics_tau = out[tau]
                metrics_tau["css_site_count"][i] = float(css_n)

                peaks = self._peak_indices_from_z(z, tau)
                peak_n = len(peaks)
                metrics_tau["peak_count"][i] = float(peak_n)

                if css_n == 0:
                    continue

                css_hits = 0
                for c in css_pos:
                    lo = max(0, int(c) - tol)
                    hi = min(valid_len, int(c) + tol + 1)
                    if bool((z[lo:hi] >= tau).any().item()):
                        css_hits += 1

                peak_hits = 0
                for p in peaks:
                    if any(abs(int(p) - int(c)) <= tol for c in css_pos):
                        peak_hits += 1

                metrics_tau["css_hit_count"][i] = float(css_hits)
                metrics_tau["peak_hit_count"][i] = float(peak_hits)

                rec = css_hits / max(css_n, 1)
                prec = peak_hits / max(peak_n, 1) if peak_n > 0 else 0.0

                metrics_tau["recall"][i] = float(rec)
                metrics_tau["precision"][i] = float(prec)
                metrics_tau["f1"][i] = (
                    0.0 if (rec + prec) <= 0 else float(2.0 * rec * prec / (rec + prec))
                )

        return out

    def _log_css_branch_metrics(
        self,
        *,
        stage: str,
        branch_name: str,
        scores: torch.Tensor | None,
        out: dict,
        batch_size: int,
    ) -> None:
        metrics_by_tau = self._css_metrics_for_scores(
            scores=scores,
            css_items=out["css"],
            mask=out["mask"],
        )

        if metrics_by_tau is None:
            return

        dataset_ids = out["dataset_ids"]

        for tau, metrics in metrics_by_tau.items():
            zlab = self._z_label(tau)

            self._log_vector(
                stage=stage,
                name=f"css_recall_{branch_name}_{zlab}_macro",
                values=metrics["recall"],
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

            self._log_vector(
                stage=stage,
                name=f"css_precision_{branch_name}_{zlab}_macro",
                values=metrics["precision"],
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

            self._log_vector(
                stage=stage,
                name=f"css_f1_{branch_name}_{zlab}_macro",
                values=metrics["f1"],
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

            hit_total = metrics["css_hit_count"].sum()
            site_total = metrics["css_site_count"].sum()

            if float(site_total.detach().cpu().item()) > 0:
                self._log_scalar(
                    f"{stage}_css_recall_{branch_name}_{zlab}_micro",
                    hit_total / site_total.clamp_min(1.0),
                    batch_size=int(site_total.detach().cpu().item()),
                )

            peak_hit_total = metrics["peak_hit_count"].sum()
            peak_total = metrics["peak_count"].sum()

            if float(peak_total.detach().cpu().item()) > 0:
                self._log_scalar(
                    f"{stage}_css_precision_{branch_name}_{zlab}_micro",
                    peak_hit_total / peak_total.clamp_min(1.0),
                    batch_size=int(peak_total.detach().cpu().item()),
                )

            if stage == "val":
                for ds_id in torch.unique(dataset_ids).detach().cpu().tolist():
                    ds_id = int(ds_id)
                    ds_mask = dataset_ids == ds_id

                    ds_hits = metrics["css_hit_count"][ds_mask].sum()
                    ds_sites = metrics["css_site_count"][ds_mask].sum()

                    if float(ds_sites.detach().cpu().item()) > 0:
                        self._log_scalar(
                            f"{stage}_css_recall_{branch_name}_{zlab}_micro_by_dataset/"
                            f"{self._dataset_name(ds_id)}",
                            ds_hits / ds_sites.clamp_min(1.0),
                            batch_size=int(ds_sites.detach().cpu().item()),
                        )

    def _log_css_metrics(
        self,
        out: dict,
        *,
        stage: str,
        batch_size: int,
    ) -> None:
        if not bool(self._cfg("metrics.log_css_recall", True)):
            return

        extras = out["extras"]
        bio_scores = self._component(extras, "L_bio", "L_queue", "bio_q_base")
        obs_scores = self._component(extras, "L_obs", "L_queue_obs", "bio_q_obs", "q")

        for branch_name, scores in [
            ("target", out["target"]),
            ("bio", bio_scores),
            ("obs", obs_scores),
            ("full", out["mu"]),
        ]:
            self._log_css_branch_metrics(
                stage=stage,
                branch_name=branch_name,
                scores=scores,
                out=out,
                batch_size=batch_size,
            )

        tmp_by_tau = self._css_metrics_for_scores(
            scores=out["target"],
            css_items=out["css"],
            mask=out["mask"],
        )

        if tmp_by_tau is not None:
            first_tau = sorted(tmp_by_tau.keys())[0]
            tmp = tmp_by_tau[first_tau]
            css_positive = (tmp["css_site_count"] > 0).float()

            self._log_vector(
                stage=stage,
                name="css_positive_sample_frac",
                values=css_positive,
                dataset_ids=out["dataset_ids"],
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

            self._log_vector(
                stage=stage,
                name="css_site_count",
                values=tmp["css_site_count"],
                dataset_ids=out["dataset_ids"],
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

    # ============================================================
    # Validation diagnostics
    # ============================================================

    def _log_regularization_diagnostics(
        self,
        *,
        stage: str,
        loss_terms: dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        self._log_scalar(
            f"{stage}_reg_obs_bias_log_l2",
            loss_terms.get("reg_obs_bias_log_l2"),
            batch_size=batch_size,
        )

        self._log_scalar(
            f"{stage}_reg_obs_gate_open",
            loss_terms.get("reg_obs_gate_open"),
            batch_size=batch_size,
        )

        self._log_scalar(
            f"{stage}_reg_obs_gate_mean_floor",
            loss_terms.get("reg_obs_gate_mean_floor"),
            batch_size=batch_size,
        )

        self._log_scalar(
            f"{stage}_reg_kappa_log_l2",
            loss_terms.get("reg_kappa_log_l2"),
            batch_size=batch_size,
        )

    def _log_observation_bias_diagnostics(
        self,
        *,
        out: dict,
        stage: str,
        batch_size: int,
    ) -> None:
        extras = out["extras"]
        dataset_ids = out["dataset_ids"]
        mask = out["mask"]
        mask_b = mask.bool()
        mask_f = mask_b.float()
        valid_len = mask_f.sum(dim=1).clamp_min(1.0)

        def frac_true(x_bool: torch.Tensor) -> torch.Tensor:
            return ((x_bool & mask_b).float().sum(dim=1) / valid_len)

        diagnostic_keys = [
            "delta_w_obs_bio_l1",
            "w_bio_zero_frac",
            "w_obs_zero_frac",
            "L_bio_zero_frac",
            "L_obs_zero_frac",
            "b_abs_log_mean",
            "b_abs_log_w_bio_weighted",
            "obs_bias_weighted_mass",
            "hazard_cap_frac",
            "hazard_cap_frac_bio",
            "hazard_cap_frac_obs",
        ]

        for key in diagnostic_keys:
            val = self._component(extras, key)
            if torch.is_tensor(val):
                self._log_vector(
                    stage=stage,
                    name=key,
                    values=val.reshape(-1),
                    dataset_ids=dataset_ids,
                    batch_size=batch_size,
                    by_dataset=(stage == "val"),
                )

        for key in [
            "obs_bias_amp",
            "obs_bias_keep_prob",
            "obs_bias_keep_gate",
            "obs_bias_keep_hard",
            "obs_bias_effective",
        ]:
            val = self._component(extras, key)
            if torch.is_tensor(val) and val.shape == mask.shape:
                mean_val = (val.detach().float() * mask_f).sum(dim=1) / valid_len
                self._log_vector(
                    stage=stage,
                    name=f"{key}_mean",
                    values=mean_val,
                    dataset_ids=dataset_ids,
                    batch_size=batch_size,
                    by_dataset=(stage == "val"),
                )

        gate_threshold = float(
            self._cfg(
                "model.dataset_bias_params."
                "dataset_multiplicative_allocation_bias_submodule_params."
                "gate_threshold",
                0.5,
            )
        )

        keep_prob = self._component(extras, "obs_bias_keep_prob", "keep_prob")
        keep_gate = self._component(extras, "obs_bias_keep_gate", "keep_gate")
        keep_hard = self._component(extras, "obs_bias_keep_hard", "keep_hard")
        gate_logits = self._component(extras, "obs_bias_gate_logits", "gate_logits")

        if torch.is_tensor(keep_prob) and keep_prob.shape == mask.shape:
            kp = keep_prob.detach().float().clamp(0.0, 1.0)

            self._log_vector(
                stage=stage,
                name="obs_bias_keep_prob_min",
                values=torch.where(mask_b, kp, torch.ones_like(kp)).min(dim=1).values,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

            for q, label in [(0.01, "p01"), (0.05, "p05"), (0.10, "p10")]:
                self._log_vector(
                    stage=stage,
                    name=f"obs_bias_keep_prob_{label}",
                    values=self._masked_quantile_per_sample(kp, mask_b, q),
                    dataset_ids=dataset_ids,
                    batch_size=batch_size,
                    by_dataset=(stage == "val"),
                )

            self._log_vector(
                stage=stage,
                name="obs_bias_keep_prob_below_threshold_frac",
                values=frac_true(kp < gate_threshold),
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

            weights = self._mixed_position_weights_from_w_bio(
                extras=extras,
                mask=mask,
            ).detach()
            weighted_mean_open = (weights * kp).sum(dim=1)

            self._log_vector(
                stage=stage,
                name="obs_bias_keep_prob_weighted_mean",
                values=weighted_mean_open,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

        if torch.is_tensor(keep_gate) and keep_gate.shape == mask.shape:
            kg = keep_gate.detach().float()
            zero_frac = frac_true(kg <= 0.0)

            self._log_vector(
                stage=stage,
                name="obs_bias_keep_gate_zero_frac",
                values=zero_frac,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

            # Backward-compatible name.
            self._log_vector(
                stage=stage,
                name="obs_bias_gate_closed_frac",
                values=zero_frac,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

        if torch.is_tensor(keep_hard) and keep_hard.shape == mask.shape:
            kh = keep_hard.detach().float()
            self._log_vector(
                stage=stage,
                name="obs_bias_keep_hard_zero_frac",
                values=frac_true(kh <= 0.0),
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

        if torch.is_tensor(gate_logits) and gate_logits.shape == mask.shape:
            gl = gate_logits.detach().float()

            self._log_vector(
                stage=stage,
                name="obs_bias_gate_logits_mean",
                values=(gl * mask_f).sum(dim=1) / valid_len,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

            self._log_vector(
                stage=stage,
                name="obs_bias_gate_logits_p01",
                values=self._masked_quantile_per_sample(gl, mask_b, 0.01),
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

        amp = self._component(extras, "obs_bias_amp")
        b_eff = self._component(extras, "obs_bias_effective", "b_effective", "b_shape")
        b_raw = self._component(extras, "obs_bias_raw", "b_raw")

        if torch.is_tensor(amp) and amp.shape == mask.shape:
            amp = amp.detach().float()

            self._log_vector(
                stage=stage,
                name="obs_bias_amp_min",
                values=torch.where(mask_b, amp, torch.full_like(amp, float("inf"))).min(dim=1).values,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

            for q, label in [(0.01, "p01"), (0.05, "p05"), (0.10, "p10")]:
                self._log_vector(
                    stage=stage,
                    name=f"obs_bias_amp_{label}",
                    values=self._masked_quantile_per_sample(amp, mask_b, q),
                    dataset_ids=dataset_ids,
                    batch_size=batch_size,
                    by_dataset=(stage == "val"),
                )

        if torch.is_tensor(b_eff) and b_eff.shape == mask.shape:
            b_eff = b_eff.detach().float()

            for q, label in [(0.01, "p01"), (0.05, "p05"), (0.10, "p10")]:
                self._log_vector(
                    stage=stage,
                    name=f"obs_bias_effective_{label}",
                    values=self._masked_quantile_per_sample(b_eff, mask_b, q),
                    dataset_ids=dataset_ids,
                    batch_size=batch_size,
                    by_dataset=(stage == "val"),
                )

        if torch.is_tensor(b_raw) and b_raw.shape == mask.shape:
            br = b_raw.detach().float()
            self._log_vector(
                stage=stage,
                name="obs_bias_raw_zero_frac",
                values=frac_true(br <= 0.0),
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

        w_obs = self._component(extras, "w_obs")
        L_obs = self._component(extras, "L_obs", "L_queue_obs", "q")

        if torch.is_tensor(w_obs) and w_obs.shape == mask.shape:
            wo = w_obs.detach().float()
            self._log_vector(
                stage=stage,
                name="w_obs_exact_zero_frac",
                values=frac_true(wo <= 0.0),
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

        if torch.is_tensor(L_obs) and L_obs.shape == mask.shape:
            lo = L_obs.detach().float()
            self._log_vector(
                stage=stage,
                name="L_obs_exact_zero_frac",
                values=frac_true(lo <= 0.0),
                dataset_ids=dataset_ids,
                batch_size=batch_size,
                by_dataset=(stage == "val"),
            )

    def _log_validation_metrics(
        self,
        out: dict,
        *,
        batch_size: int,
    ) -> None:
        target = out["target"]
        mask = out["mask"]
        dataset_ids = out["dataset_ids"]
        extras = out["extras"]

        pcc_mu = self._log_pcc(
            stage="val",
            name="mu",
            pred=out["mu"],
            target=target,
            mask=mask,
            dataset_ids=dataset_ids,
            batch_size=batch_size,
            prog_bar=True,
        )

        mu_L_bio = self._component(extras, "mu_L_bio", "mu_L_only")
        mu_L_obs = self._component(extras, "mu_L_obs", "mu_bio_only", "mu_obs")

        if mu_L_bio is None:
            mu_L_bio = self._mu_from_support(
                self._component(extras, "L_bio", "L_queue", "bio_q_base"),
                out,
            )

        if mu_L_obs is None:
            mu_L_obs = self._mu_from_support(
                self._component(extras, "L_obs", "L_queue_obs", "q", "bio_q"),
                out,
            )

        pcc_bio = self._log_pcc(
            stage="val",
            name="mu_L_bio",
            pred=mu_L_bio,
            target=target,
            mask=mask,
            dataset_ids=dataset_ids,
            batch_size=batch_size,
        )

        pcc_obs = self._log_pcc(
            stage="val",
            name="mu_L_obs",
            pred=mu_L_obs,
            target=target,
            mask=mask,
            dataset_ids=dataset_ids,
            batch_size=batch_size,
        )

        if pcc_bio is not None and pcc_obs is not None:
            self._log_vector(
                stage="val",
                name="obs_bias_gain_pcc",
                values=pcc_obs - pcc_bio,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
            )

        if pcc_mu is not None and pcc_obs is not None:
            self._log_vector(
                stage="val",
                name="full_minus_L_obs_pcc",
                values=pcc_mu - pcc_obs,
                dataset_ids=dataset_ids,
                batch_size=batch_size,
            )

        self._log_observation_bias_diagnostics(
            out=out,
            stage="val",
            batch_size=batch_size,
        )

        self._log_css_metrics(
            out,
            stage="val",
            batch_size=batch_size,
        )

    # ============================================================
    # Optional gradient diagnostics
    # ============================================================

    def _pcgrad_params(self) -> list[torch.nn.Parameter]:
        biology_only = bool(self._cfg("optim.pcgrad_biology_only", True))

        if biology_only and hasattr(self.model, "biological_model"):
            return [
                p
                for p in self.model.biological_model.parameters()
                if p.requires_grad
            ]

        return [p for p in self.model.parameters() if p.requires_grad]

    def _log_gradient_diagnostics(self) -> None:
        every_n = int(self._cfg("metrics.grad_log_every_n_steps", 0))

        if every_n <= 0 or int(self.global_step) % every_n != 0:
            return

        params = [
            p for p in self.model.parameters()
            if p.requires_grad and p.grad is not None
        ]

        if not params:
            return

        grad_norm = torch.stack(
            [p.grad.detach().float().norm() for p in params]
        ).norm()

        self.log(
            "grad/global_norm",
            grad_norm,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

    def on_after_backward(self) -> None:
        if not self.use_pcgrad:
            self._log_gradient_diagnostics()

    # ============================================================
    # Optional validation plot
    # ============================================================

    def on_validation_epoch_start(self):
        self._val_plot_logged_this_epoch = False

    def _seq_np(
        self,
        x,
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

    @staticmethod
    def _same_id(a, b) -> bool:
        return str(a) == str(b)

    def _plot_validation_example(
        self,
        out: dict,
        *,
        batch_idx: int,
    ) -> None:
        if not bool(self._cfg("metrics.log_example_plot", True)):
            return

        if self._val_plot_logged_this_epoch or batch_idx != 0:
            return

        if not getattr(self.trainer, "is_global_zero", True):
            return

        if self.logger is None or getattr(self.logger, "experiment", None) is None:
            return

        ids = out["ids"]
        dataset_ids = out["dataset_ids"]
        B = len(ids)

        if B < 1:
            return

        idx1, idx2 = 0, min(1, B - 1)
        perfect_match = False

        for i in range(B):
            for j in range(i + 1, B):
                if self._same_id(ids[i], ids[j]) and dataset_ids[i] != dataset_ids[j]:
                    idx1, idx2 = i, j
                    perfect_match = True
                    break
            if perfect_match:
                break

        extras = out["extras"]

        L_bio = self._component(extras, "L_bio", "L_queue", "bio_q_base")
        L_obs = self._component(extras, "L_obs", "L_queue_obs", "q", "bio_q")

        w_bio = self._component(extras, "w_bio")
        w_obs = self._component(extras, "w_obs")

        b_eff = self._component(extras, "obs_bias_effective")
        keep_prob = self._component(extras, "obs_bias_keep_prob")
        keep_gate = self._component(extras, "obs_bias_keep_gate")
        keep_hard = self._component(extras, "obs_bias_keep_hard")
        amp = self._component(extras, "obs_bias_amp")

        mu_L_bio = self._component(extras, "mu_L_bio", "mu_L_only")
        mu_L_obs = self._component(extras, "mu_L_obs", "mu_bio_only")

        if mu_L_bio is None:
            mu_L_bio = self._mu_from_support(L_bio, out)

        if mu_L_obs is None:
            mu_L_obs = self._mu_from_support(L_obs, out)

        fig, axes = plt.subplots(3, 2, figsize=(24, 9), sharex="col")

        for col, sample_idx in enumerate([idx1, idx2]):
            mask_i = out["mask"][sample_idx].detach().bool().cpu()
            L = int(mask_i.sum().item())

            if L < 2:
                continue

            x_axis = torch.arange(L).numpy()
            ds_id = int(dataset_ids[sample_idx].detach().cpu().item())
            title = "PAIRED" if perfect_match else "EXAMPLE"

            axes[0, col].set_title(
                f"[{title}] Dataset: {self._dataset_name(ds_id)} | "
                f"ID: {ids[sample_idx]} | L={L}"
            )

            for tensor, label, lw in [
                (out["target"], "target y", 1.0),
                (out["mu"], "mu full / mu L_obs", 1.0),
                (mu_L_obs, "mu L_obs", 0.9),
                (mu_L_bio, "mu L_bio", 0.8),
            ]:
                arr = self._seq_np(tensor, sample_idx, mask_i, L)
                if arr is not None:
                    axes[0, col].plot(x_axis, arr, label=label, linewidth=lw)

            css_pos = self._css_positions_from_item(out["css"][sample_idx], L)
            for c in css_pos:
                axes[0, col].axvline(c, linestyle="--", alpha=0.25, linewidth=0.8)

            axes[0, col].set_ylabel("profile" if col == 0 else "")
            axes[0, col].grid(True, alpha=0.3)
            axes[0, col].legend(loc="upper right")

            for tensor, label in [(L_bio, "L_bio"), (L_obs, "L_obs")]:
                arr = self._seq_np(tensor, sample_idx, mask_i, L)
                if arr is not None:
                    axes[1, col].plot(x_axis, arr, label=label, linewidth=0.9)

            for c in css_pos:
                axes[1, col].axvline(c, linestyle="--", alpha=0.25, linewidth=0.8)

            axes[1, col].set_ylabel("support" if col == 0 else "")
            axes[1, col].grid(True, alpha=0.3)
            axes[1, col].legend(loc="upper right")

            for tensor, label in [
                (w_bio, "w_bio"),
                (w_obs, "w_obs"),
                (b_eff, "b_eff"),
                (amp, "amp"),
                (keep_prob, "keep_prob"),
                (keep_gate, "keep_gate"),
                (keep_hard, "keep_hard"),
            ]:
                arr = self._seq_np(tensor, sample_idx, mask_i, L)
                if arr is not None:
                    axes[2, col].plot(x_axis, arr, label=label, linewidth=0.8)

            axes[2, col].set_ylabel("allocation / bias" if col == 0 else "")
            axes[2, col].set_xlabel("codon position")
            axes[2, col].grid(True, alpha=0.3)
            axes[2, col].legend(loc="upper right")

        fig.tight_layout()
        exp = self.logger.experiment

        if hasattr(exp, "add_figure"):
            exp.add_figure(
                "val/example_profile_comparison",
                fig,
                global_step=self.global_step,
            )
        elif hasattr(exp, "log_figure"):
            exp.log_figure(
                figure_name="val/example_profile_comparison",
                figure=fig,
                step=self.global_step,
            )

        plt.close(fig)
        self._val_plot_logged_this_epoch = True

    # ============================================================
    # PCGrad
    # ============================================================

    def _pcgrad_step(
        self,
        out: dict,
        loss_terms: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        params = self._pcgrad_params()
        profile_scalar = loss_terms["profile_loss"]
        full_loss = loss_terms["total_loss"]

        ds_losses = per_dataset_losses(
            loss_per_sample=loss_terms["nll_per_sample"],
            dataset_ids=out["dataset_ids"],
        )

        if len(ds_losses) <= 1 or len(params) == 0:
            self.manual_backward(full_loss)
            self._log_gradient_diagnostics()
            return profile_scalar.detach()

        max_datasets = int(
            self._cfg("optim.pcgrad_max_datasets_per_step", len(ds_losses))
        )

        if len(ds_losses) > max_datasets:
            perm = torch.randperm(len(ds_losses), device=profile_scalar.device)[:max_datasets]
            ds_losses = [ds_losses[int(i)] for i in perm.detach().cpu().tolist()]

        raw_grads = []
        unit_grads = []

        for ds_loss in ds_losses:
            opt.zero_grad(set_to_none=True)
            self.manual_backward(ds_loss, retain_graph=True)
            g = flatten_current_grads(params).detach()
            raw_grads.append(g)
            unit_grads.append(g / g.norm().clamp_min(1.0e-12))

        raw_grads_t = torch.stack(raw_grads, dim=0)
        unit_grads_t = torch.stack(unit_grads, dim=0)

        pcgrad = pcgrad_combine(unit_grads).detach()
        mean_grad = unit_grads_t.mean(dim=0)

        alpha = float(self._cfg("optim.pcgrad_alpha", 1.0))
        pcgrad = alpha * pcgrad + (1.0 - alpha) * mean_grad

        if bool(self._cfg("optim.pcgrad_rescale_to_mean_norm", True)):
            target_norm = raw_grads_t.norm(dim=1).mean().clamp_min(1.0e-12)
            pcgrad = pcgrad * (target_norm / pcgrad.norm().clamp_min(1.0e-12))

        opt.zero_grad(set_to_none=True)
        self.manual_backward(full_loss)
        assign_flat_grads(params=params, flat_grad=pcgrad.detach())
        self._log_gradient_diagnostics()

        return profile_scalar.detach()

    # ============================================================
    # Steps
    # ============================================================

    def _forward_batch(self, batch) -> dict:
        (
            dataset_ids,
            ids,
            seq_packed,
            target,
            lengths,
            mask,
            codon_ids,
            css,
        ) = batch

        model_out = self.model(seq_packed, codon_ids, dataset_ids, target)

        if isinstance(model_out, tuple) and len(model_out) == 4:
            mu, profile_aux, kappa_input, extras = model_out
        elif isinstance(model_out, tuple) and len(model_out) == 3:
            mu, kappa_input, extras = model_out
            profile_aux = None
        else:
            raise RuntimeError(
                "Expected model output (mu, kappa_input, extras) or "
                "(mu, profile_aux, kappa_input, extras)."
            )

        return {
            "dataset_ids": dataset_ids,
            "ids": ids,
            "target": target,
            "lengths": lengths,
            "mask": mask,
            "codon_ids": codon_ids,
            "css": css,
            "mu": mu,
            "profile_aux": profile_aux,
            "kappa_input": kappa_input,
            "extras": extras if isinstance(extras, dict) else {},
        }

    def training_step(self, batch, batch_idx):
        out = self._forward_batch(batch)
        batch_size = int(out["target"].shape[0])
        loss_terms = self._loss_terms(out)

        self._log_scalar("train_loss", loss_terms["total_loss"], batch_size=batch_size, prog_bar=True)
        self._log_scalar("train_profile_loss", loss_terms["profile_loss"], batch_size=batch_size)
        self._log_scalar("train_extra_loss", loss_terms["extra_loss"], batch_size=batch_size)

        self._log_regularization_diagnostics(
            stage="train",
            loss_terms=loss_terms,
            batch_size=batch_size,
        )

        if not self.use_pcgrad:
            return loss_terms["total_loss"]

        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        every_n = int(self._cfg("optim.pcgrad_every_n_steps", 1))
        warmup_steps = int(self._cfg("optim.pcgrad_warmup_steps", 0))
        do_pcgrad = (
            int(self.global_step) >= warmup_steps
            and (every_n <= 1 or int(self.global_step) % every_n == 0)
        )

        if do_pcgrad:
            step_loss = self._pcgrad_step(out, loss_terms)
        else:
            self.manual_backward(loss_terms["total_loss"])
            self._log_gradient_diagnostics()
            step_loss = loss_terms["profile_loss"].detach()

        grad_clip_val = float(self._cfg("trainer.gradient_clip_val", 0.0))

        if grad_clip_val > 0.0:
            self.clip_gradients(
                opt,
                gradient_clip_val=grad_clip_val,
                gradient_clip_algorithm=str(self._cfg("trainer.gradient_clip_algorithm", "norm")),
            )

        opt.step()
        opt.zero_grad(set_to_none=True)
        return step_loss

    def validation_step(self, batch, batch_idx):
        out = self._forward_batch(batch)
        batch_size = int(out["target"].shape[0])
        loss_terms = self._loss_terms(out)

        self._log_scalar("val_loss", loss_terms["total_loss"], batch_size=batch_size, prog_bar=True)
        self._log_scalar("val_profile_loss", loss_terms["profile_loss"], batch_size=batch_size)
        self._log_scalar("val_extra_loss", loss_terms["extra_loss"], batch_size=batch_size)

        self._log_regularization_diagnostics(stage="val", loss_terms=loss_terms, batch_size=batch_size)
        self._log_validation_metrics(out, batch_size=batch_size)
        self._plot_validation_example(out, batch_idx=batch_idx)
        return loss_terms["total_loss"]

    def on_validation_epoch_end(self):
        if not self.use_pcgrad:
            return

        sched = self.lr_schedulers()
        monitor = str(self._cfg("optim.scheduler.monitor", "val_loss"))

        if monitor in self.trainer.callback_metrics:
            sched.step(self.trainer.callback_metrics[monitor])

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

        result = {
            "ids": to_cpu(out["ids"]),
            "dataset_id": to_cpu(out["dataset_ids"]),
            "lengths": to_cpu(out["lengths"]),
            "mask": to_cpu(out["mask"]),
            "css": to_cpu(out["css"]),
            "y": to_cpu(out["target"]),
            "mu_obs": to_cpu(out["mu"]),
        }

        if bool(self._cfg("predict.export_kappa", False)):
            kappa = self._compute_kappa(kappa_input=out["kappa_input"], mask=out["mask"])
            result["profile_kappa"] = to_cpu(kappa)
            result["kappa_input"] = to_cpu(out["kappa_input"])

        export_all = bool(self._cfg("predict.export_all_extras", False))

        important = [
            # Biological branch
            "w_logits", "w_bio", "w_prob", "h_bio", "rho_bio",
            "L_queue", "L_bio", "bio_q_base", "J",

            # Multiplicative observation bias
            "obs_bias_raw", "obs_bias_effective", "obs_bias_weighted_mass",
            "obs_bias_amp", "obs_bias_amp_logits", "obs_bias_keep_prob",
            "obs_bias_keep_gate", "obs_bias_keep_hard", "obs_bias_gate_logits",

            # Observed branch
            "w_obs", "h_obs", "rho_obs", "L_obs", "L_queue_obs",
            "bio_q", "q", "profile_prob",

            # Mean profiles
            "total_mass", "mu_L_only", "mu_L_bio", "mu_L_obs",
            "mu_bio_smooth", "mu_bio_only",

            # Diagnostics from the model extras
            "w_bio_zero_frac", "w_obs_zero_frac", "L_bio_zero_frac",
            "L_obs_zero_frac", "delta_w_obs_bio_l1", "b_abs_log_mean",
            "b_abs_log_w_bio_weighted", "hazard_cap_frac",
            "hazard_cap_frac_bio", "hazard_cap_frac_obs",
        ]

        for key, val in out["extras"].items():
            if val is not None and (export_all or key in important):
                result[key] = to_cpu(val)

        return result

    # ============================================================
    # Optimizer
    # ============================================================

    def configure_optimizers(self):
        base_lr = float(self._cfg("optim.lr", 1.0e-3))
        bio_lr = float(self._cfg("optim.lr_biological", base_lr))
        rest_lr = float(self._cfg("optim.lr_rest", base_lr))
        weight_decay = float(self._cfg("optim.weight_decay", 1.0e-8))

        bio_params = []
        rest_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if name.startswith("biological_model."):
                bio_params.append(param)
            else:
                rest_params.append(param)

        param_groups = []

        if bio_params:
            param_groups.append({"params": bio_params, "lr": bio_lr, "weight_decay": weight_decay})

        if rest_params:
            param_groups.append({"params": rest_params, "lr": rest_lr, "weight_decay": weight_decay})

        if not param_groups:
            raise RuntimeError("No trainable parameters found.")

        opt = torch.optim.AdamW(param_groups)

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=str(self._cfg("optim.scheduler.mode", "min")),
            factor=float(self._cfg("optim.scheduler.factor", 0.9)),
            patience=int(self._cfg("optim.scheduler.patience", 10)),
            min_lr=float(self._cfg("optim.scheduler.min_lr", 1.0e-6)),
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
