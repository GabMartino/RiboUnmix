from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import lightning as pl
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use("Agg")

from matplotlib import pyplot as plt


# ============================================================
# Loss
# ============================================================

def _broadcast_profile_param(
    value: torch.Tensor,
    target_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    """Broadcast transcript-level or position-level parameters to [B, T]."""
    if value.ndim == len(target_shape) + 1 and value.shape[-1] == 1:
        value = value.squeeze(-1)

    if value.ndim == 1 and len(target_shape) == 2 and value.shape[0] == target_shape[0]:
        value = value.reshape(-1, 1)

    return torch.broadcast_to(value, target_shape)


def reduce_sequence_nll(
    nll_pos: torch.Tensor,
    mask: torch.Tensor,
    reduction: str,
    gamma: float = 0.85,
    length_ref: float = 1000.0,
    min_weight: float = 0.5,
    max_weight: float = 2.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    reduction = str(reduction)
    mask_f = mask.to(dtype=nll_pos.dtype)
    valid_len = mask_f.sum(dim=1).clamp_min(1.0)
    nll_sum = (nll_pos * mask_f).sum(dim=1)
    nll_mean = nll_sum / valid_len

    if reduction == "mean":
        return nll_mean
    if reduction == "sum":
        return nll_sum
    if reduction == "length_tempered":
        gamma_t = min(max(float(gamma), float(eps)), 1.0)
        length_ref_t = max(float(length_ref), float(eps))
        length_weight = (valid_len / length_ref_t).pow(1.0 - gamma_t)
        length_weight = length_weight.clamp(
            min=float(min_weight),
            max=float(max_weight),
        )
        return nll_mean * length_weight

    raise ValueError(
        "sequence reduction must be one of {'mean', 'sum', 'length_tempered'}, "
        f"got {reduction!r}."
    )


def sequence_length_temper_weights(
    mask: torch.Tensor,
    gamma: float = 0.85,
    length_ref: float = 1000.0,
    min_weight: float = 0.5,
    max_weight: float = 2.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    mask_f = mask.float()
    valid_len = mask_f.sum(dim=1).clamp_min(1.0)
    gamma_t = min(max(float(gamma), float(eps)), 1.0)
    length_ref_t = max(float(length_ref), float(eps))
    return (valid_len / length_ref_t).pow(1.0 - gamma_t).clamp(
        min=float(min_weight),
        max=float(max_weight),
    )


SAMPLE_REDUCTION_MODES = frozenset(
    {"global_weighted", "dataset_balanced", "transcript_balanced"}
)


def resolve_sample_reduction_mode(loss_config: Any) -> str:
    """Resolve the explicit reducer while honoring the legacy boolean.

    New configurations default to transcript balancing. Historical
    ``dataset_balanced_loss`` values remain unambiguous: false maps to the old
    global weighted mean and true maps to the old dataset-balanced mean. If an
    explicit mode is also present it must agree with that mapping.
    """

    sentinel = object()

    def read(name: str) -> Any:
        try:
            return getattr(loss_config, name)
        except (AttributeError, KeyError):
            return sentinel

    explicit = read("sample_reduction")
    legacy = read("dataset_balanced_loss")
    explicit_mode = (
        None if explicit is sentinel else str(explicit).strip().lower()
    )
    if explicit_mode is not None and explicit_mode not in SAMPLE_REDUCTION_MODES:
        raise ValueError(
            "loss.sample_reduction must be one of "
            f"{sorted(SAMPLE_REDUCTION_MODES)}, got {explicit!r}."
        )

    legacy_mode: str | None = None
    if legacy is not sentinel:
        if not isinstance(legacy, bool):
            raise TypeError("loss.dataset_balanced_loss must be a boolean when present.")
        legacy_mode = "dataset_balanced" if legacy else "global_weighted"

    if explicit_mode is not None and legacy_mode is not None:
        if explicit_mode != legacy_mode:
            raise ValueError(
                "Conflicting loss reduction settings: "
                f"sample_reduction={explicit_mode!r} but "
                f"dataset_balanced_loss={legacy!r} maps to {legacy_mode!r}."
            )
        return explicit_mode
    if explicit_mode is not None:
        return explicit_mode
    if legacy_mode is not None:
        return legacy_mode
    return "transcript_balanced"


def reduce_per_sample_quantity(
    values: torch.Tensor,
    sample_weights: torch.Tensor,
    dataset_ids: torch.Tensor,
    transcript_group_ids: torch.Tensor,
    mode: str,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Differentiably reduce one finite value per transcript--dataset pair.

    ``global_weighted`` is ``sum_s(w_s z_s) / sum_s(w_s)``.
    ``dataset_balanced`` takes that weighted mean within each represented
    dataset, then gives datasets equal outer mass. ``transcript_balanced`` does
    the analogous operation within transcript groups, then gives every
    transcript equal outer mass.

    Writing ``S_t=sum_{s in t}w_s`` and
    ``zbar_t=sum_{s in t}w_s z_s/S_t``, the global mode is
    ``sum_t [S_t/sum_u S_u] zbar_t``. It therefore equals transcript balancing
    exactly when every transcript has the same total weight mass (including
    the unit-weight, equal-dataset-count rectangle).

    Accumulation is float32 under fp16/bf16 and autograd remains connected to
    ``values``. Dataset-quality ranks are intentionally absent from this API.
    """
    mode = str(mode).strip().lower()
    if mode not in SAMPLE_REDUCTION_MODES:
        raise ValueError(
            f"mode must be one of {sorted(SAMPLE_REDUCTION_MODES)}, got {mode!r}."
        )
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and strictly positive.")
    tensors = {
        "values": values,
        "sample_weights": sample_weights,
        "dataset_ids": dataset_ids,
        "transcript_group_ids": transcript_group_ids,
    }
    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor) or tensor.ndim != 1:
            shape = tuple(tensor.shape) if torch.is_tensor(tensor) else None
            raise ValueError(f"{name} must be a one-dimensional tensor; got {shape}.")
        if tensor.device != values.device:
            raise ValueError(
                f"{name} is on {tensor.device}, but values is on {values.device}."
            )
    if not values.is_floating_point():
        raise TypeError("values must use a floating-point dtype.")
    if dataset_ids.is_floating_point() or transcript_group_ids.is_floating_point():
        raise TypeError("dataset_ids and transcript_group_ids must be integer tensors.")
    if not sample_weights.is_floating_point():
        raise TypeError("sample_weights must use a floating-point dtype.")
    pair_count = values.numel()
    if not all(tensor.numel() == pair_count for tensor in tensors.values()):
        raise ValueError(
            "values, sample_weights, dataset_ids, and transcript_group_ids "
            "must contain the same number of pairs."
        )
    if pair_count == 0:
        return values.sum() * 0.0
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Per-pair loss values must all be finite.")
    if not bool(torch.isfinite(sample_weights).all()):
        raise ValueError("Transcript reliability weights must all be finite.")
    if bool((sample_weights <= 0.0).any()):
        raise ValueError("Transcript reliability weights must all be strictly positive.")

    accumulation_dtype = (
        torch.float32
        if values.dtype in {torch.float16, torch.bfloat16}
        else values.dtype
    )
    values_acc = values.to(dtype=accumulation_dtype)
    weights_acc = sample_weights.to(dtype=accumulation_dtype)

    if mode == "global_weighted":
        return (values_acc * weights_acc).sum() / weights_acc.sum()

    group_ids = dataset_ids if mode == "dataset_balanced" else transcript_group_ids
    unique_groups, inverse = torch.unique(
        group_ids, sorted=False, return_inverse=True
    )
    group_count = int(unique_groups.numel())
    numerator = torch.zeros(
        group_count,
        device=values.device,
        dtype=accumulation_dtype,
    )
    denominator = torch.zeros_like(numerator)
    numerator.index_add_(0, inverse, values_acc * weights_acc)
    denominator.index_add_(0, inverse, weights_acc)
    valid_groups = denominator > float(eps)
    if not bool(valid_groups.any()):
        return values.sum() * 0.0
    return (numerator[valid_groups] / denominator[valid_groups]).mean()


def reduce_dataset_balanced_weighted_mean(
    values: torch.Tensor,
    dataset_ids: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible wrapper around the generic dataset reducer."""
    values_1d = values.reshape(-1)
    return reduce_per_sample_quantity(
        values=values_1d,
        sample_weights=sample_weights.reshape(-1).to(device=values_1d.device),
        dataset_ids=dataset_ids.reshape(-1).to(device=values_1d.device),
        transcript_group_ids=torch.arange(
            values_1d.numel(), device=values_1d.device, dtype=torch.long
        ),
        mode="dataset_balanced",
    )


def masked_pcc(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    min_target_var: float = 1.0e-6,
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    x = x.float()
    y = y.float()
    mask_b = mask.bool() & torch.isfinite(x) & torch.isfinite(y)
    mask_f = mask_b.to(dtype=x.dtype)

    x = torch.where(mask_b, x, torch.zeros_like(x))
    y = torch.where(mask_b, y, torch.zeros_like(y))

    w = mask_f

    w_sum = w.sum(dim=1, keepdim=True).clamp_min(float(eps))
    x_mean = (w * x).sum(dim=1, keepdim=True) / w_sum
    y_mean = (w * y).sum(dim=1, keepdim=True) / w_sum

    x_c = (x - x_mean) * mask_f
    y_c = (y - y_mean) * mask_f

    # Work with mask-normalized covariance and variances.  The historical
    # ``sum_cov / sqrt(sum_x_var * sum_y_var + eps**2)`` expression leaves a
    # singular derivative when the prediction is nearly flat: its denominator
    # is then only ``eps`` even for a variable target.  Adding eps to each
    # variance gives a finite, scale-consistent derivative while changing
    # ordinary well-conditioned Pearson values only at numerical precision.
    normalization = w_sum.squeeze(1).clamp_min(float(eps))
    covariance = (w * x_c * y_c).sum(dim=1) / normalization
    x_variance = (w * x_c.pow(2)).sum(dim=1) / normalization
    y_variance = (w * y_c.pow(2)).sum(dim=1) / normalization
    pcc = covariance / torch.sqrt(
        (x_variance + float(eps)) * (y_variance + float(eps))
    )
    pcc = torch.nan_to_num(pcc, nan=0.0, posinf=0.0, neginf=0.0)

    target_var = y_variance
    valid = target_var > float(min_target_var)
    pcc = torch.where(valid, pcc, torch.zeros_like(pcc))

    valid_f = valid.to(dtype=x.dtype)
    valid_count = valid_f.sum().clamp_min(1.0)
    valid_fraction = valid_f.mean() if valid_f.numel() > 0 else x.new_tensor(0.0)
    target_var_mean = (target_var * valid_f).sum() / valid_count

    return {
        "pcc_per_sample": pcc,
        "valid": valid,
        "valid_fraction": valid_fraction,
        "target_var_mean": target_var_mean,
    }


class NegativeBinomialProfileLoss(nn.Module):
    """
    Negative Binomial profile NLL.

    Parameterization:
        mu:        predicted mean.
        log_alpha: log dispersion, alpha = exp(log_alpha), r = 1 / alpha.
                   Var(Y) = mu + alpha * mu^2.
        sequence_reduction:
            "mean" averages valid positions within each sequence before the
            batch reduction; "sum" sums valid positions, then averages across
            batch samples.

    The loss INCLUDES the target-only lgamma(y + 1) normalization term:

        nll = lgamma(r)
              - lgamma(y + r)
              + lgamma(y + 1)
              - r * log(r)
              - y * log(mu)
              + (r + y) * log(r + mu)

    The lgamma(y + 1) term is constant w.r.t. the model parameters, so it does
    NOT change gradients, but it makes the reported NLL a proper NB log-density
    and improves cross-dataset comparability of logged NLL values. With
    non-integer (replicate-averaged) targets the gamma terms use the continuous
    lgamma extension.
    """

    def __init__(
        self,
        eps: float = 1.0e-8,
        log_alpha_min: float = -5.0,
        log_alpha_max: float = 3.0,
        sequence_reduction: str = "mean",
        length_temper_gamma: float = 0.85,
        length_temper_ref: float = 1000.0,
        length_temper_min_weight: float = 0.5,
        length_temper_max_weight: float = 2.0,
    ):
        super().__init__()

        self.eps = float(eps)
        self.log_alpha_min = float(log_alpha_min)
        self.log_alpha_max = float(log_alpha_max)
        self.sequence_reduction = sequence_reduction
        self.length_temper_gamma = float(length_temper_gamma)
        self.length_temper_ref = float(length_temper_ref)
        self.length_temper_min_weight = float(length_temper_min_weight)
        self.length_temper_max_weight = float(length_temper_max_weight)

        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("loss.eps must be finite and strictly positive.")
        if not (
            math.isfinite(self.log_alpha_min)
            and math.isfinite(self.log_alpha_max)
            and self.log_alpha_min <= self.log_alpha_max
        ):
            raise ValueError(
                "loss.nb_log_alpha_min/max must be finite and min must be <= max."
            )
        if self.sequence_reduction not in {"mean", "sum", "length_tempered"}:
            raise ValueError(
                "loss.nb_sequence_reduction must be one of "
                "{'mean', 'sum', 'length_tempered'}."
            )
        if not 0.0 <= self.length_temper_gamma <= 1.0:
            raise ValueError("loss.nb_length_temper_gamma must be in [0, 1].")
        if not math.isfinite(self.length_temper_ref) or self.length_temper_ref <= 0.0:
            raise ValueError("loss.nb_length_temper_ref must be finite and positive.")
        if not (
            math.isfinite(self.length_temper_min_weight)
            and math.isfinite(self.length_temper_max_weight)
            and 0.0 < self.length_temper_min_weight <= self.length_temper_max_weight
        ):
            raise ValueError(
                "loss NB length-tempering bounds must be finite, positive, and ordered."
            )

    def positive_mean_from_params(
        self,
        *,
        mu: torch.Tensor,
        log_sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del log_sigma
        mu = torch.nan_to_num(
            mu,
            nan=self.eps,
            posinf=torch.finfo(mu.dtype).max,
            neginf=self.eps,
        )
        return mu.clamp_min(self.eps)

    def _log_alpha_from_model_output(
        self,
        *,
        log_sigma: torch.Tensor,
        target_shape: torch.Size | tuple[int, ...],
    ) -> torch.Tensor:
        log_alpha = _broadcast_profile_param(log_sigma, target_shape)
        log_alpha = torch.nan_to_num(
            log_alpha,
            nan=0.0,
            posinf=self.log_alpha_max,
            neginf=self.log_alpha_min,
        )
        return log_alpha.clamp(min=self.log_alpha_min, max=self.log_alpha_max)

    def forward(
        self,
        mu_phys: torch.Tensor,
        log_sigma: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=mu_phys.device.type, enabled=False):
            finite_mask = mask.bool() & torch.isfinite(y_true)
            y = torch.nan_to_num(
                y_true.to(torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)

            mu = self.positive_mean_from_params(
                mu=mu_phys.to(torch.float32),
                log_sigma=None,
            ).to(torch.float32)
            log_alpha = self._log_alpha_from_model_output(
                log_sigma=log_sigma.to(torch.float32),
                target_shape=y.shape,
            )
            r = torch.exp(-log_alpha).clamp_min(self.eps)

            nll = (
                torch.lgamma(r)
                - torch.lgamma(y + r)
                + torch.lgamma(y + 1.0)
                - r * torch.log(r.clamp_min(self.eps))
                - y * torch.log(mu.clamp_min(self.eps))
                + (r + y) * torch.log((r + mu).clamp_min(self.eps))
            )

            nll = torch.nan_to_num(nll, nan=0.0, posinf=1.0e8, neginf=1.0e8)

            nll = torch.where(finite_mask, nll, torch.zeros_like(nll))
            loss_per_sample = reduce_sequence_nll(
                nll,
                finite_mask,
                self.sequence_reduction,
                gamma=self.length_temper_gamma,
                length_ref=self.length_temper_ref,
                min_weight=self.length_temper_min_weight,
                max_weight=self.length_temper_max_weight,
                eps=self.eps,
            )
            loss_per_sample = loss_per_sample.to(torch.float32)

        if return_per_sample:
            return loss_per_sample

        return loss_per_sample.mean()


# ============================================================
# Lightning module
# ============================================================

class RiboQueuingModelLightningModule(pl.LightningModule):
    """
    Minimal LightningModule for the queue-load model:

        multiplicative mean:
            mu[d,t,i] = S[d,t] * gamma[d,t,i] * L_bio[t,i]

        Active loss:

            pair_loss = replica_nb_weight * mean_replica(NB_NLL)
                      + consensus_raw_pcc_weight * (1 - PCC_consensus)
                      + consensus_nb_vst_pcc_weight
                        * (1 - PCC_NB_VST_consensus)
                      + gamma_reg_weight * mean_position(log_gamma^2)

    NB2 is evaluated against every raw replica and then averaged within the
    transcript-dataset pair. Both shape-oriented PCC terms are evaluated once
    against the arithmetic replica consensus for that pair.
    """

    # Per-position extras plotted during validation.
    PROFILE_PLOT_GROUPS = (
        ("L_bio", ("L_bio",)),
        ("rho", ("rho",)),
        ("gamma", ("gamma",)),
        ("log_sigma", ("log_sigma",)),
    )

    def __init__(
        self,
        torch_model: nn.Module,
        config: Any,
        dataset_encoding: dict,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["torch_model", "config", "dataset_encoding"])

        self.model = torch_model
        self.config = config
        self.loss_fn = self._build_loss()

        self.dataset_id_to_name = {int(v): str(k) for k, v in dataset_encoding.items()}
        self._val_profile_plot_logged_this_epoch = False
        loss_cfg = self.config.loss

        def required_nonnegative_weight(name: str) -> float:
            sentinel = object()
            raw_value = getattr(loss_cfg, name, sentinel)
            if raw_value is sentinel:
                raise ValueError(f"Missing mandatory loss coefficient loss.{name}.")
            if isinstance(raw_value, bool):
                raise TypeError(f"loss.{name} must be a number, not a boolean.")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"loss.{name} must be a finite non-negative number.") from exc
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"loss.{name} must be a finite non-negative number, got {raw_value!r}."
                )
            return value

        legacy_pcc_fields = tuple(
            name
            for name in (
                "replica_raw_pcc_weight",
                "replica_nb_vst_pcc_weight",
            )
            if getattr(loss_cfg, name, None) is not None
        )
        if legacy_pcc_fields:
            raise ValueError(
                "The replica-level PCC coefficients are no longer supported because "
                "PCC is now evaluated once against the arithmetic replica consensus. "
                "Replace "
                + ", ".join(f"loss.{name}" for name in legacy_pcc_fields)
                + " with loss.consensus_raw_pcc_weight and "
                "loss.consensus_nb_vst_pcc_weight."
            )

        self.replica_nb_weight = required_nonnegative_weight("replica_nb_weight")
        self.consensus_raw_pcc_weight = required_nonnegative_weight(
            "consensus_raw_pcc_weight"
        )
        self.consensus_nb_vst_pcc_weight = required_nonnegative_weight(
            "consensus_nb_vst_pcc_weight"
        )
        self.gamma_reg_weight = required_nonnegative_weight("gamma_reg_weight")
        if (
            self.replica_nb_weight
            + self.consensus_raw_pcc_weight
            + self.consensus_nb_vst_pcc_weight
            + self.gamma_reg_weight
            == 0.0
        ):
            raise ValueError("At least one of the four loss coefficients must be positive.")
        self.min_pcc_target_var = float(getattr(loss_cfg, "min_pcc_target_var", 1.0e-6))
        if not math.isfinite(self.min_pcc_target_var) or self.min_pcc_target_var < 0.0:
            raise ValueError("loss.min_pcc_target_var must be finite and non-negative.")
        self.pcc_detach_alpha = bool(getattr(loss_cfg, "pcc_detach_alpha", True))
        self.eps = float(getattr(loss_cfg, "eps", 1.0e-8))
        self.sample_reduction = resolve_sample_reduction_mode(loss_cfg)
        self.hparams["sample_reduction"] = self.sample_reduction
        self.alpha_learning_rate_scale = float(
            getattr(self.config.optim, "alpha_learning_rate_scale", 0.1)
        )
        if (
            not math.isfinite(self.alpha_learning_rate_scale)
            or self.alpha_learning_rate_scale <= 0.0
        ):
            raise ValueError(
                "optim.alpha_learning_rate_scale must be finite and strictly positive."
            )
        self.hparams["alpha_learning_rate_scale"] = (
            self.alpha_learning_rate_scale
        )
        execution_cfg = getattr(
            getattr(self.config, "training", None),
            "execution_microbatching",
            None,
        )
        self.execution_microbatching_enabled = bool(
            getattr(execution_cfg, "enabled", False)
        )
        execution_clip_value = getattr(execution_cfg, "gradient_clip_val", None)
        if execution_clip_value is None:
            execution_clip_value = getattr(
                getattr(self.config, "trainer", None),
                "gradient_clip_val",
                0.0,
            )
        if execution_clip_value is None:
            execution_clip_value = 0.0
        self.execution_gradient_clip_val = float(execution_clip_value)
        if (
            not math.isfinite(self.execution_gradient_clip_val)
            or self.execution_gradient_clip_val < 0.0
        ):
            raise ValueError(
                "Execution gradient clipping must be finite and non-negative."
            )
        self.execution_gradient_clip_algorithm = str(
            getattr(
                execution_cfg,
                "gradient_clip_algorithm",
                getattr(
                    getattr(self.config, "trainer", None),
                    "gradient_clip_algorithm",
                    "norm",
                ),
            )
        )
        if self.execution_microbatching_enabled and self.sample_reduction != "transcript_balanced":
            raise ValueError(
                "Complete-group execution microbatching currently requires "
                "loss.sample_reduction=transcript_balanced so chunk gradients can "
                "reconstruct the logical-batch objective exactly."
            )
        # Execution chunks are backwarded immediately and therefore require
        # manual optimization. They remain invisible to logical-batch and
        # optimizer-step semantics.
        self.automatic_optimization = not self.execution_microbatching_enabled
        self._execution_logical_batches_since_step = 0
        self._execution_optimizer_steps = 0
        self._execution_has_pending_gradients = False
        # Predicted-value floor for PCC only: predictions below this count are
        # treated as 0 ("undetected") when computing correlation. Honest (uses
        # only the prediction, never the target); 0.0 disables.
        self.pcc_prediction_floor = float(
            getattr(loss_cfg, "pcc_prediction_floor", 0.0)
        )
        metrics_cfg = getattr(self.config, "metrics", None)
        self.log_validation_transcript_mu_pcc_distribution = bool(
            getattr(
                metrics_cfg,
                "log_validation_transcript_mu_pcc_distribution",
                True,
            )
        )
        self._validation_transcript_mu_pcc: dict[str, float] = {}
        self._grouped_batch_logging_enabled = False
        self._grouped_optimizer_batch_plan: dict[str, Any] = {}
        self._train_batch_structure_records: list[dict[str, Any]] = []
        self._synthetic_ground_truth = self._load_synthetic_ground_truth()
        self._synthetic_ground_truth_epoch: dict[str, dict[str, list[float]]] = {}

    @staticmethod
    def _resolve_optional_path(raw_path: Any) -> Path:
        path = Path(str(raw_path)).expanduser()
        if path.is_absolute() or path.exists():
            return path
        return Path(__file__).resolve().parents[1] / path

    @staticmethod
    def _normalize_reference_profile(value: Any, *, label: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"{label} must be a non-empty finite profile.")
        if bool((array < 0.0).any()):
            raise ValueError(f"{label} contains a negative value.")
        mean = float(array.mean())
        if not math.isfinite(mean) or mean <= 0.0:
            raise ValueError(f"{label} must have a positive mean.")
        return (array / mean).astype(np.float32)

    def _load_synthetic_ground_truth(self) -> dict[str, dict[str, np.ndarray]] | None:
        """Load optional synthetic references for validation-only diagnostics."""
        cfg = getattr(self.config, "synthetic_ground_truth", None)
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return None

        latent_path = self._resolve_optional_path(getattr(cfg, "latent_path"))
        raw_observed_path = getattr(cfg, "observed_path", None)
        observed_path = (
            self._resolve_optional_path(raw_observed_path)
            if raw_observed_path not in (None, "", "null", "None")
            else None
        )
        if not latent_path.is_file():
            raise FileNotFoundError(f"Synthetic latent ground truth not found: {latent_path}")
        if observed_path is not None and not observed_path.is_file():
            raise FileNotFoundError(f"Synthetic observed ground truth not found: {observed_path}")

        latent_frame = pd.read_parquet(latent_path, columns=["transcript_id", "rib_profile"])
        latent: dict[str, np.ndarray] = {}
        for row in latent_frame.itertuples(index=False):
            transcript_id = str(row.transcript_id)
            if transcript_id in latent:
                raise ValueError(f"Duplicate synthetic latent transcript ID: {transcript_id}")
            latent[transcript_id] = self._normalize_reference_profile(
                row.rib_profile,
                label=f"latent profile {transcript_id}",
            )

        observed: dict[str, np.ndarray] = {}
        drop_terminal = bool(getattr(cfg, "drop_terminal_position", True))
        if observed_path is not None:
            observed_frame = pd.read_parquet(observed_path, columns=["id", "ribo"])
            for row in observed_frame.itertuples(index=False):
                transcript_id = str(row.id)
                truth = latent.get(transcript_id)
                if truth is None:
                    continue
                profile = np.asarray(row.ribo, dtype=np.float64).reshape(-1)
                expected_length = int(truth.size)
                if drop_terminal and profile.size == expected_length + 1:
                    profile = profile[:-1]
                elif profile.size != expected_length:
                    raise ValueError(
                        f"Observed profile length mismatch for {transcript_id}: "
                        f"observed={profile.size}, latent={expected_length}."
                    )
                if transcript_id in observed:
                    raise ValueError(f"Duplicate synthetic observed transcript ID: {transcript_id}")
                observed[transcript_id] = self._normalize_reference_profile(
                    profile,
                    label=f"observed profile {transcript_id}",
                )

        if not latent:
            raise ValueError(f"No profiles found in synthetic latent reference: {latent_path}")
        if observed_path is not None and not observed:
            raise ValueError(
                "No transcript IDs overlap between synthetic latent and observed references."
            )
        print(
            "[synthetic ground truth] "
            f"latent={len(latent):,} observed_overlap={len(observed):,}"
        )
        return {"latent": latent, "observed": observed}

    @staticmethod
    def _reference_profile_metrics(
        prediction: np.ndarray,
        reference: np.ndarray,
        valid_mask: np.ndarray,
    ) -> tuple[float, float] | None:
        length = min(prediction.size, reference.size, valid_mask.size)
        if length < 2:
            return None
        valid = valid_mask[:length] & np.isfinite(prediction[:length])
        if int(valid.sum()) < 2:
            return None
        pred = prediction[:length][valid].astype(np.float64)
        ref = reference[:length][valid].astype(np.float64)
        pred_mean = float(pred.mean())
        ref_mean = float(ref.mean())
        if pred_mean <= 0.0 or ref_mean <= 0.0:
            return None
        pred = pred / pred_mean
        ref = ref / ref_mean
        mse = float(np.mean((pred - ref) ** 2))
        pred_centered = pred - pred.mean()
        ref_centered = ref - ref.mean()
        denominator = float(
            np.sqrt(np.sum(pred_centered**2) * np.sum(ref_centered**2))
        )
        pcc = (
            float(np.sum(pred_centered * ref_centered) / denominator)
            if denominator > 0.0
            else float("nan")
        )
        return mse, pcc

    def _record_synthetic_ground_truth_metrics(self, out: dict[str, Any]) -> None:
        if self._synthetic_ground_truth is None:
            return
        ids = self._transcript_ids_as_strings(out["ids"], out["mask"].shape[0])
        predictions = out["extras"]["L_bio"].detach().float().cpu().numpy()
        masks = out["mask"].detach().bool().cpu().numpy()
        references = self._synthetic_ground_truth
        for index, transcript_id in enumerate(ids):
            row = self._synthetic_ground_truth_epoch.setdefault(
                transcript_id,
                {"latent_mse": [], "latent_pcc": [], "observed_mse": [], "observed_pcc": []},
            )
            latent_metrics = self._reference_profile_metrics(
                predictions[index],
                references["latent"].get(transcript_id, np.empty(0, dtype=np.float32)),
                masks[index],
            )
            if latent_metrics is not None:
                row["latent_mse"].append(latent_metrics[0])
                row["latent_pcc"].append(latent_metrics[1])
            observed_profile = references["observed"].get(transcript_id)
            if observed_profile is not None:
                observed_metrics = self._reference_profile_metrics(
                    predictions[index],
                    observed_profile,
                    masks[index],
                )
                if observed_metrics is not None:
                    row["observed_mse"].append(observed_metrics[0])
                    row["observed_pcc"].append(observed_metrics[1])

    def _log_synthetic_ground_truth_metrics(self) -> None:
        # Every DDP rank must execute the same collective sequence.  A rank can
        # legitimately have no locally valid reference rows (for example when
        # its validation shard contains no overlapping synthetic transcript),
        # so do not return based on the local epoch dictionary.
        if self._synthetic_ground_truth is None:
            return
        # These are optional diagnostics, not part of the optimized objective.
        # Do not introduce a second, manually ordered collective stream beside
        # Lightning's epoch metric reductions under DDP. Run them on a single
        # process instead; distributed training remains collective-safe.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return
        metric_names = ("latent_mse", "latent_pcc", "observed_mse", "observed_pcc")
        local_pairs: list[float] = []
        for metric_name in metric_names:
            per_transcript = [
                float(np.mean(values[metric_name]))
                for values in self._synthetic_ground_truth_epoch.values()
                if values[metric_name]
            ]
            local_pairs.extend(
                [float(np.sum(per_transcript)), float(len(per_transcript))]
            )
        local_pairs.append(float(len(self._synthetic_ground_truth_epoch)))
        totals = torch.tensor(
            local_pairs,
            device=self.device,
            dtype=torch.float64,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        for index, metric_name in enumerate(metric_names):
            metric_sum = totals[2 * index]
            metric_count = totals[2 * index + 1]
            if float(metric_count.item()) <= 0.0:
                continue
            self.log(
                f"val/synthetic_ground_truth/{metric_name}",
                (metric_sum / metric_count).float(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )
        self.log(
            "val/synthetic_ground_truth/transcripts",
            totals[-1].float(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )

    def configure_grouped_optimizer_batch_logging(
        self,
        *,
        plan: dict[str, Any],
        enabled: bool = True,
    ) -> None:
        """Attach the pre-Trainer grouped accumulation plan to this run."""
        self._grouped_batch_logging_enabled = bool(enabled)
        self._grouped_optimizer_batch_plan = dict(plan)
        # The project uses weights-only checkpoints, so the resolved Hydra config
        # and JSON manifest are the durable provenance. Keep the same compact plan
        # in Lightning hyperparameters when a full checkpoint is requested.
        self.hparams["grouped_optimizer_batch_plan"] = dict(plan)

    def _build_loss(self) -> nn.Module:
        loss_cfg = self.config.loss

        def loss_cfg_get(name: str, default: Any = None) -> Any:
            try:
                return getattr(loss_cfg, name)
            except (AttributeError, KeyError):
                return default

        sequence_reduction = str(loss_cfg_get("nb_sequence_reduction", "mean"))
        length_temper_gamma = float(loss_cfg_get("nb_length_temper_gamma", 0.85))
        length_temper_ref = float(loss_cfg_get("nb_length_temper_ref", 1000.0))
        length_temper_min_weight = float(
            loss_cfg_get("nb_length_temper_min_weight", 0.5)
        )
        length_temper_max_weight = float(
            loss_cfg_get("nb_length_temper_max_weight", 2.0)
        )

        return NegativeBinomialProfileLoss(
            eps=loss_cfg.eps,
            log_alpha_min=float(loss_cfg_get("nb_log_alpha_min", -5.0)),
            log_alpha_max=float(loss_cfg_get("nb_log_alpha_max", 3.0)),
            sequence_reduction=sequence_reduction,
            length_temper_gamma=length_temper_gamma,
            length_temper_ref=length_temper_ref,
            length_temper_min_weight=length_temper_min_weight,
            length_temper_max_weight=length_temper_max_weight,
        )

    # ============================================================
    # Forward / batch handling
    # ============================================================

    def _forward_batch(self, batch) -> dict[str, Any]:
        if len(batch) < 14:
            raise ValueError(
                "The current collate schema requires quality metadata, a local "
                "transcript-group tensor, and mandatory replica tensors; got "
                f"{len(batch)} batch fields."
            )
        (
            dataset_ids,
            ids,
            seq_packed,
            target,
            lengths,
            mask,
            codon_ids,
            css,
            sample_weights,
            dataset_quality_ranks,
            dataset_quality_weights,
            transcript_group_index,
        ) = batch[:12]
        if not (
            torch.is_tensor(dataset_quality_ranks)
            and dataset_quality_ranks.ndim == 1
            and torch.is_tensor(dataset_quality_weights)
            and dataset_quality_weights.ndim == 1
            and torch.is_tensor(transcript_group_index)
            and transcript_group_index.ndim == 1
            and not transcript_group_index.is_floating_point()
        ):
            raise ValueError(
                "Invalid batch metadata: expected one-dimensional dataset ranks, "
                "dataset quality weights, and integer transcript group indices."
            )
        optional_values = list(batch[12:])
        execution_microbatch_metadata = None
        if optional_values and (
            optional_values[-1] is None
            or isinstance(optional_values[-1], dict)
        ):
            execution_microbatch_metadata = optional_values.pop()

        dataset_bias_sequence_features = None
        if len(optional_values) == 2:
            replica_profiles, replica_mask = optional_values
        elif len(optional_values) == 3:
            dataset_bias_sequence_features, replica_profiles, replica_mask = optional_values
        else:
            raise ValueError(
                "Every batch must contain replica_profiles and replica_mask, with "
                "an optional dataset-bias tensor before them; got "
                f"{len(optional_values)} trailing fields."
            )
        if replica_profiles.ndim != 3 or replica_mask.ndim != 2:
            raise ValueError(
                "Expected replica_profiles [B, R, T] and replica_mask [B, R], "
                f"got {tuple(replica_profiles.shape)} and {tuple(replica_mask.shape)}."
            )

        mu, log_sigma, extras = self.model(
            x_packed=seq_packed,
            codon_ids=codon_ids,
            id_datasets=dataset_ids,
            mask=mask,
            target=target,
            sample_ids=ids,
            transcript_group_index=transcript_group_index,
            dataset_bias_sequence_features=dataset_bias_sequence_features,
            dataset_quality_weights=dataset_quality_weights,
        )

        out = {
            "dataset_ids": dataset_ids,
            "ids": ids,
            "lengths": lengths,
            "mask": mask.bool(),
            "target": target,
            "codon_ids": codon_ids,
            "dataset_bias_sequence_features": dataset_bias_sequence_features,
            "css": css,
            "sample_weights": sample_weights,
            "dataset_quality_ranks": dataset_quality_ranks,
            "dataset_quality_weights": dataset_quality_weights,
            "transcript_group_index": transcript_group_index,
            "mu": mu,
            "log_sigma": log_sigma,
            "extras": extras,
        }
        out["replica_profiles"] = replica_profiles
        out["replica_mask"] = replica_mask
        out["execution_microbatch_metadata"] = execution_microbatch_metadata
        return out

    def forward_batch(self, batch) -> dict[str, Any]:
        return self._forward_batch(batch)

    # ============================================================
    # Loss / metrics
    # ============================================================

    def _aggregate_per_sample(
        self,
        values: torch.Tensor,
        dataset_ids: torch.Tensor,
        sample_weights: torch.Tensor,
        transcript_group_ids: torch.Tensor,
        mode: str | None = None,
    ) -> torch.Tensor:
        """Apply the configured pair-to-batch reduction."""
        return reduce_per_sample_quantity(
            values=values,
            sample_weights=sample_weights,
            dataset_ids=dataset_ids,
            transcript_group_ids=transcript_group_ids,
            mode=self.sample_reduction if mode is None else mode,
            eps=self.eps,
        )

    def _pcc_alpha(
        self,
        log_alpha: torch.Tensor,
        target_shape: torch.Size | tuple[int, ...],
    ) -> torch.Tensor:
        bounded_log_alpha = self.loss_fn._log_alpha_from_model_output(
            log_sigma=log_alpha,
            target_shape=target_shape,
        )
        alpha = torch.exp(bounded_log_alpha)
        if self.pcc_detach_alpha:
            alpha = alpha.detach()
        return alpha

    def _nb_vst(self, x: torch.Tensor, log_alpha: torch.Tensor) -> torch.Tensor:
        alpha = self._pcc_alpha(log_alpha, x.shape).to(device=x.device, dtype=x.dtype)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        eps = self.eps
        return 2.0 / torch.sqrt(alpha + eps) * torch.asinh(
            torch.sqrt(alpha * x + eps)
        )

    def _pcc_loss_per_sample(
        self,
        out: dict[str, Any],
        *,
        transform: str,
    ) -> dict[str, torch.Tensor]:
        """Calculate one explicit PCC component without a mode selector."""
        mu = out["mu"].float()
        target = torch.nan_to_num(
            out["target"].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        mask = out["mask"].bool() & torch.isfinite(out["target"].float())
        log_alpha = out.get("log_sigma")

        def masked_position_mean(v: torch.Tensor) -> torch.Tensor:
            mask_f = mask.to(dtype=v.dtype)
            return (v * mask_f).sum() / mask_f.sum().clamp_min(1.0)

        alpha_diag = torch.ones_like(mu)
        if transform == "raw":
            x_pcc = mu
            y_pcc = target
        elif transform == "nb_vst":
            if not torch.is_tensor(log_alpha):
                raise KeyError("NB-VST PCC requires model output 'log_sigma'.")
            log_alpha_f = log_alpha.float()
            alpha_diag = self._pcc_alpha(log_alpha_f, mu.shape).to(
                device=mu.device,
                dtype=mu.dtype,
            )
            x_pcc = self._nb_vst(mu, log_alpha_f)
            y_pcc = self._nb_vst(target, log_alpha_f)
        else:
            raise ValueError(
                "PCC transform must be either 'raw' or 'nb_vst', "
                f"got {transform!r}."
            )

        pcc_out = masked_pcc(
            x=x_pcc,
            y=y_pcc,
            mask=mask,
            min_target_var=self.min_pcc_target_var,
            eps=self.eps,
        )
        valid = pcc_out["valid"]
        loss_per_sample = torch.where(
            valid,
            1.0 - pcc_out["pcc_per_sample"],
            torch.zeros_like(pcc_out["pcc_per_sample"]),
        )
        return {
            "loss_per_sample": loss_per_sample,
            "pcc_per_sample": pcc_out["pcc_per_sample"],
            "valid": valid,
            "valid_fraction": pcc_out["valid_fraction"],
            "target_var_mean": pcc_out["target_var_mean"],
            "alpha_mean": masked_position_mean(alpha_diag),
            "alpha_min": alpha_diag[mask].amin() if bool(mask.any()) else mu.new_tensor(0.0),
            "alpha_max": alpha_diag[mask].amax() if bool(mask.any()) else mu.new_tensor(0.0),
        }

    def _pearson_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1.0e-8,
    ) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        mask_b = mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
        weight_f = mask_b.to(dtype=pred.dtype)

        pred = torch.where(mask_b, pred, torch.zeros_like(pred))
        target = torch.where(mask_b, target, torch.zeros_like(target))

        valid_len = weight_f.sum(dim=1).clamp_min(1.0)
        pred_mean = (pred * weight_f).sum(dim=1, keepdim=True) / valid_len.unsqueeze(1)
        target_mean = (target * weight_f).sum(dim=1, keepdim=True) / valid_len.unsqueeze(1)

        pred_c = torch.where(mask_b, pred - pred_mean, torch.zeros_like(pred))
        target_c = torch.where(mask_b, target - target_mean, torch.zeros_like(target))

        numerator = (weight_f * pred_c * target_c).sum(dim=1)
        pred_var = (weight_f * pred_c.pow(2)).sum(dim=1)
        target_var = (weight_f * target_c.pow(2)).sum(dim=1)
        denom = torch.sqrt((pred_var * target_var).clamp_min(0.0) + eps * eps)

        pcc = numerator / denom
        valid = (pred_var > eps) & (target_var > eps)
        pcc = torch.where(valid, pcc, torch.zeros_like(pcc))
        return torch.nan_to_num(pcc, nan=0.0, posinf=0.0, neginf=0.0)

    def _gamma_regularization_per_sample(
        self,
        out: dict[str, Any],
    ) -> torch.Tensor:
        target = out["target"]
        zeros = torch.zeros(
            target.shape[0],
            device=target.device,
            dtype=target.dtype,
        )
        extras = out["extras"]
        mask = out["mask"].bool()
        mask_f = mask.to(dtype=torch.float32)
        log_gamma = extras.get("log_gamma")
        if torch.is_tensor(log_gamma):
            log_gamma = log_gamma.float()
            gamma_mask = mask & torch.isfinite(log_gamma)
            gamma_mask_f = gamma_mask.to(dtype=log_gamma.dtype)
            gamma_len = gamma_mask_f.sum(dim=1).clamp_min(1.0)
            gamma_reg = (log_gamma.pow(2) * gamma_mask_f).sum(dim=1) / gamma_len
        else:
            gamma_reg = zeros
        return gamma_reg

    def _apply_pcc_floor(self, mu: torch.Tensor) -> torch.Tensor:
        """Treat predictions below the floor as 0 ("undetected") for PCC only.

        Honest zero handling: uses only the prediction, never the target. The
        likelihood is unchanged, so the model must genuinely push mu down at
        zeros to benefit. floor <= 0 is a no-op.
        """
        floor = float(self.pcc_prediction_floor)
        if floor <= 0.0:
            return mu
        return torch.where(mu >= floor, mu, torch.zeros_like(mu))

    @staticmethod
    def _average_over_valid_replicas(
        values: torch.Tensor,
        replica_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = values.reshape(replica_mask.shape[0], replica_mask.shape[1])
        valid = replica_mask.bool().to(device=values.device)
        valid_f = valid.to(dtype=values.dtype)
        return (values * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)

    def _compute_consensus_loss_terms(
        self,
        out: dict[str, Any],
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, Any]:
        # PCC is measured on the floored prediction (values below the floor are
        # treated as 0), so the model is rewarded for pushing mu down at zeros.
        # This uses only the prediction, never the target. Both optimized PCC
        # components compare this pair-level prediction with the arithmetic
        # replica consensus supplied as ``target``.
        mu_floored = self._apply_pcc_floor(out["mu"].float())
        raw_mu_pcc_per_sample = self._pearson_per_sample(
            pred=mu_floored,
            target=target,
            mask=mask,
        )

        pcc_out = {
            **out,
            "mu": mu_floored,
            "target": target,
            "mask": mask,
        }

        def pcc_component(
            *,
            coefficient: float,
            transform: str,
        ) -> dict[str, torch.Tensor]:
            if coefficient > 0.0:
                return self._pcc_loss_per_sample(pcc_out, transform=transform)

            batch_size = int(target.shape[0])
            zero_sample = target.new_zeros(batch_size)
            zero_scalar = target.new_tensor(0.0)
            return {
                "loss_per_sample": zero_sample,
                "pcc_per_sample": zero_sample,
                "valid": torch.zeros(
                    batch_size,
                    device=target.device,
                    dtype=torch.bool,
                ),
                "valid_fraction": zero_scalar,
                "target_var_mean": zero_scalar,
                "alpha_mean": zero_scalar,
                "alpha_min": zero_scalar,
                "alpha_max": zero_scalar,
            }

        raw_pcc_diag = pcc_component(
            coefficient=self.consensus_raw_pcc_weight,
            transform="raw",
        )
        nb_vst_pcc_diag = pcc_component(
            coefficient=self.consensus_nb_vst_pcc_weight,
            transform="nb_vst",
        )

        with torch.no_grad():
            likelihood_positive_mean = self.loss_fn.positive_mean_from_params(
                mu=mu_floored,
                log_sigma=out["log_sigma"].float(),
            ).float()
        likelihood_mu_pcc_per_sample = self._pearson_per_sample(
            pred=likelihood_positive_mean,
            target=target,
            mask=mask,
        )
        log1p_mse_per_sample = self._masked_mean(
            (torch.log1p(likelihood_positive_mean) - torch.log1p(target)).pow(2),
            mask,
        )
        return {
            "raw_pcc_diag": raw_pcc_diag,
            "nb_vst_pcc_diag": nb_vst_pcc_diag,
            "raw_mu_pcc_per_sample": raw_mu_pcc_per_sample,
            "likelihood_mu_pcc_per_sample": likelihood_mu_pcc_per_sample,
            "log1p_mse_per_sample": log1p_mse_per_sample,
        }

    def _compute_replica_loss_terms(
        self,
        out: dict[str, Any],
    ) -> dict[str, Any]:
        replica_profiles = out.get("replica_profiles")
        replica_mask = out.get("replica_mask")
        if not (torch.is_tensor(replica_profiles) and torch.is_tensor(replica_mask)):
            raise RuntimeError(
                "Replica targets are mandatory: every batch must contain "
                "replica_profiles and replica_mask."
            )

        target_reps = replica_profiles.float()
        rep_mask = replica_mask.bool().to(device=target_reps.device)
        if target_reps.ndim != 3 or rep_mask.ndim != 2:
            raise ValueError(
                "Expected replica_profiles [B, R, T] and replica_mask [B, R], "
                f"got {tuple(target_reps.shape)} and {tuple(rep_mask.shape)}."
            )
        B, R, T = target_reps.shape
        if tuple(rep_mask.shape) != (B, R):
            raise ValueError(
                f"replica_mask must have shape {(B, R)}, got {tuple(rep_mask.shape)}."
            )
        if not bool(rep_mask.any(dim=1).all()):
            raise ValueError("Every sample must contain at least one valid replica.")

        base_mask = out["mask"].bool().to(device=target_reps.device)

        finite_target = torch.isfinite(target_reps)
        valid_pos = base_mask.unsqueeze(1) & rep_mask.unsqueeze(-1) & finite_target
        valid_pos_f = valid_pos.to(dtype=target_reps.dtype)
        target_clean = torch.nan_to_num(
            target_reps,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)

        valid_len = valid_pos_f.sum(dim=2).clamp_min(1.0)
        scale_rep = (target_clean * valid_pos_f).sum(dim=2) / valid_len
        scale_rep = scale_rep.clamp_min(self.eps)

        extras = out["extras"]
        if "normalized_shape" not in extras:
            raise KeyError(
                "Model output is missing extras['normalized_shape']; replica NB "
                "must reuse the exact final shape from the consensus forward."
            )
        shape = extras["normalized_shape"].float().to(device=target_reps.device)
        if tuple(shape.shape) != (B, T):
            raise ValueError(
                "normalized_shape must match [B, T] for replica NB; got "
                f"{tuple(shape.shape)} instead of {(B, T)}."
            )
        shape = torch.where(
            base_mask & torch.isfinite(shape),
            shape.clamp_min(0.0),
            torch.zeros_like(shape),
        )
        mu_rep = scale_rep.unsqueeze(-1) * shape.unsqueeze(1)
        mu_rep = torch.nan_to_num(
            mu_rep,
            nan=self.eps,
            posinf=float(torch.finfo(mu_rep.dtype).max),
            neginf=self.eps,
        )
        mu_rep = torch.where(valid_pos, mu_rep, torch.ones_like(mu_rep))

        log_sigma_rep = (
            out["log_sigma"]
            .float()
            .to(device=target_reps.device)
            .unsqueeze(1)
            .expand(B, R, T)
        )

        flat_target = target_clean.reshape(B * R, T)
        flat_mu = mu_rep.reshape(B * R, T)
        flat_mask = valid_pos.reshape(B * R, T)
        flat_log_sigma = log_sigma_rep.reshape(B * R, T)

        zero_flat = flat_mu.new_zeros(B * R)
        if self.replica_nb_weight > 0.0:
            nll_flat = self.loss_fn(
                mu_phys=flat_mu,
                log_sigma=flat_log_sigma,
                y_true=flat_target,
                mask=flat_mask,
                return_per_sample=True,
            )
        else:
            # A zero coefficient disables the NB-NLL branch completely.
            nll_flat = zero_flat
        nll_per_sample = self._average_over_valid_replicas(nll_flat, rep_mask)
        replica_counts = rep_mask.to(dtype=target_reps.dtype).sum(dim=1)
        return {
            "nll_per_sample": nll_per_sample,
            "metrics": {
                "replica_count_mean": replica_counts.mean(),
            },
        }

    def _compute_loss_and_metrics(self, out: dict[str, Any]) -> dict[str, torch.Tensor]:
        dataset_ids = out["dataset_ids"]
        sample_weights = out["sample_weights"]
        transcript_group_ids = out["transcript_group_index"]
        mask = out["mask"].bool() & torch.isfinite(out["target"].float())
        target = torch.nan_to_num(out["target"].float(), nan=0.0, posinf=0.0, neginf=0.0)

        consensus_terms = self._compute_consensus_loss_terms(out, target, mask)
        replica_terms = self._compute_replica_loss_terms(out)

        # NB2 retains raw-replica supervision. Shape-oriented PCC is evaluated
        # once per pair against the arithmetic replica consensus.
        nll_per_sample = replica_terms["nll_per_sample"]
        raw_pcc_diag = consensus_terms["raw_pcc_diag"]
        nb_vst_pcc_diag = consensus_terms["nb_vst_pcc_diag"]
        pcc_raw_loss_per_sample = raw_pcc_diag["loss_per_sample"]
        pcc_nb_vst_loss_per_sample = nb_vst_pcc_diag["loss_per_sample"]
        raw_mu_pcc_per_sample = consensus_terms["raw_mu_pcc_per_sample"]
        likelihood_mu_pcc_per_sample = consensus_terms[
            "likelihood_mu_pcc_per_sample"
        ]
        log1p_mse_per_sample = consensus_terms["log1p_mse_per_sample"]

        extras = out["extras"]
        # L_bio is the shared biological profile, so compare it directly with
        # the consensus target even when replicas provide the main objective.
        # This is a diagnostic only: detaching here makes it explicit that the
        # correlation cannot contribute gradients or otherwise affect training.
        with torch.no_grad():
            L_bio_pcc_per_sample = self._pearson_per_sample(
                pred=extras["L_bio"].detach(),
                target=target.detach(),
                mask=mask,
            )
        if self.gamma_reg_weight > 0.0:
            gamma_reg_per_sample = self._gamma_regularization_per_sample(out)
        else:
            # A zero coefficient disables gamma-regularizer computation.
            gamma_reg_per_sample = nll_per_sample.new_zeros(nll_per_sample.shape)

        effective_replica_nb_weight = nll_per_sample.new_tensor(
            self.replica_nb_weight
        )
        effective_consensus_raw_pcc_weight = nll_per_sample.new_tensor(
            self.consensus_raw_pcc_weight
        )
        effective_consensus_nb_vst_pcc_weight = nll_per_sample.new_tensor(
            self.consensus_nb_vst_pcc_weight
        )
        effective_gamma_reg_weight = nll_per_sample.new_tensor(
            self.gamma_reg_weight
        )
        pcc_loss_per_sample = (
            effective_consensus_raw_pcc_weight * pcc_raw_loss_per_sample
            + effective_consensus_nb_vst_pcc_weight * pcc_nb_vst_loss_per_sample
        )
        per_sample_total_loss = (
            effective_replica_nb_weight * nll_per_sample
            + pcc_loss_per_sample
            + effective_gamma_reg_weight * gamma_reg_per_sample
        )

        loss = self._aggregate_per_sample(
            per_sample_total_loss,
            dataset_ids,
            sample_weights,
            transcript_group_ids,
        )
        # All three diagnostics reuse the same detached pair values. Only the
        # selected objective above remains attached to the optimization graph.
        loss_by_mode = {
            mode: self._aggregate_per_sample(
                per_sample_total_loss.detach(),
                dataset_ids,
                sample_weights,
                transcript_group_ids,
                mode=mode,
            )
            for mode in sorted(SAMPLE_REDUCTION_MODES)
        }
        loss_per_sample = per_sample_total_loss

        nll = self._aggregate_per_sample(
            nll_per_sample,
            dataset_ids,
            sample_weights,
            transcript_group_ids,
        )
        pcc_loss = self._aggregate_per_sample(
            pcc_loss_per_sample,
            dataset_ids,
            sample_weights,
            transcript_group_ids,
        )
        gamma_reg = self._aggregate_per_sample(
            gamma_reg_per_sample,
            dataset_ids,
            sample_weights,
            transcript_group_ids,
        )
        pcc_raw_loss = self._aggregate_per_sample(
            pcc_raw_loss_per_sample,
            dataset_ids,
            sample_weights,
            transcript_group_ids,
        )
        pcc_nb_vst_loss = self._aggregate_per_sample(
            pcc_nb_vst_loss_per_sample,
            dataset_ids,
            sample_weights,
            transcript_group_ids,
        )

        def aggregate_active(v: torch.Tensor) -> torch.Tensor:
            return self._aggregate_per_sample(
                v,
                dataset_ids,
                sample_weights,
                transcript_group_ids,
            )

        replica_nll = aggregate_active(replica_terms["nll_per_sample"])
        pcc_weight_sum = (
            self.consensus_raw_pcc_weight
            + self.consensus_nb_vst_pcc_weight
        )
        if pcc_weight_sum > 0.0:
            raw_active_weight = (
                raw_pcc_diag["valid"].to(dtype=nll_per_sample.dtype)
                * self.consensus_raw_pcc_weight
            )
            nb_vst_active_weight = (
                nb_vst_pcc_diag["valid"].to(dtype=nll_per_sample.dtype)
                * self.consensus_nb_vst_pcc_weight
            )
            active_pcc_weight = raw_active_weight + nb_vst_active_weight
            consensus_pcc_per_sample = torch.where(
                active_pcc_weight > 0.0,
                (
                    raw_active_weight * raw_pcc_diag["pcc_per_sample"]
                    + nb_vst_active_weight * nb_vst_pcc_diag["pcc_per_sample"]
                )
                / active_pcc_weight.clamp_min(self.eps),
                torch.zeros_like(nll_per_sample),
            )
        else:
            consensus_pcc_per_sample = nll_per_sample.new_zeros(
                nll_per_sample.shape
            )
        consensus_pcc_value = aggregate_active(consensus_pcc_per_sample)
        consensus_pcc_loss = pcc_loss
        pcc_value = consensus_pcc_value
        pcc_valid = raw_pcc_diag["valid"] | nb_vst_pcc_diag["valid"]
        pcc_valid_fraction = pcc_valid.to(dtype=target.dtype).mean()
        if pcc_weight_sum > 0.0:
            pcc_target_var_mean = (
                self.consensus_raw_pcc_weight
                * raw_pcc_diag["target_var_mean"]
                + self.consensus_nb_vst_pcc_weight
                * nb_vst_pcc_diag["target_var_mean"]
            ) / pcc_weight_sum
        else:
            pcc_target_var_mean = target.new_tensor(0.0)

        mask_f = mask.to(dtype=torch.float32)
        valid_len = mask_f.sum(dim=1).clamp_min(1.0)

        def pos_mean(t: torch.Tensor) -> torch.Tensor:
            return (t.float() * mask_f).sum(dim=1) / valid_len

        def pos_max(t: torch.Tensor) -> torch.Tensor:
            return t.float().masked_fill(~mask, float("-inf")).amax(dim=1)

        valid_count = mask_f.sum().clamp_min(1.0)
        sequence_reduction = str(getattr(self.loss_fn, "sequence_reduction", "mean"))
        sequence_reduction_mode_id = {
            "mean": 0.0,
            "sum": 1.0,
            "length_tempered": 2.0,
        }.get(sequence_reduction, -1.0)
        length_weight = (
            sequence_length_temper_weights(
                mask,
                gamma=float(getattr(self.loss_fn, "length_temper_gamma", 1.0)),
                length_ref=float(getattr(self.loss_fn, "length_temper_ref", 1000.0)),
                min_weight=float(getattr(self.loss_fn, "length_temper_min_weight", 1.0)),
                max_weight=float(getattr(self.loss_fn, "length_temper_max_weight", 1.0)),
                eps=self.eps,
            ).to(device=target.device, dtype=target.dtype)
            if sequence_reduction == "length_tempered"
            else torch.ones_like(valid_len, dtype=target.dtype)
        )
        sample_lengths = mask_f.sum(dim=1)

        def length_weight_bin_mean(bin_mask: torch.Tensor) -> torch.Tensor:
            if not bool(bin_mask.any()):
                return length_weight.new_tensor(0.0)
            return length_weight[bin_mask].mean()

        def global_pos_mean(t: torch.Tensor) -> torch.Tensor:
            return (t.float() * mask_f).sum() / valid_count

        def global_pos_max(t: torch.Tensor) -> torch.Tensor:
            return t.float().masked_fill(~mask, float("-inf")).amax()

        def global_pos_min(t: torch.Tensor) -> torch.Tensor:
            return t.float().masked_fill(~mask, float("inf")).amin()

        def global_pos_std(t: torch.Tensor) -> torch.Tensor:
            vals = t.float()[mask]
            if vals.numel() <= 1:
                return target.new_tensor(0.0)
            return vals.std(unbiased=False)

        def masked_fraction(flag: torch.Tensor) -> torch.Tensor:
            return (flag.to(device=mask.device).bool() & mask).to(dtype=torch.float32).sum() / valid_count

        def finite_corr_or_zero(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x.float().reshape(-1)
            y = y.float().reshape(-1).to(device=x.device)
            valid_xy = torch.isfinite(x) & torch.isfinite(y)
            if int(valid_xy.sum().detach().cpu().item()) < 2:
                return x.new_tensor(0.0)
            x = x[valid_xy]
            y = y[valid_xy]
            x = x - x.mean()
            y = y - y.mean()
            denom = torch.sqrt(x.pow(2).sum() * y.pow(2).sum()).clamp_min(self.eps)
            return torch.nan_to_num((x * y).sum() / denom, nan=0.0, posinf=0.0, neginf=0.0)

        def masked_abs_mean(t: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
            selected = selected.to(device=mask.device).bool() & mask
            if not bool(selected.any()):
                return target.new_tensor(0.0)
            return t.float().abs()[selected].mean()

        gamma_raw_diag = extras.get("gamma_raw", torch.ones_like(extras["L_bio"]))
        log_gamma_raw_diag = extras.get(
            "log_gamma_raw",
            torch.zeros_like(extras["L_bio"]),
        )
        gamma_diag = extras.get("gamma", torch.ones_like(extras["L_bio"]))
        log_gamma_diag = extras.get(
            "log_gamma",
            torch.zeros_like(extras["L_bio"]),
        )
        gamma_reliability_diag = extras.get(
            "gamma_centering_reliability",
            torch.zeros_like(extras["L_bio"]),
        )
        gamma_eligible_diag = extras.get(
            "gamma_centering_eligible",
            torch.zeros_like(extras["L_bio"], dtype=torch.bool),
        )
        gamma_applied_diag = extras.get(
            "gamma_centering_applied",
            torch.zeros_like(extras["L_bio"], dtype=torch.bool),
        )
        gamma_distinct_diag = extras.get(
            "gamma_num_distinct_datasets",
            torch.zeros_like(extras["L_bio"]),
        )
        gamma_total_reliability_diag = extras.get(
            "gamma_total_reliability",
            torch.zeros_like(extras["L_bio"]),
        )
        gamma_constraint_error_diag = extras.get(
            "gamma_centering_constraint_error",
            torch.zeros_like(extras["L_bio"]),
        )
        gamma_positional_constraint_error_diag = extras.get(
            "gamma_positional_gauge_constraint_error",
            torch.zeros_like(extras["L_bio"]),
        )
        gamma_skipped_mask = mask & (~gamma_applied_diag.to(device=mask.device).bool())
        mean_log_gamma_per_sample = pos_mean(log_gamma_diag)
        std_log_gamma_per_sample = torch.sqrt(
            pos_mean((log_gamma_diag - mean_log_gamma_per_sample.reshape(-1, 1)).pow(2))
        )
        library_depth = (target * mask_f).sum(dim=1)
        # Retain an unweighted pair-micro diagnostic alongside the configured
        # reliability-weighted sample reduction.
        mu_pcc_unweighted = raw_mu_pcc_per_sample.mean()
        likelihood_mu_pcc_unweighted = likelihood_mu_pcc_per_sample.mean()

        metrics: dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_per_sample": loss_per_sample,
            "loss_global_weighted": loss_by_mode["global_weighted"],
            "loss_dataset_balanced": loss_by_mode["dataset_balanced"],
            "loss_transcript_balanced": loss_by_mode["transcript_balanced"],
            "nll": nll,
            "nll_per_sample": nll_per_sample,
            "pcc_loss": pcc_loss,
            "pcc_loss_per_sample": pcc_loss_per_sample,
            "consensus_pcc_loss": consensus_pcc_loss,
            "consensus_pcc_value": consensus_pcc_value,
            "replica_nll": replica_nll,
            "gamma_reg": gamma_reg,
            "gamma_reg_per_sample": gamma_reg_per_sample,
            "gamma_global_regularizer": gamma_reg,
            "gamma_reg_weight": effective_gamma_reg_weight,
            "pcc_loss_total": pcc_loss,
            "pcc_value": pcc_value,
            "replica_nb_weight": effective_replica_nb_weight,
            "consensus_raw_pcc_weight": effective_consensus_raw_pcc_weight,
            "consensus_nb_vst_pcc_weight": (
                effective_consensus_nb_vst_pcc_weight
            ),
            "replica_nb_contribution": effective_replica_nb_weight * nll,
            "pcc_valid_fraction": pcc_valid_fraction,
            "pcc_raw_loss": pcc_raw_loss,
            "pcc_raw_loss_per_sample": pcc_raw_loss_per_sample,
            "pcc_raw_value": aggregate_active(raw_pcc_diag["pcc_per_sample"]),
            "pcc_raw_valid_fraction": raw_pcc_diag["valid_fraction"],
            "pcc_nb_vst_loss": pcc_nb_vst_loss,
            "pcc_nb_vst_loss_per_sample": pcc_nb_vst_loss_per_sample,
            "pcc_nb_vst_value": aggregate_active(
                nb_vst_pcc_diag["pcc_per_sample"]
            ),
            "pcc_nb_vst_valid_fraction": nb_vst_pcc_diag["valid_fraction"],
            "pcc_raw_contribution": (
                effective_consensus_raw_pcc_weight * pcc_raw_loss
            ),
            "pcc_nb_vst_contribution": (
                effective_consensus_nb_vst_pcc_weight * pcc_nb_vst_loss
            ),
            "pcc_target_var_mean": pcc_target_var_mean,
            "pcc_alpha_mean": nb_vst_pcc_diag["alpha_mean"],
            "pcc_alpha_min": nb_vst_pcc_diag["alpha_min"],
            "pcc_alpha_max": nb_vst_pcc_diag["alpha_max"],
            # The legacy name remains an unweighted compatibility alias.
            "mu_pcc": mu_pcc_unweighted,
            "mu_pcc_unweighted": mu_pcc_unweighted,
            "mu_pcc_per_sample": raw_mu_pcc_per_sample,
            "L_bio_pcc": self._aggregate_per_sample(
                L_bio_pcc_per_sample,
                dataset_ids,
                sample_weights,
                transcript_group_ids,
            ),
            "L_bio_pcc_per_sample": L_bio_pcc_per_sample,
            "nb_sequence_reduction_mode": target.new_tensor(sequence_reduction_mode_id),
            "nb_length_temper_gamma": target.new_tensor(
                float(getattr(self.loss_fn, "length_temper_gamma", 1.0))
            ),
            "nb_length_temper_ref": target.new_tensor(
                float(getattr(self.loss_fn, "length_temper_ref", 1000.0))
            ),
            "nb_length_weight_mean": length_weight.mean(),
            "nb_length_weight_min": length_weight.amin(),
            "nb_length_weight_max": length_weight.amax(),
            "nb_length_weight_short": length_weight_bin_mean(sample_lengths < 500.0),
            "nb_length_weight_medium": length_weight_bin_mean(
                (sample_lengths >= 500.0) & (sample_lengths < 1500.0)
            ),
            "nb_length_weight_long": length_weight_bin_mean(sample_lengths >= 1500.0),
            "likelihood_mu_pcc": likelihood_mu_pcc_unweighted,
            "likelihood_mu_pcc_unweighted": likelihood_mu_pcc_unweighted,
            "likelihood_mu_pcc_per_sample": likelihood_mu_pcc_per_sample,
            "log1p_mse": self._aggregate_per_sample(
                log1p_mse_per_sample,
                dataset_ids,
                sample_weights,
                transcript_group_ids,
            ),
            "mean_ratio": extras["mean_ratio"].float().mean(),
            "mu_mean": extras["mu_mean"].float().mean(),
            "target_mean": extras["target_mean"].float().mean(),
            "J_mean": extras["J"].float().mean(),
            "J_min": extras["J"].float().amin(),
            "J_max": extras["J"].float().amax(),
            "lambda_bio_mean": global_pos_mean(extras["lambda_bio"]),
            "lambda_bio_max": global_pos_max(extras["lambda_bio"]),
            "lambda_bio_min": global_pos_min(extras["lambda_bio"]),
            "L_bio_mean": global_pos_mean(extras["L_bio"]),
            "L_bio_max": global_pos_max(extras["L_bio"]),
            "rho_mean": pos_mean(extras["rho"]).mean(),
            "rho_max": pos_max(extras["rho"]).mean(),
            "gamma_mean": global_pos_mean(
                extras.get("gamma", torch.ones_like(extras["L_bio"]))
            ),
            "gamma_raw_mean": global_pos_mean(gamma_raw_diag),
            "gamma_raw_geometric_mean": torch.exp(global_pos_mean(log_gamma_raw_diag)),
            "gamma_centered_mean": global_pos_mean(gamma_diag),
            "gamma_centered_geometric_mean": torch.exp(global_pos_mean(log_gamma_diag)),
            "gamma_log_mean": global_pos_mean(log_gamma_diag),
            "gamma_log_std": global_pos_std(log_gamma_diag),
            "gamma_fraction_below_0p1": masked_fraction(gamma_diag < 0.1),
            "gamma_fraction_above_10": masked_fraction(gamma_diag > 10.0),
            "gamma_centering_applied_fraction": masked_fraction(gamma_applied_diag),
            "gamma_centering_skipped_fraction": (
                gamma_skipped_mask.to(dtype=torch.float32).sum() / valid_count
            ),
            "gamma_eligible_fraction": masked_fraction(gamma_eligible_diag),
            "gamma_mean_reliability": global_pos_mean(gamma_reliability_diag),
            "gamma_min_reliability": global_pos_min(gamma_reliability_diag),
            "gamma_distinct_dataset_count_mean": global_pos_mean(gamma_distinct_diag),
            "gamma_total_reliability_mean": global_pos_mean(gamma_total_reliability_diag),
            "gamma_weighted_log_center_abs_mean": global_pos_mean(
                extras.get(
                    "gamma_cross_dataset_log_center",
                    torch.zeros_like(extras["L_bio"]),
                ).abs()
            ),
            "gamma_centering_constraint_error": global_pos_mean(
                gamma_constraint_error_diag
            ),
            "gamma_positional_gauge_constraint_error": global_pos_mean(
                gamma_positional_constraint_error_diag
            ),
            "gamma_reference_dataset_count": extras.get(
                "gamma_reference_dataset_count",
                torch.zeros_like(extras["scale_dt"]),
            ).float().mean(),
            "gamma_all_requested_in_reference": extras.get(
                "gamma_all_requested_in_reference",
                torch.zeros_like(extras["scale_dt"], dtype=torch.bool),
            ).float().mean(),
            "gamma_mean_log_per_sample": mean_log_gamma_per_sample.mean(),
            "gamma_std_log_per_sample": std_log_gamma_per_sample.mean(),
            "gamma_correlation_with_scale": finite_corr_or_zero(
                mean_log_gamma_per_sample.detach(),
                extras["scale_dt"].reshape(-1).detach(),
            ),
            "gamma_correlation_with_library_depth": finite_corr_or_zero(
                mean_log_gamma_per_sample.detach(),
                library_depth.detach(),
            ),
            "gamma_skipped_log_abs_mean": masked_abs_mean(
                log_gamma_diag,
                gamma_skipped_mask,
            ),
            "gamma_min": global_pos_min(
                extras.get("gamma", torch.ones_like(extras["L_bio"]))
            ),
            "gamma_max": global_pos_max(
                extras.get("gamma", torch.ones_like(extras["L_bio"]))
            ),
            "log_gamma_abs_mean": global_pos_mean(
                extras.get("log_gamma", torch.zeros_like(extras["L_bio"])).abs()
            ),
            "gamma_cross_dataset_log_center_abs_mean": global_pos_mean(
                extras.get(
                    "gamma_cross_dataset_log_center",
                    torch.zeros_like(extras["L_bio"]),
                ).abs()
            ),
            "gamma_cross_dataset_center_applied_frac": extras.get(
                "gamma_cross_dataset_center_applied",
                torch.zeros_like(extras["scale_dt"]),
            ).float().mean(),
            "gamma_cross_dataset_center_group_size_mean": extras.get(
                "gamma_cross_dataset_center_group_size",
                torch.zeros_like(extras["scale_dt"]),
            ).float().mean(),
            "nb_alpha_mean": pos_mean(extras["alpha"]).mean(),
        }
        metrics.update(replica_terms["metrics"])
        return metrics

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.bool().to(dtype=values.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (values * mask_f).sum(dim=1) / denom

    # ============================================================
    # Logging
    # ============================================================

    def _dataset_name(self, dataset_id: int) -> str:
        name = self.dataset_id_to_name.get(int(dataset_id), str(int(dataset_id)))
        return name.replace("/", "_").replace(" ", "_")

    # Scalar metrics logged every stage. `loss` and the explicit weighted /
    # unweighted mu-PCC metrics are logged in _log_stage, so they are
    # intentionally omitted here to avoid logging the same key twice with
    # different arguments.
    SCALAR_METRICS = (
        "nll",
        "pcc_loss",
        "pcc_value",
        "pcc_valid_fraction",
        "pcc_raw_value",
        "pcc_nb_vst_value",
        "consensus_pcc_value",
        "replica_nll",
        "replica_count_mean",
        "likelihood_mu_pcc",
        "likelihood_mu_pcc_unweighted",
        "log1p_mse",
        "mean_ratio",
        "J_mean",
        "L_bio_pcc",
        "L_bio_max",
        "rho_max",
        "gamma_reg",
        "gamma_mean",
        "gamma_centering_applied_fraction",
        "gamma_distinct_dataset_count_mean",
        "gamma_centering_constraint_error",
        "gamma_positional_gauge_constraint_error",
        "gamma_min",
        "gamma_max",
        "nb_alpha_mean",
    )

    def _log_validation_batch_structure(self, out: dict[str, Any]) -> None:
        """Log the positive-pair structure used by validation reducers.

        These are physical-microbatch diagnostics. They deliberately describe
        the same integer transcript groups, dataset IDs, and local reliability
        weights consumed by :func:`reduce_per_sample_quantity`.
        """
        group_ids = out["transcript_group_index"].detach().reshape(-1).long()
        dataset_ids = out["dataset_ids"].detach().reshape(-1).to(
            device=group_ids.device,
            dtype=torch.long,
        )
        sample_weights = out["sample_weights"].detach().reshape(-1).to(
            device=group_ids.device,
            dtype=torch.float32,
        )
        if not (
            group_ids.numel() == dataset_ids.numel() == sample_weights.numel()
        ):
            raise RuntimeError(
                "Validation transcript groups, dataset IDs, and weights have "
                "different row counts."
            )
        if not bool(torch.isfinite(sample_weights).all()) or bool(
            (sample_weights <= 0.0).any()
        ):
            raise ValueError(
                "Validation transcript reliability weights must be finite and "
                "strictly positive."
            )

        unique_groups, inverse = torch.unique(
            group_ids,
            sorted=False,
            return_inverse=True,
        )
        group_count = unique_groups.numel()
        group_dataset_pairs = torch.unique(
            torch.stack((inverse, dataset_ids), dim=1),
            dim=0,
        )
        datasets_per_transcript = torch.bincount(
            group_dataset_pairs[:, 0],
            minlength=group_count,
        ).float()
        rows_per_transcript = torch.bincount(
            inverse,
            minlength=group_count,
        )
        if not torch.equal(datasets_per_transcript.long(), rows_per_transcript):
            raise RuntimeError(
                "A validation microbatch contains a duplicate transcript--dataset pair."
            )
        transcript_weight_sums = torch.zeros(
            group_count,
            device=sample_weights.device,
            dtype=torch.float32,
        )
        transcript_weight_sums.index_add_(0, inverse, sample_weights)

        def standard_median(values: torch.Tensor) -> torch.Tensor:
            return torch.quantile(values.float(), 0.5)

        sync_dist = bool(getattr(self.config.trainer, "sync_dist_logs", False))
        self.log_dict(
            {
                "val_batch/unique_positive_transcripts": float(group_count),
                "val_batch/positive_dataset_pairs": float(dataset_ids.numel()),
                "val_batch/represented_datasets": float(
                    torch.unique(dataset_ids).numel()
                ),
                "val_batch/positive_datasets_per_transcript_min": (
                    datasets_per_transcript.amin()
                ),
                "val_batch/positive_datasets_per_transcript_median": (
                    standard_median(datasets_per_transcript)
                ),
                "val_batch/positive_datasets_per_transcript_mean": (
                    datasets_per_transcript.mean()
                ),
                "val_batch/positive_datasets_per_transcript_max": (
                    datasets_per_transcript.amax()
                ),
                "val_batch/transcript_total_weight_min": (
                    transcript_weight_sums.amin()
                ),
                "val_batch/transcript_total_weight_median": (
                    standard_median(transcript_weight_sums)
                ),
                "val_batch/transcript_total_weight_mean": (
                    transcript_weight_sums.mean()
                ),
                "val_batch/transcript_total_weight_max": (
                    transcript_weight_sums.amax()
                ),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=int(group_count),
            sync_dist=sync_dist,
        )

    def _log_stage(
        self,
        *,
        stage: str,
        out: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        batch_size = int(out["target"].shape[0])
        sync_dist = bool(getattr(self.config.trainer, "sync_dist_logs", False))
        transcript_count = int(
            torch.unique(out["transcript_group_index"]).numel()
        )
        represented_dataset_count = int(torch.unique(out["dataset_ids"]).numel())
        reduction_epoch_weights = {
            "global_weighted": batch_size,
            "dataset_balanced": represented_dataset_count,
            "transcript_balanced": transcript_count,
        }

        self.log(
            f"{stage}_loss",
            metrics["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=reduction_epoch_weights[self.sample_reduction],
            sync_dist=sync_dist,
        )
        for mode in sorted(SAMPLE_REDUCTION_MODES):
            self.log(
                f"{stage}/loss_{mode}",
                metrics[f"loss_{mode}"].detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=reduction_epoch_weights[mode],
                sync_dist=sync_dist,
            )
        if stage == "val":
            self._log_validation_batch_structure(out)
        # All metrics carrying the ``mu_pcc`` name are unweighted arithmetic
        # means over physical transcript-dataset pairs. The legacy alias is
        # retained for checkpoint monitoring, and the explicit suffix prevents
        # confusion with reliability-weighted objective diagnostics.
        self.log(
            f"{stage}_mu_pcc",
            metrics["mu_pcc_unweighted"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        self.log(
            f"{stage}_mu_pcc_unweighted",
            metrics["mu_pcc_unweighted"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        scalar_logs: dict[str, torch.Tensor] = {}
        for name in self.SCALAR_METRICS:
            value = metrics.get(name)
            if not torch.is_tensor(value) or value.ndim != 0:
                continue
            metric_path = {
                "pcc_raw_value": "pcc/raw_value",
                "pcc_nb_vst_value": "pcc/nb_vst_value",
                "gamma_centering_applied_fraction": "gamma/centering_applied_fraction",
                "gamma_distinct_dataset_count_mean": "gamma/distinct_dataset_count_mean",
                "gamma_centering_constraint_error": "gamma/centering_constraint_error",
                "gamma_positional_gauge_constraint_error": (
                    "gamma/positional_gauge_constraint_error"
                ),
            }.get(name)
            log_name = (
                f"{stage}/{metric_path}"
                if metric_path is not None
                else f"{stage}_{name}"
            )
            scalar_logs[log_name] = value
        self.log_dict(
            scalar_logs,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )

        log_train_per_dataset = bool(
            getattr(self.config.metrics, "log_train_per_dataset_metrics", False)
        )
        if stage != "train" or log_train_per_dataset:
            self._log_per_dataset_metrics(stage=stage, out=out, metrics=metrics)

    def _log_per_dataset_metrics(
        self,
        *,
        stage: str,
        out: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        dataset_ids = out["dataset_ids"].detach().to(device=out["target"].device)
        unique_dataset_ids = torch.unique(dataset_ids)

        per_sample = {
            "loss": metrics["loss_per_sample"],
            "nll": metrics["nll_per_sample"],
            "pcc_loss": metrics["pcc_loss_per_sample"],
            "pcc_raw_loss": metrics["pcc_raw_loss_per_sample"],
            "pcc_nb_vst_loss": metrics["pcc_nb_vst_loss_per_sample"],
            "mu_pcc": metrics["mu_pcc_per_sample"],
            "mu_pcc_unweighted": metrics["mu_pcc_per_sample"],
            "likelihood_mu_pcc": metrics["likelihood_mu_pcc_per_sample"],
            "likelihood_mu_pcc_unweighted": (
                metrics["likelihood_mu_pcc_per_sample"]
            ),
            "L_bio_pcc": metrics["L_bio_pcc_per_sample"],
        }
        extras = out["extras"]
        gamma = extras.get("gamma", torch.ones_like(extras["L_bio"])).float()

        for dataset_id_tensor in unique_dataset_ids:
            dataset_name = self._dataset_name(int(dataset_id_tensor.item()))
            sample_mask = dataset_ids == dataset_id_tensor
            dataset_sample_count = int(sample_mask.sum().item())
            dataset_logs = {
                f"{stage}_{metric_name}/{dataset_name}": values[sample_mask].mean()
                for metric_name, values in per_sample.items()
            }
            self.log_dict(
                dataset_logs,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                # Lightning weights epoch reductions by batch_size. Use this
                # dataset's physical pair count, not the full mixed batch.
                batch_size=dataset_sample_count,
                # Dataset IDs represented in a batch are data-dependent. Two
                # DDP ranks can therefore produce different metric-key sets;
                # synchronizing these dynamic keys would make ranks enter
                # different NCCL collectives. These diagnostics remain
                # rank-local; the fixed-key aggregate metrics above are still
                # synchronized.
                sync_dist=False,
            )

            position_mask = sample_mask.reshape(-1, 1) & out["mask"].bool()
            if bool(position_mask.any()):
                self.log(
                    f"{stage}/gamma/exact_zero_fraction_by_dataset/{dataset_name}",
                    (gamma[position_mask] == 0.0).float().mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    # This metric is a mean over valid positions, so weight its
                    # epoch reduction by the number of contributing positions.
                    batch_size=int(position_mask.sum().item()),
                    sync_dist=False,
                )

    # ============================================================
    # Profile plots
    # ============================================================

    @staticmethod
    def _transcript_ids_as_strings(ids: Any, expected_count: int) -> list[str]:
        if torch.is_tensor(ids):
            values = ids.detach().cpu().reshape(-1).tolist()
        elif isinstance(ids, (list, tuple)):
            values = list(ids)
        else:
            try:
                values = list(ids)
            except TypeError as exc:
                raise TypeError(
                    "Validation transcript IDs must be a tensor or sequence."
                ) from exc
        if len(values) != int(expected_count):
            raise ValueError(
                "Validation transcript ID count does not match per-pair PCC "
                f"count: ids={len(values)}, pcc={expected_count}."
            )
        return [str(value) for value in values]

    def _record_validation_transcript_mu_pcc(
        self,
        out: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        """Accumulate compact, unweighted validation PCC data on CPU.

        Complete validation transcript groups are reduced immediately to the
        equal arithmetic mean of their dataset-pair PCC values. Only one CPU
        float per transcript is retained; no profile tensor or autograd graph
        survives the validation step.
        """
        if not self.log_validation_transcript_mu_pcc_distribution:
            return

        pair_pcc = metrics["mu_pcc_per_sample"].detach().float().cpu().reshape(-1)
        group_ids = (
            out["transcript_group_index"].detach().long().cpu().reshape(-1)
        )
        transcript_ids = self._transcript_ids_as_strings(
            out["ids"],
            expected_count=pair_pcc.numel(),
        )
        if group_ids.numel() != pair_pcc.numel():
            raise ValueError(
                "Validation transcript group count does not match per-pair PCC count."
            )
        if not bool(torch.isfinite(pair_pcc).all()):
            raise ValueError("Validation per-pair mu PCC values must be finite.")

        unique_groups, inverse = torch.unique(
            group_ids,
            sorted=False,
            return_inverse=True,
        )
        group_sums = torch.zeros(unique_groups.numel(), dtype=torch.float32)
        group_counts = torch.zeros(unique_groups.numel(), dtype=torch.float32)
        group_sums.index_add_(0, inverse, pair_pcc)
        group_counts.index_add_(0, inverse, torch.ones_like(pair_pcc))
        group_means = group_sums / group_counts.clamp_min(1.0)

        for group_position in range(unique_groups.numel()):
            row_indices = torch.nonzero(
                inverse == group_position,
                as_tuple=False,
            ).reshape(-1)
            group_transcript_ids = {
                transcript_ids[int(row_index)]
                for row_index in row_indices.tolist()
            }
            if len(group_transcript_ids) != 1:
                raise RuntimeError(
                    "One validation transcript_group_index maps to multiple "
                    f"transcript IDs: {sorted(group_transcript_ids)}."
                )
            transcript_id = next(iter(group_transcript_ids))
            pcc = float(group_means[group_position].item())
            if transcript_id in self._validation_transcript_mu_pcc:
                if not math.isclose(
                    self._validation_transcript_mu_pcc[transcript_id],
                    pcc,
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                ):
                    raise RuntimeError(
                        "A validation transcript was split across physical "
                        f"batches with different PCC means: {transcript_id!r}."
                    )
                continue
            self._validation_transcript_mu_pcc[transcript_id] = pcc

    def _collect_validation_transcript_mu_pcc(
        self,
    ) -> tuple[list[str], torch.Tensor]:
        """Gather DDP shards and return one equal-dataset PCC per transcript."""
        local = self._validation_transcript_mu_pcc
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Epoch-end object collectives can race with Lightning's pending
            # reductions. Keep this diagnostic local in DDP; it is not used for
            # optimization or checkpoint selection.
            gathered: list[dict[str, float] | None] = [local]
        else:
            gathered = [local]

        merged: dict[str, float] = {}
        for rank_values in gathered:
            if rank_values is None:
                continue
            for transcript_id, pcc in rank_values.items():
                transcript_id = str(transcript_id)
                pcc = float(pcc)
                if transcript_id in merged:
                    # DDP pads the global plan by at most world_size-1 complete
                    # batches. Deduplicate those identical transcript groups.
                    if not math.isclose(
                        merged[transcript_id],
                        pcc,
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    ):
                        raise RuntimeError(
                            "DDP validation shards disagree for transcript="
                            f"{transcript_id!r}."
                        )
                    continue
                merged[transcript_id] = pcc

        transcript_ids = sorted(merged)
        transcript_pcc = torch.tensor(
            [merged[transcript_id] for transcript_id in transcript_ids],
            dtype=torch.float32,
        )
        return transcript_ids, transcript_pcc

    def _log_validation_transcript_mu_pcc_distribution(self) -> None:
        if not self.log_validation_transcript_mu_pcc_distribution:
            return
        transcript_ids, values = self._collect_validation_transcript_mu_pcc()
        if not transcript_ids:
            return

        values = values.float()
        quantiles = torch.quantile(
            values,
            torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]),
        )
        summary = {
            "val_mu_pcc_unweighted_by_transcript": values.mean(),
            "val/mu_pcc_unweighted_by_transcript/count": values.new_tensor(
                float(values.numel())
            ),
            "val/mu_pcc_unweighted_by_transcript/mean": values.mean(),
            "val/mu_pcc_unweighted_by_transcript/std": values.std(unbiased=False),
            "val/mu_pcc_unweighted_by_transcript/min": values.amin(),
            "val/mu_pcc_unweighted_by_transcript/q05": quantiles[0],
            "val/mu_pcc_unweighted_by_transcript/q25": quantiles[1],
            "val/mu_pcc_unweighted_by_transcript/median": quantiles[2],
            "val/mu_pcc_unweighted_by_transcript/q75": quantiles[3],
            "val/mu_pcc_unweighted_by_transcript/q95": quantiles[4],
            "val/mu_pcc_unweighted_by_transcript/max": values.amax(),
        }
        self.log_dict(
            {
                name: value.to(device=self.device)
                for name, value in summary.items()
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=int(values.numel()),
            # Every rank has the same all-gathered transcript distribution.
            sync_dist=False,
        )

        trainer = getattr(self, "trainer", None)
        is_global_zero = trainer is None or getattr(trainer, "is_global_zero", True)
        sanity_checking = trainer is not None and bool(
            getattr(trainer, "sanity_checking", False)
        )
        experiment = (
            getattr(self.logger, "experiment", None)
            if self.logger is not None
            else None
        )
        if (
            is_global_zero
            and not sanity_checking
            and experiment is not None
            and hasattr(experiment, "add_histogram")
        ):
            experiment.add_histogram(
                "val/mu_pcc_unweighted_by_transcript/distribution",
                values,
                global_step=self.global_step,
            )

    def on_validation_epoch_start(self) -> None:
        self._val_profile_plot_logged_this_epoch = False
        self._validation_transcript_mu_pcc = {}
        self._synthetic_ground_truth_epoch = {}

    def on_validation_epoch_end(self) -> None:
        self._log_validation_transcript_mu_pcc_distribution()
        self._log_synthetic_ground_truth_metrics()
        if (
            self.execution_microbatching_enabled
            and not bool(getattr(self.trainer, "sanity_checking", False))
        ):
            monitor = str(self.config.optim.scheduler.monitor)
            monitored_value = self.trainer.callback_metrics.get(monitor)
            if monitored_value is None:
                raise RuntimeError(
                    "Manual execution microbatching could not find scheduler "
                    f"monitor {monitor!r} at validation epoch end."
                )
            self.lr_schedulers().step(float(monitored_value.detach().cpu().item()))
        self._validation_transcript_mu_pcc = {}
        self._synthetic_ground_truth_epoch = {}

    def _likelihood_positive_mean(self, out: dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            return self.loss_fn.positive_mean_from_params(
                mu=out["mu"].detach().float(),
                log_sigma=out["log_sigma"].detach().float(),
            ).float()

    def _sequence_for_plot(self, value: torch.Tensor, *, sample_idx: int, mask_i: torch.Tensor):
        value = value.detach().float().cpu()
        mask_i = mask_i.detach().bool().cpu()
        length = int(mask_i.sum().item())

        if value.ndim == 0:
            return torch.full((length,), float(value.item())).numpy()
        if value.ndim == 1:
            if value.shape[0] == mask_i.shape[0]:
                return value[mask_i].numpy()
            return torch.full((length,), float(value[sample_idx].item())).numpy()
        if value.ndim == 2:
            sample_value = value[sample_idx]
            if sample_value.ndim == 1 and sample_value.shape[0] == mask_i.shape[0]:
                return sample_value[mask_i].numpy()
            if sample_value.numel() == 1:
                return torch.full((length,), float(sample_value.reshape(-1)[0].item())).numpy()
        return value.reshape(-1)[:length].numpy()

    def _plot_profile_example(self, out: dict[str, Any], *, batch_idx: int) -> None:
        if not bool(getattr(self.config.metrics, "log_example_plot", False)):
            return
        if self._val_profile_plot_logged_this_epoch or batch_idx != 0:
            return
        if not getattr(self.trainer, "is_global_zero", True):
            return
        if self.logger is None or getattr(self.logger, "experiment", None) is None:
            return

        dataset_ids = out["dataset_ids"].detach().cpu()
        max_plots = int(getattr(self.config.metrics, "example_plot_max_datasets", 4))
        selected_indices: list[int] = []
        seen_dataset_ids: set[int] = set()
        for sample_idx, dataset_id in enumerate(dataset_ids.tolist()):
            dataset_id = int(dataset_id)
            if dataset_id in seen_dataset_ids:
                continue
            seen_dataset_ids.add(dataset_id)
            selected_indices.append(sample_idx)
            if len(selected_indices) >= max_plots:
                break

        for sample_idx in selected_indices:
            self._plot_profile_sample(out, sample_idx=sample_idx)

        self._val_profile_plot_logged_this_epoch = True

    def _plot_profile_sample(self, out: dict[str, Any], *, sample_idx: int) -> None:
        mask_i = out["mask"][sample_idx].detach().bool().cpu()
        length = int(mask_i.sum().item())
        if length == 0:
            return

        extras = dict(out["extras"])
        extras["log_sigma"] = out["log_sigma"]
        mu_curve = self._likelihood_positive_mean(out)
        floor = float(self.pcc_prediction_floor)
        floored_curve = self._apply_pcc_floor(mu_curve) if floor > 0.0 else None

        x_axis = torch.arange(length).numpy()
        dataset_id = int(out["dataset_ids"][sample_idx].detach().cpu().item())
        dataset_name = self._dataset_name(dataset_id)
        n_axes = 1 + len(self.PROFILE_PLOT_GROUPS)

        fig, axes = plt.subplots(
            n_axes, 1, figsize=(14, 2.6 * n_axes), sharex=True, constrained_layout=True
        )

        target = self._sequence_for_plot(out["target"], sample_idx=sample_idx, mask_i=mask_i)
        mu_pred = self._sequence_for_plot(mu_curve, sample_idx=sample_idx, mask_i=mask_i)

        axes[0].plot(x_axis, target, label="target", linewidth=1.2, color="black")
        axes[0].plot(x_axis, mu_pred, label="mu prediction", linewidth=1.2, color="tab:blue")
        if floored_curve is not None:
            # The floored prediction is what PCC sees (values < floor -> 0).
            floored_pred = self._sequence_for_plot(floored_curve, sample_idx=sample_idx, mask_i=mask_i)
            axes[0].plot(
                x_axis, floored_pred, label=f"mu for PCC (floor={floor:g})",
                linewidth=0.9, color="tab:green", linestyle="--", alpha=0.8,
            )
        # Mark where the ground-truth is zero (what the model must predict low).
        zero_positions = x_axis[target <= 0.0]
        if zero_positions.size:
            axes[0].plot(
                zero_positions, [0.0] * zero_positions.size, "|", color="tab:red",
                markersize=8, label="target == 0", alpha=0.6,
            )
        title_bits = [
            f"Dataset: {dataset_name}",
            f"sample: {out['ids'][sample_idx]}",
            f"length: {length}",
            "likelihood: negative binomial (NB2)",
            f"zero_frac={float((target <= 0.0).mean()):.2f}",
        ]
        gamma_for_title = extras.get("gamma")
        if torch.is_tensor(gamma_for_title):
            gamma_curve = self._sequence_for_plot(
                gamma_for_title,
                sample_idx=sample_idx,
                mask_i=mask_i,
            )
            title_bits.append(
                f"gamma_zero_frac={float((gamma_curve == 0.0).mean()):.2f}"
            )
        axes[0].set_title(" | ".join(title_bits))
        axes[0].set_ylabel("profile")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        for axis, (ylabel, keys) in zip(axes[1:], self.PROFILE_PLOT_GROUPS, strict=True):
            plotted = False
            for key in keys:
                if key not in extras or not torch.is_tensor(extras[key]):
                    continue
                curve = self._sequence_for_plot(extras[key], sample_idx=sample_idx, mask_i=mask_i)
                axis.plot(x_axis, curve, label=key, linewidth=1.0)
                plotted = True
            axis.set_ylabel(ylabel)
            if ylabel == "gamma":
                axis.set_yscale("symlog", linthresh=0.1)
            axis.grid(True, alpha=0.3)
            if plotted:
                axis.legend(loc="upper right")

        axes[-1].set_xlabel("codon position")

        experiment = self.logger.experiment
        tag = f"val_profile/{dataset_name}"
        if hasattr(experiment, "add_figure"):
            experiment.add_figure(tag, fig, global_step=self.global_step)
        elif hasattr(experiment, "log_figure"):
            experiment.log_figure(figure_name=tag, figure=fig, step=self.global_step)

        plt.close(fig)

    # ============================================================
    # Steps
    # ============================================================

    def on_fit_start(self) -> None:
        if not self._grouped_optimizer_batch_plan:
            return
        expected = int(
            self._grouped_optimizer_batch_plan.get(
                "resolved_accumulate_grad_batches",
                1,
            )
        )
        actual = int(self.trainer.accumulate_grad_batches)
        if self.execution_microbatching_enabled:
            # Logical accumulation is performed manually; Lightning must see
            # one because its batches are execution chunks, not logical batches.
            expected = 1
        if actual != expected:
            raise RuntimeError(
                "Trainer accumulation changed after grouped batch planning: "
                f"planned={expected}, trainer={actual}. The factor must be fixed "
                "before optimizer and scheduler initialization."
            )

    def on_train_epoch_start(self) -> None:
        self._train_batch_structure_records = []
        if self.execution_microbatching_enabled:
            if (
                self._execution_logical_batches_since_step != 0
                or self._execution_has_pending_gradients
            ):
                raise RuntimeError(
                    "A pending execution-microbatch optimizer window leaked across epochs."
                )
            self.optimizers().zero_grad()

    def _record_train_batch_structure(self, batch) -> None:
        if not self._grouped_batch_logging_enabled:
            return
        dataset_ids = torch.as_tensor(batch[0]).detach().reshape(-1).to(dtype=torch.long)
        sample_weights = torch.as_tensor(batch[8]).detach().reshape(-1).float()
        group_ids = batch[11]
        if not (
            torch.is_tensor(group_ids)
            and group_ids.ndim == 1
            and not group_ids.is_floating_point()
        ):
            raise ValueError(
                "Batch field 11 must contain one integer transcript group ID per row."
            )
        group_ids = group_ids.detach().reshape(-1).to(
            device=dataset_ids.device,
            dtype=torch.long,
        )
        sample_weights = sample_weights.to(device=dataset_ids.device)
        if not (
            dataset_ids.numel() == group_ids.numel() == sample_weights.numel()
        ):
            raise RuntimeError(
                "Batch dataset IDs, transcript groups, and weights have different row counts."
            )
        if not bool(torch.isfinite(sample_weights).all()) or bool(
            (sample_weights <= 0.0).any()
        ):
            raise ValueError(
                "Batch transcript reliability weights must be finite and strictly positive."
            )

        unique_groups, inverse_groups = torch.unique(
            group_ids,
            sorted=False,
            return_inverse=True,
        )
        group_count = int(unique_groups.numel())
        group_weight_sums = torch.zeros(
            group_count,
            device=sample_weights.device,
            dtype=torch.float32,
        )
        group_weight_sums.index_add_(0, inverse_groups, sample_weights)

        # Exactly one row is expected for each transcript--dataset pair. Count
        # unique (local group, dataset) combinations and verify that invariant.
        group_dataset_pairs = torch.stack((inverse_groups, dataset_ids), dim=1)
        unique_group_dataset_pairs = torch.unique(group_dataset_pairs, dim=0)
        datasets_per_transcript_t = torch.bincount(
            unique_group_dataset_pairs[:, 0],
            minlength=group_count,
        )
        rows_per_transcript_t = torch.bincount(
            inverse_groups,
            minlength=group_count,
        )
        if not torch.equal(datasets_per_transcript_t, rows_per_transcript_t):
            raise RuntimeError(
                "A grouped microbatch contains a duplicate transcript--dataset pair."
            )
        _, rows_per_dataset_t = torch.unique(dataset_ids, return_counts=True)
        sync_dist = bool(getattr(self.config.trainer, "sync_dist_logs", False))
        datasets_per_transcript_f = datasets_per_transcript_t.float()
        self.log_dict(
            {
                "train_batch/unique_positive_transcripts": float(group_count),
                "train_batch/unique_transcripts_per_microbatch": float(group_count),
                "train_batch/positive_dataset_pairs": float(dataset_ids.numel()),
                "train_batch/represented_datasets": float(rows_per_dataset_t.numel()),
                "train_batch/datasets_per_transcript_mean_step": (
                    datasets_per_transcript_f.mean()
                ),
                "train_batch/datasets_per_transcript_min_step": (
                    datasets_per_transcript_f.amin()
                ),
                "train_batch/datasets_per_transcript_max_step": (
                    datasets_per_transcript_f.amax()
                ),
                "train_batch/transcript_weight_sum_mean_step": group_weight_sums.mean(),
                "train_batch/transcript_weight_sum_min_step": group_weight_sums.amin(),
                "train_batch/transcript_weight_sum_max_step": group_weight_sums.amax(),
            },
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=sync_dist,
        )
        self._train_batch_structure_records.append(
            {
                "pair_row_count": int(dataset_ids.numel()),
                "unique_transcript_count": group_count,
                "distinct_dataset_count": int(rows_per_dataset_t.numel()),
                "datasets_per_transcript": tuple(
                    map(int, datasets_per_transcript_t.cpu().tolist())
                ),
                "rows_per_dataset": tuple(map(int, rows_per_dataset_t.cpu().tolist())),
                "transcript_weight_sums": tuple(
                    map(float, group_weight_sums.cpu().tolist())
                ),
            }
        )

    @staticmethod
    def _gather_batch_structure_records(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return list(records)
        # Batch-structure output is diagnostic-only. Avoid a manual object
        # collective at epoch end, where it can be ordered differently from
        # Lightning's metric reductions on different ranks.
        return list(records)

    def on_train_epoch_end(self) -> None:
        if (
            self.execution_microbatching_enabled
            and self._execution_has_pending_gradients
            and self._execution_logical_batches_since_step == 0
        ):
            raise RuntimeError(
                "Training stopped in the middle of a logical batch. Execution "
                "microbatching refuses to apply a partial logical-batch gradient."
            )
        if (
            self.execution_microbatching_enabled
            and self._execution_logical_batches_since_step > 0
        ):
            # Match Lightning's usual fixed-factor final accumulation window:
            # gradients retain the configured 1/A scaling even when the last
            # epoch window contains fewer than A logical batches.
            self._execution_optimizer_step()
        if not self._grouped_batch_logging_enabled:
            return
        records = self._gather_batch_structure_records(
            self._train_batch_structure_records
        )
        if not records:
            return

        pair_rows = [float(record["pair_row_count"]) for record in records]
        unique_transcripts = [
            float(record["unique_transcript_count"]) for record in records
        ]
        datasets_per_transcript = [
            float(value)
            for record in records
            for value in record["datasets_per_transcript"]
        ]
        rows_per_dataset = [
            float(value)
            for record in records
            for value in record["rows_per_dataset"]
        ]
        transcript_weight_sums = [
            float(value)
            for record in records
            for value in record["transcript_weight_sums"]
        ]

        def summary(values: list[float]) -> tuple[float, float, float, float, float]:
            if not values:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            ordered = sorted(values)
            count = len(ordered)
            midpoint = count // 2
            if count % 2:
                median = ordered[midpoint]
            else:
                median = 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
            mean = float(sum(ordered) / count)
            variance = float(sum((value - mean) ** 2 for value in ordered) / count)
            cv = math.sqrt(variance) / mean if mean > 0.0 else 0.0
            return (
                mean,
                float(median),
                float(ordered[0]),
                float(ordered[-1]),
                float(cv),
            )

        pair_mean, _, pair_min, pair_max, _ = summary(pair_rows)
        unique_mean, unique_median, unique_min, unique_max, unique_cv = summary(
            unique_transcripts
        )
        ds_mean, ds_median, ds_min, ds_max, _ = summary(datasets_per_transcript)
        rows_ds_mean, rows_ds_median, rows_ds_min, rows_ds_max, _ = summary(
            rows_per_dataset
        )
        weight_sum_mean, _, weight_sum_min, weight_sum_max, weight_sum_cv = summary(
            transcript_weight_sums
        )
        plan = self._grouped_optimizer_batch_plan
        support = dict(plan.get("batch_support_statistics", {}))
        estimated_total_steps = float(self.trainer.estimated_stepping_batches)
        metrics = {
            "train_batch/pair_rows_mean": pair_mean,
            "train_batch/pair_rows_min": pair_min,
            "train_batch/pair_rows_max": pair_max,
            "train_batch/unique_transcripts_mean": unique_mean,
            "train_batch/unique_transcripts_median": unique_median,
            "train_batch/unique_transcripts_min": unique_min,
            "train_batch/unique_transcripts_max": unique_max,
            "train_batch/unique_transcripts_cv": unique_cv,
            "train_batch/datasets_per_transcript_mean": ds_mean,
            "train_batch/datasets_per_transcript_median": ds_median,
            "train_batch/datasets_per_transcript_min": ds_min,
            "train_batch/datasets_per_transcript_max": ds_max,
            "train_batch/rows_per_dataset_mean": rows_ds_mean,
            "train_batch/rows_per_dataset_median": rows_ds_median,
            "train_batch/rows_per_dataset_min": rows_ds_min,
            "train_batch/rows_per_dataset_max": rows_ds_max,
            "train_batch/transcript_weight_sum_mean": weight_sum_mean,
            "train_batch/transcript_weight_sum_min": weight_sum_min,
            "train_batch/transcript_weight_sum_max": weight_sum_max,
            "train_batch/transcript_weight_sum_cv": weight_sum_cv,
            "train_batch/accumulation_factor": float(
                plan.get("resolved_accumulate_grad_batches", 1)
            ),
            "train_batch/estimated_unique_transcripts_per_optimizer_step": float(
                plan.get("estimated_unique_transcripts_per_optimizer_step", 0.0)
            ),
            "train_batch/estimated_pair_rows_per_optimizer_step": float(
                plan.get("estimated_pair_rows_per_optimizer_step", 0.0)
            ),
            "train_batch/estimated_global_pair_rows_per_optimizer_step": float(
                plan.get("estimated_global_pair_rows_per_optimizer_step", 0.0)
            ),
            "train_batch/estimated_optimizer_steps_per_epoch": float(
                plan.get("estimated_optimizer_steps_per_epoch", 0)
            ),
            "train_batch/configured_target_unique_transcripts": float(
                plan.get("configured_target_unique_transcripts", 0)
            ),
            "train_batch/effective_local_target_unique_transcripts": float(
                plan.get("effective_local_target_unique_transcripts", 0)
            ),
            "train_batch/estimated_global_unique_transcripts_per_optimizer_step": float(
                plan.get(
                    "estimated_global_unique_transcripts_per_optimizer_step",
                    0.0,
                )
            ),
            "train_batch/world_size": float(plan.get("world_size", 1)),
            "train_batch/physical_pair_microbatch_size": float(
                plan.get("physical_pair_microbatch_size", 0)
            ),
            "train_batch/per_dataset_pair_microbatch_size": float(
                plan.get("per_dataset_pair_microbatch_size", 0)
            ),
            "train_batch/selected_dataset_count": float(
                plan.get("selected_dataset_count", 0)
            ),
            "train_batch/microbatches_per_epoch": float(
                plan.get("microbatches_per_epoch_per_rank", len(records))
            ),
            "train_batch/estimated_total_optimizer_steps": estimated_total_steps,
            "train_batch/optimizer_step_expectation_ratio": float(
                plan.get("optimizer_step_expectation_ratio", 0.0)
            ),
            "train_batch/transcripts_considered": float(
                support.get("transcripts_considered", 0)
            ),
            "train_batch/transcripts_with_positive_K0": float(
                support.get("transcripts_with_positive_k0", 0)
            ),
            "train_batch/transcripts_with_positive_K1": float(
                support.get("transcripts_with_positive_k1", 0)
            ),
            "train_batch/transcripts_with_positive_K2_or_more": float(
                support.get("transcripts_with_positive_k2_or_more", 0)
            ),
            "train_batch/transcripts_admitted": float(
                support.get("transcripts_admitted", 0)
            ),
            "train_batch/transcripts_excluded_for_insufficient_support": float(
                support.get(
                    "transcripts_excluded_for_insufficient_support",
                    0,
                )
            ),
            "train_batch/positive_pair_rows": float(
                support.get("positive_pair_rows", 0)
            ),
            "train_batch/group_size_min": float(
                support.get("group_size_min", 0.0)
            ),
            "train_batch/group_size_median": float(
                support.get("group_size_median", 0.0)
            ),
            "train_batch/group_size_mean": float(
                support.get("group_size_mean", 0.0)
            ),
            "train_batch/group_size_max": float(
                support.get("group_size_max", 0.0)
            ),
        }
        # Every rank has the same all-gathered summaries. Lightning writes from
        # rank zero, so sync_dist=False avoids counting the gathered data twice.
        self.log_dict(
            {
                name: torch.tensor(value, device=self.device, dtype=torch.float32)
                for name, value in metrics.items()
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )

    def _logical_accumulation_factor(self) -> int:
        return max(
            int(
                self._grouped_optimizer_batch_plan.get(
                    "resolved_accumulate_grad_batches",
                    1,
                )
            ),
            1,
        )

    def _execution_optimizer_step(self) -> None:
        """Apply one optimizer step after complete logical-batch gradients."""
        optimizer = self.optimizers()
        clip_value = self.execution_gradient_clip_val
        if clip_value > 0.0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_value,
                gradient_clip_algorithm=self.execution_gradient_clip_algorithm,
            )
        optimizer.step()
        optimizer.zero_grad()
        self._execution_logical_batches_since_step = 0
        self._execution_optimizer_steps += 1
        self._execution_has_pending_gradients = False

    @staticmethod
    def _execution_metadata(batch) -> dict[str, int] | None:
        if not batch:
            return None
        value = batch[-1]
        if value is None:
            return None
        if not isinstance(value, dict):
            return None
        required = {
            "logical_batch_index",
            "execution_chunk_index",
            "execution_chunk_count",
            "logical_group_count",
            "execution_group_count",
            "logical_pair_count",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(
                "Execution microbatch metadata is missing fields "
                f"{sorted(missing)}."
            )
        return {name: int(value[name]) for name in required}

    def training_step(self, batch, batch_idx):
        self._record_train_batch_structure(batch)
        out = self._forward_batch(batch)
        metrics = self._compute_loss_and_metrics(out)
        self._log_stage(stage="train", out=out, metrics=metrics)
        if not self.execution_microbatching_enabled:
            return metrics["loss"]

        metadata = self._execution_metadata(batch)
        if metadata is None:
            raise RuntimeError(
                "Execution microbatching is enabled, but the training sampler did "
                "not attach logical-batch metadata."
            )
        logical_groups = metadata["logical_group_count"]
        execution_groups = metadata["execution_group_count"]
        if logical_groups <= 0 or not (1 <= execution_groups <= logical_groups):
            raise ValueError(
                "Invalid execution/logical transcript-group counts: "
                f"execution={execution_groups}, logical={logical_groups}."
            )

        # The chunk loss is an equal mean over its complete transcript groups.
        # Multiplying by G_chunk/G_logical makes the sum of chunk gradients
        # exactly equal to the original transcript-balanced logical-batch loss.
        logical_accumulation = self._logical_accumulation_factor()
        gradient_loss = metrics["loss"] * (
            float(execution_groups)
            / float(logical_groups)
            / float(logical_accumulation)
        )
        self.manual_backward(gradient_loss)
        self._execution_has_pending_gradients = True

        is_last_chunk = (
            metadata["execution_chunk_index"]
            == metadata["execution_chunk_count"] - 1
        )
        if is_last_chunk:
            self._execution_logical_batches_since_step += 1
            if self._execution_logical_batches_since_step >= logical_accumulation:
                self._execution_optimizer_step()
        return metrics["loss"].detach()

    def on_before_optimizer_step(self, optimizer):
        del optimizer
        named_gradients = [
            (name, parameter.grad)
            for name, parameter in self.named_parameters()
            if parameter.grad is not None
        ]
        if not named_gradients:
            return
        finite_flags = torch.stack(
            [torch.isfinite(gradient).all() for _, gradient in named_gradients]
        ).detach().cpu().tolist()
        nonfinite = [
            name
            for (name, _), is_finite in zip(
                named_gradients,
                finite_flags,
                strict=True,
            )
            if not is_finite
        ]
        if nonfinite:
            raise FloatingPointError(
                "Non-finite gradients detected before the optimizer step. "
                "Refusing to silently discard the complete accumulated update; "
                f"first affected parameters: {nonfinite[:10]}."
            )

    def validation_step(self, batch, batch_idx):
        out = self._forward_batch(batch)
        metrics = self._compute_loss_and_metrics(out)
        self._log_stage(stage="val", out=out, metrics=metrics)
        # Sanity validation exists only to catch a broken forward/loss path.
        # Avoid expensive CPU diagnostics and Matplotlib work that will be
        # repeated during the first real validation epoch.
        if not bool(getattr(self.trainer, "sanity_checking", False)):
            self._record_synthetic_ground_truth_metrics(out)
            self._record_validation_transcript_mu_pcc(out, metrics)
            self._plot_profile_example(out, batch_idx=batch_idx)
        return metrics["loss"]

    @staticmethod
    def _detach_prediction_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu()
        return value

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        out = self._forward_batch(batch)
        extras = out["extras"]
        likelihood_positive_mean = self._likelihood_positive_mean(out)

        predictions: dict[str, Any] = {
            # Identifiers
            "ids": out["ids"],
            "dataset_id": out["dataset_ids"],
            "lengths": out["lengths"],
            "mask": out["mask"],
            "codon_ids": out["codon_ids"],
            "css": out["css"],
            "sample_weight": out["sample_weights"],
            "dataset_quality_rank": out["dataset_quality_ranks"],
            "dataset_quality_weight": out["dataset_quality_weights"],
            # Target / prediction
            "target": out["target"],
            "mu": out["mu"],
            "likelihood_positive_mean": likelihood_positive_mean,
            # Biological queue load (parquet-friendly names)
            "L_bio": extras["L_bio"],
            "rho_bio": extras["rho"],
            "J": extras["J"],
            # Dataset correction
            "gamma": extras.get("gamma", torch.ones_like(extras["L_bio"])),
            "log_gamma": extras.get(
                "log_gamma",
                torch.zeros_like(extras["L_bio"]),
            ),
            "gamma_raw": extras.get(
                "gamma_raw",
                torch.ones_like(extras["L_bio"]),
            ),
            "log_gamma_raw": extras.get(
                "log_gamma_raw",
                torch.zeros_like(extras["L_bio"]),
            ),
            "gamma_cross_dataset_log_center": extras.get(
                "gamma_cross_dataset_log_center",
                torch.zeros_like(extras["L_bio"]),
            ),
            "gamma_centering_reliability": extras.get(
                "gamma_centering_reliability",
                torch.zeros_like(extras["L_bio"]),
            ),
            "gamma_centering_eligible": extras.get(
                "gamma_centering_eligible",
                torch.zeros_like(extras["L_bio"], dtype=torch.bool),
            ),
            "gamma_centering_applied": extras.get(
                "gamma_centering_applied",
                torch.zeros_like(extras["L_bio"], dtype=torch.bool),
            ),
            "gamma_num_distinct_datasets": extras.get(
                "gamma_num_distinct_datasets",
                torch.zeros_like(extras["L_bio"]),
            ),
            "gamma_total_reliability": extras.get(
                "gamma_total_reliability",
                torch.zeros_like(extras["L_bio"]),
            ),
            "gamma_centering_constraint_error": extras.get(
                "gamma_centering_constraint_error",
                torch.zeros_like(extras["L_bio"]),
            ),
            "gamma_centering_mode": extras.get(
                "gamma_centering_mode",
                "disabled",
            ),
            "gamma_reference_dataset_count": extras.get(
                "gamma_reference_dataset_count",
                torch.zeros(
                    out["mu"].shape[0],
                    device=out["mu"].device,
                    dtype=out["mu"].dtype,
                ),
            ),
            "gamma_reference_dataset_ids": [
                extras.get(
                    "gamma_reference_dataset_ids",
                    torch.empty(0, dtype=torch.long),
                )
                .detach()
                .cpu()
                .tolist()
                for _ in range(out["mu"].shape[0])
            ],
            "gamma_reference_manifest_hash": extras.get(
                "gamma_reference_manifest_hash",
                "",
            ),
            "gamma_reference_weighting": extras.get(
                "gamma_reference_weighting",
                "equal",
            ),
            "gamma_reference_quality_rank_power": extras.get(
                "gamma_reference_quality_rank_power",
                0.0,
            ),
            "gamma_reference_chunk_size": extras.get(
                "gamma_reference_chunk_size",
                0,
            ),
            "gamma_all_requested_in_reference": extras.get(
                "gamma_all_requested_in_reference",
                torch.zeros(
                    out["mu"].shape[0],
                    device=out["mu"].device,
                    dtype=torch.bool,
                ),
            ),
            "additive_bias": extras.get(
                "additive_bias",
                torch.zeros_like(extras["L_bio"]),
            ),
            "scale_dt": extras["scale_dt"],
            "normalized_shape": extras.get(
                "normalized_shape",
                out["mu"] / extras["scale_dt"].reshape(-1, 1).clamp_min(self.eps),
            ),
            # Dispersion
            "log_sigma": out["log_sigma"],
        }

        return {
            key: self._detach_prediction_value(value)
            for key, value in predictions.items()
        }

    # ============================================================
    # Optimizer
    # ============================================================

    @staticmethod
    def _is_biological_parameter_name(name: str) -> bool:
        return name.startswith("biological_model.")

    def configure_optimizers(self):
        bio_lr = float(self.config.optim.lr_biological)
        rest_lr = float(self.config.optim.lr_rest)
        alpha_lr = rest_lr * self.alpha_learning_rate_scale
        weight_decay_bio = float(self.config.optim.weight_decay_bio)
        weight_decay_rest = float(self.config.optim.weight_decay_rest)
        bio_params = []
        rest_params = []
        alpha_params = []

        alpha_head = self.model.dataset_bias_model.log_sigma_head
        alpha_parameter_ids = {
            id(param) for param in alpha_head.parameters() if param.requires_grad
        }
        if not alpha_parameter_ids:
            raise RuntimeError("The NB2 alpha head has no trainable parameters.")

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in alpha_parameter_ids:
                alpha_params.append(param)
            elif self._is_biological_parameter_name(name):
                bio_params.append(param)
            else:
                rest_params.append(param)

        grouped_parameter_ids = [
            id(param) for param in (*bio_params, *rest_params, *alpha_params)
        ]
        if len(grouped_parameter_ids) != len(set(grouped_parameter_ids)):
            raise RuntimeError("A trainable parameter appears in multiple optimizer groups.")
        if set(grouped_parameter_ids) != {
            id(param) for param in self.model.parameters() if param.requires_grad
        }:
            raise RuntimeError("Optimizer parameter grouping omitted a trainable parameter.")
        if {id(param) for param in alpha_params} != alpha_parameter_ids:
            raise RuntimeError("Not all alpha-head parameters reached the alpha optimizer group.")

        param_groups = []
        if bio_params:
            param_groups.append(
                {
                    "params": bio_params,
                    "lr": bio_lr,
                    "weight_decay": weight_decay_bio,
                    "name": "biological",
                }
            )
        if rest_params:
            param_groups.append(
                {
                    "params": rest_params,
                    "lr": rest_lr,
                    "weight_decay": weight_decay_rest,
                    "name": "rest",
                }
            )
        if alpha_params:
            param_groups.append(
                {
                    "params": alpha_params,
                    "lr": alpha_lr,
                    "weight_decay": weight_decay_rest,
                    "name": "alpha",
                }
            )
        if not param_groups:
            raise RuntimeError("No trainable parameters found.")

        opt = torch.optim.AdamW(param_groups)

        scheduler_config = self.config.optim.scheduler
        scheduler_min_lr = float(scheduler_config.min_lr)
        scheduler_min_lrs = [
            (
                scheduler_min_lr * self.alpha_learning_rate_scale
                if group["name"] == "alpha"
                else scheduler_min_lr
            )
            for group in param_groups
        ]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=str(scheduler_config.mode),
            factor=float(scheduler_config.factor),
            patience=int(scheduler_config.patience),
            min_lr=scheduler_min_lrs,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": str(scheduler_config.monitor),
                "interval": "epoch",
                "frequency": 1,
            },
        }
