from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path
from typing import Any

import lightning as pl
import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")

from matplotlib import pyplot as plt


# ============================================================
# Loss
# ============================================================

SAMPLE_REDUCTION_MODES = frozenset(
    {"global_weighted", "dataset_balanced", "transcript_balanced"}
)


def _config_has_field(config: Any, name: str) -> bool:
    if isinstance(config, dict):
        return name in config
    try:
        return name in config
    except TypeError:
        return hasattr(config, name)


def _config_field(config: Any, name: str) -> Any:
    return config[name] if isinstance(config, dict) else getattr(config, name)


def resolve_sample_reduction_mode(loss_config: Any) -> str:
    """Resolve the explicit reducer while preserving the legacy boolean.

    A present ``dataset_balanced_loss`` is treated as an explicit legacy
    choice, not as a default.  Supplying both interfaces is accepted only when
    they identify the same reduction, preventing an old run from being
    silently reinterpreted.
    """
    has_mode = _config_has_field(loss_config, "sample_reduction")
    has_legacy = _config_has_field(loss_config, "dataset_balanced_loss")
    mode = (
        str(_config_field(loss_config, "sample_reduction")).strip().lower()
        if has_mode
        else None
    )
    if mode is not None and mode not in SAMPLE_REDUCTION_MODES:
        raise ValueError(
            "loss.sample_reduction must be one of "
            f"{sorted(SAMPLE_REDUCTION_MODES)}, got {mode!r}."
        )

    legacy_mode = None
    if has_legacy:
        legacy_value = _config_field(loss_config, "dataset_balanced_loss")
        if not isinstance(legacy_value, bool):
            raise TypeError(
                "loss.dataset_balanced_loss must be a boolean when supplied, "
                f"got {type(legacy_value).__name__}."
            )
        legacy_mode = "dataset_balanced" if legacy_value else "global_weighted"

    if mode is not None and legacy_mode is not None and mode != legacy_mode:
        raise ValueError(
            "Conflicting sample-loss reductions: "
            f"loss.sample_reduction={mode!r}, but "
            f"loss.dataset_balanced_loss={legacy_value!r} resolves to "
            f"{legacy_mode!r}. Remove the legacy field or make the two agree."
        )
    return mode or legacy_mode or "global_weighted"


def reduce_per_sample_quantity(
    values: torch.Tensor,
    sample_weights: torch.Tensor,
    transcript_group_ids: torch.Tensor,
    dataset_ids: torch.Tensor,
    mode: str,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Differentiably reduce one value per transcript-dataset sample.

    ``global_weighted`` computes ``sum(w*zeta) / sum(w)``.  Writing
    ``S_t=sum_{s:t(s)=t} w_s`` shows that this gives transcript ``t`` outer
    mass ``S_t/sum_u S_u``.  ``transcript_balanced`` instead computes each
    transcript's weighted mean and gives every positive-weight transcript one
    equal outer contribution.  A directly useful sufficient condition for the
    modes to coincide is that every transcript has the same observation count
    and the same total sample-weight sum.  (Equal total weight is the operative
    algebraic condition.)  In particular, this holds for unit weights and
    exactly M observations per transcript.

    Automatic gradient accumulation still averages physical-microbatch means.
    It is optimizer-window exact only when accumulated microbatches contain the
    same number of transcript groups.  TODO: add an optimizer-window-exact
    group-weighted accumulation ablation if observed group-count variation is
    substantial.
    """
    tensors = {
        "values": values,
        "sample_weights": sample_weights,
        "transcript_group_ids": transcript_group_ids,
        "dataset_ids": dataset_ids,
    }
    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must have shape [B], got {tuple(tensor.shape)}.")
    batch_size = values.shape[0]
    for name, tensor in tensors.items():
        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"All reducer inputs must have the same B; values has {batch_size}, "
                f"but {name} has {tensor.shape[0]}."
            )
    if not values.is_floating_point():
        raise TypeError("values must use a floating-point dtype.")
    if not sample_weights.is_floating_point():
        raise TypeError("sample_weights must use a floating-point dtype.")
    for name, tensor in (
        ("transcript_group_ids", transcript_group_ids),
        ("dataset_ids", dataset_ids),
    ):
        if tensor.is_floating_point() or tensor.dtype == torch.bool:
            raise TypeError(f"{name} must use an integer dtype.")

    mode = str(mode).strip().lower()
    if mode not in SAMPLE_REDUCTION_MODES:
        raise ValueError(
            f"mode must be one of {sorted(SAMPLE_REDUCTION_MODES)}, got {mode!r}."
        )
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}.")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("values contains a non-finite per-sample quantity.")
    if not bool(torch.isfinite(sample_weights).all()):
        raise ValueError("sample_weights contains a non-finite value.")
    if bool((sample_weights < 0.0).any()):
        raise ValueError("sample_weights contains a negative value.")
    if batch_size == 0:
        return values.sum() * 0.0

    device = values.device
    accumulation_dtype = (
        torch.float32
        if values.dtype in {torch.float16, torch.bfloat16}
        else values.dtype
    )
    values_acc = values.to(dtype=accumulation_dtype)
    weights_acc = (
        sample_weights.detach()
        .to(device=device, dtype=accumulation_dtype)
    )
    transcript_ids = transcript_group_ids.to(device=device, dtype=torch.long)
    datasets = dataset_ids.to(device=device, dtype=torch.long)
    differentiable_zero = values_acc.sum() * 0.0

    if mode == "global_weighted":
        denominator = weights_acc.sum()
        return (values_acc * weights_acc).sum() / denominator.clamp_min(eps)

    grouping_ids = datasets if mode == "dataset_balanced" else transcript_ids
    unique_groups, inverse = torch.unique(
        grouping_ids,
        sorted=False,
        return_inverse=True,
    )
    group_count = unique_groups.numel()
    if group_count == 0:
        return differentiable_zero
    numerator = torch.zeros(
        group_count,
        device=device,
        dtype=accumulation_dtype,
    )
    denominator = torch.zeros_like(numerator)
    numerator.index_add_(0, inverse, values_acc * weights_acc)
    denominator.index_add_(0, inverse, weights_acc)
    group_means = numerator / denominator.clamp_min(eps)

    if mode == "dataset_balanced":
        # Backward-compatible semantics: every represented dataset is included;
        # a historical all-zero dataset bucket contributes a zero mean.
        return group_means.mean()

    valid_groups = denominator > eps
    if not bool(valid_groups.any()):
        return differentiable_zero
    return group_means[valid_groups].mean()


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


def masked_weighted_pcc(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    min_target_var: float = 1.0e-6,
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    x = x.float()
    y = y.float()
    mask_b = mask.bool() & torch.isfinite(x) & torch.isfinite(y)
    mask_f = mask_b.to(dtype=x.dtype)

    x = torch.where(mask_b, x, torch.zeros_like(x))
    y = torch.where(mask_b, y, torch.zeros_like(y))

    if weights is None:
        w = mask_f
    else:
        w = torch.nan_to_num(
            weights.to(device=x.device, dtype=x.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0) * mask_f

    w_sum = w.sum(dim=1, keepdim=True).clamp_min(float(eps))
    x_mean = (w * x).sum(dim=1, keepdim=True) / w_sum
    y_mean = (w * y).sum(dim=1, keepdim=True) / w_sum

    x_c = (x - x_mean) * mask_f
    y_c = (y - y_mean) * mask_f

    cov = (w * x_c * y_c).sum(dim=1)
    x_var = (w * x_c.pow(2)).sum(dim=1)
    y_var = (w * y_c.pow(2)).sum(dim=1)
    pcc = cov / torch.sqrt((x_var * y_var).clamp_min(0.0) + float(eps) * float(eps))
    pcc = torch.nan_to_num(pcc, nan=0.0, posinf=0.0, neginf=0.0)

    target_var = y_var / w_sum.squeeze(1).clamp_min(float(eps))
    valid = target_var > float(min_target_var)
    pcc = torch.where(valid, pcc, torch.zeros_like(pcc))

    valid_f = valid.to(dtype=x.dtype)
    valid_count = valid_f.sum().clamp_min(1.0)
    mean_pcc = (pcc * valid_f).sum() / valid_count
    valid_fraction = valid_f.mean() if valid_f.numel() > 0 else x.new_tensor(0.0)
    target_var_mean = (target_var * valid_f).sum() / valid_count

    return {
        "pcc_per_sample": pcc,
        "valid": valid,
        "mean_pcc": mean_pcc,
        "valid_fraction": valid_fraction,
        "target_var": target_var,
        "target_var_mean": target_var_mean,
        "weights": w,
    }


class ResidualDiagnosticsAccumulator:
    def __init__(
        self,
        *,
        dataset_id_to_name: dict[int, str],
        eps: float = 1.0e-8,
        eps_count: float = 1.0e-3,
        zero_threshold: float = 0.0,
        low_count_threshold: float = 1.0,
        tail_thresholds: tuple[float, ...] = (2.0, 3.0, 5.0),
        topk_fractions: tuple[float, ...] = (0.01, 0.05, 0.10),
        quantile_max_values: int = 1_000_000,
    ) -> None:
        self.dataset_id_to_name = dict(dataset_id_to_name)
        self.eps = float(eps)
        self.eps_count = float(eps_count)
        self.zero_threshold = float(zero_threshold)
        self.low_count_threshold = float(low_count_threshold)
        self.tail_thresholds = tuple(float(x) for x in tail_thresholds)
        self.topk_fractions = tuple(float(x) for x in topk_fractions)
        self.quantile_max_values = max(1, int(quantile_max_values))
        self.reset()

    def reset(self) -> None:
        self.position_chunks: list[dict[str, torch.Tensor]] = []
        self.sample_rows: list[dict[str, float | int | str]] = []
        self.dataset_rows: list[dict[str, float | int | str]] = []

    def _dataset_name(self, dataset_id: int) -> str:
        return self.dataset_id_to_name.get(int(dataset_id), str(int(dataset_id)))

    def _quantile_values(self, x: torch.Tensor) -> torch.Tensor:
        """Return a bounded, deterministic view used only for diagnostics.

        ``torch.quantile`` rejects tensors above an internal element-count
        limit. Large validation sets can cross that limit when all positions
        from many datasets are concatenated. Sampling evenly across the
        flattened tensor keeps the diagnostic reproducible and prevents both
        that failure and an unnecessarily expensive full-data sort.

        This does not affect losses, model outputs, or gradients: residual
        diagnostics are accumulated under ``torch.no_grad()`` on CPU.
        """
        values = x.detach().float().reshape(-1)
        if values.numel() <= self.quantile_max_values:
            return values

        # Midpoints of equally sized intervals cover the complete validation
        # tensor without allocating a full randperm of a potentially huge N.
        sample_positions = torch.arange(
            self.quantile_max_values,
            device=values.device,
            dtype=torch.float64,
        )
        indices = torch.floor(
            (sample_positions + 0.5)
            * (float(values.numel()) / float(self.quantile_max_values))
        ).to(dtype=torch.long)
        return values.index_select(0, indices.clamp_max(values.numel() - 1))

    def _safe_quantiles(
        self,
        x: torch.Tensor,
        q: float | list[float] | tuple[float, ...] | torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0:
            if torch.is_tensor(q) and q.ndim > 0:
                return torch.zeros_like(q, dtype=torch.float32)
            if isinstance(q, (list, tuple)):
                return torch.zeros(len(q), dtype=torch.float32)
            return x.new_tensor(0.0, dtype=torch.float32)
        values = self._quantile_values(x)
        q_tensor = torch.as_tensor(q, dtype=torch.float32, device=values.device)
        return torch.quantile(values, q_tensor)

    def _safe_quantile(self, x: torch.Tensor, q: float) -> torch.Tensor:
        return self._safe_quantiles(x, float(q))

    @staticmethod
    def _mean_or_zero(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.tensor(0.0)
        return x.float().mean()

    def _corr_or_zero(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(x) & torch.isfinite(y)
        if int(valid.sum().item()) < 2:
            return x.new_tensor(0.0)
        x = x[valid].float()
        y = y[valid].float()
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denom = torch.sqrt(
            x_centered.pow(2).sum() * y_centered.pow(2).sum()
        ).clamp_min(self.eps)
        return (x_centered * y_centered).sum() / denom

    def _nb_zero_probability(self, mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        alpha = alpha.clamp_min(self.eps)
        r = (1.0 / alpha).clamp_max(1.0e8)
        return torch.exp(r * (torch.log(r.clamp_min(self.eps)) - torch.log((r + mu).clamp_min(self.eps))))

    def update(self, out: dict[str, Any]) -> None:
        with torch.no_grad():
            target = out["target"].detach().float().cpu()
            mu = out["mu"].detach().float().cpu()
            log_alpha = out["log_sigma"].detach().float().cpu()
            mask = out["mask"].detach().bool().cpu() & torch.isfinite(target)
            dataset_ids = out["dataset_ids"].detach().long().cpu()
            lengths = out["lengths"].detach().long().cpu()
            transcript_ids = out.get("ids", None)
            sample_weights = out.get("sample_weights", None)
            if torch.is_tensor(sample_weights):
                sample_weights_cpu = sample_weights.detach().float().cpu().reshape(-1)
            else:
                sample_weights_cpu = None
            extras = out["extras"]
            L_bio = extras["L_bio"].detach().float().cpu()
            J = extras["J"].detach().float().cpu().reshape(-1)
            scale_dt = extras["scale_dt"].detach().float().cpu().reshape(-1)

            alpha = torch.exp(_broadcast_profile_param(log_alpha, mu.shape)).clamp(
                min=1.0e-8,
                max=1.0e8,
            )
            variance = (mu + alpha * mu.pow(2)).clamp_min(self.eps)
            nb_std = (target - mu) / torch.sqrt(variance + self.eps)
            raw_residual = target - mu
            log_residual = torch.log(target.clamp_min(0.0) + self.eps_count) - torch.log(
                mu.clamp_min(0.0) + self.eps_count
            )
            relative = raw_residual / (mu + self.eps_count)

            B, T = target.shape
            pos = torch.arange(T).reshape(1, T).expand(B, T)
            rel_pos = pos.float() / (lengths.reshape(-1, 1).float() - 1.0).clamp_min(1.0)
            dataset_pos = dataset_ids.reshape(-1, 1).expand(B, T)
            length_pos = lengths.reshape(-1, 1).expand(B, T)
            J_pos = J.reshape(-1, 1).expand(B, T)
            S_pos = scale_dt.reshape(-1, 1).expand(B, T)

            valid = mask
            if bool(valid.any()):
                self.position_chunks.append(
                    {
                        "dataset_id": dataset_pos[valid],
                        "pos": pos[valid],
                        "length": length_pos[valid],
                        "rel_pos": rel_pos[valid],
                        "target": target[valid],
                        "mu": mu[valid],
                        "alpha": alpha[valid],
                        "nb_std": nb_std[valid],
                        "raw_residual": raw_residual[valid],
                        "log_residual": log_residual[valid],
                        "relative": relative[valid],
                        "L_bio": L_bio[valid],
                        "J": J_pos[valid],
                        "S": S_pos[valid],
                    }
                )

            for sample_idx in range(B):
                m = mask[sample_idx]
                if not bool(m.any()):
                    continue
                dataset_id = int(dataset_ids[sample_idx].item())
                y_i = target[sample_idx][m]
                mu_i = mu[sample_idx][m]
                alpha_i = alpha[sample_idx][m]
                nb_i = nb_std[sample_idx][m].abs()
                nb_signed_i = nb_std[sample_idx][m]
                raw_residual_i = raw_residual[sample_idx][m]
                log_i = log_residual[sample_idx][m]
                abs_sum = nb_i.sum().clamp_min(self.eps)
                y_nonneg = y_i.clamp_min(0.0)
                mu_nonneg = mu_i.clamp_min(0.0)
                target_mean = y_i.mean()
                mu_mean = mu_i.mean()
                read_depth = y_i.sum()
                mu_depth = mu_i.sum()
                log1p_y = torch.log1p(y_nonneg)
                log1p_mu = torch.log1p(mu_nonneg)
                alpha_i = alpha_i.clamp_min(self.eps)
                r = (1.0 / alpha_i).clamp_max(1.0e8)
                nb_log_prob = (
                    torch.lgamma(y_nonneg + r)
                    - torch.lgamma(r)
                    - torch.lgamma(y_nonneg + 1.0)
                    + r
                    * (
                        torch.log(r.clamp_min(self.eps))
                        - torch.log((r + mu_nonneg).clamp_min(self.eps))
                    )
                    + y_nonneg
                    * (
                        torch.log(mu_nonneg.clamp_min(self.eps))
                        - torch.log((r + mu_nonneg).clamp_min(self.eps))
                    )
                )

                if isinstance(transcript_ids, (list, tuple)):
                    transcript_id = str(transcript_ids[sample_idx])
                elif torch.is_tensor(transcript_ids):
                    transcript_id = str(transcript_ids.detach().cpu().reshape(-1)[sample_idx].item())
                elif transcript_ids is None:
                    transcript_id = str(sample_idx)
                else:
                    try:
                        transcript_id = str(transcript_ids[sample_idx])
                    except Exception:
                        transcript_id = str(sample_idx)

                row: dict[str, float | int | str] = {
                    "dataset_id": dataset_id,
                    "dataset_name": self._dataset_name(dataset_id),
                    "transcript_id": transcript_id,
                    "valid_len": int(m.sum().item()),
                    "sample_weight": (
                        float(sample_weights_cpu[sample_idx].item())
                        if sample_weights_cpu is not None and sample_idx < sample_weights_cpu.numel()
                        else 1.0
                    ),
                    "read_depth": float(read_depth.item()),
                    "mu_depth": float(mu_depth.item()),
                    "target_mean": float(target_mean.item()),
                    "mu_mean": float(mu_mean.item()),
                    "mean_ratio": float((mu_mean / target_mean.clamp_min(self.eps)).item()),
                    "coverage_fraction": float((y_i > self.zero_threshold).float().mean().item()),
                    "low_count_fraction": float((y_i <= self.low_count_threshold).float().mean().item()),
                    "target_var": float(y_i.var(unbiased=False).item()),
                    "mu_var": float(mu_i.var(unbiased=False).item()),
                    "target_max": float(y_i.max().item()),
                    "mu_max": float(mu_i.max().item()),
                    "nb_nll_mean": float((-nb_log_prob).mean().item()),
                    "nb_std_mean": float(nb_signed_i.mean().item()),
                    "nb_std_abs_mean": float(nb_i.mean().item()),
                    "nb_std_q95_abs": float(self._safe_quantile(nb_i, 0.95).item()),
                    "mean_raw_residual": float(raw_residual_i.mean().item()),
                    "mean_abs_raw_residual": float(raw_residual_i.abs().mean().item()),
                    "mean_log_residual": float(log_i.mean().item()),
                    "mean_abs_log_residual": float(log_i.abs().mean().item()),
                    "profile_log1p_mse": float((log1p_mu - log1p_y).pow(2).mean().item()),
                    "profile_pcc": float(self._corr_or_zero(mu_i, y_i).item()),
                    "log1p_profile_pcc": float(self._corr_or_zero(log1p_mu, log1p_y).item()),
                    "J": float(J[sample_idx].item()) if sample_idx < J.numel() else 0.0,
                    "scale_dt": float(scale_dt[sample_idx].item()) if sample_idx < scale_dt.numel() else 0.0,
                }
                for thr in (3.0, 5.0):
                    tail = nb_i > thr
                    row[f"tail_fraction_gt_{thr:g}"] = float(tail.float().mean().item())
                    row[f"tail_mass_gt_{thr:g}"] = float((nb_i[tail].sum() / abs_sum).item()) if bool(tail.any()) else 0.0
                for frac in self.topk_fractions:
                    k = max(1, int(math.ceil(float(frac) * nb_i.numel())))
                    row[f"top_{int(frac * 100):g}pct_abs_mass"] = float((torch.topk(nb_i, k).values.sum() / abs_sum).item())

                for frac in (0.10, 0.05, 0.01):
                    k = max(1, int(math.ceil(float(frac) * y_i.numel())))
                    target_top = torch.topk(y_i, k).indices
                    mu_top = torch.topk(mu_i, k).indices
                    row[f"peak_mu_over_y_top_{int(frac * 100):g}"] = float(
                        (mu_i[target_top] / (y_i[target_top] + self.eps_count)).mean().item()
                    )
                    row[f"peak_log_residual_top_{int(frac * 100):g}"] = float(
                        log_i[target_top].mean().item()
                    )
                    overlap = len(set(target_top.tolist()) & set(mu_top.tolist()))
                    row[f"top{int(frac * 100):g}_overlap"] = float(overlap / k)
                self.sample_rows.append(row)

    def _concat_positions(self) -> dict[str, torch.Tensor]:
        if not self.position_chunks:
            return {}
        keys = self.position_chunks[0].keys()
        return {key: torch.cat([chunk[key] for chunk in self.position_chunks]) for key in keys}

    def _summarize_positions(self, data: dict[str, torch.Tensor], mask: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if not bool(mask.any()):
            return out
        nb = data["nb_std"][mask].float()
        abs_nb = nb.abs()
        raw_residual = data["raw_residual"][mask].float()
        log_res = data["log_residual"][mask].float()
        mu = data["mu"][mask].float()
        target = data["target"][mask].float()
        alpha = data["alpha"][mask].float()
        L_bio = data["L_bio"][mask].float()
        J = data["J"][mask].float()
        centered = nb - nb.mean()
        std = nb.std(unbiased=False).clamp_min(self.eps)

        out[f"{prefix}/nb_std_mean"] = nb.mean()
        out[f"{prefix}/nb_std_std"] = nb.std(unbiased=False)
        out[f"{prefix}/nb_std_abs_mean"] = abs_nb.mean()
        out[f"{prefix}/nb_std_median"] = nb.median()
        out[f"{prefix}/nb_std_q90_abs"] = self._safe_quantile(abs_nb, 0.90)
        out[f"{prefix}/nb_std_q95_abs"] = self._safe_quantile(abs_nb, 0.95)
        out[f"{prefix}/nb_std_q99_abs"] = self._safe_quantile(abs_nb, 0.99)
        for thr in self.tail_thresholds:
            out[f"{prefix}/fraction_abs_std_gt_{thr:g}"] = (abs_nb > thr).float().mean()
        out[f"{prefix}/skewness"] = (centered.pow(3).mean() / std.pow(3)).nan_to_num()
        out[f"{prefix}/kurtosis"] = (centered.pow(4).mean() / std.pow(4)).nan_to_num()

        abs_sum = abs_nb.sum().clamp_min(self.eps)
        for thr in (3.0, 5.0):
            tail = abs_nb > thr
            out[f"{prefix}/tail_fraction_abs_gt_{thr:g}"] = tail.float().mean()
            out[f"{prefix}/tail_mass_abs_gt_{thr:g}"] = abs_nb[tail].sum() / abs_sum if bool(tail.any()) else nb.new_tensor(0.0)
        for frac in (0.01, 0.05):
            k = max(1, int(math.ceil(frac * abs_nb.numel())))
            out[f"{prefix}/top_{int(frac * 100):g}pct_abs_mass"] = torch.topk(abs_nb, k).values.sum() / abs_sum

        low_mu_cut = self._safe_quantile(mu, 0.10)
        high_mu_cut = self._safe_quantile(mu, 0.90)
        low_mu = mu <= low_mu_cut
        high_mu = mu >= high_mu_cut
        out[f"{prefix}/low_mu_raw_residual_bias"] = self._mean_or_zero(
            raw_residual[low_mu]
        )
        out[f"{prefix}/high_mu_log_bias"] = self._mean_or_zero(log_res[high_mu])

        zero = target <= self.zero_threshold
        low_count = target <= self.low_count_threshold
        p0 = self._nb_zero_probability(mu, alpha)
        out[f"{prefix}/zero_fraction"] = zero.float().mean()
        out[f"{prefix}/low_count_fraction"] = low_count.float().mean()
        out[f"{prefix}/zero_mean_mu"] = self._mean_or_zero(mu[zero])
        out[f"{prefix}/zero_median_mu"] = mu[zero].median() if bool(zero.any()) else mu.new_tensor(0.0)
        out[f"{prefix}/zero_q95_mu"] = self._safe_quantile(mu[zero], 0.95)
        out[f"{prefix}/zero_mean_alpha"] = self._mean_or_zero(alpha[zero])
        out[f"{prefix}/zero_mean_nb_p0"] = self._mean_or_zero(p0[zero])
        out[f"{prefix}/zero_fraction_mu_gt_1"] = (mu[zero] > 1.0).float().mean() if bool(zero.any()) else mu.new_tensor(0.0)
        out[f"{prefix}/zero_fraction_mu_gt_5"] = (mu[zero] > 5.0).float().mean() if bool(zero.any()) else mu.new_tensor(0.0)

        out[f"{prefix}/alpha_mean"] = alpha.mean()
        out[f"{prefix}/L_bio_mean"] = L_bio.mean()
        out[f"{prefix}/L_bio_max"] = L_bio.max()
        out[f"{prefix}/J_mean"] = J.mean()
        out[f"{prefix}/mean_ratio"] = mu.mean() / target.mean().clamp_min(self.eps)
        return out

    def _mu_bin_metrics(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if not data:
            return out
        mu = data["mu"].float()
        raw_residual = data["raw_residual"].float()
        log_res = data["log_residual"].float()
        nb = data["nb_std"].float()
        quantiles = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0]
        edges = self._safe_quantiles(mu, quantiles)
        names = ["q0_q10", "q10_q25", "q25_q50", "q50_q75", "q75_q90", "q90_q99", "q99_q100"]
        for i, name in enumerate(names):
            if i == 0:
                m = (mu >= edges[i]) & (mu <= edges[i + 1])
            else:
                m = (mu > edges[i]) & (mu <= edges[i + 1])
            prefix = f"residual/mu_bin/{name}"
            out[f"{prefix}/mean_raw_residual"] = self._mean_or_zero(raw_residual[m])
            out[f"{prefix}/median_raw_residual"] = raw_residual[m].median() if bool(m.any()) else mu.new_tensor(0.0)
            out[f"{prefix}/mean_log_residual"] = self._mean_or_zero(log_res[m])
            out[f"{prefix}/median_log_residual"] = log_res[m].median() if bool(m.any()) else mu.new_tensor(0.0)
            out[f"{prefix}/nb_std_residual_mean"] = self._mean_or_zero(nb[m])
            out[f"{prefix}/nb_std_residual_abs_mean"] = self._mean_or_zero(nb[m].abs())
        return out

    def _basic_bin_summary(
        self,
        data: dict[str, torch.Tensor],
        mask: torch.Tensor,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        if not bool(mask.any()):
            z = data["mu"].new_tensor(0.0)
            return {
                f"{prefix}/nb_std_mean": z,
                f"{prefix}/nb_std_abs_mean": z,
                f"{prefix}/frac_abs_std_gt_3": z,
                f"{prefix}/mean_raw_residual": z,
                f"{prefix}/mean_log_residual": z,
                f"{prefix}/mean_mu": z,
                f"{prefix}/mean_target": z,
            }
        nb = data["nb_std"][mask].float()
        return {
            f"{prefix}/nb_std_mean": nb.mean(),
            f"{prefix}/nb_std_abs_mean": nb.abs().mean(),
            f"{prefix}/frac_abs_std_gt_3": (nb.abs() > 3.0).float().mean(),
            f"{prefix}/mean_raw_residual": data["raw_residual"][mask].float().mean(),
            f"{prefix}/mean_log_residual": data["log_residual"][mask].float().mean(),
            f"{prefix}/mean_mu": data["mu"][mask].float().mean(),
            f"{prefix}/mean_target": data["target"][mask].float().mean(),
        }

    def _quantile_stratification_metrics(
        self,
        data: dict[str, torch.Tensor],
        key: str,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        values = data[key].float()
        if values.numel() == 0:
            return out
        quantiles = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0]
        edges = self._safe_quantiles(values, quantiles)
        names = ["q0_q10", "q10_q25", "q25_q50", "q50_q75", "q75_q90", "q90_q99", "q99_q100"]
        for i, name in enumerate(names):
            if i == 0:
                mask = (values >= edges[i]) & (values <= edges[i + 1])
            else:
                mask = (values > edges[i]) & (values <= edges[i + 1])
            out.update(self._basic_bin_summary(data, mask, f"{prefix}/{name}"))
        return out

    def _stratification_metrics(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if not data:
            return out

        length = data["length"].float()
        out.update(self._basic_bin_summary(data, length < 500.0, "residual/length_bin/short"))
        out.update(
            self._basic_bin_summary(
                data,
                (length >= 500.0) & (length < 1500.0),
                "residual/length_bin/medium",
            )
        )
        out.update(self._basic_bin_summary(data, length >= 1500.0, "residual/length_bin/long"))

        pos = data["pos"].float()
        rel_pos = data["rel_pos"].float()
        out.update(self._basic_bin_summary(data, pos < 50.0, "residual/position_bin/start"))
        out.update(self._basic_bin_summary(data, rel_pos < 0.25, "residual/position_bin/early"))
        out.update(
            self._basic_bin_summary(
                data,
                (rel_pos >= 0.25) & (rel_pos < 0.75),
                "residual/position_bin/middle",
            )
        )
        out.update(self._basic_bin_summary(data, rel_pos >= 0.75, "residual/position_bin/late"))
        out.update(
            self._basic_bin_summary(
                data,
                pos >= (length - 50.0).clamp_min(0.0),
                "residual/position_bin/stop",
            )
        )

        out.update(self._quantile_stratification_metrics(data, "target", "residual/target_quantile"))
        out.update(self._quantile_stratification_metrics(data, "L_bio", "residual/L_bio_quantile"))
        return out

    def _dataset_csv_rows(self, data: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not data:
            return rows
        sample_by_dataset: dict[int, list[dict[str, float | int | str]]] = {}
        for row in self.sample_rows:
            sample_by_dataset.setdefault(int(row["dataset_id"]), []).append(row)

        for dataset_id_tensor in torch.unique(data["dataset_id"]):
            dataset_id = int(dataset_id_tensor.item())
            mask = data["dataset_id"] == dataset_id
            summary = self._summarize_positions(data, mask, f"residual/{self._dataset_name(dataset_id)}")
            samples = sample_by_dataset.get(dataset_id, [])

            def avg_sample(key: str) -> float:
                vals = [float(r[key]) for r in samples if key in r]
                return float(sum(vals) / len(vals)) if vals else 0.0

            rows.append(
                {
                    "dataset_name": self._dataset_name(dataset_id),
                    "n_samples": len(samples),
                    "n_positions": int(mask.sum().item()),
                    "mean_ratio": float(summary.get(f"residual/{self._dataset_name(dataset_id)}/mean_ratio", torch.tensor(0.0)).item()),
                    "nb_std_mean": float(summary.get(f"residual/{self._dataset_name(dataset_id)}/nb_std_mean", torch.tensor(0.0)).item()),
                    "nb_std_std": float(summary.get(f"residual/{self._dataset_name(dataset_id)}/nb_std_std", torch.tensor(0.0)).item()),
                    "frac_abs_std_gt_3": float(summary.get(f"residual/{self._dataset_name(dataset_id)}/fraction_abs_std_gt_3", torch.tensor(0.0)).item()),
                    "frac_abs_std_gt_5": float(summary.get(f"residual/{self._dataset_name(dataset_id)}/fraction_abs_std_gt_5", torch.tensor(0.0)).item()),
                    "tail_mass_gt_3": float(summary.get(f"residual/{self._dataset_name(dataset_id)}/tail_mass_abs_gt_3", torch.tensor(0.0)).item()),
                    "tail_mass_gt_5": float(summary.get(f"residual/{self._dataset_name(dataset_id)}/tail_mass_abs_gt_5", torch.tensor(0.0)).item()),
                    "low_mu_raw_residual_bias": float(summary.get(f"residual/{self._dataset_name(dataset_id)}/low_mu_raw_residual_bias", torch.tensor(0.0)).item()),
                    "high_mu_log_bias": float(summary.get(f"residual/{self._dataset_name(dataset_id)}/high_mu_log_bias", torch.tensor(0.0)).item()),
                    "zero_fraction": float(summary.get(f"residual/{self._dataset_name(dataset_id)}/zero_fraction", torch.tensor(0.0)).item()),
                    "zero_mean_mu": float(summary.get(f"residual/{self._dataset_name(dataset_id)}/zero_mean_mu", torch.tensor(0.0)).item()),
                    "peak_mu_over_y_top1": avg_sample("peak_mu_over_y_top_1"),
                    "top1_overlap": avg_sample("top1_overlap"),
                    "alpha_mean": float(summary.get(f"residual/{self._dataset_name(dataset_id)}/alpha_mean", torch.tensor(0.0)).item()),
                }
            )
        return rows

    def compute(self) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
        data = self._concat_positions()
        if not data:
            return {}, [], {}
        all_mask = torch.ones_like(data["mu"], dtype=torch.bool)
        metrics = self._summarize_positions(data, all_mask, "residual")
        metrics.update(self._mu_bin_metrics(data))
        metrics.update(self._stratification_metrics(data))

        for dataset_id_tensor in torch.unique(data["dataset_id"]):
            dataset_id = int(dataset_id_tensor.item())
            dataset_name = self._dataset_name(dataset_id)
            metrics.update(
                self._summarize_positions(
                    data,
                    data["dataset_id"] == dataset_id,
                    f"residual/{dataset_name}",
                )
            )

        rows = self._dataset_csv_rows(data)
        report = self._decision_report(metrics)
        return metrics, rows, report

    def _decision_report(self, metrics: dict[str, torch.Tensor]) -> dict[str, Any]:
        nb_std = float(metrics.get("residual/nb_std_std", torch.tensor(0.0)).item())
        frac3 = float(metrics.get("residual/fraction_abs_std_gt_3", torch.tensor(0.0)).item())
        tail_mass3 = float(metrics.get("residual/tail_mass_abs_gt_3", torch.tensor(0.0)).item())
        low_mu_bias = float(metrics.get("residual/low_mu_raw_residual_bias", torch.tensor(0.0)).item())
        high_mu_log_bias = float(metrics.get("residual/high_mu_log_bias", torch.tensor(0.0)).item())
        zero_mu = float(metrics.get("residual/zero_mean_mu", torch.tensor(0.0)).item())
        zero_p0 = float(metrics.get("residual/zero_mean_nb_p0", torch.tensor(1.0)).item())
        recommendations = []
        if 0.75 <= nb_std <= 1.25 and frac3 < 0.02:
            recommendations.append("NB calibration looks broadly adequate.")
        if low_mu_bias > 0.1:
            recommendations.append(
                "Positive low-mu residual bias indicates systematic underprediction "
                "in the low-mean regime."
            )
        if abs(high_mu_log_bias) > 0.25:
            recommendations.append(
                "High-mu log residual bias suggests gamma/L_bio or scale misspecification."
            )
        if frac3 < 0.05 and tail_mass3 > 0.25:
            recommendations.append("Sparse residual tail concentration suggests outlier/contamination diagnostics.")
        if zero_mu > 1.0 and zero_p0 < 0.25:
            recommendations.append("Zeros with high mu and low NB p0 suggest dropout/censoring.")
        return {
            "nb_std_std": nb_std,
            "frac_abs_std_gt_3": frac3,
            "tail_mass_abs_gt_3": tail_mass3,
            "low_mu_raw_residual_bias": low_mu_bias,
            "high_mu_log_bias": high_mu_log_bias,
            "zero_mean_mu": zero_mu,
            "zero_mean_nb_p0": zero_p0,
            "recommendations": recommendations,
        }

    @staticmethod
    def export_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
        if not rows:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if str(key) not in fieldnames:
                    fieldnames.append(str(key))
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


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
    NOT change gradients. For non-negative integer targets this is the complete
    NB log-PMF and improves cross-dataset comparability of logged NLL values.
    For non-integer (replicate-averaged) targets the same formula is only the
    continuous lgamma extension, not a proper discrete likelihood.
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
                "loss NB length-tempering weights must be finite, positive, and "
                "min_weight <= max_weight."
            )

    def sanitize_mean(self, mu: torch.Tensor) -> torch.Tensor:
        """Return the finite positive NB mean used by losses and diagnostics."""
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

            mu = self.sanitize_mean(mu_phys.to(torch.float32)).to(torch.float32)
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

        loss = w_cNLL * consensus_NB
             + w_cPCC * consensus_PCC
             + w_rNLL * mean_replica_NB
             + w_rPCC * mean_replica_PCC
             + w_gamma * mean(log_gamma^2)

    The four data-term weights are explicit and mandatory. The PCC helper can
    compute raw, log1p, NB-VST, or a normalized raw/NB-VST hybrid.
    """

    # Per-position extras plotted during validation.
    PROFILE_PLOT_GROUPS = (
        ("L_bio", ("L_bio",)),
        ("rho", ("rho",)),
        ("gamma", ("gamma",)),
        ("log_sigma", ("log_sigma",)),
    )

    @property
    def dataset_balanced_loss(self) -> bool:
        """Legacy compatibility view of the explicit sample reduction."""
        return self.sample_reduction == "dataset_balanced"

    @dataset_balanced_loss.setter
    def dataset_balanced_loss(self, enabled: bool) -> None:
        # Historical tests and analysis utilities toggle this attribute after
        # construction. Preserve that interface without allowing two active
        # reducers at once.
        self.sample_reduction = (
            "dataset_balanced" if bool(enabled) else "global_weighted"
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
        self.pcc_loss_mode = str(getattr(loss_cfg, "pcc_loss_mode", "raw")).lower()
        valid_pcc_modes = {"raw", "log1p", "nb_vst", "hybrid_raw_nb_vst"}
        if self.pcc_loss_mode not in valid_pcc_modes:
            raise ValueError(
                f"loss.pcc_loss_mode must be one of {sorted(valid_pcc_modes)}, "
                f"got {self.pcc_loss_mode!r}."
            )
        self.min_pcc_target_var = float(getattr(loss_cfg, "min_pcc_target_var", 1.0e-6))
        if not math.isfinite(self.min_pcc_target_var) or self.min_pcc_target_var < 0.0:
            raise ValueError("loss.min_pcc_target_var must be finite and non-negative.")
        self.pcc_detach_alpha = bool(getattr(loss_cfg, "pcc_detach_alpha", True))
        self.pcc_raw_component_weight = float(
            getattr(loss_cfg, "pcc_raw_component_weight", 0.5)
        )
        self.pcc_nb_vst_component_weight = float(
            getattr(loss_cfg, "pcc_nb_vst_component_weight", 0.5)
        )
        if not (
            math.isfinite(self.pcc_raw_component_weight)
            and math.isfinite(self.pcc_nb_vst_component_weight)
            and self.pcc_raw_component_weight >= 0.0
            and self.pcc_nb_vst_component_weight >= 0.0
        ):
            raise ValueError("PCC component weights must be finite and non-negative.")
        if self.pcc_loss_mode == "hybrid_raw_nb_vst":
            component_sum = (
                self.pcc_raw_component_weight
                + self.pcc_nb_vst_component_weight
            )
            if not math.isclose(component_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
                raise ValueError(
                    "For hybrid_raw_nb_vst, loss.pcc_raw_component_weight and "
                    "loss.pcc_nb_vst_component_weight must sum to 1. The overall "
                    "PCC strength is controlled only by loss.loss_terms."
                )
        self.sample_reduction = resolve_sample_reduction_mode(loss_cfg)
        self.hparams["sample_reduction"] = self.sample_reduction
        self.debug_transcript_groups = bool(
            getattr(loss_cfg, "debug_transcript_groups", False)
        )
        self.train_sampling_strategy = str(
            getattr(
                getattr(self.config, "data", None),
                "train_sampling_strategy",
                "unknown",
            )
        ).strip().lower()
        if (
            self.sample_reduction == "transcript_balanced"
            and self.train_sampling_strategy
            != "transcript_grouped_multidataset_pairs"
        ):
            warnings.warn(
                "loss.sample_reduction=transcript_balanced is exact only for "
                "complete transcript groups. With "
                f"data.train_sampling_strategy={self.train_sampling_strategy!r}, "
                "the reduction is batch-local and a transcript may be split "
                "across physical batches.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.eps = float(getattr(loss_cfg, "eps", 1.0e-8))

        # This is the only objective-selection interface. All four scalar
        # weights are mandatory, which prevents hidden preset inheritance.
        self.loss_term_weights = self._resolve_loss_term_weights(loss_cfg)

        replica_weight_requested = (
            self.loss_term_weights["replica_nll"] > 0.0
            or self.loss_term_weights["replica_pcc"] > 0.0
        )
        replicas_enabled = bool(
            getattr(getattr(self.config, "data", None), "use_ribo_replicas", False)
        )
        if replica_weight_requested and not replicas_enabled:
            raise ValueError(
                "loss.loss_terms requests a positive replica_nll or replica_pcc "
                "weight, but data.use_ribo_replicas=false. Enable replica loading "
                "or set both replica weights to zero."
            )

        # Optional regularizers around the active NB + PCC objective.
        self.gamma_reg_weight = float(getattr(loss_cfg, "gamma_reg_weight", 0.0))
        if not math.isfinite(self.gamma_reg_weight) or self.gamma_reg_weight < 0.0:
            raise ValueError("loss.gamma_reg_weight must be finite and non-negative.")
        # Predicted-value floor for PCC only: predictions below this count are
        # treated as 0 ("undetected") when computing correlation. Honest (uses
        # only the prediction, never the target); 0.0 disables.
        self.pcc_prediction_floor = float(
            getattr(loss_cfg, "pcc_prediction_floor", 0.0)
        )
        if not math.isfinite(self.pcc_prediction_floor) or self.pcc_prediction_floor < 0.0:
            raise ValueError("loss.pcc_prediction_floor must be finite and non-negative.")
        residual_cfg = getattr(self.config, "residual_diagnostics", None)

        def residual_cfg_get(name: str, default: Any) -> Any:
            if residual_cfg is None:
                return default
            return getattr(residual_cfg, name, default)

        self.residual_diag_enabled = bool(residual_cfg_get("enabled", True))
        self.residual_diag_run_on_validation = bool(
            residual_cfg_get("run_on_validation", True)
        )
        self.residual_diag_export_csv = bool(residual_cfg_get("export_csv", True))
        self.residual_diag_export_sample_csv = bool(
            residual_cfg_get("export_sample_csv", True)
        )
        self.residual_diag_export_dir = str(
            residual_cfg_get("export_dir", "residual_diagnostics")
        )
        self.residual_diag = ResidualDiagnosticsAccumulator(
            dataset_id_to_name=self.dataset_id_to_name,
            eps=float(residual_cfg_get("eps", self.eps)),
            eps_count=float(residual_cfg_get("eps_count", 1.0e-3)),
            zero_threshold=float(residual_cfg_get("zero_threshold", 0.0)),
            low_count_threshold=float(residual_cfg_get("low_count_threshold", 1.0)),
            tail_thresholds=tuple(
                float(x)
                for x in residual_cfg_get("residual_tail_thresholds", [2.0, 3.0, 5.0])
            ),
            topk_fractions=tuple(
                float(x)
                for x in residual_cfg_get("topk_fractions", [0.01, 0.05, 0.10])
            ),
            quantile_max_values=int(
                residual_cfg_get("quantile_max_values", 1_000_000)
            ),
        )
        self._grouped_batch_logging_enabled = False
        self._grouped_optimizer_batch_plan: dict[str, Any] = {}
        self._expected_pair_rows_by_transcript: dict[str, int] = {}
        self._train_batch_structure_records: list[dict[str, Any]] = []
        self._legacy_transcript_group_warning_emitted = False

    def configure_grouped_optimizer_batch_logging(
        self,
        *,
        plan: dict[str, Any],
        expected_pair_rows_by_transcript: dict[str, int],
        enabled: bool = True,
    ) -> None:
        """Attach the pre-Trainer grouped accumulation plan to this run."""
        self._grouped_batch_logging_enabled = bool(enabled)
        self._grouped_optimizer_batch_plan = dict(plan)
        self._expected_pair_rows_by_transcript = {
            str(transcript_id): int(pair_count)
            for transcript_id, pair_count in expected_pair_rows_by_transcript.items()
        }
        # The project uses weights-only checkpoints, so the resolved Hydra config
        # and JSON manifest are the durable provenance. Keep the same compact plan
        # in Lightning hyperparameters when a full checkpoint is requested.
        self.hparams["grouped_optimizer_batch_plan"] = dict(plan)

    def configure_transcript_group_validation(
        self,
        expected_pair_rows_by_transcript: dict[str, int],
    ) -> None:
        """Attach expected full-group sizes for split detection in training."""
        self._expected_pair_rows_by_transcript = {
            str(transcript_id): int(pair_count)
            for transcript_id, pair_count in expected_pair_rows_by_transcript.items()
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["sample_reduction"] = self.sample_reduction

    def _build_loss(self) -> nn.Module:
        loss_cfg = self.config.loss
        return NegativeBinomialProfileLoss(
            eps=loss_cfg.eps,
            log_alpha_min=float(loss_cfg.nb_log_alpha_min),
            log_alpha_max=float(loss_cfg.nb_log_alpha_max),
            sequence_reduction=str(loss_cfg.nb_sequence_reduction),
            length_temper_gamma=float(loss_cfg.nb_length_temper_gamma),
            length_temper_ref=float(loss_cfg.nb_length_temper_ref),
            length_temper_min_weight=float(loss_cfg.nb_length_temper_min_weight),
            length_temper_max_weight=float(loss_cfg.nb_length_temper_max_weight),
        )

    # ============================================================
    # Forward / batch handling
    # ============================================================

    @staticmethod
    def _legacy_group_indices_from_ids(
        ids: list[str] | tuple[str, ...],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        group_by_transcript: dict[str, int] = {}
        groups: list[int] = []
        for raw_transcript_id in ids:
            transcript_id = str(raw_transcript_id)
            if transcript_id not in group_by_transcript:
                group_by_transcript[transcript_id] = len(group_by_transcript)
            groups.append(group_by_transcript[transcript_id])
        return torch.tensor(groups, device=device, dtype=torch.long)

    @staticmethod
    def _assert_transcript_group_identity(
        transcript_ids: list[str] | tuple[str, ...],
        transcript_group_ids: torch.Tensor,
    ) -> None:
        group_values = transcript_group_ids.detach().cpu().reshape(-1).tolist()
        if len(group_values) != len(transcript_ids):
            raise AssertionError(
                "Transcript IDs and transcript_group_index have different lengths."
            )
        transcript_by_group: dict[int, str] = {}
        group_by_transcript: dict[str, int] = {}
        for transcript_id, group_index in zip(
            map(str, transcript_ids),
            map(int, group_values),
            strict=True,
        ):
            previous_transcript = transcript_by_group.setdefault(
                group_index,
                transcript_id,
            )
            previous_group = group_by_transcript.setdefault(
                transcript_id,
                group_index,
            )
            if previous_transcript != transcript_id or previous_group != group_index:
                raise AssertionError(
                    "transcript_group_index is not a one-to-one grouping of "
                    "the physical batch transcript IDs."
                )

    def _forward_batch(self, batch) -> dict[str, Any]:
        if len(batch) < 9:
            raise ValueError(f"Expected at least 9 batch fields, got {len(batch)}.")
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
        ) = batch[:9]

        # New batches always carry the raw dataset rank and its positive
        # rank-derived quality weight. Keep the shape check so old collated
        # batches remain readable when inspecting historical artifacts.
        has_quality_metadata = (
            len(batch) >= 11
            and torch.is_tensor(batch[9])
            and torch.is_tensor(batch[10])
            and batch[9].ndim == 1
            and batch[10].ndim == 1
        )
        if has_quality_metadata:
            dataset_quality_ranks = batch[9]
            dataset_quality_weights = batch[10]
            optional_values = list(batch[11:])
        else:
            dataset_quality_ranks = torch.full_like(
                sample_weights, float("nan"), dtype=torch.float32
            )
            dataset_quality_weights = torch.ones_like(
                sample_weights, dtype=torch.float32
            )
            optional_values = list(batch[9:])

        transcript_group_ids = None
        if optional_values:
            candidate = optional_values[0]
            if (
                torch.is_tensor(candidate)
                and candidate.ndim == 1
                and candidate.shape[0] == len(ids)
                and not candidate.is_floating_point()
                and candidate.dtype != torch.bool
            ):
                transcript_group_ids = candidate.to(dtype=torch.long)
                optional_values = optional_values[1:]
        if transcript_group_ids is None:
            # Compatibility for historical collated artifacts and hand-built
            # test batches. Current dataloaders always provide integer groups,
            # so this string path is not part of normal training.
            if not self._legacy_transcript_group_warning_emitted:
                warnings.warn(
                    "Batch has no transcript_group_index; deriving it once from "
                    "legacy transcript strings. Rebuild batches with the current "
                    "collate function for transcript-balanced training.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._legacy_transcript_group_warning_emitted = True
            transcript_group_ids = self._legacy_group_indices_from_ids(
                ids,
                device=dataset_ids.device,
            )
        if self.debug_transcript_groups:
            self._assert_transcript_group_identity(ids, transcript_group_ids)

        dataset_bias_sequence_features = None
        replica_profiles = None
        replica_mask = None
        if len(optional_values) == 1:
            dataset_bias_sequence_features = optional_values[0]
        elif len(optional_values) == 2:
            replica_profiles, replica_mask = optional_values
        elif len(optional_values) == 3:
            dataset_bias_sequence_features, replica_profiles, replica_mask = optional_values
        elif len(optional_values) != 0:
            raise ValueError(
                "Expected optional bias features and/or replica tensors after "
                f"batch metadata, got {len(optional_values)} fields."
            )

        mu, log_sigma, extras = self.model(
            x_packed=seq_packed,
            codon_ids=codon_ids,
            id_datasets=dataset_ids,
            mask=mask,
            target=target,
            sample_ids=ids,
            dataset_bias_sequence_features=dataset_bias_sequence_features,
            dataset_quality_weights=dataset_quality_weights,
        )

        out = {
            "dataset_ids": dataset_ids,
            "ids": ids,
            "transcript_group_ids": transcript_group_ids,
            "lengths": lengths,
            "mask": mask.bool(),
            "target": target,
            "codon_ids": codon_ids,
            "dataset_bias_sequence_features": dataset_bias_sequence_features,
            "css": css,
            "sample_weights": sample_weights,
            "dataset_quality_ranks": dataset_quality_ranks,
            "dataset_quality_weights": dataset_quality_weights,
            "mu": mu,
            "log_sigma": log_sigma,
            "extras": extras,
        }
        if replica_profiles is not None and replica_mask is not None:
            out["replica_profiles"] = replica_profiles
            out["replica_mask"] = replica_mask
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
        sample_weights: torch.Tensor | None = None,
        dataset_balanced: bool | None = None,
        transcript_group_ids: torch.Tensor | None = None,
        mode: str | None = None,
    ) -> torch.Tensor:
        """Compatibility wrapper around :func:`reduce_per_sample_quantity`."""
        values = values.reshape(-1)
        if mode is not None and dataset_balanced is not None:
            legacy_mode = (
                "dataset_balanced" if dataset_balanced else "global_weighted"
            )
            if str(mode) != legacy_mode:
                raise ValueError(
                    f"Conflicting reducer arguments mode={mode!r} and "
                    f"dataset_balanced={dataset_balanced!r}."
                )
        if mode is None:
            mode = (
                "dataset_balanced" if dataset_balanced else "global_weighted"
                if dataset_balanced is not None
                else self.sample_reduction
            )
        if sample_weights is None:
            sample_weights = torch.ones(
                values.shape[0],
                device=values.device,
                dtype=values.dtype,
            )
        if transcript_group_ids is None:
            if mode == "transcript_balanced":
                raise ValueError(
                    "transcript_group_ids is required for transcript_balanced reduction."
                )
            transcript_group_ids = torch.arange(
                values.shape[0],
                device=values.device,
                dtype=torch.long,
            )
        return reduce_per_sample_quantity(
            values=values,
            sample_weights=sample_weights.reshape(-1),
            transcript_group_ids=transcript_group_ids.reshape(-1),
            dataset_ids=dataset_ids.reshape(-1),
            mode=mode,
            eps=self.eps,
        )

    def _pcc_alpha(
        self,
        log_alpha: torch.Tensor,
        target_shape: torch.Size | tuple[int, ...],
    ) -> torch.Tensor:
        # PCC and NB must use the same finite, bounded dispersion. Previously
        # PCC had separate "min/max" fields that only replaced non-finite
        # values and did not actually clamp finite alpha values.
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

    def _pcc_loss_per_sample(self, out: dict[str, Any]) -> dict[str, torch.Tensor]:
        mu = out["mu"].float()
        target = torch.nan_to_num(
            out["target"].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        mask = out["mask"].bool() & torch.isfinite(out["target"].float())
        log_alpha = out.get("log_sigma")
        mode = self.pcc_loss_mode
        hybrid_modes = {"hybrid_raw_nb_vst": "nb_vst"}

        def masked_position_mean(v: torch.Tensor) -> torch.Tensor:
            mask_f = mask.to(dtype=v.dtype)
            return (v * mask_f).sum() / mask_f.sum().clamp_min(1.0)

        def valid_sample_mean(v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            valid_f = valid.to(dtype=v.dtype)
            return (v * valid_f).sum() / valid_f.sum().clamp_min(1.0)

        def component(component_mode: str) -> dict[str, torch.Tensor]:
            alpha_diag = torch.ones_like(mu)

            if component_mode == "raw":
                x_pcc = mu
                y_pcc = target
            elif component_mode == "log1p":
                x_pcc = torch.log1p(mu.clamp_min(0.0))
                y_pcc = torch.log1p(target.clamp_min(0.0))
            elif component_mode == "nb_vst":
                log_alpha_f = log_alpha.float()
                alpha_diag = self._pcc_alpha(log_alpha_f, mu.shape).to(
                    device=mu.device,
                    dtype=mu.dtype,
                )
                x_pcc = self._nb_vst(mu, log_alpha_f)
                y_pcc = self._nb_vst(target, log_alpha_f)

            else:
                raise ValueError(f"Unsupported pcc_loss_mode: {component_mode!r}.")

            pcc_out = masked_weighted_pcc(
                x=x_pcc,
                y=y_pcc,
                mask=mask,
                weights=None,
                min_target_var=self.min_pcc_target_var,
                eps=self.eps,
            )
            valid = pcc_out["valid"]
            loss_per_sample = 1.0 - pcc_out["pcc_per_sample"]
            loss_per_sample = torch.where(
                valid,
                loss_per_sample,
                torch.zeros_like(loss_per_sample),
            )
            return {
                "loss_per_sample": loss_per_sample,
                "pcc_per_sample": pcc_out["pcc_per_sample"],
                "valid": valid,
                "pcc_value": valid_sample_mean(pcc_out["pcc_per_sample"], valid),
                "loss_mean": valid_sample_mean(loss_per_sample, valid),
                "valid_fraction": pcc_out["valid_fraction"],
                "target_var_mean": pcc_out["target_var_mean"],
                "alpha_mean": masked_position_mean(alpha_diag),
                "alpha_min": alpha_diag[mask].amin() if bool(mask.any()) else mu.new_tensor(0.0),
                "alpha_max": alpha_diag[mask].amax() if bool(mask.any()) else mu.new_tensor(0.0),
            }

        if mode in hybrid_modes:
            raw_component = component("raw")
            nb_component = component(hybrid_modes[mode])
            raw_weight = float(self.pcc_raw_component_weight)
            nb_weight = float(self.pcc_nb_vst_component_weight)
            loss_per_sample = (
                raw_weight * raw_component["loss_per_sample"]
                + nb_weight * nb_component["loss_per_sample"]
            )
            active_component = nb_component
            raw_loss_per_sample = raw_component["loss_per_sample"]
            nb_loss_per_sample = nb_component["loss_per_sample"]
            pcc_per_sample = (
                raw_weight * raw_component["pcc_per_sample"]
                + nb_weight * nb_component["pcc_per_sample"]
            )
            valid = raw_component["valid"] | nb_component["valid"]
        else:
            active_component = component(mode)
            loss_per_sample = active_component["loss_per_sample"]
            raw_component = active_component if mode == "raw" else component("raw")
            nb_component = (
                active_component
                if mode == "nb_vst"
                else None
            )
            raw_loss_per_sample = raw_component["loss_per_sample"]
            nb_loss_per_sample = (
                nb_component["loss_per_sample"]
                if nb_component is not None
                else torch.zeros_like(loss_per_sample)
            )
            pcc_per_sample = active_component["pcc_per_sample"]
            valid = active_component["valid"]

        valid_f = valid.to(dtype=mu.dtype)
        valid_count = valid_f.sum().clamp_min(1.0)
        raw_weight_t = mu.new_tensor(float(self.pcc_raw_component_weight))
        nb_weight_t = mu.new_tensor(float(self.pcc_nb_vst_component_weight))
        raw_loss_mean = valid_sample_mean(raw_loss_per_sample, raw_component["valid"])
        nb_loss_mean = (
            valid_sample_mean(nb_loss_per_sample, nb_component["valid"])
            if nb_component is not None
            else mu.new_tensor(0.0)
        )

        return {
            "loss_per_sample": loss_per_sample,
            "pcc_per_sample": pcc_per_sample,
            "valid": valid,
            "pcc_value": (pcc_per_sample * valid_f).sum() / valid_count,
            "valid_fraction": active_component["valid_fraction"],
            "target_var_mean": active_component["target_var_mean"],
            "alpha_mean": active_component["alpha_mean"],
            "alpha_min": active_component["alpha_min"],
            "alpha_max": active_component["alpha_max"],
            "raw_loss_per_sample": raw_loss_per_sample,
            "raw_value": raw_component["pcc_value"],
            "raw_loss": raw_loss_mean,
            "raw_valid_fraction": raw_component["valid_fraction"],
            "raw_component_weight": raw_weight_t,
            "nb_vst_loss_per_sample": nb_loss_per_sample,
            "nb_vst_value": nb_component["pcc_value"] if nb_component is not None else mu.new_tensor(0.0),
            "nb_vst_loss": nb_loss_mean,
            "nb_vst_valid_fraction": nb_component["valid_fraction"] if nb_component is not None else mu.new_tensor(0.0),
            "nb_vst_component_weight": nb_weight_t,
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
        *,
        compute_nll: bool = True,
    ) -> dict[str, Any]:
        if compute_nll:
            nll_per_sample = self.loss_fn(
                mu_phys=out["mu"].float(),
                log_sigma=out["log_sigma"].float(),
                y_true=out["target"].float(),
                mask=out["mask"].bool(),
                return_per_sample=True,
            )
        else:
            nll_per_sample = target.new_zeros(target.shape[0])
        # PCC is measured on the floored prediction (values below the floor are
        # treated as 0), so the model is rewarded for pushing mu down at zeros.
        # This uses only the prediction, never the target.
        mu_floored = self._apply_pcc_floor(out["mu"].float())
        pcc_out = {**out, "mu": mu_floored}
        raw_mu_pcc_per_sample = self._pearson_per_sample(
            pred=mu_floored,
            target=target,
            mask=mask,
        )
        pcc_diag = self._pcc_loss_per_sample(pcc_out)

        with torch.no_grad():
            likelihood_positive_mean = self.loss_fn.sanitize_mean(mu_floored).float()
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
            "nll_per_sample": nll_per_sample,
            "raw_mu_pcc_per_sample": raw_mu_pcc_per_sample,
            "pcc_diag": pcc_diag,
            "likelihood_mu_pcc_per_sample": likelihood_mu_pcc_per_sample,
            "log1p_mse_per_sample": log1p_mse_per_sample,
        }

    def _compute_replica_loss_terms(
        self,
        out: dict[str, Any],
        *,
        compute_nll: bool = True,
        compute_pcc: bool = True,
    ) -> dict[str, Any] | None:
        replica_profiles = out.get("replica_profiles")
        replica_mask = out.get("replica_mask")
        if not (torch.is_tensor(replica_profiles) and torch.is_tensor(replica_mask)):
            return None

        target_reps = replica_profiles.float()
        rep_mask = replica_mask.bool().to(device=target_reps.device)
        B, R, T = target_reps.shape
        replica_counts = rep_mask.to(dtype=target_reps.dtype).sum(dim=1)
        result: dict[str, Any] = {
            "metrics": {"replica_count_mean": replica_counts.mean()},
        }
        if not compute_nll and not compute_pcc:
            return result

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
                "Model output is missing extras['normalized_shape']; replica NLL "
                "must reuse the exact final shape used by the consensus forward."
            )
        shape = extras["normalized_shape"].float().to(device=target_reps.device)
        if tuple(shape.shape) != (B, T):
            raise ValueError(
                "normalized_shape must match [batch, positions] for replica NLL; "
                f"got {tuple(shape.shape)} instead of {(B, T)}."
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

        if compute_nll:
            nll_flat = self.loss_fn(
                mu_phys=flat_mu,
                log_sigma=flat_log_sigma,
                y_true=flat_target,
                mask=flat_mask,
                return_per_sample=True,
            )
            result["nll_per_sample"] = self._average_over_valid_replicas(
                nll_flat,
                rep_mask,
            )

        if compute_pcc:
            flat_out = {
                "mu": flat_mu,
                "target": flat_target,
                "mask": flat_mask,
                "log_sigma": flat_log_sigma,
            }
            pcc_flat = self._pcc_loss_per_sample(flat_out)
            pcc_diag: dict[str, torch.Tensor] = dict(pcc_flat)
            for key in (
                "loss_per_sample",
                "pcc_per_sample",
                "raw_loss_per_sample",
                "nb_vst_loss_per_sample",
            ):
                if torch.is_tensor(pcc_flat.get(key)):
                    pcc_diag[key] = self._average_over_valid_replicas(
                        pcc_flat[key],
                        rep_mask,
                    )
            flat_rep_mask = rep_mask.reshape(-1).to(device=target_reps.device)
            if torch.is_tensor(pcc_flat.get("valid")):
                active_valid = (
                    pcc_flat["valid"].to(device=target_reps.device) & flat_rep_mask
                )
                sample_valid = active_valid.reshape(B, R).any(dim=1)
                pcc_diag["valid"] = sample_valid
                pcc_diag["valid_fraction"] = sample_valid.to(
                    dtype=target_reps.dtype
                ).mean()

            raw_mu_pcc_flat = self._pearson_per_sample(
                pred=flat_mu,
                target=flat_target,
                mask=flat_mask,
            )
            result["raw_mu_pcc_per_sample"] = self._average_over_valid_replicas(
                raw_mu_pcc_flat,
                rep_mask,
            )
            with torch.no_grad():
                likelihood_positive_flat = self.loss_fn.sanitize_mean(flat_mu).float()
            likelihood_mu_pcc_flat = self._pearson_per_sample(
                pred=likelihood_positive_flat,
                target=flat_target,
                mask=flat_mask,
            )
            result["likelihood_mu_pcc_per_sample"] = (
                self._average_over_valid_replicas(
                    likelihood_mu_pcc_flat,
                    rep_mask,
                )
            )
            log1p_mse_flat = self._masked_mean(
                (torch.log1p(likelihood_positive_flat) - torch.log1p(flat_target)).pow(2),
                flat_mask,
            )
            result["log1p_mse_per_sample"] = self._average_over_valid_replicas(
                log1p_mse_flat,
                rep_mask,
            )
            result["pcc_diag"] = pcc_diag

        return result

    _LOSS_TERM_NAMES = (
        "consensus_nll",
        "consensus_pcc",
        "replica_nll",
        "replica_pcc",
    )

    def _resolve_loss_term_weights(self, loss_cfg: Any) -> dict[str, float]:
        """Read the one explicit, complete four-term objective definition."""
        terms_cfg = getattr(loss_cfg, "loss_terms", None)
        if terms_cfg is None:
            raise ValueError(
                "loss.loss_terms is required and must explicitly define "
                f"{', '.join(self._LOSS_TERM_NAMES)}. Implicit objective presets "
                "are not supported."
            )

        resolved: dict[str, float] = {}
        missing: list[str] = []
        for name in self._LOSS_TERM_NAMES:
            raw = getattr(terms_cfg, name, None)
            if raw is None:
                missing.append(name)
                continue
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(
                    f"loss.loss_terms.{name} must be a scalar number, got "
                    f"{type(raw).__name__}."
                )
            value = float(raw)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"loss.loss_terms.{name} must be finite and non-negative, "
                    f"got {value}."
                )
            resolved[name] = value

        if missing:
            raise ValueError(
                "loss.loss_terms must define all four terms; missing: "
                + ", ".join(missing)
            )
        if not any(value > 0.0 for value in resolved.values()):
            raise ValueError("At least one loss.loss_terms weight must be positive.")
        return resolved

    def _compute_loss_and_metrics(self, out: dict[str, Any]) -> dict[str, torch.Tensor]:
        dataset_ids = out["dataset_ids"]
        sample_weights = out.get("sample_weights")
        transcript_group_ids = out.get("transcript_group_ids")
        if not torch.is_tensor(transcript_group_ids):
            transcript_ids = out.get("ids")
            if isinstance(transcript_ids, (list, tuple)):
                transcript_group_ids = self._legacy_group_indices_from_ids(
                    transcript_ids,
                    device=dataset_ids.device,
                )
            else:
                # Hand-built historical diagnostic dictionaries sometimes have
                # no transcript strings. Treat rows as distinct groups; current
                # collated training/validation batches always carry real groups.
                transcript_group_ids = torch.arange(
                    dataset_ids.reshape(-1).shape[0],
                    device=dataset_ids.device,
                    dtype=torch.long,
                )
            out["transcript_group_ids"] = transcript_group_ids
        mask = out["mask"].bool() & torch.isfinite(out["target"].float())
        target = torch.nan_to_num(out["target"].float(), nan=0.0, posinf=0.0, neginf=0.0)

        term_w = self.loss_term_weights
        consensus_terms = self._compute_consensus_loss_terms(
            out,
            target,
            mask,
            compute_nll=term_w["consensus_nll"] > 0.0,
        )
        replica_terms = self._compute_replica_loss_terms(
            out,
            compute_nll=term_w["replica_nll"] > 0.0,
            compute_pcc=term_w["replica_pcc"] > 0.0,
        )
        replica_available = replica_terms is not None
        if (
            term_w["replica_nll"] > 0.0 or term_w["replica_pcc"] > 0.0
        ) and not replica_available:
            raise RuntimeError(
                "The configured objective requires replica targets, but this batch "
                "does not contain replica_profiles and replica_mask."
            )

        # Choose which source (consensus vs raw-replica) feeds the reported
        # top-line NLL and PCC diagnostics: prefer whichever term actually
        # carries positive weight in the optimized loss, preferring consensus on
        # ties so existing dashboards keep the same meaning by default.
        def _report_source(consensus_key: str, replica_key: str) -> dict[str, Any]:
            consensus_on = term_w[consensus_key] > 0.0
            replica_on = replica_available and term_w[replica_key] > 0.0
            if consensus_on or not replica_on:
                return consensus_terms
            return replica_terms

        nll_source = _report_source("consensus_nll", "replica_nll")
        pcc_source = _report_source("consensus_pcc", "replica_pcc")

        nll_per_sample = nll_source["nll_per_sample"]
        raw_mu_pcc_per_sample = pcc_source["raw_mu_pcc_per_sample"]
        pcc_diag = pcc_source["pcc_diag"]
        likelihood_mu_pcc_per_sample = pcc_source["likelihood_mu_pcc_per_sample"]
        log1p_mse_per_sample = pcc_source["log1p_mse_per_sample"]

        pcc_loss_per_sample = pcc_diag["loss_per_sample"]
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
        gamma_reg_per_sample = self._gamma_regularization_per_sample(out)
        effective_gamma_reg_weight = nll_per_sample.new_tensor(self.gamma_reg_weight)
        def _term_tensor(name: str) -> torch.Tensor:
            return nll_per_sample.new_tensor(term_w[name])

        # Optimized loss: one explicit weighted sum of the four elementary
        # terms plus the gamma regularizer. A zero weight disables a term.
        per_sample_total_loss = effective_gamma_reg_weight * gamma_reg_per_sample
        per_sample_total_loss = per_sample_total_loss + _term_tensor(
            "consensus_nll"
        ) * consensus_terms["nll_per_sample"]
        per_sample_total_loss = per_sample_total_loss + _term_tensor(
            "consensus_pcc"
        ) * consensus_terms["pcc_diag"]["loss_per_sample"]

        weighted_replica_loss_per_sample = torch.zeros_like(per_sample_total_loss)
        if replica_available:
            if term_w["replica_nll"] > 0.0:
                weighted_replica_loss_per_sample = (
                    weighted_replica_loss_per_sample
                    + _term_tensor("replica_nll") * replica_terms["nll_per_sample"]
                )
            if term_w["replica_pcc"] > 0.0:
                weighted_replica_loss_per_sample = (
                    weighted_replica_loss_per_sample
                    + _term_tensor("replica_pcc")
                    * replica_terms["pcc_diag"]["loss_per_sample"]
                )
            per_sample_total_loss = (
                per_sample_total_loss + weighted_replica_loss_per_sample
            )

        def reduce_samples(
            values: torch.Tensor,
            reduction_mode: str | None = None,
            weights: torch.Tensor | None = sample_weights,
        ) -> torch.Tensor:
            return self._aggregate_per_sample(
                values,
                dataset_ids,
                weights,
                transcript_group_ids=transcript_group_ids,
                mode=reduction_mode or self.sample_reduction,
            )

        loss_global_weighted = reduce_samples(
            per_sample_total_loss,
            "global_weighted",
        )
        loss_dataset_balanced = reduce_samples(
            per_sample_total_loss,
            "dataset_balanced",
        )
        loss_transcript_balanced = reduce_samples(
            per_sample_total_loss,
            "transcript_balanced",
        )
        loss_by_reduction = {
            "global_weighted": loss_global_weighted,
            "dataset_balanced": loss_dataset_balanced,
            "transcript_balanced": loss_transcript_balanced,
        }
        loss = loss_by_reduction[self.sample_reduction]
        loss_per_sample = per_sample_total_loss

        nll_global = reduce_samples(nll_per_sample, "global_weighted")
        pcc_loss_global = reduce_samples(
            pcc_loss_per_sample,
            "global_weighted",
        )
        nll_dataset_balanced = reduce_samples(
            nll_per_sample,
            "dataset_balanced",
        )
        pcc_loss_dataset_balanced = reduce_samples(
            pcc_loss_per_sample,
            "dataset_balanced",
        )
        nll_transcript_balanced = reduce_samples(
            nll_per_sample,
            "transcript_balanced",
        )
        pcc_loss_transcript_balanced = reduce_samples(
            pcc_loss_per_sample,
            "transcript_balanced",
        )
        gamma_reg = reduce_samples(gamma_reg_per_sample)
        nll = {
            "global_weighted": nll_global,
            "dataset_balanced": nll_dataset_balanced,
            "transcript_balanced": nll_transcript_balanced,
        }[self.sample_reduction]
        pcc_loss = {
            "global_weighted": pcc_loss_global,
            "dataset_balanced": pcc_loss_dataset_balanced,
            "transcript_balanced": pcc_loss_transcript_balanced,
        }[self.sample_reduction]
        pcc_value = reduce_samples(pcc_diag["pcc_per_sample"])
        pcc_raw_loss = reduce_samples(pcc_diag["raw_loss_per_sample"])
        pcc_nb_vst_loss = reduce_samples(pcc_diag["nb_vst_loss_per_sample"])

        def aggregate_active(v: torch.Tensor) -> torch.Tensor:
            return reduce_samples(v)

        consensus_nll = aggregate_active(consensus_terms["nll_per_sample"])
        consensus_pcc_loss = aggregate_active(
            consensus_terms["pcc_diag"]["loss_per_sample"]
        )
        consensus_pcc_value = aggregate_active(
            consensus_terms["pcc_diag"]["pcc_per_sample"]
        )

        zero_metric = target.new_tensor(0.0)
        replica_nll = (
            aggregate_active(replica_terms["nll_per_sample"])
            if replica_terms is not None and "nll_per_sample" in replica_terms
            else zero_metric
        )
        replica_pcc_loss = (
            aggregate_active(replica_terms["pcc_diag"]["loss_per_sample"])
            if replica_terms is not None and "pcc_diag" in replica_terms
            else zero_metric
        )
        replica_pcc_value = (
            aggregate_active(replica_terms["pcc_diag"]["pcc_per_sample"])
            if replica_terms is not None and "pcc_diag" in replica_terms
            else zero_metric
        )

        weighted_replica_loss = aggregate_active(
            weighted_replica_loss_per_sample
        )

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
        gamma_skipped_mask = mask & (~gamma_applied_diag.to(device=mask.device).bool())
        mean_log_gamma_per_sample = pos_mean(log_gamma_diag)
        std_log_gamma_per_sample = torch.sqrt(
            pos_mean((log_gamma_diag - mean_log_gamma_per_sample.reshape(-1, 1)).pow(2))
        )
        library_depth = (target * mask_f).sum(dim=1)
        # These are global profile-level metrics, deliberately independent of
        # the configured sample reduction: one answers the usual unweighted validation
        # question, the other evaluates the reliability-weighted population.
        # The weighted value is diagnostic only and never changes gradients.
        mu_pcc_unweighted = self._aggregate_per_sample(
            raw_mu_pcc_per_sample,
            dataset_ids,
            sample_weights=None,
            dataset_balanced=False,
        )
        mu_pcc_weighted = self._aggregate_per_sample(
            raw_mu_pcc_per_sample,
            dataset_ids,
            sample_weights=sample_weights,
            dataset_balanced=False,
        )

        metrics: dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_per_sample": loss_per_sample,
            # Historical compatibility alias.
            "loss_global_unbalanced": loss_global_weighted,
            "loss_global_weighted": loss_global_weighted,
            "loss_dataset_balanced": loss_dataset_balanced,
            "loss_transcript_balanced": loss_transcript_balanced,
            "nll": nll,
            "nll_per_sample": nll_per_sample,
            "nll_global": nll_global,
            "nll_dataset_balanced": nll_dataset_balanced,
            "nll_transcript_balanced": nll_transcript_balanced,
            "pcc_loss": pcc_loss,
            "pcc_loss_per_sample": pcc_loss_per_sample,
            "pcc_loss_global": pcc_loss_global,
            "pcc_loss_dataset_balanced": pcc_loss_dataset_balanced,
            "pcc_loss_transcript_balanced": pcc_loss_transcript_balanced,
            "consensus_nll": consensus_nll,
            "consensus_pcc_loss": consensus_pcc_loss,
            "consensus_pcc_value": consensus_pcc_value,
            "weighted_replica_loss": weighted_replica_loss,
            "weighted_replica_loss_per_sample": weighted_replica_loss_per_sample,
            "replica_nll": replica_nll,
            "replica_pcc_loss": replica_pcc_loss,
            "replica_pcc_value": replica_pcc_value,
            "loss_term_weight_consensus_nll": target.new_tensor(
                self.loss_term_weights["consensus_nll"]
            ),
            "loss_term_weight_consensus_pcc": target.new_tensor(
                self.loss_term_weights["consensus_pcc"]
            ),
            "loss_term_weight_replica_nll": target.new_tensor(
                self.loss_term_weights["replica_nll"]
            ),
            "loss_term_weight_replica_pcc": target.new_tensor(
                self.loss_term_weights["replica_pcc"]
            ),
            "gamma_reg": gamma_reg,
            "gamma_reg_per_sample": gamma_reg_per_sample,
            "gamma_reg_weight": effective_gamma_reg_weight,
            "pcc_value": pcc_value,
            "pcc_valid_fraction": pcc_diag["valid_fraction"],
            "pcc_raw_loss": pcc_raw_loss,
            "pcc_raw_loss_per_sample": pcc_diag["raw_loss_per_sample"],
            "pcc_raw_value": pcc_diag["raw_value"],
            "pcc_raw_component_weight": pcc_diag["raw_component_weight"],
            "pcc_raw_valid_fraction": pcc_diag["raw_valid_fraction"],
            "pcc_nb_vst_loss": pcc_nb_vst_loss,
            "pcc_nb_vst_loss_per_sample": pcc_diag["nb_vst_loss_per_sample"],
            "pcc_nb_vst_value": pcc_diag["nb_vst_value"],
            "pcc_nb_vst_component_weight": pcc_diag["nb_vst_component_weight"],
            "pcc_nb_vst_valid_fraction": pcc_diag["nb_vst_valid_fraction"],
            "pcc_target_var_mean": pcc_diag["target_var_mean"],
            "pcc_alpha_mean": pcc_diag["alpha_mean"],
            "pcc_alpha_min": pcc_diag["alpha_min"],
            "pcc_alpha_max": pcc_diag["alpha_max"],
            # The legacy name remains an unweighted compatibility alias.
            "mu_pcc": mu_pcc_unweighted,
            "mu_pcc_unweighted": mu_pcc_unweighted,
            "mu_pcc_weighted": mu_pcc_weighted,
            "mu_pcc_per_sample": raw_mu_pcc_per_sample,
            "L_bio_pcc": reduce_samples(L_bio_pcc_per_sample),
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
            "likelihood_mu_pcc": reduce_samples(
                likelihood_mu_pcc_per_sample
            ),
            "log1p_mse": reduce_samples(log1p_mse_per_sample),
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
        if replica_terms is not None:
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
        "consensus_nll",
        "consensus_pcc_value",
        "weighted_replica_loss",
        "replica_nll",
        "replica_pcc_value",
        "replica_count_mean",
        "loss_term_weight_consensus_nll",
        "loss_term_weight_consensus_pcc",
        "loss_term_weight_replica_nll",
        "loss_term_weight_replica_pcc",
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
        "gamma_min",
        "gamma_max",
        "nb_alpha_mean",
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

        self.log(
            f"{stage}_loss",
            metrics["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        # All three views reuse the already-computed per-sample objective; no
        # second forward pass is performed. ``*_loss`` above remains the
        # configured checkpoint/early-stopping objective.
        for reduction_mode in sorted(SAMPLE_REDUCTION_MODES):
            self.log(
                f"{stage}/loss_{reduction_mode}",
                metrics[f"loss_{reduction_mode}"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )
        # `*_mu_pcc` is retained as the unweighted compatibility alias used
        # by checkpoint monitoring.  `*_mu_pcc_weighted` is reduced across
        # batches using sum(sample_weight), making it the exact weighted mean
        # over the epoch rather than a mean of batch-level weighted means.
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
        sample_weights = out.get("sample_weights")
        if sample_weights is None:
            weighted_batch_size = metrics["mu_pcc_weighted"].new_tensor(float(batch_size))
        else:
            weighted_batch_size = (
                sample_weights.detach()
                .to(
                    device=metrics["mu_pcc_weighted"].device,
                    dtype=metrics["mu_pcc_weighted"].dtype,
                )
                .clamp_min(0.0)
                .sum()
                .clamp_min(self.eps)
            )
        self.log(
            f"{stage}_mu_pcc_weighted",
            metrics["mu_pcc_weighted"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=weighted_batch_size,
            sync_dist=sync_dist,
        )

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
            }.get(name)
            log_name = (
                f"{stage}/{metric_path}"
                if metric_path is not None
                else f"{stage}_{name}"
            )
            self.log(
                log_name,
                value,
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
        sync_dist = bool(getattr(self.config.trainer, "sync_dist_logs", False))

        per_sample = {
            "loss": metrics["loss_per_sample"],
            "nll": metrics["nll_per_sample"],
            "pcc_loss": metrics["pcc_loss_per_sample"],
            "pcc_raw_loss": metrics["pcc_raw_loss_per_sample"],
            "pcc_nb_vst_loss": metrics["pcc_nb_vst_loss_per_sample"],
            "mu_pcc": metrics["mu_pcc_per_sample"],
            "mu_pcc_unweighted": metrics["mu_pcc_per_sample"],
            "L_bio_pcc": metrics["L_bio_pcc_per_sample"],
        }
        sample_weights = out.get("sample_weights")
        if sample_weights is None:
            sample_weights = torch.ones_like(metrics["mu_pcc_per_sample"])
        else:
            sample_weights = sample_weights.detach().to(
                device=metrics["mu_pcc_per_sample"].device,
                dtype=metrics["mu_pcc_per_sample"].dtype,
            ).clamp_min(0.0)

        extras = out["extras"]
        gamma = extras.get("gamma", torch.ones_like(extras["L_bio"])).float()

        for dataset_id_tensor in unique_dataset_ids:
            dataset_name = self._dataset_name(int(dataset_id_tensor.item()))
            sample_mask = dataset_ids == dataset_id_tensor
            dataset_sample_count = int(sample_mask.sum().item())

            for metric_name, values in per_sample.items():
                self.log(
                    f"{stage}_{metric_name}/{dataset_name}",
                    values[sample_mask].mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    # Lightning weights epoch reductions by batch_size. Use the
                    # number of samples from this dataset, not the full mixed
                    # batch, so this is the true per-dataset sample mean.
                    batch_size=dataset_sample_count,
                    sync_dist=sync_dist,
                )

            dataset_weights = sample_weights[sample_mask]
            weighted_pcc = (metrics["mu_pcc_per_sample"][sample_mask] * dataset_weights).sum()
            weighted_pcc = weighted_pcc / dataset_weights.sum().clamp_min(self.eps)
            self.log(
                f"{stage}_mu_pcc_weighted/{dataset_name}",
                weighted_pcc,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                # Weight this epoch reduction by the sum of transcript
                # weights, so it is exact across mixed batches.
                batch_size=dataset_weights.sum().clamp_min(self.eps),
                sync_dist=sync_dist,
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
                    sync_dist=sync_dist,
                )

    # ============================================================
    # Profile plots
    # ============================================================

    def on_validation_epoch_start(self) -> None:
        self._val_profile_plot_logged_this_epoch = False
        if self.residual_diag_enabled and self.residual_diag_run_on_validation:
            self.residual_diag.reset()

    @staticmethod
    def _metrics_to_device(
        metrics: dict[str, torch.Tensor],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Move CPU-computed scalar diagnostics to the DDP reduction device."""
        moved: dict[str, torch.Tensor] = {}
        for name, value in metrics.items():
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            if value.numel() != 1:
                raise ValueError(
                    f"Logged residual metric {name!r} must be scalar, got "
                    f"shape {tuple(value.shape)}."
                )
            moved[name] = value.detach().reshape(()).to(device=device)
        return moved

    def on_validation_epoch_end(self) -> None:
        if not (self.residual_diag_enabled and self.residual_diag_run_on_validation):
            return
        metrics, rows, report = self.residual_diag.compute()
        if metrics:
            sync_dist = bool(getattr(self.config.trainer, "sync_dist_logs", False))
            # ResidualDiagnosticsAccumulator intentionally works on CPU to keep
            # full validation profiles off GPU memory. NCCL cannot all-reduce
            # CPU tensors, so return only the final scalar summaries to the
            # Lightning device before distributed logging.
            metrics = self._metrics_to_device(metrics, self.device)
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=sync_dist,
            )
        if self.residual_diag_export_csv and rows:
            log_dir = None
            if self.logger is not None:
                log_dir = getattr(self.logger, "log_dir", None)
            base_dir = Path(log_dir) if log_dir else Path(".")
            export_dir = base_dir / self.residual_diag_export_dir
            epoch = int(getattr(self, "current_epoch", 0))
            if not hasattr(self, "trainer") or getattr(self.trainer, "is_global_zero", True):
                self.residual_diag.export_csv(
                    rows,
                    export_dir / f"residual_diagnostics_by_dataset_epoch_{epoch}.csv",
                )
                if self.residual_diag_export_sample_csv and self.residual_diag.sample_rows:
                    self.residual_diag.export_csv(
                        self.residual_diag.sample_rows,
                        export_dir / f"residual_diagnostics_by_sample_epoch_{epoch}.csv",
                    )
                if report:
                    export_dir.mkdir(parents=True, exist_ok=True)
                    (export_dir / f"residual_diagnostics_report_epoch_{epoch}.json").write_text(
                        json.dumps(report, indent=2, sort_keys=True)
                    )
        self.residual_diag.reset()

    def _likelihood_positive_mean(self, out: dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            return self.loss_fn.sanitize_mean(out["mu"].detach().float()).float()

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
        if actual != expected:
            raise RuntimeError(
                "Trainer accumulation changed after grouped batch planning: "
                f"planned={expected}, trainer={actual}. The factor must be fixed "
                "before optimizer and scheduler initialization."
            )

    def on_train_epoch_start(self) -> None:
        self._train_batch_structure_records = []

    def _record_train_batch_structure(self, out: dict[str, Any]) -> None:
        validate_complete_groups = (
            self.sample_reduction == "transcript_balanced"
            and self.train_sampling_strategy
            == "transcript_grouped_multidataset_pairs"
            and bool(self._expected_pair_rows_by_transcript)
        )
        if not self._grouped_batch_logging_enabled and not validate_complete_groups:
            return
        dataset_ids = out["dataset_ids"].detach().reshape(-1).long()
        transcript_group_ids = (
            out["transcript_group_ids"].detach().reshape(-1).long()
        )
        transcript_ids = [str(value) for value in out["ids"]]
        if dataset_ids.shape != transcript_group_ids.shape:
            raise RuntimeError(
                "Batch dataset IDs and transcript group IDs have different shapes."
            )
        unique_groups, inverse = torch.unique(
            transcript_group_ids,
            sorted=False,
            return_inverse=True,
        )
        group_count = unique_groups.numel()
        row_counts = torch.bincount(inverse, minlength=group_count)
        unique_group_dataset_pairs = torch.unique(
            torch.stack((inverse, dataset_ids), dim=1),
            dim=0,
        )
        distinct_dataset_counts = torch.bincount(
            unique_group_dataset_pairs[:, 0],
            minlength=group_count,
        )
        sample_weights = out.get("sample_weights")
        if sample_weights is None:
            sample_weights = torch.ones_like(inverse, dtype=torch.float32)
        else:
            sample_weights = sample_weights.detach().reshape(-1).float()
        transcript_weight_sums = torch.zeros(
            group_count,
            device=sample_weights.device,
            dtype=torch.float32,
        )
        transcript_weight_sums.index_add_(0, inverse, sample_weights)

        inverse_cpu = inverse.detach().cpu().tolist()
        row_counts_cpu = row_counts.detach().cpu().tolist()
        distinct_counts_cpu = distinct_dataset_counts.detach().cpu().tolist()
        transcript_by_local_group: dict[int, str] = {}
        for transcript_id, local_group in zip(
            transcript_ids,
            inverse_cpu,
            strict=True,
        ):
            previous = transcript_by_local_group.setdefault(
                int(local_group),
                transcript_id,
            )
            if self.debug_transcript_groups and previous != transcript_id:
                raise AssertionError(
                    "A transcript loss group contains multiple transcript IDs."
                )

        complete_groups = 0
        incomplete_transcripts: list[str] = []
        for local_group in range(group_count):
            transcript_id = transcript_by_local_group[local_group]
            row_count = int(row_counts_cpu[local_group])
            distinct_count = int(distinct_counts_cpu[local_group])
            expected = self._expected_pair_rows_by_transcript.get(transcript_id)
            if (
                expected is not None
                and expected > 0
                and distinct_count == expected
                and row_count >= expected
                and row_count % expected == 0
            ):
                complete_groups += 1
            elif expected is not None:
                incomplete_transcripts.append(transcript_id)
        if (
            incomplete_transcripts
            and self.sample_reduction == "transcript_balanced"
            and self.train_sampling_strategy
            == "transcript_grouped_multidataset_pairs"
        ):
            preview = ", ".join(incomplete_transcripts[:5])
            raise RuntimeError(
                "Transcript-balanced loss received incomplete grouped-sampler "
                f"transcript group(s): {preview}. The reduction would only be "
                "batch-local, so training is stopped instead of silently using "
                "an incomplete transcript."
            )

        if not self._grouped_batch_logging_enabled:
            return

        datasets_per_transcript = tuple(map(int, distinct_counts_cpu))
        transcript_weight_sum_values = tuple(
            map(float, transcript_weight_sums.detach().cpu().tolist())
        )
        self._train_batch_structure_records.append(
            {
                "pair_row_count": len(transcript_ids),
                "unique_transcript_count": group_count,
                "distinct_dataset_count": int(torch.unique(dataset_ids).numel()),
                "datasets_per_transcript": datasets_per_transcript,
                "transcript_weight_sums": transcript_weight_sum_values,
                "complete_group_count": complete_groups,
                "group_count": group_count,
            }
        )

        ds_values = distinct_dataset_counts.float()
        weight_values = transcript_weight_sums.float()
        weight_mean = weight_values.mean() if group_count else weight_values.new_tensor(0.0)
        weight_cv = (
            weight_values.std(unbiased=False) / weight_mean.clamp_min(self.eps)
            if group_count
            else weight_values.new_tensor(0.0)
        )
        step_metrics = {
            "train_batch/pairs": float(len(transcript_ids)),
            "train_batch/unique_transcripts": float(group_count),
            "train_batch/unique_transcripts_per_microbatch": float(group_count),
            "train_batch/datasets_per_transcript_mean": ds_values.mean(),
            "train_batch/datasets_per_transcript_min": ds_values.amin(),
            "train_batch/datasets_per_transcript_max": ds_values.amax(),
            "train_batch/transcript_weight_sum_mean": weight_mean,
            "train_batch/transcript_weight_sum_min": weight_values.amin(),
            "train_batch/transcript_weight_sum_max": weight_values.amax(),
            "train_batch/transcript_weight_sum_cv": weight_cv,
        }
        for name, value in step_metrics.items():
            self.log(
                name,
                torch.as_tensor(value, device=self.device, dtype=torch.float32),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
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
        gathered: list[list[dict[str, Any]] | None] = [
            None for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather_object(gathered, list(records))
        return [
            record
            for rank_records in gathered
            if rank_records is not None
            for record in rank_records
        ]

    def on_train_epoch_end(self) -> None:
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
        transcript_weight_sums = [
            float(value)
            for record in records
            for value in record["transcript_weight_sums"]
        ]
        complete_groups = sum(
            int(record["complete_group_count"]) for record in records
        )
        total_groups = sum(int(record["group_count"]) for record in records)

        def summary(values: list[float]) -> tuple[float, float, float, float]:
            if not values:
                return 0.0, 0.0, 0.0, 0.0
            ordered = sorted(values)
            count = len(ordered)
            midpoint = count // 2
            if count % 2:
                median = ordered[midpoint]
            else:
                median = 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
            return (
                float(sum(ordered) / count),
                float(median),
                float(ordered[0]),
                float(ordered[-1]),
            )

        pair_mean, _, pair_min, pair_max = summary(pair_rows)
        unique_mean, unique_median, unique_min, unique_max = summary(
            unique_transcripts
        )
        ds_mean, ds_median, ds_min, ds_max = summary(datasets_per_transcript)
        weight_mean, _, weight_min, weight_max = summary(transcript_weight_sums)
        unique_cv = (
            float(np.std(unique_transcripts) / max(unique_mean, self.eps))
            if unique_transcripts
            else 0.0
        )
        weight_cv = (
            float(np.std(transcript_weight_sums) / max(weight_mean, self.eps))
            if transcript_weight_sums
            else 0.0
        )
        complete_fraction = complete_groups / max(total_groups, 1)
        plan = self._grouped_optimizer_batch_plan
        estimated_total_steps = float(self.trainer.estimated_stepping_batches)
        metrics = {
            "train_batch/pair_rows_mean": pair_mean,
            "train_batch/pair_rows_min": pair_min,
            "train_batch/pair_rows_max": pair_max,
            "train_batch/unique_transcripts_mean": unique_mean,
            "train_batch/unique_transcripts_median": unique_median,
            "train_batch/unique_transcripts_min": unique_min,
            "train_batch/unique_transcripts_max": unique_max,
            "train_batch/unique_transcripts_per_microbatch_cv": unique_cv,
            "train_batch/datasets_per_transcript_mean": ds_mean,
            "train_batch/datasets_per_transcript_median": ds_median,
            "train_batch/datasets_per_transcript_min": ds_min,
            "train_batch/datasets_per_transcript_max": ds_max,
            "train_batch/transcript_weight_sum_mean": weight_mean,
            "train_batch/transcript_weight_sum_min": weight_min,
            "train_batch/transcript_weight_sum_max": weight_max,
            "train_batch/transcript_weight_sum_cv": weight_cv,
            "train_batch/complete_group_fraction": complete_fraction,
            "train_batch/oversized_group_count": float(
                plan.get("oversized_group_count", 0)
            ),
            "train_batch/subsampled_group_count": float(
                plan.get("subsampled_group_count", 0)
            ),
            "train_batch/accumulation_factor": float(
                plan.get("resolved_accumulate_grad_batches", 1)
            ),
            "train_batch/estimated_unique_transcripts_per_optimizer_step": float(
                plan.get("estimated_unique_transcripts_per_optimizer_step", 0.0)
            ),
            "train_batch/estimated_pair_rows_per_optimizer_step": float(
                plan.get("estimated_pair_rows_per_optimizer_step", 0.0)
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
            "train/microbatches_per_epoch": float(
                plan.get("microbatches_per_epoch_per_rank", len(records))
            ),
            "train/optimizer_steps_per_epoch": float(
                plan.get("estimated_optimizer_steps_per_epoch", 0)
            ),
            "train/estimated_total_optimizer_steps": estimated_total_steps,
            "train/accumulate_grad_batches": float(
                plan.get("resolved_accumulate_grad_batches", 1)
            ),
            "train/optimizer_step_expectation_ratio": float(
                plan.get("optimizer_step_expectation_ratio", 0.0)
            ),
        }
        # Every rank has the same all-gathered summaries. Lightning writes from
        # rank zero, so sync_dist=False avoids counting the gathered data twice.
        for name, value in metrics.items():
            self.log(
                name,
                torch.tensor(value, device=self.device, dtype=torch.float32),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )

        # TODO: if unique_transcripts_per_microbatch_cv is substantial, add an
        # optional optimizer-window-exact transcript reduction. Lightning's
        # automatic accumulation gives every physical microbatch mean equal
        # weight; it is exact across the window only for equal group counts.

    def training_step(self, batch, batch_idx):
        out = self._forward_batch(batch)
        self._record_train_batch_structure(out)
        metrics = self._compute_loss_and_metrics(out)
        self._log_stage(stage="train", out=out, metrics=metrics)
        return metrics["loss"]

    def on_before_optimizer_step(self, optimizer):
        # Zero out any non-finite gradient from a degenerate batch so a single
        # bad step cannot poison the weights (runs after gradient clipping).
        params = [p for p in self.parameters() if p.grad is not None]
        if any(not torch.isfinite(p.grad).all() for p in params):
            for p in params:
                p.grad.zero_()

    def validation_step(self, batch, batch_idx):
        out = self._forward_batch(batch)
        metrics = self._compute_loss_and_metrics(out)
        self._log_stage(stage="val", out=out, metrics=metrics)
        if self.residual_diag_enabled and self.residual_diag_run_on_validation:
            self.residual_diag.update(out)
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
        bio_lr = self.config.optim.lr_biological
        rest_lr = self.config.optim.lr_rest
        weight_decay_bio = self.config.optim.weight_decay_bio
        weight_decay_rest = self.config.optim.weight_decay_rest
        bio_params = []
        rest_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_biological_parameter_name(name):
                bio_params.append(param)
            else:
                rest_params.append(param)

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
        if not param_groups:
            raise RuntimeError("No trainable parameters found.")

        opt = torch.optim.AdamW(param_groups)

        scheduler_config = self.config.optim.scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=str(scheduler_config.mode),
            factor=float(scheduler_config.factor),
            patience=int(scheduler_config.patience),
            min_lr=float(scheduler_config.min_lr),
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
