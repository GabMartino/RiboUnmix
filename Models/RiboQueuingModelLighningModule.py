from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import lightning as pl
import matplotlib
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
            additive = target - mu
            log_residual = torch.log(target.clamp_min(0.0) + self.eps_count) - torch.log(
                mu.clamp_min(0.0) + self.eps_count
            )
            relative = additive / (mu + self.eps_count)

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
                        "additive": additive[valid],
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
                additive_i = additive[sample_idx][m]
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
                    "mean_additive_residual": float(additive_i.mean().item()),
                    "mean_abs_additive_residual": float(additive_i.abs().mean().item()),
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
        additive = data["additive"][mask].float()
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
        out[f"{prefix}/low_mu_additive_bias"] = self._mean_or_zero(additive[low_mu])
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
        additive = data["additive"].float()
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
            out[f"{prefix}/mean_additive_residual"] = self._mean_or_zero(additive[m])
            out[f"{prefix}/median_additive_residual"] = additive[m].median() if bool(m.any()) else mu.new_tensor(0.0)
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
                f"{prefix}/mean_additive_residual": z,
                f"{prefix}/mean_log_residual": z,
                f"{prefix}/mean_mu": z,
                f"{prefix}/mean_target": z,
            }
        nb = data["nb_std"][mask].float()
        return {
            f"{prefix}/nb_std_mean": nb.mean(),
            f"{prefix}/nb_std_abs_mean": nb.abs().mean(),
            f"{prefix}/frac_abs_std_gt_3": (nb.abs() > 3.0).float().mean(),
            f"{prefix}/mean_additive_residual": data["additive"][mask].float().mean(),
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
                    "low_mu_additive_bias": float(summary.get(f"residual/{self._dataset_name(dataset_id)}/low_mu_additive_bias", torch.tensor(0.0)).item()),
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
        low_mu_bias = float(metrics.get("residual/low_mu_additive_bias", torch.tensor(0.0)).item())
        high_mu_log_bias = float(metrics.get("residual/high_mu_log_bias", torch.tensor(0.0)).item())
        zero_mu = float(metrics.get("residual/zero_mean_mu", torch.tensor(0.0)).item())
        zero_p0 = float(metrics.get("residual/zero_mean_nb_p0", torch.tensor(1.0)).item())
        recommendations = []
        if 0.75 <= nb_std <= 1.25 and frac3 < 0.02:
            recommendations.append("NB calibration looks broadly adequate.")
        if low_mu_bias > 0.1:
            recommendations.append("Positive low-mu additive bias suggests additive background/floor.")
        if abs(high_mu_log_bias) > 0.25:
            recommendations.append("High-mu log residual bias suggests gamma/L_bio or additive-bias adjustment.")
        if frac3 < 0.05 and tail_mass3 > 0.25:
            recommendations.append("Sparse residual tail concentration suggests outlier/contamination diagnostics.")
        if zero_mu > 1.0 and zero_p0 < 0.25:
            recommendations.append("Zeros with high mu and low NB p0 suggest dropout/censoring.")
        return {
            "nb_std_std": nb_std,
            "frac_abs_std_gt_3": frac3,
            "tail_mass_abs_gt_3": tail_mass3,
            "low_mu_additive_bias": low_mu_bias,
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

            loss = mean_replica(
                replica_nb_weight * NB_NLL
                + replica_raw_pcc_weight * (1 - PCC)
                + replica_nb_vst_pcc_weight * (1 - PCC_NB_VST)
            ) + gamma_reg_weight * mean(log_gamma^2)

    Every data-fit term uses the raw replicas. The arithmetic-mean consensus is
    retained only for diagnostics and never enters the optimized objective.
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

        self.replica_nb_weight = required_nonnegative_weight("replica_nb_weight")
        self.replica_raw_pcc_weight = required_nonnegative_weight(
            "replica_raw_pcc_weight"
        )
        self.replica_nb_vst_pcc_weight = required_nonnegative_weight(
            "replica_nb_vst_pcc_weight"
        )
        self.gamma_reg_weight = required_nonnegative_weight("gamma_reg_weight")
        if (
            self.replica_nb_weight
            + self.replica_raw_pcc_weight
            + self.replica_nb_vst_pcc_weight
            + self.gamma_reg_weight
            == 0.0
        ):
            raise ValueError("At least one of the four loss coefficients must be positive.")
        self.min_pcc_target_var = float(getattr(loss_cfg, "min_pcc_target_var", 1.0e-6))
        self.pcc_alpha_min = float(getattr(loss_cfg, "pcc_alpha_min", 1.0e-5))
        self.pcc_alpha_max = float(getattr(loss_cfg, "pcc_alpha_max", 20.0))
        self.pcc_detach_alpha = bool(getattr(loss_cfg, "pcc_detach_alpha", True))
        self.eps = float(getattr(loss_cfg, "eps", 1.0e-8))
        # Predicted-value floor for PCC only: predictions below this count are
        # treated as 0 ("undetected") when computing correlation. Honest (uses
        # only the prediction, never the target); 0.0 disables.
        self.pcc_prediction_floor = float(
            getattr(loss_cfg, "pcc_prediction_floor", 0.0)
        )
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
            remaining_values = batch[11:]
        else:
            dataset_quality_ranks = torch.full_like(
                sample_weights, float("nan"), dtype=torch.float32
            )
            dataset_quality_weights = torch.ones_like(
                sample_weights, dtype=torch.float32
            )
            remaining_values = batch[9:]

        transcript_group_index = None
        if (
            remaining_values
            and torch.is_tensor(remaining_values[0])
            and remaining_values[0].ndim == 1
            and not remaining_values[0].is_floating_point()
        ):
            transcript_group_index = remaining_values[0]
            optional_values = remaining_values[1:]
        else:
            optional_values = remaining_values

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
    ) -> torch.Tensor:
        """Return the mandatory transcript-weighted, dataset-balanced mean.

        For per-sample values ``z_s`` and transcript reliability weights ``w_s``,
        first calculate ``sum_s(w_s z_s) / sum_s(w_s)`` independently inside
        every represented dataset. The returned scalar is the arithmetic mean
        of those dataset means, so each represented dataset has one equal outer
        vote. This reduction is fixed and has no configuration selector.
        """
        values = values.reshape(-1)
        ids = dataset_ids.reshape(-1).to(device=values.device)
        w = (
            sample_weights.reshape(-1)
            .to(device=values.device, dtype=values.dtype)
            .clamp_min(0.0)
        )
        if values.numel() != ids.numel() or values.numel() != w.numel():
            raise ValueError(
                "values, dataset_ids, and sample_weights must contain the same "
                f"number of samples; got {values.numel()}, {ids.numel()}, and "
                f"{w.numel()}."
            )
        if values.numel() == 0:
            return values.sum() * 0.0

        unique_ids, inverse = torch.unique(ids, return_inverse=True)
        # unique_ids is sorted/deduped, so its element count is the number of
        # dataset buckets. Reading .numel() avoids the per-call .item() GPU sync
        # (this helper runs ~12x per step).
        K = unique_ids.numel()

        w_sums = torch.zeros(
            K, device=values.device, dtype=values.dtype
        ).scatter_add_(0, inverse, w)
        wv_sums = torch.zeros_like(w_sums).scatter_add_(0, inverse, values * w)
        per_ds_mean = wv_sums / w_sums.clamp_min(self.eps)

        return per_ds_mean.mean()

    def _pcc_alpha(
        self,
        log_alpha: torch.Tensor,
        target_shape: torch.Size | tuple[int, ...],
    ) -> torch.Tensor:
        alpha = torch.exp(_broadcast_profile_param(log_alpha, target_shape))
        alpha = torch.nan_to_num(
            alpha,
            nan=self.pcc_alpha_min,
            posinf=self.pcc_alpha_max,
            neginf=self.pcc_alpha_min,
        )
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

        def valid_sample_mean(v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            valid_f = valid.to(dtype=v.dtype)
            return (v * valid_f).sum() / valid_f.sum().clamp_min(1.0)

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

        pcc_out = masked_weighted_pcc(
            x=x_pcc,
            y=y_pcc,
            mask=mask,
            weights=None,
            min_target_var=self.min_pcc_target_var,
            eps=self.eps,
        )
        valid = pcc_out["valid"]
        loss_per_sample = torch.where(
            valid,
            1.0 - pcc_out["pcc_per_sample"],
            torch.zeros_like(pcc_out["pcc_per_sample"]),
        )
        valid_f = valid.to(dtype=mu.dtype)
        valid_count = valid_f.sum().clamp_min(1.0)
        return {
            "loss_per_sample": loss_per_sample,
            "pcc_per_sample": pcc_out["pcc_per_sample"],
            "valid": valid,
            "pcc_value": (pcc_out["pcc_per_sample"] * valid_f).sum() / valid_count,
            "loss_mean": valid_sample_mean(loss_per_sample, valid),
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
        # This uses only the prediction, never the target. It is a diagnostic;
        # the consensus is not part of the optimized loss.
        mu_floored = self._apply_pcc_floor(out["mu"].float())
        raw_mu_pcc_per_sample = self._pearson_per_sample(
            pred=mu_floored,
            target=target,
            mask=mask,
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

        flat_out = {
            "mu": self._apply_pcc_floor(flat_mu),
            "target": flat_target,
            "mask": flat_mask,
            "log_sigma": flat_log_sigma,
        }
        flat_rep_mask = rep_mask.reshape(-1).to(device=target_reps.device)

        def pcc_component(
            *,
            coefficient: float,
            transform: str,
        ) -> dict[str, torch.Tensor]:
            if coefficient == 0.0:
                zero_sample = flat_mu.new_zeros(B)
                zero_scalar = flat_mu.new_tensor(0.0)
                return {
                    "loss_per_sample": zero_sample,
                    "pcc_per_sample": zero_sample,
                    "valid": torch.zeros(B, device=flat_mu.device, dtype=torch.bool),
                    "pcc_value": zero_scalar,
                    "loss_mean": zero_scalar,
                    "valid_fraction": zero_scalar,
                    "target_var_mean": zero_scalar,
                    "alpha_mean": zero_scalar,
                    "alpha_min": zero_scalar,
                    "alpha_max": zero_scalar,
                }

            flat_diag = self._pcc_loss_per_sample(flat_out, transform=transform)
            active_valid = flat_diag["valid"].to(device=target_reps.device) & flat_rep_mask
            sample_valid = active_valid.reshape(B, R).any(dim=1)
            return {
                **flat_diag,
                "loss_per_sample": self._average_over_valid_replicas(
                    flat_diag["loss_per_sample"], rep_mask
                ),
                "pcc_per_sample": self._average_over_valid_replicas(
                    flat_diag["pcc_per_sample"], rep_mask
                ),
                "valid": sample_valid,
                "valid_fraction": sample_valid.to(dtype=target_reps.dtype).mean(),
            }

        raw_pcc_diag = pcc_component(
            coefficient=self.replica_raw_pcc_weight,
            transform="raw",
        )
        nb_vst_pcc_diag = pcc_component(
            coefficient=self.replica_nb_vst_pcc_weight,
            transform="nb_vst",
        )

        with torch.no_grad():
            likelihood_positive_flat = self.loss_fn.positive_mean_from_params(
                mu=flat_mu,
                log_sigma=flat_log_sigma,
            ).float()
        likelihood_mu_pcc_flat = self._pearson_per_sample(
            pred=likelihood_positive_flat,
            target=flat_target,
            mask=flat_mask,
        )
        likelihood_mu_pcc_per_sample = self._average_over_valid_replicas(
            likelihood_mu_pcc_flat,
            rep_mask,
        )
        log1p_mse_flat = self._masked_mean(
            (torch.log1p(likelihood_positive_flat) - torch.log1p(flat_target)).pow(2),
            flat_mask,
        )
        log1p_mse_per_sample = self._average_over_valid_replicas(
            log1p_mse_flat,
            rep_mask,
        )
        replica_counts = rep_mask.to(dtype=target_reps.dtype).sum(dim=1)
        return {
            "nll_per_sample": nll_per_sample,
            "raw_pcc_diag": raw_pcc_diag,
            "nb_vst_pcc_diag": nb_vst_pcc_diag,
            "likelihood_mu_pcc_per_sample": likelihood_mu_pcc_per_sample,
            "log1p_mse_per_sample": log1p_mse_per_sample,
            "metrics": {
                "replica_count_mean": replica_counts.mean(),
            },
        }

    def _compute_loss_and_metrics(self, out: dict[str, Any]) -> dict[str, torch.Tensor]:
        dataset_ids = out["dataset_ids"]
        sample_weights = out.get("sample_weights")
        mask = out["mask"].bool() & torch.isfinite(out["target"].float())
        target = torch.nan_to_num(out["target"].float(), nan=0.0, posinf=0.0, neginf=0.0)

        consensus_terms = self._compute_consensus_loss_terms(out, target, mask)
        replica_terms = self._compute_replica_loss_terms(out)

        # Fixed objective contract: all three data-fit terms use raw replicas.
        # The consensus is diagnostic-only.
        nll_per_sample = replica_terms["nll_per_sample"]
        raw_pcc_diag = replica_terms["raw_pcc_diag"]
        nb_vst_pcc_diag = replica_terms["nb_vst_pcc_diag"]
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
        effective_replica_raw_pcc_weight = nll_per_sample.new_tensor(
            self.replica_raw_pcc_weight
        )
        effective_replica_nb_vst_pcc_weight = nll_per_sample.new_tensor(
            self.replica_nb_vst_pcc_weight
        )
        effective_gamma_reg_weight = nll_per_sample.new_tensor(
            self.gamma_reg_weight
        )
        pcc_loss_per_sample = (
            effective_replica_raw_pcc_weight * pcc_raw_loss_per_sample
            + effective_replica_nb_vst_pcc_weight * pcc_nb_vst_loss_per_sample
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
        )
        loss_per_sample = per_sample_total_loss

        nll = self._aggregate_per_sample(
            nll_per_sample,
            dataset_ids,
            sample_weights,
        )
        pcc_loss = self._aggregate_per_sample(
            pcc_loss_per_sample,
            dataset_ids,
            sample_weights,
        )
        gamma_reg = self._aggregate_per_sample(
            gamma_reg_per_sample,
            dataset_ids,
            sample_weights,
        )
        pcc_raw_loss = self._aggregate_per_sample(
            pcc_raw_loss_per_sample,
            dataset_ids,
            sample_weights,
        )
        pcc_nb_vst_loss = self._aggregate_per_sample(
            pcc_nb_vst_loss_per_sample,
            dataset_ids,
            sample_weights,
        )

        def aggregate_active(v: torch.Tensor) -> torch.Tensor:
            return self._aggregate_per_sample(v, dataset_ids, sample_weights)

        consensus_pcc_value = aggregate_active(raw_mu_pcc_per_sample)
        consensus_pcc_loss = aggregate_active(1.0 - raw_mu_pcc_per_sample)

        replica_nll = aggregate_active(replica_terms["nll_per_sample"])
        replica_pcc_loss = pcc_loss
        pcc_weight_sum = self.replica_raw_pcc_weight + self.replica_nb_vst_pcc_weight
        if pcc_weight_sum > 0.0:
            replica_pcc_per_sample = (
                self.replica_raw_pcc_weight * raw_pcc_diag["pcc_per_sample"]
                + self.replica_nb_vst_pcc_weight
                * nb_vst_pcc_diag["pcc_per_sample"]
            ) / pcc_weight_sum
        else:
            replica_pcc_per_sample = nll_per_sample.new_zeros(nll_per_sample.shape)
        replica_pcc_value = aggregate_active(replica_pcc_per_sample)
        pcc_value = replica_pcc_value
        pcc_valid = raw_pcc_diag["valid"] | nb_vst_pcc_diag["valid"]
        pcc_valid_fraction = pcc_valid.to(dtype=target.dtype).mean()
        if pcc_weight_sum > 0.0:
            pcc_target_var_mean = (
                self.replica_raw_pcc_weight * raw_pcc_diag["target_var_mean"]
                + self.replica_nb_vst_pcc_weight
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
        gamma_skipped_mask = mask & (~gamma_applied_diag.to(device=mask.device).bool())
        mean_log_gamma_per_sample = pos_mean(log_gamma_diag)
        std_log_gamma_per_sample = torch.sqrt(
            pos_mean((log_gamma_diag - mean_log_gamma_per_sample.reshape(-1, 1)).pow(2))
        )
        library_depth = (target * mask_f).sum(dim=1)
        # Retain an unweighted pair-micro diagnostic alongside the mandatory
        # transcript-weighted, dataset-balanced PCC aggregation.
        mu_pcc_unweighted = raw_mu_pcc_per_sample.mean()
        mu_pcc_weighted = self._aggregate_per_sample(
            raw_mu_pcc_per_sample,
            dataset_ids,
            sample_weights=sample_weights,
        )

        metrics: dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_per_sample": loss_per_sample,
            "nll": nll,
            "nll_per_sample": nll_per_sample,
            "pcc_loss": pcc_loss,
            "pcc_loss_per_sample": pcc_loss_per_sample,
            "consensus_pcc_loss": consensus_pcc_loss,
            "consensus_pcc_value": consensus_pcc_value,
            "replica_nll": replica_nll,
            "replica_pcc_loss": replica_pcc_loss,
            "replica_pcc_value": replica_pcc_value,
            "gamma_reg": gamma_reg,
            "gamma_reg_per_sample": gamma_reg_per_sample,
            "gamma_global_regularizer": gamma_reg,
            "gamma_reg_weight": effective_gamma_reg_weight,
            "pcc_loss_total": pcc_loss,
            "pcc_value": pcc_value,
            "replica_nb_weight": effective_replica_nb_weight,
            "replica_raw_pcc_weight": effective_replica_raw_pcc_weight,
            "replica_nb_vst_pcc_weight": effective_replica_nb_vst_pcc_weight,
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
                effective_replica_raw_pcc_weight * pcc_raw_loss
            ),
            "pcc_nb_vst_contribution": (
                effective_replica_nb_vst_pcc_weight * pcc_nb_vst_loss
            ),
            "pcc_target_var_mean": pcc_target_var_mean,
            "pcc_alpha_mean": nb_vst_pcc_diag["alpha_mean"],
            "pcc_alpha_min": nb_vst_pcc_diag["alpha_min"],
            "pcc_alpha_max": nb_vst_pcc_diag["alpha_max"],
            # The legacy name remains an unweighted compatibility alias.
            "mu_pcc": mu_pcc_unweighted,
            "mu_pcc_unweighted": mu_pcc_unweighted,
            "mu_pcc_weighted": mu_pcc_weighted,
            "mu_pcc_per_sample": raw_mu_pcc_per_sample,
            "L_bio_pcc": self._aggregate_per_sample(
                L_bio_pcc_per_sample,
                dataset_ids,
                sample_weights,
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
            "likelihood_mu_pcc": self._aggregate_per_sample(
                likelihood_mu_pcc_per_sample,
                dataset_ids,
                sample_weights,
            ),
            "log1p_mse": self._aggregate_per_sample(log1p_mse_per_sample, dataset_ids, sample_weights),
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
        if actual != expected:
            raise RuntimeError(
                "Trainer accumulation changed after grouped batch planning: "
                f"planned={expected}, trainer={actual}. The factor must be fixed "
                "before optimizer and scheduler initialization."
            )

    def on_train_epoch_start(self) -> None:
        self._train_batch_structure_records = []

    def _record_train_batch_structure(self, batch) -> None:
        if not self._grouped_batch_logging_enabled:
            return
        dataset_ids = batch[0]
        transcript_ids = [str(value) for value in batch[1]]
        if torch.is_tensor(dataset_ids):
            dataset_id_values = dataset_ids.detach().reshape(-1).cpu().tolist()
        else:
            dataset_id_values = list(dataset_ids)
        if len(dataset_id_values) != len(transcript_ids):
            raise RuntimeError(
                "Batch dataset IDs and transcript IDs have different row counts."
            )

        rows_by_transcript: dict[str, int] = {}
        datasets_by_transcript: dict[str, set[int]] = {}
        for transcript_id, dataset_id in zip(
            transcript_ids,
            dataset_id_values,
            strict=True,
        ):
            rows_by_transcript[transcript_id] = (
                rows_by_transcript.get(transcript_id, 0) + 1
            )
            datasets_by_transcript.setdefault(transcript_id, set()).add(
                int(dataset_id)
            )

        datasets_per_transcript = tuple(
            len(datasets_by_transcript[transcript_id])
            for transcript_id in rows_by_transcript
        )
        complete_groups = 0
        for transcript_id, row_count in rows_by_transcript.items():
            expected = self._expected_pair_rows_by_transcript.get(transcript_id)
            distinct_count = len(datasets_by_transcript[transcript_id])
            if (
                expected is not None
                and expected > 0
                and distinct_count == expected
                and row_count >= expected
                and row_count % expected == 0
            ):
                complete_groups += 1

        group_count = len(rows_by_transcript)
        self._train_batch_structure_records.append(
            {
                "pair_row_count": len(transcript_ids),
                "unique_transcript_count": group_count,
                "distinct_dataset_count": len(set(map(int, dataset_id_values))),
                "datasets_per_transcript": datasets_per_transcript,
                "complete_group_count": complete_groups,
                "group_count": group_count,
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
            "train_batch/datasets_per_transcript_mean": ds_mean,
            "train_batch/datasets_per_transcript_median": ds_median,
            "train_batch/datasets_per_transcript_min": ds_min,
            "train_batch/datasets_per_transcript_max": ds_max,
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

        # TODO: if variable group sizes make microbatch-mean accumulation too
        # approximate, add a separate exact group-weighted manual accumulation
        # ablation. Do not change the active loss reduction in this feature.

    def training_step(self, batch, batch_idx):
        self._record_train_batch_structure(batch)
        out = self._forward_batch(batch)
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
