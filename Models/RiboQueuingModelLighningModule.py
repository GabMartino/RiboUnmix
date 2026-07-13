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
import torch.nn.functional as F

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


def dataset_balanced_reduce(
    per_sample_loss: torch.Tensor,
    dataset_ids: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    per_sample_loss = per_sample_loss.reshape(-1)
    dataset_ids = dataset_ids.reshape(-1).to(device=per_sample_loss.device)
    if per_sample_loss.numel() == 0:
        return per_sample_loss.mean()

    dataset_means = []
    for dataset_id in torch.unique(dataset_ids):
        idx = dataset_ids == dataset_id
        if bool(idx.any()):
            dataset_means.append(per_sample_loss[idx].mean())

    if not dataset_means:
        return per_sample_loss.sum() * 0.0 + float(eps) * 0.0

    return torch.stack(dataset_means).mean()


def summarize_gradient_conflict_from_flat_grads(
    flat_grads: dict[int, torch.Tensor],
    eps: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    if not flat_grads:
        z = torch.tensor(0.0)
        return {
            "num_datasets_present": z,
            "mean_cosine": z,
            "min_cosine": z,
            "max_cosine": z,
            "negative_cosine_fraction": z,
            "mean_dot": z,
            "min_dot": z,
            "mean_norm": z,
            "max_norm": z,
            "min_norm": z,
            "norm_ratio_max_min": z,
            "conflict_score": z,
            "dataset_ids": torch.empty(0, dtype=torch.long),
            "cosine_matrix": torch.empty(0, 0),
            "dot_matrix": torch.empty(0, 0),
            "norms": torch.empty(0),
        }

    dataset_ids = sorted(int(k) for k in flat_grads)
    grads = torch.stack([flat_grads[k].detach().float().reshape(-1) for k in dataset_ids])
    norms = torch.linalg.norm(grads, dim=1)
    dot = grads @ grads.T
    denom = (norms[:, None] * norms[None, :]).clamp_min(float(eps))
    cosine = torch.nan_to_num(dot / denom, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(dataset_ids)
    if n > 1:
        offdiag = ~torch.eye(n, dtype=torch.bool, device=cosine.device)
        offdiag_cos = cosine[offdiag]
        offdiag_dot = dot[offdiag]
        mean_cosine = offdiag_cos.mean()
        min_cosine = offdiag_cos.min()
        max_cosine = offdiag_cos.max()
        negative_fraction = (offdiag_cos < 0.0).float().mean()
        conflict_score = torch.relu(-offdiag_cos).mean()
        mean_dot = offdiag_dot.mean()
        min_dot = offdiag_dot.min()
    else:
        mean_cosine = cosine.new_tensor(0.0)
        min_cosine = cosine.new_tensor(0.0)
        max_cosine = cosine.new_tensor(0.0)
        negative_fraction = cosine.new_tensor(0.0)
        conflict_score = cosine.new_tensor(0.0)
        mean_dot = cosine.new_tensor(0.0)
        min_dot = cosine.new_tensor(0.0)

    norm_min = norms.min()
    norm_max = norms.max()
    return {
        "num_datasets_present": torch.tensor(float(n), device=grads.device),
        "mean_cosine": mean_cosine,
        "min_cosine": min_cosine,
        "max_cosine": max_cosine,
        "negative_cosine_fraction": negative_fraction,
        "mean_dot": mean_dot,
        "min_dot": min_dot,
        "mean_norm": norms.mean(),
        "max_norm": norm_max,
        "min_norm": norm_min,
        "norm_ratio_max_min": norm_max / norm_min.clamp_min(float(eps)),
        "conflict_score": conflict_score,
        "dataset_ids": torch.tensor(dataset_ids, device=grads.device, dtype=torch.long),
        "cosine_matrix": cosine,
        "dot_matrix": dot,
        "norms": norms,
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
    ) -> None:
        self.dataset_id_to_name = dict(dataset_id_to_name)
        self.eps = float(eps)
        self.eps_count = float(eps_count)
        self.zero_threshold = float(zero_threshold)
        self.low_count_threshold = float(low_count_threshold)
        self.tail_thresholds = tuple(float(x) for x in tail_thresholds)
        self.topk_fractions = tuple(float(x) for x in topk_fractions)
        self.reset()

    def reset(self) -> None:
        self.position_chunks: list[dict[str, torch.Tensor]] = []
        self.sample_rows: list[dict[str, float | int | str]] = []
        self.dataset_rows: list[dict[str, float | int | str]] = []

    def _dataset_name(self, dataset_id: int) -> str:
        return self.dataset_id_to_name.get(int(dataset_id), str(int(dataset_id)))

    @staticmethod
    def _safe_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
        if x.numel() == 0:
            return torch.tensor(0.0)
        return torch.quantile(x.float(), float(q))

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
        edges = torch.quantile(mu, torch.tensor(quantiles))
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
        edges = torch.quantile(values, torch.tensor(quantiles))
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


class PoissonProfileLoss(nn.Module):
    """
    Poisson profile NLL (kept for completeness; NB is the default).

        NLL = mu - y * log(mu) + lgamma(y + 1)

    The lgamma(y + 1) term is constant w.r.t. the model parameters (no gradient
    effect) but makes the reported NLL a proper Poisson log-density.
    """

    def __init__(
        self,
        eps: float = 1.0e-8,
        mu_min: float = 1.0e-8,
        mu_max: float = 1.0e8,
        sequence_reduction: str = "mean",
        length_temper_gamma: float = 0.85,
        length_temper_ref: float = 1000.0,
        length_temper_min_weight: float = 0.5,
        length_temper_max_weight: float = 2.0,
    ):
        super().__init__()

        self.eps = float(eps)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
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
            nan=self.mu_min,
            posinf=self.mu_max,
            neginf=self.mu_min,
        )
        return mu.clamp(min=self.mu_min, max=self.mu_max)

    def forward(
        self,
        mu_phys: torch.Tensor,
        log_sigma: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        del log_sigma

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

            nll = (
                mu
                - y * torch.log(mu.clamp_min(self.eps))
                + torch.lgamma(y + 1.0)
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
        mu_min: float = 1.0e-8,
        mu_max: float = 1.0e8,
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
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
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
            nan=self.mu_min,
            posinf=self.mu_max,
            neginf=self.mu_min,
        )
        return mu.clamp(min=self.mu_min, max=self.mu_max)

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

        gated-additive mean:
            mu[d,t,i] = S[d,t] * (gamma[d,t,i] * L_bio[t,i] + additive[d,t,i])

        Active loss:

            loss = NB_NLL
             + effective_pcc_weight * pcc_loss
             + effective_zero_calibration_weight * zero_calibration_loss
             + effective_support_calibration_weight * support_bce
             + gamma_reg_weight * mean(log_gamma^2)
             + additive_bias_l1_weight * mean(additive_bias)

    The PCC helper can compute raw, log1p, or NB-VST PCC between the output mu
    and the ground-truth profile y. All previous auxiliary
    objectives (shape-MSE / profile-CE / contamination / additive-residual /
    cross-dataset gamma neutrality / pooling / schedules) are removed. Optional
    CAGrad can be applied only to the biological branch.
    """

    # Per-position extras plotted during validation.
    PROFILE_PLOT_GROUPS = (
        ("L_bio", ("L_bio",)),
        ("rho", ("rho",)),
        ("gamma", ("gamma", "gamma_amplitude", "gamma_sparse_gate")),
        ("support", ("gamma_support_logits",)),
        ("additive_bias", ("additive_bias",)),
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
        self._cagrad_bio_grad_overrides: dict[int, torch.Tensor] = {}
        self._cagrad_hook_handles: list[Any] = []

        loss_cfg = self.config.loss
        legacy_pcc_weight = getattr(loss_cfg, "mu_pcc_loss_weight", None)
        self.pcc_loss_weight = float(
            getattr(
                loss_cfg,
                "pcc_loss_weight",
                0.0 if legacy_pcc_weight is None else legacy_pcc_weight,
            )
        )
        self.pcc_loss_enabled = bool(
            getattr(loss_cfg, "pcc_loss_enabled", self.pcc_loss_weight > 0.0)
        )
        self.pcc_loss_warmup_epochs = int(
            getattr(loss_cfg, "pcc_loss_warmup_epochs", 0)
        )
        self.pcc_loss_mode = str(getattr(loss_cfg, "pcc_loss_mode", "raw")).lower()
        self.allowed_pcc_loss_modes = {
            "raw",
            "log1p",
            "nb_vst",
            "nb_vst_weighted",
            "nb_vst_weighted_mean_ratio_gated",
            "hybrid_raw_nb_vst",
            "hybrid_raw_nb_vst_weighted",
            "hybrid_raw_nb_vst_weighted_mean_ratio_gated",
        }
        self.min_pcc_target_var = float(getattr(loss_cfg, "min_pcc_target_var", 1.0e-6))
        self.pcc_mean_ratio_gate_tau = float(
            getattr(loss_cfg, "pcc_mean_ratio_gate_tau", 0.25)
        )
        self.pcc_reliability_min = float(getattr(loss_cfg, "pcc_reliability_min", 0.05))
        self.pcc_reliability_max = float(getattr(loss_cfg, "pcc_reliability_max", 1.0))
        self.pcc_alpha_min = float(getattr(loss_cfg, "pcc_alpha_min", 1.0e-5))
        self.pcc_alpha_max = float(getattr(loss_cfg, "pcc_alpha_max", 20.0))
        self.pcc_detach_alpha = bool(getattr(loss_cfg, "pcc_detach_alpha", True))
        self.pcc_detach_reliability = bool(
            getattr(loss_cfg, "pcc_detach_reliability", True)
        )
        self.pcc_detach_mean_ratio_gate = bool(
            getattr(loss_cfg, "pcc_detach_mean_ratio_gate", True)
        )
        self.pcc_raw_component_weight = float(
            getattr(loss_cfg, "pcc_raw_component_weight", 0.03)
        )
        self.pcc_nb_vst_component_weight = float(
            getattr(loss_cfg, "pcc_nb_vst_component_weight", 0.07)
        )
        self.pcc_mean_ratio_gate_floor = float(
            getattr(loss_cfg, "pcc_mean_ratio_gate_floor", 0.0)
        )
        self.dataset_balanced_loss = bool(getattr(loss_cfg, "dataset_balanced_loss", True))
        self.eps = float(getattr(loss_cfg, "eps", 1.0e-8))

        self.replica_objective = str(
            getattr(loss_cfg, "replica_objective", "replica")
        ).lower()
        allowed_replica_objectives = {
            "replica",
            "consensus",
            "consensus_plus_replica",
        }
        self.replica_nll_weight = max(
            0.0,
            float(getattr(loss_cfg, "replica_nll_weight", 0.0)),
        )
        self.replica_pcc_loss_weight = max(
            0.0,
            float(getattr(loss_cfg, "replica_pcc_loss_weight", 0.0)),
        )

        self.zero_calibration_weight = max(
            0.0,
            float(getattr(loss_cfg, "zero_calibration_weight", 0.0)),
        )
        self.zero_calibration_warmup_epochs = max(
            0,
            int(getattr(loss_cfg, "zero_calibration_warmup_epochs", 0)),
        )
        self.zero_calibration_detach_alpha = bool(
            getattr(loss_cfg, "zero_calibration_detach_alpha", True)
        )
        self.zero_calibration_threshold = float(
            getattr(loss_cfg, "zero_calibration_threshold", 0.0)
        )
        self.zero_calibration_source = str(
            getattr(loss_cfg, "zero_calibration_source", "consensus")
        ).lower()
        allowed_zero_calibration_sources = {"consensus", "replica", "main"}
        if self.zero_calibration_source not in allowed_zero_calibration_sources:
            raise ValueError(
                "zero_calibration_source must be one of "
                f"{sorted(allowed_zero_calibration_sources)}, got "
                f"{self.zero_calibration_source!r}."
            )

        # Direct supervision of the learned entmax support. Unlike the NB zero
        # calibration score above, this term teaches the gate which positions
        # should remain active; it never inserts target zeros into the forward
        # prediction.
        self.support_calibration_weight = max(
            0.0,
            float(getattr(loss_cfg, "support_calibration_weight", 0.0)),
        )
        self.support_calibration_warmup_epochs = max(
            0,
            int(getattr(loss_cfg, "support_calibration_warmup_epochs", 0)),
        )
        self.support_calibration_threshold = float(
            getattr(loss_cfg, "support_calibration_threshold", 0.0)
        )
        # Optional regularizers around the active NB + PCC objective.
        self.gamma_reg_weight = float(getattr(loss_cfg, "gamma_reg_weight", 0.0))
        self.gamma_reg_warmup_epochs = int(
            getattr(loss_cfg, "gamma_reg_warmup_epochs", 0)
        )
        self.additive_bias_l1_weight = float(
            getattr(
                loss_cfg,
                "additive_bias_l1_weight",
                getattr(loss_cfg, "additive_bias_weight", 0.0),
            )
        )
        # Predicted-value floor for PCC only: predictions below this count are
        # treated as 0 ("undetected") when computing correlation. Honest (uses
        # only the prediction, never the target); 0.0 disables.
        self.pcc_prediction_floor = float(
            getattr(loss_cfg, "pcc_prediction_floor", 0.0)
        )
        model_cfg = getattr(self.config, "model", None)
        gamma_center_cfg = (
            getattr(model_cfg, "gamma_centering", None)
            if model_cfg is not None
            else None
        )

        def gamma_center_cfg_get(name: str, default: Any) -> Any:
            if gamma_center_cfg is None:
                return default
            return getattr(gamma_center_cfg, name, default)

        self.gamma_reference_anchor_weight = float(
            gamma_center_cfg_get(
                "reference_anchor_weight",
                getattr(loss_cfg, "gamma_reference_anchor_weight", 0.0),
            )
        )
        self.gamma_reference_anchor_warmup_epochs = int(
            gamma_center_cfg_get(
                "reference_anchor_warmup_epochs",
                getattr(loss_cfg, "gamma_reference_anchor_warmup_epochs", 0),
            )
        )
        self.additive_reference_anchor_weight = float(
            getattr(
                loss_cfg,
                "additive_reference_anchor_weight",
                getattr(model_cfg, "additive_reference_anchor_weight", 0.0)
                if model_cfg is not None
                else 0.0,
            )
        )
        self.additive_reference_anchor_warmup_epochs = int(
            getattr(
                loss_cfg,
                "additive_reference_anchor_warmup_epochs",
                getattr(model_cfg, "additive_reference_anchor_warmup_epochs", 0)
                if model_cfg is not None
                else 0,
            )
        )

        grad_cfg = getattr(self.config, "gradient_conflict_diagnostics", None)

        def grad_cfg_get(name: str, default: Any) -> Any:
            if grad_cfg is None:
                return default
            return getattr(grad_cfg, name, default)

        self.grad_conflict_enabled = bool(grad_cfg_get("enabled", False))
        self.grad_conflict_every_n_train_steps = int(
            grad_cfg_get("every_n_train_steps", 200)
        )
        self.grad_conflict_max_batches_per_epoch = int(
            grad_cfg_get("max_batches_per_epoch", 5)
        )
        self.grad_conflict_parameter_groups = list(
            grad_cfg_get("parameter_groups", ["biological", "rest", "all"])
        )
        self.grad_conflict_include_pcc = bool(
            grad_cfg_get("include_pcc_in_dataset_loss", True)
        )
        self.grad_conflict_include_nll = bool(
            grad_cfg_get("include_nll_in_dataset_loss", True)
        )
        self.grad_conflict_component_modes = list(
            grad_cfg_get("component_modes", ["full"])
        )
        self.grad_conflict_log_pairwise_matrix = bool(
            grad_cfg_get("log_pairwise_matrix", True)
        )
        self.grad_conflict_log_per_dataset_norms = bool(
            grad_cfg_get("log_per_dataset_norms", True)
        )
        self.grad_conflict_eps = float(grad_cfg_get("eps", 1.0e-12))
        self._grad_conflict_batches_this_epoch = 0
        self._grad_conflict_warned_empty_groups: set[str] = set()

        cagrad_cfg = getattr(self.config, "cagrad", None)

        def cagrad_cfg_get(name: str, default: Any) -> Any:
            if cagrad_cfg is None:
                return default
            return getattr(cagrad_cfg, name, default)

        self.cagrad_enabled = bool(cagrad_cfg_get("enabled", False))
        self.cagrad_apply_to = str(cagrad_cfg_get("apply_to", "biological"))
        self.cagrad_c = float(cagrad_cfg_get("c", 0.5))
        self.cagrad_weight_lr = float(cagrad_cfg_get("weight_lr", 0.25))
        self.cagrad_num_weight_steps = int(cagrad_cfg_get("num_weight_steps", 25))
        self.cagrad_rescale = str(cagrad_cfg_get("rescale", "c")).lower()
        self.cagrad_eps = float(cagrad_cfg_get("eps", 1.0e-12))
        self.cagrad_log_diagnostics = bool(cagrad_cfg_get("log_diagnostics", True))
        self.automatic_optimization = True
        if self.cagrad_enabled:
            self._register_cagrad_gradient_hooks()

        residual_cfg = getattr(self.config, "residual_diagnostics", None)

        def residual_cfg_get(name: str, default: Any) -> Any:
            if residual_cfg is None:
                return default
            return getattr(residual_cfg, name, default)

        self.residual_diag_enabled = bool(residual_cfg_get("enabled", True))
        self.residual_diag_run_on_validation = bool(
            residual_cfg_get("run_on_validation", True)
        )
        self.residual_diag_run_on_train = bool(residual_cfg_get("run_on_train", False))
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
        )

    def _build_loss(self) -> nn.Module:
        loss_cfg = self.config.loss

        def loss_cfg_get(name: str, default: Any = None) -> Any:
            try:
                return getattr(loss_cfg, name)
            except (AttributeError, KeyError):
                return default

        likelihood = str(loss_cfg.profile_likelihood).lower()
        poisson_likelihoods = {"poisson", "poisson_pseudo"}
        nb_likelihoods = {"negative_binomial", "nb", "nb_pseudo"}
        valid_likelihoods = poisson_likelihoods | nb_likelihoods

        sequence_reduction = str(loss_cfg_get("nb_sequence_reduction", "mean"))
        length_temper_gamma = float(loss_cfg_get("nb_length_temper_gamma", 0.85))
        length_temper_ref = float(loss_cfg_get("nb_length_temper_ref", 1000.0))
        length_temper_min_weight = float(
            loss_cfg_get("nb_length_temper_min_weight", 0.5)
        )
        length_temper_max_weight = float(
            loss_cfg_get("nb_length_temper_max_weight", 2.0)
        )

        if likelihood in poisson_likelihoods:
            return PoissonProfileLoss(
                eps=loss_cfg.eps,
                mu_min=loss_cfg.mu_min,
                mu_max=loss_cfg.mu_max,
                sequence_reduction=sequence_reduction,
                length_temper_gamma=length_temper_gamma,
                length_temper_ref=length_temper_ref,
                length_temper_min_weight=length_temper_min_weight,
                length_temper_max_weight=length_temper_max_weight,
            )

        return NegativeBinomialProfileLoss(
            eps=loss_cfg.eps,
            mu_min=loss_cfg.mu_min,
            mu_max=loss_cfg.mu_max,
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
        if len(batch) == 11:
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
                replica_profiles,
                replica_mask,
            ) = batch
        elif len(batch) == 9:
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
            ) = batch
            replica_profiles = None
            replica_mask = None
        else:
            raise ValueError(
                f"Expected a 9-tuple batch or 11-tuple replica batch, got {len(batch)}."
            )

        mu, log_sigma, extras = self.model(
            x_packed=seq_packed,
            codon_ids=codon_ids,
            id_datasets=dataset_ids,
            mask=mask,
            target=target,
            current_epoch=int(self.current_epoch),
            sample_ids=ids,
            replica_profiles=replica_profiles,
            replica_mask=replica_mask,
        )

        out = {
            "dataset_ids": dataset_ids,
            "ids": ids,
            "lengths": lengths,
            "mask": mask.bool(),
            "target": target,
            "codon_ids": codon_ids,
            "css": css,
            "sample_weights": sample_weights,
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
    ) -> torch.Tensor:
        """
        Combine a per-sample quantity into a scalar.

        If sample_weights is provided, uses a weighted mean within each dataset
        (or globally when dataset_balanced_loss is False).

        With dataset_balanced_loss, each dataset's (weighted) mean is given
        EQUAL weight regardless of how many samples it contributes.
        """
        values = values.reshape(-1)

        w: torch.Tensor | None = None
        if sample_weights is not None:
            w = sample_weights.reshape(-1).to(device=values.device, dtype=values.dtype).clamp_min(0.0)

        if dataset_balanced is None:
            dataset_balanced = self.dataset_balanced_loss

        if not dataset_balanced:
            if w is not None:
                return (values * w).sum() / w.sum().clamp_min(1.0e-8)
            return values.mean()

        ids = dataset_ids.to(device=values.device)
        unique_ids, inverse = torch.unique(ids, return_inverse=True)
        # unique_ids is sorted/deduped, so its element count is the number of
        # dataset buckets. Reading .numel() avoids the per-call .item() GPU sync
        # (this helper runs ~12x per step).
        K = unique_ids.numel()

        if w is not None:
            w_sums = torch.zeros(K, device=values.device, dtype=values.dtype).scatter_add_(0, inverse, w)
            wv_sums = torch.zeros(K, device=values.device, dtype=values.dtype).scatter_add_(0, inverse, values * w)
            per_ds_mean = wv_sums / w_sums.clamp_min(1.0e-8)
        else:
            sums = torch.zeros(K, device=values.device, dtype=values.dtype).scatter_add_(0, inverse, values)
            counts = torch.bincount(inverse, minlength=K).to(dtype=values.dtype)
            per_ds_mean = sums / counts.clamp_min(1.0)

        return per_ds_mean.mean()

    def _effective_pcc_loss_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.pcc_loss_enabled:
            return torch.tensor(0.0, device=device, dtype=dtype)

        base_weight = float(self.pcc_loss_weight)
        return self._scheduled_weight(
            base_weight=base_weight,
            warmup_epochs=int(self.pcc_loss_warmup_epochs),
            device=device,
            dtype=dtype,
        )

    def _scheduled_weight(
        self,
        *,
        base_weight: float,
        warmup_epochs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base_weight = float(base_weight)
        if base_weight <= 0.0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        warmup_epochs = int(warmup_epochs)
        if warmup_epochs <= 0:
            factor = 1.0
        else:
            epoch = float(getattr(self, "current_epoch", 0))
            factor = min(1.0, max(0.0, epoch / float(warmup_epochs)))
        return torch.tensor(base_weight * factor, device=device, dtype=dtype)

    def _effective_gamma_reg_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._scheduled_weight(
            base_weight=self.gamma_reg_weight,
            warmup_epochs=self.gamma_reg_warmup_epochs,
            device=device,
            dtype=dtype,
        )

    def _effective_zero_calibration_weight(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._scheduled_weight(
            base_weight=self.zero_calibration_weight,
            warmup_epochs=self.zero_calibration_warmup_epochs,
            device=device,
            dtype=dtype,
        )

    def _effective_support_calibration_weight(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._scheduled_weight(
            base_weight=self.support_calibration_weight,
            warmup_epochs=self.support_calibration_warmup_epochs,
            device=device,
            dtype=dtype,
        )

    def _effective_gamma_reference_anchor_weight(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._scheduled_weight(
            base_weight=self.gamma_reference_anchor_weight,
            warmup_epochs=self.gamma_reference_anchor_warmup_epochs,
            device=device,
            dtype=dtype,
        )

    def _effective_additive_reference_anchor_weight(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._scheduled_weight(
            base_weight=self.additive_reference_anchor_weight,
            warmup_epochs=self.additive_reference_anchor_warmup_epochs,
            device=device,
            dtype=dtype,
        )

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
        hybrid_modes = {
            "hybrid_raw_nb_vst": "nb_vst",
            "hybrid_raw_nb_vst_weighted": "nb_vst_weighted",
            "hybrid_raw_nb_vst_weighted_mean_ratio_gated": "nb_vst_weighted_mean_ratio_gated",
        }

        def masked_position_mean(v: torch.Tensor) -> torch.Tensor:
            mask_f = mask.to(dtype=v.dtype)
            return (v * mask_f).sum() / mask_f.sum().clamp_min(1.0)

        def valid_sample_mean(v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            valid_f = valid.to(dtype=v.dtype)
            return (v * valid_f).sum() / valid_f.sum().clamp_min(1.0)

        def mean_ratio_gate() -> torch.Tensor:
            mask_f = mask.to(dtype=mu.dtype)
            valid_len = mask_f.sum(dim=1).clamp_min(1.0)
            mu_mean = (mu * mask_f).sum(dim=1) / valid_len
            target_mean = (target * mask_f).sum(dim=1) / valid_len
            mean_ratio = mu_mean / target_mean.clamp_min(self.eps)
            tau = max(float(self.pcc_mean_ratio_gate_tau), self.eps)
            gate_raw = torch.exp(
                -torch.abs(torch.log(mean_ratio.clamp_min(self.eps))) / tau
            )
            gate_raw = torch.nan_to_num(gate_raw, nan=0.0, posinf=0.0, neginf=0.0)
            floor = min(max(float(self.pcc_mean_ratio_gate_floor), 0.0), 1.0)
            gate = floor + (1.0 - floor) * gate_raw
            if self.pcc_detach_mean_ratio_gate:
                gate = gate.detach()
            return gate

        def component(component_mode: str) -> dict[str, torch.Tensor]:
            alpha_diag = torch.ones_like(mu)
            reliability = torch.ones_like(mu)
            weights: torch.Tensor | None = None
            gate = torch.ones(mu.shape[0], device=mu.device, dtype=mu.dtype)

            if component_mode == "raw":
                x_pcc = mu
                y_pcc = target
            elif component_mode == "log1p":
                x_pcc = torch.log1p(mu.clamp_min(0.0))
                y_pcc = torch.log1p(target.clamp_min(0.0))
            elif component_mode in {
                "nb_vst",
                "nb_vst_weighted",
                "nb_vst_weighted_mean_ratio_gated",
            }:
                log_alpha_f = log_alpha.float()
                alpha_diag = self._pcc_alpha(log_alpha_f, mu.shape).to(
                    device=mu.device,
                    dtype=mu.dtype,
                )
                x_pcc = self._nb_vst(mu, log_alpha_f)
                y_pcc = self._nb_vst(target, log_alpha_f)

                if component_mode in {"nb_vst_weighted", "nb_vst_weighted_mean_ratio_gated"}:
                    reliability = 1.0 / (1.0 + alpha_diag * mu.detach().clamp_min(0.0))
                    reliability = reliability.clamp(
                        min=self.pcc_reliability_min,
                        max=self.pcc_reliability_max,
                    )
                    weights = reliability
                    if self.pcc_detach_reliability:
                        weights = weights.detach()

                if component_mode == "nb_vst_weighted_mean_ratio_gated":
                    gate = mean_ratio_gate()
            else:
                raise ValueError(f"Unsupported pcc_loss_mode: {component_mode!r}.")

            pcc_out = masked_weighted_pcc(
                x=x_pcc,
                y=y_pcc,
                mask=mask,
                weights=weights,
                min_target_var=self.min_pcc_target_var,
                eps=self.eps,
            )
            valid = pcc_out["valid"]
            loss_per_sample = gate * (1.0 - pcc_out["pcc_per_sample"])
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
                "reliability_mean": masked_position_mean(reliability),
                "reliability_min": reliability[mask].amin() if bool(mask.any()) else mu.new_tensor(1.0),
                "reliability_max": reliability[mask].amax() if bool(mask.any()) else mu.new_tensor(1.0),
                "mean_ratio_gate_mean": gate.mean() if gate.numel() > 0 else mu.new_tensor(1.0),
                "mean_ratio_gate_min": gate.amin() if gate.numel() > 0 else mu.new_tensor(1.0),
                "mean_ratio_gate_max": gate.amax() if gate.numel() > 0 else mu.new_tensor(1.0),
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
                if mode in {"nb_vst", "nb_vst_weighted", "nb_vst_weighted_mean_ratio_gated"}
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
            "reliability_mean": active_component["reliability_mean"],
            "reliability_min": active_component["reliability_min"],
            "reliability_max": active_component["reliability_max"],
            "mean_ratio_gate_mean": active_component["mean_ratio_gate_mean"],
            "mean_ratio_gate_min": active_component["mean_ratio_gate_min"],
            "mean_ratio_gate_max": active_component["mean_ratio_gate_max"],
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

    def _gated_additive_regularization_per_sample(
        self,
        out: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = out["target"]
        zeros = torch.zeros(
            target.shape[0],
            device=target.device,
            dtype=target.dtype,
        )
        extras = out["extras"]
        mask = out["mask"].bool()
        mask_f = mask.to(dtype=torch.float32)
        valid_len = mask_f.sum(dim=1).clamp_min(1.0)

        log_gamma = extras.get("log_gamma")
        if torch.is_tensor(log_gamma):
            log_gamma = log_gamma.float()
            gamma_mask = mask & torch.isfinite(log_gamma)
            gamma_mask_f = gamma_mask.to(dtype=log_gamma.dtype)
            gamma_len = gamma_mask_f.sum(dim=1).clamp_min(1.0)
            gamma_reg = (log_gamma.pow(2) * gamma_mask_f).sum(dim=1) / gamma_len
        else:
            gamma_reg = zeros

        additive_bias = extras.get("additive_bias")
        if torch.is_tensor(additive_bias):
            additive_bias = additive_bias.float()
            additive_mask = mask & torch.isfinite(additive_bias)
            additive_mask_f = additive_mask.to(dtype=additive_bias.dtype)
            additive_len = additive_mask_f.sum(dim=1).clamp_min(1.0)
            additive_l1 = (
                additive_bias.clamp_min(0.0) * additive_mask_f
            ).sum(dim=1) / additive_len
        else:
            additive_l1 = zeros

        return gamma_reg, additive_l1

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

    def _reference_anchor_regularization_per_sample(
        self,
        out: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = out["target"]
        zeros = torch.zeros(
            target.shape[0],
            device=target.device,
            dtype=target.dtype,
        )
        extras = out["extras"]
        anchor_mask = extras.get("gamma_reference_anchor_mask")
        if not torch.is_tensor(anchor_mask):
            return zeros, zeros

        mask = out["mask"].bool() & anchor_mask.to(device=target.device).bool()
        mask_f = mask.to(dtype=torch.float32)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        has_anchor = mask_f.sum(dim=1) > 0.0

        log_gamma = extras.get("log_gamma")
        if torch.is_tensor(log_gamma):
            values = log_gamma.float().to(device=target.device)
            gamma_anchor = (values.pow(2) * mask_f).sum(dim=1) / denom
            gamma_anchor = torch.where(has_anchor, gamma_anchor, zeros.float())
        else:
            gamma_anchor = zeros.float()

        additive_bias = extras.get("additive_bias")
        if torch.is_tensor(additive_bias):
            values = additive_bias.float().to(device=target.device).clamp_min(0.0)
            additive_anchor = (values * mask_f).sum(dim=1) / denom
            additive_anchor = torch.where(has_anchor, additive_anchor, zeros.float())
        else:
            additive_anchor = zeros.float()

        return gamma_anchor.to(dtype=target.dtype), additive_anchor.to(dtype=target.dtype)

    @staticmethod
    def _average_over_valid_replicas(
        values: torch.Tensor,
        replica_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = values.reshape(replica_mask.shape[0], replica_mask.shape[1])
        valid = replica_mask.bool().to(device=values.device)
        valid_f = valid.to(dtype=values.dtype)
        return (values * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)

    def _zero_calibration_terms(
        self,
        *,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score the coarsened NB event ``target == 0``.

        The score includes both zero and nonzero observations, so lowering every
        predicted mean is not a solution. By default alpha is detached only for
        this auxiliary score; the main NB NLL still learns dispersion normally.
        """
        with torch.amp.autocast(device_type=mu.device.type, enabled=False):
            valid = mask.bool() & torch.isfinite(target)
            target_f = torch.nan_to_num(
                target.float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)
            mu_f = self.loss_fn.positive_mean_from_params(
                mu=mu.float(),
                log_sigma=None,
            ).float()

            if hasattr(self.loss_fn, "log_alpha_min"):
                log_alpha = _broadcast_profile_param(log_sigma.float(), target_f.shape)
                log_alpha = torch.nan_to_num(
                    log_alpha,
                    nan=0.0,
                    posinf=float(self.loss_fn.log_alpha_max),
                    neginf=float(self.loss_fn.log_alpha_min),
                ).clamp(
                    min=float(self.loss_fn.log_alpha_min),
                    max=float(self.loss_fn.log_alpha_max),
                )
                alpha = torch.exp(log_alpha).clamp_min(self.eps)
                if self.zero_calibration_detach_alpha:
                    alpha = alpha.detach()
                # P_NB(Y=0) = (1 + alpha * mu)^(-1 / alpha).
                log_p0 = -torch.log1p(alpha * mu_f) / alpha
            else:
                # Poisson fallback: P(Y=0) = exp(-mu).
                log_p0 = -mu_f

            log_p0 = torch.nan_to_num(
                log_p0,
                nan=-1.0e8,
                posinf=0.0,
                neginf=-1.0e8,
            ).clamp(max=0.0)
            zero_target = target_f <= float(self.zero_calibration_threshold)

            # -expm1(log_p0) evaluates 1 - p0 accurately even when p0 ~= 1.
            log_p_nonzero = torch.log(
                (-torch.expm1(log_p0)).clamp_min(self.eps)
            )
            position_loss = torch.where(
                zero_target,
                -log_p0,
                -log_p_nonzero,
            )
            position_loss = torch.where(valid, position_loss, torch.zeros_like(position_loss))
            loss_per_sample = reduce_sequence_nll(
                position_loss,
                valid,
                str(getattr(self.loss_fn, "sequence_reduction", "mean")),
                gamma=float(getattr(self.loss_fn, "length_temper_gamma", 1.0)),
                length_ref=float(getattr(self.loss_fn, "length_temper_ref", 1000.0)),
                min_weight=float(getattr(self.loss_fn, "length_temper_min_weight", 1.0)),
                max_weight=float(getattr(self.loss_fn, "length_temper_max_weight", 1.0)),
                eps=self.eps,
            ).float()

            p0 = torch.exp(log_p0).clamp(min=0.0, max=1.0)
            valid_f = valid.to(dtype=p0.dtype)
            zero_valid = valid & zero_target
            nonzero_valid = valid & (~zero_target)

            def selected_mean(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
                selected_f = selected.to(dtype=values.dtype)
                return (values * selected_f).sum() / selected_f.sum().clamp_min(1.0)

            brier = selected_mean(
                (p0 - zero_target.to(dtype=p0.dtype)).pow(2),
                valid,
            )
            valid_count = valid_f.sum().clamp_min(1.0)

        return {
            "loss_per_sample": loss_per_sample,
            "p0_mean": (p0 * valid_f).sum() / valid_count,
            "p0_on_zeros": selected_mean(p0, zero_valid),
            "p0_on_nonzeros": selected_mean(p0, nonzero_valid),
            "target_zero_fraction": zero_valid.to(dtype=p0.dtype).sum() / valid_count,
            "brier": brier,
        }

    def _support_calibration_terms(
        self,
        *,
        out: dict[str, Any],
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Balanced consensus supervision for the learned entmax support.

        BCE is applied to the pre-entmax logits, so an incorrectly excluded
        positive position continues receiving a recovery gradient even where
        the sparse transform itself has a zero derivative.
        """
        valid = mask.bool() & torch.isfinite(target)
        # Use the same sequence-centered score passed to entmax.  An arbitrary
        # head intercept is removed by the forward transform and therefore must
        # not make this diagnostic look calibrated when the actual gate is not.
        logits = out["extras"].get("gamma_support_logits")
        if not torch.is_tensor(logits):
            logits = out["extras"].get("gamma_support_logits_raw")
        if not torch.is_tensor(logits):
            logits = torch.zeros_like(target)
        logits = torch.nan_to_num(
            logits.float().to(device=target.device),
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        ).clamp(min=-30.0, max=30.0)
        active_target = target.float() > float(self.support_calibration_threshold)
        position_loss = F.binary_cross_entropy_with_logits(
            logits,
            active_target.to(dtype=logits.dtype),
            reduction="none",
        )
        position_loss = torch.where(
            valid,
            position_loss,
            torch.zeros_like(position_loss),
        )

        active_valid = valid & active_target
        zero_valid = valid & (~active_target)
        active_f = active_valid.to(dtype=logits.dtype)
        zero_f = zero_valid.to(dtype=logits.dtype)
        active_count = active_f.sum(dim=1)
        zero_count = zero_f.sum(dim=1)
        active_loss = (position_loss * active_f).sum(dim=1) / active_count.clamp_min(1.0)
        zero_loss = (position_loss * zero_f).sum(dim=1) / zero_count.clamp_min(1.0)
        has_active = active_count > 0.0
        has_zero = zero_count > 0.0
        class_count = has_active.to(dtype=logits.dtype) + has_zero.to(dtype=logits.dtype)
        loss_per_sample = (
            active_loss * has_active.to(dtype=logits.dtype)
            + zero_loss * has_zero.to(dtype=logits.dtype)
        ) / class_count.clamp_min(1.0)

        active_probability = torch.sigmoid(logits)
        predicted_active = logits >= 0.0
        predicted_zero = valid & (~predicted_active)
        true_zero = zero_valid
        true_positive = active_valid

        def selected_mean(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
            selected_f = selected.to(dtype=values.dtype)
            return (values * selected_f).sum() / selected_f.sum().clamp_min(1.0)

        zero_true_positive = (predicted_zero & true_zero).to(dtype=logits.dtype).sum()
        predicted_zero_count = predicted_zero.to(dtype=logits.dtype).sum()
        true_zero_count = true_zero.to(dtype=logits.dtype).sum()
        false_zero_count = (predicted_zero & true_positive).to(dtype=logits.dtype).sum()
        true_positive_count = true_positive.to(dtype=logits.dtype).sum()
        valid_count = valid.to(dtype=logits.dtype).sum().clamp_min(1.0)

        return {
            "loss_per_sample": loss_per_sample,
            "active_probability_mean": selected_mean(active_probability, valid),
            "active_probability_on_nonzeros": selected_mean(
                active_probability,
                active_valid,
            ),
            "active_probability_on_zeros": selected_mean(
                active_probability,
                zero_valid,
            ),
            "target_zero_fraction": true_zero_count / valid_count,
            "predicted_zero_fraction": predicted_zero_count / valid_count,
            "zero_precision": zero_true_positive / predicted_zero_count.clamp_min(1.0),
            "zero_recall": zero_true_positive / true_zero_count.clamp_min(1.0),
            "false_zero_rate": false_zero_count / true_positive_count.clamp_min(1.0),
        }

    def _compute_consensus_loss_terms(
        self,
        out: dict[str, Any],
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, Any]:
        nll_per_sample = self.loss_fn(
            mu_phys=out["mu"].float(),
            log_sigma=out["log_sigma"].float(),
            y_true=out["target"].float(),
            mask=out["mask"].bool(),
            return_per_sample=True,
        )
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
        zero_calibration = self._zero_calibration_terms(
            mu=out["mu"],
            log_sigma=out["log_sigma"],
            target=out["target"],
            mask=out["mask"],
        )

        return {
            "nll_per_sample": nll_per_sample,
            "raw_mu_pcc_per_sample": raw_mu_pcc_per_sample,
            "pcc_diag": pcc_diag,
            "likelihood_mu_pcc_per_sample": likelihood_mu_pcc_per_sample,
            "log1p_mse_per_sample": log1p_mse_per_sample,
            "zero_calibration": zero_calibration,
        }

    def _compute_replica_loss_terms(
        self,
        out: dict[str, Any],
    ) -> dict[str, Any] | None:
        replica_profiles = out.get("replica_profiles")
        replica_mask = out.get("replica_mask")
        if not (torch.is_tensor(replica_profiles) and torch.is_tensor(replica_mask)):
            return None

        target_reps = replica_profiles.float()
        rep_mask = replica_mask.bool().to(device=target_reps.device)
        B, R, T = target_reps.shape

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
        L_bio = extras["L_bio"].float().to(device=target_reps.device)
        gamma = extras.get("gamma", torch.ones_like(L_bio))
        gamma = gamma.float().to(device=target_reps.device)
        additive_bias = extras.get("additive_bias", torch.zeros_like(L_bio))
        additive_bias = additive_bias.float().to(device=target_reps.device)
        shape = gamma * L_bio + additive_bias
        shape = torch.where(
            base_mask & torch.isfinite(shape),
            shape.clamp_min(0.0),
            torch.zeros_like(shape),
        )
        mu_rep = (scale_rep.unsqueeze(-1) * shape.unsqueeze(1)).clamp(
            min=self.eps,
            max=float(getattr(self.model, "mu_max", 1.0e8)),
        )
        mu_rep = torch.nan_to_num(
            mu_rep,
            nan=self.eps,
            posinf=float(getattr(self.model, "mu_max", 1.0e8)),
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

        nll_flat = self.loss_fn(
            mu_phys=flat_mu,
            log_sigma=flat_log_sigma,
            y_true=flat_target,
            mask=flat_mask,
            return_per_sample=True,
        )
        nll_per_sample = self._average_over_valid_replicas(nll_flat, rep_mask)

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
                pcc_diag[key] = self._average_over_valid_replicas(pcc_flat[key], rep_mask)
        flat_rep_mask = rep_mask.reshape(-1).to(device=target_reps.device)
        if torch.is_tensor(pcc_flat.get("valid")):
            active_valid = pcc_flat["valid"].to(device=target_reps.device) & flat_rep_mask
            sample_valid = active_valid.reshape(B, R).any(dim=1)
            pcc_diag["valid"] = sample_valid
            pcc_diag["valid_fraction"] = sample_valid.to(dtype=target_reps.dtype).mean()

        raw_mu_pcc_flat = self._pearson_per_sample(
            pred=flat_mu,
            target=flat_target,
            mask=flat_mask,
        )
        raw_mu_pcc_per_sample = self._average_over_valid_replicas(
            raw_mu_pcc_flat,
            rep_mask,
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
        zero_calibration_flat = self._zero_calibration_terms(
            mu=flat_mu,
            log_sigma=flat_log_sigma,
            target=flat_target,
            mask=flat_mask,
        )
        zero_calibration = dict(zero_calibration_flat)
        zero_calibration["loss_per_sample"] = self._average_over_valid_replicas(
            zero_calibration_flat["loss_per_sample"],
            rep_mask,
        )

        replica_counts = rep_mask.to(dtype=target_reps.dtype).sum(dim=1)
        return {
            "nll_per_sample": nll_per_sample,
            "raw_mu_pcc_per_sample": raw_mu_pcc_per_sample,
            "pcc_diag": pcc_diag,
            "likelihood_mu_pcc_per_sample": likelihood_mu_pcc_per_sample,
            "log1p_mse_per_sample": log1p_mse_per_sample,
            "zero_calibration": zero_calibration,
            "metrics": {
                "replica_count_mean": replica_counts.mean(),
                "replica_count_min": replica_counts.amin(),
                "replica_count_max": replica_counts.amax(),
            },
        }

    def _compute_loss_and_metrics(self, out: dict[str, Any]) -> dict[str, torch.Tensor]:
        dataset_ids = out["dataset_ids"]
        sample_weights = out.get("sample_weights")
        mask = out["mask"].bool() & torch.isfinite(out["target"].float())
        target = torch.nan_to_num(out["target"].float(), nan=0.0, posinf=0.0, neginf=0.0)

        consensus_terms = self._compute_consensus_loss_terms(out, target, mask)
        replica_terms = self._compute_replica_loss_terms(out)

        use_replica_as_main = (
            replica_terms is not None
            and self.replica_objective == "replica"
        )
        if use_replica_as_main:
            nll_per_sample = replica_terms["nll_per_sample"]
            raw_mu_pcc_per_sample = replica_terms["raw_mu_pcc_per_sample"]
            pcc_diag = replica_terms["pcc_diag"]
            likelihood_mu_pcc_per_sample = replica_terms["likelihood_mu_pcc_per_sample"]
            log1p_mse_per_sample = replica_terms["log1p_mse_per_sample"]
        else:
            nll_per_sample = consensus_terms["nll_per_sample"]
            raw_mu_pcc_per_sample = consensus_terms["raw_mu_pcc_per_sample"]
            pcc_diag = consensus_terms["pcc_diag"]
            likelihood_mu_pcc_per_sample = consensus_terms[
                "likelihood_mu_pcc_per_sample"
            ]
            log1p_mse_per_sample = consensus_terms["log1p_mse_per_sample"]

        use_replica_zero_calibration = (
            replica_terms is not None
            and (
                self.zero_calibration_source == "replica"
                or (self.zero_calibration_source == "main" and use_replica_as_main)
            )
        )
        zero_calibration_terms = (
            replica_terms["zero_calibration"]
            if use_replica_zero_calibration
            else consensus_terms["zero_calibration"]
        )
        zero_calibration_per_sample = zero_calibration_terms["loss_per_sample"]
        support_calibration_terms = self._support_calibration_terms(
            out=out,
            target=target,
            mask=mask,
        )
        support_calibration_per_sample = support_calibration_terms["loss_per_sample"]

        pcc_loss_per_sample = pcc_diag["loss_per_sample"]
        extras = out["extras"]
        gamma_reg_per_sample, additive_bias_l1_per_sample = (
            self._gated_additive_regularization_per_sample(out)
        )
        (
            gamma_reference_anchor_per_sample,
            additive_reference_anchor_per_sample,
        ) = self._reference_anchor_regularization_per_sample(out)

        effective_pcc_weight = self._effective_pcc_loss_weight(
            device=nll_per_sample.device,
            dtype=nll_per_sample.dtype,
        )
        effective_gamma_reg_weight = self._effective_gamma_reg_weight(
            device=nll_per_sample.device,
            dtype=nll_per_sample.dtype,
        )
        effective_zero_calibration_weight = self._effective_zero_calibration_weight(
            device=nll_per_sample.device,
            dtype=nll_per_sample.dtype,
        )
        effective_support_calibration_weight = (
            self._effective_support_calibration_weight(
                device=nll_per_sample.device,
                dtype=nll_per_sample.dtype,
            )
        )
        effective_gamma_reference_anchor_weight = (
            self._effective_gamma_reference_anchor_weight(
                device=nll_per_sample.device,
                dtype=nll_per_sample.dtype,
            )
        )
        effective_additive_reference_anchor_weight = (
            self._effective_additive_reference_anchor_weight(
                device=nll_per_sample.device,
                dtype=nll_per_sample.dtype,
            )
        )
        per_sample_total_loss = (
            nll_per_sample
            + effective_pcc_weight * pcc_loss_per_sample
            + effective_zero_calibration_weight * zero_calibration_per_sample
            + effective_support_calibration_weight
            * support_calibration_per_sample
            + effective_gamma_reg_weight * gamma_reg_per_sample
            + effective_gamma_reference_anchor_weight
            * gamma_reference_anchor_per_sample
            + self.additive_bias_l1_weight * additive_bias_l1_per_sample
            + effective_additive_reference_anchor_weight
            * additive_reference_anchor_per_sample
        )
        replica_aux_per_sample = torch.zeros_like(per_sample_total_loss)
        if (
            replica_terms is not None
            and self.replica_objective == "consensus_plus_replica"
        ):
            replica_aux_per_sample = (
                self.replica_nll_weight * replica_terms["nll_per_sample"]
                + self.replica_pcc_loss_weight
                * replica_terms["pcc_diag"]["loss_per_sample"]
            )
            per_sample_total_loss = per_sample_total_loss + replica_aux_per_sample

        loss_global_unbalanced = self._aggregate_per_sample(
            per_sample_total_loss,
            dataset_ids,
            sample_weights,
            dataset_balanced=False,
        )
        loss_dataset_balanced = self._aggregate_per_sample(
            per_sample_total_loss,
            dataset_ids,
            sample_weights,
            dataset_balanced=True,
        )
        loss = loss_dataset_balanced if self.dataset_balanced_loss else loss_global_unbalanced
        loss_per_sample = per_sample_total_loss

        nll_global = self._aggregate_per_sample(
            nll_per_sample,
            dataset_ids,
            sample_weights,
            dataset_balanced=False,
        )
        pcc_loss_global = self._aggregate_per_sample(
            pcc_loss_per_sample,
            dataset_ids,
            sample_weights,
            dataset_balanced=False,
        )
        nll_dataset_balanced = self._aggregate_per_sample(
            nll_per_sample,
            dataset_ids,
            sample_weights,
            dataset_balanced=True,
        )
        pcc_loss_dataset_balanced = self._aggregate_per_sample(
            pcc_loss_per_sample,
            dataset_ids,
            sample_weights,
            dataset_balanced=True,
        )
        zero_calibration_loss = self._aggregate_per_sample(
            zero_calibration_per_sample,
            dataset_ids,
            sample_weights,
            dataset_balanced=self.dataset_balanced_loss,
        )
        support_calibration_loss = self._aggregate_per_sample(
            support_calibration_per_sample,
            dataset_ids,
            sample_weights,
            dataset_balanced=self.dataset_balanced_loss,
        )
        gamma_reg = self._aggregate_per_sample(
            gamma_reg_per_sample,
            dataset_ids,
            sample_weights,
            dataset_balanced=self.dataset_balanced_loss,
        )
        gamma_reference_anchor_loss = self._aggregate_per_sample(
            gamma_reference_anchor_per_sample,
            dataset_ids,
            sample_weights,
            dataset_balanced=self.dataset_balanced_loss,
        )
        additive_bias_l1_reg = self._aggregate_per_sample(
            additive_bias_l1_per_sample,
            dataset_ids,
            sample_weights,
            dataset_balanced=self.dataset_balanced_loss,
        )
        additive_reference_anchor_loss = self._aggregate_per_sample(
            additive_reference_anchor_per_sample,
            dataset_ids,
            sample_weights,
            dataset_balanced=self.dataset_balanced_loss,
        )
        nll = nll_dataset_balanced if self.dataset_balanced_loss else nll_global
        pcc_loss = (
            pcc_loss_dataset_balanced
            if self.dataset_balanced_loss
            else pcc_loss_global
        )
        pcc_value = self._aggregate_per_sample(
            pcc_diag["pcc_per_sample"],
            dataset_ids,
            sample_weights,
        )
        pcc_raw_loss = self._aggregate_per_sample(
            pcc_diag["raw_loss_per_sample"],
            dataset_ids,
            sample_weights,
        )
        pcc_nb_vst_loss = self._aggregate_per_sample(
            pcc_diag["nb_vst_loss_per_sample"],
            dataset_ids,
            sample_weights,
        )

        def aggregate_active(v: torch.Tensor) -> torch.Tensor:
            return self._aggregate_per_sample(v, dataset_ids, sample_weights)

        consensus_nll = aggregate_active(consensus_terms["nll_per_sample"])
        consensus_pcc_loss = aggregate_active(
            consensus_terms["pcc_diag"]["loss_per_sample"]
        )
        consensus_pcc_value = aggregate_active(
            consensus_terms["pcc_diag"]["pcc_per_sample"]
        )

        zero_metric = target.new_tensor(0.0)
        if replica_terms is not None:
            replica_nll = aggregate_active(replica_terms["nll_per_sample"])
            replica_pcc_loss = aggregate_active(
                replica_terms["pcc_diag"]["loss_per_sample"]
            )
            replica_pcc_value = aggregate_active(
                replica_terms["pcc_diag"]["pcc_per_sample"]
            )
        else:
            replica_nll = zero_metric
            replica_pcc_loss = zero_metric
            replica_pcc_value = zero_metric

        replica_aux_loss = aggregate_active(replica_aux_per_sample)
        replica_objective_mode_id = target.new_tensor(
            {
                "replica": 0.0,
                "consensus": 1.0,
                "consensus_plus_replica": 2.0,
            }[self.replica_objective]
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
        gamma_amplitude_diag = extras.get(
            "gamma_amplitude",
            gamma_diag,
        )
        gamma_sparse_gate_diag = extras.get(
            "gamma_sparse_gate",
            torch.ones_like(gamma_diag),
        )
        gamma_support_logits_diag = extras.get(
            "gamma_support_logits",
            torch.zeros_like(gamma_diag),
        )
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
        gamma_reference_mask_diag = extras.get(
            "gamma_reference_anchor_mask",
            torch.zeros_like(extras["L_bio"], dtype=torch.bool),
        )
        gamma_high_reliability_mask = gamma_reliability_diag >= float(
            getattr(self.model, "gamma_centering_min_reliability", 0.0)
        )
        gamma_low_reliability_mask = ~gamma_high_reliability_mask
        gamma_skipped_mask = mask & (~gamma_applied_diag.to(device=mask.device).bool())
        mean_log_gamma_per_sample = pos_mean(log_gamma_diag)
        std_log_gamma_per_sample = torch.sqrt(
            pos_mean((log_gamma_diag - mean_log_gamma_per_sample.reshape(-1, 1)).pow(2))
        )
        library_depth = (target * mask_f).sum(dim=1)
        target_zero_diag = mask & (
            target <= float(self.support_calibration_threshold)
        )
        target_active_diag = mask & (~target_zero_diag)
        sparse_zero_diag = mask & (gamma_sparse_gate_diag == 0.0)
        sparse_zero_true_positive = (sparse_zero_diag & target_zero_diag).float().sum()
        sparse_zero_count = sparse_zero_diag.float().sum()
        target_zero_count = target_zero_diag.float().sum()
        target_active_count = target_active_diag.float().sum()

        metrics: dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_per_sample": loss_per_sample,
            "loss_global_unbalanced": loss_global_unbalanced,
            "loss_dataset_balanced": loss_dataset_balanced,
            "nll": nll,
            "nll_per_sample": nll_per_sample,
            "nll_global": nll_global,
            "nll_dataset_balanced": nll_dataset_balanced,
            "pcc_loss": pcc_loss,
            "pcc_loss_per_sample": pcc_loss_per_sample,
            "pcc_loss_global": pcc_loss_global,
            "pcc_loss_dataset_balanced": pcc_loss_dataset_balanced,
            "consensus_nll": consensus_nll,
            "consensus_pcc_loss": consensus_pcc_loss,
            "consensus_pcc_value": consensus_pcc_value,
            "replica_objective_mode": replica_objective_mode_id,
            "replica_aux_loss": replica_aux_loss,
            "replica_aux_loss_per_sample": replica_aux_per_sample,
            "replica_nll": replica_nll,
            "replica_pcc_loss": replica_pcc_loss,
            "replica_pcc_value": replica_pcc_value,
            "replica_nll_weight": target.new_tensor(self.replica_nll_weight),
            "replica_pcc_loss_weight": target.new_tensor(
                self.replica_pcc_loss_weight
            ),
            "zero_calibration_loss": zero_calibration_loss,
            "zero_calibration_loss_per_sample": zero_calibration_per_sample,
            "zero_calibration_weight": effective_zero_calibration_weight,
            "zero_calibration_weight_configured": target.new_tensor(
                self.zero_calibration_weight
            ),
            "zero_calibration_uses_replicas": target.new_tensor(
                1.0 if use_replica_zero_calibration else 0.0
            ),
            "zero_calibration_p0_mean": zero_calibration_terms["p0_mean"],
            "zero_calibration_p0_on_zeros": zero_calibration_terms["p0_on_zeros"],
            "zero_calibration_p0_on_nonzeros": zero_calibration_terms[
                "p0_on_nonzeros"
            ],
            "zero_calibration_target_zero_fraction": zero_calibration_terms[
                "target_zero_fraction"
            ],
            "zero_calibration_brier": zero_calibration_terms["brier"],
            "support_calibration_loss": support_calibration_loss,
            "support_calibration_loss_per_sample": support_calibration_per_sample,
            "support_calibration_weight": effective_support_calibration_weight,
            "support_calibration_weight_configured": target.new_tensor(
                self.support_calibration_weight
            ),
            "support_active_probability_mean": support_calibration_terms[
                "active_probability_mean"
            ],
            "support_active_probability_on_nonzeros": support_calibration_terms[
                "active_probability_on_nonzeros"
            ],
            "support_active_probability_on_zeros": support_calibration_terms[
                "active_probability_on_zeros"
            ],
            "support_target_zero_fraction": support_calibration_terms[
                "target_zero_fraction"
            ],
            "support_predicted_zero_fraction": support_calibration_terms[
                "predicted_zero_fraction"
            ],
            "support_zero_precision": support_calibration_terms["zero_precision"],
            "support_zero_recall": support_calibration_terms["zero_recall"],
            "support_false_zero_rate": support_calibration_terms["false_zero_rate"],
            "gamma_reg": gamma_reg,
            "gamma_reg_per_sample": gamma_reg_per_sample,
            "gamma_global_regularizer": gamma_reg,
            "gamma_reg_weight": effective_gamma_reg_weight,
            "gamma_reg_weight_configured": target.new_tensor(self.gamma_reg_weight),
            "additive_bias_l1_reg": additive_bias_l1_reg,
            "additive_bias_l1_reg_per_sample": additive_bias_l1_per_sample,
            "additive_bias_l1_weight": target.new_tensor(
                self.additive_bias_l1_weight
            ),
            "gamma_reference_anchor_loss": gamma_reference_anchor_loss,
            "gamma_reference_anchor_loss_per_sample": gamma_reference_anchor_per_sample,
            "gamma_reference_anchor_weight": effective_gamma_reference_anchor_weight,
            "additive_reference_anchor_loss": additive_reference_anchor_loss,
            "additive_reference_anchor_loss_per_sample": additive_reference_anchor_per_sample,
            "additive_reference_anchor_weight": effective_additive_reference_anchor_weight,
            "pcc_loss_total": pcc_loss,
            "pcc_value": pcc_value,
            "pcc_loss_weight_effective": effective_pcc_weight,
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
            "pcc_hybrid_raw_contribution": (
                effective_pcc_weight
                * pcc_diag["raw_component_weight"]
                * pcc_raw_loss
            ),
            "pcc_hybrid_nb_vst_contribution": (
                effective_pcc_weight
                * pcc_diag["nb_vst_component_weight"]
                * pcc_nb_vst_loss
            ),
            "pcc_target_var_mean": pcc_diag["target_var_mean"],
            "pcc_alpha_mean": pcc_diag["alpha_mean"],
            "pcc_alpha_min": pcc_diag["alpha_min"],
            "pcc_alpha_max": pcc_diag["alpha_max"],
            "pcc_reliability_mean": pcc_diag["reliability_mean"],
            "pcc_reliability_min": pcc_diag["reliability_min"],
            "pcc_reliability_max": pcc_diag["reliability_max"],
            "pcc_mean_ratio_gate_mean": pcc_diag["mean_ratio_gate_mean"],
            "pcc_mean_ratio_gate_min": pcc_diag["mean_ratio_gate_min"],
            "pcc_mean_ratio_gate_max": pcc_diag["mean_ratio_gate_max"],
            "mu_pcc": self._aggregate_per_sample(raw_mu_pcc_per_sample, dataset_ids, sample_weights),
            "mu_pcc_per_sample": raw_mu_pcc_per_sample,
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
            "gamma_exact_zero_fraction": masked_fraction(gamma_diag == 0.0),
            "gamma_amplitude_mean": global_pos_mean(gamma_amplitude_diag),
            "gamma_amplitude_max": global_pos_max(gamma_amplitude_diag),
            "gamma_sparse_gate_mean": global_pos_mean(gamma_sparse_gate_diag),
            "gamma_sparse_gate_max": global_pos_max(gamma_sparse_gate_diag),
            "gamma_sparse_gate_zero_fraction": masked_fraction(
                gamma_sparse_gate_diag == 0.0
            ),
            "gamma_sparse_zero_precision": (
                sparse_zero_true_positive / sparse_zero_count.clamp_min(1.0)
            ),
            "gamma_sparse_zero_recall": (
                sparse_zero_true_positive / target_zero_count.clamp_min(1.0)
            ),
            "gamma_sparse_false_zero_rate": (
                (sparse_zero_diag & target_active_diag).float().sum()
                / target_active_count.clamp_min(1.0)
            ),
            "gamma_support_logit_mean": global_pos_mean(gamma_support_logits_diag),
            "gamma_support_logit_std": global_pos_std(gamma_support_logits_diag),
            "gamma_split_support_head_enabled": extras.get(
                "gamma_split_support_head_enabled",
                target.new_tensor(0.0),
            ).float().mean(),
            "gamma_gate_additive_bias_enabled": extras.get(
                "gamma_gate_additive_bias_enabled",
                target.new_tensor(0.0),
            ).float().mean(),
            "gamma_sparse_transform_enabled": extras.get(
                "gamma_sparse_transform_enabled",
                torch.zeros_like(extras["scale_dt"]),
            ).float().mean(),
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
            "gamma_reference_log_abs_mean": masked_abs_mean(
                log_gamma_diag,
                gamma_reference_mask_diag,
            ),
            "gamma_nonreference_log_abs_mean": masked_abs_mean(
                log_gamma_diag,
                ~gamma_reference_mask_diag.to(device=mask.device).bool(),
            ),
            "gamma_high_reliability_log_abs_mean": masked_abs_mean(
                log_gamma_diag,
                gamma_high_reliability_mask,
            ),
            "gamma_low_reliability_log_abs_mean": masked_abs_mean(
                log_gamma_diag,
                gamma_low_reliability_mask,
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
            "additive_bias_mean": global_pos_mean(
                extras.get("additive_bias", torch.zeros_like(extras["L_bio"]))
            ),
            "additive_bias_max": global_pos_max(
                extras.get("additive_bias", torch.zeros_like(extras["L_bio"]))
            ),
            "nb_alpha_mean": pos_mean(extras["alpha"]).mean(),
        }
        if replica_terms is not None:
            metrics.update(replica_terms["metrics"])
        return metrics

    def get_gradient_diagnostic_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        biological_params = [
            p
            for name, p in self.model.named_parameters()
            if p.requires_grad and self._is_biological_parameter_name(name)
        ]
        biological_param_ids = {id(p) for p in biological_params}
        rest_params = [
            p
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in biological_param_ids
        ]
        return {
            "biological": biological_params,
            "rest": rest_params,
            "all": biological_params + rest_params,
        }

    def _gradient_diagnostic_per_sample_components(
        self,
        metrics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        zeros = torch.zeros_like(metrics["nll_per_sample"])
        pcc_component = (
            metrics["pcc_loss_weight_effective"] * metrics["pcc_loss_per_sample"]
            if self.pcc_loss_enabled
            else zeros
        )
        full = zeros
        if self.grad_conflict_include_nll:
            full = full + metrics["nll_per_sample"]
        if self.grad_conflict_include_pcc:
            full = full + pcc_component
        zero_calibration_component = (
            metrics["zero_calibration_weight"]
            * metrics["zero_calibration_loss_per_sample"]
        )
        full = full + zero_calibration_component
        support_calibration_component = (
            metrics["support_calibration_weight"]
            * metrics["support_calibration_loss_per_sample"]
        )
        full = full + support_calibration_component
        full = full + metrics["gamma_reg_weight"] * metrics["gamma_reg_per_sample"]
        full = (
            full
            + metrics["gamma_reference_anchor_weight"]
            * metrics["gamma_reference_anchor_loss_per_sample"]
        )
        full = (
            full
            + self.additive_bias_l1_weight
            * metrics["additive_bias_l1_reg_per_sample"]
        )
        full = (
            full
            + metrics["additive_reference_anchor_weight"]
            * metrics["additive_reference_anchor_loss_per_sample"]
        )

        return {
            "full": full,
            "nll": metrics["nll_per_sample"],
            "pcc": pcc_component,
            "zero_calibration": zero_calibration_component,
            "support_calibration": support_calibration_component,
            "gamma_reg": metrics["gamma_reg_weight"] * metrics["gamma_reg_per_sample"],
            "gamma_reference_anchor": metrics["gamma_reference_anchor_weight"]
            * metrics["gamma_reference_anchor_loss_per_sample"],
            "additive_bias_l1_reg": self.additive_bias_l1_weight
            * metrics["additive_bias_l1_reg_per_sample"],
            "additive_reference_anchor": metrics["additive_reference_anchor_weight"]
            * metrics["additive_reference_anchor_loss_per_sample"],
        }

    @staticmethod
    def _build_dataset_losses(
        per_sample_loss: torch.Tensor,
        dataset_ids: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        dataset_ids = dataset_ids.reshape(-1).to(device=per_sample_loss.device)
        per_sample_loss = per_sample_loss.reshape(-1)
        dataset_losses: dict[int, torch.Tensor] = {}
        for dataset_id_tensor in torch.unique(dataset_ids):
            sample_mask = dataset_ids == dataset_id_tensor
            if bool(sample_mask.any()):
                dataset_losses[int(dataset_id_tensor.detach().cpu().item())] = (
                    per_sample_loss[sample_mask].mean()
                )
        return dataset_losses

    def compute_dataset_gradient_conflict_for_group(
        self,
        dataset_losses: dict[int, torch.Tensor],
        params: list[nn.Parameter],
        group_name: str,
        eps: float = 1.0e-12,
    ) -> dict[str, torch.Tensor]:
        params = [p for p in params if p.requires_grad]
        if not params:
            return summarize_gradient_conflict_from_flat_grads({}, eps=eps)

        flat_grads: dict[int, torch.Tensor] = {}
        for dataset_id, loss_d in dataset_losses.items():
            if not loss_d.requires_grad:
                flat_grads[int(dataset_id)] = torch.cat(
                    [torch.zeros_like(p).reshape(-1) for p in params]
                ).float()
                continue
            grad_tensors = torch.autograd.grad(
                loss_d,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            flat_parts = []
            for grad, param in zip(grad_tensors, params, strict=True):
                if grad is None:
                    flat_parts.append(torch.zeros_like(param).reshape(-1))
                else:
                    flat_parts.append(grad.detach().reshape(-1))
            flat_grads[int(dataset_id)] = torch.cat(flat_parts).float()

        del group_name
        return summarize_gradient_conflict_from_flat_grads(flat_grads, eps=eps)

    def _register_cagrad_gradient_hooks(self) -> None:
        if self._cagrad_hook_handles:
            return

        for name, param in self.model.named_parameters():
            if not param.requires_grad or not self._is_biological_parameter_name(name):
                continue

            def make_hook(p: nn.Parameter):
                def hook(grad: torch.Tensor) -> torch.Tensor:
                    override = self._cagrad_bio_grad_overrides.get(id(p))
                    if override is None:
                        return grad
                    return override.to(device=grad.device, dtype=grad.dtype)

                return hook

            self._cagrad_hook_handles.append(param.register_hook(make_hook(param)))

    @staticmethod
    def _flatten_grads_for_params(
        grad_tensors: tuple[torch.Tensor | None, ...],
        params: list[nn.Parameter],
    ) -> torch.Tensor:
        flat_parts = []
        for grad, param in zip(grad_tensors, params, strict=True):
            if grad is None:
                flat_parts.append(torch.zeros_like(param).reshape(-1))
            else:
                flat_parts.append(grad.reshape(-1))
        if not flat_parts:
            return torch.empty(0)
        return torch.cat(flat_parts)

    @staticmethod
    def _unflatten_like_params(
        flat_grad: torch.Tensor,
        params: list[nn.Parameter],
    ) -> list[torch.Tensor]:
        out = []
        offset = 0
        for param in params:
            n = param.numel()
            out.append(flat_grad[offset : offset + n].reshape_as(param))
            offset += n
        return out

    @staticmethod
    def _softmax_cagrad_weights(
        grads: torch.Tensor,
        *,
        c: float,
        num_steps: int,
        weight_lr: float,
        eps: float,
    ) -> torch.Tensor:
        task_count = grads.shape[0]
        if task_count <= 1 or float(c) <= 0.0 or int(num_steps) <= 0:
            return torch.full(
                (task_count,),
                1.0 / max(task_count, 1),
                device=grads.device,
                dtype=grads.dtype,
            )

        grads = grads.detach()
        g0 = grads.mean(dim=0)
        g0_norm = torch.linalg.norm(g0).clamp_min(float(eps))
        c_scaled = float(c) * g0_norm
        logits = torch.zeros(task_count, device=grads.device, dtype=grads.dtype)

        for _ in range(int(num_steps)):
            logits = logits.detach().requires_grad_(True)
            weights = torch.softmax(logits, dim=0)
            gw = weights @ grads
            objective = torch.dot(gw, g0) + c_scaled * torch.linalg.norm(gw).clamp_min(float(eps))
            grad_logits = torch.autograd.grad(objective, logits, retain_graph=False)[0]
            step_scale = grad_logits.norm().clamp_min(1.0)
            logits = logits - float(weight_lr) * grad_logits / step_scale

        return torch.softmax(logits.detach(), dim=0)

    @classmethod
    def _combine_cagrad_flat_grads(
        cls,
        flat_grads: dict[int, torch.Tensor],
        *,
        c: float = 0.5,
        num_steps: int = 25,
        weight_lr: float = 0.25,
        rescale: str = "c",
        eps: float = 1.0e-12,
    ) -> dict[str, torch.Tensor]:
        if not flat_grads:
            z = torch.tensor(0.0)
            return {
                "combined_grad": torch.empty(0),
                "weights": torch.empty(0),
                "dataset_ids": torch.empty(0, dtype=torch.long),
                "mean_grad_norm": z,
                "combined_grad_norm": z,
            }

        dataset_ids = sorted(int(k) for k in flat_grads)
        grads = torch.stack([flat_grads[k].detach().float().reshape(-1) for k in dataset_ids])
        mean_grad = grads.mean(dim=0)
        weights = cls._softmax_cagrad_weights(
            grads,
            c=float(c),
            num_steps=int(num_steps),
            weight_lr=float(weight_lr),
            eps=float(eps),
        )

        if grads.shape[0] <= 1 or float(c) <= 0.0:
            combined = mean_grad
        else:
            gw = weights @ grads
            g0_norm = torch.linalg.norm(mean_grad).clamp_min(float(eps))
            gw_norm = torch.linalg.norm(gw).clamp_min(float(eps))
            combined = mean_grad + (float(c) * g0_norm / gw_norm) * gw

            if rescale == "c":
                combined = combined / (1.0 + float(c))
            elif rescale in {"c2", "c_squared", "cagrad"}:
                combined = combined / (1.0 + float(c) * float(c))
            elif rescale in {"none", "off", "false"}:
                pass
            else:
                raise ValueError(
                    "cagrad.rescale must be one of {'c', 'c_squared', 'none'}, "
                    f"got {rescale!r}."
                )

        return {
            "combined_grad": combined,
            "weights": weights,
            "dataset_ids": torch.tensor(dataset_ids, device=grads.device, dtype=torch.long),
            "mean_grad_norm": torch.linalg.norm(mean_grad),
            "combined_grad_norm": torch.linalg.norm(combined),
        }

    def _prepare_biological_cagrad(
        self,
        *,
        out: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        self._cagrad_bio_grad_overrides = {}
        if not self.cagrad_enabled:
            return {}

        params = self.get_gradient_diagnostic_parameter_groups().get("biological", [])
        params = [p for p in params if p.requires_grad]
        if not params:
            return {}

        dataset_losses = self._build_dataset_losses(
            metrics["loss_per_sample"],
            out["dataset_ids"],
        )
        if len(dataset_losses) <= 1:
            return {
                "cagrad/biological/num_datasets_present": torch.tensor(
                    float(len(dataset_losses)),
                    device=metrics["loss"].device,
                )
            }

        flat_grads: dict[int, torch.Tensor] = {}
        for dataset_id, loss_d in dataset_losses.items():
            grad_tensors = torch.autograd.grad(
                loss_d,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            flat_grads[int(dataset_id)] = self._flatten_grads_for_params(grad_tensors, params).detach()

        combined = self._combine_cagrad_flat_grads(
            flat_grads,
            c=self.cagrad_c,
            num_steps=self.cagrad_num_weight_steps,
            weight_lr=self.cagrad_weight_lr,
            rescale=self.cagrad_rescale,
            eps=self.cagrad_eps,
        )
        combined_parts = self._unflatten_like_params(combined["combined_grad"], params)
        self._cagrad_bio_grad_overrides = {
            id(param): grad.to(device=param.device, dtype=param.dtype)
            for param, grad in zip(params, combined_parts, strict=True)
        }

        if not self.cagrad_log_diagnostics:
            return {}

        conflict = summarize_gradient_conflict_from_flat_grads(flat_grads, eps=self.cagrad_eps)
        logs: dict[str, torch.Tensor] = {
            "cagrad/biological/enabled": metrics["loss"].new_tensor(1.0),
            "cagrad/biological/num_datasets_present": conflict["num_datasets_present"].detach(),
            "cagrad/biological/pre_mean_cosine": conflict["mean_cosine"].detach(),
            "cagrad/biological/pre_conflict_score": conflict["conflict_score"].detach(),
            "cagrad/biological/pre_negative_cosine_fraction": conflict["negative_cosine_fraction"].detach(),
            "cagrad/biological/pre_norm_ratio_max_min": conflict["norm_ratio_max_min"].detach(),
            "cagrad/biological/mean_grad_norm": combined["mean_grad_norm"].detach(),
            "cagrad/biological/combined_grad_norm": combined["combined_grad_norm"].detach(),
        }
        for idx, dataset_id in enumerate(combined["dataset_ids"].detach().cpu().tolist()):
            dataset_name = self._dataset_name(int(dataset_id))
            logs[f"cagrad/biological/weight/{dataset_name}"] = combined["weights"][idx].detach()
            logs[f"cagrad/biological/loss/{dataset_name}"] = dataset_losses[int(dataset_id)].detach()
        return logs

    def _should_run_gradient_conflict_diagnostics(self) -> bool:
        if not self.grad_conflict_enabled:
            return False
        if self.training is False:
            return False
        if self.grad_conflict_max_batches_per_epoch >= 0:
            if self._grad_conflict_batches_this_epoch >= self.grad_conflict_max_batches_per_epoch:
                return False
        every_n = max(int(self.grad_conflict_every_n_train_steps), 1)
        global_step = int(getattr(self, "global_step", 0))
        return global_step % every_n == 0

    def _log_gradient_conflict_diagnostics(
        self,
        *,
        out: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        if not self._should_run_gradient_conflict_diagnostics():
            return
        if hasattr(self, "trainer") and self.trainer is not None:
            if not getattr(self.trainer, "is_global_zero", True):
                return

        components = self._gradient_diagnostic_per_sample_components(metrics)
        parameter_groups = self.get_gradient_diagnostic_parameter_groups()
        logs: dict[str, torch.Tensor] = {}
        dataset_ids = out["dataset_ids"]

        for component_name in self.grad_conflict_component_modes:
            if component_name not in components:
                continue
            dataset_losses = self._build_dataset_losses(
                components[component_name],
                dataset_ids,
            )
            if not dataset_losses:
                continue

            for dataset_id, loss_d in dataset_losses.items():
                dataset_name = self._dataset_name(dataset_id)
                logs[f"grad_conflict/{component_name}/loss/{dataset_name}"] = loss_d.detach()

            for group_name in self.grad_conflict_parameter_groups:
                params = parameter_groups.get(str(group_name), [])
                if not params:
                    if group_name not in self._grad_conflict_warned_empty_groups:
                        self._grad_conflict_warned_empty_groups.add(str(group_name))
                    continue
                diag = self.compute_dataset_gradient_conflict_for_group(
                    dataset_losses=dataset_losses,
                    params=params,
                    group_name=str(group_name),
                    eps=self.grad_conflict_eps,
                )
                prefix = f"grad_conflict/{component_name}/{group_name}"
                for key in (
                    "num_datasets_present",
                    "mean_cosine",
                    "min_cosine",
                    "max_cosine",
                    "negative_cosine_fraction",
                    "mean_dot",
                    "min_dot",
                    "mean_norm",
                    "max_norm",
                    "min_norm",
                    "norm_ratio_max_min",
                    "conflict_score",
                ):
                    logs[f"{prefix}/{key}"] = diag[key].detach()

                diag_dataset_ids = [int(x) for x in diag["dataset_ids"].detach().cpu().tolist()]
                if self.grad_conflict_log_per_dataset_norms:
                    for i, dataset_id in enumerate(diag_dataset_ids):
                        dataset_name = self._dataset_name(dataset_id)
                        logs[f"{prefix}/norm/{dataset_name}"] = diag["norms"][i].detach()

                if self.grad_conflict_log_pairwise_matrix and len(diag_dataset_ids) <= 10:
                    cosine = diag["cosine_matrix"]
                    dot = diag["dot_matrix"]
                    for i, dataset_a in enumerate(diag_dataset_ids):
                        name_a = self._dataset_name(dataset_a)
                        for j, dataset_b in enumerate(diag_dataset_ids):
                            if j <= i:
                                continue
                            name_b = self._dataset_name(dataset_b)
                            pair = f"{name_a}__{name_b}"
                            logs[f"{prefix}/cosine/{pair}"] = cosine[i, j].detach()
                            logs[f"{prefix}/dot/{pair}"] = dot[i, j].detach()

        if logs:
            self.log_dict(
                logs,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=int(out["target"].shape[0]),
                sync_dist=False,
            )
            self._grad_conflict_batches_this_epoch += 1

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

    # Scalar metrics logged every stage. `loss` and `mu_pcc` are logged
    # explicitly (prog_bar) in _log_stage, so they are intentionally omitted
    # here to avoid logging the same key twice with different arguments.
    SCALAR_METRICS = (
        "nll",
        "nll_global",
        "nll_dataset_balanced",
        "pcc_loss",
        "pcc_loss_global",
        "pcc_loss_dataset_balanced",
        "pcc_loss_total",
        "loss_global_unbalanced",
        "loss_dataset_balanced",
        "pcc_value",
        "consensus_nll",
        "consensus_pcc_loss",
        "consensus_pcc_value",
        "replica_objective_mode",
        "replica_aux_loss",
        "replica_nll",
        "replica_pcc_loss",
        "replica_pcc_value",
        "replica_nll_weight",
        "replica_pcc_loss_weight",
        "zero_calibration_loss",
        "zero_calibration_weight",
        "zero_calibration_weight_configured",
        "zero_calibration_uses_replicas",
        "zero_calibration_p0_mean",
        "zero_calibration_p0_on_zeros",
        "zero_calibration_p0_on_nonzeros",
        "zero_calibration_target_zero_fraction",
        "zero_calibration_brier",
        "support_calibration_loss",
        "support_calibration_weight",
        "support_calibration_weight_configured",
        "support_active_probability_mean",
        "support_active_probability_on_nonzeros",
        "support_active_probability_on_zeros",
        "support_target_zero_fraction",
        "support_predicted_zero_fraction",
        "support_zero_precision",
        "support_zero_recall",
        "support_false_zero_rate",
        "pcc_loss_weight_effective",
        "pcc_valid_fraction",
        "pcc_raw_loss",
        "pcc_raw_value",
        "pcc_raw_component_weight",
        "pcc_raw_valid_fraction",
        "pcc_nb_vst_loss",
        "pcc_nb_vst_value",
        "pcc_nb_vst_component_weight",
        "pcc_nb_vst_valid_fraction",
        "pcc_hybrid_raw_contribution",
        "pcc_hybrid_nb_vst_contribution",
        "pcc_target_var_mean",
        "pcc_alpha_mean",
        "pcc_alpha_min",
        "pcc_alpha_max",
        "pcc_reliability_mean",
        "pcc_reliability_min",
        "pcc_reliability_max",
        "pcc_mean_ratio_gate_mean",
        "pcc_mean_ratio_gate_min",
        "pcc_mean_ratio_gate_max",
        "nb_sequence_reduction_mode",
        "nb_length_temper_gamma",
        "nb_length_temper_ref",
        "nb_length_weight_mean",
        "nb_length_weight_min",
        "nb_length_weight_max",
        "nb_length_weight_short",
        "nb_length_weight_medium",
        "nb_length_weight_long",
        "log1p_mse",
        "mean_ratio",
        "mu_mean",
        "target_mean",
        "J_mean",
        "J_min",
        "J_max",
        "lambda_bio_mean",
        "lambda_bio_max",
        "lambda_bio_min",
        "L_bio_mean",
        "L_bio_max",
        "rho_mean",
        "rho_max",
        "gamma_reg",
        "gamma_global_regularizer",
        "gamma_reg_weight",
        "gamma_reg_weight_configured",
        "gamma_reference_anchor_loss",
        "gamma_reference_anchor_weight",
        "additive_reference_anchor_loss",
        "additive_reference_anchor_weight",
        "gamma_mean",
        "gamma_raw_mean",
        "gamma_raw_geometric_mean",
        "gamma_centered_mean",
        "gamma_centered_geometric_mean",
        "gamma_log_mean",
        "gamma_log_std",
        "gamma_fraction_below_0p1",
        "gamma_fraction_above_10",
        "gamma_exact_zero_fraction",
        "gamma_amplitude_mean",
        "gamma_amplitude_max",
        "gamma_sparse_gate_mean",
        "gamma_sparse_gate_max",
        "gamma_sparse_gate_zero_fraction",
        "gamma_sparse_zero_precision",
        "gamma_sparse_zero_recall",
        "gamma_sparse_false_zero_rate",
        "gamma_support_logit_mean",
        "gamma_support_logit_std",
        "gamma_split_support_head_enabled",
        "gamma_gate_additive_bias_enabled",
        "gamma_sparse_transform_enabled",
        "gamma_centering_applied_fraction",
        "gamma_centering_skipped_fraction",
        "gamma_eligible_fraction",
        "gamma_mean_reliability",
        "gamma_min_reliability",
        "gamma_distinct_dataset_count_mean",
        "gamma_total_reliability_mean",
        "gamma_weighted_log_center_abs_mean",
        "gamma_centering_constraint_error",
        "gamma_mean_log_per_sample",
        "gamma_std_log_per_sample",
        "gamma_correlation_with_scale",
        "gamma_correlation_with_library_depth",
        "gamma_reference_log_abs_mean",
        "gamma_nonreference_log_abs_mean",
        "gamma_high_reliability_log_abs_mean",
        "gamma_low_reliability_log_abs_mean",
        "gamma_skipped_log_abs_mean",
        "gamma_min",
        "gamma_max",
        "log_gamma_abs_mean",
        "gamma_cross_dataset_log_center_abs_mean",
        "gamma_cross_dataset_center_applied_frac",
        "gamma_cross_dataset_center_group_size_mean",
        "additive_bias_l1_reg",
        "additive_bias_mean",
        "additive_bias_max",
        "nb_alpha_mean",
        "replica_count_mean",
        "replica_count_min",
        "replica_count_max",
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
        self.log(
            f"{stage}_mu_pcc",
            metrics["mu_pcc"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )

        for name in self.SCALAR_METRICS:
            value = metrics.get(name)
            if not torch.is_tensor(value) or value.ndim != 0:
                continue
            log_name = f"{stage}_{name}"
            if name in {
                "nll_global",
                "pcc_loss_global",
                "loss_global_unbalanced",
                "nll_dataset_balanced",
                "pcc_loss_dataset_balanced",
                "loss_dataset_balanced",
            }:
                log_name = f"{stage}/{name}"
            pcc_log_name = {
                "pcc_loss_total": "pcc/loss_total",
                "pcc_loss_weight_effective": "pcc/loss_weight_effective",
                "pcc_raw_loss": "pcc/raw_loss",
                "pcc_raw_value": "pcc/raw_value",
                "pcc_raw_component_weight": "pcc/raw_component_weight",
                "pcc_raw_valid_fraction": "pcc/raw_valid_fraction",
                "pcc_nb_vst_loss": "pcc/nb_vst_loss",
                "pcc_nb_vst_value": "pcc/nb_vst_value",
                "pcc_nb_vst_component_weight": "pcc/nb_vst_component_weight",
                "pcc_nb_vst_valid_fraction": "pcc/nb_vst_valid_fraction",
                "pcc_hybrid_raw_contribution": "pcc/hybrid_raw_contribution",
                "pcc_hybrid_nb_vst_contribution": "pcc/hybrid_nb_vst_contribution",
                "pcc_reliability_mean": "pcc/reliability_mean",
                "pcc_reliability_min": "pcc/reliability_min",
                "pcc_reliability_max": "pcc/reliability_max",
                "pcc_mean_ratio_gate_mean": "pcc/mean_ratio_gate_mean",
                "pcc_mean_ratio_gate_min": "pcc/mean_ratio_gate_min",
                "pcc_mean_ratio_gate_max": "pcc/mean_ratio_gate_max",
                "nb_sequence_reduction_mode": "nb/sequence_reduction_mode",
                "nb_length_temper_gamma": "nb/length_temper_gamma",
                "nb_length_temper_ref": "nb/length_temper_ref",
                "nb_length_weight_mean": "nb/length_weight_mean",
                "nb_length_weight_min": "nb/length_weight_min",
                "nb_length_weight_max": "nb/length_weight_max",
                "nb_length_weight_short": "nb/length_weight_short",
                "nb_length_weight_medium": "nb/length_weight_medium",
                "nb_length_weight_long": "nb/length_weight_long",
            }.get(name)
            if pcc_log_name is not None:
                log_name = f"{stage}/{pcc_log_name}"
            zero_calibration_log_name = {
                "zero_calibration_loss": "zero_calibration/loss",
                "zero_calibration_weight": "zero_calibration/weight",
                "zero_calibration_weight_configured": "zero_calibration/weight_configured",
                "zero_calibration_uses_replicas": "zero_calibration/uses_replicas",
                "zero_calibration_p0_mean": "zero_calibration/p0_mean",
                "zero_calibration_p0_on_zeros": "zero_calibration/p0_on_zeros",
                "zero_calibration_p0_on_nonzeros": "zero_calibration/p0_on_nonzeros",
                "zero_calibration_target_zero_fraction": "zero_calibration/target_zero_fraction",
                "zero_calibration_brier": "zero_calibration/brier",
            }.get(name)
            if zero_calibration_log_name is not None:
                log_name = f"{stage}/{zero_calibration_log_name}"
            support_calibration_log_name = {
                "support_calibration_loss": "support/loss",
                "support_calibration_weight": "support/weight",
                "support_calibration_weight_configured": "support/weight_configured",
                "support_active_probability_mean": "support/active_probability_mean",
                "support_active_probability_on_nonzeros": "support/active_probability_on_nonzeros",
                "support_active_probability_on_zeros": "support/active_probability_on_zeros",
                "support_target_zero_fraction": "support/target_zero_fraction",
                "support_predicted_zero_fraction": "support/predicted_zero_fraction",
                "support_zero_precision": "support/zero_precision",
                "support_zero_recall": "support/zero_recall",
                "support_false_zero_rate": "support/false_zero_rate",
            }.get(name)
            if support_calibration_log_name is not None:
                log_name = f"{stage}/{support_calibration_log_name}"
            gamma_log_name = {
                "gamma_raw_mean": "gamma/raw_mean",
                "gamma_raw_geometric_mean": "gamma/raw_geometric_mean",
                "gamma_centered_mean": "gamma/centered_mean",
                "gamma_centered_geometric_mean": "gamma/centered_geometric_mean",
                "gamma_log_mean": "gamma/log_mean",
                "gamma_log_std": "gamma/log_std",
                "gamma_fraction_below_0p1": "gamma/fraction_below_0.1",
                "gamma_fraction_above_10": "gamma/fraction_above_10",
                "gamma_exact_zero_fraction": "gamma/exact_zero_fraction",
                "gamma_amplitude_mean": "gamma/amplitude_mean",
                "gamma_amplitude_max": "gamma/amplitude_max",
                "gamma_sparse_gate_mean": "gamma/sparse_gate_mean",
                "gamma_sparse_gate_max": "gamma/sparse_gate_max",
                "gamma_sparse_gate_zero_fraction": "gamma/sparse_gate_zero_fraction",
                "gamma_sparse_zero_precision": "gamma/sparse_zero_precision",
                "gamma_sparse_zero_recall": "gamma/sparse_zero_recall",
                "gamma_sparse_false_zero_rate": "gamma/sparse_false_zero_rate",
                "gamma_support_logit_mean": "gamma/support_logit_mean",
                "gamma_support_logit_std": "gamma/support_logit_std",
                "gamma_split_support_head_enabled": "gamma/split_support_head_enabled",
                "gamma_gate_additive_bias_enabled": "gamma/gate_additive_bias_enabled",
                "gamma_sparse_transform_enabled": "gamma/sparse_transform_enabled",
                "gamma_centering_applied_fraction": "gamma/centering_applied_fraction",
                "gamma_centering_skipped_fraction": "gamma/centering_skipped_fraction",
                "gamma_eligible_fraction": "gamma/eligible_fraction",
                "gamma_mean_reliability": "gamma/mean_reliability",
                "gamma_min_reliability": "gamma/min_reliability",
                "gamma_distinct_dataset_count_mean": "gamma/distinct_dataset_count_mean",
                "gamma_total_reliability_mean": "gamma/total_reliability_mean",
                "gamma_weighted_log_center_abs_mean": "gamma/weighted_log_center_abs_mean",
                "gamma_centering_constraint_error": "gamma/centering_constraint_error",
                "gamma_reference_anchor_loss": "gamma/reference_anchor_loss",
                "gamma_global_regularizer": "gamma/global_regularizer",
                "gamma_correlation_with_scale": "gamma/correlation_with_scale",
                "gamma_correlation_with_library_depth": "gamma/correlation_with_library_depth",
                "gamma_mean_log_per_sample": "gamma/mean_log_gamma_per_sample",
                "gamma_std_log_per_sample": "gamma/std_log_gamma_per_sample",
                "gamma_reference_log_abs_mean": "gamma/reference/log_abs_mean",
                "gamma_nonreference_log_abs_mean": "gamma/non_reference/log_abs_mean",
                "gamma_high_reliability_log_abs_mean": "gamma/high_reliability/log_abs_mean",
                "gamma_low_reliability_log_abs_mean": "gamma/low_reliability/log_abs_mean",
                "gamma_skipped_log_abs_mean": "gamma/skipped/log_abs_mean",
            }.get(name)
            if gamma_log_name is not None:
                log_name = f"{stage}/{gamma_log_name}"
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
        if unique_dataset_ids.numel() <= 1:
            return

        batch_size = int(out["target"].shape[0])
        sync_dist = bool(getattr(self.config.trainer, "sync_dist_logs", False))

        per_sample = {
            "loss": metrics["loss_per_sample"],
            "nll": metrics["nll_per_sample"],
            "pcc_loss": metrics["pcc_loss_per_sample"],
            "zero_calibration_loss": metrics["zero_calibration_loss_per_sample"],
            "support_calibration_loss": metrics[
                "support_calibration_loss_per_sample"
            ],
            "pcc_raw_loss": metrics["pcc_raw_loss_per_sample"],
            "pcc_nb_vst_loss": metrics["pcc_nb_vst_loss_per_sample"],
            "mu_pcc": metrics["mu_pcc_per_sample"],
        }

        for dataset_id_tensor in unique_dataset_ids:
            dataset_id = int(dataset_id_tensor.item())
            dataset_name = self._dataset_name(dataset_id)
            sample_mask = dataset_ids == dataset_id_tensor

            for metric_name, values in per_sample.items():
                value = values[sample_mask].mean()
                self.log(
                    f"{stage}_{metric_name}/{dataset_name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size,
                    sync_dist=sync_dist,
                )
                by_dataset_name = {
                    "loss": "loss",
                    "nll": "nll",
                    "pcc_loss": "pcc",
                    "zero_calibration_loss": "zero_calibration/loss",
                    "support_calibration_loss": "support/loss",
                    "pcc_raw_loss": "pcc/raw_loss",
                    "pcc_nb_vst_loss": "pcc/nb_vst_loss",
                }.get(metric_name)
                if by_dataset_name is not None:
                    self.log(
                        f"{stage}/{by_dataset_name}_by_dataset/{dataset_name}",
                        value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        batch_size=batch_size,
                        sync_dist=sync_dist,
                    )

            extras = out["extras"]
            pos_mask = sample_mask.reshape(-1, 1) & out["mask"].bool()
            if bool(pos_mask.any()):
                gamma = extras.get("gamma", torch.ones_like(extras["L_bio"])).float()
                log_gamma = extras.get(
                    "log_gamma",
                    torch.zeros_like(extras["L_bio"]),
                ).float()
                gamma_raw = extras.get("gamma_raw", torch.ones_like(extras["L_bio"])).float()
                reliability = extras.get(
                    "gamma_centering_reliability",
                    torch.zeros_like(extras["L_bio"]),
                ).float()
                applied = extras.get(
                    "gamma_centering_applied",
                    torch.zeros_like(extras["L_bio"], dtype=torch.bool),
                ).bool()
                for log_key, value in {
                    f"{stage}/gamma/log_mean_by_dataset/{dataset_name}": log_gamma[pos_mask].mean(),
                    f"{stage}/gamma/log_std_by_dataset/{dataset_name}": (
                        log_gamma[pos_mask].std(unbiased=False)
                        if int(pos_mask.sum().item()) > 1
                        else log_gamma.new_tensor(0.0)
                    ),
                    f"{stage}/gamma/raw_mean_by_dataset/{dataset_name}": gamma_raw[pos_mask].mean(),
                    f"{stage}/gamma/centered_mean_by_dataset/{dataset_name}": gamma[pos_mask].mean(),
                    f"{stage}/gamma/mean_reliability_by_dataset/{dataset_name}": reliability[pos_mask].mean(),
                    f"{stage}/gamma/centering_applied_fraction_by_dataset/{dataset_name}": applied[pos_mask].float().mean(),
                }.items():
                    self.log(
                        log_key,
                        value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        batch_size=batch_size,
                        sync_dist=sync_dist,
                    )

    # ============================================================
    # Profile plots
    # ============================================================

    def on_train_epoch_start(self) -> None:
        self._grad_conflict_batches_this_epoch = 0

    def on_validation_epoch_start(self) -> None:
        self._val_profile_plot_logged_this_epoch = False
        if self.residual_diag_enabled and self.residual_diag_run_on_validation:
            self.residual_diag.reset()

    def on_validation_epoch_end(self) -> None:
        if not (self.residual_diag_enabled and self.residual_diag_run_on_validation):
            return
        metrics, rows, report = self.residual_diag.compute()
        if metrics:
            sync_dist = bool(getattr(self.config.trainer, "sync_dist_logs", False))
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
            f"likelihood: {self.config.loss.profile_likelihood}",
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

    def training_step(self, batch, batch_idx):
        out = self._forward_batch(batch)
        metrics = self._compute_loss_and_metrics(out)
        self._log_stage(stage="train", out=out, metrics=metrics)
        cagrad_logs = self._prepare_biological_cagrad(out=out, metrics=metrics)
        if cagrad_logs:
            self.log_dict(
                cagrad_logs,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
            )
        self._log_gradient_conflict_diagnostics(out=out, metrics=metrics)
        return metrics["loss"]

    def on_before_optimizer_step(self, optimizer):
        # Zero out any non-finite gradient from a degenerate batch so a single
        # bad step cannot poison the weights (runs after gradient clipping).
        params = [p for p in self.parameters() if p.grad is not None]
        if any(not torch.isfinite(p.grad).all() for p in params):
            for p in params:
                p.grad.zero_()
        self._cagrad_bio_grad_overrides = {}

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
            "gamma_amplitude": extras.get(
                "gamma_amplitude",
                extras.get("gamma", torch.ones_like(extras["L_bio"])),
            ),
            "gamma_sparse_gate": extras.get(
                "gamma_sparse_gate",
                torch.ones_like(extras["L_bio"]),
            ),
            "gamma_support_logits": extras.get(
                "gamma_support_logits",
                torch.zeros_like(extras["L_bio"]),
            ),
            "gamma_support_logits_raw": extras.get(
                "gamma_support_logits_raw",
                torch.zeros_like(extras["L_bio"]),
            ),
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
            "additive_bias": extras.get(
                "additive_bias",
                torch.zeros_like(extras["L_bio"]),
            ),
            "scale_dt": extras["scale_dt"],
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
