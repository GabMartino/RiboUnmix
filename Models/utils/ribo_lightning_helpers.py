from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence


def cfg_get(config: Any, path: str, default: Any = None) -> Any:
    obj = config

    for part in path.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return default

    return obj


def unpack_batch(batch: Any) -> dict[str, Any]:
    """
    Supports both old and new batch layouts.

    Old 7-item layout:
        ids_datasets, ids, x_packed, y, lengths, mask, css

    New 8-item layout:
        ids_datasets, ids, x_packed, y, lengths, mask, codon_ids, css
    """
    if len(batch) == 8:
        (
            ids_datasets_sorted,
            ids,
            packed_sequence,
            profiles_target,
            lengths,
            mask,
            codon_ids,
            css,
        ) = batch
    elif len(batch) == 7:
        (
            ids_datasets_sorted,
            ids,
            packed_sequence,
            profiles_target,
            lengths,
            mask,
            css,
        ) = batch
        codon_ids = None
    else:
        raise ValueError(f"Expected batch with 7 or 8 elements, got {len(batch)}.")

    return {
        "ids_datasets_sorted": ids_datasets_sorted,
        "ids": ids,
        "packed_sequence": packed_sequence,
        "profiles_target": profiles_target,
        "lengths": lengths,
        "mask": mask,
        "codon_ids": codon_ids,
        "css": css,
    }


def get_css_item(css: Any, sample_idx: int) -> Any:
    if css is None:
        return None

    if isinstance(css, (list, tuple)):
        if sample_idx >= len(css):
            return None
        return css[sample_idx]

    if torch.is_tensor(css):
        if css.ndim == 0:
            return css
        if sample_idx >= css.shape[0]:
            return None
        return css[sample_idx]

    try:
        return css[sample_idx]
    except Exception:
        return None


def normalize_css_positions(
    css_i: Any,
    L: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Converts one sample's CSS annotation into valid integer positions.

    Supports:
      - list/array/tensor of positions
      - dense boolean mask of length L
      - dense 0/1 mask of length L
    """
    if css_i is None:
        return torch.empty(0, dtype=torch.long, device=device)

    try:
        if torch.is_tensor(css_i):
            arr = css_i.detach().cpu()
        else:
            arr = torch.as_tensor(css_i)
    except Exception:
        return torch.empty(0, dtype=torch.long, device=device)

    arr = arr.reshape(-1)

    if arr.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    if arr.dtype == torch.bool:
        if arr.numel() >= L:
            pos = torch.nonzero(arr[:L], as_tuple=False).reshape(-1)
        else:
            pos = torch.nonzero(arr, as_tuple=False).reshape(-1)
    else:
        if torch.is_floating_point(arr):
            arr = arr[torch.isfinite(arr)]

        if arr.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        arr_long = arr.to(torch.long)

        if arr_long.numel() == L and torch.all((arr_long == 0) | (arr_long == 1)):
            pos = torch.nonzero(arr_long.bool(), as_tuple=False).reshape(-1)
        else:
            pos = arr_long.reshape(-1)

    pos = pos.to(dtype=torch.long)
    pos = pos[(pos >= 0) & (pos < int(L))]

    if pos.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    pos = torch.unique(pos, sorted=True)
    return pos.to(device=device)


def css_window_mask(
    css_pos: torch.Tensor,
    L: int,
    window: int,
    device: torch.device,
) -> torch.Tensor:
    css_mask = torch.zeros(int(L), dtype=torch.bool, device=device)

    if css_pos.numel() == 0:
        return css_mask

    window = max(0, int(window))

    for p in css_pos.detach().cpu().tolist():
        left = max(0, int(p) - window)
        right = min(int(L), int(p) + window + 1)
        css_mask[left:right] = True

    return css_mask


def css_rank_recall_enrichment_for_score(
    *,
    score: torch.Tensor,
    mask_b: torch.Tensor,
    css: Any,
    top_frac: float,
    min_k: int,
    window: int,
    eps: float = 1e-8,
) -> tuple[dict[str, torch.Tensor], int]:
    device = score.device

    rank_percentiles = []
    recalls = []
    enrichments = []

    B = int(score.shape[0])

    for i in range(B):
        L = int(mask_b[i].sum().detach().cpu().item())

        if L < 3:
            continue

        score_i = score[i, :L].detach().float()
        css_i = get_css_item(css, i)

        css_pos = normalize_css_positions(
            css_i=css_i,
            L=L,
            device=device,
        )

        if css_pos.numel() == 0:
            continue

        css_win = css_window_mask(
            css_pos=css_pos,
            L=L,
            window=window,
            device=device,
        )

        non_css_win = ~css_win

        if css_win.sum() == 0 or non_css_win.sum() == 0:
            continue

        css_scores = score_i[css_pos]

        percentiles = []
        for s in css_scores:
            percentiles.append((score_i <= s).float().mean())

        rank_percentiles.append(torch.stack(percentiles).mean())

        k = max(int(min_k), int(math.ceil(float(top_frac) * L)))
        k = min(k, L)

        if k > 0:
            top_idx = torch.topk(score_i, k=k, largest=True).indices

            distances = (
                css_pos.reshape(-1, 1)
                - top_idx.reshape(1, -1)
            ).abs()

            hit = distances.min(dim=1).values <= int(window)
            recalls.append(hit.float().mean())

        css_mean = score_i[css_win].mean()
        bg_mean = score_i[non_css_win].mean().clamp_min(eps)

        enrichments.append(css_mean / bg_mean)

    if not rank_percentiles:
        return {}, 0

    out = {
        "css_rank_percentile": torch.stack(rank_percentiles).mean(),
        "css_recall_topk_window": (
            torch.stack(recalls).mean()
            if recalls
            else torch.zeros((), device=device)
        ),
        "css_enrichment": torch.stack(enrichments).mean(),
    }

    return out, len(rank_percentiles)


def css_delta_for_values(
    *,
    values: torch.Tensor,
    mask_b: torch.Tensor,
    css: Any,
    window: int,
) -> tuple[torch.Tensor | None, int]:
    device = values.device
    deltas = []

    B = int(values.shape[0])

    for i in range(B):
        L = int(mask_b[i].sum().detach().cpu().item())

        if L < 3:
            continue

        values_i = values[i, :L].detach().float()
        css_i = get_css_item(css, i)

        css_pos = normalize_css_positions(
            css_i=css_i,
            L=L,
            device=device,
        )

        if css_pos.numel() == 0:
            continue

        css_win = css_window_mask(
            css_pos=css_pos,
            L=L,
            window=window,
            device=device,
        )

        non_css_win = ~css_win

        if css_win.sum() == 0 or non_css_win.sum() == 0:
            continue

        deltas.append(values_i[css_win].mean() - values_i[non_css_win].mean())

    if not deltas:
        return None, 0

    return torch.stack(deltas).mean(), len(deltas)


def compute_css_diagnostics(
    *,
    L_queue: torch.Tensor,
    mask_b: torch.Tensor,
    css: Any,
    top_frac: float,
    min_k: int,
    window: int,
    eps: float,
    L_effective: torch.Tensor | None = None,
    mu_base: torch.Tensor | None = None,
    additive_bg: torch.Tensor | None = None,
    b_offset: torch.Tensor | None = None,
    phi: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], int]:
    logs: dict[str, torch.Tensor] = {}

    L_metrics, css_count = css_rank_recall_enrichment_for_score(
        score=L_queue,
        mask_b=mask_b,
        css=css,
        top_frac=top_frac,
        min_k=min_k,
        window=window,
        eps=eps,
    )

    if css_count == 0:
        return logs, 0

    for name, value in L_metrics.items():
        logs[f"css_L_queue_{name}"] = value

    if L_effective is not None:
        L_eff_metrics, _ = css_rank_recall_enrichment_for_score(
            score=L_effective,
            mask_b=mask_b,
            css=css,
            top_frac=top_frac,
            min_k=min_k,
            window=window,
            eps=eps,
        )

        for name, value in L_eff_metrics.items():
            logs[f"css_L_effective_{name}"] = value

    if mu_base is not None:
        mu_base_metrics, _ = css_rank_recall_enrichment_for_score(
            score=mu_base,
            mask_b=mask_b,
            css=css,
            top_frac=top_frac,
            min_k=min_k,
            window=window,
            eps=eps,
        )

        for name, value in mu_base_metrics.items():
            logs[f"css_mu_base_{name}"] = value

    if additive_bg is not None:
        A_metrics, _ = css_rank_recall_enrichment_for_score(
            score=additive_bg,
            mask_b=mask_b,
            css=css,
            top_frac=top_frac,
            min_k=min_k,
            window=window,
            eps=eps,
        )

        for name, value in A_metrics.items():
            logs[f"css_additive_bg_{name}"] = value

    if b_offset is not None:
        b_delta, b_count = css_delta_for_values(
            values=b_offset,
            mask_b=mask_b,
            css=css,
            window=window,
        )

        if b_delta is not None and b_count > 0:
            logs["css_b_delta"] = b_delta

    if phi is not None:
        phi_delta, phi_count = css_delta_for_values(
            values=phi,
            mask_b=mask_b,
            css=css,
            window=window,
        )

        if phi_delta is not None and phi_count > 0:
            logs["css_phi_delta"] = phi_delta

    return logs, css_count


def get_pcgrad_target_parameters(
    model: nn.Module,
    *,
    biology_only: bool = True,
) -> list[nn.Parameter]:
    if biology_only:
        biological_model = getattr(model, "biological_model", None)

        if biological_model is None:
            raise AttributeError(
                "pcgrad_biology_only=True, but model.biological_model does not exist."
            )

        return [
            p for p in biological_model.parameters()
            if p.requires_grad
        ]

    return [
        p for p in model.parameters()
        if p.requires_grad
    ]


def flatten_current_grads(
    params: list[nn.Parameter],
) -> torch.Tensor:
    flats = []

    for p in params:
        if p.grad is None:
            flats.append(torch.zeros_like(p).reshape(-1))
        else:
            flats.append(p.grad.detach().clone().reshape(-1))

    if not flats:
        return torch.empty(0)

    return torch.cat(flats, dim=0)


def assign_flat_grads(
    params: list[nn.Parameter],
    flat_grad: torch.Tensor,
) -> None:
    offset = 0

    for p in params:
        n = p.numel()
        g = flat_grad[offset:offset + n].view_as(p)
        offset += n

        if p.grad is None:
            p.grad = g.detach().clone()
        else:
            p.grad.detach().copy_(g)


def pcgrad_combine(
    flat_grads: list[torch.Tensor],
    eps: float = 1e-12,
) -> torch.Tensor:
    if len(flat_grads) == 0:
        raise ValueError("No gradients passed to PCGrad.")

    if len(flat_grads) == 1:
        return flat_grads[0]

    projected = []

    for i, g_i_original in enumerate(flat_grads):
        g_i = g_i_original.clone()

        order = torch.randperm(len(flat_grads), device=g_i.device)

        for j_tensor in order:
            j = int(j_tensor.item())

            if j == i:
                continue

            g_j = flat_grads[j]

            dot = torch.dot(g_i, g_j)
            denom = torch.dot(g_j, g_j).clamp_min(eps)

            if dot < 0:
                g_i = g_i - (dot / denom) * g_j

        projected.append(g_i)

    return torch.stack(projected, dim=0).mean(dim=0)


def pcgrad_pairwise_stats(
    flat_grads: list[torch.Tensor],
    eps: float = 1e-12,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if len(flat_grads) < 2:
        return None, None

    cosines = []
    conflicts = []

    for i in range(len(flat_grads)):
        for j in range(i + 1, len(flat_grads)):
            g_i = flat_grads[i]
            g_j = flat_grads[j]

            denom = (g_i.norm() * g_j.norm()).clamp_min(eps)
            cosine = torch.dot(g_i, g_j) / denom

            cosines.append(cosine)
            conflicts.append((cosine < 0).float())

    return torch.stack(cosines).mean(), torch.stack(conflicts).mean()


def per_dataset_losses(
    loss_per_sample: torch.Tensor,
    dataset_ids: torch.Tensor,
) -> list[torch.Tensor]:
    dataset_ids = dataset_ids.reshape(-1)
    loss_per_sample = loss_per_sample.reshape(-1)

    losses = []

    for dataset_id in torch.unique(dataset_ids.detach()):
        ds_mask = dataset_ids == dataset_id

        if torch.any(ds_mask):
            losses.append(loss_per_sample[ds_mask].mean())

    return losses


def get_shift_values(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    shift_head = getattr(
        getattr(model, "dataset_bias_model", None),
        "dataset_shift_head",
        None,
    )

    if shift_head is None:
        return None

    shifts = getattr(shift_head, "shifts", None)

    if shifts is None:
        return None

    return torch.tensor(
        list(shifts),
        device=device,
        dtype=dtype,
    )


def make_validation_profile_figure(
    *,
    y: torch.Tensor,
    mu: torch.Tensor,
    phi: torch.Tensor,
    tweedie_p: torch.Tensor,
    L_queue: torch.Tensor,
    mask_b: torch.Tensor,
    mu_pcc_per_sample: torch.Tensor,
    L_queue_pcc_per_sample: torch.Tensor,
    sample_idx: int,
    mu_base: torch.Tensor | None = None,
    additive_bg: torch.Tensor | None = None,
    additive_rel: torch.Tensor | None = None,
    css: Any | None = None,
) -> Any | None:
    B = y.shape[0]

    if B == 0:
        return None

    sample_idx = max(0, min(int(sample_idx), B - 1))

    with torch.no_grad():
        valid = mask_b[sample_idx].detach().bool().cpu()
        L = int(valid.sum().item())

        if L < 2:
            return None

        y_i = y[sample_idx].detach().float().cpu()[valid]
        mu_i = mu[sample_idx].detach().float().cpu()[valid]
        phi_i = phi[sample_idx].detach().float().cpu()[valid]
        L_queue_i = L_queue[sample_idx].detach().float().cpu()[valid]

        p_scalar = (
            tweedie_p.detach()
            .float()
            .reshape(-1)
            .mean()
            .cpu()
            .clamp(1.0001, 1.9999)
        )

        tweedie_var_i = phi_i.clamp_min(1e-8) * torch.pow(
            mu_i.clamp_min(1e-8),
            p_scalar,
        )

        mu_pcc_i = float(mu_pcc_per_sample[sample_idx].detach().float().cpu())
        L_queue_pcc_i = float(
            L_queue_pcc_per_sample[sample_idx].detach().float().cpu()
        )
        p_value = float(p_scalar)

        x = torch.arange(y_i.numel()).numpy()

        y_np = y_i.numpy()
        mu_np = mu_i.numpy()
        L_queue_np = L_queue_i.numpy()
        var_np = tweedie_var_i.numpy()
        phi_np = phi_i.numpy()

        if mu_base is not None:
            mu_base_np = mu_base[sample_idx].detach().float().cpu()[valid].numpy()
        else:
            mu_base_np = None

        if additive_bg is not None:
            additive_bg_np = additive_bg[sample_idx].detach().float().cpu()[valid].numpy()
        else:
            additive_bg_np = None

        if additive_rel is not None:
            additive_rel_np = additive_rel[sample_idx].detach().float().cpu()[valid].numpy()
        else:
            additive_rel_np = None

        css_i = get_css_item(css, sample_idx)
        css_pos = normalize_css_positions(
            css_i=css_i,
            L=L,
            device=torch.device("cpu"),
        )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0, 1.0]},
    )

    fig.suptitle(
        f"Validation profile diagnostic | sample={sample_idx} | "
        f"PCC(mu, y)={mu_pcc_i:.4f} | "
        f"PCC(L_queue, y)={L_queue_pcc_i:.4f} | "
        f"Tweedie p={p_value:.4f}",
        fontsize=12,
    )

    axes[0].plot(x, y_np, label="target y", linewidth=1.2)
    axes[0].plot(x, mu_np, label="mu = mu_base + additive", linewidth=1.2)

    if mu_base_np is not None:
        axes[0].plot(x, mu_base_np, label="mu_base", linewidth=1.0)

    if additive_bg_np is not None:
        axes[0].plot(x, additive_bg_np, label="additive_bg", linewidth=1.0)

    axes[0].set_ylabel("profile")
    axes[0].set_title("Target profile vs predicted mean")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, L_queue_np, label="L_queue", linewidth=1.2)
    axes[1].set_ylabel("L_queue")
    axes[1].set_title("Biological queueing prediction")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        x,
        var_np,
        label=r"Tweedie variance $\phi\mu^p$",
        linewidth=1.2,
    )
    axes[2].plot(
        x,
        phi_np,
        label=r"$\phi$",
        linewidth=1.0,
        linestyle=":",
    )

    if additive_rel_np is not None:
        ax2 = axes[2].twinx()
        ax2.plot(
            x,
            additive_rel_np,
            label="additive_rel = A/S",
            linewidth=1.0,
            linestyle="--",
        )
        ax2.set_ylabel("additive_rel")
        ax2.legend(loc="upper left")

    axes[2].set_ylabel("variance / phi")
    axes[2].set_xlabel("codon position")
    axes[2].set_title("Tweedie variance, phi, and additive_rel diagnostic")
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        for j, css_position in enumerate(css_pos.detach().cpu().tolist()):
            ax.axvline(
                int(css_position),
                linestyle="--",
                linewidth=0.8,
                alpha=0.35,
                label="CSS" if j == 0 else None,
            )
        ax.legend(loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig

