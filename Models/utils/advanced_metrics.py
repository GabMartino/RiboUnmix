import torch
from torchmetrics.functional.classification import binary_average_precision
from torchmetrics.functional import spearman_corrcoef


def masked_1d_wasserstein(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the Earth Mover's Distance (1D Wasserstein) between predicted and target profiles.
    Rewards models for spatial proximity to the true stall site.
    """
    # Normalize profiles into probability mass functions (PMF)
    pred_sum = (pred * mask).sum(dim=1, keepdim=True).clamp_min(1e-8)
    target_sum = (target * mask).sum(dim=1, keepdim=True).clamp_min(1e-8)

    pred_pmf = (pred * mask) / pred_sum
    target_pmf = (target * mask) / target_sum

    # Calculate Cumulative Distribution Functions (CDF)
    pred_cdf = torch.cumsum(pred_pmf, dim=1) * mask
    target_cdf = torch.cumsum(target_pmf, dim=1) * mask

    # 1D EMD is the exact L1 distance between the CDFs
    emd = torch.abs(pred_cdf - target_cdf).sum(dim=1)
    return emd.mean()


def masked_auprc_peak_caller(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                             top_percentile: float = 0.95) -> torch.Tensor:
    """
    Binarizes the target to find the top X% of queuing peaks, then calculates AUPRC.
    Evaluates the model as a strict Codon Stalling Site detector.
    """
    auprc_scores = []

    for i in range(pred.shape[0]):
        valid_len = int(mask[i].sum().item())
        if valid_len < 10:
            continue

        p_i = pred[i, :valid_len]
        t_i = target[i, :valid_len]

        # Define ground truth peaks (e.g., counts above the 95th percentile)
        threshold = torch.quantile(t_i, top_percentile)
        binary_target = (t_i >= threshold).long()

        # Only compute if we actually have peaks and non-peaks
        if binary_target.sum() > 0 and binary_target.sum() < valid_len:
            score = binary_average_precision(p_i, binary_target)
            auprc_scores.append(score)

    if not auprc_scores:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(auprc_scores).mean()


def physics_asymmetry_ratio(L_queue: torch.Tensor, css_list: list, window: int = 5) -> torch.Tensor:
    """
    Checks if the model learned that traffic jams propagate upstream (5') of the CSS,
    leaving a downstream (3') shadow. Expectation: Ratio > 1.0.
    """
    ratios = []

    for b in range(L_queue.shape[0]):
        css_idx = css_list[b]
        if css_idx is None or len(css_idx) == 0:
            continue

        up_sum, down_sum = 0.0, 0.0
        seq_len = L_queue.shape[1]

        for idx in css_idx:
            idx = int(idx)
            # Upstream (assuming smaller index is 5')
            up_start = max(0, idx - window)
            up_sum += L_queue[b, up_start:idx].sum().item()

            # Downstream shadow
            down_end = min(seq_len, idx + window + 1)
            down_sum += L_queue[b, idx + 1:down_end].sum().item()

        # Add epsilon to prevent division by zero
        ratio = (up_sum + 1e-8) / (down_sum + 1e-8)
        ratios.append(ratio)

    if not ratios:
        return torch.tensor(0.0, device=L_queue.device)
    return torch.tensor(ratios).mean()


def compute_advanced_diagnostics(target, L_queue, mask, css):
    """Wraps all advanced metrics into a single call."""
    with torch.no_grad():
        mae = torch.abs((L_queue - target) * mask).sum() / mask.sum().clamp_min(1.0)
        emd = masked_1d_wasserstein(L_queue, target, mask)
        auprc = masked_auprc_peak_caller(L_queue, target, mask)
        phys_ratio = physics_asymmetry_ratio(L_queue, css)

    return {
        "MAE": mae,
        "EMD": emd,
        "AUPRC": auprc,
        "Upstream_Ratio": phys_ratio
    }


from torchmetrics.functional.classification import binary_average_precision


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error over valid sequence lengths."""
    return torch.abs((pred - target) * mask).sum() / mask.sum().clamp_min(1.0)


def masked_1d_wasserstein(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Earth Mover's Distance: measures spatial displacement of ribosome density."""
    pred_sum = (pred * mask).sum(dim=1, keepdim=True).clamp_min(1e-8)
    target_sum = (target * mask).sum(dim=1, keepdim=True).clamp_min(1e-8)

    pred_pmf = (pred * mask) / pred_sum
    target_pmf = (target * mask) / target_sum

    pred_cdf = torch.cumsum(pred_pmf, dim=1) * mask
    target_cdf = torch.cumsum(target_pmf, dim=1) * mask

    return torch.abs(pred_cdf - target_cdf).sum(dim=1).mean()

import torch.nn.functional as F
def masked_auprc_peak_caller(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        top_percentile: float = 0.95,
        window: int = 1
) -> torch.Tensor:
    """
    Evaluates the model as a CSS (Codon Stalling Site) detector.
    Allows a +/- `window` codon tolerance for peak matching using 1D dilation.
    """
    auprc_scores = []

    for i in range(pred.shape[0]):
        valid_len = int(mask[i].sum().item())
        if valid_len < 10:
            continue

        p_i = pred[i, :valid_len]
        t_i = target[i, :valid_len]

        # 1. Binarize the ground truth based on the top percentile
        threshold = torch.quantile(t_i, top_percentile)
        binary_target = (t_i >= threshold).float()  # Float for pooling

        # 2. Apply window tolerance via 1D Dilation (Max Pooling)
        if window > 0:
            # Reshape for max_pool1d: (Batch=1, Channels=1, Length=L)
            binary_target_reshaped = binary_target.view(1, 1, -1)

            # Kernel size of 3 with padding 1 creates a [-1, 0, +1] window
            kernel_size = 2 * window + 1
            dilated_target = F.max_pool1d(
                binary_target_reshaped,
                kernel_size=kernel_size,
                stride=1,
                padding=window
            )

            # Flatten back to 1D
            binary_target = dilated_target.view(-1)

        # Convert back to integers for the classification metric
        binary_target = binary_target.long()

        # 3. Calculate AUPRC
        if binary_target.sum() > 0 and binary_target.sum() < valid_len:
            score = binary_average_precision(p_i, binary_target)
            auprc_scores.append(score)

    return torch.stack(auprc_scores).mean() if auprc_scores else torch.tensor(0.0, device=pred.device)


def physics_asymmetry_ratio(L_queue: torch.Tensor, css_list: list, window: int = 5) -> torch.Tensor:
    """Checks for the 5' upstream traffic jam vs 3' downstream shadow."""
    ratios = []
    for b in range(L_queue.shape[0]):
        css_idx = css_list[b]
        if css_idx is None or len(css_idx) == 0: continue

        up_sum, down_sum = 0.0, 0.0
        seq_len = L_queue.shape[1]

        for idx in css_idx:
            idx = int(idx)
            up_sum += L_queue[b, max(0, idx - window):idx].sum().item()
            down_sum += L_queue[b, idx + 1:min(seq_len, idx + window + 1)].sum().item()

        ratios.append((up_sum + 1e-8) / (down_sum + 1e-8))

    return torch.tensor(ratios).mean() if ratios else torch.tensor(0.0, device=L_queue.device)