import torch

@torch.no_grad()
def compute_S_mean(
        y_true: torch.Tensor,
        mask: torch.Tensor,
        censor_threshold: float = 0.5,
        use_censor_threshold: bool = True,
        use_nonzero_only: bool = False,
) -> torch.Tensor:
    """
    Computes the mean of valid reads for each transcript.
    Preserves the total mass/abundance of the transcript.
    """
    y = y_true.to(torch.float32)
    m = mask.bool()

    if use_censor_threshold:
        m = m & (y > float(censor_threshold))
    elif use_nonzero_only:
        m = m & (y > 0)

    # Sum the valid reads
    y_valid = y * m.float()
    total_reads = y_valid.sum(dim=1, keepdim=True)  # [B, 1]

    # Count the valid positions to find the denominator
    counts = m.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]

    S_mean = total_reads / counts

    return S_mean


@torch.no_grad()
def compute_S_trimmed_mean(y, mask, trim_top_frac=0.02, eps=1e-8):
    mask_b = mask.bool()
    out = []

    for i in range(y.shape[0]):
        vals = y[i][mask_b[i]].float()
        vals = vals[torch.isfinite(vals)]

        if vals.numel() == 0:
            out.append(torch.tensor(eps, device=y.device, dtype=y.dtype))
            continue

        vals = vals.sort().values
        k = int((1.0 - trim_top_frac) * vals.numel())
        k = max(1, k)

        vals = vals[:k]
        out.append(vals.mean().clamp_min(eps))

    return torch.stack(out)