import torch


@torch.no_grad()
def compute_S_quantile(
        y_true: torch.Tensor,
        mask: torch.Tensor,
        q: float,
        eps: float,
        censor_threshold: float,
        use_censor_threshold: bool,
        use_nonzero_only: bool,
) -> torch.Tensor:
    y = y_true.to(torch.float32)
    m = mask.bool()

    if use_censor_threshold:
        m = m & (y > float(censor_threshold))
    elif use_nonzero_only:
        m = m & (y > 0)

    counts = m.sum(dim=1)  # [B]
    B, _T = y.shape

    vals = y.masked_fill(~m, float("inf"))
    vals_sorted, _ = torch.sort(vals, dim=1)

    q = float(q)
    k = (q * (counts.clamp_min(1) - 1).float()).floor().long()
    k_max = (counts.clamp_min(1) - 1).long()
    k = torch.minimum(k.clamp_min(0), k_max)

    S = vals_sorted.gather(1, k.view(B, 1))
    S = torch.where(counts.view(B, 1) > 0, S, torch.full_like(S, float(eps)))
    return S.clamp_min(float(eps))


import torch


@torch.no_grad()
def compute_S_median(
        y_true: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-8,
        censor_threshold: float = 0.5,
        use_censor_threshold: bool = True,
        use_nonzero_only: bool = False,
) -> torch.Tensor:
    """
    Computes the median of valid reads for each transcript.
    Acts as the baseline scale (S_base) for the M/M/1 exponential queue model.
    """
    y = y_true.to(torch.float32)
    m = mask.bool()

    if use_censor_threshold:
        m = m & (y > float(censor_threshold))
    elif use_nonzero_only:
        m = m & (y > 0)

    counts = m.sum(dim=1)  # [B]
    B, _T = y.shape

    # Fill invalid positions with infinity so they sort to the very end
    vals = y.masked_fill(~m, float("inf"))
    vals_sorted, _ = torch.sort(vals, dim=1)

    # Find the exact middle index for each sequence in the batch
    # For N elements, the median index is floor((N - 1) / 2)
    k_mid = ((counts.clamp_min(1) - 1).float() / 2.0).floor().long()

    # Gather the median values
    S_median = vals_sorted.gather(1, k_mid.view(B, 1))

    # If a sequence had 0 valid counts, fallback to eps to prevent NaNs
    S_median = torch.where(counts.view(B, 1) > 0, S_median, torch.full_like(S_median, float(eps)))

    return S_median.clamp_min(float(eps))


@torch.no_grad()
def compute_S_mean(
        y_true: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-8,
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

    # Fallback to eps if sequence is entirely empty
    S_mean = torch.where(counts > 0, S_mean, torch.full_like(S_mean, float(eps)))

    return S_mean.clamp_min(float(eps))