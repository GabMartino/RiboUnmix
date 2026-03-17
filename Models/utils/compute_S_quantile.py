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
