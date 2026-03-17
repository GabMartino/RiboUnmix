from __future__ import annotations

from dataclasses import dataclass

import torch






def mu_total_from_median_lognormal(mu_phys_median: torch.Tensor, pi: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Unconditional mean of a ZI-LogNormal when `mu_phys_median` is the positive median.

    LogNormal mean = median * exp(0.5*sigma^2)
    Unconditional mean = (1-pi) * mean_pos
    """
    return (1.0 - pi) * mu_phys_median * torch.exp(0.5 * (sigma ** 2))

def masked_mean(x: torch.Tensor, mask: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.to(x.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask_f).sum(dim=1, keepdim=True) / (denom + float(eps))
