from __future__ import annotations

import torch
import torch.nn as nn


class MultinomialProfileLoss(nn.Module):
    """
    Conditional profile loss.

    Interprets y as counts allocated over codon positions, conditional on
    total transcript mass N = sum_i y_i.

    Loss:
        - sum_i (y_i / N) log pi_i

    where:
        pi_i = mu_i / sum_j mu_j

    This is equivalent to per-read multinomial NLL up to constants.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        normalize_by_total: bool = True,
    ):
        super().__init__()
        self.eps = float(eps)
        self.normalize_by_total = bool(normalize_by_total)

    def forward(
        self,
        *,
        mu: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=mu.dtype)

        mu = mu.float().clamp_min(0.0) * mask_f
        y = y_true.float().clamp_min(0.0) * mask_f

        mu_mass = mu.sum(dim=1, keepdim=True).clamp_min(self.eps)
        y_mass = y.sum(dim=1, keepdim=True).clamp_min(self.eps)

        pi = mu / mu_mass
        pi = pi.clamp_min(self.eps)
        pi = pi / (pi * mask_f).sum(dim=1, keepdim=True).clamp_min(self.eps)

        if self.normalize_by_total:
            y_weight = y / y_mass
            loss_per_sample = -(y_weight * torch.log(pi) * mask_f).sum(dim=1)
        else:
            loss_per_sample = -(y * torch.log(pi) * mask_f).sum(dim=1)

        if return_per_sample:
            return loss_per_sample

        return loss_per_sample.mean()