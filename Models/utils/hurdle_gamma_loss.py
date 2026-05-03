from __future__ import annotations

import torch
import torch.nn as nn


class HurdleGammaLoss(nn.Module):
    """
    Hurdle Gamma loss for continuous nonnegative targets.

    mu:
        positive mean prediction, shape [B, T]

    pi:
        probability that y is zero / censored, shape [B, T]

    phi:
        Gamma relative dispersion, shape [B, T]
        Var[Y | positive] = phi * mu^2

    y_true:
        continuous nonnegative target, shape [B, T]

    mask:
        valid-position mask, shape [B, T]
    """

    def __init__(
        self,
        censor_threshold: float = 0.5,
        eps: float = 1e-8,
        phi_min: float = 1e-4,
        phi_max: float = 10.0,
    ):
        super().__init__()
        self.censor_threshold = float(censor_threshold)
        self.eps = float(eps)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)

    def forward(
        self,
        mu: torch.Tensor,
        pi: torch.Tensor,
        phi: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        y = y_true.float()
        m = mask.bool()
        eps = self.eps

        mu = mu.float().clamp_min(eps)
        pi = pi.float().clamp(eps, 1.0 - eps)
        phi = phi.float().clamp(self.phi_min, self.phi_max)

        low = m & (y <= self.censor_threshold)
        pos = m & (y > self.censor_threshold)

        nll = torch.zeros_like(y)

        # Hurdle / censoring component.
        if low.any():
            nll[low] = -torch.log(pi[low])

        # Positive continuous component.
        if pos.any():
            y_pos = y[pos].clamp_min(eps)
            mu_pos = mu[pos]
            phi_pos = phi[pos]

            shape = (1.0 / phi_pos).clamp_min(eps)
            scale = (phi_pos * mu_pos).clamp_min(eps)

            log_prob_gamma = (
                (shape - 1.0) * torch.log(y_pos)
                - y_pos / scale
                - shape * torch.log(scale)
                - torch.lgamma(shape)
            )

            nll[pos] = -torch.log1p(-pi[pos]) - log_prob_gamma

        mask_f = m.float()
        lengths = mask_f.sum(dim=1).clamp_min(1.0)

        loss_per_sample = (nll * mask_f).sum(dim=1) / lengths

        if return_per_sample:
            return loss_per_sample

        return loss_per_sample.mean()