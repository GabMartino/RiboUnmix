from __future__ import annotations

import torch
import torch.nn as nn


class NBVarianceQuasiLoss(nn.Module):
    """
    Continuous NB-like quasi-likelihood.

    Variance:
        Var[Y_i] = mu_i + alpha_i * mu_i^2

    This is not a discrete NB/ZINB likelihood.
    It is a continuous heteroscedastic loss with NB-like variance.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        alpha_min: float = 1e-4,
        alpha_max: float = 100.0,
        zero_censor_to_zero: bool = True,
        censor_threshold: float = 0.0,
    ):
        super().__init__()

        self.eps = float(eps)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.zero_censor_to_zero = bool(zero_censor_to_zero)
        self.censor_threshold = float(censor_threshold)

    def forward(
        self,
        *,
        mu_phys: torch.Tensor,
        alpha: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        m = mask.bool()

        with torch.amp.autocast(device_type=mu_phys.device.type, enabled=False):
            y = y_true.to(torch.float64).clamp_min(0.0)
            mu = mu_phys.to(torch.float64).clamp(min=1e-6, max=1e8)
            alpha = alpha.to(torch.float64).clamp(self.alpha_min, self.alpha_max)

            if self.zero_censor_to_zero and self.censor_threshold > 0.0:
                y = torch.where(y <= self.censor_threshold, torch.zeros_like(y), y)

            var = mu + alpha * mu.pow(2)
            var = var.clamp_min(self.eps)

            resid2 = (y - mu).pow(2)

            loss_pos = 0.5 * (
                resid2 / var
                + torch.log1p(var)
            )

            loss_pos = torch.nan_to_num(
                loss_pos,
                nan=0.0,
                posinf=1e8,
                neginf=1e8,
            )

            mask_f = m.to(torch.float64)
            valid_lengths = mask_f.sum(dim=1).clamp_min(1.0)

            loss_per_sample = (loss_pos * mask_f).sum(dim=1) / valid_lengths
            loss_per_sample = loss_per_sample.to(torch.float32)

        if return_per_sample:
            return loss_per_sample

        return loss_per_sample.mean()