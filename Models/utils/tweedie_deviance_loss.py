from __future__ import annotations

import torch
import torch.nn as nn


class TweedieDevianceLoss(nn.Module):
    """
    Tweedie deviance loss for nonnegative continuous targets.

    Uses:

        E[Y] = mu
        Var[Y] = phi * mu^p

    For 1 < p < 2, Tweedie supports exact zeros and continuous positive values.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        phi_min: float = 1e-4,
        phi_max: float = 10.0,
        censor_threshold: float = 0.0,
        zero_censor_to_zero: bool = True,
        include_log_phi: bool = True,
    ):
        super().__init__()

        self.eps = float(eps)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.censor_threshold = float(censor_threshold)
        self.zero_censor_to_zero = bool(zero_censor_to_zero)
        self.include_log_phi = bool(include_log_phi)

        if not (0.0 < self.phi_min <= self.phi_max):
            raise ValueError("Require 0 < phi_min <= phi_max.")

    def forward(
        self,
        mu_phys: torch.Tensor,
        power: torch.Tensor | float,
        phi: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        m = mask.bool()

        with torch.amp.autocast(device_type=mu_phys.device.type, enabled=False):
            y = y_true.float().clamp_min(0.0)
            mu = mu_phys.float().clamp(min=self.eps, max=1e8)
            phi = phi.float().clamp(min=self.phi_min, max=self.phi_max)

            p = torch.as_tensor(power, device=mu.device, dtype=mu.dtype)
            p = p.clamp(min=1.0001, max=1.9999)

            if self.zero_censor_to_zero and self.censor_threshold > 0.0:
                y = torch.where(
                    y <= self.censor_threshold,
                    torch.zeros_like(y),
                    y,
                )

            term_y = torch.pow(y, 2.0 - p) / ((1.0 - p) * (2.0 - p))
            term_cross = -y * torch.pow(mu, 1.0 - p) / (1.0 - p)
            term_mu = torch.pow(mu, 2.0 - p) / (2.0 - p)

            deviance = 2.0 * (term_y + term_cross + term_mu)

            deviance = torch.nan_to_num(
                deviance,
                nan=0.0,
                posinf=1e8,
                neginf=1e8,
            ).clamp_min(0.0)

            nll = deviance / (2.0 * phi)

            if self.include_log_phi:
                nll = nll + 0.5 * torch.log(phi.clamp_min(self.eps))

            mask_f = m.float()
            valid_lengths = mask_f.sum(dim=1).clamp_min(1.0)

            loss_per_sample = (nll * mask_f).sum(dim=1) / valid_lengths

        if return_per_sample:
            return loss_per_sample

        return loss_per_sample.mean()