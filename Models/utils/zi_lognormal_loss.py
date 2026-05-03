from __future__ import annotations

import math

import torch
import torch.nn as nn

class ScaledZeroInflatedLogNormalLoss(nn.Module):
    def __init__(self, censor_threshold: float = 0.5, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.censor_threshold = censor_threshold
        self.register_buffer("half_log_2pi", torch.tensor(0.5 * math.log(2.0 * math.pi)))
        self.register_buffer("sqrt2", torch.tensor(math.sqrt(2.0)))

    def forward(
        self,
        mu_phys: torch.Tensor,
        pi: torch.Tensor,
        sigma: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:

        y = y_true.to(torch.float32)
        m = mask.bool()
        eps = self.eps

        pi = pi.clamp(eps, 1.0 - eps)
        mu = mu_phys.clamp(min=eps, max=1e8)
        sigma = sigma.clamp_min(eps)

        mu_stat = torch.log(mu)

        delta = y.new_tensor(self.censor_threshold)
        log_delta = torch.log(delta.clamp_min(eps))

        low = m & (y <= delta)
        pos = m & (y > delta)

        nll = torch.zeros_like(y)

        if low.any():
            z = (log_delta - mu_stat[low]) / sigma[low]
            logcdf = self._log_ndtr(z)
            log_pi = torch.log(pi[low])
            log_1m_pi = torch.log1p(-pi[low])
            nll[low] = -torch.logaddexp(log_pi, log_1m_pi + logcdf)

        if pos.any():
            y_pos = y[pos].clamp_min(eps)
            logy = torch.log(y_pos)
            z = (logy - mu_stat[pos]) / sigma[pos]

            nll_ln = (
                logy
                + torch.log(sigma[pos])
                + self.half_log_2pi.to(dtype=logy.dtype, device=logy.device)
                + 0.5 * (z ** 2)
            )
            nll[pos] = -torch.log1p(-pi[pos]) + nll_ln

        mask_f = m.to(dtype=y.dtype)
        L = mask_f.sum(dim=1).clamp_min(1.0)

        loss_per_sample = (nll * mask_f).sum(dim=1).div(L)  # [B]

        if return_per_sample:
            return loss_per_sample

        return loss_per_sample.mean()
    def _log_ndtr(self, z: torch.Tensor) -> torch.Tensor:
        """Stable log Phi(z)."""
        if hasattr(torch.special, "log_ndtr"):
            return torch.special.log_ndtr(z)

        # fallback: log(0.5*erfc(-z/sqrt(2)))
        return torch.log(
            0.5
            * torch.erfc(-z / self.sqrt2.to(dtype=z.dtype, device=z.device)).clamp_min(self.eps)
        )

