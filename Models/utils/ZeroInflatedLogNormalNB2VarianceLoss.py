from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn


class ZeroInflatedLogNormalNB2Loss(nn.Module):
    """
    Simplified zero-inflated log-normal NLL with positive-component NB2 variance.

    Model:

        P(Y = 0) = pi_zero

        Y | Y > 0 ~ LogNormal(log_loc, log_sigma)

    Mean convention:

        mu = E[Y]

    Therefore:

        mu_pos = E[Y | Y > 0] = mu / (1 - pi_zero)

    Positive-component variance:

        Var[Y | Y > 0] = mu_pos + phi * mu_pos^2

    Conversion to log-normal parameters:

        log_sigma^2 = log(1 + Var[Y | Y > 0] / mu_pos^2)

        log_loc = log(mu_pos) - 0.5 * log_sigma^2

    Recommended for your current model:

        mu = rho * beta

    Use a small censor_threshold if you want zeros/near-zeros to give gradient
    to mu through the log-normal CDF.
    """

    def __init__(
        self,
        eps: float = 1.0e-8,
        mu_min: float = 1.0e-8,
        mu_max: float = 1.0e8,
        phi_min: float = 1.0e-4,
        phi_max: float = 10.0,
        log_sigma_min: float = 0.05,
        log_sigma_max: float = 2.0,
        censor_threshold: float = 0.0,
        zero_input_is_logits: bool = True,
        phi_input_is_log: bool = False,
    ):
        super().__init__()

        self.eps = float(eps)

        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)

        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)

        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)

        self.censor_threshold = float(censor_threshold)
        self.zero_input_is_logits = bool(zero_input_is_logits)
        self.phi_input_is_log = bool(phi_input_is_log)

        if not (0.0 < self.mu_min <= self.mu_max):
            raise ValueError("Require 0 < mu_min <= mu_max.")

        if not (0.0 < self.phi_min <= self.phi_max):
            raise ValueError("Require 0 < phi_min <= phi_max.")

        if not (0.0 < self.log_sigma_min <= self.log_sigma_max):
            raise ValueError("Require 0 < log_sigma_min <= log_sigma_max.")

    @staticmethod
    def _log_normal_cdf_standard(z: torch.Tensor) -> torch.Tensor:
        if hasattr(torch.special, "log_ndtr"):
            return torch.special.log_ndtr(z)

        return torch.log(
            0.5 * torch.erfc(-z / math.sqrt(2.0))
        ).clamp_min(-1.0e30)

    def zero_prob_from_param(self, zero_param: torch.Tensor) -> torch.Tensor:
        if self.zero_input_is_logits:
            pi = torch.sigmoid(zero_param)
        else:
            pi = zero_param

        return pi.clamp(min=self.eps, max=1.0 - self.eps)

    def _prepare_phi(self, phi: torch.Tensor) -> torch.Tensor:
        if self.phi_input_is_log:
            phi = torch.exp(phi)

        return phi.clamp(min=self.phi_min, max=self.phi_max)

    def _lognormal_params(
        self,
        *,
        mu: torch.Tensor,
        pi_zero: torch.Tensor,
        phi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one_minus_pi = (1.0 - pi_zero).clamp_min(self.eps)

        # mu is E[Y], so the positive-component mean is larger.
        mu_pos = mu / one_minus_pi
        mu_pos = mu_pos.clamp(min=self.mu_min, max=self.mu_max)

        # Positive-component NB2 variance.
        var_pos = mu_pos + phi * mu_pos.pow(2)
        var_pos = var_pos.clamp_min(self.eps)

        log_sigma2 = torch.log1p(
            var_pos / mu_pos.pow(2).clamp_min(self.eps)
        )

        log_sigma = torch.sqrt(log_sigma2.clamp_min(self.eps))
        log_sigma = log_sigma.clamp(
            min=self.log_sigma_min,
            max=self.log_sigma_max,
        )

        log_loc = torch.log(mu_pos.clamp_min(self.eps)) - 0.5 * log_sigma.pow(2)

        return log_loc, log_sigma

    def forward(
        self,
        mu_phys: torch.Tensor,
        zero_param: torch.Tensor,
        phi: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        if mu_phys.shape != y_true.shape:
            raise ValueError(
                f"mu_phys and y_true must have same shape. "
                f"Got mu_phys={tuple(mu_phys.shape)}, y_true={tuple(y_true.shape)}."
            )

        if mask.shape != y_true.shape:
            raise ValueError(
                f"mask and y_true must have same shape. "
                f"Got mask={tuple(mask.shape)}, y_true={tuple(y_true.shape)}."
            )

        with torch.amp.autocast(device_type=mu_phys.device.type, enabled=False):
            y = y_true.to(torch.float64).clamp_min(0.0)

            mu = mu_phys.to(torch.float64).clamp(
                min=self.mu_min,
                max=self.mu_max,
            )

            pi_zero = self.zero_prob_from_param(
                zero_param.to(torch.float64)
            )

            phi_t = self._prepare_phi(
                phi.to(torch.float64)
            )

            pi_zero = torch.broadcast_to(pi_zero, y.shape)
            phi_t = torch.broadcast_to(phi_t, y.shape)

            log_loc, log_sigma = self._lognormal_params(
                mu=mu,
                pi_zero=pi_zero,
                phi=phi_t,
            )

            one_minus_pi = (1.0 - pi_zero).clamp_min(self.eps)

            # --------------------------------------------------------
            # Positive branch:
            #
            # -log[(1 - pi) LogNormal(y)]
            # --------------------------------------------------------
            log_y = torch.log(y.clamp_min(self.eps))
            z = (log_y - log_loc) / log_sigma

            positive_nll = (
                -torch.log(one_minus_pi)
                + log_y
                + torch.log(log_sigma.clamp_min(self.eps))
                + 0.5 * math.log(2.0 * math.pi)
                + 0.5 * z.pow(2)
            )

            # --------------------------------------------------------
            # Zero / censored-zero branch
            # --------------------------------------------------------
            if self.censor_threshold > 0.0:
                c = torch.tensor(
                    self.censor_threshold,
                    device=y.device,
                    dtype=torch.float64,
                ).clamp_min(self.eps)

                z_c = (torch.log(c) - log_loc) / log_sigma
                log_cdf = self._log_normal_cdf_standard(z_c)

                # log[pi + (1 - pi) F(c)]
                log_zero_mass = torch.logaddexp(
                    torch.log(pi_zero.clamp_min(self.eps)),
                    torch.log(one_minus_pi) + log_cdf,
                )

                zero_nll = -log_zero_mass
                is_zero = y <= self.censor_threshold

            else:
                zero_nll = -torch.log(pi_zero.clamp_min(self.eps))
                is_zero = y <= 0.0

            nll = torch.where(is_zero, zero_nll, positive_nll)

            nan_count = int(torch.isnan(nll).sum().item())
            if nan_count > 0:
                warnings.warn(
                    f"ZILNLoss: {nan_count} NaN(s) in NLL before masking — replacing with 0.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            nll = torch.nan_to_num(
                nll,
                nan=0.0,
                posinf=1.0e8,
                neginf=1.0e8,
            )

            mask_f = mask.bool().to(torch.float64)
            valid_len = mask_f.sum(dim=1).clamp_min(1.0)

            loss_per_sample = (nll * mask_f).sum(dim=1) / valid_len
            loss_per_sample = loss_per_sample.to(torch.float32)

        if return_per_sample:
            return loss_per_sample

        return loss_per_sample.mean()