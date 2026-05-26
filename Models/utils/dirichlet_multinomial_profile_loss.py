from __future__ import annotations

import torch
import torch.nn as nn


class DirichletMultinomialProfileLoss(nn.Module):
    """
    Dirichlet-multinomial profile loss.

    This is appropriate when the model predicts the within-transcript
    allocation profile, conditional on the observed total mass.

    Model:
        y | N, pi, kappa ~ DirichletMultinomial(N, alpha)

    with:
        pi_i    = mu_i / sum_j mu_j
        alpha_i = kappa * pi_i
        alpha_0 = sum_i alpha_i = kappa

    Interpretation:
        pi:
            predicted mean profile over codon positions.

        kappa:
            profile-level concentration / precision.

            high kappa -> close to multinomial, less overdispersion.
            low kappa  -> more overdispersion, more tolerant loss.

    Important:
        kappa is continuous positive. It is NOT an integer.

    Supported kappa shapes:
        scalar      -> same kappa for all samples
        [B]         -> one kappa per sample
        [B, 1]      -> one kappa per sample
        [B, T]      -> per-position values are pooled to [B, 1]

    If y_true is non-integer, this becomes a gamma-function extension
    / quasi-likelihood rather than an exact discrete probability model.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        kappa: float | None = 100.0,
        kappa_min: float = 1e-2,
        kappa_max: float = 1e5,
        normalize_by_total: bool = True,
        include_multinomial_constant: bool = False,
        pool_position_kappa: str = "mean",
    ) -> None:
        super().__init__()

        self.eps = float(eps)
        self.fixed_kappa = None if kappa is None else float(kappa)

        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)

        self.normalize_by_total = bool(normalize_by_total)
        self.include_multinomial_constant = bool(include_multinomial_constant)

        if pool_position_kappa not in {"mean", "median", "error"}:
            raise ValueError(
                "pool_position_kappa must be one of: 'mean', 'median', 'error'."
            )

        self.pool_position_kappa = str(pool_position_kappa)

        if not (0.0 < self.kappa_min <= self.kappa_max):
            raise ValueError("Require 0 < kappa_min <= kappa_max.")

    def _prepare_kappa(
        self,
        *,
        kappa: torch.Tensor | float | None,
        mask_f: torch.Tensor,
        B: int,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Converts kappa to shape [B, 1].

        If kappa is [B, T], we pool it over valid positions because the
        standard Dirichlet-multinomial has one concentration scalar per profile.
        """
        dtype = torch.float64

        if kappa is None:
            if self.fixed_kappa is None:
                raise ValueError(
                    "kappa was None and fixed_kappa is also None. "
                    "Pass kappa to forward or set kappa in __init__."
                )

            kappa_t = torch.full(
                (B, 1),
                float(self.fixed_kappa),
                device=device,
                dtype=dtype,
            )

            return kappa_t.clamp(self.kappa_min, self.kappa_max)

        if not torch.is_tensor(kappa):
            kappa_t = torch.tensor(
                float(kappa),
                device=device,
                dtype=dtype,
            )
        else:
            kappa_t = kappa.to(device=device, dtype=dtype)

        if kappa_t.ndim == 0:
            kappa_t = kappa_t.reshape(1, 1).expand(B, 1)

        elif kappa_t.ndim == 1:
            if kappa_t.shape[0] != B:
                raise ValueError(
                    f"Expected kappa shape [B], got {tuple(kappa_t.shape)} "
                    f"with B={B}."
                )
            kappa_t = kappa_t.reshape(B, 1)

        elif kappa_t.ndim == 2:
            if kappa_t.shape == (B, 1):
                pass

            elif kappa_t.shape == (B, T):
                if self.pool_position_kappa == "error":
                    raise ValueError(
                        "Received per-position kappa [B, T], but "
                        "pool_position_kappa='error'. For standard "
                        "Dirichlet-multinomial use one kappa per profile."
                    )

                valid_lengths = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

                if self.pool_position_kappa == "mean":
                    kappa_t = (kappa_t * mask_f).sum(dim=1, keepdim=True) / valid_lengths

                elif self.pool_position_kappa == "median":
                    # Mask invalid positions to NaN, then use nanmedian.
                    kappa_masked = torch.where(
                        mask_f.bool(),
                        kappa_t,
                        torch.full_like(kappa_t, float("nan")),
                    )
                    kappa_t = torch.nanmedian(kappa_masked, dim=1, keepdim=True).values

                    # Fallback if an entire row somehow became NaN.
                    kappa_t = torch.nan_to_num(
                        kappa_t,
                        nan=float(self.fixed_kappa if self.fixed_kappa is not None else 100.0),
                    )

            else:
                raise ValueError(
                    f"Expected kappa shape [B, 1] or [B, T], got "
                    f"{tuple(kappa_t.shape)} with B={B}, T={T}."
                )

        else:
            raise ValueError(f"Unsupported kappa shape: {tuple(kappa_t.shape)}")

        kappa_t = torch.nan_to_num(
            kappa_t,
            nan=float(self.fixed_kappa if self.fixed_kappa is not None else 100.0),
            posinf=self.kappa_max,
            neginf=self.kappa_min,
        )

        return kappa_t.clamp(self.kappa_min, self.kappa_max)

    def forward(
        self,
        *,
        mu: torch.Tensor | None = None,
        mu_phys: torch.Tensor | None = None,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        kappa: torch.Tensor | float | None = None,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            mu / mu_phys:
                Predicted nonnegative profile mean. Either name is accepted
                for compatibility with your existing Lightning code.

            y_true:
                Observed profile/counts, shape [B, T].

            mask:
                Valid-position mask, shape [B, T].

            kappa:
                Optional concentration parameter. If None, fixed_kappa from
                __init__ is used.

        Returns:
            loss scalar or loss_per_sample [B].
        """
        if mu is None:
            if mu_phys is None:
                raise ValueError("Provide either mu or mu_phys.")
            mu = mu_phys

        mask_b = mask.bool()

        with torch.amp.autocast(device_type=mu.device.type, enabled=False):
            y = y_true.to(torch.float64).clamp_min(0.0)
            mu = mu.to(torch.float64).clamp_min(0.0)

            mask_f = mask_b.to(torch.float64)

            y = y * mask_f
            mu = mu * mask_f

            B, T = mu.shape

            # ------------------------------------------------------------
            # Predicted profile probability pi
            # ------------------------------------------------------------
            mu_mass = mu.sum(dim=1, keepdim=True).clamp_min(self.eps)

            pi = mu / mu_mass
            pi = pi * mask_f

            # Avoid zero alpha on valid positions.
            pi = torch.where(
                mask_b,
                pi.clamp_min(self.eps),
                torch.zeros_like(pi),
            )

            # Renormalize after epsilon floor.
            pi = pi / (pi * mask_f).sum(dim=1, keepdim=True).clamp_min(self.eps)
            pi = pi * mask_f

            # ------------------------------------------------------------
            # Total observed mass N
            # ------------------------------------------------------------
            N = y.sum(dim=1, keepdim=True)

            # ------------------------------------------------------------
            # Kappa: one scalar per profile
            # ------------------------------------------------------------
            kappa_t = self._prepare_kappa(
                kappa=kappa,
                mask_f=mask_f,
                B=B,
                T=T,
                device=mu.device,
            )

            # ------------------------------------------------------------
            # Dirichlet parameters
            # ------------------------------------------------------------
            alpha = kappa_t * pi
            alpha = torch.where(
                mask_b,
                alpha.clamp_min(self.eps),
                torch.zeros_like(alpha),
            )

            alpha0 = alpha.sum(dim=1, keepdim=True).clamp_min(self.eps)

            # ------------------------------------------------------------
            # Dirichlet-multinomial log-probability
            #
            # log p(y | N, alpha)
            # =
            #   log Gamma(N + 1) - sum_i log Gamma(y_i + 1)
            # + log Gamma(alpha0) - log Gamma(N + alpha0)
            # + sum_i [log Gamma(y_i + alpha_i) - log Gamma(alpha_i)]
            #
            # The multinomial constant is optional because it does not depend
            # on model parameters.
            # ------------------------------------------------------------
            log_prob = (
                torch.lgamma(alpha0)
                - torch.lgamma(N + alpha0)
            )

            pos_terms = (
                torch.lgamma(y + alpha.clamp_min(self.eps))
                - torch.lgamma(alpha.clamp_min(self.eps))
            )

            pos_terms = pos_terms * mask_f
            log_prob = log_prob + pos_terms.sum(dim=1, keepdim=True)

            if self.include_multinomial_constant:
                const = (
                    torch.lgamma(N + 1.0)
                    - (torch.lgamma(y + 1.0) * mask_f).sum(dim=1, keepdim=True)
                )
                log_prob = log_prob + const

            nll = -log_prob.squeeze(1)

            # If N == 0, the profile carries no information.
            zero_mass = N.squeeze(1) <= 0.0
            nll = torch.where(zero_mass, torch.zeros_like(nll), nll)

            if self.normalize_by_total:
                nll = nll / N.squeeze(1).clamp_min(1.0)

            nll = torch.nan_to_num(
                nll,
                nan=0.0,
                posinf=1e8,
                neginf=1e8,
            )

            nll = nll.to(torch.float32)

        if return_per_sample:
            return nll

        return nll.mean()