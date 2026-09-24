"""Stable scalar/profile algebra for AMP. Neural layers keep their AMP policy.

No clipping of finite model scores, surrogate gradients, or nan_to_num repairs.
FP32 is the reduction dtype under AMP; FP64 is retained for reference tests.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NUMERICAL_FORMULATION_VERSION = "log-space-nb2-v1"


def working_float(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float64 else x.float()


def require_finite(x: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(f"Non-finite {label}; refusing to replace it with a finite value.")


def log_softplus(x: torch.Tensor) -> torch.Tensor:
    x = working_float(x)
    # Below this threshold log(softplus(x)) = x to working-dtype accuracy.
    # Both where branches must be safe: log(softplus(-1000)) is not.
    cutoff = -36.0 if x.dtype == torch.float64 else -20.0
    return torch.where(x < cutoff, x, F.softplus(x.clamp_min(cutoff)).log())


def masked_logmeanexp(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = working_float(x)
    mask = mask.bool()
    count = mask.sum(dim=-1, keepdim=True)
    masked = torch.where(mask, x, -torch.inf)
    # All-padding rows must not create logsumexp(-inf, ...)'s NaN backward.
    masked = torch.where(count > 0, masked, torch.zeros_like(masked))
    value = torch.logsumexp(masked, dim=-1, keepdim=True) - count.clamp_min(1).to(x.dtype).log()
    return torch.where(count > 0, value, torch.zeros_like(value))


def masked_mean(x: torch.Tensor, mask: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    x = working_float(x)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(x.dtype)
    # Divide before reducing: the sum can overflow even when the mean fits.
    return (torch.where(mask.bool(), x, 0.0) / count).sum(dim=-1, keepdim=keepdim)


def nb2_nll_from_log_mean(y: torch.Tensor, log_mu: torch.Tensor,
                         log_alpha: torch.Tensor) -> torch.Tensor:
    """NB2 NLL, including the continuous-target gamma normalization.

    r=exp(-log_alpha), z=log_mu+log_alpha:
      lgamma(r) + lgamma(y+1)-lgamma(y+r) + r*softplus(z) + y*softplus(-z).
    Never materializes mu or alpha*mu, nor subtracts log(r+mu)-log(mu).
    """
    y, log_mu, log_alpha = map(working_float, (y, log_mu, log_alpha))
    r = (-log_alpha).exp()
    z = log_mu + log_alpha
    # For large targets, direct lgamma subtraction loses the entire ratio.
    # Stirling difference with log1p(delta/a), through the a^-7 term. At
    # a>=1e4 and bounded NB dispersion its error is below FP64 precision.
    large = y >= 1.0e4
    small_y = torch.where(large, torch.zeros_like(y), y)
    direct = torch.lgamma(small_y + 1.0) - torch.lgamma(small_y + r)
    a = torch.where(large, y, torch.full_like(y, 1.0e4)) + 1.0
    delta = r - 1.0
    b = a + delta

    def correction(t):
        inv = t.reciprocal()
        return inv / 12.0 - inv.pow(3) / 360.0 + inv.pow(5) / 1260.0 - inv.pow(7) / 1680.0

    ratio = (-delta * a.log() - (b - 0.5) * torch.log1p(delta / a)
             + delta + correction(a) - correction(b))
    return torch.lgamma(r) + torch.where(large, ratio, direct) + r * F.softplus(z) + y * F.softplus(-z)


def nb_vst(x: torch.Tensor, alpha: torch.Tensor, eps: float) -> torch.Tensor:
    """2/sqrt(alpha+eps) * asinh(sqrt(alpha*x+eps)), without alpha*x.

    If u=log(alpha*x+eps), asinh(exp(u/2)) =
    logaddexp(u/2, softplus(u)/2). Zero x retains its finite derivative.
    """
    x, alpha = working_float(x), working_float(alpha)
    u = alpha.log() + (x + eps / alpha).log()
    asinh_sqrt = torch.logaddexp(0.5 * u, 0.5 * F.softplus(u))
    return (2.0 / (alpha + eps).sqrt()) * asinh_sqrt


def standardized_profile(x: torch.Tensor, mask: torch.Tensor, eps: float):
    """(x-mean)/sqrt(var+eps), never squares unscaled profile amplitudes."""
    x = working_float(x)
    # Subtract a reference first so a large constant offset does not swallow
    # small profile differences or produce a spurious nonzero flat variance.
    anchor = x.gather(-1, mask.to(torch.int64).argmax(dim=-1, keepdim=True))
    shifted = torch.where(mask, x - anchor, 0.0)
    centered = torch.where(mask, shifted - masked_mean(shifted, mask), 0.0)
    scale = centered.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    variance_scaled = masked_mean((centered / scale).square(), mask)
    log_variance = torch.where(
        variance_scaled > 0,
        variance_scaled.clamp_min(torch.finfo(x.dtype).tiny).log() + 2.0 * scale.log(),
        -torch.inf,
    )
    log_std = 0.5 * torch.logaddexp(log_variance, x.new_tensor(eps).log())
    if eps == 0:  # unregularized reporting PCC: give constant rows a unit std
        log_std = torch.where(torch.isfinite(log_std), log_std, 0.0)
    return centered / log_std.exp(), log_variance.squeeze(-1)


@torch.no_grad()
def clip_grad_norm_stable(parameters, max_norm: float) -> None:
    """The usual global L2 clipping coefficient, without overflowing sum(g^2).

    Used after the existing finite-gradient guard, never to repair NaN/Inf.
    Scaling the reductions does not clip individual coordinates or gate states.
    """
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return
    scale = torch.stack([g.detach().abs().amax() for g in gradients]).amax().clamp_min(1.0)
    norm_scaled = torch.stack([(working_float(g) / scale).square().sum() for g in gradients]).sum().sqrt()
    coefficient = ((max_norm / scale) / (norm_scaled + 1e-6 / scale)).clamp_max(1.0)
    require_finite(coefficient, "global gradient clipping coefficient")
    for gradient in gradients:
        gradient.mul_(coefficient.to(gradient.dtype))
