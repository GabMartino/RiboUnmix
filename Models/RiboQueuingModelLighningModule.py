from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import lightning as pl
import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")

from matplotlib import pyplot as plt


# ============================================================
# Loss
# ============================================================

class ZeroInflatedLogNormalNB2Loss(nn.Module):
    """
    Simplified zero-inflated log-normal NLL with positive-component NB2 variance.

    Model:

        P(Y = 0) = pi_zero

        Y | Y > 0 ~ LogNormal(log_loc, log_sigma)

    Mu convention:

        positive_mean:

            mu = E[Y | Y > 0]

        unconditional_mean:

            mu = E[Y]
            E[Y | Y > 0] = mu / (1 - pi_zero)

        median:

            mu = median[Y | Y > 0] = exp(log_loc)

    Positive-component variance target:

        Var[Y | Y > 0] = E[Y | Y > 0] + phi * E[Y | Y > 0]^2

    Conversion:

        For mean parameterizations:
            log_sigma^2 = log(1 + Var / E[Y | Y > 0]^2)
            log_loc = log(E[Y | Y > 0]) - 0.5 * log_sigma^2

        For median parameterization:
            solve r^3 - (1 + phi) * r - 1 / median = 0,
            where r = exp(log_sigma^2 / 2), then
            log_sigma^2 = 2 * log(r)
            log_loc = log(mu)
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
        detach_mu_for_zero_branch: bool = True,
        mu_input_is_positive_mean: bool = True,
        mu_parameterization: str | None = None,
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
        self.detach_mu_for_zero_branch = bool(detach_mu_for_zero_branch)
        self.mu_input_is_positive_mean = bool(mu_input_is_positive_mean)
        if mu_parameterization is None:
            mu_parameterization = (
                "positive_mean"
                if self.mu_input_is_positive_mean
                else "unconditional_mean"
            )
        self.mu_parameterization = str(mu_parameterization)
        valid_parameterizations = {
            "positive_mean",
            "unconditional_mean",
            "median",
        }
        if self.mu_parameterization not in valid_parameterizations:
            raise ValueError(
                "mu_parameterization must be one of "
                f"{sorted(valid_parameterizations)}, got {self.mu_parameterization!r}."
            )

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
        if self.mu_parameterization == "unconditional_mean":
            one_minus_pi = (1.0 - pi_zero).clamp_min(self.eps)
            mu_ref = mu / one_minus_pi
        else:
            mu_ref = mu

        mu_ref = mu_ref.clamp(min=self.mu_min, max=self.mu_max)

        if self.mu_parameterization == "median":
            log_sigma2 = self._median_log_sigma2_from_nb2_dispersion(
                median=mu_ref,
                phi=phi,
            )
        else:
            var_pos = mu_ref + phi * mu_ref.pow(2)
            var_pos = var_pos.clamp_min(self.eps)
            log_sigma2 = torch.log1p(
                var_pos / mu_ref.pow(2).clamp_min(self.eps)
            )

        log_sigma = torch.sqrt(log_sigma2.clamp_min(self.eps))
        log_sigma = log_sigma.clamp(
            min=self.log_sigma_min,
            max=self.log_sigma_max,
        )

        if self.mu_parameterization == "median":
            log_loc = torch.log(mu_ref.clamp_min(self.eps))
        else:
            log_loc = torch.log(mu_ref.clamp_min(self.eps)) - 0.5 * log_sigma.pow(2)

        return log_loc, log_sigma

    def _median_log_sigma2_from_nb2_dispersion(
        self,
        *,
        median: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """Convert NB2 mean-variance dispersion to log-normal sigma^2.

        For median m and r = exp(sigma^2 / 2), the positive mean is m * r.
        Equating log-normal variance to mean + phi * mean^2 yields:

            r^3 - (1 + phi) * r - 1 / m = 0
        """
        a = 1.0 + phi.clamp_min(self.eps)
        b = 1.0 / median.clamp_min(self.eps)

        lower = torch.sqrt(a).clamp_min(1.0 + self.eps)
        small_b_approx = lower + b / (2.0 * a.clamp_min(self.eps))
        large_b_approx = b.clamp_min(self.eps).pow(1.0 / 3.0)
        r = torch.maximum(small_b_approx, large_b_approx).clamp_min(lower)

        for _ in range(8):
            f = r.pow(3) - a * r - b
            fp = (3.0 * r.pow(2) - a).clamp_min(self.eps)
            r = (r - f / fp).clamp_min(lower)

        return 2.0 * torch.log(r.clamp_min(1.0 + self.eps))

    def positive_mean_from_params(
        self,
        *,
        mu: torch.Tensor,
        pi_zero: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        log_loc, log_sigma = self._lognormal_params(
            mu=mu,
            pi_zero=pi_zero,
            phi=phi,
        )
        mean = torch.exp(log_loc + 0.5 * log_sigma.pow(2))
        return mean.clamp(min=self.mu_min, max=self.mu_max)

    def forward(
        self,
        mu_phys: torch.Tensor,
        zero_param: torch.Tensor,
        phi: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
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

            # Positive log-normal branch.
            log_y = torch.log(y.clamp_min(self.eps))
            z = (log_y - log_loc) / log_sigma

            positive_nll = (
                -torch.log(one_minus_pi)
                + log_y
                + torch.log(log_sigma.clamp_min(self.eps))
                + 0.5 * math.log(2.0 * math.pi)
                + 0.5 * z.pow(2)
            )

            # Zero / censored-zero branch.
            if self.censor_threshold > 0.0:
                c = torch.tensor(
                    self.censor_threshold,
                    device=y.device,
                    dtype=torch.float64,
                ).clamp_min(self.eps)

                if self.detach_mu_for_zero_branch:
                    # Dataset zeros train the dropout/zero branch, not the
                    # biological queueing support or visibility mean. Detach pi
                    # only inside the log-normal CDF parameters; the mixture
                    # probability below still learns from zeros.
                    log_loc_zero, log_sigma_zero = self._lognormal_params(
                        mu=mu.detach(),
                        pi_zero=pi_zero.detach(),
                        phi=phi_t.detach(),
                    )
                else:
                    log_loc_zero = log_loc
                    log_sigma_zero = log_sigma

                z_c = (torch.log(c) - log_loc_zero) / log_sigma_zero
                log_cdf = self._log_normal_cdf_standard(z_c)
                if self.detach_mu_for_zero_branch:
                    log_cdf = log_cdf.detach()

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


class ZeroInflatedNegativeBinomialLoss(nn.Module):
    """
    Zero-inflated NB2 NLL.

    Positive component:

        Y | not_dropout ~ NB2(mean=mu_pos, var=mu_pos + phi * mu_pos^2)

    Zero-inflated mixture:

        P(Y=0) = pi_zero + (1 - pi_zero) * NB2(Y=0)
        P(Y>0) = (1 - pi_zero) * NB2(Y)

    target_transform controls how non-integer targets are handled:

        real : gamma-function continuation of NB to non-negative real values.
        floor : floor target before NB.
        round : round target before NB.
    """

    def __init__(
        self,
        eps: float = 1.0e-8,
        mu_min: float = 1.0e-8,
        mu_max: float = 1.0e8,
        phi_min: float = 1.0e-4,
        phi_max: float = 10.0,
        zero_input_is_logits: bool = True,
        phi_input_is_log: bool = False,
        detach_mu_for_zero_branch: bool = True,
        mu_input_is_positive_mean: bool = True,
        target_transform: str = "real",
        zero_threshold: float = 0.0,
    ):
        super().__init__()

        self.eps = float(eps)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.zero_input_is_logits = bool(zero_input_is_logits)
        self.phi_input_is_log = bool(phi_input_is_log)
        self.detach_mu_for_zero_branch = bool(detach_mu_for_zero_branch)
        self.mu_input_is_positive_mean = bool(mu_input_is_positive_mean)
        self.target_transform = str(target_transform).lower()
        self.zero_threshold = float(zero_threshold)

        allowed = {"real", "floor", "round"}
        if self.target_transform not in allowed:
            raise ValueError(
                f"target_transform must be one of {sorted(allowed)}, "
                f"got {self.target_transform!r}."
            )

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

    def _prepare_target(self, y: torch.Tensor) -> torch.Tensor:
        y = y.clamp_min(0.0)

        if self.target_transform == "floor":
            return torch.floor(y)

        if self.target_transform == "round":
            return torch.round(y)

        return y

    def _positive_mean(
        self,
        *,
        mu: torch.Tensor,
        pi_zero: torch.Tensor,
    ) -> torch.Tensor:
        if self.mu_input_is_positive_mean:
            mu_pos = mu
        else:
            mu_pos = mu / (1.0 - pi_zero).clamp_min(self.eps)

        return mu_pos.clamp(min=self.mu_min, max=self.mu_max)

    def _nb_log_prob(
        self,
        *,
        y: torch.Tensor,
        mu: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        mu = mu.clamp(min=self.mu_min, max=self.mu_max)
        phi = phi.clamp(min=self.phi_min, max=self.phi_max)

        size = (1.0 / phi).clamp_min(self.eps)
        log_total = torch.log(size + mu)

        return (
            torch.lgamma(y + size)
            - torch.lgamma(size)
            - torch.lgamma(y + 1.0)
            + size * (torch.log(size.clamp_min(self.eps)) - log_total)
            + y * (torch.log(mu.clamp_min(self.eps)) - log_total)
        )

    def forward(
        self,
        mu_phys: torch.Tensor,
        zero_param: torch.Tensor,
        phi: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=mu_phys.device.type, enabled=False):
            y = self._prepare_target(y_true.to(torch.float64))

            mu = mu_phys.to(torch.float64).clamp(
                min=self.mu_min,
                max=self.mu_max,
            )
            pi_zero = self.zero_prob_from_param(zero_param.to(torch.float64))
            phi_t = self._prepare_phi(phi.to(torch.float64))

            pi_zero = torch.broadcast_to(pi_zero, y.shape)
            phi_t = torch.broadcast_to(phi_t, y.shape)

            mu_pos = self._positive_mean(mu=mu, pi_zero=pi_zero)
            one_minus_pi = (1.0 - pi_zero).clamp_min(self.eps)

            log_nb_y = self._nb_log_prob(y=y, mu=mu_pos, phi=phi_t)
            positive_nll = -(torch.log(one_minus_pi) + log_nb_y)

            if self.detach_mu_for_zero_branch:
                mu_zero = mu_pos.detach()
                phi_zero = phi_t.detach()
            else:
                mu_zero = mu_pos
                phi_zero = phi_t

            y_zero = torch.zeros_like(y)
            log_nb_zero = self._nb_log_prob(y=y_zero, mu=mu_zero, phi=phi_zero)
            if self.detach_mu_for_zero_branch:
                log_nb_zero = log_nb_zero.detach()

            log_zero_mass = torch.logaddexp(
                torch.log(pi_zero.clamp_min(self.eps)),
                torch.log(one_minus_pi) + log_nb_zero,
            )
            zero_nll = -log_zero_mass

            is_zero = y <= self.zero_threshold
            nll = torch.where(is_zero, zero_nll, positive_nll)
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


# ============================================================
# Lightning module
# ============================================================

class RiboQueuingModelLightningModule(pl.LightningModule):
    """
    Minimal LightningModule for:

        mu = S_dt * support * beta

    where support is rho_bio or q_bio depending on loss.pcc_target.
    """

    DATASET_DIAGNOSTIC_KEYS = (
        "rho_mean",
        "q_mean",
        "q_max",
        "q_zero_frac",
        "queue_alpha",
        "queue_propagation_enabled",
        "queue_alpha_trainable",
        "p_bio_mean",
        "p_visible_mean",
        "beta_mean",
        "mu_mass",
        "lambda_mass",
        "mu_unconditional_mass",
        "dropout_prob_mean",
        "visibility_log_abs_mean",
        "visibility_log_q_weighted_mean",
        "visibility_log_center_abs",
        "rho_zero_frac",
        "beta_zero_frac",
        "mu_zero_frac",
        "J",
        "scale_dt",
        "log_scale_dt",
        "dataset_scale",
        "transcript_scale",
    )

    # Scalar objective/diagnostic metrics. _log_stage logs each only when
    # present, so pcc_only mode does not require likelihood metrics.
    OPTIONAL_SCALAR_METRICS = (
        "nll",
        "pcc_loss",
        "mu_unconditional_pcc_loss",
        "mu_mse",
        "mu_unconditional_mse",
        "mu_log1p_mse",
        "mu_unconditional_log1p_mse",
        "ziln_mean_pcc",
        "ziln_unconditional_mean_pcc",
        "ziln_mean_log1p_mse",
        "ziln_unconditional_mean_log1p_mse",
        "target_zero_frac",
        "zero_prob_on_zero",
        "zero_prob_on_positive",
        "zero_prob_gap",
        "j_centering_loss",
    )

    PROFILE_PLOT_GROUPS = (
        ("biology/support", ("rho_bio", "q_bio", "p_visible")),
        ("dropout", ("zero_prob",)),
        ("visibility", ("obs_beta", "log_visibility_bias")),
        (
            "mean/scale",
            ("mu_bio", "lambda_pre_dropout", "mu_positive", "mu_unconditional"),
        ),
        (
            "likelihood",
            (
                "phi",
                "likelihood_sd",
                "likelihood_shape",
                "likelihood_positive_mean",
                "likelihood_unconditional_mean",
            ),
        ),
    )

    def __init__(
        self,
        torch_model: nn.Module,
        config: Any,
        dataset_encoding: dict,
    ):
        super().__init__()

        self.save_hyperparameters(
            ignore=["torch_model", "config", "dataset_encoding"]
        )

        self.model = torch_model
        self.config = config
        self.loss_fn = self._build_loss()

        self.dataset_id_to_name = {int(v): str(k) for k, v in dataset_encoding.items()}
        self.use_cagrad = self.config.optim.use_cagrad
        self.automatic_optimization = not self.use_cagrad
        self._val_profile_plot_logged_this_epoch = False

    def _build_loss(self) -> nn.Module:
        loss_cfg = self.config.loss

        common_kwargs = {
            "eps": loss_cfg.eps,
            "mu_min": loss_cfg.mu_min,
            "mu_max": loss_cfg.mu_max,
            "phi_min": loss_cfg.phi_min,
            "phi_max": loss_cfg.phi_max,
            "zero_input_is_logits": loss_cfg.zero_input_is_logits,
            "phi_input_is_log": loss_cfg.phi_input_is_log,
            "detach_mu_for_zero_branch": loss_cfg.detach_mu_for_zero_branch,
            # mu_input_is_positive_mean is not read from config: for ZILN the active
            # parameterization is set by ziln_mu_parameterization (explicit overrides
            # the bool default); for ZINB the code default (True) is correct since
            # mu = scale_dt * support * beta is always the pre-dropout positive mean.
            "mu_input_is_positive_mean": True,
        }

        loss_builders = {
            "ziln": lambda: ZeroInflatedLogNormalNB2Loss(
                **common_kwargs,
                mu_parameterization=getattr(
                    loss_cfg,
                    "ziln_mu_parameterization",
                    None,
                ),
                log_sigma_min=loss_cfg.log_sigma_min,
                log_sigma_max=loss_cfg.log_sigma_max,
                censor_threshold=loss_cfg.censor_threshold,
            ),
            "zinb": lambda: ZeroInflatedNegativeBinomialLoss(
                **common_kwargs,
                target_transform=loss_cfg.zinb_target_transform,
                zero_threshold=loss_cfg.zinb_zero_threshold,
            ),
        }

        return loss_builders[loss_cfg.profile_likelihood]()

    # ============================================================
    # Forward / batch handling
    # ============================================================

    def _forward_batch(self, batch) -> dict[str, Any]:
        (
            dataset_ids,
            ids,
            seq_packed,
            target,
            lengths,
            mask,
            codon_ids,
            css,
        ) = batch

        mu, phi, extras = self.model(
            x_packed=seq_packed,
            codon_ids=codon_ids,
            id_datasets=dataset_ids,
            mask=mask,
        )

        return {
            "dataset_ids": dataset_ids,
            "ids": ids,
            "lengths": lengths,
            "mask": mask.bool(),
            "target": target,
            "codon_ids": codon_ids,
            "css": css,
            "mu": mu,
            "phi": phi,
            "extras": extras,
        }

    def forward_batch(self, batch) -> dict[str, Any]:
        return self._forward_batch(batch)

    # ============================================================
    # Zero inflation
    # ============================================================

    def _zero_param_from_out(self, out: dict[str, Any]) -> torch.Tensor:
        """
        Priority:
            1. loss.zero_fixed_prob, if provided.
            2. extras["zero_logits"]
        """
        target = out["target"]
        mask = out["mask"].bool()
        extras = out["extras"]

        dtype = target.dtype
        device = target.device

        fixed_prob = self.config.loss.zero_fixed_prob

        if fixed_prob is not None:
            p = float(fixed_prob)
            p = min(max(p, 1.0e-6), 1.0 - 1.0e-6)

            prob = torch.full_like(
                target,
                fill_value=p,
                dtype=dtype,
                device=device,
            )

            prob = torch.where(
                mask,
                prob,
                torch.full_like(prob, 1.0e-6),
            )

            if self.config.loss.zero_input_is_logits:
                return torch.logit(prob.clamp(1.0e-6, 1.0 - 1.0e-6))

            return prob

        zero_logits = extras["zero_logits"].to(device=device, dtype=dtype)

        if self.config.loss.zero_input_is_logits:
            return zero_logits

        return torch.sigmoid(zero_logits)

    # ============================================================
    # Loss / metrics
    # ============================================================

    def _profile_nll_per_sample(self, out: dict[str, Any]) -> torch.Tensor:
        zero_param = self._zero_param_from_out(out)

        return self.loss_fn(
            mu_phys=out["mu"].float(),
            zero_param=zero_param.float(),
            phi=out["phi"].float(),
            y_true=out["target"].float(),
            mask=out["mask"].bool(),
            return_per_sample=True,
        )

    def _aggregate_per_sample(
        self,
        values: torch.Tensor,
        dataset_ids: torch.Tensor,
        _ds_split: "tuple[torch.Tensor, int] | None" = None,
    ) -> torch.Tensor:
        if not self.config.loss.dataset_balanced_loss:
            return values.mean()

        if _ds_split is None:
            _, inverse = torch.unique(
                dataset_ids.to(device=values.device), return_inverse=True
            )
            K = int(inverse.max().item()) + 1
        else:
            inverse, K = _ds_split
            inverse = inverse.to(device=values.device)

        sums = torch.zeros(K, device=values.device, dtype=values.dtype).scatter_add_(
            0, inverse, values
        )
        counts = torch.bincount(inverse, minlength=K).to(dtype=values.dtype)
        return (sums / counts.clamp_min(1.0)).mean()

    def _pearson_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1.0e-8,
    ) -> torch.Tensor:
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=pred.dtype)

        pred = pred.float() * mask_f
        target = target.float() * mask_f

        valid_len = mask_f.sum(dim=1).clamp_min(1.0)

        pred_mean = pred.sum(dim=1, keepdim=True) / valid_len.unsqueeze(1)
        target_mean = target.sum(dim=1, keepdim=True) / valid_len.unsqueeze(1)

        pred_centered = (pred - pred_mean) * mask_f
        target_centered = (target - target_mean) * mask_f

        numerator = (pred_centered * target_centered).sum(dim=1)

        pred_var = pred_centered.pow(2).sum(dim=1)
        target_var = target_centered.pow(2).sum(dim=1)

        denom = torch.sqrt(pred_var * target_var).clamp_min(eps)

        pcc = numerator / denom

        valid = (pred_var > eps) & (target_var > eps)
        pcc = torch.where(valid, pcc, torch.zeros_like(pcc))

        return pcc

    def _dataset_name(self, dataset_id: int) -> str:
        name = self.dataset_id_to_name.get(int(dataset_id), str(int(dataset_id)))
        return name.replace("/", "_").replace(" ", "_")

    def _zero_probability_from_out(self, out: dict[str, Any]) -> torch.Tensor:
        zero_param = self._zero_param_from_out(out)

        if self.config.loss.zero_input_is_logits:
            return torch.sigmoid(zero_param)

        return zero_param

    def _likelihood_curves_for_plot(
        self,
        out: dict[str, Any],
        zero_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mu = out["mu"].detach().float()
        phi = out["phi"].detach().float().clamp_min(self.loss_fn.eps)

        if self.config.loss.profile_likelihood == "ziln":
            with torch.no_grad():
                log_loc, log_sigma = self.loss_fn._lognormal_params(
                    mu=mu,
                    pi_zero=zero_prob.detach().float(),
                    phi=phi,
                )
                log_sigma2 = log_sigma.pow(2)
                positive_mean = torch.exp(log_loc + 0.5 * log_sigma2)
                variance = (
                    torch.expm1(log_sigma2)
                    * torch.exp(2.0 * log_loc + log_sigma2)
                )
                unconditional_mean = (
                    (1.0 - zero_prob.detach().float())
                    * positive_mean
                )
            return {
                "likelihood_sd": variance.clamp_min(self.loss_fn.eps).sqrt(),
                "likelihood_shape": log_sigma.float(),
                "likelihood_positive_mean": positive_mean.float(),
                "likelihood_unconditional_mean": unconditional_mean.float(),
            }

        variance = mu + phi * mu.pow(2)
        return {
            "likelihood_sd": variance.clamp_min(self.loss_fn.eps).sqrt(),
            "likelihood_shape": (1.0 / phi).clamp_max(1.0e6),
        }

    def _mean_for_dataset(
        self,
        value: torch.Tensor,
        *,
        sample_mask: torch.Tensor,
        position_mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        value = value.detach().float()

        if value.ndim == 0:
            return value

        if value.ndim == 1 and value.shape[0] == batch_size:
            return value[sample_mask].mean()

        if value.ndim == 2 and value.shape[0] == batch_size and value.shape[1] == 1:
            return value[sample_mask].mean()

        if value.ndim == 2 and value.shape == position_mask.shape:
            dataset_values = value[sample_mask]
            dataset_position_mask = position_mask[sample_mask]

            return dataset_values[dataset_position_mask].mean()

        return value.mean()

    def _log_scalar(
        self,
        name: str,
        value: torch.Tensor,
        *,
        batch_size: int,
        sync_dist: bool,
        prog_bar: bool = False,
        reduce_fx: str = "mean",
    ) -> None:
        self.log(
            name,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            reduce_fx=reduce_fx,
        )

    def _log_per_dataset_metrics(
        self,
        *,
        stage: str,
        out: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        """
        Logs per-dataset metrics under the same metric namespace as the
        global average, so TensorBoard can compare datasets side by side:
            val_mu_pcc
            val_mu_pcc/<dataset_name>
            val_support_pcc
            val_support_pcc/<dataset_name>
            val_loss
            val_loss/<dataset_name>
        """
        dataset_ids = out["dataset_ids"].detach().to(device=out["target"].device)
        unique_dataset_ids = torch.unique(dataset_ids)

        batch_size = int(out["target"].shape[0])
        sync_dist = self.config.trainer.sync_dist_logs

        extras = out["extras"]
        mask = out["mask"].bool()
        zero_probability = self._zero_probability_from_out(out)

        for dataset_id_tensor in unique_dataset_ids:
            dataset_id = int(dataset_id_tensor.item())
            dataset_name = self._dataset_name(dataset_id)
            sample_mask = dataset_ids == dataset_id_tensor

            n_samples = int(sample_mask.sum().item())

            scalar_metrics = {
                "loss": metrics["loss_per_sample"][sample_mask].mean(),
                "mu_pcc": metrics["mu_pcc_per_sample"][sample_mask].mean(),
                "mu_unconditional_pcc": metrics[
                    "mu_unconditional_pcc_per_sample"
                ][sample_mask].mean(),
                "mu_unconditional_pcc_loss": metrics[
                    "mu_unconditional_pcc_loss_per_sample"
                ][sample_mask].mean(),
                "support_pcc": metrics["support_pcc_per_sample"][sample_mask].mean(),
                "mu_mse": metrics["mu_mse_per_sample"][sample_mask].mean(),
                "mu_unconditional_mse": metrics[
                    "mu_unconditional_mse_per_sample"
                ][sample_mask].mean(),
                "mu_log1p_mse": metrics["mu_log1p_mse_per_sample"][
                    sample_mask
                ].mean(),
                "mu_unconditional_log1p_mse": metrics[
                    "mu_unconditional_log1p_mse_per_sample"
                ][sample_mask].mean(),
                "target_zero_frac": metrics[
                    "target_zero_frac_per_sample"
                ][sample_mask].mean(),
                "zero_prob_on_zero": metrics[
                    "zero_prob_on_zero_per_sample"
                ][sample_mask].mean(),
                "zero_prob_on_positive": metrics[
                    "zero_prob_on_positive_per_sample"
                ][sample_mask].mean(),
                "zero_prob_gap": metrics[
                    "zero_prob_gap_per_sample"
                ][sample_mask].mean(),
                "n_samples": torch.as_tensor(
                    float(n_samples),
                    device=self.device,
                ),
                "zero_prob_mean": self._mean_for_dataset(
                    zero_probability,
                    sample_mask=sample_mask,
                    position_mask=mask,
                    batch_size=batch_size,
                ),
                "phi_mean": self._mean_for_dataset(
                    out["phi"],
                    sample_mask=sample_mask,
                    position_mask=mask,
                    batch_size=batch_size,
                ),
            }

            # Optional per-sample metrics exist only on the full ZILN + reg path.
            for name in self.OPTIONAL_SCALAR_METRICS:
                key = f"{name}_per_sample"
                if key in metrics:
                    scalar_metrics[name] = metrics[key][sample_mask].mean()

            for key in self.DATASET_DIAGNOSTIC_KEYS:
                scalar_metrics[key] = self._mean_for_dataset(
                    extras[key],
                    sample_mask=sample_mask,
                    position_mask=mask,
                    batch_size=batch_size,
                )

            for metric_name, metric_value in scalar_metrics.items():
                self._log_scalar(
                    f"{stage}_{metric_name}/{dataset_name}",
                    metric_value,
                    batch_size=n_samples,
                    sync_dist=sync_dist,
                    reduce_fx="sum" if metric_name == "n_samples" else "mean",
                )

    def _mse_per_sample(
        self,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        log1p: bool = False,
    ) -> torch.Tensor:
        mask_f = mask.bool().to(dtype=pred.dtype)
        valid_len = mask_f.sum(dim=1).clamp_min(1.0)

        pred = pred.float().clamp_min(0.0)
        target = target.float().clamp_min(0.0)

        if log1p:
            pred = torch.log1p(pred)
            target = torch.log1p(target)

        return ((pred - target).pow(2) * mask_f).sum(dim=1) / valid_len

    def _zero_threshold_for_metrics(self) -> float:
        if self.config.loss.profile_likelihood == "ziln":
            return float(self.config.loss.censor_threshold)

        return float(self.config.loss.zinb_zero_threshold)

    def _zero_diagnostics_per_sample(
        self,
        *,
        out: dict[str, Any],
        zero_probability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target = out["target"].float()
        mask = out["mask"].bool()
        mask_f = mask.to(dtype=target.dtype)

        zero_threshold = self._zero_threshold_for_metrics()
        zero_mask = (target <= zero_threshold) & mask
        positive_mask = (target > zero_threshold) & mask

        valid_len = mask_f.sum(dim=1).clamp_min(1.0)
        zero_count = zero_mask.to(dtype=target.dtype).sum(dim=1)
        positive_count = positive_mask.to(dtype=target.dtype).sum(dim=1)

        zero_prob = zero_probability.float()
        zero_prob_on_zero = (
            zero_prob * zero_mask.to(dtype=zero_prob.dtype)
        ).sum(dim=1) / zero_count.clamp_min(1.0)
        zero_prob_on_positive = (
            zero_prob * positive_mask.to(dtype=zero_prob.dtype)
        ).sum(dim=1) / positive_count.clamp_min(1.0)

        zero_prob_on_zero = torch.where(
            zero_count > 0.0,
            zero_prob_on_zero,
            torch.zeros_like(zero_prob_on_zero),
        )
        zero_prob_on_positive = torch.where(
            positive_count > 0.0,
            zero_prob_on_positive,
            torch.zeros_like(zero_prob_on_positive),
        )

        return {
            "target_zero_frac_per_sample": zero_count / valid_len,
            "zero_prob_on_zero_per_sample": zero_prob_on_zero,
            "zero_prob_on_positive_per_sample": zero_prob_on_positive,
            "zero_prob_gap_per_sample": zero_prob_on_zero - zero_prob_on_positive,
        }

    def _compute_loss_and_metrics(
        self,
        out: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        mode = self.config.loss.mode
        dataset_ids = out["dataset_ids"]

        # Pre-compute dataset grouping once; passed to every _aggregate_per_sample call
        # to avoid 15 redundant torch.unique() calls per training step.
        _ds_split: tuple[torch.Tensor, int] | None = None
        if self.config.loss.dataset_balanced_loss:
            _, _ds_inv = torch.unique(
                dataset_ids.to(device=out["target"].device), return_inverse=True
            )
            _ds_split = (_ds_inv, int(_ds_inv.max().item()) + 1)

        # Quantity correlated against normalized profile shape:
        #   rho = raw local occupancy (baseline)
        #   q   = occupancy AFTER causal queue propagation. This is the ONLY path
        #         by which the queue coupling (and a learnable alpha) reaches the
        #         loss; with propagation off, q_bio == rho_bio.
        support = out["extras"][
            {"q": "q_bio", "rho": "rho_bio"}[self.config.loss.pcc_target]
        ]

        target = out["target"]
        mask = out["mask"].bool()
        mask_f = mask.to(dtype=support.dtype)

        # Normalize target per-transcript: y_norm_i = y_i / max_j(y_j), in [0, 1].
        # clamp_min(1.0) keeps zero-profile transcripts from producing NaN.
        target_max = (target * mask_f).max(dim=1, keepdim=True).values.clamp_min(1.0)
        target_norm = (target / target_max * mask_f).clamp(0.0, 1.0)

        support_pcc_per_sample = self._pearson_per_sample(
            pred=support,
            target=target_norm,
            mask=mask,
        )
        mu_pcc_per_sample = self._pearson_per_sample(
            pred=out["mu"].float(),
            target=target.float(),
            mask=mask,
        )
        mu_unconditional_pcc_per_sample = self._pearson_per_sample(
            pred=out["extras"]["mu_unconditional"].float(),
            target=target.float(),
            mask=mask,
        )
        mu_mse_per_sample = self._mse_per_sample(
            pred=out["mu"],
            target=target,
            mask=mask,
        )
        mu_unconditional_mse_per_sample = self._mse_per_sample(
            pred=out["extras"]["mu_unconditional"],
            target=target,
            mask=mask,
        )
        mu_log1p_mse_per_sample = self._mse_per_sample(
            pred=out["mu"],
            target=target,
            mask=mask,
            log1p=True,
        )
        mu_unconditional_log1p_mse_per_sample = self._mse_per_sample(
            pred=out["extras"]["mu_unconditional"],
            target=target,
            mask=mask,
            log1p=True,
        )

        zero_probability = self._zero_probability_from_out(out)
        zero_metrics = self._zero_diagnostics_per_sample(
            out=out,
            zero_probability=zero_probability,
        )
        ziln_mean_metrics = {}
        if self.config.loss.profile_likelihood == "ziln":
            with torch.no_grad():
                ziln_positive_mean = self.loss_fn.positive_mean_from_params(
                    mu=out["mu"].detach().float(),
                    pi_zero=zero_probability.detach().float(),
                    phi=out["phi"].detach().float(),
                ).float()
                ziln_unconditional_mean = (
                    (1.0 - zero_probability.detach().float())
                    * ziln_positive_mean
                )

            ziln_mean_pcc_per_sample = self._pearson_per_sample(
                pred=ziln_positive_mean,
                target=target.float(),
                mask=mask,
            )
            ziln_unconditional_mean_pcc_per_sample = self._pearson_per_sample(
                pred=ziln_unconditional_mean,
                target=target.float(),
                mask=mask,
            )
            ziln_mean_log1p_mse_per_sample = self._mse_per_sample(
                pred=ziln_positive_mean,
                target=target,
                mask=mask,
                log1p=True,
            )
            ziln_unconditional_mean_log1p_mse_per_sample = self._mse_per_sample(
                pred=ziln_unconditional_mean,
                target=target,
                mask=mask,
                log1p=True,
            )
            ziln_mean_metrics = {
                "ziln_mean_pcc": self._aggregate_per_sample(
                    ziln_mean_pcc_per_sample, dataset_ids, _ds_split,
                ),
                "ziln_mean_pcc_per_sample": ziln_mean_pcc_per_sample,
                "ziln_unconditional_mean_pcc": self._aggregate_per_sample(
                    ziln_unconditional_mean_pcc_per_sample, dataset_ids, _ds_split,
                ),
                "ziln_unconditional_mean_pcc_per_sample": (
                    ziln_unconditional_mean_pcc_per_sample
                ),
                "ziln_mean_log1p_mse": self._aggregate_per_sample(
                    ziln_mean_log1p_mse_per_sample, dataset_ids, _ds_split,
                ),
                "ziln_mean_log1p_mse_per_sample": ziln_mean_log1p_mse_per_sample,
                "ziln_unconditional_mean_log1p_mse": self._aggregate_per_sample(
                    ziln_unconditional_mean_log1p_mse_per_sample, dataset_ids, _ds_split,
                ),
                "ziln_unconditional_mean_log1p_mse_per_sample": (
                    ziln_unconditional_mean_log1p_mse_per_sample
                ),
            }

        # Loss = 1 - PCC so that minimising loss maximises correlation.
        pcc_loss_per_sample = 1.0 - support_pcc_per_sample
        mu_unconditional_pcc_loss_per_sample = 1.0 - mu_unconditional_pcc_per_sample
        pcc_loss = self._aggregate_per_sample(pcc_loss_per_sample, dataset_ids)
        mu_unconditional_pcc_loss = self._aggregate_per_sample(
            mu_unconditional_pcc_loss_per_sample,
            dataset_ids,
        )
        support_pcc = self._aggregate_per_sample(support_pcc_per_sample, dataset_ids)
        mu_pcc = self._aggregate_per_sample(mu_pcc_per_sample, dataset_ids)
        mu_unconditional_pcc = self._aggregate_per_sample(
            mu_unconditional_pcc_per_sample,
            dataset_ids,
        )
        mu_mse = self._aggregate_per_sample(mu_mse_per_sample, dataset_ids)
        mu_unconditional_mse = self._aggregate_per_sample(
            mu_unconditional_mse_per_sample,
            dataset_ids,
        )
        mu_log1p_mse = self._aggregate_per_sample(
            mu_log1p_mse_per_sample,
            dataset_ids,
        )
        mu_unconditional_log1p_mse = self._aggregate_per_sample(
            mu_unconditional_log1p_mse_per_sample,
            dataset_ids,
        )

        metrics = {
            "pcc_loss": pcc_loss,
            "pcc_loss_per_sample": pcc_loss_per_sample,
            "mu_unconditional_pcc_loss": mu_unconditional_pcc_loss,
            "mu_unconditional_pcc_loss_per_sample": (
                mu_unconditional_pcc_loss_per_sample
            ),
            "support_pcc": support_pcc,
            "support_pcc_per_sample": support_pcc_per_sample,
            "mu_pcc": mu_pcc,
            "mu_pcc_per_sample": mu_pcc_per_sample,
            "mu_unconditional_pcc": mu_unconditional_pcc,
            "mu_unconditional_pcc_per_sample": mu_unconditional_pcc_per_sample,
            "mu_mse": mu_mse,
            "mu_mse_per_sample": mu_mse_per_sample,
            "mu_unconditional_mse": mu_unconditional_mse,
            "mu_unconditional_mse_per_sample": mu_unconditional_mse_per_sample,
            "mu_log1p_mse": mu_log1p_mse,
            "mu_log1p_mse_per_sample": mu_log1p_mse_per_sample,
            "mu_unconditional_log1p_mse": mu_unconditional_log1p_mse,
            "mu_unconditional_log1p_mse_per_sample": (
                mu_unconditional_log1p_mse_per_sample
            ),
        }
        for name, value in zero_metrics.items():
            metrics[name] = value
            metrics[name.removesuffix("_per_sample")] = self._aggregate_per_sample(
                value,
                dataset_ids,
            )
        metrics.update(ziln_mean_metrics)

        # Batch-mean log(J) centering — breaks the J/scale_dt identifiability degeneracy.
        # J is a per-transcript biological scale (ribosome flux amplitude). Without this
        # constraint, the optimizer slides along the flat J×scale_dt manifold, pushing J→0
        # and scale_dt→∞ while leaving mu unchanged. Penalising the BATCH MEAN of log(J)
        # drifting away from 0 prevents global collapse while leaving per-transcript
        # J variation (the biological signal) completely free.
        j_centering_weight = float(getattr(self.config.loss, "j_centering_weight", 0.0))
        j_centering_loss = torch.zeros((), device=support.device, dtype=support.dtype)
        if j_centering_weight > 0.0:
            J = out["extras"]["J"].float()  # [B, 1]
            log_J_mean = torch.log(J.clamp_min(1e-6)).mean()
            j_centering_loss = j_centering_weight * log_J_mean.pow(2)
            metrics["j_centering_loss"] = j_centering_loss

        if mode == "pcc_only":
            metrics["loss"] = pcc_loss + mu_unconditional_pcc_loss + j_centering_loss
            metrics["loss_per_sample"] = (
                pcc_loss_per_sample + mu_unconditional_pcc_loss_per_sample
            )
            return metrics

        nll_per_sample = self._profile_nll_per_sample(out)
        nll = self._aggregate_per_sample(nll_per_sample, dataset_ids)

        metrics["nll"] = nll
        metrics["nll_per_sample"] = nll_per_sample

        if mode == "likelihood_only":
            metrics["loss"] = nll + j_centering_loss
            metrics["loss_per_sample"] = nll_per_sample
            return metrics

        if mode == "pcc_likelihood":
            metrics["loss"] = pcc_loss + mu_unconditional_pcc_loss + nll + j_centering_loss
            metrics["loss_per_sample"] = (
                pcc_loss_per_sample
                + mu_unconditional_pcc_loss_per_sample
                + nll_per_sample
            )
            return metrics

        raise KeyError(mode)

    # ============================================================
    # Logging helpers
    # ============================================================

    def _log_stage(
        self,
        *,
        stage: str,
        out: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        batch_size = int(out["target"].shape[0])
        sync_dist = self.config.trainer.sync_dist_logs

        self.log(
            f"{stage}_loss",
            metrics["loss"],
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )

        self.log(
            f"{stage}_mu_pcc",
            metrics["mu_pcc"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )

        self.log(
            f"{stage}_mu_unconditional_pcc",
            metrics["mu_unconditional_pcc"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )

        self.log(
            f"{stage}_support_pcc",
            metrics["support_pcc"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )

        # Optional scalar metrics produced only by the active objective terms.
        # In pcc_only mode the likelihood NLL is absent.
        for name in self.OPTIONAL_SCALAR_METRICS:
            if name in metrics:
                self.log(
                    f"{stage}_{name}",
                    metrics[name],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size,
                    sync_dist=sync_dist,
                )

        extras = out["extras"]

        for key in self.DATASET_DIAGNOSTIC_KEYS:
            self.log(
                f"{stage}_{key}",
                extras[key].float().mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )

        zero_probability = self._zero_probability_from_out(out)
        self.log(
            f"{stage}_zero_prob_mean",
            zero_probability.float()[out["mask"].bool()].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )

        self.log(
            f"{stage}_phi_mean",
            out["phi"].float()[out["mask"].bool()].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        self._log_per_dataset_metrics(
            stage=stage,
            out=out,
            metrics=metrics,
        )

    # ============================================================
    # Profile plots
    # ============================================================

    def on_validation_epoch_start(self) -> None:
        self._val_profile_plot_logged_this_epoch = False

    def _sequence_for_plot(
        self,
        value: torch.Tensor,
        *,
        sample_idx: int,
        mask_i: torch.Tensor,
    ):
        value = value.detach().float().cpu()
        mask_i = mask_i.detach().bool().cpu()
        length = int(mask_i.sum().item())

        if value.ndim == 0:
            return torch.full((length,), float(value.item())).numpy()

        if value.ndim == 1:
            if value.shape[0] == mask_i.shape[0]:
                return value[mask_i].numpy()

            return torch.full(
                (length,),
                float(value[sample_idx].item()),
            ).numpy()

        if value.ndim == 2:
            sample_value = value[sample_idx]

            if sample_value.ndim == 1 and sample_value.shape[0] == mask_i.shape[0]:
                return sample_value[mask_i].numpy()

            if sample_value.numel() == 1:
                return torch.full(
                    (length,),
                    float(sample_value.reshape(-1)[0].item()),
                ).numpy()

        return value.reshape(-1)[:length].numpy()

    def _plot_profile_example(
        self,
        out: dict[str, Any],
        *,
        batch_idx: int,
    ) -> None:
        if not self.config.metrics.log_example_plot:
            return

        if self._val_profile_plot_logged_this_epoch or batch_idx != 0:
            return

        if not getattr(self.trainer, "is_global_zero", True):
            return

        if self.logger is None or getattr(self.logger, "experiment", None) is None:
            return

        dataset_ids = out["dataset_ids"].detach().cpu()
        max_plots = int(self.config.metrics.example_plot_max_datasets)
        selected_indices = []
        seen_dataset_ids = set()

        for sample_idx, dataset_id in enumerate(dataset_ids.tolist()):
            dataset_id = int(dataset_id)

            if dataset_id in seen_dataset_ids:
                continue

            seen_dataset_ids.add(dataset_id)
            selected_indices.append(sample_idx)

            if len(selected_indices) >= max_plots:
                break

        for sample_idx in selected_indices:
            self._plot_profile_sample(out, sample_idx=sample_idx)

        self._val_profile_plot_logged_this_epoch = True

    def _plot_profile_sample(
        self,
        out: dict[str, Any],
        *,
        sample_idx: int,
    ) -> None:
        mask_i = out["mask"][sample_idx].detach().bool().cpu()
        length = int(mask_i.sum().item())

        extras = dict(out["extras"])
        zero_prob = self._zero_probability_from_out(out)
        extras["zero_prob"] = zero_prob
        extras["phi"] = out["phi"]
        extras.update(self._likelihood_curves_for_plot(out, zero_prob))

        x_axis = torch.arange(length).numpy()
        dataset_id = int(out["dataset_ids"][sample_idx].detach().cpu().item())
        dataset_name = self._dataset_name(dataset_id)
        n_axes = 1 + len(self.PROFILE_PLOT_GROUPS)

        fig, axes = plt.subplots(
            n_axes,
            1,
            figsize=(14, 2.6 * n_axes),
            sharex=True,
            constrained_layout=True,
        )

        target = self._sequence_for_plot(
            out["target"],
            sample_idx=sample_idx,
            mask_i=mask_i,
        )
        prediction = self._sequence_for_plot(
            out["mu"],
            sample_idx=sample_idx,
            mask_i=mask_i,
        )

        axes[0].plot(x_axis, target, label="target", linewidth=1.2)
        axes[0].plot(x_axis, prediction, label="prediction", linewidth=1.2)
        if "mu_unconditional" in extras:
            mu_unconditional = self._sequence_for_plot(
                extras["mu_unconditional"],
                sample_idx=sample_idx,
                mask_i=mask_i,
            )
            axes[0].plot(
                x_axis,
                mu_unconditional,
                label="mu_unconditional",
                linewidth=1.0,
                linestyle="--",
            )

        axes[0].set_title(
            " | ".join(
                (
                    f"Dataset: {dataset_name}",
                    f"sample: {out['ids'][sample_idx]}",
                    f"length: {length}",
                    f"likelihood: {self.config.loss.profile_likelihood}",
                )
            )
        )
        axes[0].set_ylabel("profile")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        for axis, (ylabel, keys) in zip(
            axes[1:],
            self.PROFILE_PLOT_GROUPS,
            strict=True,
        ):
            for key in keys:
                curve = self._sequence_for_plot(
                    extras[key],
                    sample_idx=sample_idx,
                    mask_i=mask_i,
                )

                axis.plot(x_axis, curve, label=key, linewidth=1.0)

            axis.set_ylabel(ylabel)
            if ylabel == "dropout":
                axis.set_ylim(-0.05, 1.05)
            axis.grid(True, alpha=0.3)
            axis.legend(loc="upper right")

        axes[-1].set_xlabel("codon position")

        experiment = self.logger.experiment
        tag = f"val_profile/{dataset_name}"

        if hasattr(experiment, "add_figure"):
            experiment.add_figure(tag, fig, global_step=self.global_step)
        elif hasattr(experiment, "log_figure"):
            experiment.log_figure(
                figure_name=tag,
                figure=fig,
                step=self.global_step,
            )

        log_dir = getattr(self.logger, "log_dir", None)
        if log_dir is not None:
            plot_dir = Path(log_dir) / "profile_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                plot_dir
                / (
                    f"epoch_{int(self.current_epoch):04d}_"
                    f"step_{int(self.global_step):08d}_{dataset_name}.png"
                ),
                dpi=140,
            )

        plt.close(fig)

    # ============================================================
    # CAGrad-style biological gradient aggregation
    # ============================================================

    @staticmethod
    def _is_biological_parameter_name(name: str) -> bool:
        return name.startswith("biological_model.") or name == "queue_raw_alpha"

    def _bio_parameters(self) -> list[torch.nn.Parameter]:
        return [
            param
            for name, param in self.model.named_parameters()
            if param.requires_grad and self._is_biological_parameter_name(name)
        ]

    def _dataset_losses_from_per_sample(
        self,
        *,
        loss_per_sample: torch.Tensor,
        dataset_ids: torch.Tensor,
    ) -> list[torch.Tensor]:
        dataset_ids = dataset_ids.to(device=loss_per_sample.device)
        losses = []

        for dataset_id in torch.unique(dataset_ids):
            sample_mask = dataset_ids == dataset_id
            if torch.any(sample_mask):
                losses.append(loss_per_sample[sample_mask].mean())

        return losses

    @staticmethod
    def _flatten_autograd_grads(
        grads: tuple[torch.Tensor | None, ...],
        params: list[torch.nn.Parameter],
    ) -> torch.Tensor:
        flat = []

        for grad, param in zip(grads, params, strict=True):
            if grad is None:
                flat.append(torch.zeros_like(param).reshape(-1))
            else:
                flat.append(grad.reshape(-1))

        if len(flat) == 0:
            raise RuntimeError("Cannot flatten an empty parameter list.")

        return torch.cat(flat)

    @staticmethod
    def _assign_flat_grads(
        *,
        params: list[torch.nn.Parameter],
        flat_grad: torch.Tensor,
    ) -> None:
        offset = 0

        for param in params:
            n = param.numel()
            grad_view = flat_grad[offset : offset + n].view_as(param)

            if param.grad is None:
                param.grad = grad_view.detach().clone()
            else:
                param.grad.detach().copy_(grad_view)

            offset += n

        if offset != flat_grad.numel():
            raise RuntimeError("Flat gradient size did not match parameter sizes.")

    @staticmethod
    def _project_simplex(v: torch.Tensor) -> torch.Tensor:
        if v.ndim != 1:
            raise ValueError(f"Expected vector, got {tuple(v.shape)}.")

        n = v.numel()
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0) - 1.0
        ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
        cond = u - cssv / ind > 0

        if not torch.any(cond):
            return torch.full_like(v, 1.0 / float(n))

        rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
        theta = cssv[rho] / (rho.to(dtype=v.dtype) + 1.0)
        return torch.clamp(v - theta, min=0.0)

    def _cagrad_style_combine(self, grads: torch.Tensor) -> torch.Tensor:
        if grads.ndim != 2:
            raise ValueError(f"Expected grads [K, P], got {tuple(grads.shape)}.")

        if grads.shape[0] == 1:
            return grads[0]

        mean_grad = grads.mean(dim=0)

        # MGDA-style minimum-norm convex combination over dataset gradients,
        # blended with the mean gradient. This is the practical CAGrad-like
        # conflict-averse update used only for shared biological parameters.
        gram = grads @ grads.t()
        k = grads.shape[0]
        weights = torch.full(
            (k,),
            1.0 / float(k),
            device=grads.device,
            dtype=grads.dtype,
        )

        trace = torch.trace(gram).abs().clamp_min(1.0e-12)
        step_size = 1.0 / trace
        n_iter = self.config.optim.cagrad_iterations

        for _ in range(max(n_iter, 1)):
            grad_w = 2.0 * (gram @ weights)
            weights = self._project_simplex(weights - step_size * grad_w)

        conflict_grad = weights @ grads
        alpha = self.config.optim.cagrad_alpha
        alpha = min(max(alpha, 0.0), 1.0)
        combined = (1.0 - alpha) * mean_grad + alpha * conflict_grad

        if self.config.optim.cagrad_rescale_to_mean_norm:
            target_norm = mean_grad.norm().clamp_min(1.0e-12)
            combined = combined * (target_norm / combined.norm().clamp_min(1.0e-12))

        return combined

    def _manual_cagrad_training_step(
        self,
        out: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        # Ordinary dataset-balanced loss updates all non-biological heads.
        self.manual_backward(metrics["loss"], retain_graph=True)

        bio_params = self._bio_parameters()
        dataset_losses = self._dataset_losses_from_per_sample(
            loss_per_sample=metrics["loss_per_sample"],
            dataset_ids=out["dataset_ids"],
        )

        if len(bio_params) > 0 and len(dataset_losses) > 1:
            flat_grads = []

            for dataset_loss in dataset_losses:
                grads = torch.autograd.grad(
                    dataset_loss,
                    bio_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                flat_grads.append(self._flatten_autograd_grads(grads, bio_params))

            combined = self._cagrad_style_combine(torch.stack(flat_grads, dim=0))
            self._assign_flat_grads(params=bio_params, flat_grad=combined.detach())

        grad_clip_val = self.config.trainer.gradient_clip_val

        if grad_clip_val > 0.0:
            self.clip_gradients(
                opt,
                gradient_clip_val=grad_clip_val,
                gradient_clip_algorithm=str(
                    self.config.trainer.gradient_clip_algorithm
                ),
            )

        opt.step()
        opt.zero_grad(set_to_none=True)

        return metrics["loss"].detach()

    # ============================================================
    # Steps
    # ============================================================

    def training_step(self, batch, batch_idx):
        out = self._forward_batch(batch)
        metrics = self._compute_loss_and_metrics(out)

        self._log_stage(
            stage="train",
            out=out,
            metrics=metrics,
        )

        if self.use_cagrad:
            return self._manual_cagrad_training_step(out, metrics)

        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._forward_batch(batch)
        metrics = self._compute_loss_and_metrics(out)

        self._log_stage(
            stage="val",
            out=out,
            metrics=metrics,
        )
        self._plot_profile_example(out, batch_idx=batch_idx)

        return metrics["loss"]

    def on_validation_epoch_end(self) -> None:
        if not self.use_cagrad:
            return

        scheduler = self.lr_schedulers()
        monitor = self.config.optim.scheduler.monitor
        metric = self.trainer.callback_metrics.get(monitor)

        if metric is not None:
            scheduler.step(metric)

    # ============================================================
    # Optimizer
    # ============================================================

    def configure_optimizers(self):
        base_lr = self.config.optim.lr
        bio_lr = self.config.optim.lr_biological
        rest_lr = self.config.optim.lr_rest
        weight_decay = self.config.optim.weight_decay

        bio_params = []
        rest_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if self._is_biological_parameter_name(name):
                bio_params.append(param)
            else:
                rest_params.append(param)

        param_groups = []

        if bio_params:
            param_groups.append(
                {
                    "params": bio_params,
                    "lr": bio_lr,
                    "weight_decay": weight_decay,
                    "name": "biological",
                }
            )

        if rest_params:
            param_groups.append(
                {
                    "params": rest_params,
                    "lr": rest_lr,
                    "weight_decay": weight_decay,
                    "name": "rest",
                }
            )

        if not param_groups:
            raise RuntimeError("No trainable parameters found.")

        opt = torch.optim.AdamW(
            param_groups,
        )

        scheduler_config = self.config.optim.scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=str(scheduler_config.mode),
            factor=float(scheduler_config.factor),
            patience=int(scheduler_config.patience),
            min_lr=float(scheduler_config.min_lr),
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": str(scheduler_config.monitor),
                "interval": "epoch",
                "frequency": 1,
            },
        }
