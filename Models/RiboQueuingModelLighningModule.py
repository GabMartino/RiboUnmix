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

class LeftCensoredLogNormalNB2Loss(nn.Module):
    """
    Left-censored log-normal NLL with NB2-style variance.

    Model for threshold c > 0:

        P(Y <= c) = LogNormalCDF(c; log_loc, log_sigma)

        Y | Y > c uses the LogNormal(log_loc, log_sigma) density.

    There is no zero-inflated mixture and no fixed/manual pi parameter.

    Mu convention:

        positive_mean:

            mu = E[Y | Y > 0]

        median:

            mu = median[Y | Y > 0] = exp(log_loc)

    Positive-component variance target:

        positive_mean:
            Var[Y | Y > 0] = E[Y | Y > 0] + phi * E[Y | Y > 0]^2

        median:
            Var[Y | Y > 0] = median[Y | Y > 0] + phi * median[Y | Y > 0]^2

    Conversion:

        For mean parameterizations:
            log_sigma^2 = log(1 + Var / E[Y | Y > 0]^2)
            log_loc = log(E[Y | Y > 0]) - 0.5 * log_sigma^2

        For median parameterization:
            x = (1 + sqrt(1 + 4 * (1 / median + phi))) / 2
            log_sigma^2 = log(x)
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
        censor_threshold: float = 0.0,
        phi_input_is_log: bool = False,
        mu_parameterization: str | None = None,
    ):
        super().__init__()

        self.eps = float(eps)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)

        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)

        self.log_sigma_min = float(log_sigma_min)

        self.censor_threshold = float(censor_threshold)
        self.phi_input_is_log = bool(phi_input_is_log)
        if mu_parameterization is None:
            mu_parameterization = "positive_mean"
        if str(mu_parameterization) == "unconditional_mean":
            mu_parameterization = "positive_mean"
        self.mu_parameterization = str(mu_parameterization)
        valid_parameterizations = {
            "positive_mean",
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

    def _prepare_phi(self, phi: torch.Tensor) -> torch.Tensor:
        if self.phi_input_is_log:
            phi = torch.exp(phi)

        return phi.clamp(min=self.phi_min)  # no upper clamp — phi_reg handles it

    def _lognormal_params(
        self,
        *,
        mu: torch.Tensor,
        phi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        log_sigma = log_sigma.clamp(min=self.log_sigma_min)

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
        """Convert median-anchored NB2-style dispersion to sigma^2.

        For median m and x = exp(sigma^2), equating log-normal variance to
        the median-anchored target m + phi * m^2 yields:

            m^2 * x * (x - 1) = m + phi * m^2

        Therefore:

            x^2 - x - (1 / m + phi) = 0
        """
        a = (1.0 / median.clamp_min(self.eps)) + phi.clamp_min(self.eps)
        x = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * a))
        return torch.log(x.clamp_min(1.0 + self.eps))

    def positive_mean_from_params(
        self,
        *,
        mu: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        log_loc, log_sigma = self._lognormal_params(
            mu=mu,
            phi=phi,
        )
        mean = torch.exp(log_loc + 0.5 * log_sigma.pow(2))
        return mean.clamp(min=self.mu_min, max=self.mu_max)

    def forward(
        self,
        mu_phys: torch.Tensor,
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

            phi_t = self._prepare_phi(
                phi.to(torch.float64)
            )

            phi_t = torch.broadcast_to(phi_t, y.shape)

            log_loc, log_sigma = self._lognormal_params(
                mu=mu,
                phi=phi_t,
            )

            # Positive log-normal branch.
            log_y = torch.log(y.clamp_min(self.eps))
            z = (log_y - log_loc) / log_sigma

            positive_nll = (
                log_y
                + torch.log(log_sigma.clamp_min(self.eps))
                + 0.5 * math.log(2.0 * math.pi)
                + 0.5 * z.pow(2)
            )

            # Left-censored branch. Values y <= c contribute the probability
            # mass below c under the same log-normal, with no zero mixture.
            if self.censor_threshold > 0.0:
                c = torch.tensor(
                    self.censor_threshold,
                    device=y.device,
                    dtype=torch.float64,
                ).clamp_min(self.eps)

                z_c = (torch.log(c) - log_loc) / log_sigma
                log_cdf = self._log_normal_cdf_standard(z_c)
                censored_nll = -log_cdf
                is_censored = y <= self.censor_threshold

            else:
                # With no left-censoring threshold, exact zeros have no
                # continuous log-normal mass. Keep this branch explicit so a
                # c <= 0 configuration fails loudly through a large loss.
                censored_nll = torch.full_like(positive_nll, 1.0e8)
                is_censored = y <= 0.0

            nll = torch.where(is_censored, censored_nll, positive_nll)

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
        "p_visible_mean",
        "beta_mean",
        "mu_mass",
        "lambda_mass",
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
        "mu_pcc_loss",
        "mu_mse",
        "mu_log1p_mse",
        "target_zero_frac",
        "j_centering_loss",
        "phi_reg_loss",
        "transcript_scale_centering_loss",
    )

    PROFILE_PLOT_GROUPS = (
        ("beta", ("obs_beta",)),
        ("phi", ("phi",)),
        ("rho", ("rho_bio",)),
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

        valid_likelihoods = {"left_censored_lognormal", "lcln", "lognormal"}
        if str(loss_cfg.profile_likelihood) not in valid_likelihoods:
            raise ValueError(
                "Only left-censored log-normal likelihoods are supported. "
                f"Use one of {sorted(valid_likelihoods)}. "
                f"Got {loss_cfg.profile_likelihood!r}."
            )

        common_kwargs = {
            "eps": loss_cfg.eps,
            "mu_min": loss_cfg.mu_min,
            "mu_max": loss_cfg.mu_max,
            "phi_min": loss_cfg.phi_min,
            "phi_max": loss_cfg.phi_max,
            "phi_input_is_log": loss_cfg.phi_input_is_log,
        }

        return LeftCensoredLogNormalNB2Loss(
            **common_kwargs,
            mu_parameterization=getattr(
                loss_cfg,
                "lognormal_mu_parameterization",
                None,
            ),
            log_sigma_min=loss_cfg.log_sigma_min,
            censor_threshold=loss_cfg.censor_threshold,
        )

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
    # Loss / metrics
    # ============================================================

    def _profile_nll_per_sample(self, out: dict[str, Any]) -> torch.Tensor:
        return self.loss_fn(
            mu_phys=out["mu"].float(),
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

    def _likelihood_curves_for_plot(
        self,
        out: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        mu = out["mu"].detach().float()
        phi = out["phi"].detach().float().clamp_min(self.loss_fn.eps)

        with torch.no_grad():
            log_loc, log_sigma = self.loss_fn._lognormal_params(
                mu=mu,
                phi=phi,
            )
            positive_mean = torch.exp(log_loc + 0.5 * log_sigma.pow(2))

        return {
            "likelihood_positive_mean": positive_mean.float(),
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

        for dataset_id_tensor in unique_dataset_ids:
            dataset_id = int(dataset_id_tensor.item())
            dataset_name = self._dataset_name(dataset_id)
            sample_mask = dataset_ids == dataset_id_tensor

            n_samples = int(sample_mask.sum().item())

            scalar_metrics = {
                "loss": metrics["loss_per_sample"][sample_mask].mean(),
                "mu_pcc": metrics["mu_pcc_per_sample"][sample_mask].mean(),
                "mu_pcc_loss": metrics["mu_pcc_loss_per_sample"][
                    sample_mask
                ].mean(),
                "support_pcc": metrics["support_pcc_per_sample"][sample_mask].mean(),
                "mu_mse": metrics["mu_mse_per_sample"][sample_mask].mean(),
                "mu_log1p_mse": metrics["mu_log1p_mse_per_sample"][
                    sample_mask
                ].mean(),
                "n_samples": torch.as_tensor(
                    float(n_samples),
                    device=self.device,
                ),
                "phi_mean": self._mean_for_dataset(
                    out["phi"],
                    sample_mask=sample_mask,
                    position_mask=mask,
                    batch_size=batch_size,
                ),
            }

            # Optional per-sample metrics exist only when those objective terms
            # are active.
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

    def _zero_diagnostics_per_sample(
        self,
        out: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        target = out["target"].float()
        mask = out["mask"].bool()
        mask_f = mask.to(dtype=target.dtype)

        zero_count = (
            (target <= float(self.config.loss.censor_threshold)) & mask
        ).to(dtype=target.dtype).sum(dim=1)
        valid_len = mask_f.sum(dim=1).clamp_min(1.0)

        return {"target_zero_frac_per_sample": zero_count / valid_len}

    def _support_target_scale(
        self,
        *,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        strategy = str(
            getattr(self.config.loss, "support_target_normalization", "max")
        ).lower()
        min_scale = float(
            getattr(self.config.loss, "support_target_min_scale", 1.0)
        )
        min_scale = max(min_scale, float(self.config.loss.eps))

        target_f = target.float().clamp_min(0.0)
        mask_b = mask.bool()

        if strategy in {"max", "maximum"}:
            mask_f = mask_b.to(dtype=target_f.dtype)
            return (target_f * mask_f).max(dim=1, keepdim=True).values.clamp_min(
                min_scale
            )

        if strategy in {"nonzero_quantile", "quantile_nonzero"}:
            q = float(getattr(self.config.loss, "support_target_quantile", 0.99))
            q = min(max(q, 0.0), 1.0)
            scales = []
            for target_i, mask_i in zip(target_f, mask_b, strict=True):
                valid = mask_i & torch.isfinite(target_i) & (target_i > 0.0)
                values = target_i[valid]
                if values.numel() == 0:
                    scale_i = target_i.new_tensor(min_scale)
                else:
                    scale_i = torch.quantile(values, q).clamp_min(min_scale)
                scales.append(scale_i)
            return torch.stack(scales).reshape(-1, 1)

        raise ValueError(
            "loss.support_target_normalization must be one of "
            "{'max', 'nonzero_quantile'}, got "
            f"{strategy!r}."
        )

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

        # Robust per-transcript normalization for the support-shape PCC target.
        # nonzero_quantile avoids letting one noisy extreme footprint set the
        # scale for the whole profile; values above the scale are winsorized to 1.
        target_scale = self._support_target_scale(target=target, mask=mask)
        target_norm = (target.float() / target_scale * mask_f).clamp(0.0, 1.0)

        with torch.no_grad():
            likelihood_positive_mean = self.loss_fn.positive_mean_from_params(
                mu=out["mu"].float(),
                phi=out["phi"].float(),
            ).float()

        support_pcc_per_sample = self._pearson_per_sample(
            pred=support,
            target=target_norm,
            mask=mask,
        )
        mu_pcc_per_sample = self._pearson_per_sample(
            pred=likelihood_positive_mean,
            target=target.float(),
            mask=mask,
        )
        mu_mse_per_sample = self._mse_per_sample(
            pred=likelihood_positive_mean,
            target=target,
            mask=mask,
        )
        mu_log1p_mse_per_sample = self._mse_per_sample(
            pred=likelihood_positive_mean,
            target=target,
            mask=mask,
            log1p=True,
        )

        zero_metrics = self._zero_diagnostics_per_sample(out)

        # Loss = 1 - PCC so that minimising loss maximises correlation.
        pcc_loss_per_sample = 1.0 - support_pcc_per_sample
        mu_pcc_loss_per_sample = 1.0 - mu_pcc_per_sample
        pcc_loss = self._aggregate_per_sample(
            pcc_loss_per_sample,
            dataset_ids,
            _ds_split,
        )
        mu_pcc_loss = self._aggregate_per_sample(
            mu_pcc_loss_per_sample,
            dataset_ids,
            _ds_split,
        )
        support_pcc = self._aggregate_per_sample(
            support_pcc_per_sample,
            dataset_ids,
            _ds_split,
        )
        mu_pcc = self._aggregate_per_sample(
            mu_pcc_per_sample,
            dataset_ids,
            _ds_split,
        )
        mu_mse = self._aggregate_per_sample(
            mu_mse_per_sample,
            dataset_ids,
            _ds_split,
        )
        mu_log1p_mse = self._aggregate_per_sample(
            mu_log1p_mse_per_sample,
            dataset_ids,
            _ds_split,
        )

        metrics = {
            "pcc_loss": pcc_loss,
            "pcc_loss_per_sample": pcc_loss_per_sample,
            "mu_pcc_loss": mu_pcc_loss,
            "mu_pcc_loss_per_sample": mu_pcc_loss_per_sample,
            "support_pcc": support_pcc,
            "support_pcc_per_sample": support_pcc_per_sample,
            "mu_pcc": mu_pcc,
            "mu_pcc_per_sample": mu_pcc_per_sample,
            "mu_mse": mu_mse,
            "mu_mse_per_sample": mu_mse_per_sample,
            "mu_log1p_mse": mu_log1p_mse,
            "mu_log1p_mse_per_sample": mu_log1p_mse_per_sample,
        }
        for name, value in zero_metrics.items():
            metrics[name] = value
            metrics[name.removesuffix("_per_sample")] = self._aggregate_per_sample(
                value,
                dataset_ids,
                _ds_split,
            )
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

        phi_reg_weight = float(getattr(self.config.loss, "phi_reg_weight", 0.0))
        phi_reg_loss = torch.zeros((), device=support.device, dtype=support.dtype)
        if phi_reg_weight > 0.0:
            log_phi = out["extras"]["log_phi"].float()  # [B, 1]
            log_phi_target = math.log(float(getattr(self.config.loss, "phi_reg_target", 10.0)))
            phi_reg_loss = phi_reg_weight * (log_phi - log_phi_target).pow(2).mean()
            metrics["phi_reg_loss"] = phi_reg_loss

        ts_centering_weight = float(getattr(self.config.loss, "transcript_scale_centering_weight", 0.0))
        transcript_scale_centering_loss = torch.zeros((), device=support.device, dtype=support.dtype)
        if ts_centering_weight > 0.0:
            transcript_log_scale = out["extras"]["transcript_log_scale"].float().squeeze(-1)  # [B]
            unique_ds_ids = torch.unique(dataset_ids)
            per_ds = [
                transcript_log_scale[dataset_ids == ds_id].mean().pow(2)
                for ds_id in unique_ds_ids
                if (dataset_ids == ds_id).sum() > 1
            ]
            if per_ds:
                transcript_scale_centering_loss = ts_centering_weight * torch.stack(per_ds).mean()
            metrics["transcript_scale_centering_loss"] = transcript_scale_centering_loss

        if mode == "pcc_only":
            metrics["loss"] = pcc_loss + mu_pcc_loss + j_centering_loss + phi_reg_loss + transcript_scale_centering_loss
            metrics["loss_per_sample"] = (
                pcc_loss_per_sample + mu_pcc_loss_per_sample
            )
            return metrics

        nll_per_sample = self._profile_nll_per_sample(out)
        nll = self._aggregate_per_sample(nll_per_sample, dataset_ids, _ds_split)

        metrics["nll"] = nll
        metrics["nll_per_sample"] = nll_per_sample

        if mode == "likelihood_only":
            metrics["loss"] = nll + j_centering_loss + phi_reg_loss + transcript_scale_centering_loss
            metrics["loss_per_sample"] = nll_per_sample
            return metrics

        if mode == "pcc_likelihood":
            metrics["loss"] = pcc_loss + mu_pcc_loss + nll + j_centering_loss + phi_reg_loss + transcript_scale_centering_loss
            metrics["loss_per_sample"] = (
                pcc_loss_per_sample
                + mu_pcc_loss_per_sample
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

        self.log(
            f"{stage}_phi_mean",
            out["phi"].float()[out["mask"].bool()].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )

        log_train_per_dataset = bool(
            getattr(self.config.metrics, "log_train_per_dataset_metrics", False)
        )
        if stage != "train" or log_train_per_dataset:
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
        extras["phi"] = out["phi"]
        likelihood_curves = self._likelihood_curves_for_plot(out)
        extras.update(likelihood_curves)

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
            likelihood_curves["likelihood_positive_mean"],
            sample_idx=sample_idx,
            mask_i=mask_i,
        )

        axes[0].plot(x_axis, target, label="target", linewidth=1.2)
        axes[0].plot(x_axis, prediction, label="mu prediction", linewidth=1.2)

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
        return (
            name.startswith("biological_model.")
            or name.startswith("queue_alpha_head.")
            or name == "queue_raw_alpha"
        )

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

    @staticmethod
    def _detach_prediction_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu()

        return value

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        out = self._forward_batch(batch)
        extras = out["extras"]

        with torch.no_grad():
            likelihood_positive_mean = self.loss_fn.positive_mean_from_params(
                mu=out["mu"].detach().float(),
                phi=out["phi"].detach().float(),
            ).float()

        predictions: dict[str, Any] = {
            # Identifiers
            "ids": out["ids"],
            "dataset_id": out["dataset_ids"],
            "lengths": out["lengths"],
            "mask": out["mask"],
            "codon_ids": out["codon_ids"],
            "css": out["css"],
            # Target
            "target": out["target"],
            # Prediction: log-normal mean.
            "likelihood_positive_mean": likelihood_positive_mean,
            # Raw model output (median parameterization)
            "mu": out["mu"],
            # Biological branch
            "rho_bio": extras["rho_bio"],
            "q_bio": extras["q_bio"],
            "h_bio": extras["h_bio"],
            "J": extras["J"],
            "p_visible": extras["p_visible"],
            # Observation branch
            "lambda_pre_dropout": extras["lambda_pre_dropout"],
            "obs_beta": extras["obs_beta"],
            "log_visibility_bias": extras["log_visibility_bias"],
            "scale_dt": extras["scale_dt"],
            # Model params
            "phi": out["phi"],
        }

        return {
            key: self._detach_prediction_value(value)
            for key, value in predictions.items()
        }

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
        weight_decay_bio = self.config.optim.weight_decay_bio
        weight_decay_rest = self.config.optim.weight_decay_rest
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
                    "weight_decay": weight_decay_bio,
                    "name": "biological",
                }
            )

        if rest_params:
            param_groups.append(
                {
                    "params": rest_params,
                    "lr": rest_lr,
                    "weight_decay": weight_decay_rest,
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
