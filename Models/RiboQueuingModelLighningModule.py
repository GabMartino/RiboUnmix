from __future__ import annotations

import math
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

def _broadcast_profile_param(
    value: torch.Tensor,
    target_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    """Broadcast transcript-level or position-level parameters to [B, T]."""
    if value.ndim == len(target_shape) + 1 and value.shape[-1] == 1:
        value = value.squeeze(-1)

    if value.ndim == 1 and len(target_shape) == 2 and value.shape[0] == target_shape[0]:
        value = value.reshape(-1, 1)

    return torch.broadcast_to(value, target_shape)

class PoissonProfileLoss(nn.Module):
    """
    Poisson profile NLL.

    The target-only lgamma(y + 1) term is dropped, matching the optimized
    objective used for training:

        NLL = mu - y * log(mu)

    The dropped term does not affect gradients with respect to model
    parameters, but the logged NLL is not an absolute normalized likelihood.
    """

    def __init__(
        self,
        eps: float = 1.0e-8,
        mu_min: float = 1.0e-8,
        mu_max: float = 1.0e8,
    ):
        super().__init__()

        self.eps = float(eps)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)

    def positive_mean_from_params(
        self,
        *,
        mu: torch.Tensor,
        log_sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del log_sigma
        mu = torch.nan_to_num(
            mu,
            nan=self.mu_min,
            posinf=self.mu_max,
            neginf=self.mu_min,
        )
        return mu.clamp(min=self.mu_min, max=self.mu_max)

    def forward(
        self,
        mu_phys: torch.Tensor,
        log_sigma: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        del log_sigma

        with torch.amp.autocast(device_type=mu_phys.device.type, enabled=False):
            finite_mask = mask.bool() & torch.isfinite(y_true)
            y = torch.nan_to_num(
                y_true.to(torch.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)

            mu = self.positive_mean_from_params(
                mu=mu_phys.to(torch.float64),
                log_sigma=None,
            ).to(torch.float64)

            nll = (
                mu
                - y * torch.log(mu.clamp_min(self.eps))
            )
            nll = torch.nan_to_num(
                nll,
                nan=0.0,
                posinf=1.0e8,
                neginf=1.0e8,
            )

            mask_f = finite_mask.to(torch.float64)
            valid_len = mask_f.sum(dim=1).clamp_min(1.0)

            loss_per_sample = (nll * mask_f).sum(dim=1) / valid_len
            loss_per_sample = loss_per_sample.to(torch.float32)

        if return_per_sample:
            return loss_per_sample

        return loss_per_sample.mean()


class NegativeBinomialProfileLoss(nn.Module):
    """
    Negative Binomial profile NLL optimized for training.

    Parameterization:
        mu:
            predicted mean.
        log_alpha:
            log dispersion, alpha = exp(log_alpha), r = 1 / alpha.
        sequence_reduction:
            "mean" averages valid positions within each sequence before the
            batch reduction. "sum" sums valid positions within each sequence,
            then averages only across batch samples.

    The loss drops the target-only lgamma(y + 1) term, matching the optimized
    training objective supplied by the user. With non-integer averaged targets,
    the remaining gamma terms are evaluated by the continuous lgamma extension.
    """

    def __init__(
        self,
        eps: float = 1.0e-8,
        mu_min: float = 1.0e-8,
        mu_max: float = 1.0e8,
        log_alpha_min: float = -10.0,
        log_alpha_max: float = 10.0,
        sequence_reduction: str = "mean",
    ):
        super().__init__()

        self.eps = float(eps)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.log_alpha_min = float(log_alpha_min)
        self.log_alpha_max = float(log_alpha_max)
        if self.log_alpha_max <= self.log_alpha_min:
            raise ValueError(
                "log_alpha_max must be > log_alpha_min, "
                f"got {self.log_alpha_max} <= {self.log_alpha_min}."
            )
        if sequence_reduction not in {"mean", "sum"}:
            raise ValueError(
                "sequence_reduction must be one of {'mean', 'sum'}, "
                f"got {sequence_reduction!r}."
            )
        self.sequence_reduction = sequence_reduction

    def positive_mean_from_params(
        self,
        *,
        mu: torch.Tensor,
        log_sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del log_sigma
        mu = torch.nan_to_num(
            mu,
            nan=self.mu_min,
            posinf=self.mu_max,
            neginf=self.mu_min,
        )
        return mu.clamp(min=self.mu_min, max=self.mu_max)

    def _log_alpha_from_model_output(
        self,
        *,
        log_sigma: torch.Tensor,
        target_shape: torch.Size | tuple[int, ...],
    ) -> torch.Tensor:
        log_alpha = _broadcast_profile_param(
            log_sigma,
            target_shape,
        )
        log_alpha = torch.nan_to_num(
            log_alpha,
            nan=0.0,
            posinf=self.log_alpha_max,
            neginf=self.log_alpha_min,
        )
        return log_alpha.clamp(min=self.log_alpha_min, max=self.log_alpha_max)

    def forward(
        self,
        mu_phys: torch.Tensor,
        log_sigma: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=mu_phys.device.type, enabled=False):
            finite_mask = mask.bool() & torch.isfinite(y_true)
            y = torch.nan_to_num(
                y_true.to(torch.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)

            mu = self.positive_mean_from_params(
                mu=mu_phys.to(torch.float64),
                log_sigma=None,
            ).to(torch.float64)
            log_alpha = self._log_alpha_from_model_output(
                log_sigma=log_sigma.to(torch.float64),
                target_shape=y.shape,
            )
            r = torch.exp(-log_alpha).clamp_min(self.eps)

            # Full NB negative log-likelihood, including the lgamma(y+1)
            # normalization (the continuous generalization of log(y!)). This
            # term is constant w.r.t. the model parameters, so it does NOT
            # change gradients, but it makes the reported/combined NLL a proper
            # log-density. Without it the NLL magnitude is dominated by count
            # scale (large-count datasets look strongly negative regardless of
            # fit quality), which makes cross-dataset NLLs incomparable.
            nll = (
                torch.lgamma(r)
                - torch.lgamma(y + r)
                + torch.lgamma(y + 1.0)
                - r * torch.log(r.clamp_min(self.eps))
                - y * torch.log(mu.clamp_min(self.eps))
                + (r + y) * torch.log((r + mu).clamp_min(self.eps))
            )
            nll = torch.nan_to_num(
                nll,
                nan=0.0,
                posinf=1.0e8,
                neginf=1.0e8,
            )

            mask_f = finite_mask.to(torch.float64)
            valid_len = mask_f.sum(dim=1).clamp_min(1.0)

            nll = torch.where(finite_mask, nll, torch.zeros_like(nll))
            loss_per_sample = (nll * mask_f).sum(dim=1)
            if self.sequence_reduction == "mean":
                loss_per_sample = loss_per_sample / valid_len
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

    where support is unbounded rho_bio traffic intensity or queue-propagated
    q_bio traffic intensity depending on loss.pcc_target.
    """

    DATASET_DIAGNOSTIC_KEYS = (
        "rho_mean",
        "q_mean",
        "q_max",
        "rho_utilization_mean",
        "q_utilization_mean",
        "q_utilization_max",
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
        "log_sigma_reg_loss",
        "transcript_scale_centering_loss",
    )

    PROFILE_PLOT_GROUPS = (
        ("beta", ("obs_beta",)),
        ("log_sigma", ("log_sigma",)),
        ("rho", ("rho_bio",)),
        ("utilization", ("rho_utilization", "q_utilization")),
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

        # Per-dataset loss-scale normalization (ablation, see
        # "memory documents/dataset_loss_scale_normalization.md").
        # When enabled, the NLL term of dataset d is divided by a detached
        # EMA of its own NLL magnitude before datasets are combined, so each
        # dataset contributes a comparable gradient to the shared biological
        # parameters regardless of its absolute count scale.
        loss_cfg = self.config.loss
        # Power-space mu PCC for the OPTIMIZED loss term only (the reported
        # mu_pcc metric stays in raw space): raises values to an exponent > 1 so
        # peaks dominate the optimized correlation (peak / CSS recovery). See
        # loss.mu_pcc_power_space in the config.
        self.mu_pcc_power_space = bool(getattr(loss_cfg, "mu_pcc_power_space", False))
        self.mu_pcc_power_exponent = float(
            getattr(loss_cfg, "mu_pcc_power_exponent", 2.0)
        )
        self.dataset_loss_scale_norm_enabled = bool(
            getattr(loss_cfg, "dataset_loss_scale_normalization", False)
        )
        self.dataset_loss_scale_ema_decay = float(
            getattr(loss_cfg, "dataset_loss_scale_ema_decay", 0.99)
        )
        self.dataset_loss_scale_eps = float(
            getattr(loss_cfg, "dataset_loss_scale_eps", 1.0e-6)
        )
        num_datasets = (
            max(self.dataset_id_to_name) + 1 if self.dataset_id_to_name else 1
        )
        # Running per-dataset NLL magnitude (EMA). Persisted in checkpoints so
        # the normalization survives resume; updated during training only.
        self.register_buffer(
            "dataset_loss_scale",
            torch.ones(num_datasets, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "dataset_loss_scale_initialized",
            torch.zeros(num_datasets, dtype=torch.bool),
            persistent=True,
        )

        # Per-dataset plateau loss-weight schedule (soft per-task early stopping).
        # When a dataset's val_nll stops improving for `patience` validations,
        # its training loss weight is multiplied by `decay` (floored at
        # `min_weight`). The small/sparse dataset that overfits early thus stops
        # driving updates and its val curve flattens, while datasets still
        # improving keep weight 1.0. Updated in on_validation_epoch_end from the
        # logged val_nll/<name>. See
        # "memory documents/dataset_loss_scale_normalization.md".
        self.dataset_loss_weight_schedule_enabled = bool(
            getattr(loss_cfg, "dataset_loss_weight_schedule", False)
        )
        self.dataset_loss_weight_patience = int(
            getattr(loss_cfg, "dataset_loss_weight_patience", 3)
        )
        self.dataset_loss_weight_decay = float(
            getattr(loss_cfg, "dataset_loss_weight_decay", 0.5)
        )
        self.dataset_loss_weight_min = float(
            getattr(loss_cfg, "dataset_loss_weight_min", 0.1)
        )
        self.dataset_loss_weight_min_delta = float(
            getattr(loss_cfg, "dataset_loss_weight_min_delta", 0.0)
        )
        # w_d (applied to the loss), best val_nll seen, and the no-improvement
        # counter — per global dataset id, persisted so a resume keeps the
        # schedule state.
        self.register_buffer(
            "dataset_loss_weight",
            torch.ones(num_datasets, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "dataset_loss_weight_best_nll",
            torch.full((num_datasets,), float("inf"), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "dataset_loss_weight_bad_epochs",
            torch.zeros(num_datasets, dtype=torch.int64),
            persistent=True,
        )

        # Running per-dataset sample count (training only), used to scale the
        # optional hierarchical-shrinkage penalty so the smallest dataset shrinks
        # the most (the "partial pooling" inductive bias). See _dataset_pooling_loss.
        self.register_buffer(
            "dataset_seen_count",
            torch.zeros(num_datasets, dtype=torch.float64),
            persistent=True,
        )

        # Per-dataset EMA self-shrinkage of log_sigma (option 2a). Anchors each
        # dataset's dispersion to its OWN running history (not the cross-dataset
        # mean), so a head cannot drift to extremes late in training (overfit)
        # while legitimately different levels (e.g. kutay's sharp sigma) are kept.
        # See _dataset_log_sigma_self_target and the pooling guide.
        self.dataset_log_sigma_self_reg_weight = float(
            getattr(loss_cfg, "dataset_log_sigma_self_reg_weight", 0.0)
        )
        self.dataset_log_sigma_self_reg_ema_decay = float(
            getattr(loss_cfg, "dataset_log_sigma_self_reg_ema_decay", 0.99)
        )
        self.register_buffer(
            "dataset_log_sigma_ema",
            torch.zeros(num_datasets, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "dataset_log_sigma_ema_initialized",
            torch.zeros(num_datasets, dtype=torch.bool),
            persistent=True,
        )

    def _build_loss(self) -> nn.Module:
        loss_cfg = self.config.loss

        def loss_cfg_get(name: str, default: Any = None) -> Any:
            try:
                return getattr(loss_cfg, name)
            except (AttributeError, KeyError):
                return default

        likelihood = str(loss_cfg.profile_likelihood).lower()
        poisson_likelihoods = {"poisson", "poisson_pseudo"}
        nb_likelihoods = {"negative_binomial", "nb", "nb_pseudo"}
        valid_likelihoods = poisson_likelihoods | nb_likelihoods
        if likelihood not in valid_likelihoods:
            raise ValueError(
                "Unsupported profile likelihood. "
                f"Use one of {sorted(valid_likelihoods)}. "
                f"Got {loss_cfg.profile_likelihood!r}."
            )

        if likelihood in poisson_likelihoods:
            return PoissonProfileLoss(
                eps=loss_cfg.eps,
                mu_min=loss_cfg.mu_min,
                mu_max=loss_cfg.mu_max,
            )

        return NegativeBinomialProfileLoss(
            eps=loss_cfg.eps,
            mu_min=loss_cfg.mu_min,
            mu_max=loss_cfg.mu_max,
            log_alpha_min=float(loss_cfg_get("nb_log_alpha_min", -10.0)),
            log_alpha_max=float(loss_cfg_get("nb_log_alpha_max", 10.0)),
            sequence_reduction=str(loss_cfg_get("nb_sequence_reduction", "mean")),
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

        mu, log_sigma, extras = self.model(
            x_packed=seq_packed,
            codon_ids=codon_ids,
            id_datasets=dataset_ids,
            mask=mask,
            target=target,
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
            "log_sigma": log_sigma,
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
            log_sigma=out["log_sigma"].float(),
            y_true=out["target"].float(),
            mask=out["mask"].bool(),
            return_per_sample=True,
        )

    def _aggregate_per_sample(
        self,
        values: torch.Tensor,
        dataset_ids: torch.Tensor,
        _ds_split: "tuple[torch.Tensor, int] | None" = None,
        apply_scale_norm: bool = False,
        apply_task_weight: bool = False,
    ) -> torch.Tensor:
        """
        Combine a per-sample quantity into a single scalar.

        Three orthogonal axes (all ablatable via config):
          - dataset_balanced_loss: average per-dataset means with EQUAL weight
            per dataset (removes the sample-COUNT imbalance) instead of pooling
            all samples (which weights by count).
          - apply_scale_norm + dataset_loss_scale_normalization: divide each
            dataset's mean by a detached EMA of its own MAGNITUDE before
            combining (removes the gradient-MAGNITUDE imbalance). Only passed
            True for the NLL term; PCC/MSE are already comparable across
            datasets.
          - apply_task_weight + dataset_loss_weight_schedule: multiply each
            dataset's mean by its plateau loss weight w_d (soft per-task early
            stopping). Passed True only for the loss-component terms (mu_pcc_loss
            and the NLL), never for logged metrics. NOT renormalized: datasets
            still at w_d=1 keep their full gradient; a decayed dataset simply
            fades out. See "memory documents/dataset_loss_scale_normalization.md".
        """
        scale_norm = apply_scale_norm and self.dataset_loss_scale_norm_enabled
        task_weight = apply_task_weight and self.dataset_loss_weight_schedule_enabled
        balanced = bool(self.config.loss.dataset_balanced_loss)

        # Fast path: nothing to group by.
        if not balanced and not scale_norm and not task_weight:
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
        per_ds_mean = sums / counts.clamp_min(1.0)

        # sorted unique global ids correspond 1:1 to group order in `inverse`
        global_ids = (
            torch.unique(dataset_ids.to(device=values.device))
            if (scale_norm or task_weight)
            else None
        )

        if scale_norm:
            scale = self._dataset_loss_scale_for(global_ids, per_ds_mean)
            per_ds_mean = per_ds_mean / scale

        if task_weight:
            w = self.dataset_loss_weight[global_ids.long()].to(per_ds_mean.dtype)
            per_ds_mean = per_ds_mean * w

        if balanced:
            # equal weight per dataset
            return per_ds_mean.mean()
        # count-weighted recombination (preserves the un-balanced semantics
        # while still applying the per-dataset scale normalization)
        return (per_ds_mean * counts).sum() / counts.sum().clamp_min(1.0)

    def _dataset_loss_scale_for(
        self,
        global_ids: torch.Tensor,
        per_ds_mean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return a detached, positive per-dataset scale used to normalize the
        loss magnitude. During training, update the EMA buffer in place:
            first time a dataset is seen -> initialize to its current magnitude
            afterwards                   -> EMA of |per-dataset mean|.
        During val/test the buffer is frozen, so the same (training-derived)
        normalization is applied — keeping val metrics comparable to train.
        """
        ids = global_ids.long()
        magnitude = per_ds_mean.detach().abs().to(self.dataset_loss_scale.dtype)

        if self.training:
            decay = self.dataset_loss_scale_ema_decay
            current = self.dataset_loss_scale[ids]
            seen = self.dataset_loss_scale_initialized[ids]
            updated = torch.where(
                seen,
                decay * current + (1.0 - decay) * magnitude,
                magnitude,
            )
            self.dataset_loss_scale[ids] = updated
            self.dataset_loss_scale_initialized[ids] = torch.ones_like(seen)

        scale = self.dataset_loss_scale[ids].clamp_min(self.dataset_loss_scale_eps)
        return scale.to(per_ds_mean.dtype)

    def _dataset_pool_size_weights(
        self,
        global_ids: torch.Tensor,
        dataset_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Detached per-dataset shrinkage weights `s_d = (mean_count / N_d)^exponent`,
        so smaller datasets are shrunk harder toward the shared baseline (the
        hierarchical `theta_d ~ N(theta_global, tau^2)` prior gets stronger as
        N_d shrinks). `exponent = 0` recovers a uniform ridge (no size scaling).
        N_d is estimated online from the training stream so no datamodule
        plumbing is needed.
        """
        if self.training:
            counts = torch.bincount(
                dataset_ids.long(), minlength=self.dataset_seen_count.numel()
            ).to(self.dataset_seen_count.dtype)
            self.dataset_seen_count += counts

        exponent = float(getattr(self.config.loss, "dataset_pool_size_exponent", 1.0))
        c = self.dataset_seen_count[global_ids.long()].clamp_min(1.0)
        if exponent == 0.0:
            return torch.ones_like(c, dtype=torch.float32)
        w = (c.mean() / c).pow(exponent)
        return w.detach().to(torch.float32)

    def _dataset_pooling_loss(
        self,
        out: dict[str, Any],
        dataset_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optional hierarchical-shrinkage (partial-pooling) penalty on the
        per-dataset heads — a pure loss add-on; touches no head class.

        - Visibility: pulls each dataset's `log_visibility_bias` toward 0
          (beta -> 1, the shared "no idiosyncratic visibility bias" baseline).
        - Dispersion (cross-dataset): pulls each dataset's mean `log_sigma`
          toward the detached cross-dataset mean (datasets share a dispersion
          level; the small one borrows it). UNSAFE when a dataset's dispersion is
          legitimately different — see the pooling guide.
        - Dispersion (self, option 2a): pulls each dataset's mean `log_sigma`
          toward its OWN running EMA, so the head cannot drift to extremes late
          in training (overfit) while each dataset keeps its own level. No
          cross-dataset coupling, so it is safe even when levels differ.

        Each dataset's term is weighted by `_dataset_pool_size_weights`, then a
        size-weighted mean is taken so the strength stays interpretable and the
        smallest dataset is shrunk the most. Off (returns 0) when all weights
        are 0. See "memory documents/hierarchical_pooling_hyperparameters.md".
        """
        target = out["target"]
        zero = torch.zeros((), device=target.device, dtype=torch.float32)

        vw = float(getattr(self.config.loss, "dataset_pool_visibility_weight", 0.0))
        sw = float(getattr(self.config.loss, "dataset_pool_log_sigma_weight", 0.0))
        self_w = self.dataset_log_sigma_self_reg_weight
        if vw <= 0.0 and sw <= 0.0 and self_w <= 0.0:
            return zero

        mask_f = out["mask"].bool().to(torch.float32)
        valid_len = mask_f.sum(dim=1).clamp_min(1.0)  # [B]

        global_ids, inverse = torch.unique(
            dataset_ids.to(device=target.device), return_inverse=True
        )
        K = int(global_ids.numel())
        size_w = self._dataset_pool_size_weights(global_ids, dataset_ids)
        size_w_sum = size_w.sum().clamp_min(1.0e-8)

        def _per_dataset_mean(per_sample: torch.Tensor) -> torch.Tensor:
            sums = torch.zeros(K, device=per_sample.device, dtype=per_sample.dtype)
            sums = sums.scatter_add_(0, inverse, per_sample)
            counts = torch.bincount(inverse, minlength=K).to(per_sample.dtype)
            return sums / counts.clamp_min(1.0)

        loss = zero
        if vw > 0.0:
            log_bias = out["extras"]["log_visibility_bias"].to(torch.float32)
            # mean bias^2 over valid positions per transcript -> per dataset
            per_sample = (log_bias.pow(2) * mask_f).sum(dim=1) / valid_len
            per_ds = _per_dataset_mean(per_sample)
            loss = loss + vw * (size_w * per_ds).sum() / size_w_sum

        if sw > 0.0 or self_w > 0.0:
            log_sigma = out["extras"]["log_sigma"].to(torch.float32)
            per_sample_ls = (log_sigma * mask_f).sum(dim=1) / valid_len
            per_ds_ls = _per_dataset_mean(per_sample_ls)

            if sw > 0.0:
                global_ls = per_ds_ls.mean().detach()  # pooled target (stop-grad)
                loss = loss + sw * (size_w * (per_ds_ls - global_ls).pow(2)).sum() / size_w_sum

            if self_w > 0.0:
                # Anchor to each dataset's OWN EMA history (detached) — resists
                # late overfit drift without coupling datasets together.
                self_target = self._dataset_log_sigma_self_target(global_ids, per_ds_ls)
                loss = loss + self_w * (size_w * (per_ds_ls - self_target).pow(2)).sum() / size_w_sum

        return loss

    def _dataset_log_sigma_self_target(
        self,
        global_ids: torch.Tensor,
        per_ds_ls: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the detached per-dataset EMA of mean log_sigma used as the
        self-shrinkage target, and (training only) advance the EMA toward the
        current value. First time a dataset is seen the target equals its current
        value (no penalty), so the anchor never injects a transient. The penalty
        `(per_ds_ls - target)^2` therefore measures drift away from the dataset's
        own recent history; a slower `ema_decay` gives a longer-memory leash.
        """
        ids = global_ids.long()
        value = per_ds_ls.detach().to(self.dataset_log_sigma_ema.dtype)
        seen = self.dataset_log_sigma_ema_initialized[ids]
        old = self.dataset_log_sigma_ema[ids]
        # pre-update target: old EMA where seen, else current value
        target = torch.where(seen, old, value)

        if self.training:
            decay = self.dataset_log_sigma_self_reg_ema_decay
            updated = torch.where(seen, decay * old + (1.0 - decay) * value, value)
            self.dataset_log_sigma_ema[ids] = updated
            self.dataset_log_sigma_ema_initialized[ids] = torch.ones_like(seen)

        return target.to(per_ds_ls.dtype)

    def _beta_cross_dataset_center_loss(
        self,
        out: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-dataset visibility-bias neutrality (identifiability gauge).

        For a transcript t measured in datasets D(t), the centered log
        visibility bias is required to average to zero across datasets at every
        position:

            mean_{d in D(t)} log beta_{d,t,i} = 0   for all i.

        Any per-position shape that is COMMON to all datasets is thereby forced
        out of the bias branch and into the shared biological factor rho (the
        only factor that can still fit it), keeping biology out of beta.
        Dataset-SPECIFIC deviations are unconstrained (they can arrange a zero
        cross-dataset mean). Only transcripts present in >= 2 datasets within the
        batch contribute, so a transcript-grouped sampler
        (train_sampling_strategy=transcript_grouped_multidataset_pairs) is
        required for the same transcript's dataset copies to co-occur in a batch.

        Returns (loss, mean_abs_diag), both 0-dim tensors. Uses the centered
        log_visibility_bias (the quantity that actually forms beta in mu), so it
        is in the same gauge as the prediction.
        """
        log_vis = out["extras"]["log_visibility_bias"].float()  # [B, T] centered, masked
        mask_f = out["mask"].bool().to(dtype=log_vis.dtype)      # [B, T]
        device = log_vis.device
        zero = torch.zeros((), device=device, dtype=log_vis.dtype)

        ids = out.get("ids")
        B, T = log_vis.shape
        if ids is None or len(ids) != B or B == 0:
            return zero, zero

        # Transcript identity is a list of string ids; map to integer group idx.
        id_to_group: dict[str, int] = {}
        inv: list[int] = []
        for x in ids:
            key = str(x)
            g = id_to_group.get(key)
            if g is None:
                g = len(id_to_group)
                id_to_group[key] = g
            inv.append(g)
        G = len(id_to_group)
        group_idx = torch.as_tensor(inv, device=device, dtype=torch.long)  # [B]

        # Per (transcript group, position): sum and count of valid log beta.
        sum_gt = torch.zeros((G, T), device=device, dtype=log_vis.dtype)
        cnt_gt = torch.zeros((G, T), device=device, dtype=log_vis.dtype)
        sum_gt.index_add_(0, group_idx, log_vis * mask_f)
        cnt_gt.index_add_(0, group_idx, mask_f)
        mean_gt = sum_gt / cnt_gt.clamp_min(1.0)                 # [G, T] cross-dataset mean
        valid_gt = (cnt_gt > 0).to(log_vis.dtype)

        # rows-per-group == distinct datasets per group (flat pairs are unique
        # (transcript, dataset)); single-dataset transcripts give no meaningful
        # cross-dataset mean and must be excluded.
        rows_per_group = torch.zeros((G,), device=device, dtype=log_vis.dtype)
        rows_per_group.index_add_(
            0, group_idx, torch.ones((B,), device=device, dtype=log_vis.dtype)
        )
        qualifying = (rows_per_group >= 2.0).to(log_vis.dtype)   # [G]
        denom = qualifying.sum()
        if float(denom) == 0.0:
            return zero, zero

        pos_per_group = valid_gt.sum(dim=1).clamp_min(1.0)       # [G]
        per_group_sq = (mean_gt.pow(2) * valid_gt).sum(dim=1) / pos_per_group   # [G]
        per_group_abs = (mean_gt.abs() * valid_gt).sum(dim=1) / pos_per_group   # [G]

        loss = (per_group_sq * qualifying).sum() / denom
        diag = (per_group_abs * qualifying).sum() / denom
        return loss, diag

    def _pearson_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1.0e-8,
        power: float = 1.0,
    ) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        mask_b = mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
        mask_f = mask_b.to(dtype=pred.dtype)

        pred = torch.where(mask_b, pred, torch.zeros_like(pred)) * mask_f
        target = torch.where(mask_b, target, torch.zeros_like(target)) * mask_f

        # Optional power transform on the non-negative profiles before
        # correlating (the reported mu_pcc passes power=1.0 and stays raw).
        # power > 1 gives peak emphasis -- high positions dominate the fit
        # (CSS / stalling-site recovery). 0^p = 0 keeps masked positions at zero.
        if power != 1.0:
            p = max(float(power), eps)
            pred = pred.clamp_min(0.0).pow(p) * mask_f
            target = target.clamp_min(0.0).pow(p) * mask_f

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
        log_sigma = out["log_sigma"].detach().float()

        with torch.no_grad():
            positive_mean = self.loss_fn.positive_mean_from_params(
                mu=mu,
                log_sigma=log_sigma,
            ).float()

        return {
            "likelihood_positive_mean": positive_mean,
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

        # With a single active dataset the per-dataset series duplicates the
        # aggregate (e.g. val_mu_pcc == val_mu_pcc/<name>). Skip the duplication.
        if unique_dataset_ids.numel() <= 1:
            return

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
                "log_sigma_mean": self._mean_for_dataset(
                    out["log_sigma"],
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
        pred = pred.float()
        target = target.float()
        mask_b = mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
        mask_f = mask_b.to(dtype=pred.dtype)
        valid_len = mask_f.sum(dim=1).clamp_min(1.0)

        pred = torch.where(mask_b, pred, torch.zeros_like(pred)).clamp_min(0.0)
        target = torch.where(mask_b, target, torch.zeros_like(target)).clamp_min(0.0)

        if log1p:
            pred = torch.log1p(pred)
            target = torch.log1p(target)

        return ((pred - target).pow(2) * mask_f).sum(dim=1) / valid_len

    def _zero_diagnostics_per_sample(
        self,
        out: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        target = out["target"].float()
        mask = out["mask"].bool() & torch.isfinite(target)
        mask_f = mask.to(dtype=target.dtype)

        zero_count = (
            (target <= float(self.config.loss.censor_threshold)) & mask
        ).to(dtype=target.dtype).sum(dim=1)
        valid_len = mask_f.sum(dim=1).clamp_min(1.0)

        return {"target_zero_frac_per_sample": zero_count / valid_len}

    @staticmethod
    def _masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_b = mask.bool() & torch.isfinite(values)
        values = torch.where(mask_b, values, torch.zeros_like(values))
        return values.sum()

    @classmethod
    def _masked_mean(cls, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_b = mask.bool() & torch.isfinite(values)
        denom = mask_b.to(dtype=values.dtype).sum().clamp_min(1.0)
        return cls._masked_sum(values, mask_b) / denom

    @classmethod
    def _masked_std(cls, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_b = mask.bool() & torch.isfinite(values)
        mean = cls._masked_mean(values, mask_b)
        centered = torch.where(mask_b, values - mean, torch.zeros_like(values))
        denom = mask_b.to(dtype=values.dtype).sum().clamp_min(1.0)
        return torch.sqrt(centered.pow(2).sum() / denom)

    @staticmethod
    def _masked_fraction(mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = valid_mask.bool()
        dtype = torch.float32
        denom = valid_mask.to(dtype=dtype).sum().clamp_min(1.0)
        return (mask.bool() & valid_mask).to(dtype=dtype).sum() / denom

    # Quantile cut-points (of positive ground truth) that define the positive-side
    # calibration bins. The low bin is fixed by the censor threshold; the positive
    # side is partitioned by these quantiles of valid positive y in the batch.
    _POSITIVE_BIN_QUANTILES = (0.5, 0.75, 0.9, 0.99)

    @classmethod
    def _add_target_bin_calibration_metrics(
        cls,
        *,
        metrics: dict[str, torch.Tensor],
        bins: tuple[tuple[str, torch.Tensor], ...],
        y: torch.Tensor,
        pred: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """
        Per-target-bin calibration view:
          - mean_target_by_target_bin_<label>: average observed y in the bin
          - mean_pred_by_target_bin_<label>:   average predicted mu in the bin
          - log1p_mse_by_target_bin_<label>:   MSE on log1p scale (comparable
                                               across bins of different scale)
          - target_bin_<label>_frac:           fraction of valid positions in bin

        Calibration read: if mean_pred ~ mean_target per bin the model extracts
        that magnitude regime correctly. log1p_mse quantifies spread on a scale
        where the zero bin and the top-1% bin are directly comparable.
        """
        # Match dtypes; pred may be float32 while y is float64.
        pred_d = pred.to(dtype=y.dtype) if pred.dtype != y.dtype else pred
        log1p_sq_err = (torch.log1p(pred_d.clamp_min(0.0))
                        - torch.log1p(y.clamp_min(0.0))).pow(2)

        for label, bin_mask_raw in bins:
            bin_mask = valid_mask & bin_mask_raw & torch.isfinite(y) & torch.isfinite(pred_d)
            metrics[f"mean_target_by_target_bin_{label}"] = cls._masked_mean(
                y, bin_mask
            ).float()
            metrics[f"mean_pred_by_target_bin_{label}"] = cls._masked_mean(
                pred_d, bin_mask
            ).float()
            metrics[f"log1p_mse_by_target_bin_{label}"] = cls._masked_mean(
                log1p_sq_err, bin_mask
            ).float()
            metrics[f"target_bin_{label}_frac"] = cls._masked_fraction(
                bin_mask, valid_mask
            ).to(device=pred_d.device)

    @classmethod
    def _add_positive_quantile_threshold_metrics(
        cls,
        *,
        metrics: dict[str, torch.Tensor],
        y: torch.Tensor,
        valid_mask: torch.Tensor,
        threshold: float,
    ) -> None:
        q_vals = cls._positive_quantile_thresholds(
            y=y,
            valid_mask=valid_mask,
            threshold=threshold,
        )
        if q_vals is None:
            return
        for q, v in zip(cls._POSITIVE_BIN_QUANTILES, q_vals, strict=True):
            label = f"q{int(round(q * 100)):02d}"
            metrics[f"target_pos_{label}_threshold"] = torch.tensor(
                float(v), device=y.device, dtype=torch.float32
            )

    @classmethod
    def _positive_quantile_thresholds(
        cls,
        *,
        y: torch.Tensor,
        valid_mask: torch.Tensor,
        threshold: float,
    ) -> tuple[float, ...] | None:
        positive_mask = valid_mask & (y > threshold) & torch.isfinite(y)
        positive_y = y[positive_mask]
        if positive_y.numel() < max(int(1.0 / (1.0 - cls._POSITIVE_BIN_QUANTILES[-1])), 16):
            return None
        qs = torch.tensor(
            cls._POSITIVE_BIN_QUANTILES,
            device=positive_y.device,
            dtype=positive_y.dtype,
        )
        return tuple(torch.quantile(positive_y, qs).tolist())

    @classmethod
    def _positive_quantile_bins(
        cls,
        *,
        values: torch.Tensor,
        y: torch.Tensor,
        valid_mask: torch.Tensor,
        threshold: float,
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        """
        Build target-magnitude bins for calibration diagnostics.

        - One zero/censored bin (`le_c`) at y <= threshold.
        - Positive-side bins partitioned by quantiles of the in-batch positive y,
          so the high-value bin is "actually high values" rather than a fixed
          count cutoff that may be dataset-inappropriate.

        Quantile thresholds are recomputed per batch; with the default sampler
        (~16 sequences x O(1k) positions) the per-batch p50..p99 are stable
        enough for epoch aggregation.
        """
        q_vals = cls._positive_quantile_thresholds(
            y=y,
            valid_mask=valid_mask,
            threshold=threshold,
        )
        if q_vals is None:
            return (
                ("le_c", values <= threshold),
                ("positive", values > threshold),
            )
        q50, q75, q90, q99 = q_vals

        return (
            ("le_c", values <= threshold),
            ("pos_q0_50", (values > threshold) & (values <= q50)),
            ("pos_q50_75", (values > q50) & (values <= q75)),
            ("pos_q75_90", (values > q75) & (values <= q90)),
            ("pos_q90_99", (values > q90) & (values <= q99)),
            ("pos_gt_q99", values > q99),
        )

    def _likelihood_position_diagnostics(
        self,
        out: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if isinstance(self.loss_fn, NegativeBinomialProfileLoss):
            return self._negative_binomial_position_diagnostics(out)

        if isinstance(self.loss_fn, PoissonProfileLoss):
            return self._poisson_position_diagnostics(out)

        return {}

    def _poisson_position_diagnostics(
        self,
        out: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad(), torch.amp.autocast(
            device_type=out["mu"].device.type,
            enabled=False,
        ):
            target_raw = out["target"].float()
            valid_mask = out["mask"].bool() & torch.isfinite(target_raw)
            y = torch.nan_to_num(
                target_raw.to(torch.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)
            mu = self.loss_fn.positive_mean_from_params(
                mu=out["mu"].float().to(torch.float64),
                log_sigma=None,
            ).to(torch.float64)

            nll = (
                mu
                - y * torch.log(mu.clamp_min(self.loss_fn.eps))
            )
            nll = torch.nan_to_num(nll, nan=0.0, posinf=1.0e8, neginf=1.0e8)
            valid_mask = valid_mask & torch.isfinite(nll)

            threshold = float(getattr(self.config.loss, "censor_threshold", 0.0))
            low_mask = valid_mask & (y <= threshold)
            positive_mask = valid_mask & (y > threshold)
            valid_count = valid_mask.to(dtype=torch.float64).sum().clamp_min(1.0)

            residual = y - mu
            pearson_residual = residual / torch.sqrt(mu.clamp_min(self.loss_fn.eps))

            metrics: dict[str, torch.Tensor] = {
                "nll_censored": self._masked_mean(nll, low_mask).float(),
                "nll_positive": self._masked_mean(nll, positive_mask).float(),
                "nll_censored_contrib": (
                    self._masked_sum(nll, low_mask) / valid_count
                ).float(),
                "nll_positive_contrib": (
                    self._masked_sum(nll, positive_mask) / valid_count
                ).float(),
                "nll_censored_frac": self._masked_fraction(
                    low_mask,
                    valid_mask,
                ).to(device=nll.device),
                "nll_positive_frac": self._masked_fraction(
                    positive_mask,
                    valid_mask,
                ).to(device=nll.device),
                "poisson_residual_mean": self._masked_mean(
                    residual,
                    valid_mask,
                ).float(),
                "poisson_abs_residual_mean": self._masked_mean(
                    residual.abs(),
                    valid_mask,
                ).float(),
                "poisson_pearson_residual_mean": self._masked_mean(
                    pearson_residual,
                    valid_mask,
                ).float(),
                "poisson_pearson_residual_std": self._masked_std(
                    pearson_residual,
                    valid_mask,
                ).float(),
                "poisson_abs_pearson_residual_mean": self._masked_mean(
                    pearson_residual.abs(),
                    valid_mask,
                ).float(),
                "pred_mean_censored_mean": self._masked_mean(
                    mu,
                    low_mask,
                ).float(),
                "pred_mean_positive_mean": self._masked_mean(
                    mu,
                    positive_mask,
                ).float(),
            }

            bins = self._positive_quantile_bins(
                values=y,
                y=y,
                valid_mask=valid_mask,
                threshold=threshold,
            )
            self._add_target_bin_calibration_metrics(
                metrics=metrics,
                bins=bins,
                y=y,
                pred=mu,
                valid_mask=valid_mask,
            )
            self._add_positive_quantile_threshold_metrics(
                metrics=metrics,
                y=y,
                valid_mask=valid_mask,
                threshold=threshold,
            )

            return metrics

    def _negative_binomial_position_diagnostics(
        self,
        out: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad(), torch.amp.autocast(
            device_type=out["mu"].device.type,
            enabled=False,
        ):
            target_raw = out["target"].float()
            valid_mask = out["mask"].bool() & torch.isfinite(target_raw)
            y = torch.nan_to_num(
                target_raw.to(torch.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)
            mu = self.loss_fn.positive_mean_from_params(
                mu=out["mu"].float().to(torch.float64),
                log_sigma=None,
            ).to(torch.float64)
            log_alpha = self.loss_fn._log_alpha_from_model_output(
                log_sigma=out["log_sigma"].float().to(torch.float64),
                target_shape=y.shape,
            )
            alpha = torch.exp(log_alpha)
            r = torch.exp(-log_alpha).clamp_min(self.loss_fn.eps)

            nll = (
                torch.lgamma(r)
                - torch.lgamma(y + r)
                - r * torch.log(r.clamp_min(self.loss_fn.eps))
                - y * torch.log(mu.clamp_min(self.loss_fn.eps))
                + (r + y) * torch.log((r + mu).clamp_min(self.loss_fn.eps))
            )
            nll = torch.nan_to_num(nll, nan=0.0, posinf=1.0e8, neginf=1.0e8)
            valid_mask = valid_mask & torch.isfinite(nll)

            threshold = float(getattr(self.config.loss, "censor_threshold", 0.0))
            low_mask = valid_mask & (y <= threshold)
            positive_mask = valid_mask & (y > threshold)
            valid_count = valid_mask.to(dtype=torch.float64).sum().clamp_min(1.0)

            residual = y - mu
            variance = mu + alpha * mu.pow(2)
            pearson_residual = residual / torch.sqrt(variance.clamp_min(self.loss_fn.eps))

            metrics: dict[str, torch.Tensor] = {
                "nll_censored": self._masked_mean(nll, low_mask).float(),
                "nll_positive": self._masked_mean(nll, positive_mask).float(),
                "nll_censored_contrib": (
                    self._masked_sum(nll, low_mask) / valid_count
                ).float(),
                "nll_positive_contrib": (
                    self._masked_sum(nll, positive_mask) / valid_count
                ).float(),
                "nll_censored_frac": self._masked_fraction(
                    low_mask,
                    valid_mask,
                ).to(device=nll.device),
                "nll_positive_frac": self._masked_fraction(
                    positive_mask,
                    valid_mask,
                ).to(device=nll.device),
                "nb_alpha_mean": self._masked_mean(alpha, valid_mask).float(),
                "nb_alpha_censored_mean": self._masked_mean(
                    alpha,
                    low_mask,
                ).float(),
                "nb_alpha_positive_mean": self._masked_mean(
                    alpha,
                    positive_mask,
                ).float(),
                "nb_r_mean": self._masked_mean(r, valid_mask).float(),
                "nb_residual_mean": self._masked_mean(
                    residual,
                    valid_mask,
                ).float(),
                "nb_abs_residual_mean": self._masked_mean(
                    residual.abs(),
                    valid_mask,
                ).float(),
                "nb_pearson_residual_mean": self._masked_mean(
                    pearson_residual,
                    valid_mask,
                ).float(),
                "nb_pearson_residual_std": self._masked_std(
                    pearson_residual,
                    valid_mask,
                ).float(),
                "nb_abs_pearson_residual_mean": self._masked_mean(
                    pearson_residual.abs(),
                    valid_mask,
                ).float(),
                "pred_mean_censored_mean": self._masked_mean(
                    mu,
                    low_mask,
                ).float(),
                "pred_mean_positive_mean": self._masked_mean(
                    mu,
                    positive_mask,
                ).float(),
            }

            bins = self._positive_quantile_bins(
                values=y,
                y=y,
                valid_mask=valid_mask,
                threshold=threshold,
            )
            self._add_target_bin_calibration_metrics(
                metrics=metrics,
                bins=bins,
                y=y,
                pred=mu,
                valid_mask=valid_mask,
            )
            self._add_positive_quantile_threshold_metrics(
                metrics=metrics,
                y=y,
                valid_mask=valid_mask,
                threshold=threshold,
            )

            return metrics

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

        target_raw = target.float()
        mask_b = mask.bool()
        finite_mask = mask_b & torch.isfinite(target_raw)
        target_f = torch.where(
            finite_mask,
            target_raw.clamp_min(0.0),
            torch.zeros_like(target_raw),
        )

        if strategy in {"max", "maximum"}:
            mask_f = finite_mask.to(dtype=target_f.dtype)
            return (target_f * mask_f).max(dim=1, keepdim=True).values.clamp_min(
                min_scale
            )

        if strategy in {"nonzero_quantile", "quantile_nonzero"}:
            q = float(getattr(self.config.loss, "support_target_quantile", 0.99))
            q = min(max(q, 0.0), 1.0)
            scales = []
            for target_i, mask_i in zip(target_f, finite_mask, strict=True):
                valid = mask_i & (target_i > 0.0)
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

        # Quantity correlated against normalized profile shape. Two axes:
        #   intensity (rho_bio / q_bio) — unbounded, full dynamic range
        #   occupancy (rho_utilization / q_utilization) — bounded in [0, 1),
        #                                                 Poisson "at least one
        #                                                 footprint" probability
        # and queue propagation off / on. q_bio == rho_bio and q_utilization ==
        # rho_utilization when propagation is off.
        support_key = {
            "rho": "rho_bio",                  # intensity, no queue
            "q": "q_bio",                      # intensity, queue-propagated
            "occupancy": "rho_utilization",    # bounded occupancy, no queue
            "q_occupancy": "q_utilization",    # bounded occupancy, queue-propagated
        }
        pcc_target = str(self.config.loss.pcc_target)
        if pcc_target not in support_key:
            raise ValueError(
                f"loss.pcc_target must be one of {sorted(support_key)}, "
                f"got {pcc_target!r}."
            )
        support = out["extras"][support_key[pcc_target]]

        target_raw = out["target"].float()
        mask = out["mask"].bool() & torch.isfinite(target_raw)
        target = torch.nan_to_num(
            target_raw,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        mask_f = mask.to(dtype=support.dtype)

        # Robust per-transcript normalization for the support-shape PCC target.
        # nonzero_quantile avoids letting one noisy extreme footprint set the
        # scale for the whole profile; values above the scale are winsorized to 1.
        target_scale = self._support_target_scale(target=target, mask=mask)
        target_norm = (target.float() / target_scale * mask_f).clamp(0.0, 1.0)

        with torch.no_grad():
            likelihood_positive_mean = self.loss_fn.positive_mean_from_params(
                mu=out["mu"].float(),
                log_sigma=out["log_sigma"].float(),
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
        # Quantity that actually drives the loss. Identical to mu_pcc_per_sample
        # unless power-space (peak emphasis) is configured. The raw mu_pcc above
        # remains the reported metric / val monitor.
        if self.mu_pcc_power_space and self.mu_pcc_power_exponent != 1.0:
            mu_pcc_for_loss_per_sample = self._pearson_per_sample(
                pred=likelihood_positive_mean,
                target=target.float(),
                mask=mask,
                power=self.mu_pcc_power_exponent,
            )
        else:
            mu_pcc_for_loss_per_sample = mu_pcc_per_sample
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
        mu_pcc_loss_per_sample = 1.0 - mu_pcc_for_loss_per_sample
        pcc_loss = self._aggregate_per_sample(
            pcc_loss_per_sample,
            dataset_ids,
            _ds_split,
        )
        mu_pcc_loss = self._aggregate_per_sample(
            mu_pcc_loss_per_sample,
            dataset_ids,
            _ds_split,
            apply_task_weight=True,
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

        log_sigma_reg_weight = float(getattr(self.config.loss, "log_sigma_reg_weight", 0.0))
        log_sigma_reg_loss = torch.zeros((), device=support.device, dtype=support.dtype)
        if log_sigma_reg_weight > 0.0 and not isinstance(self.loss_fn, PoissonProfileLoss):
            log_sigma_t = out["extras"]["log_sigma_t"].float()  # [B, 1]
            # Magnitude (L2-toward-zero) shrinkage: keep the dispersion low
            # without anchoring it to a specific target value. log_sigma is the
            # NB log_alpha, so this discourages large overdispersion (the main
            # NLL overfitting channel) while leaving its sign/shape free.
            log_sigma_reg_loss = log_sigma_reg_weight * log_sigma_t.pow(2).mean()
            metrics["log_sigma_reg_loss"] = log_sigma_reg_loss

        ts_centering_weight = float(getattr(self.config.loss, "transcript_scale_centering_weight", 0.0))
        transcript_scale_centering_loss = torch.zeros((), device=support.device, dtype=support.dtype)
        scale_dt_is_target = out["extras"].get("scale_dt_is_target")
        using_target_scale = (
            scale_dt_is_target is not None
            and bool((scale_dt_is_target.float() > 0.5).all().item())
        )
        if ts_centering_weight > 0.0 and not using_target_scale:
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

        # Optional hierarchical-shrinkage (partial pooling) of the per-dataset
        # heads. Zero unless dataset_pool_*_weight > 0. Shrinks small datasets
        # harder toward the shared baseline so they cannot overfit their own
        # visibility/dispersion heads.
        dataset_pool_loss = self._dataset_pooling_loss(out, dataset_ids).to(support.dtype)
        if dataset_pool_loss.requires_grad or float(dataset_pool_loss) != 0.0:
            metrics["dataset_pool_loss"] = dataset_pool_loss

        # Cross-dataset beta-neutrality (identifiability gauge): forces per-position
        # profile shape that is COMMON across datasets out of the bias branch and
        # into the shared biological factor rho. Requires a transcript-grouped
        # sampler. Zero (no behavior change) unless the weight is > 0.
        beta_xdc_weight = float(
            getattr(self.config.loss, "beta_cross_dataset_center_weight", 0.0)
        )
        beta_xdc_loss = torch.zeros((), device=support.device, dtype=support.dtype)
        if beta_xdc_weight > 0.0:
            raw_xdc, xdc_diag = self._beta_cross_dataset_center_loss(out)
            beta_xdc_loss = (beta_xdc_weight * raw_xdc).to(support.dtype)
            metrics["beta_cross_dataset_center"] = beta_xdc_loss
            metrics["beta_cross_dataset_logabs"] = xdc_diag.to(support.dtype)

        if mode == "pcc_only":
            metrics["loss"] = mu_pcc_loss + j_centering_loss + log_sigma_reg_loss + transcript_scale_centering_loss + dataset_pool_loss + beta_xdc_loss
            metrics["loss_per_sample"] = (
                mu_pcc_loss_per_sample
            )
            return metrics

        nll_per_sample = self._profile_nll_per_sample(out)
        # Raw, interpretable aggregate NLL — logged as <stage>_nll and directly
        # comparable to the per-dataset <stage>_nll/<name> series (it is their
        # equal- or count-weighted mean depending on dataset_balanced_loss).
        # Does NOT touch the per-dataset EMA buffers.
        nll = self._aggregate_per_sample(nll_per_sample, dataset_ids, _ds_split)
        # Scale-normalized aggregate actually summed into the optimized loss.
        # Identical to `nll` when dataset_loss_scale_normalization is false;
        # otherwise each dataset is divided by its detached EMA magnitude (this
        # call is what updates the EMA buffers). Kept separate from `nll` so the
        # logged diagnostic stays interpretable.
        nll_for_loss = self._aggregate_per_sample(
            nll_per_sample, dataset_ids, _ds_split,
            apply_scale_norm=True, apply_task_weight=True
        )

        metrics["nll"] = nll
        metrics["nll_per_sample"] = nll_per_sample
        if self.dataset_loss_scale_norm_enabled:
            # What the optimizer actually sees (in normalized units, ~O(1)).
            metrics["nll_scaled"] = nll_for_loss
        metrics.update(self._likelihood_position_diagnostics(out))

        if mode == "likelihood_only":
            metrics["loss"] = nll_for_loss + j_centering_loss + log_sigma_reg_loss + transcript_scale_centering_loss + dataset_pool_loss + beta_xdc_loss
            metrics["loss_per_sample"] = nll_per_sample
            return metrics

        if mode == "pcc_likelihood":
            metrics["loss"] = mu_pcc_loss + nll_for_loss + j_centering_loss + log_sigma_reg_loss + transcript_scale_centering_loss + dataset_pool_loss + beta_xdc_loss
            metrics["loss_per_sample"] = (
                mu_pcc_loss_per_sample
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
            on_step=False,
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
        # This also logs detached likelihood diagnostics such as branch NLLs
        # and target/prediction-bin contributions.
        already_logged = {"loss", "mu_pcc", "support_pcc"}
        for name, value in metrics.items():
            if name in already_logged or name.endswith("_per_sample"):
                continue
            if not torch.is_tensor(value) or value.ndim != 0:
                continue

            self.log(
                f"{stage}_{name}",
                value,
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
            f"{stage}_log_sigma_mean",
            out["log_sigma"].float()[out["mask"].bool()].mean(),
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
        extras["log_sigma"] = out["log_sigma"]
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
                log_sigma=out["log_sigma"].detach().float(),
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
            "rho_utilization": extras["rho_utilization"],
            "q_utilization": extras["q_utilization"],
            "J": extras["J"],
            "p_visible": extras["p_visible"],
            # Observation branch
            "lambda_pre_dropout": extras["lambda_pre_dropout"],
            "obs_beta": extras["obs_beta"],
            "log_visibility_bias": extras["log_visibility_bias"],
            "scale_dt": extras["scale_dt"],
            # Model params
            "log_sigma": out["log_sigma"],
        }

        return {
            key: self._detach_prediction_value(value)
            for key, value in predictions.items()
        }

    def on_validation_epoch_end(self) -> None:
        # Log a scale-free, cross-dataset-comparable objective: the UNWEIGHTED
        # mean of the per-dataset val_mu_pcc. The pooled val_loss is dominated
        # by the high-count / large-magnitude-NLL datasets, so we monitor this
        # instead (see config optim.scheduler.monitor: val_mean_mu_pcc).
        self._log_mean_mu_pcc()
        self._update_dataset_loss_weight_schedule()

        if not self.use_cagrad:
            return

        scheduler = self.lr_schedulers()
        monitor = self.config.optim.scheduler.monitor
        metric = self.trainer.callback_metrics.get(monitor)

        if metric is not None:
            scheduler.step(metric)

    def _log_mean_mu_pcc(self) -> None:
        """
        Aggregates the per-dataset ``val_mu_pcc/<name>`` series (already
        epoch-reduced into ``trainer.callback_metrics``) into a single
        ``val_mean_mu_pcc`` with equal weight per dataset. This is the metric
        the scheduler / checkpoint / early-stopping monitor.

        With a single active dataset the per-dataset series is not emitted
        (see ``_log_per_dataset_metrics``); fall back to the aggregate
        ``val_mu_pcc`` so the monitored metric is always present.
        """
        callback_metrics = self.trainer.callback_metrics
        prefix = "val_mu_pcc/"
        per_dataset = [
            value
            for name, value in callback_metrics.items()
            if name.startswith(prefix) and value is not None
        ]

        if per_dataset:
            mean_mu_pcc = torch.stack(
                [v.detach().to(self.device).float() for v in per_dataset]
            ).mean()
        else:
            fallback = callback_metrics.get("val_mu_pcc")
            if fallback is None:
                return
            mean_mu_pcc = fallback.detach().to(self.device).float()

        self.log(
            "val_mean_mu_pcc",
            mean_mu_pcc,
            prog_bar=True,
            sync_dist=self.config.trainer.sync_dist_logs,
        )

    def _update_dataset_loss_weight_schedule(self) -> None:
        """
        Soft per-task early stopping. After each validation, compare every
        dataset's ``val_nll/<name>`` against its best-so-far. A dataset that has
        not improved by ``min_delta`` for ``patience`` consecutive validations
        has its training loss weight ``w_d`` multiplied by ``decay`` (floored at
        ``min_weight``) and its counter reset, so the decay happens once per
        ``patience`` window. The resulting ``w_d`` is applied in
        ``_aggregate_per_sample`` (apply_task_weight=True) and logged as
        ``dataset_loss_weight/<name>``.

        Skipped during the sanity-check validation (no real training has
        happened yet) so a transient first value cannot latch the schedule.
        """
        if not self.dataset_loss_weight_schedule_enabled:
            return
        if getattr(self.trainer, "sanity_checking", False):
            return

        callback_metrics = self.trainer.callback_metrics
        name_to_id = {name: ds_id for ds_id, name in self.dataset_id_to_name.items()}
        sync_dist = self.config.trainer.sync_dist_logs

        for name, ds_id in name_to_id.items():
            metric = callback_metrics.get(f"val_nll/{name}")
            if metric is None:
                continue
            cur = float(metric.detach().float().item())

            best = float(self.dataset_loss_weight_best_nll[ds_id].item())
            if cur < best - self.dataset_loss_weight_min_delta:
                self.dataset_loss_weight_best_nll[ds_id] = cur
                self.dataset_loss_weight_bad_epochs[ds_id] = 0
            else:
                self.dataset_loss_weight_bad_epochs[ds_id] += 1
                if int(self.dataset_loss_weight_bad_epochs[ds_id].item()) >= self.dataset_loss_weight_patience:
                    decayed = max(
                        float(self.dataset_loss_weight[ds_id].item()) * self.dataset_loss_weight_decay,
                        self.dataset_loss_weight_min,
                    )
                    self.dataset_loss_weight[ds_id] = decayed
                    self.dataset_loss_weight_bad_epochs[ds_id] = 0

            self.log(
                f"dataset_loss_weight/{name}",
                float(self.dataset_loss_weight[ds_id].item()),
                prog_bar=False,
                sync_dist=sync_dist,
            )

    # ============================================================
    # Optimizer
    # ============================================================

    def configure_optimizers(self):
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
