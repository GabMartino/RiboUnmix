from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from Models.RiboQueuingModel.DatasetBiasSubmodel import DatasetBiasSubmodel
from Models.RiboQueuingModel.QueuingBiologicalModel import QueuingBiologicalModel


class RiboQueuingModel(nn.Module):
    """
    Minimal interpretable queue-load model:

        mu[d,t,i] = S[d,t] * L_bio[t,i] * beta[d,t,i]

    where
        S[d,t]     = mean over valid positions of the ground-truth target
                     (target-derived mean gauge, NOT learned),
        L_bio[d,t,i] = queueing-theory biological load
                       = expm1(J_dt * w_norm),
        J_dt       = J_t * J_d, where J_t is transcript-derived and J_d is an
                     optional centered dataset multiplier,
        beta[d,t,i]= dataset-specific multiplicative visibility correction,
                     optionally centered under the (detached) L_bio weights:

            log_beta_i = log_beta_raw_i - eta * center
            center     = log sum_i a_i * exp(log_beta_raw_i)
            a_i        = L_bio_i / sum_j L_bio_j.

    eta=1 is hard arithmetic centering, eta=0 is free bounded beta, and
    intermediate eta partially centers beta. If dataset J is enabled, eta=1 is
    the identifiable setting: J_d carries dataset flux and beta carries
    position/codon visibility shape.
    """

    def __init__(
        self,
        model_configs: dict,
        eps: float = 1.0e-8,
        mu_max: float = 1.0e8,
        active_dataset_ids: Sequence[int] | None = None,
    ):
        super().__init__()

        self.eps = float(eps)
        self.mu_max = float(mu_max)

        # Retained for backward compatibility; only the mean gauge is supported
        # in this simplified model (S = mean_valid(target)).
        self.scale_gauge_mode = str(
            model_configs.get("scale_gauge_mode", "mean")
        ).lower()
        if self.scale_gauge_mode != "mean":
            raise ValueError(
                "This simplified model only supports scale_gauge_mode='mean', "
                f"got {self.scale_gauge_mode!r}."
            )
        self.beta_center_strength = min(
            1.0,
            max(0.0, float(model_configs.get("beta_center_strength", 0.3))),
        )

        biological_params = dict(model_configs["biological_params"])
        biological_params.setdefault("eps", self.eps)
        self.biological_model = QueuingBiologicalModel(config_params=biological_params)

        dataset_bias_params = model_configs["dataset_bias_params"]
        self.position_features = list(dataset_bias_params["position_features"])
        self.position_scale = float(dataset_bias_params.get("position_scale", 5000.0))
        self.position_edge_tau = float(dataset_bias_params.get("position_edge_tau", 30.0))

        self.dataset_bias_model = DatasetBiasSubmodel(config_params=dataset_bias_params)

        dataset_J_params = dict(
            model_configs.get(
                "dataset_J_params",
                model_configs.get("dataset_flux_params", {}),
            )
            or {}
        )
        self.dataset_J_enabled = bool(dataset_J_params.get("enabled", False))
        self.dataset_J_center_strength = min(
            1.0,
            max(0.0, float(dataset_J_params.get("center_strength", 1.0))),
        )
        self.dataset_J_center_mode = str(
            dataset_J_params.get("center_mode", "configured")
        ).lower()
        allowed_center_modes = {"configured", "active_configured", "batch_unique"}
        if self.dataset_J_center_mode not in allowed_center_modes:
            raise ValueError(
                "dataset_J_params.center_mode must be one of "
                f"{sorted(allowed_center_modes)}, got {self.dataset_J_center_mode!r}."
            )
        self.log_J_dataset_max = float(dataset_J_params.get("log_J_dataset_max", 0.0))

        num_datasets = int(dataset_bias_params["num_datasets"])
        center_dataset_ids = dataset_J_params.get("center_dataset_ids", None)
        if (
            center_dataset_ids is None
            and self.dataset_J_center_mode == "active_configured"
        ):
            if active_dataset_ids is None:
                raise ValueError(
                    "dataset_J_params.center_mode='active_configured' requires "
                    "active_dataset_ids to be passed to RiboQueuingModel."
                )
            center_dataset_ids = list(active_dataset_ids)

        if center_dataset_ids is not None:
            center_ids = torch.as_tensor(center_dataset_ids, dtype=torch.long)
            if center_ids.numel() == 0:
                raise ValueError(
                    "dataset_J_params center ids cannot be empty."
                )
            if bool(((center_ids < 0) | (center_ids >= num_datasets)).any()):
                raise ValueError(
                    "dataset_J_params.center_dataset_ids contains ids outside "
                    f"[0, {num_datasets})."
                )
            self.register_buffer(
                "_dataset_J_center_ids",
                center_ids,
                persistent=False,
            )
        else:
            self.register_buffer(
                "_dataset_J_center_ids",
                torch.arange(num_datasets, dtype=torch.long),
                persistent=False,
            )

        if self.dataset_J_enabled:
            init_log_J_dataset = float(dataset_J_params.get("init_log_J_dataset", 0.0))
            self.log_J_dataset_raw = nn.Parameter(
                torch.full((num_datasets,), init_log_J_dataset)
            )
        else:
            self.register_parameter("log_J_dataset_raw", None)

    # ============================================================
    # Dataset flux J_d
    # ============================================================

    def _dataset_J_factor(
        self,
        id_datasets: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        B = int(id_datasets.reshape(-1).shape[0])
        if not self.dataset_J_enabled or self.log_J_dataset_raw is None:
            zeros = torch.zeros(B, device=device, dtype=dtype)
            ones = torch.ones(B, device=device, dtype=dtype)
            return {
                "J_dataset": ones,
                "log_J_dataset": zeros,
                "log_J_dataset_raw": zeros,
                "log_J_dataset_center": torch.zeros((), device=device, dtype=dtype),
            }

        ids = id_datasets.reshape(-1).to(device=device, dtype=torch.long)
        raw_all = self.log_J_dataset_raw.to(device=device, dtype=dtype)
        if self.log_J_dataset_max > 0.0:
            max_abs = float(self.log_J_dataset_max)
            raw_all = max_abs * torch.tanh(raw_all / max_abs)

        if self.dataset_J_center_mode == "batch_unique":
            center_ids = torch.unique(ids)
        else:
            center_ids = self._dataset_J_center_ids.to(device=device, dtype=torch.long)
        center = raw_all[center_ids].mean()

        raw_sample = raw_all[ids]
        log_J_dataset = raw_sample - float(self.dataset_J_center_strength) * center
        return {
            "J_dataset": torch.exp(log_J_dataset),
            "log_J_dataset": log_J_dataset,
            "log_J_dataset_raw": raw_sample,
            "log_J_dataset_center": center,
        }

    # ============================================================
    # Position features
    # ============================================================

    def make_position_features(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        mask_b = mask.bool()
        B, T = mask_b.shape
        device = mask_b.device

        pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(0).expand(B, T)
        lengths = mask_b.sum(dim=1, keepdim=True).to(dtype=dtype).clamp_min(1.0)
        last_pos = (lengths - 1.0).clamp_min(1.0)
        tau = max(float(self.position_edge_tau), 1.0e-6)

        rel_pos = pos / last_pos
        dist_start = pos
        dist_stop = (lengths - 1.0 - pos).clamp_min(0.0)

        feature_map = {
            "rel_pos": rel_pos,
            "abs_pos": pos / float(self.position_scale),
            "abs_pos_log": torch.log1p(pos) / torch.log1p(
                torch.tensor(float(self.position_scale), device=device, dtype=dtype)
            ),
            "start_window": torch.exp(-dist_start / tau),
            "stop_window": torch.exp(-dist_stop / tau),
            "dist_to_start": rel_pos,
            "dist_to_stop": 1.0 - rel_pos,
        }

        missing = [n for n in self.position_features if n not in feature_map]
        if missing:
            raise KeyError(f"Unknown position feature(s): {missing}")

        x_pos = torch.stack([feature_map[n] for n in self.position_features], dim=-1)
        return x_pos * mask_b.unsqueeze(-1).to(dtype=dtype)

    # ============================================================
    # Beta centering
    # ============================================================

    def _center_log_beta(
        self,
        log_beta_raw: torch.Tensor,
        L_bio: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Arithmetic (log-sum-exp) centering under the detached L_bio weights:

            a_i    = L_bio_i / sum_j L_bio_j           (detached, masked)
            center = log sum_i a_i * exp(log_beta_raw_i)
            log_beta_i = log_beta_raw_i - eta * center

        eta=1 gives sum_i a_i * exp(log_beta_i) = 1. The per-row max shift
        keeps the log-sum-exp numerically stable; masked positions carry
        a_i = 0 and never contribute.
        """
        mask_f = mask_b.to(dtype=log_beta_raw.dtype)
        weights = L_bio.detach().to(dtype=log_beta_raw.dtype) * mask_f
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        eta = float(self.beta_center_strength)

        b_shift = log_beta_raw.masked_fill(~mask_b, float("-inf")).amax(
            dim=1, keepdim=True
        )
        b_shift = torch.where(
            torch.isfinite(b_shift), b_shift, torch.zeros_like(b_shift)
        )
        center = b_shift + torch.log(
            (weights * torch.exp(log_beta_raw - b_shift))
            .sum(dim=1, keepdim=True)
            .clamp_min(self.eps)
        )
        log_beta = (log_beta_raw - eta * center) * mask_f
        return log_beta, center.reshape(-1), weights

    # ============================================================
    # Target-derived mean scale
    # ============================================================

    def _target_scale_dt(
        self,
        target: torch.Tensor,
        mask_b: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        target = target.to(device=device, dtype=dtype)
        valid = mask_b & torch.isfinite(target)
        mask_f = valid.to(dtype=dtype)
        target = torch.where(valid, target.clamp_min(0.0), torch.zeros_like(target))
        target_sum = (target * mask_f).sum(dim=1, keepdim=True)
        valid_len = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (target_sum / valid_len).clamp_min(self.eps)

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        x_packed,
        codon_ids: torch.Tensor,
        id_datasets: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor | None = None,
        current_epoch: int | None = None,
    ):
        del current_epoch  # unused; kept for interface compatibility

        # --------------------------------------------------------
        # 1. Biological branch -> queue load
        # --------------------------------------------------------
        bio = self.biological_model(x_packed, mask)
        w_norm = bio["w_norm"]
        J_transcript = bio["J"]

        dtype = w_norm.dtype
        device = w_norm.device
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=dtype)

        dataset_J = self._dataset_J_factor(id_datasets, dtype=dtype, device=device)
        J_dataset = dataset_J["J_dataset"].reshape(-1, 1)
        J = (J_transcript * J_dataset).clamp(
            min=float(self.biological_model.J_min),
            max=float(self.biological_model.J_max),
        )
        lambda_bio = (J * w_norm).clamp(
            min=float(self.biological_model.lambda_bio_min),
            max=float(self.biological_model.lambda_bio_max),
        ) * mask_f
        rho = (-torch.expm1(-lambda_bio)).clamp(0.0, 1.0 - 1.0e-6) * mask_f
        L_bio = torch.expm1(lambda_bio).clamp_min(self.eps) * mask_f

        # --------------------------------------------------------
        # 2. Dataset bias branch -> raw log beta + log_sigma
        # --------------------------------------------------------
        position_features = self.make_position_features(mask=mask_b, dtype=dtype)
        bias = self.dataset_bias_model(
            dataset_ids=id_datasets,
            mask=mask_b,
            codon_ids=codon_ids,
            position_features=position_features,
        )
        log_beta_raw = bias["log_visibility_bias_raw"].to(dtype=dtype) * mask_f
        log_sigma = bias["log_sigma"].to(dtype=dtype)
        log_sigma = torch.where(mask_b, log_sigma, torch.zeros_like(log_sigma))

        # --------------------------------------------------------
        # 3. Beta centering under detached L_bio weights
        # --------------------------------------------------------
        log_beta, log_beta_center, beta_weights = self._center_log_beta(
            log_beta_raw=log_beta_raw,
            L_bio=L_bio,
            mask_b=mask_b,
        )
        beta = torch.exp(log_beta) * mask_f

        # --------------------------------------------------------
        # 4. Target-derived mean scale S
        # --------------------------------------------------------
        if target is None:
            raise ValueError(
                "RiboQueuingModel.forward requires target so S = mean_valid(target) "
                "can be computed."
            )
        scale_dt = self._target_scale_dt(target, mask_b, dtype, device)  # [B, 1]

        # --------------------------------------------------------
        # 5. Prediction: mu = S * L_bio * beta
        # --------------------------------------------------------
        mu = (scale_dt * L_bio * beta).clamp(self.eps, self.mu_max)
        mu = torch.nan_to_num(mu, nan=self.eps, posinf=self.mu_max, neginf=self.eps)
        mu = torch.where(mask_b, mu, torch.ones_like(mu))

        # --------------------------------------------------------
        # 6. Diagnostics (no effect on the loss)
        # --------------------------------------------------------
        valid_len = mask_f.sum(dim=1).clamp_min(1.0)
        target_mean = scale_dt.reshape(-1)
        mu_mean = (mu * mask_f).sum(dim=1) / valid_len
        mass_ratio = mu_mean / target_mean.clamp_min(self.eps)
        alpha = torch.exp(log_sigma) * mask_f
        beta_weighted_mean = (beta * beta_weights).sum(dim=1)
        beta_weighted_log_mean = (log_beta * beta_weights).sum(dim=1)
        log_beta_abs_mean = (log_beta.abs() * mask_f).sum(dim=1) / valid_len
        beta_valid = torch.where(mask_b, beta, torch.zeros_like(beta))
        beta_max = beta_valid.amax(dim=1)
        beta_min = torch.where(mask_b, beta, torch.full_like(beta, float("inf"))).amin(dim=1)
        beta_min = torch.where(torch.isfinite(beta_min), beta_min, torch.zeros_like(beta_min))

        lambda_valid = torch.where(mask_b, lambda_bio, torch.zeros_like(lambda_bio))
        L_bio_valid = torch.where(mask_b, L_bio, torch.zeros_like(L_bio))
        w_norm_valid = torch.where(mask_b, w_norm, torch.zeros_like(w_norm))
        lambda_bio_min_cfg = float(self.biological_model.lambda_bio_min)
        lambda_bio_max_cfg = float(self.biological_model.lambda_bio_max)
        clamp_tol = 1.0e-6
        lambda_bio_mean = lambda_valid.sum(dim=1) / valid_len
        lambda_bio_max = lambda_valid.amax(dim=1)
        lambda_bio_min = torch.where(
            mask_b,
            lambda_bio,
            torch.full_like(lambda_bio, float("inf")),
        ).amin(dim=1)
        lambda_bio_min = torch.where(
            torch.isfinite(lambda_bio_min),
            lambda_bio_min,
            torch.zeros_like(lambda_bio_min),
        )
        lambda_bio_at_min_frac = (
            ((lambda_bio <= lambda_bio_min_cfg + clamp_tol) & mask_b)
            .to(dtype=dtype)
            .sum(dim=1)
            / valid_len
        )
        lambda_bio_at_max_frac = (
            ((lambda_bio >= lambda_bio_max_cfg - clamp_tol) & mask_b)
            .to(dtype=dtype)
            .sum(dim=1)
            / valid_len
        )
        L_bio_mean = L_bio_valid.sum(dim=1) / valid_len
        L_bio_max = L_bio_valid.amax(dim=1)
        J_flat = J.reshape(-1)
        J_transcript_flat = J_transcript.reshape(-1)
        J_dataset_flat = J_dataset.reshape(-1)
        w_norm_mean = w_norm_valid.sum(dim=1) / valid_len
        w_norm_max = w_norm_valid.amax(dim=1)

        extras = {
            "scale_dt": scale_dt.reshape(-1),
            "target_mean": target_mean,
            "mu_mean": mu_mean,
            "mass_ratio": mass_ratio,
            "w_norm": w_norm,
            "J": J,
            "J_transcript": J_transcript,
            "J_dataset": J_dataset_flat,
            "log_J_dataset": dataset_J["log_J_dataset"],
            "log_J_dataset_raw": dataset_J["log_J_dataset_raw"],
            "log_J_dataset_center": dataset_J["log_J_dataset_center"].expand_as(target_mean),
            "lambda_bio": lambda_bio,
            "rho": rho,
            "L_bio": L_bio,
            "log_beta_raw": torch.where(
                mask_b, log_beta_raw, torch.zeros_like(log_beta_raw)
            ),
            "log_beta": log_beta,
            "log_beta_center": log_beta_center,
            "beta": torch.where(mask_b, beta, torch.ones_like(beta)),
            "beta_center_strength": torch.full_like(target_mean, self.beta_center_strength),
            "beta_weighted_mean": beta_weighted_mean,
            "beta_weighted_log_mean": beta_weighted_log_mean,
            "beta_mass_gain": beta_weighted_mean,
            "log_beta_abs_mean": log_beta_abs_mean,
            "beta_max": beta_max,
            "beta_min": beta_min,
            "lambda_bio_mean": lambda_bio_mean,
            "lambda_bio_max": lambda_bio_max,
            "lambda_bio_min": lambda_bio_min,
            "lambda_bio_at_min_frac": lambda_bio_at_min_frac,
            "lambda_bio_at_max_frac": lambda_bio_at_max_frac,
            "L_bio_mean": L_bio_mean,
            "L_bio_max": L_bio_max,
            "J_mean": J_flat,
            "J_min": J_flat,
            "J_max": J_flat,
            "J_transcript_mean": J_transcript_flat,
            "J_dataset_mean": J_dataset_flat,
            "w_norm_mean": w_norm_mean,
            "w_norm_max": w_norm_max,
            "mu": mu,
            "log_sigma": log_sigma,
            "log_sigma_t": (log_sigma * mask_f).sum(dim=1, keepdim=True)
            / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0),
            "alpha": alpha,
            "valid_len": valid_len,
        }

        return mu, log_sigma, extras
