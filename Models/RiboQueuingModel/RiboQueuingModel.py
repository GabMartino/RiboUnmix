from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn as nn
from entmax import entmax15

from Models.RiboQueuingModel.DatasetBiasSubmodel import DatasetBiasSubmodel
from Models.RiboQueuingModel.QueuingBiologicalModel import QueuingBiologicalModel


class RiboQueuingModel(nn.Module):
    """
    Minimal interpretable queue-load shape model with a gated-additive mean:

        mu[d,t,i] = S[d,t] * (gamma[d,t,i] * L_bio[t,i] + a[d,t,i])

    where
        S[d,t]     = mean over valid positions of the ground-truth target
                     (target-derived mean gauge, NOT learned),
        L_bio[t,i] = normalized biological load, so it is mean-one by
                     construction while S carries target scale,
        rho[t,i]   = L_bio / (1 + L_bio), so high rho marks saturated local
                     load.
        gamma      = exp(centered_amplitude_score) times an optional
                     length-scaled entmax support gate; neutral scores give 1
                     and sparse mode permits exact 0,
        a          = nonnegative additive background, regularized toward 0.

    Gamma-score centering is a multiplicative reference/gauge constraint only. It
    does not identify the additive branch; additive-bias regularization remains
    a separate objective term.
    """

    def __init__(
        self,
        model_configs: dict,
        eps: float = 1.0e-8,
        mu_max: float = 1.0e8,
    ):
        super().__init__()

        self.eps = float(eps)
        self.mu_max = float(mu_max)

        # Scale gauge is always the mean gauge S = mean_valid(target).
        self.init_gamma = float(model_configs.get("init_gamma", 1.0))
        self.gamma_log_init = math.log(self.init_gamma)
        self.gamma_transform = str(
            model_configs.get("gamma_transform", "exponential")
        ).lower()
        self.gamma_entmax_temperature = max(
            float(model_configs.get("gamma_entmax_temperature", 10.0)),
            self.eps,
        )
        self.gamma_split_support_head = bool(
            model_configs.get("gamma_split_support_head", False)
        )
        self.gamma_gate_additive_bias = bool(
            model_configs.get("gamma_gate_additive_bias", False)
        )
        self._validate_gamma_transform()
        self._configure_gamma_centering(model_configs)

        # Mass conservation: renormalize the shape (gamma*L_bio + a) to mean 1
        # over valid positions before applying S, so mean_valid(mu) = S exactly
        # (=> sum(mu) = sum(target)). This makes suppressing zeros mass-neutral:
        # mass pushed off zeros is redistributed to the peaks automatically.
        self.mass_conservation = bool(model_configs.get("mass_conservation", True))

        feature_config = model_configs.get("additional_sequence_features", {}) or {}
        biological_extra_dim = 0
        dataset_bias_extra_dim = 0
        allowed_feature_routes = {"none", "biological", "dataset_bias", "both"}
        for feature_name, raw_spec in feature_config.items():
            spec = dict(raw_spec or {})
            route = str(spec.get("route", "none")).lower()
            if route not in allowed_feature_routes:
                raise ValueError(
                    f"Invalid route {route!r} for sequence feature {feature_name!r}; "
                    f"expected one of {sorted(allowed_feature_routes)}."
                )
            if route == "none":
                continue
            dimension = int(spec.get("dimension", 1))
            if dimension <= 0:
                raise ValueError(
                    f"Feature {feature_name!r} must have a positive dimension."
                )
            if route in {"biological", "both"}:
                biological_extra_dim += dimension
            if route in {"dataset_bias", "both"}:
                dataset_bias_extra_dim += dimension

        biological_params = dict(model_configs["biological_params"])
        biological_params["input_size"] = (
            int(biological_params["input_size"]) + biological_extra_dim
        )
        biological_params.setdefault("eps", self.eps)
        self.biological_model = QueuingBiologicalModel(config_params=biological_params)

        dataset_bias_params = dict(model_configs["dataset_bias_params"])
        dataset_bias_params["additional_sequence_feature_dim"] = dataset_bias_extra_dim
        self.position_features = list(dataset_bias_params["position_features"])
        self.position_scale = float(dataset_bias_params.get("position_scale", 5000.0))
        self.position_edge_tau = float(dataset_bias_params.get("position_edge_tau", 30.0))

        self.dataset_bias_model = DatasetBiasSubmodel(config_params=dataset_bias_params)

    def _validate_gamma_transform(self) -> None:
        aliases = {
            "exponential": "exponential",
            "exp": "exponential",
            "entmax15": "entmax15_gated_exponential",
            "entmax15_gated_exponential": "entmax15_gated_exponential",
        }
        if self.gamma_transform not in aliases:
            raise ValueError(
                "gamma_transform must be one of "
                f"{sorted(aliases)}, got {self.gamma_transform!r}."
            )
        self.gamma_transform = aliases[self.gamma_transform]

    def set_gamma_transform(
        self,
        transform: str,
        *,
        entmax_temperature: float | None = None,
        split_support_head: bool | None = None,
        gate_additive_bias: bool | None = None,
    ) -> None:
        """Restore gamma semantics that are not encoded by tensor weights."""
        self.gamma_transform = str(transform).lower()
        if entmax_temperature is not None:
            self.gamma_entmax_temperature = max(
                float(entmax_temperature),
                self.eps,
            )
        if split_support_head is not None:
            self.gamma_split_support_head = bool(split_support_head)
        if gate_additive_bias is not None:
            self.gamma_gate_additive_bias = bool(gate_additive_bias)
        self._validate_gamma_transform()

    def _gamma_from_centered_score(
        self,
        score: torch.Tensor,
        mask_b: torch.Tensor,
        support_score: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map amplitude and optional support scores to nonnegative gamma.

        The sparse transform retains the exponential amplitude but multiplies it
        by a valid-length-scaled entmax allocation. At a neutral score every
        valid position has gate=1 and gamma=1. Entmax support exclusions give
        exact gamma zeros; selected positions can still form large peaks.
        """
        mask_b = mask_b.bool()
        mask_f = mask_b.to(dtype=score.dtype)
        amplitude = torch.exp(score).clamp_min(self.eps) * mask_f

        if self.gamma_transform == "exponential":
            sparse_gate = mask_f
            return amplitude, amplitude, sparse_gate

        neg_large = torch.finfo(score.dtype).min
        if support_score is None:
            support_score = score
        logits = (
            support_score / float(self.gamma_entmax_temperature)
        ).masked_fill(
            ~mask_b,
            neg_large,
        )
        allocation = entmax15(logits, dim=1)
        allocation = torch.nan_to_num(
            allocation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0) * mask_f
        allocation_mass = allocation.sum(dim=1, keepdim=True)
        uniform = mask_f / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        allocation = torch.where(
            allocation_mass > self.eps,
            allocation / allocation_mass.clamp_min(self.eps),
            uniform,
        ) * mask_f
        valid_len = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        sparse_gate = allocation * valid_len
        gamma = amplitude * sparse_gate
        gamma = torch.nan_to_num(
            gamma,
            nan=0.0,
            posinf=self.mu_max,
            neginf=0.0,
        ).clamp_min(0.0) * mask_f
        return gamma, amplitude, sparse_gate

    # ============================================================
    # Equal-dataset gamma centering
    # ============================================================

    def _configure_gamma_centering(self, model_configs: dict) -> None:
        cfg = dict(model_configs.get("gamma_centering", {}))
        scope = str(cfg.get("scope", "batch_grouped")).lower()

        if scope not in {"batch_grouped", "disabled"}:
            raise ValueError("gamma_centering.scope must be 'batch_grouped' or 'disabled'.")

        self.gamma_cross_dataset_centering_enabled = bool(
            cfg.get("enabled", False)
        ) and scope != "disabled"
        self.gamma_cross_dataset_centering_strength = float(cfg.get("strength", 1.0))
        self.gamma_centering_min_distinct_datasets = int(
            cfg.get("min_distinct_datasets", 2)
        )
        self.gamma_log_min = float(cfg.get("log_gamma_min", -8.0))
        self.gamma_log_max = float(cfg.get("log_gamma_max", 8.0))

    @staticmethod
    def _normalize_sample_ids(
        sample_ids: Sequence[str] | torch.Tensor | None,
        batch_size: int,
    ) -> list[str] | None:
        if sample_ids is None:
            return None
        if torch.is_tensor(sample_ids):
            values = sample_ids.detach().cpu().reshape(-1).tolist()
        else:
            values = list(sample_ids)
        return [str(value) for value in values]

    def _center_log_gamma_across_transcripts(
        self,
        log_gamma_raw: torch.Tensor,
        *,
        mask_b: torch.Tensor,
        sample_ids: Sequence[str] | torch.Tensor | None,
        id_datasets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Center log-amplitude equally across datasets for each transcript."""
        B, T = log_gamma_raw.shape
        dtype = log_gamma_raw.dtype
        device = log_gamma_raw.device
        mask_b = mask_b.bool()
        mask_f = mask_b.to(dtype=dtype)
        zeros = torch.zeros_like(log_gamma_raw)
        false = torch.zeros_like(mask_b)
        dataset_ids = id_datasets.reshape(-1).to(device=device, dtype=torch.long)
        sample_id_list = self._normalize_sample_ids(sample_ids, B)
        if sample_id_list is None:
            candidate = false
        else:
            valid_ids = [
                sid.lower() not in {"", "none", "nan", "null"}
                for sid in sample_id_list
            ]
            transcript_valid = torch.as_tensor(valid_ids, device=device).reshape(-1, 1)
            candidate = (
                mask_b
                & transcript_valid
                & (dataset_ids >= 0).reshape(-1, 1)
                & torch.isfinite(log_gamma_raw)
            )

        if (
            sample_id_list is None
            or not self.gamma_cross_dataset_centering_enabled
            or self.gamma_cross_dataset_centering_strength <= 0.0
            or B <= 1
        ):
            return {
                "log_gamma": log_gamma_raw * mask_f,
                "gamma_center": zeros,
                "weights": zeros,
                "applied": false,
                "num_distinct_datasets": zeros,
                "total_weight": zeros,
                "constraint_error": zeros,
                "eligible": candidate,
            }

        group_lookup: dict[str, int] = {}
        group_index = torch.as_tensor(
            [group_lookup.setdefault(sid, len(group_lookup)) for sid in sample_id_list],
            device=device,
            dtype=torch.long,
        )
        num_groups = len(group_lookup)
        unique_datasets, dataset_index = torch.unique(dataset_ids, return_inverse=True)
        num_datasets = int(unique_datasets.numel())
        num_cells = num_groups * num_datasets
        cell_index = group_index * num_datasets + dataset_index
        cell_group = torch.arange(num_cells, device=device, dtype=torch.long) // num_datasets

        candidate_f = candidate.to(dtype=dtype)
        log_safe = torch.where(candidate, log_gamma_raw, zeros)

        def _scatter_sum(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
            return torch.zeros(size, T, device=device, dtype=src.dtype).index_add(
                0, index, src
            )

        count_cell = _scatter_sum(candidate_f, cell_index, num_cells)
        mean_cell = _scatter_sum(
            log_safe, cell_index, num_cells
        ) / count_cell.clamp_min(1.0)
        cell_active = count_cell > 0
        active_f = cell_active.to(dtype=dtype)
        num_distinct_group = _scatter_sum(active_f, cell_group, num_groups)
        sum_means_group = _scatter_sum(mean_cell * active_f, cell_group, num_groups)
        center_group = sum_means_group / num_distinct_group.clamp_min(1.0)
        apply_group = num_distinct_group >= float(
            self.gamma_centering_min_distinct_datasets
        )

        apply_sample = apply_group.index_select(0, group_index)
        center_sample = center_group.index_select(0, group_index)
        applied = apply_sample & mask_b
        log_center = torch.where(applied, center_sample, zeros)
        num_distinct = torch.where(
            applied, num_distinct_group.index_select(0, group_index), zeros
        )
        dataset_weight_cell = active_f / num_distinct_group.index_select(
            0, cell_group
        ).clamp_min(1.0)
        sample_weights = dataset_weight_cell.index_select(
            0, cell_index
        ) / count_cell.index_select(0, cell_index).clamp_min(1.0)
        sample_weights = torch.where(
            candidate & apply_sample, sample_weights, torch.zeros_like(sample_weights)
        )
        strength = float(self.gamma_cross_dataset_centering_strength)
        ce_contrib = sample_weights * (log_safe - strength * center_sample)
        constraint_error_group = _scatter_sum(ce_contrib, group_index, num_groups).abs()
        constraint_error = torch.where(
            applied, constraint_error_group.index_select(0, group_index), zeros
        )
        log_gamma = (log_gamma_raw - strength * log_center) * mask_f

        return {
            "log_gamma": log_gamma,
            "gamma_center": log_center * mask_f,
            "weights": sample_weights * mask_f,
            "applied": applied & mask_b,
            "num_distinct_datasets": num_distinct * mask_f,
            "total_weight": num_distinct * mask_f,
            "constraint_error": constraint_error * mask_f,
            "eligible": candidate,
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
        target: torch.Tensor,
        sample_ids: Sequence[str] | torch.Tensor | None = None,
        dataset_bias_sequence_features: torch.Tensor | None = None,
    ):
        # --------------------------------------------------------
        # 1. Biological branch -> queue load
        # --------------------------------------------------------
        bio = self.biological_model(x_packed, mask)
        L_bio = bio["L_bio"]

        dtype = L_bio.dtype
        device = L_bio.device
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=dtype)

        # Direct-load queue: L_bio is already mean-one, no lambda clamp.
        J = bio["J"].to(dtype=dtype, device=device)
        lambda_bio = bio["lambda_bio"].to(dtype=dtype, device=device) * mask_f
        rho = bio["rho"].to(dtype=dtype, device=device).clamp(
            0.0,
            1.0 - 1.0e-6,
        ) * mask_f
        L_bio = L_bio.to(dtype=dtype, device=device) * mask_f

        # --------------------------------------------------------
        # 2. Dataset bias branch -> gamma residual + additive background
        # --------------------------------------------------------
        position_features = self.make_position_features(mask=mask_b, dtype=dtype)
        bias = self.dataset_bias_model(
            dataset_ids=id_datasets,
            mask=mask_b,
            codon_ids=codon_ids,
            position_features=position_features,
            sequence_features=dataset_bias_sequence_features,
        )
        gamma_log_residual = bias["gamma_raw"].to(dtype=dtype, device=device) * mask_f
        gamma_support_logits_raw = bias.get("gamma_support_logits")
        if torch.is_tensor(gamma_support_logits_raw):
            gamma_support_logits_raw = (
                gamma_support_logits_raw.to(dtype=dtype, device=device) * mask_f
            )
        else:
            gamma_support_logits_raw = gamma_log_residual
        additive_bias = bias["additive_bias"].to(dtype=dtype, device=device) * mask_f
        log_sigma = bias["log_sigma"].to(dtype=dtype)
        log_sigma = torch.where(mask_b, log_sigma, torch.zeros_like(log_sigma))

        # --------------------------------------------------------
        # 3. Observation mean branch
        # --------------------------------------------------------
        log_gamma_raw = (
            gamma_log_residual + float(self.gamma_log_init)
        ).clamp(min=self.gamma_log_min, max=self.gamma_log_max) * mask_f
        gamma_raw = torch.exp(log_gamma_raw).clamp_min(self.eps) * mask_f
        centered = self._center_log_gamma_across_transcripts(
            log_gamma_raw,
            mask_b=mask_b,
            sample_ids=sample_ids,
            id_datasets=id_datasets,
        )
        gamma_eligible = centered["eligible"]
        gamma_uniform_weight = gamma_eligible.to(dtype=dtype)
        log_gamma = centered["log_gamma"]
        gamma_cross_dataset_log_center = centered["gamma_center"]
        gamma_cross_dataset_center_group_size = centered[
            "num_distinct_datasets"
        ].amax(dim=1)
        gamma_cross_dataset_center_applied = centered["applied"].any(dim=1).to(
            dtype=dtype
        )
        if self.gamma_split_support_head:
            # Entmax is invariant to a constant shift, but explicit masked
            # sequence centering improves numerical conditioning and makes the
            # exported support scores easier to compare within a transcript.
            support_mean = gamma_support_logits_raw.sum(dim=1, keepdim=True) / (
                mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            gamma_support_logits = (
                gamma_support_logits_raw - support_mean
            ) * mask_f
        else:
            # Exact legacy behavior: the amplitude score also controls support.
            gamma_support_logits = log_gamma
        gamma, gamma_amplitude, gamma_sparse_gate = self._gamma_from_centered_score(
            log_gamma,
            mask_b,
            support_score=gamma_support_logits,
        )

        # --------------------------------------------------------
        # 4. Target-derived mean scale S
        # --------------------------------------------------------
        scale_dt = self._target_scale_dt(target, mask_b, dtype, device)  # [B, 1]

        # --------------------------------------------------------
        # 5. Prediction
        # --------------------------------------------------------
        if self.gamma_gate_additive_bias:
            # A support zero must suppress the complete dataset-specific mean,
            # including the additive branch; otherwise additive bias can leak
            # positive mass through an exact gamma zero.
            mu_inner = gamma * L_bio + gamma_sparse_gate * additive_bias
        else:
            mu_inner = gamma * L_bio + additive_bias
        if self.mass_conservation:
            # Renormalize the shape to mean 1 over valid positions so that
            # mean_valid(mu) = S exactly. gamma, L_bio and additive_bias keep
            # their per-position meaning; only the overall shape level is pinned.
            inner_valid_len = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            inner_mean = (mu_inner * mask_f).sum(dim=1, keepdim=True) / inner_valid_len
            mu_inner = mu_inner / inner_mean.clamp_min(self.eps)
        mu = (scale_dt * mu_inner).clamp(self.eps, self.mu_max)
        mu = torch.nan_to_num(mu, nan=self.eps, posinf=self.mu_max, neginf=self.eps)
        mu = torch.where(mask_b, mu, torch.ones_like(mu))

        # --------------------------------------------------------
        # 6. Diagnostics (no effect on the loss)
        # --------------------------------------------------------
        valid_len = mask_f.sum(dim=1).clamp_min(1.0)
        target_mean = scale_dt.reshape(-1)
        mu_mean = (mu * mask_f).sum(dim=1) / valid_len
        mean_ratio = mu_mean / target_mean.clamp_min(self.eps)
        alpha = torch.exp(log_sigma) * mask_f

        lambda_valid = torch.where(mask_b, lambda_bio, torch.zeros_like(lambda_bio))
        L_bio_valid = torch.where(mask_b, L_bio, torch.zeros_like(L_bio))
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
        L_bio_mean = L_bio_valid.sum(dim=1) / valid_len
        L_bio_max = L_bio_valid.amax(dim=1)
        J_flat = J.reshape(-1)

        extras = {
            "scale_dt": scale_dt.reshape(-1),
            "target_mean": target_mean,
            "mu_mean": mu_mean,
            "mean_ratio": mean_ratio,
            "J": J,
            "lambda_bio": lambda_bio,
            "rho": rho,
            "L_bio": L_bio,
            "gamma_logits": torch.where(
                mask_b,
                log_gamma_raw,
                torch.zeros_like(log_gamma_raw),
            ),
            "gamma_raw": torch.where(mask_b, gamma_raw, torch.ones_like(gamma_raw)),
            "log_gamma_raw": torch.where(
                mask_b,
                log_gamma_raw,
                torch.zeros_like(log_gamma_raw),
            ),
            "gamma_cross_dataset_log_center": torch.where(
                mask_b,
                gamma_cross_dataset_log_center,
                torch.zeros_like(gamma_cross_dataset_log_center),
            ),
            "gamma_cross_dataset_center_group_size": gamma_cross_dataset_center_group_size,
            "gamma_cross_dataset_center_applied": gamma_cross_dataset_center_applied,
            "gamma_centering_reliability": torch.where(
                mask_b,
                gamma_uniform_weight,
                torch.zeros_like(gamma_uniform_weight),
            ),
            "gamma_centering_eligible": gamma_eligible & mask_b,
            "gamma_centering_applied": centered["applied"] & mask_b,
            "gamma_num_distinct_datasets": torch.where(
                mask_b,
                centered["num_distinct_datasets"],
                torch.zeros_like(centered["num_distinct_datasets"]),
            ),
            "gamma_total_reliability": torch.where(
                mask_b,
                centered["total_weight"],
                torch.zeros_like(centered["total_weight"]),
            ),
            "gamma_centering_constraint_error": torch.where(
                mask_b,
                centered["constraint_error"],
                torch.zeros_like(centered["constraint_error"]),
            ),
            "gamma": torch.where(mask_b, gamma, torch.ones_like(gamma)),
            "gamma_amplitude": torch.where(
                mask_b,
                gamma_amplitude,
                torch.ones_like(gamma_amplitude),
            ),
            "gamma_sparse_gate": torch.where(
                mask_b,
                gamma_sparse_gate,
                torch.ones_like(gamma_sparse_gate),
            ),
            "gamma_support_logits_raw": torch.where(
                mask_b,
                gamma_support_logits_raw,
                torch.zeros_like(gamma_support_logits_raw),
            ),
            "gamma_support_logits": torch.where(
                mask_b,
                gamma_support_logits,
                torch.zeros_like(gamma_support_logits),
            ),
            "log_gamma": torch.where(
                mask_b,
                log_gamma,
                torch.zeros_like(log_gamma),
            ),
            "additive_bias": torch.where(
                mask_b,
                additive_bias,
                torch.zeros_like(additive_bias),
            ),
            "lambda_bio_mean": lambda_bio_mean,
            "lambda_bio_max": lambda_bio_max,
            "lambda_bio_min": lambda_bio_min,
            "L_bio_mean": L_bio_mean,
            "L_bio_max": L_bio_max,
            "J_mean": J_flat,
            "J_min": J_flat,
            "J_max": J_flat,
            "mu": mu,
            "log_sigma": log_sigma,
            "log_sigma_t": (log_sigma * mask_f).sum(dim=1, keepdim=True)
            / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0),
            "alpha": alpha,
            "valid_len": valid_len,
        }

        return mu, log_sigma, extras
