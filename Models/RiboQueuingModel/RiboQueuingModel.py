from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from Models.RiboQueuingModel.DatasetBiasSubmodel import DatasetBiasSubmodel
from Models.RiboQueuingModel.QueuingBiologicalModel import QueuingBiologicalModel


def build_bias_sequence_embedding_tables(
    *,
    nt_encoding: Mapping[str, Sequence[float]],
    codon_to_aa_encoding: Mapping[str, str],
    codon_encoding: Mapping[str, int],
    aa_encoding: Mapping[str, int],
) -> dict[str, object]:
    """Build codon -> nucleotide/amino-acid token lookup tables."""
    num_codons = len(codon_encoding)
    nt_base_to_id = {
        str(base).upper(): int(torch.as_tensor(value).argmax().item())
        for base, value in nt_encoding.items()
    }
    num_nucleotides = len(nt_base_to_id)

    codon_nucleotide_ids = [[0, 0, 0] for _ in range(num_codons)]
    codon_amino_acid_ids = [0 for _ in range(num_codons)]
    for codon, raw_id in codon_encoding.items():
        codon = str(codon).upper().replace("U", "T")
        codon_id = int(raw_id)
        amino_acid_id = int(aa_encoding[str(codon_to_aa_encoding[codon])])
        codon_nucleotide_ids[codon_id] = [nt_base_to_id[base] for base in codon]
        codon_amino_acid_ids[codon_id] = amino_acid_id
    return {
        "num_nucleotides": num_nucleotides,
        "num_amino_acids": len(aa_encoding),
        "codon_nucleotide_ids": codon_nucleotide_ids,
        "codon_amino_acid_ids": codon_amino_acid_ids,
    }


class RiboQueuingModel(nn.Module):
    """
    Minimal interpretable queue-load shape model with a multiplicative mean:

        mu[d,t,i] = S[d,t] * normalize_i(gamma[d,t,i] * L_bio[t,i])

    where
        S[d,t]     = mean over valid positions of the ground-truth target
                     (target-derived mean gauge, NOT learned),
        L_bio[t,i] = normalized biological load, so it is mean-one by
                     construction while S carries target scale,
        rho[t,i]   = L_bio / (1 + L_bio), so high rho marks saturated local
                     load.
        gamma      = exp(centered_log_score), so it is strictly positive on
                     valid positions and neutral scores give 1,
    Gamma uses two joint log-space gauges: the configured weighted reference
    datasets have zero weighted log mean at every position, and every active
    dataset-transcript profile has zero positional log mean. Thus gamma has
    geometric mean one along both identifiable axes without altering mu.
    """

    def __init__(
        self,
        model_configs: dict,
        eps: float = 1.0e-8,
        selected_dataset_names: Sequence[str] | None = None,
        selected_dataset_ids: Sequence[int] | None = None,
        reference_dataset_names: Sequence[str] | None = None,
        reference_dataset_ids: Sequence[int] | None = None,
        reference_dataset_quality_weights: Sequence[float] | None = None,
        nt_encoding: Mapping[str, Sequence[float]] | None = None,
        codon_to_aa_encoding: Mapping[str, str] | None = None,
        codon_encoding: Mapping[str, int] | None = None,
        aa_encoding: Mapping[str, int] | None = None,
    ):
        super().__init__()

        self.eps = float(eps)
        self.selected_dataset_names = tuple(str(x) for x in (selected_dataset_names or ()))
        self.selected_dataset_ids = tuple(int(x) for x in (selected_dataset_ids or ()))

        # Scale gauge is always the mean gauge S = mean_valid(target).
        self.init_gamma = float(model_configs.get("init_gamma", 1.0))
        self.gamma_log_init = math.log(self.init_gamma)
        self._configure_gamma_centering(
            model_configs,
            reference_dataset_names=reference_dataset_names,
            reference_dataset_ids=reference_dataset_ids,
            reference_dataset_quality_weights=reference_dataset_quality_weights,
        )

        # Mass conservation: renormalize the shape (gamma*L_bio + a) to mean 1
        # over valid positions before applying S, so mean_valid(mu) = S exactly
        # (=> sum(mu) = sum(target)). This makes suppressing zeros mass-neutral:
        # mass pushed off zeros is redistributed to the peaks automatically.
        self.mass_conservation = bool(model_configs.get("mass_conservation", True))

        self.alpha_mode = str(model_configs.get("alpha_mode", "learned")).lower()
        if self.alpha_mode not in {"learned", "fixed"}:
            raise ValueError("model.alpha_mode must be 'learned' or 'fixed'.")
        self.fixed_alpha = float(model_configs.get("fixed_alpha", 0.1))
        if not math.isfinite(self.fixed_alpha) or self.fixed_alpha <= 0.0:
            raise ValueError("model.fixed_alpha must be finite and strictly positive.")

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
        if bool(dataset_bias_params.get("use_nucleotide_amino_acid_embeddings", False)):
            tables = build_bias_sequence_embedding_tables(
                nt_encoding=nt_encoding,
                codon_to_aa_encoding=codon_to_aa_encoding,
                codon_encoding=codon_encoding,
                aa_encoding=aa_encoding,
            )
            dataset_bias_params.update(tables)
        self.position_features = list(dataset_bias_params["position_features"])
        self.position_scale = float(dataset_bias_params.get("position_scale", 5000.0))
        self.position_edge_tau = float(dataset_bias_params.get("position_edge_tau", 30.0))

        self.dataset_bias_model = DatasetBiasSubmodel(config_params=dataset_bias_params)
        if self.alpha_mode == "fixed":
            # Keep the architecture/checkpoint schema unchanged, but make the
            # diagnostic oracle bypass the alpha head and optimizer entirely.
            self.dataset_bias_model.log_sigma_head.requires_grad_(False)
        self.register_buffer(
            "fixed_log_alpha",
            torch.tensor(math.log(self.fixed_alpha), dtype=torch.float32),
            persistent=False,
        )

    # ============================================================
    # Cross-dataset gamma centering
    # ============================================================

    def _configure_gamma_centering(
        self,
        model_configs: dict,
        *,
        reference_dataset_names: Sequence[str] | None,
        reference_dataset_ids: Sequence[int] | None,
        reference_dataset_quality_weights: Sequence[float] | None,
    ) -> None:
        cfg = dict(model_configs.get("gamma_centering", {}))
        configured_mode = str(cfg.get("mode", "disabled")).lower()
        valid_modes = {"batch_grouped", "fixed_reference", "disabled"}
        if configured_mode not in valid_modes:
            raise ValueError(
                "gamma_centering.mode must be one of 'fixed_reference', "
                "'batch_grouped', or 'disabled'."
            )
        mode = configured_mode
        dataset_constant_scale_gauge = str(
            cfg.get("dataset_constant_scale_gauge", "geometric_mean_one")
        ).lower()
        if dataset_constant_scale_gauge not in {
            "geometric_mean_one",
            "disabled",
        }:
            raise ValueError(
                "gamma_centering.dataset_constant_scale_gauge must be "
                "'geometric_mean_one' or 'disabled'."
            )

        reference_cfg = dict(cfg.get("reference", {}) or {})
        weighting = str(reference_cfg.get("weighting", "equal")).lower()
        if weighting not in {"equal", "quality_rank"}:
            raise ValueError(
                "gamma_centering weighting must be 'equal' or 'quality_rank'."
            )
        quality_rank_power = float(reference_cfg.get("quality_rank_power", 1.0))
        if not math.isfinite(quality_rank_power) or quality_rank_power < 0.0:
            raise ValueError(
                "gamma_centering.quality_rank_power must be finite and nonnegative."
            )

        chunk_size = int(reference_cfg.get("chunk_size", 32))
        minimum_datasets = int(reference_cfg.get("minimum_datasets", 2))
        if chunk_size <= 0:
            raise ValueError("gamma_centering.reference.chunk_size must be positive.")
        if minimum_datasets < 1:
            raise ValueError(
                "gamma_centering.reference.minimum_datasets must be at least one."
            )

        names = tuple(str(x) for x in (reference_dataset_names or ()))
        ids = tuple(int(x) for x in (reference_dataset_ids or ()))
        quality = tuple(float(x) for x in (reference_dataset_quality_weights or ()))
        if len(self.selected_dataset_names) != len(self.selected_dataset_ids):
            raise ValueError(
                "Selected experiment dataset names and IDs must have equal length."
            )
        if len(set(self.selected_dataset_ids)) != len(self.selected_dataset_ids):
            raise ValueError("Duplicate selected experiment dataset IDs are not allowed.")
        if len(names) != len(ids):
            raise ValueError(
                "Resolved gamma reference dataset names and IDs must have equal length."
            )
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate gamma reference dataset IDs are not allowed.")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate gamma reference dataset names are not allowed.")
        if mode == "fixed_reference" and not self.selected_dataset_ids:
            raise ValueError(
                "fixed_reference gamma centering requires selected experiment IDs."
            )
        inactive_ids = sorted(set(ids) - set(self.selected_dataset_ids))
        if inactive_ids:
            raise ValueError(
                "Gamma reference IDs must be selected in the current experiment; "
                f"inactive IDs: {inactive_ids}."
            )
        if quality and len(quality) != len(ids):
            raise ValueError(
                "Resolved gamma reference quality weights must match the reference IDs."
            )
        if any((not math.isfinite(x)) or x <= 0.0 for x in quality):
            raise ValueError(
                "Resolved gamma reference quality weights must be finite and positive."
            )
        if mode == "fixed_reference" and not ids:
            raise ValueError(
                "fixed_reference gamma centering requires a resolved reference panel."
            )

        if weighting == "quality_rank" and quality:
            final_weights = torch.as_tensor(quality, dtype=torch.float64).pow(
                quality_rank_power
            )
        else:
            # p=0 is exactly equal weighting, including in quality-rank mode.
            final_weights = torch.ones(len(ids), dtype=torch.float64)

        manifest = {
            "dataset_names": list(names),
            "dataset_ids": list(ids),
            "weights": [float(x) for x in final_weights.tolist()],
            "weighting": weighting,
            "quality_rank_power": quality_rank_power,
        }
        manifest_hash = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        self.gamma_centering_mode = mode
        self.gamma_cross_dataset_centering_enabled = mode != "disabled"
        self.gamma_dataset_constant_scale_gauge = dataset_constant_scale_gauge
        self.gamma_centering_weighting = weighting
        self.gamma_centering_quality_rank_power = quality_rank_power
        self.gamma_reference_chunk_size = chunk_size
        self.gamma_reference_minimum_datasets = minimum_datasets
        self.gamma_reference_dataset_names = names
        self.gamma_reference_manifest_hash = manifest_hash
        self.register_buffer(
            "gamma_reference_dataset_ids",
            torch.as_tensor(ids, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "gamma_selected_dataset_ids",
            torch.as_tensor(self.selected_dataset_ids, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "gamma_reference_weights",
            final_weights.to(dtype=torch.float32),
            persistent=True,
        )

    def set_gamma_centering_mode(self, mode: str) -> None:
        valid_modes = {"batch_grouped", "fixed_reference", "disabled"}
        key = str(mode).lower()
        if key not in valid_modes:
            raise ValueError(f"Unknown gamma-centering mode: {mode!r}.")
        if key == "fixed_reference" and self.gamma_reference_dataset_ids.numel() == 0:
            raise ValueError("Cannot enable fixed_reference without a reference panel.")
        self.gamma_centering_mode = key
        self.gamma_cross_dataset_centering_enabled = key != "disabled"

    def get_extra_state(self) -> dict:
        return {
            "version": 4,
            "alpha_mode": self.alpha_mode,
            "fixed_alpha": self.fixed_alpha,
            "raw_log_gamma_bound": self.dataset_bias_model.raw_log_gamma_bound,
            "gamma_centering_mode": self.gamma_centering_mode,
            "gamma_dataset_constant_scale_gauge": (
                self.gamma_dataset_constant_scale_gauge
            ),
            "gamma_reference_dataset_names": list(self.gamma_reference_dataset_names),
            "gamma_reference_manifest_hash": self.gamma_reference_manifest_hash,
            "gamma_centering_weighting": self.gamma_centering_weighting,
            "gamma_centering_quality_rank_power": self.gamma_centering_quality_rank_power,
            "gamma_reference_chunk_size": self.gamma_reference_chunk_size,
            "gamma_reference_minimum_datasets": self.gamma_reference_minimum_datasets,
            "selected_dataset_names": list(self.selected_dataset_names),
            "selected_dataset_ids": list(self.selected_dataset_ids),
        }

    def set_extra_state(self, state: dict | None) -> None:
        if not state:
            return
        self.alpha_mode = str(state.get("alpha_mode", self.alpha_mode))
        self.fixed_alpha = float(state.get("fixed_alpha", self.fixed_alpha))
        self.fixed_log_alpha.fill_(math.log(self.fixed_alpha))
        raw_log_gamma_bound = state.get(
            "raw_log_gamma_bound",
            self.dataset_bias_model.raw_log_gamma_bound,
        )
        self.dataset_bias_model.raw_log_gamma_bound = (
            None
            if raw_log_gamma_bound is None
            else float(raw_log_gamma_bound)
        )
        if self.dataset_bias_model.raw_log_gamma_bound is not None and (
            not math.isfinite(self.dataset_bias_model.raw_log_gamma_bound)
            or self.dataset_bias_model.raw_log_gamma_bound <= 0.0
        ):
            raise ValueError(
                "Checkpoint raw_log_gamma_bound must be null or finite and "
                "strictly positive."
            )
        self.gamma_centering_mode = str(
            state.get("gamma_centering_mode", self.gamma_centering_mode)
        )
        self.gamma_cross_dataset_centering_enabled = (
            self.gamma_centering_mode != "disabled"
        )
        self.gamma_dataset_constant_scale_gauge = str(
            state.get(
                "gamma_dataset_constant_scale_gauge",
                self.gamma_dataset_constant_scale_gauge,
            )
        )
        self.gamma_reference_dataset_names = tuple(
            str(x)
            for x in state.get(
                "gamma_reference_dataset_names",
                self.gamma_reference_dataset_names,
            )
        )
        self.gamma_reference_manifest_hash = str(
            state.get(
                "gamma_reference_manifest_hash",
                self.gamma_reference_manifest_hash,
            )
        )
        self.gamma_centering_weighting = str(
            state.get("gamma_centering_weighting", self.gamma_centering_weighting)
        )
        self.gamma_centering_quality_rank_power = float(
            state.get(
                "gamma_centering_quality_rank_power",
                self.gamma_centering_quality_rank_power,
            )
        )
        self.gamma_reference_chunk_size = int(
            state.get("gamma_reference_chunk_size", self.gamma_reference_chunk_size)
        )
        self.gamma_reference_minimum_datasets = int(
            state.get(
                "gamma_reference_minimum_datasets",
                self.gamma_reference_minimum_datasets,
            )
        )
        self.selected_dataset_names = tuple(
            str(x) for x in state.get("selected_dataset_names", self.selected_dataset_names)
        )
        self.selected_dataset_ids = tuple(
            int(x) for x in state.get("selected_dataset_ids", self.selected_dataset_ids)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Reference-panel length is experiment-specific. Resize the registered
        # buffers before PyTorch performs its normal shape checks.
        for buffer_name in (
            "gamma_reference_dataset_ids",
            "gamma_reference_weights",
            "gamma_selected_dataset_ids",
        ):
            key = prefix + buffer_name
            if key in state_dict:
                setattr(self, buffer_name, state_dict[key].detach().clone())
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        if len(values) != int(batch_size):
            raise ValueError(
                f"Expected {batch_size} sample IDs, got {len(values)}."
            )
        return [str(value) for value in values]

    def _center_log_gamma_across_transcripts(
        self,
        log_gamma_raw: torch.Tensor,
        *,
        mask_b: torch.Tensor,
        sample_ids: Sequence[str] | torch.Tensor | None,
        id_datasets: torch.Tensor,
        dataset_quality_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Apply joint cross-dataset and positional log-gamma gauges.

        For raw scores ``a[d,t,i]`` this computes

        ``g = a - c[t,i] - m[d,t] + a_bar[t]``

        where ``c`` is the configured weighted mean across distinct datasets,
        ``m`` is the valid-position mean for one dataset-transcript pair, and
        ``a_bar`` is the same dataset-weighted mean of ``m``. This makes both
        the weighted cross-dataset log mean and each pair's positional log mean
        zero. Fewer than two distinct datasets retain the documented raw-score
        fallback. Dataset quality weights are used only for this gauge.
        """
        B, T = log_gamma_raw.shape
        dtype = log_gamma_raw.dtype
        device = log_gamma_raw.device
        mask_b = mask_b.bool()
        accum_dtype = torch.float32
        mask_f = mask_b.to(dtype=accum_dtype)
        raw_f = log_gamma_raw.to(dtype=accum_dtype)
        zeros = torch.zeros(B, T, device=device, dtype=accum_dtype)
        false = torch.zeros_like(mask_b)
        dataset_ids = id_datasets.reshape(-1).to(device=device, dtype=torch.long)
        sample_id_list = self._normalize_sample_ids(sample_ids, B)
        if (
            self.gamma_cross_dataset_centering_enabled
            and self.gamma_centering_weighting == "quality_rank"
        ):
            if dataset_quality_weights is None:
                raise ValueError(
                    "gamma_centering.weighting='quality_rank' requires "
                    "dataset_quality_weights in the batch."
                )
            raw_dataset_weights = dataset_quality_weights.reshape(-1).to(
                device=device, dtype=accum_dtype
            )
            if raw_dataset_weights.numel() != B:
                raise ValueError(
                    f"Expected {B} dataset quality weights, got "
                    f"{raw_dataset_weights.numel()}."
                )
            if not torch.isfinite(raw_dataset_weights).all() or torch.any(
                raw_dataset_weights <= 0.0
            ):
                raise ValueError("Dataset quality weights must be finite and positive.")
            raw_dataset_weights = raw_dataset_weights.pow(
                self.gamma_centering_quality_rank_power
            )
        else:
            raw_dataset_weights = torch.ones(B, device=device, dtype=accum_dtype)
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
            or B <= 1
        ):
            return {
                "log_gamma": raw_f * mask_f,
                "gamma_center": zeros,
                "dataset_constant_log_shift": zeros,
                "weights": zeros,
                "applied": false,
                "num_distinct_datasets": zeros,
                "total_weight": zeros,
                "constraint_error": zeros,
                "positional_constraint_error": zeros,
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

        candidate_f = candidate.to(dtype=accum_dtype)
        log_safe = torch.where(candidate, raw_f, zeros)

        def _scatter_sum(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
            return torch.zeros(size, T, device=device, dtype=src.dtype).index_add(
                0, index, src
            )

        count_cell = _scatter_sum(candidate_f, cell_index, num_cells)
        mean_cell = _scatter_sum(
            log_safe, cell_index, num_cells
        ) / count_cell.clamp_min(1.0)
        cell_active = count_cell > 0
        active_f = cell_active.to(dtype=accum_dtype)
        num_distinct_group = _scatter_sum(active_f, cell_group, num_groups)
        # Average duplicate samples within each dataset before weighting the
        # distinct dataset means. Replicas therefore never multiply a
        # dataset's influence on the centering gauge.
        raw_weight_samples = candidate_f * raw_dataset_weights.reshape(-1, 1)
        raw_weight_cell = _scatter_sum(
            raw_weight_samples, cell_index, num_cells
        ) / count_cell.clamp_min(1.0)
        raw_weight_cell = raw_weight_cell * active_f
        total_weight_group = _scatter_sum(raw_weight_cell, cell_group, num_groups)
        weighted_sum_group = _scatter_sum(
            mean_cell * raw_weight_cell, cell_group, num_groups
        )
        center_group = weighted_sum_group / total_weight_group.clamp_min(self.eps)
        # A cross-dataset constraint is defined exactly when at least two
        # distinct datasets contribute at this transcript position.
        apply_group = num_distinct_group >= 2.0

        apply_sample = apply_group.index_select(0, group_index)
        center_sample = center_group.index_select(0, group_index)
        applied = apply_sample & mask_b
        log_center = torch.where(applied, center_sample, zeros)
        num_distinct = torch.where(
            applied, num_distinct_group.index_select(0, group_index), zeros
        )
        dataset_weight_cell = raw_weight_cell / total_weight_group.index_select(
            0, cell_group
        ).clamp_min(self.eps)
        sample_weights = dataset_weight_cell.index_select(
            0, cell_index
        ) / count_cell.index_select(0, cell_index).clamp_min(1.0)
        sample_weights = torch.where(
            candidate & apply_sample, sample_weights, torch.zeros_like(sample_weights)
        )

        # Dataset-level positional means and their group-weighted mean use
        # float32 accumulation even when the forward runs in bf16/fp16.
        cell_position_count = active_f.sum(dim=1).clamp_min(1.0)
        positional_mean_cell = (mean_cell * active_f).sum(dim=1) / cell_position_count
        row_active = candidate.any(dim=1).to(dtype=accum_dtype)
        row_count_cell = torch.zeros(
            num_cells, device=device, dtype=accum_dtype
        ).index_add(0, cell_index, row_active)
        cell_weight = torch.zeros(
            num_cells, device=device, dtype=accum_dtype
        ).index_add(
            0,
            cell_index,
            raw_dataset_weights * row_active,
        ) / row_count_cell.clamp_min(1.0)
        cell_weight = cell_weight * cell_active.any(dim=1).to(dtype=accum_dtype)
        total_weight_scalar = torch.zeros(
            num_groups, device=device, dtype=accum_dtype
        ).index_add(0, cell_group, cell_weight)
        weighted_positional_mean = torch.zeros(
            num_groups, device=device, dtype=accum_dtype
        ).index_add(0, cell_group, cell_weight * positional_mean_cell)
        a_bar_group = weighted_positional_mean / total_weight_scalar.clamp_min(self.eps)

        valid_len = candidate_f.sum(dim=1).clamp_min(1.0)
        requested_positional_mean = (log_safe * candidate_f).sum(dim=1) / valid_len
        a_bar_sample = a_bar_group.index_select(0, group_index)
        positional_shift = (
            requested_positional_mean - a_bar_sample
            if self.gamma_dataset_constant_scale_gauge == "geometric_mean_one"
            else torch.zeros_like(requested_positional_mean)
        )
        positional_shift_2d = positional_shift.reshape(-1, 1).expand(B, T)
        log_gamma = torch.where(
            applied,
            raw_f - center_sample - positional_shift_2d,
            raw_f,
        ) * mask_f

        # Diagnostics are evaluated from the final constrained scores rather
        # than inferred from algebra, making numerical residuals observable.
        final_cell = _scatter_sum(
            log_gamma * candidate_f,
            cell_index,
            num_cells,
        ) / count_cell.clamp_min(1.0)
        cross_residual_group = _scatter_sum(
            final_cell * raw_weight_cell,
            cell_group,
            num_groups,
        ) / total_weight_group.clamp_min(self.eps)
        constraint_error = torch.where(
            applied,
            cross_residual_group.abs().index_select(0, group_index),
            zeros,
        )
        positional_residual = (
            (log_gamma * candidate_f).sum(dim=1) / valid_len
        ).abs().reshape(-1, 1).expand(B, T)

        return {
            "log_gamma": log_gamma,
            "gamma_center": log_center * mask_f,
            "dataset_constant_log_shift": torch.where(
                applied,
                positional_shift_2d,
                zeros,
            ) * mask_f,
            "weights": sample_weights * mask_f,
            "applied": applied & mask_b,
            "num_distinct_datasets": num_distinct * mask_f,
            "total_weight": torch.where(
                applied,
                total_weight_group.index_select(0, group_index),
                zeros,
            ) * mask_f,
            "constraint_error": constraint_error * mask_f,
            "positional_constraint_error": torch.where(
                applied,
                positional_residual,
                zeros,
            ) * mask_f,
            "eligible": candidate,
        }

    @staticmethod
    def _assert_transcript_tensor_consistency(
        *,
        name: str,
        tensor: torch.Tensor | None,
        row_indices: list[int],
    ) -> None:
        if tensor is None or len(row_indices) < 2:
            return
        reference = tensor[row_indices[0]]
        for row_index in row_indices[1:]:
            candidate = tensor[row_index]
            if torch.is_floating_point(reference):
                same = torch.allclose(
                    reference,
                    candidate,
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
            else:
                same = torch.equal(reference, candidate)
            if not same:
                raise ValueError(
                    "Fixed-reference gamma centering requires transcript-invariant "
                    f"{name}, but repeated rows for one transcript differ "
                    f"(rows {row_indices[0]} and {row_index})."
                )

    def _center_log_gamma_fixed_reference(
        self,
        log_gamma_raw: torch.Tensor,
        *,
        mask_b: torch.Tensor,
        sample_ids: Sequence[str] | torch.Tensor | None,
        id_datasets: torch.Tensor,
        codon_ids: torch.Tensor,
        position_features: torch.Tensor,
        dataset_bias_sequence_features: torch.Tensor | None,
        biological_sequence_features: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Center requested raw scores using a fixed, checkpointed dataset panel.

        The reference panel is evaluated for one canonical sequence-derived
        input per transcript. Its membership never comes from the requested
        minibatch. Chunking only changes evaluation memory, not the definition
        of the center, and all weighted sums remain in the autograd graph.
        """
        B, T = log_gamma_raw.shape
        device = log_gamma_raw.device
        dtype = log_gamma_raw.dtype
        mask_b = mask_b.bool()
        accum_dtype = torch.float32
        mask_f = mask_b.to(dtype=accum_dtype)
        raw_requested_f = log_gamma_raw.to(dtype=accum_dtype)
        zeros = torch.zeros(B, T, device=device, dtype=accum_dtype)
        false = torch.zeros_like(mask_b)
        reference_ids = self.gamma_reference_dataset_ids.to(device=device)
        reference_weights = self.gamma_reference_weights.to(
            device=device,
            dtype=accum_dtype,
        )
        reference_count = int(reference_ids.numel())

        if (
            not self.gamma_cross_dataset_centering_enabled
            or reference_count < self.gamma_reference_minimum_datasets
        ):
            return {
                "log_gamma": raw_requested_f * mask_f,
                "gamma_center": zeros,
                "dataset_constant_log_shift": zeros,
                "weights": zeros,
                "applied": false,
                "num_distinct_datasets": zeros,
                "total_weight": zeros,
                "constraint_error": zeros,
                "positional_constraint_error": zeros,
                "eligible": mask_b & torch.isfinite(log_gamma_raw),
                "reference_count": torch.full(
                    (B,),
                    reference_count,
                    device=device,
                    dtype=dtype,
                ),
                "all_requested_in_reference": torch.zeros(
                    B, device=device, dtype=torch.bool
                ),
            }
        if reference_weights.numel() != reference_count:
            raise RuntimeError(
                "Checkpoint gamma reference IDs and weights have different lengths."
            )
        if (
            not torch.isfinite(reference_weights).all()
            or torch.any(reference_weights <= 0.0)
        ):
            raise RuntimeError("Gamma reference weights must be finite and positive.")

        sample_id_list = self._normalize_sample_ids(sample_ids, B)
        if sample_id_list is None:
            # A singleton/manual call need not supply IDs. Treat each row as a
            # separate transcript; identical inputs still receive identical
            # deterministic centers.
            sample_id_list = [f"__gamma_reference_row_{index}" for index in range(B)]

        grouped_rows: dict[str, list[int]] = {}
        for row_index, sample_id in enumerate(sample_id_list):
            grouped_rows.setdefault(sample_id, []).append(row_index)
        canonical_rows = [indices[0] for indices in grouped_rows.values()]
        group_index_by_row = torch.empty(B, device=device, dtype=torch.long)
        for group_index, indices in enumerate(grouped_rows.values()):
            self._assert_transcript_tensor_consistency(
                name="codon IDs",
                tensor=codon_ids,
                row_indices=indices,
            )
            self._assert_transcript_tensor_consistency(
                name="valid-position mask",
                tensor=mask_b,
                row_indices=indices,
            )
            self._assert_transcript_tensor_consistency(
                name="position features",
                tensor=position_features,
                row_indices=indices,
            )
            self._assert_transcript_tensor_consistency(
                name="dataset-bias optional sequence features",
                tensor=dataset_bias_sequence_features,
                row_indices=indices,
            )
            self._assert_transcript_tensor_consistency(
                name="biological sequence features",
                tensor=biological_sequence_features,
                row_indices=indices,
            )
            group_index_by_row[indices] = group_index

        canonical_index = torch.as_tensor(
            canonical_rows,
            device=device,
            dtype=torch.long,
        )
        canonical_mask = mask_b.index_select(0, canonical_index)
        canonical_codons = codon_ids.index_select(0, canonical_index)
        canonical_position = position_features.index_select(0, canonical_index)
        canonical_optional = (
            None
            if dataset_bias_sequence_features is None
            else dataset_bias_sequence_features.index_select(0, canonical_index)
        )
        num_transcripts = len(canonical_rows)
        weighted_sum = torch.zeros(
            num_transcripts,
            T,
            device=device,
            dtype=accum_dtype,
        )
        weighted_positional_mean_sum = torch.zeros(
            num_transcripts,
            device=device,
            dtype=accum_dtype,
        )
        weight_sum = reference_weights.sum()
        chunk_size = min(self.gamma_reference_chunk_size, reference_count)
        requested_ids = id_datasets.reshape(-1).to(device=device, dtype=torch.long)

        # In evaluation mode, complete transcript groups already contain the
        # raw score for every requested reference dataset. Reusing those values
        # is exactly equivalent because dropout is disabled, and avoids a
        # second dataset-bias BiGRU pass during sanity checking/validation.
        # Training deliberately retains the original independent reference
        # evaluation so its dropout semantics are unchanged.
        reusable_reference_rows: list[list[int]] = []
        can_reuse_requested_reference = not self.training
        if can_reuse_requested_reference:
            reference_id_list = [int(value) for value in reference_ids.tolist()]
            for indices in grouped_rows.values():
                row_by_dataset: dict[int, int] = {}
                for row_index in indices:
                    dataset_id = int(requested_ids[row_index].item())
                    if dataset_id in row_by_dataset:
                        can_reuse_requested_reference = False
                        break
                    row_by_dataset[dataset_id] = int(row_index)
                if not can_reuse_requested_reference or any(
                    dataset_id not in row_by_dataset
                    for dataset_id in reference_id_list
                ):
                    can_reuse_requested_reference = False
                    break
                reusable_reference_rows.append(
                    [row_by_dataset[dataset_id] for dataset_id in reference_id_list]
                )

        reference_mask = canonical_mask.to(dtype=accum_dtype).unsqueeze(1)
        reference_valid_len = reference_mask.sum(dim=2).clamp_min(1.0)
        if can_reuse_requested_reference:
            row_matrix = torch.as_tensor(
                reusable_reference_rows,
                device=device,
                dtype=torch.long,
            )
            raw_reference = raw_requested_f.index_select(
                0,
                row_matrix.reshape(-1),
            ).reshape(num_transcripts, reference_count, T)
            weighted_sum = (
                raw_reference * reference_weights.reshape(1, reference_count, 1)
            ).sum(dim=1)
            reference_positional_mean = (
                raw_reference * reference_mask
            ).sum(dim=2) / reference_valid_len
            weighted_positional_mean_sum = (
                reference_positional_mean
                * reference_weights.reshape(1, reference_count)
            ).sum(dim=1)
        else:
            for start in range(0, reference_count, chunk_size):
                stop = min(start + chunk_size, reference_count)
                ids_chunk = reference_ids[start:stop]
                weights_chunk = reference_weights[start:stop]
                chunk_count = int(ids_chunk.numel())
                synthetic_ids = ids_chunk.repeat(num_transcripts)
                synthetic_mask = canonical_mask.repeat_interleave(chunk_count, dim=0)
                synthetic_codons = canonical_codons.repeat_interleave(chunk_count, dim=0)
                synthetic_position = canonical_position.repeat_interleave(
                    chunk_count, dim=0
                )
                synthetic_optional = (
                    None
                    if canonical_optional is None
                    else canonical_optional.repeat_interleave(chunk_count, dim=0)
                )
                raw_reference = self.dataset_bias_model(
                    dataset_ids=synthetic_ids,
                    mask=synthetic_mask,
                    codon_ids=synthetic_codons,
                    position_features=synthetic_position,
                    sequence_features=synthetic_optional,
                    compute_log_sigma=False,
                    embedding_center_ids=self.gamma_selected_dataset_ids,
                )["gamma_raw"]
                raw_reference = (
                    raw_reference.to(device=device, dtype=accum_dtype)
                    + float(self.gamma_log_init)
                )
                raw_reference = raw_reference.reshape(
                    num_transcripts,
                    chunk_count,
                    T,
                )
                weighted_sum = weighted_sum + (
                    raw_reference * weights_chunk.reshape(1, chunk_count, 1)
                ).sum(dim=1)
                reference_positional_mean = (
                    raw_reference * reference_mask
                ).sum(dim=2) / reference_valid_len
                weighted_positional_mean_sum = weighted_positional_mean_sum + (
                    reference_positional_mean * weights_chunk.reshape(1, chunk_count)
                ).sum(dim=1)

        center_by_transcript = weighted_sum / weight_sum.clamp_min(self.eps)
        a_bar_by_transcript = (
            weighted_positional_mean_sum / weight_sum.clamp_min(self.eps)
        )
        center = center_by_transcript.index_select(0, group_index_by_row)
        applied = mask_b & torch.isfinite(log_gamma_raw)
        log_center = torch.where(applied, center, zeros)
        requested_valid_len = mask_f.sum(dim=1).clamp_min(1.0)
        requested_positional_mean = (
            torch.where(applied, raw_requested_f, zeros) * mask_f
        ).sum(dim=1) / requested_valid_len
        a_bar = a_bar_by_transcript.index_select(0, group_index_by_row)
        positional_shift = (
            requested_positional_mean - a_bar
            if self.gamma_dataset_constant_scale_gauge == "geometric_mean_one"
            else torch.zeros_like(requested_positional_mean)
        )
        positional_shift_2d = positional_shift.reshape(-1, 1).expand(B, T)
        log_gamma = torch.where(
            applied,
            raw_requested_f - center - positional_shift_2d,
            zeros,
        ) * mask_f

        # The weighted residual is computed without retaining all reference
        # scores: sum w(q-c) = sum(wq) - c*sum(w).
        constraint_numerator = weighted_sum - center_by_transcript * weight_sum
        if self.gamma_dataset_constant_scale_gauge == "geometric_mean_one":
            constraint_numerator = (
                constraint_numerator
                - weighted_positional_mean_sum.reshape(-1, 1)
                + a_bar_by_transcript.reshape(-1, 1) * weight_sum
            )
        constraint_by_transcript = (
            constraint_numerator.abs() / weight_sum.clamp_min(self.eps)
        )
        constraint = constraint_by_transcript.index_select(0, group_index_by_row)
        positional_constraint = (
            (log_gamma * mask_f).sum(dim=1) / requested_valid_len
        ).abs().reshape(-1, 1).expand(B, T)

        requested_in_reference = (
            requested_ids.reshape(-1, 1) == reference_ids.reshape(1, -1)
        ).any(dim=1)
        normalized_reference_weights = reference_weights / weight_sum.clamp_min(self.eps)
        requested_weights = (
            (
                requested_ids.reshape(-1, 1)
                == reference_ids.reshape(1, -1)
            ).to(dtype=accum_dtype)
            * normalized_reference_weights.reshape(1, -1)
        ).sum(dim=1)

        return {
            "log_gamma": log_gamma,
            "gamma_center": log_center * mask_f,
            "dataset_constant_log_shift": positional_shift_2d * mask_f,
            "weights": requested_weights.reshape(-1, 1).expand(B, T) * mask_f,
            "applied": applied,
            "num_distinct_datasets": torch.full_like(
                log_gamma_raw,
                float(reference_count),
            )
            * mask_f,
            "total_weight": weight_sum.expand_as(log_gamma_raw) * mask_f,
            "constraint_error": constraint * mask_f,
            "positional_constraint_error": positional_constraint * mask_f,
            "eligible": applied,
            "reference_count": torch.full(
                (B,),
                reference_count,
                device=device,
                dtype=dtype,
            ),
            "all_requested_in_reference": requested_in_reference.all().expand(B),
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

    @staticmethod
    def _canonical_transcript_rows(
        transcript_group_index: torch.Tensor,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, int]:
        """Return the first pair row for every contiguous transcript group.

        Collate assigns group IDs in first-occurrence order, starting from zero.
        Keeping that order makes the unique packed biological batch and this
        pair-to-transcript gather map agree without any string processing.
        """
        groups = transcript_group_index.reshape(-1).long()
        if groups.numel() != int(batch_size):
            raise ValueError(
                "transcript_group_index must contain one entry per pair row; "
                f"got {groups.numel()} entries for batch size {batch_size}."
            )
        if groups.numel() == 0 or bool(torch.any(groups < 0)):
            raise ValueError("transcript_group_index must be nonempty and nonnegative.")
        unique_groups = torch.unique(groups, sorted=True)
        expected = torch.arange(
            unique_groups.numel(),
            device=groups.device,
            dtype=groups.dtype,
        )
        if not torch.equal(unique_groups, expected):
            raise ValueError(
                "transcript_group_index must be contiguous from zero within each batch."
            )
        canonical_rows = torch.stack(
            [
                torch.nonzero(groups == group, as_tuple=False)[0, 0]
                for group in unique_groups
            ]
        )
        return canonical_rows, int(unique_groups.numel())

    def _forward_biological_unique_transcripts(
        self,
        *,
        x_packed,
        mask_b: torch.Tensor,
        transcript_group_index: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Evaluate the shared branch once per transcript and gather to pairs.

        Gathering a unique transcript output into all of its dataset rows keeps
        the complete autograd graph. Consequently, gradients contributed by all
        pair losses are summed at the gather operation before flowing through
        the single biological evaluation.
        """
        batch_size, maximum_length = mask_b.shape
        if transcript_group_index is None:
            padded, _ = pad_packed_sequence(
                x_packed,
                batch_first=True,
                total_length=maximum_length,
            )
            if padded.shape[0] != batch_size:
                raise ValueError(
                    "A pair-row biological packed batch must contain one sequence "
                    f"per pair row; got {padded.shape[0]} for {batch_size} rows."
                )
            return self.biological_model(x_packed, mask_b), padded

        groups = transcript_group_index.to(device=mask_b.device, dtype=torch.long)
        canonical_rows, unique_count = self._canonical_transcript_rows(
            groups,
            batch_size=batch_size,
        )
        unique_mask = mask_b.index_select(0, canonical_rows)
        unique_padded, _ = pad_packed_sequence(
            x_packed,
            batch_first=True,
            total_length=maximum_length,
        )
        if unique_padded.shape[0] != unique_count:
            raise ValueError(
                "The packed biological batch must contain exactly one sequence "
                "per transcript group; got "
                f"{unique_padded.shape[0]} sequences for {unique_count} groups."
            )

        pair_padded = unique_padded.index_select(0, groups)
        if self.training and float(self.biological_model.dropout) > 0.0:
            # Preserve the former independent per-pair dropout masks for any
            # nonbaseline configuration that enables biological dropout.
            pair_lengths = mask_b.sum(dim=1).clamp_min(1).to("cpu")
            pair_packed = pack_padded_sequence(
                pair_padded,
                pair_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            return self.biological_model(pair_packed, mask_b), pair_padded

        biological_unique = self.biological_model(x_packed, unique_mask)
        biological_pairs: dict[str, torch.Tensor] = {}
        for name, value in biological_unique.items():
            if name == "h_n":
                # Hidden state is not consumed downstream. Retain a correctly
                # gathered diagnostic form without changing recurrent compute.
                biological_pairs[name] = value.index_select(1, groups)
            else:
                biological_pairs[name] = value.index_select(0, groups)
        return biological_pairs, pair_padded

    def forward(
        self,
        x_packed,
        codon_ids: torch.Tensor,
        id_datasets: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        sample_ids: Sequence[str] | torch.Tensor | None = None,
        transcript_group_index: torch.Tensor | None = None,
        dataset_bias_sequence_features: torch.Tensor | None = None,
        dataset_quality_weights: torch.Tensor | None = None,
    ):
        # --------------------------------------------------------
        # 1. Biological branch -> queue load
        # --------------------------------------------------------
        mask_b = mask.bool()
        bio, biological_sequence_features = (
            self._forward_biological_unique_transcripts(
                x_packed=x_packed,
                mask_b=mask_b,
                transcript_group_index=transcript_group_index,
            )
        )
        L_bio = bio["L_bio"]

        dtype = L_bio.dtype
        device = L_bio.device
        B, _ = mask_b.shape
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
        # 2. Dataset branch -> gamma residual + NB dispersion
        # --------------------------------------------------------
        position_features = self.make_position_features(mask=mask_b, dtype=dtype)
        bias = self.dataset_bias_model(
            dataset_ids=id_datasets,
            mask=mask_b,
            codon_ids=codon_ids,
            position_features=position_features,
            sequence_features=dataset_bias_sequence_features,
            embedding_center_ids=(
                self.gamma_selected_dataset_ids
                if self.gamma_centering_mode == "fixed_reference"
                else None
            ),
            compute_log_sigma=self.alpha_mode == "learned",
        )
        gamma_log_residual = bias["gamma_raw"].to(dtype=dtype, device=device) * mask_f
        gamma_raw_bound_active_fraction = bias.get(
            "gamma_raw_bound_active_fraction",
            torch.zeros(B, device=device, dtype=dtype),
        ).to(dtype=dtype, device=device)
        if self.alpha_mode == "learned":
            log_sigma = bias["log_sigma"].to(dtype=dtype)
        else:
            # The synthetic oracle uses the known generative NB2 dispersion.
            # Invalid padded positions retain the historical zero sentinel;
            # every valid position is exactly log(fixed_alpha).
            log_sigma = self.fixed_log_alpha.to(device=device).expand(
                L_bio.shape
            )
        log_sigma = torch.where(mask_b, log_sigma, torch.zeros_like(log_sigma))

        # --------------------------------------------------------
        # 3. Observation mean branch
        # --------------------------------------------------------
        # Center the unconstrained raw log-score directly. No transformation is
        # applied after centering other than exp, so the exact zero-mean log
        # constraint is preserved by the final gamma.
        log_gamma_raw = torch.where(
            mask_b,
            gamma_log_residual + float(self.gamma_log_init),
            torch.zeros_like(gamma_log_residual),
        )
        gamma_raw = torch.exp(log_gamma_raw)
        if self.gamma_centering_mode == "fixed_reference":
            centered = self._center_log_gamma_fixed_reference(
                log_gamma_raw,
                mask_b=mask_b,
                sample_ids=sample_ids,
                id_datasets=id_datasets,
                codon_ids=codon_ids,
                position_features=position_features,
                dataset_bias_sequence_features=dataset_bias_sequence_features,
                biological_sequence_features=biological_sequence_features,
            )
        else:
            centered = self._center_log_gamma_across_transcripts(
                log_gamma_raw,
                mask_b=mask_b,
                sample_ids=sample_ids,
                id_datasets=id_datasets,
                dataset_quality_weights=dataset_quality_weights,
            )
        gamma_eligible = centered["eligible"]
        log_gamma = centered["log_gamma"]
        gamma_cross_dataset_log_center = centered["gamma_center"]
        gamma_cross_dataset_center_group_size = centered[
            "num_distinct_datasets"
        ].amax(dim=1)
        gamma_cross_dataset_center_applied = centered["applied"].any(dim=1).to(
            dtype=dtype
        )
        # This is the final multiplicative correction. Padded positions have
        # log_gamma=0 and therefore gamma=1; all entries are strictly positive.
        gamma = torch.exp(log_gamma)

        # --------------------------------------------------------
        # 4. Target-derived mean scale S
        # --------------------------------------------------------
        scale_dt = self._target_scale_dt(target, mask_b, dtype, device)  # [B, 1]

        # --------------------------------------------------------
        # 5. Prediction
        # --------------------------------------------------------
        mu_inner = gamma * L_bio
        if self.mass_conservation:
            # Renormalize the shape to mean 1 over valid positions so that
            # mean_valid(mu) = S exactly. gamma and L_bio keep their
            # per-position meaning; only the overall shape level is pinned.
            inner_valid_len = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            inner_mean = (mu_inner * mask_f).sum(dim=1, keepdim=True) / inner_valid_len
            mu_inner = mu_inner / inner_mean.clamp_min(self.eps)
        mu = scale_dt * mu_inner
        # The mean has no configured upper bound. Keep only finite numerical
        # values for pathological overflow/underflow cases.
        mu = torch.nan_to_num(
            mu,
            nan=self.eps,
            posinf=torch.finfo(mu.dtype).max,
            neginf=self.eps,
        )
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
            "gamma_raw_bound_active_fraction": gamma_raw_bound_active_fraction,
            "gamma_raw": gamma_raw,
            "log_gamma_raw": log_gamma_raw,
            "gamma_cross_dataset_log_center": torch.where(
                mask_b,
                gamma_cross_dataset_log_center,
                torch.zeros_like(gamma_cross_dataset_log_center),
            ),
            "gamma_dataset_constant_log_shift": torch.where(
                mask_b,
                centered.get(
                    "dataset_constant_log_shift",
                    torch.zeros_like(log_gamma),
                ),
                torch.zeros_like(log_gamma),
            ),
            "gamma_cross_dataset_center_group_size": gamma_cross_dataset_center_group_size,
            "gamma_cross_dataset_center_applied": gamma_cross_dataset_center_applied,
            "gamma_centering_reliability": torch.where(
                mask_b,
                centered["weights"],
                torch.zeros_like(centered["weights"]),
            ),
            "gamma_centering_weight": torch.where(
                mask_b,
                centered["weights"],
                torch.zeros_like(centered["weights"]),
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
            "gamma_positional_gauge_constraint_error": torch.where(
                mask_b,
                centered.get(
                    "positional_constraint_error",
                    torch.zeros_like(log_gamma),
                ),
                torch.zeros_like(log_gamma),
            ),
            "gamma_centering_mode": self.gamma_centering_mode,
            "gamma_dataset_constant_scale_gauge": (
                self.gamma_dataset_constant_scale_gauge
            ),
            "gamma_reference_dataset_count": centered.get(
                "reference_count",
                torch.zeros(B, device=device, dtype=dtype),
            ),
            "gamma_reference_dataset_ids": self.gamma_reference_dataset_ids,
            "gamma_reference_manifest_hash": self.gamma_reference_manifest_hash,
            "gamma_reference_weighting": self.gamma_centering_weighting,
            "gamma_reference_quality_rank_power": float(
                self.gamma_centering_quality_rank_power
            ),
            "gamma_reference_chunk_size": int(self.gamma_reference_chunk_size),
            "gamma_all_requested_in_reference": centered.get(
                "all_requested_in_reference",
                torch.zeros(B, device=device, dtype=torch.bool),
            ),
            "gamma": gamma,
            "log_gamma": log_gamma,
            "lambda_bio_mean": lambda_bio_mean,
            "lambda_bio_max": lambda_bio_max,
            "lambda_bio_min": lambda_bio_min,
            "L_bio_mean": L_bio_mean,
            "L_bio_max": L_bio_max,
            "J_mean": J_flat,
            "J_min": J_flat,
            "J_max": J_flat,
            "mu": mu,
            "normalized_shape": mu_inner,
            "log_sigma": log_sigma,
            "log_sigma_t": (log_sigma * mask_f).sum(dim=1, keepdim=True)
            / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0),
            "alpha": alpha,
            "valid_len": valid_len,
        }

        return mu, log_sigma, extras
