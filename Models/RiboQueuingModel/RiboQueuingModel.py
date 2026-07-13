from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    does not identify the additive branch; additive-bias regularization and
    optional reference anchors remain separate objective terms.
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


        biological_params = dict(model_configs["biological_params"])
        biological_params.setdefault("eps", self.eps)
        self.biological_model = QueuingBiologicalModel(config_params=biological_params)

        # ``active_dataset_ids`` remains in the public constructor because the
        # training entry point and historical callers pass it. Gamma no longer
        # owns per-dataset response parameters, so it has no active use here.
        del active_dataset_ids
        dataset_bias_params = dict(model_configs["dataset_bias_params"])
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
    # Gamma-centering configuration
    # ============================================================

    @staticmethod
    def _as_plain_dict(value) -> dict:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return {str(k): v for k, v in value.items()}
        return dict(value)

    @staticmethod
    def _parse_reference_dataset_ids(values) -> set[int]:
        if values is None:
            return set()
        ids: set[int] = set()
        for value in list(values):
            try:
                ids.add(int(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "gamma_centering.reference_dataset_ids must contain integer "
                    f"dataset ids; got {value!r}."
                ) from exc
        return ids

    def _configure_gamma_centering(self, model_configs: Mapping) -> None:
        old_cfg = self._as_plain_dict(
            model_configs.get("gamma_cross_dataset_centering", {})
        )
        new_cfg = self._as_plain_dict(model_configs.get("gamma_centering", {}))

        if old_cfg and new_cfg:
            conflict_checks = {
                "enabled": ("enabled", bool),
                "strength": ("strength", float),
                "min_datasets": ("min_distinct_datasets", int),
            }
            for old_key, (new_key, caster) in conflict_checks.items():
                if old_key not in old_cfg or new_key not in new_cfg:
                    continue
                old_value = caster(old_cfg[old_key])
                new_value = caster(new_cfg[new_key])

        cfg = dict(new_cfg)
        if old_cfg:
            cfg.setdefault("enabled", old_cfg.get("enabled", False))
            cfg.setdefault("strength", old_cfg.get("strength", 1.0))
            cfg.setdefault(
                "min_distinct_datasets",
                old_cfg.get("min_datasets", old_cfg.get("min_distinct_datasets", 2)),
            )

        self.gamma_cross_dataset_centering_enabled = bool(cfg.get("enabled", False))
        self.gamma_centering_scope = str(cfg.get("scope", "batch_grouped")).lower()
        allowed_scopes = {"batch_grouped", "global_running", "disabled"}
        if self.gamma_centering_scope == "global_running":
            raise NotImplementedError(
                "gamma_centering.scope='global_running' is not implemented. Use "
                "'batch_grouped' with transcript-grouped batches or 'disabled'."
            )
        if self.gamma_centering_scope == "disabled":
            self.gamma_cross_dataset_centering_enabled = False

        self.gamma_cross_dataset_centering_strength = float(cfg.get("strength", 1.0))
        self.gamma_centering_eps = float(cfg.get("eps", 1.0e-8))
        self.gamma_reference_mode = str(
            cfg.get("reference_mode", "reliable_geometric_center")
        ).lower()
        allowed_modes = {
            "all_eligible",
            "reliable_geometric_center",
            "trusted_datasets_only",
            "single_reference_dataset",
        }
        self.gamma_reference_dataset_ids = self._parse_reference_dataset_ids(
            cfg.get("reference_dataset_ids", [])
        )
        self.gamma_reference_hard_anchor = bool(cfg.get("reference_hard_anchor", False))

        self.gamma_centering_reliability_mode = str(
            cfg.get("reliability_mode", "uniform")
        ).lower()
        allowed_reliability_modes = {
            "uniform",
            "transcript_depth",
            "local_coverage",
            "replicate_agreement",
            "combined",
        }
        self.gamma_centering_reliability_detach = bool(
            cfg.get("reliability_detach", True)
        )
        self.gamma_centering_min_reliability = float(cfg.get("min_reliability", 0.0))
        self.gamma_centering_min_total_reliability = float(
            cfg.get("min_total_reliability", 0.0)
        )
        self.gamma_centering_min_distinct_datasets = int(
            cfg.get("min_distinct_datasets", 2)
        )
        # Backward-compatible alias used by older tests and metrics.
        self.gamma_cross_dataset_centering_min_datasets = (
            self.gamma_centering_min_distinct_datasets
        )

        self.gamma_centering_transcript_depth_kappa = float(
            cfg.get("transcript_depth_kappa", 100.0)
        )
        self.gamma_centering_local_coverage_window = int(
            cfg.get("local_coverage_window", 5)
        )
        self.gamma_centering_local_coverage_kappa = float(
            cfg.get("local_coverage_kappa", 1.0)
        )
        self.gamma_centering_replicate_agreement_tau = float(
            cfg.get("replicate_agreement_tau", 0.5)
        )
        self.gamma_centering_duplicate_dataset_reliability_reduction = str(
            cfg.get("duplicate_dataset_reliability_reduction", "mean")
        ).lower()
        allowed_duplicate_reductions = {"mean", "maximum", "max", "capped_sum"}
        self.gamma_centering_combined_reduction = str(
            cfg.get("combined_reliability_reduction", "geometric_mean")
        ).lower()

        self.gamma_log_min = float(cfg.get("log_gamma_min", -8.0))
        self.gamma_log_max = float(cfg.get("log_gamma_max", 8.0))
        self.gamma_centering_log_diagnostics = bool(cfg.get("log_diagnostics", True))

    @staticmethod
    def _inv_softplus(x: float) -> float:
        x = float(x)
        if x > 20.0:
            return x
        return math.log(math.expm1(x))

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

    def _transcript_ids_valid_mask(
        self,
        sample_ids: Sequence[str] | torch.Tensor | None,
        batch_size: int,
        *,
        device: torch.device,
    ) -> tuple[list[str] | None, torch.Tensor]:
        sample_id_list = self._normalize_sample_ids(sample_ids, batch_size)
        valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
        if sample_id_list is None:
            return None, valid
        valid_values = [
            (sid is not None)
            and str(sid) != ""
            and str(sid).lower() not in {"none", "nan", "null"}
            for sid in sample_id_list
        ]
        valid = torch.as_tensor(valid_values, device=device, dtype=torch.bool)
        return sample_id_list, valid

    def _local_coverage_reliability(
        self,
        target: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> torch.Tensor:
        dtype = target.dtype
        device = target.device
        h = max(int(self.gamma_centering_local_coverage_window), 0)
        width = 2 * h + 1
        y = torch.nan_to_num(target.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        y = y.clamp_min(0.0) * mask_b.to(dtype=dtype)
        m = mask_b.to(dtype=dtype)

        kernel = torch.ones(1, 1, width, device=device, dtype=dtype)
        y_sum = F.conv1d(y.unsqueeze(1), kernel, padding=h).squeeze(1)
        m_sum = F.conv1d(m.unsqueeze(1), kernel, padding=h).squeeze(1)
        local_mean = y_sum / m_sum.clamp_min(1.0)
        kappa = max(float(self.gamma_centering_local_coverage_kappa), 0.0)
        return local_mean / (local_mean + kappa + self.gamma_centering_eps)

    def _transcript_depth_reliability(
        self,
        target: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> torch.Tensor:
        dtype = target.dtype
        y = torch.nan_to_num(target.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        y = y.clamp_min(0.0) * mask_b.to(dtype=dtype)
        depth = y.sum(dim=1, keepdim=True)
        kappa = max(float(self.gamma_centering_transcript_depth_kappa), 0.0)
        rel = depth / (depth + kappa + self.gamma_centering_eps)
        return rel.expand_as(target)

    def _replicate_agreement_reliability(
        self,
        replica_profiles: torch.Tensor | None,
        replica_mask: torch.Tensor | None,
        target_shape: torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        B, T = int(target_shape[0]), int(target_shape[1])
        rel = torch.ones(B, T, device=device, dtype=dtype)
        if replica_profiles is None or replica_mask is None:
            return rel

        reps = replica_profiles.detach().to(device=device, dtype=dtype)
        rep_mask = replica_mask.detach().to(device=device).bool()

        tau = max(float(self.gamma_centering_replicate_agreement_tau), self.gamma_centering_eps)
        for b in range(B):
            valid_reps = rep_mask[b]
            if int(valid_reps.sum().item()) < 2:
                continue
            values = torch.log1p(reps[b, valid_reps].clamp_min(0.0))
            median = values.median(dim=0).values
            mad = (values - median.unsqueeze(0)).abs().median(dim=0).values
            rel[b] = torch.exp(-mad / tau)
        return rel

    def _build_gamma_centering_reliability_and_eligibility(
        self,
        *,
        target: torch.Tensor,
        mask_b: torch.Tensor,
        sample_ids: Sequence[str] | torch.Tensor | None,
        id_datasets: torch.Tensor,
        gamma_centering_reliability: torch.Tensor | None = None,
        gamma_centering_eligible: torch.Tensor | None = None,
        replica_profiles: torch.Tensor | None = None,
        replica_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str] | None]:
        B, T = target.shape
        dtype = target.dtype
        device = target.device
        sample_id_list, transcript_valid = self._transcript_ids_valid_mask(
            sample_ids,
            B,
            device=device,
        )
        dataset_valid = id_datasets.reshape(-1).to(device=device, dtype=torch.long) >= 0

        if gamma_centering_reliability is not None:
            reliability = gamma_centering_reliability.to(device=device, dtype=dtype)
            reliability = reliability.detach()
        else:
            mode = self.gamma_centering_reliability_mode
            components: list[torch.Tensor] = []
            if mode == "uniform":
                reliability = torch.ones_like(target, dtype=dtype)
            else:
                if mode in {"transcript_depth", "combined"}:
                    components.append(self._transcript_depth_reliability(target, mask_b))
                if mode in {"local_coverage", "combined"}:
                    components.append(self._local_coverage_reliability(target, mask_b))
                if mode in {"replicate_agreement", "combined"}:
                    components.append(
                        self._replicate_agreement_reliability(
                            replica_profiles,
                            replica_mask,
                            target.shape,
                            dtype=dtype,
                            device=device,
                        )
                    )
                if not components:
                    reliability = torch.ones_like(target, dtype=dtype)
                elif mode == "combined":
                    stacked = torch.stack(
                        [c.clamp(0.0, 1.0) for c in components],
                        dim=0,
                    )
                    if self.gamma_centering_combined_reduction == "product":
                        reliability = stacked.prod(dim=0)
                    else:
                        reliability = torch.exp(
                            torch.log(stacked.clamp_min(self.gamma_centering_eps)).mean(dim=0)
                        )
                else:
                    reliability = components[0]

            if self.gamma_centering_reliability_detach:
                reliability = reliability.detach()

        reliability = torch.nan_to_num(
            reliability,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        eligible = (
            mask_b
            & transcript_valid.reshape(-1, 1)
            & dataset_valid.reshape(-1, 1)
            & (reliability >= float(self.gamma_centering_min_reliability))
        )
        if gamma_centering_eligible is not None:
            supplied = gamma_centering_eligible.to(device=device).bool()
            eligible = eligible & supplied.detach()

        reliability = reliability * mask_b.to(dtype=dtype)
        return reliability.detach(), eligible.detach(), sample_id_list

    def _dataset_level_reliability(self, values: torch.Tensor) -> torch.Tensor:
        reduction = self.gamma_centering_duplicate_dataset_reliability_reduction
        if reduction == "mean":
            return values.mean()
        if reduction in {"maximum", "max"}:
            return values.max()
        if reduction == "capped_sum":
            return values.sum().clamp(max=1.0)
        raise RuntimeError(
            "Unsupported duplicate_dataset_reliability_reduction "
            f"{reduction!r}."
        )

    def _center_log_gamma_across_transcripts(
        self,
        log_gamma_raw: torch.Tensor,
        *,
        mask_b: torch.Tensor,
        sample_ids: Sequence[str] | torch.Tensor | None = None,
        sample_id_list: list[str] | None = None,
        id_datasets: torch.Tensor,
        reliability: torch.Tensor,
        eligible: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B = int(log_gamma_raw.shape[0])
        dtype = log_gamma_raw.dtype
        device = log_gamma_raw.device
        centering_enabled = self.gamma_cross_dataset_centering_enabled
        centering_strength = self.gamma_cross_dataset_centering_strength
        zeros_pos = torch.zeros_like(log_gamma_raw)
        zeros_bool = torch.zeros_like(mask_b, dtype=torch.bool)
        zeros_count = torch.zeros_like(log_gamma_raw)
        dataset_ids = id_datasets.reshape(-1).to(device=device, dtype=torch.long)
        reference_ids = set(self.gamma_reference_dataset_ids)
        if sample_id_list is None:
            sample_id_list = self._normalize_sample_ids(sample_ids, B)

        # Reference-dataset anchor mask (empty unless a reference mode is used).
        if reference_ids:
            reference_dataset_ids_t = torch.as_tensor(
                sorted(reference_ids), device=device, dtype=torch.long
            )
            is_reference_sample = (
                dataset_ids.reshape(-1, 1) == reference_dataset_ids_t.reshape(1, -1)
            ).any(dim=1)
            reference_anchor_mask = (
                is_reference_sample.reshape(-1, 1) & eligible.bool() & mask_b.bool()
            )
        else:
            is_reference_sample = torch.zeros(B, device=device, dtype=torch.bool)
            reference_anchor_mask = torch.zeros_like(mask_b, dtype=torch.bool)

        if (
            sample_id_list is None
            or not centering_enabled
            or centering_strength <= 0.0
            or B <= 1
        ):
            log_gamma = log_gamma_raw * mask_b.to(dtype=dtype)
            return {
                "log_gamma": log_gamma,
                "gamma_center": zeros_pos,
                "weights": zeros_pos,
                "applied": zeros_bool,
                "num_distinct_datasets": zeros_count,
                "total_reliability": zeros_pos,
                "constraint_error": zeros_pos,
                "reference_anchor_mask": reference_anchor_mask,
            }

        # ------------------------------------------------------------------
        # Vectorized reliability-weighted cross-dataset log-gamma centering.
        #
        # Same semantics as the earlier per-(group, position, dataset) Python
        # loop, but every reduction is a masked scatter over the flat [B, T]
        # tensors. There is no Python loop over codon positions and no
        # host<->device synchronization inside the batch, so the cost drops from
        # O(groups * T) tiny CUDA launches to a handful of full-tensor kernels.
        #
        #   center[g, i] = sum_d w_d * ( sum_{s in (g,d)} r_s * loggamma_s
        #                                / sum_{s in (g,d)} r_s )
        # with dataset weights w_d proportional to the dataset-level reliability,
        # applied only where at least `required_datasets` distinct datasets are
        # present and the total reliability clears the configured floor.
        # ------------------------------------------------------------------
        T = int(log_gamma_raw.shape[1])
        eps = float(self.gamma_centering_eps)
        strength = float(centering_strength)
        min_datasets = int(self.gamma_centering_min_distinct_datasets)
        required_datasets = (
            1
            if self.gamma_reference_mode == "single_reference_dataset"
            else min_datasets
        )
        mask_f = mask_b.to(dtype=dtype)

        # Transcript groups: map each (string) transcript id to a dense integer.
        # This single pass is over the batch dimension (B), never over T.
        group_lookup: dict[str, int] = {}
        group_index = torch.as_tensor(
            [group_lookup.setdefault(sid, len(group_lookup)) for sid in sample_id_list],
            device=device,
            dtype=torch.long,
        )
        num_groups = len(group_lookup)

        # Datasets present in the batch, compacted to [0, num_datasets).
        unique_dataset_ids, dataset_index = torch.unique(
            dataset_ids, return_inverse=True
        )
        dataset_index = dataset_index.to(device=device, dtype=torch.long)
        num_datasets = int(unique_dataset_ids.numel())

        # Level-1 cell = (group, dataset); cell index in [0, num_groups*num_datasets).
        num_cells = num_groups * num_datasets
        cell_index = group_index * num_datasets + dataset_index  # [B]
        cell_group = torch.arange(num_cells, device=device, dtype=torch.long) // num_datasets

        # Candidate samples: eligible, in-mask, finite, and (for reference modes)
        # restricted to the reference datasets.
        candidate = eligible.bool() & mask_b.bool() & torch.isfinite(log_gamma_raw)
        if self.gamma_reference_mode in {
            "trusted_datasets_only",
            "single_reference_dataset",
        }:
            candidate = candidate & is_reference_sample.reshape(-1, 1)
        candidate_f = candidate.to(dtype=dtype)

        r = reliability.to(dtype=dtype).clamp_min(0.0)
        rc = torch.where(candidate, r, torch.zeros_like(r))  # [B, T]
        log_safe = torch.where(
            candidate, log_gamma_raw, torch.zeros_like(log_gamma_raw)
        )

        def _scatter_sum(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
            return torch.zeros(size, T, device=device, dtype=src.dtype).index_add(
                0, index, src
            )

        # Level 1: per (group, dataset) reliability-weighted mean of log-gamma.
        rsum_cell = _scatter_sum(rc, cell_index, num_cells)          # [C, T]
        wlog_cell = _scatter_sum(rc * log_safe, cell_index, num_cells)
        count_cell = _scatter_sum(candidate_f, cell_index, num_cells)
        cell_active = rsum_cell > 0.0
        dataset_log_cell = wlog_cell / rsum_cell.clamp_min(eps)      # [C, T]

        # Dataset-level reliability reduction over the cell's candidate samples.
        reduction = self.gamma_centering_duplicate_dataset_reliability_reduction
        if reduction == "mean":
            dataset_rel_cell = rsum_cell / count_cell.clamp_min(1.0)
        elif reduction in {"maximum", "max"}:
            cell_index_bt = cell_index.reshape(-1, 1).expand(B, T)
            dataset_rel_cell = torch.zeros(
                num_cells, T, device=device, dtype=dtype
            ).scatter_reduce(0, cell_index_bt, rc, reduce="amax", include_self=True)
        elif reduction == "capped_sum":
            dataset_rel_cell = rsum_cell.clamp(max=1.0)
        else:
            raise RuntimeError(
                "Unsupported duplicate_dataset_reliability_reduction "
                f"{reduction!r}."
            )

        if self.gamma_reference_mode == "all_eligible":
            center_rel_cell = cell_active.to(dtype=dtype)
        else:
            center_rel_cell = dataset_rel_cell
        center_rel_cell = torch.nan_to_num(
            center_rel_cell, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        center_rel_cell = torch.where(
            cell_active, center_rel_cell, torch.zeros_like(center_rel_cell)
        )

        # Level 2: aggregate across the datasets present in each group.
        rel_total_group = _scatter_sum(center_rel_cell, cell_group, num_groups)
        num_distinct_group = _scatter_sum(
            cell_active.to(dtype=dtype), cell_group, num_groups
        )
        weighted_log_group = _scatter_sum(
            center_rel_cell
            * torch.where(cell_active, dataset_log_cell, torch.zeros_like(dataset_log_cell)),
            cell_group,
            num_groups,
        )
        center_group = weighted_log_group / rel_total_group.clamp_min(eps)  # [G, T]

        apply_group = (
            (num_distinct_group >= float(required_datasets))
            & (rel_total_group >= float(self.gamma_centering_min_total_reliability))
            & (rel_total_group > 0.0)
        )  # [G, T]

        # Broadcast group-level results back to every masked sample in the group.
        apply_sample = apply_group.index_select(0, group_index)      # [B, T] bool
        center_sample = center_group.index_select(0, group_index)    # [B, T]
        applied = apply_sample & mask_b.bool()
        zero_bt = torch.zeros_like(center_sample)

        log_center = torch.where(applied, center_sample, zero_bt)
        num_distinct = torch.where(
            applied, num_distinct_group.index_select(0, group_index), zero_bt
        )
        total_reliability = torch.where(
            applied, rel_total_group.index_select(0, group_index), zero_bt
        )

        # Per-sample contribution weights (diagnostic; hierarchical dataset mix).
        ds_weights_cell = center_rel_cell / rel_total_group.index_select(
            0, cell_group
        ).clamp_min(eps)
        sample_weights = (
            ds_weights_cell.index_select(0, cell_index)
            * rc
            / rsum_cell.index_select(0, cell_index).clamp_min(eps)
        )
        sample_weights = torch.where(
            candidate & apply_sample, sample_weights, torch.zeros_like(sample_weights)
        )

        # Constraint diagnostic: |sum_s w_s (loggamma_s - strength*center)| / group.
        ce_contrib = sample_weights * (log_safe - strength * center_sample)
        constraint_error_group = _scatter_sum(ce_contrib, group_index, num_groups).abs()
        constraint_error = torch.where(
            applied, constraint_error_group.index_select(0, group_index), zero_bt
        )

        if reference_ids:
            reference_anchor_mask = reference_anchor_mask | (
                is_reference_sample.reshape(-1, 1) & applied
            )

        log_gamma = (log_gamma_raw - strength * log_center) * mask_f
        if self.gamma_reference_hard_anchor and reference_ids:
            log_gamma = torch.where(
                reference_anchor_mask & applied,
                torch.zeros_like(log_gamma),
                log_gamma,
            )

        return {
            "log_gamma": log_gamma,
            "gamma_center": log_center * mask_f,
            "weights": sample_weights * mask_f,
            "applied": applied & mask_b,
            "num_distinct_datasets": num_distinct * mask_f,
            "total_reliability": total_reliability * mask_f,
            "constraint_error": constraint_error * mask_f,
            "reference_anchor_mask": reference_anchor_mask & mask_b,
        }

    def _center_log_gamma_across_sample_ids(
        self,
        log_gamma_raw: torch.Tensor,
        *,
        mask_b: torch.Tensor,
        sample_ids: Sequence[str] | torch.Tensor | None,
        id_datasets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward-compatible wrapper around reliability-weighted centering."""
        reliability = torch.ones_like(log_gamma_raw)
        eligible = mask_b.bool()
        centered = self._center_log_gamma_across_transcripts(
            log_gamma_raw,
            mask_b=mask_b,
            sample_ids=sample_ids,
            id_datasets=id_datasets,
            reliability=reliability,
            eligible=eligible,
        )
        group_size = centered["num_distinct_datasets"].amax(dim=1)
        applied = centered["applied"].any(dim=1).to(dtype=log_gamma_raw.dtype)
        return (
            centered["log_gamma"],
            centered["gamma_center"],
            group_size,
            applied,
        )

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
        target: torch.Tensor | None = None,
        current_epoch: int | None = None,
        sample_ids: Sequence[str] | torch.Tensor | None = None,
        gamma_centering_reliability: torch.Tensor | None = None,
        gamma_centering_eligible: torch.Tensor | None = None,
        replica_profiles: torch.Tensor | None = None,
        replica_mask: torch.Tensor | None = None,
    ):
        del current_epoch  # unused; kept for interface compatibility

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
        target_for_reliability = target.to(device=device, dtype=dtype)
        log_gamma_raw = (
            gamma_log_residual + float(self.gamma_log_init)
        ).clamp(min=self.gamma_log_min, max=self.gamma_log_max) * mask_f
        gamma_raw = torch.exp(log_gamma_raw).clamp_min(self.eps) * mask_f
        (
            gamma_reliability,
            gamma_eligible,
            sample_id_list,
        ) = self._build_gamma_centering_reliability_and_eligibility(
            target=target_for_reliability,
            mask_b=mask_b,
            sample_ids=sample_ids,
            id_datasets=id_datasets,
            gamma_centering_reliability=gamma_centering_reliability,
            gamma_centering_eligible=gamma_centering_eligible,
            replica_profiles=replica_profiles,
            replica_mask=replica_mask,
        )
        centered = self._center_log_gamma_across_transcripts(
            log_gamma_raw,
            mask_b=mask_b,
            sample_ids=sample_ids,
            sample_id_list=sample_id_list,
            id_datasets=id_datasets,
            reliability=gamma_reliability,
            eligible=gamma_eligible,
        )
        log_gamma = centered["log_gamma"]
        gamma_cross_dataset_log_center = centered["gamma_center"]
        gamma_cross_dataset_center_group_size = centered["num_distinct_datasets"].amax(dim=1)
        gamma_cross_dataset_center_applied = centered["applied"].any(dim=1).to(dtype=dtype)
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
            "gamma_center": torch.where(
                mask_b,
                gamma_cross_dataset_log_center,
                torch.zeros_like(gamma_cross_dataset_log_center),
            ),
            "gamma_cross_dataset_log_center": torch.where(
                mask_b,
                gamma_cross_dataset_log_center,
                torch.zeros_like(gamma_cross_dataset_log_center),
            ),
            "gamma_cross_dataset_center_group_size": gamma_cross_dataset_center_group_size,
            "gamma_cross_dataset_center_applied": gamma_cross_dataset_center_applied,
            "gamma_cross_dataset_center_strength": torch.full_like(
                target_mean,
                self.gamma_cross_dataset_centering_strength
                if self.gamma_cross_dataset_centering_enabled
                else 0.0,
            ),
            "gamma_centering_weights": torch.where(
                mask_b,
                centered["weights"],
                torch.zeros_like(centered["weights"]),
            ),
            "gamma_centering_reliability": torch.where(
                mask_b,
                gamma_reliability,
                torch.zeros_like(gamma_reliability),
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
                centered["total_reliability"],
                torch.zeros_like(centered["total_reliability"]),
            ),
            "gamma_centering_constraint_error": torch.where(
                mask_b,
                centered["constraint_error"],
                torch.zeros_like(centered["constraint_error"]),
            ),
            "gamma_reference_anchor_mask": centered["reference_anchor_mask"] & mask_b,
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
            "gamma_split_support_head_enabled": target.new_tensor(
                1.0 if self.gamma_split_support_head else 0.0
            ),
            "gamma_gate_additive_bias_enabled": target.new_tensor(
                1.0 if self.gamma_gate_additive_bias else 0.0
            ),
            "gamma_sparse_transform_enabled": torch.full_like(
                target_mean,
                1.0 if self.gamma_transform == "entmax15_gated_exponential" else 0.0,
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
