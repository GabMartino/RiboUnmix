from __future__ import annotations

import torch
import torch.nn as nn

from Models.RiboQueuingModel.DatasetBiasSubmodel import DatasetBiasSubmodel
from Models.RiboQueuingModel.QueuingBiologicalModel import QueuingBiologicalModel
from Models.RiboQueuingModel.submodels.DatasetDispersionHead import DatasetDispersionHead
from Models.RiboQueuingModel.submodels.DatasetPositionTweediePowerHead import DatasetPositionTweediePowerHead
from Models.utils.compute_S_mean import compute_S_mean, compute_S_trimmed_mean


class RiboQueuingModel(nn.Module):
    def __init__(
            self,
            model_configs: dict,
            eps: float = 1e-8,
            mu_max: float = 1e8,
    ):
        super().__init__()

        self.eps = float(eps)
        self.mu_max = float(mu_max)

        self.position_edge_tau = float(
            model_configs["dataset_bias_params"].get("position_edge_tau", 30.0)
        )
        self.position_features = list(model_configs["dataset_bias_params"]["position_features"])
        self.position_scale = float(model_configs["dataset_bias_params"]["position_scale"])

        # Biological Model setup
        self.biological_model = QueuingBiologicalModel(config_params=model_configs["biological_params"])

        # Dataset Bias Submodel setup
        self.dataset_bias_model = DatasetBiasSubmodel(config_params=model_configs["dataset_bias_params"])
        self.gate_additive = bool(
            model_configs["dataset_bias_params"].get("gate_additive", False)
        )

    def make_position_features(
            self,
            mask: torch.Tensor,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        mask_b = mask.bool()
        B, T = mask_b.shape
        device = mask_b.device

        pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(0).expand(B, T)

        lengths = mask_b.sum(dim=1, keepdim=True).to(dtype=dtype).clamp_min(1.0)
        last_pos = (lengths - 1.0).clamp_min(1.0)

        rel_pos = pos / last_pos

        # Absolute position, compressed. This avoids huge linear extrapolation.
        abs_pos = pos / float(self.position_scale)
        abs_pos_log = torch.log1p(pos) / torch.log1p(
            torch.tensor(float(self.position_scale), device=device, dtype=dtype)
        )

        # Local edge features. These are high near start/stop and decay away.
        # Tau is in codons.
        tau = float(getattr(self, "position_edge_tau", 30.0))

        dist_start_codons = pos
        dist_stop_codons = (lengths - 1.0 - pos).clamp_min(0.0)

        start_window = torch.exp(-dist_start_codons / tau)
        stop_window = torch.exp(-dist_stop_codons / tau)

        feature_map = {
            "rel_pos": rel_pos,
            "abs_pos": abs_pos,
            "abs_pos_log": abs_pos_log,
            "start_window": start_window,
            "stop_window": stop_window,
            "dist_to_start": rel_pos,
            "dist_to_stop": 1.0 - rel_pos,
        }

        features = [feature_map[name] for name in self.position_features]

        x_pos = torch.stack(features, dim=-1)
        x_pos = x_pos * mask_b.unsqueeze(-1).to(dtype=dtype)

        return x_pos

    def _apply_L_mass_preserving_b(
            self,
            L_queue: torch.Tensor,
            b: torch.Tensor,
            mask: torch.Tensor,
            eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_f = mask.bool().to(dtype=L_queue.dtype)

        L = L_queue.clamp_min(0.0) * mask_f
        b = b.clamp_min(eps)

        L_mass = L.detach().sum(dim=1, keepdim=True).clamp_min(eps)
        Lb_mass = (L.detach() * b * mask_f).sum(dim=1, keepdim=True).clamp_min(eps)

        b_shape = b * (L_mass / Lb_mass)
        b_shape = torch.where(mask.bool(), b_shape, torch.ones_like(b_shape))

        bio_q = L * b_shape
        bio_q = bio_q * mask_f

        return bio_q, b_shape

    def forward(
            self,
            x_packed,
            codon_ids: torch.Tensor,
            id_datasets: torch.Tensor,
            y_raw_target: torch.Tensor,
    ):
        # ============================================================
        # 1. BIOLOGY
        # ============================================================
        L_queue, rho_diag, w_prob, J, mask = self.biological_model(x_packed)

        B, T = L_queue.shape

        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=L_queue.dtype)

        position_features = self.make_position_features(
            mask=mask_b,
            dtype=L_queue.dtype,
        )

        # ============================================================
        # 2. DATASET / OBSERVATION HEADS
        # ============================================================
        (
            b,  # b_total from head = b_smooth * keep_gate, before mass preservation
            log_b,  # log of b_smooth, not log of gated b
            beta_centered,
            b_smooth,
            keep_gate,
            keep_prob,
            keep_hard,
            gate_logits,

            additive_rel,  # relative additive support = lambda_frac * R_shape
            R_shape,
            lambda_bg,  # semantically: lambda_frac
            r_logits,
            lambda_raw,

            phi_raw,
        ) = self.dataset_bias_model(
            dataset_ids=id_datasets,
            codon_ids=codon_ids,
            mask=mask_b,
            position_features=position_features,
        )

        # Dtype/device consistency.
        b = b.to(dtype=L_queue.dtype)
        b_smooth = b_smooth.to(dtype=L_queue.dtype)
        log_b = log_b.to(dtype=L_queue.dtype)
        beta_centered = beta_centered.to(dtype=L_queue.dtype)

        keep_gate = keep_gate.to(dtype=L_queue.dtype) * mask_f
        keep_prob = keep_prob.to(dtype=L_queue.dtype) * mask_f
        keep_hard = keep_hard.to(dtype=L_queue.dtype) * mask_f
        gate_logits = gate_logits.to(dtype=L_queue.dtype) * mask_f

        additive_rel = additive_rel.to(dtype=L_queue.dtype) * mask_f
        R_shape = R_shape.to(dtype=L_queue.dtype) * mask_f
        lambda_bg = lambda_bg.to(dtype=L_queue.dtype)
        lambda_raw = lambda_raw.to(dtype=L_queue.dtype)

        # ============================================================
        # 3. MASS-PRESERVED SMOOTH MULTIPLICATIVE BIOLOGY
        # ============================================================
        # Apply mass preservation only to b_smooth.
        # Do NOT mass-preserve the keep gate.
        bio_q_smooth, b_shape = self._apply_L_mass_preserving_b(
            L_queue=L_queue,
            b=b_smooth,
            mask=mask_b,
            eps=self.eps,
        )

        bio_q_smooth = bio_q_smooth * mask_f
        b_shape = torch.where(mask_b, b_shape, torch.ones_like(b_shape))

        # ============================================================
        # 4. RELATIVE ADDITIVE SUPPORT -> ABSOLUTE ADDITIVE SUPPORT
        # ============================================================
        # additive_rel is dimensionless:
        #   additive_rel_i = lambda_frac * R_i
        #
        # Convert it into support using the detached biological scale:
        #   additive_noise_raw_i = additive_rel_i * mean_valid(bio_q_smooth)
        #
        # This makes lambda_frac interpretable as a relative additive fraction.
        valid_lengths = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        bio_mean_for_additive = (
                                        bio_q_smooth.detach().clamp_min(0.0) * mask_f
                                ).sum(dim=1, keepdim=True) / valid_lengths

        bio_mean_for_additive = bio_mean_for_additive.clamp_min(self.eps)

        additive_rel = additive_rel.clamp_min(0.0) * mask_f

        additive_noise_raw = additive_rel * bio_mean_for_additive
        additive_noise_raw = additive_noise_raw * mask_f

        # ============================================================
        # 5. APPLY KEEP GATE AFTER BIOLOGY + ADDITIVE SUPPORT
        # ============================================================
        # Pre-gate support:
        #   q_pre_gate_i = L_i * b_shape_i + additive_noise_raw_i
        q_pre_gate = bio_q_smooth + additive_noise_raw
        q_pre_gate = q_pre_gate * mask_f

        # Effective biological support after gate:
        #   bio_q_i = keep_gate_i * L_i * b_shape_i
        bio_q = keep_gate * bio_q_smooth
        bio_q = bio_q * mask_f

        if self.gate_additive:
            # Strict observation-zero interpretation:
            # q_i = gate_i * (bio_i + additive_i)
            additive_noise_eff = keep_gate * additive_noise_raw
        else:
            # Additive-rescue interpretation:
            # q_i = gate_i * bio_i + additive_i
            additive_noise_eff = additive_noise_raw

        additive_noise_eff = additive_noise_eff * mask_f

        q = bio_q + additive_noise_eff
        q = q * mask_f

        # ============================================================
        # 6. ROBUST PROFILE NORMALIZATION
        # ============================================================
        # If the gate closes every valid position, fallback to pre-gate support.
        # If pre-gate support is also zero, fallback to uniform over valid positions.
        q_mass_raw = q.sum(dim=1, keepdim=True)

        use_q = q_mass_raw > self.eps

        q_fallback = q_pre_gate
        q_fallback_mass = q_fallback.sum(dim=1, keepdim=True)

        use_pre_gate = (~use_q) & (q_fallback_mass > self.eps)

        q_uniform = mask_f

        q_for_profile = torch.where(
            use_q,
            q,
            torch.where(
                use_pre_gate,
                q_fallback,
                q_uniform,
            ),
        )

        q_for_profile = q_for_profile * mask_f
        q_mass = q_for_profile.sum(dim=1, keepdim=True).clamp_min(self.eps)

        profile_prob = q_for_profile / q_mass
        profile_prob = profile_prob * mask_f

        # ============================================================
        # 7. TOTAL-MASS-CONDITIONED MEAN
        # ============================================================
        total_mass = (
                y_raw_target.float().clamp_min(0.0) * mask_f.float()
        ).sum(dim=1, keepdim=True).detach().clamp_min(self.eps)

        mu = total_mass * profile_prob
        mu = mu * mask_f

        # ============================================================
        # 8. CLEANUP
        # ============================================================
        mu = torch.nan_to_num(
            mu,
            nan=self.eps,
            posinf=self.mu_max,
            neginf=self.eps,
        )

        mu = mu.clamp(min=self.eps, max=self.mu_max)
        mu = torch.where(mask_b, mu, torch.ones_like(mu))

        phi = torch.where(
            mask_b,
            phi_raw,
            torch.ones_like(phi_raw),
        )

        # ============================================================
        # 9. DIAGNOSTIC MEANS
        # ============================================================
        b_effective = b_shape * keep_gate
        b_effective = torch.where(mask_b, b_effective, torch.ones_like(b_effective))

        bio_q_smooth_mass = bio_q_smooth.sum(dim=1, keepdim=True).clamp_min(self.eps)
        bio_q_mass = bio_q.sum(dim=1, keepdim=True).clamp_min(self.eps)

        mu_bio_smooth = total_mass * (bio_q_smooth / bio_q_smooth_mass)
        mu_bio_smooth = mu_bio_smooth * mask_f

        mu_bio_only = total_mass * (bio_q / bio_q_mass)
        mu_bio_only = mu_bio_only * mask_f

        q_used_pre_gate_fallback = use_pre_gate.reshape(B).float()
        q_used_uniform_fallback = ((~use_q) & (~use_pre_gate)).reshape(B).float()

        # Useful mass diagnostics.
        bio_mass = (bio_q_smooth * mask_f).sum(dim=1, keepdim=True).clamp_min(self.eps)
        additive_mass_raw = (additive_noise_raw * mask_f).sum(dim=1, keepdim=True)
        additive_mass_eff = (additive_noise_eff * mask_f).sum(dim=1, keepdim=True)

        additive_frac_raw = additive_mass_raw / (bio_mass + additive_mass_raw).clamp_min(self.eps)
        additive_frac_eff = additive_mass_eff / (
                (bio_q * mask_f).sum(dim=1, keepdim=True) + additive_mass_eff
        ).clamp_min(self.eps)

        # ============================================================
        # 10. EXTRAS
        # ============================================================
        extras = {
            # ------------------------------------------------------------
            # Biology
            # ------------------------------------------------------------
            "L_queue_raw": L_queue,
            "L_queue": L_queue,
            "L_queue_shape": L_queue,

            "L_effective": L_queue,
            "L_shape": L_queue,

            "rho": rho_diag,
            "w_prob": w_prob,
            "J": J,

            # ------------------------------------------------------------
            # Observation support decomposition
            # ------------------------------------------------------------
            "bio_q_smooth": bio_q_smooth,
            "bio_q": bio_q,

            # Relative additive branch from the head.
            "additive_rel": additive_rel,

            # Absolute additive support before and after gate.
            "additive_noise_raw": additive_noise_raw,
            "additive_noise": additive_noise_eff,

            "bio_mean_for_additive": bio_mean_for_additive.reshape(B),

            "R_shape": R_shape,

            # Backward-compatible name.
            "lambda_bg": lambda_bg,

            # Correct semantic name.
            "lambda_frac": lambda_bg,

            "r_logits": r_logits,
            "lambda_raw": lambda_raw,

            "q_pre_gate": q_pre_gate,
            "q": q,
            "q_for_profile": q_for_profile,
            "q_mass": q_mass.reshape(B),
            "q_mass_raw": q_mass_raw.reshape(B),
            "profile_prob": profile_prob,

            "q_used_pre_gate_fallback": q_used_pre_gate_fallback,
            "q_used_uniform_fallback": q_used_uniform_fallback,

            "total_mass": total_mass.reshape(B),

            "bio_mass": bio_mass.reshape(B),
            "additive_mass_raw": additive_mass_raw.reshape(B),
            "additive_mass_eff": additive_mass_eff.reshape(B),
            "additive_frac_raw": additive_frac_raw.reshape(B),
            "additive_frac_eff": additive_frac_eff.reshape(B),

            # ------------------------------------------------------------
            # Means
            # ------------------------------------------------------------
            "mu_obs": mu,
            "mu_base": mu_bio_only,
            "mu_bio_only": mu_bio_only,
            "mu_bio_smooth": mu_bio_smooth,

            # ------------------------------------------------------------
            # Multiplicative correction and gate
            # ------------------------------------------------------------
            "exp_b": b_effective,
            "exp_b_raw": b,

            "b_total": b,
            "b_effective": b_effective,
            "b_shape": b_shape,
            "b_smooth": b_smooth,

            "log_b": log_b,
            "control": log_b,
            "b_control": log_b,
            "beta_centered": beta_centered,

            "keep_gate": keep_gate,
            "keep_prob": keep_prob,
            "keep_hard": keep_hard,
            "gate_logits": gate_logits,

            # ------------------------------------------------------------
            # Disabled old heads
            # ------------------------------------------------------------
            "beta_per_position_raw": None,
            "beta_per_position": None,
            "beta_raw": None,

            "base_shape": None,
            "corrected_shape": None,

            # ------------------------------------------------------------
            # Observation uncertainty / profile-dispersion head
            # ------------------------------------------------------------
            "phi": phi,
            "phi_raw": phi_raw,
            "kappa_input": phi,

            # ------------------------------------------------------------
            # Disabled old shift/context diagnostics
            # ------------------------------------------------------------
            "local_context": None,
            "shift_weights_used": None,
            "shift_weights_soft": None,
        }

        return mu, phi, extras
