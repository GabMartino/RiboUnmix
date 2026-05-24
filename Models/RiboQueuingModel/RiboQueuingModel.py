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
        mask_f = mask_b.float()

        position_features = self.make_position_features(
            mask=mask_b,
            dtype=L_queue.dtype,
        )

        (
            b,
            control,
            beta_centered,
            additive_noise,
            R_shape,
            lambda_bg,
            r_logits,
            lambda_raw,
            phi_raw,
            p, p_extras
        ) = self.dataset_bias_model(
            dataset_ids=id_datasets,
            codon_ids=codon_ids,
            mask=mask_b,
            position_features=position_features,
        )

        bio_q = L_queue.clamp_min(0.0) * b
        bio_q = bio_q * mask_f

        additive_noise = additive_noise.clamp_min(0.0) * mask_f

        q = bio_q + additive_noise
        q = q * mask_f

        q_mass = q.sum(dim=1, keepdim=True).clamp_min(self.eps)

        # ============================================================
        # TOTAL-MASS-CONDITIONED MEAN
        # ============================================================
        total_mass = (
                y_raw_target.float().clamp_min(0.0) * mask_f
        ).sum(dim=1, keepdim=True).detach().clamp_min(self.eps)

        profile_prob = q / q_mass
        profile_prob = profile_prob * mask_f

        mu = total_mass * profile_prob
        mu = mu * mask_f

        # ============================================================
        # CLEANUP
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
        # 6. EXTRAS
        # ============================================================
        # ============================================================
        # 6. EXTRAS
        # ============================================================
        p_extras = p_extras or {}

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
            # bio_q is the queue-dependent part:
            #   bio_q_i = L_queue_i * b_i
            "bio_q": bio_q,

            # additive_noise is the background/residual part:
            #   additive_noise_i = lambda_bg * R_i
            "additive_noise": additive_noise,
            "R_shape": R_shape,
            "lambda_bg": lambda_bg,
            "r_logits": r_logits,
            "lambda_raw": lambda_raw,

            # total support:
            #   q_i = bio_q_i + additive_noise_i
            "q": q,
            "q_mass": q_mass.reshape(B),
            "profile_prob": profile_prob,

            # total target-conditioned mass:
            "total_mass": total_mass.reshape(B),

            # ------------------------------------------------------------
            # Means
            # ------------------------------------------------------------
            "mu_obs": mu,
            "mu_base": total_mass * (bio_q / bio_q.sum(dim=1, keepdim=True).clamp_min(self.eps)),
            "mu_bio_only": total_mass * (bio_q / bio_q.sum(dim=1, keepdim=True).clamp_min(self.eps)),

            # ------------------------------------------------------------
            # Multiplicative dataset/protocol correction
            # ------------------------------------------------------------
            # Use exp_b for the multiplier so the Lightning logger can find it.
            "exp_b": b,
            "exp_b_raw": b,
            "b": control,  # backward-compatible: b means log-bias in old plots
            "control": control,
            "beta_centered": beta_centered,

            # ------------------------------------------------------------
            # Disabled old heads
            # ------------------------------------------------------------
            "beta_per_position_raw": None,
            "beta_per_position": None,
            "beta_raw": None,

            "base_shape": None,
            "corrected_shape": None,

            # ------------------------------------------------------------
            # Observation distribution heads
            # ------------------------------------------------------------
            "phi": phi,
            "phi_raw": phi_raw,

            # ------------------------------------------------------------
            # Tweedie p diagnostics
            # ------------------------------------------------------------
            "tweedie_p_position": p,
            "tweedie_p": p,
            "tweedie_p_raw": p_extras.get("tweedie_p_raw"),
            "tweedie_p_dataset_raw": p_extras.get("tweedie_p_dataset_raw"),
            "tweedie_p_local_delta": p_extras.get("tweedie_p_local_delta"),

            # ------------------------------------------------------------
            # Disabled old shift/context diagnostics
            # ------------------------------------------------------------
            "local_context": None,
            "shift_weights_used": None,
            "shift_weights_soft": None,
        }

        return mu, p, phi, extras
