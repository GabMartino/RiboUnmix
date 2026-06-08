from __future__ import annotations

import math

import torch
import torch.nn as nn

from Models.RiboQueuingModel.DatasetBiasSubmodel import DatasetBiasSubmodel
from Models.RiboQueuingModel.QueuingBiologicalModel import QueuingBiologicalModel


def _safe_logit(p: float) -> float:
    p = min(max(float(p), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(p / (1.0 - p))


def upstream_queue_propagation(
    rho: torch.Tensor,
    mask: torch.Tensor,
    alpha: float | torch.Tensor,
    q_max: float = 10.0,
) -> torch.Tensor:
    """
    Causal downstream-to-upstream queue propagation.

        q_i = rho_i + alpha * (1 - rho_i) * q_{i+1}

    Codon index increases from start to stop, so downstream bottlenecks
    propagate upstream by scanning right-to-left.
    """
    dtype = rho.dtype
    device = rho.device
    mask_b = mask.bool()
    mask_f = mask_b.to(dtype=dtype)

    if torch.is_tensor(alpha):
        alpha_t = alpha.to(device=device, dtype=dtype)
    else:
        alpha_t = torch.tensor(float(alpha), device=device, dtype=dtype)

    if alpha_t.ndim > 0:
        alpha_t = alpha_t.reshape(rho.shape[0])

    rho = rho.clamp(0.0, 1.0) * mask_f

    q = torch.zeros_like(rho)
    carry = torch.zeros(rho.shape[0], device=device, dtype=dtype)

    for i in range(rho.shape[1] - 1, -1, -1):
        rho_i = rho[:, i]
        q_i = rho_i + alpha_t * (1.0 - rho_i) * carry
        q_i = q_i * mask_f[:, i]
        q[:, i] = q_i
        carry = q_i

    q_max = float(max(q_max, 0.0))
    return q.clamp(0.0, q_max) * mask_f


class RiboQueuingModel(nn.Module):
    def __init__(
        self,
        model_configs: dict,
        eps: float = 1.0e-8,
        mu_max: float = 1.0e8,
    ):
        super().__init__()

        self.eps = float(eps)
        self.mu_max = float(mu_max)

        dataset_bias_params = model_configs["dataset_bias_params"]
        biological_params = model_configs["biological_params"]
        queue_params = {**biological_params, **model_configs.get("queue_propagation_params", {})}

        self.position_features = list(dataset_bias_params["position_features"])
        self.position_scale = float(dataset_bias_params.get("position_scale", 5000.0))
        self.position_edge_tau = float(dataset_bias_params.get("position_edge_tau", 30.0))

        self.biological_model = QueuingBiologicalModel(config_params=biological_params)

        biological_context_size = (
            int(biological_params["num_layers"]) * 2 * int(biological_params["hidden_size"])
        )
        self.dataset_bias_model = DatasetBiasSubmodel(
            config_params=dataset_bias_params,
            biological_context_size=biological_context_size,
        )

        self.use_queue_propagation = bool(queue_params.get("use_queue_propagation", False))
        self.queue_alpha_min = float(queue_params.get("queue_propagation_alpha_min", 0.0))
        self.queue_alpha_max = float(queue_params.get("queue_propagation_alpha_max", 0.95))
        self.queue_q_max = float(queue_params.get("queue_propagation_q_max", 10.0))
        self.queue_alpha_learnable = self.use_queue_propagation and bool(
            queue_params.get("queue_propagation_learnable", False)
        )
        self.queue_alpha_per_transcript = self.queue_alpha_learnable and bool(
            queue_params.get("queue_propagation_per_transcript", True)
        )

        init_alpha = float(queue_params.get("queue_propagation_alpha", 0.5))
        init_alpha = min(max(init_alpha, self.queue_alpha_min), self.queue_alpha_max)
        span = max(self.queue_alpha_max - self.queue_alpha_min, 1.0e-6)
        alpha_unit = (init_alpha - self.queue_alpha_min) / span
        alpha_raw_init = _safe_logit(alpha_unit)

        if self.queue_alpha_learnable and self.queue_alpha_per_transcript:
            self.queue_alpha_head = nn.Linear(biological_context_size, 1)
            nn.init.zeros_(self.queue_alpha_head.weight)
            nn.init.constant_(self.queue_alpha_head.bias, alpha_raw_init)
            self.queue_raw_alpha = None
        elif self.queue_alpha_learnable:
            self.queue_alpha_head = None
            self.queue_raw_alpha = nn.Parameter(
                torch.tensor(alpha_raw_init, dtype=torch.float32)
            )
        else:
            self.queue_alpha_head = None
            self.queue_raw_alpha = None

        self.register_buffer(
            "queue_alpha_fixed",
            torch.tensor(init_alpha, dtype=torch.float32),
            persistent=False,
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
    # Helpers
    # ============================================================

    def _center_log_visibility_bias(
        self,
        log_bias_raw: torch.Tensor,
        p_bio: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_f = mask.bool().to(dtype=log_bias_raw.dtype)
        weights = p_bio.detach().to(dtype=log_bias_raw.dtype) * mask_f
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        center = (log_bias_raw * weights).sum(dim=1, keepdim=True)
        return (log_bias_raw - center) * mask_f, center.reshape(-1)

    def _queue_alpha(
        self,
        *,
        h_n: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.queue_alpha_learnable and self.queue_alpha_per_transcript:
            h_flat = h_n.permute(1, 0, 2).reshape(h_n.shape[1], -1)
            alpha_unit = torch.sigmoid(
                self.queue_alpha_head(h_flat.to(device=device, dtype=dtype))
            )
            alpha = self.queue_alpha_min + (self.queue_alpha_max - self.queue_alpha_min) * alpha_unit
        elif self.queue_alpha_learnable:
            alpha_unit = torch.sigmoid(self.queue_raw_alpha.to(device=device, dtype=dtype))
            alpha = self.queue_alpha_min + (self.queue_alpha_max - self.queue_alpha_min) * alpha_unit
        else:
            alpha = self.queue_alpha_fixed.to(device=device, dtype=dtype)
        return alpha.clamp(self.queue_alpha_min, self.queue_alpha_max)

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        x_packed,
        codon_ids: torch.Tensor,
        id_datasets: torch.Tensor,
        mask: torch.Tensor,
    ):
        # --------------------------------------------------------
        # 1. Biological branch
        # --------------------------------------------------------
        rho, J, h_bio, h_n = self.biological_model(x_packed, mask)

        dtype = rho.dtype
        device = rho.device
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=dtype)

        rho = rho.clamp(0.0, 1.0) * mask_f
        h_bio = h_bio.clamp_min(0.0) * mask_f
        alpha = self._queue_alpha(h_n=h_n, device=device, dtype=dtype)

        if self.use_queue_propagation:
            q = upstream_queue_propagation(rho=rho, mask=mask_b, alpha=alpha, q_max=self.queue_q_max)
        else:
            q = rho.clamp(0.0, self.queue_q_max) * mask_f

        # --------------------------------------------------------
        # 2. Observation heads
        # --------------------------------------------------------
        position_features = self.make_position_features(mask=mask_b, dtype=dtype)

        bias = self.dataset_bias_model(
            dataset_ids=id_datasets,
            mask=mask_b,
            codon_ids=codon_ids,
            position_features=position_features,
            biological_context=h_n,
        )

        # --------------------------------------------------------
        # 3. Visibility correction
        # --------------------------------------------------------
        log_visibility_raw = bias["log_visibility_bias_raw"].to(dtype=dtype) * mask_f
        log_visibility_bias, log_visibility_center = self._center_log_visibility_bias(
            log_bias_raw=log_visibility_raw, p_bio=q, mask=mask_b,
        )
        beta = torch.exp(log_visibility_bias).clamp_min(self.eps) * mask_f
        p_visible = q * beta * mask_f

        # --------------------------------------------------------
        # 4. Dataset-transcript scale
        # --------------------------------------------------------
        scale_dt = bias["scale_dt"].to(dtype=dtype)
        if scale_dt.ndim == 1:
            scale_dt = scale_dt.reshape(-1, 1)

        mu_bio = (scale_dt * q).clamp(self.eps, self.mu_max) * mask_f
        lambda_pre_dropout = (scale_dt * q * beta).clamp(self.eps, self.mu_max) * mask_f

        # --------------------------------------------------------
        # 5. Likelihood mu parameter (returned as main model output)
        # --------------------------------------------------------
        mu = torch.nan_to_num(lambda_pre_dropout, nan=self.eps, posinf=self.mu_max, neginf=self.eps)
        mu = mu.clamp(self.eps, self.mu_max)
        mu = torch.where(mask_b, mu, torch.ones_like(mu))

        # --------------------------------------------------------
        # 6. Dispersion
        # --------------------------------------------------------
        phi = bias["phi"].to(dtype=dtype)
        phi = torch.where(mask_b, phi, torch.ones_like(phi))

        # --------------------------------------------------------
        # 7. Diagnostics
        # --------------------------------------------------------
        valid_len = mask_f.sum(dim=1).clamp_min(1.0)

        rho_mean = (rho * mask_f).sum(dim=1) / valid_len
        q_mean = (q * mask_f).sum(dim=1) / valid_len
        q_max_val = q.masked_fill(~mask_b, 0.0).max(dim=1).values
        q_zero_frac = ((q <= 0.0) & mask_b).float().sum(dim=1) / valid_len
        p_visible_mean = (p_visible * mask_f).sum(dim=1) / valid_len
        beta_mean = (beta * mask_f).sum(dim=1) / valid_len
        lambda_mass = (lambda_pre_dropout * mask_f).sum(dim=1)
        visibility_log_abs_mean = (log_visibility_bias.abs() * mask_f).sum(dim=1) / valid_len
        vis_weights = q.detach() * mask_f
        vis_weights = vis_weights / vis_weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        visibility_log_q_weighted_mean = (log_visibility_bias * vis_weights).sum(dim=1)
        visibility_log_center_abs = log_visibility_center.abs()
        rho_zero_frac = ((rho <= 0.0) & mask_b).float().sum(dim=1) / valid_len
        beta_zero_frac = ((beta <= 0.0) & mask_b).float().sum(dim=1) / valid_len
        mu_zero_frac = ((lambda_pre_dropout <= 0.0) & mask_b).float().sum(dim=1) / valid_len

        if alpha.ndim == 0:
            queue_alpha = alpha.reshape(()).expand(rho.shape[0])
        else:
            queue_alpha = alpha.reshape(rho.shape[0])
        queue_propagation_enabled = torch.full(
            (rho.shape[0],), float(self.use_queue_propagation), device=device, dtype=dtype,
        )
        queue_alpha_trainable = torch.full(
            (rho.shape[0],), float(self.queue_alpha_learnable), device=device, dtype=dtype,
        )
        queue_alpha_per_transcript = torch.full(
            (rho.shape[0],), float(self.queue_alpha_per_transcript), device=device, dtype=dtype,
        )

        extras = {
            # Biological
            "rho_bio": rho,
            "q_bio": q,
            "h_bio": h_bio,
            "J": J,
            "queue_alpha": queue_alpha,
            "queue_propagation_enabled": queue_propagation_enabled,
            "queue_alpha_trainable": queue_alpha_trainable,
            "queue_alpha_per_transcript": queue_alpha_per_transcript,

            # Observation
            "log_visibility_bias": torch.where(mask_b, log_visibility_bias, torch.zeros_like(log_visibility_bias)),
            "obs_beta": torch.where(mask_b, beta, torch.ones_like(beta)),
            "p_visible": p_visible,
            "mu_bio": torch.where(mask_b, mu_bio, torch.ones_like(mu_bio)),
            "lambda_pre_dropout": torch.where(mask_b, lambda_pre_dropout, torch.ones_like(lambda_pre_dropout)),

            # Scale
            "scale_dt": scale_dt.reshape(-1),
            "log_scale_dt": bias["log_scale_dt"],
            "dataset_scale": bias["dataset_scale"],
            "transcript_scale": bias["transcript_scale"],
            "transcript_log_scale": bias["transcript_log_scale"],  # [B, 1]

            # Dispersion (also returned as second value of forward)
            "phi": phi,
            "log_phi": bias["log_phi"],  # [B, 1]

            # Scalar diagnostics (all in DATASET_DIAGNOSTIC_KEYS)
            "rho_mean": rho_mean,
            "q_mean": q_mean,
            "q_max": q_max_val,
            "q_zero_frac": q_zero_frac,
            "p_visible_mean": p_visible_mean,
            "beta_mean": beta_mean,
            "mu_mass": lambda_mass,
            "lambda_mass": lambda_mass,
            "visibility_log_abs_mean": visibility_log_abs_mean,
            "visibility_log_q_weighted_mean": visibility_log_q_weighted_mean,
            "visibility_log_center_abs": visibility_log_center_abs,
            "rho_zero_frac": rho_zero_frac,
            "beta_zero_frac": beta_zero_frac,
            "mu_zero_frac": mu_zero_frac,
        }

        return mu, phi, extras
