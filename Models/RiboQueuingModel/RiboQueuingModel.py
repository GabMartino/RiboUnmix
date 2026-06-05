from __future__ import annotations

import math
import warnings

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
    if rho.ndim != 2:
        raise ValueError(f"Expected rho [B, T], got {tuple(rho.shape)}.")

    if mask.ndim != 2:
        raise ValueError(f"Expected mask [B, T], got {tuple(mask.shape)}.")

    if mask.shape != rho.shape:
        raise ValueError(
            f"mask/rho shape mismatch: mask={tuple(mask.shape)}, rho={tuple(rho.shape)}."
        )

    B, T = rho.shape
    dtype = rho.dtype
    device = rho.device

    mask_b = mask.bool()
    mask_f = mask_b.to(dtype=dtype)

    if torch.is_tensor(alpha):
        alpha_t = alpha.to(device=device, dtype=dtype)
    else:
        alpha_t = torch.tensor(float(alpha), device=device, dtype=dtype)

    if alpha_t.numel() != 1:
        raise ValueError(f"Expected scalar alpha, got shape {tuple(alpha_t.shape)}.")

    alpha_t = alpha_t.reshape(())
    q_max = float(max(q_max, 0.0))

    nan_rho = int(torch.isnan(rho).sum().item())
    if nan_rho > 0:
        warnings.warn(
            f"upstream_queue_propagation: {nan_rho} NaN(s) in rho input — clamping to 0.",
            RuntimeWarning,
            stacklevel=2,
        )
    rho = torch.nan_to_num(
        rho.to(dtype=dtype),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(min=0.0, max=1.0)
    rho = rho * mask_f

    q = torch.zeros_like(rho)
    carry = torch.zeros((B,), device=device, dtype=dtype)

    for i in range(T - 1, -1, -1):
        rho_i = rho[:, i]
        q_i = rho_i + alpha_t * (1.0 - rho_i) * carry
        q_i = torch.where(mask_b[:, i], q_i, torch.zeros_like(q_i))
        q[:, i] = q_i
        carry = q_i

    nan_q = int(torch.isnan(q).sum().item())
    if nan_q > 0:
        warnings.warn(
            f"upstream_queue_propagation: {nan_q} NaN(s) in q after propagation — clamping to 0.",
            RuntimeWarning,
            stacklevel=2,
        )
    q = torch.nan_to_num(
        q,
        nan=0.0,
        posinf=q_max,
        neginf=0.0,
    )

    if q_max > 0.0:
        q = q.clamp(min=0.0, max=q_max)
    else:
        q = q.clamp_min(0.0)

    q = q * mask_f

    if q.shape != rho.shape:
        raise RuntimeError(f"Expected q [B, T], got {tuple(q.shape)}.")

    return q


class RiboQueuingModel(nn.Module):
    """
    Anchor-free queueing biology plus dataset observation model.

    Shared biological queueing support:

        rho_ti = 1 - exp(-h_ti)
        q_ti = rho_ti + alpha * (1 - rho_ti) * q_t,i+1

    Dataset observation model:

        log_b_dti = centered dataset visibility correction
        beta_dti = exp(log_b_dti)
        lambda_dti = scale_dt * q_ti * beta_dti
        mu_unconditional_dti = (1 - dropout_prob_dti) * lambda_dti

    Dropout/zero inflation is dataset-specific and separate from biological
    support. This prevents dataset zeros from becoming biological zeros.
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

        # When False, the dataset bias heads are bypassed in forward() and all
        # bias terms are forced to identity (beta=1, scale=1, zero_prob~0, phi=1).
        self.use_dataset_bias = bool(model_configs.get("use_dataset_bias", True))

        dataset_bias_params = model_configs["dataset_bias_params"]
        biological_params = model_configs["biological_params"]
        queue_params = dict(biological_params)
        queue_params.update(dict(model_configs.get("queue_propagation_params", {})))

        self.position_features = list(dataset_bias_params["position_features"])
        self.position_scale = float(dataset_bias_params.get("position_scale", 5000.0))
        self.position_edge_tau = float(dataset_bias_params.get("position_edge_tau", 30.0))

        self.biological_model = QueuingBiologicalModel(
            config_params=biological_params,
        )

        biological_context_size = (
            int(biological_params["num_layers"])
            * 2
            * int(biological_params["hidden_size"])
        )

        self.dataset_bias_model = DatasetBiasSubmodel(
            config_params=dataset_bias_params,
            biological_context_size=biological_context_size,
        )

        self.use_queue_propagation = bool(
            queue_params.get("use_queue_propagation", False)
        )
        self.queue_alpha_min = float(
            queue_params.get("queue_propagation_alpha_min", 0.0)
        )
        self.queue_alpha_max = float(
            queue_params.get("queue_propagation_alpha_max", 0.95)
        )
        self.queue_q_max = float(queue_params.get("queue_propagation_q_max", 10.0))
        self.queue_alpha_learnable = self.use_queue_propagation and bool(
            queue_params.get("queue_propagation_learnable", False)
        )

        if self.queue_alpha_min > self.queue_alpha_max:
            raise ValueError(
                "queue_propagation_alpha_min must be <= queue_propagation_alpha_max, "
                f"got {self.queue_alpha_min} > {self.queue_alpha_max}."
            )

        init_alpha = float(queue_params.get("queue_propagation_alpha", 0.5))
        init_alpha = min(max(init_alpha, self.queue_alpha_min), self.queue_alpha_max)

        if self.queue_alpha_learnable:
            span = max(self.queue_alpha_max - self.queue_alpha_min, 1.0e-6)
            alpha_unit = (init_alpha - self.queue_alpha_min) / span
            self.queue_raw_alpha = nn.Parameter(
                torch.tensor(_safe_logit(alpha_unit), dtype=torch.float32)
            )
            self.register_buffer(
                "queue_alpha_fixed",
                torch.tensor(init_alpha, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.queue_raw_alpha = None
            self.register_buffer(
                "queue_alpha_fixed",
                torch.tensor(init_alpha, dtype=torch.float32),
                persistent=False,
            )

    # ============================================================
    # Position features
    # ============================================================

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
        abs_pos = pos / float(self.position_scale)

        abs_pos_log = torch.log1p(pos) / torch.log1p(
            torch.tensor(
                float(self.position_scale),
                device=device,
                dtype=dtype,
            )
        )

        tau = max(float(self.position_edge_tau), 1.0e-6)

        dist_start = pos
        dist_stop = (lengths - 1.0 - pos).clamp_min(0.0)

        start_window = torch.exp(-dist_start / tau)
        stop_window = torch.exp(-dist_stop / tau)

        feature_map = {
            "rel_pos": rel_pos,
            "abs_pos": abs_pos,
            "abs_pos_log": abs_pos_log,
            "start_window": start_window,
            "stop_window": stop_window,
            "dist_to_start": rel_pos,
            "dist_to_stop": 1.0 - rel_pos,
        }

        missing = [name for name in self.position_features if name not in feature_map]
        if missing:
            raise KeyError(f"Unknown position feature(s): {missing}")

        x_pos = torch.stack(
            [feature_map[name] for name in self.position_features],
            dim=-1,
        )

        return x_pos * mask_b.unsqueeze(-1).to(dtype=dtype)

    # ============================================================
    # Normalization helpers
    # ============================================================

    def _masked_normalize(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        fallback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=x.dtype)

        x = x.clamp_min(0.0) * mask_f
        mass = x.sum(dim=1, keepdim=True)

        uniform = mask_f / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        if fallback is not None:
            fallback = fallback.to(device=x.device, dtype=x.dtype).clamp_min(0.0) * mask_f
            fallback_mass = fallback.sum(dim=1, keepdim=True)
            fallback = torch.where(
                fallback_mass > self.eps,
                fallback / fallback_mass.clamp_min(self.eps),
                uniform,
            )
        else:
            fallback = uniform

        out = torch.where(mass > self.eps, x / mass.clamp_min(self.eps), fallback)
        return out * mask_f

    def _center_log_visibility_bias(
        self,
        log_bias_raw: torch.Tensor,
        p_bio: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_f = mask.bool().to(dtype=log_bias_raw.dtype)

        # Gauge: no transcript-wide scale is allowed to hide inside the
        # positional visibility correction. The scale head owns that degree of
        # freedom. Detaching p_bio keeps this constraint from pushing biology.
        weights = p_bio.detach().to(dtype=log_bias_raw.dtype) * mask_f
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)

        center = (log_bias_raw * weights).sum(dim=1, keepdim=True)
        log_bias = (log_bias_raw - center) * mask_f
        return log_bias, center.reshape(-1)

    def _queue_alpha(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.queue_alpha_learnable:
            if self.queue_raw_alpha is None:
                raise RuntimeError("queue_alpha_learnable=True but queue_raw_alpha is None.")

            alpha_unit = torch.sigmoid(self.queue_raw_alpha.to(device=device, dtype=dtype))
            alpha = self.queue_alpha_min + (
                self.queue_alpha_max - self.queue_alpha_min
            ) * alpha_unit
        else:
            alpha = self.queue_alpha_fixed.to(device=device, dtype=dtype)

        return alpha.clamp(min=self.queue_alpha_min, max=self.queue_alpha_max)

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
        # ------------------------------------------------------------
        # 1. Shared biological queueing branch
        # ------------------------------------------------------------
        rho, J, h_bio, h_n = self.biological_model(x_packed, mask)

        if rho.ndim != 2:
            raise ValueError(f"Expected rho [B, T], got {tuple(rho.shape)}.")

        mask_b = mask.bool()

        if mask_b.shape != rho.shape:
            raise ValueError(
                f"mask shape {tuple(mask_b.shape)} does not match rho shape {tuple(rho.shape)}."
            )

        if codon_ids.shape[:2] != rho.shape:
            raise ValueError(
                f"codon_ids shape {tuple(codon_ids.shape)} does not match rho shape {tuple(rho.shape)}."
            )

        dtype = rho.dtype
        device = rho.device
        mask_f = mask_b.to(dtype=dtype)

        rho = rho.to(dtype=dtype).clamp(min=0.0, max=1.0) * mask_f
        h_bio = h_bio.to(device=device, dtype=dtype).clamp_min(0.0) * mask_f

        alpha = self._queue_alpha(device=device, dtype=dtype)

        if self.use_queue_propagation:
            q = upstream_queue_propagation(
                rho=rho,
                mask=mask_b,
                alpha=alpha,
                q_max=self.queue_q_max,
            )
        else:
            q = torch.nan_to_num(
                rho,
                nan=0.0,
                posinf=self.queue_q_max,
                neginf=0.0,
            )
            q = q.clamp(min=0.0, max=self.queue_q_max) * mask_f

        if q.shape != rho.shape:
            raise RuntimeError(f"Expected q [B, T], got {tuple(q.shape)}.")

        # Compatibility plotting keys. These are no longer probability
        # distributions; they are local occupancy/support signals.
        p_bio = rho

        # ------------------------------------------------------------
        # 2. Dataset/protocol observation heads
        # ------------------------------------------------------------
        position_features = self.make_position_features(
            mask=mask_b,
            dtype=dtype,
        )

        if self.use_dataset_bias:
            bias = self.dataset_bias_model(
                dataset_ids=id_datasets,
                mask=mask_b,
                codon_ids=codon_ids,
                position_features=position_features,
                biological_context=h_n,
            )
        else:
            # Identity bias: beta=1 (log_visibility=0), scale_dt=1, zero_prob~0,
            # phi=1. scale_dt / zero_prob / phi fall through to the identity
            # fallbacks below; the *_scale keys are provided so the diagnostic
            # pass-through in extras does not see None.
            zeros_bt = torch.zeros_like(rho)
            ones_b1 = torch.ones((rho.shape[0], 1), device=device, dtype=dtype)
            zeros_b1 = torch.zeros((rho.shape[0], 1), device=device, dtype=dtype)
            bias = {
                "log_visibility_bias_raw": zeros_bt,
                "log_scale_dt": zeros_b1,
                "global_log_scale": zeros_b1,
                "dataset_log_scale": zeros_b1,
                "transcript_log_scale": zeros_b1,
                "dataset_scale": ones_b1,
                "transcript_scale": ones_b1,
            }

        log_visibility_raw = bias.get("log_visibility_bias_raw")

        if not torch.is_tensor(log_visibility_raw):
            beta_fallback = bias.get("obs_beta", bias.get("obs_bias_raw"))
            if not torch.is_tensor(beta_fallback):
                raise KeyError(
                    "Dataset bias model must return log_visibility_bias_raw or obs_beta."
                )
            log_visibility_raw = torch.log(
                beta_fallback.to(device=device, dtype=dtype).clamp_min(self.eps)
            )
        else:
            log_visibility_raw = log_visibility_raw.to(device=device, dtype=dtype)

        if log_visibility_raw.shape != rho.shape:
            raise ValueError(
                "log_visibility_bias_raw shape "
                f"{tuple(log_visibility_raw.shape)} does not match rho shape {tuple(rho.shape)}."
            )

        log_visibility_raw = log_visibility_raw * mask_f
        log_visibility_bias, log_visibility_center = self._center_log_visibility_bias(
            log_bias_raw=log_visibility_raw,
            p_bio=q,
            mask=mask_b,
        )

        beta = torch.exp(log_visibility_bias).clamp_min(self.eps) * mask_f

        visible_support = q * beta
        p_visible = visible_support * mask_f

        # ------------------------------------------------------------
        # 3. Dataset-transcript scale
        # ------------------------------------------------------------
        scale_dt = bias.get("scale_dt", None)

        if torch.is_tensor(scale_dt):
            scale_dt = scale_dt.to(device=device, dtype=dtype)
            if scale_dt.ndim == 1:
                scale_dt = scale_dt.reshape(-1, 1)
            if scale_dt.shape != (rho.shape[0], 1):
                raise ValueError(
                    f"Expected scale_dt [B, 1] or [B], got {tuple(scale_dt.shape)}."
                )
        else:
            scale_dt = torch.ones((rho.shape[0], 1), device=device, dtype=dtype)

        lambda_pre_dropout = scale_dt * q * beta
        lambda_pre_dropout = lambda_pre_dropout.clamp(min=self.eps, max=self.mu_max)
        lambda_pre_dropout = lambda_pre_dropout * mask_f

        mu_bio = (scale_dt * q).clamp(min=self.eps, max=self.mu_max) * mask_f

        # ------------------------------------------------------------
        # 4. Dataset/transcript/position dropout zero inflation
        # ------------------------------------------------------------
        zero_prob = bias.get("zero_prob", bias.get("dropout_prob", None))
        zero_logits = bias.get("zero_logits", bias.get("dropout_logits", None))

        if torch.is_tensor(zero_prob):
            zero_prob = zero_prob.to(device=device, dtype=dtype)
            if zero_prob.shape != rho.shape:
                raise ValueError(
                    f"zero_prob shape {tuple(zero_prob.shape)} does not match rho shape {tuple(rho.shape)}."
                )
            zero_prob = zero_prob.clamp(1.0e-6, 1.0 - 1.0e-6)
            zero_prob = torch.where(mask_b, zero_prob, torch.full_like(zero_prob, 1.0e-6))
        elif torch.is_tensor(zero_logits):
            zero_logits = zero_logits.to(device=device, dtype=dtype)
            zero_prob = torch.sigmoid(zero_logits).clamp(1.0e-6, 1.0 - 1.0e-6)
            zero_prob = torch.where(mask_b, zero_prob, torch.full_like(zero_prob, 1.0e-6))
        else:
            zero_prob = torch.full_like(rho, 1.0e-6)
            zero_prob = torch.where(mask_b, zero_prob, torch.full_like(zero_prob, 1.0e-6))

        if torch.is_tensor(zero_logits):
            zero_logits = zero_logits.to(device=device, dtype=dtype)
        else:
            zero_logits = torch.logit(zero_prob.clamp(1.0e-6, 1.0 - 1.0e-6))

        mu_unconditional = (1.0 - zero_prob) * lambda_pre_dropout
        mu_unconditional = mu_unconditional * mask_f

        # Main prediction is the positive-component intensity. Dropout is a
        # separate observation process in the likelihood and diagnostics; it
        # should not erase profile-shape correlation.
        mu_raw = lambda_pre_dropout * mask_f

        mu = torch.nan_to_num(
            mu_raw,
            nan=self.eps,
            posinf=self.mu_max,
            neginf=self.eps,
        )
        mu = mu.clamp(min=self.eps, max=self.mu_max)
        mu = torch.where(mask_b, mu, torch.ones_like(mu))

        # ------------------------------------------------------------
        # 5. Dispersion
        # ------------------------------------------------------------
        phi = bias.get("phi", None)

        if phi is None:
            phi = torch.ones_like(mu)
        else:
            phi = phi.to(device=device, dtype=dtype)
            if phi.shape != mu.shape:
                raise ValueError(
                    f"phi shape {tuple(phi.shape)} does not match mu shape {tuple(mu.shape)}."
                )
            phi = torch.where(mask_b, phi, torch.ones_like(phi))

        # ------------------------------------------------------------
        # 6. Diagnostics
        # ------------------------------------------------------------
        valid_len = mask_f.sum(dim=1).clamp_min(1.0)

        rho_mean = (rho * mask_f).sum(dim=1) / valid_len
        q_mean = (q * mask_f).sum(dim=1) / valid_len
        q_max_sample = q.masked_fill(~mask_b, 0.0).max(dim=1).values
        p_bio_mean = (p_bio * mask_f).sum(dim=1) / valid_len
        p_visible_mean = (p_visible * mask_f).sum(dim=1) / valid_len
        beta_mean = (beta * mask_f).sum(dim=1) / valid_len
        mu_mass = (mu_raw * mask_f).sum(dim=1)
        lambda_mass = (lambda_pre_dropout * mask_f).sum(dim=1)
        mu_unconditional_mass = (mu_unconditional * mask_f).sum(dim=1)
        dropout_prob_mean = (zero_prob * mask_f).sum(dim=1) / valid_len
        visibility_log_abs_mean = (log_visibility_bias.abs() * mask_f).sum(dim=1) / valid_len
        visibility_weights = q.detach() * mask_f
        visibility_weights = visibility_weights / visibility_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)
        visibility_log_q_weighted_mean = (
            log_visibility_bias * visibility_weights
        ).sum(dim=1)
        visibility_log_center_abs = log_visibility_center.abs()

        rho_zero_frac = ((rho <= 0.0) & mask_b).float().sum(dim=1) / valid_len
        q_zero_frac = ((q <= 0.0) & mask_b).float().sum(dim=1) / valid_len
        beta_zero_frac = ((beta <= 0.0) & mask_b).float().sum(dim=1) / valid_len
        mu_zero_frac = ((mu_raw <= 0.0) & mask_b).float().sum(dim=1) / valid_len

        ones = torch.ones_like(rho)
        queue_alpha = alpha.reshape(()).expand(rho.shape[0])
        queue_propagation_enabled = torch.full(
            (rho.shape[0],),
            float(self.use_queue_propagation),
            device=device,
            dtype=dtype,
        )
        queue_alpha_trainable = torch.full(
            (rho.shape[0],),
            float(self.queue_alpha_learnable),
            device=device,
            dtype=dtype,
        )

        extras = {
            # Biological branch
            "rho_bio": rho,
            "L_bio": q,
            "q_bio": q,
            "h_bio": h_bio,
            "p_bio": p_bio,
            "mu_bio": torch.where(mask_b, mu_bio, torch.ones_like(mu_bio)),
            "J": J,
            "queue_alpha": queue_alpha,
            "queue_propagation_enabled": queue_propagation_enabled,
            "queue_alpha_trainable": queue_alpha_trainable,
            "w_bio": torch.where(mask_b, q, torch.zeros_like(q)),

            # Anchor-free visibility branch
            "log_visibility_bias_raw": torch.where(
                mask_b,
                log_visibility_raw,
                torch.zeros_like(log_visibility_raw),
            ),
            "log_visibility_bias": torch.where(
                mask_b,
                log_visibility_bias,
                torch.zeros_like(log_visibility_bias),
            ),
            "log_visibility_center": log_visibility_center,
            "obs_beta": torch.where(mask_b, beta, torch.ones_like(beta)),
            "obs_bias_raw": torch.where(mask_b, beta, torch.ones_like(beta)),
            "obs_bias_amp": torch.where(mask_b, beta, torch.ones_like(beta)),
            "obs_bias_amp_logits": torch.where(
                mask_b,
                log_visibility_bias,
                torch.zeros_like(log_visibility_bias),
            ),
            "obs_bias_geomean": torch.ones(
                (rho.shape[0],),
                dtype=dtype,
                device=device,
            ),
            "p_visible": p_visible,
            "lambda_pre_dropout": torch.where(
                mask_b,
                lambda_pre_dropout,
                torch.ones_like(lambda_pre_dropout),
            ),
            "mu_visible": torch.where(
                mask_b,
                lambda_pre_dropout,
                torch.ones_like(lambda_pre_dropout),
            ),
            "mu_unconditional": torch.where(
                mask_b,
                mu_unconditional,
                torch.ones_like(mu_unconditional),
            ),
            "mu_positive": mu,

            # Compatibility gate diagnostics. Visibility no longer gates zeros.
            "obs_bias_keep_prob": torch.where(mask_b, 1.0 - zero_prob, ones),
            "obs_bias_keep_gate": ones,
            "obs_bias_keep_gate_effective": ones,
            "obs_bias_keep_hard": ones,
            "obs_bias_gate_logits": torch.logit((1.0 - zero_prob).clamp(1.0e-6, 1.0 - 1.0e-6)),

            # Scale branch
            "scale_dt": scale_dt.reshape(-1),
            "log_scale_dt": bias.get("log_scale_dt"),
            "global_log_scale": bias.get("global_log_scale"),
            "dataset_log_scale": bias.get("dataset_log_scale"),
            "transcript_log_scale": bias.get("transcript_log_scale"),
            "dataset_scale": bias.get("dataset_scale"),
            "transcript_scale": bias.get("transcript_scale"),

            # Dropout / zero inflation
            "dropout_prob": zero_prob,
            "dropout_logits": zero_logits,
            "zero_prob": zero_prob,
            "zero_logits": zero_logits,

            # Mean / likelihood
            "mu_raw": mu_raw,
            "mu_obs": torch.where(
                mask_b,
                mu_unconditional,
                torch.ones_like(mu_unconditional),
            ),
            "mu_unconditional_raw": mu_unconditional,

            # Dispersion
            "phi": phi,
            "phi_raw": bias.get("phi"),
            "kappa_input": phi,

            # Simple diagnostics
            "rho_mean": rho_mean,
            "q_mean": q_mean,
            "q_max": q_max_sample,
            "q_zero_frac": q_zero_frac,
            "p_bio_mean": p_bio_mean,
            "p_visible_mean": p_visible_mean,
            "beta_mean": beta_mean,
            "mu_mass": mu_mass,
            "lambda_mass": lambda_mass,
            "mu_unconditional_mass": mu_unconditional_mass,
            "dropout_prob_mean": dropout_prob_mean,
            "visibility_log_abs_mean": visibility_log_abs_mean,
            "visibility_log_q_weighted_mean": visibility_log_q_weighted_mean,
            "visibility_log_center_abs": visibility_log_center_abs,
            "rho_zero_frac": rho_zero_frac,
            "beta_zero_frac": beta_zero_frac,
            "mu_zero_frac": mu_zero_frac,
        }

        return mu, phi, extras
