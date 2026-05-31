from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def inv_softplus(x: float) -> float:
    """
    Numerically stable inverse softplus.

    Returns y such that:

        softplus(y) = x

    Requires x > 0.
    """
    x = float(x)

    if x <= 0.0:
        raise ValueError(f"inv_softplus requires x > 0, got {x}.")

    if x > 20.0:
        return x

    return math.log(math.expm1(x))


def logit(p: float) -> float:
    """
    Numerically stable scalar logit.
    """
    p = float(p)
    p = min(max(p, 1.0e-6), 1.0 - 1.0e-6)
    return math.log(p / (1.0 - p))


class DatasetMultiplicativeAllocationBiasHead(nn.Module):
    """
    Dataset/protocol multiplicative allocation-bias head.

    Single-score gated-amplitude formulation.

    It predicts one scalar score per position:

        beta_i = f_d(x_i)

    Then:

        amp_i = amp_min + softplus(beta_i)

        keep_prob_i = sigmoid((beta_i - gate_cutoff) / gate_temperature)

        keep_gate_i =
            keep_prob_i                         if soft/continuous
            hard 0/1 with straight-through       if hard_forward=True during training
            hard 0/1                             if hard_eval=True during eval

        b_raw_i = keep_gate_i * amp_i

    The outer model should normalize:

        b_eff_i = b_raw_i / sum_j w_bio_j b_raw_j

        w_obs_i = w_bio_i * b_eff_i

    Therefore:

        sum_i w_obs_i = 1

    Why this version is cleaner than two independent heads:
        - low beta means low amplitude and low keep probability
        - high beta means high amplitude and high keep probability
        - the gate is a thresholded amplitude score, not an independent
          second multiplicative branch
        - exact zeros remain possible when keep_gate is hard 0

    Neutral initialization:
        beta_0 is chosen so that:

            amp = 1

        gate_cutoff is chosen so that:

            keep_prob = init_keep_prob

        at beta = beta_0.
    """

    def __init__(
        self,
        config_params: dict,
        input_size: int,
    ) -> None:
        super().__init__()

        self.hidden_size = int(config_params.get("hidden_size", 128))
        self.dropout = float(config_params.get("dropout", 0.0))

        # ------------------------------------------------------------
        # Amplitude transform
        # ------------------------------------------------------------
        # amp = amp_min + softplus(beta)
        #
        # Use amp_min self.hidden_size = int(config_params.get("hidden_size", 128))
        self.dropout = float(config_params.get("dropout", 0.0))

        # ------------------------------------------------------------
        # Am=0.0 for maximum multiplicative capacity.
        # Use amp_min>0 only if you explicitly want the hard gate to be the
        # only mechanism capable of reaching exact/near-zero visibility.
        self.amp_min = float(config_params.get("amp_min", 0.0))
        self.amp_max = float(config_params.get("amp_max", 20.0))
        self.amp_eps = float(config_params.get("amp_eps", 1.0e-8))

        if not (0.0 <= self.amp_min < 1.0):
            raise ValueError(
                f"amp_min must be in [0, 1) so neutral amp=1 is possible. "
                f"Got amp_min={self.amp_min}."
            )

        if self.amp_max > 0.0 and self.amp_max < 1.0:
            raise ValueError(
                f"amp_max must be >= 1 if enabled, otherwise neutral amp=1 "
                f"would be clipped. Got amp_max={self.amp_max}."
            )

        if self.amp_eps <= 0.0:
            raise ValueError(f"amp_eps must be > 0. Got {self.amp_eps}.")

        # Neutral beta gives amp = 1:
        #
        #   amp_min + softplus(beta_0) = 1
        #
        self.beta_neutral = inv_softplus(1.0 - self.amp_min)

        # ------------------------------------------------------------
        # Gate transform from the same beta
        # ------------------------------------------------------------
        self.gate_temperature = max(
            float(config_params.get("gate_temperature", 0.5)),
            1.0e-6,
        )

        self.gate_threshold = float(config_params.get("gate_threshold", 0.5))
        self.init_keep_prob = float(config_params.get("init_keep_prob", 0.95))

        if not (0.0 < self.gate_threshold < 1.0):
            raise ValueError(
                f"gate_threshold must be in (0, 1). Got {self.gate_threshold}."
            )

        if not (0.0 < self.init_keep_prob < 1.0):
            raise ValueError(
                f"init_keep_prob must be in (0, 1). Got {self.init_keep_prob}."
            )

        # Choose cutoff such that:
        #
        #   sigmoid((beta_neutral - gate_cutoff) / tau) = init_keep_prob
        #
        # Therefore:
        #
        #   gate_cutoff = beta_neutral - tau * logit(init_keep_prob)
        #
        self.gate_cutoff = (
            self.beta_neutral
            - self.gate_temperature * logit(self.init_keep_prob)
        )

        # Recommended for performance first:
        #   hard_forward=False
        #   hard_eval=False
        #
        # Recommended only for exact-zero diagnostic:
        #   hard_eval=True
        self.hard_forward = bool(config_params.get("hard_forward", False))
        self.hard_eval = bool(config_params.get("hard_eval", False))

        # ------------------------------------------------------------
        # Network
        # ------------------------------------------------------------
        self.shared = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
        )

        # One score controls both amplitude and gate.
        self.bias_score_head = nn.Linear(self.hidden_size, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize to neutral observation bias:

            beta = beta_neutral
            amp = 1
            keep_prob = init_keep_prob
            b_raw ≈ init_keep_prob

        Since the outer model normalizes b_raw by its w_bio-weighted mass,
        a constant b_raw is neutral for w_obs.
        """
        nn.init.zeros_(self.bias_score_head.weight)
        nn.init.constant_(self.bias_score_head.bias, self.beta_neutral)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, T, F], got {tuple(x.shape)}.")

        if mask.ndim != 2:
            raise ValueError(f"Expected mask [B, T], got {tuple(mask.shape)}.")

        if x.shape[:2] != mask.shape:
            raise ValueError(
                f"x and mask shape mismatch: x={tuple(x.shape)}, "
                f"mask={tuple(mask.shape)}."
            )

        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=x.dtype)

        x = x * mask_f.unsqueeze(-1)

        h = self.shared(x)

        # ============================================================
        # Single visibility score
        # ============================================================
        beta = self.bias_score_head(h).squeeze(-1)

        # ============================================================
        # Amplitude from beta
        # ============================================================
        amp = self.amp_min + F.softplus(beta)
        amp = amp.clamp_min(self.amp_eps)

        if self.amp_max > 0.0:
            amp = amp.clamp_max(self.amp_max)

        amp = amp * mask_f

        # ============================================================
        # Gate from same beta
        # ============================================================
        gate_logits = (beta - self.gate_cutoff) / self.gate_temperature

        keep_prob = torch.sigmoid(gate_logits)
        keep_prob = keep_prob * mask_f

        keep_hard = (keep_prob > self.gate_threshold).to(dtype=x.dtype)
        keep_hard = keep_hard * mask_f

        if self.training:
            if self.hard_forward:
                # Straight-through estimator:
                #
                # forward: hard 0/1
                # backward: soft keep_prob gradient
                keep_gate = keep_hard + keep_prob - keep_prob.detach()
            else:
                keep_gate = keep_prob
        else:
            keep_gate = keep_hard if self.hard_eval else keep_prob

        keep_gate = keep_gate * mask_f

        # Single-score gated amplitude.
        b_raw = keep_gate * amp
        b_raw = b_raw * mask_f

        # ============================================================
        # Padding-neutral diagnostic outputs
        # ============================================================
        beta_diag = torch.where(
            mask_b,
            beta,
            torch.full_like(beta, self.beta_neutral),
        )

        amp_diag = torch.where(
            mask_b,
            amp,
            torch.ones_like(amp),
        )

        keep_prob_diag = torch.where(
            mask_b,
            keep_prob,
            torch.ones_like(keep_prob),
        )

        keep_hard_diag = torch.where(
            mask_b,
            keep_hard,
            torch.ones_like(keep_hard),
        )

        keep_gate_diag = torch.where(
            mask_b,
            keep_gate,
            torch.ones_like(keep_gate),
        )

        gate_logits_diag = torch.where(
            mask_b,
            gate_logits,
            torch.zeros_like(gate_logits),
        )

        b_raw_diag = torch.where(
            mask_b,
            b_raw,
            torch.ones_like(b_raw),
        )

        # Keep backward-compatible keys:
        #   obs_bias_amp_logits now means beta / visibility score.
        #   obs_bias_gate_logits means the actual pre-sigmoid gate logit.
        return {
            "obs_bias_raw": b_raw_diag,

            "obs_bias_beta": beta_diag,
            "obs_bias_score": beta_diag,

            "obs_bias_amp": amp_diag,
            "obs_bias_amp_logits": beta_diag,

            "obs_bias_keep_prob": keep_prob_diag,
            "obs_bias_keep_gate": keep_gate_diag,
            "obs_bias_keep_hard": keep_hard_diag,
            "obs_bias_gate_logits": gate_logits_diag,

            # Useful constants for diagnostics/debugging.
            "obs_bias_beta_neutral": torch.full(
                (x.shape[0],),
                float(self.beta_neutral),
                dtype=x.dtype,
                device=x.device,
            ),
            "obs_bias_gate_cutoff": torch.full(
                (x.shape[0],),
                float(self.gate_cutoff),
                dtype=x.dtype,
                device=x.device,
            ),
        }