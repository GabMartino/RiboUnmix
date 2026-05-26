from __future__ import annotations

import torch
import torch.nn as nn
from entmax import entmax15


class DatasetAdditiveBiasHead(nn.Module):
    """
    Dataset/protocol additive relative-background head.

    This head does NOT directly output final additive support.

    It predicts:

        R_i = T * entmax/softmax(r_i)

    so:

        mean_valid(R) = 1

    and a non-negative transcript-level relative amplitude:

        lambda_frac = lambda_max * sigmoid(lambda_raw)

    The returned relative additive shape is:

        additive_rel_i = lambda_frac * R_i

    Intended usage in the full model:

        bio_q_smooth_i = L_i * b_i

        bio_mean = mean_valid(bio_q_smooth).detach()

        additive_noise_i = additive_rel_i * bio_mean

        q_i = keep_gate_i * (bio_q_smooth_i + additive_noise_i)

    Therefore lambda_frac is interpretable as an approximate additive/background
    fraction relative to the biological support scale.

    Example:
        lambda_frac = 0.05 means additive support is roughly 5% of the
        mean biological support per valid position.
    """

    def __init__(self, config_params: dict, input_size: int):
        super().__init__()

        self.hidden_size = int(config_params["hidden_size"])
        self.dropout = float(config_params.get("dropout", 0.0))

        # Now interpreted as maximum relative amplitude, not absolute support.
        self.lambda_max = float(config_params.get("lambda_max", 0.05))
        self.lambda_init = float(config_params.get("lambda_init", 0.001))

        self.use_entmax = bool(config_params.get("use_entmax", True))

        self.r_ff = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        self.lambda_ff = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        # R uniform at initialization:
        # r_logits = 0 -> uniform -> R_i = 1 over valid positions.
        nn.init.zeros_(self.r_ff[-1].weight)
        nn.init.zeros_(self.r_ff[-1].bias)

        # lambda_frac starts small.
        nn.init.zeros_(self.lambda_ff[-1].weight)

        init_frac = self.lambda_init / max(self.lambda_max, 1e-8)
        init_frac = min(max(init_frac, 1e-6), 1.0 - 1e-6)
        lambda_bias = torch.logit(torch.tensor(init_frac)).item()
        nn.init.constant_(self.lambda_ff[-1].bias, lambda_bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=x.dtype)

        x = x * mask_f.unsqueeze(-1)

        B, T, _ = x.shape
        valid_lengths = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        # ------------------------------------------------------------
        # 1. Relative background shape R
        # ------------------------------------------------------------
        r_logits = self.r_ff(x).squeeze(-1)
        r_logits = r_logits.masked_fill(~mask_b, -torch.inf)

        if self.use_entmax:
            r_prob = entmax15(r_logits, dim=1)
        else:
            r_prob = torch.softmax(r_logits, dim=1)

        r_prob = r_prob * mask_f

        # Mean-one background shape.
        R_shape = valid_lengths * r_prob
        R_shape = R_shape * mask_f

        # ------------------------------------------------------------
        # 2. Relative amplitude lambda_frac
        # ------------------------------------------------------------
        pooled_x = (x * mask_f.unsqueeze(-1)).sum(dim=1) / valid_lengths

        lambda_raw = self.lambda_ff(pooled_x).squeeze(-1)
        lambda_frac = self.lambda_max * torch.sigmoid(lambda_raw)
        lambda_frac = lambda_frac.reshape(B, 1)

        # Dimensionless relative additive shape.
        additive_rel = lambda_frac * R_shape
        additive_rel = additive_rel * mask_f

        return additive_rel, R_shape, lambda_frac, r_logits, lambda_raw