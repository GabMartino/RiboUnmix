from __future__ import annotations

import entmax
import torch
import torch.nn as nn
from entmax import entmax15


class DatasetAdditiveBiasHead(nn.Module):
    """
    Dataset/protocol additive background head.

    This is NOT multiplicative.

    It predicts:

        R_i = T * softmax(r_i)

    so:

        mean_valid(R) = 1

    and a non-negative transcript-level amplitude:

        lambda_bg = lambda_max * sigmoid(lambda_raw)

    The additive observation support is:

        additive_noise_i = lambda_bg * R_i

    Intended usage:

        bio_q = L_queue * b
        q     = bio_q + additive_noise
        mu    = total_mass * q / sum(q)

    Neutral-ish initialization:
        R is uniform.
        lambda_bg starts small.
    """

    def __init__(self, config_params: dict, input_size: int):
        super().__init__()

        self.hidden_size = int(config_params["hidden_size"])
        self.dropout = float(config_params.get("dropout", 0.0))

        self.lambda_max = float(config_params.get("lambda_max", 1.0))
        self.lambda_init = float(config_params.get("lambda_init", 0.01))

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
        # r_logits = 0 -> softmax uniform -> R_i = 1 over valid positions.
        nn.init.zeros_(self.r_ff[-1].weight)
        nn.init.zeros_(self.r_ff[-1].bias)

        # lambda starts small.
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
        # 1. Background shape R
        # ------------------------------------------------------------
        r_logits = self.r_ff(x).squeeze(-1)
        r_logits = r_logits.masked_fill(~mask_b, -torch.inf)

        r_prob = entmax15(r_logits, dim=1)
        r_prob = r_prob * mask_f

        # Mean-one background shape.
        R_shape = valid_lengths * r_prob
        R_shape = R_shape * mask_f

        # ------------------------------------------------------------
        # 2. Background amplitude lambda
        # ------------------------------------------------------------
        pooled_x = (x * mask_f.unsqueeze(-1)).sum(dim=1) / valid_lengths

        lambda_raw = self.lambda_ff(pooled_x).squeeze(-1)
        lambda_bg = self.lambda_max * torch.sigmoid(lambda_raw)
        lambda_bg = lambda_bg.reshape(B, 1)

        additive_noise = lambda_bg * R_shape
        additive_noise = additive_noise * mask_f

        return additive_noise, R_shape, lambda_bg, r_logits, lambda_raw