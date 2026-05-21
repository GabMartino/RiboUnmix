from __future__ import annotations

import torch
import torch.nn as nn


class DatasetZerosHead(nn.Module):
    """
    Hard keep gate.

    keep_hard = 1: keep continuous mean.
    keep_hard = 0: force mean close to zero.

    The head returns both hard and soft gates.
    """

    def __init__(self, config_params: dict, input_size: int):
        super().__init__()

        hidden_size = int(config_params["hidden_size"])
        dropout = float(config_params.get("dropout", 0.0))

        self.temperature = float(config_params.get("temperature", 1.0))
        self.init_keep_prob = float(config_params.get("init_keep_prob", 0.90))

        self.keep_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        nn.init.zeros_(self.keep_head[-1].weight)

        init_p = min(max(self.init_keep_prob, 1e-4), 1.0 - 1e-4)
        init_bias = torch.logit(torch.tensor(init_p)).item()
        nn.init.constant_(self.keep_head[-1].bias, init_bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=x.dtype)

        x = x * mask_f.unsqueeze(-1)

        keep_logits = self.keep_head(x).squeeze(-1)

        keep_prob = torch.sigmoid(
            keep_logits / max(self.temperature, 1e-6)
        )
        keep_prob = keep_prob * mask_f

        keep_hard = (keep_prob > 0.5).to(dtype=x.dtype)
        keep_hard = keep_hard * mask_f

        return keep_hard, keep_prob, keep_logits