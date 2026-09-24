from __future__ import annotations

import torch
import torch.nn as nn


class DatasetLogSigmaHead(nn.Module):
    """
    Per-position log-dispersion head for the NB2 observation model.

    ``log_sigma`` is retained as the public key for checkpoint and loss
    compatibility. The head only depends on the already assembled feature
    tensor; dataset and codon context are included in that tensor by the
    caller.

    Input:
        x:    [B, T, D]
        mask: [B, T]

    Output dict:
        log_sigma: [B, T]
    """

    def __init__(self, config_params: dict, input_size: int):
        super().__init__()

        self.hidden_size = int(config_params["hidden_size"])
        self.dropout = float(config_params.get("dropout", 0.0))
        self.input_size = input_size

        self.init_log_sigma = float(config_params.get("init_log_sigma", 0.0))
        self.ff = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        nn.init.zeros_(self.ff[-1].weight)
        nn.init.constant_(self.ff[-1].bias, self.init_log_sigma)

    def forward(
        self,
        *,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask_f = mask.to(dtype=x.dtype)

        log_sigma = self.ff(x).squeeze(-1) * mask_f

        return {"log_sigma": log_sigma}  # [B, T]
