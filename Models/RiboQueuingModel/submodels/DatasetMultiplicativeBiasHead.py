from __future__ import annotations

import torch
from torch import nn


class DatasetMultiplicativeBiasHead(nn.Module):
    """
    Dataset/protocol multiplicative correction head.

    Produces a bounded log-space multiplier:

        beta_i = f(x_i)
        beta_centered_i = beta_i - mean_valid(beta)
        log_b_i = log_b_max * tanh(beta_centered_i / tanh_temperature)
        b_i = exp(log_b_i)

    Therefore:

        b_i in [exp(-log_b_max), exp(log_b_max)]

    Neutral initialization gives b_i = 1.
    """

    def __init__(self, config_params: dict, input_size: int) -> None:
        super().__init__()

        self.hidden_size = int(config_params["hidden_size"])
        self.dropout = float(config_params.get("dropout", 0.0))
        self.log_b_max = float(config_params.get("log_b_max", 0.1))
        self.tanh_temperature = float(config_params.get("tanh_temperature", 3.0))

        self.bias_ff = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1, bias=False),
        )

        nn.init.zeros_(self.bias_ff[-1].weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=x.dtype)

        x = x * mask_f.unsqueeze(-1)

        beta = self.bias_ff(x).squeeze(-1)
        beta = beta * mask_f

        valid_lengths = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        beta_mean = beta.sum(dim=1, keepdim=True) / valid_lengths
        beta_centered = (beta - beta_mean) * mask_f

        log_b = self.log_b_max * torch.tanh(
            beta_centered / self.tanh_temperature
        )
        log_b = log_b * mask_f

        b = torch.exp(log_b)
        b = torch.where(mask_b, b, torch.ones_like(b))

        return b, log_b, beta_centered