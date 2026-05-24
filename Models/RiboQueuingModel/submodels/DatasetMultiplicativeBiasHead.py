from __future__ import annotations

import torch
from torch import nn


class DatasetMultiplicativeBiasHead(nn.Module):
    """
    Dataset/protocol multiplicative correction head.

    Produces a bounded multiplier that supports exact zeros.

    Let c_i = log_b_max * tanh(beta_centered_i / tanh_temperature)

    To allow massive spikes but also exact zeros, we use a C1-continuous piecewise activation:
        If c_i >= 0: b_i = exp(c_i)         # Range: [1, exp(log_b_max)]
        If c_i <  0: b_i = ReLU(1.0 + c_i)  # Range: [0, 1)

    b_i = 0 is achieved exactly when c_i <= -1.
    Neutral initialization (c_i = 0) gives b_i = 1.
    """

    def __init__(self, config_params: dict, input_size: int) -> None:
        super().__init__()

        self.hidden_size = int(config_params["hidden_size"])
        self.dropout = float(config_params.get("dropout", 0.0))
        self.log_b_max = float(config_params.get("log_b_max", 5.0))
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

        # 1. Calculate the bounded control signal
        control = self.log_b_max * torch.tanh(
            beta_centered / self.tanh_temperature
        )
        control = control * mask_f

        # 2. Apply C1-continuous piecewise mapping
        b_pos = torch.exp(control)  # Exponential growth for spikes
        b_neg = torch.relu(1.0 + control)  # Linear decay to exact zero

        b = torch.where(control >= 0.0, b_pos, b_neg)
        b = torch.where(mask_b, b, torch.ones_like(b))

        # We return 'control' as the second argument instead of actual log(b).
        # See the critical warning below.
        return b, control, beta_centered