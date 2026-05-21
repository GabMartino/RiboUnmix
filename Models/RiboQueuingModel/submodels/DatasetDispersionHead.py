from __future__ import annotations

import math
import torch
import torch.nn as nn


class DatasetDispersionHead(nn.Module):
    """
    Dataset/sequence-dependent Tweedie dispersion.
    Now accepts a pre-computed local sequence context.
    """

    def __init__(
            self,
            config_params: dict,
            input_size: int,
    ):
        super().__init__()

        self.hidden_size = config_params["hidden_size"]
        self.phi_min = config_params["phi_min"]
        self.phi_max = config_params["phi_max"]
        self.dropout = config_params["dropout"]

        self.ff = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        init_unit = (float(0.5) - self.phi_min) / (self.phi_max - self.phi_min)
        init_unit = min(max(init_unit, 1e-6), 1.0 - 1e-6)
        init_raw = math.log(init_unit / (1.0 - init_unit))

        nn.init.zeros_(self.ff[-1].weight)
        nn.init.constant_(self.ff[-1].bias, init_raw)

    def forward(self, x, mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        phi_unit = torch.sigmoid(self.ff(x).squeeze(-1))
        phi = self.phi_min + phi_unit * (self.phi_max - self.phi_min)
        phi = torch.where(mask, phi, torch.ones_like(phi))
        return phi