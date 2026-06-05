from __future__ import annotations

import math
import torch
import torch.nn as nn


class DatasetDispersionHead(nn.Module):
    """
    Position-level dispersion head.

    Produces one dispersion value per valid transcript position.

    Input:
        x:    [B, T, F]
        mask: [B, T]

    Output:
        phi: [B, T]

    Internally:
        phi_unit: [B, T]
        phi:      [B, T]
    """

    def __init__(
        self,
        config_params: dict,
        input_size: int,
    ):
        super().__init__()

        self.hidden_size = int(config_params["hidden_size"])
        self.phi_min = float(config_params["phi_min"])
        self.phi_max = float(config_params["phi_max"])
        self.dropout = float(config_params.get("dropout", 0.0))

        self.init_phi = float(config_params.get("init_phi", 0.5))
        self.init_phi = min(max(self.init_phi, self.phi_min), self.phi_max)

        self.ff = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        self.log_phi_min = math.log(self.phi_min)
        self.log_phi_max = math.log(self.phi_max)

        init_unit = (math.log(self.init_phi) - self.log_phi_min) / (
            self.log_phi_max - self.log_phi_min
        )
        init_unit = min(max(init_unit, 1e-6), 1.0 - 1e-6)
        init_raw = math.log(init_unit / (1.0 - init_unit))

        nn.init.zeros_(self.ff[-1].weight)
        nn.init.constant_(self.ff[-1].bias, init_raw)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=x.dtype)

        x = x * mask_f.unsqueeze(-1)

        phi_unit = torch.sigmoid(self.ff(x).squeeze(-1))

        log_phi = self.log_phi_min + phi_unit * (
            self.log_phi_max - self.log_phi_min
        )
        phi = torch.exp(log_phi).clamp(self.phi_min, self.phi_max)

        phi = torch.where(
            mask_b,
            phi,
            torch.ones_like(phi),
        )

        return phi
