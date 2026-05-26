from __future__ import annotations

import math
import torch
import torch.nn as nn


class DatasetDispersionHead(nn.Module):
    """
    Dataset/transcript-level dispersion head.

    Produces one scalar dispersion value per transcript-dataset pair,
    then broadcasts it over valid positions.

    Input:
        x:    [B, T, F]
        mask: [B, T]

    Output:
        phi: [B, T]

    Internally:
        pooled_x:   [B, F]
        phi_scalar: [B, 1]
        phi:        [B, T]
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

        init_unit = (self.init_phi - self.phi_min) / (self.phi_max - self.phi_min)
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

        B, T, _ = x.shape

        x = x * mask_f.unsqueeze(-1)

        valid_lengths = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        # One pooled representation per transcript-dataset pair.
        pooled_x = x.sum(dim=1) / valid_lengths

        phi_unit = torch.sigmoid(self.ff(pooled_x))  # [B, 1]

        phi_scalar = self.phi_min + phi_unit * (self.phi_max - self.phi_min)
        phi_scalar = phi_scalar.clamp(self.phi_min, self.phi_max)  # [B, 1]

        # Broadcast the same scalar over valid positions.
        phi = phi_scalar.expand(B, T)
        phi = torch.where(
            mask_b,
            phi,
            torch.ones_like(phi),
        )

        return phi