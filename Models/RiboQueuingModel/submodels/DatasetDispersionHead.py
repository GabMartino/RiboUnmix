from __future__ import annotations

import torch
import torch.nn as nn


class DatasetDispersionHead(nn.Module):
    """
    Dataset-level Tweedie dispersion.

        Var[Y] = phi_d * mu^p

    phi is dataset-level, not position-specific.
    """

    def __init__(
        self,
        num_datasets: int,
        phi_min: float = 0.05,
        phi_max: float = 5.0,
        init_phi: float = 1.0,
    ):
        super().__init__()

        if not (0.0 < phi_min <= init_phi <= phi_max):
            raise ValueError("Require 0 < phi_min <= init_phi <= phi_max.")

        self.num_datasets = int(num_datasets)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)

        self.phi_raw = nn.Embedding(self.num_datasets, 1)

        p = (init_phi - phi_min) / (phi_max - phi_min)
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        init_raw = torch.logit(torch.tensor(p)).item()

        nn.init.constant_(self.phi_raw.weight, init_raw)

    def forward(
        self,
        dataset_ids: torch.Tensor,
        T: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B = dataset_ids.shape[0]
        device = dataset_ids.device

        dataset_ids = dataset_ids.to(device=device, dtype=torch.long)
        mask_b = mask.to(device=device, dtype=torch.bool)

        phi_unit = torch.sigmoid(self.phi_raw(dataset_ids)).squeeze(-1)
        phi = self.phi_min + phi_unit * (self.phi_max - self.phi_min)

        phi = phi.reshape(B, 1).expand(B, T)
        phi = phi.to(device=device, dtype=torch.float32)

        # Padding value is harmless.
        phi = torch.where(mask_b, phi, torch.ones_like(phi))

        return phi