from __future__ import annotations

import math

import torch
import torch.nn as nn


class DatasetCodonDispersionHead(nn.Module):
    """
    Dataset/codon-dependent Tweedie dispersion.

    Mathematical form:

        Var[Y_{d,t,i}] = phi[d, codon_i] * mu_{d,t,i}^p

    where:

        phi[d, codon_i] =
            phi_min + (phi_max - phi_min) * sigmoid(g(dataset_d, codon_i))

    This head deliberately sees only dataset ID and codon ID.
    It should not see L_queue, mu, y, or RNN hidden states.
    """

    def __init__(
        self,
        num_datasets: int,
        num_codons: int = 64,
        dataset_emb_dim: int = 16,
        codon_emb_dim: int = 8,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        phi_min: float = 0.05,
        phi_max: float = 5.0,
        init_phi: float = 1.0,
    ):
        super().__init__()

        if num_datasets <= 0:
            raise ValueError("num_datasets must be > 0.")

        if num_codons <= 0:
            raise ValueError("num_codons must be > 0.")

        if not (0.0 < phi_min <= init_phi <= phi_max):
            raise ValueError(
                "Require 0 < phi_min <= init_phi <= phi_max. "
                f"Got phi_min={phi_min}, init_phi={init_phi}, phi_max={phi_max}."
            )

        self.num_datasets = int(num_datasets)
        self.num_codons = int(num_codons)

        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)

        self.dataset_embedding = nn.Embedding(
            self.num_datasets,
            int(dataset_emb_dim),
        )

        self.codon_embedding = nn.Embedding(
            self.num_codons,
            int(codon_emb_dim),
        )

        in_dim = int(dataset_emb_dim) + int(codon_emb_dim)

        self.ff = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize phi approximately to init_phi everywhere.
        init_unit = (float(init_phi) - self.phi_min) / (self.phi_max - self.phi_min)
        init_unit = min(max(init_unit, 1e-6), 1.0 - 1e-6)
        init_raw = math.log(init_unit / (1.0 - init_unit))

        nn.init.zeros_(self.ff[-1].weight)
        nn.init.constant_(self.ff[-1].bias, init_raw)

    def _check_dataset_ids(self, dataset_ids: torch.Tensor) -> None:
        min_id = int(dataset_ids.min().detach().cpu())
        max_id = int(dataset_ids.max().detach().cpu())

        if min_id < 0 or max_id >= self.num_datasets:
            raise ValueError(
                f"dataset_ids out of range: min={min_id}, max={max_id}, "
                f"num_datasets={self.num_datasets}. Need num_datasets >= {max_id + 1}."
            )

    def forward(
        self,
        *,
        dataset_ids: torch.Tensor,
        codon_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        dataset_ids:
            [B]

        codon_ids:
            [B, T]

        mask:
            [B, T] bool

        Returns
        -------
        phi:
            [B, T]
        """
        if codon_ids.ndim != 2:
            raise ValueError(f"codon_ids must have shape [B, T], got {codon_ids.shape}.")

        B, T = codon_ids.shape
        device = codon_ids.device

        dataset_ids = dataset_ids.to(device=device, dtype=torch.long)
        codon_ids = codon_ids.to(device=device, dtype=torch.long)
        mask_b = mask.to(device=device, dtype=torch.bool)

        self._check_dataset_ids(dataset_ids)

        codon_ids = codon_ids.clamp(min=0, max=self.num_codons - 1)

        dataset_emb = self.dataset_embedding(dataset_ids)       # [B, D]
        dataset_emb = dataset_emb.unsqueeze(1).expand(B, T, -1) # [B, T, D]

        codon_emb = self.codon_embedding(codon_ids)             # [B, T, C]

        x = torch.cat([dataset_emb, codon_emb], dim=-1)

        phi_unit = torch.sigmoid(self.ff(x).squeeze(-1))

        phi = self.phi_min + phi_unit * (self.phi_max - self.phi_min)

        # Padding value should be harmless.
        phi = torch.where(mask_b, phi, torch.ones_like(phi))

        return phi