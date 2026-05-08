from __future__ import annotations

import math

import torch
import torch.nn as nn


class DatasetDispersionHead(nn.Module):
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
            config_params: dict,
        init_phi: float = 1.0,
    ):
        super().__init__()

        self.num_datasets = config_params["num_datasets"]
        self.dataset_embeddings_size = config_params["dataset_embeddings_size"]
        self.num_codons = config_params["num_codons"]
        self.codon_embeddings_size = config_params["codon_embeddings_size"]
        self.hidden_size = config_params["hidden_size"]
        self.dropout = config_params["dropout"]
        self.phi_min = config_params["phi_min"]
        self.phi_max = config_params["phi_max"]

        self.dataset_embedding = nn.Embedding(self.num_datasets,self.dataset_embeddings_size)

        self.codon_embedding = nn.Embedding( self.num_codons,self.codon_embeddings_size)

        in_dim = self.dataset_embeddings_size + self.codon_embeddings_size

        self.ff = nn.Sequential(
            nn.Linear(in_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        # Initialize phi approximately to init_phi everywhere.
        init_unit = (float(init_phi) - self.phi_min) / (self.phi_max - self.phi_min)
        init_unit = min(max(init_unit, 1e-6), 1.0 - 1e-6)
        init_raw = math.log(init_unit / (1.0 - init_unit))

        nn.init.zeros_(self.ff[-1].weight)
        nn.init.constant_(self.ff[-1].bias, init_raw)


    def forward( self, dataset_ids: torch.Tensor, codon_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:


        B, T = codon_ids.shape

        dataset_embeddings = self.dataset_embedding(dataset_ids)   .unsqueeze(1).expand(B, T, -1) # [B, T, D]
        codon_embeddings = self.codon_embedding(codon_ids)             # [B, T, C]

        x = torch.cat([dataset_embeddings, codon_embeddings], dim=-1)

        phi_unit = torch.sigmoid(self.ff(x).squeeze(-1))

        phi = self.phi_min + phi_unit * (self.phi_max - self.phi_min)

        phi = torch.where(mask, phi, torch.ones_like(phi))

        return phi