from __future__ import annotations

import torch
import torch.nn as nn


class DatasetAdditiveBiasHead(nn.Module):
    def __init__(self, config_params: dict):
        super().__init__()

        self.num_datasets = config_params["num_datasets"]
        self.dataset_embeddings_size = config_params["dataset_embeddings_size"]

        self.num_codons = config_params["num_codons"]
        self.codon_embeddings_size = config_params["codon_embeddings_size"]
        self.hidden_size = config_params["hidden_size"]
        self.dropout = config_params["dropout"]

        self.dataset_embedding = nn.Embedding(self.num_datasets, self.dataset_embeddings_size)

        self.codon_embedding = nn.Embedding(self.num_codons, self.codon_embeddings_size)

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
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.constant_(self.ff[-1].bias, -8.0)

    def forward(
        self,
        dataset_ids: torch.Tensor,
        codon_ids: torch.Tensor,
        S_mean: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:


        B, T = codon_ids.shape

        dataset_embeddings = self.dataset_embedding(dataset_ids)          # [B, D]
        dataset_embeddings = dataset_embeddings.unsqueeze(1).expand(B, T, -1)    # [B, T, D]

        codon_embeddings = self.codon_embedding(codon_ids)                # [B, T, C]

        x = torch.cat([dataset_embeddings, codon_embeddings], dim=-1)

        beta_per_position = torch.nn.functional.softplus(
            self.ff(x).squeeze(-1)
        )

        additive_bias = S_mean.reshape(B, 1) * beta_per_position * mask

        return additive_bias, beta_per_position