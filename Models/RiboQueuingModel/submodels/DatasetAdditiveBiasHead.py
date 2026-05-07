from __future__ import annotations

import torch
import torch.nn as nn


class DatasetAdditiveBiasHead(nn.Module):
    def __init__(
        self,
        num_datasets: int,
        num_codons: int = 64,
        dataset_emb_dim: int = 16,
        codon_emb_dim: int = 8,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        init_bias: float = -8.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.num_datasets = int(num_datasets)
        self.num_codons = int(num_codons)
        self.eps = float(eps)

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

        # Start almost off.
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.constant_(self.ff[-1].bias, float(init_bias))

    def forward(
        self,
        *,
        dataset_ids: torch.Tensor,
        codon_ids: torch.Tensor,
        S_mean: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if codon_ids.ndim != 2:
            raise ValueError(f"codon_ids must have shape [B, T], got {codon_ids.shape}.")

        B, T = codon_ids.shape
        device = codon_ids.device

        dataset_ids = dataset_ids.to(device=device, dtype=torch.long)
        codon_ids = codon_ids.to(device=device, dtype=torch.long)
        mask_b = mask.to(device=device, dtype=torch.bool)
        mask_f = mask_b.float()


        codon_ids = codon_ids.clamp(min=0, max=self.num_codons - 1)

        dataset_emb = self.dataset_embedding(dataset_ids)          # [B, D]
        dataset_emb = dataset_emb.unsqueeze(1).expand(B, T, -1)    # [B, T, D]

        codon_emb = self.codon_embedding(codon_ids)                # [B, T, C]

        x = torch.cat([dataset_emb, codon_emb], dim=-1)

        additive_rel = torch.nn.functional.softplus(
            self.ff(x).squeeze(-1)
        )

        additive_rel = additive_rel * mask_f

        S = S_mean.reshape(B, 1).to(device=device, dtype=additive_rel.dtype)
        S = S.clamp_min(self.eps)

        additive_bg = S * additive_rel
        additive_bg = additive_bg * mask_f

        return additive_bg, additive_rel