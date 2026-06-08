from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DatasetDispersionHead(nn.Module):
    """
    Transcript-level dispersion head using mean-pooled codon embeddings.

    phi = softplus(raw) ∈ (0, +∞), approaching 0 as raw → -∞.
    No hard bounds — a symmetric L2 phi-regularisation loss applied externally
    keeps phi near a target without creating gradient deadzones.

    Input:
        dataset_ids:      [B]
        codon_embeddings: [B, T, D_codon]  (already masked by caller)
        mask:             [B, T]

    Output dict:
        phi:     [B, 1]   ∈ (0, +∞)
        log_phi: [B, 1]   log(phi), used by the external phi-reg loss
    """

    def __init__(self, config_params: dict):
        super().__init__()

        self.num_datasets     = int(config_params["num_datasets"])
        self.dataset_emb_size = int(config_params.get("dataset_embedding_size", 16))
        self.codon_input_size = int(config_params["codon_input_size"])
        self.hidden_size      = int(config_params["hidden_size"])
        self.dropout          = float(config_params.get("dropout", 0.0))

        self.init_phi = float(config_params.get("init_phi", 5.0))
        assert self.init_phi > 0, "init_phi must be positive"

        self.dataset_embedding = nn.Embedding(self.num_datasets, self.dataset_emb_size)
        self.codon_norm = nn.LayerNorm(self.codon_input_size)

        mlp_input_size = self.dataset_emb_size + self.codon_input_size

        self.ff = nn.Sequential(
            nn.Linear(mlp_input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        # phi = softplus(bias)  →  bias = softplus_inv(init_phi) = log(exp(init_phi) - 1)
        # Valid for any init_phi > 0.
        init_bias = math.log(math.exp(self.init_phi) - 1.0)
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.constant_(self.ff[-1].bias, init_bias)

    def forward(
        self,
        *,
        dataset_ids: torch.Tensor,
        codon_embeddings: torch.Tensor,
        mask: torch.Tensor,
        biological_context: torch.Tensor = None,  # unused, kept for interface compat
    ) -> dict[str, torch.Tensor]:
        B = int(dataset_ids.shape[0])
        device = dataset_ids.device

        dataset_ids = dataset_ids.to(device=device, dtype=torch.long)
        dataset_weight = self.dataset_embedding.weight
        dataset_weight = dataset_weight - dataset_weight.mean(dim=0, keepdim=True)
        dataset_emb = F.embedding(dataset_ids, dataset_weight)
        dtype = dataset_emb.dtype

        # Masked mean-pool codon embeddings → transcript composition [B, D_codon]
        mask_f = mask.to(dtype=dtype)
        count = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (codon_embeddings.to(dtype=dtype) * mask_f.unsqueeze(-1)).sum(dim=1) / count

        x = torch.cat([dataset_emb, self.codon_norm(pooled)], dim=-1)

        # phi = softplus(raw) ∈ (0, +∞), gradient = sigmoid(raw) ∈ (0, 1) always
        phi = F.softplus(self.ff(x))
        log_phi = torch.log(phi.clamp_min(1e-8))

        return {"phi": phi, "log_phi": log_phi}  # both [B, 1]
