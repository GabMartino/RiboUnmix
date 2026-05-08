import torch
from torch import nn


class DatasetMultiplicativeBiasHead(nn.Module):
    def __init__(self, config_params: dict):
        super().__init__()

        self.num_datasets = config_params["num_datasets"]
        self.dataset_embeddings_size = config_params["dataset_embeddings_size"]
        self.num_codons = config_params["num_codons"]
        self.codon_embeddings_size = config_params["codon_embeddings_size"]
        self.hidden_size = config_params["hidden_size"]
        self.dropout = config_params["dropout"]

        self.dataset_embedding = nn.Embedding(self.num_datasets, self.dataset_embeddings_size)
        self.codon_embedding = nn.Embedding(self.num_codons,self.codon_embeddings_size)

        ff_in_dim = self.dataset_embeddings_size + self.codon_embeddings_size

        self.bias_ff = nn.Sequential(
            nn.Linear(ff_in_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        # Start with no multiplicative bias.
        nn.init.zeros_(self.bias_ff[-1].weight)
        nn.init.zeros_(self.bias_ff[-1].bias)



    def forward(self, dataset_ids: torch.Tensor, codon_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        B, T = codon_ids.shape


        codon_embeddings = self.codon_embedding(codon_ids)  # [B, T, C]
        dataset_embeddings = self.dataset_embedding(dataset_ids)  # [B, D]

        dataset_embeddings = dataset_embeddings.unsqueeze(1).expand(B, T, -1)

        ff_input = torch.cat([codon_embeddings, dataset_embeddings], dim=-1)
        ff_input = ff_input * mask.unsqueeze(-1)

        b = self.bias_ff(ff_input).squeeze(-1)
        b = b * mask

        valid_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        # Center before clipping.
        b_mean = b.sum(dim=1, keepdim=True) / valid_lengths
        b = (b - b_mean) * mask


        exp_b = torch.exp(b) * mask

        # Zero-centered b does not imply mean(exp(b)) = 1.
        # This normalization keeps the multiplicative branch mostly redistributive.
        exp_b_mean = exp_b.sum(dim=1, keepdim=True) / valid_lengths
        exp_b = exp_b / exp_b_mean
        exp_b = exp_b * mask

        return exp_b, b

