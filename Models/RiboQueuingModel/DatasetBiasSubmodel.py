from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from Models.RiboQueuingModel.submodels.DatasetShiftHead import DatasetShiftHead
from Models.RiboQueuingModel.submodels.DatasetAdditiveBiasHead import DatasetAdditiveBiasHead


class DatasetBiasSubmodel(nn.Module):
    """
    Dataset-specific technical bias model.

    It applies:

        1. Dataset-level coordinate shift:
              L_queue -> L_effective

        2. Dataset/codon-dependent multiplicative log-bias:
              multiplier_i = exp(b_i)

           where b_i is centered per transcript and exp(b_i) is mean-normalized.

        3. Dataset/codon-dependent additive residual:
              additive_bg_i = S_mean * additive_rel_i

           The additive branch has no fixed beta budget. Its use should be
           controlled by an explicit loss penalty on additive_rel.

    Final external usage:

        mu_base = S_mean * L_effective * multiplier
        mu = mu_base + additive_bg
    """

    def __init__(
        self,
        num_datasets: int,
        shifts: Sequence[int] = (-1, 0, 1),
        init_strength: float = 1.0,
        temperature: float = 0.25,
        hard_eval: bool = True,
        straight_through_train: bool = False,
        num_codons: int = 64,
        dataset_emb_dim: int = 16,
        codon_emb_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        b_clip: float = 1.0,
        additive_dataset_emb_dim: int = 16,
        additive_codon_emb_dim: int = 8,
        additive_hidden_dim: int = 32,
        additive_init_bias: float = -8.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.num_datasets = int(num_datasets)
        self.num_codons = int(num_codons)
        self.dataset_emb_dim = int(dataset_emb_dim)
        self.codon_emb_dim = int(codon_emb_dim)
        self.hidden_dim = int(hidden_dim)

        self.b_clip = float(b_clip)
        self.eps = float(eps)

        self.dataset_shift_head = DatasetShiftHead(
            num_datasets=self.num_datasets,
            shifts=shifts,
            init_strength=init_strength,
            temperature=temperature,
            hard_eval=hard_eval,
            straight_through_train=straight_through_train,
        )

        # Embeddings used by multiplicative b branch.
        self.dataset_embedding = nn.Embedding(
            self.num_datasets,
            self.dataset_emb_dim,
        )

        self.codon_embedding = nn.Embedding(
            self.num_codons,
            self.codon_emb_dim,
        )

        ff_in_dim = self.dataset_emb_dim + self.codon_emb_dim

        self.bias_ff = nn.Sequential(
            nn.Linear(ff_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Start with no multiplicative bias.
        nn.init.zeros_(self.bias_ff[-1].weight)
        nn.init.zeros_(self.bias_ff[-1].bias)

        # Additive residual branch.
        self.additive_bias_head = DatasetAdditiveBiasHead(
            num_datasets=self.num_datasets,
            num_codons=self.num_codons,
            dataset_emb_dim=additive_dataset_emb_dim,
            codon_emb_dim=additive_codon_emb_dim,
            hidden_dim=additive_hidden_dim,
            dropout=dropout,
            init_bias=additive_init_bias,
            eps=self.eps,
        )

    def _check_dataset_ids(
        self,
        dataset_ids: torch.Tensor,
    ) -> None:
        min_id = int(dataset_ids.min().detach().cpu())
        max_id = int(dataset_ids.max().detach().cpu())

        if min_id < 0 or max_id >= self.num_datasets:
            raise ValueError(
                f"dataset_ids out of range: min={min_id}, max={max_id}, "
                f"num_datasets={self.num_datasets}. Need num_datasets >= {max_id + 1}."
            )

    def compute_log_bias(
        self,
        codon_ids: torch.Tensor,
        dataset_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        codon_ids:
            [B, T] integer codon IDs in [0, num_codons - 1]

        dataset_ids:
            [B] integer dataset IDs

        mask:
            [B, T] bool

        Returns
        -------
        b:
            [B, T] centered log-bias

        multiplier:
            [B, T] mean-normalized exp(b)
        """
        if codon_ids.ndim != 2:
            raise ValueError(f"codon_ids must be [B, T], got {codon_ids.shape}.")

        B, T = codon_ids.shape
        device = codon_ids.device

        dataset_ids = dataset_ids.to(device=device, dtype=torch.long)
        codon_ids = codon_ids.to(device=device, dtype=torch.long)
        mask_b = mask.to(device=device, dtype=torch.bool)
        mask_f = mask_b.float()

        self._check_dataset_ids(dataset_ids)

        codon_ids = codon_ids.clamp(min=0, max=self.num_codons - 1)

        codon_emb = self.codon_embedding(codon_ids)  # [B, T, C]

        dataset_emb = self.dataset_embedding(dataset_ids)  # [B, D]
        dataset_emb = dataset_emb.unsqueeze(1).expand(B, T, -1)

        ff_input = torch.cat([codon_emb, dataset_emb], dim=-1)
        ff_input = ff_input * mask_f.unsqueeze(-1)

        b = self.bias_ff(ff_input).squeeze(-1)
        b = b * mask_f

        valid_lengths = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        # Center before clipping.
        b_mean = b.sum(dim=1, keepdim=True) / valid_lengths
        b = (b - b_mean) * mask_f

        # Clip to prevent multiplicative technical bias from dominating.
        if self.b_clip > 0.0:
            b = b.clamp(-self.b_clip, self.b_clip) * mask_f

            # Recenter after clipping because clipping breaks exact centering.
            b_mean = b.sum(dim=1, keepdim=True) / valid_lengths
            b = (b - b_mean) * mask_f

        multiplier = torch.exp(b) * mask_f

        # Zero-centered b does not imply mean(exp(b)) = 1.
        # This normalization keeps the multiplicative branch mostly redistributive.
        multiplier_mean = multiplier.sum(dim=1, keepdim=True) / valid_lengths
        multiplier = multiplier / multiplier_mean.clamp_min(self.eps)
        multiplier = multiplier * mask_f

        return b, multiplier

    def forward(
        self,
        L_queue: torch.Tensor,
        dataset_ids: torch.Tensor,
        mask: torch.Tensor,
        codon_ids: torch.Tensor,
        S_mean: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns
        -------
        L_effective:
            [B, T]

        b:
            [B, T] centered log-bias

        multiplier:
            [B, T] mean-normalized exp(b)

        additive_bg:
            [B, T] additive residual in target/profile units

        additive_rel:
            [B, T] additive_bg / S_mean

        shift_weights_used:
            [B, K]

        shift_weights_soft:
            [B, K]
        """
        mask_b = mask.bool()
        mask_f = mask_b.to(device=L_queue.device, dtype=L_queue.dtype)

        L_effective, shift_weights_used, shift_weights_soft = self.dataset_shift_head(
            L_queue=L_queue,
            dataset_ids=dataset_ids,
            mask_f=mask_f,
        )

        L_effective = L_effective.to(device=L_queue.device, dtype=L_queue.dtype)
        L_effective = L_effective.clamp_min(0.0) * mask_f

        b, multiplier = self.compute_log_bias(
            codon_ids=codon_ids,
            dataset_ids=dataset_ids,
            mask=mask_b,
        )

        additive_bg, additive_rel = self.additive_bias_head(
            dataset_ids=dataset_ids,
            codon_ids=codon_ids,
            S_mean=S_mean,
            mask=mask_b,
        )

        b = b.to(device=L_queue.device, dtype=L_queue.dtype)
        multiplier = multiplier.to(device=L_queue.device, dtype=L_queue.dtype)
        additive_bg = additive_bg.to(device=L_queue.device, dtype=L_queue.dtype)
        additive_rel = additive_rel.to(device=L_queue.device, dtype=L_queue.dtype)

        return (
            L_effective,
            b,
            multiplier,
            additive_bg,
            additive_rel,
            shift_weights_used,
            shift_weights_soft,
        )