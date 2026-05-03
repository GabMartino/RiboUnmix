from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

from Models.RiboQueuingModel.DatasetBiasSubmodel import DatasetBiasSubmodel
from Models.RiboQueuingModel.QueuingBiologicalModel import QueuingBiologicalModel
from Models.RiboQueuingModel.submodels.DatasetCodonDispersionHead import DatasetCodonDispersionHead
from Models.RiboQueuingModel.submodels.DatasetDispersionHead import DatasetDispersionHead
from Models.utils.compute_S_quantile import compute_S_mean


class RiboQueuingModel(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 2,
            dropout: float = 0.1,
            num_datasets: int = 32,
            eps: float = 1e-8,
            mu_max: float = 1e8,
            codon_feature_start: int = 12,
            num_codons: int = 64,
            dataset_emb_dim: int = 16,
            codon_emb_dim: int = 16,
            bias_hidden_dim: int = 64,
            b_clip: float = 1.0,
            additive_dataset_emb_dim: int = 16,
            additive_codon_emb_dim: int = 8,
            additive_hidden_dim: int = 32,
            additive_init_bias: float = -8.0,
            phi_min: float = 0.05,
            phi_max: float = 5.0,
            init_phi: float = 1.0,
            phi_dataset_emb_dim: int = 16,
            phi_codon_emb_dim: int = 8,
            phi_hidden_dim: int = 32,
    ):
        super().__init__()

        self.eps = float(eps)
        self.mu_max = float(mu_max)

        self.codon_feature_start = int(codon_feature_start)
        self.num_codons = int(num_codons)
        self.codon_feature_end = self.codon_feature_start + self.num_codons

        if input_size < self.codon_feature_end:
            raise ValueError(
                f"input_size={input_size} is too small for codon slice "
                f"[{self.codon_feature_start}:{self.codon_feature_end}]."
            )

        self.biological_model = QueuingBiologicalModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.dataset_bias_model = DatasetBiasSubmodel(
            num_datasets=num_datasets,
            shifts=(-1, 0, 1),
            num_codons=num_codons,
            dataset_emb_dim=dataset_emb_dim,
            codon_emb_dim=codon_emb_dim,
            hidden_dim=bias_hidden_dim,
            dropout=dropout,
            b_clip=b_clip,
            additive_dataset_emb_dim=additive_dataset_emb_dim,
            additive_codon_emb_dim=additive_codon_emb_dim,
            additive_hidden_dim=additive_hidden_dim,
            additive_init_bias=additive_init_bias,
            eps=eps,
        )
        self.dispersion_head = DatasetCodonDispersionHead(
            num_datasets=num_datasets,
            num_codons=num_codons,
            dataset_emb_dim=phi_dataset_emb_dim,
            codon_emb_dim=phi_codon_emb_dim,
            hidden_dim=phi_hidden_dim,
            dropout=dropout,
            phi_min=phi_min,
            phi_max=phi_max,
            init_phi=init_phi,
        )

        # raw=0 gives p=1.5.
        self.p_raw = nn.Parameter(torch.zeros(()))

    def tweedie_power(self) -> torch.Tensor:
        return 1.1 + 0.8 * torch.sigmoid(self.p_raw)

    def extract_codon_ids_from_packed(
        self,
        x_packed,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Extracts codon IDs from the padded input sequence.

        Assumes codon one-hot features are:

            x[..., codon_feature_start : codon_feature_start + num_codons]

        Returns:
            codon_ids [B, T]
        """
        x_padded, _ = pad_packed_sequence(x_packed, batch_first=True)

        x_padded = x_padded.to(device=device)

        if x_padded.size(1) != T:
            if x_padded.size(1) > T:
                x_padded = x_padded[:, :T, :]
            else:
                pad_T = T - x_padded.size(1)
                x_padded = torch.cat(
                    [
                        x_padded,
                        torch.zeros(
                            x_padded.size(0),
                            pad_T,
                            x_padded.size(2),
                            device=x_padded.device,
                            dtype=x_padded.dtype,
                        ),
                    ],
                    dim=1,
                )

        codon_onehot = x_padded[
            ...,
            self.codon_feature_start:self.codon_feature_end,
        ]

        codon_ids = codon_onehot.argmax(dim=-1).long()

        return codon_ids

    def forward(
        self,
        x_packed,
        id_datasets: torch.Tensor,
        y_raw_target: torch.Tensor,
    ):
        p = self.tweedie_power()

        L_queue, rho_diag, w_prob, J, mask = self.biological_model(x_packed)

        mask_b = mask.bool()
        mask_f = mask_b.to(device=L_queue.device, dtype=L_queue.dtype)

        S_mean = compute_S_mean(
            y_raw_target,
            mask_b,
            censor_threshold=0.0,
            use_censor_threshold=False,
            use_nonzero_only=False,
        )

        B, T = L_queue.shape

        S_mean = S_mean.reshape(B, 1).to(
            device=L_queue.device,
            dtype=L_queue.dtype,
        )

        codon_ids = self.extract_codon_ids_from_packed(
            x_packed=x_packed,
            T=T,
            device=L_queue.device,
        )

        (
            L_effective,
            b,
            multiplier,
            additive_bg,
            additive_rel,
            shift_weights_used,
            shift_weights_soft,
        ) = self.dataset_bias_model(
            L_queue=L_queue,
            dataset_ids=id_datasets,
            mask=mask_b,
            codon_ids=codon_ids,
            S_mean=S_mean,
        )

        L_effective = L_effective.clamp_min(0.0) * mask_f

        # Final mean:
        #   mu = L_effective * S_mean * exp(b)

        mu_base = S_mean * L_effective * multiplier
        mu = mu_base + additive_bg

        mu = torch.nan_to_num(
            mu,
            nan=self.eps,
            posinf=self.mu_max,
            neginf=self.eps,
        )

        mu = mu.clamp(min=self.eps, max=self.mu_max)
        mu = torch.where(mask_b, mu, torch.ones_like(mu))

        phi = self.dispersion_head(
            dataset_ids=id_datasets,
            codon_ids=codon_ids,
            mask=mask_b,
        )

        phi = phi.to(device=mu.device, dtype=mu.dtype)

        extras = (
            rho_diag,  # 0
            w_prob,  # 1
            L_queue,  # 2
            J,  # 3
            S_mean.squeeze(-1),  # 4
            L_effective,  # 5
            p,  # 6
            phi,  # 7
            b,  # 8
            multiplier,  # 9
            mu_base,  # 10
            additive_bg,  # 11
            additive_rel,  # 12
            codon_ids,  # 13
            shift_weights_used,  # 14
            shift_weights_soft,  # 15
        )

        return mu, p, phi, extras