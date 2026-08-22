from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.RiboQueuingModel.submodels.DatasetLogSigmaHead import DatasetLogSigmaHead
from Models.RiboQueuingModel.submodels.DatasetMultiplicativeAllocationBiasHead import (
    DatasetMultiplicativeAllocationBiasHead,
)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class BiGRUContextEncoder(nn.Module):
    """
    Bidirectional GRU context encoder for the dataset-bias branch.

    It maps [B, C_in, T] -> [B, C_out, T]. This is a *separate* encoder from
    the biological BiGRU on purpose: gamma must stay an independent function
    of (sequence, dataset_id), so an XAI attribution on gamma resolves to the
    sequence directly and never flows through biology-derived features. That
    keeps `L_bio <- GRU_bio(seq)` and `gamma <- GRU_bias(seq, dataset)` cleanly
    separable at interpretation time.

    Padding is handled with packed sequences: a bidirectional GRU run on padded
    input would let the backward pass integrate padding into the valid region.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.out_channels = 2 * self.hidden_size  # bidirectional

        self.rnn = nn.GRU(
            input_size=self.in_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.output_norm = ChannelLayerNorm(self.out_channels)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, C_in, T] -> GRU wants [B, T, C_in]
        seq = x.transpose(1, 2)
        B, T, _ = seq.shape

        if mask is not None:
            mask_b = mask.bool()
            lengths = mask_b.sum(dim=1).clamp_min(1).to("cpu")
            packed = nn.utils.rnn.pack_padded_sequence(
                seq, lengths, batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed, batch_first=True, total_length=T
            )
        else:
            out, _ = self.rnn(seq)

        out = out.transpose(1, 2)  # [B, C_out, T]
        out = self.output_norm(out)
        if mask is not None:
            out = out * mask.bool().to(dtype=out.dtype).unsqueeze(1)
        return out


class DatasetBiasSubmodel(nn.Module):

    def __init__(
        self,
        config_params: dict,
    ):
        super().__init__()

        self.position_features = list(config_params["position_features"])
        self.position_dim = len(self.position_features)

        self.codon_embeddings_size = int(config_params["codon_embeddings_size"])
        self.use_nucleotide_amino_acid_embeddings = bool(
            config_params.get("use_nucleotide_amino_acid_embeddings", False)
        )
        self.dataset_embeddings_size = int(config_params["dataset_embeddings_size"])
        self.num_datasets = int(config_params["num_datasets"])
        self.num_codons = int(config_params["num_codons"])
        self.additional_sequence_feature_dim = int(
            config_params.get("additional_sequence_feature_dim", 0)
        )
        self.dataset_embedding = nn.Embedding(
            self.num_datasets,
            self.dataset_embeddings_size,
        )

        self.codon_embedding = nn.Embedding(
            self.num_codons,
            self.codon_embeddings_size,
        )

        nucleotide_feature_dim = 0
        amino_acid_feature_dim = 0
        if self.use_nucleotide_amino_acid_embeddings:
            nucleotide_embeddings_size = int(config_params["nucleotide_embeddings_size"])
            amino_acid_embeddings_size = int(config_params["amino_acid_embeddings_size"])
            num_nucleotides = int(config_params["num_nucleotides"])
            num_amino_acids = int(config_params["num_amino_acids"])
            codon_nucleotide_ids = torch.as_tensor(
                config_params["codon_nucleotide_ids"], dtype=torch.long
            )
            codon_amino_acid_ids = torch.as_tensor(
                config_params["codon_amino_acid_ids"], dtype=torch.long
            )
            self.nucleotide_embedding = nn.Embedding(
                num_nucleotides,
                nucleotide_embeddings_size,
            )
            self.amino_acid_embedding = nn.Embedding(
                num_amino_acids,
                amino_acid_embeddings_size,
            )
            self.register_buffer("codon_nucleotide_ids", codon_nucleotide_ids)
            self.register_buffer("codon_amino_acid_ids", codon_amino_acid_ids)
            nucleotide_feature_dim = 3 * nucleotide_embeddings_size
            amino_acid_feature_dim = amino_acid_embeddings_size

        # This branch deliberately owns a GRU separate from the biological GRU,
        # preserving gamma as a function of only (sequence, dataset_id).
        encoder_in = (
            self.codon_embeddings_size
            + nucleotide_feature_dim
            + amino_acid_feature_dim
            + self.dataset_embeddings_size
            + self.additional_sequence_feature_dim
        )
        self.local_context_gru = BiGRUContextEncoder(
            in_channels=encoder_in,
            hidden_size=int(config_params.get("context_gru_hidden_size", 128)),
            num_layers=int(config_params.get("context_gru_num_layers", 2)),
            dropout=float(config_params.get("context_gru_dropout", 0.0)),
        )
        self.context_dim = self.local_context_gru.out_channels

        head_input_size = (
            self.dataset_embeddings_size
            + self.context_dim
            + self.position_dim
        )
        self.observation_bias_head = DatasetMultiplicativeAllocationBiasHead(
            config_params=config_params.get(
            "dataset_multiplicative_allocation_bias_submodule_params"),
            input_size=head_input_size,
        )

        log_sigma_cfg = dict(config_params["dataset_log_sigma_submodule_params"])
        self.log_sigma_head = DatasetLogSigmaHead(config_params=log_sigma_cfg, input_size=head_input_size)

    def forward(
        self,
        dataset_ids: torch.Tensor,
        mask: torch.Tensor,
        codon_ids: torch.Tensor,
        position_features: torch.Tensor,
        sequence_features: torch.Tensor | None = None,
        compute_log_sigma: bool = True,
        embedding_center_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, T = codon_ids.shape

        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=position_features.dtype)

        device = codon_ids.device
        dtype = position_features.dtype

        dataset_ids = dataset_ids.to(device=device, dtype=torch.long)
        dataset_weight = self.dataset_embedding.weight
        if embedding_center_ids is None:
            embedding_center = dataset_weight.mean(dim=0, keepdim=True)
        else:
            embedding_center_ids = embedding_center_ids.to(
                device=dataset_weight.device,
                dtype=torch.long,
            ).reshape(-1)
            if embedding_center_ids.numel() == 0:
                raise ValueError("embedding_center_ids cannot be empty when provided.")
            if torch.unique(embedding_center_ids).numel() != embedding_center_ids.numel():
                raise ValueError("embedding_center_ids must contain distinct IDs.")
            embedding_center = dataset_weight.index_select(
                0, embedding_center_ids
            ).mean(dim=0, keepdim=True)
        dataset_weight = dataset_weight - embedding_center
        dataset_emb = F.embedding(dataset_ids, dataset_weight).to(dtype=dtype)
        dataset_emb = dataset_emb.unsqueeze(1).expand(B, T, -1)

        codon_ids = codon_ids.to(device=device, dtype=torch.long)
        codon_emb = self.codon_embedding(codon_ids).to(dtype=dtype)
        codon_emb = codon_emb * mask_f.unsqueeze(-1)

        encoder_features = [dataset_emb, codon_emb]
        if self.use_nucleotide_amino_acid_embeddings:
            nucleotide_ids = self.codon_nucleotide_ids[codon_ids]
            nucleotide_emb = self.nucleotide_embedding(nucleotide_ids).reshape(B, T, -1)
            amino_acid_ids = self.codon_amino_acid_ids[codon_ids]
            amino_acid_emb = self.amino_acid_embedding(amino_acid_ids)
            encoder_features.extend(
                [
                    nucleotide_emb.to(dtype=dtype) * mask_f.unsqueeze(-1),
                    amino_acid_emb.to(dtype=dtype) * mask_f.unsqueeze(-1),
                ]
            )
        if self.additional_sequence_feature_dim > 0:
            if sequence_features is None:
                raise ValueError(
                    "The dataset-bias branch expects additional sequence features "
                    f"with dimension {self.additional_sequence_feature_dim}, but none "
                    "were provided."
                )
            if sequence_features.shape != (
                B,
                T,
                self.additional_sequence_feature_dim,
            ):
                raise ValueError(
                    "Unexpected dataset-bias sequence feature shape: got "
                    f"{tuple(sequence_features.shape)}, expected "
                    f"{(B, T, self.additional_sequence_feature_dim)}."
                )
            sequence_features = sequence_features.to(device=device, dtype=dtype)
            sequence_features = sequence_features * mask_f.unsqueeze(-1)
            encoder_features.append(sequence_features)
        encoder_input = torch.cat(encoder_features, dim=-1).transpose(1, 2)
        local_context = self.local_context_gru(
            encoder_input,
            mask=mask_b,
        ).transpose(1, 2)
        features = [dataset_emb, local_context, position_features]


        x = torch.cat(features, dim=-1)
        x = x * mask_f.unsqueeze(-1)

        out = self.observation_bias_head(
            x=x,
            mask=mask_b,
        )

        if compute_log_sigma:
            # Dispersion sees the exact same dataset-conditioned context values,
            # but its NB2 gradients stop here. The alpha-specific head remains
            # fully trainable; gamma keeps the attached `x` path above.
            log_sigma_out = self.log_sigma_head(
                x=x.detach(),
                mask=mask_b,
            )
            log_sigma = log_sigma_out["log_sigma"]  # [B, T]
            log_sigma = torch.where(mask_b, log_sigma, torch.zeros_like(log_sigma))
            out["log_sigma"] = log_sigma
            log_sigma_t = (log_sigma * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)
            out["log_sigma_t"] = log_sigma_t  # [B, 1] — masked mean log_sigma

        return out
