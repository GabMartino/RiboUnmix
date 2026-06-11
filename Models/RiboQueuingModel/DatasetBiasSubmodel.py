from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.RiboQueuingModel.submodels.DatasetLogSigmaHead import DatasetLogSigmaHead
from Models.RiboQueuingModel.submodels.DatasetMultiplicativeAllocationBiasHead import (
    DatasetMultiplicativeAllocationBiasHead,
)
from Models.RiboQueuingModel.submodels.DatasetTranscriptScaleFactorHead import DatasetTranscriptScaleFactorHead


class ResidualDilatedConvBlock(nn.Module):
    """
    Pre-norm residual dilated convolution block.

    Args:
        x:
            Tensor with shape [B, C, T].
        mask:
            Optional valid-position mask with shape [B, T].
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
        gated: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length padding.")

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.gated = bool(gated)
        self.residual_scale = float(residual_scale)

        padding = self.dilation * (self.kernel_size - 1) // 2

        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.channels)

        conv_out_channels = 2 * self.channels if self.gated else self.channels

        self.conv = nn.Conv1d(
            in_channels=self.channels,
            out_channels=conv_out_channels,
            kernel_size=self.kernel_size,
            padding=padding,
            dilation=self.dilation,
            padding_mode="replicate",
            bias=True,
        )

        self.dropout = nn.Dropout(p=float(dropout))

        self.pointwise = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            bias=True,
        )

        # Identity-like residual initialization.
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x

        if mask is not None:
            mask_f = mask.bool().to(dtype=x.dtype).unsqueeze(1)
            x = x * mask_f

        z = self.norm(x)
        z = self.conv(z)

        if self.gated:
            value, gate = z.chunk(2, dim=1)
            z = value * torch.sigmoid(gate)
        else:
            z = F.gelu(z)

        z = self.dropout(z)
        z = self.pointwise(z)

        out = residual + self.residual_scale * z

        if mask is not None:
            out = out * mask.bool().to(dtype=out.dtype).unsqueeze(1)

        return out


class DilatedContextCNN(nn.Module):
    """
    Masked residual dilated TCN for local codon-context features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        dilations: Sequence[int] = (1, 2, 4, 8, 16),
        dropout: float = 0.0,
        gated: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length padding.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)

        self.input_proj = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=True,
        )

        self.blocks = nn.ModuleList(
            [
                ResidualDilatedConvBlock(
                    channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    dilation=d,
                    dropout=dropout,
                    gated=gated,
                    residual_scale=residual_scale,
                )
                for d in self.dilations
            ]
        )

        self.output_norm = nn.GroupNorm(num_groups=1, num_channels=self.out_channels)

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, C, T], got {tuple(x.shape)}.")

        if mask is not None:
            if mask.ndim != 2:
                raise ValueError(f"Expected mask [B, T], got {tuple(mask.shape)}.")
            if x.shape[0] != mask.shape[0] or x.shape[-1] != mask.shape[1]:
                raise ValueError(
                    f"x/mask mismatch: x={tuple(x.shape)}, mask={tuple(mask.shape)}."
                )

            x = x * mask.bool().to(dtype=x.dtype).unsqueeze(1)

        x = self.input_proj(x)

        if mask is not None:
            x = x * mask.bool().to(dtype=x.dtype).unsqueeze(1)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.output_norm(x)

        if mask is not None:
            x = x * mask.bool().to(dtype=x.dtype).unsqueeze(1)

        return x



class DatasetBiasSubmodel(nn.Module):

    def __init__(
        self,
        config_params: dict,
        biological_context_size: int,
    ):
        super().__init__()

        self.position_features = list(config_params["position_features"])
        self.position_dim = len(self.position_features)

        self.codon_embeddings_size = int(config_params["codon_embeddings_size"])
        self.dataset_embeddings_size = int(config_params["dataset_embeddings_size"])
        self.num_filters = int(config_params["num_filters"])
        self.kernel_size = int(config_params["kernel_size"])
        self.num_datasets = int(config_params["num_datasets"])
        self.num_codons = int(config_params["num_codons"])


        self.dataset_embedding = nn.Embedding(
            self.num_datasets,
            self.dataset_embeddings_size,
        )

        self.codon_embedding = nn.Embedding(
            self.num_codons,
            self.codon_embeddings_size,
        )

        self.use_local_context_cnn = bool(config_params.get("use_local_context_cnn", True))

        self.local_context_cnn = DilatedContextCNN(
            in_channels=self.codon_embeddings_size,
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            dilations=config_params.get("context_cnn_dilations", (1, 2, 4, 8, 16)),
            dropout=float(config_params.get("context_cnn_dropout", 0.0)),
            gated=bool(config_params.get("context_cnn_gated", True)),
            residual_scale=float(config_params.get("context_cnn_residual_scale", 1.0)),
        )

        head_input_size = (
            self.dataset_embeddings_size
            + (self.num_filters if self.use_local_context_cnn else self.codon_embeddings_size)
            + self.position_dim
        )


        self.observation_bias_head = DatasetMultiplicativeAllocationBiasHead(
            config_params=config_params.get(
            "dataset_multiplicative_allocation_bias_submodule_params"),
            input_size=head_input_size
        )

        log_sigma_cfg = dict(config_params["dataset_log_sigma_submodule_params"])
        log_sigma_cfg["num_datasets"] = self.num_datasets
        log_sigma_cfg["codon_input_size"] = self.codon_embeddings_size

        self.log_sigma_head = DatasetLogSigmaHead(config_params=log_sigma_cfg, input_size=head_input_size)

        scale_cfg = config_params.get("dataset_transcript_scale_factor_params", None)

        scale_cfg = dict(scale_cfg)
        scale_cfg["num_datasets"] = self.num_datasets
        scale_cfg["biological_context_size"] = int(biological_context_size)

        self.dataset_transcript_scale = DatasetTranscriptScaleFactorHead(
            config_params=scale_cfg,
        )


    def forward(
        self,
        dataset_ids: torch.Tensor,
        mask: torch.Tensor,
        codon_ids: torch.Tensor,
        position_features: torch.Tensor,
        biological_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, T = codon_ids.shape

        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=position_features.dtype)

        device = codon_ids.device
        dtype = position_features.dtype

        dataset_ids = dataset_ids.to(device=device, dtype=torch.long)
        dataset_weight = self.dataset_embedding.weight
        dataset_weight = dataset_weight - dataset_weight.mean(dim=0, keepdim=True)
        dataset_emb = F.embedding(dataset_ids, dataset_weight).to(dtype=dtype)
        dataset_emb = dataset_emb.unsqueeze(1).expand(B, T, -1)

        codon_ids = codon_ids.to(device=device, dtype=torch.long)
        codon_emb = self.codon_embedding(codon_ids).to(dtype=dtype)
        codon_emb = codon_emb * mask_f.unsqueeze(-1)

        features = [dataset_emb]
        if self.use_local_context_cnn:
            local_context = self.local_context_cnn(
                codon_emb.transpose(1, 2),
                mask=mask_b,
            ).transpose(1, 2)
            features.append(local_context)
        else:
            features.append(codon_emb)
        features.append(position_features)


        x = torch.cat(features, dim=-1)
        x = x * mask_f.unsqueeze(-1)

        out = self.observation_bias_head(
            x=x,
            mask=mask_b,
        )

        log_sigma_out = self.log_sigma_head(
            dataset_ids=dataset_ids,
            codon_embeddings=codon_emb,
            x = x,
            mask=mask_b,
        )
        log_sigma = log_sigma_out["log_sigma"]  # [B, T]
        log_sigma = torch.where(mask_b, log_sigma, torch.zeros_like(log_sigma))
        out["log_sigma"] = log_sigma
        log_sigma_t = (log_sigma * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        out["log_sigma_t"] = log_sigma_t  # [B, 1] — masked mean for log_sigma reg loss

        scale_out = self.dataset_transcript_scale(
            dataset_ids=dataset_ids,
            biological_context=biological_context,
        )
        out.update(scale_out)

        return out
