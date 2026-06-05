from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.RiboQueuingModel.submodels.DatasetDispersionHead import DatasetDispersionHead
from Models.RiboQueuingModel.submodels.DatasetDropoutHead import DatasetDropoutHead
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


class DatasetTranscriptScaleHead(nn.Module):
    """
    Predicts a positive dataset-transcript scale for absolute ZILN training.

    The returned scale is transcript-level, not position-level:

        S_{d,t} = exp(s0 + s_d + r_{d,t})

    It should be applied outside the queueing exponential by the outer model:

        mu_{d,t,i} = S_{d,t} * L_bio_{t,i} * b_raw_{d,t,i}

    This head may read the biological h_n context. By default the context is
    detached, so the absolute scale objective does not reshape the biological
    encoder directly.
    """

    def __init__(
        self,
        *,
        num_datasets: int,
        biological_context_size: int,
        config_params: dict,
    ):
        super().__init__()

        self.num_datasets = int(num_datasets)
        self.biological_context_size = int(biological_context_size)

        self.dataset_scale_embedding_size = int(
            config_params.get("dataset_scale_embedding_size", 16)
        )
        self.hidden_size = int(config_params.get("hidden_size", 128))
        self.dropout = float(config_params.get("dropout", 0.0))

        self.log_dataset_scale_max = float(
            config_params.get("log_dataset_scale_max", 5.0)
        )
        self.log_transcript_scale_max = float(
            config_params.get("log_transcript_scale_max", 5.0)
        )

        self.detach_biological_context = bool(
            config_params.get("detach_biological_context", True)
        )

        self.global_log_scale = nn.Parameter(
            torch.tensor(
                float(config_params.get("init_global_log_scale", 0.0)),
                dtype=torch.float32,
            )
        )

        self.dataset_log_scale = nn.Embedding(self.num_datasets, 1)
        nn.init.zeros_(self.dataset_log_scale.weight)

        self.dataset_embedding = nn.Embedding(
            self.num_datasets,
            self.dataset_scale_embedding_size,
        )

        if self.biological_context_size > 0:
            self.context_norm = nn.LayerNorm(self.biological_context_size)
        else:
            self.context_norm = None

        input_size = self.dataset_scale_embedding_size + self.biological_context_size

        self.residual_head = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        final = self.residual_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @staticmethod
    def _bound_log_scale(raw: torch.Tensor, max_abs: float) -> torch.Tensor:
        max_abs = float(max(max_abs, 1.0e-6))
        return max_abs * torch.tanh(raw / max_abs)

    def _prepare_context(
        self,
        *,
        biological_context: torch.Tensor | None,
        B: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.biological_context_size <= 0:
            return None

        if biological_context is None:
            raise ValueError(
                "DatasetTranscriptScaleHead requires biological_context, but received None."
            )

        ctx = biological_context

        if self.detach_biological_context:
            ctx = ctx.detach()

        if ctx.ndim == 3:
            if ctx.shape[1] != B:
                raise ValueError(
                    "Expected biological_context h_n with shape "
                    f"[layers*directions, B, H], got {tuple(ctx.shape)} for B={B}."
                )
            ctx = ctx.permute(1, 0, 2).reshape(B, -1)
        elif ctx.ndim == 2:
            if ctx.shape[0] != B:
                raise ValueError(
                    f"Expected biological_context [B, C], got {tuple(ctx.shape)} for B={B}."
                )
        else:
            raise ValueError(
                "biological_context must have shape [B, C] or "
                f"[layers*directions, B, H], got {tuple(ctx.shape)}."
            )

        if ctx.shape[-1] != self.biological_context_size:
            raise ValueError(
                "biological_context dimension mismatch in scale head. "
                f"Expected {self.biological_context_size}, got {ctx.shape[-1]}."
            )

        ctx = ctx.to(device=device, dtype=dtype)

        if self.context_norm is not None:
            ctx = self.context_norm(ctx)

        return ctx

    def forward(
        self,
        *,
        dataset_ids: torch.Tensor,
        biological_context: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        B = int(dataset_ids.shape[0])
        device = dataset_ids.device

        dataset_ids = dataset_ids.to(device=device, dtype=torch.long)

        dataset_log_raw = self.dataset_log_scale(dataset_ids)
        dataset_log_scale = self._bound_log_scale(
            dataset_log_raw,
            self.log_dataset_scale_max,
        )

        dataset_emb = self.dataset_embedding(dataset_ids)

        ctx = self._prepare_context(
            biological_context=biological_context,
            B=B,
            dtype=dataset_emb.dtype,
            device=device,
        )

        if ctx is not None:
            x = torch.cat([dataset_emb, ctx], dim=-1)
        else:
            x = dataset_emb

        transcript_log_raw = self.residual_head(x)
        transcript_log_scale = self._bound_log_scale(
            transcript_log_raw,
            self.log_transcript_scale_max,
        )

        global_log_scale = self.global_log_scale.to(
            device=device,
            dtype=dataset_log_scale.dtype,
        ).reshape(1, 1)

        log_scale = global_log_scale + dataset_log_scale + transcript_log_scale
        scale = torch.exp(log_scale)

        return {
            "scale_dt": scale.to(dtype=dtype),
            "log_scale_dt": log_scale.to(dtype=dtype),
            "global_log_scale": global_log_scale.expand(B, 1).to(dtype=dtype),
            "dataset_log_scale": dataset_log_scale.to(dtype=dtype),
            "transcript_log_scale": transcript_log_scale.to(dtype=dtype),
            "dataset_scale": torch.exp(dataset_log_scale).to(dtype=dtype),
            "transcript_scale": torch.exp(transcript_log_scale).to(dtype=dtype),
        }


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
            + self.num_filters
            + self.position_dim
        )


        self.observation_bias_head = DatasetMultiplicativeAllocationBiasHead(
            config_params=config_params.get(
            "dataset_multiplicative_allocation_bias_submodule_params"),
            input_size=head_input_size
        )

        self.dispersion_head = DatasetDispersionHead(
            config_params=config_params["dataset_dispersion_submodule_params"],
            input_size=head_input_size,
        )

        self.dropout_head = DatasetDropoutHead(
            config_params=config_params.get("dataset_dropout_submodule_params", {}),
            input_size=head_input_size,
        )

        scale_cfg = config_params.get("dataset_transcript_scale_factor_params", None)

        scale_cfg = dict(scale_cfg)
        scale_cfg["num_datasets"] = self.num_datasets
        scale_cfg["biological_context_size"] = int(biological_context_size)

        self.dataset_transcript_scale = DatasetTranscriptScaleFactorHead(
            config_params=scale_cfg,
            input_size=head_input_size,
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

        local_context = self.local_context_cnn(
            codon_emb.transpose(1, 2),
            mask=mask_b,
        ).transpose(1, 2)

        features = [dataset_emb, local_context, position_features]


        x = torch.cat(features, dim=-1)
        x = x * mask_f.unsqueeze(-1)

        out = self.observation_bias_head(
            x=x,
            mask=mask_b,
        )

        phi = self.dispersion_head(x, mask_b)
        out["phi"] = phi

        dropout_out = self.dropout_head(x, mask_b)
        out.update(dropout_out)

        scale_out = self.dataset_transcript_scale(
            dataset_ids=dataset_ids,
            biological_context=biological_context,
        )
        out.update(scale_out)

        return out
