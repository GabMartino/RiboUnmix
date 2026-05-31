from __future__ import annotations

import torch
import torch.nn as nn

from Models.RiboQueuingModel.submodels.DatasetDispersionHead import DatasetDispersionHead
from Models.RiboQueuingModel.submodels.DatasetMultiplicativeAllocationBiasHead import (
    DatasetMultiplicativeAllocationBiasHead,
)


class DilatedContextCNN(nn.Module):
    """
    3-layer dilated temporal CNN for local codon-context features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length padding.")

        pad1 = 1 * (kernel_size - 1) // 2
        pad2 = 2 * (kernel_size - 1) // 2
        pad3 = 4 * (kernel_size - 1) // 2

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=pad1,
                dilation=1,
                padding_mode="replicate",
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=pad2,
                dilation=2,
                padding_mode="replicate",
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=pad3,
                dilation=4,
                padding_mode="replicate",
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DatasetBiasSubmodel(nn.Module):
    """
    Dataset/protocol submodel for multiplicative allocation bias.

    It predicts dataset-specific observation bias:

        b_raw_{d,t,i}

    The outer model applies:

        b_eff_i = b_raw_i / sum_j w_bio_j b_raw_j

        w_obs_i = w_bio_i * b_eff_i

    so that:

        sum_i w_obs_i = 1

    Inputs per position:

        dataset embedding
        local codon-context CNN features
        position features
        detached global biological context h_n

    The detached biological context lets the protocol-bias branch know the
    transcript-level state without sending gradients back into the biological RNN.
    """

    def __init__(
        self,
        config_params: dict,
        biological_context_size: int = 0,
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

        self.use_biological_context = bool(
            config_params.get("use_biological_context", biological_context_size > 0)
        )

        self.biological_context_size = (
            int(biological_context_size) if self.use_biological_context else 0
        )

        if self.biological_context_size < 0:
            raise ValueError(
                f"biological_context_size must be >= 0. "
                f"Got {self.biological_context_size}."
            )

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
        )

        if self.biological_context_size > 0:
            self.biological_context_norm = nn.LayerNorm(self.biological_context_size)
        else:
            self.biological_context_norm = None

        head_input_size = (
            self.dataset_embeddings_size
            + self.num_filters
            + self.position_dim
            + self.biological_context_size
        )

        head_cfg = config_params.get(
            "dataset_multiplicative_allocation_bias_submodule_params",
            None,
        )

        if head_cfg is None:
            raise KeyError(
                "Missing config key: "
                "model.dataset_bias_params."
                "dataset_multiplicative_allocation_bias_submodule_params"
            )

        self.multiplicative_allocation_bias_head = DatasetMultiplicativeAllocationBiasHead(
            config_params=head_cfg,
            input_size=head_input_size,
        )

        self.dispersion_head = DatasetDispersionHead(
            config_params=config_params["dataset_dispersion_submodule_params"],
            input_size=head_input_size,
        )

    def _prepare_biological_context(
        self,
        *,
        biological_context: torch.Tensor | None,
        B: int,
        T: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.biological_context_size <= 0:
            return None

        if biological_context is None:
            raise ValueError(
                "DatasetBiasSubmodel was configured with "
                f"biological_context_size={self.biological_context_size}, "
                "but biological_context=None was provided."
            )

        ctx = biological_context.detach()

        # Accept raw GRU h_n:
        #   [num_layers * num_directions, B, H]
        # and flatten to:
        #   [B, num_layers * num_directions * H]
        if ctx.ndim == 3:
            if ctx.shape[1] != B:
                raise ValueError(
                    "Expected biological_context h_n with shape "
                    f"[layers*directions, B, H], got {tuple(ctx.shape)} "
                    f"for B={B}."
                )

            ctx = ctx.permute(1, 0, 2).reshape(B, -1)

        # Also accept already flattened context:
        #   [B, C]
        elif ctx.ndim == 2:
            if ctx.shape[0] != B:
                raise ValueError(
                    "Expected biological_context with shape [B, C], "
                    f"got {tuple(ctx.shape)} for B={B}."
                )

        else:
            raise ValueError(
                "biological_context must have shape [B, C] or "
                f"[layers*directions, B, H], got {tuple(ctx.shape)}."
            )

        if ctx.shape[-1] != self.biological_context_size:
            raise ValueError(
                "biological_context last dimension mismatch. "
                f"Expected {self.biological_context_size}, got {ctx.shape[-1]}."
            )

        ctx = ctx.to(device=device, dtype=dtype)

        if self.biological_context_norm is not None:
            ctx = self.biological_context_norm(ctx)

        ctx = ctx.unsqueeze(1).expand(B, T, self.biological_context_size)

        return ctx

    def forward(
        self,
        dataset_ids: torch.Tensor,
        mask: torch.Tensor,
        codon_ids: torch.Tensor,
        position_features: torch.Tensor,
        biological_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, T = codon_ids.shape

        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=position_features.dtype)

        device = codon_ids.device
        dtype = position_features.dtype

        dataset_ids = dataset_ids.to(
            device=device,
            dtype=torch.long,
        )

        dataset_emb = self.dataset_embedding(dataset_ids)
        dataset_emb = dataset_emb.unsqueeze(1).expand(B, T, -1)

        codon_ids = codon_ids.to(
            device=device,
            dtype=torch.long,
        )

        codon_emb = self.codon_embedding(codon_ids)

        local_context = self.local_context_cnn(
            codon_emb.transpose(1, 2)
        ).transpose(1, 2)

        features = [
            dataset_emb,
            local_context,
            position_features,
        ]

        bio_context_features = self._prepare_biological_context(
            biological_context=biological_context,
            B=B,
            T=T,
            dtype=dtype,
            device=device,
        )

        if bio_context_features is not None:
            features.append(bio_context_features)

        x = torch.cat(features, dim=-1)
        x = x * mask_f.unsqueeze(-1)

        out = self.multiplicative_allocation_bias_head(
            x=x,
            mask=mask_b,
        )

        phi = self.dispersion_head(
            x,
            mask_b,
        )

        out["phi"] = phi

        return out