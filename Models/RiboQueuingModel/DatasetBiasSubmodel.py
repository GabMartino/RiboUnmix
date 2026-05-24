from __future__ import annotations

import torch
import torch.nn as nn

from Models.RiboQueuingModel.submodels.DatasetDispersionHead import DatasetDispersionHead
from Models.RiboQueuingModel.submodels.DatasetMultiplicativeBiasHead import DatasetMultiplicativeBiasHead
from Models.RiboQueuingModel.submodels.DatasetAdditiveBiasHead import DatasetAdditiveBiasHead
from Models.RiboQueuingModel.submodels.DatasetPositionTweediePowerHead import DatasetPositionTweediePowerHead


class DilatedContextCNN(nn.Module):
    """
    Replaces a single Conv1d with a 3-layer Dilated Temporal Convolutional Network (TCN).
    Receptive field expands exponentially while maintaining position-specific precision.
    Uses replicate padding to prevent signal dilution at transcript edges.

    RF Calculation for kernel_size=5:
    Layer 1 (dilation=1): sees 5 codons
    Layer 2 (dilation=2): sees 13 codons
    Layer 3 (dilation=4): sees 29 codons
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding=2,  # padding = dilation * (kernel_size - 1) // 2
            dilation=1,
            padding_mode='replicate',
            bias=False
        )
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding=4,  # 2 * 2
            dilation=2,
            padding_mode='replicate',
            bias=False
        )
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding=8,  # 4 * 2
            dilation=4,
            padding_mode='replicate',
            bias=False
        )
        # We leave the final output linear (no GELU) so the downstream heads
        # can process negative and positive shifts evenly.

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.conv3(x)
        return x


class DatasetBiasSubmodel(nn.Module):
    """
    Dataset/protocol observation submodel.

    All observation heads receive the same dataset-conditioned local codon-context
    representation:

        x = [dataset_emb, local_context_cnn(codon_emb), position_features]

    This is less restrictive than splitting R/b/p/phi into separate feature streams.
    Leakage is controlled by:
        - bounded b via log_b_max
        - bounded additive branch via lambda_max
        - bounded/centered Tweedie p local_delta
        - diagnostics: R_vs_y, R_vs_L_queue, bg_frac, mu_bio_only vs mu_full
    """

    def __init__(self, config_params: dict):
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
        )

        self.bias_input_size = (
            self.dataset_embeddings_size
            + self.num_filters
            + self.position_dim
        )

        self.additive_bias_head = DatasetAdditiveBiasHead(
            config_params=config_params["dataset_additive_bias_submodule_params"],
            input_size=self.bias_input_size,
        )

        self.dataset_multiplicative_bias_head = DatasetMultiplicativeBiasHead(
            config_params=config_params["dataset_multiplicative_bias_submodule_params"],
            input_size=self.bias_input_size,
        )

        self.dispersion_head = DatasetDispersionHead(
            config_params=config_params["dataset_dispersion_submodule_params"],
            input_size=self.bias_input_size,
        )

        self.tweedie_power_head = DatasetPositionTweediePowerHead(
            config_params=config_params["dataset_tweedie_power_params"],
            input_size=self.bias_input_size,
            num_datasets=self.num_datasets,
        )

    def forward(
        self,
        dataset_ids: torch.Tensor,
        mask: torch.Tensor,
        codon_ids: torch.Tensor,
        position_features: torch.Tensor,
    ):
        B, T = codon_ids.shape

        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=position_features.dtype)

        dataset_emb = self.dataset_embedding(dataset_ids)
        dataset_emb = dataset_emb.unsqueeze(1).expand(B, T, -1)

        codon_emb = self.codon_embedding(codon_ids)

        codon_emb_swapped = codon_emb.transpose(1, 2)
        local_context = self.local_context_cnn(codon_emb_swapped)
        local_context = local_context.transpose(1, 2)

        x = torch.cat(
            [
                dataset_emb,
                local_context,
                position_features,
            ],
            dim=-1,
        )
        x = x * mask_f.unsqueeze(-1)

        additive_noise, R_shape, lambda_bg, r_logits, lambda_raw = self.additive_bias_head(
            x,
            mask_b,
        )

        b, control, beta_centered = self.dataset_multiplicative_bias_head(
            x,
            mask_b,
        )

        phi = self.dispersion_head(
            x,
            mask_b,
        )

        p, p_extras = self.tweedie_power_head(
            dataset_ids=dataset_ids,
            x=x,
            mask=mask_b,
        )

        return (
            b,
            control,
            beta_centered,
            additive_noise,
            R_shape,
            lambda_bg,
            r_logits,
            lambda_raw,
            phi,
            p,
            p_extras,
        )