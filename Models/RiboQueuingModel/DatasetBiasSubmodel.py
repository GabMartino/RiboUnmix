from __future__ import annotations

import torch
import torch.nn as nn

from Models.RiboQueuingModel.submodels.DatasetDispersionHead import DatasetDispersionHead
from Models.RiboQueuingModel.submodels.DatasetMultiplicativeBiasHead import DatasetMultiplicativeBiasHead
from Models.RiboQueuingModel.submodels.DatasetAdditiveBiasHead import DatasetAdditiveBiasHead
from Models.RiboQueuingModel.submodels.DatasetPositionTweediePowerHead import DatasetPositionTweediePowerHead


class DilatedContextCNN(nn.Module):
    """
    3-layer dilated temporal CNN for local codon-context features.

    For kernel_size=5:
        layer 1 dilation=1 -> local window
        layer 2 dilation=2 -> wider window
        layer 3 dilation=4 -> wider context

    Output shape:
        input:  [B, C_in, T]
        output: [B, C_out, T]
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

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=pad1,
            dilation=1,
            padding_mode="replicate",
            bias=False,
        )
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=pad2,
            dilation=2,
            padding_mode="replicate",
            bias=False,
        )
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=pad3,
            dilation=4,
            padding_mode="replicate",
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.conv3(x)
        return x


class DatasetBiasSubmodel(nn.Module):
    """
    Dataset/protocol observation submodel.

    Shared observation features:

        x = [
            dataset_embedding,
            local_context_cnn(codon_embedding),
            position_features,
        ]

    Heads:
        additive_bias_head:
            additive observation/background component.

        dataset_multiplicative_bias_head:
            smooth multiplicative correction plus optional keep gate.

        dispersion_head:
            currently used as profile-level uncertainty/kappa input downstream.

        tweedie_power_head:
            legacy auxiliary output if still required by the outer model.
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
            kernel_size=self.kernel_size,
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

        # ============================================================
        # 1. Shared observation features
        # ============================================================
        dataset_emb = self.dataset_embedding(dataset_ids)
        dataset_emb = dataset_emb.unsqueeze(1).expand(B, T, -1)

        codon_emb = self.codon_embedding(codon_ids)

        local_context = self.local_context_cnn(
            codon_emb.transpose(1, 2)
        ).transpose(1, 2)

        x = torch.cat(
            [
                dataset_emb,
                local_context,
                position_features,
            ],
            dim=-1,
        )

        x = x * mask_f.unsqueeze(-1)

        # ============================================================
        # 2. Additive/background branch
        # ============================================================
        (
            additive_rel, R_shape, lambda_frac, r_logits, lambda_raw
        ) = self.additive_bias_head(
            x,
            mask_b,
        )

        # ============================================================
        # 3. Multiplicative + gate branch
        # ============================================================
        (
            b_total,
            log_b,
            beta_centered,
            b_smooth,
            keep_gate,
            keep_prob,
            keep_hard,
            gate_logits,
        ) = self.dataset_multiplicative_bias_head(
            x,
            mask_b,
        )

        # ============================================================
        # 4. Profile uncertainty / kappa input
        # ============================================================
        phi = self.dispersion_head(
            x,
            mask_b,
        )


        # ============================================================
        # 6. Return
        # ============================================================
        return (
            b_total,
            log_b,
            beta_centered,
            b_smooth,
            keep_gate,
            keep_prob,
            keep_hard,
            gate_logits,
            additive_rel,
            R_shape,
            lambda_frac,
            r_logits,
            lambda_raw,
            phi
        )