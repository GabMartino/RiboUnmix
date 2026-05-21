from __future__ import annotations


import torch
import torch.nn as nn

from Models.RiboQueuingModel.submodels.DatasetDispersionHead import DatasetDispersionHead
from Models.RiboQueuingModel.submodels.DatasetMultiplicativeBiasHead import DatasetMultiplicativeBiasHead
from Models.RiboQueuingModel.submodels.DatasetAdditiveBiasHead import  DatasetAdditiveBiasHead
from Models.RiboQueuingModel.submodels.DatasetPositionTweediePowerHead import DatasetPositionTweediePowerHead


class DatasetBiasSubmodel(nn.Module):

    def __init__(self, config_params: dict):
        super().__init__()

        self.position_features = list(config_params["position_features"])
        self.position_dim = len(self.position_features)
        self.codon_embeddings_size = config_params["codon_embeddings_size"]
        self.dataset_embeddings_size = config_params["dataset_embeddings_size"]
        self.num_filters = config_params["num_filters"]
        self.kernel_size = config_params["kernel_size"]
        self.num_datasets = config_params["num_datasets"]
        self.num_codons = config_params["num_codons"]


        self.dataset_embedding = nn.Embedding(self.num_datasets, self.dataset_embeddings_size)
        self.codon_embedding = nn.Embedding(self.num_codons , self.codon_embeddings_size)

        self.local_context_cnn = nn.Conv1d(
            in_channels=self.codon_embeddings_size,
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,  # 'same' padding
            bias=False
        )
        self.bias_input_size = (
                self.num_filters
                + self.position_dim
                + self.dataset_embeddings_size
        )
        self.dataset_multiplicative_bias_head = DatasetMultiplicativeBiasHead(
            config_params=config_params["dataset_multiplicative_bias_submodule_params"],
            input_size=self.bias_input_size,
        )

        self.additive_bias_head = DatasetAdditiveBiasHead(
            config_params=config_params["dataset_additive_bias_submodule_params"],
            input_size=self.bias_input_size,
        )

        self.dispersion_head = DatasetDispersionHead(
            config_params=config_params["dataset_dispersion_submodule_params"],
            input_size=self.bias_input_size,
        )

        self.tweedie_power_head = DatasetPositionTweediePowerHead(
            config_params=config_params["dataset_tweedie_power_params"],
            input_size=self.bias_input_size,
            num_datasets=self.num_datasets
        )


    def forward(
            self,
            dataset_ids: torch.Tensor,
            mask: torch.Tensor,
            codon_ids: torch.Tensor,
            position_features: torch.Tensor,
    ):
        '''

            Dataset bias inputs
        '''
        B, T = codon_ids.shape
        dataset_emb = self.dataset_embedding(dataset_ids).unsqueeze(1).expand(B, T, -1)
        codon_emb = self.codon_embedding(codon_ids)  # [B, T, C_emb]
        codon_emb_swapped = codon_emb.transpose(1, 2)
        local_context = self.local_context_cnn(codon_emb_swapped)  # [B, num_filters, T]
        local_context = local_context.transpose(1, 2)  # Back to [B, T, num_filters]

        x = torch.cat([dataset_emb, local_context, position_features], dim=-1)
        x = x * mask.unsqueeze(-1)


        phi = self.dispersion_head(x, mask)

        additive_noise, R_shape, lambda_bg, r_logits, lambda_raw = self.additive_bias_head(x, mask)
        b, log_b, beta_centered = self.dataset_multiplicative_bias_head(x, mask)
        p, p_extras = self.tweedie_power_head(dataset_ids=dataset_ids, x=x, mask=mask)



        return b, log_b, beta_centered, additive_noise, R_shape, lambda_bg, r_logits, lambda_raw, phi, p, p_extras
