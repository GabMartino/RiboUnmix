from __future__ import annotations


import torch
import torch.nn as nn

from Models.RiboQueuingModel.submodels.DatasetMultiplicativeBiasHead import DatasetMultiplicativeBiasHead
from Models.RiboQueuingModel.submodels.DatasetShiftHead import DatasetShiftHead
from Models.RiboQueuingModel.submodels.DatasetAdditiveBiasHead import DatasetAdditiveBiasHead


class DatasetBiasSubmodel(nn.Module):

    def __init__(self, config_params: dict):
        super().__init__()

        self.dataset_shift_head = DatasetShiftHead(config_params=config_params["dataset_shift_submodule_params"])

        self.dataset_multiplicative_bias_head = DatasetMultiplicativeBiasHead(config_params=config_params["dataset_multiplicative_bias_submodule_params"])

        self.additive_bias_head = DatasetAdditiveBiasHead( config_params=config_params["dataset_additive_bias_submodule_params"])


    def forward(
            self,
            L_queue: torch.Tensor,
            dataset_ids: torch.Tensor,
            mask: torch.Tensor,
            codon_ids: torch.Tensor,
            S_mean: torch.Tensor,
        ):
        L_effective, shift_weights_used, shift_weights_soft = self.dataset_shift_head(L_queue=L_queue, dataset_ids=dataset_ids, mask_f=mask)

        exp_b, b = self.dataset_multiplicative_bias_head(dataset_ids= dataset_ids, codon_ids=codon_ids, mask=mask)

        additive_bias, beta_per_position = self.additive_bias_head( dataset_ids=dataset_ids, codon_ids=codon_ids, S_mean=S_mean, mask=mask)

        return L_effective, exp_b, additive_bias, (shift_weights_used, shift_weights_soft, b, beta_per_position)