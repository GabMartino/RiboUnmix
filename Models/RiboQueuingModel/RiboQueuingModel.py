from __future__ import annotations

import torch
import torch.nn as nn

from Models.RiboQueuingModel.DatasetBiasSubmodel import DatasetBiasSubmodel
from Models.RiboQueuingModel.QueuingBiologicalModel import QueuingBiologicalModel
from Models.RiboQueuingModel.submodels.DatasetDispersionHead import DatasetDispersionHead
from Models.utils.compute_S_mean import compute_S_mean



class RiboQueuingModel(nn.Module):
    def __init__(
            self,
            model_configs: dict,
            eps: float = 1e-8,
            mu_max: float = 1e8,
    ):
        super().__init__()

        self.eps = float(eps)
        self.mu_max = float(mu_max)
        '''
            Biological Model setup
        '''
        self.biological_model = QueuingBiologicalModel(config_params=model_configs["biological_params"])
        '''
            Dataset Bias Submodel setup
        '''
        self.dataset_bias_model = DatasetBiasSubmodel(config_params=model_configs["dataset_bias_params"])



        self.dispersion_head = DatasetDispersionHead(config_params=model_configs["dataset_bias_params"]["dataset_dispersion_submodule_params"])

        # raw=0 gives p=1.5.
        self.p_raw = nn.Parameter(torch.zeros(()))

    def tweedie_power(self) -> torch.Tensor:
        return 1.1 + 0.8 * torch.sigmoid(self.p_raw)


    def forward(
            self,
            x_packed,
            codon_ids: torch.Tensor,
            id_datasets: torch.Tensor,
            y_raw_target: torch.Tensor,
    ):

        p = self.tweedie_power()

        '''
            BIOLOGY
        '''
        L_queue, rho_diag, w_prob, J, mask = self.biological_model(x_packed)

        B, T = L_queue.shape

        '''
        
            DATASET BIAS
        '''
        mask_b = mask.bool()
        #mask_f = mask_b.to(device=L_queue.device, dtype=L_queue.dtype)

        S_mean = compute_S_mean( y_raw_target, mask_b, censor_threshold=0.0, use_censor_threshold=False, use_nonzero_only=False )

        L_effective, exp_b, additive_bias, _ = self.dataset_bias_model(L_queue, dataset_ids=id_datasets, codon_ids=codon_ids, mask=mask, S_mean=S_mean)


        '''
            FINAL OUTPUT
        '''
        mu_base = S_mean.reshape(B, 1) * L_effective * exp_b
        mu = mu_base + additive_bias

        mu = torch.nan_to_num(
            mu,
            nan=self.eps,
            posinf=self.mu_max,
            neginf=self.eps,
        )

        mu = mu.clamp(min=self.eps, max=self.mu_max)
        mu = torch.where(mask_b, mu, torch.ones_like(mu))

        '''
            PROBABILITY DISTRIBUTION DEPENDENT VARIANCE PARAM
        '''
        phi = self.dispersion_head(dataset_ids=id_datasets,codon_ids=codon_ids,  mask=mask_b)

        extras = {
            "L_queue": L_queue,
            "L_effective": L_effective,
            "mu_base": mu_base,
            "additive_bias": additive_bias,
            "exp_b": exp_b,
            "phi": phi,
            "S_mean": S_mean.squeeze(-1),
            "rho": rho_diag,
            "w_prob": w_prob,
            "J": J,
        }

        return mu, p, phi, extras