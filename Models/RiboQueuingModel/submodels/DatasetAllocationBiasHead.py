from __future__ import annotations

import torch
from torch import nn


class DatasetAllocationBiasHead(nn.Module):
    """
    Dataset/protocol allocation-logit bias head.

    This replaces post-L multiplicative b and the gate.

    It predicts a bounded, centered logit perturbation:

        beta_raw_i = beta_max * tanh(r_i / beta_temperature)

        beta_i = beta_raw_i - mean_valid(beta_raw)

    Then the observed allocation is computed in the outer model as:

        w_obs = entmax(a_bio + beta)

    where a_bio are the biological allocation logits.

    Centering beta fixes the translation gauge because entmax/softmax are
    invariant to adding a constant to all logits.
    """

    def __init__(self, config_params: dict, input_size: int) -> None:
        super().__init__()

        self.hidden_size = int(config_params["hidden_size"])
        self.dropout = float(config_params.get("dropout", 0.0))

        self.beta_max = float(config_params.get("beta_max", 2.0))
        self.beta_temperature = max(
            float(config_params.get("beta_temperature", 3.0)),
            1e-6,
        )

        self.net = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        # Neutral initialization:
        # beta_logits = 0 -> beta_raw = 0 -> beta = 0.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, T, F], got {tuple(x.shape)}")

        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=x.dtype)

        x = x * mask_f.unsqueeze(-1)

        beta_logits = self.net(x).squeeze(-1)
        beta_logits = beta_logits * mask_f

        beta_raw = torch.sigmoid(
            beta_logits / self.beta_temperature
        )
        beta_raw = beta_raw * mask_f

        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        #beta_mean = beta_raw.sum(dim=1, keepdim=True) / denom
        ##TODO: eliminate
        beta = beta_raw / denom #(beta_raw - beta_mean) * mask_f

        # Padding is neutral in logit space.
        beta = torch.where(mask_b, beta, torch.zeros_like(beta))
        beta_raw = torch.where(mask_b, beta_raw, torch.zeros_like(beta_raw))
        beta_logits = torch.where(mask_b, beta_logits, torch.zeros_like(beta_logits))

        return {
            "allocation_logit_bias": beta,
            "allocation_logit_bias_raw": beta_raw,
            "allocation_logit_bias_logits": beta_logits,
            "allocation_logit_bias_mean": None,
        }