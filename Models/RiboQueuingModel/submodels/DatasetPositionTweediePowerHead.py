from __future__ import annotations

import torch
import torch.nn as nn


class DatasetPositionTweediePowerHead(nn.Module):
    """
    Predicts per-position Tweedie power p_{b,i}.

    p is bounded:

        p = p_min + (p_max - p_min) * sigmoid(raw_p)

    raw_p has:
        global component
        dataset component
        local dataset/codon/position/context component

    The local component is centered over valid positions so it does not
    simply replace the dataset/global p.
    """

    def __init__(self, config_params: dict, input_size: int, num_datasets: int) -> None:
        super().__init__()

        self.num_datasets = num_datasets
        self.hidden_size = int(config_params["hidden_size"])
        self.dropout = float(config_params.get("dropout", 0.0))

        self.p_min = float(config_params.get("p_min", 1.1))
        self.p_max = float(config_params.get("p_max", 1.9))

        self.local_delta_max = float(config_params.get("local_delta_max", 0.25))

        # raw = 0 gives p halfway between p_min and p_max.
        self.global_raw = nn.Parameter(torch.zeros(()))

        self.dataset_delta = nn.Embedding(self.num_datasets, 1)
        nn.init.zeros_(self.dataset_delta.weight)

        self.local_ff = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        # Neutral local effect at start.
        nn.init.zeros_(self.local_ff[-1].weight)
        nn.init.zeros_(self.local_ff[-1].bias)

    def forward(
        self,
        dataset_ids: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=x.dtype)

        x = x * mask_f.unsqueeze(-1)

        local_raw = self.local_ff(x).squeeze(-1)
        local_raw = local_raw * mask_f

        local_delta = self.local_delta_max * torch.tanh(local_raw)
        local_delta = local_delta * mask_f

        # Center local delta over valid positions.
        valid_lengths = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        local_mean = local_delta.sum(dim=1, keepdim=True) / valid_lengths
        local_delta = (local_delta - local_mean) * mask_f

        dataset_raw = self.dataset_delta(dataset_ids).squeeze(-1).unsqueeze(1)

        raw_p = self.global_raw + dataset_raw + local_delta

        p = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(raw_p)

        neutral_p = torch.full_like(p, 0.5 * (self.p_min + self.p_max))
        p = torch.where(mask_b, p, neutral_p)

        extras = {
            "tweedie_p": p,
            "tweedie_p_raw": raw_p,
            "tweedie_p_dataset_raw": dataset_raw.squeeze(1),
            "tweedie_p_local_delta": local_delta,
        }

        return p, extras