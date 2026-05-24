from __future__ import annotations

import torch
import torch.nn as nn


class DatasetPositionTweediePowerHead(nn.Module):
    """
    Predicts a pure, scalar Tweedie power p_{b} per dataset.

    The local sequence-based component has been completely removed to prevent
    the optimizer from 'variance hacking' the NLL loss at the codon level.
    """

    def __init__(self, config_params: dict, input_size: int, num_datasets: int) -> None:
        super().__init__()

        self.num_datasets = num_datasets
        self.p_min = float(config_params.get("p_min", 1.1))
        self.p_max = float(config_params.get("p_max", 1.9))

        # raw = 0 gives p halfway between p_min and p_max.
        self.global_raw = nn.Parameter(torch.zeros(()))

        self.dataset_delta = nn.Embedding(self.num_datasets, 1)
        nn.init.zeros_(self.dataset_delta.weight)

    def forward(
            self,
            dataset_ids: torch.Tensor,
            x: torch.Tensor,
            mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask_b = mask.bool()

        # 1. Compute the dataset-level scalar raw score
        dataset_raw = self.dataset_delta(dataset_ids).squeeze(-1).unsqueeze(1)
        raw_p = self.global_raw + dataset_raw

        # 2. Map to the bounded [p_min, p_max] range
        p_val = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(raw_p)

        # 3. Expand the scalar to match the sequence length [B, T]
        # so the downstream loss functions receive the correct tensor shape.
        B, T = mask.shape
        p = p_val.expand(B, T)

        # 4. Mask neutral padding positions
        neutral_p = torch.full_like(p, 0.5 * (self.p_min + self.p_max))
        p = torch.where(mask_b, p, neutral_p)

        extras = {
            "tweedie_p": p,
            "tweedie_p_raw": raw_p,
            "tweedie_p_dataset_raw": dataset_raw.squeeze(1),
        }

        return p, extras