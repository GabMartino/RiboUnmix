from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DatasetShiftHead(nn.Module):
    """
    Dataset-level coordinate-shift correction.

    Soft training:
        L_effective[i] = sum_k alpha[d,k] * L[i+k]

    Hard eval/predict:
        L_effective[i] = L[i+k*]
        where k* = argmax_k alpha[d,k]

    Positive shift k means:
        output position i uses biological L at i+k.
    """

    def __init__(self, config_params: dict):
        super().__init__()

        self.num_datasets = config_params["num_datasets"]
        self.shifts = config_params["shifts"]
        self.init_strength = config_params["init_strength"]
        self.temperature = config_params["temperature"]
        self.hard_eval = config_params["hard_eval"]
        self.straight_through_train = config_params["straight_through_train"]

        self.shift_logits = nn.Embedding(self.num_datasets, len(self.shifts))

        # Smaller init_strength makes it easier to move away from zero-shift.
        nn.init.constant_(self.shift_logits.weight, -float(self.init_strength))
        zero_idx = self.shifts.index(0)

        with torch.no_grad():
            self.shift_logits.weight[:, zero_idx] = float(self.init_strength)

    def _shift_tensor(self, L_queue: torch.Tensor, k: int) -> torch.Tensor:
        shifted = torch.zeros_like(L_queue)

        if k > 0:
            shifted[:, :-k] = L_queue[:, k:]
        elif k < 0:
            kk = -k
            shifted[:, kk:] = L_queue[:, :-kk]
        else:
            shifted = L_queue

        return shifted

    def forward(
        self,
        L_queue: torch.Tensor,
        dataset_ids: torch.Tensor,
        mask_f: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        L_effective:
            [B, T]
        shift_weights_used:
            [B, K], hard or soft depending on mode
        shift_weights_soft:
            [B, K], always differentiable softmax weights
        """
        temp = max(float(self.temperature), 1e-6)

        logits = self.shift_logits(dataset_ids)
        shift_weights_soft = torch.softmax(logits / temp, dim=-1)

        use_hard = (not self.training and self.hard_eval) or (
            self.training and self.straight_through_train
        )

        if use_hard:
            hard_idx = shift_weights_soft.argmax(dim=-1)
            shift_weights_hard = F.one_hot(
                hard_idx,
                num_classes=len(self.shifts),
            ).to(dtype=shift_weights_soft.dtype)

            if self.training and self.straight_through_train:
                # Forward hard, backward soft.
                shift_weights_used = (
                    shift_weights_hard
                    + shift_weights_soft
                    - shift_weights_soft.detach()
                )
            else:
                # Pure hard eval/predict.
                shift_weights_used = shift_weights_hard
        else:
            shift_weights_used = shift_weights_soft

        L_effective = torch.zeros_like(L_queue)

        for j, k in enumerate(self.shifts):
            shifted = self._shift_tensor(L_queue, k)
            L_effective = L_effective + shift_weights_used[:, j].unsqueeze(1) * shifted

        L_effective = L_effective * mask_f

        return L_effective, shift_weights_used, shift_weights_soft