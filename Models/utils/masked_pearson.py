from __future__ import annotations

import torch
import torch.nn as nn


class MaskedPearsonCorrelation(nn.Module):
    """
    Computes per-sample Pearson correlation over valid masked positions.

    Inputs:
        pred:   [B, T]
        target: [B, T]
        mask:   [B, T] bool or 0/1

    Returns:
        pcc_per_sample: [B]
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        mask_b = mask.bool()
        mask_f = mask_b.float()

        pred = pred * mask_f
        target = target * mask_f

        n = mask_f.sum(dim=1).clamp_min(1.0)

        pred_mean = pred.sum(dim=1) / n
        target_mean = target.sum(dim=1) / n

        pred_centered = (pred - pred_mean.unsqueeze(1)) * mask_f
        target_centered = (target - target_mean.unsqueeze(1)) * mask_f

        cov = (pred_centered * target_centered).sum(dim=1)
        pred_var = (pred_centered ** 2).sum(dim=1)
        target_var = (target_centered ** 2).sum(dim=1)

        denom = torch.sqrt(pred_var * target_var).clamp_min(self.eps)
        pcc = cov / denom

        invalid = (
            (mask_f.sum(dim=1) < 2)
            | (pred_var <= self.eps)
            | (target_var <= self.eps)
        )

        return torch.where(invalid, torch.zeros_like(pcc), pcc)