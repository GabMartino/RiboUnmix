from __future__ import annotations

import torch
from torch import nn


class DatasetMultiplicativeAllocationBiasHead(nn.Module):
    """Dataset-conditioned log-gamma residual head."""

    def __init__(
        self,
        config_params: dict,
        input_size: int,
    ) -> None:
        super().__init__()

        config_params = dict(config_params or {})

        self.hidden_size = int(config_params.get("hidden_size", 128))
        self.dropout = float(config_params.get("dropout", 0.0))

        self.shared = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
        )
        self.log_bias_head = nn.Linear(self.hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Neutral gamma residual; the outer model adds init_gamma.
        nn.init.zeros_(self.log_bias_head.weight)
        nn.init.zeros_(self.log_bias_head.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, T, F], got {tuple(x.shape)}.")

        if mask.ndim != 2:
            raise ValueError(f"Expected mask [B, T], got {tuple(mask.shape)}.")

        if x.shape[:2] != mask.shape:
            raise ValueError(
                f"x and mask shape mismatch: x={tuple(x.shape)}, "
                f"mask={tuple(mask.shape)}."
            )

        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=x.dtype)

        h = self.shared(x * mask_f.unsqueeze(-1))
        gamma_raw = self.log_bias_head(h).squeeze(-1) * mask_f

        gamma_raw_diag = torch.where(
            mask_b,
            gamma_raw,
            torch.zeros_like(gamma_raw),
        )
        return {"gamma_raw": gamma_raw_diag}
