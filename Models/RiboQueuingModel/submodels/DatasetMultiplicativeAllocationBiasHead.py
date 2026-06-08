from __future__ import annotations

import torch
from torch import nn


class DatasetMultiplicativeAllocationBiasHead(nn.Module):
    """
    Anchor-free continuous dataset visibility head.

    Despite the legacy class name, this no longer owns a keep gate. It only
    predicts a bounded log visibility correction per position. The outer model
    applies the identifiability gauge by centering this correction under the
    biological profile before forming:

        p_visible_i proportional to p_bio_i * exp(log_visibility_bias_i)

    Dataset zeros are handled by the left-censored likelihood, not by this
    visibility head.
    """

    def __init__(
        self,
        config_params: dict,
        input_size: int,
    ) -> None:
        super().__init__()

        config_params = dict(config_params or {})

        self.hidden_size = int(config_params.get("hidden_size", 128))
        self.dropout = float(config_params.get("dropout", 0.0))
        self.log_bias_max = float(config_params.get("log_bias_max", 3.0))

        if self.log_bias_max <= 0.0:
            raise ValueError(f"log_bias_max must be > 0, got {self.log_bias_max}.")

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
        # Neutral visibility: exp(log_bias)=1 before outer centering.
        nn.init.zeros_(self.log_bias_head.weight)
        nn.init.zeros_(self.log_bias_head.bias)

    def _bound_log_bias(self, raw: torch.Tensor) -> torch.Tensor:
        return self.log_bias_max * torch.tanh(raw / self.log_bias_max)

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
        log_bias_raw = self._bound_log_bias(self.log_bias_head(h).squeeze(-1))
        log_bias_raw = log_bias_raw * mask_f

        log_bias_diag = torch.where(
            mask_b,
            log_bias_raw,
            torch.zeros_like(log_bias_raw),
        )
        beta_diag = torch.exp(log_bias_diag)

        ones = torch.ones_like(beta_diag)
        zeros = torch.zeros_like(beta_diag)

        return {
            # Main visibility outputs. The outer model performs p_bio-weighted
            # centering and normalization.
            "log_visibility_bias_raw": log_bias_diag,
            "log_visibility_bias": log_bias_diag,
            "obs_beta": beta_diag,
            "obs_bias_raw": beta_diag,
            "obs_bias_amp": beta_diag,
            "obs_bias_amp_logits": log_bias_diag,
            "obs_bias_geomean": torch.ones(
                (x.shape[0],),
                dtype=x.dtype,
                device=x.device,
            ),

            # Compatibility diagnostics for older analysis code; active
            # visibility no longer has a keep gate.
            "obs_bias_keep_prob": ones,
            "obs_bias_keep_gate": ones,
            "obs_bias_keep_gate_effective": ones,
            "obs_bias_keep_hard": ones,
            "obs_bias_gate_logits": zeros,
        }
