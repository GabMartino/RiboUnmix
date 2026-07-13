from __future__ import annotations

import math

import torch
from torch import nn


def inv_softplus(x: float) -> float:
    x = float(x)
    if x <= 0.0:
        raise ValueError(f"inv_softplus requires x > 0, got {x}.")
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


class DatasetMultiplicativeAllocationBiasHead(nn.Module):
    """
    Dataset-conditioned gated-additive mean head.

    The class name is kept for checkpoint/config compatibility, but the active
    outputs are now only:

        gamma_raw_i       -> unbounded log-amplitude residual in the outer model
        gamma_support_logits_i -> learned active/zero support score
        additive_bias_i   -> nonnegative additive background

    Dataset zeros are handled by the likelihood, not by this head.
    """

    def __init__(
        self,
        config_params: dict,
        input_size: int,
        additive_input_size: int | None = None,
    ) -> None:
        super().__init__()

        config_params = dict(config_params or {})

        self.hidden_size = int(config_params.get("hidden_size", 128))
        self.dropout = float(config_params.get("dropout", 0.0))
        self.additive_input_size = (
            None if additive_input_size is None else int(additive_input_size)
        )
        self.additive_bias_activation = str(
            config_params.get("additive_bias_activation", "softplus")
        ).lower()
        additive_activation_aliases = {
            "softplus": "softplus",
            "relu": "relu",
            "threshold": "relu",
            "thresholded": "relu",
        }
        if self.additive_bias_activation not in additive_activation_aliases:
            raise ValueError(
                "additive_bias_activation must be one of "
                f"{sorted(additive_activation_aliases)}, got "
                f"{self.additive_bias_activation!r}."
            )
        self.additive_bias_activation = additive_activation_aliases[
            self.additive_bias_activation
        ]
        self.init_additive_bias = float(
            config_params.get("init_additive_bias", 1.0e-4)
        )

        if self.additive_bias_activation == "softplus" and self.init_additive_bias <= 0.0:
            raise ValueError(
                "init_additive_bias must be > 0 when additive_bias_activation="
                "'softplus', got "
                f"{self.init_additive_bias}."
            )
        if self.additive_bias_activation == "relu" and self.init_additive_bias < 0.0:
            raise ValueError(
                "init_additive_bias must be >= 0 when additive_bias_activation="
                "'relu', got "
                f"{self.init_additive_bias}."
            )

        self.shared = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
        )
        self.additive_shared = None
        if self.additive_input_size is not None:
            self.additive_shared = nn.Sequential(
                nn.Linear(self.additive_input_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(p=self.dropout),
            )

        self.log_bias_head = nn.Linear(self.hidden_size, 1)
        self.support_logit_head = nn.Linear(self.hidden_size, 1)
        self.additive_bias_head = nn.Linear(self.hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Neutral gamma residual; the outer model adds init_gamma.
        nn.init.zeros_(self.log_bias_head.weight)
        nn.init.zeros_(self.log_bias_head.bias)
        # A constant support score gives a uniform entmax allocation, hence a
        # length-normalized support gate of exactly one at initialization.
        nn.init.zeros_(self.support_logit_head.weight)
        nn.init.zeros_(self.support_logit_head.bias)
        # Near-zero nonnegative additive background for the gated-additive mean.
        nn.init.zeros_(self.additive_bias_head.weight)
        if self.additive_bias_activation == "softplus":
            additive_bias_init = inv_softplus(self.init_additive_bias)
        elif self.additive_bias_activation == "relu":
            additive_bias_init = self.init_additive_bias
        else:
            raise RuntimeError(
                f"Unsupported additive_bias_activation {self.additive_bias_activation!r}."
            )
        nn.init.constant_(self.additive_bias_head.bias, additive_bias_init)

    def _activate_additive_bias(self, raw: torch.Tensor) -> torch.Tensor:
        if self.additive_bias_activation == "softplus":
            return torch.nn.functional.softplus(raw)
        if self.additive_bias_activation == "relu":
            return torch.relu(raw)
        raise RuntimeError(
            f"Unsupported additive_bias_activation {self.additive_bias_activation!r}."
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        additive_x: torch.Tensor | None = None,
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
        if additive_x is not None:
            if self.additive_shared is None:
                raise ValueError(
                    "additive_x was provided, but this head was constructed "
                    "without additive_input_size."
                )
            if additive_x.ndim != 3 or additive_x.shape[:2] != mask.shape:
                raise ValueError(
                    "Expected additive_x [B, T, F_add] matching mask, got "
                    f"{tuple(additive_x.shape)}."
                )
            h_additive = self.additive_shared(
                additive_x.to(device=x.device, dtype=x.dtype) * mask_f.unsqueeze(-1)
            )
        else:
            h_additive = h
        gamma_raw = self.log_bias_head(h).squeeze(-1) * mask_f
        gamma_support_logits = self.support_logit_head(h).squeeze(-1) * mask_f
        additive_bias_raw = self.additive_bias_head(h_additive).squeeze(-1)
        additive_bias = self._activate_additive_bias(additive_bias_raw) * mask_f

        gamma_raw_diag = torch.where(
            mask_b,
            gamma_raw,
            torch.zeros_like(gamma_raw),
        )
        additive_bias_raw_diag = torch.where(
            mask_b,
            additive_bias_raw,
            torch.zeros_like(additive_bias_raw),
        )
        additive_bias_diag = torch.where(
            mask_b,
            additive_bias,
            torch.zeros_like(additive_bias),
        )

        return {
            "gamma_raw": gamma_raw_diag,
            "gamma_support_logits": torch.where(
                mask_b,
                gamma_support_logits,
                torch.zeros_like(gamma_support_logits),
            ),
            "additive_bias_raw": additive_bias_raw_diag,
            "additive_bias": additive_bias_diag,
        }
