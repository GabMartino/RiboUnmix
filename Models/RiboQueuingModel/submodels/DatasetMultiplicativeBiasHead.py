from __future__ import annotations

import torch
from torch import nn


class DatasetMultiplicativeBiasHead(nn.Module):
    """
    Dataset/protocol multiplicative correction head.

    Outputs:
        b_smooth:
            positive smooth multiplier around 1.

        log_b:
            log of b_smooth before mass-preserving normalization.

        keep_gate:
            effective gate used in q. If hard_forward/hard_eval are active,
            this is already floored:

                keep_gate = gate_min + (1 - gate_min) * gate_raw

            so a closed gate suppresses support but does not make it exactly zero.

        keep_prob:
            differentiable gate probability. Use this for regularization.

        keep_hard:
            raw binary hard gate before gate_min floor.
    """

    def __init__(self, config_params: dict, input_size: int) -> None:
        super().__init__()

        self.hidden_size = int(config_params["hidden_size"])
        self.dropout = float(config_params.get("dropout", 0.0))

        self.log_b_max = float(config_params.get("log_b_max", 1.0))
        self.tanh_temperature = max(float(config_params.get("tanh_temperature", 3.0)), 1e-6)

        self.gate_temperature = max(float(config_params.get("gate_temperature", 1.0)), 1e-6)
        self.gate_threshold = float(config_params.get("gate_threshold", 0.5))
        self.init_keep_prob = float(config_params.get("init_keep_prob", 0.995))

        self.gate_min = float(config_params.get("gate_min", 0.0))
        self.gate_min = min(max(self.gate_min, 0.0), 1.0 - 1e-6)

        self.hard_forward = bool(config_params.get("hard_forward", False))
        self.hard_eval = bool(config_params.get("hard_eval", self.hard_forward))

        self.bias_ff = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1, bias=False),
        )

        self.gate_ff = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        # Neutral b: beta=0 -> log_b=0 -> b_smooth=1.
        nn.init.zeros_(self.bias_ff[-1].weight)

        # Temperature-aware gate initialization:
        # keep_prob = sigmoid(gate_logits / gate_temperature)
        # therefore gate_bias = gate_temperature * logit(init_keep_prob)
        nn.init.zeros_(self.gate_ff[-1].weight)

        init_p = min(max(self.init_keep_prob, 1e-5), 1.0 - 1e-5)
        init_bias = self.gate_temperature * torch.logit(torch.tensor(init_p)).item()
        nn.init.constant_(self.gate_ff[-1].bias, init_bias)

    def _effective_gate(self, gate_raw: torch.Tensor) -> torch.Tensor:
        return self.gate_min + (1.0 - self.gate_min) * gate_raw

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, T, F], got {tuple(x.shape)}")

        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=x.dtype)

        x = x * mask_f.unsqueeze(-1)
        valid_lengths = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        # ------------------------------------------------------------
        # b branch
        # ------------------------------------------------------------
        beta = self.bias_ff(x).squeeze(-1)
        beta = beta * mask_f

        beta_mean = beta.sum(dim=1, keepdim=True) / valid_lengths
        beta_centered = (beta - beta_mean) * mask_f

        log_b = self.log_b_max * torch.tanh(beta_centered / self.tanh_temperature)
        log_b = log_b * mask_f

        log_b_mean = log_b.sum(dim=1, keepdim=True) / valid_lengths
        log_b = (log_b - log_b_mean) * mask_f

        b_smooth = torch.exp(log_b)
        b_smooth = torch.where(mask_b, b_smooth, torch.ones_like(b_smooth))

        # ------------------------------------------------------------
        # gate branch
        # ------------------------------------------------------------
        gate_logits = self.gate_ff(x).squeeze(-1)
        gate_logits = gate_logits * mask_f

        keep_prob = torch.sigmoid(gate_logits / self.gate_temperature)
        keep_prob = keep_prob * mask_f

        keep_hard = (keep_prob > self.gate_threshold).to(dtype=x.dtype)
        keep_hard = keep_hard * mask_f

        if self.training:
            if self.hard_forward:
                gate_raw = keep_hard + keep_prob - keep_prob.detach()
            else:
                gate_raw = keep_prob
        else:
            gate_raw = keep_hard if self.hard_eval else keep_prob

        gate_raw = gate_raw * mask_f
        keep_gate = self._effective_gate(gate_raw)
        keep_gate = keep_gate * mask_f

        # Padding should be neutral.
        keep_gate = torch.where(mask_b, keep_gate, torch.ones_like(keep_gate))
        keep_prob = torch.where(mask_b, keep_prob, torch.ones_like(keep_prob))
        keep_hard = torch.where(mask_b, keep_hard, torch.ones_like(keep_hard))
        gate_logits = torch.where(mask_b, gate_logits, torch.zeros_like(gate_logits))

        return {
            "b_smooth": b_smooth,
            "log_b": log_b,
            "beta_centered": beta_centered,
            "keep_gate": keep_gate,
            "keep_prob": keep_prob,
            "keep_hard": keep_hard,
            "gate_logits": gate_logits,
        }