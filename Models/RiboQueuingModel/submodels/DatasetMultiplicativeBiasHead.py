from __future__ import annotations

import torch
from torch import nn


class DatasetMultiplicativeBiasHead(nn.Module):
    """
    Dataset/protocol multiplicative correction with an internal keep gate.

    Decomposition:

        b_smooth_i = exp(log_b_i)

        gate_i in [0, 1]

        b_total_i = gate_i * b_smooth_i

    Interpretation:

        b_smooth:
            smooth positive multiplicative correction around 1.

        gate:
            observation keep/drop gate.
            It can suppress positions toward zero.

        b_total:
            final multiplier applied to L_queue.

    Important:
        log_b is the log of b_smooth, NOT log(b_total).
        If gate becomes zero, log(b_total) would be -inf, so use log_b
        for regularization/diagnostics of smooth multiplicative bias.

    Default behavior:
        - Training uses soft gate_prob unless hard_forward=True.
        - Evaluation uses hard gate if hard_eval=True, otherwise soft gate.
        - Gate initializes near 1 everywhere.
    """

    def __init__(self, config_params: dict, input_size: int) -> None:
        super().__init__()

        self.hidden_size = int(config_params["hidden_size"])
        self.dropout = float(config_params.get("dropout", 0.0))

        # Smooth multiplicative branch.
        self.log_b_max = float(config_params.get("log_b_max", 1.0))
        self.tanh_temperature = float(config_params.get("tanh_temperature", 3.0))

        # Gate branch.
        self.gate_temperature = float(config_params.get("gate_temperature", 2.0))
        self.gate_threshold = float(config_params.get("gate_threshold", 0.5))
        self.init_keep_prob = float(config_params.get("init_keep_prob", 0.995))

        self.hard_forward = bool(config_params.get("hard_forward", False))
        self.hard_eval = bool(config_params.get("hard_eval", True))

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
            nn.Linear(self.hidden_size, 1),
        )

        # Neutral smooth multiplier:
        # beta = 0 -> log_b = 0 -> b_smooth = 1.
        nn.init.zeros_(self.bias_ff[-1].weight)

        # Gate starts almost fully open.
        nn.init.zeros_(self.gate_ff[-1].weight)

        init_p = min(max(self.init_keep_prob, 1e-5), 1.0 - 1e-5)
        init_bias = torch.logit(torch.tensor(init_p)).item()
        nn.init.constant_(self.gate_ff[-1].bias, init_bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,  # b_total
        torch.Tensor,  # log_b_smooth
        torch.Tensor,  # beta_centered
        torch.Tensor,  # b_smooth
        torch.Tensor,  # keep_gate
        torch.Tensor,  # keep_prob
        torch.Tensor,  # keep_hard
        torch.Tensor,  # gate_logits
    ]:
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=x.dtype)

        x = x * mask_f.unsqueeze(-1)

        valid_lengths = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        # ============================================================
        # 1. Smooth multiplicative correction b_smooth
        # ============================================================
        beta = self.bias_ff(x).squeeze(-1)
        beta = beta * mask_f

        beta_mean = beta.sum(dim=1, keepdim=True) / valid_lengths
        beta_centered = (beta - beta_mean) * mask_f

        log_b = self.log_b_max * torch.tanh(
            beta_centered / max(self.tanh_temperature, 1e-6)
        )
        log_b = log_b * mask_f

        # Center after nonlinearity.
        # This makes geometric mean(b_smooth) approximately 1 over valid positions.
        log_b_mean = log_b.sum(dim=1, keepdim=True) / valid_lengths
        log_b = (log_b - log_b_mean) * mask_f

        b_smooth = torch.exp(log_b)
        b_smooth = torch.where(
            mask_b,
            b_smooth,
            torch.ones_like(b_smooth),
        )

        # ============================================================
        # 2. Observation keep gate
        # ============================================================
        gate_logits = self.gate_ff(x).squeeze(-1)
        gate_logits = gate_logits * mask_f

        keep_prob = torch.sigmoid(
            gate_logits / max(self.gate_temperature, 1e-6)
        )
        keep_prob = keep_prob * mask_f

        keep_hard = (keep_prob > self.gate_threshold).to(dtype=x.dtype)
        keep_hard = keep_hard * mask_f

        if self.training:
            if self.hard_forward:
                # Straight-through hard gate:
                # forward = hard, backward = soft.
                keep_gate = keep_hard + keep_prob - keep_prob.detach()
            else:
                # Safer early training.
                keep_gate = keep_prob
        else:
            keep_gate = keep_hard if self.hard_eval else keep_prob

        keep_gate = keep_gate * mask_f

        # ============================================================
        # 3. Final multiplier
        # ============================================================
        b_total = b_smooth * keep_gate

        # Outside valid positions use neutral multiplier 1.
        b_total = torch.where(
            mask_b,
            b_total,
            torch.ones_like(b_total),
        )

        keep_gate = torch.where(
            mask_b,
            keep_gate,
            torch.ones_like(keep_gate),
        )

        keep_prob = torch.where(
            mask_b,
            keep_prob,
            torch.ones_like(keep_prob),
        )

        keep_hard = torch.where(
            mask_b,
            keep_hard,
            torch.ones_like(keep_hard),
        )

        return (
            b_total,
            log_b,
            beta_centered,
            b_smooth,
            keep_gate,
            keep_prob,
            keep_hard,
            gate_logits,
        )