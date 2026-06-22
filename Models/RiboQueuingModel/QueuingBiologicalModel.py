from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


def inv_softplus(x: float) -> float:
    x = float(x)
    if x <= 0.0:
        raise ValueError(f"inv_softplus requires x > 0, got {x}.")
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


class QueuingBiologicalModel(nn.Module):
    """
    Shared, dataset-blind biological queue-load model (sequence only).

    Pipeline (mean-normalized local factor + transcript flux -> queue load):

        w_raw_i    = softplus(local_head(out)_i)          >= 0
        w_norm_i   = w_raw_i / mean_valid(w_raw)           (mean_valid(w_norm)=1)
        J          = clamp(softplus(J_head(h_n)), J_min, J_max)   > 0
        lambda_i   = clamp(J * w_norm_i, lambda_bio_min, lambda_bio_max)
        rho_i      = 1 - exp(-lambda_i)            (utilization in [0, 1))
        L_bio_i    = expm1(lambda_i) = rho_i / (1 - rho_i)   (avg jobs in queue)

    `L_bio` is the queueing-theory average number of jobs in the queue and is
    the biological load multiplied downstream by the target-derived scale and
    the dataset visibility correction:

        mu = S[d,t] * L_bio[t,i] * beta[d,t,i].

    `forward` returns a dict with: w_raw, w_norm, J, lambda_bio, rho, L_bio, h_n.
    """

    def __init__(self, config_params: dict):
        super().__init__()

        self.input_size = int(config_params["input_size"])
        self.hidden_size = int(config_params["hidden_size"])
        self.num_layers = int(config_params["num_layers"])
        self.dropout = float(config_params.get("dropout", 0.0))

        self.J_min = float(config_params.get("J_min", 1.0e-4))
        self.J_max = float(config_params.get("J_max", 5.0))
        self.init_J = float(config_params.get("init_J", 0.5))
        self.lambda_bio_min = float(config_params.get("lambda_bio_min", 1.0e-3))
        self.lambda_bio_max = float(config_params.get("lambda_bio_max", 3.0))
        self.eps = float(config_params.get("eps", 1.0e-8))

        if self.lambda_bio_min < 0.0:
            raise ValueError(
                f"lambda_bio_min must be >= 0, got {self.lambda_bio_min}."
            )
        if self.lambda_bio_max <= self.lambda_bio_min:
            raise ValueError(
                "lambda_bio_max must be greater than lambda_bio_min, got "
                f"{self.lambda_bio_max} <= {self.lambda_bio_min}."
            )

        self.init_local_hazard_factor = float(
            config_params.get("init_local_hazard_factor", 1.0)
        )
        # Std of the random init for the final local-factor weights. Must be > 0
        # so the per-position factor is NOT flat at init (a flat profile gives a
        # degenerate / zero shape signal on arrival).
        self.init_local_hazard_weight_std = float(
            config_params.get("init_local_hazard_weight_std", 1.0e-2)
        )

        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        feat_dim = self.hidden_size * 2
        h_dim = self.num_layers * 2 * self.hidden_size

        self.ff_local_hazard = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(feat_dim, 1),
            nn.Softplus(),
        )

        self.ff_J = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(h_dim, 1),
            nn.Softplus(),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        final_j = self.ff_J[-2]
        if isinstance(final_j, nn.Linear):
            nn.init.zeros_(final_j.weight)
            init_j = min(max(self.init_J, self.J_min), self.J_max)
            nn.init.constant_(final_j.bias, inv_softplus(init_j))

        final_h = self.ff_local_hazard[-2]
        if isinstance(final_h, nn.Linear):
            # Small random (not zero) weights so the local factor varies across
            # positions at init; the bias centers the mean factor at
            # init_local_hazard_factor.
            std = max(float(self.init_local_hazard_weight_std), 0.0)
            if std > 0.0:
                nn.init.normal_(final_h.weight, mean=0.0, std=std)
            else:
                nn.init.zeros_(final_h.weight)
            init_factor = max(float(self.init_local_hazard_factor), 1.0e-6)
            nn.init.constant_(final_h.bias, inv_softplus(init_factor))

    def forward(self, x_packed, mask) -> dict[str, torch.Tensor]:
        out_packed, h_n = self.rnn(x_packed)
        out, _ = pad_packed_sequence(
            out_packed,
            batch_first=True,
            total_length=mask.shape[1],
        )
        B, T, _ = out.shape
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=out.dtype)

        if mask_b.shape != (B, T):
            raise ValueError(
                f"mask shape {tuple(mask_b.shape)} does not match RNN output {(B, T)}."
            )

        # 1. Local mean-normalized biological factor: mean_valid(w_norm) = 1.
        w_raw = self.ff_local_hazard(out).squeeze(-1)
        w_raw = w_raw * mask_f
        valid_len = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        w_mean = (w_raw * mask_f).sum(dim=1, keepdim=True) / valid_len
        w_norm = w_raw / w_mean.clamp_min(self.eps)
        w_norm = w_norm * mask_f

        # 2. Transcript-level flux / intensity J > 0.
        h_n_flat = h_n.permute(1, 0, 2).reshape(B, -1)
        J = self.ff_J(h_n_flat).clamp(min=self.J_min, max=self.J_max)  # [B, 1]

        # 3. Queue load. lambda = J * w_norm (clamped); L_bio = expm1(lambda).
        lambda_bio = (J * w_norm).clamp(
            min=self.lambda_bio_min,
            max=self.lambda_bio_max,
        ) * mask_f
        rho = (-torch.expm1(-lambda_bio)).clamp(0.0, 1.0 - 1.0e-6) * mask_f
        L_bio = torch.expm1(lambda_bio).clamp_min(self.eps) * mask_f

        return {
            "w_raw": w_raw,
            "w_norm": w_norm,
            "J": J,
            "lambda_bio": lambda_bio,
            "rho": rho,
            "L_bio": L_bio,
            "h_n": h_n,
        }
