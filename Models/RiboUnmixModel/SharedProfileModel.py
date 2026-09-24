from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from Models.utils.stable_numerics import log_softplus, masked_logmeanexp, masked_mean
from Models.utils.gru_precision import gru_precision_context


def inv_softplus(x: float) -> float:
    x = float(x)
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


class SharedProfileModel(nn.Module):
    """
    Shared, dataset-blind biological profile model (sequence only).

    Pipeline (mean-normalized local factor -> shared profile):

        w_raw_i    = softplus(local_head(out)_i)          >= 0
        w_norm_i   = w_raw_i / mean_valid(w_raw)           (mean_valid(w_norm)=1)
        L_bio_i    = w_norm_i, so mean_valid(L_bio)=1
        rho_i      = L_bio_i / (1 + L_bio_i)
        lambda_i   = log(1 + L_bio_i)
        J          = mean_valid(lambda_i)                  (diagnostic only)

    `L_bio` is the mean-one shared profile multiplied downstream by the
    target-derived scale and the dataset bias correction:

        mu = S[d,t] * (gamma[d,t,i] * L_bio[t,i] + a[d,t,i]).

    `forward` returns a dict with: w_raw, w_norm, J, lambda_bio, rho, L_bio, h_n.
    """

    def __init__(self, config_params: dict):
        super().__init__()

        self.input_size = int(config_params["input_size"])
        self.hidden_size = int(config_params["hidden_size"])
        self.num_layers = int(config_params["num_layers"])
        self.dropout = float(config_params.get("dropout", 0.0))

        self.eps = float(config_params.get("eps", 1.0e-8))
        # direct_load is the only mode: L_bio = w_norm is already mean-one, so
        # lambda is never clamped. These are kept as fixed attributes purely so
        # the downstream lambda-clamp diagnostics keep resolving.
        self.lambda_bio_min = 0.0
        self.lambda_bio_max = None

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

        self.ff_local_hazard = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(feat_dim, 1),
            nn.Softplus(),
        )

        # No learned transcript-flux head in direct_load mode.
        self.ff_J = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
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
        # The biological GRU uses the same CUDA-AMP protection as the bias
        # GRU; the feed-forward head below retains the caller's AMP policy.
        with gru_precision_context(self.rnn, x_packed) as recurrent_input:
            out_packed, h_n = self.rnn(recurrent_input)
        out, _ = pad_packed_sequence(
            out_packed,
            batch_first=True,
            total_length=mask.shape[1],
        )
        B, T, _ = out.shape
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=out.dtype)


        # 1. Local mean-normalized biological factor: mean_valid(w_norm) = 1.
        # The final Softplus has no parameters: bypass only its materialized
        # value, retaining every existing checkpoint key and neural AMP op.
        log_w_raw = log_softplus(self.ff_local_hazard[:-1](out).squeeze(-1))
        log_w_norm = log_w_raw - masked_logmeanexp(log_w_raw, mask_b)
        log_w_norm = torch.where(mask_b, log_w_norm, 0.0)
        w_raw = torch.where(mask_b, log_w_raw.exp(), 0.0)
        w_norm = torch.where(mask_b, log_w_norm.exp(), 0.0)
        mask_f = mask_b.to(w_norm.dtype)

        # 2. Direct-load queue: L_bio = w_norm (mean-one), rho = L/(1+L),
        #    lambda = log(1 + L_bio). J is the mean-lambda diagnostic.
        L_bio = w_norm
        rho = (L_bio / (1.0 + L_bio).clamp_min(self.eps)) * mask_f
        lambda_bio = torch.log1p(L_bio) * mask_f
        J = masked_mean(lambda_bio, mask_b)

        return {
            "w_raw": w_raw,
            "w_norm": w_norm,
            "J": J,
            "lambda_bio": lambda_bio,
            "rho": rho,
            "L_bio": L_bio,
            "log_L_bio": log_w_norm,
            "h_n": h_n,
        }
