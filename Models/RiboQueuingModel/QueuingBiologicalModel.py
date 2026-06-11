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
    Local biological traffic-intensity model.

        h_i      >= 0                    (softplus output)
        rho_i    = J * h_i               (per-codon traffic intensity, unbounded)

    Semantics: rho_i here is *traffic intensity* in queueing-theory terms
    (analogous to lambda / mu_service in M/M/1), not stationary occupancy
    probability. It is intentionally unbounded above so the downstream
    prediction mu = scale_dt * rho * beta can express the full dynamic range
    of observed ribo-seq footprint counts (zero through ~thousands).

    The bounded probabilistic occupancy is recovered downstream in
    RiboQueuingModel as `rho_utilization = 1 - exp(-rho)`, which is the
    Poisson "at least one footprint" probability and lives in [0, 1). That
    is the quantity used for queue propagation (where bounded inputs are
    required for stability); rho itself stays unbounded for the prediction
    head. See _intensity_to_utilization in RiboQueuingModel.

    There is intentionally no transcript-level sum_i w_i = 1 allocation
    constraint.
    """

    def __init__(self, config_params: dict):
        super().__init__()

        self.input_size = int(config_params["input_size"])
        self.hidden_size = int(config_params["hidden_size"])
        self.num_layers = int(config_params["num_layers"])
        self.dropout = float(config_params.get("dropout", 0.0))

        self.J_min = float(config_params.get("J_min", 1.0e-6))
        self.J_max = float(config_params.get("J_max", 10.0))
        self.init_J = float(config_params.get("init_J", max(self.J_min, 0.5)))
        self.init_local_hazard_factor = float(
            config_params.get("init_local_hazard_factor", 1.0)
        )
        # Std of the random init for the final local-hazard weights. Must be > 0
        # so the per-position hazard is NOT flat at init: a perfectly constant
        # rho makes a Pearson-correlation loss gradient identically zero (the
        # PCC validity gate detaches it), which would freeze training on arrival.
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

        self.ff_J_conditioned = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(h_dim, 1),
            nn.Softplus(),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        final_j = self.ff_J_conditioned[-2]
        if isinstance(final_j, nn.Linear):
            nn.init.zeros_(final_j.weight)
            init_j = min(max(self.init_J, self.J_min), self.J_max)
            nn.init.constant_(final_j.bias, inv_softplus(init_j))

        final_h = self.ff_local_hazard[-2]
        if isinstance(final_h, nn.Linear):
            # Small random (not zero) weights so the local hazard varies across
            # positions at init; the bias still centers the mean factor at
            # init_local_hazard_factor. See init_local_hazard_weight_std above.
            std = max(float(self.init_local_hazard_weight_std), 0.0)
            if std > 0.0:
                nn.init.normal_(final_h.weight, mean=0.0, std=std)
            else:
                nn.init.zeros_(final_h.weight)
            init_factor = max(float(self.init_local_hazard_factor), 1.0e-6)
            nn.init.constant_(final_h.bias, inv_softplus(init_factor))

    def forward(self, x_packed, mask) -> tuple:
        out_packed, h_n = self.rnn(x_packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        B, T, _ = out.shape
        mask_b = mask.bool()
        mask_f = mask.to(dtype=out.dtype)

        if mask_b.shape != (B, T):
            raise ValueError(
                f"mask shape {tuple(mask_b.shape)} does not match RNN output {(B, T)}."
            )

        # ------------------------------------------------------------
        # 1. Local non-normalized hazard factor
        # ------------------------------------------------------------
        local_factor = self.ff_local_hazard(out).squeeze(-1)
        local_factor = local_factor * mask_f

        # ------------------------------------------------------------
        # 2. Transcript-level hazard scale
        # ------------------------------------------------------------
        h_n_flat = h_n.permute(1, 0, 2).reshape(B, -1)

        J = self.ff_J_conditioned(h_n_flat)
        J = J.clamp(min=self.J_min, max=self.J_max)

        # ------------------------------------------------------------
        # 3. Local unbounded traffic/intensity. No sum_i w_i = 1 constraint.
        # ------------------------------------------------------------
        h_bio = J.to(dtype=out.dtype).reshape(B, 1) * local_factor
        h_bio = torch.nan_to_num(
            h_bio,
            nan=0.0,
            posinf=1.0e8,
            neginf=0.0,
        )
        h_bio = h_bio.clamp_min(0.0) * mask_f

        rho_bio = h_bio

        return rho_bio, J, h_bio, h_n
