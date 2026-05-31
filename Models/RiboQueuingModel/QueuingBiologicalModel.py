import torch
from entmax import entmax15
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence


class QueuingBiologicalModel(nn.Module):
    """
    Shared biological allocator + transcript-level hazard model.

    This module produces the shared biological allocation logits a_{t,i}.

    Biological path:

        a_{t,i} = biological allocation logits
        w_bio   = entmax(a)
        h_bio   = J_t * T_t * w_bio
        L_bio   = exp(h_bio) - 1

    The outer model can then form the dataset-specific observed allocation:

        w_obs = entmax(a + beta_d)

    where beta_d is a centered dataset/protocol logit bias.
    """

    def __init__(self, config_params: dict):
        super().__init__()

        self.input_size = int(config_params["input_size"])
        self.hidden_size = int(config_params["hidden_size"])
        self.num_layers = int(config_params["num_layers"])
        self.dropout = float(config_params.get("dropout", 0.0))

        self.w_temperature = float(config_params.get("w_temperature", 1.0))
        self.w_transform = str(config_params.get("w_transform", "entmax15")).lower()

        self.J_min = float(config_params.get("J_min", 1e-6))
        self.J_max = float(config_params.get("J_max", 10.0))
        self.hazard_max = float(config_params.get("hazard_max", 8.0))

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

        self.ff_w_logits = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(feat_dim, 1),
        )

        self.ff_J_conditioned = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(h_dim, 1),
            nn.Softplus(),
        )

    def allocation_from_logits(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts allocation logits into a valid probability allocation.

        With entmax15, exact zeros are possible:

            w_i = 0
            h_i = 0
            L_i = 0

        This is the mechanism we want for biological or observed zero support.
        """
        if logits.ndim != 2:
            raise ValueError(f"Expected logits [B, T], got {tuple(logits.shape)}")

        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=logits.dtype)

        temperature = max(float(self.w_temperature), 1e-6)

        # Avoid NaNs in mixed precision by using dtype min instead of -inf.
        neg_large = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask_b, neg_large)

        z = logits / temperature

        if self.w_transform == "entmax15":
            w = entmax15(z, dim=1)
        elif self.w_transform == "softmax":
            w = torch.softmax(z, dim=1)
        else:
            raise ValueError(
                f"Unknown w_transform={self.w_transform!r}. "
                "Use 'entmax15' or 'softmax'."
            )

        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = w.clamp_min(0.0) * mask_f

        # Numerical safety. For entmax this should already sum to one, but the
        # renormalization protects against fully masked / underflow cases.
        w_mass = w.sum(dim=1, keepdim=True)
        uniform = mask_f / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        w = torch.where(
            w_mass > 1e-8,
            w / w_mass.clamp_min(1e-8),
            uniform,
        )

        w = w * mask_f

        return w

    def hazard_from_allocation(
        self,
        *,
        w: torch.Tensor,
        J: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Builds hazard from allocation:

            h_i = J * T * w_i

        Since sum_i w_i = 1, the mean valid hazard is approximately J.
        """
        mask_f = mask.bool().to(dtype=w.dtype)

        L_seq = lengths.reshape(-1, 1).to(device=w.device, dtype=w.dtype)

        h = J.to(dtype=w.dtype) * L_seq * w
        h = torch.nan_to_num(
            h,
            nan=0.0,
            posinf=self.hazard_max,
            neginf=0.0,
        )
        h = h.clamp(min=0.0, max=self.hazard_max)
        h = h * mask_f

        return h

    def rho_from_hazard(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = mask.bool().to(dtype=h.dtype)
        rho = -torch.expm1(-h.clamp(min=0.0, max=self.hazard_max))
        return rho * mask_f

    def L_queue_from_hazard(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = mask.bool().to(dtype=h.dtype)
        h = h.clamp(min=0.0, max=self.hazard_max)
        L = torch.expm1(h)
        L = torch.nan_to_num(L, nan=0.0, posinf=1e8, neginf=0.0)
        return L * mask_f

    def forward(self, x_packed) -> dict[str, torch.Tensor]:
        out_packed, h_n = self.rnn(x_packed)
        out, lengths = pad_packed_sequence(out_packed, batch_first=True)

        device = out.device
        lengths = lengths.to(device)

        B, T, _ = out.shape

        arange = torch.arange(T, device=device)
        mask = arange[None, :] < lengths[:, None]
        mask_f = mask.to(dtype=out.dtype)

        # ------------------------------------------------------------
        # Biological allocation logits a_{t,i}
        # ------------------------------------------------------------
        w_logits = self.ff_w_logits(out).squeeze(-1)
        w_logits = w_logits.masked_fill(~mask, torch.finfo(w_logits.dtype).min)

        w_bio = self.allocation_from_logits(
            logits=w_logits,
            mask=mask,
        )

        # ------------------------------------------------------------
        # Transcript-level biological hazard J_t
        # ------------------------------------------------------------
        h_n_flat = h_n.permute(1, 0, 2).reshape(B, -1)

        J = self.ff_J_conditioned(h_n_flat)
        J = J.clamp(min=self.J_min, max=self.J_max)

        # ------------------------------------------------------------
        # Biological hazard/support
        # ------------------------------------------------------------
        h_bio = self.hazard_from_allocation(
            w=w_bio,
            J=J,
            lengths=lengths,
            mask=mask,
        )

        rho_bio = self.rho_from_hazard(h_bio, mask)
        L_bio = self.L_queue_from_hazard(h_bio, mask)

        w_zero_frac = ((w_bio <= 0.0) & mask).float().sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        L_zero_frac = ((L_bio <= 0.0) & mask).float().sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

        return {
            "encoded": out,
            "lengths": lengths,
            "mask": mask,

            "w_logits": w_logits,
            "w_bio": w_bio,
            "w_zero_frac_bio": w_zero_frac,

            "J": J,
            "h_n": h_n,
            "h_bio": h_bio,
            "rho_bio": rho_bio,
            "L_bio": L_bio,
            "L_zero_frac_bio": L_zero_frac,
        }