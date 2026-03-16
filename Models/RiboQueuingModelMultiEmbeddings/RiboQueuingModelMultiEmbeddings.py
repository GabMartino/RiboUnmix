from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import math


@torch.no_grad()
def compute_S_quantile(
        y_true: torch.Tensor,
        mask: torch.Tensor,
        q: float,
        eps: float,
        censor_threshold: float,
        use_censor_threshold: bool,
        use_nonzero_only: bool,
) -> torch.Tensor:
    y = y_true.to(torch.float32)
    m = mask.bool()

    if use_censor_threshold:
        m = m & (y > float(censor_threshold))
    elif use_nonzero_only:
        m = m & (y > 0)

    counts = m.sum(dim=1)  # [B]
    B, _T = y.shape

    vals = y.masked_fill(~m, float("inf"))
    vals_sorted, _ = torch.sort(vals, dim=1)

    q = float(q)
    k = (q * (counts.clamp_min(1) - 1).float()).floor().long()
    k_max = (counts.clamp_min(1) - 1).long()
    k = torch.minimum(k.clamp_min(0), k_max)

    S = vals_sorted.gather(1, k.view(B, 1))
    S = torch.where(counts.view(B, 1) > 0, S, torch.full_like(S, float(eps)))
    return S.clamp_min(float(eps))


class RiboQueuingModelMultiEmbeddings(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 2,
            dropout: float = 0.1,
            w_temperature: float = 1.0,
            rho_eps: float = 1e-6,
            pi_max: float = 0.99,
            num_datasets: int = 32,
            dataset_emb_dim: int | None = None,
            eps: float = 1e-8,
            dataset_log_offset_clip: float = 6.0,
            log_sigma_min: float = -4.0,
            log_sigma_max: float = 1.0,
            S_quantile: float = 0.99,
            S_eps: float = 1e-8,
            S_use_censor_threshold: bool = True,
            S_use_nonzero_only: bool = False,
            censor_threshold: float = 0.5,
    ):
        super().__init__()

        if not (0.0 < pi_max < 1.0):
            raise ValueError("Require 0 < pi_max < 1")

        self.num_layers = int(num_layers)
        self.w_temperature = float(w_temperature)
        self.rho_eps = float(rho_eps)
        self.pi_max = float(pi_max)

        self.num_datasets = int(num_datasets)
        self.eps = float(eps)

        self.dataset_log_offset_clip = float(dataset_log_offset_clip)
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)

        self.S_quantile = float(S_quantile)
        self.S_eps = float(S_eps)
        self.S_use_censor_threshold = bool(S_use_censor_threshold)
        self.S_use_nonzero_only = bool(S_use_nonzero_only)
        self.censor_threshold = float(censor_threshold)

        if dataset_emb_dim is None:
            dataset_emb_dim = hidden_size


        self.dataset_emb_dim = int(dataset_emb_dim)

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        feat_dim = hidden_size * 2
        h_dim = self.num_layers * 2 * hidden_size

        # --- Intrinsic biology ---
        self.ff_w_logits = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 1),
        )

        # THE FIX 1: Change input from feat_dim (local) to h_dim (global)
        self.ff_pi = nn.Sequential(
            nn.Linear(h_dim + self.dataset_emb_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid(),
        )

        # predicts log_sigma, then sigma = exp(log_sigma)
        self.ff_log_sigma = nn.Sequential(
            nn.Linear(feat_dim + self.dataset_emb_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 1),
        )

        self.ff_J_conditioned = nn.Sequential(
            nn.Linear(h_dim + self.dataset_emb_dim, h_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 1),
            nn.Softplus(beta=1.0),
        )

        # --- Transcript and dataset scaling ---
        self.dataset_embeddings = nn.Embedding(self.num_datasets, self.dataset_emb_dim)

        self.dataset_offset_ff = nn.Sequential(
            nn.Linear(feat_dim + self.dataset_emb_dim, h_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_dim, h_dim),  # <-- The new expansion layer
            nn.GELU(),
            nn.Linear(h_dim, 2),
        )

        # --- Initialization ---
        nn.init.constant_(self.ff_w_logits[-1].bias, 0.0)
        nn.init.constant_(self.ff_J_conditioned[-2].bias, -1.0)

        # pi_max * Sigmoid(-2.0) sets initial dropout probability very low (approx 12%)
        nn.init.constant_(self.ff_pi[-2].bias, -2.0)

        # log_sigma = -1.0 -> sigma = 0.36 (starts with low variance assumption)
        nn.init.constant_(self.ff_log_sigma[-1].bias, -1.0)

        # Force the dataset offset network to start at true neutral (a_raw=0, b_raw=0)
        nn.init.constant_(self.dataset_offset_ff[-1].weight, 0.0)
        nn.init.constant_(self.dataset_offset_ff[-1].bias, 0.0)

    def rnn_out(self, x_packed):
        out_packed, h_n = self.rnn(x_packed)
        out, lengths = pad_packed_sequence(out_packed, batch_first=True)

        device = out.device
        lengths = lengths.to(device)

        B, T, _ = out.shape
        arange = torch.arange(T, device=device)
        mask = arange[None, :] < lengths[:, None]
        mask_f = mask.to(dtype=out.dtype)

        return out, h_n, mask, mask_f, B, T, lengths

    def utilization_rate(self, x, h_n, dataset_embeddings, mask, mask_f, lengths, B):
        w_logits = self.ff_w_logits(x).squeeze(-1)  # [B,T]
        w_logits = w_logits.masked_fill(~mask, float("-inf"))
        w_prob = torch.softmax(w_logits / max(self.w_temperature, 1e-6), dim=1)
        w_prob = w_prob * mask_f

        # Format h_n to [B, h_dim]
        h_n_flat = h_n.permute(1, 0, 2).reshape(B, -1)

        h_n_dataset = torch.cat([h_n_flat.detach(), dataset_embeddings], -1)
        J = self.ff_J_conditioned(h_n_dataset)

        L = lengths.unsqueeze(1).to(dtype=x.dtype)  # [B,1]
        x_flux = J * L * w_prob  # [B,T]
        rho = (1.0 - torch.exp(-x_flux)).clamp_max(1.0 - self.rho_eps) * mask_f
        return rho, w_prob, J

    def transcript_and_dataset_scaling(self, out, dataset_embeddings, y_raw_target, mask, detach_out: bool = True):
        transcript_scale_S = compute_S_quantile(
            y_raw_target, mask, q=self.S_quantile, eps=self.S_eps,
            censor_threshold=self.censor_threshold, use_censor_threshold=self.S_use_censor_threshold,
            use_nonzero_only=self.S_use_nonzero_only,
        )

        log_transcript_scale = torch.log(transcript_scale_S.clamp_min(self.eps))

        B, T, _ = out.shape
        mask_f = mask.unsqueeze(-1).float()

        dataset_emb_expanded = dataset_embeddings.unsqueeze(1).expand(B, T, -1)
        log_transcript_scale_pos = log_transcript_scale.unsqueeze(1).expand(B, T, 1)

        out_for_scale = out.detach() if detach_out else out

        ff_input = torch.cat(
            [out_for_scale, dataset_emb_expanded],
            dim=-1,
        )

        ab = self.dataset_offset_ff(ff_input)
        a, b = ab.chunk(2, dim=-1)

        a = 1.0 + 0.1 * torch.tanh(a)
        b = b * mask_f
        valid_lengths = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        b_mean = b.sum(dim=1, keepdim=True) / valid_lengths
        b = (b - b_mean) * mask_f
        b = b.clamp(-self.dataset_log_offset_clip, self.dataset_log_offset_clip) * mask_f
        log_total_scale = a * log_transcript_scale_pos + b
        total_scale = torch.exp(log_total_scale)

        return total_scale.squeeze(-1), transcript_scale_S, log_transcript_scale, a.squeeze(-1), b.squeeze(-1)

    def compute_conditional_sigma(self, out, dataset_embeddings):
        B, T, _ = out.shape

        dataset_emb_expanded = dataset_embeddings.unsqueeze(1).expand(B, T, -1)

        sigma_ff_input = torch.cat([out.detach(), dataset_emb_expanded], dim=-1)

        log_sigma_raw = self.ff_log_sigma(sigma_ff_input).squeeze(-1)  # [B, T]

        log_sigma = log_sigma_raw.clamp(self.log_sigma_min, self.log_sigma_max)

        sigma = torch.exp(log_sigma)  # [B, T]
        return sigma, log_sigma

    def forward(self, x_packed, id_datasets: torch.Tensor, y_raw_target: torch.Tensor):
        out, h_n, mask, mask_f, B, T, lengths = self.rnn_out(x_packed)

        '''
            Intrinsic biology
        '''
        dataset_embeddings = self.dataset_embeddings(id_datasets)

        rho, w_prob, J = self.utilization_rate(out, h_n, dataset_embeddings, mask, mask_f, lengths, B)

        total_scale, transcript_scale_S, log_transcript_scale, a, b = self.transcript_and_dataset_scaling(
            out, dataset_embeddings, y_raw_target, mask
        )

        mu = rho * total_scale

        # THE FIX 2: Global Pi Execution
        # 1. Format and detach the global h_n vector [B, h_dim]
        h_n_flat_detached = h_n.permute(1, 0, 2).reshape(B, -1).detach()

        # 2. Concatenate with dataset embeddings [B, h_dim + E]
        pi_ff_input = torch.cat([h_n_flat_detached, dataset_embeddings], dim=-1)

        # 3. Predict a single, global dropout scalar per transcript [B, 1]
        pi_global = self.ff_pi(pi_ff_input) * self.pi_max

        # 4. Expand it across the sequence length and apply padding mask [B, T]
        pi = pi_global.expand(B, T) * mask_f

        sigma, log_sigma = self.compute_conditional_sigma(out, dataset_embeddings)

        return mu, pi, sigma, (
            rho,
            w_prob,
            J,
            transcript_scale_S,
            log_transcript_scale,
            total_scale,
            a,
            b,
            log_sigma * mask_f,
        )