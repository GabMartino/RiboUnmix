from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import math

from Models.utils.compute_S_quantile import compute_S_quantile, compute_S_median, compute_S_mean


class RiboQueuingModel(nn.Module):
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
            cnn_hidden_dim: int = 128,  # NEW: Dimension for the RNase Convolutional Head
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
        self.cnn_hidden_dim = int(cnn_hidden_dim)

        feat_dim = hidden_size * 2
        h_dim = self.num_layers * 2 * hidden_size
        # --- NEW: The RNase Convolutional Head ---
        # A 1D CNN that slides a 5-codon window over the raw sequence
        # to detect sequence-specific RNase enzyme cleavage biases.
        # padding=2 ensures the output sequence length (T) perfectly matches the input.
        self.rnase_cnn = nn.Sequential(
            nn.Conv1d(in_channels=feat_dim + self.dataset_emb_dim, out_channels=self.cnn_hidden_dim, kernel_size=5,
                      padding=2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=self.cnn_hidden_dim, out_channels=self.cnn_hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )


        # --- Intrinsic biology ---
        self.ff_w_logits = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 1),
        )

        self.ff_pi = nn.Sequential(
            nn.Linear(feat_dim + self.dataset_emb_dim, feat_dim),
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
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 1),
            nn.Softplus(beta=1.0),
        )

        # --- Transcript and dataset scaling ---
        self.dataset_embeddings = nn.Embedding(self.num_datasets, self.dataset_emb_dim)

        # THE FIX: Input dimension is now cnn_hidden_dim + dataset_emb_dim
        self.dataset_offset_ff = nn.Sequential(
            nn.Linear(self.cnn_hidden_dim + self.dataset_emb_dim, h_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_dim, h_dim*2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_dim*2, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 1),
        )

        # --- Initialization ---
        nn.init.constant_(self.ff_w_logits[-1].bias, 0.0)
        nn.init.constant_(self.ff_J_conditioned[-2].bias, -1.0)
        nn.init.constant_(self.ff_pi[-2].bias, -2.0)
        nn.init.constant_(self.ff_log_sigma[-1].bias, -1.0)
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

    def utilization_rate(self, x, h_n, mask, mask_f, lengths, B):
        w_logits = self.ff_w_logits(x).squeeze(-1)  # [B,T]
        w_logits = w_logits.masked_fill(~mask, float("-inf"))

        w_prob = torch.softmax(w_logits / max(self.w_temperature, 1e-6), dim=1)
        w_prob = w_prob * mask_f

        h_n_flat = h_n.permute(1, 0, 2).reshape(B, -1)
        J = self.ff_J_conditioned(h_n_flat).clamp(1e-6, 100.0)  # [B,1]

        safe_w_prob = w_prob.clamp_min(1e-10)
        L_seq = lengths.unsqueeze(1).to(dtype=x.dtype)  # [B,1]

        x_flux = J * L_seq * safe_w_prob
        x_flux = x_flux.clamp_max(200.0)

        L_queue = torch.expm1(x_flux) * mask_f
        rho_diagnostic = (1.0 - torch.exp(-x_flux)) * mask_f

        return L_queue, rho_diagnostic, w_prob, J

    def transcript_and_dataset_scaling(self, out, dataset_embeddings, y_raw_target, mask, mask_f, B, T):
        dataset_emb_expanded = dataset_embeddings.unsqueeze(1).expand(B, T, -1)

        x_cnn_input = torch.cat([out.detach(), dataset_emb_expanded], dim=-1)
        x_cnn_input_t = x_cnn_input.transpose(1, 2)

        cnn_out_t = self.rnase_cnn(x_cnn_input_t)
        cnn_out = cnn_out_t.transpose(1, 2) * mask_f.unsqueeze(-1)

        mask_f_3d = mask.unsqueeze(-1).float()

        ff_input = torch.cat([cnn_out, dataset_emb_expanded], dim=-1)
        b = self.dataset_offset_ff(ff_input)
        b = b * mask_f_3d

        valid_lengths = mask_f_3d.sum(dim=1, keepdim=True).clamp_min(1.0)
        b_mean = b.sum(dim=1, keepdim=True) / valid_lengths
        b = (b - b_mean) * mask_f_3d
        b = b.clamp(-self.dataset_log_offset_clip, self.dataset_log_offset_clip) * mask_f_3d

        transcript_scale_S = compute_S_mean(
            y_raw_target,
            mask,
            eps=self.S_eps,
            censor_threshold=self.censor_threshold,
            use_censor_threshold=self.S_use_censor_threshold,
            use_nonzero_only=self.S_use_nonzero_only,
        )

        log_transcript_scale = torch.log(transcript_scale_S.clamp_min(self.eps))
        log_transcript_scale_pos = log_transcript_scale.unsqueeze(1).expand(B, T, 1)

        log_total_scale = log_transcript_scale_pos + b

        max_log = math.log(torch.finfo(log_total_scale.dtype).max) - 2.0
        total_scale = torch.exp(log_total_scale.clamp(max=max_log))

        return total_scale.squeeze(-1), transcript_scale_S, log_transcript_scale, b.squeeze(-1)

    def compute_conditional_sigma(self, out, dataset_embeddings, mu_phys, mask_f):
        B, T, _ = out.shape
        dataset_emb_expanded = dataset_embeddings.unsqueeze(1).expand(B, T, -1)
        alpha_ff_input = torch.cat([out.detach(), dataset_emb_expanded], dim=-1)

        alpha = F.softplus(self.ff_log_sigma(alpha_ff_input).squeeze(-1))
        alpha = alpha * mask_f

        safe_mu = mu_phys.clamp_min(1e-6)
        rel_var = 1.0 / safe_mu + alpha

        X = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * rel_var))
        sigma_sq = torch.log(X)
        sigma = torch.sqrt(sigma_sq.clamp_min(1e-8))

        log_sigma = torch.log(sigma).clamp(self.log_sigma_min, self.log_sigma_max)
        sigma = torch.exp(log_sigma) * mask_f
        log_sigma = log_sigma * mask_f

        return sigma, log_sigma, alpha

    def compute_conditional_pi(self, out, dataset_embeddings, mask_f, B, T):
        dataset_emb_expanded = dataset_embeddings.unsqueeze(1).expand(B, T, -1)
        pi_ff_input = torch.cat([out.detach(), dataset_emb_expanded], dim=-1)  # Shape: [B, T, Hidden + Emb]

        pi_local = self.ff_pi(pi_ff_input) * self.pi_max
        pi = pi_local.squeeze(-1) * mask_f  # Shape: [B, T]
        return pi

    def forward(self, x_packed, id_datasets: torch.Tensor, y_raw_target: torch.Tensor):
        out, h_n, mask, mask_f, B, T, lengths = self.rnn_out(x_packed)

        L_queue, rho_diag, w_prob, J = self.utilization_rate(out, h_n, mask, mask_f, lengths, B)

        dataset_embeddings = self.dataset_embeddings(id_datasets)
        total_scale, S_mean, log_S, b = self.transcript_and_dataset_scaling(
            out, dataset_embeddings, y_raw_target, mask, mask_f, B, T
        )

        mu = L_queue * total_scale
        pi = self.compute_conditional_pi(out, dataset_embeddings, mask_f, B, T)
        sigma, log_sigma, alpha = self.compute_conditional_sigma(out, dataset_embeddings, mu, mask_f)

        return mu, pi, sigma, (
                rho_diag,  # 0: Occupancy (0-1)
                w_prob,  # 1: Intrinsic slowness
                L_queue,  # 2: Queue intensity (Pile-up)
                J,  # 3: Initiation rate
                S_mean,  # 4: Library baseline
                total_scale,  # 5: S_mean * exp(b)
                b,  # 6: Spatial bias (footprint)
                alpha,  # 7: Overdispersion (the "Noise" factor)
                log_sigma,  # 8: Clamped log variance
            )