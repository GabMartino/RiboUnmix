import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence

from Models.utils.compute_S_quantile import compute_S_quantile


class RiboQueuingModelMultiEmbeddings(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            embeddings_list: list = None,  # NEW: Pass the list of embeddings here
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

        # --- THE EMBEDDING PROJECTION HEAD ---
        self.embeddings_list = embeddings_list or []

        # If we have embeddings, we project them into a dense 32-dim vector
        self.emb_hidden_dim = 32 if len(self.embeddings_list) > 0 else 0

        if self.emb_hidden_dim > 0:
            raw_emb_dim = 0
            for emb in self.embeddings_list:
                raw_emb_dim += 3 if emb in ['dom', 'exo', 'fra', 'gmp', 'tmp', 'openen'] else 1

            self.emb_proj = nn.Sequential(
                nn.Linear(raw_emb_dim, self.emb_hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout)
            )

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        feat_dim = hidden_size * 2
        h_dim = self.num_layers * 2 * hidden_size

        # The new augmented dimension for the downstream physics heads
        augmented_feat_dim = feat_dim + self.emb_hidden_dim

        # --- Intrinsic biology ---
        self.ff_w_logits = nn.Sequential(
            nn.Linear(augmented_feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 1),
        )

        self.ff_pi = nn.Sequential(
            nn.Linear(h_dim + self.dataset_emb_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid(),
        )

        self.ff_log_sigma = nn.Sequential(
            nn.Linear(augmented_feat_dim + self.dataset_emb_dim, feat_dim),
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
            nn.Linear(augmented_feat_dim + self.dataset_emb_dim, h_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 2),
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

    # UPDATED: Now takes augmented_out instead of raw out
    def utilization_rate(self, augmented_out, h_n, dataset_embeddings, mask, mask_f, lengths, B):
        w_logits = self.ff_w_logits(augmented_out).squeeze(-1)  # [B,T]
        w_logits = w_logits.masked_fill(~mask, float("-inf"))
        w_prob = torch.softmax(w_logits / max(self.w_temperature, 1e-6), dim=1)
        w_prob = w_prob * mask_f

        h_n_flat = h_n.permute(1, 0, 2).reshape(B, -1)
        h_n_dataset = torch.cat([h_n_flat.detach(), dataset_embeddings], -1)
        J = self.ff_J_conditioned(h_n_dataset)

        L = lengths.unsqueeze(1).to(dtype=augmented_out.dtype)  # [B,1]
        x_flux = J * L * w_prob
        rho = (1.0 - torch.exp(-x_flux)).clamp_max(1.0 - self.rho_eps) * mask_f
        return rho, w_prob, J

    # UPDATED: Now takes augmented_out
    def transcript_and_dataset_scaling(self, augmented_out, dataset_embeddings, y_raw_target, mask,
                                       detach_out: bool = True):
        transcript_scale_S = compute_S_quantile(
            y_raw_target, mask, q=self.S_quantile, eps=self.S_eps,
            censor_threshold=self.censor_threshold, use_censor_threshold=self.S_use_censor_threshold,
            use_nonzero_only=self.S_use_nonzero_only,
        )

        log_transcript_scale = torch.log(transcript_scale_S.clamp_min(self.eps))

        B, T, _ = augmented_out.shape
        mask_f = mask.unsqueeze(-1).float()

        dataset_emb_expanded = dataset_embeddings.unsqueeze(1).expand(B, T, -1)
        log_transcript_scale_pos = log_transcript_scale.unsqueeze(1).expand(B, T, 1)

        out_for_scale = augmented_out.detach() if detach_out else augmented_out

        ff_input = torch.cat([out_for_scale, dataset_emb_expanded], dim=-1)

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

    # UPDATED: Now takes augmented_out
    def compute_conditional_sigma(self, augmented_out, dataset_embeddings):
        B, T, _ = augmented_out.shape
        dataset_emb_expanded = dataset_embeddings.unsqueeze(1).expand(B, T, -1)
        sigma_ff_input = torch.cat([augmented_out.detach(), dataset_emb_expanded], dim=-1)
        log_sigma_raw = self.ff_log_sigma(sigma_ff_input).squeeze(-1)
        log_sigma = log_sigma_raw.clamp(self.log_sigma_min, self.log_sigma_max)
        sigma = torch.exp(log_sigma)
        return sigma, log_sigma

    # NEW SIGNATURE: Accepts batch_embeddings
    def forward(self, x_packed, id_datasets: torch.Tensor, y_raw_target: torch.Tensor, batch_embeddings: dict = None):
        out, h_n, mask, mask_f, B, T, lengths = self.rnn_out(x_packed)

        # --- THE EMBEDDING FUSION ---
        if self.emb_hidden_dim > 0 and batch_embeddings is not None:
            emb_tensors = []
            # Extract in the exact order specified in __init__
            for emb_name in self.embeddings_list:
                # Shape is [B, Tmax, 3]
                emb_tensors.append(batch_embeddings[emb_name])

            # Concatenate along the last dimension.
            # E.g., 9 embeddings * 3 nts = 27 features [B, Tmax, 27]
            fused_embs = torch.cat(emb_tensors, dim=-1)

            # Pass through the projection head to get a [B, Tmax, 32] vector
            projected_embs = self.emb_proj(fused_embs)

            # Concatenate the biological features with the GRU's memory state
            augmented_out = torch.cat([out, projected_embs], dim=-1)
        else:
            augmented_out = out

        # ----------------------------

        dataset_embeddings = self.dataset_embeddings(id_datasets)

        # Feed the augmented representation to the physics heads
        rho, w_prob, J = self.utilization_rate(augmented_out, h_n, dataset_embeddings, mask, mask_f, lengths, B)

        total_scale, transcript_scale_S, log_transcript_scale, a, b = self.transcript_and_dataset_scaling(
            augmented_out, dataset_embeddings, y_raw_target, mask
        )

        mu = rho * total_scale

        h_n_flat_detached = h_n.permute(1, 0, 2).reshape(B, -1).detach()
        pi_ff_input = torch.cat([h_n_flat_detached, dataset_embeddings], dim=-1)
        pi_global = self.ff_pi(pi_ff_input) * self.pi_max
        pi = pi_global.expand(B, T) * mask_f

        sigma, log_sigma = self.compute_conditional_sigma(augmented_out, dataset_embeddings)

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