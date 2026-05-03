import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence


class QueuingBiologicalModel(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int = 32,
                 num_layers: int = 2,
                 dropout: float = 0.0,):
        super().__init__()

        self.w_temperature = 1
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        feat_dim = hidden_size * 2
        h_dim = num_layers * 2 * hidden_size

        self.ff_w_logits = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 1),
        )

        self.ff_J_conditioned = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 1),
            nn.Softplus(),
        )
    def rnn_out(self, x_packed):
        x_padded, _ = pad_packed_sequence(x_packed, batch_first=True)

        out_packed, h_n = self.rnn(x_packed)
        out, lengths = pad_packed_sequence(out_packed, batch_first=True)

        device = out.device
        lengths = lengths.to(device)

        B, T, _ = out.shape

        if x_padded.size(1) != T:
            if x_padded.size(1) > T:
                x_padded = x_padded[:, :T, :]
            else:
                pad_T = T - x_padded.size(1)
                x_padded = torch.cat(
                    [
                        x_padded,
                        torch.zeros(
                            x_padded.size(0),
                            pad_T,
                            x_padded.size(2),
                            device=x_padded.device,
                            dtype=x_padded.dtype,
                        ),
                    ],
                    dim=1,
                )

        x_padded = x_padded.to(device=device, dtype=out.dtype)

        arange = torch.arange(T, device=device)
        mask = arange[None, :] < lengths[:, None]
        mask_f = mask.to(dtype=out.dtype)

        return x_padded, out, h_n, mask, mask_f, B, T, lengths

    def utilization_rate(
        self,
        x: torch.Tensor,
        h_n: torch.Tensor,
        mask: torch.Tensor,
        mask_f: torch.Tensor,
        lengths: torch.Tensor,
        B: int,
    ):
        w_logits = self.ff_w_logits(x).squeeze(-1)
        w_logits = w_logits.masked_fill(~mask, float("-inf"))

        temperature = max(float(self.w_temperature), 1e-6)

        w_prob = torch.softmax(w_logits / temperature, dim=1)
        w_prob = w_prob * mask_f

        h_n_flat = h_n.permute(1, 0, 2).reshape(B, -1)
        J = self.ff_J_conditioned(h_n_flat).clamp(1e-6, 100.0)

        safe_w_prob = w_prob.clamp_min(1e-10)
        L_seq = lengths.unsqueeze(1).to(dtype=x.dtype)

        x_flux = J * L_seq * safe_w_prob
        x_flux = x_flux.clamp_max(12.0)

        L_queue = torch.expm1(x_flux) * mask_f
        rho_diagnostic = (1.0 - torch.exp(-x_flux)) * mask_f

        return L_queue, rho_diagnostic, w_prob, J

    def forward(self, x_packed):
        x_padded, out, h_n, mask, mask_f, B, T, lengths = self.rnn_out(x_packed)

        L_queue, rho_diag, w_prob, J = self.utilization_rate(
            out,
            h_n,
            mask,
            mask_f,
            lengths,
            B,
        )

        return L_queue, rho_diag, w_prob, J, mask