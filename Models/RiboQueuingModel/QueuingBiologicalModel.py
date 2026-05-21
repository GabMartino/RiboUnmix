import torch
from entmax import entmax15
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence


class QueuingBiologicalModel(nn.Module):
    def __init__(self,
                 config_params: dict):
        super().__init__()

        self.input_size = config_params["input_size"]
        self.hidden_size = config_params["hidden_size"]
        self.num_layers = config_params["num_layers"]
        self.w_temperature = config_params["w_temperature"]
        self.dropout = config_params["dropout"]

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
            #nn.Linear(feat_dim, feat_dim),
            #nn.GELU(),
            nn.Linear(feat_dim, 1),
        )

        self.ff_J_conditioned = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            #nn.Linear(h_dim, h_dim),
            #nn.GELU(),
            nn.Linear(h_dim, 1),
            nn.Softplus(),
        )

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

        #w_prob = torch.softmax(w_logits / temperature, dim=1)
        w_prob = entmax15(w_logits, dim=1)
        w_prob = w_prob * mask_f

        h_n_flat = h_n.permute(1, 0, 2).reshape(B, -1)
        J = self.ff_J_conditioned(h_n_flat).clamp(1e-6, 100.0)

        L_seq = lengths.unsqueeze(1).to(dtype=x.dtype)

        x_flux = J * L_seq * w_prob
        x_flux = x_flux.clamp_max(5.0)

        L_queue  = torch.expm1(x_flux) * mask_f
        rho_diagnostic = (1.0 - torch.exp(-x_flux)) * mask_f

        return L_queue, rho_diagnostic, w_prob, J

    def forward(self, x_packed):

        out_packed, h_n = self.rnn(x_packed)
        out, lengths = pad_packed_sequence(out_packed, batch_first=True)

        device = out.device
        lengths = lengths.to(device)
        B, T, _ = out.shape
        arange = torch.arange(T, device=device)
        mask = arange[None, :] < lengths[:, None]
        mask_f = mask.to(dtype=out.dtype)

        L_queue, rho_diag, w_prob, J = self.utilization_rate(
            out,
            h_n,
            mask,
            mask_f,
            lengths,
            B,
        )

        return L_queue, rho_diag, w_prob, J, mask