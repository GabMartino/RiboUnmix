from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DatasetTranscriptScaleFactorHead(nn.Module):
    """
    Simple dataset-transcript scale factor.

    Predicts one positive scalar per transcript/dataset pair:

        S_{d,t} = exp(s_d + r_{d,t})

    where:

        s_d     = dataset-specific log scale
        r_{d,t} = transcript-specific residual log scale

    Intended usage in the outer model:

        mu_{d,t,i} = S_{d,t} * rho_{t,i} * beta_{d,t,i}

    This head does not modify rho or beta. It only handles absolute scale.
    """

    def __init__(
        self,
        config_params: dict,
    ):
        super().__init__()

        biological_context_size = int(config_params["biological_context_size"])

        self.num_datasets = int(config_params["num_datasets"])
        self.dataset_embedding_size = int(config_params["dataset_scale_embedding_size"])
        self.hidden_size = int(config_params["hidden_size"])
        self.dropout = float(config_params["dropout"])
        self.log_dataset_scale_max = float(config_params["log_dataset_scale_max"])
        self.log_transcript_scale_max = float(config_params["log_transcript_scale_max"])

        self.dataset_log_scale = nn.Embedding(self.num_datasets, 1)
        nn.init.zeros_(self.dataset_log_scale.weight)

        self.dataset_embedding = nn.Embedding(
            self.num_datasets,
            self.dataset_embedding_size,
        )

        self.context_norm = nn.LayerNorm(biological_context_size)

        mlp_input_size = self.dataset_embedding_size + biological_context_size

        self.transcript_residual = nn.Sequential(
            nn.Linear(mlp_input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize scale neutrally:

            dataset_log_scale = 0
            transcript_log_scale = 0

        Therefore:

            scale_dt = exp(0) = 1
        """
        final = self.transcript_residual[-1]

        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @staticmethod
    def _bound_log_scale(
        raw: torch.Tensor,
        max_abs: float,
    ) -> torch.Tensor:
        max_abs = max(float(max_abs), 1.0e-6)
        return max_abs * torch.tanh(raw / max_abs)

    def _prepare_biological_context(
        self,
        ctx: torch.Tensor,
        B: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        ctx = ctx.detach()
        if ctx.ndim == 3:
            ctx = ctx.permute(1, 0, 2).reshape(B, -1)
        return self.context_norm(ctx.to(device=device, dtype=dtype))

    def forward(
        self,
        *,
        dataset_ids: torch.Tensor,
        biological_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            dataset_ids:
                [B]

            biological_context:
                Biological GRU final state h_n, either [layers*directions, B, H]
                or already flattened as [B, layers*directions*H].

        Returns:
            scale_dt:
                [B, 1], positive scale factor.
        """
        B = int(dataset_ids.shape[0])
        device = dataset_ids.device

        dataset_ids = dataset_ids.to(device=device, dtype=torch.long)

        dataset_weight = self.dataset_embedding.weight
        dataset_weight = dataset_weight - dataset_weight.mean(dim=0, keepdim=True)
        dataset_emb = F.embedding(dataset_ids, dataset_weight)
        dtype = dataset_emb.dtype

        dataset_log_weight = self.dataset_log_scale.weight
        dataset_log_weight = dataset_log_weight - dataset_log_weight.mean(dim=0, keepdim=True)
        dataset_log_raw = F.embedding(dataset_ids, dataset_log_weight).to(dtype=dtype)
        dataset_log_scale = self._bound_log_scale(
            dataset_log_raw,
            self.log_dataset_scale_max,
        )

        biological_context = self._prepare_biological_context(biological_context, B, dtype, device)

        mlp_input = torch.cat([dataset_emb, biological_context], dim=-1)

        transcript_log_raw = self.transcript_residual(mlp_input)
        transcript_log_scale = self._bound_log_scale(
            transcript_log_raw,
            self.log_transcript_scale_max,
        )

        log_scale_dt = (
            dataset_log_scale
            + transcript_log_scale
        )

        scale_dt = torch.exp(log_scale_dt)

        return {
            "scale_dt": scale_dt,
            "log_scale_dt": log_scale_dt,
            "dataset_log_scale": dataset_log_scale,
            "transcript_log_scale": transcript_log_scale,
            "dataset_scale": torch.exp(dataset_log_scale),
            "transcript_scale": torch.exp(transcript_log_scale),
        }
