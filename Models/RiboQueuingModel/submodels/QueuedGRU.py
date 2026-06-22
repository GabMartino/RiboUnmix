from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    PackedSequence,
    pad_packed_sequence,
    pack_padded_sequence,
)


class QueuedGRU(nn.GRU):
    """
    Drop-in replacement for torch.nn.GRU with an internal reverse queuing layer.

    Standard usage, identical to nn.GRU:

        output, h_n = gru(x)

    Extended usage:

        output, h_n, queue = gru(x, return_queue=True)

    where queue contains:

        queue["rho"]   : local latent occupancy      [B, L, 1]
        queue["alpha"] : propagation gate            [B, L, 1]
        queue["q"]     : effective queued occupancy  [B, L, 1]

    Queue recurrence:

        q_i = rho_i + alpha_i * (1 - rho_i) * q_{i+1}

    with q_{L+1} = 0.

    Parameters
    ----------
    queue_mode:
        "aux":
            Return the original GRU output unchanged.
            The queue is only returned when return_queue=True.

        "modulate":
            Multiply the GRU output by q_i, preserving the same output shape.
            This keeps the class shape-compatible with nn.GRU, but changes the hidden
            representation using the queue.

            output_i <- q_i * output_i

    Notes
    -----
    If you want the queuing model to be scientifically meaningful, the cleanest
    use is usually queue_mode="aux", followed by a prediction head that uses q.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        *,
        queue_mode: Literal["aux", "modulate"] = "aux",
        rho_init_bias: float = -2.0,
        alpha_init_bias: float = -2.0,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            **factory_kwargs,
        )

        if queue_mode not in {"aux", "modulate"}:
            raise ValueError(
                f"queue_mode must be 'aux' or 'modulate', got {queue_mode!r}"
            )

        self.queue_mode = queue_mode

        gru_output_size = hidden_size * (2 if bidirectional else 1)

        self.rho_head = nn.Linear(
            gru_output_size,
            1,
            device=device,
            dtype=dtype,
        )

        self.alpha_head = nn.Linear(
            gru_output_size,
            1,
            device=device,
            dtype=dtype,
        )

        self.reset_queue_parameters(
            rho_init_bias=rho_init_bias,
            alpha_init_bias=alpha_init_bias,
        )

    def reset_queue_parameters(
        self,
        rho_init_bias: float = -2.0,
        alpha_init_bias: float = -2.0,
    ) -> None:
        """
        Initialize the queue heads.

        rho_init_bias=-2 gives initial rho around sigmoid(-2) ~= 0.12.
        alpha_init_bias=-2 gives initial alpha around sigmoid(-2) ~= 0.12.

        This makes the queue initially weak, so the model starts close to a normal GRU.
        """

        nn.init.xavier_uniform_(self.rho_head.weight)
        nn.init.constant_(self.rho_head.bias, rho_init_bias)

        nn.init.xavier_uniform_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, alpha_init_bias)

    def forward(
        self,
        input: torch.Tensor | PackedSequence,
        hx: torch.Tensor | None = None,
        *,
        lengths: torch.Tensor | list[int] | None = None,
        return_queue: bool = False,
    ):
        """
        Parameters
        ----------
        input:
            Same as nn.GRU input.

            If batch_first=True:
                input shape is [B, L, D]

            If batch_first=False:
                input shape is [L, B, D]

            PackedSequence is also supported.

        hx:
            Same as nn.GRU initial hidden state.

        lengths:
            Optional sequence lengths for padded inputs.
            Not needed for PackedSequence.

        return_queue:
            If False:
                returns exactly like nn.GRU:

                    output, h_n

            If True:
                returns:

                    output, h_n, queue

        Returns
        -------
        output:
            Same shape/type as nn.GRU output.

        h_n:
            Same as nn.GRU h_n.

        queue:
            Only returned if return_queue=True.
        """

        is_packed = isinstance(input, PackedSequence)

        output, h_n = super().forward(input, hx)

        if is_packed:
            return self._forward_packed_output(
                packed_output=output,
                h_n=h_n,
                return_queue=return_queue,
            )

        return self._forward_tensor_output(
            output=output,
            h_n=h_n,
            lengths=lengths,
            return_queue=return_queue,
        )

    def _forward_tensor_output(
        self,
        output: torch.Tensor,
        h_n: torch.Tensor,
        lengths: torch.Tensor | list[int] | None,
        return_queue: bool,
    ):
        # Convert GRU output to batch-first internally: [B, L, H]
        if self.batch_first:
            output_blh = output
        else:
            output_blh = output.transpose(0, 1)

        if return_queue or self.queue_mode == "modulate":
            queue = self.compute_queue(output_blh, lengths=lengths)

            if self.queue_mode == "modulate":
                output_blh = output_blh * queue["q"]

                if self.batch_first:
                    output = output_blh
                else:
                    output = output_blh.transpose(0, 1)

            if return_queue:
                return output, h_n, queue

        return output, h_n

    def _forward_packed_output(
        self,
        packed_output: PackedSequence,
        h_n: torch.Tensor,
        return_queue: bool,
    ):
        # If queue is not requested and not used, preserve exact PackedSequence output.
        if not return_queue and self.queue_mode == "aux":
            return packed_output, h_n

        padded_output, lengths = pad_packed_sequence(
            packed_output,
            batch_first=True,
        )

        queue = self.compute_queue(
            padded_output,
            lengths=lengths.to(padded_output.device),
        )

        if self.queue_mode == "modulate":
            padded_output = padded_output * queue["q"]

            packed_output = pack_padded_sequence(
                padded_output,
                lengths=lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

        if return_queue:
            return packed_output, h_n, queue

        return packed_output, h_n

    def compute_queue(
        self,
        output_blh: torch.Tensor,
        lengths: torch.Tensor | list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute rho, alpha and q from GRU outputs.

        Parameters
        ----------
        output_blh:
            GRU output in batch-first format [B, L, H].

        lengths:
            Optional valid sequence lengths [B].

        Returns
        -------
        queue:
            Dictionary with rho, alpha, q.
        """

        rho = torch.sigmoid(self.rho_head(output_blh))
        alpha = torch.sigmoid(self.alpha_head(output_blh))

        q = self.reverse_queue_recurrence(
            rho=rho,
            alpha=alpha,
            lengths=lengths,
        )

        if lengths is not None:
            mask = self._length_mask(
                lengths=lengths,
                max_len=output_blh.size(1),
                device=output_blh.device,
            ).unsqueeze(-1)

            rho = rho * mask
            alpha = alpha * mask
            q = q * mask

        return {
            "rho": rho,
            "alpha": alpha,
            "q": q,
        }

    @staticmethod
    def reverse_queue_recurrence(
        rho: torch.Tensor,
        alpha: torch.Tensor,
        lengths: torch.Tensor | list[int] | None = None,
    ) -> torch.Tensor:
        """
        Compute:

            q_i = rho_i + alpha_i * (1 - rho_i) * q_{i+1}

        in reverse order.

        rho shape:
            [B, L, 1]

        alpha shape:
            [B, L, 1]

        q shape:
            [B, L, 1]
        """

        if rho.ndim != 3 or rho.size(-1) != 1:
            raise ValueError(f"rho must have shape [B, L, 1], got {rho.shape}")

        if alpha.shape != rho.shape:
            raise ValueError(
                f"alpha must have the same shape as rho, got {alpha.shape} and {rho.shape}"
            )

        batch_size, seq_len, _ = rho.shape

        q = torch.zeros_like(rho)
        q_next = rho.new_zeros(batch_size, 1)

        if lengths is not None:
            lengths = torch.as_tensor(lengths, device=rho.device)
            valid = QueuedGRU._length_mask(
                lengths=lengths,
                max_len=seq_len,
                device=rho.device,
            )
        else:
            valid = None

        for i in range(seq_len - 1, -1, -1):
            rho_i = rho[:, i, :]
            alpha_i = alpha[:, i, :]

            q_i = rho_i + alpha_i * (1.0 - rho_i) * q_next

            if valid is not None:
                q_i = q_i * valid[:, i].unsqueeze(-1)

            q[:, i, :] = q_i
            q_next = q_i

        return q

    @staticmethod
    def _length_mask(
        lengths: torch.Tensor | list[int],
        max_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        lengths = torch.as_tensor(lengths, device=device)
        positions = torch.arange(max_len, device=device).unsqueeze(0)
        return positions < lengths.unsqueeze(1)