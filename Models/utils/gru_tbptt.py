"""State-carrying TBPTT for a stacked, bidirectional ``nn.GRU``.

The forward recurrence and parameterization are unchanged. Only temporal
derivatives across window boundaries are removed. Windows start at the first
valid token in each direction (the true sequence end for the reverse GRU).
Both directions of a layer are completed before evaluating the next layer.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from Models.utils.gru_precision import gru_precision_context


def _reverse_valid(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(x.size(1), device=x.device)[None, :]
    indices = (lengths[:, None] - 1 - positions).clamp_min(0)
    reversed_x = x.gather(1, indices[..., None].expand_as(x))
    return reversed_x.masked_fill((positions >= lengths[:, None])[..., None], 0)


def _direction(
    rnn: nn.GRU,
    x: torch.Tensor,
    lengths: torch.Tensor,
    window: int,
    layer: int,
    reverse: bool,
) -> torch.Tensor:
    """One direction, with lengths sorted descending on CPU."""
    suffix = f"_l{layer}" + ("_reverse" if reverse else "")
    names = ["weight_ih", "weight_hh"]
    if rnn.bias:
        names += ["bias_ih", "bias_hh"]
    weights = [getattr(rnn, name + suffix) for name in names]
    batch, padded_length, _ = x.shape
    hidden = x.new_zeros(1, batch, rnn.hidden_size)
    outputs = []
    max_length = int(lengths[0])
    for start in range(0, max_length, window):
        width = min(window, max_length - start)
        active = int((lengths > start).sum())
        chunk_lengths = (lengths[:active] - start).clamp_max(width)
        packed = pack_padded_sequence(
            x[:active, start:start + width], chunk_lengths,
            batch_first=True, enforce_sorted=True,
        )
        # Values propagate across the full CDS; only their history is detached.
        initial = hidden[:, :active].detach().contiguous()
        # Same packed ATen operator as nn.GRU.forward, using the ORIGINAL
        # Parameter objects. No extra modules, checkpoint keys, or per-codon
        # Python unroll. Isolated here because _VF is an internal PyTorch API.
        data, hidden = torch._VF.gru(
            packed.data, packed.batch_sizes, initial, weights,
            rnn.bias, 1, 0.0, rnn.training, False,
        )
        output, _ = pad_packed_sequence(
            PackedSequence(data, packed.batch_sizes),
            batch_first=True, total_length=width,
        )
        outputs.append(F.pad(output, (0, 0, 0, 0, 0, batch - active)))
    return F.pad(torch.cat(outputs, dim=1), (0, 0, 0, padded_length - max_length))


def gru_tbptt(
    rnn: nn.GRU,
    sequence: torch.Tensor,
    cpu_lengths: torch.Tensor | tuple[int, ...],
    window: int,
) -> torch.Tensor:
    """Return [B,T,directions*H]; detach each direction's state every K tokens.

    Right-padding is excluded from recurrence. Recurrent dropout must be zero:
    splitting a fused stacked GRU cannot preserve its inter-layer dropout mask.
    Head dropout is unaffected. This helper is used only for training with grad.
    """
    if window <= 0 or rnn.dropout != 0:
        raise ValueError("GRU TBPTT requires a positive window and recurrent dropout=0.")
    # _VF.gru bypasses nn.GRU.forward, so protect direct helper calls too.
    with gru_precision_context(rnn, sequence) as recurrent_input:
        return _gru_tbptt_impl(rnn, recurrent_input, cpu_lengths, window)


def _gru_tbptt_impl(rnn, sequence, cpu_lengths, window):
    lengths = torch.as_tensor(cpu_lengths, dtype=torch.long, device="cpu")
    lengths, order_cpu = lengths.sort(descending=True, stable=True)
    order = order_cpu.to(sequence.device)
    device_lengths = lengths.to(sequence.device)
    x = sequence.index_select(0, order)
    for layer in range(rnn.num_layers):
        forward = _direction(rnn, x, lengths, window, layer, reverse=False)
        if rnn.bidirectional:
            backward = _direction(
                rnn, _reverse_valid(x, device_lengths), lengths, window, layer,
                reverse=True,
            )
            backward = _reverse_valid(backward, device_lengths)
            x = torch.cat((forward, backward), dim=-1)
        else:
            x = forward
    return x.index_select(0, order.argsort())
