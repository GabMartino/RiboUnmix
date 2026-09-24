"""Protect GRUs from cuDNN's FP16 autocast path, including under BF16 AMP.

Keep original FP32 Parameters and checkpoint keys. Only recurrence execution
changes; surrounding heads still use the caller's mixed-precision context.
"""
from contextlib import contextmanager

import torch
from torch.nn.utils.rnn import PackedSequence


GRU_COMPUTE_POLICY = "cuda-amp-gru-fp32-v1"


@contextmanager
def gru_precision_context(rnn, inputs, *, force_float32=False):
    """Yield FP32 input under disabled CUDA autocast; restore AMP on exit.

PyTorch 2.10's cuDNN autocast wrapper hardcodes FP16 even for BF16 AMP.
Legacy 'inherit' configurations therefore still need this CUDA protection.
CPU and explicit non-AMP FP64 reference calculations are unchanged unless
force_float32 is requested. No weight conversion or gradient repair occurs.
"""
    data = inputs.data if isinstance(inputs, PackedSequence) else inputs
    protect = force_float32 or (
        data.device.type == "cuda" and torch.is_autocast_enabled("cuda")
    )
    if not protect:
        yield inputs
        return
    if rnn.weight_ih_l0.dtype != torch.float32:
        raise ValueError(
            "Protected GRU execution requires FP32 model weights; use mixed "
            "precision or 32-true, not model.half()/bfloat16() or true-16 training."
        )
    with torch.autocast(device_type=data.device.type, enabled=False):
        yield inputs.to(dtype=torch.float32)
