"""Opt-in capture of the *first bad fused GRU backward*, without repairing it.

The saved payload is a diagnostic, not a training checkpoint. It contains the
actual packed input, GRU weights, incoming adjoints and RNG/runtime settings.
It isolates the recurrence from the loss and earlier accumulated gradients.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import torch
from torch.nn.utils.rnn import PackedSequence


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def finite_flags(values):
    return [None if item is None else bool(torch.isfinite(item).all()) for item in values]


def capture_if_nonfinite(directory, payload_factory, grad_inputs, grad_outputs):
    """A node hook: observe, save, then fail. Never return replacement gradients."""
    incoming = finite_flags(grad_outputs)
    generated = finite_flags(grad_inputs)
    if False not in incoming and False not in generated:
        return
    origin = "upstream_adjoint_already_nonfinite" if False in incoming else "inside_fused_gru_backward"
    message = (
        f"Non-finite GRU backward: origin={origin}; "
        f"incoming_adjoint_finite={incoming}; generated_gradient_finite={generated}."
    )
    try:
        payload = payload_factory()
        payload.update(
            origin=origin, incoming_adjoint_finite=incoming,
            generated_gradient_finite=generated, grad_outputs=_cpu(grad_outputs),
        )
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        # Unique name and exclusive creation: never overwrite a previous capture.
        with tempfile.NamedTemporaryFile(prefix="gru_failure_", suffix=".pt",
                                         dir=destination, delete=False) as stream:
            torch.save(payload, stream)
            capture_path = stream.name
        message += f" Replay capture: {capture_path} (diagnostic, not a resume checkpoint)."
    except Exception as error:
        message += f" Capture failed: {type(error).__name__}: {error}."
    raise FloatingPointError(message)


def begin_gru_capture(rnn, packed):
    if not isinstance(packed, PackedSequence) or packed.data.device.type != "cuda":
        raise ValueError("GRU failure capture currently requires a CUDA packed-sequence GRU.")
    if rnn.dropout != 0:
        # cuDNN's internal dropout state is not the public CUDA RNG alone.
        raise ValueError("Exact fused GRU capture currently requires context_gru_dropout=0.")
    device = packed.data.device
    return {
        "format_version": 1,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(device),
        "autocast_enabled": torch.is_autocast_enabled("cuda"),
        "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
        "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(device),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "rnn_config": dict(input_size=rnn.input_size, hidden_size=rnn.hidden_size,
                           num_layers=rnn.num_layers, bias=rnn.bias,
                           batch_first=rnn.batch_first, dropout=rnn.dropout,
                           bidirectional=rnn.bidirectional),
    }


def attach_gru_capture(rnn, packed, output, before, directory):
    node = output.data.grad_fn
    # Do not pretend a hook on a view/stack observes the recurrent backward.
    if node is None or "CudnnRnnBackward" not in node.name():
        name = None if node is None else node.name()
        raise RuntimeError(f"Expected a fused cuDNN GRU backward node for capture; found {name!r}.")
    node_name = node.name()  # Never capture the node in its own hook closure.
    data = packed.data.detach()
    batch_sizes = packed.batch_sizes
    sorted_indices = packed.sorted_indices
    unsorted_indices = packed.unsorted_indices

    def payload_factory():
        return dict(
            **_cpu(before), node_name=node_name,
            input_data=_cpu(data), batch_sizes=_cpu(batch_sizes),
            sorted_indices=_cpu(sorted_indices), unsorted_indices=_cpu(unsorted_indices),
            rnn_state_dict=_cpu(rnn.state_dict()),
            note="Actual failed GRU invocation. No loss reconstruction and no optimizer update. "
                 "Replay on different hardware/software is not guaranteed bitwise identical.",
        )

    node.register_hook(lambda grad_inputs, grad_outputs: capture_if_nonfinite(
        directory, payload_factory, grad_inputs, grad_outputs
    ))
