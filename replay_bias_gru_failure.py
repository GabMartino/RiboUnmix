#!/usr/bin/env python3
"""Replay a saved failed GRU operation, not an epoch and not a synthetic loss.

Only load trusted captures produced by this repository: torch.load uses pickle.
No model/optimizer update, checkpoint replacement, or training-result mutation.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import time

import torch
from torch.nn.utils.rnn import PackedSequence


def replay(payload, *, device, mode, disable_cudnn=False, return_gradients=False):
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported GRU capture format.")
    if mode not in {"original", "float32", "float64"}:
        raise ValueError(f"Unknown precision mode: {mode}")
    if payload["rnn_config"]["dropout"] != 0:
        raise ValueError("Cannot exactly restore cuDNN internal dropout state; require dropout=0.")
    adjoints = payload["grad_outputs"]
    if any(x is not None and bool(x.any()) for x in adjoints[2:]):
        raise ValueError("Unexpected nonzero adjoint for a cuDNN auxiliary output.")
    torch.manual_seed(0)  # Initialization is overwritten; restore forward RNG below.
    original_dtype = next(iter(payload["rnn_state_dict"].values())).dtype
    dtype = original_dtype if mode == "original" else getattr(torch, mode)
    rnn = torch.nn.GRU(**payload["rnn_config"]).to(device=device, dtype=dtype).train()
    rnn.load_state_dict(payload["rnn_state_dict"], strict=True)
    data_dtype = payload["input_data"].dtype if mode == "original" else dtype
    data = payload["input_data"].to(device=device, dtype=data_dtype).detach().requires_grad_(True)
    packed = PackedSequence(data, payload["batch_sizes"].cpu(),
                            None if payload["sorted_indices"] is None else payload["sorted_indices"].to(device),
                            None if payload["unsorted_indices"] is None else payload["unsorted_indices"].to(device))
    amp_dtype = getattr(torch, payload["autocast_dtype"].removeprefix("torch."))
    torch.set_rng_state(payload["rng_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(payload["rng_cuda"].cpu(), device)
    cudnn_context = torch.backends.cudnn.flags(
        enabled=payload["cudnn_enabled"] and not disable_cudnn,
        benchmark=payload["cudnn_benchmark"],
        deterministic=payload["cudnn_deterministic"],
        allow_tf32=payload["cudnn_allow_tf32"],
    ) if device.type == "cuda" else contextlib.nullcontext()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    start = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = payload["matmul_allow_tf32"]
        with cudnn_context:
            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=mode == "original" and payload["autocast_enabled"]):
                output, hidden = rnn(packed)
            # Fused cuDNN hidden-state adjoints use sorted batch order.
            if packed.sorted_indices is not None:
                hidden = hidden.index_select(1, packed.sorted_indices)
            used_outputs, used_adjoints = [], []
            for value, adjoint in zip((output.data, hidden), adjoints[:2]):
                if adjoint is not None:
                    used_outputs.append(value)
                    used_adjoints.append(adjoint.to(device=device, dtype=value.dtype))
            if not used_outputs:
                raise ValueError("Capture contains no incoming output/hidden adjoint.")
            torch.autograd.backward(used_outputs, used_adjoints)
        gradients = {"input_data": data.grad, **{name: p.grad for name, p in rnn.named_parameters()}}
        stats = {}
        for name, gradient in gradients.items():
            if gradient is None:
                stats[name] = {"present": False}
            else:
                finite = bool(gradient.isfinite().all())
                stats[name] = {"present": True, "finite": finite,
                               "max_abs": float(gradient.abs().max()) if finite else None}
        result = dict(mode=mode, device=str(device), disable_cudnn=disable_cudnn,
                      output_finite=bool(output.data.isfinite().all()),
                      nonfinite_gradients=[name for name, stat in stats.items() if stat.get("finite") is False],
                      gradients=stats, seconds=time.monotonic() - start)
        if return_gradients:  # For full VJP regression checks, never the JSON CLI.
            result["gradient_tensors"] = {name: None if g is None else g.detach().cpu().clone()
                                          for name, g in gradients.items()}
        return result
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--modes", default="original,float32,float64")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("No CUDA GPU available. CPU replay uses a different kernel and is not a CUDA reproduction.")
    report = dict(capture=str(args.capture.resolve()), captured_origin=payload["origin"],
                  captured_torch=payload["torch_version"], replay_torch=str(torch.__version__),
                  captured_gpu=payload["gpu_name"],
                  replay_gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  note="Same captured input, weights and incoming adjoints; no training or optimizer step. "
                       "Different precision/hardware/software can change the trajectory.", results=[])
    for mode in args.modes.split(","):
        if mode not in {"original", "float32", "float64"}:
            parser.error(f"Unknown mode: {mode}")
        try:
            result = replay(payload, device=device, mode=mode, disable_cudnn=args.disable_cudnn)
        except Exception as error:
            result = dict(mode=mode, error=f"{type(error).__name__}: {error}")
        report["results"].append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Diagnostics must not overwrite an earlier result without an explicit new name.
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    return int(any("error" in result or result.get("nonfinite_gradients") for result in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
