#!/usr/bin/env python3
"""Measure actual GRU precision and isolate a controlled overflow mechanism.

This constructed recurrence is NOT a replay of a training failure. At x=h=0,
the forward GRU remains exactly zero, but its forward-direction derivative is
dh_t/dh_{t-1} = 0.5 + 0.25*2.2 = 1.05 before rounding. The incoming loss
adjoint is small and finite. Thus any failure can be localized to backward,
independently of loss magnitude, data, optimizer, or exponentiating log-mu.
The autocast_bf16 baseline explicitly bypasses the new encoder protection;
protected_inherit and the TBPTT cases use the current production code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from diagnose_bias_gru_cuda import tensor_stats, trace_gru
from Models.RiboUnmixModel.DatasetBiasSubmodel import BiGRUContextEncoder
from Models.utils.stable_numerics import nb2_nll_from_log_mean
from Models.utils.gru_precision import GRU_COMPUTE_POLICY


def run_case(mode, length, capture_dir):
    model = BiGRUContextEncoder(1, 1).cuda().train()
    model.output_norm = torch.nn.Identity()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.rnn.weight_hh_l0[2, 0] = 2.2
    if mode == "gru_fp32":
        model.precision = "float32"
    elif mode.startswith("tbptt"):
        model.tbptt_window = int(mode.removeprefix("tbptt"))
    elif mode == "native_bf16":
        # Explicit BF16 parameters/input bypass cuDNN eligibility in torch 2.10.
        # This tests backend capability, not an optimizer implementation.
        model.bfloat16()
    if mode == "autocast_bf16":
        model.failure_capture_dir = str(capture_dir)
    dtype = torch.bfloat16 if mode == "native_bf16" else torch.float32
    x = torch.zeros(1, 1, length, device="cuda", dtype=dtype, requires_grad=True)
    mask = torch.ones(1, length, device="cuda", dtype=torch.bool)
    traces = []
    handle = trace_gru(model.rnn, traces)
    row = dict(mode=mode, length=length, theoretical_local_gain=1.05)
    start = time.monotonic()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode != "native_bf16"):
            out = (model._forward_impl(x, mask, (length,)) if mode == "autocast_bf16"
                   else model(x, mask, (length,)))
            eta = out[0, 0, -1].float()
            loss = 0.002 * nb2_nll_from_log_mean(eta.new_tensor(2), eta, eta.new_tensor(0))
            row.update(output=tensor_stats(out), loss=float(loss.detach()))
        loss.backward()
        gradients = {k: tensor_stats(p.grad) for k, p in model.named_parameters() if p.grad is not None}
        row["gradients"] = gradients
        row["nonfinite_parameters"] = [k for k, v in gradients.items() if not v["finite"]]
    except (RuntimeError, FloatingPointError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        handle.remove()
    row.update(gru_traces=traces, seconds=time.monotonic() - start)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=int, default=1013)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                 gru_compute_policy=GRU_COMPUTE_POLICY,
                 cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
                 float16_max=torch.finfo(torch.float16).max,
                 bfloat16_max=torch.finfo(torch.bfloat16).max, note=__doc__)
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    with (args.output_dir / "probes.jsonl").open("x") as stream:
        for mode in ("autocast_bf16", "protected_inherit", "gru_fp32", "native_bf16", "tbptt1024", "tbptt512", "tbptt256"):
            row = run_case(mode, args.length, args.output_dir / "captures")
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps({k: v for k, v in row.items() if k not in ("gradients", "gru_traces")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
