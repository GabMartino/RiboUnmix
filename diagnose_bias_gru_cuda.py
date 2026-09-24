#!/usr/bin/env python3
"""Read-only checkpoint/sequence CUDA probe. No training or optimizer updates.

Stage 'bias' uses an explicitly diagnostic NB loss (y=2, alpha=1, raw log-gamma
as log-mean), not a reconstruction of the production training objective.
Full-GRU failures are captured with their actual incoming adjoints for replay.
Mode 'bf16' names the REQUESTED autocast dtype, not the measured GRU dtype.
Current model code protects recurrence in FP32; inspect gru_traces and the
recorded compute-policy version when comparing with pre-fix reports.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import pyarrow.parquet as pq
import torch
import yaml

from Models.RiboUnmixModel.RiboUnmixModel import RiboUnmixModel
from Models.utils.stable_numerics import nb2_nll_from_log_mean
from Models.utils.gru_precision import GRU_COMPUTE_POLICY

ROOT = Path(__file__).resolve().parent


def load_model(task, checkpoint=None):
    config = yaml.safe_load((task / "resolved_config.yaml").read_text())
    if checkpoint is None:
        checkpoint = next((task / "checkpoints").rglob("last.ckpt"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {k.removeprefix("model."): v for k, v in payload["state_dict"].items()
             if k.startswith("model.")}
    extra = state["_extra_state"]
    enc = {key: yaml.safe_load((ROOT / config["paths"]["encodings"][name]).read_text())
           for key, name in (("nt_encoding", "nt"), ("codon_to_aa_encoding", "codon_to_aa"),
                             ("codon_encoding", "codon"), ("aa_encoding", "aa"))}
    model = RiboUnmixModel(
        model_configs=config["model"],
        selected_dataset_names=extra["selected_dataset_names"],
        selected_dataset_ids=extra["selected_dataset_ids"],
        reference_dataset_names=extra["gamma_reference_dataset_names"],
        reference_dataset_ids=state["gamma_reference_dataset_ids"].tolist(),
        **enc,
    )
    model.load_state_dict(state, strict=True)  # Restores exact reference weights/gauge.
    bad_weights = [k for k, v in state.items() if torch.is_tensor(v) and not bool(v.isfinite().all())]
    bad_optimizer = []
    for opt in payload.get("optimizer_states", []):
        for index, values in opt["state"].items():
            for key, value in values.items():
                if torch.is_tensor(value) and not bool(value.isfinite().all()):
                    bad_optimizer.append([index, key])
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    audit = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest,
                 epoch=payload["epoch"], global_step=payload["global_step"],
                 bad_weights=bad_weights, bad_optimizer=bad_optimizer,
                 saved_hyperparameters=payload.get("hyper_parameters", {}))
    if bad_weights or bad_optimizer:
        raise ValueError(f"Nonfinite saved state: {audit}")
    return model, config, enc, audit


def tensor_stats(value):
    if value is None:
        return None
    value = value.detach()
    finite = bool(value.isfinite().all())
    return dict(shape=list(value.shape), dtype=str(value.dtype), finite=finite,
                max_abs=float(value.abs().max()) if finite and value.numel() else None)


def trace_gru(rnn, traces, *, branch="bias"):
    def after_forward(_module, _args, output):
        node = output[0].data.grad_fn
        row = dict(branch=branch, node=node.name(), output=tensor_stats(output[0].data),
                   autocast_enabled=torch.is_autocast_enabled("cuda"),
                   autocast_dtype=str(torch.get_autocast_dtype("cuda")))
        traces.append(row)
        def backward(inputs, outputs):
            row["incoming_adjoints"] = [tensor_stats(x) for x in outputs]
            row["generated_gradients"] = [tensor_stats(x) for x in inputs]
        node.register_hook(backward)
    return rnn.register_forward_hook(after_forward)


def probe_bias(model, codons, dataset_ids, mode, *, seed, capture_dir, dropout):
    encoder = model.dataset_bias_model.local_context_gru
    encoder.precision = "float32" if mode == "fp32" else "inherit"
    encoder.tbptt_window = int(mode.removeprefix("tbptt")) if mode.startswith("tbptt") else 0
    encoder.failure_capture_dir = str(capture_dir) if encoder.tbptt_window == 0 else None
    model.train()
    if not dropout:
        for layer in model.modules():
            if isinstance(layer, torch.nn.Dropout):
                layer.eval()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    ids = torch.tensor(dataset_ids, device="cuda")
    x = codons.cuda().expand(len(ids), -1)
    mask = torch.ones_like(x, dtype=torch.bool)
    traces = []
    handle = trace_gru(encoder.rnn, traces)
    result = dict(mode=mode, dataset_ids=dataset_ids, length=x.size(1), seed=seed,
                  dropout=dropout, diagnostic_loss="NB y=2, alpha=1, eta=raw log-gamma")
    start = time.monotonic()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bias = model.dataset_bias_model(
                dataset_ids=ids, mask=mask, codon_ids=x,
                position_features=model.make_position_features(mask=mask, dtype=torch.float32),
                embedding_center_ids=model.gamma_selected_dataset_ids,
                compute_log_sigma=False, cpu_lengths=(x.size(1),) * len(ids),
            )
            eta = bias["gamma_raw"]
            loss = nb2_nll_from_log_mean(torch.full_like(eta, 2), eta, torch.zeros_like(eta)).mean()
        result.update(loss=float(loss.detach()), output=tensor_stats(eta))
        loss.backward()
        gradients = {k: tensor_stats(p.grad) for k, p in model.dataset_bias_model.named_parameters()
                     if p.grad is not None}
        result["nonfinite_parameters"] = [k for k, v in gradients.items() if not v["finite"]]
        result["max_abs_gradient"] = max(v["max_abs"] or 0 for v in gradients.values())
    except (RuntimeError, FloatingPointError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        handle.remove()
    result.update(gru_traces=traces, seconds=time.monotonic() - start,
                  peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--transcripts", required=True)
    p.add_argument("--dataset-ids", help="Default: all selected datasets, in checkpoint order.")
    p.add_argument("--dataset-chunk", type=int, default=8)
    p.add_argument("--modes", default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dropout", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model, config, enc, audit = load_model(args.task_dir, args.checkpoint)
    audit.update(torch=str(torch.__version__), cuda=torch.version.cuda,
                 gru_compute_policy=GRU_COMPUTE_POLICY,
                 cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                 task_dir=str(args.task_dir.resolve()), args={k: str(v) for k, v in vars(args).items()},
                 limitation="Saved checkpoint/real sequences; diagnostic loss, not epoch failure replay.")
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("epoch", "global_step", "gpu", "bad_weights", "bad_optimizer")}), flush=True)
    ids = ([int(i) for i in args.dataset_ids.split(",")] if args.dataset_ids
           else model.gamma_selected_dataset_ids.tolist())
    transcripts = args.transcripts.split(",")
    sequences = pq.read_table(
        ROOT / "Datasets/data/sequence/MANE.selection.sequence_embeddings_with_css.parquet",
        columns=["transcript_id", "codons"], filters=[("transcript_id", "in", transcripts)],
    ).to_pylist()
    sequences = {r["transcript_id"]: r["codons"] for r in sequences}
    model.cuda()
    with (args.output_dir / "probes.jsonl").open("x") as stream:
        for transcript in transcripts:
            codons = torch.tensor([[enc["codon_encoding"][c] for c in sequences[transcript]]])
            for start in range(0, len(ids), args.dataset_chunk):
                for mode in args.modes.split(","):
                    if mode not in {"bf16", "fp32", "tbptt1024", "tbptt512", "tbptt256"}:
                        raise ValueError(mode)
                    row = probe_bias(model, codons, ids[start:start + args.dataset_chunk], mode,
                                     seed=args.seed, capture_dir=args.output_dir / "captures",
                                     dropout=args.dropout)
                    row["transcript"] = transcript
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    print(json.dumps({k: v for k, v in row.items() if k not in ("gru_traces", "output")}), flush=True)
                    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
