#!/usr/bin/env python3
"""Probe a saved checkpoint with real counts and the production training loss.

No optimizer steps or checkpoint changes. Each probe contains all observed
dataset pairs for one transcript, and all saved fixed-reference datasets. This
preserves the transcript-balanced objective but not the cluster's batch shape,
dropout realization, or updates between the checkpoint and the reported crash.
The 'bf16' label is the REQUESTED autocast setting; gru_traces records actual
output precision for both branches. Current code protects both GRUs in FP32.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import PackedSequence
import yaml

from diagnose_bias_gru_cuda import ROOT, load_model, tensor_stats, trace_gru
from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDataset import RiboUnmixMultiDataset
from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import load_dataset_quality_ranking
from Models.RiboUnmixLightningModule import RiboUnmixLightningModule
from Models.utils.gru_precision import GRU_COMPUTE_POLICY
from Utils.reliability_references import apply_dataset_reliability_reference


def local_data_path(path):
    """Relocate only the copied project's Datasets paths, not arbitrary files."""
    path = Path(path)
    if not path.is_absolute():
        return ROOT / path
    if path.exists():
        return path
    return ROOT / "Datasets" / str(path).split("/Datasets/", 1)[1]


def load_real_dataset(task, model, config, enc, transcripts):
    sequences = pq.read_table(
        local_data_path(config["paths"]["sequences_path"]),
        columns=["transcript_id", "codons", "conserved_stalling_sites"],
        filters=[("transcript_id", "in", transcripts)],
    ).to_pylist()
    sequences = {r["transcript_id"]: r for r in sequences}
    data = dict(
        transcript_id=transcripts, sequence_representation="codon_tokens",
        ref=[sequences[t]["codons"] for t in transcripts],
        css=[sequences[t]["conserved_stalling_sites"] for t in transcripts],
        ribo_profiles={t: {} for t in transcripts},
        ribo_replicas={t: {} for t in transcripts},
        sample_weights={t: {} for t in transcripts},
    )
    references = json.loads((task / "reliability_reference_manifest.json").read_text())["datasets"]
    quality = config["data"]["dataset_quality_ranking"]
    # Use the frozen table copied with this experiment, never refit a ranking.
    ranking_path = task.parent.parent / Path(quality["path"]).name
    if not ranking_path.exists():
        ranking_path = local_data_path(quality["path"])
    data["dataset_quality_ranks"], data["dataset_quality_weights"] = load_dataset_quality_ranking(
        str(ranking_path), dataset_column=quality["dataset_column"], rank_column=quality["rank_column"],
    )
    source_rows = []
    for i, name in enumerate(model.selected_dataset_names):
        path = local_data_path(config["dataset_config"]["dataset_path"][name])
        frame = pq.read_table(
            path, columns=["id", "ribo", "ribo_cds_replicas", "weight", "read_density", "coverage"],
            filters=[("id", "in", transcripts)], use_threads=False,
        ).to_pandas()
        # Same eligibility and arithmetic-replica consensus as production.
        frame = frame.loc[frame["weight"] > 0].copy()
        frame = frame.loc[frame["ribo"].map(lambda x: np.asarray(x).sum(dtype=np.float64) > 0)]
        weights = apply_dataset_reliability_reference(frame, dataset_name=name, reference=references[name])
        for row, weight in zip(frame.itertuples(index=False), weights, strict=True):
            replicas = np.asarray(list(row.ribo_cds_replicas), dtype=np.float32)
            data["ribo_profiles"][row.id][name] = replicas.mean(axis=0)
            data["ribo_replicas"][row.id][name] = replicas
            data["sample_weights"][row.id][name] = float(weight)
            source_rows.append(dict(transcript_id=row.id, dataset=name, replicas=len(replicas),
                                    length=replicas.shape[1], reliability_weight=float(weight),
                                    mean_count=float(replicas.mean()), max_count=float(replicas.max())))
        if i % 20 == 0:
            print(f"Loaded {i + 1}/{len(model.selected_dataset_names)} count files", flush=True)
    dataset_encoding = yaml.safe_load(local_data_path(config["paths"]["encodings"]["datasets"]).read_text())
    dataset = RiboUnmixMultiDataset(
        **enc, datasets_encoding=dataset_encoding, transcripts_ids=transcripts,
        data=data, lengths=np.array([len(x) for x in data["ref"]]),
        additional_sequence_features=config["model"]["additional_sequence_features"],
    )
    return dataset, dataset_encoding, source_rows


def probe(module, batch_cpu, mode, seed, capture_dir, gradient_scale, *, offload=False):
    encoder = module.model.dataset_bias_model.local_context_gru
    encoder.precision = "float32" if mode == "fp32" else "inherit"
    encoder.tbptt_window = int(mode.removeprefix("tbptt")) if mode.startswith("tbptt") else 0
    encoder.failure_capture_dir = str(capture_dir) if encoder.tbptt_window == 0 else None
    module.train()  # Keep training dropout and the training reference-panel path.
    module.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    batch = tuple(v.cuda() if isinstance(v, (torch.Tensor, PackedSequence)) else v for v in batch_cpu)
    traces = []
    handles = [trace_gru(encoder.rnn, traces),
               trace_gru(module.model.biological_model.rnn, traces, branch="biology")]
    row = dict(mode=mode, seed=seed, gradient_scale=gradient_scale,
               pair_count=len(batch_cpu[1]), transcript_ids=sorted(set(batch_cpu[1])),
               saved_tensor_cpu_offload=offload)
    start = time.monotonic()
    try:
        # Optional lossless activation storage for the 8-GB diagnostic GPU;
        # forward shapes, precision, and the objective are unchanged.
        storage = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext()
        with storage, torch.autocast("cuda", dtype=torch.bfloat16):
            out = module._forward_batch(batch)
            metrics = module._compute_loss_and_metrics(out, optimize_with_reweighted_nb=True)
            loss = metrics["loss"] * gradient_scale
        row["loss_terms"] = {k: float(metrics[k].detach()) for k in
                             ("loss", "nll", "pcc_raw_loss", "pcc_nb_vst_loss")}
        loss.backward()
        gradients = {k: tensor_stats(p.grad) for k, p in module.named_parameters() if p.grad is not None}
        row["nonfinite_parameters"] = [k for k, v in gradients.items() if not v["finite"]]
        row["max_abs_gradient"] = max(v["max_abs"] or 0 for v in gradients.values())
        row["gradients"] = gradients
    except (RuntimeError, FloatingPointError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        for handle in handles:
            handle.remove()
    row.update(gru_traces=traces, seconds=time.monotonic() - start,
               peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--prepared-batches", type=Path, help="Reuse this diagnostic's saved CPU batches for the same checkpoint.")
    p.add_argument("--offload-saved-tensors", action="store_true", help="Store backward activations on CPU to fit diagnostic GPUs; no precision or loss changes.")
    p.add_argument("--transcripts", required=True)
    p.add_argument("--modes", default="bf16")
    p.add_argument("--seeds", default="42")
    p.add_argument("--logical-groups", type=int, default=33)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model, config, enc, audit = load_model(args.task_dir, args.checkpoint)
    transcripts = args.transcripts.split(",")
    if args.prepared_batches:
        prepared = torch.load(args.prepared_batches, map_location="cpu", weights_only=False)
        if prepared["checkpoint_sha256"] != audit["checkpoint_sha256"]:
            raise ValueError("Prepared batches belong to a different checkpoint.")
        batches, dataset_encoding, source_rows = (prepared[k] for k in ("batches", "dataset_encoding", "source_rows"))
    else:
        dataset, dataset_encoding, source_rows = load_real_dataset(args.task_dir, model, config, enc, transcripts)
        batches = {t: dataset.collate_fn([dataset[int(i)] for i in np.flatnonzero(dataset.flat_transcript_ids == t)])
                   for t in transcripts}
        torch.save(dict(batches=batches, dataset_encoding=dataset_encoding, source_rows=source_rows,
                        checkpoint_sha256=audit["checkpoint_sha256"]), args.output_dir / "prepared_batches.pt")
    pd.DataFrame(source_rows).to_csv(args.output_dir / "observed_pairs.csv", index=False)
    module = RiboUnmixLightningModule(model, OmegaConf.create(config), dataset_encoding).cuda()
    audit.update(torch=str(torch.__version__), cuda=torch.version.cuda,
                 gru_compute_policy=GRU_COMPUTE_POLICY,
                 cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                 loss_config=config["loss"], args={k: str(v) for k, v in vars(args).items()},
                 limitation=__doc__)
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    with (args.output_dir / "probes.jsonl").open("x") as stream:
        for transcript in transcripts:
            batch = batches[transcript]
            for seed in map(int, args.seeds.split(",")):
                for mode in args.modes.split(","):
                    if mode not in {"bf16", "fp32", "tbptt1024", "tbptt512", "tbptt256"}:
                        raise ValueError(mode)
                    row = probe(module, batch, mode, seed, args.output_dir / "captures", 1 / args.logical_groups,
                                offload=args.offload_saved_tensors)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    print(json.dumps({k: v for k, v in row.items() if k not in ("gradients", "gru_traces")}), flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
