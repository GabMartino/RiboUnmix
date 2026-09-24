#!/usr/bin/env python3
"""Audit ``PCC(L_hat, K)`` on the exact main-figure cohorts and masks.

This is a read-only, post-hoc calculation over the frozen within-depth
synthetic experiments.  It deliberately reuses the run inventory, matched
validation cohorts, and evaluation-mask audit produced by
``analyze_synthetic_reference_target_audit.py``.  No checkpoint is loaded and
no model is trained.  The output is intended to accompany, rather than replace,
the more proximal ``L_hat``--``q_bar`` comparison.

The implementation streams prediction Parquet files one record batch at a
time.  Only one learned profile per transcript is evaluated; the source audit
has already verified that this shared profile and its mask are identical over
all dataset rows belonging to that transcript.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyses.analyze_synthetic_reference_target_audit import (  # noqa: E402
    DEPTH_ORDER,
    bootstrap_draws,
    bootstrap_interval,
    mean_one,
    profile_metrics,
    sha256,
    text_hash,
)


DEFAULT_AUDIT_DIR = (
    ROOT
    / "analyses/artifacts/synthetic/manuscript_revision/"
    "reference_target_best_val_loss"
)
DEFAULT_KINETICS = (
    ROOT
    / "Datasets/Synthetic_data/"
    "artificial_ground_truth_kinetics_target_mean_one.parquet"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "analyses/artifacts/manuscript_main_figures_20260924/kinetic_alignment"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--kinetics", type=Path, default=DEFAULT_KINETICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def load_kinetics(path: Path, transcript_ids: set[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reader = pq.ParquetFile(path)
    metadata = {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in (reader.metadata.metadata or {}).items()
    }
    expected_schema = "riboart.kinetics_target_profile.v1"
    if metadata.get("riboart.schema_version") != expected_schema:
        raise ValueError(
            f"Unexpected kinetic-target schema: {metadata.get('riboart.schema_version')!r}"
        )
    profiles: dict[str, np.ndarray] = {}
    maximum_mean_deviation = 0.0
    for batch in reader.iter_batches(
        columns=["sample", "transcript_id", "rib_profile"],
        batch_size=256,
        use_threads=False,
    ):
        for row in batch.to_pylist():
            transcript_id = str(row["transcript_id"])
            if transcript_id not in transcript_ids:
                continue
            if row["sample"] != "kinetics_target":
                raise ValueError(f"{transcript_id}: unexpected kinetic-target sample")
            if transcript_id in profiles:
                raise ValueError(f"{transcript_id}: duplicate kinetic target")
            values = np.asarray(row["rib_profile"], dtype=np.float64)
            if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
                raise ValueError(f"{transcript_id}: invalid kinetic target")
            if np.any(values <= 0):
                raise ValueError(f"{transcript_id}: nonpositive kinetic target")
            maximum_mean_deviation = max(
                maximum_mean_deviation, abs(float(values.mean()) - 1.0)
            )
            profiles[transcript_id] = values
    missing = sorted(transcript_ids.difference(profiles))
    if missing:
        raise KeyError(f"Kinetic target misses {len(missing)} transcripts: {missing[:5]}")
    if maximum_mean_deviation > 2e-12:
        raise ValueError(
            "Kinetic targets are not mean one within tolerance: "
            f"maximum deviation={maximum_mean_deviation:.3e}"
        )
    return profiles, {
        "schema_version": expected_schema,
        "n_profiles": len(profiles),
        "maximum_full_profile_mean_one_deviation": maximum_mean_deviation,
        "mean_one_tolerance": 2e-12,
    }


def iter_prediction_rows(path: Path, cohort: set[str], batch_size: int) -> Iterator[dict[str, Any]]:
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(
        columns=["transcript_id", "dataset_id", "mask", "L_bio"],
        batch_size=batch_size,
        use_threads=False,
    ):
        for row in batch.to_pylist():
            if str(row["transcript_id"]) in cohort:
                yield row


def mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


def analyze_run(
    run: pd.Series,
    cohort_ids: list[str],
    kinetics: dict[str, np.ndarray],
    audited_masks: set[tuple[str, int, int, str]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohort = set(cohort_ids)
    prediction_path = ROOT / str(run.prediction_path)
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    first_seen: set[str] = set()
    dataset_ids: dict[str, set[int]] = defaultdict(set)
    rows: list[dict[str, Any]] = []
    maximum_l_mean_deviation = 0.0
    maximum_k_mean_deviation = 0.0
    terminal_removed_count = 0
    for row in iter_prediction_rows(prediction_path, cohort, batch_size):
        transcript_id = str(row["transcript_id"])
        dataset_id = int(row["dataset_id"])
        if dataset_id in dataset_ids[transcript_id]:
            raise ValueError(f"{run.run_id}/{transcript_id}: duplicate dataset row {dataset_id}")
        dataset_ids[transcript_id].add(dataset_id)
        if transcript_id in first_seen:
            continue
        first_seen.add(transcript_id)
        k_full = kinetics[transcript_id]
        l_full = np.asarray(row["L_bio"], dtype=np.float64)
        mask_full = np.asarray(row["mask"], dtype=bool)
        if l_full.shape != mask_full.shape or not np.isfinite(l_full).all():
            raise ValueError(f"{run.run_id}/{transcript_id}: invalid L/mask export")
        if l_full.size == k_full.size + 1:
            l_sense = l_full[:-1]
            mask_sense = mask_full[:-1]
            terminal_removed = True
            terminal_removed_count += 1
        elif l_full.size == k_full.size:
            l_sense = l_full
            mask_sense = mask_full
            terminal_removed = False
        else:
            raise ValueError(
                f"{run.run_id}/{transcript_id}: L={l_full.size}, K={k_full.size}"
            )
        positions = np.arange(k_full.size)
        take = mask_sense & (positions >= 10) & (positions < k_full.size - 10)
        n_positions = int(take.sum())
        if n_positions < 3:
            raise ValueError(f"{run.run_id}/{transcript_id}: fewer than 3 retained codons")
        digest = mask_digest(take)
        mask_key = (transcript_id, int(l_full.size), int(k_full.size), digest)
        if mask_key not in audited_masks:
            raise ValueError(
                f"{run.run_id}/{transcript_id}: mask does not match the source audit"
            )
        l = mean_one(l_sense[take])
        k = mean_one(k_full[take])
        maximum_l_mean_deviation = max(maximum_l_mean_deviation, abs(float(l.mean()) - 1.0))
        maximum_k_mean_deviation = max(maximum_k_mean_deviation, abs(float(k.mean()) - 1.0))
        pcc = float(profile_metrics(l, k)["pearson"])
        reason = "" if np.isfinite(pcc) else "constant_profile"
        rows.append(
            {
                "run_id": str(run.run_id),
                "depth": str(run.depth),
                "n_datasets": int(run.n_datasets),
                "reference_weighting": str(run.reference_weighting),
                "training_seed": int(run.training_seed),
                "transcript_id": transcript_id,
                "evaluated_positions": n_positions,
                "terminal_entry_removed": terminal_removed,
                "mask_sha256": digest,
                "pcc_L_vs_K": pcc,
                "pcc_defined": bool(np.isfinite(pcc)),
                "undefined_reason": reason,
            }
        )
    missing = sorted(cohort.difference(first_seen))
    extra = sorted(first_seen.difference(cohort))
    if missing or extra:
        raise ValueError(
            f"{run.run_id}: cohort mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    expected_datasets = int(run.n_datasets)
    malformed = {
        transcript_id: len(ids)
        for transcript_id, ids in dataset_ids.items()
        if len(ids) != expected_datasets
    }
    if malformed:
        raise ValueError(
            f"{run.run_id}: incomplete dataset rows, examples={list(malformed.items())[:5]}"
        )
    return rows, {
        "run_id": str(run.run_id),
        "n_transcripts": len(rows),
        "terminal_entry_removed_count": terminal_removed_count,
        "maximum_L_mean_one_deviation": maximum_l_mean_deviation,
        "maximum_K_mean_one_deviation_after_mask": maximum_k_mean_deviation,
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_size_bytes": prediction_path.stat().st_size,
        "prediction_mtime_ns": prediction_path.stat().st_mtime_ns,
    }


def summarize(
    metrics: pd.DataFrame,
    cohort_by_depth: dict[str, list[str]],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for depth_index, depth in enumerate(DEPTH_ORDER):
        cohort = cohort_by_depth[depth]
        draws = bootstrap_draws(len(cohort), replicates, seed + depth_index)
        depth_metrics = metrics.loc[metrics.depth.eq(depth)]
        complete = depth_metrics.pivot(
            index="transcript_id", columns="n_datasets", values="pcc_L_vs_K"
        ).reindex(cohort)
        if list(complete.columns) != list(range(2, 11)):
            raise ValueError(f"{depth}: incomplete N=2,...,10 curve")
        if not np.isfinite(complete.to_numpy(float)).all():
            bad = int((~np.isfinite(complete.to_numpy(float))).sum())
            raise ValueError(
                f"{depth}: {bad} undefined PCC values prevent a fixed-cohort summary"
            )
        for n_datasets in range(2, 11):
            values = complete[n_datasets].to_numpy(float)
            low, high = bootstrap_interval(values, draws)
            run_row = depth_metrics.loc[depth_metrics.n_datasets.eq(n_datasets)].iloc[0]
            summaries.append(
                {
                    "run_id": run_row.run_id,
                    "depth": depth,
                    "n_datasets": n_datasets,
                    "reference_weighting": "equal",
                    "metric": "pearson",
                    "comparison": "L_vs_K",
                    "n_transcripts": len(values),
                    "n_undefined": 0,
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "mean": float(np.mean(values)),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_replicates": replicates,
                    "bootstrap_cluster": "transcript",
                    "bootstrap_seed": seed + depth_index,
                }
            )
    return pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    audit_dir = args.audit_dir.resolve()
    kinetics_path = args.kinetics.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.bootstrap_replicates <= 0 or args.batch_size <= 0:
        raise ValueError("Bootstrap replicates and batch size must be positive")

    manifest_path = audit_dir / "run_manifest.csv"
    cohort_path = audit_dir / "matched_transcript_ids.csv"
    masks_path = audit_dir / "evaluation_masks.csv"
    sanity_path = audit_dir / "sanity_checks.csv"
    source_provenance_path = audit_dir / "provenance.json"
    for path in (manifest_path, cohort_path, masks_path, sanity_path, source_provenance_path, kinetics_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    sanity = pd.read_csv(sanity_path)
    if (sanity["status"] == "FAIL").any():
        failed = sanity.loc[sanity.status.eq("FAIL"), "check"].tolist()
        raise ValueError(f"Source reference-target audit contains failures: {failed}")
    run_manifest = pd.read_csv(manifest_path)
    runs = run_manifest.loc[run_manifest.family.eq("within_depth")].copy()
    if (
        len(runs) != 27
        or not runs.reference_weighting.eq("equal").all()
        or not runs.training_seed.eq(42).all()
        or not runs.checkpoint_selection_metric.eq("val_loss").all()
    ):
        raise ValueError(
            "Expected 27 equal-reference, seed-42, best-validation-loss within-depth runs"
        )
    if not runs.groupby("depth").n_datasets.apply(
        lambda values: sorted(values.tolist()) == list(range(2, 11))
    ).all():
        raise ValueError("The within-depth run matrix is incomplete")

    cohort_table = pd.read_csv(cohort_path)
    cohort_by_depth = {
        depth: cohort_table.loc[
            cohort_table.cohort_key.eq(f"within_depth::{depth}"), "transcript_id"
        ].astype(str).tolist()
        for depth in DEPTH_ORDER
    }
    if any(len(ids) != len(set(ids)) for ids in cohort_by_depth.values()):
        raise ValueError("Matched cohort contains duplicate transcript IDs")
    union_ids = set().union(*(set(ids) for ids in cohort_by_depth.values()))

    source_provenance = json.loads(source_provenance_path.read_text(encoding="utf-8"))
    for depth, ids in cohort_by_depth.items():
        key = f"within_depth::{depth}"
        expected = source_provenance["cohorts"][key]
        if len(ids) != int(expected["n_transcripts"]) or text_hash(ids) != expected["identity_sha256"]:
            raise ValueError(f"{depth}: cohort does not match source-audit provenance")

    mask_table = pd.read_csv(masks_path)
    if not mask_table.boundary_trim_codons.eq(10).all():
        raise ValueError("Source audit did not use a uniform ten-codon trim")
    audited_masks = {
        (
            str(row.transcript_id),
            int(row.model_length),
            int(row.sense_length),
            str(row.mask_sha256),
        )
        for row in mask_table.itertuples(index=False)
    }
    kinetics, kinetic_validation = load_kinetics(kinetics_path, union_ids)

    metric_rows: list[dict[str, Any]] = []
    run_validations: list[dict[str, Any]] = []
    ordered_runs = runs.assign(
        depth_order=runs.depth.map({depth: i for i, depth in enumerate(DEPTH_ORDER)})
    ).sort_values(["depth_order", "n_datasets"])
    for index, run in enumerate(ordered_runs.itertuples(index=False), start=1):
        cohort = cohort_by_depth[str(run.depth)]
        print(
            f"[{index:02d}/27] {run.depth} N={run.n_datasets}: "
            f"{len(cohort):,} transcripts",
            flush=True,
        )
        rows, validation = analyze_run(
            pd.Series(run._asdict()), cohort, kinetics, audited_masks, args.batch_size
        )
        metric_rows.extend(rows)
        run_validations.append(validation)

    metrics = pd.DataFrame(metric_rows)
    expected_rows = sum(len(ids) for ids in cohort_by_depth.values()) * 9
    if len(metrics) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} metric rows, found {len(metrics):,}")
    summary = summarize(
        metrics,
        cohort_by_depth,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    metrics.to_parquet(output_dir / "L_vs_K_per_transcript.parquet", index=False)
    summary.to_csv(output_dir / "L_vs_K_summary.csv", index=False)
    validation_frame = pd.DataFrame(run_validations)
    validation_frame.to_csv(output_dir / "validation_checks.csv", index=False)

    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Measure transcript-level PCC(L_hat, K) for the frozen main-figure "
            "models on exactly the audited within-depth cohorts and masks."
        ),
        "claim_scope": (
            "Secondary alignment with upstream programmed kinetics; not the direct "
            "fixed-reference estimand and not an independent test-set result."
        ),
        "method": {
            "checkpoint_variant": "best validation loss",
            "training_seed": 42,
            "reference_weighting": "equal",
            "family": "within_depth",
            "depths": list(DEPTH_ORDER),
            "n_datasets": list(range(2, 11)),
            "terminal_convention": "remove one appended terminal entry when present",
            "boundary_trim_codons_each_end": 10,
            "normalization": "arithmetic mean one on the exact retained mask",
            "metric": "per-transcript Pearson correlation",
            "aggregation": "median across the fixed depth-specific transcript cohort",
            "interval": "95% percentile bootstrap over whole transcripts",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed_by_depth": {
                depth: args.bootstrap_seed + index
                for index, depth in enumerate(DEPTH_ORDER)
            },
        },
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                manifest_path,
                cohort_path,
                masks_path,
                sanity_path,
                source_provenance_path,
                kinetics_path,
            )
        },
        "kinetic_validation": kinetic_validation,
        "cohorts": {
            depth: {
                "n_transcripts": len(ids),
                "identity_sha256": text_hash(ids),
            }
            for depth, ids in cohort_by_depth.items()
        },
        "undefined_pcc": int((~metrics.pcc_defined).sum()),
        "source_audit_failures": 0,
        "models_loaded": False,
        "models_retrained": False,
        "prediction_inputs": run_validations,
        "failure_modes": [
            "validation cohorts were also used for best-validation-loss checkpoint selection",
            "one training seed and one cumulative bias order are available",
            "K is upstream of stochastic TASEP traffic and is not the direct L estimand",
            "bootstrap intervals condition on the 27 fitted models and their fixed splits",
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# Matched synthetic kinetic-alignment audit",
        "",
        "This read-only calculation uses the same 27 best-validation-loss, "
        "equal-reference fits, depth-specific cohorts, terminal convention, and "
        "ten-codon masks as the main shared-profile figure.",
        "",
        "The result is a secondary mechanistic comparison: $K_t$ precedes stochastic "
        "TASEP traffic, so PCC($\\widehat L_t$, $K_t$) must not be described as "
        "recovery of the direct learned-profile estimand.",
        "",
        f"Undefined PCC values: **{int((~metrics.pcc_defined).sum())}**.",
        "",
        "| depth | N=2 median PCC | N=10 median PCC |",
        "|---|---:|---:|",
    ]
    for depth in DEPTH_ORDER:
        selected = summary.loc[summary.depth.eq(depth)].set_index("n_datasets")
        report_lines.append(
            f"| {depth} | {selected.loc[2, 'median']:.4f} | "
            f"{selected.loc[10, 'median']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "Intervals in `L_vs_K_summary.csv` are 2,000-resample whole-transcript "
            "bootstrap intervals conditional on these fitted models.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'L_vs_K_summary.csv'}")


if __name__ == "__main__":
    main()
