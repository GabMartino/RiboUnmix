#!/usr/bin/env python3
"""Compare every programmed synthetic bias across replicas, truth, and peers.

The analysis consumes the scalar transcript metrics produced by
``analyze_synthetic_input_data.py`` for the replica/truth box plots.  It then
streams the replica-aware weighted Parquets to calculate, within each read
depth, every pairwise dataset-consensus correlation on each transcript.

Cross-dataset work is parallelized by read depth.  Each worker keeps only the
10 biased consensus profiles for one current transcript, avoiding a dense
transcript-by-position-by-dataset array.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import html
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Iterator

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyses.analyze_synthetic_input_data import iter_rows, sha256
from analyses.analyze_synthetic_single_dataset_mu_pcc import (
    BIAS_TYPES as MODEL_BIASES,
    DEPTHS as MODEL_DEPTHS,
    RunArtifact,
    analyze_prediction,
    discover_run_directory,
    resolve_prediction_artifact,
)
from Utils.publication_plot_style import publication_rc


WEIGHTED_ROOT = ROOT / "Datasets" / "data" / "weighted_synthetic"
DEFAULT_INPUT_METRICS = (
    ROOT
    / "analyses"
    / "artifacts"
    / "synthetic"
    / "input_data"
    / "per_transcript_metrics.parquet"
)
DEFAULT_OUTPUT = ROOT / "analyses" / "artifacts" / "synthetic" / "individual_dataset"
DEFAULT_SINGLE_MODEL_RESULTS = ROOT / "results" / "riboai_synthetic_experiments"
DEFAULT_SINGLE_MODEL_RUN_ID = "single_20260830_205110"
DEPTHS: tuple[tuple[str, float, str], ...] = (
    ("0p25_per_codon", 0.25, "0.25 reads/codon"),
    ("2_per_codon", 2.0, "2 reads/codon"),
    ("20_per_codon", 20.0, "20 reads/codon"),
)

AVAILABLE_DATASET_ORDER = (
    "artificial_ground_truth",
    "artificial_bias_5prime_aa",
    "artificial_bias_5prime_cc",
    "artificial_bias_5prime_gg",
    "artificial_bias_5prime_uu",
    "artificial_bias_3prime_aa",
    "artificial_bias_3prime_cc",
    "artificial_bias_3prime_gg",
    "artificial_bias_3prime_uu",
    "artificial_bias_au_fraction_gt_0p7",
    "artificial_bias_gc_fraction_gt_0p7",
)
DATASET_ORDER = tuple(
    dataset
    for dataset in AVAILABLE_DATASET_ORDER
    if dataset != "artificial_ground_truth"
)
DISPLAY_NAMES = {
    "artificial_bias_5prime_aa": "5′ AA",
    "artificial_bias_5prime_cc": "5′ CC",
    "artificial_bias_5prime_gg": "5′ GG",
    "artificial_bias_5prime_uu": "5′ UU",
    "artificial_bias_3prime_aa": "3′ AA",
    "artificial_bias_3prime_cc": "3′ CC",
    "artificial_bias_3prime_gg": "3′ GG",
    "artificial_bias_3prime_uu": "3′ UU",
    "artificial_bias_au_fraction_gt_0p7": "AU fraction > 0.7",
    "artificial_bias_gc_fraction_gt_0p7": "GC fraction > 0.7",
}
PLOT_DISPLAY_NAMES = {
    "artificial_bias_5prime_aa": r"$5^\prime$ AA",
    "artificial_bias_5prime_cc": r"$5^\prime$ CC",
    "artificial_bias_5prime_gg": r"$5^\prime$ GG",
    "artificial_bias_5prime_uu": r"$5^\prime$ UU",
    "artificial_bias_3prime_aa": r"$3^\prime$ AA",
    "artificial_bias_3prime_cc": r"$3^\prime$ CC",
    "artificial_bias_3prime_gg": r"$3^\prime$ GG",
    "artificial_bias_3prime_uu": r"$3^\prime$ UU",
    "artificial_bias_au_fraction_gt_0p7": r"AU fraction $>0.7$",
    "artificial_bias_gc_fraction_gt_0p7": r"GC fraction $>0.7$",
}
REPRESENTATIONS = (
    ("rep1_kinetic_pcc", "Replica 1", "#3676B8"),
    ("rep2_kinetic_pcc", "Replica 2", "#E6862F"),
    ("consensus_kinetic_pcc", "Arithmetic mean", "#3A923A"),
)
CROSS_SCHEMA = pa.schema(
    [
        ("depth", pa.string()),
        ("nominal_reads_per_codon", pa.float64()),
        ("transcript_id", pa.string()),
        ("dataset_a", pa.string()),
        ("dataset_b", pa.string()),
        ("consensus_pcc", pa.float32()),
    ]
)


def discover_datasets(depth: str) -> list[str]:
    available = {path.stem for path in (WEIGHTED_ROOT / depth).glob("*.parquet")}
    expected = set(AVAILABLE_DATASET_ORDER)
    if available != expected:
        raise ValueError(
            f"{depth}: expected datasets differ from files; "
            f"missing={sorted(expected-available)}, extra={sorted(available-expected)}"
        )
    # Keep K_t as the deterministic target, but do not display the unbiased
    # sampled observation condition alongside the ten programmed biases.
    return list(DATASET_ORDER)


def box_statistics(values: Iterable[float], label: str) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "label": label,
            "whislo": np.nan,
            "q1": np.nan,
            "med": np.nan,
            "q3": np.nan,
            "whishi": np.nan,
            "fliers": [],
        }
    p05, p25, p50, p75, p95 = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "label": label,
        "whislo": float(p05),
        "q1": float(p25),
        "med": float(p50),
        "q3": float(p75),
        "whishi": float(p95),
        "fliers": [],
    }


def centered_correlation_matrix(profiles: np.ndarray) -> np.ndarray:
    """Return row-wise Pearson correlations for aligned dataset profiles."""
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim != 2 or profiles.shape[1] < 3:
        raise ValueError(f"Expected [dataset, position] profiles, got {profiles.shape}")
    if not np.isfinite(profiles).all():
        raise ValueError("Cross-dataset profiles contain non-finite values")
    centered = profiles - profiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    denominator = np.outer(norms, norms)
    correlations = np.full(denominator.shape, np.nan, dtype=np.float64)
    np.divide(
        centered @ centered.T,
        denominator,
        out=correlations,
        where=denominator > np.finfo(np.float64).eps * profiles.shape[1],
    )
    finite = np.isfinite(correlations)
    correlations[finite] = np.clip(correlations[finite], -1.0, 1.0)
    return correlations


def _flush_cross_rows(
    writer: pq.ParquetWriter, rows: list[dict[str, Any]]
) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=CROSS_SCHEMA))
        rows.clear()


def cross_dataset_worker(
    depth: str,
    nominal_depth: float,
    output_path_string: str,
    batch_size: int,
    max_transcripts: int | None,
) -> dict[str, Any]:
    """Process one complete read depth in an isolated process."""
    output_path = Path(output_path_string)
    datasets = discover_datasets(depth)
    paths = [WEIGHTED_ROOT / depth / f"{dataset}.parquet" for dataset in datasets]
    row_counts = [pq.ParquetFile(path).metadata.num_rows for path in paths]
    if len(set(row_counts)) != 1:
        raise ValueError(f"{depth}: weighted tables have unequal row counts {row_counts}")
    expected_rows = row_counts[0]
    if max_transcripts is not None:
        expected_rows = min(expected_rows, max_transcripts)

    iterators = [
        iter_rows(path, ["id", "ribo_cds_replicas"], batch_size=batch_size)
        for path in paths
    ]
    pair_indices = [(i, j) for i in range(len(datasets)) for j in range(i + 1, len(datasets))]
    pair_values = np.full((expected_rows, len(pair_indices)), np.nan, dtype=np.float32)
    exact_position_counts = np.zeros(len(pair_indices), dtype=np.int64)
    total_position_counts = np.zeros(len(pair_indices), dtype=np.int64)
    identical_profile_counts = np.zeros(len(pair_indices), dtype=np.int64)
    buffer: list[dict[str, Any]] = []
    processed = 0
    with pq.ParquetWriter(
        output_path, CROSS_SCHEMA, compression="zstd", use_dictionary=True
    ) as writer:
        for row_number, aligned_rows in enumerate(zip(*iterators)):
            if max_transcripts is not None and row_number >= max_transcripts:
                break
            ids = [str(row["id"]) for row in aligned_rows]
            if len(set(ids)) != 1:
                raise ValueError(f"{depth}: transcript streams are misaligned: {ids}")
            transcript_id = ids[0]
            profiles: list[np.ndarray] = []
            sense_length: int | None = None
            for dataset, row in zip(datasets, aligned_rows):
                replicas = np.asarray(row["ribo_cds_replicas"], dtype=np.float64)
                if replicas.ndim != 2 or replicas.shape[0] != 2:
                    raise ValueError(
                        f"{depth}/{dataset}/{transcript_id}: expected exactly two replicas"
                    )
                if np.any(replicas[:, -1] != 0.0):
                    raise ValueError(
                        f"{depth}/{dataset}/{transcript_id}: appended terminal is nonzero"
                    )
                consensus = replicas[:, :-1].mean(axis=0)
                if sense_length is None:
                    sense_length = consensus.size
                elif consensus.size != sense_length:
                    raise ValueError(
                        f"{depth}/{transcript_id}: dataset profiles have unequal lengths"
                    )
                profiles.append(consensus)
            profile_matrix = np.vstack(profiles)
            matrix = centered_correlation_matrix(profile_matrix)
            for pair_number, (i, j) in enumerate(pair_indices):
                value = float(matrix[i, j])
                pair_values[processed, pair_number] = value
                exact = profile_matrix[i] == profile_matrix[j]
                exact_position_counts[pair_number] += int(np.count_nonzero(exact))
                total_position_counts[pair_number] += int(exact.size)
                identical_profile_counts[pair_number] += int(np.all(exact))
                buffer.append(
                    {
                        "depth": depth,
                        "nominal_reads_per_codon": nominal_depth,
                        "transcript_id": transcript_id,
                        "dataset_a": datasets[i],
                        "dataset_b": datasets[j],
                        "consensus_pcc": value,
                    }
                )
            if len(buffer) >= 5500:
                _flush_cross_rows(writer, buffer)
            processed += 1
            if processed % 2000 == 0:
                print(
                    f"[{depth}] cross-dataset PCC: {processed:,}/{expected_rows:,} transcripts",
                    flush=True,
                )
        _flush_cross_rows(writer, buffer)
    if processed != expected_rows:
        raise RuntimeError(f"{depth}: processed {processed} rows, expected {expected_rows}")

    summaries: list[dict[str, Any]] = []
    for pair_number, (i, j) in enumerate(pair_indices):
        values = pair_values[:, pair_number].astype(np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            p05, p25, p50, p75, p95 = np.quantile(
                finite, [0.05, 0.25, 0.5, 0.75, 0.95]
            )
            mean = finite.mean()
        else:
            p05 = p25 = p50 = p75 = p95 = mean = np.nan
        summaries.append(
            {
                "depth": depth,
                "nominal_reads_per_codon": nominal_depth,
                "dataset_a": datasets[i],
                "dataset_b": datasets[j],
                "valid_transcripts": int(finite.size),
                "undefined_transcripts": int(processed - finite.size),
                "identical_profile_transcripts": int(
                    identical_profile_counts[pair_number]
                ),
                "identical_profile_fraction": float(
                    identical_profile_counts[pair_number] / processed
                ),
                "exact_position_fraction": float(
                    exact_position_counts[pair_number]
                    / total_position_counts[pair_number]
                ),
                "mean_pcc": float(mean),
                "p05_pcc": float(p05),
                "q25_pcc": float(p25),
                "median_pcc": float(p50),
                "q75_pcc": float(p75),
                "p95_pcc": float(p95),
            }
        )
    return {
        "depth": depth,
        "nominal_reads_per_codon": nominal_depth,
        "processed_transcripts": processed,
        "output_path": str(output_path),
        "summaries": summaries,
    }


def load_metrics(path: Path, max_transcripts: int | None) -> pd.DataFrame:
    columns = [
        "depth",
        "nominal_reads_per_codon",
        "dataset",
        "transcript_id",
        "replicate_pcc",
        "rep1_kinetic_pcc",
        "rep2_kinetic_pcc",
        "consensus_kinetic_pcc",
        "bias_affected_fraction",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[frame["dataset"].isin(DATASET_ORDER)].copy()
    if max_transcripts is not None:
        frame = (
            frame.sort_values(["depth", "dataset", "transcript_id"])
            .groupby(["depth", "dataset"], sort=False, group_keys=False)
            .head(max_transcripts)
            .reset_index(drop=True)
        )
    expected = set(DATASET_ORDER)
    for depth, _, _ in DEPTHS:
        observed = set(frame.loc[frame["depth"] == depth, "dataset"])
        if observed != expected:
            raise ValueError(
                f"Input metrics for {depth} do not contain the expected datasets"
            )
    return frame


def metric_summaries(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    replica_rows: list[dict[str, Any]] = []
    kinetic_rows: list[dict[str, Any]] = []
    for (depth, nominal, dataset), group in metrics.groupby(
        ["depth", "nominal_reads_per_codon", "dataset"], sort=False
    ):
        stats = box_statistics(group["replicate_pcc"], DISPLAY_NAMES[dataset])
        replica_rows.append(
            {
                "depth": depth,
                "nominal_reads_per_codon": nominal,
                "dataset": dataset,
                "bias_label": DISPLAY_NAMES[dataset],
                "n_valid": int(group["replicate_pcc"].notna().sum()),
                "p05": stats["whislo"],
                "q25": stats["q1"],
                "median": stats["med"],
                "q75": stats["q3"],
                "p95": stats["whishi"],
            }
        )
        for column, label, _ in REPRESENTATIONS:
            stats = box_statistics(group[column], label)
            kinetic_rows.append(
                {
                    "depth": depth,
                    "nominal_reads_per_codon": nominal,
                    "dataset": dataset,
                    "bias_label": DISPLAY_NAMES[dataset],
                    "representation": label,
                    "n_valid": int(group[column].notna().sum()),
                    "p05": stats["whislo"],
                    "q25": stats["q1"],
                    "median": stats["med"],
                    "q75": stats["q3"],
                    "p95": stats["whishi"],
                }
            )
    return pd.DataFrame(replica_rows), pd.DataFrame(kinetic_rows)


def _mean_t_interval(values: pd.Series) -> tuple[float, float, float, float]:
    """Mean and two-sided 95% t interval over independent design units."""
    finite = values.to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        raise ValueError("A mean confidence interval requires at least two units")
    estimate = float(np.mean(finite))
    standard_error = float(np.std(finite, ddof=1) / np.sqrt(finite.size))
    half_width = float(student_t.ppf(0.975, finite.size - 1) * standard_error)
    return estimate, half_width, estimate - half_width, estimate + half_width


def headline_mean_ci_summary(
    replica_summary: pd.DataFrame,
    kinetic_summary: pd.DataFrame,
    cross_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create the manuscript headline means and dependence-aware 95% CIs.

    Within-dataset and kinetic-target summaries treat each programmed bias as
    one design unit (n=10).  Cross-dataset summaries average the 45 pairwise
    medians, but estimate uncertainty by deleting one of the ten datasets at a
    time; this avoids pretending that the 45 graph edges are independent.
    """
    rows: list[dict[str, Any]] = []
    for depth, nominal, _ in DEPTHS:
        rep = replica_summary.loc[replica_summary["depth"] == depth]
        kinetic = kinetic_summary.loc[kinetic_summary["depth"] == depth]
        kinetic_wide = kinetic.pivot(
            index="dataset", columns="representation", values="median"
        )
        condition_metrics = {
            "replica_agreement": rep.set_index("dataset")["median"],
            "single_replica_vs_Kt": kinetic_wide[["Replica 1", "Replica 2"]].mean(
                axis=1
            ),
            "replica_mean_vs_Kt": kinetic_wide["Arithmetic mean"],
        }
        for metric, values in condition_metrics.items():
            estimate, half_width, ci_low, ci_high = _mean_t_interval(values)
            rows.append(
                {
                    "depth": depth,
                    "nominal_reads_per_codon": nominal,
                    "metric": metric,
                    "estimate": estimate,
                    "ci95_half_width": half_width,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "design_units": int(values.size),
                    "interval_method": "two-sided Student-t CI across bias conditions",
                }
            )

        pairs = cross_summary.loc[cross_summary["depth"] == depth].copy()
        datasets = sorted(set(pairs["dataset_a"]) | set(pairs["dataset_b"]))
        expected_pairs = len(datasets) * (len(datasets) - 1) // 2
        if len(datasets) != len(DATASET_ORDER) or len(pairs) != expected_pairs:
            raise ValueError(
                f"Cross-dataset summary for {depth} is incomplete: "
                f"datasets={len(datasets)}, pairs={len(pairs)}"
            )
        estimate = float(pairs["median_pcc"].mean())
        leave_one_out = np.asarray(
            [
                pairs.loc[
                    (pairs["dataset_a"] != dataset)
                    & (pairs["dataset_b"] != dataset),
                    "median_pcc",
                ].mean()
                for dataset in datasets
            ],
            dtype=np.float64,
        )
        n_units = len(datasets)
        jackknife_se = float(
            np.sqrt(
                (n_units - 1)
                / n_units
                * np.sum((leave_one_out - leave_one_out.mean()) ** 2)
            )
        )
        half_width = float(student_t.ppf(0.975, n_units - 1) * jackknife_se)
        rows.append(
            {
                "depth": depth,
                "nominal_reads_per_codon": nominal,
                "metric": "cross_dataset_agreement",
                "estimate": estimate,
                "ci95_half_width": half_width,
                "ci95_low": estimate - half_width,
                "ci95_high": estimate + half_width,
                "design_units": n_units,
                "interval_method": (
                    "two-sided t interval from leave-one-dataset-out jackknife SE"
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def additional_summaries(
    metrics: pd.DataFrame,
    replica_summary: pd.DataFrame,
    kinetic_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize bias prevalence and reproducibility-versus-fidelity coupling."""
    first_depth = DEPTHS[0][0]
    prevalence_rows: list[dict[str, Any]] = []
    for dataset, group in metrics.loc[metrics["depth"] == first_depth].groupby(
        "dataset", sort=False
    ):
        affected = group["bias_affected_fraction"].to_numpy(dtype=np.float64)
        prevalence_rows.append(
            {
                "dataset": dataset,
                "bias_label": DISPLAY_NAMES[dataset],
                "transcripts": int(affected.size),
                "zero_affected_transcript_fraction": float(np.mean(affected == 0.0)),
                "mean_affected_codon_fraction": float(np.mean(affected)),
                "median_affected_codon_fraction": float(np.median(affected)),
                "p95_affected_codon_fraction": float(np.quantile(affected, 0.95)),
            }
        )
    prevalence = pd.DataFrame(prevalence_rows)

    consensus = kinetic_summary.loc[
        kinetic_summary["representation"] == "Arithmetic mean",
        [
            "depth",
            "nominal_reads_per_codon",
            "dataset",
            "bias_label",
            "median",
        ],
    ].rename(columns={"median": "median_consensus_kinetic_pcc"})
    relationship = replica_summary[
        [
            "depth",
            "nominal_reads_per_codon",
            "dataset",
            "bias_label",
            "median",
        ]
    ].rename(columns={"median": "median_replicate_pcc"})
    relationship = relationship.merge(
        consensus,
        on=["depth", "nominal_reads_per_codon", "dataset", "bias_label"],
        validate="one_to_one",
    ).merge(
        prevalence[
            [
                "dataset",
                "zero_affected_transcript_fraction",
                "median_affected_codon_fraction",
            ]
        ],
        on="dataset",
        validate="many_to_one",
    )
    association_rows: list[dict[str, Any]] = []
    for (depth, nominal), group in relationship.groupby(
        ["depth", "nominal_reads_per_codon"], sort=False
    ):
        rho = group["median_replicate_pcc"].rank().corr(
            group["median_consensus_kinetic_pcc"].rank()
        )
        association_rows.append(
            {
                "depth": depth,
                "nominal_reads_per_codon": nominal,
                "datasets": int(len(group)),
                "spearman_rho_across_dataset_medians": float(rho),
            }
        )
    return prevalence, relationship, pd.DataFrame(association_rows)


def analyze_single_model_vs_replica_agreement(
    *,
    input_metrics_path: Path,
    results_root: Path,
    run_id: str,
    training_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Compare held-out single-dataset mu PCC with matched replica agreement.

    Model PCC is read from the best-validation-loss prediction export and uses
    the model-valid mask. Replica PCC comes from the corresponding synthetic
    input audit. The comparison cohort is the intersection of valid values for
    both metrics, within the exact held-out transcript panel of each run.
    """
    input_frame = pd.read_parquet(
        input_metrics_path,
        columns=["depth", "dataset", "transcript_id", "replicate_pcc"],
    )
    input_frame = input_frame.loc[input_frame["dataset"].isin(DATASET_ORDER)].copy()
    if input_frame.duplicated(["depth", "dataset", "transcript_id"]).any():
        raise ValueError("Input audit contains duplicate depth/dataset/transcript rows")

    expected_biases = {bias.slug for bias in MODEL_BIASES}
    if expected_biases != set(DATASET_ORDER):
        raise AssertionError(
            "Single-model bias matrix and displayed synthetic biases differ: "
            f"models_only={sorted(expected_biases-set(DATASET_ORDER))}, "
            f"plots_only={sorted(set(DATASET_ORDER)-expected_biases)}"
        )

    depth_to_input = {
        "0p25": "0p25_per_codon",
        "2": "2_per_codon",
        "20": "20_per_codon",
    }
    model_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    validation_ids_by_depth: dict[str, set[str]] = {}
    for depth_index, model_depth in enumerate(MODEL_DEPTHS):
        input_depth = depth_to_input[model_depth.slug]
        for bias_index, bias in enumerate(MODEL_BIASES):
            task_index = depth_index * len(MODEL_BIASES) + bias_index
            run_directory = discover_run_directory(
                results_root=results_root,
                depth=model_depth,
                bias=bias,
                seed=training_seed,
                run_id=run_id,
                expected_task_index=task_index,
            )
            prediction_path, checkpoint_path = resolve_prediction_artifact(
                run_directory=run_directory,
                bias=bias,
                checkpoint_variant="best_val_loss",
            )
            artifact = RunArtifact(
                depth=model_depth.slug,
                depth_label=model_depth.label,
                bias=bias.slug,
                bias_label=bias.label.replace("\n", " "),
                task_index=task_index,
                run_directory=str(run_directory.resolve()),
                prediction_path=str(prediction_path.resolve()),
                checkpoint_path=checkpoint_path,
            )
            rows, transcript_ids = analyze_prediction(
                artifact=artifact,
                pcc_prediction_floor=0.0,
            )
            reference_ids = validation_ids_by_depth.get(input_depth)
            if reference_ids is None:
                validation_ids_by_depth[input_depth] = transcript_ids
            elif transcript_ids != reference_ids:
                raise AssertionError(
                    f"Held-out transcript IDs differ across biases at {input_depth}"
                )
            for row in rows:
                row["depth"] = input_depth
                row["dataset"] = row.pop("bias")
                row["model_mu_pcc"] = (
                    float(row["mu_pcc"])
                    if bool(row["pcc_variance_valid"])
                    else np.nan
                )
            model_rows.extend(rows)
            artifact_rows.append(
                {
                    "depth": input_depth,
                    "dataset": bias.slug,
                    "task_index": task_index,
                    "run_directory": str(run_directory.resolve()),
                    "prediction_path": str(prediction_path.resolve()),
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_variant": "best_val_loss",
                    "n_validation_transcripts": len(transcript_ids),
                }
            )

    model_frame = pd.DataFrame(model_rows)
    matched = model_frame.merge(
        input_frame,
        on=["depth", "dataset", "transcript_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not (matched["_merge"] == "both").all():
        missing = matched.loc[
            matched["_merge"] != "both", ["depth", "dataset", "transcript_id"]
        ].head(10)
        raise ValueError(f"Model transcripts missing from input audit:\n{missing}")
    matched = matched.drop(columns=["_merge", "mu_pcc"])
    matched["comparison_valid"] = (
        np.isfinite(matched["model_mu_pcc"])
        & np.isfinite(matched["replicate_pcc"])
    )
    matched["paired_pcc_difference"] = np.where(
        matched["comparison_valid"],
        matched["model_mu_pcc"] - matched["replicate_pcc"],
        np.nan,
    )

    summary_rows: list[dict[str, Any]] = []
    for (depth, dataset), group in matched.groupby(["depth", "dataset"], sort=False):
        valid = group.loc[group["comparison_valid"]]
        if valid.empty:
            raise ValueError(f"No valid matched comparison rows for {depth}/{dataset}")
        replica = valid["replicate_pcc"].to_numpy(dtype=np.float64)
        model = valid["model_mu_pcc"].to_numpy(dtype=np.float64)
        delta = model - replica
        summary_rows.append(
            {
                "depth": depth,
                "nominal_reads_per_codon": next(
                    nominal for item, nominal, _ in DEPTHS if item == depth
                ),
                "dataset": dataset,
                "bias_label": DISPLAY_NAMES[dataset],
                "held_out_transcripts": int(len(group)),
                "matched_valid_transcripts": int(len(valid)),
                "median_replica_pcc": float(np.median(replica)),
                "q25_replica_pcc": float(np.quantile(replica, 0.25)),
                "q75_replica_pcc": float(np.quantile(replica, 0.75)),
                "mean_replica_pcc": float(np.mean(replica)),
                "median_model_mu_pcc": float(np.median(model)),
                "q25_model_mu_pcc": float(np.quantile(model, 0.25)),
                "q75_model_mu_pcc": float(np.quantile(model, 0.75)),
                "mean_model_mu_pcc": float(np.mean(model)),
                "median_paired_difference": float(np.median(delta)),
                "mean_paired_difference": float(np.mean(delta)),
                "spearman_across_transcripts": float(
                    valid["replicate_pcc"].rank().corr(
                        valid["model_mu_pcc"].rank()
                    )
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    association_rows: list[dict[str, Any]] = []
    for (depth, nominal), group in summary.groupby(
        ["depth", "nominal_reads_per_codon"], sort=False
    ):
        rho = group["median_replica_pcc"].rank().corr(
            group["median_model_mu_pcc"].rank()
        )
        association_rows.append(
            {
                "depth": depth,
                "nominal_reads_per_codon": nominal,
                "bias_datasets": int(len(group)),
                "spearman_across_bias_medians": float(rho),
            }
        )
    return matched, summary, pd.DataFrame(association_rows), artifact_rows


def _style_boxplot(
    result: dict[str, Any], color: str, alpha: float = 0.78
) -> None:
    for box in result["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(alpha)
        box.set_edgecolor("#25343d")
    for median in result["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.7)
    for part in ("whiskers", "caps"):
        for artist in result[part]:
            artist.set_color("#53636c")


def configure_iclr_typography() -> dict[str, Any]:
    """Apply one bold LaTeX publication style to every generated figure."""
    style = publication_rc()
    style.update(
        {
            "font.size": 18.0,
            "axes.labelsize": 20.0,
            "axes.titlesize": 21.0,
            "figure.titlesize": 24.0,
            "xtick.labelsize": 17.0,
            "ytick.labelsize": 17.0,
            "legend.fontsize": 17.0,
            "legend.title_fontsize": 17.0,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    if style["text.usetex"]:
        style["text.latex.preamble"] += (
            r"\usepackage{bm}"
            r"\renewcommand{\seriesdefault}{\bfdefault}"
            r"\AtBeginDocument{\boldmath}"
        )
    matplotlib.rcParams.update(style)
    return {
        key: style[key]
        for key in (
            "text.usetex",
            "font.family",
            "font.serif",
            "font.size",
            "font.weight",
            "axes.labelweight",
            "axes.titleweight",
            "pdf.fonttype",
            "savefig.dpi",
        )
    }


def save_figure(figure: plt.Figure, stem: Path) -> list[Path]:
    # Matplotlib does not reliably propagate font.weight to every legend and
    # colorbar Text instance, so enforce the requested bold style explicitly.
    for text_artist in figure.findobj(match=Text):
        text_artist.set_fontweight("bold")
    paths = [
        stem.with_suffix(".pdf"),
        stem.with_suffix(".png"),
        stem.with_suffix(".svg"),
    ]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=300, bbox_inches="tight")
    figure.savefig(paths[2], bbox_inches="tight")
    plt.close(figure)
    return paths


def plot_replica_boxes(metrics: pd.DataFrame, figure_dir: Path) -> list[Path]:
    """Merge the three read depths into one grouped per-bias box plot."""
    figure, axis = plt.subplots(figsize=(18.0, 8.8), constrained_layout=True)
    base_positions = np.arange(1, len(DATASET_ORDER) + 1, dtype=np.float64)
    offsets = (-0.27, 0.0, 0.27)
    colors = ("#0072B2", "#E69F00", "#009E73")
    for (depth, _, title), color, offset in zip(DEPTHS, colors, offsets):
        subset = metrics.loc[metrics["depth"] == depth]
        stats = [
            box_statistics(
                subset.loc[subset["dataset"] == dataset, "replicate_pcc"],
                PLOT_DISPLAY_NAMES[dataset],
            )
            for dataset in DATASET_ORDER
        ]
        result = axis.bxp(
            stats,
            positions=base_positions + offset,
            widths=0.23,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
        )
        _style_boxplot(result, color)
        result["boxes"][0].set_label(title)
    axis.axhline(0.0, color="#777777", linewidth=0.9)
    axis.set_ylim(-0.25, 1.02)
    axis.set_xlim(0.48, len(DATASET_ORDER) + 0.52)
    axis.set_xticks(
        base_positions,
        [PLOT_DISPLAY_NAMES[dataset] for dataset in DATASET_ORDER],
    )
    axis.tick_params(axis="x", labelrotation=28)
    axis.set_ylabel(r"$\mathrm{PCC}(\mathrm{replica}\ 1,\mathrm{replica}\ 2)$")
    axis.grid(axis="y", alpha=0.28)
    axis.legend(
        title="Read depth",
        frameon=False,
        ncol=3,
        loc="upper left",
    )
    figure.suptitle("Within-dataset replica agreement across read depths")
    stem = figure_dir / "replica_agreement_boxplots_by_dataset_and_depth"
    return save_figure(figure, stem)


def plot_kinetic_boxes(metrics: pd.DataFrame, figure_dir: Path) -> list[Path]:
    figure, axes = plt.subplots(3, 1, figsize=(22.0, 19.0), constrained_layout=True)
    base_positions = np.arange(1, len(DATASET_ORDER) + 1, dtype=np.float64)
    offsets = (-0.25, 0.0, 0.25)
    outputs: list[Path] = []
    for axis, (depth, nominal, title) in zip(axes, DEPTHS):
        subset = metrics.loc[metrics["depth"] == depth]
        for (column, label, color), offset in zip(REPRESENTATIONS, offsets):
            stats = [
                box_statistics(
                    subset.loc[subset["dataset"] == dataset, column], label
                )
                for dataset in DATASET_ORDER
            ]
            result = axis.bxp(
                stats,
                positions=base_positions + offset,
                widths=0.22,
                showfliers=False,
                patch_artist=True,
                manage_ticks=False,
            )
            _style_boxplot(result, color, alpha=0.8)
            result["boxes"][0].set_label(label)
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_ylim(-0.25, 1.02)
        axis.set_xlim(0.45, len(DATASET_ORDER) + 0.55)
        axis.set_xticks(base_positions, [PLOT_DISPLAY_NAMES[d] for d in DATASET_ORDER])
        axis.tick_params(axis="x", labelrotation=32)
        axis.set_ylabel(r"$\mathrm{PCC}(\mathrm{sampled\ profile},K_t)$")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False, ncol=3, loc="lower left")
    figure.suptitle(
        r"Each replica and their arithmetic mean versus the common kinetic target $K_t$",
        fontsize=24,
    )
    stem = figure_dir / "kinetic_target_agreement_boxplots_by_dataset_and_depth"
    outputs.extend(save_figure(figure, stem))
    return outputs


def plot_single_model_comparison_boxes(
    matched: pd.DataFrame, figure_dir: Path
) -> list[Path]:
    """Plot matched held-out distributions for repeatability and model fit."""
    figure, axes = plt.subplots(3, 1, figsize=(22.0, 19.0), constrained_layout=True)
    base_positions = np.arange(1, len(DATASET_ORDER) + 1, dtype=np.float64)
    specifications = (
        ("replicate_pcc", "Replica 1 vs replica 2", "#6E8796", -0.19),
        (
            "model_mu_pcc",
            r"Model $\mu$ vs two-replica consensus",
            "#9B5C8F",
            0.19,
        ),
    )
    for axis, (depth, _, title) in zip(axes, DEPTHS):
        subset = matched.loc[
            (matched["depth"] == depth) & matched["comparison_valid"]
        ]
        for column, label, color, offset in specifications:
            stats = [
                box_statistics(
                    subset.loc[subset["dataset"] == dataset, column], label
                )
                for dataset in DATASET_ORDER
            ]
            result = axis.bxp(
                stats,
                positions=base_positions + offset,
                widths=0.34,
                showfliers=False,
                patch_artist=True,
                manage_ticks=False,
            )
            _style_boxplot(result, color, alpha=0.82)
            result["boxes"][0].set_label(label)
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_ylim(-0.25, 1.02)
        axis.set_xlim(0.45, len(DATASET_ORDER) + 0.55)
        axis.set_xticks(base_positions, [PLOT_DISPLAY_NAMES[d] for d in DATASET_ORDER])
        axis.tick_params(axis="x", labelrotation=32)
        axis.set_ylabel("Transcript-level PCC")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False, ncol=2, loc="lower left")
    figure.suptitle(
        "Single-dataset prediction versus input replica agreement\n"
        "best-validation-loss checkpoints; identical held-out transcripts within each comparison",
        fontsize=24,
    )
    stem = figure_dir / "single_dataset_prediction_vs_replica_agreement_boxplots"
    return save_figure(figure, stem)


def plot_single_model_comparison_scatter(
    summary: pd.DataFrame,
    association: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    """Relate condition-median repeatability to condition-median model fit."""
    # A horizontal strip makes each square panel and its labels too small at
    # manuscript width.  Center the lowest-depth panel above the other two so
    # that all three retain the same physical size and axis limits.
    figure = plt.figure(figsize=(16.0, 14.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 4)
    axes = np.asarray(
        [
            figure.add_subplot(grid[0, 1:3]),
            figure.add_subplot(grid[1, 0:2]),
            figure.add_subplot(grid[1, 2:4]),
        ],
        dtype=object,
    )
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(DATASET_ORDER)))
    for axis, (depth, _, title) in zip(axes, DEPTHS):
        subset = summary.loc[summary["depth"] == depth].set_index("dataset")
        subset = subset.loc[list(DATASET_ORDER)]
        axis.plot([-0.25, 1.0], [-0.25, 1.0], color="#999999", linestyle="--", linewidth=1)
        for color, dataset in zip(colors, DATASET_ORDER):
            row = subset.loc[dataset]
            x = float(row["median_replica_pcc"])
            y = float(row["median_model_mu_pcc"])
            axis.scatter(
                x,
                y,
                s=66,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
                label=PLOT_DISPLAY_NAMES[dataset] if axis is axes[0] else None,
            )
        rho = association.loc[
            association["depth"] == depth, "spearman_across_bias_medians"
        ].item()
        axis.text(
            0.04,
            0.96,
            rf"Spearman $\rho$ across biases $= {rho:.2f}$",
            transform=axis.transAxes,
            va="top",
            fontsize=17.0,
        )
        axis.set_xlim(-0.12, 1.0)
        axis.set_ylim(-0.12, 1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.set_xlabel("Median PCC(replica 1, replica 2)")
        axis.set_ylabel(r"Median $\mathrm{PCC}(\mathrm{model}\ \mu,\mathrm{replica\ consensus})$")
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Does single-dataset predictive agreement track input repeatability?",
        fontsize=24,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.755, 0.73),
        ncol=1,
        frameon=False,
        fontsize=15.0,
        labelspacing=0.45,
    )
    stem = figure_dir / "single_dataset_prediction_vs_replica_agreement_scatter"
    return save_figure(figure, stem)


def plot_cross_heatmaps(summary: pd.DataFrame, figure_dir: Path) -> list[Path]:
    # A horizontal 1x3 strip makes the ten condition labels and 100 annotated
    # cells unreadable at article width.  Give every square a full half-page
    # cell instead: one centred panel above two equally sized lower panels.
    figure = plt.figure(figsize=(20.0, 18.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 4)
    axes = np.asarray(
        [
            figure.add_subplot(grid[0, 1:3]),
            figure.add_subplot(grid[1, 0:2]),
            figure.add_subplot(grid[1, 2:4]),
        ],
        dtype=object,
    )
    labels = [PLOT_DISPLAY_NAMES[name] for name in DATASET_ORDER]
    image = None
    for axis, (depth, nominal, title) in zip(axes, DEPTHS):
        matrix = np.eye(len(DATASET_ORDER), dtype=np.float64)
        subset = summary.loc[summary["depth"] == depth]
        lookup = {name: index for index, name in enumerate(DATASET_ORDER)}
        for row in subset.itertuples(index=False):
            i, j = lookup[row.dataset_a], lookup[row.dataset_b]
            matrix[i, j] = matrix[j, i] = row.median_pcc
        image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_xticks(
            range(len(labels)), labels, rotation=48, ha="right", fontsize=15.5
        )
        axis.set_yticks(range(len(labels)), labels, fontsize=15.5)
        axis.set_title(title)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = matrix[i, j]
                color = "white" if value < 0.55 else "#111111"
                axis.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12.5,
                    color=color,
                )
    assert image is not None
    figure.colorbar(
        image,
        ax=list(axes),
        shrink=0.90,
        pad=0.025,
        label="Median transcript-level consensus PCC",
    )
    figure.suptitle(
        "Cross-dataset agreement of arithmetic replica consensuses",
        fontsize=24,
    )
    stem = figure_dir / "cross_dataset_consensus_correlation_by_depth"
    return save_figure(figure, stem)


def format_table(frame: pd.DataFrame, digits: int = 3) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        if pd.api.types.is_integer_dtype(display[column]):
            continue
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    return display.to_html(index=False, border=0, classes="data-table", escape=True)


def render_html(
    replica_summary: pd.DataFrame,
    kinetic_summary: pd.DataFrame,
    cross_summary: pd.DataFrame,
    bias_prevalence: pd.DataFrame,
    reproducibility_fidelity: pd.DataFrame,
    association_summary: pd.DataFrame,
    single_model_summary: pd.DataFrame,
    single_model_association: pd.DataFrame,
    provenance: dict[str, Any],
) -> str:
    depth_rows: list[dict[str, Any]] = []
    extrema_rows: list[dict[str, Any]] = []
    for depth, nominal, title in DEPTHS:
        rep = replica_summary.loc[replica_summary["depth"] == depth]
        kin = kinetic_summary.loc[
            (kinetic_summary["depth"] == depth)
            & (kinetic_summary["representation"] == "Arithmetic mean")
        ]
        pairs = cross_summary.loc[cross_summary["depth"] == depth]
        highest = pairs.loc[pairs["median_pcc"].idxmax()]
        lowest = pairs.loc[pairs["median_pcc"].idxmin()]
        depth_rows.append(
            {
                "depth": nominal,
                "median replica PCC across datasets": float(rep["median"].median()),
                "replica-PCC range": f"{rep['median'].min():.3f}–{rep['median'].max():.3f}",
                "median consensus–K PCC across datasets": float(kin["median"].median()),
                "consensus–K range": f"{kin['median'].min():.3f}–{kin['median'].max():.3f}",
                "median off-diagonal cross-dataset PCC": float(pairs["median_pcc"].median()),
            }
        )
        extrema_rows.extend(
            [
                {
                    "depth": nominal,
                    "extreme": "most similar",
                    "dataset A": DISPLAY_NAMES[highest.dataset_a],
                    "dataset B": DISPLAY_NAMES[highest.dataset_b],
                    "median PCC": highest.median_pcc,
                    "IQR": f"{highest.q25_pcc:.3f}–{highest.q75_pcc:.3f}",
                },
                {
                    "depth": nominal,
                    "extreme": "least similar",
                    "dataset A": DISPLAY_NAMES[lowest.dataset_a],
                    "dataset B": DISPLAY_NAMES[lowest.dataset_b],
                    "median PCC": lowest.median_pcc,
                    "IQR": f"{lowest.q25_pcc:.3f}–{lowest.q75_pcc:.3f}",
                },
            ]
        )
    depth_table = pd.DataFrame(depth_rows)
    extrema_table = pd.DataFrame(extrema_rows)
    replica_medians = replica_summary.pivot(
        index="bias_label", columns="nominal_reads_per_codon", values="median"
    ).reindex([DISPLAY_NAMES[name] for name in DATASET_ORDER]).reset_index()
    replica_medians.columns = [
        "bias",
        "0.25 reads/codon",
        "2 reads/codon",
        "20 reads/codon",
    ]
    consensus_medians = kinetic_summary.loc[
        kinetic_summary["representation"] == "Arithmetic mean"
    ].pivot(
        index="bias_label", columns="nominal_reads_per_codon", values="median"
    ).reindex([DISPLAY_NAMES[name] for name in DATASET_ORDER]).reset_index()
    consensus_medians.columns = [
        "bias",
        "0.25 reads/codon",
        "2 reads/codon",
        "20 reads/codon",
    ]
    association_display = association_summary.rename(
        columns={
            "nominal_reads_per_codon": "reads/codon",
            "spearman_rho_across_dataset_medians": "Spearman rho: replica agreement vs K agreement",
        }
    )[["reads/codon", "datasets", "Spearman rho: replica agreement vs K agreement"]]
    prevalence_display = bias_prevalence[
        [
            "bias_label",
            "zero_affected_transcript_fraction",
            "median_affected_codon_fraction",
            "mean_affected_codon_fraction",
            "p95_affected_codon_fraction",
        ]
    ].rename(
        columns={
            "bias_label": "bias",
            "zero_affected_transcript_fraction": "fraction transcripts with no selected codon",
            "median_affected_codon_fraction": "median selected-codon fraction",
            "mean_affected_codon_fraction": "mean selected-codon fraction",
            "p95_affected_codon_fraction": "p95 selected-codon fraction",
        }
    )
    identity_rows: list[dict[str, Any]] = []
    for depth, nominal, _ in DEPTHS:
        candidates = cross_summary.loc[cross_summary["depth"] == depth]
        row = candidates.loc[candidates["identical_profile_fraction"].idxmax()]
        identity_rows.append(
            {
                "reads/codon": nominal,
                "bias A": DISPLAY_NAMES[row.dataset_a],
                "bias B": DISPLAY_NAMES[row.dataset_b],
                "identical complete profiles": row.identical_profile_fraction,
                "identical codon values": row.exact_position_fraction,
            }
        )
    identity_display = pd.DataFrame(identity_rows)
    model_display = single_model_summary[
        [
            "nominal_reads_per_codon",
            "bias_label",
            "matched_valid_transcripts",
            "median_replica_pcc",
            "median_model_mu_pcc",
            "median_paired_difference",
            "spearman_across_transcripts",
        ]
    ].rename(
        columns={
            "nominal_reads_per_codon": "reads/codon",
            "bias_label": "bias",
            "matched_valid_transcripts": "matched transcripts",
            "median_replica_pcc": "median replica PCC",
            "median_model_mu_pcc": "median model–consensus PCC",
            "median_paired_difference": "median paired difference",
            "spearman_across_transcripts": "transcript-level Spearman rho",
        }
    )
    model_association_display = single_model_association[
        [
            "nominal_reads_per_codon",
            "bias_datasets",
            "spearman_across_bias_medians",
        ]
    ].rename(
        columns={
            "nominal_reads_per_codon": "reads/codon",
            "bias_datasets": "bias datasets",
            "spearman_across_bias_medians": "Spearman rho across bias medians",
        }
    )
    total_transcripts = int(provenance["transcripts_per_dataset"])
    created = html.escape(provenance["created_at"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Individual synthetic dataset agreement</title>
<style>
:root{{--ink:#1d2a32;--muted:#58666f;--blue:#155f84;--line:#d8e1e6;--soft:#f3f7f9;--warn:#fff4df}}
*{{box-sizing:border-box}}body{{max-width:1380px;margin:30px auto;padding:0 28px;color:var(--ink);font:17px/1.58 Georgia,serif}}
h1,h2,h3{{line-height:1.22}}h1{{font-size:36px}}h2{{margin-top:2.2em;padding-top:16px;border-top:1px solid var(--line)}}
p,li{{max-width:108ch}}a{{color:var(--blue)}}code,pre{{font-family:ui-monospace,Consolas,monospace}}.meta{{color:var(--muted);font-size:15px}}
.finding,.warning{{padding:15px 20px;margin:18px 0;background:var(--soft);border-left:5px solid var(--blue)}}.warning{{background:var(--warn);border-color:#bd762e}}
figure{{margin:26px 0}}figure img{{width:100%;height:auto}}figcaption{{font-size:14px;color:var(--muted)}}.table-wrap{{overflow-x:auto;margin:18px 0}}
table{{border-collapse:collapse;width:100%;font:13px/1.4 system-ui,sans-serif}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th{{background:#eaf1f4}}td:nth-child(2),th:nth-child(2),td:nth-child(3),th:nth-child(3){{text-align:left}}
nav{{display:flex;gap:10px 22px;flex-wrap:wrap;font:14px system-ui,sans-serif}}@media(max-width:700px){{body{{padding:0 14px}}h1{{font-size:27px}}}}
</style></head><body>
<header><h1>Individual synthetic datasets: replica, latent-target, and cross-dataset agreement</h1>
<p class="meta">Generated {created}. The input-audit boxes contain transcript-level PCCs over {total_transcripts:,} aligned transcripts per bias dataset;
the model comparison uses each run's matched held-out validation panel. Whiskers are the 5th and 95th percentiles; boxes are the IQR;
the line is the median. Outliers are intentionally not drawn.</p>
<nav><a href="#replicas">Replica agreement</a><a href="#occupancy">Against q</a><a href="#kinetic">Against K</a><a href="#model">Single-model prediction</a><a href="#cross">Cross-dataset</a><a href="#interpretation">Interpretation</a><a href="#sources">Sources</a></nav></header>

<div class="finding"><strong>This is an individual-dataset analysis.</strong> Every named bias has its own distribution.
No codons are pooled across transcripts, no biased datasets are averaged into one condition, and the stored integerized mean is not treated as a replicate.</div>

<div class="table-wrap">{format_table(depth_table)}</div>

<section id="replicas"><h2>1. Replica agreement within every dataset</h2>
<figure><img src="figures/replica_agreement_boxplots_by_dataset_and_depth.svg" alt="Replica PCC boxplots for each bias and depth">
<figcaption>Each bias has three adjacent transcript-level boxes, one per sampling depth. This merged layout compares the read-depth
effect directly while preserving separate distributions for every programmed bias.</figcaption></figure>
<p>These boxes measure technical repeatability of the sampled profiles. A high value can result from a strongly reproduced technical bias;
it is not evidence that the dataset is close to the latent kinetic target.</p></section>
<div class="table-wrap">{format_table(replica_medians)}</div>

<section id="occupancy"><h2>2. Sampled profiles against matched pre-bias TASEP occupancy</h2>
<figure><img src="figures/tasep_occupancy_agreement_boxplots_by_dataset_and_depth.svg" alt="Replica and mean versus matched TASEP occupancy boxplots">
<figcaption>Sampled replica r is compared with its own normalized TASEP occupancy q<sub>t</sub><sup>(r)</sup>;
the arithmetic sampled consensus is compared with the arithmetic occupancy target q-bar. These are replicate-matched
pre-bias targets rather than the programmed dwell-time profile.</figcaption></figure>
<p>The median consensus agreement across biases is 0.323, 0.499, and 0.549 at increasing depth. Matching each observation
to q removes traffic and finite-trajectory differences from the target discrepancy, but it does not remove the injected
technical multiplier or NB2 sampling noise. Across the 19,283 retained transcripts, PCC(q<sup>(1)</sup>,q<sup>(2)</sup>)
has median 0.866 and IQR 0.849–0.882, so finite trajectory recording contributes nonzero variation even before count sampling.</p></section>

<section id="kinetic"><h2>2b. Secondary comparison against the programmed kinetic target</h2>
<figure><img src="figures/kinetic_target_agreement_boxplots_by_dataset_and_depth.svg" alt="Replica and mean versus K boxplots">
<figcaption>For each named bias, blue and orange are the individual replicas and green is their exact arithmetic mean.
All three are correlated against the same transcript-specific programmed profile K<sub>t</sub>.</figcaption></figure>
<p>The green boxes generally move upward relative to the individual replicas because averaging reduces count noise.
At high depth, differences among biases persist because additional reads cannot remove a shared sequence-dependent multiplier.</p></section>
<div class="table-wrap">{format_table(consensus_medians)}</div>

<h3>Repeatability is not fidelity</h3>
<p>Across the 10 programmed-bias conditions, biases that create large reproducible peaks can have higher replicate PCC but lower
agreement with the common kinetic target. The association below is descriptive across deliberately constructed conditions,
not an inferential test over biological datasets.</p>
<div class="table-wrap">{format_table(association_display)}</div>

<section id="model"><h2>3. Single-dataset prediction versus replica agreement</h2>
<figure><img src="figures/single_dataset_prediction_vs_replica_agreement_boxplots.svg" alt="Matched replica and model prediction PCC boxplots">
<figcaption>For every bias and depth, both boxes use the same held-out validation transcripts. Grey is PCC(replica 1, replica 2);
purple is PCC(predicted μ, arithmetic two-replica consensus) from the best-validation-loss checkpoint.</figcaption></figure>
<figure><img src="figures/single_dataset_prediction_vs_replica_agreement_scatter.svg" alt="Condition median model prediction versus replica agreement">
<figcaption>Each point is one bias dataset. The dashed line is equality and is a visual reference, not a performance threshold.</figcaption></figure>
<div class="warning"><strong>Important non-independence:</strong> the model target is the arithmetic mean of the same two replicas.
Averaging reduces sampling noise, and the fitted model can denoise across training transcripts. Consequently, model–consensus PCC
may exceed replica–replica PCC; this does not mean that the model has surpassed a formal experimental noise ceiling.</div>
<div class="table-wrap">{format_table(model_association_display)}</div>
<div class="table-wrap">{format_table(model_display)}</div></section>

<section id="cross"><h2>4. Cross-dataset correlations within each read depth</h2>
<figure><img src="figures/cross_dataset_consensus_correlation_by_depth.svg" alt="Cross-dataset consensus correlation matrices">
<figcaption>Each off-diagonal cell is the median, across transcripts, of the PCC between two datasets' arithmetic two-replica consensuses.
Diagonal values are defined as one. All panels use the same 0–1 color scale.</figcaption></figure>
<div class="table-wrap">{format_table(extrema_table)}</div>
<div class="warning"><strong>What this matrix measures:</strong> observed-profile similarity at a fixed read depth.
At low depth it is attenuated by count noise; at high depth it increasingly exposes persistent differences among bias mechanisms.
It is not a correlation between fitted model representations and it is not an estimate based on independent transcripts across cells.</div></section>

<div class="warning"><strong>Paired-simulation caveat.</strong> The bias conditions reuse the same two TASEP occupancy trajectories,
and unchanged positions can retain exactly the same sampled counts in the supplied files. The most exactly matched displayed bias pair at each depth is shown below.
Therefore the cross-dataset matrix is a descriptive comparison of these paired synthetic files; it must not be interpreted as the agreement
expected between independently resampled experiments.</div>
<div class="table-wrap">{format_table(identity_display)}</div>

<h3>How often each programmed rule is active</h3>
<p>The bias annotations are depth-independent. Rare rules can remain close to the baseline generator even when their multiplier is large,
because the multiplier is applied at very few codons.</p>
<div class="table-wrap">{format_table(prevalence_display)}</div>

<section id="interpretation"><h2>5. Scientific interpretation and pitfalls</h2>
<ul>
<li><strong>Replica agreement and truth agreement answer different questions.</strong> Shared systematic bias increases the first while potentially decreasing the second.</li>
<li><strong>Model–consensus PCC is also a different quantity.</strong> It measures predictive fit to the averaged observation, not recovery of K<sub>t</sub>, and the consensus reuses both replicas.</li>
<li><strong>The arithmetic mean is the relevant consensus.</strong> The raw <code>mean</code>/<code>ribo</code> row is a deterministic integerization and not a third stochastic replicate.</li>
<li><strong>q and K answer different questions.</strong> The exported q profiles are the immediate pre-bias occupancy targets; K is the earlier programmed dwell-time target before traffic and finite recording.</li>
<li><strong>Cross-dataset PCC is calculated transcript by transcript.</strong> A single pooled correlation would let long transcripts dominate and would create misleadingly narrow uncertainty.</li>
<li><strong>These are descriptive distributions from one simulator/count seed.</strong> They do not quantify variation over regenerated synthetic datasets.</li>
</ul></section>

<section id="sources"><h2>6. Source tables and reproduction</h2>
<pre>RIBOUNMIX_PLOT_TEX=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
.venv/bin/python analyses/analyze_synthetic_individual_datasets.py --workers 3 --overwrite</pre>
<pre>RIBOUNMIX_PLOT_TEX=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
.venv/bin/python analyses/analyze_synthetic_tasep_occupancy_agreement.py --overwrite</pre>
<p>Every figure is exported as vector PDF and SVG plus a 300-dpi PNG. Plot text is rendered by LaTeX in bold Latin Modern serif.</p>
<ul>
<li><a href="replica_agreement_summary.csv">Replica box-plot source</a></li>
<li><a href="kinetic_target_agreement_summary.csv">Replica/mean versus K box-plot source</a></li>
<li><a href="tasep_occupancy_agreement_summary.csv">Replica/mean versus matched q box-plot source</a></li>
<li><a href="tasep_occupancy_replica_agreement_summary.csv">q replicate-agreement summary</a></li>
<li><a href="tasep_occupancy_per_transcript.parquet">Per-transcript matched q metrics</a></li>
<li><a href="tasep_occupancy_provenance.json">Matched q analysis provenance</a></li>
<li><a href="cross_dataset_correlation_summary.csv">Cross-dataset pair summaries</a></li>
<li><a href="bias_prevalence_summary.csv">Programmed bias prevalence</a></li>
<li><a href="dataset_level_reproducibility_vs_fidelity.csv">Dataset-level repeatability and kinetic fidelity</a></li>
<li><a href="reproducibility_fidelity_association.csv">Across-dataset descriptive associations</a></li>
<li><a href="single_model_vs_replica_per_transcript.parquet">Matched held-out model/replica metrics</a></li>
<li><a href="single_model_vs_replica_summary.csv">Per-bias matched comparison summary</a></li>
<li><a href="single_model_vs_replica_association.csv">Across-bias median association</a></li>
<li><a href="cross_dataset_per_transcript_0p25_per_codon.parquet">Low-depth pairwise transcript metrics</a></li>
<li><a href="cross_dataset_per_transcript_2_per_codon.parquet">Medium-depth pairwise transcript metrics</a></li>
<li><a href="cross_dataset_per_transcript_20_per_codon.parquet">High-depth pairwise transcript metrics</a></li>
<li><a href="provenance.json">Provenance and exact definitions</a></li>
</ul></section>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-metrics", type=Path, default=DEFAULT_INPUT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-transcripts", type=int, help="Smoke-test limit per dataset")
    parser.add_argument(
        "--single-model-results-root",
        type=Path,
        default=DEFAULT_SINGLE_MODEL_RESULTS,
        help="Root containing the completed 30-run single-dataset matrix.",
    )
    parser.add_argument(
        "--single-model-run-id",
        default=DEFAULT_SINGLE_MODEL_RUN_ID,
    )
    parser.add_argument("--single-model-training-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.batch_size <= 0:
        parser.error("--workers and --batch-size must be positive")
    if args.max_transcripts is not None and args.max_transcripts <= 0:
        parser.error("--max-transcripts must be positive")
    input_metrics = args.input_metrics.resolve()
    if not input_metrics.exists():
        raise FileNotFoundError(
            f"Missing {input_metrics}. First run analyses/analyze_synthetic_input_data.py."
        )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(input_metrics, args.max_transcripts)
    replica_summary, kinetic_summary = metric_summaries(metrics)
    bias_prevalence, reproducibility_fidelity, association_summary = (
        additional_summaries(metrics, replica_summary, kinetic_summary)
    )
    replica_summary.to_csv(output_dir / "replica_agreement_summary.csv", index=False)
    kinetic_summary.to_csv(
        output_dir / "kinetic_target_agreement_summary.csv", index=False
    )
    bias_prevalence.to_csv(output_dir / "bias_prevalence_summary.csv", index=False)
    reproducibility_fidelity.to_csv(
        output_dir / "dataset_level_reproducibility_vs_fidelity.csv", index=False
    )
    association_summary.to_csv(
        output_dir / "reproducibility_fidelity_association.csv", index=False
    )

    single_model_results_root = args.single_model_results_root.resolve()
    if not single_model_results_root.is_dir():
        raise FileNotFoundError(
            f"Missing single-model results root: {single_model_results_root}"
        )
    (
        single_model_matched,
        single_model_summary,
        single_model_association,
        single_model_artifacts,
    ) = analyze_single_model_vs_replica_agreement(
        input_metrics_path=input_metrics,
        results_root=single_model_results_root,
        run_id=args.single_model_run_id,
        training_seed=args.single_model_training_seed,
    )
    single_model_matched.to_parquet(
        output_dir / "single_model_vs_replica_per_transcript.parquet",
        index=False,
        compression="zstd",
    )
    single_model_summary.to_csv(
        output_dir / "single_model_vs_replica_summary.csv", index=False
    )
    single_model_association.to_csv(
        output_dir / "single_model_vs_replica_association.csv", index=False
    )

    tasks = []
    worker_count = min(args.workers, len(DEPTHS))
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        for depth, nominal, _ in DEPTHS:
            output_path = output_dir / f"cross_dataset_per_transcript_{depth}.parquet"
            if output_path.exists():
                output_path.unlink()
            tasks.append(
                executor.submit(
                    cross_dataset_worker,
                    depth,
                    nominal,
                    str(output_path),
                    args.batch_size,
                    args.max_transcripts,
                )
            )
        worker_results = [future.result() for future in as_completed(tasks)]
    worker_results.sort(key=lambda item: item["nominal_reads_per_codon"])
    cross_summary = pd.DataFrame.from_records(
        row for result in worker_results for row in result["summaries"]
    )
    cross_summary.to_csv(
        output_dir / "cross_dataset_correlation_summary.csv", index=False
    )
    headline_mean_ci_summary(
        replica_summary, kinetic_summary, cross_summary
    ).to_csv(output_dir / "headline_mean_ci_summary.csv", index=False)

    typography = configure_iclr_typography()
    figures: list[Path] = []
    figures.extend(plot_replica_boxes(metrics, figure_dir))
    figures.extend(plot_kinetic_boxes(metrics, figure_dir))
    figures.extend(
        plot_single_model_comparison_boxes(single_model_matched, figure_dir)
    )
    figures.extend(
        plot_single_model_comparison_scatter(
            single_model_summary, single_model_association, figure_dir
        )
    )
    figures.extend(plot_cross_heatmaps(cross_summary, figure_dir))
    counts = metrics.groupby(["depth", "dataset"]).size()
    if counts.nunique() != 1:
        raise ValueError(f"Metric groups have unequal transcript counts: {counts.describe()}")
    transcripts_per_dataset = int(counts.iloc[0])
    provenance = {
        "analysis": "synthetic_individual_dataset_agreement",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "repository_root": str(ROOT),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "matplotlib": matplotlib.__version__,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_metrics": str(input_metrics),
        "input_metrics_sha256": sha256(input_metrics),
        "datasets": list(DATASET_ORDER),
        "excluded_observation_datasets": {
            "artificial_ground_truth": (
                "Excluded from all displayed dataset comparisons at user request; "
                "the deterministic K_t profile remains the truth target."
            )
        },
        "display_names": DISPLAY_NAMES,
        "plot_display_names": PLOT_DISPLAY_NAMES,
        "typography": {
            **typography,
            "style": "bold LaTeX serif typography for ICLR publication figures",
            "formats": ["PDF vector", "PNG 300 dpi", "SVG vector"],
        },
        "depths": [
            {"directory": depth, "nominal_reads_per_codon": nominal}
            for depth, nominal, _ in DEPTHS
        ],
        "transcripts_per_dataset": transcripts_per_dataset,
        "workers": worker_count,
        "batch_size": args.batch_size,
        "full_run": args.max_transcripts is None,
        "max_transcripts": args.max_transcripts,
        "boxplot_definition": "transcript PCC distribution; box=IQR, line=median, whiskers=p05/p95, no displayed fliers",
        "cross_dataset_definition": "for each transcript, PCC between arithmetic two-replica consensus profiles over all sense codons; matrix cell=median across transcripts",
        "single_model_comparison": {
            "run_id": args.single_model_run_id,
            "training_seed": args.single_model_training_seed,
            "checkpoint_variant": "best_val_loss",
            "definition": (
                "On each model's held-out transcript IDs, compare input "
                "PCC(replica1, replica2) with PCC(predicted mu, arithmetic "
                "two-replica consensus), retaining only rows where both PCCs "
                "are defined. No cross-depth intersection is imposed."
            ),
            "target_dependence_caveat": (
                "The observed target is formed from the same two replicas; "
                "model-consensus PCC and replica-replica PCC are not independent."
            ),
            "artifacts": single_model_artifacts,
        },
        "exact_identity_audit": "exact float equality of arithmetic consensus values; used to expose paired/common-random-number construction, not as a biological similarity metric",
        "terminal_convention": "remove exactly one validated appended zero terminal position",
        "parallelization": "one isolated process per read depth; no profile arrays shared between processes",
        "worker_results": [
            {key: value for key, value in result.items() if key != "summaries"}
            for result in worker_results
        ],
        "outputs": [str(path) for path in figures],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = render_html(
        replica_summary,
        kinetic_summary,
        cross_summary,
        bias_prevalence,
        reproducibility_fidelity,
        association_summary,
        single_model_summary,
        single_model_association,
        provenance,
    )
    report_path = output_dir / "individual_dataset_analysis.html"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
