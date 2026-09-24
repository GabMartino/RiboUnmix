#!/usr/bin/env python3
"""Audit the synthetic inputs and their weighted training artifacts.

This is an input-data analysis, not a model evaluation.  It streams the raw
synthetic count tables together with the corresponding weighted tables and
computes transcript-level shape metrics on the sense-codon coordinates.  The
primary sampled profile is the arithmetic mean of the two raw replicas, which
is also the consensus used by the multidataset loss.  The integerized ``ribo``
column is audited separately and is never treated as a third replicate.

Outputs include a compact Parquet table, CSV summaries, figures, provenance,
and a self-contained scientific HTML narrative (apart from local figure links).
No trained model or checkpoint is loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
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
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "Datasets" / "Synthetic_data"
WEIGHTED_ROOT = ROOT / "Datasets" / "data" / "weighted_synthetic"
SEQUENCE_PATH = (
    ROOT
    / "Datasets"
    / "data"
    / "sequence"
    / "MANE.selection.sequence_embeddings_with_css.parquet"
)
DEFAULT_OUTPUT = ROOT / "analyses" / "artifacts" / "synthetic" / "input_data"

DEPTHS: tuple[tuple[str, float], ...] = (
    ("0p25_per_codon", 0.25),
    ("2_per_codon", 2.0),
    ("20_per_codon", 20.0),
)
METRIC_SCHEMA = pa.schema(
    [
        ("depth", pa.string()),
        ("nominal_reads_per_codon", pa.float64()),
        ("dataset", pa.string()),
        ("is_biased", pa.bool_()),
        ("transcript_id", pa.string()),
        ("sense_length", pa.int32()),
        ("replicate_pcc", pa.float64()),
        ("replicate_pcc_interior10", pa.float64()),
        ("rep1_kinetic_pcc", pa.float64()),
        ("rep2_kinetic_pcc", pa.float64()),
        ("consensus_kinetic_pcc", pa.float64()),
        ("consensus_kinetic_pcc_interior10", pa.float64()),
        ("stored_kinetic_pcc", pa.float64()),
        ("consensus_bias_proxy_pcc", pa.float64()),
        ("consensus_gain_over_single_replica", pa.float64()),
        ("reproducibility_minus_kinetic", pa.float64()),
        ("bias_proxy_advantage", pa.float64()),
        ("consensus_kinetic_rmse_mean1", pa.float64()),
        ("consensus_bias_proxy_rmse_mean1", pa.float64()),
        ("weight", pa.float64()),
        ("coverage", pa.float64()),
        ("read_density", pa.float64()),
        ("bias_affected_fraction", pa.float64()),
        ("stored_vs_arithmetic_mae", pa.float64()),
    ]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_metadata(path: Path) -> dict[str, str]:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    return {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in metadata.items()
        if key != b"ARROW:schema" and key != b"pandas"
    }


def iter_rows(
    path: Path, columns: list[str], *, batch_size: int = 128
) -> Iterator[dict[str, Any]]:
    """Yield small Python row objects without materializing a full Parquet."""
    with pq.ParquetFile(path) as reader:
        for batch in reader.iter_batches(
            columns=columns, batch_size=batch_size, use_threads=False
        ):
            yield from batch.to_pylist()


def sample_role(sample: str) -> str:
    if sample == "rep1" or sample.endswith("_rep1"):
        return "rep1"
    if sample == "rep2" or sample.endswith("_rep2"):
        return "rep2"
    if sample == "mean" or sample.endswith("_mean"):
        return "mean"
    raise ValueError(f"Unrecognized synthetic sample label: {sample!r}")


def iter_grouped_profiles(
    path: Path, value_column: str, *, batch_size: int = 128
) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    """Yield the three sample rows belonging to each sorted transcript."""
    current_id: str | None = None
    current: dict[str, np.ndarray] = {}
    for row in iter_rows(
        path, ["sample", "transcript_id", value_column], batch_size=batch_size
    ):
        transcript_id = str(row["transcript_id"])
        if current_id is not None and transcript_id != current_id:
            if set(current) != {"rep1", "rep2", "mean"}:
                raise ValueError(
                    f"{path}: transcript {current_id!r} has roles {sorted(current)}"
                )
            yield current_id, current
            current = {}
        current_id = transcript_id
        role = sample_role(str(row["sample"]))
        if role in current:
            raise ValueError(f"{path}: duplicate {role} for {transcript_id}")
        current[role] = np.asarray(row[value_column], dtype=np.float64)
    if current_id is not None:
        if set(current) != {"rep1", "rep2", "mean"}:
            raise ValueError(
                f"{path}: transcript {current_id!r} has roles {sorted(current)}"
            )
        yield current_id, current


def iter_truth(path: Path, *, batch_size: int = 128) -> Iterator[tuple[str, np.ndarray]]:
    for row in iter_rows(
        path, ["transcript_id", "rib_profile"], batch_size=batch_size
    ):
        yield str(row["transcript_id"]), np.asarray(row["rib_profile"], dtype=np.float64)


@dataclass
class SortedCursor:
    iterator: Iterator[tuple[str, Any]]
    label: str

    def __post_init__(self) -> None:
        self.current = next(self.iterator, None)
        self.skipped_ids: list[str] = []
        self.last_requested: str | None = None

    def get(self, transcript_id: str) -> Any:
        if self.last_requested is not None and transcript_id <= self.last_requested:
            raise ValueError(
                f"Weighted IDs are not strictly sorted: {transcript_id!r} after "
                f"{self.last_requested!r}"
            )
        self.last_requested = transcript_id
        while self.current is not None and self.current[0] < transcript_id:
            self.skipped_ids.append(self.current[0])
            self.current = next(self.iterator, None)
        if self.current is None or self.current[0] != transcript_id:
            seen = None if self.current is None else self.current[0]
            raise ValueError(
                f"{self.label}: could not align {transcript_id!r}; next ID is {seen!r}"
            )
        value = self.current[1]
        self.current = next(self.iterator, None)
        return value


def pearson(x: np.ndarray, y: np.ndarray, *, trim: int = 0) -> float:
    """Pearson correlation without coercing undefined cases to zero."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Profile shape mismatch: {x.shape} versus {y.shape}")
    if trim:
        if x.size <= 2 * trim:
            return float("nan")
        x = x[trim:-trim]
        y = y[trim:-trim]
    if x.size < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    dx = x - x.mean()
    dy = y - y.mean()
    denominator = float(np.linalg.norm(dx) * np.linalg.norm(dy))
    if denominator <= np.finfo(np.float64).eps * x.size:
        return float("nan")
    return float(np.dot(dx, dy) / denominator)


def mean_one(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean()) if values.size else float("nan")
    if not np.isfinite(values).all() or not np.isfinite(mean) or mean <= 0.0:
        return None
    return values / mean


def rmse_mean_one(observed: np.ndarray, target_mean_one: np.ndarray) -> float:
    normalized = mean_one(observed)
    if normalized is None:
        return float("nan")
    if normalized.shape != target_mean_one.shape:
        raise ValueError("RMSE profiles are not aligned")
    return float(np.sqrt(np.mean(np.square(normalized - target_mean_one))))


def finite_mean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def q(series: pd.Series, probability: float) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, probability)) if values.size else float("nan")


def summarize_group(
    frame: pd.DataFrame,
    *,
    raw_rows: int,
    raw_transcripts: int,
    raw_replica_mismatches: int,
    raw_mean_mismatches: int,
    terminal_nonzero_rows: int,
    total_positions: int,
    total_reads: float,
    zero_replica_positions: int,
    zero_consensus_positions: int,
    skipped_raw_ids: int,
) -> dict[str, Any]:
    first = frame.iloc[0]
    result: dict[str, Any] = {
        "depth": first["depth"],
        "nominal_reads_per_codon": first["nominal_reads_per_codon"],
        "dataset": first["dataset"],
        "is_biased": bool(first["is_biased"]),
        "raw_rows": raw_rows,
        "raw_transcripts": raw_transcripts,
        "weighted_transcripts": len(frame),
        "raw_only_transcripts": skipped_raw_ids,
        "actual_reads_per_codon_per_replica": total_reads / (2.0 * total_positions),
        "replica_zero_fraction": zero_replica_positions / (2.0 * total_positions),
        "consensus_zero_fraction": zero_consensus_positions / total_positions,
        "raw_replica_mismatches": raw_replica_mismatches,
        "raw_integerized_mean_mismatches": raw_mean_mismatches,
        "terminal_nonzero_rows": terminal_nonzero_rows,
        "stored_vs_arithmetic_mae": float(frame["stored_vs_arithmetic_mae"].mean()),
        "median_weight": q(frame["weight"], 0.5),
        "weight_p05": q(frame["weight"], 0.05),
        "weight_p95": q(frame["weight"], 0.95),
        "median_coverage": q(frame["coverage"], 0.5),
        "median_read_density": q(frame["read_density"], 0.5),
        "median_bias_affected_fraction": q(frame["bias_affected_fraction"], 0.5),
    }
    metric_names = (
        "replicate_pcc",
        "replicate_pcc_interior10",
        "rep1_kinetic_pcc",
        "rep2_kinetic_pcc",
        "consensus_kinetic_pcc",
        "consensus_kinetic_pcc_interior10",
        "stored_kinetic_pcc",
        "consensus_bias_proxy_pcc",
        "consensus_gain_over_single_replica",
        "reproducibility_minus_kinetic",
        "bias_proxy_advantage",
        "consensus_kinetic_rmse_mean1",
        "consensus_bias_proxy_rmse_mean1",
    )
    for name in metric_names:
        finite = pd.to_numeric(frame[name], errors="coerce").dropna()
        result[f"valid_{name}"] = int(len(finite))
        result[f"median_{name}"] = q(finite, 0.5)
        result[f"q25_{name}"] = q(finite, 0.25)
        result[f"q75_{name}"] = q(finite, 0.75)
    result["spearman_weight_vs_replicate_pcc"] = float(
        frame[["weight", "replicate_pcc"]].corr(method="spearman").iloc[0, 1]
    )
    result["spearman_weight_vs_consensus_kinetic_pcc"] = float(
        frame[["weight", "consensus_kinetic_pcc"]]
        .corr(method="spearman")
        .iloc[0, 1]
    )
    result["spearman_weight_vs_consensus_bias_proxy_pcc"] = float(
        frame[["weight", "consensus_bias_proxy_pcc"]]
        .corr(method="spearman")
        .iloc[0, 1]
    )
    return result


def write_metric_group(
    writer: pq.ParquetWriter, records: list[dict[str, Any]]
) -> pd.DataFrame:
    table = pa.Table.from_pylist(records, schema=METRIC_SCHEMA)
    writer.write_table(table, row_group_size=1000)
    return table.to_pandas()


def process_dataset(
    *,
    depth: str,
    nominal_depth: float,
    dataset: str,
    max_transcripts: int | None,
    batch_size: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    pd.DataFrame,
    dict[str, Any],
]:
    weighted_path = WEIGHTED_ROOT / depth / f"{dataset}.parquet"
    raw_path = RAW_ROOT / depth / f"{dataset}_psite_counts_{depth}.parquet"
    truth_path = RAW_ROOT / "artificial_ground_truth_kinetics_target_mean_one.parquet"
    bias_path = (
        RAW_ROOT
        / "bias_profile"
        / f"{dataset}_compendium_added_bias_only.parquet"
    )
    is_biased = dataset != "artificial_ground_truth"
    if is_biased and not bias_path.exists():
        raise FileNotFoundError(f"Missing bias annotation: {bias_path}")

    raw_cursor = SortedCursor(
        iter_grouped_profiles(raw_path, "rib_profile", batch_size=batch_size),
        str(raw_path),
    )
    truth_cursor = SortedCursor(iter_truth(truth_path, batch_size=batch_size), str(truth_path))
    bias_cursor = (
        SortedCursor(
            iter_grouped_profiles(bias_path, "added_bias", batch_size=batch_size),
            str(bias_path),
        )
        if is_biased
        else None
    )

    raw_pf = pq.ParquetFile(raw_path)
    records: list[dict[str, Any]] = []
    raw_replica_mismatches = 0
    raw_mean_mismatches = 0
    terminal_nonzero_rows = 0
    total_positions = 0
    total_reads = 0.0
    zero_replica_positions = 0
    zero_consensus_positions = 0

    weighted_columns = [
        "id",
        "ribo",
        "ribo_cds_replicas",
        "weight",
        "coverage",
        "read_density",
    ]
    for row_number, row in enumerate(
        iter_rows(weighted_path, weighted_columns, batch_size=batch_size)
    ):
        if max_transcripts is not None and row_number >= max_transcripts:
            break
        transcript_id = str(row["id"])
        raw = raw_cursor.get(transcript_id)
        kinetic = np.asarray(truth_cursor.get(transcript_id), dtype=np.float64)
        replicas_with_stop = np.asarray(row["ribo_cds_replicas"], dtype=np.float64)
        stored_with_stop = np.asarray(row["ribo"], dtype=np.float64)
        if replicas_with_stop.ndim != 2 or replicas_with_stop.shape[0] != 2:
            raise ValueError(
                f"{weighted_path}: {transcript_id} does not contain exactly two replicas"
            )
        if replicas_with_stop.shape[1] != kinetic.size + 1:
            raise ValueError(
                f"{weighted_path}: {transcript_id} length {replicas_with_stop.shape[1]} "
                f"does not equal kinetic length + 1 ({kinetic.size + 1})"
            )
        terminal_nonzero_rows += int(
            np.any(replicas_with_stop[:, -1] != 0.0) or stored_with_stop[-1] != 0.0
        )
        rep1 = replicas_with_stop[0, :-1]
        rep2 = replicas_with_stop[1, :-1]
        stored = stored_with_stop[:-1]
        raw_replica_mismatches += int(not np.array_equal(rep1, raw["rep1"]))
        raw_replica_mismatches += int(not np.array_equal(rep2, raw["rep2"]))
        raw_mean_mismatches += int(not np.array_equal(stored, raw["mean"]))
        if not np.isfinite(kinetic).all() or np.any(kinetic <= 0.0):
            raise ValueError(f"Invalid kinetic target for {transcript_id}")
        if not math.isclose(float(kinetic.mean()), 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"Kinetic target is not mean-one for {transcript_id}")

        if bias_cursor is None:
            added_bias = np.zeros_like(kinetic)
        else:
            bias_profiles = bias_cursor.get(transcript_id)
            added_bias = np.asarray(bias_profiles["rep1"], dtype=np.float64)
            if not (
                np.array_equal(added_bias, bias_profiles["rep2"])
                and np.array_equal(added_bias, bias_profiles["mean"])
            ):
                raise ValueError(
                    f"Bias annotation differs among samples for {dataset}/{transcript_id}"
                )
            if added_bias.shape != kinetic.shape or np.any(added_bias < 0.0):
                raise ValueError(f"Invalid bias annotation for {dataset}/{transcript_id}")

        consensus = 0.5 * (rep1 + rep2)
        bias_proxy = mean_one(kinetic * (1.0 + added_bias))
        if bias_proxy is None:
            raise ValueError(f"Invalid bias-aware proxy for {dataset}/{transcript_id}")
        r1_kinetic = pearson(rep1, kinetic)
        r2_kinetic = pearson(rep2, kinetic)
        consensus_kinetic = pearson(consensus, kinetic)
        replicate_pcc = pearson(rep1, rep2)
        consensus_proxy = pearson(consensus, bias_proxy)
        mean_single = finite_mean((r1_kinetic, r2_kinetic))

        total_positions += kinetic.size
        total_reads += float(rep1.sum() + rep2.sum())
        zero_replica_positions += int(np.count_nonzero(rep1 == 0.0))
        zero_replica_positions += int(np.count_nonzero(rep2 == 0.0))
        zero_consensus_positions += int(np.count_nonzero(consensus == 0.0))
        records.append(
            {
                "depth": depth,
                "nominal_reads_per_codon": float(nominal_depth),
                "dataset": dataset,
                "is_biased": is_biased,
                "transcript_id": transcript_id,
                "sense_length": int(kinetic.size),
                "replicate_pcc": replicate_pcc,
                "replicate_pcc_interior10": pearson(rep1, rep2, trim=10),
                "rep1_kinetic_pcc": r1_kinetic,
                "rep2_kinetic_pcc": r2_kinetic,
                "consensus_kinetic_pcc": consensus_kinetic,
                "consensus_kinetic_pcc_interior10": pearson(
                    consensus, kinetic, trim=10
                ),
                "stored_kinetic_pcc": pearson(stored, kinetic),
                "consensus_bias_proxy_pcc": consensus_proxy,
                "consensus_gain_over_single_replica": (
                    consensus_kinetic - mean_single
                    if np.isfinite(consensus_kinetic) and np.isfinite(mean_single)
                    else float("nan")
                ),
                "reproducibility_minus_kinetic": (
                    replicate_pcc - consensus_kinetic
                    if np.isfinite(replicate_pcc) and np.isfinite(consensus_kinetic)
                    else float("nan")
                ),
                "bias_proxy_advantage": (
                    consensus_proxy - consensus_kinetic
                    if np.isfinite(consensus_proxy) and np.isfinite(consensus_kinetic)
                    else float("nan")
                ),
                "consensus_kinetic_rmse_mean1": rmse_mean_one(consensus, kinetic),
                "consensus_bias_proxy_rmse_mean1": rmse_mean_one(
                    consensus, bias_proxy
                ),
                "weight": float(row["weight"]),
                "coverage": float(row["coverage"]),
                "read_density": float(row["read_density"]),
                "bias_affected_fraction": float(np.mean(added_bias > 0.0)),
                "stored_vs_arithmetic_mae": float(np.mean(np.abs(stored - consensus))),
            }
        )

    if not records:
        raise RuntimeError(f"No rows processed for {depth}/{dataset}")
    frame = pd.DataFrame.from_records(records)
    summary = summarize_group(
        frame,
        raw_rows=raw_pf.metadata.num_rows,
        raw_transcripts=raw_pf.metadata.num_rows // 3,
        raw_replica_mismatches=raw_replica_mismatches,
        raw_mean_mismatches=raw_mean_mismatches,
        terminal_nonzero_rows=terminal_nonzero_rows,
        total_positions=total_positions,
        total_reads=total_reads,
        zero_replica_positions=zero_replica_positions,
        zero_consensus_positions=zero_consensus_positions,
        skipped_raw_ids=len(raw_cursor.skipped_ids),
    )

    ordered = frame.sort_values(["weight", "transcript_id"]).reset_index(drop=True)
    ordered["weight_decile"] = (
        np.floor(np.arange(len(ordered), dtype=np.float64) * 10.0 / len(ordered))
        .astype(int)
        .clip(0, 9)
        + 1
    )
    calibration: list[dict[str, Any]] = []
    for decile, subset in ordered.groupby("weight_decile", sort=True):
        calibration.append(
            {
                "depth": depth,
                "nominal_reads_per_codon": nominal_depth,
                "dataset": dataset,
                "is_biased": is_biased,
                "weight_decile": int(decile),
                "n": len(subset),
                "median_weight": q(subset["weight"], 0.5),
                "median_replicate_pcc": q(subset["replicate_pcc"], 0.5),
                "median_consensus_kinetic_pcc": q(
                    subset["consensus_kinetic_pcc"], 0.5
                ),
                "median_consensus_bias_proxy_pcc": q(
                    subset["consensus_bias_proxy_pcc"], 0.5
                ),
            }
        )

    sample_size = min(400, len(frame))
    indices = np.linspace(0, len(frame) - 1, sample_size, dtype=int)
    plot_sample = frame.iloc[indices].copy()
    source = {
        "raw_path": str(raw_path.relative_to(ROOT)),
        "weighted_path": str(weighted_path.relative_to(ROOT)),
        "bias_path": str(bias_path.relative_to(ROOT)) if is_biased else None,
        "raw_metadata": parquet_metadata(raw_path),
    }
    return records, summary, calibration, plot_sample, source


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_outputs(
    output_dir: Path,
    summaries: pd.DataFrame,
    calibration: pd.DataFrame,
    plot_sample: pd.DataFrame,
) -> list[Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    depth_order = [value for _, value in DEPTHS]
    depth_colors = {0.25: "#3B75AF", 2.0: "#E6862F", 20.0: "#3A923A"}
    outputs: list[Path] = []

    unbiased = summaries.loc[~summaries["is_biased"]].sort_values(
        "nominal_reads_per_codon"
    )
    fig, axis = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    metrics = [
        ("median_rep1_kinetic_pcc", "replicate 1 vs kinetic target", "o"),
        ("median_rep2_kinetic_pcc", "replicate 2 vs kinetic target", "s"),
        ("median_consensus_kinetic_pcc", "two-replica mean vs kinetic target", "D"),
        ("median_replicate_pcc", "replicate 1 vs replicate 2", "^")
    ]
    for column, label, marker in metrics:
        axis.plot(
            unbiased["nominal_reads_per_codon"],
            unbiased[column],
            marker=marker,
            linewidth=2.2,
            markersize=7,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_xticks(depth_order, ["0.25", "2", "20"])
    axis.set_ylim(-0.05, 1.0)
    axis.set_xlabel("Nominal unbiased reads per codon")
    axis.set_ylabel("Median transcript-level PCC")
    axis.set_title("Unbiased observations: sampling depth and profile agreement")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    stem = figure_dir / "unbiased_recovery_by_depth"
    save_figure(fig, stem)
    outputs.extend([stem.with_suffix(".svg"), stem.with_suffix(".png")])

    dataset_order = sorted(summaries["dataset"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.2), constrained_layout=True)
    heat_metrics = [
        ("median_consensus_kinetic_pcc", "Consensus vs kinetic target", -0.1, 1.0, "viridis"),
        ("median_replicate_pcc", "Replicate reproducibility", -0.1, 1.0, "viridis"),
        ("median_bias_proxy_advantage", "Bias-proxy PCC advantage", 0.0, 0.55, "magma"),
    ]
    for axis, (column, title, vmin, vmax, cmap) in zip(axes, heat_metrics):
        pivot = (
            summaries.pivot(
                index="dataset", columns="nominal_reads_per_codon", values=column
            )
            .reindex(index=dataset_order, columns=depth_order)
        )
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_xticks(range(3), ["0.25", "2", "20"])
        axis.set_yticks(range(len(dataset_order)), dataset_order, fontsize=8)
        axis.set_xlabel("Nominal reads per codon")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, shrink=0.75)
    stem = figure_dir / "dataset_metric_heatmaps"
    save_figure(fig, stem)
    outputs.extend([stem.with_suffix(".svg"), stem.with_suffix(".png")])

    fig, axis = plt.subplots(figsize=(7.3, 6.0), constrained_layout=True)
    for depth_value in depth_order:
        subset = summaries.loc[summaries["nominal_reads_per_codon"] == depth_value]
        for biased, marker, label_suffix in ((False, "*", "unbiased"), (True, "o", "biased")):
            part = subset.loc[subset["is_biased"] == biased]
            axis.scatter(
                part["median_replicate_pcc"],
                part["median_consensus_kinetic_pcc"],
                color=depth_colors[depth_value],
                marker=marker,
                s=110 if not biased else 55,
                edgecolor="white",
                linewidth=0.6,
                label=f"{depth_value:g} reads/codon, {label_suffix}",
            )
    axis.axline((0, 0), (1, 1), color="#777777", linestyle="--", linewidth=1)
    axis.set_xlim(-0.1, 1.0)
    axis.set_ylim(-0.1, 1.0)
    axis.set_xlabel("Median replicate--replicate PCC")
    axis.set_ylabel("Median consensus--kinetic-target PCC")
    axis.set_title("Reproducibility does not guarantee latent-target agreement")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    stem = figure_dir / "reproducibility_vs_kinetic_agreement"
    save_figure(fig, stem)
    outputs.extend([stem.with_suffix(".svg"), stem.with_suffix(".png")])

    aggregate = (
        calibration.groupby(["nominal_reads_per_codon", "weight_decile"], as_index=False)
        .agg(
            median_replicate_pcc=("median_replicate_pcc", "median"),
            median_consensus_kinetic_pcc=("median_consensus_kinetic_pcc", "median"),
            median_consensus_bias_proxy_pcc=(
                "median_consensus_bias_proxy_pcc", "median"
            ),
            median_weight=("median_weight", "median"),
            datasets=("dataset", "nunique"),
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), constrained_layout=True)
    for depth_value in depth_order:
        subset = aggregate.loc[aggregate["nominal_reads_per_codon"] == depth_value]
        axes[0].plot(
            subset["weight_decile"],
            subset["median_replicate_pcc"],
            marker="o",
            color=depth_colors[depth_value],
            label=f"{depth_value:g}",
        )
        axes[1].plot(
            subset["weight_decile"],
            subset["median_consensus_kinetic_pcc"],
            marker="o",
            color=depth_colors[depth_value],
            label=f"{depth_value:g}",
        )
        axes[2].plot(
            subset["weight_decile"],
            subset["median_consensus_bias_proxy_pcc"],
            marker="o",
            color=depth_colors[depth_value],
            label=f"{depth_value:g}",
        )
    axes[0].set_title("Replicate reproducibility")
    axes[1].set_title("Kinetic-target agreement")
    axes[2].set_title("Bias-aware proxy agreement")
    for axis in axes:
        axis.set_xlabel("Within-dataset reliability-weight decile")
        axis.set_ylabel("Median transcript-level PCC")
        axis.set_xticks(range(1, 11))
        axis.grid(alpha=0.2)
        axis.legend(title="reads/codon", frameon=False)
    stem = figure_dir / "weight_calibration"
    save_figure(fig, stem)
    outputs.extend([stem.with_suffix(".svg"), stem.with_suffix(".png")])

    # The retained sample is a source table for optional later distribution
    # plots; it is intentionally not used to calculate any reported summary.
    plot_sample.to_csv(output_dir / "plot_sample.csv", index=False)
    return outputs


def format_table(frame: pd.DataFrame, *, digits: int = 3) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        if pd.api.types.is_integer_dtype(display[column]):
            continue
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    return display.to_html(index=False, border=0, classes="data-table", escape=True)


def render_html(
    output_dir: Path,
    summaries: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    dropped_ids: list[str],
    provenance: dict[str, Any],
) -> str:
    unbiased = (
        summaries.loc[~summaries["is_biased"]]
        .sort_values("nominal_reads_per_codon")
        .copy()
    )
    unbiased_table = unbiased[
        [
            "nominal_reads_per_codon",
            "actual_reads_per_codon_per_replica",
            "median_rep1_kinetic_pcc",
            "median_rep2_kinetic_pcc",
            "median_consensus_kinetic_pcc",
            "median_replicate_pcc",
            "median_consensus_gain_over_single_replica",
            "median_consensus_kinetic_rmse_mean1",
        ]
    ].rename(
        columns={
            "nominal_reads_per_codon": "nominal reads/codon",
            "actual_reads_per_codon_per_replica": "actual reads/codon/replica",
            "median_rep1_kinetic_pcc": "median PCC(rep1,K)",
            "median_rep2_kinetic_pcc": "median PCC(rep2,K)",
            "median_consensus_kinetic_pcc": "median PCC(mean,K)",
            "median_replicate_pcc": "median PCC(rep1,rep2)",
            "median_consensus_gain_over_single_replica": "median consensus gain",
            "median_consensus_kinetic_rmse_mean1": "median mean-one RMSE",
        }
    )
    compact = summaries[
        [
            "nominal_reads_per_codon",
            "dataset",
            "actual_reads_per_codon_per_replica",
            "replica_zero_fraction",
            "median_replicate_pcc",
            "median_consensus_kinetic_pcc",
            "median_consensus_bias_proxy_pcc",
            "median_bias_proxy_advantage",
            "median_weight",
            "spearman_weight_vs_replicate_pcc",
            "spearman_weight_vs_consensus_bias_proxy_pcc",
        ]
    ].sort_values(["dataset", "nominal_reads_per_codon"])
    compact = compact.rename(
        columns={
            "nominal_reads_per_codon": "nominal depth",
            "actual_reads_per_codon_per_replica": "actual depth",
            "replica_zero_fraction": "zero fraction",
            "median_replicate_pcc": "median replica PCC",
            "median_consensus_kinetic_pcc": "median consensus--K PCC",
            "median_consensus_bias_proxy_pcc": "median consensus--proxy PCC",
            "median_bias_proxy_advantage": "proxy advantage",
            "median_weight": "median weight",
            "spearman_weight_vs_replicate_pcc": "Spearman(weight, replica PCC)",
            "spearman_weight_vs_consensus_bias_proxy_pcc": "Spearman(weight, proxy PCC)",
        }
    )

    low = unbiased.iloc[0]
    high = unbiased.iloc[-1]
    biased = summaries.loc[summaries["is_biased"]]
    median_proxy_advantage_high = q(
        biased.loc[
            biased["nominal_reads_per_codon"] == 20.0,
            "median_bias_proxy_advantage",
        ],
        0.5,
    )
    median_weight_rep_association = q(
        summaries["spearman_weight_vs_replicate_pcc"], 0.5
    )
    weight_association_by_depth = (
        summaries.groupby("nominal_reads_per_codon")
        .agg(
            rho_replicate=("spearman_weight_vs_replicate_pcc", "median"),
            rho_kinetic=("spearman_weight_vs_consensus_kinetic_pcc", "median"),
            rho_bias_proxy=(
                "spearman_weight_vs_consensus_bias_proxy_pcc", "median"
            ),
        )
        .reindex([0.25, 2.0, 20.0])
    )
    stored_mae = q(summaries["stored_vs_arithmetic_mae"], 0.5)
    invalid_replica = int(
        (summaries["weighted_transcripts"] - summaries["valid_replicate_pcc"]).sum()
    )
    invalid_consensus = int(
        (
            summaries["weighted_transcripts"]
            - summaries["valid_consensus_kinetic_pcc"]
        ).sum()
    )
    report_time = html.escape(provenance["created_at"])
    dropped = ", ".join(f"<code>{html.escape(item)}</code>" for item in dropped_ids)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic input-data audit</title>
<style>
:root{{--ink:#1d2a32;--muted:#58666f;--blue:#155f84;--line:#d8e1e6;--soft:#f3f7f9;--warn:#fff4df}}
*{{box-sizing:border-box}} body{{max-width:1280px;margin:30px auto;padding:0 28px;color:var(--ink);font:17px/1.58 Georgia,serif}}
h1,h2,h3{{line-height:1.22}} h1{{font-size:36px}} h2{{margin-top:2.2em;padding-top:16px;border-top:1px solid var(--line)}}
p,li{{max-width:105ch}} nav{{display:flex;gap:10px 22px;flex-wrap:wrap;font:14px system-ui,sans-serif}}
a{{color:var(--blue)}} code,pre{{font-family:ui-monospace,Consolas,monospace}} .meta{{color:var(--muted);font-size:15px}}
.finding,.warning{{padding:15px 20px;margin:18px 0;background:var(--soft);border-left:5px solid var(--blue)}}
.warning{{background:var(--warn);border-color:#bd762e}} .formula{{padding:12px 18px;background:var(--soft);overflow:auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:20px}} figure{{margin:18px 0}}
figure img{{width:100%;height:auto}} figcaption{{font-size:14px;color:var(--muted)}} .table-wrap{{overflow-x:auto;margin:18px 0}}
table{{border-collapse:collapse;width:100%;font:13px/1.4 system-ui,sans-serif}} th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th{{background:#eaf1f4;position:sticky;top:0}} td:nth-child(2),th:nth-child(2){{text-align:left}}
@media(max-width:700px){{body{{padding:0 15px}}h1{{font-size:28px}}.grid{{display:block}}}}
@media print{{body{{max-width:none}}nav{{display:none}}figure,.finding,.warning{{break-inside:avoid}}}}
</style>
</head>
<body>
<header>
<h1>From synthetic truth to weighted training profiles</h1>
<p class="meta">Generated {report_time}. This report analyzes <code>Datasets/Synthetic_data</code> and
<code>Datasets/data/weighted_synthetic</code>. It does not load a model or inspect fitted predictions.</p>
<nav><a href="#map">Data map</a><a href="#metrics">Definitions</a><a href="#results">Results</a>
<a href="#bias">Bias</a><a href="#weights">Weights</a><a href="#integrity">Integrity</a><a href="#limits">Limitations</a></nav>
</header>

<div class="finding"><strong>Main measured result.</strong> In the unbiased observation, the median transcript-level
PCC between the arithmetic two-replica consensus and the programmed mean-one kinetic profile rises from
<strong>{low['median_consensus_kinetic_pcc']:.3f}</strong> at 0.25 nominal reads/codon to
<strong>{high['median_consensus_kinetic_pcc']:.3f}</strong> at 20 reads/codon. Replica--replica PCC rises from
<strong>{low['median_replicate_pcc']:.3f}</strong> to <strong>{high['median_replicate_pcc']:.3f}</strong>.
These are descriptive profile-wise medians over the same validated transcript universe, not codon-pooled correlations.</div>

<div class="warning"><strong>Important interpretation.</strong> The exported “ground truth” is the programmed
dwell-time/kinetic profile <em>K</em>. Counts were generated after ribosome traffic simulation, sequence-bias injection,
and NB2 sampling. The unnoised trajectory-occupancy profiles are not exported. Therefore PCC(consensus, K) measures
agreement with the kinetic target; it is not a pure sampling-noise ceiling and cannot isolate traffic distortion.</div>

<section id="map"><h2>1. What is in the two directories?</h2>
<ul>
<li><code>artificial_ground_truth_kinetics_target_mean_one.parquet</code>: 19,290 positive, mean-one programmed kinetic profiles.</li>
<li>Three depth directories: 0.25, 2, and 20 nominal unbiased reads/codon. Each contains one unbiased and ten biased datasets.</li>
<li>Every raw count table contains two NB2 replicas and one deterministic integerization of their mean. The latter is not a third draw.</li>
<li>The weighted artifact retains 19,283 transcripts, appends one zero terminal-boundary position, preserves both replicas, and adds coverage, density, and reliability weights.</li>
</ul>
<p>The seven removed IDs were absent from the master sequence table, not removed because of their synthetic profiles: {dropped}.</p>
</section>

<section id="metrics"><h2>2. Definitions and comparison hierarchy</h2>
<div class="formula"><code>K_t</code> = programmed mean-one kinetic profile;<br>
<code>Y_dt1, Y_dt2</code> = the two sampled count replicas;<br>
<code>Ybar_dt = (Y_dt1 + Y_dt2)/2</code> = exact arithmetic consensus used for PCC supervision;<br>
<code>H_dt = mean_one(K_t * b_dt)</code> = bias-aware kinetic proxy, where the annotation stores <code>b_dt - 1</code>.</div>
<ol>
<li><strong>Kinetic-target agreement:</strong> PCC(<code>Ybar</code>, <code>K</code>), plus each individual replica versus K.</li>
<li><strong>Technical reproducibility:</strong> PCC(replica 1, replica 2). High reproducibility does not prove truth recovery.</li>
<li><strong>Consensus gain:</strong> PCC(<code>Ybar</code>, K) minus the mean of the two single-replica PCCs.</li>
<li><strong>Systematic-discordance diagnostic:</strong> PCC(rep1, rep2) minus PCC(<code>Ybar</code>, K). A positive value can reflect shared systematic bias, but is not itself a bias estimator.</li>
<li><strong>Known-bias diagnostic:</strong> PCC(<code>Ybar</code>, H) minus PCC(<code>Ybar</code>, K). H is a proxy, not the unavailable traffic-aware generating mean.</li>
<li><strong>Amplitude error:</strong> RMSE after independently scaling the sampled consensus to mean one. PCC itself is scale invariant.</li>
</ol>
<p>All primary metrics use every aligned sense codon. A separately exported interior-10 PCC removes ten codons from each end.
Constant profiles produce undefined PCC and remain missing: {invalid_replica:,} replica PCCs and {invalid_consensus:,} consensus--K PCCs
were undefined across all dataset--transcript rows.</p>
</section>

<section id="results"><h2>3. Unbiased observations: what sampling depth changes</h2>
<div class="table-wrap">{format_table(unbiased_table, digits=3)}</div>
<figure><img src="figures/unbiased_recovery_by_depth.svg" alt="Unbiased profile correlations by depth">
<figcaption>Each point is the median of transcript-level PCCs. The consensus curve demonstrates the benefit of averaging two noisy replicas.</figcaption></figure>
<p>The improvement with depth is large but should not be described as “model recovery”: these are input observations.
The consensus gain quantifies how much the arithmetic two-replica average improves kinetic-target agreement relative to a typical single replica.</p>
</section>

<section id="bias"><h2>4. Sequence bias and the reproducibility trap</h2>
<div class="grid">
<figure><img src="figures/dataset_metric_heatmaps.svg" alt="Dataset metric heatmaps"><figcaption>Dataset-level medians across all 11 conditions and three depths.</figcaption></figure>
<figure><img src="figures/reproducibility_vs_kinetic_agreement.svg" alt="Reproducibility versus kinetic agreement"><figcaption>Each point is one dataset-depth summary. Stars mark the unbiased control.</figcaption></figure>
</div>
<p>At 20 reads/codon, the median across the ten biased datasets of
PCC(consensus, bias-aware proxy) minus PCC(consensus, K) is <strong>{median_proxy_advantage_high:.3f}</strong>.
A positive proxy advantage is expected when the injected bias is visible. It is useful precisely because replica reproducibility alone
cannot reveal a bias shared by both replicas.</p>
<div class="warning">Do not call H an exact biased ground truth. The simulator applies the multiplier to a traffic-derived occupancy profile,
whereas H substitutes the available kinetic profile K. H tests whether the known bias annotation points in the expected direction.</div>
</section>

<section id="weights"><h2>5. Does the artifact weight track empirical reliability?</h2>
<p>The artifact uses <code>0.70 * sqrt(D)/(sqrt(D)+sqrt(tau_d)) + 0.30 * coverage</code>, followed by a
dataset-specific median normalization. These preprocessing references were fitted on all eligible artifact rows, not on a later train split.
The median dataset-level Spearman association between weight and replica PCC is <strong>{median_weight_rep_association:.3f}</strong>.</p>
<p>Across depths 0.25, 2, and 20, the median dataset-level Spearman associations with kinetic-target PCC are
<strong>{weight_association_by_depth.iloc[0]['rho_kinetic']:.3f}, {weight_association_by_depth.iloc[1]['rho_kinetic']:.3f},
and {weight_association_by_depth.iloc[2]['rho_kinetic']:.3f}</strong>; the corresponding associations with the bias-aware proxy are
<strong>{weight_association_by_depth.iloc[0]['rho_bias_proxy']:.3f}, {weight_association_by_depth.iloc[1]['rho_bias_proxy']:.3f},
and {weight_association_by_depth.iloc[2]['rho_bias_proxy']:.3f}</strong>. Thus the weight tracks supported observed structure,
including reproducible technical bias; it is deliberately not a weight for proximity to the latent kinetic truth.</p>
<p>One implementation detail matters when reproducing these values: the artifact's <code>D</code> and coverage are calculated from the
stored integerized <code>ribo</code> profile after its terminal zero is appended, not from the floating arithmetic replica consensus.
The model can still reconstruct the latter from <code>ribo_cds_replicas</code>.</p>
<figure><img src="figures/weight_calibration.svg" alt="Weight deciles and empirical correlations"><figcaption>
Each line first forms weight deciles within a dataset; the plotted value is the median of the 11 dataset-specific decile medians.
This is a calibration diagnostic, not evidence that the weight is an inverse-variance optimum.</figcaption></figure>
<div class="table-wrap">{format_table(calibration_summary, digits=3)}</div>
</section>

<section id="integrity"><h2>6. Raw-to-weighted integrity checks</h2>
<ul>
<li>All weighted replica arrays exactly matched their corresponding raw replica rows.</li>
<li>All weighted stored <code>ribo</code> arrays matched the raw deterministic integerized-mean row.</li>
<li>Every weighted profile had exactly one appended terminal position and it was zero.</li>
<li>The median across dataset-depth conditions of the position-wise absolute difference between stored integerized <code>ribo</code>
and the exact arithmetic two-replica mean was {stored_mae:.3f} count units. This is why the report uses the replicas to reconstruct the consensus.</li>
</ul>
<div class="table-wrap">{format_table(compact, digits=3)}</div>
</section>

<section id="limits"><h2>7. What is reasonable to conclude?</h2>
<ul>
<li><strong>Supported:</strong> greater count depth improves both replicate reproducibility and agreement of the unbiased sampled consensus with K.</li>
<li><strong>Supported:</strong> averaging the two replicas improves agreement with K relative to a typical single replica.</li>
<li><strong>Supported:</strong> biased datasets can be internally reproducible while diverging from K; the known-bias proxy often explains part of that divergence.</li>
<li><strong>Not supported:</strong> these PCCs are not performance estimates for RiboUnmix, because no model prediction is analyzed.</li>
<li><strong>Not supported:</strong> PCC(rep1, rep2) is not biological accuracy, and H is not an exact traffic-aware oracle.</li>
<li><strong>Dependence:</strong> the same transcripts and shared simulator design recur across conditions. Millions of codons must not be treated as independent replicates.</li>
<li><strong>Single simulation realization:</strong> the report describes the supplied seeds. It does not quantify uncertainty over new simulator or count-sampling seeds.</li>
</ul>
</section>

<section><h2>8. Reproduction and source tables</h2>
<pre>OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
.venv/bin/python analyses/analyze_synthetic_input_data.py --overwrite</pre>
<ul>
<li><a href="per_transcript_metrics.parquet">Per-transcript scalar metrics</a></li>
<li><a href="dataset_summary.csv">Dataset-depth summaries</a></li>
<li><a href="weight_calibration_by_dataset.csv">Dataset-specific weight calibration</a></li>
<li><a href="weight_calibration_summary.csv">Equal-dataset calibration summary</a></li>
<li><a href="provenance.json">Analysis provenance</a></li>
</ul>
</section>
</body></html>"""


def discover_datasets() -> list[str]:
    per_depth: list[list[str]] = []
    for depth, _ in DEPTHS:
        names = sorted(path.stem for path in (WEIGHTED_ROOT / depth).glob("*.parquet"))
        if not names:
            raise FileNotFoundError(f"No weighted synthetic files below {WEIGHTED_ROOT / depth}")
        per_depth.append(names)
    if any(names != per_depth[0] for names in per_depth[1:]):
        raise ValueError("Synthetic dataset names differ among depth directories")
    if "artificial_ground_truth" not in per_depth[0]:
        raise ValueError("The unbiased artificial_ground_truth dataset is missing")
    return per_depth[0]


def dropped_transcript_ids() -> list[str]:
    raw_path = (
        RAW_ROOT
        / "2_per_codon"
        / "artificial_ground_truth_psite_counts_2_per_codon.parquet"
    )
    weighted_path = WEIGHTED_ROOT / "2_per_codon" / "artificial_ground_truth.parquet"
    raw_ids = set(
        pq.read_table(raw_path, columns=["transcript_id"])
        .column("transcript_id")
        .to_pylist()
    )
    weighted_ids = set(
        pq.read_table(weighted_path, columns=["id"]).column("id").to_pylist()
    )
    sequence_ids = set(
        pq.read_table(SEQUENCE_PATH, columns=["transcript_id"])
        .column("transcript_id")
        .to_pylist()
    )
    dropped = sorted(raw_ids - weighted_ids)
    if set(dropped) != raw_ids - sequence_ids:
        raise ValueError("Dropped synthetic IDs are not explained by the master sequence table")
    return dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-transcripts", type=int, help="Development smoke-test limit per dataset")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_transcripts is not None and args.max_transcripts <= 0:
        parser.error("--max-transcripts must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "per_transcript_metrics.parquet"
    if metrics_path.exists():
        metrics_path.unlink()

    datasets = discover_datasets()
    dropped_ids = dropped_transcript_ids()
    summaries: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    plot_samples: list[pd.DataFrame] = []
    inputs: list[dict[str, Any]] = []
    with pq.ParquetWriter(
        metrics_path, METRIC_SCHEMA, compression="zstd", use_dictionary=True
    ) as writer:
        for depth, nominal_depth in DEPTHS:
            for dataset in datasets:
                print(f"[{depth}] {dataset}", flush=True)
                records, summary, calibration, plot_sample, source = process_dataset(
                    depth=depth,
                    nominal_depth=nominal_depth,
                    dataset=dataset,
                    max_transcripts=args.max_transcripts,
                    batch_size=args.batch_size,
                )
                frame = write_metric_group(writer, records)
                summaries.append(summary)
                calibrations.extend(calibration)
                plot_samples.append(plot_sample)
                inputs.append(
                    {
                        "depth": depth,
                        "dataset": dataset,
                        **source,
                    }
                )
                print(
                    f"  n={len(frame):,}; median r12={summary['median_replicate_pcc']:.3f}; "
                    f"median r(mean,K)={summary['median_consensus_kinetic_pcc']:.3f}",
                    flush=True,
                )

    summary_frame = pd.DataFrame.from_records(summaries)
    calibration_frame = pd.DataFrame.from_records(calibrations)
    plot_sample_frame = pd.concat(plot_samples, ignore_index=True)
    summary_frame.to_csv(output_dir / "dataset_summary.csv", index=False)
    calibration_frame.to_csv(
        output_dir / "weight_calibration_by_dataset.csv", index=False
    )
    calibration_summary = (
        calibration_frame.groupby(
            ["nominal_reads_per_codon", "weight_decile"], as_index=False
        )
        .agg(
            datasets=("dataset", "nunique"),
            median_weight=("median_weight", "median"),
            median_replicate_pcc=("median_replicate_pcc", "median"),
            median_consensus_kinetic_pcc=(
                "median_consensus_kinetic_pcc",
                "median",
            ),
            median_consensus_bias_proxy_pcc=(
                "median_consensus_bias_proxy_pcc",
                "median",
            ),
        )
        .sort_values(["nominal_reads_per_codon", "weight_decile"])
    )
    calibration_summary.to_csv(
        output_dir / "weight_calibration_summary.csv", index=False
    )
    figure_paths = plot_outputs(
        output_dir, summary_frame, calibration_frame, plot_sample_frame
    )

    integrity_columns = [
        "raw_replica_mismatches",
        "raw_integerized_mean_mismatches",
        "terminal_nonzero_rows",
    ]
    integrity_totals = {
        column: int(summary_frame[column].sum()) for column in integrity_columns
    }
    if any(integrity_totals.values()):
        raise RuntimeError(f"Raw-to-weighted integrity checks failed: {integrity_totals}")
    provenance = {
        "analysis": "synthetic_input_and_weighted_artifact_audit",
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
        "depths": [{"label": label, "nominal": value} for label, value in DEPTHS],
        "datasets": datasets,
        "full_run": args.max_transcripts is None,
        "max_transcripts_per_dataset": args.max_transcripts,
        "batch_size": args.batch_size,
        "dropped_ids": dropped_ids,
        "metric_definition": {
            "coordinate_domain": "all sense codons; exactly one appended terminal zero removed",
            "sampled_representation": "arithmetic mean of the two raw replicas",
            "primary_correlation": "transcript-level Pearson correlation; undefined remains NaN",
            "kinetic_target": "programmed mean-one dwell-time profile K; not unnoised trajectory occupancy",
            "bias_proxy": "mean_one(K * (1 + added_bias)); diagnostic, not exact traffic-aware truth",
            "interior_sensitivity": "remove first and last 10 sense codons",
            "weight_calibration": "deciles formed separately within every dataset-depth table",
        },
        "integrity_totals": integrity_totals,
        "inputs": inputs,
        "outputs": {
            "metrics": str(metrics_path),
            "dataset_summary": str(output_dir / "dataset_summary.csv"),
            "weight_calibration": str(
                output_dir / "weight_calibration_by_dataset.csv"
            ),
            "figures": [str(path) for path in figure_paths],
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = render_html(
        output_dir,
        summary_frame,
        calibration_summary,
        dropped_ids,
        provenance,
    )
    report_path = output_dir / "synthetic_input_data_analysis.html"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
