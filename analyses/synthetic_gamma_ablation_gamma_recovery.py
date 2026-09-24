"""Evaluate recovery of the programmed dataset-specific gamma bias profiles.

This is the dataset-bias counterpart of ``13_gamma_ablation_recovery.py``.  That
report scores the shared ``L_bio`` profile against the deterministic latent
kinetics ``K``; this one scores the learned per-dataset ``gamma`` against the
bias profiles that were actually programmed into the synthetic compendium
(``Datasets/Synthetic_data/bias_profile``), where ``added_bias`` is stored as
multiplier minus one so the physical bias is ``b = 1 + added_bias``.

Raw ``b`` is not directly comparable with ``gamma`` because the model leaves a
common positional profile and a dataset-constant scale unidentifiable.  Both
the programmed and the learned log profiles are therefore put through the same
joint two-way gauge (``joint_log_gamma_gauge``) over one identical position
domain, the CDS-observable interior ``5 <= i < L - 5``.  The figure and the
summary tables are produced in exactly the format used by the shared-profile
recovery report so the two can be read side by side.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pyarrow_compute
import pyarrow.dataset as pyarrow_dataset
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    # Support both ``python results/20_...py`` (which puts ``results`` on the
    # path) and ``python -m results.20_...`` (which puts the repository root
    # on it), matching the other synthetic reports.
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analyses.gamma_ablation.common import (
    DEFAULT_CONFIG,
    DEFAULT_DATASET_ENCODING,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RESULTS_ROOT,
    bootstrap_summary,
    discover_runs,
    filter_runs,
    fisher_summary,
    inventory_dataframe,
    load_config,
    select_latest_runs,
)
from analyses.analyze_synthetic_gamma_recovery import _base_bias_name
from analyses.analyze_synthetic_recovery import (
    BOUNDARY_TRIM_CODONS,
    _encoding_from_config,
    _validation_manifest,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None


METRICS = ("pearson", "shape_rmse", "mae")
DEFAULT_LATENT_TRUTH = (
    REPOSITORY_ROOT
    / "Datasets"
    / "Synthetic_data"
    / "artificial_ground_truth_kinetics_target_mean_one.parquet"
)
DEFAULT_BIAS_ROOT = REPOSITORY_ROOT / "Datasets" / "Synthetic_data" / "bias_profile"

# Compact, colour-blind-safe defaults suitable for a two-column ICLR/NeurIPS
# paper figure: restrained grid, no top/right box, embedded editable text in
# vector outputs, and enough marker contrast to remain readable in grayscale.
ICLR_NEURIPS_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 8.5,
    "legend.frameon": False,
    "lines.linewidth": 2.1,
    "lines.markersize": 5.5,
    "grid.color": "#D0D0D0",
    "grid.linewidth": 0.55,
    "grid.alpha": 0.55,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 300,
}

READ_DEPTH_ORDER = ("0p25_per_codon", "2_per_codon", "20_per_codon")
READ_DEPTH_LABELS = {
    "0p25_per_codon": "0.25 reads/codon",
    "2_per_codon": "2 reads/codon",
    "20_per_codon": "20 reads/codon",
}
# Okabe--Ito colours: reliable for common forms of colour-vision deficiency.
READ_DEPTH_STYLES = {
    "0p25_per_codon": {"color": "#0072B2", "marker": "o"},
    "2_per_codon": {"color": "#E69F00", "marker": "s"},
    "20_per_codon": {"color": "#009E73", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "gamma_recovery"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-encoding", type=Path, default=DEFAULT_DATASET_ENCODING)
    parser.add_argument(
        "--bias-root",
        type=Path,
        default=DEFAULT_BIAS_ROOT,
        help="Directory holding *_compendium_added_bias_only.parquet bias profiles.",
    )
    parser.add_argument(
        "--latent-truth",
        type=Path,
        default=DEFAULT_LATENT_TRUTH,
        help=(
            "Deterministic synthetic K profile parquet; used only for the "
            "canonical per-transcript P-site length."
        ),
    )
    parser.add_argument(
        "--boundary-trim-codons",
        type=int,
        default=BOUNDARY_TRIM_CODONS,
        help=(
            "Exclude this many codons from each CDS end in every profile metric "
            "(default: 5)."
        ),
    )
    parser.add_argument("--strategy", action="append", default=None)
    parser.add_argument("--feature-preset", action="append", default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--n-datasets", action="append", type=int, default=None)
    parser.add_argument(
        "--exclude-n-datasets",
        action="append",
        type=int,
        default=None,
        help="Exclude these training dataset counts (repeatable), e.g. --exclude-n-datasets 80.",
    )
    parser.add_argument("--quality-power", action="append", type=float, default=None)
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help=(
            "Prediction splits to score (default: main_val). Gamma truth is only "
            "defined for the held-out transcripts in the run split manifest."
        ),
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Worker processes used to score runs in parallel (default: one per "
            "run up to 4 and the CPU count; 1 disables the process pool)."
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Regenerate the compact gamma-recovery figure from the existing "
            "gamma_recovery_by_condition.csv without reopening prediction parquets."
        ),
    )
    return parser.parse_args()


def load_latent_lengths(path: Path) -> dict[str, int]:
    """Return the canonical 0-based P-site profile length per transcript.

    The synthetic bias annotations omit the terminal stop, so this length is
    also the axis on which learned gamma is truncated and compared.
    """
    table = pq.read_table(path, columns=["transcript_id", "rib_profile"])
    identifiers = table.column("transcript_id").to_pylist()
    profile_lengths = _list_column(table, "rib_profile")[2]
    if int(profile_lengths.min(initial=2)) < 2:
        raise ValueError(f"A latent profile in {path} has fewer than two positions.")
    lengths: dict[str, int] = {}
    for identifier, length in zip(identifiers, profile_lengths.tolist()):
        identifier = str(identifier)
        if identifier in lengths:
            raise ValueError(f"Duplicate latent transcript ID: {identifier}")
        lengths[identifier] = int(length)
    if not lengths:
        raise ValueError(f"No latent profiles found in {path}")
    return lengths


def _list_column(table: pa.Table, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (flat values, row start offsets, row lengths) for a list column.

    Reading whole columns as flat NumPy buffers instead of per-row Python lists
    is what makes this report fast; every downstream operation then works on
    contiguous segments of one array.
    """
    array = table.column(name).combine_chunks()
    if isinstance(array, pa.ChunkedArray):
        array = array.chunk(0) if array.num_chunks == 1 else pa.concat_arrays(array.chunks)
    lengths = array.value_lengths().to_numpy(zero_copy_only=False).astype(np.int64)
    values = np.asarray(array.flatten().to_numpy(zero_copy_only=False), dtype=np.float64)
    starts = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=starts[1:])
    return values, starts[:-1], lengths


def _interior_gather(
    row_starts: np.ndarray, counts: np.ndarray, offsets: np.ndarray, trim: int
) -> np.ndarray:
    """Flat indices of every interior codon, transcript-major.

    ``offsets`` are the compact per-transcript output offsets and ``row_starts``
    the source offsets, so this is the vectorised form of concatenating
    ``source[start + trim : start + length - trim]`` over all transcripts.
    """
    total = int(offsets[-1])
    positions = np.arange(total, dtype=np.int64)
    positions -= np.repeat(offsets[:-1], counts)
    positions += np.repeat(row_starts + trim, counts)
    return positions


def _compact_interior(
    values: np.ndarray,
    row_starts: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    trim: int,
) -> np.ndarray:
    return values[_interior_gather(row_starts, counts, offsets, trim)]


def load_bias_interior(
    *,
    bias_root: Path,
    dataset_name: str,
    transcript_ids: list[str],
    expected_lengths: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    trim: int,
) -> np.ndarray:
    """Load programmed ``added_bias`` for one dataset, interior codons only."""
    base_name = _base_bias_name(dataset_name)
    if base_name == "artificial_ground_truth":
        return np.zeros(int(offsets[-1]), dtype=np.float64)
    path = bias_root / f"{base_name}_compendium_added_bias_only.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Bias profile not found for {dataset_name}: {path}")
    # Only the ``*_mean`` sample is the programmed profile; pushing that filter
    # into the reader avoids materialising the replicate rows at all.
    table = pyarrow_dataset.dataset(path).to_table(
        columns=["sample", "transcript_id", "added_bias"],
        filter=pyarrow_compute.ends_with(pyarrow_dataset.field("sample"), pattern="_mean"),
    )
    values, starts, lengths = _list_column(table, "added_bias")
    index: dict[str, int] = {}
    for position, identifier in enumerate(table.column("transcript_id").to_pylist()):
        identifier = str(identifier)
        if identifier in index:
            raise ValueError(f"Duplicate mean bias profile: {dataset_name}/{identifier}")
        index[identifier] = position
    missing = [identifier for identifier in transcript_ids if identifier not in index]
    if missing:
        raise ValueError(
            f"Bias file {path} is missing {len(missing)} validation transcripts; "
            f"first={sorted(missing)[0]}."
        )
    rows = np.fromiter(
        (index[identifier] for identifier in transcript_ids),
        dtype=np.int64,
        count=len(transcript_ids),
    )
    if not np.array_equal(lengths[rows], expected_lengths):
        mismatch = int(np.flatnonzero(lengths[rows] != expected_lengths)[0])
        raise ValueError(
            f"Bias length mismatch for {dataset_name}/{transcript_ids[mismatch]}: "
            f"bias={int(lengths[rows][mismatch])}, "
            f"expected={int(expected_lengths[mismatch])}."
        )
    interior = _compact_interior(values, starts[rows], counts, offsets, trim)
    if not np.isfinite(interior).all() or bool((interior < 0.0).any()):
        raise ValueError(f"Invalid added_bias values in {path}.")
    return interior


def load_learned_gamma_interior(
    *,
    prediction_path: Path,
    dataset_names: list[str],
    dataset_id_to_name: dict[int, str],
    transcript_ids: list[str],
    expected_lengths: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    trim: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Load learned ``log_gamma`` and its reference weights, interior only.

    Returns per-dataset compact interior log-gamma, per-dataset per-transcript
    gauge reference weights, and the largest ``|exp(log_gamma) - gamma|`` seen,
    which is the writer self-consistency check kept from the original report.
    """
    required = ("transcript_id", "dataset_id", "log_gamma", "gamma", "gamma_centering_reliability")
    parquet_file = pq.ParquetFile(prediction_path)
    missing = sorted(set(required) - set(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"{prediction_path} is missing gamma columns: {missing}")
    table = pq.read_table(prediction_path, columns=list(required))
    log_values, log_starts, log_lengths = _list_column(table, "log_gamma")
    gamma_values, _, _ = _list_column(table, "gamma")
    if log_values.size != gamma_values.size:
        raise ValueError(f"{prediction_path}: gamma and log_gamma differ in size.")
    exp_difference = float(np.max(np.abs(np.exp(log_values) - gamma_values))) if log_values.size else 0.0
    del gamma_values
    reliability, reliability_starts, reliability_lengths = _list_column(
        table, "gamma_centering_reliability"
    )
    identifiers = table.column("transcript_id").to_pylist()
    dataset_ids = table.column("dataset_id").to_numpy()
    del table

    row_of: dict[tuple[str, str], int] = {}
    for position, (identifier, dataset_id) in enumerate(zip(identifiers, dataset_ids.tolist())):
        name = dataset_id_to_name.get(int(dataset_id))
        if name is None:
            raise KeyError(f"No dataset name for prediction dataset ID {dataset_id}.")
        key = (str(identifier), name)
        if key in row_of:
            raise ValueError(f"Duplicate prediction row: {name}/{identifier}")
        row_of[key] = position

    interiors: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    for dataset in dataset_names:
        try:
            rows = np.fromiter(
                (row_of[(identifier, dataset)] for identifier in transcript_ids),
                dtype=np.int64,
                count=len(transcript_ids),
            )
        except KeyError as error:
            raise KeyError(
                f"{prediction_path} has no row for dataset {dataset} and transcript {error}."
            ) from error
        if not np.array_equal(np.minimum(log_lengths[rows], expected_lengths), expected_lengths):
            mismatch = int(np.flatnonzero(log_lengths[rows] < expected_lengths)[0])
            raise ValueError(
                f"Short gamma profile: {dataset}/{transcript_ids[mismatch]}"
            )
        interior = _compact_interior(log_values, log_starts[rows], counts, offsets, trim)
        if not np.isfinite(interior).all():
            raise ValueError(f"Non-finite learned gamma for dataset {dataset}.")
        interiors[dataset] = interior
        dataset_weights = np.empty(len(transcript_ids), dtype=np.float64)
        for position, row in enumerate(rows.tolist()):
            segment = reliability[
                reliability_starts[row] : reliability_starts[row] + reliability_lengths[row]
            ]
            positive = segment[segment > 0.0]
            if positive.size == 0 or not np.isfinite(positive).all():
                raise ValueError(
                    "Missing positive gamma reference weight: "
                    f"{dataset}/{transcript_ids[position]}"
                )
            dataset_weights[position] = np.median(positive)
        weights[dataset] = dataset_weights
    return interiors, weights, exp_difference


def _gauge_block(values: np.ndarray, pi: np.ndarray, pi_expanded: np.ndarray, segment_starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Vectorised ``joint_log_gamma_gauge`` over many transcripts at once.

    ``values`` is ``[datasets, all interior positions]`` with one contiguous
    segment per transcript, so both gauge terms are segment-wise reductions.
    """
    position_center = np.einsum("dn,dn->n", pi_expanded, values)
    dataset_means = np.add.reduceat(values, segment_starts, axis=1) / counts
    weighted_dataset_mean = np.einsum("dt,dt->t", pi, dataset_means)
    gauged = values - position_center
    gauged -= np.repeat(dataset_means, counts, axis=1)
    gauged += np.repeat(weighted_dataset_mean, counts)
    return gauged


def _segment_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    support: np.ndarray,
    segment_starts: np.ndarray,
    counts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-transcript, per-dataset gamma metrics as ``[datasets, transcripts]``.

    These are exactly the quantities ``gamma_vector_metrics`` returns, computed
    segment-wise with ``np.add.reduceat`` instead of one Python call per pair.
    """
    def reduce(values: np.ndarray) -> np.ndarray:
        return np.add.reduceat(values, segment_starts, axis=1)

    predicted_centered = predicted - np.repeat(reduce(predicted) / counts, counts, axis=1)
    truth_centered = truth - np.repeat(reduce(truth) / counts, counts, axis=1)
    covariance = reduce(predicted_centered * truth_centered)
    predicted_variance = reduce(predicted_centered**2)
    truth_variance = reduce(truth_centered**2)
    denominator = np.sqrt(predicted_variance * truth_variance)
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = np.where(denominator > 0.0, covariance / denominator, np.nan)
    error = predicted - truth
    squared = reduce(error**2)
    absolute = reduce(np.abs(error))
    truth_energy = reduce(truth**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(truth_energy > 0.0, reduce(truth * predicted) / truth_energy, np.nan)
    relative = np.abs(np.expm1(error))
    return {
        "pearson": pearson,
        "shape_rmse": np.sqrt(squared / counts),
        "mae": absolute / counts,
        "calibration_slope": slope,
        "mean_absolute_relative_error": reduce(relative) / counts,
        "fraction_within_5pct": reduce((relative <= 0.05).astype(np.float64)) / counts,
        "fraction_within_10pct": reduce((relative <= 0.10).astype(np.float64)) / counts,
        "programmed_bias_fraction": reduce(support.astype(np.float64)) / counts,
    }


METRIC_COLUMNS = (
    "pearson",
    "shape_rmse",
    "mae",
    "calibration_slope",
    "mean_absolute_relative_error",
    "fraction_within_5pct",
    "fraction_within_10pct",
    "programmed_bias_fraction",
)


def process_run(
    run_metadata: dict[str, Any],
    *,
    run_path: Path,
    config: dict[str, Any],
    prediction_path: Path,
    split: str,
    checkpoint_variant: str,
    latent_lengths: dict[str, int],
    bias_root: Path,
    bias_cache: dict[tuple[str, str, int], np.ndarray],
    boundary_trim_codons: int,
    block_elements: int = 4_000_000,
) -> pd.DataFrame:
    """Score learned gamma against programmed bias for one prediction file.

    One row is emitted per held-out transcript and dataset.  Truth and
    prediction are gauged jointly over the same interior coordinates, so the
    only quantities compared are the identifiable ones.
    """
    datasets = config.get("experiment", {}).get("dataset", [])
    if not isinstance(datasets, (list, tuple)):
        datasets = [] if datasets in (None, "") else [datasets]
    datasets = [str(value) for value in datasets]
    if len(datasets) < 2:
        # The cross-dataset gauge is undefined for a single dataset, so a
        # single-dataset run has no identifiable gamma to score.
        return pd.DataFrame()

    validation_ids_list, validation_hash = _validation_manifest(run_path)
    validation_ids = set(validation_ids_list)
    missing_latent = sorted(validation_ids - latent_lengths.keys())
    if missing_latent:
        raise KeyError(
            f"{run_metadata['run_id']}: {len(missing_latent)} validation transcripts "
            f"are absent from the latent truth table; first={missing_latent[0]}."
        )
    trim = int(boundary_trim_codons)
    ordered_ids = sorted(validation_ids)
    full_lengths = np.fromiter(
        (latent_lengths[identifier] for identifier in ordered_ids),
        dtype=np.int64,
        count=len(ordered_ids),
    )
    # Transcripts with no fair CDS-only coordinate contribute to neither metric.
    keep = full_lengths > 2 * trim
    ordered_ids = [identifier for identifier, flag in zip(ordered_ids, keep.tolist()) if flag]
    full_lengths = full_lengths[keep]
    if not ordered_ids:
        return pd.DataFrame()
    counts = full_lengths - 2 * trim
    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    id_to_name = _encoding_from_config(config, REPOSITORY_ROOT)
    # Older completed synthetic runs stored a temporary dataset-encoding path
    # that no longer exists; for those the experiment dataset list is written in
    # the same order as the compact prediction dataset IDs.
    if not set(datasets).issubset(set(id_to_name.values())):
        id_to_name = {index: dataset for index, dataset in enumerate(datasets)}

    predicted_by_dataset, weight_by_dataset, _ = load_learned_gamma_interior(
        prediction_path=prediction_path,
        dataset_names=datasets,
        dataset_id_to_name=id_to_name,
        transcript_ids=ordered_ids,
        expected_lengths=full_lengths,
        counts=counts,
        offsets=offsets,
        trim=trim,
    )
    truth_by_dataset: dict[str, np.ndarray] = {}
    for dataset in datasets:
        cache_key = (_base_bias_name(dataset), validation_hash, trim)
        if cache_key not in bias_cache:
            bias_cache[cache_key] = load_bias_interior(
                bias_root=bias_root,
                dataset_name=dataset,
                transcript_ids=ordered_ids,
                expected_lengths=full_lengths,
                counts=counts,
                offsets=offsets,
                trim=trim,
            )
        truth_by_dataset[dataset] = bias_cache[cache_key]

    n_datasets = len(datasets)
    weights = np.stack([weight_by_dataset[dataset] for dataset in datasets])
    if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
        raise ValueError("Reference weights must be finite and strictly positive.")
    normalized_weights = weights / weights.sum(axis=0, keepdims=True)

    # Blocks keep peak memory bounded while still amortising every NumPy call
    # over thousands of transcripts.
    block_transcripts = max(1, int(block_elements // max(1, n_datasets * int(counts.max()))))
    metrics: dict[str, list[np.ndarray]] = {name: [] for name in METRIC_COLUMNS}
    for start in range(0, len(ordered_ids), block_transcripts):
        stop = min(start + block_transcripts, len(ordered_ids))
        block_counts = counts[start:stop]
        first, last = int(offsets[start]), int(offsets[stop])
        segment_starts = offsets[start:stop] - first
        raw_bias = np.stack([truth_by_dataset[dataset][first:last] for dataset in datasets])
        support = raw_bias > 0.0
        truth = np.log1p(raw_bias)
        predicted = np.stack(
            [predicted_by_dataset[dataset][first:last] for dataset in datasets]
        )
        pi = normalized_weights[:, start:stop]
        pi_expanded = np.repeat(pi, block_counts, axis=1)
        gauged_truth = _gauge_block(truth, pi, pi_expanded, segment_starts, block_counts)
        gauged_predicted = _gauge_block(
            predicted, pi, pi_expanded, segment_starts, block_counts
        )
        block_metrics = _segment_metrics(
            gauged_predicted, gauged_truth, support, segment_starts, block_counts
        )
        for name in METRIC_COLUMNS:
            metrics[name].append(block_metrics[name])

    frame = pd.DataFrame(
        {
            # Transcript-major ordering: every dataset of one transcript is
            # adjacent, matching the per-transcript reference implementation.
            "transcript_id": np.repeat(np.asarray(ordered_ids, dtype=object), n_datasets),
            "dataset": np.tile(np.asarray(datasets, dtype=object), len(ordered_ids)),
            "n_positions": np.repeat(counts, n_datasets),
            **{
                name: np.concatenate(metrics[name], axis=1).T.reshape(-1)
                for name in METRIC_COLUMNS
            },
        }
    )
    for key, value in run_metadata.items():
        frame[key] = value
    frame["model_id"] = run_metadata["run_id"]
    frame["split"] = split
    frame["checkpoint_variant"] = checkpoint_variant
    frame["component"] = "gamma"
    frame["reference_column"] = "added_bias"
    frame["reference_kind"] = "programmed_dataset_bias"
    frame["scope"] = "gauged_log_gamma_interior"
    del predicted_by_dataset, weight_by_dataset
    gc.collect()
    return frame


_WORKER_STATE: dict[str, Any] = {}


def _worker_initializer(latent_truth_path: str) -> None:
    _WORKER_STATE["latent_lengths"] = load_latent_lengths(Path(latent_truth_path))
    _WORKER_STATE["bias_cache"] = {}


def _worker_process_run(payload: dict[str, Any]) -> pd.DataFrame:
    """Entry point for ``ProcessPoolExecutor``; runs are fully independent."""
    return process_run(
        payload["run_metadata"],
        run_path=Path(payload["run_path"]),
        config=payload["config"],
        prediction_path=Path(payload["prediction_path"]),
        split=payload["split"],
        checkpoint_variant=payload["checkpoint_variant"],
        latent_lengths=_WORKER_STATE["latent_lengths"],
        bias_root=Path(payload["bias_root"]),
        bias_cache=_WORKER_STATE["bias_cache"],
        boundary_trim_codons=payload["boundary_trim_codons"],
    )


RUN_KEYS = [
    "run_id",
    "strategy",
    "training_scope",
    "depth",
    "mass_condition",
    "n_datasets",
    "quality_rank_power",
    "gamma_weighting",
    "feature_preset",
    "seed",
    "model_id",
    "split",
    "checkpoint_variant",
    "component",
    "reference_column",
    "reference_kind",
    "scope",
]


def _summarize(group: pd.DataFrame, *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "n_transcripts": int(group["transcript_id"].nunique()),
        "n_profiles": int(len(group)),
        "mean_programmed_bias_fraction": float(group["programmed_bias_fraction"].mean()),
    }
    for metric in METRICS:
        summary = bootstrap_summary(group[metric], n_bootstrap=n_bootstrap, seed=seed)
        for name, value in summary.items():
            record[f"{metric}_{name}"] = value
    record.update(
        {
            f"pearson_{name}": value
            for name, value in fisher_summary(
                group["pearson"], group["n_positions"]
            ).items()
        }
    )
    for name in ("calibration_slope", "mean_absolute_relative_error", "fraction_within_5pct", "fraction_within_10pct"):
        record[f"{name}_mean"] = float(group[name].mean())
    return record


def dataset_summary_frame(
    rows: pd.DataFrame, *, n_bootstrap: int, seed: int
) -> pd.DataFrame:
    """One row per run and programmed bias case."""
    if rows.empty:
        return pd.DataFrame()
    output: list[dict[str, Any]] = []
    keys = RUN_KEYS + ["dataset"]
    for key, group in rows.groupby(keys, dropna=False, sort=False):
        output.append(
            {**dict(zip(keys, key)), **_summarize(group, n_bootstrap=n_bootstrap, seed=seed)}
        )
    return pd.DataFrame(output)


def condition_summary_frame(
    rows: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """One plotted row per run: all transcript-by-dataset profiles pooled."""
    if rows.empty:
        return pd.DataFrame()
    output: list[dict[str, Any]] = []
    for key, group in rows.groupby(RUN_KEYS, dropna=False, sort=False):
        base = dict(zip(RUN_KEYS, key))
        summary = _summarize(group, n_bootstrap=n_bootstrap, seed=seed)
        cases = dataset_summary[dataset_summary["run_id"] == base["run_id"]]
        output.append(
            {
                **base,
                "n_bias_cases": int(group["dataset"].nunique()),
                **summary,
                "log_gamma_pearson_fisher": summary["pearson_fisher"],
                "log_gamma_rmse_mean": summary["shape_rmse_mean"],
                "log_gamma_pearson_worst_case": (
                    float(cases["pearson_fisher"].min()) if not cases.empty else np.nan
                ),
                "log_gamma_rmse_worst_case": (
                    float(cases["shape_rmse_mean"].max()) if not cases.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(output)


def _depth_sort_key(depth: Any) -> tuple[int, str]:
    value = str(depth)
    try:
        return (READ_DEPTH_ORDER.index(value), value)
    except ValueError:
        return (len(READ_DEPTH_ORDER), value)


def _safe_plot_token(value: Any) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "p")


def plot_gamma_recovery_by_depth(summary: pd.DataFrame, output: Path) -> None:
    """Plot gauge-fixed gamma recovery against the programmed bias profiles.

    ``log_gamma_pearson_fisher`` is the profile-length Fisher-weighted PCC over
    every held-out transcript-by-dataset profile; ``log_gamma_rmse_mean`` is the
    mean of the same profiles' log-space RMSEs.  The layout deliberately
    mirrors the shared-``L_bio``-versus-``K`` figure.
    """
    if plt is None or summary.empty:
        return
    required = {
        "depth",
        "n_datasets",
        "log_gamma_pearson_fisher",
        "log_gamma_rmse_mean",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise KeyError(f"Gamma recovery summary is missing plot columns: {missing}")

    data = summary[summary["split"] == "main_val"].copy()
    if data.empty:
        return
    if data.duplicated(["depth", "n_datasets"]).any():
        duplicate = data.loc[
            data.duplicated(["depth", "n_datasets"], keep=False),
            ["depth", "n_datasets", "run_id"],
        ]
        raise ValueError(
            "Expected one gamma-recovery row per read-depth/dataset-count; "
            f"found duplicates: {duplicate.to_dict('records')[:5]}"
        )

    depths = sorted(data["depth"].dropna().unique(), key=_depth_sort_key)
    dataset_counts = sorted(int(value) for value in data["n_datasets"].dropna().unique())
    if not depths or not dataset_counts:
        return

    with matplotlib.rc_context(ICLR_NEURIPS_RC):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8.4, 3.35),
            sharex=True,
            constrained_layout=True,
        )
        panel_specs = (
            (
                "log_gamma_pearson_fisher",
                "Fisher-weighted transcript PCC",
                "a  Bias-profile shape recovery",
            ),
            (
                "log_gamma_rmse_mean",
                r"Mean transcript RMSE of $\log\gamma$ vs $\log b$",
                "b  Bias-profile error",
            ),
        )
        handles = []
        for depth in depths:
            line = data[data["depth"] == depth].sort_values("n_datasets")
            style = READ_DEPTH_STYLES.get(
                str(depth),
                {"color": "#6C6C6C", "marker": "D"},
            )
            label = READ_DEPTH_LABELS.get(str(depth), str(depth).replace("_", " "))
            for axis_index, (metric, ylabel, title) in enumerate(panel_specs):
                line_handle = axes[axis_index].plot(
                    line["n_datasets"],
                    line[metric],
                    label=label,
                    color=style["color"],
                    marker=style["marker"],
                    markeredgecolor="white",
                    markeredgewidth=0.65,
                    zorder=3,
                )[0]
                if axis_index == 0:
                    handles.append(line_handle)
                axes[axis_index].set_ylabel(ylabel)
                axes[axis_index].set_title(title, loc="left", pad=8)
                axes[axis_index].grid(axis="y")
                axes[axis_index].set_axisbelow(True)
                axes[axis_index].set_xlabel("Number of training datasets")
                axes[axis_index].set_xticks(dataset_counts)
                axes[axis_index].set_xlim(min(dataset_counts) - 0.25, max(dataset_counts) + 0.25)

        # Gamma recovery is near its ceiling, so the shape panel keeps the
        # legend clear of the curves while the error panel stays uncluttered.
        axes[0].legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
            title="Read depth",
            loc="lower right",
            handlelength=2.2,
        )
        fig.text(
            0.5,
            -0.015,
            r"Learned $\gamma$ evaluated against the programmed dataset bias "
            r"$b = 1 + \mathrm{added\ bias}$ under the joint identifiability "
            r"gauge; interior codons only ($5 \leq i < L - 5$).",
            ha="center",
            va="top",
            fontsize=8.3,
            color="#404040",
        )
        for suffix in (".png", ".pdf", ".svg"):
            fig.savefig(output.with_suffix(suffix), bbox_inches="tight")
        plt.close(fig)


def write_gamma_recovery_plots(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write one two-panel depth-comparison figure per non-depth condition."""
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = summary[summary["split"] == "main_val"].copy()
    if primary.empty:
        return []
    controls = (
        "strategy",
        "training_scope",
        "mass_condition",
        "quality_rank_power",
        "gamma_weighting",
        "feature_preset",
        "seed",
    )
    missing_controls = [column for column in controls if column not in primary.columns]
    if missing_controls:
        raise KeyError(f"Gamma recovery summary is missing control columns: {missing_controls}")

    written: list[Path] = []
    for key, group in primary.groupby(list(controls), dropna=False, sort=True):
        values = dict(zip(controls, key))
        filename = (
            "gamma_recovery_main_val_by_read_depth"
            f"__{_safe_plot_token(values['strategy'])}"
            f"__{_safe_plot_token(values['mass_condition'])}"
            f"__rankp{_safe_plot_token(values['quality_rank_power'])}.png"
        )
        output = output_dir / filename
        plot_gamma_recovery_by_depth(group, output)
        written.append(output)
    return written


def clear_gamma_recovery_plot_outputs(output_dir: Path) -> None:
    """Remove only figures generated by this script, never summary tables."""
    plot_dir = output_dir / "plots_by_depth"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in plot_dir.glob("gamma_recovery_main_val_by_read_depth__*"):
        if old_plot.is_file():
            old_plot.unlink()


def _default_workers(n_jobs: int) -> int:
    # Deep-panel prediction parquets are ~350 MB on disk, so the pool is capped
    # conservatively; raise it with --workers when memory allows.
    return max(1, min(4, n_jobs, os.cpu_count() or 1))


def main() -> None:
    args = parse_args()
    if args.boundary_trim_codons < 0:
        raise ValueError("--boundary-trim-codons must be non-negative.")
    if args.plot_only:
        summary_path = args.output_dir / "gamma_recovery_by_condition.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"--plot-only requires an existing gamma summary: {summary_path}"
            )
        conditions = pd.read_csv(summary_path)
        clear_gamma_recovery_plot_outputs(args.output_dir)
        written = write_gamma_recovery_plots(
            conditions, args.output_dir / "plots_by_depth"
        )
        if not written:
            raise RuntimeError(
                "No main-validation gamma-recovery rows were available for plotting."
            )
        print("Regenerated compact gamma-recovery plot(s):")
        for path in written:
            print(f"  {path}")
        return

    config = load_config(args.config)
    statistics = config.get("statistics", {})
    n_bootstrap = args.bootstrap if args.bootstrap is not None else int(statistics.get("bootstrap_replicates", 500))
    bootstrap_seed = args.bootstrap_seed if args.bootstrap_seed is not None else int(statistics.get("bootstrap_seed", 42))
    splits = set(args.split or ["main_val"])

    bias_root = args.bias_root.expanduser()
    if not bias_root.is_absolute():
        bias_root = REPOSITORY_ROOT / bias_root
    if not bias_root.is_dir():
        raise FileNotFoundError(f"Bias profile directory does not exist: {bias_root}")
    latent_truth_path = args.latent_truth.expanduser()
    if not latent_truth_path.is_absolute():
        latent_truth_path = REPOSITORY_ROOT / latent_truth_path
    latent_lengths = load_latent_lengths(latent_truth_path)

    all_runs = discover_runs(args.results_root)
    latest = select_latest_runs(all_runs)
    selected = filter_runs(
        latest,
        strategies=set(args.strategy) if args.strategy else None,
        feature_presets=set(args.feature_preset) if args.feature_preset else None,
        seeds=set(args.seed) if args.seed else None,
        dataset_counts=set(args.n_datasets) if args.n_datasets else None,
        quality_powers=set(args.quality_power) if args.quality_power else None,
        max_runs=args.max_runs,
    )
    excluded_dataset_counts = set(args.exclude_n_datasets or [])
    selected = [run for run in selected if run.n_datasets not in excluded_dataset_counts]
    if not selected:
        raise RuntimeError("No readable prediction runs matched the requested filters.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis_settings.json").write_text(
        json.dumps(
            {
                "results_root": str(args.results_root.resolve()),
                "config": str(args.config.resolve()),
                "bias_root": str(bias_root.resolve()),
                "latent_truth": str(latent_truth_path.resolve()),
                "primary_metric": (
                    "learned gamma versus programmed bias b = 1 + added_bias, both "
                    "jointly gauged in log space over 5 <= i < L-5; one score per "
                    "held-out transcript and dataset"
                ),
                "gauge": (
                    "position centre under the checkpointed gamma reference weights "
                    "plus dataset-constant mean removal (joint_log_gamma_gauge)"
                ),
                "boundary_trim_codons": int(args.boundary_trim_codons),
                "strategies": args.strategy,
                "feature_presets": args.feature_preset,
                "seeds": args.seed,
                "dataset_counts": args.n_datasets,
                "excluded_dataset_counts": sorted(excluded_dataset_counts),
                "quality_powers": args.quality_power,
                "splits": sorted(splits),
                "plot_partition_keys": [
                    "strategy",
                    "training_scope",
                    "mass_condition",
                    "quality_rank_power",
                    "gamma_weighting",
                    "feature_preset",
                    "seed",
                ],
                "read_depths_combined_in_primary_plots": True,
                "primary_figure": (
                    "One row of two panels: Fisher-weighted transcript PCC and mean "
                    "per-profile log-gamma RMSE; colour/marker encodes read depth."
                ),
                "primary_figure_style": "compact ICLR/NeurIPS-inspired publication style",
                "bootstrap_replicates": n_bootstrap,
                "bootstrap_seed": bootstrap_seed,
                "worker_processes": args.workers,
                "selected_run_ids": [run.run_id for run in selected],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    inventory_dataframe(all_runs, selected).to_csv(
        args.output_dir / "run_inventory.csv", index=False
    )

    payloads: list[dict[str, Any]] = []
    for run in selected:
        for prediction in (file for file in run.usable_files if file.split in splits):
            payloads.append(
                {
                    "run_metadata": run.metadata(),
                    "run_path": str(run.path),
                    "config": run.config,
                    "prediction_path": str(prediction.path),
                    "split": prediction.split,
                    "checkpoint_variant": prediction.checkpoint_variant,
                    "bias_root": str(bias_root),
                    "boundary_trim_codons": int(args.boundary_trim_codons),
                }
            )
    if not payloads:
        raise RuntimeError(f"No prediction files matched splits {sorted(splits)}.")

    workers = args.workers if args.workers is not None else _default_workers(len(payloads))
    frames: list[pd.DataFrame] = []
    if workers > 1:
        # Runs share nothing but the read-only bias parquets, so a process pool
        # scales almost linearly; each worker keeps its own bias cache.
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_initializer,
            initargs=(str(latent_truth_path),),
        ) as executor:
            for index, frame in enumerate(
                executor.map(_worker_process_run, payloads), start=1
            ):
                print(f"[{index}/{len(payloads)}] {payloads[index - 1]['run_metadata']['run_id']}")
                frames.append(frame)
    else:
        bias_cache: dict[tuple[str, str, int], np.ndarray] = {}
        for index, payload in enumerate(payloads, start=1):
            print(f"[{index}/{len(payloads)}] {payload['run_metadata']['run_id']}")
            frames.append(
                process_run(
                    payload["run_metadata"],
                    run_path=Path(payload["run_path"]),
                    config=payload["config"],
                    prediction_path=Path(payload["prediction_path"]),
                    split=payload["split"],
                    checkpoint_variant=payload["checkpoint_variant"],
                    latent_lengths=latent_lengths,
                    bias_root=bias_root,
                    bias_cache=bias_cache,
                    boundary_trim_codons=int(args.boundary_trim_codons),
                )
            )

    frames = [frame for frame in frames if not frame.empty]
    profiles = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if profiles.empty:
        raise RuntimeError("No transcript-by-dataset gamma comparisons were produced.")
    dataset_summary = dataset_summary_frame(
        profiles, n_bootstrap=n_bootstrap, seed=bootstrap_seed
    )
    conditions = condition_summary_frame(
        profiles, dataset_summary, n_bootstrap=n_bootstrap, seed=bootstrap_seed
    )
    dataset_summary.to_csv(args.output_dir / "gamma_recovery_by_dataset.csv", index=False)
    conditions.to_csv(args.output_dir / "gamma_recovery_by_condition.csv", index=False)
    profiles.to_csv(
        args.output_dir / "gamma_recovery_by_transcript_dataset.csv.gz",
        index=False,
        compression="gzip",
    )

    clear_gamma_recovery_plot_outputs(args.output_dir)
    written = write_gamma_recovery_plots(conditions, args.output_dir / "plots_by_depth")
    if written:
        print("Saved compact gamma-recovery plot(s):")
        for path in written:
            print(f"  {path}")
    print(f"Saved gamma recovery analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
