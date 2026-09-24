#!/usr/bin/env python3
"""Audit observation reconstruction from single to cumulative synthetic fits.

The analysis compares frozen minimum-validation-loss checkpoints.  For every
read depth, cumulative panel size, named bias condition and held-out
transcript, it computes the Pearson correlation between the exported fitted
mean ``mu`` and exported observed target on the model-valid positions.  It
then pairs each cumulative value with the independently trained N=1 model for
the same depth, bias condition and transcript.

This is an observation-reconstruction diagnostic.  It is not a test of K_t,
L_t or gamma recovery and it does not isolate a pure dataset-count effect,
because the cumulative panels add biases in a fixed order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utils.publication_plot_style import publication_rc
from analyses.analyze_synthetic_recovery import _encoding_from_config
from analyses.analyze_synthetic_single_dataset_mu_pcc import transcript_mu_pcc


DEPTHS = ("0p25_per_codon", "2_per_codon", "20_per_codon")
DEPTH_LABELS = {
    "0p25_per_codon": "0.25 reads/codon",
    "2_per_codon": "2 reads/codon",
    "20_per_codon": "20 reads/codon",
}
DEPTH_COLORS = {
    "0p25_per_codon": "#0072B2",
    "2_per_codon": "#E69F00",
    "20_per_codon": "#009E73",
}
BIAS_ORDER = (
    "artificial_bias_3prime_aa",
    "artificial_bias_3prime_cc",
    "artificial_bias_3prime_gg",
    "artificial_bias_3prime_uu",
    "artificial_bias_5prime_aa",
    "artificial_bias_5prime_cc",
    "artificial_bias_5prime_gg",
    "artificial_bias_5prime_uu",
    "artificial_bias_gc_fraction_gt_0p7",
    "artificial_bias_au_fraction_gt_0p7",
)
BIAS_LABELS = {
    "artificial_bias_3prime_aa": r"$3^\prime$-AA",
    "artificial_bias_3prime_cc": r"$3^\prime$-CC",
    "artificial_bias_3prime_gg": r"$3^\prime$-GG",
    "artificial_bias_3prime_uu": r"$3^\prime$-UU",
    "artificial_bias_5prime_aa": r"$5^\prime$-AA",
    "artificial_bias_5prime_cc": r"$5^\prime$-CC",
    "artificial_bias_5prime_gg": r"$5^\prime$-GG",
    "artificial_bias_5prime_uu": r"$5^\prime$-UU",
    "artificial_bias_gc_fraction_gt_0p7": "GC-rich",
    "artificial_bias_au_fraction_gt_0p7": "AU-rich",
}
PANEL_SIZES = tuple(range(2, 11))
CHECKPOINT_VARIANT = "best_val_loss"
EXPECTED_METRICS_PER_TRANSCRIPT = len(BIAS_ORDER) + sum(PANEL_SIZES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort_hash(transcript_ids: list[str] | set[str]) -> str:
    payload = "\n".join(sorted(transcript_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configure_style() -> dict[str, Any]:
    style = publication_rc()
    style.update(
        {
            "font.size": 10.5,
            "font.weight": "bold",
            "axes.labelsize": 11.5,
            "axes.labelweight": "bold",
            "axes.titlesize": 12.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if style.get("text.usetex"):
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}"
            r"\AtBeginDocument{\boldmath}"
        )
    return style


def load_single_metrics(
    metric_path: Path,
    provenance_path: Path,
) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, Any]]:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    comparison = provenance.get("single_model_comparison", {})
    if comparison.get("checkpoint_variant") != CHECKPOINT_VARIANT:
        raise ValueError(
            f"Single-model audit does not use {CHECKPOINT_VARIANT}: {provenance_path}"
        )
    if int(comparison.get("training_seed", -1)) != 42:
        raise ValueError("The single-model audit must use training seed 42.")

    columns = [
        "depth",
        "dataset",
        "transcript_id",
        "model_mu_pcc",
        "pcc_variance_valid",
        "n_valid_positions",
        "profile_length",
        "prediction_path",
    ]
    frame = pd.read_parquet(metric_path, columns=columns)
    frame = frame.loc[
        frame["depth"].isin(DEPTHS) & frame["dataset"].isin(BIAS_ORDER)
    ].copy()
    keys = ["depth", "dataset", "transcript_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate single-model metric rows were found.")
    conditions = set(zip(frame["depth"], frame["dataset"]))
    expected = {(depth, dataset) for depth in DEPTHS for dataset in BIAS_ORDER}
    if conditions != expected:
        raise ValueError(
            f"Incomplete single-model grid: missing={sorted(expected-conditions)}"
        )

    frame["mu_pcc"] = frame["model_mu_pcc"].where(
        frame["pcc_variance_valid"] & np.isfinite(frame["model_mu_pcc"])
    )
    frame["n_datasets"] = 1
    frame["model_type"] = "single"
    validation: dict[str, set[str]] = {}
    for depth in DEPTHS:
        depth_frame = frame.loc[frame["depth"] == depth]
        sets = [
            set(group["transcript_id"].astype(str))
            for _, group in depth_frame.groupby("dataset", sort=False)
        ]
        if not sets or any(ids != sets[0] for ids in sets[1:]):
            raise ValueError(f"Single-model validation IDs vary by bias at {depth}.")
        validation[depth] = sets[0]
    return frame, validation, provenance


def resolve_prediction(run_directory: Path) -> tuple[Path, str]:
    predictions = sorted(
        path
        for path in run_directory.rglob(
            f"predictions_main_val_{CHECKPOINT_VARIANT}_*.parquet"
        )
        if path.stat().st_size > 0
    )
    if len(predictions) != 1:
        raise FileNotFoundError(
            f"Expected one {CHECKPOINT_VARIANT} prediction below {run_directory}; "
            f"found {len(predictions)}."
        )
    manifests = sorted(run_directory.rglob("prediction_checkpoint_manifest.json"))
    if len(manifests) != 1:
        raise FileNotFoundError(
            f"Expected one checkpoint manifest below {run_directory}; found {len(manifests)}."
        )
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    entry = payload.get(CHECKPOINT_VARIANT)
    if not isinstance(entry, dict):
        raise KeyError(f"{manifests[0]} has no {CHECKPOINT_VARIANT} entry.")
    recorded = entry.get("output_path")
    if recorded and Path(str(recorded)).name != predictions[0].name:
        raise ValueError("Checkpoint manifest and prediction filename disagree.")
    return predictions[0], str(entry.get("checkpoint_path", ""))


def validate_run(
    run: dict[str, Any],
    single_validation: dict[str, set[str]],
) -> dict[str, Any]:
    depth = str(run["depth"])
    datasets = [str(value) for value in run["datasets"]]
    n_datasets = len(datasets)
    if depth not in DEPTHS or n_datasets not in PANEL_SIZES:
        raise ValueError(f"Unexpected cumulative condition: {depth}, N={n_datasets}")
    expected_datasets = list(BIAS_ORDER[:n_datasets])
    if datasets != expected_datasets:
        raise ValueError(
            f"Cumulative membership/order changed for {depth}, N={n_datasets}."
        )
    config_path = ROOT / str(run["config_path"])
    if sha256(config_path) != str(run["config_sha256"]):
        raise ValueError(f"Resolved configuration changed: {config_path}")
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
    reference = config["model"]["gamma_centering"]["reference"]
    if int(config["experiment"]["seed"]) != 42:
        raise ValueError(f"Training seed is not 42 in {config_path}")
    if list(config["experiment"]["dataset"]) != datasets:
        raise ValueError(f"Experiment datasets disagree with provenance in {config_path}")
    if config["model"]["gamma_centering"]["mode"] != "fixed_reference":
        raise ValueError(f"Run is not fixed-reference: {config_path}")
    if reference["weighting"] != "equal":
        raise ValueError(f"Run is not uniform-reference: {config_path}")

    split_path = ROOT / str(run["split_path"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validation_ids = set(map(str, split["validation_ids"]))
    train_ids = set(map(str, split["train_ids"]))
    if validation_ids & train_ids:
        raise ValueError(f"Training/validation overlap in {split_path}")
    if validation_ids != single_validation[depth]:
        raise ValueError(
            f"Single and cumulative validation IDs differ at {depth}, N={n_datasets}."
        )
    if cohort_hash(validation_ids) != str(run["validation_hash"]):
        raise ValueError(f"Validation hash mismatch for {split_path}")

    run_directory = ROOT / "results" / "riboai_synthetic_experiments" / str(run["run"])
    prediction_path, checkpoint_path = resolve_prediction(run_directory)
    id_to_name = _encoding_from_config(config, ROOT)
    return {
        "run": str(run["run"]),
        "depth": depth,
        "n_datasets": n_datasets,
        "datasets": datasets,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "split_path": str(split_path),
        "split_sha256": sha256(split_path),
        "validation_n": len(validation_ids),
        "validation_hash": cohort_hash(validation_ids),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "checkpoint_path": checkpoint_path,
        "dataset_id_to_name": id_to_name,
    }


def stream_cumulative_prediction(run: dict[str, Any]) -> pd.DataFrame:
    prediction_path = Path(run["prediction_path"])
    parquet = pq.ParquetFile(prediction_path)
    required = {"transcript_id", "dataset_id", "target", "mu", "mask", "length"}
    missing = required.difference(parquet.schema_arrow.names)
    if missing:
        raise KeyError(f"{prediction_path} lacks {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=16):
        columns = batch.to_pydict()
        for row_index in range(batch.num_rows):
            transcript_id = str(columns["transcript_id"][row_index])
            dataset_id = int(columns["dataset_id"][row_index])
            dataset = run["dataset_id_to_name"].get(dataset_id)
            if dataset not in run["datasets"]:
                raise ValueError(
                    f"Unexpected dataset_id={dataset_id} ({dataset}) in {prediction_path}"
                )
            key = (transcript_id, dataset)
            if key in seen:
                raise ValueError(f"Duplicate prediction row {key} in {prediction_path}")
            seen.add(key)
            pcc, valid, n_positions = transcript_mu_pcc(
                target_value=columns["target"][row_index],
                mu_value=columns["mu"][row_index],
                mask_value=columns["mask"][row_index],
                pcc_prediction_floor=0.0,
            )
            rows.append(
                {
                    "run": run["run"],
                    "depth": run["depth"],
                    "n_datasets": run["n_datasets"],
                    "dataset": dataset,
                    "transcript_id": transcript_id,
                    "mu_pcc": pcc if valid else np.nan,
                    "pcc_variance_valid": bool(valid),
                    "n_valid_positions": n_positions,
                    "profile_length": int(columns["length"][row_index]),
                    "prediction_path": str(prediction_path),
                    "model_type": "cumulative",
                }
            )
        del columns, batch
    expected_rows = int(run["validation_n"]) * int(run["n_datasets"])
    if len(rows) != expected_rows:
        raise ValueError(
            f"{prediction_path} has {len(rows)} rows; expected {expected_rows}."
        )
    return pd.DataFrame.from_records(rows)


def fixed_complete_cohorts(
    single: pd.DataFrame,
    cumulative: pd.DataFrame,
    validation: dict[str, set[str]],
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    validity = pd.concat(
        [
            single[["depth", "transcript_id", "mu_pcc"]],
            cumulative[["depth", "transcript_id", "mu_pcc"]],
        ],
        ignore_index=True,
    )
    validity["valid"] = np.isfinite(validity["mu_pcc"])
    grouped = validity.groupby(["depth", "transcript_id"], sort=False).agg(
        records=("valid", "size"), valid_records=("valid", "sum")
    )
    cohorts: dict[str, list[str]] = {}
    exclusion_rows: list[dict[str, Any]] = []
    for depth in DEPTHS:
        depth_counts = grouped.loc[depth]
        complete = depth_counts.loc[
            (depth_counts["records"] == EXPECTED_METRICS_PER_TRANSCRIPT)
            & (depth_counts["valid_records"] == EXPECTED_METRICS_PER_TRANSCRIPT)
        ]
        ids = sorted(set(complete.index.astype(str)) & validation[depth])
        cohorts[depth] = ids
        for transcript_id in sorted(validation[depth] - set(ids)):
            if transcript_id not in depth_counts.index:
                reason = "missing metric record"
                records = valid_records = 0
            else:
                row = depth_counts.loc[transcript_id]
                records = int(row["records"])
                valid_records = int(row["valid_records"])
                reason = (
                    "missing metric record"
                    if records != EXPECTED_METRICS_PER_TRANSCRIPT
                    else "undefined PCC from constant or nearly constant profile"
                )
            exclusion_rows.append(
                {
                    "depth": depth,
                    "transcript_id": transcript_id,
                    "reason": reason,
                    "records": records,
                    "valid_records": valid_records,
                    "expected_records": EXPECTED_METRICS_PER_TRANSCRIPT,
                }
            )
    return cohorts, pd.DataFrame(
        exclusion_rows,
        columns=(
            "depth",
            "transcript_id",
            "reason",
            "records",
            "valid_records",
            "expected_records",
        ),
    )


def bootstrap_mean(
    values: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap input must contain at least two finite values.")
    estimates = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 250):
        stop = min(draws, start + 250)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        estimates[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def pair_and_summarize(
    single: pd.DataFrame,
    cumulative: pd.DataFrame,
    cohorts: dict[str, list[str]],
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = single[
        ["depth", "dataset", "transcript_id", "mu_pcc"]
    ].rename(columns={"mu_pcc": "single_mu_pcc"})
    paired = cumulative.merge(
        baseline,
        on=["depth", "dataset", "transcript_id"],
        how="left",
        validate="many_to_one",
    )
    paired["paired_delta"] = paired["mu_pcc"] - paired["single_mu_pcc"]
    cohort_sets = {depth: set(ids) for depth, ids in cohorts.items()}
    paired["in_fixed_complete_cohort"] = [
        transcript_id in cohort_sets[depth]
        for depth, transcript_id in zip(paired["depth"], paired["transcript_id"])
    ]
    paired = paired.loc[paired["in_fixed_complete_cohort"]].copy()

    rng = np.random.default_rng(bootstrap_seed)
    condition_rows: list[dict[str, Any]] = []
    for (depth, n_datasets, dataset), group in paired.groupby(
        ["depth", "n_datasets", "dataset"], sort=False
    ):
        values = group["mu_pcc"].to_numpy(dtype=np.float64)
        baseline_values = group["single_mu_pcc"].to_numpy(dtype=np.float64)
        delta = group["paired_delta"].to_numpy(dtype=np.float64)
        delta_low, delta_high = bootstrap_mean(
            delta, draws=bootstrap_draws, rng=rng
        )
        condition_rows.append(
            {
                "depth": depth,
                "depth_label": DEPTH_LABELS[depth],
                "n_datasets": int(n_datasets),
                "dataset": dataset,
                "bias_label": BIAS_LABELS[dataset],
                "n_transcripts": len(group),
                "cohort_hash": cohort_hash(cohorts[depth]),
                "mean_mu_pcc": float(np.mean(values)),
                "median_mu_pcc": float(np.median(values)),
                "p05_mu_pcc": float(np.quantile(values, 0.05)),
                "q25_mu_pcc": float(np.quantile(values, 0.25)),
                "q75_mu_pcc": float(np.quantile(values, 0.75)),
                "p95_mu_pcc": float(np.quantile(values, 0.95)),
                "mean_single_mu_pcc": float(np.mean(baseline_values)),
                "median_single_mu_pcc": float(np.median(baseline_values)),
                "mean_paired_delta": float(np.mean(delta)),
                "median_paired_delta": float(np.median(delta)),
                "paired_delta_ci95_low": delta_low,
                "paired_delta_ci95_high": delta_high,
            }
        )
    condition_summary = pd.DataFrame.from_records(condition_rows)

    aggregate_rows: list[dict[str, Any]] = []
    per_transcript_rows: list[dict[str, Any]] = []
    for (depth, n_datasets, transcript_id), group in paired.groupby(
        ["depth", "n_datasets", "transcript_id"], sort=False
    ):
        if len(group) != int(n_datasets):
            raise ValueError(
                f"Incomplete panel for {depth}, N={n_datasets}, {transcript_id}."
            )
        per_transcript_rows.append(
            {
                "depth": depth,
                "n_datasets": int(n_datasets),
                "transcript_id": transcript_id,
                "cumulative_mean_mu_pcc": float(group["mu_pcc"].mean()),
                "matched_single_mean_mu_pcc": float(
                    group["single_mu_pcc"].mean()
                ),
                "paired_mean_delta": float(group["paired_delta"].mean()),
            }
        )
    aggregate_per_transcript = pd.DataFrame.from_records(per_transcript_rows)
    for (depth, n_datasets), group in aggregate_per_transcript.groupby(
        ["depth", "n_datasets"], sort=False
    ):
        cumulative_values = group["cumulative_mean_mu_pcc"].to_numpy(float)
        baseline_values = group["matched_single_mean_mu_pcc"].to_numpy(float)
        delta = group["paired_mean_delta"].to_numpy(float)
        cumulative_low, cumulative_high = bootstrap_mean(
            cumulative_values, draws=bootstrap_draws, rng=rng
        )
        baseline_low, baseline_high = bootstrap_mean(
            baseline_values, draws=bootstrap_draws, rng=rng
        )
        delta_low, delta_high = bootstrap_mean(
            delta, draws=bootstrap_draws, rng=rng
        )
        aggregate_rows.append(
            {
                "depth": depth,
                "depth_label": DEPTH_LABELS[depth],
                "n_datasets": int(n_datasets),
                "n_transcripts": len(group),
                "cohort_hash": cohort_hash(cohorts[depth]),
                "mean_cumulative_mu_pcc": float(cumulative_values.mean()),
                "cumulative_ci95_low": cumulative_low,
                "cumulative_ci95_high": cumulative_high,
                "mean_matched_single_mu_pcc": float(baseline_values.mean()),
                "matched_single_ci95_low": baseline_low,
                "matched_single_ci95_high": baseline_high,
                "mean_paired_delta": float(delta.mean()),
                "paired_delta_ci95_low": delta_low,
                "paired_delta_ci95_high": delta_high,
            }
        )
    return paired, condition_summary, pd.DataFrame(aggregate_rows)


def add_single_cells(
    single: pd.DataFrame,
    cohorts: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (depth, dataset), group in single.groupby(["depth", "dataset"], sort=False):
        selected = group.loc[group["transcript_id"].isin(cohorts[depth])]
        values = selected["mu_pcc"].to_numpy(float)
        rows.append(
            {
                "depth": depth,
                "n_datasets": 1,
                "dataset": dataset,
                "n_transcripts": len(values),
                "mean_mu_pcc": float(values.mean()),
                "median_mu_pcc": float(np.median(values)),
            }
        )
    return pd.DataFrame.from_records(rows)


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def plot_aggregate(summary: pd.DataFrame, output: Path) -> None:
    style = configure_style()
    with matplotlib.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.25))
        figure.subplots_adjust(left=0.10, right=0.985, bottom=0.19, top=0.82, wspace=0.31)
        for depth in DEPTHS:
            cell = summary.loc[summary["depth"] == depth].sort_values("n_datasets")
            n_values = cell["n_datasets"].to_numpy(int)
            color = DEPTH_COLORS[depth]
            axes[0].plot(
                n_values,
                cell["mean_cumulative_mu_pcc"],
                color=color,
                marker="o",
                linewidth=2.0,
                markersize=4.2,
                label=DEPTH_LABELS[depth],
            )
            axes[0].plot(
                n_values,
                cell["mean_matched_single_mu_pcc"],
                color=color,
                linestyle="--",
                linewidth=1.35,
                marker="x",
                markersize=4.6,
                markeredgewidth=1.1,
                alpha=0.72,
                zorder=4,
            )
            axes[1].errorbar(
                n_values,
                cell["mean_paired_delta"],
                yerr=[
                    cell["mean_paired_delta"] - cell["paired_delta_ci95_low"],
                    cell["paired_delta_ci95_high"] - cell["mean_paired_delta"],
                ],
                color=color,
                marker="o",
                linewidth=2.0,
                markersize=4.2,
                elinewidth=0.8,
                capsize=2.0,
            )
        axes[0].set_title("A  Absolute reconstruction", loc="left")
        axes[0].set_ylabel(r"Mean PCC$(\mu_{dt},\overline{Y}_{dt})$")
        axes[1].set_title("B  Change from matched $N=1$ fits", loc="left")
        axes[1].set_ylabel(r"Mean paired $\Delta$PCC")
        axes[1].axhline(0.0, color="#666666", linestyle=":", linewidth=1.2)
        for axis in axes:
            axis.set_xlabel("Datasets in cumulative model, $N$")
            axis.set_xticks(PANEL_SIZES)
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
        depth_handles = [
            Line2D([], [], color=DEPTH_COLORS[depth], marker="o", label=DEPTH_LABELS[depth])
            for depth in DEPTHS
        ]
        model_handles = [
            Line2D([], [], color="#333333", linewidth=2.0, label="Cumulative model"),
            Line2D([], [], color="#333333", linestyle="--", marker="x", linewidth=1.35,
                   label="Matched single-model mean"),
        ]
        figure.legend(
            handles=depth_handles + model_handles,
            loc="upper center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.54, 1.01),
            columnspacing=0.9,
            handletextpad=0.35,
        )
        save_figure(figure, output / "synthetic_mu_robustness_aggregate")


def _heatmap_matrix(
    table: pd.DataFrame,
    depth: str,
    columns: tuple[int, ...],
    value: str,
) -> np.ndarray:
    matrix = np.full((len(BIAS_ORDER), len(columns)), np.nan, dtype=float)
    selected = table.loc[table["depth"] == depth]
    for row_index, dataset in enumerate(BIAS_ORDER):
        for column_index, n_datasets in enumerate(columns):
            cell = selected.loc[
                (selected["dataset"] == dataset)
                & (selected["n_datasets"] == n_datasets),
                value,
            ]
            if len(cell) > 1:
                raise ValueError(f"Duplicate heatmap cell: {depth}, N={n_datasets}, {dataset}")
            if len(cell) == 1:
                matrix[row_index, column_index] = float(cell.iloc[0])
    return matrix


def plot_heatmaps(
    absolute_table: pd.DataFrame,
    condition_summary: pd.DataFrame,
    output: Path,
) -> None:
    style = configure_style()
    specifications = (
        (
            "synthetic_mu_pcc_condition_resolved",
            tuple(range(1, 11)),
            absolute_table,
            "mean_mu_pcc",
            "Mean transcript PCC",
            "viridis",
            None,
        ),
        (
            "synthetic_mu_pcc_delta_from_single",
            PANEL_SIZES,
            condition_summary,
            "mean_paired_delta",
            r"Mean paired $\Delta$PCC (cumulative $-$ single)",
            "RdBu_r",
            0.0,
        ),
    )
    with matplotlib.rc_context(style):
        for stem, columns, table, value, colorbar_label, cmap_name, center in specifications:
            matrices = [_heatmap_matrix(table, depth, columns, value) for depth in DEPTHS]
            finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
            if center is None:
                vmin = min(0.35, float(finite.min()))
                vmax = max(0.95, float(finite.max()))
                norm = None
            else:
                limit = float(np.max(np.abs(finite)))
                vmin, vmax = -limit, limit
                norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("#EEEEEE")
            figure, axes = plt.subplots(3, 1, figsize=(7.15, 9.25))
            figure.subplots_adjust(left=0.17, right=0.89, bottom=0.07, top=0.95, hspace=0.36)
            image = None
            for panel_index, (axis, depth, matrix) in enumerate(zip(axes, DEPTHS, matrices)):
                image = axis.imshow(
                    matrix,
                    aspect="auto",
                    interpolation="none",
                    cmap=cmap,
                    vmin=None if norm is not None else vmin,
                    vmax=None if norm is not None else vmax,
                    norm=norm,
                )
                axis.set_title(
                    f"{chr(65 + panel_index)}  {DEPTH_LABELS[depth]}",
                    loc="left",
                    pad=5,
                )
                axis.set_xticks(np.arange(len(columns)), [str(value) for value in columns])
                axis.set_yticks(
                    np.arange(len(BIAS_ORDER)),
                    [BIAS_LABELS[dataset] for dataset in BIAS_ORDER],
                )
                axis.set_xlabel(
                    "Training configuration: single model" if columns[0] == 1 else "Datasets in cumulative model, $N$"
                )
                if columns[0] == 1:
                    axis.set_xlabel(
                        r"Training configuration ($N=1$ is bias-specific; $N\geq2$ is cumulative)"
                    )
                for row_index in range(matrix.shape[0]):
                    for column_index in range(matrix.shape[1]):
                        cell = matrix[row_index, column_index]
                        if not np.isfinite(cell):
                            continue
                        scaled = (cell - vmin) / max(vmax - vmin, np.finfo(float).eps)
                        text_color = "white" if scaled < 0.30 or scaled > 0.78 else "black"
                        axis.text(
                            column_index,
                            row_index,
                            f"{cell:.3f}" if center is None else f"{cell:+.4f}",
                            ha="center",
                            va="center",
                            fontsize=6.2 if center is None else 5.7,
                            fontweight="bold",
                            color=text_color,
                            fontfamily="Latin Modern Roman",
                            usetex=False,
                        )
                axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
                axis.set_yticks(np.arange(-0.5, len(BIAS_ORDER), 1), minor=True)
                axis.grid(which="minor", color="white", linewidth=0.55)
                axis.tick_params(which="minor", bottom=False, left=False)
            if image is None:
                raise RuntimeError("No heatmap was rendered.")
            colorbar_axis = figure.add_axes([0.915, 0.12, 0.022, 0.76])
            colorbar = figure.colorbar(image, cax=colorbar_axis)
            colorbar.set_label(colorbar_label, fontweight="bold")
            save_figure(figure, output / stem)


def write_report(
    output: Path,
    aggregate: pd.DataFrame,
    cohorts: dict[str, list[str]],
    exclusions: pd.DataFrame,
) -> None:
    endpoint_rows = []
    for depth in DEPTHS:
        selected = aggregate.loc[aggregate["depth"] == depth].set_index("n_datasets")
        smallest = selected.loc[2]
        largest = selected.loc[10]
        endpoint_rows.append(
            (
                DEPTH_LABELS[depth],
                float(selected["mean_paired_delta"].min()),
                float(selected["mean_paired_delta"].max()),
                float(smallest["mean_cumulative_mu_pcc"]),
                float(largest["mean_cumulative_mu_pcc"]),
                float(largest["mean_paired_delta"]),
                float(largest["paired_delta_ci95_low"]),
                float(largest["paired_delta_ci95_high"]),
            )
        )
    lines = [
        "# Synthetic observation-prediction robustness",
        "",
        "Frozen minimum-validation-loss checkpoints were used throughout.",
        "The metric is transcript-level PCC between exported mu and the exported observed",
        "two-replica consensus on model-valid positions. Undefined PCCs are excluded by a",
        "fixed complete-cohort rule; they are never replaced by zero.",
        "",
        "| Depth | Fixed cohort | Cohort hash | Excluded |",
        "|---|---:|---|---:|",
    ]
    for depth in DEPTHS:
        excluded = int((exclusions["depth"] == depth).sum()) if not exclusions.empty else 0
        lines.append(
            f"| {DEPTH_LABELS[depth]} | {len(cohorts[depth]):,} | "
            f"`{cohort_hash(cohorts[depth])[:12]}` | {excluded} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate results",
            "",
            "| Depth | Paired delta range over N | cumulative PCC N=2 | cumulative PCC N=10 | N=10 paired delta [95% CI] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, low, high, n2, n10, delta, ci_low, ci_high in endpoint_rows:
        lines.append(
            f"| {label} | {low:+.4f} to {high:+.4f} | {n2:.4f} | {n10:.4f} | "
            f"{delta:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The dashed baselines are recomputed at every N from the same first N named bias conditions; this prevents changing panel composition from masquerading as a single-versus-cumulative effect.",
            "- A near-zero paired delta means that joint training preserves observation reconstruction relative to separate fits. It does not establish recovery of K_t or correct L/gamma separation.",
            "- The condition-resolved maps should be consulted before making an aggregate robustness claim: later biases occur in fewer cumulative panels, and a panel mean can hide a condition-specific loss.",
            "- This is one training seed and validation data are used both for checkpoint selection (via validation loss) and evaluation. Transcript-bootstrap intervals condition on the trained models and are not training-seed uncertainty.",
            "- Increasing N changes training-set size and bias composition simultaneously. The curves are a fixed-order stress test, not a causal estimate of dataset count alone.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single-metrics",
        type=Path,
        default=ROOT / "analyses/artifacts/synthetic/individual_dataset/single_model_vs_replica_per_transcript.parquet",
    )
    parser.add_argument(
        "--single-provenance",
        type=Path,
        default=ROOT / "analyses/artifacts/synthetic/individual_dataset/provenance.json",
    )
    parser.add_argument(
        "--cumulative-provenance",
        type=Path,
        default=ROOT / "analyses/artifacts/synthetic/read_depth/multidataset_mu_reconstruction/provenance.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analyses/artifacts/synthetic/prediction_robustness",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_draws < 100:
        raise ValueError("Use at least 100 bootstrap draws.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    single, validation, single_provenance = load_single_metrics(
        args.single_metrics.resolve(), args.single_provenance.resolve()
    )
    cumulative_source = json.loads(
        args.cumulative_provenance.resolve().read_text(encoding="utf-8")
    )
    source_runs = [
        run
        for run in cumulative_source.get("runs", [])
        if run.get("depth") in DEPTHS and len(run.get("datasets", [])) in PANEL_SIZES
    ]
    expected_grid = {(depth, n) for depth in DEPTHS for n in PANEL_SIZES}
    actual_grid = {(str(run["depth"]), len(run["datasets"])) for run in source_runs}
    if len(source_runs) != len(expected_grid) or actual_grid != expected_grid:
        raise ValueError("Cumulative provenance does not contain one complete 3 x 9 grid.")

    audited_runs: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for run in sorted(source_runs, key=lambda item: (DEPTHS.index(item["depth"]), len(item["datasets"]))):
        audited = validate_run(run, validation)
        print(
            f"[{audited['depth']}] N={audited['n_datasets']}: "
            f"{Path(audited['prediction_path']).name}",
            flush=True,
        )
        frames.append(stream_cumulative_prediction(audited))
        audited["dataset_id_to_name"] = {
            str(key): value for key, value in audited["dataset_id_to_name"].items()
        }
        audited_runs.append(audited)
    cumulative = pd.concat(frames, ignore_index=True)

    cohorts, exclusions = fixed_complete_cohorts(single, cumulative, validation)
    if any(len(ids) < 2 for ids in cohorts.values()):
        raise ValueError("At least one fixed complete cohort has fewer than two transcripts.")
    paired, condition_summary, aggregate_summary = pair_and_summarize(
        single,
        cumulative,
        cohorts,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    single_cells = add_single_cells(single, cohorts)
    absolute_table = pd.concat(
        [single_cells, condition_summary[single_cells.columns]], ignore_index=True
    )

    paired.to_parquet(output / "per_transcript_paired_metrics.parquet", index=False)
    condition_summary.to_csv(output / "condition_resolved_summary.csv", index=False)
    aggregate_summary.to_csv(output / "panel_aggregate_summary.csv", index=False)
    exclusions.to_csv(output / "exclusions.csv", index=False)
    pd.DataFrame(
        [
            {
                "depth": depth,
                "transcript_id": transcript_id,
                "cohort_hash": cohort_hash(cohorts[depth]),
            }
            for depth in DEPTHS
            for transcript_id in cohorts[depth]
        ]
    ).to_csv(output / "fixed_cohort_ids.csv", index=False)

    plot_aggregate(aggregate_summary, output)
    plot_heatmaps(absolute_table, condition_summary, output)
    write_report(output, aggregate_summary, cohorts, exclusions)

    command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    provenance = {
        "analysis": "synthetic_observation_prediction_robustness",
        "command": command,
        "checkpoint_variant": CHECKPOINT_VARIANT,
        "training_seed": 42,
        "reference_weighting": "equal",
        "panel_order": list(BIAS_ORDER),
        "metric": (
            "Within-transcript Pearson PCC(exported mu, exported observed target) "
            "on finite model-mask positions; undefined correlations remain missing."
        ),
        "single_baseline": (
            "For every cumulative dataset/transcript row, the separately trained N=1 "
            "model is matched by depth, named bias, and transcript ID."
        ),
        "cohorts": {
            depth: {
                "n": len(ids),
                "sha256": cohort_hash(ids),
                "validation_candidates": len(validation[depth]),
            }
            for depth, ids in cohorts.items()
        },
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
            "unit": "transcript",
            "paired": True,
            "interval": "pointwise percentile 95%",
            "training_seed_uncertainty": False,
        },
        "inputs": {
            "single_metrics": str(args.single_metrics.resolve()),
            "single_metrics_sha256": sha256(args.single_metrics.resolve()),
            "single_provenance": str(args.single_provenance.resolve()),
            "single_provenance_sha256": sha256(args.single_provenance.resolve()),
            "cumulative_provenance": str(args.cumulative_provenance.resolve()),
            "cumulative_provenance_sha256": sha256(args.cumulative_provenance.resolve()),
        },
        "single_model_comparison": single_provenance["single_model_comparison"],
        "cumulative_runs": audited_runs,
        "limitations": [
            "Validation data are used for checkpoint selection and evaluation.",
            "Only one training seed is represented.",
            "Increasing N changes both sample size and bias composition.",
            "Observation reconstruction does not establish recovery of K_t, L_t, or gamma.",
        ],
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (output / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "RIBOUNMIX_PLOT_TEX=1 .venv/bin/python "
        "analyses/analyze_synthetic_prediction_robustness.py\n",
        encoding="utf-8",
    )
    print(aggregate_summary.to_string(index=False))
    print(output)


if __name__ == "__main__":
    main()
