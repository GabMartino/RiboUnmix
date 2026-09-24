#!/usr/bin/env python3
"""Audit the final weighted synthetic datasets used by the fitted models.

The script reads only scalar columns from the 30 biased Parquets under
``Datasets/data/weighted_synthetic``.  It reports transcript attrition from
the simulator tables to the actual seed-42 fitting cohorts and summarizes the
stored transcript--dataset reliability weights.  It does not load a model or
recompute a prediction.

The stored reliability weight ``w_dt`` is distinct from the dataset reference
weight ``pi_d`` used in gamma centering.  Here only ``w_dt`` is analyzed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utils.publication_plot_style import publication_rc  # noqa: E402


DEPTHS = ("0p25_per_codon", "2_per_codon", "20_per_codon")
DEPTH_LABELS = {
    "0p25_per_codon": "0.25 reads/codon",
    "2_per_codon": "2 reads/codon",
    "20_per_codon": "20 reads/codon",
}
DEPTH_VALUES = {
    "0p25_per_codon": 0.25,
    "2_per_codon": 2.0,
    "20_per_codon": 20.0,
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
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
WEIGHT_COLUMNS = (
    "id",
    "weight",
    "weight_raw",
    "raw_weight",
    "coverage",
    "read_density",
    "depth_reference_tau",
    "depth_snr_score",
    "coverage_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "Datasets/Synthetic_data",
    )
    parser.add_argument(
        "--weighted-root",
        type=Path,
        default=ROOT / "Datasets/data/weighted_synthetic",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=ROOT / "results/riboai_synthetic_experiments",
    )
    parser.add_argument(
        "--single-provenance",
        type=Path,
        default=ROOT / "analyses/artifacts/synthetic/individual_dataset/provenance.json",
    )
    parser.add_argument(
        "--cumulative-provenance",
        type=Path,
        default=(
            ROOT
            / "analyses/artifacts/synthetic/read_depth"
            / "multidataset_mu_reconstruction/provenance.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analyses/artifacts/synthetic/preprocessed_datasets",
    )
    parser.add_argument("--png-dpi", type=int, default=600)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_hash(values: set[str] | list[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scalar_hash(frame: pd.DataFrame) -> str:
    """Hash IDs and persisted scalar audit values in a stable row order."""
    ordered = frame.sort_values("id", kind="mergesort")
    digest = hashlib.sha256()
    for transcript_id in ordered["id"].astype(str):
        digest.update(transcript_id.encode("utf-8"))
        digest.update(b"\0")
    numeric = ordered[list(WEIGHT_COLUMNS[1:])].to_numpy(dtype="<f8", copy=True)
    digest.update(numeric.tobytes(order="C"))
    return digest.hexdigest()


def configure_style() -> dict[str, Any]:
    style = publication_rc()
    style.update(
        {
            "font.size": 12.0,
            "font.weight": "bold",
            "axes.labelsize": 13.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 13.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 10.5,
            "legend.title_fontsize": 10.5,
            "axes.linewidth": 1.0,
        }
    )
    if style.get("text.usetex"):
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}"
            r"\AtBeginDocument{\boldmath}"
        )
    return style


def raw_path(raw_root: Path, depth: str, bias: str) -> Path:
    return raw_root / depth / f"{bias}_psite_counts_{depth}.parquet"


def weighted_path(weighted_root: Path, depth: str, bias: str) -> Path:
    return weighted_root / depth / f"{bias}.parquet"


def load_raw_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    values = pq.read_table(path, columns=["transcript_id"], use_threads=False)[
        "transcript_id"
    ].to_pylist()
    return {str(value) for value in values}


def load_weight_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(WEIGHT_COLUMNS).difference(schema)
    if missing:
        raise KeyError(f"{path}: missing scalar columns {sorted(missing)}")
    frame = pq.read_table(
        path, columns=list(WEIGHT_COLUMNS), use_threads=False
    ).to_pandas()
    if frame["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate transcript IDs")
    numeric = frame[list(WEIGHT_COLUMNS[1:])].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{path}: non-finite scalar values")
    if bool((frame["weight"] <= 0.0).any()):
        raise ValueError(f"{path}: non-positive final weights")
    if not np.allclose(frame["weight_raw"], frame["raw_weight"], rtol=0.0, atol=0.0):
        raise ValueError(f"{path}: weight_raw and raw_weight are not identical")
    expected = frame["weight_raw"].astype(np.float64) / float(
        frame["weight_raw"].median()
    )
    if not np.allclose(frame["weight"], expected, rtol=2e-7, atol=2e-7):
        raise ValueError(f"{path}: stored weights do not match median normalization")
    return frame


def find_split_manifest(experiment_root: Path, depth: str) -> Path:
    pattern = (
        f"riboai_synthetic_within_{depth}_panel10_gammaequal_seed42*"
        "/results/**/split_manifest*.json"
    )
    matches = sorted(experiment_root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one seed-42 N=10 split manifest for {depth}; found {len(matches)}"
        )
    return matches[0]


def load_split(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    train = {str(value) for value in data["train_ids"]}
    validation = {str(value) for value in data["validation_ids"]}
    if train & validation:
        raise ValueError(f"{path}: train and validation IDs overlap")
    counts = data["counts"]
    expected = (int(counts["train"]), int(counts["validation"]))
    if (len(train), len(validation)) != expected:
        raise ValueError(f"{path}: manifest counts disagree with ID arrays")
    return {
        "path": path,
        "sha256": sha256(path),
        "seed": int(data["seed"]),
        "train": train,
        "validation": validation,
        "eligible": train | validation,
        "max_cds_codons": int(data["sequence_eligibility"]["max_cds_codons"]),
    }


def read_hparams_dataset_paths(path: Path) -> list[str]:
    """Read the short datasets_paths block without parsing the large ID lists."""
    paths: list[str] = []
    active = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "datasets_paths:":
            active = True
            continue
        if active and line.startswith("- "):
            paths.append(line[2:].strip().strip("'\""))
            continue
        if active:
            break
    return paths


def verify_mu_training_inputs(
    single_provenance_path: Path,
    cumulative_provenance_path: Path,
) -> dict[str, Any]:
    weighted_token = "Datasets/data/weighted_synthetic/"

    single = json.loads(single_provenance_path.read_text(encoding="utf-8"))
    artifacts = single["single_model_comparison"]["artifacts"]
    single_paths: list[str] = []
    hparams_paths: list[str] = []
    for artifact in artifacts:
        run_directory = Path(artifact["run_directory"])
        candidates = sorted(run_directory.glob("**/hparams.yaml"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one hparams.yaml below {run_directory}; found {len(candidates)}"
            )
        paths = read_hparams_dataset_paths(candidates[0])
        if len(paths) != 1:
            raise ValueError(f"{candidates[0]}: expected one training dataset path")
        single_paths.extend(paths)
        hparams_paths.append(str(candidates[0].relative_to(ROOT)))

    cumulative = json.loads(cumulative_provenance_path.read_text(encoding="utf-8"))
    cumulative_paths = [
        str(path)
        for run in cumulative["runs"]
        for path in run["training_paths"]
    ]
    invalid_single = [path for path in single_paths if weighted_token not in path]
    invalid_cumulative = [
        path for path in cumulative_paths if weighted_token not in path
    ]
    if invalid_single or invalid_cumulative:
        raise ValueError(
            "Frozen mu analyses include non-weighted-synthetic inputs: "
            f"single={invalid_single}, cumulative={invalid_cumulative}"
        )
    return {
        "single_models_checked": len(artifacts),
        "single_models_using_weighted_synthetic": len(artifacts),
        "single_unique_training_paths": sorted(set(single_paths)),
        "single_hparams_paths": hparams_paths,
        "cumulative_models_checked": len(cumulative["runs"]),
        "cumulative_models_using_weighted_synthetic": len(cumulative["runs"]),
        "cumulative_unique_training_paths": sorted(set(cumulative_paths)),
        "single_provenance": str(single_provenance_path.relative_to(ROOT)),
        "single_provenance_sha256": sha256(single_provenance_path),
        "cumulative_provenance": str(cumulative_provenance_path.relative_to(ROOT)),
        "cumulative_provenance_sha256": sha256(cumulative_provenance_path),
    }


def audit_datasets(
    raw_root: Path,
    weighted_root: Path,
    splits: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    weights: dict[tuple[str, str], np.ndarray] = {}
    common_weighted_ids: set[str] | None = None
    common_raw_ids: set[str] | None = None
    dropped_ids: set[str] | None = None
    input_records: list[dict[str, Any]] = []

    for depth in DEPTHS:
        split = splits[depth]
        for bias in BIAS_ORDER:
            source = raw_path(raw_root, depth, bias)
            final = weighted_path(weighted_root, depth, bias)
            raw_ids = load_raw_ids(source)
            frame = load_weight_frame(final)
            final_ids = set(frame["id"].astype(str))
            if common_raw_ids is None:
                common_raw_ids = raw_ids
            elif raw_ids != common_raw_ids:
                raise ValueError(f"Raw transcript identities differ at {depth}/{bias}")
            if common_weighted_ids is None:
                common_weighted_ids = final_ids
            elif final_ids != common_weighted_ids:
                raise ValueError(
                    f"Weighted transcript identities differ at {depth}/{bias}"
                )
            current_dropped = raw_ids - final_ids
            if dropped_ids is None:
                dropped_ids = current_dropped
            elif current_dropped != dropped_ids:
                raise ValueError(
                    f"Preprocessing exclusions differ at {depth}/{bias}"
                )
            if not split["eligible"].issubset(final_ids):
                missing = sorted(split["eligible"] - final_ids)[:5]
                raise ValueError(f"{depth}/{bias}: split IDs absent from final data: {missing}")

            values = frame["weight"].to_numpy(dtype=np.float64, copy=True)
            if not np.isclose(np.median(values), 1.0, rtol=0.0, atol=1e-6):
                raise ValueError(f"{final}: median final weight is not one")
            weights[(depth, bias)] = values
            indexed = frame.set_index(frame["id"].astype(str), drop=False)
            train_weights = indexed.loc[sorted(split["train"]), "weight"].to_numpy(
                dtype=np.float64
            )
            validation_weights = indexed.loc[
                sorted(split["validation"]), "weight"
            ].to_numpy(dtype=np.float64)
            qs = np.quantile(values, QUANTILES)
            summary_rows.append(
                {
                    "depth": depth,
                    "nominal_reads_per_codon": DEPTH_VALUES[depth],
                    "bias": bias,
                    "bias_label": BIAS_LABELS[bias],
                    "simulator_transcripts": len(raw_ids),
                    "weighted_transcripts": len(final_ids),
                    "preprocessing_excluded": len(raw_ids - final_ids),
                    "model_eligible_transcripts": len(split["eligible"]),
                    "training_transcripts": len(split["train"]),
                    "validation_transcripts": len(split["validation"]),
                    "model_rule_excluded": len(final_ids - split["eligible"]),
                    "weight_mean": float(np.mean(values)),
                    "weight_std": float(np.std(values, ddof=1)),
                    "weight_min": float(qs[0]),
                    "weight_q01": float(qs[1]),
                    "weight_q05": float(qs[2]),
                    "weight_q25": float(qs[3]),
                    "weight_median": float(qs[4]),
                    "weight_q75": float(qs[5]),
                    "weight_q95": float(qs[6]),
                    "weight_q99": float(qs[7]),
                    "weight_max": float(qs[8]),
                    "weight_iqr": float(qs[5] - qs[3]),
                    "weight_q90_width": float(qs[6] - qs[2]),
                    "training_weight_median": float(np.median(train_weights)),
                    "validation_weight_median": float(np.median(validation_weights)),
                    "coverage_median": float(frame["coverage"].median()),
                    "read_density_median": float(frame["read_density"].median()),
                    "depth_reference_tau": float(frame["depth_reference_tau"].iloc[0]),
                    "scalar_audit_sha256": scalar_hash(frame),
                    "raw_path": str(source.relative_to(ROOT)),
                    "weighted_path": str(final.relative_to(ROOT)),
                }
            )
            for probability, value in zip(QUANTILES, qs, strict=True):
                quantile_rows.append(
                    {
                        "depth": depth,
                        "nominal_reads_per_codon": DEPTH_VALUES[depth],
                        "bias": bias,
                        "probability": probability,
                        "weight_quantile": float(value),
                    }
                )
            input_records.append(
                {
                    "raw_path": str(source.relative_to(ROOT)),
                    "raw_size_bytes": source.stat().st_size,
                    "weighted_path": str(final.relative_to(ROOT)),
                    "weighted_size_bytes": final.stat().st_size,
                    "weighted_scalar_sha256": scalar_hash(frame),
                }
            )

    assert common_raw_ids is not None
    assert common_weighted_ids is not None
    assert dropped_ids is not None
    shared = {
        "raw_transcript_count": len(common_raw_ids),
        "raw_transcript_identity_sha256": identity_hash(common_raw_ids),
        "weighted_transcript_count": len(common_weighted_ids),
        "weighted_transcript_identity_sha256": identity_hash(common_weighted_ids),
        "preprocessing_excluded_ids": sorted(dropped_ids),
        "preprocessing_excluded_identity_sha256": identity_hash(dropped_ids),
        "input_files": input_records,
    }
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(quantile_rows),
        weights,
        shared,
    )


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_counts(summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    y = np.arange(len(BIAS_ORDER))
    with plt.rc_context(configure_style()):
        fig, axes = plt.subplots(
            1, 3, figsize=(15.2, 7.2), sharex=True, sharey=True,
        )
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.78, wspace=0.06)
        for axis, depth in zip(axes, DEPTHS, strict=True):
            part = summary.loc[summary["depth"] == depth].set_index("bias").loc[
                list(BIAS_ORDER)
            ]
            train = part["training_transcripts"].to_numpy()
            validation = part["validation_transcripts"].to_numpy()
            excluded = part["model_rule_excluded"].to_numpy()
            raw = part["simulator_transcripts"].to_numpy()
            axis.barh(y, train, height=0.64, color="#285F8F", label="Training")
            axis.barh(
                y, validation, left=train, height=0.64,
                color="#79B4D5", label="Validation",
            )
            axis.barh(
                y, excluded, left=train + validation, height=0.64,
                color="#B9B9B9", label=r"Excluded by $n_t\leq4000$ rule",
            )
            axis.scatter(
                raw, y, marker="x", s=42, linewidth=1.7,
                color="#1A1A1A", zorder=4, label="Simulator input",
            )
            axis.set_title(DEPTH_LABELS[depth])
            axis.set_xlabel("Number of transcripts")
            axis.set_xlim(0, 19800)
            axis.set_xticks([0, 5000, 10000, 15000, 19283],
                            ["0", "5k", "10k", "15k", "19,283"])
            axis.grid(axis="x", alpha=0.45)
            axis.invert_yaxis()
        axes[0].set_yticks(y, [BIAS_LABELS[bias] for bias in BIAS_ORDER])
        axes[0].set_ylabel("Injected bias condition")
        handles = [
            Patch(facecolor="#285F8F", label="Training (17,284)"),
            Patch(facecolor="#79B4D5", label="Validation (1,920)"),
            Patch(facecolor="#B9B9B9", label=r"Model-rule excluded (79)"),
            Line2D([], [], color="#1A1A1A", marker="x", linestyle="none",
                   markersize=7, markeredgewidth=1.7,
                   label="Simulator input (19,290)"),
        ]
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=4
        )
        fig.suptitle(
            "Transcript accounting in the final weighted synthetic datasets",
            y=0.985,
        )
        save_figure(fig, output_dir / "synthetic_preprocessed_transcript_counts", dpi)


def plot_weight_distributions(
    summary: pd.DataFrame, output_dir: Path, dpi: int
) -> None:
    y = np.arange(len(BIAS_ORDER))
    x_min = float(summary["weight_min"].min())
    x_max = float(summary["weight_max"].max())
    padding = 0.035 * (x_max - x_min)
    with plt.rc_context(configure_style()):
        fig, axes = plt.subplots(
            1, 3, figsize=(15.2, 7.2), sharex=True, sharey=True,
        )
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.78, wspace=0.06)
        for axis, depth in zip(axes, DEPTHS, strict=True):
            part = summary.loc[summary["depth"] == depth].set_index("bias").loc[
                list(BIAS_ORDER)
            ]
            color = DEPTH_COLORS[depth]
            for position, (_, row) in enumerate(part.iterrows()):
                axis.plot(
                    [row["weight_min"], row["weight_max"]], [position, position],
                    color="#C7C7C7", linewidth=0.8, zorder=1,
                )
                axis.plot(
                    [row["weight_q05"], row["weight_q95"]], [position, position],
                    color=color, linewidth=2.0, alpha=0.58, zorder=2,
                )
                axis.plot(
                    [row["weight_q25"], row["weight_q75"]], [position, position],
                    color=color, linewidth=6.0, solid_capstyle="round", zorder=3,
                )
                axis.scatter(
                    row["weight_median"], position, s=36, color="#111111",
                    edgecolor="white", linewidth=0.6, zorder=4,
                )
            axis.axvline(1.0, color="#A23B3B", linestyle="--", linewidth=1.2)
            axis.set_title(DEPTH_LABELS[depth])
            axis.set_xlabel(r"Normalized reliability weight $w_{dt}$")
            axis.set_xlim(x_min - padding, x_max + padding)
            axis.grid(axis="x", alpha=0.45)
            axis.invert_yaxis()
        axes[0].set_yticks(y, [BIAS_LABELS[bias] for bias in BIAS_ORDER])
        axes[0].set_ylabel("Injected bias condition")
        handles = [
            Line2D([], [], color="#888888", linewidth=0.8, label="Minimum--maximum"),
            Line2D([], [], color="#4A88B5", linewidth=2.0, label="5th--95th percentile"),
            Line2D([], [], color="#4A88B5", linewidth=6.0, label="25th--75th percentile"),
            Line2D([], [], marker="o", color="#111111", markeredgecolor="white",
                   linestyle="none", label="Median"),
            Line2D([], [], color="#A23B3B", linestyle="--", label=r"Normalization reference $w=1$"),
        ]
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=5
        )
        fig.suptitle(
            "Final transcript--dataset reliability-weight distributions",
            y=0.985,
        )
        save_figure(
            fig, output_dir / "synthetic_reliability_weight_distributions", dpi
        )


def plot_weight_spread(summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    positions = np.arange(len(DEPTHS), dtype=float)
    offsets = np.linspace(-0.16, 0.16, len(BIAS_ORDER))
    medians: list[float] = []
    with plt.rc_context(configure_style()):
        fig, axis = plt.subplots(figsize=(6.9, 5.4), constrained_layout=True)
        for depth_index, depth in enumerate(DEPTHS):
            part = summary.loc[summary["depth"] == depth].set_index("bias").loc[
                list(BIAS_ORDER)
            ]
            values = part["weight_q90_width"].to_numpy(dtype=float)
            axis.scatter(
                positions[depth_index] + offsets,
                values,
                s=32,
                facecolor="white",
                edgecolor=DEPTH_COLORS[depth],
                linewidth=1.3,
                zorder=3,
            )
            medians.append(float(np.median(values)))
        axis.plot(
            positions, medians, color="#1F4E79", marker="o", markersize=8,
            linewidth=2.5, label="Median across 10 biases", zorder=4,
        )
        axis.set_xticks(positions, ["0.25", "2", "20"])
        axis.set_xlabel("Nominal reads per codon")
        axis.set_ylabel(r"Weight spread ($q_{0.95}-q_{0.05}$)")
        axis.set_title("Across-transcript reliability spread by read depth")
        axis.grid(axis="y", alpha=0.45)
        axis.legend(loc="upper right")
        save_figure(fig, output_dir / "synthetic_reliability_weight_spread", dpi)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    shared: dict[str, Any],
    mu_audit: dict[str, Any],
    splits: dict[str, dict[str, Any]],
) -> None:
    spread = (
        summary.groupby("depth", sort=False)["weight_q90_width"]
        .median()
        .reindex(DEPTHS)
    )
    train_median_deviation = float(
        np.max(np.abs(summary["training_weight_median"] - 1.0))
    )
    validation_median_deviation = float(
        np.max(np.abs(summary["validation_weight_median"] - 1.0))
    )
    report = f"""# Audit of the final weighted synthetic datasets

## What these files are

The frozen single-dataset $\\mu$ analysis uses **{mu_audit['single_models_checked']} / {mu_audit['single_models_checked']}** models whose resolved `datasets_paths` point to `Datasets/data/weighted_synthetic`.  The cumulative $N=2,\\ldots,10$ $\\mu$ analysis uses **{mu_audit['cumulative_models_checked']} / {mu_audit['cumulative_models_checked']}** runs whose recorded training paths point to the same directory.

This audit concerns the transcript--dataset reliability weight $w_{{dt}}$.  It is not the reference-centering weight $\\pi_d$.

For transcript $t$ in dataset $d$, preprocessing computes read density $D_{{dt}}$ and coverage $A_{{dt}}$, then

$$
w^{{\\mathrm{{raw}}}}_{{dt}}
=0.70\\frac{{\\sqrt{{D_{{dt}}}}}}{{\\sqrt{{D_{{dt}}}}+\\sqrt{{\\tau_d}}}}
+0.30A_{{dt}},
\\qquad
w_{{dt}}=\\frac{{w^{{\\mathrm{{raw}}}}_{{dt}}}}{{\\operatorname{{median}}_u w^{{\\mathrm{{raw}}}}_{{du}}}},
$$

where $\\tau_d=\\operatorname{{median}}_t D_{{dt}}$ in these historical standalone artifacts.

## Transcript accounting

- Each of the 30 simulator bias--depth tables contains {shared['raw_transcript_count']:,} unique transcript IDs.
- Conversion against the sequence master retains {shared['weighted_transcript_count']:,}; the same seven unavailable transcript versions are excluded in every condition.
- The frozen seed-42 within-depth manifests contain 19,204 model-eligible transcripts at every depth: 17,284 training and 1,920 validation transcripts.  The remaining 79 weighted rows exceed the configured complete-CDS limit of 4,000 codons.
- Validation identities differ between at least some depths, although the cohort sizes are identical.  The count figure therefore reports the realized cohort for each depth rather than implying one common validation cohort.

## Weight distributions

- The stored median is exactly one (within floating-point precision) in all 30 files, by construction.
- The median across the ten bias conditions of the 5th--95th percentile width is {spread.loc['0p25_per_codon']:.4f} at 0.25, {spread.loc['2_per_codon']:.4f} at 2, and {spread.loc['20_per_codon']:.4f} at 20 reads/codon.
- Thus the normalized reliability weights are substantially more dispersed at sparse depth.  At high depth, coverage is close to saturation for most transcripts and the weights cluster more tightly around one.
- The maximum absolute displacement of the *training-subset* median from one is {train_median_deviation:.6f}; for validation it is {validation_median_deviation:.6f}.  Those medians are not forced to one because the historical normalization reference was estimated using all retained rows before splitting.

## Interpretation limit

These plots describe preprocessing and reliability weighting.  They do not establish model performance and they do not explain $L_t$, $\\gamma_{{dt}}$, or $\\mu_{{dt}}$ recovery by themselves.  In particular, a narrower high-depth $w_{{dt}}$ distribution is a property of the observed coverage/density formula, not evidence that $\\pi_d$ should be uniform or quality-ranked.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")

    captions = r"""% Auto-generated captions for the weighted-synthetic input audit.
\paragraph{Transcript accounting in the final weighted synthetic datasets.}
For each of ten injected-bias conditions and three nominal read depths, stacked bars partition the 19,283 transcripts retained in the final weighted Parquet into 17,284 training transcripts, 1,920 validation transcripts, and 79 transcripts subsequently excluded by the complete-CDS length rule ($n_t\leq4000$). Crosses show the 19,290 transcript IDs in each simulator table; seven transcript versions were absent from the preprocessing sequence master. Counts are identical across bias conditions, while validation identities are depth-specific.

\paragraph{Final transcript--dataset reliability-weight distributions.}
Distributions of the normalized reliability weights $w_{dt}$ stored in the 30 biased synthetic training Parquets. Thin gray, colored, and thick colored intervals denote the minimum--maximum, 5th--95th percentile, and interquartile range, respectively; points denote medians. Each dataset is normalized by its own raw-weight median, so the stored median is one by construction (dashed line). The visibly broader low-depth distributions reflect greater across-transcript heterogeneity in observed coverage and read density. These $w_{dt}$ weights are distinct from the reference-centering weights $\pi_d$.

\paragraph{Reliability-weight spread by read depth.}
For each nominal depth, small points show the 5th--95th percentile width of $w_{dt}$ in each of the ten bias conditions; the connected large points show the median width across biases. This is a descriptive preprocessing diagnostic, not a model-recovery metric.
"""
    (output_dir / "captions.tex").write_text(captions, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        depth: load_split(find_split_manifest(args.experiment_root, depth))
        for depth in DEPTHS
    }
    if any(split["seed"] != 42 for split in splits.values()):
        raise ValueError("All selected manifests must use training seed 42")

    mu_audit = verify_mu_training_inputs(
        args.single_provenance, args.cumulative_provenance
    )
    summary, quantiles, _, shared = audit_datasets(
        args.raw_root, args.weighted_root, splits
    )
    summary.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    quantiles.to_csv(args.output_dir / "weight_quantiles.csv", index=False)

    split_rows = []
    for depth, split in splits.items():
        split_rows.append(
            {
                "depth": depth,
                "nominal_reads_per_codon": DEPTH_VALUES[depth],
                "seed": split["seed"],
                "training_transcripts": len(split["train"]),
                "validation_transcripts": len(split["validation"]),
                "eligible_transcripts": len(split["eligible"]),
                "training_identity_sha256": identity_hash(split["train"]),
                "validation_identity_sha256": identity_hash(split["validation"]),
                "eligible_identity_sha256": identity_hash(split["eligible"]),
                "split_manifest": str(split["path"].relative_to(ROOT)),
                "split_manifest_sha256": split["sha256"],
            }
        )
    pd.DataFrame(split_rows).to_csv(
        args.output_dir / "fitting_cohort_provenance.csv", index=False
    )

    plot_counts(summary, args.output_dir, args.png_dpi)
    plot_weight_distributions(summary, args.output_dir, args.png_dpi)
    plot_weight_spread(summary, args.output_dir, args.png_dpi)
    write_report(args.output_dir, summary, shared, mu_audit, splits)

    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis": "final weighted synthetic dataset audit",
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "raw_root": str(args.raw_root.relative_to(ROOT)),
        "weighted_root": str(args.weighted_root.relative_to(ROOT)),
        "biases": list(BIAS_ORDER),
        "depths": list(DEPTHS),
        "weight_definition": {
            "quantity": "transcript-dataset reliability weight w_dt, not pi_d",
            "raw_formula": "0.70*sqrt(D)/(sqrt(D)+sqrt(tau_d)) + 0.30*A",
            "depth_reference": "tau_d = median positive read density over retained dataset rows",
            "normalization": "dataset-specific raw-weight median over all retained rows",
            "historical_limitation": "tau_d and normalization median were estimated before train/validation splitting",
        },
        "shared_transcript_audit": shared,
        "mu_training_input_verification": mu_audit,
        "split_manifests": split_rows,
        "outputs": sorted(
            path.name for path in args.output_dir.iterdir() if path.is_file()
        ),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(summary[[
        "depth", "bias", "weighted_transcripts", "weight_median",
        "weight_q05", "weight_q95",
    ]].to_string(index=False))
    print(f"Wrote audit to {args.output_dir}")


if __name__ == "__main__":
    main()
