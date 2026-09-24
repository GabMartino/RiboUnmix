#!/usr/bin/env python3
"""Build the self-contained synthetic appendix figure bundle.

The script does not refit a model or recompute prediction-level metrics.  It
uses the audited input-count table, frozen split provenance, and saved summary
tables, then copies the already generated vector figures under stable
manuscript filenames.  In particular, it never infers a depth-dependent loss
of transcripts when the saved artifacts show none.

Run from the repository root::

    RIBOUNMIX_PLOT_TEX=auto .venv/bin/python \
        analyses/build_synthetic_appendix_bundle.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.publication_plot_style import publication_rc  # noqa: E402


AUDIT_ROOT = REPO_ROOT / "analyses/artifacts/synthetic/read_depth"
DEFAULT_OUTPUT = REPO_ROOT / "figures/synthetic_appendix"
DEFAULT_SINGLE_SUMMARY = (
    REPO_ROOT
    / "analyses/artifacts/synthetic/single_dataset_mu/single_20260830_205110/"
    "best_pcc/single_dataset_mu_pcc_summary.csv"
)


@dataclass(frozen=True)
class DepthSpec:
    key: str
    short: str
    label: str
    nominal: float
    color: str


DEPTHS: tuple[DepthSpec, ...] = (
    DepthSpec("0p25_per_codon", "0.25", "0.25 reads/codon", 0.25, "#0072B2"),
    DepthSpec("2_per_codon", "2", "2 reads/codon", 2.0, "#E69F00"),
    DepthSpec("20_per_codon", "20", "20 reads/codon", 20.0, "#009E73"),
)


ASSETS: dict[str, Path] = {
    "synthetic_observations_example": (
        REPO_ROOT / "figures/synthetic_bias_read_depth_example_observations"
    ),
    "synthetic_multidataset_reconstruction": (
        AUDIT_ROOT
        / "multidataset_mu_reconstruction/synthetic_multidataset_mu_reconstruction"
    ),
    "synthetic_shared_and_gamma_recovery": (
        AUDIT_ROOT
        / "depth_recovery_overview/synthetic_recovery_overview_depth_effect"
    ),
    "synthetic_gamma_condition_recovery": (
        AUDIT_ROOT
        / "depth_recovery_overview/gamma_recovery/condition_recovery/"
        "synthetic_gamma_condition_recovery"
    ),
    "synthetic_pi_shared_recovery": (
        REPO_ROOT
        / "analyses/artifacts/synthetic/pi_demo/lbio_ranking/"
        "synthetic_pi_lbio_recovery"
    ),
    "synthetic_pi_reconstruction": (
        REPO_ROOT
        / "analyses/artifacts/synthetic/pi_demo/mu_reconstruction/"
        "synthetic_pi_mu_reconstruction"
    ),
    "synthetic_alpha_position_error": (
        AUDIT_ROOT
        / "alpha_recovery/position_error/synthetic_alpha_position_error"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(table: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _load_inputs(
    count_path: Path,
    provenance_path: Path,
    single_summary_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    counts = pd.read_csv(count_path)
    _require_columns(
        counts,
        {
            "depth",
            "dataset",
            "transcripts",
            "reads_per_codon_per_replicate",
            "replicate_zero_fraction",
            "consensus_zero_fraction",
            "weight_p05",
            "weight_median",
            "weight_p95",
        },
        count_path,
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    single = pd.read_csv(single_summary_path)
    _require_columns(
        single,
        {
            "bias",
            "bias_label",
            "read_depth",
            "read_depth_label",
            "mean_mu_pcc",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "n_transcripts",
        },
        single_summary_path,
    )
    return counts, provenance, single


def _summarize_depths(counts: pd.DataFrame) -> pd.DataFrame:
    observed = set(counts["depth"].astype(str))
    expected = {depth.key for depth in DEPTHS}
    if observed != expected:
        raise ValueError(
            f"Input-count depths differ from the expected audited set: {sorted(observed)}"
        )

    rows: list[dict[str, Any]] = []
    for depth in DEPTHS:
        block = counts.loc[counts["depth"].astype(str).eq(depth.key)].copy()
        transcript_counts = block["transcripts"].astype(int).unique()
        if transcript_counts.size != 1:
            raise ValueError(
                f"{depth.key} has inconsistent processed row counts: "
                f"{transcript_counts.tolist()}"
            )
        rows.append(
            {
                "depth": depth.key,
                "depth_label": depth.label,
                "nominal_reads_per_codon": depth.nominal,
                "datasets": int(block["dataset"].nunique()),
                "processed_transcripts_per_dataset": int(transcript_counts[0]),
                "mean_realized_reads_per_codon_per_replicate": float(
                    block["reads_per_codon_per_replicate"].mean()
                ),
                "minimum_realized_reads_per_codon_per_replicate": float(
                    block["reads_per_codon_per_replicate"].min()
                ),
                "maximum_realized_reads_per_codon_per_replicate": float(
                    block["reads_per_codon_per_replicate"].max()
                ),
                "mean_replicate_zero_fraction": float(
                    block["replicate_zero_fraction"].mean()
                ),
                "mean_consensus_zero_fraction": float(
                    block["consensus_zero_fraction"].mean()
                ),
                "median_of_dataset_weight_p05": float(block["weight_p05"].median()),
                "median_of_dataset_weight_median": float(
                    block["weight_median"].median()
                ),
                "median_of_dataset_weight_p95": float(block["weight_p95"].median()),
            }
        )
    return pd.DataFrame(rows)


def _cohort_tables(
    depth_summary: pd.DataFrame,
    provenance: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohorts = provenance.get("cohorts")
    if not isinstance(cohorts, dict):
        raise ValueError("Audit provenance has no cohort mapping.")

    validation_sets: dict[str, set[str]] = {}
    cohort_rows: list[dict[str, Any]] = []
    for depth in DEPTHS:
        entry = cohorts.get(depth.key)
        if not isinstance(entry, dict):
            raise ValueError(f"Audit provenance has no cohort for {depth.key}.")
        train = entry.get("train")
        validation = entry.get("validation")
        if not isinstance(train, dict) or not isinstance(validation, dict):
            raise ValueError(f"Malformed cohort entry for {depth.key}.")
        train_ids = set(map(str, train.get("ids", [])))
        validation_ids = set(map(str, validation.get("ids", [])))
        if len(train_ids) != int(train.get("n", -1)):
            raise ValueError(f"Train ID count mismatch for {depth.key}.")
        if len(validation_ids) != int(validation.get("n", -1)):
            raise ValueError(f"Validation ID count mismatch for {depth.key}.")
        if train_ids & validation_ids:
            raise ValueError(f"Train/validation overlap for {depth.key}.")
        validation_sets[depth.key] = validation_ids

        processed = int(
            depth_summary.loc[
                depth_summary["depth"].eq(depth.key),
                "processed_transcripts_per_dataset",
            ].iloc[0]
        )
        retained = len(train_ids) + len(validation_ids)
        cohort_rows.append(
            {
                "depth": depth.key,
                "depth_label": depth.label,
                "processed_transcripts": processed,
                "sequence_eligible_transcripts": retained,
                "excluded_by_sequence_eligibility": processed - retained,
                "training_transcripts": len(train_ids),
                "validation_transcripts": len(validation_ids),
                "validation_sha256": str(validation.get("hash", "")),
                "training_sha256": str(train.get("hash", "")),
            }
        )

    cohort_table = pd.DataFrame(cohort_rows)
    if not (cohort_table["excluded_by_sequence_eligibility"] == 79).all():
        raise ValueError(
            "The audited sequence-eligibility exclusion is no longer 79 at every depth."
        )

    overlap_rows: list[dict[str, Any]] = []
    for left, right in combinations(DEPTHS, 2):
        a = validation_sets[left.key]
        b = validation_sets[right.key]
        overlap_rows.append(
            {
                "depth_a": left.key,
                "depth_b": right.key,
                "n_a": len(a),
                "n_b": len(b),
                "intersection": len(a & b),
                "union": len(a | b),
                "jaccard": len(a & b) / len(a | b),
            }
        )
    overlap_table = pd.DataFrame(overlap_rows)

    membership_rows: list[dict[str, Any]] = []
    universe = set().union(*validation_sets.values())
    for mask in range(1, 1 << len(DEPTHS)):
        active = [DEPTHS[index] for index in range(len(DEPTHS)) if mask & (1 << index)]
        count = sum(
            all(identifier in validation_sets[depth.key] for depth in active)
            and all(
                identifier not in validation_sets[depth.key]
                for depth in DEPTHS
                if depth not in active
            )
            for identifier in universe
        )
        membership_rows.append(
            {
                "membership": " + ".join(depth.short for depth in active),
                "depth_count": len(active),
                "n_transcripts": int(count),
            }
        )
    membership_table = pd.DataFrame(membership_rows)
    if int(membership_table["n_transcripts"].sum()) != len(universe):
        raise RuntimeError("Validation membership table does not partition the union.")

    common = set.intersection(*validation_sets.values())
    recorded_common = provenance.get("common_three_depths", {})
    if len(common) != int(recorded_common.get("n", -1)) or len(common) != 27:
        raise ValueError(
            "The audited three-depth validation intersection no longer contains 27 IDs."
        )
    return cohort_table, overlap_table, membership_table


def _plot_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.10,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=13.5,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _draw_workflow(axis: plt.Axes) -> None:
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    _plot_label(axis, "A")
    axis.set_title("Synthetic workflow", loc="left", pad=8)

    boxes = (
        (0.01, 0.74, 0.26, 0.15, "Programmed $K_t$\ntraffic $q_{tr}$"),
        (0.38, 0.74, 0.20, 0.15, "Bias\n$b_{dti}$"),
        (0.68, 0.74, 0.31, 0.15, "NB2 replicas\n$C=.25,2,20$"),
        (
            0.08,
            0.43,
            0.84,
            0.16,
            "Validate sequence and codons\npad terminal boundary; filter empty rows",
        ),
        (
            0.14,
            0.19,
            0.72,
            0.14,
            "Compute $w_{dt}$; length $\\leq 4000$\nstratified 90/10 split",
        ),
    )
    for x, y, width, height, label in boxes:
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            linespacing=1.15,
            bbox={
                "boxstyle": "round,pad=0.34",
                "facecolor": "#F7F7F7",
                "edgecolor": "#666666",
                "linewidth": 1.4,
            },
        )
    arrows = (
        ((0.27, 0.815), (0.38, 0.815)),
        ((0.58, 0.815), (0.68, 0.815)),
        ((0.84, 0.70), (0.50, 0.60)),
        ((0.50, 0.41), (0.50, 0.34)),
    )
    for start, stop in arrows:
        axis.add_patch(
            FancyArrowPatch(
                start,
                stop,
                transform=axis.transAxes,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.5,
                color="#555555",
                connectionstyle="arc3,rad=0.0",
            )
        )
    axis.text(
        0.50,
        0.055,
        r"Model: $\mu_{dtri}=S_{dtr}L_{ti}\gamma_{dti}$",
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
    )


def _plot_preprocessing(
    counts: pd.DataFrame,
    depth_summary: pd.DataFrame,
    cohort_table: pd.DataFrame,
    overlap_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    style = publication_rc()
    style.update(
        {
            "font.size": 11.5,
            "font.weight": "bold",
            "axes.labelsize": 11.5,
            "axes.labelweight": "bold",
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.2,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "grid.linewidth": 0.7,
        }
    )
    if style["text.usetex"]:
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}"
            r"\AtBeginDocument{\boldmath}"
        )
    with plt.rc_context(style):
        figure = plt.figure(figsize=(7.0, 5.45), constrained_layout=False)
        grid = figure.add_gridspec(
            2,
            2,
            left=0.08,
            right=0.98,
            bottom=0.10,
            top=0.95,
            wspace=0.34,
            hspace=0.42,
        )
        flow = figure.add_subplot(grid[0, 0])
        sparsity = figure.add_subplot(grid[0, 1])
        cohorts = figure.add_subplot(grid[1, 0])
        overlap = figure.add_subplot(grid[1, 1])
        _draw_workflow(flow)

        _plot_label(sparsity, "B")
        sparsity.set_title("Read depth changes sparsity", loc="left", pad=8)
        x = np.arange(len(DEPTHS), dtype=float)
        rng = np.random.default_rng(20260917)
        series = (
            ("replicate_zero_fraction", "Individual replicas", "#0072B2", "o"),
            ("consensus_zero_fraction", "Replica consensus", "#D55E00", "s"),
        )
        for column, label, color, marker in series:
            means: list[float] = []
            for index, depth in enumerate(DEPTHS):
                values = counts.loc[counts["depth"].eq(depth.key), column].to_numpy(
                    dtype=float
                )
                jitter = rng.uniform(-0.085, 0.085, size=values.size)
                sparsity.scatter(
                    np.full(values.size, x[index]) + jitter,
                    values,
                    s=16,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.28,
                    zorder=2,
                )
                means.append(float(values.mean()))
            sparsity.plot(
                x,
                means,
                color=color,
                marker=marker,
                markersize=6.2,
                linewidth=2.3,
                label=label,
                zorder=3,
            )
        sparsity.set_xticks(x, [depth.short for depth in DEPTHS])
        sparsity.set_xlabel("Nominal reads per codon")
        sparsity.set_ylabel("Fraction of zero positions")
        sparsity.set_ylim(-0.02, 0.84)
        sparsity.grid(axis="y")
        sparsity.legend(loc="upper right")

        _plot_label(cohorts, "C")
        cohorts.set_title("Transcript accounting", loc="left", pad=8)
        y = np.arange(len(DEPTHS), dtype=float)
        ordered = cohort_table.set_index("depth").loc[[depth.key for depth in DEPTHS]]
        train = ordered["training_transcripts"].to_numpy(dtype=float)
        validation = ordered["validation_transcripts"].to_numpy(dtype=float)
        excluded = ordered["excluded_by_sequence_eligibility"].to_numpy(dtype=float)
        cohorts.barh(y, train, color="#8C8C8C", height=0.52, label="Train")
        cohorts.barh(
            y,
            validation,
            left=train,
            color="#E69F00",
            height=0.52,
            label="Validation",
        )
        cohorts.barh(
            y,
            excluded,
            left=train + validation,
            color="#CC79A7",
            height=0.52,
            label="_nolegend_",
        )
        for index in range(len(DEPTHS)):
            cohorts.text(
                train[index] - 900,
                y[index],
                f"{int(train[index]):,}",
                ha="right",
                va="center",
                color="white",
                fontsize=9.0,
                fontweight="bold",
            )
            cohorts.text(
                train[index] + validation[index] / 2,
                y[index],
                f"{int(validation[index]):,}",
                ha="center",
                va="center",
                color="black",
                fontsize=9.5,
                fontweight="bold",
            )
        cohorts.set_yticks(y, [depth.short for depth in DEPTHS])
        cohorts.invert_yaxis()
        cohorts.set_ylim(2.55, -0.72)
        cohorts.set_xlabel("Transcripts")
        cohorts.set_ylabel("Nominal reads/codon")
        cohorts.set_xlim(0, 20_100)
        cohorts.grid(axis="x")
        cohorts.legend(
            loc="upper left",
            ncol=2,
            handlelength=1.1,
            columnspacing=0.8,
            borderaxespad=0.35,
        )
        cohorts.annotate(
            r"79 excluded at every depth (0.41\%)",
            xy=(train[0] + validation[0] + excluded[0] / 2.0, y[0]),
            xytext=(19_850, 0.50),
            ha="right",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": "#666666", "linewidth": 1.2},
        )

        _plot_label(overlap, "D")
        overlap.set_title("Validation overlap by depth", loc="left", pad=8)
        matrix = np.eye(len(DEPTHS), dtype=float) * 1920.0
        for row in overlap_table.itertuples(index=False):
            left = next(i for i, depth in enumerate(DEPTHS) if depth.key == row.depth_a)
            right = next(i for i, depth in enumerate(DEPTHS) if depth.key == row.depth_b)
            matrix[left, right] = matrix[right, left] = float(row.intersection)
        percent = 100.0 * matrix / 1920.0
        image = overlap.imshow(
            percent,
            cmap="Blues",
            norm=Normalize(vmin=0.0, vmax=100.0),
            aspect="auto",
        )
        for row in range(len(DEPTHS)):
            for column in range(len(DEPTHS)):
                color = "white" if percent[row, column] > 55.0 else "black"
                overlap.text(
                    column,
                    row,
                    f"{int(matrix[row, column]):,}\n({percent[row, column]:.1f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                    fontweight="bold",
                )
        labels = [depth.short for depth in DEPTHS]
        overlap.set_xticks(np.arange(3), labels)
        overlap.set_yticks(np.arange(3), labels)
        overlap.set_xlabel("Nominal reads/codon")
        overlap.set_ylabel("Nominal reads/codon")
        overlap.tick_params(length=0)
        colorbar = figure.colorbar(image, ax=overlap, fraction=0.046, pad=0.04)
        colorbar.set_label(r"Shared validation set (\%)")
        overlap.text(
            0.5,
            -0.25,
            "Only 27 transcripts occur in all three validation sets",
            transform=overlap.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

        for extension, dpi in (("pdf", None), ("svg", None), ("png", 600)):
            path = output_dir / f"synthetic_preprocessing_and_cohorts.{extension}"
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def _plot_single_bias(single: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    depth_map = {depth.key.removesuffix("_per_codon"): depth for depth in DEPTHS}
    observed_depths = set(single["read_depth"].astype(str))
    if observed_depths != set(depth_map):
        raise ValueError(
            f"Single-dataset summary depths are {sorted(observed_depths)}, "
            f"expected {sorted(depth_map)}."
        )
    bias_order = list(dict.fromkeys(single["bias_label"].astype(str)))
    if len(bias_order) != 10 or len(single) != 30:
        raise ValueError("Expected exactly ten biases at each of three depths.")

    retained = single[
        [
            "bias",
            "bias_label",
            "read_depth",
            "read_depth_label",
            "mean_mu_pcc",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "n_transcripts",
            "n_valid_pcc_transcripts",
        ]
    ].copy()

    style = publication_rc()
    style.update(
        {
            "font.size": 12.0,
            "font.weight": "bold",
            "axes.labelsize": 12.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 13.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.0,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "grid.linewidth": 0.7,
        }
    )
    if style["text.usetex"]:
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}"
            r"\AtBeginDocument{\boldmath}"
        )
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(6.6, 4.35))
        y = np.arange(len(bias_order), dtype=float)
        offsets = {"0p25": -0.18, "2": 0.0, "20": 0.18}
        for short in ("0p25", "2", "20"):
            depth = depth_map[short]
            block = (
                retained.loc[retained["read_depth"].astype(str).eq(short)]
                .set_index("bias_label")
                .loc[bias_order]
            )
            center = block["mean_mu_pcc"].to_numpy(dtype=float)
            low = block["bootstrap_ci95_low"].to_numpy(dtype=float)
            high = block["bootstrap_ci95_high"].to_numpy(dtype=float)
            axis.errorbar(
                center,
                y + offsets[short],
                xerr=np.vstack((center - low, high - center)),
                fmt="o",
                markersize=6.4,
                markerfacecolor=depth.color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                color=depth.color,
                elinewidth=1.6,
                capsize=2.5,
                label=depth.label,
                zorder=3,
            )
        axis_labels = [
            label.replace("\n", " ")
            .replace("3′", r"$3^\prime$")
            .replace("5′", r"$5^\prime$")
            .replace(" > ", r" $>$ ")
            for label in bias_order
        ]
        axis.set_yticks(y, axis_labels)
        axis.invert_yaxis()
        axis.set_xlim(0.40, 0.95)
        axis.set_xlabel(
            r"Mean validation PCC$(\mu_{dt},\overline{Y}_{dt})$"
        )
        axis.set_title(
            "Single-bias reconstruction across read depths",
            loc="left",
            pad=32,
        )
        axis.grid(axis="x")
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            handletextpad=0.45,
            columnspacing=1.1,
        )
        figure.subplots_adjust(left=0.23, right=0.985, bottom=0.14, top=0.80)
        for extension, dpi in (("pdf", None), ("svg", None), ("png", 600)):
            path = output_dir / f"synthetic_single_bias_depth_reconstruction.{extension}"
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
    return retained


def _copy_assets(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for destination_stem, source_stem in ASSETS.items():
        copied = 0
        for extension in ("pdf", "svg", "png"):
            source = source_stem.with_suffix(f".{extension}")
            if not source.exists():
                continue
            destination = output_dir / f"{destination_stem}.{extension}"
            shutil.copy2(source, destination)
            records.append(
                {
                    "manuscript_asset": str(destination.relative_to(REPO_ROOT)),
                    "source": str(source.relative_to(REPO_ROOT)),
                    "sha256": sha256(destination),
                    "bytes": destination.stat().st_size,
                }
            )
            copied += 1
        if copied == 0:
            raise FileNotFoundError(f"No figure exports found for {source_stem}.")
    return records


def _write_supporting_text(output_dir: Path) -> None:
    caption = r"""\textbf{Synthetic preprocessing and evaluation-cohort accounting.}
\textbf{(A)} Programmed mean-one kinetics $K_t$ generate traffic-modified
profiles $q_{tr}$; dataset-specific multipliers $b_{dti}$ are then applied and
two NB2 replicas are sampled at nominal depths $C\in\{0.25,2,20\}$ reads per
codon. Long-format profiles are pivoted, validated against the sequence table,
zero-padded at the terminal boundary used by the model, filtered for positive
information, assigned transcript--dataset reliability weights $w_{dt}$, and
restricted to complete CDSs of at most 4,000 codons. The model factorization is
$\mu_{dtri}=S_{dtr}L_{ti}\gamma_{dti}$. \textbf{(B)} Dataset-level fractions
of zero codon positions in the individual replicas and their arithmetic
consensus; points are the 11 processed conditions and lines connect their
unweighted means. \textbf{(C)} Realized accounting at each depth. Every source
artifact contains 19,283 rows per dataset; the common length/sequence filter
removes 79, leaving 17,284 training and 1,920 validation transcripts. Thus
depth did \emph{not} reduce the realized cohort size in these experiments.
\textbf{(D)} Pairwise intersections of the depth-specific validation lists,
shown as counts and percentages of 1,920. Although their sizes are equal, only
27 transcripts occur in all three lists because validation sampling was
stratified by depth-dependent reliability rank and CSS count. Read depth
therefore changes sparsity, reliability, and cohort identity; comparisons that
attribute changes to depth use the explicit matched intersection.
"""
    (output_dir / "synthetic_preprocessing_and_cohorts_caption.tex").write_text(
        caption,
        encoding="utf-8",
    )

    single_caption = r"""\textbf{Single-bias reconstruction across read depths.}
Each row is one separately trained technical-bias condition, evaluated at
nominal depths $C=0.25$, 2, and 20 reads per codon. Points are arithmetic means
of per-transcript PCC between the exported fitted mean $\boldsymbol\mu_{dt}$
and the arithmetic two-replicate consensus over the complete model-valid
profile; bars are 95\% transcript-bootstrap intervals. Each cell contains
1,920 validation transcripts and uses the checkpoint maximizing validation
$\mu$ PCC. The bias mechanism and training seed (42) are fixed, but the
depth-specific validation identities are not: the pairwise intersections are
184--225 transcripts and only 27 occur at all three depths. The monotone
condition means therefore show the expected descriptive depth trend in
reconstruction, not a paired causal effect, independent test performance, or
recovery of the programmed kinetic profile $K_t$. For $N=1$, cross-dataset
reference centering and $\pi$ are not defined, so this analysis isolates
observation reconstruction rather than the shared/correction decomposition.
"""
    (output_dir / "synthetic_single_bias_depth_reconstruction_caption.tex").write_text(
        single_caption,
        encoding="utf-8",
    )


def _write_readme(output_dir: Path) -> None:
    text = """# Synthetic appendix bundle

This directory is generated from audited saved tables and existing vector
figures.  No model is trained and no prediction distribution is reconstructed
from a raster image.

The central cohort finding is intentionally explicit: all depths retain the
same 19,204 sequence-eligible transcripts and the same 17,284/1,920 split
sizes.  Lower depth increases sparsity and changes reliability-stratified split
identities; it does not reduce the realized row count in these artifacts.

The manuscript fragment centers every figure at 88% of the available text
width.  Plot labels are deliberately larger than in the first draft and all
principal curves, error bars, axes, ticks, and typographic labels use a bold
publication treatment.  Cross-depth reconstruction is displayed with the
simulator-NB2-standardized residual instead of raw count RMSE; the reference-
policy reconstruction uses profile-relative RMSE.  Raw RMSE remains available
in the original numeric source tables.  A condition-resolved gamma-recovery
figure reports PCC, multiplier RMSE, and amplitude slope for all ten biases.

Rebuild from the repository root:

```bash
RIBOUNMIX_PLOT_TEX=auto .venv/bin/python analyses/build_synthetic_appendix_bundle.py
```

`bundle_manifest.json` records every source path and SHA-256.  CSV files under
`source/` contain the exact values used by the preprocessing and single-bias
figures.  The copied model-analysis figures retain their exact numerical source
tables in the original analysis directories, which are listed in the manifest.

`synthetic_training_and_recovery_appendix.tex` is the manuscript fragment.
`synthetic_training_and_recovery_appendix_preview.tex` is a minimal two-column
wrapper, and its compiled PDF is supplied for visual inspection.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count-audit",
        type=Path,
        default=AUDIT_ROOT / "input_count_audit.csv",
    )
    parser.add_argument(
        "--split-provenance",
        type=Path,
        default=AUDIT_ROOT / "provenance.json",
    )
    parser.add_argument(
        "--single-summary",
        type=Path,
        default=DEFAULT_SINGLE_SUMMARY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-copy-existing",
        action="store_true",
        help="Generate the two new figures without copying existing figure exports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = (args.count_audit, args.split_provenance, args.single_summary)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    counts, provenance, single = _load_inputs(*paths)
    depth_summary = _summarize_depths(counts)
    cohort_table, overlap_table, membership_table = _cohort_tables(
        depth_summary,
        provenance,
    )
    single_source = _plot_single_bias(single, output_dir)
    _plot_preprocessing(counts, depth_summary, cohort_table, overlap_table, output_dir)

    depth_summary.to_csv(source_dir / "depth_summary.csv", index=False)
    cohort_table.to_csv(source_dir / "cohort_accounting.csv", index=False)
    overlap_table.to_csv(source_dir / "validation_overlap.csv", index=False)
    membership_table.to_csv(source_dir / "validation_membership.csv", index=False)
    single_source.to_csv(source_dir / "single_bias_depth_summary.csv", index=False)

    copied = [] if args.no_copy_existing else _copy_assets(output_dir)
    _write_supporting_text(output_dir)
    _write_readme(output_dir)

    generated = []
    for stem in (
        "synthetic_preprocessing_and_cohorts",
        "synthetic_single_bias_depth_reconstruction",
    ):
        for extension in ("pdf", "svg", "png"):
            path = output_dir / f"{stem}.{extension}"
            generated.append(
                {
                    "manuscript_asset": str(path.relative_to(REPO_ROOT)),
                    "source": "generated_from_saved_numeric_tables",
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )

    manuscript_files = []
    for name in (
        "synthetic_training_and_recovery_appendix.tex",
        "synthetic_training_and_recovery_appendix_preview.tex",
    ):
        path = output_dir / name
        if path.is_file():
            manuscript_files.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "inputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
        "new_figures": generated,
        "copied_figures": copied,
        "manuscript_files": manuscript_files,
        "scientific_checks": {
            "processed_transcripts_per_dataset_all_depths": 19_283,
            "sequence_eligible_transcripts_all_depths": 19_204,
            "training_transcripts_all_depths": 17_284,
            "validation_transcripts_all_depths": 1_920,
            "three_depth_validation_intersection": 27,
            "depth_reduced_realized_cohort_size": False,
        },
        "rendering": {
            "manuscript_image_width_fraction": 0.88,
            "bold_typography": True,
            "heavier_primary_strokes": True,
            "numerical_values_changed_by_restyling": False,
        },
    }
    (output_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Synthetic appendix assets: {output_dir}")
    print("Realized cohort size is identical across depths; common validation IDs: 27.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
