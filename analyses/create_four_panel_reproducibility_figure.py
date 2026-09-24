#!/usr/bin/env python3
"""Build the primary publication figure for four-panel profile reproducibility.

This script reads the verified, saved ``common_test_L_profiles.parquet``
artifacts produced from each panel's best-validation-loss checkpoint.  It does
not reconstruct distributions from summary statistics, digitize a prior
figure, smooth profiles, or renormalize profiles after loading them.

The output contains the compact two-panel main figure, all plotted source tables,
the deterministic example-selection table, a LaTeX figure snippet/caption,
and the exact regeneration command.  Pooled histograms, median heatmaps, and
peak/boundary/residual diagnostics are intentionally not included in this
primary figure; they belong in supplementary figures.

Example
-------
::

    .venv/bin/python analyses/create_four_panel_reproducibility_figure.py \\
        --run-root results/my_panels_a100_b32_20260906_114323
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
from matplotlib import pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.publication_plot_style import LATEX_PAPER_RC
from analyses.paths import artifact_directory

DEFAULT_RUN_NAME = "my_panels_qrank_a100_b32_20260908_103510"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "results" / DEFAULT_RUN_NAME
DEFAULT_OUTPUT_DIR = artifact_directory(
    "real_data", DEFAULT_RUN_ROOT, "publication_figure"
)
PANEL_NAMES = tuple(f"panel_{index:02d}" for index in range(1, 5))
PANEL_LABELS = {
    "panel_01": "Panel 1",
    "panel_02": "Panel 2",
    "panel_03": "Panel 3",
    "panel_04": "Panel 4",
}
PANEL_COLORS = {
    "panel_01": "#0072B2",
    "panel_02": "#E69F00",
    "panel_03": "#009E73",
    "panel_04": "#CC79A7",
}


# Render all figure text through the local TeX installation.  Latin Modern is
# the modernized Computer Modern family used by LaTeX; this avoids Matplotlib
# font substitution and embeds the actual TeX-rendered glyphs in the PDF.
PAPER_RC = LATEX_PAPER_RC


@dataclass(frozen=True)
class PanelProfiles:
    """One panel's original, saved mean-one shared profiles."""

    panel: str
    artifact_path: Path
    scientific_manifest_path: Path
    checkpoint_variant: str
    prediction_split: str
    profiles: dict[str, np.ndarray]
    lengths: dict[str, int]
    maximum_abs_mean_one_deviation: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=(
            "Completed four-panel run root (default: "
            f"results/{DEFAULT_RUN_NAME})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-root>/analysis/publication_figure).",
    )
    parser.add_argument(
        "--mean-one-tolerance",
        type=float,
        default=1.0e-4,
        help="Maximum accepted |mean(L_t)-1| before failing (default: 1e-4).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution in dots per inch (default: 300).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png"),
        default=("pdf", "png"),
        help="Main-figure formats to export (default: pdf png).",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


def _require_columns(frame: pd.DataFrame, names: Iterable[str], path: Path) -> None:
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing columns: {', '.join(missing)}")


def _profile_artifact(panel_directory: Path) -> Path:
    candidates = sorted(panel_directory.rglob("common_test_L_profiles.parquet"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"{panel_directory.name}: expected exactly one saved compact "
            f"common-test L-profile artifact, found {len(candidates)}: {candidates}"
        )
    return candidates[0].resolve()


def _load_panel_profiles(
    panel: str,
    panel_directory: Path,
    expected_ids: set[str],
    mean_one_tolerance: float,
) -> PanelProfiles:
    """Load and validate one panel's unmodified saved shared-profile arrays."""
    scientific_manifest_path = panel_directory / "scientific_checkpoint_manifest.json"
    scientific = _read_json(scientific_manifest_path)
    if scientific.get("checkpoint_variant") != "best_val_loss":
        raise ValueError(
            f"{panel}: required best_val_loss checkpoint, found "
            f"{scientific.get('checkpoint_variant')!r}."
        )
    if scientific.get("prediction_split") != "common_test":
        raise ValueError(
            f"{panel}: required common_test prediction split, found "
            f"{scientific.get('prediction_split')!r}."
        )

    artifact_path = _profile_artifact(panel_directory)
    frame = pd.read_parquet(
        artifact_path,
        columns=["transcript_id", "transcript_length", "L_t"],
    )
    _require_columns(frame, ("transcript_id", "transcript_length", "L_t"), artifact_path)
    frame["transcript_id"] = frame["transcript_id"].astype(str)
    if frame["transcript_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["transcript_id"].duplicated(keep=False), "transcript_id"]
            .unique()
            .tolist()
        )
        raise ValueError(f"{panel}: duplicate compact-profile IDs: {duplicates[:10]}")

    observed_ids = set(frame["transcript_id"])
    if observed_ids != expected_ids:
        raise ValueError(
            f"{panel}: saved profile IDs do not equal the common held-out set; "
            f"missing={sorted(expected_ids - observed_ids)[:5]}, "
            f"extra={sorted(observed_ids - expected_ids)[:5]}."
        )

    profiles: dict[str, np.ndarray] = {}
    lengths: dict[str, int] = {}
    maximum_deviation = 0.0
    for row in frame.itertuples(index=False):
        transcript_id = str(row.transcript_id)
        values = np.asarray(row.L_t, dtype=np.float64).reshape(-1)
        length = int(row.transcript_length)
        if values.size != length or length < 2:
            raise ValueError(
                f"{panel}/{transcript_id}: L_t has length {values.size}; expected {length}."
            )
        if not np.isfinite(values).all() or bool((values <= 0.0).any()):
            raise ValueError(f"{panel}/{transcript_id}: L_t is non-finite or non-positive.")
        deviation = abs(float(values.mean()) - 1.0)
        if deviation > mean_one_tolerance:
            raise ValueError(
                f"{panel}/{transcript_id}: mean(L_t) differs from one by {deviation:.3e}; "
                "the figure must not post-hoc normalize this profile."
            )
        profiles[transcript_id] = values
        lengths[transcript_id] = length
        maximum_deviation = max(maximum_deviation, deviation)

    return PanelProfiles(
        panel=panel,
        artifact_path=artifact_path,
        scientific_manifest_path=scientific_manifest_path,
        checkpoint_variant=str(scientific["checkpoint_variant"]),
        prediction_split=str(scientific["prediction_split"]),
        profiles=profiles,
        lengths=lengths,
        maximum_abs_mean_one_deviation=maximum_deviation,
    )


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation of aligned complete-CDS arrays, with no masking added."""
    if left.shape != right.shape or left.size < 2:
        return float("nan")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def _pair_code(panel_a: str, panel_b: str) -> str:
    return f"P{int(panel_a[-2:])}\N{EN DASH}P{int(panel_b[-2:])}"


def calculate_pairwise_pcc(
    panels: dict[str, PanelProfiles],
    common_ids: set[str],
) -> pd.DataFrame:
    """Calculate all six per-transcript PCC distributions from saved arrays."""
    rows: list[dict[str, Any]] = []
    for panel_a, panel_b in itertools.combinations(PANEL_NAMES, 2):
        left = panels[panel_a]
        right = panels[panel_b]
        pair = _pair_code(panel_a, panel_b)
        for transcript_id in sorted(common_ids):
            left_values = left.profiles[transcript_id]
            right_values = right.profiles[transcript_id]
            length = left.lengths[transcript_id]
            if length != right.lengths[transcript_id] or left_values.shape != right_values.shape:
                raise ValueError(
                    f"{pair}/{transcript_id}: panel profiles have inconsistent CDS lengths."
                )
            rows.append(
                {
                    "transcript_id": transcript_id,
                    "panel_a": panel_a,
                    "panel_b": panel_b,
                    "panel_pair": pair,
                    "transcript_length": length,
                    "PCC": _pearson(left_values, right_values),
                    "profile_domain": "complete_CDS",
                }
            )
    agreement = pd.DataFrame(rows)
    expected_rows = len(common_ids) * math.comb(len(PANEL_NAMES), 2)
    if len(agreement) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} PCC rows, found {len(agreement)}.")
    return agreement


def calculate_transcript_medians(agreement: pd.DataFrame) -> pd.DataFrame:
    """Compute m_t, the median PCC across the six panel pairs, per transcript."""
    pair_count = math.comb(len(PANEL_NAMES), 2)
    pivot = agreement.pivot(index="transcript_id", columns="panel_pair", values="PCC")
    if pivot.shape[1] != pair_count:
        raise ValueError(f"Expected {pair_count} panel-pair PCC columns, found {pivot.shape[1]}.")
    finite_counts = pivot.notna().sum(axis=1)
    usable = pivot.loc[finite_counts == pair_count].copy()
    if usable.empty:
        raise RuntimeError("No transcript has all six finite panel-pair PCC values.")
    lengths = agreement.drop_duplicates("transcript_id").set_index("transcript_id")[
        "transcript_length"
    ]
    output = pd.DataFrame(
        {
            "transcript_id": usable.index.astype(str),
            "m_t": usable.median(axis=1).to_numpy(dtype=np.float64),
            "usable_panel_pairs": pair_count,
            "transcript_length": lengths.loc[usable.index].to_numpy(dtype=int),
        }
    )
    return output.sort_values("transcript_id", kind="stable").reset_index(drop=True)


def select_examples(transcript_medians: pd.DataFrame) -> pd.DataFrame:
    """Select the sole profile example by the empirical median of m_t."""
    if transcript_medians.empty:
        raise ValueError("Cannot select examples from an empty transcript table.")
    specifications = (("B_typical_agreement", "B", "Typical-agreement transcript", 0.50),)
    rows: list[dict[str, Any]] = []
    for label, panel_letter, display_label, quantile in specifications:
        target = float(transcript_medians["m_t"].quantile(quantile))
        candidates = transcript_medians.assign(
            absolute_distance_to_empirical_quantile=(transcript_medians["m_t"] - target).abs()
        ).sort_values(
            ["absolute_distance_to_empirical_quantile", "transcript_id"],
            kind="stable",
        )
        chosen = candidates.iloc[0]
        rows.append(
            {
                "selection_label": label,
                "figure_panel": panel_letter,
                "display_label": display_label,
                "selection_quantile": quantile,
                "empirical_quantile_m_t": target,
                "transcript_id": str(chosen["transcript_id"]),
                "m_t": float(chosen["m_t"]),
                "absolute_distance_to_empirical_quantile": float(
                    chosen["absolute_distance_to_empirical_quantile"]
                ),
                "transcript_length": int(chosen["transcript_length"]),
                "usable_panel_pairs": int(chosen["usable_panel_pairs"]),
                "selection_rule": (
                    "Closest m_t to the empirical quantile across all transcripts; "
                    "ties broken lexicographically by transcript_id."
                ),
            }
        )
    return pd.DataFrame(rows)


def selected_profile_table(
    panels: dict[str, PanelProfiles], selections: pd.DataFrame
) -> pd.DataFrame:
    """Write every plotted, original full-CDS L_t value in tidy form."""
    rows: list[dict[str, Any]] = []
    for selection in selections.itertuples(index=False):
        for panel in PANEL_NAMES:
            values = panels[panel].profiles[str(selection.transcript_id)]
            for position, value in enumerate(values, start=1):
                rows.append(
                    {
                        "selection_label": selection.selection_label,
                        "figure_panel": selection.figure_panel,
                        "transcript_id": selection.transcript_id,
                        "panel": panel,
                        "codon_position": position,
                        "L_t": float(value),
                        "profile_domain": "complete_CDS",
                        "normalization_applied_after_loading": False,
                    }
                )
    return pd.DataFrame(rows)


def _support_limited_violin(
    axis: matplotlib.axes.Axes,
    values: np.ndarray,
    position: float,
    *,
    half_width: float = 0.36,
) -> None:
    """Draw a KDE violin only over the observed data range, never beyond it."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("A violin requires at least two finite observations.")
    low, high = float(values[0]), float(values[-1])
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1.0e-12):
        axis.plot(
            [position - half_width, position + half_width],
            [low, high],
            color="#4B5563",
            linewidth=1.0,
            zorder=1,
        )
        return
    support = np.linspace(low, high, 512)
    density = gaussian_kde(values, bw_method="scott")(support)
    scaled_width = half_width * density / float(density.max())
    axis.fill_betweenx(
        support,
        position - scaled_width,
        position + scaled_width,
        facecolor="#4C78A8",
        edgecolor="#315B7D",
        linewidth=0.55,
        alpha=0.34,
        zorder=1,
    )


def plot_pcc_distributions(axis: matplotlib.axes.Axes, agreement: pd.DataFrame) -> None:
    """Panel A: full actual distributions shown in a deliberately cropped view."""
    pair_order = [_pair_code(left, right) for left, right in itertools.combinations(PANEL_NAMES, 2)]
    distributions = [
        agreement.loc[agreement["panel_pair"] == pair, "PCC"].dropna().to_numpy(dtype=np.float64)
        for pair in pair_order
    ]
    if any(values.size == 0 for values in distributions):
        raise ValueError("At least one requested panel-pair PCC distribution is empty.")

    positions = np.arange(1, len(pair_order) + 1, dtype=float)
    for position, values in zip(positions, distributions):
        _support_limited_violin(axis, values, position)

    boxes = axis.boxplot(
        distributions,
        positions=positions,
        widths=0.20,
        whis=1.5,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#9B2226", "linewidth": 1.25},
        boxprops={"facecolor": "#F9FAFB", "edgecolor": "#303030", "linewidth": 0.7},
        whiskerprops={"color": "#303030", "linewidth": 0.7},
        capprops={"color": "#303030", "linewidth": 0.7},
    )
    for patch in boxes["boxes"]:
        patch.set_zorder(3)
    for element in ("medians", "whiskers", "caps"):
        for artist in boxes[element]:
            artist.set_zorder(4)

    # The user-facing manuscript view intentionally crops sparse low-PCC
    # tails; the complete, uncropped observations remain in the source table.
    axis.set_xlim(0.45, len(pair_order) + 0.55)
    axis.set_ylim(0.4, 1.0)
    axis.set_yticks(np.arange(0.4, 1.01, 0.1))
    # TeX's conventional ``--`` renders an en dash, while the source table
    # keeps the literal Unicode en dash used for the panel-pair identifier.
    axis.set_xticks(positions, [pair.replace("\N{EN DASH}", "--") for pair in pair_order])
    axis.tick_params(axis="x", labelrotation=28)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    axis.set_xlabel("Independent panel pair")
    axis.set_ylabel("Full-CDS per-transcript PCC")
    axis.grid(axis="y")
    axis.set_axisbelow(True)


def draw_pcc_header(axis: matplotlib.axes.Axes) -> None:
    """Draw the deliberately minimal Panel-A title."""
    axis.axis("off")
    axis.text(
        0.0,
        0.82,
        r"\textbf{A.} PCC agreement",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=12.0,
    )


def _profile_axis_limits(values: list[np.ndarray]) -> tuple[float, float]:
    minimum = float(min(np.min(value) for value in values))
    maximum = float(max(np.max(value) for value in values))
    span = maximum - minimum
    padding = max(0.05, 0.055 * span)
    return min(0.0, minimum - padding), maximum + padding


def plot_selected_transcript(
    axis: matplotlib.axes.Axes,
    selection: pd.Series,
    panels: dict[str, PanelProfiles],
) -> list[matplotlib.lines.Line2D]:
    """Panel B: direct overlay of all four original complete-CDS profiles."""
    transcript_id = str(selection["transcript_id"])
    arrays = [panels[panel].profiles[transcript_id] for panel in PANEL_NAMES]
    length = arrays[0].size
    if any(array.size != length for array in arrays):
        raise ValueError(f"{transcript_id}: selected profiles have inconsistent lengths.")

    handles: list[matplotlib.lines.Line2D] = []
    positions = np.arange(1, length + 1)
    for panel, values in zip(PANEL_NAMES, arrays):
        handle = axis.plot(
            positions,
            values,
            color=PANEL_COLORS[panel],
            linewidth=0.72,
            alpha=0.92,
            label=PANEL_LABELS[panel],
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )[0]
        handles.append(handle)
    axis.set_xlim(1, length)
    axis.set_ylim(*_profile_axis_limits(arrays))
    axis.set_xlabel("Codon position")
    axis.set_ylabel(r"Shared profile $L_t$ (mean one)")
    axis.grid(axis="both")
    axis.set_axisbelow(True)
    return handles


def draw_example_header(axis: matplotlib.axes.Axes, selection: pd.Series) -> None:
    """Place the compact deterministic-selection annotation above Panel B."""
    axis.axis("off")
    axis.text(
        0.0,
        0.82,
        r"\textbf{B.} Typical profile",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=12.0,
    )
    axis.text(
        0.0,
        0.43,
        f"{selection['transcript_id']}; $m_t$={float(selection['m_t']):.3f}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.0,
    )


def build_figure(
    agreement: pd.DataFrame,
    selections: pd.DataFrame,
    panels: dict[str, PanelProfiles],
):
    """Build one compact manuscript-width row: A left, B right."""
    with matplotlib.rc_context(PAPER_RC):
        figure = plt.figure(figsize=(7.15, 3.3), layout="constrained")
        grid = figure.add_gridspec(
            2,
            2,
            width_ratios=(1.0, 1.0),
            height_ratios=(0.48, 1.0),
            hspace=0.03,
            wspace=0.10,
        )
        pcc_header_axis = figure.add_subplot(grid[0, 0])
        typical_header_axis = figure.add_subplot(grid[0, 1])
        distribution_axis = figure.add_subplot(grid[1, 0])
        typical_axis = figure.add_subplot(grid[1, 1])

        plot_pcc_distributions(distribution_axis, agreement)
        selection_by_label = selections.set_index("selection_label")
        typical_selection = selection_by_label.loc["B_typical_agreement"]
        draw_pcc_header(pcc_header_axis)
        draw_example_header(typical_header_axis, typical_selection)
        handles = plot_selected_transcript(
            typical_axis, typical_selection, panels
        )
        typical_header_axis.legend(
            handles=handles,
            labels=[f"P{index}" for index in range(1, len(handles) + 1)],
            loc="lower center",
            bbox_to_anchor=(0.50, 0.0),
            ncol=4,
            columnspacing=0.70,
            handlelength=1.45,
            handletextpad=0.38,
            borderaxespad=0.0,
            fontsize=12.0,
            frameon=False,
        )
        return figure


def _hash_ids(identifiers: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(identifiers)).encode("utf-8")).hexdigest()


def write_source_tables(
    *,
    source_dir: Path,
    agreement: pd.DataFrame,
    transcript_medians: pd.DataFrame,
    selections: pd.DataFrame,
    selected_profiles: pd.DataFrame,
    panels: dict[str, PanelProfiles],
    common_ids: set[str],
) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    agreement.to_csv(source_dir / "panel_pair_pcc_values.csv", index=False)
    transcript_medians.to_csv(source_dir / "transcript_median_pcc.csv", index=False)
    selections.to_csv(source_dir / "example_selection.csv", index=False)
    selected_profiles.to_csv(source_dir / "selected_transcript_profiles.csv", index=False)
    provenance = pd.DataFrame(
        [
            {
                "panel": panel,
                "profile_artifact": str(record.artifact_path),
                "scientific_checkpoint_manifest": str(record.scientific_manifest_path),
                "checkpoint_variant": record.checkpoint_variant,
                "prediction_split": record.prediction_split,
                "common_heldout_transcript_count": len(common_ids),
                "common_heldout_transcript_id_hash": _hash_ids(common_ids),
                "maximum_abs_mean_one_deviation": record.maximum_abs_mean_one_deviation,
                "normalization_applied_after_loading": False,
            }
            for panel, record in panels.items()
        ]
    )
    provenance.to_csv(source_dir / "panel_prediction_provenance.csv", index=False)
    (source_dir / "README.md").write_text(
        "# Primary-figure source tables\n\n"
        "- `panel_pair_pcc_values.csv`: the six full-CDS PCC distributions, "
        "computed directly from saved complete-CDS `L_t` arrays. It retains all "
        "values, including observations below the 0.4 lower display limit in A.\n"
        "- `transcript_median_pcc.csv`: `m_t`, the median of the six PCC values "
        "for each transcript with all six usable pairs.\n"
        "- `example_selection.csv`: deterministic typical-agreement (B) selection result.\n"
        "- `selected_transcript_profiles.csv`: every unmodified plotted profile "
        "value, indexed by one-based CDS codon position.\n"
        "- `panel_prediction_provenance.csv`: verified prediction-artifact and "
        "common-held-out-set provenance.\n",
        encoding="utf-8",
    )


def latex_snippet(common_count: int) -> str:
    return rf"""% Requires \usepackage{{graphicx}}
\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{four_panel_reproducibility.pdf}}
  \caption{{\textbf{{Reproducibility of the inferred shared profile across four independent dataset panels.}}
  All comparisons use the same common held-out test set of {common_count:,} transcripts. \textbf{{A}}, full-CDS Pearson correlations between the original saved mean-one shared profiles $L_t$ for each of the six panel pairs. The visible PCC range is deliberately restricted to 0.4--1.0 for compact presentation; the complete uncropped distributions remain in the source table. Violin KDEs use all observations and are restricted to observed support; boxes show the interquartile range, red lines show medians, and whiskers extend to the most extreme values within $1.5\times$IQR. \textbf{{B}}, unmodified complete-CDS $L_t$ profiles from all four panels for the transcript selected solely as closest to the empirical 50th percentile of $m_t$, its median across six pairwise PCCs, with transcript-ID lexicographic tie-breaking. Pairwise comparisons are dependent because they share held-out transcripts and each fitted panel contributes to multiple pairs; the distributions are descriptive rather than independent replicates. PCC quantifies agreement between inferred profiles, not biological accuracy.}}
  \label{{fig:four-panel-profile-reproducibility}}
\end{{figure*}}
"""


def write_handoff_files(
    output_dir: Path,
    *,
    common_count: int,
    args: argparse.Namespace,
) -> None:
    (output_dir / "four_panel_reproducibility.tex").write_text(
        latex_snippet(common_count), encoding="utf-8"
    )
    run_root_arg = args.run_root
    output_dir_arg = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)
    )
    command = (
        ".venv/bin/python analyses/create_four_panel_reproducibility_figure.py \\\n"
        f"  --run-root {run_root_arg} \\\n"
        f"  --output-dir {output_dir_arg}\n"
    )
    (output_dir / "REGENERATION.md").write_text(
        "# Exact regeneration command\n\n```bash\n" + command + "```\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Two-panel reproducibility publication figure\n\n"
        "This directory contains the compact A/B main manuscript figure only. It deliberately "
        "excludes the pooled PCC histogram, median heatmap, and peak/boundary/residual "
        "diagnostics, which should remain supplementary figures.\n\n"
        "The primary figure was calculated directly from verified saved `common_test_L_profiles.parquet` "
        "artifacts selected from each panel's `best_val_loss` checkpoint. No profile was "
        "smoothed, rescaled, or reconstructed from summary statistics.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.mean_one_tolerance < 0.0:
        raise ValueError("--mean-one-tolerance must be non-negative.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else artifact_directory("real_data", run_root, "publication_figure")
    )
    common_split = _read_json(run_root / "common_split_manifest.json")
    common_ids = {str(value) for value in common_split.get("common_test_ids", [])}
    if not common_ids:
        raise ValueError("common_split_manifest.json has no common_test_ids.")

    analysis_manifest = _read_json(
        artifact_directory("real_data", run_root, "panel_convergence")
        / "analysis_manifest.json"
    )
    if not bool(analysis_manifest.get("analysis_complete_for_planned_panels")):
        raise ValueError("The existing analysis manifest does not certify all planned panels.")
    available_panels = tuple(analysis_manifest.get("panels", []))
    if available_panels != PANEL_NAMES:
        raise ValueError(
            f"Expected exactly {PANEL_NAMES} in the verified analysis manifest; "
            f"found {available_panels}."
        )

    panels = {
        panel: _load_panel_profiles(
            panel,
            run_root / panel,
            common_ids,
            args.mean_one_tolerance,
        )
        for panel in PANEL_NAMES
    }
    agreement = calculate_pairwise_pcc(panels, common_ids)
    if agreement["PCC"].isna().any():
        bad = int(agreement["PCC"].isna().sum())
        raise RuntimeError(f"Found {bad} unusable panel-pair PCC values.")
    transcript_medians = calculate_transcript_medians(agreement)
    selections = select_examples(transcript_medians)
    selected_profiles = selected_profile_table(panels, selections)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_source_tables(
        source_dir=output_dir / "source_tables",
        agreement=agreement,
        transcript_medians=transcript_medians,
        selections=selections,
        selected_profiles=selected_profiles,
        panels=panels,
        common_ids=common_ids,
    )
    outputs: list[Path] = []
    # ``usetex`` is evaluated at draw/save time, so the TeX rc context must
    # remain active through export rather than only while axes are constructed.
    with matplotlib.rc_context(PAPER_RC):
        figure = build_figure(agreement, selections, panels)
        for suffix in args.formats:
            output = output_dir / f"four_panel_reproducibility.{suffix}"
            save_kwargs = {"bbox_inches": "tight"}
            if suffix == "png":
                save_kwargs["dpi"] = args.dpi
            figure.savefig(output, **save_kwargs)
            outputs.append(output)
    plt.close(figure)
    write_handoff_files(output_dir, common_count=len(common_ids), args=args)

    print("Wrote four-panel reproducibility publication figure:")
    for output in outputs:
        print(output)
    print(f"Source tables: {output_dir / 'source_tables'}")
    print(f"Example selection: {output_dir / 'source_tables' / 'example_selection.csv'}")
    print(f"LaTeX snippet: {output_dir / 'four_panel_reproducibility.tex'}")
    print(f"Common held-out transcripts: {len(common_ids):,}; PCC rows: {len(agreement):,}.")


if __name__ == "__main__":
    main()
