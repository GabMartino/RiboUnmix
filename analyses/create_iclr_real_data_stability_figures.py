#!/usr/bin/env python3
"""Create the PCC-focused real-data stability figures for the ICLR appendix.

The script separates experimental design from model results:

1. panel rank balance and cross-panel transcript-level L_t PCC;
2. cumulative reference concentration and direction; and
3. cumulative L_t PCC to one common equal-reference N=2 anchor; and
4. a compact two-panel summary for the main text.

No result is reconstructed from a figure.  Every plotted value is read from a
frozen design table or a saved analysis table, and a source CSV is written next
to each figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.publication_plot_style import publication_rc


PANEL_ORDER = [f"panel_{index:02d}" for index in range(1, 5)]
PANEL_LABELS = {panel: f"P{index}" for index, panel in enumerate(PANEL_ORDER, 1)}
PANEL_COLORS = {
    "panel_01": "#0072B2",
    "panel_02": "#E69F00",
    "panel_03": "#009E73",
    "panel_04": "#CC79A7",
}

ARM_ORDER = [
    "equal",
    "ranked_p1",
    "reverse_p1",
    "ranked_p3",
    "reverse_p3",
]
CROSS_PANEL_LAYOUT = [
    "ranked_p1",
    "ranked_p3",
    "equal",
    "reverse_p1",
    "reverse_p3",
]
ARM_LABELS = {
    "equal": "Equal",
    "ranked_p1": r"Ranked $p=1$",
    "reverse_p1": r"Reversed $p=1$",
    "ranked_p3": r"Ranked $p=3$",
    "reverse_p3": r"Reversed $p=3$",
}
ARM_COLORS = {
    "equal": "#333333",
    "ranked_p1": "#0072B2",
    "reverse_p1": "#56B4E9",
    "ranked_p3": "#D55E00",
    "reverse_p3": "#E69F00",
}
ARM_LINESTYLES = {
    "equal": "--",
    "ranked_p1": "-",
    "reverse_p1": ":",
    "ranked_p3": "-",
    "reverse_p3": ":",
}
PAIR_COLORS = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--four-panel-root",
        type=Path,
        default=PROJECT_ROOT / "results/four_panel_stability_seed42",
    )
    parser.add_argument(
        "--cumulative-result-root",
        type=Path,
        default=PROJECT_ROOT / "results/cumulative_stability_seed42",
        help="Partial run containing validated prediction exports.",
    )
    parser.add_argument(
        "--cumulative-design-root",
        type=Path,
        default=PROJECT_ROOT / "results/cumulative_stability_fixed_cohort_seed42",
        help="Confirmatory design whose concentration is fixed before training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "figures/assets_5_real_datasets_4_panels",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def save_figure(fig: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_panel_assignment(four_panel_root: Path) -> pd.DataFrame:
    assignment_path = four_panel_root / "inputs/panel_assignment.csv"
    assignment = pd.read_csv(assignment_path)

    require_columns(
        assignment,
        {"dataset_name", "panel", "source_identifier", "global_rank"},
        assignment_path,
    )
    if len(assignment) != 114 or assignment["dataset_name"].duplicated().any():
        raise ValueError("The four-panel assignment must contain 114 unique datasets.")
    if sorted(assignment["panel"].unique()) != PANEL_ORDER:
        raise ValueError("Unexpected four-panel identifiers.")
    if (assignment.groupby("source_identifier")["panel"].nunique() > 1).any():
        raise ValueError("At least one source family is split across panels.")
    panel_sizes = assignment.groupby("panel").size().reindex(PANEL_ORDER)
    if panel_sizes.tolist() != [29, 29, 28, 28]:
        raise ValueError(f"Unexpected panel sizes: {panel_sizes.to_dict()}")
    if not assignment["global_rank"].between(1, 115).all():
        raise ValueError("A panel global rank falls outside 1--115.")
    return assignment


def emphasize_axis_text(ax: plt.Axes) -> None:
    """Keep labels legible after a six-panel figure is scaled to page width."""

    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")


def bold_grid_figure_rc(font_size: float) -> dict[str, object]:
    """Large, explicit bold typography for page-width multi-panel figures."""

    style = publication_rc()
    style.update(
        {
            "text.usetex": False,
            "text.latex.preamble": "",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": font_size,
            "font.weight": "bold",
            "axes.labelsize": font_size,
            "axes.labelweight": "bold",
            "axes.titlesize": font_size,
            "axes.titleweight": "bold",
            "xtick.labelsize": font_size - 2.5,
            "ytick.labelsize": font_size - 2.5,
        }
    )
    return style


def draw_panel_rank_balance(ax: plt.Axes, assignment: pd.DataFrame) -> None:
    """Draw the frozen global-rank distribution in a compact grid cell."""

    positions = np.arange(1, len(PANEL_ORDER) + 1)
    groups = [
        assignment.loc[assignment["panel"].eq(panel), "global_rank"].to_numpy(dtype=float)
        for panel in PANEL_ORDER
    ]
    for lower, upper in [(0.5, 29.5), (58.5, 87.5)]:
        ax.axhspan(lower, upper, color="#6B7280", alpha=0.055, zorder=-3)

    boxplots = ax.boxplot(
        groups,
        positions=positions,
        vert=True,
        widths=0.56,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#202124", "linewidth": 1.7},
        whiskerprops={"color": "#4B5563", "linewidth": 1.0},
        capprops={"color": "#4B5563", "linewidth": 1.0},
    )
    for panel, position, ranks, box in zip(
        PANEL_ORDER, positions, groups, boxplots["boxes"], strict=True
    ):
        box.set_facecolor(PANEL_COLORS[panel])
        box.set_alpha(0.14)
        box.set_edgecolor(PANEL_COLORS[panel])
        box.set_linewidth(1.1)
        jitter = 0.12 * np.sin(ranks * 2.399963229728653)
        ax.scatter(
            position + jitter,
            ranks,
            s=27,
            color=PANEL_COLORS[panel],
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        ax.plot(
            position,
            ranks.mean(),
            marker="D",
            color="#202124",
            markerfacecolor="white",
            markersize=5.5,
            markeredgewidth=1.1,
            zorder=3,
        )

    ax.axhline(
        float(assignment["global_rank"].mean()),
        color="#202124",
        linestyle="--",
        linewidth=1.1,
        alpha=0.75,
        zorder=-1,
    )
    ax.set_xticks(positions, [PANEL_LABELS[panel] for panel in PANEL_ORDER])
    ax.set_yticks([1, 29, 58, 87, 115])
    ax.set_ylim(115.5, 0.5)
    ax.set_xlabel("Source-disjoint panel")
    ax.set_ylabel("Frozen global QC rank\n(1 = best)")
    ax.set_title("A   Panel QC-rank balance", loc="left", fontweight="bold")
    ax.grid(axis="y")
    emphasize_axis_text(ax)


def transcript_id_hash(transcript_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(transcript_ids).encode("utf-8")).hexdigest()


def audit_profile_export(
    prediction_path: Path, expected_ids: list[str], label: str
) -> tuple[str, float]:
    """Validate one shared-profile export and summarize its fixed geometry."""

    profiles = pd.read_parquet(
        prediction_path,
        columns=["transcript_id", "transcript_length", "L_t", "valid_position_mask"],
    )
    profiles["transcript_id"] = profiles["transcript_id"].astype(str)
    if profiles["transcript_id"].duplicated().any():
        raise ValueError(f"{label} has duplicate test transcripts.")
    if set(profiles["transcript_id"]) != set(expected_ids):
        raise ValueError(f"{label} does not contain the exact frozen test-ID set.")
    profiles = profiles.set_index("transcript_id").loc[expected_ids]

    coordinate_digest = hashlib.sha256()
    max_mean_one_error = 0.0
    for transcript_id, row in profiles.iterrows():
        values = np.asarray(row["L_t"], dtype=float)
        mask = np.asarray(row["valid_position_mask"], dtype=bool)
        length = int(row["transcript_length"])
        if len(values) != length or len(mask) != length or mask.sum() < 2:
            raise ValueError(f"{label}/{transcript_id} has inconsistent profile geometry.")
        if not np.isfinite(values[mask]).all():
            raise ValueError(f"{label}/{transcript_id} has a non-finite shared profile.")
        max_mean_one_error = max(
            max_mean_one_error, abs(float(values[mask].mean()) - 1.0)
        )
        coordinate_digest.update(transcript_id.encode("utf-8"))
        coordinate_digest.update(np.int64(length).tobytes())
        coordinate_digest.update(np.packbits(mask).tobytes())
    return coordinate_digest.hexdigest(), max_mean_one_error


def audit_four_panel_comparison(four_panel_root: Path, output_dir: Path) -> list[str]:
    """Require identical held-out IDs and coordinates for every plotted model."""

    split_path = four_panel_root / "inputs/split.json"
    experiment_path = four_panel_root / "experiment_manifest.json"
    availability_path = four_panel_root / "analysis/availability.csv"
    split = json.loads(split_path.read_text())
    experiment = json.loads(experiment_path.read_text())
    availability = pd.read_csv(availability_path)

    validation_ids = [str(value) for value in split["common_validation_ids"]]
    test_ids = [str(value) for value in split["common_test_ids"]]
    if len(validation_ids) != 1593 or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("The frozen four-panel validation cohort is not 1,593 unique IDs.")
    if len(test_ids) != 1593 or len(set(test_ids)) != len(test_ids):
        raise ValueError("The frozen four-panel test cohort is not 1,593 unique IDs.")
    if set(validation_ids).intersection(test_ids):
        raise ValueError("The frozen validation and test cohorts overlap.")

    expected_validation_hash = transcript_id_hash(validation_ids)
    expected_test_hash = transcript_id_hash(test_ids)
    if split["fold_id_hashes"]["validation"] != expected_validation_hash:
        raise ValueError("The stored validation-ID hash does not match the frozen IDs.")
    if split["fold_id_hashes"]["test"] != expected_test_hash:
        raise ValueError("The stored test-ID hash does not match the frozen IDs.")

    for panel in PANEL_ORDER:
        fold = experiment["source_folds"][panel]
        if [str(value) for value in fold["validation_ids"]] != validation_ids:
            raise ValueError(f"{panel} does not use the common validation IDs.")
        if [str(value) for value in fold["test_ids"]] != test_ids:
            raise ValueError(f"{panel} does not use the common test IDs.")
        train_ids = {str(value) for value in fold["train_ids"]}
        if train_ids.intersection(validation_ids) or train_ids.intersection(test_ids):
            raise ValueError(f"{panel} training IDs overlap a held-out fold.")

    require_columns(
        availability,
        {
            "arm",
            "panel_id",
            "status",
            "runtime_config_verified",
            "prediction_path",
            "runtime_manifest",
            "n_test_transcripts",
        },
        availability_path,
    )
    selected = availability.loc[availability["arm"].isin(ARM_ORDER)].copy()
    expected_cells = {(arm, panel) for arm in ARM_ORDER for panel in PANEL_ORDER}
    observed_cells = set(zip(selected["arm"], selected["panel_id"], strict=False))
    if len(selected) != len(expected_cells) or observed_cells != expected_cells:
        raise ValueError(
            "Availability does not contain exactly the 20 plotted panel-policy models."
        )

    baseline_coordinate_hash: str | None = None
    audit_rows: list[dict[str, object]] = []
    for record in selected.sort_values(["arm", "panel_id"]).to_dict("records"):
        arm = str(record["arm"])
        panel = str(record["panel_id"])
        if record["status"] != "validated_predictions" or not bool(
            record["runtime_config_verified"]
        ):
            raise ValueError(f"{panel}/{arm} is not a runtime-verified prediction export.")
        if int(record["n_test_transcripts"]) != len(test_ids):
            raise ValueError(f"{panel}/{arm} reports the wrong test-cohort size.")

        checkpoint_manifest_path = Path(str(record["runtime_manifest"]))
        checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text())["best_val_loss"]
        if checkpoint_manifest["split_name"] != "test":
            raise ValueError(f"{panel}/{arm} was not exported on the test split.")
        if not checkpoint_manifest["sequence_only_shared_profile_prediction"]:
            raise ValueError(f"{panel}/{arm} is not a sequence-only shared-profile export.")
        if int(checkpoint_manifest["transcript_count"]) != len(test_ids):
            raise ValueError(f"{panel}/{arm} manifest has the wrong transcript count.")
        if checkpoint_manifest["transcript_id_hash"] != expected_test_hash:
            raise ValueError(f"{panel}/{arm} manifest has the wrong test-ID hash.")

        coordinate_hash, max_mean_one_error = audit_profile_export(
            Path(str(record["prediction_path"])), test_ids, f"{panel}/{arm}"
        )
        if baseline_coordinate_hash is None:
            baseline_coordinate_hash = coordinate_hash
        elif coordinate_hash != baseline_coordinate_hash:
            raise ValueError(f"{panel}/{arm} does not use the common CDS coordinates/masks.")

        audit_rows.append(
            {
                "panel": panel,
                "arm": arm,
                "validation_ids_common": True,
                "test_ids_common": True,
                "train_validation_overlap": 0,
                "train_test_overlap": 0,
                "checkpoint_selection": "best_val_loss",
                "prediction_split": "test",
                "sequence_only_shared_profile": True,
                "n_test_transcripts": len(test_ids),
                "test_id_hash": expected_test_hash,
                "coordinate_mask_hash": coordinate_hash,
                "max_mean_one_error": max_mean_one_error,
            }
        )

    pd.DataFrame(audit_rows).to_csv(
        output_dir / "appendix_cross_panel_comparison_audit.csv", index=False
    )
    return test_ids


def load_cross_panel_pcc(
    four_panel_root: Path, output_dir: Path
) -> tuple[pd.DataFrame, list[str]]:
    test_ids = audit_four_panel_comparison(four_panel_root, output_dir)
    path = four_panel_root / "analysis/transcript_agreement.csv"
    data = pd.read_csv(path)
    require_columns(
        data,
        {"kind", "panel_a", "panel_b", "arm", "transcript_id", "PCC", "reason"},
        path,
    )
    data = data.loc[
        data["kind"].eq("cross_panel") & data["arm"].isin(CROSS_PANEL_LAYOUT)
    ].copy()
    if data["PCC"].isna().any() or not data["reason"].eq("ok").all():
        raise ValueError("A plotted cross-panel comparison is not a finite, valid PCC.")
    pairs = [(a, b) for index, a in enumerate(PANEL_ORDER) for b in PANEL_ORDER[index + 1 :]]
    pair_labels = [f"{PANEL_LABELS[a]}--{PANEL_LABELS[b]}" for a, b in pairs]
    expected = pd.MultiIndex.from_product(
        [CROSS_PANEL_LAYOUT, pair_labels], names=["arm", "pair_label"]
    )
    data["pair_label"] = data.apply(
        lambda row: f"{PANEL_LABELS[row['panel_a']]}--{PANEL_LABELS[row['panel_b']]}", axis=1
    )
    counts = data.groupby(["arm", "pair_label"]).size().reindex(expected)
    if counts.isna().any() or counts.nunique() != 1 or int(counts.iloc[0]) != len(test_ids):
        raise ValueError(f"Cross-panel PCC cohorts are incomplete: {counts.to_dict()}")
    for (arm, pair_label), group in data.groupby(["arm", "pair_label"]):
        observed_ids = group["transcript_id"].astype(str)
        if observed_ids.duplicated().any() or set(observed_ids) != set(test_ids):
            raise ValueError(f"{arm}/{pair_label} does not use the exact frozen test cohort.")
    return data, pair_labels


def adaptive_common_pcc_limits(values: np.ndarray) -> tuple[float, float]:
    """Use robust pooled limits while preserving a common scale across policies."""

    lower, upper = np.quantile(values, [0.005, 0.995])
    span = float(upper - lower)
    return max(-1.0, float(lower - 0.04 * span)), min(1.0, float(upper + 0.04 * span))


def draw_cross_panel_pcc_facet(
    ax: plt.Axes,
    data: pd.DataFrame,
    pair_labels: list[str],
    arm: str,
    letter: str,
    y_limits: tuple[float, float],
    show_ylabel: bool,
) -> None:
    groups = [
        data.loc[data["arm"].eq(arm) & data["pair_label"].eq(pair), "PCC"].to_numpy()
        for pair in pair_labels
    ]
    positions = np.arange(1, len(groups) + 1)
    violins = ax.violinplot(
        groups,
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.18,
    )
    for body, color in zip(violins["bodies"], PAIR_COLORS, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.42)
        body.set_linewidth(0.8)
    ax.boxplot(
        groups,
        positions=positions,
        widths=0.17,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#A60000", "linewidth": 1.7},
        boxprops={"facecolor": "white", "edgecolor": "#202124", "linewidth": 0.9},
        whiskerprops={"color": "#202124", "linewidth": 0.8},
        capprops={"color": "#202124", "linewidth": 0.8},
    )
    combined = np.concatenate(groups)
    outside_fraction = float(
        np.mean((combined < y_limits[0]) | (combined > y_limits[1]))
    )
    percent_symbol = r"\%" if matplotlib.rcParams["text.usetex"] else "%"
    ax.set_title(
        f"{letter}   {ARM_LABELS[arm]}  (mean {combined.mean():.3f})",
        loc="left",
        fontweight="bold",
    )
    display_pair_labels = [label.replace("--", "–") for label in pair_labels]
    ax.set_xticks(positions, display_pair_labels, rotation=31, ha="right")
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Panel pair")
    if show_ylabel:
        ax.set_ylabel(r"Transcript PCC of $\mathbf{L}_t$")
    ax.grid(axis="y")
    ax.text(
        0.985,
        0.035,
        f"{100.0 * outside_fraction:.1f}{percent_symbol} outside axis",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#5F6368",
    )
    emphasize_axis_text(ax)


def plot_four_panel_overview(
    assignment: pd.DataFrame, four_panel_root: Path, output_dir: Path
) -> tuple[pd.DataFrame, list[str]]:
    data, pair_labels = load_cross_panel_pcc(four_panel_root, output_dir)
    y_limits = adaptive_common_pcc_limits(data["PCC"].to_numpy(dtype=float))
    style = bold_grid_figure_rc(13.5)
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.9), constrained_layout=True)
        draw_panel_rank_balance(axes[0, 0], assignment)
        layout = [
            (axes[0, 1], "ranked_p1", "B", True),
            (axes[0, 2], "ranked_p3", "C", False),
            (axes[1, 0], "equal", "D", True),
            (axes[1, 1], "reverse_p1", "E", False),
            (axes[1, 2], "reverse_p3", "F", False),
        ]
        for ax, arm, letter, show_ylabel in layout:
            draw_cross_panel_pcc_facet(
                ax, data, pair_labels, arm, letter, y_limits, show_ylabel
            )
        save_figure(fig, output_dir / "appendix_four_panel_rank_and_cross_panel_pcc")

    assignment.sort_values(["panel", "global_rank"]).to_csv(
        output_dir / "appendix_four_panel_rank_and_cross_panel_pcc_rank_source.csv",
        index=False,
    )
    data[
        ["panel_a", "panel_b", "pair_label", "arm", "transcript_id", "PCC", "reason"]
    ].sort_values(["arm", "panel_a", "panel_b", "transcript_id"]).to_csv(
        output_dir / "appendix_four_panel_rank_and_cross_panel_pcc_source.csv",
        index=False,
    )
    return data, pair_labels


def load_concentration(cumulative_design_root: Path) -> pd.DataFrame:
    path = cumulative_design_root / "reference_concentration.csv"
    data = pd.read_csv(path)
    require_columns(data, {"N", "arm", "N_ref", "weighted_mean_rank"}, path)
    expected_ns = [2, 5, 10, 20, 40, 80, 114]
    expected = pd.MultiIndex.from_product(
        [expected_ns, ARM_ORDER], names=["N", "arm"]
    )
    observed = pd.MultiIndex.from_frame(data[["N", "arm"]])
    if len(data) != len(expected) or set(observed) != set(expected):
        raise ValueError(
            "The cumulative concentration table does not contain all 35 design cells."
        )
    data = data.copy()
    data["effective_fraction"] = data["N_ref"] / data["N"]
    for exponent in (1, 3):
        ranked = data.loc[data["arm"].eq(f"ranked_p{exponent}")].sort_values("N")
        reverse = data.loc[data["arm"].eq(f"reverse_p{exponent}")].sort_values("N")
        if not np.allclose(ranked["N_ref"], reverse["N_ref"], rtol=0.0, atol=1e-10):
            raise ValueError("Ranked and reversed policies should have identical concentration.")
    return data.sort_values(["N", "arm"])


def plot_cumulative_concentration(data: pd.DataFrame, output_dir: Path) -> None:
    ns = sorted(data["N"].unique())
    with matplotlib.rc_context(publication_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.55), constrained_layout=True)

        ax = axes[0]
        for arm in ["equal", "ranked_p1", "ranked_p3"]:
            part = data.loc[data["arm"].eq(arm)].sort_values("N")
            label = {
                "equal": "Equal",
                "ranked_p1": r"$p=1$ (ranked/reversed)",
                "ranked_p3": r"$p=3$ (ranked/reversed)",
            }[arm]
            ax.plot(
                part["N"],
                part["effective_fraction"],
                color=ARM_COLORS[arm],
                linestyle=ARM_LINESTYLES[arm],
                marker="o",
                markersize=4.5,
                linewidth=1.7,
                label=label,
            )
        ax.axvspan(1.8, 22.5, color="#777777", alpha=0.08, zorder=-2)
        ax.text(3.0, 0.70, r"Near-uniform through $N=20$", fontsize=9.5, color="#555555")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_xlim(1.8, 125)
        ax.set_ylim(0.40, 1.025)
        ax.set_xlabel("Number of datasets, $N$")
        ax.set_ylabel(r"Effective reference fraction, $N_{\rm eff}/N$")
        ax.set_title("A   Reference concentration", loc="left", fontweight="bold")
        ax.grid(axis="both")
        ax.legend(loc="lower left", fontsize=9.5)

        ax = axes[1]
        for arm in ARM_ORDER:
            part = data.loc[data["arm"].eq(arm)].sort_values("N")
            ax.plot(
                part["N"],
                part["weighted_mean_rank"],
                color=ARM_COLORS[arm],
                linestyle=ARM_LINESTYLES[arm],
                marker="o",
                markersize=4.2,
                linewidth=1.55,
                label=ARM_LABELS[arm],
            )
        ax.axvspan(1.8, 22.5, color="#777777", alpha=0.08, zorder=-2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_xlim(1.8, 125)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Number of datasets, $N$")
        ax.set_ylabel("Reference-weighted mean global rank")
        ax.set_title(
            "B   Which end of the ranking defines the reference",
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="both")
        ax.legend(ncol=2, loc="upper left", columnspacing=0.9, fontsize=9.5)

        save_figure(fig, output_dir / "appendix_cumulative_reference_concentration")

    data.to_csv(
        output_dir / "appendix_cumulative_reference_concentration_source.csv", index=False
    )


def audit_cumulative_anchor_comparison(
    cumulative_result_root: Path, selected_summary: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """Audit the exact legacy test cohort while retaining its fold limitation."""

    split_path = cumulative_result_root / "inputs/split.json"
    experiment_path = cumulative_result_root / "experiment_manifest.json"
    availability_path = cumulative_result_root / "analysis/availability.csv"
    transcript_path = cumulative_result_root / "analysis/transcript_stability.csv"
    split = json.loads(split_path.read_text())
    experiment = json.loads(experiment_path.read_text())
    availability = pd.read_csv(availability_path)
    transcript_rows = pd.read_csv(transcript_path)

    test_ids = [str(value) for value in split["common_test_ids"]]
    if len(test_ids) != 1771 or len(set(test_ids)) != len(test_ids):
        raise ValueError("The legacy cumulative test cohort is not 1,771 unique IDs.")
    expected_test_hash = transcript_id_hash(test_ids)
    if split["fold_id_hashes"]["test"] != expected_test_hash:
        raise ValueError("The legacy cumulative test-ID hash does not match the frozen IDs.")

    comparison_cells = {
        (int(row.N_b), str(row.arm)) for row in selected_summary.itertuples(index=False)
    }
    required_models = comparison_cells | {(2, "equal")}
    if len(comparison_cells) != len(selected_summary):
        raise ValueError("The cumulative anchor summary contains duplicate model cells.")

    task_cells = {
        (int(task["N"]), str(task["arm"])): task for task in experiment["tasks"]
    }
    validation_hashes: dict[int, str] = {}
    for n_value in sorted({n_value for n_value, _ in required_models}):
        fold = experiment["source_folds"][str(n_value)]
        fold_test_ids = [str(value) for value in fold["test_ids"]]
        validation_ids = [str(value) for value in fold["validation_ids"]]
        train_ids = {str(value) for value in fold["train_ids"]}
        if fold_test_ids != test_ids:
            raise ValueError(f"N={n_value} does not use the common cumulative test IDs.")
        if len(validation_ids) != len(set(validation_ids)):
            raise ValueError(f"N={n_value} has duplicate validation IDs.")
        if set(validation_ids).intersection(test_ids):
            raise ValueError(f"N={n_value} validation and test folds overlap.")
        if train_ids.intersection(validation_ids) or train_ids.intersection(test_ids):
            raise ValueError(f"N={n_value} training IDs overlap a held-out fold.")
        validation_hashes[n_value] = transcript_id_hash(validation_ids)
        for arm in {arm for n, arm in required_models if n == n_value}:
            task = task_cells[(n_value, arm)]
            if task["source_panel"] != fold["source_panel"]:
                raise ValueError(f"N={n_value}/{arm} does not reuse the declared N-specific fold.")

    require_columns(
        availability,
        {
            "arm",
            "N",
            "status",
            "runtime_config_verified",
            "prediction_path",
            "runtime_manifest",
            "n_test_transcripts",
        },
        availability_path,
    )
    selected_models = availability.loc[
        availability.apply(lambda row: (int(row["N"]), str(row["arm"])) in required_models, axis=1)
    ].copy()
    observed_models = set(
        zip(selected_models["N"].astype(int), selected_models["arm"].astype(str), strict=False)
    )
    if len(selected_models) != len(required_models) or observed_models != required_models:
        raise ValueError("Availability is missing a model used by the cumulative anchor plot.")

    baseline_coordinate_hash: str | None = None
    audit_rows: list[dict[str, object]] = []
    for record in selected_models.sort_values(["N", "arm"]).to_dict("records"):
        n_value = int(record["N"])
        arm = str(record["arm"])
        label = f"N={n_value}/{arm}"
        if record["status"] != "validated_predictions" or not bool(
            record["runtime_config_verified"]
        ):
            raise ValueError(f"{label} is not a runtime-verified prediction export.")
        if int(record["n_test_transcripts"]) != len(test_ids):
            raise ValueError(f"{label} reports the wrong test-cohort size.")

        checkpoint_manifest = json.loads(Path(str(record["runtime_manifest"])).read_text())[
            "best_val_loss"
        ]
        if (
            checkpoint_manifest["split_name"] != "test"
            or not checkpoint_manifest["sequence_only_shared_profile_prediction"]
            or int(checkpoint_manifest["transcript_count"]) != len(test_ids)
            or checkpoint_manifest["transcript_id_hash"] != expected_test_hash
        ):
            raise ValueError(f"{label} has an inconsistent prediction manifest.")

        coordinate_hash, max_mean_one_error = audit_profile_export(
            Path(str(record["prediction_path"])), test_ids, label
        )
        if baseline_coordinate_hash is None:
            baseline_coordinate_hash = coordinate_hash
        elif coordinate_hash != baseline_coordinate_hash:
            raise ValueError(f"{label} does not use the common CDS coordinates/masks.")

        audit_rows.append(
            {
                "N": n_value,
                "arm": arm,
                "role": "common_anchor" if (n_value, arm) == (2, "equal") else "comparison",
                "validation_id_hash": validation_hashes[n_value],
                "test_ids_common": True,
                "train_validation_overlap": 0,
                "train_test_overlap": 0,
                "checkpoint_selection": "best_val_loss",
                "prediction_split": "test",
                "n_test_transcripts": len(test_ids),
                "test_id_hash": expected_test_hash,
                "coordinate_mask_hash": coordinate_hash,
                "max_mean_one_error": max_mean_one_error,
            }
        )

    relevant_rows = transcript_rows.loc[
        transcript_rows["kind"].eq("shared_anchor")
        & transcript_rows.apply(
            lambda row: (int(row["N_b"]), str(row["arm"])) in comparison_cells, axis=1
        )
    ].copy()
    if relevant_rows["PCC"].isna().any() or not relevant_rows["reason"].eq("ok").all():
        raise ValueError("A cumulative anchor comparison is not a finite, valid PCC.")
    for (n_value, arm), group in relevant_rows.groupby(["N_b", "arm"]):
        observed_ids = group["transcript_id"].astype(str)
        if observed_ids.duplicated().any() or set(observed_ids) != set(test_ids):
            raise ValueError(f"N={n_value}/{arm} does not compare the exact frozen test cohort.")
    observed_comparisons = set(
        zip(relevant_rows["N_b"].astype(int), relevant_rows["arm"].astype(str), strict=False)
    )
    if observed_comparisons != comparison_cells:
        raise ValueError("The cumulative transcript-level table is missing a plotted comparison.")

    pd.DataFrame(audit_rows).to_csv(
        output_dir / "appendix_cumulative_anchor_comparison_audit.csv", index=False
    )
    return relevant_rows.sort_values(["N_b", "arm", "transcript_id"])


def transcript_bootstrap_mean_intervals(
    transcript_rows: pd.DataFrame,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired transcript bootstrap intervals for each cumulative comparison.

    The same resampled transcript indices are used for every model cell.  The
    intervals therefore describe sampling over held-out transcripts; they do
    not represent uncertainty over training seeds or dataset collections.
    """

    pivot = transcript_rows.pivot(
        index="transcript_id", columns=["N_b", "arm"], values="PCC"
    ).sort_index(axis=1)
    if pivot.isna().any().any():
        raise ValueError("The cumulative PCC table is incomplete after transcript alignment.")

    values = pivot.to_numpy(dtype=float)
    n_transcripts, n_cells = values.shape
    if n_transcripts != 1771:
        raise ValueError("Bootstrap intervals require the audited 1,771 transcripts.")

    rng = np.random.default_rng(seed)
    bootstrapped_means = np.empty((n_resamples, n_cells), dtype=float)
    chunk_size = 100
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, n_transcripts, size=(stop - start, n_transcripts))
        bootstrapped_means[start:stop] = values[indices].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrapped_means, [tail, 1.0 - tail], axis=0)
    rows: list[dict[str, object]] = []
    for column_index, (n_value, arm) in enumerate(pivot.columns):
        rows.append(
            {
                "N_b": int(n_value),
                "arm": str(arm),
                "mean": float(values[:, column_index].mean()),
                "ci_lower": float(lower[column_index]),
                "ci_upper": float(upper[column_index]),
                "confidence": confidence,
                "bootstrap_resamples": n_resamples,
                "n_transcripts": n_transcripts,
            }
        )
    return pd.DataFrame(rows).sort_values(["N_b", "arm"])


def plot_cumulative_anchor_pcc(
    cumulative_result_root: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = cumulative_result_root / "analysis/stability_summary.csv"
    data = pd.read_csv(path)
    require_columns(
        data,
        {"kind", "N_a", "N_b", "metric", "n_valid", "arm", "mean", "median"},
        path,
    )
    selected = data.loc[
        data["kind"].eq("shared_anchor") & data["metric"].eq("PCC")
    ].copy()
    if selected.empty or not selected["N_a"].eq(2).all():
        raise ValueError("No PCC comparisons to the shared N=2 anchor were found.")
    if not selected["n_valid"].eq(1771).all():
        raise ValueError(
            "The shared-anchor PCC rows do not use the common 1,771-transcript cohort."
        )
    transcript_rows = audit_cumulative_anchor_comparison(
        cumulative_result_root, selected, output_dir
    )
    intervals = transcript_bootstrap_mean_intervals(transcript_rows)
    mean_check = selected[["N_b", "arm", "mean"]].merge(
        intervals[["N_b", "arm", "mean"]],
        on=["N_b", "arm"],
        how="outer",
        suffixes=("_summary", "_transcripts"),
        validate="one_to_one",
    )
    if mean_check.isna().any().any() or not np.allclose(
        mean_check["mean_summary"],
        mean_check["mean_transcripts"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Cumulative summary means disagree with transcript-level PCC values.")
    selected = selected.merge(
        intervals.drop(columns=["mean"]), on=["N_b", "arm"], how="left", validate="one_to_one"
    )
    if selected[["ci_lower", "ci_upper"]].isna().any().any():
        raise ValueError("A cumulative anchor mean is missing its bootstrap interval.")

    style = bold_grid_figure_rc(12.5)
    with matplotlib.rc_context(style):
        fig, ax = plt.subplots(figsize=(8.5, 4.35), constrained_layout=True)
        ax.axvspan(4.5, 22.5, color="#777777", alpha=0.075, zorder=-3)
        ax.text(
            5.25,
            0.632,
            r"Reference weights nearly uniform through $N=20$",
            color="#5F6368",
            fontsize=10.0,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        for arm in ARM_ORDER:
            part = selected.loc[selected["arm"].eq(arm)].sort_values("N_b")
            if part.empty:
                continue
            ax.fill_between(
                part["N_b"].to_numpy(dtype=float),
                part["ci_lower"].to_numpy(dtype=float),
                part["ci_upper"].to_numpy(dtype=float),
                color=ARM_COLORS[arm],
                alpha=0.11,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                part["N_b"],
                part["mean"],
                color=ARM_COLORS[arm],
                linestyle=ARM_LINESTYLES[arm],
                marker="o",
                markersize=6.0,
                linewidth=2.2,
                label=ARM_LABELS[arm],
                zorder=2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([5, 10, 20, 40, 80, 114], ["5", "10", "20", "40", "80", "114"])
        ax.set_xlim(4.5, 125)
        ax.set_ylim(0.62, 0.88)
        ax.set_xlabel("Datasets in the cumulative best-$N$ collection")
        ax.set_ylabel(r"Mean PCC to the common $N=2$ anchor")
        ax.set_title(
            r"Cumulative stability of $\mathbf{L}_t$",
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="both")
        ax.legend(ncol=3, loc="upper right", columnspacing=0.9, fontsize=10.0)
        arms_at_114 = set(selected.loc[selected["N_b"].eq(114), "arm"])
        if arms_at_114 != set(ARM_ORDER):
            status = "pending" if not arms_at_114 else "partial"
            ax.text(
                114,
                0.626,
                rf"$N=114$ {status}",
                color="#777777",
                fontsize=10.0,
                fontweight="bold",
                ha="right",
                va="bottom",
            )
        emphasize_axis_text(ax)
        save_figure(fig, output_dir / "appendix_cumulative_anchor_L_PCC")

    selected.sort_values(["N_b", "arm"]).to_csv(
        output_dir / "appendix_cumulative_anchor_L_PCC_source.csv", index=False
    )
    intervals.to_csv(
        output_dir / "appendix_cumulative_anchor_L_PCC_bootstrap_intervals.csv",
        index=False,
    )
    return selected, transcript_rows


def plot_main_text_summary(
    cross_panel_data: pd.DataFrame,
    pair_labels: list[str],
    cumulative_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Combine the two real-data claims in one main-text figure."""

    equal_pcc = cross_panel_data.loc[
        cross_panel_data["arm"].eq("equal"), "PCC"
    ].to_numpy(dtype=float)
    if len(equal_pcc) != 6 * 1593:
        raise ValueError("The main-text panel comparison requires six complete equal-arm pairs.")
    left_limits = adaptive_common_pcc_limits(equal_pcc)

    displayed_arms = ["equal", "ranked_p1", "reverse_p3"]
    cumulative = cumulative_summary.loc[
        cumulative_summary["arm"].isin(displayed_arms)
    ].copy()
    if cumulative.empty:
        raise ValueError("The main-text cumulative panel has no available comparisons.")
    lower = float(cumulative["ci_lower"].min())
    upper = float(cumulative["ci_upper"].max())
    padding = 0.07 * (upper - lower)
    right_limits = (max(-1.0, lower - padding), min(1.0, upper + padding))

    style = bold_grid_figure_rc(14.0)
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.25), constrained_layout=True)

        ax = axes[0]
        draw_cross_panel_pcc_facet(
            ax,
            cross_panel_data,
            pair_labels,
            "equal",
            "A",
            left_limits,
            True,
        )
        ax.set_title(
            rf"A   Four balanced panels, equal reference  (mean {equal_pcc.mean():.3f})",
            loc="left",
            fontweight="bold",
        )
        ax.set_xlabel("Source-disjoint panel pair")
        emphasize_axis_text(ax)

        ax = axes[1]
        ax.axvspan(4.5, 22.5, color="#777777", alpha=0.075, zorder=-3)
        for arm in displayed_arms:
            part = cumulative.loc[cumulative["arm"].eq(arm)].sort_values("N_b")
            ax.fill_between(
                part["N_b"].to_numpy(dtype=float),
                part["ci_lower"].to_numpy(dtype=float),
                part["ci_upper"].to_numpy(dtype=float),
                color=ARM_COLORS[arm],
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                part["N_b"],
                part["mean"],
                color=ARM_COLORS[arm],
                linestyle=ARM_LINESTYLES[arm],
                marker="o",
                markersize=7.0,
                linewidth=2.6,
                label=ARM_LABELS[arm],
                zorder=2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([5, 10, 20, 40, 80, 114], ["5", "10", "20", "40", "80", "114"])
        ax.set_xlim(4.5, 125)
        ax.set_ylim(*right_limits)
        ax.set_xlabel("Datasets in the cumulative best-$N$ collection")
        ax.set_ylabel(r"Mean PCC to the common $N=2$ anchor")
        ax.set_title("B   Cumulative shared-profile stability", loc="left", fontweight="bold")
        ax.grid(axis="both")
        ax.legend(loc="upper right", fontsize=11.0)
        ax.text(
            5.25,
            right_limits[0] + 0.025 * (right_limits[1] - right_limits[0]),
            r"Near-uniform reference through $N=20$",
            color="#5F6368",
            fontsize=10.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        displayed_at_114 = set(cumulative.loc[cumulative["N_b"].eq(114), "arm"])
        if displayed_at_114 != set(displayed_arms):
            status = "pending" if not displayed_at_114 else "partial"
            ax.text(
                114,
                right_limits[0] + 0.025 * (right_limits[1] - right_limits[0]),
                rf"$N=114$ {status}",
                color="#777777",
                fontsize=10.5,
                fontweight="bold",
                ha="right",
                va="bottom",
            )
        emphasize_axis_text(ax)

        save_figure(fig, output_dir / "main_text_real_data_stability")

    cross_panel_data.loc[cross_panel_data["arm"].eq("equal")].sort_values(
        ["panel_a", "panel_b", "transcript_id"]
    ).to_csv(output_dir / "main_text_four_panel_equal_source.csv", index=False)
    cumulative.sort_values(["N_b", "arm"]).to_csv(
        output_dir / "main_text_cumulative_anchor_source.csv", index=False
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignment = load_panel_assignment(args.four_panel_root)
    cross_panel_data, pair_labels = plot_four_panel_overview(
        assignment, args.four_panel_root, args.output_dir
    )
    concentration = load_concentration(args.cumulative_design_root)
    plot_cumulative_concentration(concentration, args.output_dir)
    cumulative_summary, _ = plot_cumulative_anchor_pcc(
        args.cumulative_result_root, args.output_dir
    )
    plot_main_text_summary(
        cross_panel_data,
        pair_labels,
        cumulative_summary,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
