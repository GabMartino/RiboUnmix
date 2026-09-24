#!/usr/bin/env python3
"""Create manuscript figures from audited real-data scalar summaries.

The script never loads checkpoints or profile arrays.  It combines the frozen
four-panel transcript metrics with the fixed-cohort cumulative quality-score
summaries, and writes the exact scalar source tables beside the figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Utils.publication_plot_style import publication_rc

POLICIES = ("equal", "score_p1", "score_p3", "score_p5")
POLICY_LABELS = {
    "equal": "Equal reference",
    "score_p1": r"Score weighted, $p=1$",
    "score_p3": r"Score weighted, $p=3$",
    "score_p5": r"Score weighted, $p=5$",
}
POLICY_COLORS = {
    "equal": "#4C566A",
    "score_p1": "#0072B2",
    "score_p3": "#D55E00",
    "score_p5": "#7B3FA1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="both", alpha=0.28, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", which="major", width=1.45, length=5.0)
    for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        label.set_fontweight("bold")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_summary(panel_metrics: pd.DataFrame) -> pd.DataFrame:
    equal = panel_metrics.loc[panel_metrics["arm"].eq("equal")].copy()
    if equal.shape[0] != 6 * 1593 or equal["transcript_id"].nunique() != 1593:
        raise ValueError("Expected six complete equal-reference panel pairs on 1,593 transcripts.")
    rows = []
    for pair, group in equal.groupby("pair_label", sort=False):
        values = group["PCC"].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite panel PCC in {pair}.")
        rows.append({
            "pair_label": pair,
            "n_transcripts": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
        })
    return pd.DataFrame(rows)


def selection_overlap(membership: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in (2, 5, 10, 20, 40, 80):
        best = set(membership.loc[
            membership["selection_direction"].eq("best_first") & membership["N"].eq(n),
            "dataset_id",
        ])
        worst = set(membership.loc[
            membership["selection_direction"].eq("worst_first") & membership["N"].eq(n),
            "dataset_id",
        ])
        if len(best) != n or len(worst) != n:
            raise ValueError(f"Invalid best/worst membership at N={n}.")
        intersection = len(best & worst)
        rows.append({
            "N": n,
            "intersection": intersection,
            "jaccard": intersection / len(best | worst),
            "forced_minimum_intersection": max(0, 2 * n - 114),
        })
    rows.append({"N": 114, "intersection": 114, "jaccard": 1.0, "forced_minimum_intersection": 114})
    return pd.DataFrame(rows)


def plot_main(
    panel_metrics: pd.DataFrame,
    panel: pd.DataFrame,
    anchor: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    equal_panel = panel_metrics.loc[panel_metrics["arm"].eq("equal")].copy()
    best_anchor = anchor.loc[anchor["selection_direction"].eq("best_first")].copy()
    sizes = [2, 5, 10, 20, 40, 80, 114]
    if not best_anchor.anchor_policy.eq(best_anchor.reference_policy).all() or not best_anchor.anchor_N.eq(2).all():
        raise ValueError("Each policy must use its own N=2 fit, not a shared equal-reference anchor.")
    for policy in POLICIES:
        curve = best_anchor.loc[best_anchor["reference_policy"].eq(policy)].sort_values("N")
        if curve.N.tolist() != sizes:
            raise ValueError(f"Best-first anchor comparison is incomplete for {policy}.")
        if not (curve["n_valid"].eq(714).all() and curve["n_excluded"].eq(0).all()):
            raise ValueError(f"Best-first anchor comparison is not complete on 714 transcripts for {policy}.")
        if not np.allclose(curve.loc[curve.N.eq(2), ["mean", "ci_low", "ci_high"]], 1, rtol=0, atol=1e-12):
            raise ValueError(f"N=2 must be an exact self-comparison for {policy}.")
    style = publication_rc()
    style.update({
        "font.size": 15.0,
        "font.weight": "bold",
        "axes.labelsize": 15.8,
        "axes.labelweight": "bold",
        "axes.titlesize": 16.5,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.45,
        "legend.fontsize": 14.0,
        "xtick.labelsize": 14.0,
        "ytick.labelsize": 14.0,
        "xtick.major.width": 1.45,
        "ytick.major.width": 1.45,
    })
    if style["text.usetex"]:
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}"
        )
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))

        x = np.arange(len(panel))
        distributions = [
            equal_panel.loc[equal_panel["pair_label"].eq(pair), "PCC"].to_numpy(float)
            for pair in panel["pair_label"]
        ]
        violins = axes[0].violinplot(
            distributions, positions=x, widths=0.86,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for body in violins["bodies"]:
            body.set_facecolor("#56B4E9")
            body.set_edgecolor("#075985")
            body.set_linewidth(1.25)
            body.set_alpha(0.70)
        axes[0].vlines(x, panel.q05, panel.q95, color="#263238", linewidth=1.35, zorder=3)
        axes[0].vlines(x, panel.q25, panel.q75, color="white", linewidth=5.8, zorder=4)
        axes[0].vlines(x, panel.q25, panel.q75, color="#075985", linewidth=3.2, zorder=5)
        axes[0].scatter(x, panel["median"], color="#111827", s=46, zorder=6)
        axes[0].set_xticks(x, panel.pair_label, rotation=24, ha="right")
        axes[0].set_ylabel(r"Transcript PCC of $L_t$")
        axes[0].set_xlabel("Source-disjoint panel pair")
        axes[0].set_title("A  Independent-panel reproducibility", loc="left")
        axes[0].set_ylim(0.0, 1.005)
        style_axis(axes[0])

        positions = np.arange(len(sizes))
        marker_styles = {"equal": "o", "score_p1": "s", "score_p3": "^", "score_p5": "D"}
        line_styles = {"equal": "-", "score_p1": "--", "score_p3": "-.", "score_p5": ":"}
        for policy in POLICIES:
            curve = best_anchor.loc[
                best_anchor["reference_policy"].eq(policy)
            ].sort_values("N")
            y = curve["mean"].to_numpy(float)
            low = curve["ci_low"].to_numpy(float)
            high = curve["ci_high"].to_numpy(float)
            axes[1].errorbar(
                positions, y, yerr=np.vstack([y - low, high - y]),
                color=POLICY_COLORS[policy], marker=marker_styles[policy],
                linestyle=line_styles[policy], linewidth=2.7, elinewidth=1.6,
                markersize=6.5, markeredgewidth=1.15, capsize=2.5,
                label=POLICY_LABELS[policy].replace("Score weighted,", "Score,"),
            )
        axes[1].set_xticks(positions, sizes)
        axes[1].set_xlabel(r"Included datasets $N$, best-first")
        axes[1].set_ylabel(r"Mean PCC with own $N=2$ fit")
        axes[1].set_title(r"B  Agreement with the initial $N=2$ profile", loc="left")
        axes[1].legend(loc="lower left", frameon=False, ncol=2,
                       columnspacing=0.8, handlelength=2.1)
        axes[1].set_ylim(0.54, 1.015)
        style_axis(axes[1])
        fig.subplots_adjust(left=0.075, right=0.995, bottom=0.24, top=0.91, wspace=0.20)
        save_figure(fig, output_dir, "real_data_stability", dpi)

    panel.to_csv(output_dir / "real_data_panel_reproducibility_source.csv", index=False)
    equal_panel[["pair_label", "transcript_id", "PCC"]].to_csv(
        output_dir / "real_data_panel_reproducibility_per_transcript_source.csv", index=False
    )
    best_anchor.to_csv(output_dir / "real_data_best_first_own_anchor_source.csv", index=False)


def plot_directional_results(anchor: pd.DataFrame, direct: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    if not anchor.anchor_policy.eq(anchor.reference_policy).all() or not anchor.anchor_N.eq(2).all():
        raise ValueError("Every curve must use its own policy's N=2 anchor.")
    style = publication_rc()
    style.update({"font.size": 12.2, "axes.labelsize": 13.0, "axes.titlesize": 14.0,
                  "legend.fontsize": 11.0, "font.weight": "bold",
                  "axes.labelweight": "bold", "axes.titleweight": "bold", "axes.linewidth": 1.4})
    if style['text.usetex']:
        style['text.latex.preamble'] += r'\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}'
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5))
        sizes = [2, 5, 10, 20, 40, 80, 114]
        xmap = {n: index for index, n in enumerate(sizes)}
        for index, direction in enumerate(("best_first", "worst_first")):
            for policy in POLICIES:
                curve = anchor.loc[
                    anchor["selection_direction"].eq(direction)
                    & anchor["reference_policy"].eq(policy)
                ].sort_values("N")
                if curve.empty:
                    continue
                x = np.asarray([xmap[int(n)] for n in curve.N])
                axes[index].plot(
                    x, curve["mean"], "o-", color=POLICY_COLORS[policy],
                    linewidth=2.0, markersize=5.2, label=POLICY_LABELS[policy],
                )
                axes[index].fill_between(x, curve.ci_low, curve.ci_high,
                                         color=POLICY_COLORS[policy], alpha=0.10)
            title = "Best-first" if direction == "best_first" else "Worst-first"
            axes[index].set_title(f"{'A' if index == 0 else 'B'}  {title}: own-policy $N=2$ anchor", loc="left")
            axes[index].set_ylabel(r"Mean transcript PCC of $L_t$")
            axes[index].set_ylim(0.18 if index else 0.64, 1.015)
            if index == 0:
                axes[index].legend(frameon=False, loc="lower left")
        for axis in axes:
            axis.set_xticks(np.arange(len(sizes)), sizes)
            axis.set_xlim(-0.25, len(sizes) - 0.75)
            axis.set_xlabel("Selected datasets $N$")
            style_axis(axis)
        fig.subplots_adjust(left=0.065, right=0.995, bottom=0.17, top=0.91, wspace=0.21)
        save_figure(fig, output_dir, "cumulative_directional_stability", dpi)


def plot_reference_geometry(geometry: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    frame = geometry.copy()
    shared = frame.loc[frame.selection_direction.eq('shared_full') & frame.reference_policy.eq('equal') & frame.N.eq(114)]
    if len(shared) != 1:
        raise ValueError('Expected the single shared full-collection uniform reference.')
    frame = frame.loc[frame["selection_direction"].isin(["best_first", "worst_first"])]
    # This is the same physical fit at the endpoint of both selection paths.
    frame = pd.concat([frame, shared.assign(selection_direction='best_first'),
                       shared.assign(selection_direction='worst_first')], ignore_index=True)
    frame.to_csv(output_dir / 'cumulative_reference_geometry_plotted_source.csv', index=False)
    style = publication_rc()
    style.update({"font.size": 12.2, "axes.labelsize": 13.0, "axes.titlesize": 14.0,
                  "legend.fontsize": 11.0, "font.weight": "bold",
                  "axes.labelweight": "bold", "axes.titleweight": "bold", "axes.linewidth": 1.4})
    if style['text.usetex']:
        style['text.latex.preamble'] += r'\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}'
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
        sizes = [2, 5, 10, 20, 40, 80, 114]
        xmap = {n: index for index, n in enumerate(sizes)}
        for direction, line_style in (("best_first", "-"), ("worst_first", "--")):
            for policy in POLICIES:
                curve = frame.loc[
                    frame["selection_direction"].eq(direction)
                    & frame["reference_policy"].eq(policy)
                ].sort_values("N")
                if curve.N.tolist() != sizes:
                    raise ValueError(f'Incomplete reference geometry: {direction}/{policy}')
                x = np.asarray([xmap[int(n)] for n in curve.N])
                label = POLICY_LABELS[policy] if direction == "best_first" else None
                axes[0].plot(x, curve.effective_fraction, marker="o", linestyle=line_style,
                             color=POLICY_COLORS[policy], linewidth=1.9, markersize=4.7, label=label)
                axes[1].plot(x, curve.weighted_mean_rank, marker="o", linestyle=line_style,
                             color=POLICY_COLORS[policy], linewidth=1.9, markersize=4.7)
        axes[0].set_title(r"A  Reference concentration $N_{\rm eff}/N$", loc="left")
        axes[0].set_ylabel("Effective reference fraction")
        axes[0].set_ylim(0, 1.04)
        axes[1].set_title("B  Reference location along the QC ordering", loc="left")
        axes[1].set_ylabel("Reference-weighted global QC rank")
        axes[1].set_ylim(0, 116)
        for axis in axes:
            axis.set_xticks(np.arange(len(sizes)), sizes)
            axis.set_xlim(-0.25, len(sizes) - 0.75)
            axis.set_xlabel("Selected datasets $N$")
            style_axis(axis)
        axes[0].legend(frameon=False, fontsize=9.2, ncol=2, loc="lower left")
        # Direction line styles are documented in the caption, not over data.
        fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.91, wspace=0.18)
        save_figure(fig, output_dir, "cumulative_reference_geometry", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-metrics", type=Path,
        default=ROOT / "figures/assets_5_real_datasets_4_panels/main_text_four_panel_equal_source.csv",
    )
    parser.add_argument(
        "--cumulative-dir", type=Path,
        default=ROOT / "analyses/artifacts/real_data/cumulative_selection_quality_score",
    )
    parser.add_argument(
        "--membership", type=Path,
        default=ROOT / "results/cumulative_selection_direction_quality_score_directional_seed42/inputs/collection_membership.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "analyses/artifacts/real_data/manuscript_revision/cumulative_quality_score",
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--main-only", action="store_true", help="Regenerate only the main-text figure, leaving appendix assets unchanged.")
    args = parser.parse_args()

    paths = {
        "panel_metrics": args.panel_metrics.resolve(),
        "adjacent": (args.cumulative_dir / "adjacent_summary.csv").resolve(),
        "anchor": (args.cumulative_dir / "own_policy_anchor_summary.csv").resolve(),
        "direct": (args.cumulative_dir / "best_vs_worst_summary.csv").resolve(),
        "geometry": (args.cumulative_dir / "reference_geometry.csv").resolve(),
        "membership": args.membership.resolve(),
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    require_columns(frames["panel_metrics"], {"arm", "pair_label", "transcript_id", "PCC"}, "panel metrics")
    require_columns(frames["adjacent"], {"selection_direction", "reference_policy", "N_a", "N_b", "n_valid", "n_excluded", "mean", "ci_low", "ci_high"}, "adjacent summary")
    require_columns(frames["anchor"], {"selection_direction", "reference_policy", "N", "mean", "ci_low", "ci_high"}, "anchor summary")
    require_columns(frames["direct"], {"reference_policy", "N", "mean", "ci_low", "ci_high"}, "direct summary")
    require_columns(frames["geometry"], {"selection_direction", "reference_policy", "N", "effective_fraction", "weighted_mean_rank"}, "reference geometry")
    require_columns(frames["membership"], {"selection_direction", "N", "dataset_id"}, "membership")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    panel = panel_summary(frames["panel_metrics"])
    overlap = selection_overlap(frames["membership"])
    plot_main(frames["panel_metrics"], panel, frames["anchor"], out, args.dpi)
    if not args.main_only:
        plot_directional_results(frames["anchor"], frames["direct"], out, args.dpi)
        plot_reference_geometry(frames["geometry"], out, args.dpi)
    frames["anchor"].to_csv(out / "cumulative_anchor_summary_source.csv", index=False)
    frames["direct"].merge(overlap, on="N", how="left").to_csv(
        out / "cumulative_best_vs_worst_source.csv", index=False
    )
    frames["geometry"].to_csv(out / "cumulative_reference_geometry_source.csv", index=False)
    (out / "provenance.json").write_text(json.dumps({
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "models_retrained": False,
        "checkpoint_selection": "best validation loss, inherited from audited source analyses",
        "cumulative_test_transcripts": 714,
        "panel_test_transcripts": 1593,
        "bootstrap_intervals": "95% whole-transcript intervals copied from audited cumulative summaries",
        "main_panel_a": "six pairwise distributions from four balanced source-disjoint panels; violins show complete transcript-level PCC distributions",
        "main_panel_b": "best-first agreement with each reference policy's own N=2 fit; no shared or cross-policy anchor",
        "nested_panel_warning": "nested collections: agreement measures retention of the initial representation, not independent-panel evidence or biological accuracy; N=2 is a self-comparison",
    }, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
