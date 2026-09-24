#!/usr/bin/env python3
"""Create the compact equal-reference synthetic component-recovery figure.

The script reads the audited scalar summary produced by
``analyze_synthetic_reference_target_audit.py``.  It does not load model
checkpoints or recompute transcript metrics.  The main figure reports PCC and
RMSE for end-to-end L--qbar recovery at three separate depths, with the
secondary PCC(L, K) kinetic-alignment audit in its own panel.  The correction
panel reads ``analyze_synthetic_gamma_matched_depths.py`` outputs: PCC and RMSE
for the same 27 best-validation-loss fits and ten-codon masks, with
constant-target transcripts excluded on a fixed within-depth cohort across N.

PCC and RMSE are shown in the same *panel* but never on the same numerical
axis: each panel has an explicitly labelled left PCC axis and right error axis.
This compact layout must not be read as equating their vertical scales.
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Utils.publication_plot_style import publication_rc

DEPTH_ORDER = ("0p25_per_codon", "2_per_codon", "20_per_codon")
DEPTH_LABELS = {
    "0p25_per_codon": "0.25 reads/codon",
    "2_per_codon": "2 reads/codon",
    "20_per_codon": "20 reads/codon",
}
DEPTH_STYLE = {
    "0p25_per_codon": ("#0072B2", "o", "--"),
    "2_per_codon": ("#E69F00", "s", ":"),
    "20_per_codon": ("#009E73", "^", "-"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_summary(frame: pd.DataFrame) -> None:
    required = {
        "family", "depth", "n_datasets", "reference_weighting", "comparison",
        "metric", "n_bias_families", "run_id", "n_transcripts", "median", "bootstrap_ci_low", "bootstrap_ci_high",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Aggregate summary is missing columns: {missing}")
    selected = frame.loc[
        frame["comparison"].eq("analysis3_L_vs_Q")
        & frame["metric"].isin({"pearson", "clr_rmse"})
    ]
    if selected.empty:
        raise ValueError("No audited L--qbar PCC/RMSE summaries were found.")
    values = selected[["median", "bootstrap_ci_low", "bootstrap_ci_high"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Recovery summaries contain non-finite values.")
    if np.any(values[:, 1] > values[:, 0]) or np.any(values[:, 0] > values[:, 2]):
        raise ValueError("Bootstrap intervals do not enclose their medians.")
    pcc = selected.loc[selected.metric.eq("pearson"), [
        "median", "bootstrap_ci_low", "bootstrap_ci_high"
    ]].to_numpy(float)
    rmse = selected.loc[selected.metric.eq("clr_rmse"), [
        "median", "bootstrap_ci_low", "bootstrap_ci_high"
    ]].to_numpy(float)
    if np.any(pcc < -1) or np.any(pcc > 1) or np.any(rmse < 0):
        raise ValueError("PCC or RMSE lies outside its mathematical domain.")


def validate_gamma(gamma: pd.DataFrame) -> None:
    required = {
        "run_id", "depth", "n_datasets", "reference_weighting",
        "n_transcripts", "n_cohort_transcripts", "excluded_transcripts",
        "pcc_median", "pcc_bootstrap_ci_low", "pcc_bootstrap_ci_high",
        "log_rmse_median", "log_rmse_bootstrap_ci_low",
        "log_rmse_bootstrap_ci_high", "calibration_slope_median",
        "checkpoint_variant", "boundary_trim_codons",
    }
    missing = sorted(required.difference(gamma.columns))
    if missing:
        raise ValueError(f"Gamma summary is missing columns: {missing}")
    if (not gamma.checkpoint_variant.eq("best_val_loss").all()
            or not gamma.boundary_trim_codons.eq(10).all()
            or not gamma.reference_weighting.eq("equal").all()
            or len(gamma) != 27):
        raise ValueError(
            "Gamma must contain 27 equal-reference best-val-loss runs with ten-codon masks."
        )
    pcc = gamma[["pcc_median", "pcc_bootstrap_ci_low", "pcc_bootstrap_ci_high"]].to_numpy(float)
    rmse = gamma[[
        "log_rmse_median", "log_rmse_bootstrap_ci_low", "log_rmse_bootstrap_ci_high"
    ]].to_numpy(float)
    if (not np.isfinite(pcc).all() or not np.isfinite(rmse).all()
            or np.any(pcc < -1) or np.any(pcc > 1) or np.any(rmse < 0)):
        raise ValueError("Gamma PCC or RMSE summary is invalid.")
    if (np.any(pcc[:, 1] > pcc[:, 0]) or np.any(pcc[:, 0] > pcc[:, 2])
            or np.any(rmse[:, 1] > rmse[:, 0]) or np.any(rmse[:, 0] > rmse[:, 2])):
        raise ValueError("Gamma bootstrap intervals do not enclose their medians.")


def validate_kinetic(kinetic: pd.DataFrame) -> None:
    required = {
        "run_id", "depth", "n_datasets", "reference_weighting", "metric",
        "comparison", "n_transcripts", "n_undefined", "median",
        "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_replicates",
    }
    missing = sorted(required.difference(kinetic.columns))
    if missing:
        raise ValueError(f"Kinetic-alignment summary is missing columns: {missing}")
    if (
        len(kinetic) != 27
        or not kinetic.reference_weighting.eq("equal").all()
        or not kinetic.metric.eq("pearson").all()
        or not kinetic.comparison.eq("L_vs_K").all()
        or not kinetic.n_undefined.eq(0).all()
    ):
        raise ValueError(
            "Kinetic alignment must contain 27 defined equal-reference L--K PCC summaries."
        )
    values = kinetic[["median", "bootstrap_ci_low", "bootstrap_ci_high"]].to_numpy(float)
    if (
        not np.isfinite(values).all()
        or np.any(values < -1)
        or np.any(values > 1)
        or np.any(values[:, 1] > values[:, 0])
        or np.any(values[:, 0] > values[:, 2])
    ):
        raise ValueError("Kinetic PCC summaries or bootstrap intervals are invalid.")
    for depth in DEPTH_ORDER:
        if kinetic.loc[kinetic.depth.eq(depth), "n_datasets"].sort_values().tolist() != list(range(2, 11)):
            raise ValueError(f"Incomplete kinetic-alignment curve: {depth}")


def _errorbar(
    axis,
    frame: pd.DataFrame,
    value: str,
    low: str,
    high: str,
    *,
    color: str,
    marker: str,
    linestyle: str,
    markerfacecolor: str | None = None,
    alpha: float = 1.0,
) -> None:
    x = frame.n_datasets.to_numpy()
    y = frame[value].to_numpy(float)
    axis.errorbar(
        x,
        y,
        yerr=np.vstack([y - frame[low].to_numpy(float), frame[high].to_numpy(float) - y]),
        color=color,
        marker=marker,
        markerfacecolor=markerfacecolor,
        markeredgecolor=color,
        markeredgewidth=1.35,
        linestyle=linestyle,
        linewidth=2.45,
        markersize=6.0,
        capsize=2.0,
        alpha=alpha,
        zorder=3,
    )


def plot(
    summary: pd.DataFrame,
    gamma: pd.DataFrame,
    kinetic: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> pd.DataFrame:
    source = summary.loc[
        summary["comparison"].eq("analysis3_L_vs_Q")
        & summary["metric"].isin({"pearson", "clr_rmse"})
        & summary["reference_weighting"].eq("equal")
        & summary["family"].eq("within_depth")
    ].copy()
    within = source.loc[source["metric"].eq("pearson")]
    if set(within["depth"].dropna()) != set(DEPTH_ORDER):
        raise ValueError("The within-depth summary does not contain all three depths.")
    validate_gamma(gamma)
    validate_kinetic(kinetic)
    matched = within.merge(gamma, on="run_id", suffixes=("_shared", "_gamma"), validate="one_to_one")
    if (len(matched) != 27
            or not matched.n_transcripts_shared.eq(matched.n_cohort_transcripts).all()
            or not matched.n_transcripts_gamma.add(matched.excluded_transcripts).eq(matched.n_cohort_transcripts).all()
            or not gamma.groupby("depth").n_transcripts.nunique().eq(1).all()):
        raise ValueError("Gamma must start from the A/C cohorts and document fixed-cohort exclusions.")
    if set(within.run_id) != set(kinetic.run_id):
        raise ValueError("The L--K audit and L--qbar source do not describe the same 27 fits.")

    style = publication_rc()
    style.update({
        "font.size": 17.0,
        "font.weight": "bold",
        "axes.labelsize": 17.0,
        "axes.labelweight": "bold",
        "axes.titlesize": 19.0,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.45,
        "legend.fontsize": 14.5,
        "xtick.labelsize": 15.0,
        "ytick.labelsize": 15.0,
        "xtick.major.width": 1.45,
        "ytick.major.width": 1.45,
    })
    if style["text.usetex"]:
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}"
        )
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(14.2, 4.0),
            gridspec_kw={"width_ratios": [1.08, 0.90, 1.08]},
        )
        shared_error_axis = axes[0].twinx()
        gamma_error_axis = axes[2].twinx()

        # Panel A: proximal shared-profile recovery.  Color encodes depth;
        # line/fill encodes the metric.
        for depth in DEPTH_ORDER:
            color, marker, _ = DEPTH_STYLE[depth]
            q_pcc = source.loc[
                source.metric.eq("pearson") & source.depth.eq(depth)
            ].sort_values("n_datasets")
            q_rmse = source.loc[
                source.metric.eq("clr_rmse") & source.depth.eq(depth)
            ].sort_values("n_datasets")
            expected = list(range(2, 11))
            if (
                q_pcc.n_bias_families.tolist() != expected
                or q_rmse.n_bias_families.tolist() != expected
            ):
                raise ValueError(f"Incomplete equal-reference L--qbar curve: {depth}")
            _errorbar(
                axes[0], q_pcc, "median", "bootstrap_ci_low", "bootstrap_ci_high",
                color=color, marker=marker, linestyle="-", markerfacecolor=color,
            )
            _errorbar(
                shared_error_axis, q_rmse, "median", "bootstrap_ci_low", "bootstrap_ci_high",
                color=color, marker=marker, linestyle="--", markerfacecolor="white",
                alpha=0.78,
            )

        # Panel B: upstream programmed-kinetic alignment, separated from panel
        # A so it does not compete with the shared-profile PCC/RMSE encoding.
        for depth in DEPTH_ORDER:
            color, marker, _ = DEPTH_STYLE[depth]
            k_pcc = kinetic.loc[kinetic.depth.eq(depth)].sort_values("n_datasets")
            _errorbar(
                axes[1], k_pcc, "median", "bootstrap_ci_low", "bootstrap_ci_high",
                color=color, marker=marker, linestyle="-", markerfacecolor=color,
            )

        # Panel C: correction shape and amplitude error in the identifiable
        # equal-reference, position-centered log gauge.
        for depth in DEPTH_ORDER:
            curve = gamma.loc[gamma.depth.eq(depth)].sort_values("n_datasets")
            if curve.n_datasets.tolist() != list(range(2, 11)):
                raise ValueError(f"Incomplete gamma curve: {depth}")
            color, marker, _ = DEPTH_STYLE[depth]
            _errorbar(
                axes[2], curve, "pcc_median", "pcc_bootstrap_ci_low",
                "pcc_bootstrap_ci_high", color=color, marker=marker,
                linestyle="-", markerfacecolor=color,
            )
            _errorbar(
                gamma_error_axis, curve, "log_rmse_median",
                "log_rmse_bootstrap_ci_low", "log_rmse_bootstrap_ci_high",
                color=color, marker=marker, linestyle="--",
                markerfacecolor="white", alpha=0.78,
            )

        axes[0].set_title(
            r"A  $\widehat L_t$ vs. $\bar q_t$",
            loc="left",
            y=1.29,
        )
        axes[1].set_title(
            r"B  $\widehat L_t$ vs. $K_t$",
            loc="left",
            y=1.29,
        )
        axes[2].set_title(
            r"C  $\log\widehat\gamma_{t,d}$ vs. $g^\star_{t,d}$",
            loc="left",
            y=1.29,
        )
        axes[0].set_ylabel(r"Median PCC $\uparrow$")
        shared_error_axis.set_ylabel(r"Median RMSE $\downarrow$", labelpad=11)
        axes[1].set_ylabel(r"Median PCC $\uparrow$")
        axes[2].set_ylabel(r"Median PCC $\uparrow$")
        gamma_error_axis.set_ylabel(r"Median RMSE $\downarrow$", labelpad=11)

        shared_pcc_min = float(
            source.loc[source.metric.eq("pearson"), "bootstrap_ci_low"].min()
        )
        shared_lower = max(-1.0, np.floor((shared_pcc_min - 0.015) * 20.0) / 20.0)
        axes[0].set_ylim(shared_lower, 1.005)
        # Add display-only headroom above the largest uncertainty bound.  This
        # keeps the RMSE trajectories visually distinct from the PCC curves on
        # the overlaid twin axes without changing either statistic.
        shared_rmse_ceiling = np.ceil(
            source.loc[source.metric.eq("clr_rmse"), "bootstrap_ci_high"].max()
            * 1.20
            * 20.0
        ) / 20.0
        shared_error_axis.set_ylim(0, max(0.05, shared_rmse_ceiling))
        kinetic_lower = max(
            -1.0,
            np.floor((float(kinetic.bootstrap_ci_low.min()) - 0.015) * 20.0) / 20.0,
        )
        axes[1].set_ylim(kinetic_lower, 1.005)
        lower = np.floor((gamma.pcc_bootstrap_ci_low.min() - .0005) * 1000) / 1000
        axes[2].set_ylim(lower, 1.001)
        axes[2].set_yticks([.996, .998, 1.000])
        gamma_rmse_ceiling = max(
            .04,
            np.ceil(gamma.log_rmse_bootstrap_ci_high.max() * 1.20 * 100) / 100,
        )
        gamma_error_axis.set_ylim(0, gamma_rmse_ceiling)

        for axis in axes:
            axis.set_xlabel("Included datasets $N$")
            axis.set_xticks([2, 4, 6, 8, 10])
            axis.spines[["top", "right"]].set_visible(False)

        for error_axis in (shared_error_axis, gamma_error_axis):
            error_axis.spines[["top", "left"]].set_visible(False)
            error_axis.spines["right"].set_linewidth(1.45)
            error_axis.tick_params(axis="y", width=1.45, colors="#4D4D4D")
            error_axis.yaxis.label.set_color("#4D4D4D")

        shared_metric_handles = [
            Line2D([], [], color="black", marker="o", linewidth=2.45,
                   label="PCC"),
            Line2D([], [], color="black", marker="o", markerfacecolor="white",
                   linestyle="--", linewidth=2.45,
                   label="RMSE"),
        ]
        gamma_metric_handles = [
            Line2D([], [], color="black", marker="o", linewidth=2.45,
                   label="PCC"),
            Line2D([], [], color="black", marker="o", markerfacecolor="white",
                   linestyle="--", linewidth=2.45,
                   label="RMSE"),
        ]
        axes[0].legend(
            handles=shared_metric_handles, loc="lower left", bbox_to_anchor=(0.0, 1.015),
            ncol=2, frameon=False, fontsize=13.5, handlelength=1.8,
            columnspacing=0.7, borderaxespad=0.0, labelspacing=0.35,
        )
        axes[2].legend(
            handles=gamma_metric_handles, loc="lower left", bbox_to_anchor=(0.0, 1.015),
            ncol=2, frameon=False, fontsize=13.5, handlelength=1.8,
            columnspacing=0.9, borderaxespad=0.0,
        )
        depth_handles = [
            Line2D([], [], color=DEPTH_STYLE[depth][0], marker=DEPTH_STYLE[depth][1],
                   linewidth=2.6, label=DEPTH_LABELS[depth])
            for depth in DEPTH_ORDER
        ]
        fig.legend(
            handles=depth_handles, loc="lower center", ncol=3, frameon=False,
            fontsize=15.0, bbox_to_anchor=(0.5, 0.002),
        )
        # Reserve explicit space for the right-hand RMSE label; relying on a
        # tight bounding box makes the exported PDF dimensions less stable.
        fig.subplots_adjust(left=.055, right=.905, bottom=.24, top=.72, wspace=.82)
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "png"):
            fig.savefig(output_dir / f"synthetic_occupancy_recovery_compact.{suffix}", dpi=dpi)
        plt.close(fig)
        shared_source = source.copy()
        shared_source["source_quantity"] = "L_vs_qbar"
        kinetic_source = kinetic.copy()
        kinetic_source["source_quantity"] = "L_vs_K"
        pd.concat([shared_source, kinetic_source], ignore_index=True, sort=False).to_csv(
            output_dir / "synthetic_occupancy_recovery_compact_source.csv", index=False
        )
        gamma.to_csv(output_dir / "synthetic_gamma_recovery_source.csv", index=False)
        return pd.concat([shared_source, kinetic_source], ignore_index=True, sort=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "analyses/artifacts/synthetic/manuscript_revision/reference_target_best_val_loss/aggregate_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analyses/artifacts/manuscript_main_figures_20260924/synthetic",
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--gamma-summary",
        type=Path,
        default=ROOT / "analyses/artifacts/manuscript_main_figures_20260923/gamma_pcc_rmse/gamma_summary.csv",
    )
    parser.add_argument(
        "--kinetic-summary",
        type=Path,
        default=ROOT / "analyses/artifacts/manuscript_main_figures_20260924/kinetic_alignment/L_vs_K_summary.csv",
    )
    args = parser.parse_args()
    summary_path = args.summary.resolve()
    frame = pd.read_csv(summary_path)
    validate_summary(frame)
    kinetic_path = args.kinetic_summary.resolve()
    source = plot(
        frame,
        pd.read_csv(args.gamma_summary),
        pd.read_csv(kinetic_path),
        args.output_dir.resolve(),
        args.dpi,
    )
    provenance = {
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "input": str(summary_path),
        "input_sha256": sha256(summary_path),
        "comparisons": [
            "analysis3_L_vs_Q",
            "L_vs_K",
            "log_gamma_vs_centered_log_bias",
        ],
        "metrics": [
            "per-transcript PCC",
            "per-transcript RMSE between position-centered log L and qbar",
            "per-transcript PCC for mean-one L and upstream K",
            "per-transcript joint dataset-by-position RMSE for centered log gamma and g-star",
        ],
        "aggregation": "median across transcripts",
        "interval": "95% whole-transcript bootstrap interval from audited source table",
        "rows_plotted": int(len(source)),
        "panel_a": (
            "equal-reference PCC(L, qbar) and RMSE of their position-centered "
            "log profiles; PCC and error use separate labelled axes"
        ),
        "panel_b": "equal-reference PCC(L, K) on its own numerical axis",
        "panel_c": (
            "equal-reference PCC and RMSE for centered log gamma and g-star; "
            "PCC and error use separate labelled axes"
        ),
        "gamma_cohort": "starts from the shared-profile cohorts and exact ten-codon masks; excludes constant targets on a fixed cohort across N for both PCC and RMSE",
        "gamma_target": (
            "log b and saved log gamma are each transformed by the same equal-weight "
            "cross-dataset reference centering and positional mean centering"
        ),
        "shared_profile_error": (
            "RMSE after log transforming the retained positive profiles and "
            "subtracting each profile's positional mean log value"
        ),
        "gamma_error": "RMSE over the complete centered-log dataset-by-position array within transcript",
        "gamma_input": str(args.gamma_summary.resolve()),
        "gamma_input_sha256": sha256(args.gamma_summary),
        "kinetic_input": str(kinetic_path),
        "kinetic_input_sha256": sha256(kinetic_path),
        "kinetic_scope": (
            "secondary upstream-kinetic alignment on the exact L-qbar cohorts and "
            "ten-codon masks; K is not the direct fixed-reference estimand"
        ),
        "gamma_bootstrap": "2000 whole-transcript resamples; common indices across N within depth; seeds 20260922, 20260923, 20260924",
        "models_retrained": False,
        "checkpoint_variant": "best validation loss (inherited from audited summary)",
        "dual_axis_warning": (
            "PCC and RMSE share panels only; their separate y axes have different units "
            "and vertical positions/slopes must not be compared across axes"
        ),
        "rmse_axis_headroom": (
            "display ceilings are rounded upward after adding 20% above the "
            "largest RMSE bootstrap upper bound; estimates are unchanged"
        ),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(args.output_dir / "synthetic_occupancy_recovery_compact.pdf")


if __name__ == "__main__":
    main()
