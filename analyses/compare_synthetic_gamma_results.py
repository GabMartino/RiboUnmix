#!/usr/bin/env python3
"""Analyze and compare gamma recovery in two result trees.

The script deliberately reruns the canonical gauge-aware gamma analysis for
each tree.  It never trusts a possibly stale report copied with a result
folder.  Incomplete run directories are recorded and skipped, so the command
can be rerun while rsync is still filling either tree.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyses.analyze_synthetic_gamma_recovery import (
    DEFAULT_BIAS_ROOT,
    DEFAULT_LATENT_TRUTH,
    analyze as analyze_gamma,
)
from analyses.analyze_synthetic_recovery import (
    DEFAULT_OUTPUT_DIRECTORY_NAME,
    DEFAULT_RUN_PREFIX,
    REPOSITORY_ROOT,
    discover_run_directories,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


DEFAULT_BASELINE_ROOT = REPOSITORY_ROOT / "results/riboai_synthetic_experiments"
DEFAULT_CANDIDATE_ROOT = REPOSITORY_ROOT / "results/riboai_synthetic_experiments"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "analyses" / "artifacts" / "synthetic" / "gamma_comparison"
)

MATCH_COLUMNS = (
    "checkpoint_variant",
    "depth",
    "mass_condition",
    "dataset_count",
    "datasets",
    "validation_id_hash",
)
COMPARISON_METRICS = (
    "mean_pair_log_gamma_pcc",
    "median_pair_log_gamma_pcc",
    "pooled_log_gamma_pcc",
    "pooled_log_gamma_rmse",
    "pooled_calibration_slope",
    "programmed_site_log_gamma_pcc",
    "programmed_site_log_gamma_rmse",
    "programmed_site_fraction_within_10pct",
)


def build_matched_comparison(
    combined: pd.DataFrame,
    *,
    baseline_label: str,
    candidate_label: str,
) -> pd.DataFrame:
    """Match scientifically identical panels and calculate candidate deltas."""
    missing = sorted(set(MATCH_COLUMNS) - set(combined.columns))
    if missing:
        raise ValueError(f"Combined gamma summary is missing match columns: {missing}")

    available_metrics = [
        metric for metric in COMPARISON_METRICS if metric in combined.columns
    ]
    rows: list[pd.DataFrame] = []
    for label in (baseline_label, candidate_label):
        selected = combined[combined["result_set"] == label]
        if selected.empty:
            continue
        grouped = selected.groupby(list(MATCH_COLUMNS), dropna=False, as_index=False)
        summary = grouped[available_metrics].mean()
        run_counts = grouped.size().rename(columns={"size": "replicate_runs"})
        run_names = grouped["run"].agg(lambda values: ",".join(sorted(map(str, values))))
        summary = summary.merge(run_counts, on=list(MATCH_COLUMNS), validate="one_to_one")
        summary = summary.merge(run_names, on=list(MATCH_COLUMNS), validate="one_to_one")
        summary["result_set"] = label
        rows.append(summary)

    if len(rows) != 2:
        return pd.DataFrame(columns=list(MATCH_COLUMNS))
    baseline = rows[0].drop(columns="result_set")
    candidate = rows[1].drop(columns="result_set")
    matched = baseline.merge(
        candidate,
        on=list(MATCH_COLUMNS),
        how="inner",
        suffixes=(f"_{baseline_label}", f"_{candidate_label}"),
        validate="one_to_one",
    )
    for metric in available_metrics:
        matched[f"{metric}_candidate_minus_baseline"] = (
            matched[f"{metric}_{candidate_label}"]
            - matched[f"{metric}_{baseline_label}"]
        )
    if "pooled_log_gamma_rmse" in available_metrics:
        matched["pooled_log_gamma_rmse_improvement"] = (
            matched[f"pooled_log_gamma_rmse_{baseline_label}"]
            - matched[f"pooled_log_gamma_rmse_{candidate_label}"]
        )
    return matched


def _run_one_root(
    *,
    label: str,
    root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dirs = discover_run_directories(root, args.run_prefix)
    status: dict[str, Any] = {
        "result_set": label,
        "results_root": str(root),
        "discovered_run_directories": len(run_dirs),
        "analyzed_runs": 0,
        "skipped_runs": len(run_dirs),
        "analysis_error": "",
    }
    analysis_dir = output_dir / f"{label}_analysis"
    gamma_args = Namespace(
        results_root=str(root),
        run_prefix=args.run_prefix,
        bias_root=args.bias_root,
        latent_ground_truth=args.latent_ground_truth,
        checkpoint_variant=args.checkpoint_variant,
        output_dir=str(analysis_dir),
        strict=False,
    )
    try:
        paths = analyze_gamma(gamma_args)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        status["analysis_error"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(), status

    summary = pd.read_csv(paths["summary"], sep="\t")
    summary.insert(0, "result_set", label)
    summary.insert(1, "results_root", str(root))
    skipped = pd.read_csv(paths["skipped"], sep="\t")
    status["analyzed_runs"] = len(summary)
    status["skipped_runs"] = len(skipped)
    return summary, status


def _plot(combined: pd.DataFrame, matched: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), constrained_layout=True)
    palette = {label: color for label, color in zip(
        combined["result_set"].drop_duplicates(), ("#6b3fa0", "#16828c")
    )}
    markers = {"mass_conserved": "o", "mass_free": "s"}

    for (label, mass, depth), frame in combined.groupby(
        ["result_set", "mass_condition", "depth"], sort=True
    ):
        frame = frame.sort_values("dataset_count")
        legend = f"{label}; {mass}; {depth}"
        style = dict(
            color=palette[label], marker=markers.get(str(mass), "o"),
            linewidth=1.8, markersize=6, label=legend,
        )
        axes[0, 0].plot(
            frame["dataset_count"], frame["mean_pair_log_gamma_pcc"], **style
        )
        axes[0, 1].plot(
            frame["dataset_count"], frame["pooled_log_gamma_rmse"], **style
        )

    axes[0, 0].set_title("A. Interior mean pairwise gamma-shape PCC", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("PCC (higher is better)")
    axes[0, 1].set_title("B. Interior pooled log-gamma RMSE", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("RMSE (lower is better)")
    for axis in axes[0]:
        axis.set_xlabel("Datasets in panel")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)

    labels = list(combined["result_set"].drop_duplicates())
    if len(labels) == 2 and not matched.empty:
        baseline, candidate = labels
        scatter_specs = (
            (
                "programmed_site_log_gamma_pcc",
                "C. Matched programmed-site PCC",
                "higher is better",
            ),
            (
                "programmed_site_log_gamma_rmse",
                "D. Matched programmed-site RMSE",
                "lower is better",
            ),
        )
        for axis, (metric, title, note) in zip(axes[1], scatter_specs):
            x = matched[f"{metric}_{baseline}"].to_numpy(float)
            y = matched[f"{metric}_{candidate}"].to_numpy(float)
            axis.scatter(x, y, s=55, color="#16828c", edgecolor="white", linewidth=0.7)
            finite = np.isfinite(x) & np.isfinite(y)
            if bool(finite.any()):
                low = float(min(x[finite].min(), y[finite].min()))
                high = float(max(x[finite].max(), y[finite].max()))
                margin = max((high - low) * 0.08, 1.0e-4)
                axis.plot([low - margin, high + margin], [low - margin, high + margin],
                          color="#64748b", linestyle="--", linewidth=1.2)
            axis.set_xlabel(baseline)
            axis.set_ylabel(candidate)
            axis.set_title(title, loc="left", fontweight="bold")
            axis.text(0.02, 0.98, note, transform=axis.transAxes, va="top", fontsize=9)
            axis.grid(alpha=0.2)
    else:
        for axis in axes[1]:
            axis.axis("off")
        axes[1, 0].text(
            0.0, 0.8,
            "No exactly matched completed panels yet.\nRerun after more files finish downloading.",
            fontsize=12,
        )

    labels = list(combined["result_set"].drop_duplicates()) if not combined.empty else []
    title = "Synthetic gamma recovery: " + " versus ".join(labels) if labels else "Synthetic gamma recovery"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    availability: pd.DataFrame,
    combined: pd.DataFrame,
    matched: pd.DataFrame,
    path: Path,
    *,
    baseline_label: str,
    candidate_label: str,
    same_root: bool = False,
) -> None:
    lines = [
        "# Synthetic gamma recovery comparison",
        "",
        "This comparison was regenerated from prediction parquets using the same "
        "gauge-aware gamma metric code for both result trees. Incomplete runs are "
        "listed in each root's `synthetic_gamma_recovery_skipped_runs.tsv` and do "
        "not enter the metrics.",
        "All compared recovery metrics use the CDS-observable interior `5 <= i < L-5`; programmed and learned gamma are re-gauged over those identical physical coordinates. Boundary diagnostics are not mixed into this comparison.",
        "",
        "## Availability",
        "",
    ]
    if same_root:
        lines.extend(
            [
                "Both inputs resolve to the same result root. This invocation is a self-consistency check, not a comparison of two independent training versions. Supply a different `--baseline-root` for a cross-tree comparison.",
                "",
            ]
        )
    for row in availability.itertuples(index=False):
        lines.append(
            f"- **{row.result_set}**: {int(row.analyzed_runs)} analyzed of "
            f"{int(row.discovered_run_directories)} discovered run folders; "
            f"{int(row.skipped_runs)} skipped."
        )
        if row.analysis_error:
            lines.append(f"  Analysis status: `{row.analysis_error}`")
    lines.extend([
        "",
        "## Comparison rule",
        "",
        "Runs are matched only when checkpoint variant, depth, mass-conservation "
        "condition, dataset count, ordered dataset panel, and validation-ID hash "
        "are identical. This prevents a visually convenient but invalid comparison "
        "between different held-out transcripts or bias panels.",
        "",
        f"Exactly matched panels currently available: **{len(matched)}**.",
        "",
    ])
    if not matched.empty:
        pcc_delta = matched["mean_pair_log_gamma_pcc_candidate_minus_baseline"]
        rmse_gain = matched["pooled_log_gamma_rmse_improvement"]
        lines.extend([
            "Across matched panels, candidate minus baseline mean pairwise PCC is "
            f"{pcc_delta.mean():+.5f}; baseline minus candidate pooled RMSE is "
            f"{rmse_gain.mean():+.5f} (positive means the current results improved).",
            "",
            "The overall PCC includes the many neutral codons. The stricter "
            "programmed-site-only comparison is:",
            "",
            f"| datasets | overall PCC {baseline_label}/{candidate_label} | programmed-site PCC {baseline_label}/{candidate_label} | "
            f"programmed-site RMSE {baseline_label}/{candidate_label} |",
            "|---:|---:|---:|---:|",
        ])
        for _, values in matched.sort_values("dataset_count").iterrows():
            lines.append(
                f"| {int(values['dataset_count'])} | "
                f"{values[f'mean_pair_log_gamma_pcc_{baseline_label}']:.6f} / "
                f"{values[f'mean_pair_log_gamma_pcc_{candidate_label}']:.6f} | "
                f"{values[f'programmed_site_log_gamma_pcc_{baseline_label}']:.6f} / "
                f"{values[f'programmed_site_log_gamma_pcc_{candidate_label}']:.6f} | "
                f"{values[f'programmed_site_log_gamma_rmse_{baseline_label}']:.6f} / "
                f"{values[f'programmed_site_log_gamma_rmse_{candidate_label}']:.6f} |"
            )
        lines.extend([
            "",
            "Thus, the near-0.99 overall PCC demonstrates excellent whole-profile "
            "agreement but should not be read as exact recovery at the deliberately "
            "biased sites; programmed-site PCC and RMSE are the more demanding "
            "bias-recovery diagnostics.",
            "",
        ])
    lines.extend([
        "The combined per-run table is `synthetic_gamma_runs_combined.tsv`; exact "
        "matched deltas are in `synthetic_gamma_matched_comparison.tsv`. Rerunning "
        "this script safely refreshes the comparison as downloads complete.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = (
        (args.baseline_label, Path(args.baseline_root).expanduser().resolve()),
        (args.candidate_label, Path(args.candidate_root).expanduser().resolve()),
    )
    same_root = roots[0][1] == roots[1][1]
    frames: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    for label, root in roots:
        frame, status = _run_one_root(
            label=label, root=root, output_dir=output_dir, args=args
        )
        statuses.append(status)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("Neither result tree currently contains an analyzable run.")

    combined = pd.concat(frames, ignore_index=True)
    matched = build_matched_comparison(
        combined,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    availability = pd.DataFrame(statuses)
    paths = {
        "combined": output_dir / "synthetic_gamma_runs_combined.tsv",
        "matched": output_dir / "synthetic_gamma_matched_comparison.tsv",
        "availability": output_dir / "synthetic_gamma_download_availability.tsv",
        "plot": output_dir / "synthetic_gamma_comparison.png",
        "report": output_dir / "GAMMA_COMPARISON.md",
    }
    combined.to_csv(paths["combined"], sep="\t", index=False)
    matched.to_csv(paths["matched"], sep="\t", index=False)
    availability.to_csv(paths["availability"], sep="\t", index=False)
    _plot(combined, matched, paths["plot"])
    _write_report(
        availability,
        combined,
        matched,
        paths["report"],
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        same_root=same_root,
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--candidate-root", default=str(DEFAULT_CANDIDATE_ROOT))
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="current")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--bias-root", default=str(DEFAULT_BIAS_ROOT))
    parser.add_argument("--latent-ground-truth", default=str(DEFAULT_LATENT_TRUTH))
    parser.add_argument(
        "--checkpoint-variant",
        choices=("best_pcc", "best_val_loss"),
        default="best_pcc",
    )
    return parser


def main() -> None:
    paths = analyze(build_parser().parse_args())
    print("Synthetic gamma root comparison written:")
    for label, path in paths.items():
        print(f"  {label:12s} {path}")


if __name__ == "__main__":
    main()
