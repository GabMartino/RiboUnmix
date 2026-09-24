#!/usr/bin/env python3
"""Analyze inter-dataset artificial-bias synthetic experiments.

This is the inter-bias companion to the two general synthetic reports. By
default it selects the cumulative inter artificial-bias runs and delegates
the actual metric calculations to:

* :mod:`analyze_synthetic_recovery` for the shared ``L_bio`` signal and
  reconstructed-profile recovery; and
* :mod:`analyze_synthetic_gamma_recovery` for gauge-fixed programmed-bias
  recovery.

It then adds paired equal-reference versus quality-rank-reference comparisons
for both targets. In addition, this script writes an experiment-setup audit.
The audit is
deliberately derived from each run's resolved ``config.yaml`` rather than from
the directory name, so it exposes the actual dataset panel, read-depth case,
gamma-centering mode, reference weighting, sample reduction, seed, and batch
settings.  Incomplete runs are retained in a skipped-run table instead of
being mistaken for failed recovery.

Example::

    python analyses/analyze_synthetic_inter_artificial_bias.py

The detailed reports are written below ``inter_artificial_bias_analysis/``
unless ``--output-dir`` is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyses.analyze_synthetic_gamma_recovery import analyze as analyze_gamma
from analyses.analyze_synthetic_recovery import (
    DEFAULT_LATENT_TRUTH,
    DEFAULT_RESULTS_ROOT,
    _find_prediction,
    _read_yaml,
    discover_run_directories,
    synthetic_mass_conservation,
    synthetic_depth_label,
)
from analyses.analyze_synthetic_inter_shared_signal import (
    REFERENCE_WEIGHTINGS,
    REFERENCE_WEIGHTING_COLORS,
    REFERENCE_WEIGHTING_LABELS,
    analyze as analyze_signal,
    annotate_selected_attempts,
    build_paired_ranking_comparison,
    ranking_from_run_name,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


DEFAULT_RUN_PREFIX = "riboai_synthetic_inter_artificial_bias_cumulative_"
DEFAULT_OUTPUT_DIRECTORY_NAME = "inter_artificial_bias_analysis"
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "analyses"
    / "artifacts"
    / "synthetic"
    / "inter_artificial_bias"
)


def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _depth_from_config(config: dict[str, Any]) -> str:
    return synthetic_depth_label(config)


def _setup_row(
    run_dir: Path,
    config: dict[str, Any],
    *,
    checkpoint_variant: str,
) -> dict[str, Any]:
    datasets = _as_list(_nested(config, "experiment", "dataset", default=[]))
    reference_names = _nested(
        config,
        "model",
        "gamma_centering",
        "reference",
        "dataset_names",
        default=None,
    )
    mass_conservation, mass_condition = synthetic_mass_conservation(
        config, run_dir.name
    )
    gamma_reference_weighting = str(
        _nested(
            config,
            "model",
            "gamma_centering",
            "reference",
            "weighting",
            default="missing",
        )
    ).strip().lower()
    run_name_weighting = ranking_from_run_name(run_dir.name)
    return {
        "run": run_dir.name,
        "depth": _depth_from_config(config),
        "dataset_count": len(datasets),
        "mass_conservation": mass_conservation,
        "mass_condition": mass_condition,
        "datasets": ",".join(datasets),
        "bias_cases": ",".join(
            sorted(
                {
                    dataset.removesuffix("_0p25_per_codon")
                    .removesuffix("_2_per_codon")
                    .removesuffix("_20_per_codon")
                    for dataset in datasets
                }
            )
        ),
        "seed": _nested(config, "experiment", "seed", default=""),
        "train": _nested(config, "experiment", "train", default=""),
        "predict": _nested(config, "experiment", "predict", default=""),
        "sample_reduction": _nested(
            config, "loss", "sample_reduction", default="missing"
        ),
        "batch_size": _nested(config, "data", "batch_size", default=""),
        "target_unique_transcripts_per_optimizer_step": _nested(
            config,
            "training",
            "grouped_optimizer_batch",
            "target_unique_transcripts_per_optimizer_step",
            default="",
        ),
        "max_accumulate_grad_batches": _nested(
            config,
            "training",
            "grouped_optimizer_batch",
            "max_accumulate_grad_batches",
            default="",
        ),
        "execution_microbatching": _nested(
            config,
            "training",
            "execution_microbatching",
            "enabled",
            default=False,
        ),
        "max_pair_rows_per_forward": _nested(
            config,
            "training",
            "execution_microbatching",
            "max_pair_rows_per_forward",
            default="",
        ),
        "gamma_centering_mode": _nested(
            config, "model", "gamma_centering", "mode", default="missing"
        ),
        "gamma_scale_gauge": _nested(
            config,
            "model",
            "gamma_centering",
            "dataset_constant_scale_gauge",
            default="missing",
        ),
        "gamma_reference_weighting": gamma_reference_weighting,
        "run_name_gamma_reference_weighting": run_name_weighting,
        "gamma_reference_weighting_matches_run_name": (
            run_name_weighting == "missing"
            or run_name_weighting == gamma_reference_weighting
        ),
        "gamma_quality_rank_power": _nested(
            config,
            "model",
            "gamma_centering",
            "reference",
            "quality_rank_power",
            default="",
        ),
        "gamma_reference_chunk_size": _nested(
            config,
            "model",
            "gamma_centering",
            "reference",
            "chunk_size",
            default="",
        ),
        "gamma_reference_dataset_names": (
            "all_selected"
            if reference_names is None
            else ",".join(_as_list(reference_names))
        ),
        "prediction_checkpoint_variant": checkpoint_variant,
        "prediction_available": _find_prediction(
            run_dir,
            checkpoint_variant=checkpoint_variant,
        )
        is not None,
        "config_path": str(next(run_dir.rglob("config.yaml"), "")),
    }


def _write_setup_report(
    setup: pd.DataFrame,
    skipped: list[dict[str, str]],
    output_dir: Path,
) -> dict[str, Path]:
    setup_path = output_dir / "inter_artificial_bias_setup.tsv"
    skipped_path = output_dir / "inter_artificial_bias_setup_skipped.tsv"
    report_path = output_dir / "INTER_ARTIFICIAL_BIAS_ANALYSIS.md"
    setup.to_csv(setup_path, sep="\t", index=False)
    pd.DataFrame(skipped, columns=["run", "reason"]).to_csv(
        skipped_path, sep="\t", index=False
    )

    plot_paths: list[Path] = []
    for mass_condition, condition_setup in setup.groupby(
        "mass_condition", sort=True
    ) if not setup.empty else []:
        condition_setup = condition_setup[
            condition_setup["selected_attempt"].fillna(False).astype(bool)
        ].copy()
        condition_setup["dataset_count"] = pd.to_numeric(
            condition_setup["dataset_count"], errors="coerce"
        )
        condition_setup = condition_setup.dropna(subset=["dataset_count"])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        all_counts = sorted(condition_setup["dataset_count"].unique())
        for row_index, weighting in enumerate(REFERENCE_WEIGHTINGS):
            policy = condition_setup[
                condition_setup["gamma_reference_weighting"] == weighting
            ]
            available = policy["prediction_available"].fillna(False).astype(bool)
            colour = REFERENCE_WEIGHTING_COLORS[weighting]
            axes[0].scatter(
                policy.loc[available, "dataset_count"],
                [row_index] * int(available.sum()),
                marker="o",
                s=70,
                color=colour,
                label=f"{REFERENCE_WEIGHTING_LABELS[weighting]}: available",
            )
            axes[0].scatter(
                policy.loc[~available, "dataset_count"],
                [row_index] * int((~available).sum()),
                marker="x",
                s=70,
                color=colour,
                linewidths=2,
                label=f"{REFERENCE_WEIGHTING_LABELS[weighting]}: missing",
            )
        axes[0].set_xticks(all_counts)
        axes[0].set_yticks(
            range(len(REFERENCE_WEIGHTINGS)),
            [REFERENCE_WEIGHTING_LABELS[value] for value in REFERENCE_WEIGHTINGS],
        )
        axes[0].set_xlabel("Number of datasets in cumulative panel")
        axes[0].set_title("Selected-checkpoint artifact coverage")
        axes[0].grid(alpha=0.25)
        axes[0].legend(frameon=False, fontsize=7, loc="best")

        configured = []
        completed = []
        for weighting in REFERENCE_WEIGHTINGS:
            policy = condition_setup[
                condition_setup["gamma_reference_weighting"] == weighting
            ]
            configured.append(len(policy))
            completed.append(
                int(policy["prediction_available"].fillna(False).astype(bool).sum())
            )
        x = list(range(len(REFERENCE_WEIGHTINGS)))
        axes[1].bar(
            [value - 0.18 for value in x],
            configured,
            width=0.36,
            color="#9ca3af",
            label="Configured",
        )
        axes[1].bar(
            [value + 0.18 for value in x],
            completed,
            width=0.36,
            color=[REFERENCE_WEIGHTING_COLORS[value] for value in REFERENCE_WEIGHTINGS],
            label="Prediction available",
        )
        axes[1].set_xticks(
            x,
            [REFERENCE_WEIGHTING_LABELS[value] for value in REFERENCE_WEIGHTINGS],
        )
        axes[1].set_ylabel("Selected attempts")
        axes[1].set_title("Coverage by gamma-reference weighting")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(frameon=False)
        fig.suptitle(f"Inter artificial-bias setup ({mass_condition})")
        plot_path = output_dir / f"inter_artificial_bias_setup_metrics_{mass_condition}.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        plot_paths.append(plot_path)

    lines = [
        "# Inter artificial-bias synthetic analysis",
        "",
        "This report is restricted to runs beginning with "
        f"`{DEFAULT_RUN_PREFIX}`.",
        "",
        "The shared-signal metrics come from the latent mean-one kinetics comparison "
        "(`L_bio` versus deterministic ground truth). The gamma metrics compare learned "
        "dataset-specific log-gamma shapes with the programmed bias profiles after the "
        "same joint cross-dataset and positional gauge is applied to both.",
        "",
        "The setup table is read from each resolved `config.yaml`; the run name is not "
        "treated as the source of truth.",
        "",
    ]
    if setup.empty:
        lines.append("No complete setup records were found.")
    else:
        lines.extend(
            [
                f"Resolved setup records: **{len(setup)}**; selected logical "
                f"attempts: **{int(setup['selected_attempt'].sum())}**.",
                "",
                "Important setup dimensions:",
                "",
                "- `depth` is the synthetic read-depth condition.",
                "- `bias_cases` identifies the programmed artificial sequence-bias cases.",
                "- `dataset_count` is the number of artificial dataset "
                "observations jointly trained.",
                "- `mass_condition` distinguishes strict mass conservation "
                "from mass-free predictions.",
                "- `gamma_reference_weighting` distinguishes equal and "
                "quality-rank gamma references.",
                "- `selected_attempt` prevents an original run and its "
                "`_retryN` from being counted as independent replicates.",
                "- `sample_reduction` is the pair-loss aggregation mode.",
                "- `prediction_available` indicates whether the run can "
                "contribute recovery metrics.",
                "- `prediction_checkpoint_variant` records the exact "
                "checkpoint family requested by this report.",
                "",
                "Files:",
                "",
                f"- [setup table]({setup_path.name})",
                f"- [skipped setup records]({skipped_path.name})",
                "- [shared-signal ranking comparison]"
                "(shared_signal/INTER_SHARED_SIGNAL_RECOVERY.md)",
                f"- [gamma ranking comparison](gamma/INTER_GAMMA_RANKING_COMPARISON.md)",
                "- [underlying generic gamma metric report]"
                "(gamma/GAMMA_RECOVERY.md) — use the policy-aware report for "
                "panel-size comparisons",
            ]
        )
        for plot_path in plot_paths:
            lines.append(f"- [setup plot: {plot_path.stem}]({plot_path.name})")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths = {
        "setup": setup_path,
        "skipped": skipped_path,
        "report": report_path,
    }
    for plot_path in plot_paths:
        paths[f"plot_{plot_path.stem}"] = plot_path
    return paths


def _merge_gamma_metrics_with_setup(
    metrics: pd.DataFrame,
    setup: pd.DataFrame,
) -> pd.DataFrame:
    """Attach authoritative reference-weighting metadata to gamma metrics."""
    metrics = metrics.rename(
        columns={"checkpoint_variant": "prediction_checkpoint_variant"}
    )
    merged = setup.merge(
        metrics,
        on="run",
        how="left",
        suffixes=("_config", ""),
        validate="one_to_one",
        indicator="gamma_metric_merge_status",
    )
    for name in (
        "depth",
        "dataset_count",
        "mass_conservation",
        "mass_condition",
        "datasets",
        "prediction_checkpoint_variant",
    ):
        config_name = f"{name}_config"
        if config_name not in merged:
            continue
        if name in merged:
            mismatch = (
                merged[config_name].notna()
                & merged[name].notna()
                & ~merged[config_name].eq(merged[name]).fillna(False)
            )
            if bool(mismatch.any()):
                run = str(merged.loc[mismatch, "run"].iloc[0])
                raise ValueError(
                    f"Gamma metrics and resolved config disagree on {name!r} "
                    f"for {run}."
                )
            merged[name] = merged[name].combine_first(merged[config_name])
        else:
            merged[name] = merged[config_name]
        merged.drop(columns=config_name, inplace=True)
    merged["dataset_count"] = pd.to_numeric(
        merged["dataset_count"], errors="raise"
    ).astype("Int64")
    merged["gamma_metrics_available"] = (
        merged["gamma_metric_merge_status"] == "both"
    )
    merged = annotate_selected_attempts(
        merged, availability_column="gamma_metrics_available"
    )
    return merged.sort_values(
        [
            "mass_condition",
            "depth",
            "dataset_count",
            "gamma_reference_weighting",
            "run",
        ]
    ).reset_index(drop=True)


def _make_gamma_ranking_plots(
    paired: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    metric_specs = (
        ("pooled_log_gamma_pcc", "Pooled log-gamma PCC", "higher is better"),
        ("pooled_log_gamma_rmse", "Pooled log-gamma RMSE", "lower is better"),
        (
            "programmed_site_mean_absolute_relative_error",
            "Biased-site mean relative error",
            "lower is better",
        ),
        (
            "programmed_site_fraction_within_10pct",
            "Biased sites within 10%",
            "higher is better",
        ),
    )
    paths: list[Path] = []
    for mass_condition, condition in paired.groupby(
        "mass_condition", sort=True
    ):
        counts = sorted(condition["dataset_count"].unique())
        fig, axes = plt.subplots(
            2, 2, figsize=(15, 10), sharex=True, constrained_layout=True
        )
        for axis, (metric, ylabel, direction) in zip(axes.ravel(), metric_specs):
            valid = condition[
                condition[f"{metric}_pair_comparable"]
                .fillna(False)
                .astype(bool)
            ]
            for weighting in REFERENCE_WEIGHTINGS:
                values = (
                    valid.groupby("dataset_count", as_index=True)[
                        f"{metric}_{weighting}"
                    ]
                    .mean()
                    .reindex(counts)
                )
                axis.plot(
                    counts,
                    values.to_numpy(),
                    marker="o",
                    linestyle="-" if weighting == "equal" else "--",
                    color=REFERENCE_WEIGHTING_COLORS[weighting],
                    label=REFERENCE_WEIGHTING_LABELS[weighting],
                )
            axis.set_title(f"{ylabel} ({direction})")
            axis.set_ylabel(ylabel)
            axis.set_xticks(counts)
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, fontsize=8)
        for axis in axes[-1]:
            axis.set_xlabel("Number of inter-bias datasets in the panel")
        fig.suptitle(
            f"Gamma recovery by reference weighting ({mass_condition})"
        )
        path = output_dir / (
            f"inter_gamma_recovery_by_reference_weighting_{mass_condition}.png"
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        condition_pairs = condition[
            condition["pair_comparable"].fillna(False).astype(bool)
        ]
        if condition_pairs.empty:
            continue
        fig, axes = plt.subplots(
            2, 2, figsize=(15, 10), sharex=True, constrained_layout=True
        )
        for axis, (metric, ylabel, direction) in zip(axes.ravel(), metric_specs):
            delta = f"{metric}_quality_rank_minus_equal"
            values = (
                condition_pairs.groupby("dataset_count", as_index=True)[delta]
                .mean()
                .sort_index()
            )
            axis.axhline(0.0, color="#6b7280", linestyle=":", linewidth=1.2)
            axis.plot(
                values.index,
                values.to_numpy(),
                marker="o",
                color=REFERENCE_WEIGHTING_COLORS["quality_rank"],
            )
            favored = (
                "positive favors quality-rank"
                if direction == "higher is better"
                else "negative favors quality-rank"
            )
            axis.set_title(f"{ylabel} ({favored})")
            axis.set_ylabel("Quality-rank minus equal")
            axis.set_xticks(counts)
            axis.grid(alpha=0.25)
        for axis in axes[-1]:
            axis.set_xlabel("Number of inter-bias datasets in the panel")
        fig.suptitle(
            f"Paired gamma-reference weighting effect ({mass_condition})"
        )
        path = output_dir / (
            f"inter_gamma_recovery_paired_weighting_delta_{mass_condition}.png"
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_gamma_ranking_report(
    gamma_paths: dict[str, Path],
    setup: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    summary = pd.read_csv(gamma_paths["summary"], sep="\t")
    comparison = _merge_gamma_metrics_with_setup(summary, setup)
    by_run_path = output_dir / "inter_gamma_recovery_by_run.tsv"
    comparison.to_csv(by_run_path, sep="\t", index=False)

    cases = pd.read_csv(gamma_paths["cases"], sep="\t")
    case_metadata = comparison[
        [
            "run",
            "logical_run",
            "retry_index",
            "selected_attempt",
            "seed",
            "gamma_reference_weighting",
            "gamma_quality_rank_power",
        ]
    ]
    cases = cases.merge(
        case_metadata, on="run", how="left", validate="many_to_one"
    )
    by_case_path = output_dir / "inter_gamma_recovery_by_bias_case.tsv"
    cases.to_csv(by_case_path, sep="\t", index=False)

    paired = build_paired_ranking_comparison(
        comparison,
        metric_columns=(
            "mean_pair_log_gamma_pcc",
            "pooled_log_gamma_pcc",
            "pooled_log_gamma_rmse",
            "programmed_site_mean_absolute_relative_error",
            "programmed_site_fraction_within_10pct",
        ),
        availability_column="gamma_metrics_available",
        checkpoint_column="prediction_checkpoint_variant",
        matching_columns=(
            "gamma_centering_mode",
            "gamma_scale_gauge",
            "gamma_reference_dataset_names",
            "gamma_reference_chunk_size",
            "sample_reduction",
            "batch_size",
        ),
    )
    paired_path = output_dir / "inter_gamma_equal_vs_quality_rank.tsv"
    paired.to_csv(paired_path, sep="\t", index=False)
    plot_paths = _make_gamma_ranking_plots(paired, output_dir)

    selected = comparison[
        comparison["selected_attempt"].fillna(False).astype(bool)
    ]
    complete_runs = int(selected["gamma_metrics_available"].sum())
    comparable_pairs = int(paired["pair_comparable"].sum())
    pooled_pcc_pairs = int(
        paired["pooled_log_gamma_pcc_pair_comparable"].sum()
    )
    undefined_pcc_counts = paired.loc[
        paired["pair_comparable"]
        & ~paired["pooled_log_gamma_pcc_pair_comparable"],
        "dataset_count",
    ].tolist()
    pcc_deltas = paired.loc[
        paired["pooled_log_gamma_pcc_pair_comparable"],
        "pooled_log_gamma_pcc_quality_rank_minus_equal",
    ].dropna()
    rmse_deltas = paired.loc[
        paired["pooled_log_gamma_rmse_pair_comparable"],
        "pooled_log_gamma_rmse_quality_rank_minus_equal",
    ].dropna()
    relative_error_deltas = paired.loc[
        paired[
            "programmed_site_mean_absolute_relative_error_pair_comparable"
        ],
        "programmed_site_mean_absolute_relative_error_quality_rank_minus_equal",
    ].dropna()
    within_tolerance_deltas = paired.loc[
        paired["programmed_site_fraction_within_10pct_pair_comparable"],
        "programmed_site_fraction_within_10pct_quality_rank_minus_equal",
    ].dropna()
    seeds = ", ".join(
        sorted(str(value) for value in selected["seed"].dropna().unique())
    )
    report_path = output_dir / "INTER_GAMMA_RANKING_COMPARISON.md"
    lines = [
        "# Inter artificial-bias gamma recovery by reference weighting",
        "",
        f"Complete selected runs: **{complete_runs}**. Comparable "
        f"equal/quality-rank pairs: **{comparable_pairs} of {len(paired)} "
        "configured panels**.",
        f"Pooled log-gamma PCC is defined for **{pooled_pcc_pairs} of "
        f"{comparable_pairs}** comparable panels.",
    ]
    if undefined_pcc_counts:
        counts = ", ".join(f"N={int(value)}" for value in undefined_pcc_counts)
        lines.extend(
            [
                "",
                f"Pooled PCC is omitted for {counts} because at least one "
                "policy has zero-variance gauge-fixed programmed gamma; "
                "correlation is undefined there. Other well-defined metrics "
                "for that panel are retained.",
            ]
        )
    lines.extend(
        [
            "",
            "The reference weighting is read from the resolved configuration. "
            "Equal and quality-rank runs are never joined into one panel-size "
            "series. Pairing requires the same mass condition, dataset panel, "
            "seed, checkpoint variant, gamma gauge, reference set, sample "
            "reduction, and batch size; deltas additionally require identical "
            "validation transcript IDs.",
            "",
            "All deltas are `quality_rank - equal`. Positive is favorable for "
            "PCC and fractions-within-tolerance; negative is favorable for "
            "RMSE and relative error.",
            "",
            "## Descriptive paired result",
            "",
            f"- Pooled log-gamma PCC: mean delta "
            f"**{pcc_deltas.mean():+.6f}**, median "
            f"**{pcc_deltas.median():+.6f}**; quality-rank is higher in "
            f"**{int((pcc_deltas > 0).sum())}/{len(pcc_deltas)}** defined "
            "panels.",
            f"- Pooled log-gamma RMSE: mean delta "
            f"**{rmse_deltas.mean():+.6f}**, median "
            f"**{rmse_deltas.median():+.6f}**; quality-rank is lower in "
            f"**{int((rmse_deltas < 0).sum())}/{len(rmse_deltas)}** panels.",
            f"- Biased-site mean relative error: mean delta "
            f"**{relative_error_deltas.mean():+.6f}**, median "
            f"**{relative_error_deltas.median():+.6f}**; quality-rank is "
            f"lower in **{int((relative_error_deltas < 0).sum())}/"
            f"{len(relative_error_deltas)}** panels.",
            f"- Biased sites within 10%: mean delta "
            f"**{100.0 * within_tolerance_deltas.mean():+.2f} percentage "
            f"points**, median **{100.0 * within_tolerance_deltas.median():+.2f} "
            f"points**; quality-rank is higher in "
            f"**{int((within_tolerance_deltas > 0).sum())}/"
            f"{len(within_tolerance_deltas)}** panels.",
            "",
            f"These are descriptive results for seed(s) **{seeds}**. Panel "
            "size is cumulative, so each step also adds a new bias family; "
            "the trend is not a pure sample-size ablation.",
            "",
            "Files:",
            "",
            "- [policy-aware per-run gamma metrics]"
            "(inter_gamma_recovery_by_run.tsv)",
            "- [policy-aware per-bias gamma metrics]"
            "(inter_gamma_recovery_by_bias_case.tsv)",
            "- [paired equal vs quality-rank comparison]"
            "(inter_gamma_equal_vs_quality_rank.tsv)",
            f"- [underlying generic gamma metric report]"
            f"({gamma_paths['report'].name}) — use the policy-aware tables and "
            "plots above for panel-size comparisons",
        ]
    )
    for path in plot_paths:
        lines.append(f"- [{path.stem}]({path.name})")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths = {
        "report": report_path,
        "by_run": by_run_path,
        "by_case": by_case_path,
        "paired": paired_path,
    }
    for index, path in enumerate(plot_paths):
        paths[f"plot_{index}"] = path
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--bias-root", default="Datasets/Synthetic_data/bias_profile")
    parser.add_argument("--latent-ground-truth", default=str(DEFAULT_LATENT_TRUTH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--selection-metric",
        choices=("val_mu_pcc", "val_loss"),
        default="val_loss",
        help=(
            "Checkpoint used for both shared-signal and gamma recovery. The "
            "scientific default is the minimum validation-loss checkpoint."
        ),
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = discover_run_directories(results_root, args.run_prefix)
    if not run_dirs:
        raise RuntimeError(
            f"No run directories beginning {args.run_prefix!r} found below {results_root}."
        )

    checkpoint_variant = (
        "best_val_loss" if args.selection_metric == "val_loss" else "best_pcc"
    )
    setup_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for run_dir in run_dirs:
        configs = sorted(run_dir.rglob("config.yaml"))
        if len(configs) != 1:
            skipped.append(
                {
                    "run": run_dir.name,
                    "reason": (
                        f"expected one config.yaml, found {len(configs)}"
                    ),
                }
            )
            if args.strict:
                raise ValueError(skipped[-1]["reason"])
            continue
        try:
            config = _read_yaml(configs[0])
            setup_rows.append(
                _setup_row(
                    run_dir,
                    config,
                    checkpoint_variant=checkpoint_variant,
                )
            )
        except (OSError, ValueError, TypeError) as exc:
            skipped.append({"run": run_dir.name, "reason": str(exc)})
            if args.strict:
                raise

    setup = annotate_selected_attempts(
        pd.DataFrame(setup_rows), availability_column="prediction_available"
    )
    if (
        args.strict
        and "gamma_reference_weighting_matches_run_name" in setup
        and not bool(setup["gamma_reference_weighting_matches_run_name"].all())
    ):
        mismatch = setup[
            ~setup["gamma_reference_weighting_matches_run_name"]
        ].iloc[0]
        raise ValueError(
            "Resolved gamma-reference weighting disagrees with the run name "
            f"for {mismatch['run']}."
        )
    paths = _write_setup_report(setup, skipped, output_dir)

    common = dict(
        results_root=str(results_root),
        run_prefix=args.run_prefix,
        latent_ground_truth=args.latent_ground_truth,
        output_dir=str(output_dir / "shared_signal"),
        selection_metric=args.selection_metric,
        strict=args.strict,
    )
    signal_paths = analyze_signal(argparse.Namespace(**common))
    gamma_options = dict(common)
    gamma_options["output_dir"] = str(output_dir / "gamma")
    gamma_options["checkpoint_variant"] = checkpoint_variant
    gamma_paths = analyze_gamma(
        argparse.Namespace(
            **gamma_options,
            bias_root=args.bias_root,
        )
    )
    gamma_ranking_paths = _write_gamma_ranking_report(
        gamma_paths, setup, output_dir / "gamma"
    )
    print(f"Analyzed {len(run_dirs)} inter-bias run directories.")
    print(f"Setup report: {paths['report']}")
    print(f"Shared signal report: {signal_paths['report']}")
    print(f"Gamma ranking report: {gamma_ranking_paths['report']}")
    print(f"Underlying generic gamma report: {gamma_paths['report']}")


if __name__ == "__main__":
    main()
