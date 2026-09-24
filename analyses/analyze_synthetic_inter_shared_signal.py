#!/usr/bin/env python3
"""Compare shared-signal recovery across inter artificial-bias settings.

The default run filter selects the cumulative inter artificial-bias series.
Recovery metrics are computed by the established latent-kinetics pipeline;
this wrapper adds a configuration-aware comparison of equal versus
quality-rank gamma references,
artificial-bias cases, and inter-bias panels with increasing dataset count. The
purpose is to answer
whether the shared ``L_bio`` signal changes when the nuisance-bias setup or its
gamma-reference ranking changes.

Example::

    python analyses/analyze_synthetic_inter_shared_signal.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyses.analyze_synthetic_recovery import (
    DEFAULT_LATENT_TRUTH,
    DEFAULT_RESULTS_ROOT,
    _find_prediction,
    _read_yaml,
    analyze as analyze_recovery,
    discover_run_directories,
    synthetic_mass_conservation,
    synthetic_depth_label,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


DEFAULT_RUN_PREFIX = "riboai_synthetic_inter_artificial_bias_cumulative_"
DEFAULT_OUTPUT_DIRECTORY_NAME = "inter_shared_signal_analysis"
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "analyses"
    / "artifacts"
    / "synthetic"
    / "inter_shared_signal"
)

REFERENCE_WEIGHTINGS = ("equal", "quality_rank")
REFERENCE_WEIGHTING_LABELS = {
    "equal": "Equal reference",
    "quality_rank": "Quality-rank reference",
}
REFERENCE_WEIGHTING_COLORS = {
    "equal": "#4338ca",
    "quality_rank": "#dc2626",
}
_RETRY_PATTERN = re.compile(r"_retry(?P<retry>[0-9]+)(?=_[^_]+$)")


def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _dataset_names(config: dict[str, Any]) -> list[str]:
    value = _nested(config, "experiment", "dataset", default=[])
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else [str(value)]


def _bias_case(datasets: list[str]) -> str:
    suffixes = ("_0p25_per_codon", "_2_per_codon", "_20_per_codon")
    bases = {
        next(
            (name[: -len(s)] for s in suffixes if name.endswith(s)), name
        )
        for name in datasets
    }
    return ",".join(sorted(bases))


def ranking_from_run_name(run_name: str) -> str:
    """Return the advisory gamma-reference label encoded in a run name."""
    if "_gammaquality_rank_" in run_name:
        return "quality_rank"
    if "_gammaequal_" in run_name:
        return "equal"
    return "missing"


def annotate_selected_attempts(
    frame: pd.DataFrame,
    *,
    availability_column: str,
) -> pd.DataFrame:
    """Mark one usable attempt per logical run, preferring complete retries.

    A retry has the same final run identifier as its original attempt, with a
    ``_retryN`` token immediately before that identifier.  If both attempts
    were downloaded, treating them as replicates would overweight one
    scientific setting.  A complete attempt wins over an incomplete one;
    otherwise the highest retry number wins.
    """
    annotated = frame.copy()
    if annotated.empty:
        annotated["logical_run"] = pd.Series(dtype="object")
        annotated["retry_index"] = pd.Series(dtype="int64")
        annotated["selected_attempt"] = pd.Series(dtype="bool")
        return annotated
    annotated["logical_run"] = annotated["run"].astype(str).map(
        lambda value: _RETRY_PATTERN.sub("", value)
    )
    annotated["retry_index"] = annotated["run"].astype(str).map(
        lambda value: (
            int(match.group("retry"))
            if (match := _RETRY_PATTERN.search(value)) is not None
            else 0
        )
    )
    available = annotated[availability_column].fillna(False).astype(bool)
    annotated["selected_attempt"] = False
    priority = (
        annotated.assign(_available=available)
        .sort_values(
            ["logical_run", "_available", "retry_index", "run"],
            ascending=[True, False, False, False],
        )
        .drop_duplicates("logical_run", keep="first")
    )
    annotated.loc[priority.index, "selected_attempt"] = True
    return annotated


def build_paired_ranking_comparison(
    frame: pd.DataFrame,
    *,
    metric_columns: tuple[str, ...],
    availability_column: str,
    checkpoint_column: str,
    matching_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Pair equal and quality-rank runs from the same resolved experiment.

    Metrics are differenced only when both artifacts exist and their held-out
    transcript hashes agree.  Every delta is defined as quality-rank minus
    equal; therefore positive is favorable for PCC-like metrics and negative
    is favorable for error metrics.
    """
    required = {
        "run",
        "mass_condition",
        "depth",
        "dataset_count",
        "datasets",
        "seed",
        "gamma_reference_weighting",
        availability_column,
        checkpoint_column,
        *matching_columns,
        *metric_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Cannot build the ranking comparison; missing columns: "
            + ", ".join(missing)
        )

    selected = frame.copy()
    if "selected_attempt" in selected:
        selected = selected[
            selected["selected_attempt"].fillna(False).astype(bool)
        ]
    selected = selected[
        selected["gamma_reference_weighting"].isin(REFERENCE_WEIGHTINGS)
    ].copy()
    pair_keys = [
        "mass_condition",
        "depth",
        "dataset_count",
        "datasets",
        "seed",
        checkpoint_column,
        *matching_columns,
    ]
    duplicates = (
        selected.groupby(
            pair_keys + ["gamma_reference_weighting"], dropna=False
        )
        .size()
        .loc[lambda values: values > 1]
    )
    if not duplicates.empty:
        raise ValueError(
            "More than one selected run exists for a ranking policy and "
            f"resolved panel: {duplicates.index[0]!r}."
        )

    policy_frames: dict[str, pd.DataFrame] = {}
    optional = [
        name
        for name in ("validation_id_hash", "gamma_quality_rank_power")
        if name in selected
    ]
    for weighting in REFERENCE_WEIGHTINGS:
        columns = pair_keys + [
            "run",
            availability_column,
            *optional,
            *metric_columns,
        ]
        policy = selected.loc[
            selected["gamma_reference_weighting"] == weighting, columns
        ].copy()
        policy.rename(
            columns={
                name: f"{name}_{weighting}"
                for name in columns
                if name not in pair_keys
            },
            inplace=True,
        )
        policy_frames[weighting] = policy

    paired = policy_frames["equal"].merge(
        policy_frames["quality_rank"],
        on=pair_keys,
        how="outer",
        validate="one_to_one",
    )
    equal_available = paired[f"{availability_column}_equal"].fillna(False).astype(bool)
    quality_available = paired[
        f"{availability_column}_quality_rank"
    ].fillna(False).astype(bool)
    paired["both_artifacts_available"] = equal_available & quality_available
    if "validation_id_hash_equal" in paired:
        paired["validation_ids_match"] = (
            paired["validation_id_hash_equal"].notna()
            & paired["validation_id_hash_quality_rank"].notna()
            & (
                paired["validation_id_hash_equal"].astype(str)
                == paired["validation_id_hash_quality_rank"].astype(str)
            )
        )
    else:
        paired["validation_ids_match"] = True
    paired["pair_comparable"] = (
        paired["both_artifacts_available"] & paired["validation_ids_match"]
    )
    for metric in metric_columns:
        metric_comparable = f"{metric}_pair_comparable"
        paired[metric_comparable] = (
            paired["pair_comparable"]
            & paired[f"{metric}_equal"].notna()
            & paired[f"{metric}_quality_rank"].notna()
        )
        delta_column = f"{metric}_quality_rank_minus_equal"
        paired[delta_column] = (
            paired[f"{metric}_quality_rank"] - paired[f"{metric}_equal"]
        ).where(paired[metric_comparable])
    return paired.sort_values(pair_keys).reset_index(drop=True)


def _resolve_matching_merge_column(frame: pd.DataFrame, name: str) -> None:
    """Coalesce one setup/recovery field and reject contradictory metadata."""
    left = f"{name}_x"
    right = f"{name}_y"
    if left not in frame and right not in frame:
        return
    if left in frame and right in frame:
        mismatch = frame[left].notna() & frame[right].notna() & (
            ~frame[left].eq(frame[right]).fillna(False)
        )
        if bool(mismatch.any()):
            runs = ", ".join(frame.loc[mismatch, "run"].astype(str).head(3))
            raise ValueError(
                f"Recovery and resolved-config {name!r} disagree for: {runs}."
            )
        frame[name] = frame[right].combine_first(frame[left])
    elif right in frame:
        frame[name] = frame[right]
    else:
        frame[name] = frame[left]
    frame.drop(columns=[column for column in (left, right) if column in frame], inplace=True)


def _setup_frame(
    run_dirs: list[Path],
    *,
    checkpoint_variant: str,
    strict: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for run_dir in run_dirs:
        configs = sorted(run_dir.rglob("config.yaml"))
        if len(configs) != 1:
            reason = f"expected one config.yaml, found {len(configs)}"
            skipped.append({"run": run_dir.name, "reason": reason})
            if strict:
                raise ValueError(f"{run_dir.name}: {reason}")
            continue
        try:
            config = _read_yaml(configs[0])
            mass_conservation, mass_condition = synthetic_mass_conservation(
                config, run_dir.name
            )
            datasets = _dataset_names(config)
            reference_names = _nested(
                config,
                "model",
                "gamma_centering",
                "reference",
                "dataset_names",
                default=None,
            )
            weighting = str(
                _nested(
                    config,
                    "model",
                    "gamma_centering",
                    "reference",
                    "weighting",
                    default="missing",
                )
            ).strip().lower()
            name_weighting = ranking_from_run_name(run_dir.name)
            rows.append(
                {
                    "run": run_dir.name,
                    "depth": synthetic_depth_label(config),
                    "bias_case": _bias_case(datasets),
                    "dataset_count": len(datasets),
                    "mass_conservation": mass_conservation,
                    "mass_condition": mass_condition,
                    "datasets": ",".join(datasets),
                    "gamma_reference_weighting": weighting,
                    "run_name_gamma_reference_weighting": name_weighting,
                    "gamma_reference_weighting_matches_run_name": (
                        name_weighting == "missing" or name_weighting == weighting
                    ),
                    "gamma_quality_rank_power": _nested(
                        config,
                        "model",
                        "gamma_centering",
                        "reference",
                        "quality_rank_power",
                        default="",
                    ),
                    "gamma_centering_mode": _nested(
                        config,
                        "model",
                        "gamma_centering",
                        "mode",
                        default="missing",
                    ),
                    "gamma_scale_gauge": _nested(
                        config,
                        "model",
                        "gamma_centering",
                        "dataset_constant_scale_gauge",
                        default="missing",
                    ),
                    "gamma_reference_dataset_names": (
                        "all_selected"
                        if reference_names is None
                        else ",".join(
                            str(value)
                            for value in (
                                reference_names
                                if isinstance(reference_names, (list, tuple))
                                else [reference_names]
                            )
                        )
                    ),
                    "gamma_reference_chunk_size": _nested(
                        config,
                        "model",
                        "gamma_centering",
                        "reference",
                        "chunk_size",
                        default="",
                    ),
                    "sample_reduction": _nested(
                        config,
                        "loss",
                        "sample_reduction",
                        default="missing",
                    ),
                    "batch_size": _nested(config, "data", "batch_size", default=""),
                    "seed": _nested(config, "experiment", "seed", default=""),
                    "prediction_checkpoint_variant": checkpoint_variant,
                    "prediction_available": _find_prediction(
                        run_dir,
                        checkpoint_variant=checkpoint_variant,
                    )
                    is not None,
                }
            )
        except (OSError, ValueError, TypeError) as exc:
            skipped.append({"run": run_dir.name, "reason": str(exc)})
            if strict:
                raise
    setup = annotate_selected_attempts(
        pd.DataFrame(rows), availability_column="prediction_available"
    )
    return setup, skipped


def _make_condition_plots(
    frame: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    if frame.empty:
        return paths
    conditions = frame["mass_condition"].unique()
    if len(conditions) != 1:
        raise ValueError("Each inter shared-signal figure must use one mass condition.")
    mass_condition = str(conditions[0])
    frame = frame[
        frame["selected_attempt"].fillna(False).astype(bool)
    ].copy()
    frame["dataset_count"] = pd.to_numeric(
        frame["dataset_count"], errors="coerce"
    )
    frame = frame.dropna(subset=["dataset_count"])
    ordered = (
        frame.sort_values(
            [
                "depth",
                "dataset_count",
                "gamma_reference_weighting",
                "run",
            ]
        )
        .reset_index(drop=True)
    )
    x = list(range(len(ordered)))
    labels = [f"{int(value):02d}" for value in ordered["run_number"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for weighting in REFERENCE_WEIGHTINGS:
        mask = ordered["gamma_reference_weighting"] == weighting
        if not bool(mask.any()):
            continue
        positions = [index for index, present in enumerate(mask) if present]
        label = REFERENCE_WEIGHTING_LABELS[weighting]
        colour = REFERENCE_WEIGHTING_COLORS[weighting]
        axes[0].scatter(
            positions,
            ordered.loc[mask, "L_vs_K_PCC_interior"],
            color=colour,
            label=label,
            s=48,
        )
        axes[1].scatter(
            positions,
            ordered.loc[mask, "L_vs_K_RMSE_interior"],
            color=colour,
            label=label,
            s=48,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("L_bio PCC versus latent ground truth")
    axes[0].set_xlabel("Run number (matching inter_shared_signal_by_run.tsv)")
    axes[0].set_title("Shared-signal recovery per run")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Mean-one L_bio RMSE")
    axes[1].set_xlabel("Run number (matching inter_shared_signal_by_run.tsv)")
    axes[1].set_title("Shared-signal error per run")
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.legend(frameon=False)
    fig.suptitle(f"Inter shared-signal results ({mass_condition})")
    path = output_dir / f"inter_shared_signal_by_run_{mass_condition}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # Overlaying the two policies is the primary scientific comparison.  A
    # missing selected-checkpoint parquet remains NaN after reindexing and is
    # therefore rendered as a gap instead of being joined across panel sizes.
    all_counts = sorted(frame["dataset_count"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharex=True, constrained_layout=True)
    for weighting in REFERENCE_WEIGHTINGS:
        subset = frame[frame["gamma_reference_weighting"] == weighting]
        seeds = sorted(str(value) for value in subset["seed"].dropna().unique())
        for seed_index, seed in enumerate(seeds):
            group = subset[subset["seed"].astype(str) == seed]
            grouped = (
                group.groupby("dataset_count", as_index=True)
                .agg(
                    L_vs_K_PCC_interior=("L_vs_K_PCC_interior", "mean"),
                    L_vs_K_RMSE_interior=("L_vs_K_RMSE_interior", "mean"),
                )
                .reindex(all_counts)
            )
            label = REFERENCE_WEIGHTING_LABELS[weighting]
            if len(seeds) > 1:
                label += f", seed {seed}"
            colour = REFERENCE_WEIGHTING_COLORS[weighting]
            axes[0].plot(
                all_counts,
                grouped["L_vs_K_PCC_interior"].to_numpy(),
                marker="o",
                color=colour,
                alpha=max(0.45, 1.0 - seed_index * 0.12),
                linestyle="-" if weighting == "equal" else "--",
                label=label,
            )
            axes[1].plot(
                all_counts,
                grouped["L_vs_K_RMSE_interior"].to_numpy(),
                marker="o",
                color=colour,
                alpha=max(0.45, 1.0 - seed_index * 0.12),
                linestyle="-" if weighting == "equal" else "--",
                label=label,
            )
    axes[0].set_title("Shared-signal PCC")
    axes[1].set_title("Shared-signal RMSE")
    axes[0].set_ylabel("L_bio PCC")
    axes[1].set_ylabel("Mean-one L_bio RMSE")
    for axis in axes:
        axis.set_xlabel("Number of inter-bias datasets in the panel")
        axis.set_xticks(all_counts)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Inter shared-signal recovery ({mass_condition})")
    path = output_dir / (
        f"inter_shared_signal_by_reference_weighting_{mass_condition}.png"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    condition_pairs = paired[
        paired["mass_condition"].astype(str) == mass_condition
    ].copy()
    condition_pairs = condition_pairs[
        condition_pairs["pair_comparable"].fillna(False).astype(bool)
    ]
    if not condition_pairs.empty:
        delta_metrics = (
            (
                "L_vs_K_PCC_interior_quality_rank_minus_equal",
                "Quality-rank minus equal L_bio PCC",
            ),
            (
                "L_vs_K_RMSE_interior_quality_rank_minus_equal",
                "Quality-rank minus equal L_bio RMSE",
            ),
        )
        fig, axes = plt.subplots(
            1, 2, figsize=(15, 5.5), sharex=True, constrained_layout=True
        )
        for axis, (column, ylabel) in zip(axes, delta_metrics):
            values = (
                condition_pairs.groupby("dataset_count", as_index=True)[column]
                .mean()
                .sort_index()
            )
            axis.axhline(0.0, color="#6b7280", linewidth=1.2, linestyle=":")
            axis.plot(
                values.index,
                values.to_numpy(),
                marker="o",
                color=REFERENCE_WEIGHTING_COLORS["quality_rank"],
            )
            axis.set_xticks(all_counts)
            axis.set_xlabel("Number of inter-bias datasets in the panel")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
        axes[0].set_title("Positive favors quality-rank")
        axes[1].set_title("Negative favors quality-rank")
        fig.suptitle(f"Paired reference-weighting effect ({mass_condition})")
        path = output_dir / (
            f"inter_shared_signal_paired_weighting_delta_{mass_condition}.png"
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _make_plots(
    frame: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for _, condition_frame in frame.groupby("mass_condition", sort=True):
        paths.extend(
            _make_condition_plots(condition_frame.copy(), paired, output_dir)
        )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--latent-ground-truth", default=str(DEFAULT_LATENT_TRUTH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--selection-metric",
        choices=("val_mu_pcc", "val_loss"),
        default="val_loss",
        help=(
            "Checkpoint used for prediction analysis. The scientific default "
            "is the minimum validation-loss checkpoint."
        ),
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def analyze(args: argparse.Namespace) -> dict[str, Path]:
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
            f"No run directories beginning {args.run_prefix!r} found below "
            f"{results_root}."
        )

    checkpoint_variant = (
        "best_val_loss" if args.selection_metric == "val_loss" else "best_pcc"
    )
    setup, skipped = _setup_frame(
        run_dirs,
        checkpoint_variant=checkpoint_variant,
        strict=args.strict,
    )
    mismatched_names = (
        setup[~setup["gamma_reference_weighting_matches_run_name"]]
        if "gamma_reference_weighting_matches_run_name" in setup
        else setup.iloc[0:0]
    )
    if args.strict and not mismatched_names.empty:
        raise ValueError(
            "Resolved gamma-reference weighting disagrees with the run name "
            f"for {mismatched_names.iloc[0]['run']}."
        )
    recovery_dir = output_dir / "recovery"
    recovery_paths = analyze_recovery(
        argparse.Namespace(
            results_root=str(results_root),
            run_prefix=args.run_prefix,
            latent_ground_truth=args.latent_ground_truth,
            output_dir=str(recovery_dir),
            selection_metric=args.selection_metric,
            strict=args.strict,
            # Panel-size trends are scientifically comparable only when the
            # held-out transcript IDs are identical within each condition.
            allow_validation_id_variation=False,
        )
    )
    panel = pd.read_csv(recovery_paths["summary"], sep="\t")
    comparison = panel.merge(
        setup, on="run", how="outer", validate="one_to_one"
    )
    # Both layers expose key setup fields. Coalesce them only after checking
    # equality, so a stale/misassociated prediction cannot acquire a plausible
    # label from the wrapper's config table.
    for column in (
        "depth",
        "dataset_count",
        "mass_conservation",
        "mass_condition",
        "datasets",
        "prediction_checkpoint_variant",
    ):
        _resolve_matching_merge_column(comparison, column)
    comparison["dataset_count"] = pd.to_numeric(
        comparison["dataset_count"], errors="raise"
    ).astype("Int64")
    comparison = comparison.sort_values(
        [
            "mass_condition",
            "depth",
            "dataset_count",
            "gamma_reference_weighting",
            "run",
        ]
    ).reset_index(drop=True)
    comparison.insert(0, "run_number", comparison.index + 1)
    comparison["analysis_artifact_available"] = (
        comparison["predictions_available"].fillna(False).astype(bool)
        & comparison["prediction_available"].fillna(False).astype(bool)
    )
    comparison = annotate_selected_attempts(
        comparison, availability_column="analysis_artifact_available"
    )
    by_run_path = output_dir / "inter_shared_signal_by_run.tsv"
    comparison.to_csv(by_run_path, sep="\t", index=False)

    detailed = comparison[
        comparison["selected_attempt"].fillna(False).astype(bool)
        & comparison["analysis_artifact_available"]
        & comparison["L_vs_K_PCC_interior"].notna()
        & comparison["L_vs_K_RMSE_interior"].notna()
    ].copy()
    if detailed.empty:
        raise RuntimeError(
            f"No complete {checkpoint_variant} prediction artifacts produced "
            "interior shared-signal metrics."
        )

    grouped = (
        detailed.groupby(
            [
                "mass_condition",
                "depth",
                "dataset_count",
                "datasets",
                "bias_case",
                "seed",
                "gamma_reference_weighting",
                "gamma_quality_rank_power",
            ],
            dropna=False,
        )
        .agg(
            runs=("run", "count"),
            mean_l_bio_pcc=("L_vs_K_PCC_interior", "mean"),
            std_l_bio_pcc=("L_vs_K_PCC_interior", "std"),
            mean_l_bio_rmse=("L_vs_K_RMSE_interior", "mean"),
            mean_l_bio_mse=("L_vs_K_MSE_interior", "mean"),
        )
        .reset_index()
    )
    by_setting_path = output_dir / "inter_shared_signal_by_setting.tsv"
    grouped.to_csv(by_setting_path, sep="\t", index=False)

    paired = build_paired_ranking_comparison(
        comparison,
        metric_columns=(
            "L_vs_K_PCC_interior",
            "L_vs_K_RMSE_interior",
            "L_vs_K_MSE_interior",
        ),
        availability_column="analysis_artifact_available",
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
    paired_path = output_dir / "inter_shared_signal_equal_vs_quality_rank.tsv"
    paired.to_csv(paired_path, sep="\t", index=False)
    plot_paths = _make_plots(comparison, paired, output_dir)
    skipped_path = output_dir / "inter_shared_signal_skipped_setup.tsv"
    pd.DataFrame(skipped, columns=["run", "reason"]).to_csv(
        skipped_path, sep="\t", index=False
    )

    comparable_pairs = int(paired["pair_comparable"].sum())
    configured_pairs = len(paired)
    pcc_deltas = paired.loc[
        paired["L_vs_K_PCC_interior_pair_comparable"],
        "L_vs_K_PCC_interior_quality_rank_minus_equal",
    ].dropna()
    rmse_deltas = paired.loc[
        paired["L_vs_K_RMSE_interior_pair_comparable"],
        "L_vs_K_RMSE_interior_quality_rank_minus_equal",
    ].dropna()
    seeds = ", ".join(
        sorted(str(value) for value in detailed["seed"].dropna().unique())
    )
    report = [
        "# Inter artificial-bias shared-signal recovery",
        "",
        f"Complete selected prediction runs analyzed: **{len(detailed)}**; "
        f"run prefix: `{args.run_prefix}`.",
        f"Checkpoint variant: **`{checkpoint_variant}`**.",
        f"Comparable equal/quality-rank pairs: **{comparable_pairs} of "
        f"{configured_pairs} configured panels**.",
        "",
        "The primary recovery target is the dataset-independent `L_bio` "
        "profile, compared after mean-one normalization with the deterministic "
        "kinetic ground truth. `mu` is not used as the shared-signal target "
        "because it intentionally contains dataset-specific bias.",
        "",
        "Equal and quality-rank gamma-reference weighting are separate series "
        "read from each resolved `config.yaml`. The run-name token is retained "
        "only as an audit field. Runs are paired only when mass condition, "
        "dataset panel, seed, checkpoint variant, gamma gauge, reference set, "
        "sample reduction, and batch size match; metric differences also "
        "require identical validation-ID hashes.",
        "",
        "Every paired delta is `quality_rank - equal`: positive favors "
        "quality-rank for PCC, while negative favors quality-rank for RMSE/MSE. "
        "Missing prediction artifacts remain explicit and create gaps. If an "
        "original attempt and `_retryN` both exist, only the best available "
        "attempt contributes to summaries and pairs.",
        "",
        "## Descriptive paired result",
        "",
        f"- `L_bio` PCC: mean delta **{pcc_deltas.mean():+.6f}**, median "
        f"**{pcc_deltas.median():+.6f}**; quality-rank is higher in "
        f"**{int((pcc_deltas > 0).sum())}/{len(pcc_deltas)}** panels.",
        f"- `L_bio` RMSE: mean delta **{rmse_deltas.mean():+.6f}**, median "
        f"**{rmse_deltas.median():+.6f}**; quality-rank is lower in "
        f"**{int((rmse_deltas < 0).sum())}/{len(rmse_deltas)}** panels.",
        "",
        f"These are descriptive results for seed(s) **{seeds}**. Panel size "
        "is cumulative, so each step also adds a new bias family; the trend "
        "is not a pure sample-size ablation.",
        "",
        "Files:",
        "",
        "- [per-run results](inter_shared_signal_by_run.tsv)",
        "- [grouped comparison](inter_shared_signal_by_setting.tsv)",
        "- [paired equal vs quality-rank comparison]"
        "(inter_shared_signal_equal_vs_quality_rank.tsv)",
        "- [underlying generic metric report](recovery/README.md) — use the "
        "policy-aware tables and plots above for panel-size comparisons",
        "- [skipped setup records](inter_shared_signal_skipped_setup.tsv)",
    ]
    for path in plot_paths:
        report.append(f"- [{path.stem}]({path.name})")
    report_path = output_dir / "INTER_SHARED_SIGNAL_RECOVERY.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Analyzed {len(comparison)} inter-bias runs.")
    print(f"Report: {report_path}")
    print(f"Grouped table: {by_setting_path}")
    print(f"Paired comparison: {paired_path}")
    paths = {
        "report": report_path,
        "summary": by_run_path,
        "by_run": by_run_path,
        "by_setting": by_setting_path,
        "paired": paired_path,
        "skipped": skipped_path,
        "recovery_report": recovery_paths["report"],
        "recovery_summary": recovery_paths["summary"],
    }
    for index, path in enumerate(plot_paths):
        paths[f"plot_{index}"] = path
    return paths


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
