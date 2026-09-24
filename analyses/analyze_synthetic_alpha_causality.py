#!/usr/bin/env python3
"""Build the matched A/B/C alpha-causality comparison.

This utility consumes the outputs produced by
``run_synthetic_alpha_causality_local.sh``. With no arguments it discovers the
newest complete A/B/C trio and generates any missing prerequisite recovery
analyses. It never chooses a checkpoint from the synthetic ground truth:
upstream tables and predictions are required to use the minimum-validation-
loss checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Utils.tensorboard_scalars import find_event_runs, load_scalars


CONDITIONS = {
    "A_standard": "A: standard learned alpha",
    "B_beta1_decoupled": "B: beta=1 decoupled mean gradient",
    "C_fixed_true_alpha": "C: fixed alpha=0.1",
}

DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results" / "riboai_synthetic_experiments"


def _discover_latest_complete_group(results_root: Path) -> tuple[str, dict[str, Path]]:
    """Return the newest run-name prefix containing exactly one A/B/C trio."""
    groups: dict[str, dict[str, Path]] = {}
    for path in results_root.iterdir():
        if not path.is_dir():
            continue
        try:
            condition = _condition_from_run(path.name)
        except ValueError:
            continue
        prefix = path.name[: -len(condition)]
        if condition in groups.setdefault(prefix, {}):
            raise ValueError(f"Duplicate {condition} run for prefix {prefix!r}.")
        groups[prefix][condition] = path

    complete = [
        (prefix, runs)
        for prefix, runs in groups.items()
        if set(runs) == set(CONDITIONS)
    ]
    if not complete:
        raise FileNotFoundError(
            f"No complete A/B/C alpha-causality run group found below {results_root}."
        )
    return max(
        complete,
        key=lambda item: max(path.stat().st_mtime for path in item[1].values()),
    )


def _group_id(run_dirs: dict[str, Path]) -> str:
    config_paths = list(run_dirs["A_standard"].rglob("config.yaml"))
    if len(config_paths) != 1:
        raise ValueError(
            f"{run_dirs['A_standard'].name}: expected one resolved config, "
            f"found {len(config_paths)}."
        )
    config = yaml.safe_load(config_paths[0].read_text(encoding="utf-8"))
    value = config.get("experiment", {}).get("alpha_causality_group_id")
    if value is None or not str(value).strip():
        raise ValueError("Resolved config has no experiment.alpha_causality_group_id.")
    return str(value)


def _run_command(command: list[str]) -> None:
    print("Running prerequisite:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def _ensure_prerequisite_analyses(
    *,
    results_root: Path,
    run_prefix: str,
    analysis_root: Path,
    workers: int,
    refresh: bool,
) -> None:
    """Generate shared, gamma, and compensation tables when absent."""
    script_dir = Path(__file__).resolve().parent
    scripts = {
        "shared": script_dir / "analyze_synthetic_recovery.py",
        "gamma": script_dir / "analyze_synthetic_gamma_recovery.py",
        "compensation": script_dir / "analyze_synthetic_gamma_compensation.py",
    }
    missing_scripts = [str(path) for path in scripts.values() if not path.is_file()]
    if missing_scripts:
        raise FileNotFoundError(
            "The direct comparison requires these analysis scripts to be present:\n  "
            + "\n  ".join(missing_scripts)
        )

    shared_dir = analysis_root / "shared_recovery"
    shared_table = shared_dir / "synthetic_recovery_by_panel.tsv"
    shared_is_current = (
        shared_table.is_file()
        and "L_vs_K_PCC_interior"
        in pd.read_csv(shared_table, sep="\t", nrows=1).columns
    )
    if refresh or not shared_is_current:
        _run_command(
            [
                sys.executable,
                str(scripts["shared"]),
                "--results-root",
                str(results_root),
                "--run-prefix",
                run_prefix,
                "--selection-metric",
                "val_loss",
                "--output-dir",
                str(shared_dir),
                "--strict",
            ]
        )

    gamma_dir = analysis_root / "gamma_recovery"
    gamma_table = gamma_dir / "synthetic_gamma_recovery_by_panel.tsv"
    gamma_is_current = (
        gamma_table.is_file()
        and "pooled_log_gamma_pcc_interior"
        in pd.read_csv(gamma_table, sep="\t", nrows=1).columns
    )
    if refresh or not gamma_is_current:
        _run_command(
            [
                sys.executable,
                str(scripts["gamma"]),
                "--results-root",
                str(results_root),
                "--run-prefix",
                run_prefix,
                "--checkpoint-variant",
                "best_val_loss",
                "--output-dir",
                str(gamma_dir),
                "--strict",
            ]
        )

    for condition in CONDITIONS:
        output_dir = analysis_root / "gamma_compensation" / condition
        required = (
            output_dir / "condition_bias_summary.tsv",
            output_dir / "site_group_summary.tsv",
        )
        compensation_is_current = all(path.is_file() for path in required)
        if compensation_is_current:
            compensation_is_current = (
                "evaluation_domain"
                in pd.read_csv(required[0], sep="\t", nrows=1).columns
            )
        if refresh or not compensation_is_current:
            _run_command(
                [
                    sys.executable,
                    str(scripts["compensation"]),
                    "--results-root",
                    str(results_root),
                    "--run-prefix",
                    f"{run_prefix}{condition}",
                    "--output-dir",
                    str(output_dir),
                    "--workers",
                    str(workers),
                    "--strict",
                ]
            )


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def _condition_from_run(run: str) -> str:
    matches = [condition for condition in CONDITIONS if run.endswith(condition)]
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve alpha-causality condition from {run!r}.")
    return matches[0]


def _drop_path(tree: dict[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    node: Any = tree
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(parts[-1], None)


def _validate_resolved_configs(run_dirs: dict[str, Path]) -> None:
    allowed_differences = (
        "loss.experiment_mode",
        "loss.nb_mean_gradient_beta",
        "model.alpha_mode",
        "model.fixed_alpha",
        "experiment.alpha_causality_condition",
        "paths.checkpoints",
        "paths.logs",
        "paths.results",
    )
    normalized: dict[str, dict[str, Any]] = {}
    validation_hashes = set()
    for condition, run_dir in run_dirs.items():
        configs = list(run_dir.rglob("config.yaml"))
        if len(configs) != 1:
            raise ValueError(f"{run_dir.name}: expected one resolved config, found {len(configs)}.")
        config = yaml.safe_load(configs[0].read_text(encoding="utf-8"))
        if config["prediction"]["checkpoint_variants"] != ["best_val_loss"]:
            raise ValueError(f"{run_dir.name}: predictions are not best-val-loss-only.")
        if bool(config["model"]["mass_conservation"]):
            raise ValueError(f"{run_dir.name}: alpha-causality suite must be mass-free.")
        if bool(config["callbacks"].get("save_best_pcc_checkpoint", True)):
            raise ValueError(f"{run_dir.name}: PCC checkpointing must be disabled.")
        manifests = list(run_dir.rglob("split_manifest_*.json"))
        if len(manifests) != 1:
            raise ValueError(f"{run_dir.name}: expected one split manifest.")
        import json

        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        validation_ids = [str(value) for value in manifest.get("validation_ids", [])]
        validation_hashes.add(
            hashlib.sha256(
                "\n".join(sorted(validation_ids)).encode("utf-8")
            ).hexdigest()[:12]
            if validation_ids
            else ""
        )
        clean = copy.deepcopy(config)
        for path in allowed_differences:
            _drop_path(clean, path)
        normalized[condition] = clean

    reference = normalized["A_standard"]
    for condition, config in normalized.items():
        if config != reference:
            raise ValueError(
                f"Resolved configuration for {condition} differs outside the "
                "permitted alpha-causality keys."
            )
    if len(validation_hashes) != 1 or "" in validation_hashes:
        raise ValueError("The three runs do not share one recorded validation-ID hash.")


def _scalar_at_step(run_dir: Path, tag: str, step: int) -> float:
    event_runs = find_event_runs(run_dir)
    if len(event_runs) != 1:
        raise ValueError(f"{run_dir.name}: expected one TensorBoard run, found {len(event_runs)}.")
    scalars = load_scalars(event_runs[0])
    events = scalars.get(tag, [])
    matches = [float(event.value) for event in events if int(event.step) == int(step)]
    if len(matches) != 1:
        raise KeyError(f"{run_dir.name}: missing unique scalar {tag!r} at step {step}.")
    return matches[0]


def _weighted_mean(frame: pd.DataFrame, value: str, weight: str) -> float:
    valid = frame[value].notna() & frame[weight].notna() & (frame[weight] > 0)
    if not bool(valid.any()):
        return float("nan")
    return float(
        (frame.loc[valid, value] * frame.loc[valid, weight]).sum()
        / frame.loc[valid, weight].sum()
    )


def _summarize_compensation(path: Path, *, fixed_alpha: bool) -> tuple[dict[str, float], pd.DataFrame]:
    condition = _read_tsv(path / "condition_bias_summary.tsv")
    groups = _read_tsv(path / "site_group_summary.tsv")
    strong = float(condition["strong_positions"].sum())
    missed = float(condition["missed_strong_positions"].sum())
    summary: dict[str, float] = {
        "strong_site_miss_rate": missed / strong if strong > 0 else float("nan"),
        "gamma_underestimate_log_alpha_correlation": _weighted_mean(
            condition,
            "gamma_underestimate_log_alpha_correlation",
            "programmed_positions",
        ),
        "gamma_underestimate_log_alpha_partial_correlation": _weighted_mean(
            condition,
            "gamma_underestimate_log_alpha_partial_correlation",
            "programmed_positions",
        ),
        "alpha_stratified_log_difference": _weighted_mean(
            condition,
            "alpha_missed_minus_correct_stratified",
            "alpha_stratified_matched_support",
        ),
        "replicate_cv_missed": _weighted_mean(
            condition,
            "mean_replicate_cv_missed",
            "missed_strong_positions",
        ),
        "replicate_cv_correct": _weighted_mean(
            condition,
            "mean_replicate_cv_correct",
            "correct_strong_positions",
        ),
    }
    for short, bias in (
        ("au", "artificial_bias_au_fraction_gt_0p7"),
        ("gc", "artificial_bias_gc_fraction_gt_0p7"),
    ):
        row = condition.loc[condition["bias_name"] == bias]
        summary[f"{short}_strong_site_miss_rate"] = (
            float(row.iloc[0]["strong_site_miss_rate"])
            if len(row) == 1
            else float("nan")
        )

    missed_groups = groups.loc[groups["site_group"] == "missed_strong"]
    correct_groups = groups.loc[groups["site_group"] == "correct_strong"]
    if fixed_alpha:
        summary.update(
            {
                "mean_alpha_missed": 0.1,
                "median_alpha_missed": 0.1,
                "mean_alpha_correct": 0.1,
                "median_alpha_correct": 0.1,
                "alpha_missed_correct_ratio": 1.0,
            }
        )
    else:
        mean_missed = _weighted_mean(missed_groups, "mean_alpha", "positions")
        mean_correct = _weighted_mean(correct_groups, "mean_alpha", "positions")
        summary.update(
            {
                "mean_alpha_missed": mean_missed,
                "median_alpha_missed": _weighted_mean(
                    missed_groups, "median_alpha", "positions"
                ),
                "mean_alpha_correct": mean_correct,
                "median_alpha_correct": _weighted_mean(
                    correct_groups, "median_alpha", "positions"
                ),
                "alpha_missed_correct_ratio": (
                    mean_missed / mean_correct
                    if math.isfinite(mean_missed)
                    and math.isfinite(mean_correct)
                    and mean_correct > 0
                    else float("nan")
                ),
            }
        )

    bias_table = condition[
        [
            "bias_name",
            "strong_positions",
            "missed_strong_positions",
            "strong_site_miss_rate",
            "mean_log_alpha_missed",
            "mean_log_alpha_correct",
            "alpha_missed_minus_correct_stratified",
            "mean_replicate_cv_missed",
            "mean_replicate_cv_correct",
        ]
    ].copy()
    return summary, bias_table


def _classification(rows: pd.DataFrame) -> str:
    miss = dict(zip(rows["condition"], rows["strong_site_miss_rate"], strict=True))
    a, b, c = miss["A_standard"], miss["B_beta1_decoupled"], miss["C_fixed_true_alpha"]
    if not all(math.isfinite(value) for value in (a, b, c)) or a <= 0:
        return "Inconclusive: the baseline has no finite, positive strong-site miss rate."

    def improves(value: float) -> bool:
        return value <= 0.5 * a and (a - value) >= 0.001

    def unchanged(value: float) -> bool:
        return abs(value - a) <= max(0.002, 0.2 * a)

    if improves(b) and improves(c):
        return (
            "Outcome 1: both gradient decoupling and fixed true alpha improve gamma; "
            "causal mean-gradient attenuation is supported."
        )
    if improves(c) and not improves(b):
        return (
            "Outcome 2: fixed true alpha improves gamma but beta=1 does not; alpha is "
            "causally involved beyond the proposed attenuation correction."
        )
    if unchanged(b) and unchanged(c):
        return (
            "Outcome 3: neither intervention materially changes the miss rate; elevated "
            "alpha is primarily a consequence/correlate of gamma failure. Investigate "
            "rarity, loss allocation, representation, and L/gamma attribution."
        )
    if improves(b) and not improves(c):
        return (
            "Outcome 4: beta=1 improves gamma but fixed true alpha does not. This is "
            "unexpected; verify gradients, checkpoint selection, and exact matching "
            "before interpretation."
        )
    return "Inconclusive: the three miss rates do not satisfy a prespecified outcome rule."


def _format(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(numeric) else f"{numeric:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Run-name prefix. Default: newest complete A/B/C group.",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=None,
        help="Default: <results-root>/<alpha-causality-group-id>_analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <analysis-root>/matched_comparison.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--refresh-prerequisites",
        action="store_true",
        help="Regenerate shared, gamma, and compensation analyses even if present.",
    )
    parser.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help="Require existing prerequisite tables instead of generating them.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least one.")
    results_root = args.results_root.expanduser().resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(results_root)
    run_prefix = (
        args.run_prefix
        if args.run_prefix is not None
        else _discover_latest_complete_group(results_root)[0]
    )
    run_dirs = {
        _condition_from_run(path.name): path
        for path in results_root.iterdir()
        if path.is_dir() and path.name.startswith(run_prefix)
    }
    if set(run_dirs) != set(CONDITIONS):
        raise ValueError(
            f"Expected exactly {sorted(CONDITIONS)}, found {sorted(run_dirs)}."
        )
    group_id = _group_id(run_dirs)
    analysis_root = (
        args.analysis_root.expanduser().resolve()
        if args.analysis_root is not None
        else REPOSITORY_ROOT
        / "analyses"
        / "artifacts"
        / "synthetic"
        / "alpha_causality"
        / group_id
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else analysis_root / "matched_comparison"
    )
    print(f"Selected matched group: {group_id}")
    print(f"Run prefix: {run_prefix}")
    print(f"Analysis root: {analysis_root}")

    _validate_resolved_configs(run_dirs)
    if not args.skip_prerequisites:
        _ensure_prerequisite_analyses(
            results_root=results_root,
            run_prefix=run_prefix,
            analysis_root=analysis_root,
            workers=args.workers,
            refresh=args.refresh_prerequisites,
        )

    shared = _read_tsv(
        analysis_root / "shared_recovery" / "synthetic_recovery_by_panel.tsv"
    )
    gamma = _read_tsv(
        analysis_root / "gamma_recovery" / "synthetic_gamma_recovery_by_panel.tsv"
    )
    rows = []
    bias_rows = []
    for condition, run_dir in run_dirs.items():
        run = run_dir.name
        shared_row = shared.loc[shared["run"] == run]
        gamma_row = gamma.loc[gamma["run"] == run]
        if len(shared_row) != 1 or len(gamma_row) != 1:
            raise ValueError(f"{run}: missing unique shared/gamma recovery rows.")
        shared_row = shared_row.iloc[0]
        gamma_row = gamma_row.iloc[0]
        if shared_row["selection_metric"] != "val_loss":
            raise ValueError(f"{run}: shared recovery was not selected by val_loss.")
        if gamma_row["checkpoint_variant"] != "best_val_loss":
            raise ValueError(f"{run}: gamma recovery did not use best_val_loss.")

        compensation_dir = analysis_root / "gamma_compensation" / condition
        compensation, by_bias = _summarize_compensation(
            compensation_dir,
            fixed_alpha=condition == "C_fixed_true_alpha",
        )
        by_bias.insert(0, "condition", condition)
        bias_rows.append(by_bias)
        step = int(shared_row["selection_step"])
        row = {
            "condition": condition,
            "label": CONDITIONS[condition],
            "run": run,
            "selection_step": step,
            "validation_id_hash": shared_row["validation_id_hash"],
            "L_vs_K_PCC": shared_row["L_vs_K_PCC_interior"],
            "L_vs_K_RMSE": shared_row["L_vs_K_RMSE_interior"],
            "L_vs_K_PCC_full_previous_definition": shared_row[
                "L_vs_K_PCC_full_previous_definition"
            ],
            "L_vs_K_RMSE_full_previous_definition": shared_row[
                "L_vs_K_RMSE_full_previous_definition"
            ],
            "gamma_mean_pair_PCC": gamma_row["mean_pair_log_gamma_pcc"],
            "gamma_pooled_PCC": gamma_row["pooled_log_gamma_pcc_interior"],
            "gamma_log_RMSE": gamma_row["pooled_log_gamma_rmse_interior"],
            "gamma_pooled_PCC_full_previous_definition": gamma_row[
                "pooled_log_gamma_pcc_full_previous_definition"
            ],
            "gamma_log_RMSE_full_previous_definition": gamma_row[
                "pooled_log_gamma_rmse_full_previous_definition"
            ],
            "gamma_calibration_slope": gamma_row["pooled_calibration_slope"],
            "biased_site_multiplicative_error": gamma_row[
                "programmed_site_mean_absolute_relative_error"
            ],
            "biased_sites_within_10pct": gamma_row[
                "programmed_site_fraction_within_10pct"
            ],
            "boundary_strong_site_miss_rate": gamma_row[
                "boundary_strong_site_miss_rate"
            ],
            "n_boundary_strong_sites": gamma_row["n_boundary_strong_sites"],
            "boundary_trim_codons": gamma_row["boundary_trim_codons"],
            "evaluation_domain": gamma_row["evaluation_domain"],
            "boundary_positions_excluded": gamma_row[
                "boundary_positions_excluded"
            ],
            "raw_val_NB2_NLL": _scalar_at_step(run_dir, "val_nb_nll_raw", step),
            "val_optimization_surrogate": _scalar_at_step(
                run_dir, "val_optimization_surrogate", step
            ),
            "val_mu_PCC": _scalar_at_step(run_dir, "val_mu_pcc_unweighted", step),
            "val_raw_profile_PCC": _scalar_at_step(run_dir, "val/pcc/raw_value", step),
            "val_NB_VST_profile_PCC": _scalar_at_step(
                run_dir, "val/pcc/nb_vst_value", step
            ),
            **compensation,
        }
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values("condition")
    per_bias = pd.concat(bias_rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "alpha_causality_matched_comparison.tsv", sep="\t", index=False)
    per_bias.to_csv(output_dir / "alpha_causality_strong_sites_by_bias.tsv", sep="\t", index=False)

    table_metrics = [
        ("L vs K PCC", "L_vs_K_PCC"),
        ("L vs K RMSE", "L_vs_K_RMSE"),
        ("gamma pooled PCC", "gamma_pooled_PCC"),
        ("gamma log-RMSE", "gamma_log_RMSE"),
        ("gamma slope", "gamma_calibration_slope"),
        ("strong-site miss rate", "strong_site_miss_rate"),
        ("AU strong-site miss rate", "au_strong_site_miss_rate"),
        ("GC strong-site miss rate", "gc_strong_site_miss_rate"),
        ("mean alpha missed", "mean_alpha_missed"),
        ("mean alpha correct", "mean_alpha_correct"),
        ("alpha missed/correct ratio", "alpha_missed_correct_ratio"),
        ("raw val NB2 NLL", "raw_val_NB2_NLL"),
        ("val mu PCC", "val_mu_PCC"),
    ]
    indexed = comparison.set_index("condition")
    lines = [
        "# Matched synthetic alpha-causality comparison",
        "",
        "All three runs use the same resolved configuration and validation IDs outside "
        "the explicitly permitted alpha/NB-gradient keys. Predictions and metrics use "
        "only the minimum-validation-loss checkpoint.",
        "",
        "| Metric | A: standard learned alpha | B: beta=1 decoupled mean gradient | C: fixed alpha=0.1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, column in table_metrics:
        values = [_format(indexed.loc[condition, column]) for condition in CONDITIONS]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(
        [
            "",
            "## Prespecified interpretation",
            "",
            _classification(comparison),
            "",
            "Operationally, 'substantial improvement' means at least a two-fold miss-rate "
            "reduction and at least 0.001 absolute reduction. 'Approximately unchanged' "
            "means within max(0.002, 20% of the baseline rate).",
            "",
            "The beta=1 decoupled objective is an optimization surrogate, not a likelihood. "
            "Cross-condition likelihood comparisons use `raw_val_NB2_NLL` only.",
        ]
    )
    (output_dir / "ALPHA_CAUSALITY_COMPARISON.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote matched alpha-causality comparison to {output_dir}")


if __name__ == "__main__":
    main()
