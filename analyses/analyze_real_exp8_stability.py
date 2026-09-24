#!/usr/bin/env python3
"""Analyze overlap-controlled shared-profile stability for real Experiment 8."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.paths import artifact_directory


def _discover_run_root() -> Path:
    """Return the newest locally available Experiment-8 design."""
    results_root = PROJECT_ROOT / "results"
    candidates: list[tuple[float, Path]] = []
    for manifest in results_root.rglob("experiment_manifest.json"):
        try:
            data = _read_json(manifest)
        except Exception:
            continue
        if data.get("experiment_name") == "real_exp8_L_stability":
            candidates.append((manifest.stat().st_mtime, manifest.parent))
    if not candidates:
        raise FileNotFoundError(
            f"No Experiment-8 experiment_manifest.json found under {results_root}. "
            "Pass --run-root explicitly."
        )
    return max(candidates, key=lambda item: item[0])[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute designated-disjoint stability, convergence to the empirical "
            "full 114-dataset representation, and overlap diagnostics."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Experiment root. Default: newest local Experiment-8 design.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=202608)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--mean-one-tolerance", type=float, default=1.0e-4)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.var(left) <= 0.0 or np.var(right) <= 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(
        pd.Series(left).rank(method="average").to_numpy(dtype=np.float64),
        pd.Series(right).rank(method="average").to_numpy(dtype=np.float64),
    )


def _metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return {
        "PCC": _pearson(left, right),
        "Spearman": _spearman(left, right),
        "RMSE": float(np.sqrt(np.mean(np.square(left - right)))),
    }


def _task_directory(run_root: Path, task: Mapping[str, Any]) -> Path:
    base = run_root / f"N{int(task['N']):03d}"
    if task["kind"] == "designated_disjoint_pair":
        result = base / f"{task['pair_id']}_{task['side']}"
    elif task["kind"] == "large_N_subset":
        result = base / str(task["subset_id"])
    else:
        result = base / "full"
    if "base_run_id" in task:
        result = result / f"trainseed{task['training_seed']}"
    return result


def _load_profiles(
    *,
    path: Path,
    expected_ids: Sequence[str],
    mean_one_tolerance: float,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    required = {
        "transcript_id",
        "transcript_length",
        "L_t",
        "valid_position_mask",
    }
    frame = pd.read_parquet(path)
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing shared-profile columns {sorted(missing)}.")
    frame["transcript_id"] = frame["transcript_id"].astype(str)
    if frame["transcript_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate transcript IDs.")
    expected = list(map(str, expected_ids))
    observed = frame["transcript_id"].tolist()
    if set(observed) != set(expected):
        raise ValueError(
            f"Common-test identity mismatch in {path}: "
            f"missing={sorted(set(expected) - set(observed))[:10]}, "
            f"extra={sorted(set(observed) - set(expected))[:10]}."
        )
    indexed = frame.set_index("transcript_id")
    profiles: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    for transcript_id in expected:
        row = indexed.loc[transcript_id]
        values = np.asarray(row["L_t"], dtype=np.float64)
        mask = np.asarray(row["valid_position_mask"], dtype=bool)
        length = int(row["transcript_length"])
        if values.ndim != 1 or mask.ndim != 1 or values.shape != mask.shape:
            raise ValueError(f"Invalid profile/mask shape for {transcript_id} in {path}.")
        if int(mask.sum()) != length or length <= 0:
            raise ValueError(f"Mask/length mismatch for {transcript_id} in {path}.")
        valid = values[mask]
        if not np.isfinite(valid).all():
            raise ValueError(f"Non-finite L_t for {transcript_id} in {path}.")
        mean = float(valid.mean())
        deviation = abs(mean - 1.0)
        if deviation > mean_one_tolerance:
            raise ValueError(
                f"L_t is not mean-one for {transcript_id}: mean={mean:.8g}. "
                "The analysis does not renormalize predictions."
            )
        profiles[transcript_id] = {"values": values, "mask": mask, "length": length}
        checks.append(
            {
                "transcript_id": transcript_id,
                "transcript_length": length,
                "L_mean": mean,
                "absolute_mean_one_deviation": deviation,
            }
        )
    return profiles, pd.DataFrame(checks)


def _compare_profiles(
    *,
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    transcript_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transcript_id in transcript_ids:
        left_row = left[str(transcript_id)]
        right_row = right[str(transcript_id)]
        left_values = np.asarray(left_row["values"], dtype=np.float64)
        right_values = np.asarray(right_row["values"], dtype=np.float64)
        left_mask = np.asarray(left_row["mask"], dtype=bool)
        right_mask = np.asarray(right_row["mask"], dtype=bool)
        if left_values.shape != right_values.shape or not np.array_equal(
            left_mask, right_mask
        ):
            raise ValueError(
                f"Prediction position alignment differs for transcript {transcript_id}."
            )
        full_left = left_values[left_mask]
        full_right = right_values[right_mask]
        full = _metrics(full_left, full_right)
        length = int(left_mask.sum())
        interior_mask = left_mask.copy()
        valid_indices = np.flatnonzero(left_mask)
        if valid_indices.size > 10:
            interior_mask[valid_indices[:5]] = False
            interior_mask[valid_indices[-5:]] = False
            interior = _metrics(left_values[interior_mask], right_values[interior_mask])
        else:
            interior = {"PCC": np.nan, "Spearman": np.nan, "RMSE": np.nan}
        rows.append(
            {
                "transcript_id": str(transcript_id),
                "transcript_length": length,
                **full,
                "PCC_interior5": interior["PCC"],
                "Spearman_interior5": interior["Spearman"],
                "RMSE_interior5": interior["RMSE"],
            }
        )
    return rows


def _distribution_summary(values: pd.Series) -> dict[str, float | int]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if not len(array):
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "IQR": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
        }
    q = np.percentile(array, [5, 25, 50, 75, 95])
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "IQR": float(q[3] - q[1]),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
    }


def _hierarchical_bootstrap(
    frame: pd.DataFrame,
    *,
    grouping_column: str,
    metric: str,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    groups = {
        str(group): pd.to_numeric(values[metric], errors="coerce")
        .dropna()
        .to_numpy(dtype=np.float64)
        for group, values in frame.groupby(grouping_column, sort=True)
    }
    groups = {key: value for key, value in groups.items() if len(value)}
    if not groups or replicates <= 0:
        return float("nan"), float("nan")
    names = sorted(groups)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        selected_groups = rng.choice(names, size=len(names), replace=True)
        group_means = []
        for name in selected_groups:
            values = groups[str(name)]
            group_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        estimates[index] = float(np.mean(group_means))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def _save_stability_figure(
    pair_summary: pd.DataFrame,
    n_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
    rng = np.random.default_rng(42)
    for row in pair_summary.itertuples(index=False):
        axis.scatter(
            float(row.N) * np.exp(rng.uniform(-0.018, 0.018)),
            row.mean_PCC,
            color="#4c78a8",
            s=35,
            alpha=0.72,
            zorder=2,
        )
    ordered = n_summary.sort_values("N")
    axis.plot(ordered["N"], ordered["mean_PCC"], color="#d62728", marker="o", lw=1.8)
    axis.fill_between(
        ordered["N"].to_numpy(dtype=float),
        ordered["bootstrap_ci_low"].to_numpy(dtype=float),
        ordered["bootstrap_ci_high"].to_numpy(dtype=float),
        color="#d62728",
        alpha=0.16,
    )
    axis.set_xscale("log")
    axis.set_xticks(sorted(ordered["N"].unique()), [str(value) for value in sorted(ordered["N"].unique())])
    axis.set_ylim(-0.05, 1.02)
    axis.set_xlabel("Number of datasets in each source-disjoint panel, N")
    axis.set_ylabel("Mean held-out transcript PCC")
    axis.set_title("Independent shared-profile agreement versus dataset count")
    axis.grid(alpha=0.2)
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        figure.savefig(figure_dir / f"stability_vs_N.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(figure)


def _save_convergence_figure(
    run_summary: pd.DataFrame,
    n_summary: pd.DataFrame,
    full_N: int,
    figure_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
    axis.scatter(run_summary["N"], run_summary["mean_PCC"], color="#59a14f", alpha=0.68, s=30)
    ordered = n_summary.sort_values("N")
    axis.plot(ordered["N"], ordered["mean_PCC"], color="#1f6d35", marker="o", lw=1.8)
    axis.scatter([full_N], [1.0], marker="*", s=120, color="#777777", label=f"N={full_N} self-reference")
    axis.set_xscale("log")
    ticks = sorted(set(ordered["N"].tolist()) | {full_N})
    axis.set_xticks(ticks, [str(value) for value in ticks])
    axis.set_ylim(-0.05, 1.02)
    axis.set_xlabel("Number of training datasets, N")
    axis.set_ylabel(f"Mean PCC with full N={full_N} representation")
    axis.set_title(f"Convergence to the full {full_N}-dataset representation")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        figure.savefig(
            figure_dir / f"convergence_to_full_vs_N.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(figure)


def _save_overlap_figure(all_pairs: pd.DataFrame, figure_dir: Path) -> None:
    secondary = all_pairs.loc[~all_pairs["is_designated_disjoint_pair"]].copy()
    figure, axis = plt.subplots(figsize=(6.6, 4.7), constrained_layout=True)
    if not secondary.empty:
        image = axis.scatter(
            secondary["jaccard"],
            secondary["mean_transcript_PCC"],
            c=secondary["N"],
            cmap="viridis",
            s=34,
            alpha=0.74,
        )
        figure.colorbar(image, ax=axis, label="N")
    axis.set_xlabel("Jaccard overlap of training datasets")
    axis.set_ylabel("Mean held-out L agreement (PCC)")
    axis.set_title("Subset overlap can inflate apparent same-N stability")
    axis.grid(alpha=0.2)
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        figure.savefig(figure_dir / f"overlap_vs_agreement.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(figure)


def _save_quality_figure(quality: pd.DataFrame, figure_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    positive_floor = max(
        float(
            quality.loc[quality["quality_mismatch"] > 0, "quality_mismatch"].min()
        )
        * 0.5
        if bool((quality["quality_mismatch"] > 0).any())
        else 1.0e-12,
        1.0e-12,
    )
    for N, group in quality.groupby("N", sort=True):
        axis.scatter(
            np.full(len(group), N),
            group["quality_mismatch"].clip(lower=positive_floor),
            s=28,
            alpha=0.68,
        )
    medians = (
        quality.assign(
            plotted_quality_mismatch=quality["quality_mismatch"].clip(
                lower=positive_floor
            )
        )
        .groupby("N", sort=True)["plotted_quality_mismatch"]
        .median()
    )
    axis.plot(medians.index, medians.values, color="#d62728", marker="o")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(medians.index, [str(value) for value in medians.index])
    axis.set_xlabel("N")
    axis.set_ylabel("Mismatch from full-pool observed-QC distribution")
    axis.set_title("Quality matching of source-family-atomic subsets")
    axis.grid(alpha=0.2)
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        figure.savefig(figure_dir / f"subset_quality_balance.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(figure)


def _representative_profiles(
    *,
    disjoint: pd.DataFrame,
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    profiles: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output_dir: Path,
    figure_dir: Path,
) -> None:
    largest_N = int(disjoint["N"].max())
    candidates = disjoint.loc[disjoint["N"] == largest_N]
    pair_id = sorted(candidates["pair_id"].unique())[0]
    training_seed = int(candidates["training_seed"].min())
    pair = candidates.loc[
        (candidates["pair_id"] == pair_id)
        & (candidates["training_seed"] == training_seed)
    ]
    pooled = pair.groupby("transcript_id", sort=True)["PCC"].mean().reset_index()
    labels = [("low", 0.10), ("median", 0.50), ("high", 0.90)]
    selected: list[dict[str, Any]] = []
    for label, quantile in labels:
        target = float(pooled["PCC"].quantile(quantile))
        row = pooled.assign(distance=(pooled["PCC"] - target).abs()).sort_values(
            ["distance", "transcript_id"]
        ).iloc[0]
        selected.append(
            {
                "selection": label,
                "quantile": quantile,
                "transcript_id": str(row["transcript_id"]),
                "PCC": float(row["PCC"]),
            }
        )
    selected_frame = pd.DataFrame(selected)
    selected_frame.to_csv(output_dir / "representative_transcripts.csv", index=False)
    run_a = str(pair["run_a"].iloc[0])
    run_b = str(pair["run_b"].iloc[0])
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 8.0), constrained_layout=True)
    source_rows: list[dict[str, Any]] = []
    for axis, row in zip(axes, selected, strict=True):
        transcript_id = row["transcript_id"]
        for run_id, color in ((run_a, "#4c78a8"), (run_b, "#e45756")):
            record = profiles[run_id][transcript_id]
            values = np.asarray(record["values"])[np.asarray(record["mask"], dtype=bool)]
            positions = np.arange(len(values))
            axis.plot(positions, values, lw=0.9, color=color, label=run_id)
            source_rows.extend(
                {
                    "selection": row["selection"],
                    "transcript_id": transcript_id,
                    "run_id": run_id,
                    "codon_position_0based": int(position),
                    "L_t": float(value),
                }
                for position, value in zip(positions, values, strict=True)
            )
        axis.set_title(
            f"{row['selection'].title()} agreement: {transcript_id} (PCC={row['PCC']:.3f})"
        )
        axis.set_xlabel("Modeled codon position (0-based)")
        axis.set_ylabel("Mean-one L_t")
        axis.grid(alpha=0.15)
    axes[0].legend(frameon=False, fontsize=8)
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        figure.savefig(
            figure_dir / f"representative_L_profiles.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(figure)
    pd.DataFrame(source_rows).to_parquet(
        output_dir / "representative_L_profiles.parquet", index=False
    )


def _saturation_fit(summary: pd.DataFrame) -> dict[str, Any]:
    finite = summary.loc[
        np.isfinite(summary["mean_PCC"]) & (summary["N"] > 0)
    ].sort_values("N")
    if len(finite) < 4:
        return {"status": "not_fitted", "reason": "fewer than four finite N values"}
    try:
        from scipy.optimize import curve_fit

        def model(N, asymptote, amplitude, tau):
            return asymptote - amplitude * np.exp(-N / tau)

        x = finite["N"].to_numpy(dtype=np.float64)
        y = finite["mean_PCC"].to_numpy(dtype=np.float64)
        parameters, covariance = curve_fit(
            model,
            x,
            y,
            p0=(min(1.0, max(y) + 0.02), max(0.01, max(y) - min(y)), 15.0),
            bounds=([-1.0, 0.0, 1.0e-6], [1.0, 2.0, 1.0e5]),
            maxfev=20000,
        )
        prediction = model(x, *parameters)
        residual = float(np.sum(np.square(y - prediction)))
        total = float(np.sum(np.square(y - y.mean())))
        errors = np.sqrt(np.diag(covariance))
        return {
            "status": "fitted_descriptive_only",
            "formula": "C(N)=C_inf-A*exp(-N/tau)",
            "C_inf": float(parameters[0]),
            "A": float(parameters[1]),
            "tau": float(parameters[2]),
            "C_inf_standard_error": float(errors[0]),
            "A_standard_error": float(errors[1]),
            "tau_standard_error": float(errors[2]),
            "C_inf_approximate_95ci": [
                float(parameters[0] - 1.96 * errors[0]),
                float(parameters[0] + 1.96 * errors[0]),
            ],
            "tau_approximate_95ci": [
                float(max(0.0, parameters[2] - 1.96 * errors[2])),
                float(parameters[2] + 1.96 * errors[2]),
            ],
            "R_squared": float(1.0 - residual / total) if total > 0 else np.nan,
            "interpretation": "descriptive convergence scale; not a biological constant",
        }
    except Exception as exc:
        return {"status": "not_fitted", "reason": f"{type(exc).__name__}: {exc}"}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    automatic_root = args.run_root is None
    run_root = (args.run_root or _discover_run_root()).expanduser().resolve()
    print(f"Experiment root: {run_root}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else artifact_directory("real_data", run_root, "stability")
    )
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    experiment = _read_json(run_root / "experiment_manifest.json")
    common_test = _read_json(run_root / "common_test_manifest.json")
    expected_ids = list(map(str, common_test["common_test_ids"]))
    if not expected_ids:
        raise ValueError("Common test set is empty.")
    tasks = list(experiment["tasks"])
    tasks_by_id = {str(task["run_id"]): task for task in tasks}
    if len(tasks_by_id) != len(tasks):
        raise ValueError("Experiment manifest has duplicate run IDs.")

    # A no-argument invocation is intended to be convenient during a running
    # experiment too.  Preserve the strict complete-matrix behavior for an
    # explicitly supplied --run-root, but automatically use the partial
    # analysis when the newest discovered run is not consolidated yet.
    missing_selected = [
        str(task["run_id"])
        for task in tasks
        if not (_task_directory(run_root, task) / "selected_checkpoint.json").is_file()
    ]
    if automatic_root and missing_selected:
        from analyses.analyze_real_exp8_partial import main as partial_main

        print(
            f"Complete analysis is not ready ({len(missing_selected)}/{len(tasks)} "
            "runs lack selected_checkpoint.json); running the partial analysis."
        )
        partial_argv = ["--run-root", str(run_root)]
        if args.output_dir is not None:
            partial_argv.extend(["--output-dir", str(args.output_dir)])
        return partial_main(partial_argv)

    profiles: dict[str, dict[str, dict[str, Any]]] = {}
    mean_checks: list[pd.DataFrame] = []
    for task in tasks:
        run_id = str(task["run_id"])
        directory = _task_directory(run_root, task)
        selected = _read_json(directory / "selected_checkpoint.json")
        if selected.get("checkpoint_variant") != "best_val_loss":
            raise ValueError(f"{run_id} is not based on best validation loss.")
        path = Path(str(selected["shared_profile_path"]))
        profiles[run_id], checks = _load_profiles(
            path=path,
            expected_ids=expected_ids,
            mean_one_tolerance=args.mean_one_tolerance,
        )
        checks.insert(0, "run_id", run_id)
        checks.insert(1, "N", int(task["N"]))
        mean_checks.append(checks)
    mean_check_table = pd.concat(mean_checks, ignore_index=True)
    mean_check_table.to_csv(output_dir / "L_mean_one_checks.csv", index=False)

    overlap = pd.read_csv(run_root / "overlap_report.csv")
    overlap_lookup = {
        frozenset((str(row.run_a), str(row.run_b))): row._asdict()
        for row in overlap.itertuples(index=False)
    }
    designated_rows: list[dict[str, Any]] = []
    designated_groups: dict[tuple[int, str, int], dict[str, str]] = {}
    for task in tasks:
        if task["kind"] != "designated_disjoint_pair":
            continue
        key = (int(task["N"]), str(task["pair_id"]), int(task["training_seed"]))
        designated_groups.setdefault(key, {})[str(task["side"])] = str(task["run_id"])
    for (N, pair_id, training_seed), sides in sorted(designated_groups.items()):
        if set(sides) != {"A", "B"}:
            raise ValueError(f"Incomplete designated pair N={N}, {pair_id}.")
        run_a, run_b = sides["A"], sides["B"]
        overlap_row = overlap_lookup[frozenset((run_a, run_b))]
        if int(overlap_row["intersection_count"]) or int(
            overlap_row["source_family_intersection_count"]
        ):
            raise AssertionError(f"Designated pair {run_a}/{run_b} overlaps.")
        for row in _compare_profiles(
            left=profiles[run_a], right=profiles[run_b], transcript_ids=expected_ids
        ):
            designated_rows.append(
                {
                    "N": N,
                    "pair_id": pair_id,
                    "training_seed": training_seed,
                    "run_a": run_a,
                    "run_b": run_b,
                    "dataset_overlap": int(overlap_row["intersection_count"]),
                    "source_family_overlap": int(
                        overlap_row["source_family_intersection_count"]
                    ),
                    **row,
                }
            )
    disjoint = pd.DataFrame(designated_rows)
    disjoint.to_parquet(
        output_dir / "stability_disjoint_per_transcript.parquet", index=False
    )
    pair_summary_rows: list[dict[str, Any]] = []
    for keys, group in disjoint.groupby(
        ["N", "pair_id", "training_seed", "run_a", "run_b"], sort=True
    ):
        row = dict(zip(("N", "pair_id", "training_seed", "run_a", "run_b"), keys, strict=True))
        for metric in ("PCC", "Spearman", "RMSE", "PCC_interior5", "RMSE_interior5"):
            summary = _distribution_summary(group[metric])
            row.update({f"{name}_{metric}": value for name, value in summary.items()})
        pair_summary_rows.append(row)
    pair_summary = pd.DataFrame(pair_summary_rows)
    pair_summary.to_csv(output_dir / "stability_disjoint_pair_summary.csv", index=False)
    n_summary_rows: list[dict[str, Any]] = []
    for N, group in disjoint.groupby("N", sort=True):
        distribution = _distribution_summary(group["PCC"])
        ci_low, ci_high = _hierarchical_bootstrap(
            group,
            grouping_column="pair_id",
            metric="PCC",
            seed=args.bootstrap_seed + int(N),
            replicates=args.bootstrap_replicates,
        )
        # Optional repeated optimization seeds are averaged within the same
        # dataset-selection pair before estimating between-pair uncertainty.
        pair_estimates = (
            pair_summary.loc[pair_summary["N"] == N]
            .groupby("pair_id", sort=True)["mean_PCC"]
            .mean()
        )
        n_summary_rows.append(
            {
                "N": int(N),
                **{f"transcript_distribution_{key}": value for key, value in distribution.items()},
                "mean_PCC": float(pair_estimates.mean()),
                "median_pair_estimate_PCC": float(pair_estimates.median()),
                "number_of_independent_subset_pairs": int(len(pair_estimates)),
                "pair_estimate_standard_deviation": float(pair_estimates.std(ddof=1))
                if len(pair_estimates) > 1
                else np.nan,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_seed": args.bootstrap_seed + int(N),
                "bootstrap_replicates": args.bootstrap_replicates,
            }
        )
    stability_summary = pd.DataFrame(n_summary_rows)
    stability_summary.to_csv(output_dir / "stability_disjoint_summary.csv", index=False)

    full_tasks = [task for task in tasks if task["kind"] == "full_collection"]
    full_by_seed: dict[int, Mapping[str, Any]] = {}
    for task in full_tasks:
        seed = int(task["training_seed"])
        if seed in full_by_seed:
            raise ValueError(f"Multiple full-collection references for training seed {seed}.")
        full_by_seed[seed] = task
    training_seeds = sorted({int(task["training_seed"]) for task in tasks})
    if set(full_by_seed) != set(training_seeds):
        raise ValueError(
            "Every optional training seed needs exactly one matching full-collection "
            f"reference; full={sorted(full_by_seed)}, all={training_seeds}."
        )
    full_N_values = {int(task["N"]) for task in full_tasks}
    if len(full_N_values) != 1:
        raise ValueError(f"Full-collection references disagree on N: {full_N_values}.")
    full_N = next(iter(full_N_values))
    full_run_ids = [str(full_by_seed[seed]["run_id"]) for seed in training_seeds]
    convergence_rows: list[dict[str, Any]] = []
    for task in tasks:
        run_id = str(task["run_id"])
        training_seed = int(task["training_seed"])
        full_run = str(full_by_seed[training_seed]["run_id"])
        if run_id == full_run:
            continue
        for row in _compare_profiles(
            left=profiles[run_id], right=profiles[full_run], transcript_ids=expected_ids
        ):
            convergence_rows.append(
                {
                    "N": int(task["N"]),
                    "run_id": run_id,
                    "subset_design_id": str(task.get("base_run_id", run_id)),
                    "training_seed": training_seed,
                    "full_reference_run_id": full_run,
                    **row,
                }
            )
    convergence = pd.DataFrame(convergence_rows)
    convergence.to_parquet(
        output_dir / "convergence_to_full_per_transcript.parquet", index=False
    )
    convergence_run_rows: list[dict[str, Any]] = []
    for keys, group in convergence.groupby(
        ["N", "subset_design_id", "training_seed", "run_id", "full_reference_run_id"],
        sort=True,
    ):
        N, subset_design_id, training_seed, run_id, full_reference_run_id = keys
        row = {
            "N": int(N),
            "subset_design_id": str(subset_design_id),
            "training_seed": int(training_seed),
            "run_id": str(run_id),
            "full_reference_run_id": str(full_reference_run_id),
        }
        for metric in ("PCC", "Spearman", "RMSE", "PCC_interior5", "RMSE_interior5"):
            row.update(
                {
                    f"{name}_{metric}": value
                    for name, value in _distribution_summary(group[metric]).items()
                }
            )
        convergence_run_rows.append(row)
    convergence_run_summary = pd.DataFrame(convergence_run_rows)
    convergence_run_summary.to_csv(
        output_dir / "convergence_to_full_run_summary.csv", index=False
    )
    convergence_n_rows: list[dict[str, Any]] = []
    convergence_design_summary = (
        convergence_run_summary.groupby(["N", "subset_design_id"], as_index=False)
        .agg(
            mean_PCC=("mean_PCC", "mean"),
            optimization_seed_standard_deviation=("mean_PCC", "std"),
            number_of_training_seeds=("training_seed", "nunique"),
        )
    )
    convergence_design_summary.to_csv(
        output_dir / "convergence_to_full_subset_summary.csv", index=False
    )
    for N, group in convergence_design_summary.groupby("N", sort=True):
        values = group["mean_PCC"]
        convergence_n_rows.append(
            {
                "N": int(N),
                "number_of_subsets": len(values),
                "mean_PCC": float(values.mean()),
                "median_subset_mean_PCC": float(values.median()),
                "subset_standard_deviation": float(values.std(ddof=1))
                if len(values) > 1
                else np.nan,
                "minimum_subset_mean_PCC": float(values.min()),
                "maximum_subset_mean_PCC": float(values.max()),
            }
        )
    convergence_summary = pd.DataFrame(convergence_n_rows)
    convergence_summary.to_csv(
        output_dir / "convergence_to_full_summary.csv", index=False
    )

    all_pair_rows: list[dict[str, Any]] = []
    for N, same_n_tasks in itertools.groupby(
        sorted(tasks, key=lambda task: (int(task["N"]), str(task["run_id"]))),
        key=lambda task: int(task["N"]),
    ):
        same_n = list(same_n_tasks)
        for left_task, right_task in itertools.combinations(same_n, 2):
            run_a, run_b = str(left_task["run_id"]), str(right_task["run_id"])
            comparisons = pd.DataFrame(
                _compare_profiles(
                    left=profiles[run_a], right=profiles[run_b], transcript_ids=expected_ids
                )
            )
            overlap_row = overlap_lookup[frozenset((run_a, run_b))]
            all_pair_rows.append(
                {
                    "N": int(N),
                    "run_a": run_a,
                    "run_b": run_b,
                    "dataset_intersection": int(overlap_row["intersection_count"]),
                    "overlap_fraction_a": float(overlap_row["overlap_fraction_a"]),
                    "overlap_fraction_b": float(overlap_row["overlap_fraction_b"]),
                    "jaccard": float(overlap_row["jaccard"]),
                    "source_family_intersection_count": int(
                        overlap_row["source_family_intersection_count"]
                    ),
                    "is_designated_disjoint_pair": bool(
                        overlap_row["is_designated_disjoint_pair"]
                    ),
                    "mean_transcript_PCC": float(comparisons["PCC"].mean()),
                    "median_transcript_PCC": float(comparisons["PCC"].median()),
                    "mean_transcript_RMSE": float(comparisons["RMSE"].mean()),
                }
            )
    all_pairs = pd.DataFrame(all_pair_rows)
    all_pairs.to_csv(output_dir / "same_N_all_pairs.csv", index=False)

    diversity = pd.read_csv(run_root / "subset_diversity.csv")
    diversity.to_csv(output_dir / "subset_diversity.csv", index=False)
    diversity_response = diversity.merge(
        convergence_run_summary[["run_id", "N", "mean_PCC"]],
        on=["run_id", "N"],
        how="left",
    ).rename(columns={"mean_PCC": "convergence_to_full_mean_PCC"})
    diversity_response.to_csv(output_dir / "diversity_convergence_response.csv", index=False)
    diversity_by_run = diversity.set_index("run_id")
    pair_diversity_rows: list[dict[str, Any]] = []
    for row in pair_summary.itertuples(index=False):
        left = diversity_by_run.loc[str(row.run_a)]
        right = diversity_by_run.loc[str(row.run_b)]
        pair_diversity_rows.append(
            {
                "N": int(row.N),
                "pair_id": str(row.pair_id),
                "training_seed": int(row.training_seed),
                "run_a": str(row.run_a),
                "run_b": str(row.run_b),
                "pair_stability_mean_PCC": float(row.mean_PCC),
                "pair_diversity_raw": float(
                    np.nanmean([left["diversity_raw"], right["diversity_raw"]])
                )
                if np.isfinite([left["diversity_raw"], right["diversity_raw"]]).any()
                else np.nan,
                "pair_diversity_zscored": float(
                    np.nanmean(
                        [left["diversity_zscored"], right["diversity_zscored"]]
                    )
                )
                if np.isfinite(
                    [left["diversity_zscored"], right["diversity_zscored"]]
                ).any()
                else np.nan,
            }
        )
    pair_diversity_response = pd.DataFrame(pair_diversity_rows)
    pair_diversity_response.to_csv(
        output_dir / "diversity_pair_stability_response.csv", index=False
    )
    diversity_associations: list[dict[str, Any]] = []
    for N, group in diversity_response.groupby("N", sort=True):
        for column in ("diversity_raw", "diversity_zscored"):
            valid = group[[column, "convergence_to_full_mean_PCC"]].dropna()
            diversity_associations.append(
                {
                    "response": "convergence_to_full_mean_PCC",
                    "N": int(N),
                    "diversity_definition": column,
                    "number_of_subsets": len(valid),
                    "Spearman": _spearman(
                        valid[column].to_numpy(dtype=float),
                        valid["convergence_to_full_mean_PCC"].to_numpy(dtype=float),
                    )
                    if len(valid) >= 3
                    else np.nan,
                    "causal_interpretation": False,
                }
            )
    for N, group in pair_diversity_response.groupby("N", sort=True):
        for column in ("pair_diversity_raw", "pair_diversity_zscored"):
            valid = group[[column, "pair_stability_mean_PCC"]].dropna()
            diversity_associations.append(
                {
                    "response": "designated_pair_stability_mean_PCC",
                    "N": int(N),
                    "diversity_definition": column,
                    "number_of_subsets": len(valid),
                    "Spearman": _spearman(
                        valid[column].to_numpy(dtype=float),
                        valid["pair_stability_mean_PCC"].to_numpy(dtype=float),
                    )
                    if len(valid) >= 3
                    else np.nan,
                    "causal_interpretation": False,
                }
            )
    pd.DataFrame(diversity_associations).to_csv(
        output_dir / "diversity_associations.csv", index=False
    )

    saturation = _saturation_fit(convergence_summary)
    (output_dir / "saturation_fit.json").write_text(
        json.dumps(saturation, indent=2, sort_keys=True), encoding="utf-8"
    )
    _save_stability_figure(pair_summary, stability_summary, figure_dir)
    _save_convergence_figure(
        convergence_run_summary, convergence_summary, full_N, figure_dir
    )
    _save_overlap_figure(all_pairs, figure_dir)
    quality = pd.read_csv(run_root / "subset_quality_report.csv")
    _save_quality_figure(quality, figure_dir)
    _representative_profiles(
        disjoint=disjoint,
        tasks_by_id=tasks_by_id,
        profiles=profiles,
        output_dir=output_dir,
        figure_dir=figure_dir,
    )
    if diversity["fingerprint_table_supplied"].astype(bool).any():
        figure, axis = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)
        for N, group in diversity_response.groupby("N", sort=True):
            axis.scatter(
                group["diversity_zscored"],
                group["convergence_to_full_mean_PCC"],
                label=f"N={N}",
                alpha=0.75,
            )
        axis.set_xlabel("Subset gamma-fingerprint diversity (z-scored features)")
        axis.set_ylabel(f"Mean PCC with full N={full_N} representation")
        axis.set_title("Post-hoc diversity association (not causal)")
        axis.legend(frameon=False, ncol=2)
        axis.grid(alpha=0.2)
        for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
            figure.savefig(
                figure_dir / f"diversity_vs_stability.{suffix}",
                bbox_inches="tight",
                **kwargs,
            )
        plt.close(figure)

    manifest = {
        "experiment": "real_exp8_L_stability",
        "run_root": str(run_root),
        "common_test_transcript_count": len(expected_ids),
        "number_of_trained_models": len(tasks),
        "full_reference_run_ids": full_run_ids,
        "full_reference_is_biological_ground_truth": False,
        "primary_statistic": "designated source-family-disjoint A/B mean transcript PCC",
        "all_pair_statistic_is_secondary_and_overlap_reported": True,
        "analysis_renormalized_L": False,
        "maximum_absolute_mean_one_deviation": float(
            mean_check_table["absolute_mean_one_deviation"].max()
        ),
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "saturation_fit": saturation,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("\nPrimary designated-disjoint stability:")
    print(stability_summary.to_string(index=False, float_format=lambda value: f"{value:.5g}"))
    print(f"\nAnalysis complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
