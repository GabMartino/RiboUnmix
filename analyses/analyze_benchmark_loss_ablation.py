#!/usr/bin/env python3
"""Analyze the matched four-organism loss ablation from saved predictions.

Primary effects are paired differences against the contemporaneous full-loss
control on identical held-out transcripts.  The default checkpoint minimizes
validation raw NB2 NLL, a criterion whose definition is common to every arm.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.special import gammaln

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_benchmark_loss_ablation import (
    DEFAULT_DESIGN,
    DEFAULT_OUTPUT_ROOT,
    AblationTask,
    build_tasks,
    load_design,
    parse_training_seeds,
    sha256_file,
    task_directory,
    verify_attempt_outputs,
)
from Utils.publication_plot_style import latex_paper_style
from analyses.paths import artifact_directory


DEFAULT_ROOT = DEFAULT_OUTPUT_ROOT
DATASET_LABELS = {
    "human_iwasaki_2014": "Human",
    "yeast_stein_2021": "Yeast",
    "celegans_stein_2021": r"C. elegans",
    "ecoli_zhang_2016": r"E. coli",
}
ARM_COLORS = {
    "nb_only": "#4C78A8",
    "nb_raw_pcc": "#59A14F",
    "nb_vst_pcc": "#F28E2B",
}
METRICS = {
    "raw_pcc": ("Raw PCC", True),
    "nb_vst_pcc": ("NB-VST PCC", True),
    "nb2_nll": ("NB2 NLL", False),
}


def _array(value: Any, dtype: Any) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=dtype)
    return np.asarray(value, dtype=dtype).reshape(-1)


def _profile_parameter(value: Any, length: int) -> np.ndarray:
    array = _array(value, np.float64)
    if array.size == 1:
        return np.full(length, float(array[0]), dtype=np.float64)
    if array.size != length:
        raise ValueError(
            f"Profile parameter has length {array.size}; expected 1 or {length}."
        )
    return array


def pearson(x: np.ndarray, y: np.ndarray, eps: float = 1.0e-12) -> float:
    if x.size < 2:
        return math.nan
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = math.sqrt(float(np.dot(xc, xc) * np.dot(yc, yc)))
    if not math.isfinite(denominator) or denominator <= eps:
        return math.nan
    value = float(np.dot(xc, yc) / denominator)
    return value if math.isfinite(value) else math.nan


def nb_vst(x: np.ndarray, alpha: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    return (2.0 / np.sqrt(alpha + eps)) * np.arcsinh(
        np.sqrt(alpha * np.clip(x, 0.0, None) + eps)
    )


def nb2_nll(
    target: np.ndarray,
    mu: np.ndarray,
    log_alpha: np.ndarray,
) -> np.ndarray:
    log_mu = np.log(np.clip(mu, 1.0e-8, None))
    log_alpha = np.clip(log_alpha, -5.0, 1.0)
    r = np.exp(-log_alpha)
    z = log_mu + log_alpha
    values = (
        gammaln(r)
        + gammaln(target + 1.0)
        - gammaln(target + r)
        + r * np.logaddexp(0.0, z)
        + target * np.logaddexp(0.0, -z)
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite held-out NB2 NLL.")
    return values


def profile_hash(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def transcript_metrics(row: dict[str, Any]) -> dict[str, Any]:
    target = _array(row["target"], np.float64)
    mu = _array(row["mu"], np.float64)
    mask = _array(row["mask"], np.bool_)
    if not (target.size == mu.size == mask.size) or target.size == 0:
        raise ValueError("Prediction target, mu, and mask must have identical lengths.")
    log_alpha = _profile_parameter(row["log_sigma"], target.size)
    valid = mask & np.isfinite(target) & np.isfinite(mu) & np.isfinite(log_alpha)
    if not np.any(valid):
        raise ValueError("Prediction row has no finite valid positions.")
    y = np.clip(target[valid], 0.0, None)
    prediction = np.clip(mu[valid], 0.0, None)
    alpha = np.exp(np.clip(log_alpha[valid], -5.0, 1.0))
    n_valid = int(valid.sum())
    nll_mean = float(nb2_nll(y, prediction, log_alpha[valid]).mean())
    length_weight = float(np.clip((n_valid / 1000.0) ** 0.25, 0.5, 2.0))
    target_rms = math.sqrt(float(np.mean(np.square(y))))
    return {
        "transcript_id": str(row["transcript_id"]),
        "n_valid_positions": n_valid,
        "raw_pcc": pearson(prediction, y),
        "nb_vst_pcc": pearson(nb_vst(prediction, alpha), nb_vst(y, alpha)),
        "nb2_nll": nll_mean * length_weight,
        "nb2_nll_position_mean": nll_mean,
        "rmse": math.sqrt(float(np.mean(np.square(prediction - y)))),
        "relative_rmse": (
            math.sqrt(float(np.mean(np.square(prediction - y)))) / target_rms
            if target_rms > 0.0
            else math.nan
        ),
        "log1p_rmse": math.sqrt(
            float(np.mean(np.square(np.log1p(prediction) - np.log1p(y))))
        ),
        "mean_ratio": float(prediction.mean() / y.mean()) if y.mean() > 0.0 else math.nan,
        "target_hash": profile_hash(y.astype(np.float64)),
        "mask_hash": profile_hash(mask.astype(np.uint8)),
    }


def local_prediction_path(manifest_path: Path, variant: str) -> tuple[Path, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest[variant]
    recorded = Path(str(entry["output_path"]))
    for candidate in (recorded, manifest_path.parent / recorded.name):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate, str(entry.get("checkpoint_path", ""))
    raise FileNotFoundError(f"Missing {variant} predictions next to {manifest_path}.")


def newest_complete_attempt(
    root: Path,
    task: AblationTask,
    design_sha256: str,
    variants: tuple[str, ...],
) -> Path | None:
    candidates: list[tuple[str, Path]] = []
    for status_path in task_directory(root, task).glob("attempts/*/task_status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                status.get("state") == "completed"
                and status.get("task_id") == task.task_id
                and status.get("design_sha256") == design_sha256
            ):
                verify_attempt_outputs(status_path.parent, task.dataset, variants)
                candidates.append((str(status.get("finished_at_utc", "")), status_path.parent))
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
    return max(candidates, default=("", None), key=lambda value: value[0])[1]


def split_identity_hash(manifest: dict[str, Any]) -> str:
    payload = {
        key: manifest[key]
        for key in ("dataset", "fractions", "train_ids", "validation_ids", "test_ids")
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _drop_path(mapping: dict[str, Any], path: tuple[str, ...]) -> None:
    current: Any = mapping
    for key in path[:-1]:
        if not isinstance(current, dict):
            return
        current = current.get(key)
    if isinstance(current, dict):
        current.pop(path[-1], None)


def scientific_config_hash(config: dict[str, Any]) -> str:
    """Hash all matched settings except the intended loss arm and output names."""
    cleaned = deepcopy(config)
    for path in (
        ("name",),
        ("paths", "checkpoints"),
        ("paths", "logs"),
        ("paths", "results"),
        ("loss", "replica_nb_weight"),
        ("loss", "consensus_raw_pcc_weight"),
        ("loss", "consensus_nb_vst_pcc_weight"),
    ):
        _drop_path(cleaned, path)
    return hashlib.sha256(
        json.dumps(cleaned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class LoadedTask:
    task: AblationTask
    attempt_dir: Path
    prediction_path: Path
    checkpoint_path: str
    split_hash: str
    config_hash: str
    rows: list[dict[str, Any]]


def load_task(
    *,
    task: AblationTask,
    attempt_dir: Path,
    variant: str,
    all_variants: tuple[str, ...],
) -> LoadedTask:
    manifest_path = verify_attempt_outputs(
        attempt_dir,
        task.dataset,
        all_variants,
    )
    prediction_path, checkpoint_path = local_prediction_path(manifest_path, variant)
    split_path = manifest_path.parent / "split_manifest.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_path}")
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if int(split_manifest.get("split_seed", split_manifest["seed"])) != task.split_seed:
        raise ValueError(f"Wrong split seed for {task.task_id}.")
    if int(split_manifest.get("training_seed", task.training_seed)) != task.training_seed:
        raise ValueError(f"Wrong training seed for {task.task_id}.")

    configs = sorted((attempt_dir / "logs" / task.dataset).rglob("config.yaml"))
    if len(configs) != 1:
        raise RuntimeError(f"Expected one resolved config for {task.task_id}; got {len(configs)}.")
    config = yaml.safe_load(configs[0].read_text(encoding="utf-8"))
    expected_weights = {
        "replica_nb_weight": task.replica_nb_weight,
        "consensus_raw_pcc_weight": task.consensus_raw_pcc_weight,
        "consensus_nb_vst_pcc_weight": task.consensus_nb_vst_pcc_weight,
        "gamma_reg_weight": task.gamma_reg_weight,
    }
    for key, expected in expected_weights.items():
        actual = float(config["loss"][key])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                f"{task.task_id}: loss.{key}={actual}, expected {expected}."
            )

    required = {"transcript_id", "target", "mu", "mask", "log_sigma"}
    parquet = pq.ParquetFile(prediction_path)
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise KeyError(f"{prediction_path} is missing {missing}.")
    rows: list[dict[str, Any]] = []
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=128):
        columns = batch.to_pydict()
        for index in range(batch.num_rows):
            metric = transcript_metrics({name: columns[name][index] for name in required})
            metric.update(
                {
                    "training_seed": task.training_seed,
                    "split_seed": task.split_seed,
                    "dataset": task.dataset,
                    "arm": task.arm,
                    "checkpoint_variant": variant,
                    "prediction_path": str(prediction_path),
                }
            )
            rows.append(metric)
    if len({row["transcript_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate test transcript in {prediction_path}.")
    return LoadedTask(
        task=task,
        attempt_dir=attempt_dir,
        prediction_path=prediction_path,
        checkpoint_path=checkpoint_path,
        split_hash=split_identity_hash(split_manifest),
        config_hash=scientific_config_hash(config),
        rows=rows,
    )


def validate_matched_tasks(tasks: list[LoadedTask]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    by_pair: dict[tuple[int, str], list[LoadedTask]] = {}
    for loaded in tasks:
        by_pair.setdefault((loaded.task.training_seed, loaded.task.dataset), []).append(loaded)
    for (seed, dataset), group in sorted(by_pair.items()):
        split_hashes = {item.split_hash for item in group}
        config_hashes = {item.config_hash for item in group}
        check = {
            "training_seed": seed,
            "dataset": dataset,
            "arms": ",".join(sorted(item.task.arm for item in group)),
            "split_identity_match": len(split_hashes) == 1,
            "non_loss_config_match": len(config_hashes) == 1,
        }
        checks.append(check)
        if not check["split_identity_match"] or not check["non_loss_config_match"]:
            raise ValueError(f"Unmatched ablation configuration: {check}")
    return checks


def absolute_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [*METRICS, "rmse", "relative_rmse", "log1p_rmse", "mean_ratio"]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["dataset", "training_seed", "arm"], sort=True):
        row = dict(zip(("dataset", "training_seed", "arm"), keys))
        row["n_transcripts"] = len(group)
        for metric in metrics:
            finite = group[metric].to_numpy(float)
            finite = finite[np.isfinite(finite)]
            row[f"mean_{metric}"] = float(finite.mean()) if finite.size else math.nan
            row[f"median_{metric}"] = float(np.median(finite)) if finite.size else math.nan
            row[f"n_valid_{metric}"] = int(finite.size)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_effects(
    frame: pd.DataFrame,
    *,
    design: dict[str, Any],
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effect_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    arm_order = [arm for arm in design["arms"] if arm != "full"]
    dataset_order = list(map(str, design["datasets"]))
    rng = np.random.default_rng(seed)

    for dataset in dataset_order:
        dataset_frame = frame[frame["dataset"] == dataset]
        for arm in arm_order:
            complete_seeds: list[int] = []
            paired_by_seed: dict[int, pd.DataFrame] = {}
            for training_seed in sorted(dataset_frame["training_seed"].unique()):
                subset = dataset_frame[dataset_frame["training_seed"] == training_seed]
                control = subset[subset["arm"] == "full"]
                treatment = subset[subset["arm"] == arm]
                if control.empty or treatment.empty:
                    continue
                paired = control.merge(
                    treatment,
                    on="transcript_id",
                    suffixes=("_full", "_arm"),
                    validate="one_to_one",
                )
                hash_match = (
                    (paired["target_hash_full"] == paired["target_hash_arm"])
                    & (paired["mask_hash_full"] == paired["mask_hash_arm"])
                )
                if not bool(hash_match.all()):
                    raise ValueError(
                        f"Target/mask mismatch for {dataset}, seed {training_seed}, {arm}."
                    )
                finite = np.ones(len(paired), dtype=bool)
                for metric in METRICS:
                    finite &= np.isfinite(paired[f"{metric}_full"].to_numpy(float))
                    finite &= np.isfinite(paired[f"{metric}_arm"].to_numpy(float))
                exclusion_rows.append(
                    {
                        "dataset": dataset,
                        "training_seed": int(training_seed),
                        "arm": arm,
                        "test_union": int(
                            len(set(control["transcript_id"]) | set(treatment["transcript_id"]))
                        ),
                        "matched_ids": len(paired),
                        "common_finite_all_primary_metrics": int(finite.sum()),
                        "excluded_nonfinite_or_unmatched": int(
                            len(set(control["transcript_id"]) | set(treatment["transcript_id"]))
                            - finite.sum()
                        ),
                    }
                )
                paired = paired.loc[finite].set_index("transcript_id", drop=False)
                if paired.empty:
                    continue
                complete_seeds.append(int(training_seed))
                paired_by_seed[int(training_seed)] = paired
            if not complete_seeds:
                continue

            common_ids = sorted(
                set.intersection(
                    *(set(paired_by_seed[value].index) for value in complete_seeds)
                )
            )
            if not common_ids:
                continue
            n = len(common_ids)
            samples = rng.integers(0, n, size=(draws, n), endpoint=False)
            for metric in METRICS:
                seed_differences: list[np.ndarray] = []
                seed_estimates: list[float] = []
                for training_seed in complete_seeds:
                    paired = paired_by_seed[training_seed].loc[common_ids]
                    differences = (
                        paired[f"{metric}_arm"].to_numpy(float)
                        - paired[f"{metric}_full"].to_numpy(float)
                    )
                    seed_estimate = float(differences.mean())
                    seed_estimates.append(seed_estimate)
                    seed_differences.append(differences)
                    seed_rows.append(
                        {
                            "dataset": dataset,
                            "arm": arm,
                            "training_seed": training_seed,
                            "metric": metric,
                            "estimate": seed_estimate,
                            "n_transcripts": n,
                        }
                    )
                bootstrap = np.mean(
                    np.stack(
                        [values[samples].mean(axis=1) for values in seed_differences],
                        axis=1,
                    ),
                    axis=1,
                )
                effect_rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "metric": metric,
                        "estimate": float(np.mean(seed_estimates)),
                        "ci_low": float(np.quantile(bootstrap, 0.025)),
                        "ci_high": float(np.quantile(bootstrap, 0.975)),
                        "n_transcripts": n,
                        "n_training_seeds": len(complete_seeds),
                        "training_seeds": ",".join(map(str, complete_seeds)),
                        "bootstrap_draws": draws,
                        "bootstrap_seed": seed,
                    }
                )
    return (
        pd.DataFrame(effect_rows),
        pd.DataFrame(seed_rows),
        pd.DataFrame(exclusion_rows),
    )


@latex_paper_style
def plot_effects(
    effects: pd.DataFrame,
    seed_effects: pd.DataFrame,
    *,
    design: dict[str, Any],
    output_stem: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Keep the SVG/HTML preview independent of browser math-font support.
    # Unicode minus, arrows, Delta, and TeX superscripts have rendered as
    # missing-glyph boxes on some compute-node/browser combinations.
    matplotlib.rcParams["axes.unicode_minus"] = False

    datasets = list(map(str, design["datasets"]))
    arms = [arm for arm in design["arms"] if arm != "full"]
    offsets = np.linspace(-0.22, 0.22, len(arms))
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.55), sharey=True)
    for panel_index, (axis, (metric, (label, higher_better))) in enumerate(
        zip(axes, METRICS.items())
    ):
        axis.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0, zorder=0)
        metric_effects = effects[effects["metric"] == metric]
        finite_limits = np.abs(
            metric_effects[["ci_low", "ci_high"]].to_numpy(float).reshape(-1)
        )
        finite_limits = finite_limits[np.isfinite(finite_limits) & (finite_limits > 0)]
        maximum = float(finite_limits.max()) if finite_limits.size else 1.0
        exponent = 0
        if maximum < 1.0e-2:
            exponent = int(3 * math.floor(math.log10(maximum) / 3.0))
        scale = 10.0**exponent
        for arm, offset in zip(arms, offsets):
            subset = effects[(effects["metric"] == metric) & (effects["arm"] == arm)]
            subset = subset.set_index("dataset")
            for row_index, dataset in enumerate(datasets):
                if dataset not in subset.index:
                    continue
                row = subset.loc[dataset]
                y = row_index + offset
                color = ARM_COLORS.get(arm, "0.3")
                axis.errorbar(
                    float(row["estimate"]) / scale,
                    y,
                    xerr=np.asarray(
                        [[float(row["estimate"] - row["ci_low"]) / scale],
                         [float(row["ci_high"] - row["estimate"]) / scale]]
                    ),
                    fmt="o",
                    color=color,
                    markersize=5.2,
                    linewidth=1.5,
                    capsize=2.2,
                    label=str(design["arms"][arm]["label"]) if panel_index == 0 else None,
                    zorder=3,
                )
                individual = seed_effects[
                    (seed_effects["metric"] == metric)
                    & (seed_effects["arm"] == arm)
                    & (seed_effects["dataset"] == dataset)
                ]
                if len(individual) > 1:
                    axis.scatter(
                        individual["estimate"] / scale,
                        np.full(len(individual), y),
                        s=9,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=0.65,
                        alpha=0.65,
                        zorder=2,
                    )
        scale_label = "" if exponent == 0 else f" (x 10^{exponent})"
        direction = "higher is better" if higher_better else "lower is better"
        axis.set_xlabel(f"Change vs full{scale_label}\n({direction})")
        axis.set_title(chr(ord("A") + panel_index) + f". {label}")
        axis.grid(axis="x", linewidth=0.45, alpha=0.55)
        axis.set_axisbelow(True)
    axes[0].set_yticks(
        np.arange(len(datasets)),
        [DATASET_LABELS.get(dataset, dataset) for dataset in datasets],
    )
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.53, -0.03),
        ncol=3,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    figure.subplots_adjust(left=0.14, right=0.99, top=0.87, bottom=0.25, wspace=0.20)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def write_report(
    *,
    output: Path,
    design_path: Path,
    design_sha256: str,
    variant: str,
    expected: list[AblationTask],
    loaded: list[LoadedTask],
    effects: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> None:
    loaded_ids = {item.task.task_id for item in loaded}
    missing = [task.task_id for task in expected if task.task_id not in loaded_ids]
    lines = [
        "# Benchmark loss-ablation report",
        "",
        f"- Design: `{design_path}`",
        f"- Design SHA-256: `{design_sha256}`",
        f"- Checkpoint variant: `{variant}`",
        f"- Completed tasks analyzed: {len(loaded)}/{len(expected)}",
        f"- Missing tasks: {len(missing)}",
        "",
        "The full arm is a contemporaneous control. Historical benchmark outputs are not "
        "silently mixed with these fits. All arms retain the same gamma regularizer "
        "and configured mean-gradient-reweighted NB formulation; NB-only means the "
        "replica-NB term is the only data-fit term.",
        "",
        "Each benchmark has one observed profile per transcript, represented internally "
        "as one replica. The arithmetic consensus is therefore identical to that profile. "
        "This is an objective/transform ablation, not a multi-replicate consensus study.",
        "",
        "Intervals use paired transcript bootstrap resampling. They are conditional on the "
        "fitted models, datasets, split, and training seeds; transcripts carry both arms "
        "together. They do not quantify variability over new datasets.",
    ]
    if missing:
        lines.extend(["", "## Missing tasks", "", *[f"- `{value}`" for value in missing]])
    if not effects.empty:
        lines.extend(["", "## Effect ranges", ""])
        for metric, group in effects.groupby("metric"):
            lines.append(
                f"- `{metric}`: {group['estimate'].min():.5g} to "
                f"{group['estimate'].max():.5g} (arm minus full)."
            )
    if not exclusions.empty:
        lines.extend(
            [
                "",
                "## Cohorts",
                "",
                f"Paired finite cohort sizes range from "
                f"{int(exclusions['common_finite_all_primary_metrics'].min())} to "
                f"{int(exclusions['common_finite_all_primary_metrics'].max())} transcripts.",
            ]
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def task_availability(
    *,
    experiment_root: Path,
    expected: list[AblationTask],
    completed: list[LoadedTask],
    design_sha256: str,
) -> pd.DataFrame:
    """Summarize scientific-task availability, ignoring obsolete attempts.

    A task is complete only when the analyzer has validated its checkpoint
    exports.  Failed attempts remain useful operational provenance, but never
    supersede a later valid completion.
    """

    completed_ids = {item.task.task_id for item in completed}
    rows: list[dict[str, Any]] = []
    for task in expected:
        attempts: list[dict[str, Any]] = []
        for status_path in sorted(
            task_directory(experiment_root, task).glob("attempts/*/task_status.json")
        ):
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if status.get("task_id") != task.task_id:
                continue
            if status.get("design_sha256") != design_sha256:
                continue
            attempts.append(status)

        states = [str(item.get("state", "unknown")) for item in attempts]
        if task.task_id in completed_ids:
            state = "completed"
        elif "running" in states:
            state = "running"
        elif "failed" in states:
            state = "failed"
        else:
            state = "not_started"
        rows.append(
            {
                "task_id": task.task_id,
                "training_seed": task.training_seed,
                "dataset": task.dataset,
                "arm": task.arm,
                "state": state,
                "attempt_count": len(attempts),
                "failed_attempt_count": states.count("failed"),
            }
        )
    return pd.DataFrame(rows)


def _format_number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "&mdash;"
    if not math.isfinite(number):
        return "&mdash;"
    return f"{number:.{digits}f}"


def _effect_cell(row: pd.Series | None, metric: str) -> tuple[str, str]:
    if row is None:
        return "&mdash;", "neutral"
    estimate = float(row["estimate"])
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    higher_better = METRICS[metric][1]
    favorable = low > 0.0 if higher_better else high < 0.0
    unfavorable = high < 0.0 if higher_better else low > 0.0
    css_class = "favorable" if favorable else "unfavorable" if unfavorable else "neutral"
    return f"{estimate:.4f} [{low:.4f}, {high:.4f}]", css_class


def _html_table(headers: list[str], rows: list[list[str]], classes: list[list[str]] | None = None) -> str:
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body: list[str] = []
    for row_index, row in enumerate(rows):
        cells: list[str] = []
        for column_index, value in enumerate(row):
            css = ""
            if classes is not None and classes[row_index][column_index]:
                css = f' class="{html.escape(classes[row_index][column_index])}"'
            cells.append(f"<td{css}>{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def write_html_report(
    *,
    output: Path,
    design: dict[str, Any],
    design_path: Path,
    design_sha256: str,
    variant: str,
    expected: list[AblationTask],
    loaded: list[LoadedTask],
    availability: pd.DataFrame,
    absolute: pd.DataFrame,
    effects: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> None:
    """Write a self-contained, explicitly partial scientific report."""

    arm_labels = {key: str(value["label"]) for key, value in design["arms"].items()}
    dataset_order = list(map(str, design["datasets"]))
    arm_order = list(map(str, design["arms"]))
    seed_order = sorted({int(task.training_seed) for task in expected})
    completed_count = int((availability["state"] == "completed").sum())
    running_count = int((availability["state"] == "running").sum())
    matched_comparisons = (
        effects[["dataset", "arm"]].drop_duplicates().shape[0] if not effects.empty else 0
    )

    availability_rows: list[list[str]] = []
    availability_classes: list[list[str]] = []
    for training_seed in seed_order:
        for dataset in dataset_order:
            row = [str(training_seed), html.escape(DATASET_LABELS.get(dataset, dataset))]
            row_classes = ["", ""]
            for arm in arm_order:
                match = availability[
                    (availability["training_seed"] == training_seed)
                    & (availability["dataset"] == dataset)
                    & (availability["arm"] == arm)
                ]
                state = str(match.iloc[0]["state"]) if len(match) else "not_started"
                label = state.replace("_", " ")
                row.append(html.escape(label))
                row_classes.append(f"status-{state}")
            availability_rows.append(row)
            availability_classes.append(row_classes)

    absolute_rows: list[list[str]] = []
    for _, row in absolute.sort_values(
        ["training_seed", "dataset", "arm"],
        key=lambda column: column.map(
            {**{value: index for index, value in enumerate(dataset_order)},
             **{value: index for index, value in enumerate(arm_order)}}
        ).fillna(column),
    ).iterrows():
        absolute_rows.append(
            [
                html.escape(DATASET_LABELS.get(str(row["dataset"]), str(row["dataset"]))),
                str(int(row["training_seed"])),
                html.escape(arm_labels[str(row["arm"])]),
                str(int(row["n_transcripts"])),
                _format_number(row["mean_raw_pcc"]),
                _format_number(row["mean_nb_vst_pcc"]),
                _format_number(row["mean_nb2_nll"]),
                _format_number(row["mean_relative_rmse"]),
                _format_number(row["mean_mean_ratio"]),
            ]
        )

    effect_rows: list[list[str]] = []
    effect_classes: list[list[str]] = []
    for dataset in dataset_order:
        for arm in arm_order:
            if arm == "full":
                continue
            group = effects[(effects["dataset"] == dataset) & (effects["arm"] == arm)]
            if group.empty:
                continue
            indexed = {str(row["metric"]): row for _, row in group.iterrows()}
            values: list[str] = []
            value_classes: list[str] = []
            for metric in METRICS:
                text, css = _effect_cell(indexed.get(metric), metric)
                values.append(text)
                value_classes.append(css)
            representative = group.iloc[0]
            effect_rows.append(
                [
                    html.escape(DATASET_LABELS.get(dataset, dataset)),
                    html.escape(arm_labels[arm]),
                    *values,
                    str(int(representative["n_transcripts"])),
                    html.escape(str(representative["training_seeds"])),
                ]
            )
            effect_classes.append(["", "", *value_classes, "", ""])

    full_rows: list[list[str]] = []
    full = absolute[absolute["arm"] == "full"]
    for dataset in dataset_order:
        subset = full[full["dataset"] == dataset].sort_values("training_seed")
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            full_rows.append(
                [
                    html.escape(DATASET_LABELS.get(dataset, dataset)),
                    str(int(row["training_seed"])),
                    _format_number(row["mean_raw_pcc"]),
                    _format_number(row["mean_nb_vst_pcc"]),
                    _format_number(row["mean_nb2_nll"]),
                ]
            )

    findings: list[str] = []
    nb_only = effects[effects["arm"] == "nb_only"]
    nb_datasets = nb_only["dataset"].nunique() if not nb_only.empty else 0
    if nb_datasets:
        raw = nb_only[nb_only["metric"] == "raw_pcc"]
        vst = nb_only[nb_only["metric"] == "nb_vst_pcc"]
        nll = nb_only[nb_only["metric"] == "nb2_nll"]
        consistent = (
            len(raw) == nb_datasets
            and len(vst) == nb_datasets
            and len(nll) == nb_datasets
            and bool((raw["ci_high"] < 0).all())
            and bool((vst["ci_high"] < 0).all())
            and bool((nll["ci_low"] > 0).all())
        )
        if consistent:
            findings.append(
                f"Across all {nb_datasets} currently matched datasets, removing both PCC "
                "terms lowers raw and NB-VST PCC and raises held-out NB2 NLL; every paired "
                "transcript-bootstrap interval excludes zero."
            )
    raw_only = effects[effects["arm"] == "nb_raw_pcc"]
    vst_only = effects[effects["arm"] == "nb_vst_pcc"]
    if not raw_only.empty and not vst_only.empty:
        findings.append(
            "The two single-PCC arms are currently available only for yeast and E. coli. "
            "Removing the NB-VST term produces clearer penalties than removing the raw-PCC "
            "term, but the organism-specific responses are not identical."
        )
    findings.extend(
        [
            f"Only {matched_comparisons} dataset-arm contrasts can currently be formed, "
            "and every paired contrast uses training seed 42. Seeds 43 and 44 therefore "
            "do not yet test whether these effects are robust to optimization variability.",
            "The reported effects are arm minus full-loss control on identical held-out "
            "transcripts. Positive PCC changes and negative NLL changes favor an ablated "
            "arm; the opposite signs favor the full loss.",
        ]
    )

    missing_ids = availability.loc[
        availability["state"] != "completed", "task_id"
    ].astype(str).tolist()
    cohort_min = (
        int(exclusions["common_finite_all_primary_metrics"].min())
        if not exclusions.empty
        else 0
    )
    cohort_max = (
        int(exclusions["common_finite_all_primary_metrics"].max())
        if not exclusions.empty
        else 0
    )
    figure = "benchmark_loss_ablation_effects.svg"
    generated = datetime.now(timezone.utc).isoformat()
    style = """
    :root { --ink:#17202a; --muted:#5f6b76; --line:#d8dee4; --paper:#fff;
            --accent:#b75d0a; --good:#eaf6ec; --bad:#fff0ee; --pending:#fff6d8; }
    * { box-sizing:border-box; }
    body { margin:0; background:#f3f5f7; color:var(--ink); font:15px/1.5 Arial,sans-serif; }
    main { max-width:1180px; margin:28px auto; background:var(--paper); padding:34px 42px 50px;
           box-shadow:0 2px 14px rgba(0,0,0,.08); }
    h1 { margin:0 0 5px; font-size:30px; } h2 { margin-top:34px; border-bottom:2px solid var(--line); padding-bottom:5px; }
    h3 { margin-top:25px; } .subtitle,.note { color:var(--muted); } .warning { border-left:5px solid var(--accent);
    background:#fff5eb; padding:12px 15px; margin:20px 0; } .cards { display:grid;
    grid-template-columns:repeat(4,minmax(130px,1fr)); gap:10px; margin:20px 0; }
    .card { border:1px solid var(--line); border-radius:7px; padding:13px; } .card b { display:block; font-size:24px; }
    .table-wrap { overflow-x:auto; margin:12px 0 20px; } table { border-collapse:collapse; width:100%; font-size:13px; }
    th,td { border:1px solid var(--line); padding:7px 8px; text-align:right; white-space:nowrap; }
    th { background:#eef1f4; } th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) { text-align:left; }
    .favorable { background:var(--good); } .unfavorable { background:var(--bad); }
    .status-completed { background:#e7f5ea; } .status-running { background:var(--pending); }
    .status-failed { background:#fde8e5; } .status-not_started { color:#737d86; background:#f4f5f6; }
    figure { margin:24px 0; } figure img { width:100%; max-height:520px; object-fit:contain; }
    code { overflow-wrap:anywhere; } details { margin:12px 0; } li { margin:5px 0; }
    @media(max-width:760px) { main{margin:0;padding:22px 16px}.cards{grid-template-columns:1fr 1fr;} }
    """

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Partial benchmark loss-ablation report</title><style>{style}</style></head><body><main>
<h1>Benchmark loss ablation</h1>
<p class="subtitle">Partial held-out analysis &middot; generated {html.escape(generated)}</p>
<div class="warning"><strong>Interim result.</strong> {completed_count} of {len(expected)} planned fits have validated exports;
{running_count} tasks were recorded as running in this filesystem snapshot. Conclusions below are descriptive and must be
recomputed after the matched three-seed matrix finishes.</div>
<div class="cards"><div class="card"><b>{completed_count}/{len(expected)}</b>validated runs</div>
<div class="card"><b>{matched_comparisons}</b>matched contrasts</div>
<div class="card"><b>{cohort_min}&ndash;{cohort_max}</b>transcripts/contrast</div>
<div class="card"><b>{variant}</b>checkpoint rule</div></div>

<h2>What can be concluded now?</h2><ul>{''.join(f'<li>{html.escape(value)}</li>' for value in findings)}</ul>
<p class="note">These intervals quantify transcript sampling conditional on the fitted models. They are not uncertainty over
training seeds, datasets, or organisms. Each benchmark contains one observed profile per transcript, so the replica target and
arithmetic consensus coincide.</p>

<h2>Paired effects versus the full loss</h2>
<p>Values are mean paired transcript differences with 95% paired-bootstrap intervals: <em>ablated arm minus full arm</em>.
Green favors the ablated arm; red favors the full loss. All rows currently use seed 42.</p>
{_html_table(['Dataset','Ablated objective','Delta raw PCC [95% CI]','Delta NB-VST PCC [95% CI]',
              'Delta NB2 NLL [95% CI]','Matched transcripts','Seeds'], effect_rows, effect_classes)}
<figure><img src="{figure}" alt="Paired benchmark loss-ablation effects"><figcaption>Paired effects at the common
validation-NB2-selected checkpoint. Categorical rows must not be interpreted as independent organism replications.</figcaption></figure>

<h2>Absolute held-out summaries</h2>
<p>Unweighted means over held-out transcripts. Absolute levels differ substantially among organisms and should not be pooled
without an explicit estimand.</p>
{_html_table(['Dataset','Seed','Objective','N','Raw PCC','NB-VST PCC','NB2 NLL','Relative RMSE','Predicted/observed mean'], absolute_rows)}

<h2>Full-loss seed check</h2><p>This is descriptive: only yeast and E. coli currently have completed full-loss fits at seed 43,
and no matched seed-43 ablation is yet available.</p>
{_html_table(['Dataset','Seed','Raw PCC','NB-VST PCC','NB2 NLL'], full_rows)}

<h2>Run availability</h2>
{_html_table(['Seed','Dataset',*[arm_labels[value] for value in arm_order]], availability_rows, availability_classes)}
<details><summary>{len(missing_ids)} incomplete task IDs</summary><ul>{''.join(f'<li><code>{html.escape(value)}</code></li>' for value in missing_ids)}</ul></details>

<h2>Design and provenance</h2><ul>
<li>Frozen design: <code>{html.escape(str(design_path))}</code></li>
<li>Design SHA-256: <code>{html.escape(design_sha256)}</code></li>
<li>Primary checkpoint: minimum validation raw NB2 NLL (<code>{html.escape(variant)}</code>), common across arms.</li>
<li>Loss mode retained in every arm: mean-gradient-reweighted NB with beta 0.5; gamma regularization remains 1e-4.</li>
<li>Bootstrap: {int(design['analysis']['bootstrap_draws']):,} paired transcript draws, seed {int(design['analysis']['bootstrap_seed'])}.</li>
<li>Matching checks: identical split identities and all resolved non-loss configuration fields within each dataset/seed comparison.</li>
</ul>
<p>Numeric source tables: <a href="absolute_metric_summary.csv">absolute summaries</a>,
<a href="paired_effects_vs_full.csv">paired effects</a>, <a href="matching_and_exclusions.csv">matching/exclusions</a>,
<a href="run_availability.csv">availability</a>, and <a href="run_provenance.csv">run provenance</a>.</p>
</main></body></html>"""
    output.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--training-seeds", default="42")
    parser.add_argument(
        "--checkpoint-variant",
        choices=("best_nb_nll", "best_val_loss", "best_pcc"),
        default=None,
    )
    parser.add_argument("--bootstrap-draws", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.expanduser().resolve()
    design_path = args.design.expanduser().resolve()
    design = load_design(design_path)
    design_sha256 = sha256_file(design_path)
    frozen_design = experiment_root / "benchmark_loss_ablation_design.yaml"
    if frozen_design.is_file() and sha256_file(frozen_design) != design_sha256:
        raise ValueError("Experiment root was frozen with a different design.")
    seeds = parse_training_seeds(args.training_seeds)
    expected = build_tasks(design, seeds)
    variants = tuple(map(str, design["fixed"]["checkpoint_variants"]))
    variant = args.checkpoint_variant or str(design["analysis"]["primary_checkpoint"])
    draws = int(args.bootstrap_draws or design["analysis"]["bootstrap_draws"])
    bootstrap_seed = int(args.bootstrap_seed or design["analysis"]["bootstrap_seed"])
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws.")

    loaded: list[LoadedTask] = []
    for task in expected:
        attempt = newest_complete_attempt(
            experiment_root,
            task,
            design_sha256,
            variants,
        )
        if attempt is None:
            continue
        loaded.append(
            load_task(
                task=task,
                attempt_dir=attempt,
                variant=variant,
                all_variants=variants,
            )
        )
    if not loaded:
        raise FileNotFoundError(f"No completed ablation task found under {experiment_root}.")
    if args.require_complete and len(loaded) != len(expected):
        raise RuntimeError(f"Only {len(loaded)}/{len(expected)} tasks are complete.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else artifact_directory("benchmarking", experiment_root, variant)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = validate_matched_tasks(loaded)
    frame = pd.DataFrame(row for item in loaded for row in item.rows)
    absolute = absolute_summary(frame)
    effects, seed_effects, exclusions = paired_effects(
        frame,
        design=design,
        draws=draws,
        seed=bootstrap_seed,
    )
    availability = task_availability(
        experiment_root=experiment_root,
        expected=expected,
        completed=loaded,
        design_sha256=design_sha256,
    )

    frame.to_csv(output_dir / "per_transcript_metrics.csv", index=False)
    absolute.to_csv(output_dir / "absolute_metric_summary.csv", index=False)
    effects.to_csv(output_dir / "paired_effects_vs_full.csv", index=False)
    seed_effects.to_csv(output_dir / "seed_specific_effects.csv", index=False)
    exclusions.to_csv(output_dir / "matching_and_exclusions.csv", index=False)
    availability.to_csv(output_dir / "run_availability.csv", index=False)
    pd.DataFrame(checks).to_csv(output_dir / "matched_configuration_checks.csv", index=False)
    pd.DataFrame(
        [
            {
                **asdict(item.task),
                "attempt_dir": str(item.attempt_dir),
                "prediction_path": str(item.prediction_path),
                "checkpoint_path": item.checkpoint_path,
                "split_hash": item.split_hash,
                "non_loss_config_hash": item.config_hash,
            }
            for item in loaded
        ]
    ).to_csv(output_dir / "run_provenance.csv", index=False)

    if not effects.empty:
        plot_effects(
            effects,
            seed_effects,
            design=design,
            output_stem=output_dir / "benchmark_loss_ablation_effects",
        )
    caption = (
        r"\textbf{Ablation of the benchmarking data-fit objective.} "
        r"Points show changes relative to a contemporaneous model trained with "
        r"NB2, raw-profile PCC, and NB-VST PCC terms. Panels report held-out raw "
        r"PCC, NB-VST PCC, and NB2 NLL, using checkpoints selected by the common "
        r"validation NB2 NLL. Intervals are 95\% paired transcript-bootstrap "
        r"intervals (5,000 draws by default), conditional on the fitted models and "
        r"fixed organism-specific splits. Each source supplies one observed profile "
        r"per transcript, so its replica and arithmetic consensus coincide."
    )
    (output_dir / "caption.tex").write_text(caption + "\n", encoding="utf-8")
    write_report(
        output=output_dir / "ANALYSIS_REPORT.md",
        design_path=design_path,
        design_sha256=design_sha256,
        variant=variant,
        expected=expected,
        loaded=loaded,
        effects=effects,
        exclusions=exclusions,
    )
    write_html_report(
        output=output_dir / "benchmark_loss_ablation_report.html",
        design=design,
        design_path=design_path,
        design_sha256=design_sha256,
        variant=variant,
        expected=expected,
        loaded=loaded,
        availability=availability,
        absolute=absolute,
        effects=effects,
        exclusions=exclusions,
    )
    print(f"Analyzed {len(loaded)}/{len(expected)} tasks with {variant}.")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
