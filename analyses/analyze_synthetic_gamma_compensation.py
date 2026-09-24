#!/usr/bin/env python3
"""Diagnose whether missed synthetic gamma effects leak into L_bio or NB2 alpha.

This analysis is intentionally stricter than the older synthetic recovery
scripts: it accepts *only* ``predictions_main_val_best_val_loss_*.parquet``
artifacts (preferably recorded in ``prediction_checkpoint_manifest.json``).
It never falls back to an unsuffixed or best-PCC prediction parquet.

The identifiable gamma comparison is delegated to
``analyze_synthetic_gamma_recovery.joint_log_gamma_gauge`` and
``_load_bias_profiles``.  Consequently the programmed multiplier and learned
log-gamma receive exactly the same two-way, reference-weighted log gauge.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyses.analyze_synthetic_gamma_recovery import (  # exact existing gauge
    _base_bias_name,
    _load_bias_profiles,
    _load_learned_gamma,
    joint_log_gamma_gauge,
)
from analyses.analyze_synthetic_recovery import (
    BOUNDARY_TRIM_CODONS,
    DEFAULT_OUTPUT_DIRECTORY_NAME,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_RUN_PREFIX,
    REPOSITORY_ROOT,
    _encoding_from_config,
    _load_latent_truth,
    _read_yaml,
    _validation_manifest,
    cds_interior_mask,
    discover_run_directories,
    evaluation_domain_metadata,
    synthetic_depth_label,
    synthetic_mass_conservation,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


DEFAULT_BIAS_ROOT = REPOSITORY_ROOT / "Datasets" / "Synthetic_data" / "bias_profile"
DEFAULT_LATENT_TRUTH = (
    REPOSITORY_ROOT
    / "Datasets"
    / "Synthetic_data"
    / "artificial_ground_truth_kinetics_target_mean_one.parquet"
)
DEFAULT_OUTPUT = (
    DEFAULT_RESULTS_ROOT / DEFAULT_OUTPUT_DIRECTORY_NAME / "gamma_compensation_analysis"
)
EPS = 1.0e-8
SYNTHETIC_TRUE_ALPHA = 0.1
ALPHA_CALIBRATION_COLUMNS = (
    "run",
    "depth",
    "mass_condition",
    "dataset_count",
    "validation_id_hash",
    "prediction_path",
    "alpha_learning_rate_scale_resolved",
    "alpha_true",
    "alpha_validation_pairs",
    "alpha_validation_positions",
    "alpha_mean",
    "alpha_median",
    "alpha_std",
    "alpha_q05",
    "alpha_q25",
    "alpha_q75",
    "alpha_q95",
    "alpha_mae_from_true",
    "alpha_rmse_from_true",
    "alpha_log_bias",
    "alpha_log_rmse_from_true",
    "alpha_fraction_within_1p25fold",
    "alpha_fraction_within_1p5fold",
    "alpha_fraction_within_2fold",
    "alpha_pair_equal_mae_from_true",
    "alpha_pair_equal_log_rmse_from_true",
    "boundary_trim_codons",
    "evaluation_domain",
    "boundary_positions_excluded",
    "alpha_validation_positions_full_previous_definition",
    "alpha_mean_full_previous_definition",
    "alpha_median_full_previous_definition",
    "alpha_mae_from_true_full_previous_definition",
    "alpha_log_rmse_from_true_full_previous_definition",
)
STRONG_TRUE_LOG_GAMMA = 1.0
MISSED_LEARNED_LOG_GAMMA = 0.5
CORRECT_ABS_LOG_GAMMA_ERROR = 0.25
L_INFLATION_LOG_THRESHOLD = 0.20
ALPHA_EXCESS_LOG_THRESHOLD = 0.25
CANONICAL_BIASES = (
    "artificial_bias_3prime_aa",
    "artificial_bias_3prime_cc",
    "artificial_bias_3prime_gg",
    "artificial_bias_3prime_uu",
    "artificial_bias_5prime_aa",
    "artificial_bias_5prime_cc",
    "artificial_bias_5prime_gg",
    "artificial_bias_5prime_uu",
    "artificial_bias_au_fraction_gt_0p7",
    "artificial_bias_gc_fraction_gt_0p7",
)


def find_best_val_loss_prediction(run_dir: Path) -> tuple[Path | None, str]:
    """Return only a verified minimum-validation-loss prediction artifact.

    The return reason is saved in the skipped-run table, making it clear that a
    legacy unsuffixed prediction was deliberately not used.
    """
    manifests = sorted(run_dir.rglob("prediction_checkpoint_manifest.json"))
    if len(manifests) > 1:
        return None, f"found {len(manifests)} prediction checkpoint manifests"
    if manifests:
        try:
            payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            record = payload.get("best_val_loss")
            if not isinstance(record, dict):
                return None, "manifest has no best_val_loss entry"
            prediction = Path(str(record.get("output_path", ""))).expanduser()
            if not prediction.is_absolute():
                prediction = (manifests[0].parent / prediction).resolve()
            checkpoint = Path(str(record.get("checkpoint_path", ""))).name
            # Historical Lightning formatting is ``epoch=...-val_loss=...``;
            # current code may use another non-PCC filename, but all valid
            # minimum-loss checkpoints contain val_loss and never start pcc-.
            if checkpoint.startswith("pcc-") or "val_loss" not in checkpoint:
                return None, "best_val_loss manifest entry does not reference val-loss checkpoint"
            if "predictions_main_val_best_val_loss_" not in prediction.name:
                return None, "best_val_loss manifest output has an unexpected filename"
            if prediction.is_file():
                return prediction, "manifest"

            # Result folders are commonly copied from another host. Absolute
            # paths persisted in the manifest then keep their remote prefix
            # even though the corresponding artifact exists inside the local
            # run directory. Preserve the manifest's checkpoint/variant proof,
            # and relocate only to an exact same-basename local artifact.
            local_matches = sorted(
                path
                for path in run_dir.rglob(
                    "predictions_main_val_best_val_loss_*.parquet"
                )
                if path.name == prediction.name
            )
            if len(local_matches) == 1:
                return local_matches[0], "manifest (relocated local artifact)"
            if len(local_matches) > 1:
                return None, (
                    "manifest path is unavailable and found multiple local "
                    f"artifacts named {prediction.name}"
                )
            return None, f"best_val_loss prediction is missing: {prediction}"
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return None, f"invalid prediction checkpoint manifest: {exc}"

    explicit = sorted(run_dir.rglob("predictions_main_val_best_val_loss_*.parquet"))
    if len(explicit) == 1:
        return explicit[0], "explicit filename (manifest unavailable)"
    if len(explicit) > 1:
        return None, f"found {len(explicit)} best_val_loss prediction parquets"
    legacy = list(run_dir.rglob("predictions_main_val_*.parquet"))
    if legacy:
        return None, "only legacy/PCC-unspecified prediction parquet(s) available"
    return None, "no best_val_loss prediction parquet"


def _historical_config(run_dir: Path) -> Path:
    """Prefer the original direct-run config over a replay's version_N copy."""
    paths = sorted(run_dir.rglob("config.yaml"))
    direct = [path for path in paths if path.parent.name.startswith("direct_")]
    if len(direct) == 1:
        return direct[0]
    if len(paths) != 1:
        raise ValueError(f"expected one historical config.yaml, found {len(paths)}")
    return paths[0]


def _as_profile(value: Any, length: int, *, label: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64).reshape(-1)[:length]
    if values.size != length or not np.isfinite(values).all():
        raise ValueError(f"{label} is shorter than expected or non-finite.")
    return values


def _mean_one_positive(value: Any, length: int, *, label: str) -> np.ndarray:
    values = _as_profile(value, length, label=label)
    if bool((values <= 0.0).any()):
        raise ValueError(f"{label} contains a non-positive value.")
    return values / values.mean()


def _prediction_profiles(
    prediction_path: Path,
    *,
    expected_lengths: dict[str, int],
    dataset_id_to_name: dict[int, str],
    log_alpha_min: float,
    log_alpha_max: float,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], dict[str, np.ndarray]], float]:
    """Load L, alpha, mu and target; gamma itself uses the existing loader."""
    required = ("transcript_id", "dataset_id", "L_bio", "log_sigma", "mu", "target")
    file = pq.ParquetFile(prediction_path)
    missing = sorted(set(required) - set(file.schema_arrow.names))
    if missing:
        raise ValueError(f"{prediction_path} is missing columns: {missing}")
    l_by_transcript: dict[str, np.ndarray] = {}
    pairs: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    max_l_duplicate_difference = 0.0
    for batch in file.iter_batches(batch_size=32, columns=list(required)):
        columns = batch.to_pydict()
        for index in range(batch.num_rows):
            transcript_id = str(columns["transcript_id"][index])
            if transcript_id not in expected_lengths:
                continue
            dataset_id = int(columns["dataset_id"][index])
            dataset = dataset_id_to_name.get(dataset_id)
            if dataset is None:
                raise KeyError(f"No dataset name for prediction dataset ID {dataset_id}.")
            key = (transcript_id, dataset)
            if key in pairs:
                raise ValueError(f"Duplicate prediction pair {key}.")
            length = expected_lengths[transcript_id]
            l_bio = _mean_one_positive(columns["L_bio"][index], length, label=f"L_bio {key}")
            prior = l_by_transcript.get(transcript_id)
            if prior is None:
                l_by_transcript[transcript_id] = l_bio
            else:
                max_l_duplicate_difference = max(
                    max_l_duplicate_difference, float(np.max(np.abs(prior - l_bio)))
                )
            log_alpha = np.clip(
                _as_profile(columns["log_sigma"][index], length, label=f"log_sigma {key}"),
                log_alpha_min,
                log_alpha_max,
            )
            pairs[key] = {
                "alpha": np.exp(log_alpha),
                "log_alpha": log_alpha,
                "mu": _as_profile(columns["mu"][index], length, label=f"mu {key}"),
                "target": _as_profile(columns["target"][index], length, label=f"target {key}"),
            }
    if max_l_duplicate_difference > 1.0e-5:
        raise AssertionError(
            "L_bio differs across dataset rows for the same transcript; "
            f"maximum difference={max_l_duplicate_difference:.3e}."
        )
    return l_by_transcript, pairs, max_l_duplicate_difference


def alpha_calibration_metrics(
    prediction_pairs: dict[tuple[str, str], dict[str, np.ndarray]],
    *,
    true_alpha: float = SYNTHETIC_TRUE_ALPHA,
    interior_only: bool = True,
) -> dict[str, float | int]:
    """Summarize alpha against the known NB2 simulation value.

    The primary definition uses the same CDS-interior coordinates as gamma
    recovery. Passing ``interior_only=False`` reproduces the historical
    all-modeled-codon diagnostic. Both are independent of neutral-site
    subsampling used later by the compensation analysis.
    """
    if not math.isfinite(true_alpha) or true_alpha <= 0.0:
        raise ValueError("true_alpha must be finite and positive.")
    arrays = []
    for pair in prediction_pairs.values():
        array = np.asarray(pair["alpha"], dtype=np.float64).reshape(-1)
        if interior_only:
            array = array[cds_interior_mask(array.size)]
        if array.size:
            arrays.append(array)
    if not arrays:
        raise ValueError("No prediction pairs are available for alpha calibration.")
    if any(array.size == 0 or not np.isfinite(array).all() or bool((array <= 0).any()) for array in arrays):
        raise ValueError("Predicted alpha profiles must be finite, positive, and non-empty.")

    alpha = np.concatenate(arrays)
    log_error = np.log(alpha) - math.log(true_alpha)
    relative_factor = np.maximum(alpha / true_alpha, true_alpha / alpha)
    pair_mae = np.asarray(
        [np.mean(np.abs(array - true_alpha)) for array in arrays], dtype=np.float64
    )
    pair_log_rmse = np.asarray(
        [np.sqrt(np.mean((np.log(array) - math.log(true_alpha)) ** 2)) for array in arrays],
        dtype=np.float64,
    )
    return {
        "alpha_true": float(true_alpha),
        "alpha_validation_pairs": len(arrays),
        "alpha_validation_positions": int(alpha.size),
        "alpha_mean": float(alpha.mean()),
        "alpha_median": float(np.median(alpha)),
        "alpha_std": float(alpha.std()),
        "alpha_q05": float(np.quantile(alpha, 0.05)),
        "alpha_q25": float(np.quantile(alpha, 0.25)),
        "alpha_q75": float(np.quantile(alpha, 0.75)),
        "alpha_q95": float(np.quantile(alpha, 0.95)),
        "alpha_mae_from_true": float(np.mean(np.abs(alpha - true_alpha))),
        "alpha_rmse_from_true": float(np.sqrt(np.mean((alpha - true_alpha) ** 2))),
        "alpha_log_bias": float(log_error.mean()),
        "alpha_log_rmse_from_true": float(np.sqrt(np.mean(log_error**2))),
        "alpha_fraction_within_1p25fold": float(np.mean(relative_factor <= 1.25)),
        "alpha_fraction_within_1p5fold": float(np.mean(relative_factor <= 1.5)),
        "alpha_fraction_within_2fold": float(np.mean(relative_factor <= 2.0)),
        "alpha_pair_equal_mae_from_true": float(pair_mae.mean()),
        "alpha_pair_equal_log_rmse_from_true": float(pair_log_rmse.mean()),
    }


def _load_observations(path: Path, ids: set[str], expected_lengths: dict[str, int]) -> dict[str, dict[str, Any]]:
    """Read consensus and raw replicate counts without changing training data."""
    file = pq.ParquetFile(path)
    columns = [name for name in ("id", "ribo", "ribo_cds_replicas", "replica_ids") if name in file.schema_arrow.names]
    if not {"id", "ribo"}.issubset(columns):
        raise ValueError(f"{path} has no id/ribo observation columns.")
    result: dict[str, dict[str, Any]] = {}
    for batch in file.iter_batches(batch_size=512, columns=columns):
        values = batch.to_pydict()
        for index, raw_id in enumerate(values["id"]):
            transcript_id = str(raw_id)
            if transcript_id not in ids:
                continue
            if transcript_id in result:
                raise ValueError(f"Duplicate observation {path.name}/{transcript_id}.")
            length = expected_lengths[transcript_id]
            consensus = _as_profile(values["ribo"][index], length, label=f"consensus {path.name}/{transcript_id}")
            raw_replicas = values.get("ribo_cds_replicas", [None] * batch.num_rows)[index]
            replicas = []
            if raw_replicas is not None:
                for replica in raw_replicas:
                    profile = np.asarray(replica, dtype=np.float64).reshape(-1)[:length]
                    if profile.size == length and np.isfinite(profile).all():
                        replicas.append(profile)
            raw_replica_ids = values.get("replica_ids", [None] * batch.num_rows)[index]
            result[transcript_id] = {
                "consensus": consensus,
                "replicas": np.stack(replicas, axis=0) if replicas else None,
                "replica_ids": [] if raw_replica_ids is None else list(raw_replica_ids),
            }
    return result


def _corr(x: Iterable[float], y: Iterable[float]) -> float:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    if int(valid.sum()) < 3:
        return float("nan")
    x_array, y_array = x_array[valid], y_array[valid]
    if float(x_array.std()) == 0.0 or float(y_array.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def _slope(y: Iterable[float], x: Iterable[float]) -> float:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    if int(valid.sum()) < 3:
        return float("nan")
    x_array, y_array = x_array[valid], y_array[valid]
    denominator = float(np.sum((x_array - x_array.mean()) ** 2))
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum((x_array - x_array.mean()) * (y_array - y_array.mean())) / denominator)


def _partial_corr(frame: pd.DataFrame, x: str, y: str, controls: list[str]) -> float:
    values = frame[[x, y, *controls]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < len(controls) + 4:
        return float("nan")
    design = np.column_stack([np.ones(len(values)), values[controls].to_numpy(dtype=np.float64)])
    x_residual = values[x].to_numpy(dtype=np.float64) - design @ np.linalg.lstsq(design, values[x], rcond=None)[0]
    y_residual = values[y].to_numpy(dtype=np.float64) - design @ np.linalg.lstsq(design, values[y], rcond=None)[0]
    return _corr(x_residual, y_residual)


def classify_sites(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach explicit miss, recovery, and conservative joint-compensation labels."""
    frame = frame.copy()
    frame["gamma_error"] = frame["g_learned"] - frame["g_true"]
    frame["gamma_underestimate"] = -frame["gamma_error"]
    frame["gamma_abs_error"] = frame["gamma_error"].abs()
    frame["L_log_error"] = np.log(np.maximum(frame["L_learned"], EPS)) - np.log(np.maximum(frame["L_true"], EPS))
    frame["log_alpha"] = np.log(np.maximum(frame["alpha"], EPS))
    frame["is_strong"] = frame["g_true"] >= STRONG_TRUE_LOG_GAMMA
    frame["is_missed_strong"] = frame["is_strong"] & (frame["g_learned"] < MISSED_LEARNED_LOG_GAMMA)
    frame["is_correct_strong"] = frame["is_strong"] & (frame["gamma_error"].abs() <= CORRECT_ABS_LOG_GAMMA_ERROR)
    frame["site_group"] = np.select(
        [
            frame["is_missed_strong"],
            frame["is_correct_strong"],
            frame["programmed_bias"] & ~frame["is_strong"],
            ~frame["programmed_bias"],
        ],
        ["missed_strong", "correct_strong", "programmed_weaker", "neutral"],
        default="programmed_other",
    )
    frame["true_amplitude_bin"] = pd.cut(
        frame["g_true"], [-np.inf, 0.0, 0.5, 1.0, 1.25, 1.5, 2.0, np.inf], include_lowest=True
    ).astype(str)
    frame["count_bin"] = pd.cut(
        np.log1p(np.maximum(frame["consensus"], 0.0)), [-np.inf, 0.0, 0.7, 1.4, 2.3, 3.5, np.inf], include_lowest=True
    ).astype(str)
    # Alpha reference is a non-missed median within true amplitude and observed
    # count strata. It does not make a causal claim; it only prevents raw count
    # magnitude from masquerading as an alpha effect.
    strata = ["run", "dataset", "true_amplitude_bin", "count_bin"]
    non_missed = frame.loc[~frame["is_missed_strong"]].groupby(strata, observed=True)["log_alpha"].median().rename("_alpha_reference")
    frame = frame.join(non_missed, on=strata)
    fallback = frame.loc[~frame["is_missed_strong"]].groupby(["run", "dataset", "true_amplitude_bin"], observed=True)["log_alpha"].median().rename("_alpha_fallback")
    frame = frame.join(fallback, on=["run", "dataset", "true_amplitude_bin"])
    frame["alpha_reference_log"] = frame["_alpha_reference"].fillna(frame["_alpha_fallback"])
    frame["alpha_log_excess"] = frame["log_alpha"] - frame["alpha_reference_log"]
    frame["L_inflated"] = frame["L_log_error"] >= L_INFLATION_LOG_THRESHOLD
    frame["alpha_elevated"] = frame["alpha_log_excess"] >= ALPHA_EXCESS_LOG_THRESHOLD
    frame["joint_compensation"] = np.select(
        [frame["L_inflated"] & frame["alpha_elevated"], frame["L_inflated"], frame["alpha_elevated"]],
        ["both", "L_inflation", "alpha_elevation"],
        default="neither",
    )
    return frame.drop(columns=["_alpha_reference", "_alpha_fallback"])


def _stratified_miss_difference(frame: pd.DataFrame, column: str) -> tuple[float, int]:
    strong = frame[frame["site_group"].isin(["missed_strong", "correct_strong"])].copy()
    values: list[tuple[float, int]] = []
    for _, part in strong.groupby(["dataset", "true_amplitude_bin", "count_bin"], observed=True):
        missed = part.loc[part["site_group"] == "missed_strong", column].dropna()
        correct = part.loc[part["site_group"] == "correct_strong", column].dropna()
        if len(missed) and len(correct):
            values.append((float(missed.mean() - correct.mean()), min(len(missed), len(correct))))
    if not values:
        return float("nan"), 0
    differences, weights = zip(*values, strict=True)
    return float(np.average(differences, weights=weights)), int(sum(weights))


def summarize_bias(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if "is_interior" in frame.columns:
        frame = frame.loc[frame["is_interior"]].copy()
    rows: list[dict[str, Any]] = []
    for key, part in frame.groupby(group_columns, observed=True, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key_values, strict=True))
        programmed = part[part["programmed_bias"]]
        strong = part[part["is_strong"]]
        missed = part[part["site_group"] == "missed_strong"]
        correct = part[part["site_group"] == "correct_strong"]
        alpha_diff, alpha_strata = _stratified_miss_difference(part, "log_alpha")
        replicate_diff, replicate_strata = _stratified_miss_difference(part, "replicate_cv")
        controls = ["g_true", "log1p_consensus", "replicate_cv"]
        row.update({
            "positions": len(part), "programmed_positions": len(programmed), "strong_positions": len(strong),
            "missed_strong_positions": len(missed), "correct_strong_positions": len(correct),
            "strong_site_miss_rate": float(len(missed) / len(strong)) if len(strong) else float("nan"),
            "interior_strong_site_miss_rate": float(len(missed) / len(strong)) if len(strong) else float("nan"),
            **evaluation_domain_metadata(),
            "mean_abs_gamma_error_programmed": float(programmed["gamma_abs_error"].mean()),
            "mean_abs_gamma_error_strong": float(strong["gamma_abs_error"].mean()),
            "gamma_L_error_correlation_programmed": _corr(programmed["gamma_underestimate"], programmed["L_log_error"]),
            "gamma_L_error_slope_programmed": _slope(programmed["L_log_error"], programmed["gamma_underestimate"]),
            "mean_L_log_error_missed": float(missed["L_log_error"].mean()),
            "mean_L_log_error_correct": float(correct["L_log_error"].mean()),
            "L_log_error_missed_minus_correct": float(missed["L_log_error"].mean() - correct["L_log_error"].mean()),
            "mean_log_alpha_missed": float(missed["log_alpha"].mean()),
            "mean_log_alpha_correct": float(correct["log_alpha"].mean()),
            "alpha_missed_minus_correct_stratified": alpha_diff,
            "alpha_stratified_matched_support": alpha_strata,
            "replicate_cv_missed_minus_correct_stratified": replicate_diff,
            "replicate_cv_stratified_matched_support": replicate_strata,
            "abs_gamma_error_log_alpha_correlation": _corr(programmed["gamma_abs_error"], programmed["log_alpha"]),
            "gamma_underestimate_log_alpha_correlation": _corr(programmed["gamma_underestimate"], programmed["log_alpha"]),
            "abs_gamma_error_log_alpha_partial_correlation": _partial_corr(programmed, "gamma_abs_error", "log_alpha", controls),
            "gamma_underestimate_log_alpha_partial_correlation": _partial_corr(programmed, "gamma_underestimate", "log_alpha", controls),
            "mean_replicate_cv_missed": float(missed["replicate_cv"].mean()),
            "mean_replicate_cv_correct": float(correct["replicate_cv"].mean()),
        })
        leak = (
            len(missed) >= 20
            and row["mean_L_log_error_missed"] > 0.0
            and row["gamma_L_error_correlation_programmed"] >= 0.20
            and row["gamma_L_error_slope_programmed"] > 0.0
        )
        alpha = (
            len(missed) >= 20
            and np.isfinite(alpha_diff) and alpha_diff >= ALPHA_EXCESS_LOG_THRESHOLD
            and (not np.isfinite(replicate_diff) or replicate_diff <= 0.05)
        )
        row["conservative_interpretation"] = (
            "gamma_to_L_leakage_supported" if leak and not alpha else
            "joint_L_and_alpha_compensation_supported" if leak and alpha else
            "alpha_absorption_supported" if alpha else
            "gamma_miss_unexplained_or_insufficient_evidence"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_boundary_diagnostics(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Report boundary failures without mixing them into fair recovery scores."""
    rows: list[dict[str, Any]] = []
    for key, part in frame.groupby(group_columns, observed=True, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key_values, strict=True))
        interior = part.loc[part["is_interior"]]
        boundary = part.loc[~part["is_interior"]]
        interior_strong = interior.loc[interior["is_strong"]]
        boundary_strong = boundary.loc[boundary["is_strong"]]
        interior_missed = interior_strong.loc[interior_strong["is_missed_strong"]]
        boundary_missed = boundary_strong.loc[boundary_strong["is_missed_strong"]]
        all_misses = len(interior_missed) + len(boundary_missed)
        row.update(
            {
                "interior_strong_site_miss_rate": (
                    len(interior_missed) / len(interior_strong)
                    if len(interior_strong)
                    else float("nan")
                ),
                "boundary_strong_site_miss_rate": (
                    len(boundary_missed) / len(boundary_strong)
                    if len(boundary_strong)
                    else float("nan")
                ),
                "n_interior_strong_sites": len(interior_strong),
                "n_boundary_strong_sites": len(boundary_strong),
                "n_interior_missed_strong_sites": len(interior_missed),
                "n_boundary_missed_strong_sites": len(boundary_missed),
                "fraction_of_all_misses_at_boundary": (
                    len(boundary_missed) / all_misses
                    if all_misses
                    else float("nan")
                ),
                "boundary_evaluation_scope": (
                    "out_of_scope_for_fair_cds_only_recovery"
                ),
                "boundary_gauge_definition": "full_cds_previous_definition",
                **evaluation_domain_metadata(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _choose_primary_runs(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Choose largest selected panel per depth/mass condition for figures."""
    if run_summary.empty:
        return run_summary
    ordered = run_summary.sort_values(
        ["depth", "mass_condition", "dataset_count", "run"],
        ascending=[True, True, False, True],
    )
    return ordered.drop_duplicates(["depth", "mass_condition"], keep="first").copy()


def _site_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose all requested comparison groups, not only missed/correct sites."""
    if "is_interior" in frame.columns:
        frame = frame.loc[frame["is_interior"]]
    return frame.groupby(
        ["run", "depth", "mass_condition", "bias_name", "site_group"], observed=True
    ).agg(
        positions=("position", "size"),
        mean_L_log_error=("L_log_error", "mean"),
        median_L_log_error=("L_log_error", "median"),
        mean_alpha=("alpha", "mean"), median_alpha=("alpha", "median"),
        mean_log_alpha=("log_alpha", "mean"), median_log_alpha=("log_alpha", "median"),
        mean_replicate_cv=("replicate_cv", "mean"), median_replicate_cv=("replicate_cv", "median"),
        mean_consensus=("consensus", "mean"), median_consensus=("consensus", "median"),
    ).reset_index()


def _amplitude_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Count-/amplitude-visible alpha table used to audit the matched result."""
    if "is_interior" in frame.columns:
        frame = frame.loc[frame["is_interior"]]
    return frame.groupby(
        ["run", "depth", "mass_condition", "bias_name", "dataset", "true_amplitude_bin", "site_group"], observed=True
    ).agg(
        positions=("position", "size"), mean_log_alpha=("log_alpha", "mean"),
        mean_alpha=("alpha", "mean"), mean_consensus=("consensus", "mean"),
        mean_replicate_cv=("replicate_cv", "mean"), mean_L_log_error=("L_log_error", "mean"),
    ).reset_index()


def _joint_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if "is_interior" in frame.columns:
        frame = frame.loc[frame["is_interior"]]
    return frame[frame["is_missed_strong"]].groupby(
        ["run", "depth", "mass_condition", "bias_name", "joint_compensation"], observed=True
    ).size().rename("missed_strong_sites").reset_index()


def _plot_bias(part: pd.DataFrame, output: Path, title: str) -> None:
    if "is_interior" in part.columns:
        part = part.loc[part["is_interior"]].copy()
    rng = np.random.default_rng(42)
    if len(part) > 25000:
        part = part.iloc[rng.choice(len(part), size=25000, replace=False)].copy()
    programmed = part[part["programmed_bias"]]
    missed = programmed[programmed["is_missed_strong"]]
    correct = programmed[programmed["is_correct_strong"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, y, ylabel in (
        (axes[0, 0], "L_log_error", r"log(L learned) − log(K true)"),
        (axes[0, 1], "log_alpha", r"log learned $\alpha$"),
    ):
        axis.scatter(programmed["gamma_underestimate"], programmed[y], s=4, alpha=.12, color="#2563eb", rasterized=True, label="programmed sites")
        axis.scatter(missed["gamma_underestimate"], missed[y], s=11, alpha=.55, color="#dc2626", rasterized=True, label="missed strong")
        axis.axvline(0, color="black", lw=.8); axis.axhline(0, color="black", lw=.8)
        axis.set_xlabel(r"gamma underestimation: $g^{true}-g^{learned}$")
        axis.set_ylabel(ylabel); axis.grid(alpha=.2); axis.legend(frameon=False, fontsize=8)
    axes[0, 0].set_title("Gamma error versus shared-profile error")
    axes[0, 1].set_title("Gamma error versus NB2 dispersion")
    for axis, column, ylabel in (
        (axes[1, 0], "L_log_error", r"log(L learned) − log(K true)"),
        (axes[1, 1], "log_alpha", r"log learned $\alpha$"),
    ):
        arrays = [correct[column].dropna().to_numpy(), missed[column].dropna().to_numpy()]
        axis.boxplot(arrays, tick_labels=["correct strong", "missed strong"], showfliers=False)
        axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=.2)
    axes[1, 0].set_title("Shared-profile error at strong sites")
    axes[1, 1].set_title("Dispersion at strong sites")
    fig.suptitle(title, fontweight="bold")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(summary: pd.DataFrame, output: Path) -> None:
    frame = summary[summary["bias_name"].isin(CANONICAL_BIASES)].copy()
    frame["label"] = frame["bias_name"].str.replace("artificial_bias_", "", regex=False)
    frame = frame.sort_values("bias_name", key=lambda x: x.map({name: i for i, name in enumerate(CANONICAL_BIASES)}))
    metrics = (
        ("strong_site_miss_rate", "Strong-site miss rate"),
        ("mean_L_log_error_missed", "Mean L log-error at missed sites"),
        ("alpha_missed_minus_correct_stratified", "Stratified log-alpha elevation"),
        ("gamma_L_error_correlation_programmed", "Gamma-underestimate / L-error r"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(19, 7), sharey=True, constrained_layout=True)
    for axis, (column, label) in zip(axes, metrics, strict=True):
        axis.barh(frame["label"], frame[column], color="#2563eb")
        axis.axvline(0, color="black", lw=.7); axis.set_title(label); axis.grid(axis="x", alpha=.2)
    fig.suptitle("Synthetic gamma compensation diagnostics: primary best-val-loss panels", fontweight="bold")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_composition_comparison(summary: pd.DataFrame, output: Path) -> None:
    composition = summary[summary["bias_name"].isin(CANONICAL_BIASES[-2:])].copy()
    dinucleotide = summary[summary["bias_name"].isin(CANONICAL_BIASES[:8])]
    if composition.empty or dinucleotide.empty:
        return
    metrics = ["strong_site_miss_rate", "mean_L_log_error_missed", "alpha_missed_minus_correct_stratified"]
    baseline = dinucleotide[metrics].median(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    labels = composition["bias_name"].str.replace("artificial_bias_", "", regex=False).tolist()
    for axis, metric in zip(axes, metrics, strict=True):
        axis.bar(np.arange(len(composition)), composition[metric], color=["#7c3aed", "#db2777"][:len(composition)])
        axis.axhline(float(baseline[metric]), color="#374151", ls="--", label="dinucleotide median")
        axis.set_xticks(np.arange(len(composition)), labels, rotation=25, ha="right")
        axis.set_title(metric.replace("_", " ")); axis.grid(axis="y", alpha=.2); axis.legend(frameon=False, fontsize=8)
    fig.suptitle("AU/GC composition biases versus eight dinucleotide biases", fontweight="bold")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


MASS_COMPARISON_METRICS: tuple[tuple[str, str], ...] = (
    ("strong_site_miss_rate", "strong-site miss rate\n(lower is better)"),
    ("mean_abs_gamma_error_programmed", "mean |gamma error|\n(lower is better)"),
    ("mean_L_log_error_missed", "mean L log-error\nat missed sites"),
    ("alpha_missed_minus_correct_stratified", "matched log-alpha\nelevation"),
    ("gamma_L_error_correlation_programmed", "gamma/L error\ncorrelation"),
)


def _matched_mass_pairs(primary_runs: pd.DataFrame) -> pd.DataFrame:
    """Return only fair mass-mode pairings, never merely same-depth runs.

    A comparison requires the same depth, exact selected dataset list, dataset
    count, and validation transcript manifest.  This deliberately rejects a
    mass-free panel with fewer datasets than its mass-conserved counterpart.
    """
    columns = ["run", "depth", "mass_condition", "dataset_count", "datasets", "validation_id_hash"]
    available = primary_runs.reindex(columns=columns).dropna(subset=["run", "mass_condition"]).copy()
    keys = ["depth", "dataset_count", "datasets", "validation_id_hash"]
    conserved = available.loc[available["mass_condition"] == "mass_conserved", keys + ["run"]].rename(
        columns={"run": "mass_conserved_run"}
    )
    free = available.loc[available["mass_condition"] == "mass_free", keys + ["run"]].rename(
        columns={"run": "mass_free_run"}
    )
    return conserved.merge(free, on=keys, how="inner", validate="one_to_one")


def _paired_mass_summary(condition_summary: pd.DataFrame, matched_runs: pd.DataFrame) -> pd.DataFrame:
    """Calculate mass-free minus mass-conserved diagnostics for matched panels."""
    if matched_runs.empty or condition_summary.empty:
        return pd.DataFrame()
    metrics = [name for name, _ in MASS_COMPARISON_METRICS]
    conserved = condition_summary.loc[condition_summary["mass_condition"] == "mass_conserved", ["run", "bias_name", *metrics]].rename(
        columns={"run": "mass_conserved_run", **{name: f"{name}_mass_conserved" for name in metrics}}
    )
    free = condition_summary.loc[condition_summary["mass_condition"] == "mass_free", ["run", "bias_name", *metrics]].rename(
        columns={"run": "mass_free_run", **{name: f"{name}_mass_free" for name in metrics}}
    )
    paired = matched_runs.merge(conserved, on="mass_conserved_run", how="inner", validate="one_to_many").merge(
        free, on=["mass_free_run", "bias_name"], how="inner", validate="one_to_one"
    )
    for name in metrics:
        paired[f"{name}_mass_free_minus_mass_conserved"] = (
            paired[f"{name}_mass_free"] - paired[f"{name}_mass_conserved"]
        )
    return paired


def _plot_condition_overview(summary: pd.DataFrame, output: Path, title: str) -> None:
    """Keep one mass condition per plot; no mass-mode pooling is allowed."""
    metrics = MASS_COMPARISON_METRICS[:4]
    order = {name: index for index, name in enumerate(CANONICAL_BIASES)}
    frame = summary.copy()
    frame["label"] = frame["bias_name"].str.replace("artificial_bias_", "", regex=False)
    frame = frame.sort_values("bias_name", key=lambda value: value.map(order))
    fig, axes = plt.subplots(1, len(metrics), figsize=(19, 6.5), sharey=True, constrained_layout=True)
    for axis, (column, label) in zip(axes, metrics, strict=True):
        axis.barh(frame["label"], frame[column], color="#2563eb")
        axis.axvline(0, color="black", lw=.7)
        axis.set_title(label)
        axis.grid(axis="x", alpha=.2)
    fig.suptitle(title, fontweight="bold")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_paired_mass_delta(pair: pd.DataFrame, output: Path) -> None:
    """Heatmap of fair mass-free minus mass-conserved differences by bias."""
    if pair.empty:
        return
    order = {name: index for index, name in enumerate(CANONICAL_BIASES)}
    pair = pair.sort_values("bias_name", key=lambda value: value.map(order)).copy()
    labels = pair["bias_name"].str.replace("artificial_bias_", "", regex=False).tolist()
    fig, axes = plt.subplots(1, len(MASS_COMPARISON_METRICS), figsize=(20, 7), sharey=True, constrained_layout=True)
    for metric_index, (axis, (metric, label)) in enumerate(zip(axes, MASS_COMPARISON_METRICS, strict=True)):
        values = pair[f"{metric}_mass_free_minus_mass_conserved"].to_numpy(dtype=float)[:, None]
        finite = np.abs(values[np.isfinite(values)])
        limit = float(np.quantile(finite, .95)) if finite.size else 1.0
        limit = max(limit, 1.0e-6)
        image = axis.imshow(values, aspect="auto", cmap="coolwarm_r", vmin=-limit, vmax=limit)
        for row, value in enumerate(values[:, 0]):
            if np.isfinite(value):
                axis.text(0, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)
        axis.set_xticks([0], ["free − conserved"])
        axis.set_title(label)
        axis.figure.colorbar(image, ax=axis, fraction=.046, pad=.04)
        axis.set_yticks(np.arange(len(labels)))
        if metric_index == 0:
            axis.set_yticklabels(labels, fontsize=8)
            axis.tick_params(axis="y", labelleft=True)
        else:
            axis.tick_params(axis="y", labelleft=False)
    first = pair.iloc[0]
    fig.suptitle(
        f"Matched mass-mode comparison — {first.depth}, {first.dataset_count} datasets\n"
        "Negative values favour mass-free for the first two error metrics; L/alpha/correlation are mechanisms, not universal quality scores.",
        fontweight="bold",
    )
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_paired_mass_bias(conserved: pd.DataFrame, free: pd.DataFrame, output: Path, title: str) -> None:
    """Directly compare the gamma-error mechanisms without mixing mass modes."""
    if "is_interior" in conserved.columns:
        conserved = conserved.loc[conserved["is_interior"]].copy()
    if "is_interior" in free.columns:
        free = free.loc[free["is_interior"]].copy()
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for row, (label, frame) in enumerate((("mass conserved", conserved), ("mass free", free))):
        if len(frame) > 25000:
            frame = frame.iloc[rng.choice(len(frame), size=25000, replace=False)].copy()
        programmed = frame[frame["programmed_bias"]]
        missed = programmed[programmed["is_missed_strong"]]
        for column, ylabel, axis in (
            ("L_log_error", r"log(L learned) − log(K true)", axes[row, 0]),
            ("log_alpha", r"log learned $\alpha$", axes[row, 1]),
        ):
            axis.scatter(programmed["gamma_underestimate"], programmed[column], s=4, alpha=.12, color="#2563eb", rasterized=True)
            axis.scatter(missed["gamma_underestimate"], missed[column], s=12, alpha=.6, color="#dc2626", rasterized=True, label="missed strong")
            axis.axvline(0, color="black", lw=.8); axis.axhline(0, color="black", lw=.8)
            axis.set_xlabel(r"gamma underestimation: $g^{true}-g^{learned}$")
            axis.set_ylabel(ylabel); axis.grid(alpha=.2)
            axis.set_title(f"{label}: gamma error vs {'L' if column == 'L_log_error' else 'alpha'}")
            if row == 0: axis.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _analyze_run(
    run_dir: Path,
    config: dict[str, Any],
    prediction_path: Path,
    latent_truth: dict[str, np.ndarray],
    bias_root: Path,
    show_progress: bool = False,
    neutral_control_ratio: float = 5.0,
    neutral_control_min_per_pair: int = 16,
    include_all_neutral_sites: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    datasets = [str(value) for value in config["experiment"]["dataset"]]
    validation_ids, validation_hash = _validation_manifest(run_dir)
    validation_set = set(validation_ids)
    expected_lengths = {identifier: int(latent_truth[identifier].size) for identifier in validation_set}
    id_to_name = _encoding_from_config(config, REPOSITORY_ROOT)
    if not set(datasets).issubset(set(id_to_name.values())):
        id_to_name = {index: dataset for index, dataset in enumerate(datasets)}
    loss = config.get("loss", {})
    l_by_transcript, prediction_pairs, l_duplicate_difference = _prediction_profiles(
        prediction_path, expected_lengths=expected_lengths, dataset_id_to_name=id_to_name,
        log_alpha_min=float(loss.get("nb_log_alpha_min", -5.0)),
        log_alpha_max=float(loss.get("nb_log_alpha_max", 3.0)),
    )
    alpha_metrics = alpha_calibration_metrics(prediction_pairs, interior_only=True)
    alpha_metrics_full = alpha_calibration_metrics(
        prediction_pairs, interior_only=False
    )
    learned_gamma, gamma_weights, gamma_exp_difference = _load_learned_gamma(
        prediction_path, expected_lengths=expected_lengths, dataset_id_to_name=id_to_name,
    )
    programmed = {
        dataset: _load_bias_profiles(dataset_name=dataset, bias_root=bias_root, validation_ids=validation_set, expected_lengths=expected_lengths)
        for dataset in datasets
    }
    paths = config.get("dataset_config", {}).get("dataset_path", {})
    observations = {
        dataset: _load_observations((REPOSITORY_ROOT / str(paths[dataset])).resolve() if not Path(str(paths[dataset])).is_absolute() else Path(str(paths[dataset])), validation_set, expected_lengths)
        for dataset in datasets
    }
    depth = synthetic_depth_label(config)
    mass_conservation, mass_condition = synthetic_mass_conservation(config, run_dir.name)
    # Keeping a Python dict per codon was the dominant runtime cost. Accumulate
    # contiguous NumPy chunks instead; a panel has millions of positions but
    # only transcript × dataset chunks.
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)

    def add(name: str, values: Any, length: int, *, dtype: Any | None = None) -> None:
        array = np.asarray(values if np.ndim(values) else np.full(length, values), dtype=dtype)
        if array.ndim == 0:
            array = np.full(length, array.item(), dtype=dtype)
        chunks[name].append(array.reshape(-1))
    transcript_iterator: Iterable[str] = validation_ids
    if show_progress:
        transcript_iterator = tqdm(
            validation_ids,
            desc=f"transcripts: {run_dir.name[:54]}",
            unit="transcript",
            leave=False,
            mininterval=0.75,
        )
    for transcript_id in transcript_iterator:
        participants = [dataset for dataset in datasets if (transcript_id, dataset) in learned_gamma and (transcript_id, dataset) in prediction_pairs]
        if len(participants) < 2:
            continue
        length = expected_lengths[transcript_id]
        weights = np.asarray([gamma_weights[(transcript_id, dataset)] for dataset in participants], dtype=np.float64)
        raw_truth = np.stack([np.log1p(programmed[dataset][transcript_id]) for dataset in participants])
        raw_learned = np.stack([learned_gamma[(transcript_id, dataset)] for dataset in participants])
        interior_mask = cds_interior_mask(length)
        g_true_full = joint_log_gamma_gauge(raw_truth, weights)
        g_learned_full = joint_log_gamma_gauge(raw_learned, weights)
        # Keep boundary values only as a historical full-CDS-gauge diagnostic.
        # The primary interior values are re-gauged jointly over exactly the
        # same selected physical coordinates for truth and prediction.
        g_true_eval = g_true_full.copy()
        g_learned_eval = g_learned_full.copy()
        if bool(interior_mask.any()):
            g_true_eval[:, interior_mask] = joint_log_gamma_gauge(
                raw_truth, weights, position_mask=interior_mask
            )
            g_learned_eval[:, interior_mask] = joint_log_gamma_gauge(
                raw_learned, weights, position_mask=interior_mask
            )
        l_true = latent_truth[transcript_id].astype(np.float64)
        l_learned = l_by_transcript[transcript_id]
        for dataset_index, dataset in enumerate(participants):
            observed = observations[dataset].get(transcript_id)
            if observed is None:
                raise KeyError(f"Observation missing for {dataset}/{transcript_id}.")
            replicas = observed["replicas"]
            consensus = observed["consensus"]
            pair = prediction_pairs[(transcript_id, dataset)]
            base_bias = _base_bias_name(dataset)
            added_bias = programmed[dataset][transcript_id]
            # All programmed sites enter the continuous gamma/L and gamma/alpha
            # analyses. Neutral sites are controls only, so a deterministic
            # matched sample is sufficient and avoids materializing tens of
            # millions of duplicate neutral rows from cumulative panels.
            programmed_mask = added_bias > 0.0
            if include_all_neutral_sites:
                selected = np.ones(length, dtype=bool)
            else:
                neutral_positions = np.flatnonzero(~programmed_mask)
                control_count = min(
                    neutral_positions.size,
                    max(neutral_control_min_per_pair, int(math.ceil(neutral_control_ratio * int(programmed_mask.sum())))),
                )
                selected = programmed_mask.copy()
                if control_count:
                    seed_text = f"{run_dir.name}|{dataset}|{transcript_id}"
                    seed = int.from_bytes(hashlib.blake2b(seed_text.encode(), digest_size=8).digest(), "little")
                    chosen = np.random.default_rng(seed).choice(neutral_positions, size=control_count, replace=False)
                    selected[chosen] = True
            positions = np.flatnonzero(selected)
            selected_length = int(positions.size)
            if selected_length == 0:
                continue
            if replicas is not None and replicas.shape[0] >= 2:
                replicate_std = replicas.std(axis=0, ddof=1)
                replicate_cv = replicate_std / (replicas.mean(axis=0) + EPS)
                replicate_count = np.full(length, replicas.shape[0], dtype=np.int16)
            else:
                replicate_std = np.full(length, np.nan, dtype=np.float32)
                replicate_cv = np.full(length, np.nan, dtype=np.float32)
                replicate_count = np.zeros(length, dtype=np.int16) if replicas is None else np.full(length, replicas.shape[0], dtype=np.int16)
            add("run", run_dir.name, selected_length, dtype=object); add("depth", depth, selected_length, dtype=object)
            add("mass_condition", mass_condition, selected_length, dtype=object); add("dataset_count", len(datasets), selected_length, dtype=np.int16)
            add("validation_id_hash", validation_hash, selected_length, dtype=object); add("bias_name", base_bias, selected_length, dtype=object)
            add("dataset", dataset, selected_length, dtype=object); add("transcript_id", transcript_id, selected_length, dtype=object)
            add("position", positions, selected_length, dtype=np.int32)
            add("profile_length", length, selected_length, dtype=np.int32)
            add("is_interior", interior_mask[positions], selected_length, dtype=bool)
            add(
                "evaluation_domain",
                np.where(
                    interior_mask[positions],
                    "cds_interior",
                    "boundary_out_of_scope",
                ),
                selected_length,
                dtype=object,
            )
            add(
                "gauge_domain",
                np.where(
                    interior_mask[positions],
                    "cds_interior",
                    "full_cds_previous_definition",
                ),
                selected_length,
                dtype=object,
            )
            add("site_selection", np.where(programmed_mask[positions], "programmed", "neutral_control"), selected_length, dtype=object)
            add("g_true", g_true_eval[dataset_index, positions], selected_length, dtype=np.float32); add("g_learned", g_learned_eval[dataset_index, positions], selected_length, dtype=np.float32)
            add("programmed_bias_multiplier", 1.0 + added_bias[positions], selected_length, dtype=np.float32); add("programmed_bias", programmed_mask[positions], selected_length, dtype=bool)
            add("L_true", l_true[positions], selected_length, dtype=np.float32); add("L_learned", l_learned[positions], selected_length, dtype=np.float32)
            add("alpha", pair["alpha"][positions], selected_length, dtype=np.float32); add("mu", pair["mu"][positions], selected_length, dtype=np.float32)
            add("consensus", consensus[positions], selected_length, dtype=np.float32); add("replicate_count", replicate_count[positions], selected_length, dtype=np.int16)
            add("replicate_std", replicate_std[positions], selected_length, dtype=np.float32); add("replicate_cv", replicate_cv[positions], selected_length, dtype=np.float32)
            add("log1p_consensus", np.log1p(np.maximum(consensus[positions], 0.0)), selected_length, dtype=np.float32)
    if not chunks:
        raise ValueError("No complete transcript groups were available for gamma compensation analysis.")
    frame = pd.DataFrame({name: np.concatenate(values) for name, values in chunks.items()})
    for category in ("run", "depth", "mass_condition", "validation_id_hash", "bias_name", "dataset", "transcript_id", "evaluation_domain", "gauge_domain", "site_selection"):
        frame[category] = frame[category].astype("category")
    frame = classify_sites(frame)
    # Raw per-replica vectors are intentionally materialized only for the
    # missed-site table. Repeating a Python list at every neutral codon made
    # the former full diagnostic table orders of magnitude slower and larger.
    missed = frame.loc[frame["is_missed_strong"]].copy()
    if not missed.empty:
        raw_counts: list[list[float]] = []
        raw_ids: list[list[str]] = []
        for row in missed[["dataset", "transcript_id", "position"]].itertuples(index=False):
            observed = observations[str(row.dataset)][str(row.transcript_id)]
            replicas = observed["replicas"]
            raw_counts.append([] if replicas is None else replicas[:, int(row.position)].astype(float).tolist())
            raw_ids.append(observed["replica_ids"])
        missed["replicate_counts"] = raw_counts
        missed["replica_ids"] = raw_ids
    else:
        missed["replicate_counts"] = pd.Series(dtype=object)
        missed["replica_ids"] = pd.Series(dtype=object)
    metadata = {
        "run": run_dir.name, "depth": depth, "mass_condition": mass_condition, "dataset_count": len(datasets),
        "datasets": ",".join(datasets), "prediction_path": str(prediction_path), "validation_id_hash": validation_hash,
        "positions": len(frame), "l_duplicate_max_abs_difference": l_duplicate_difference,
        "gamma_exp_max_abs_difference": gamma_exp_difference,
        "site_rows_full": int(sum(expected_lengths[identifier] for identifier in validation_ids) * len(datasets)),
        "site_rows_selected": len(frame),
        "all_neutral_sites": bool(include_all_neutral_sites),
        **evaluation_domain_metadata(),
        "alpha_learning_rate_scale_resolved": config.get("optim", {}).get(
            "alpha_learning_rate_scale", float("nan")
        ),
        **alpha_metrics,
    }
    for name, value in alpha_metrics_full.items():
        if name == "alpha_true":
            continue
        metadata[f"{name}_full_previous_definition"] = value
    return frame, metadata, missed


def _analyze_run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Process-isolated worker: write a frame, never pickle it back to parent."""
    try:
        run_dir = Path(payload["run_dir"])
        frame, metadata, missed = _analyze_run(
            run_dir,
            _read_yaml(Path(payload["config_path"])),
            Path(payload["prediction_path"]),
            _load_latent_truth(Path(payload["latent_truth"])),
            Path(payload["bias_root"]),
            show_progress=False,
            neutral_control_ratio=float(payload["neutral_control_ratio"]),
            neutral_control_min_per_pair=int(payload["neutral_control_min_per_pair"]),
            include_all_neutral_sites=bool(payload["all_neutral_sites"]),
        )
        frame_path = Path(payload["work_dir"]) / f"{int(payload['index']):03d}.parquet"
        missed_path = Path(payload["work_dir"]) / f"{int(payload['index']):03d}_missed.parquet"
        frame.to_parquet(frame_path, index=False)
        missed.to_parquet(missed_path, index=False)
        return {"ok": True, "metadata": metadata, "frame_path": str(frame_path), "missed_path": str(missed_path), "selection": payload["selection"]}
    except (FileNotFoundError, KeyError, ValueError, AssertionError, OSError) as exc:
        return {"ok": False, "run": payload["run_name"], "reason": str(exc).replace("\n", " ")}


def _write_report(
    output: Path,
    summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    skipped: pd.DataFrame,
) -> None:
    lines = [
        "# Synthetic gamma compensation diagnostics", "",
        "This analysis accepts only prediction artifacts explicitly named `best_val_loss`; unsuffixed and `best_pcc` artifacts are excluded. Programmed and learned gamma are gauge-fixed using the existing weighted two-way gamma recovery gauge.", "",
        f"All primary gamma/alpha compensation metrics and plots use the CDS-observable interior `5 <= i < L-5` (boundary trim={BOUNDARY_TRIM_CODONS} codons per side). Truth and prediction are re-gauged over those identical coordinates. Boundary sites remain in `boundary_out_of_scope_diagnostics.tsv` and the labelled site table, but are out-of-scope for fair CDS-only recovery evaluation.", "",
        "Every programmed-bias site is retained. By default, neutral sites are a deterministic matched control sample (five per programmed site, with a small per-pair minimum); use `--all-neutral-sites` only when a full neutral-site export is specifically required.", "",
        "## Definitions", "",
        "For every transcript, dataset, and P-site coordinate: `gamma_error = g_learned - g_true`, `gamma_underestimate = -gamma_error`, and `L_log_error = log(L_learned) - log(K_true)`. A strong missed site has `g_true >= 1` and `g_learned < 0.5`; a correctly recovered strong site has `g_true >= 1` and `abs(gamma_error) <= 0.25`.", "",
        "`alpha_missed_minus_correct_stratified` compares log-alpha inside dataset, true-gamma-amplitude, and consensus-count strata. Replicate CV is reported separately, so an alpha difference is not called absorption merely because biological/technical replicate variation is high.", "",
        "## Conservative interpretations", "",
    ]
    if condition_summary.empty:
        lines.append("No verified best-validation-loss prediction artifacts were available. See `skipped_runs.tsv`; no scientific conclusion is drawn.")
    else:
        lines.append("## Per-condition conclusions\n")
        for row in condition_summary.sort_values(["depth", "mass_condition", "bias_name"]).itertuples(index=False):
            lines.append(
                f"- `{row.depth}` / `{row.mass_condition}` / `{row.bias_name}`: **{row.conservative_interpretation}**; "
                f"strong miss rate={row.strong_site_miss_rate:.3f}, "
                f"mean |gamma error|={row.mean_abs_gamma_error_programmed:.3f}, "
                f"mean missed L log-error={row.mean_L_log_error_missed:.3f}, "
                f"stratified log-alpha difference={row.alpha_missed_minus_correct_stratified:.3f}, "
                f"gamma/L correlation={row.gamma_L_error_correlation_programmed:.3f}."
            )
        lines.append("\n## Mass-conservation comparison\n")
        if paired_summary.empty:
            lines.append(
                "No fair mass-conserved versus mass-free pair was available. A pair is eligible only when depth, exact dataset list, dataset count, and validation transcript manifest all match."
            )
        else:
            grouped = paired_summary.groupby(["depth", "dataset_count", "mass_conserved_run", "mass_free_run"], observed=True)
            for key, part in grouped:
                mean_miss = part["strong_site_miss_rate_mass_free_minus_mass_conserved"].mean()
                mean_mae = part["mean_abs_gamma_error_programmed_mass_free_minus_mass_conserved"].mean()
                if mean_miss < 0.0 and mean_mae < 0.0:
                    verdict = "mass-free has lower gamma error on both aggregate recovery measures"
                elif mean_miss > 0.0 and mean_mae > 0.0:
                    verdict = "mass-conserved has lower gamma error on both aggregate recovery measures"
                else:
                    verdict = "the two gamma-recovery measures disagree; neither mode is declared better"
                lines.append(
                    f"- `{key[0]}`, {key[1]} matched datasets: {verdict} "
                    f"(free − conserved mean strong-miss difference={mean_miss:+.4f}; mean |gamma-error| difference={mean_mae:+.4f}). "
                    "The L/alpha deltas are mechanism diagnostics, not independent quality scores."
                )
    lines += [
        "", "## Interpretation rule", "",
        "Gamma-to-L leakage is reported only when missed strong sites have positive L error and the continuous gamma-underestimation/L-error association is positive. Alpha absorption is reported only when alpha remains elevated after amplitude/count matching and is not accompanied by a comparable increase in replicate CV. Otherwise a gamma miss is labelled unexplained or insufficiently evidenced; this analysis does not infer a mechanism from gamma error alone.", "",
        "## Files", "",
        "- `site_diagnostics.parquet`: complete numeric per-site diagnostics, including replicate count, spread, and CV.",
        "- `missed_strong_site_diagnostics.tsv.gz`: all missed strong sites with their raw replicate-count vectors serialized as JSON.",
        "- `per_bias_summary.tsv`, `condition_bias_summary.tsv`, `mass_conservation_paired_comparison.tsv`, `site_group_summary.tsv`, `alpha_by_amplitude.tsv`, and `joint_compensation_summary.tsv`: requested group-, mass-mode-, amplitude-, and joint-compensation summaries.",
        "- `run_summary.tsv`, `alpha_calibration_by_run.tsv`, `primary_runs.tsv`, and `skipped_runs.tsv`: run selection and full-validation alpha calibration audit.",
        "- `figures/`: condition-specific summaries, fair mass-mode heatmaps/direct comparisons, and AU/GC versus dinucleotide comparison. No figure pools mass modes as though they were replicates.",
        "- `representative_missed_sites.tsv`, `REPRESENTATIVE_MISSES.md`, and `figures/representative_missed_profiles/`: matched full-profile examples selected from the persisted missed-site diagnostics; generated by `plot_synthetic_missed_gamma_examples.py` without checkpoint inference.",
        "", "Only `predictions_main_val_*` is emitted by the current training entry point, so the produced analysis is a validation analysis. Test coordinates will be included only after the entry point exports same-checkpoint test predictions with an equally explicit `best_val_loss` artifact name.",
    ]
    (output / "GAMMA_COMPENSATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_diagnostics(
    sites: pd.DataFrame,
    run_summary: pd.DataFrame,
    skipped: pd.DataFrame,
    output: Path,
    figures: Path,
    *,
    write_site_parquet: bool,
) -> None:
    """Summarize already aligned site data without re-reading checkpoints.

    The raw analysis is expensive because it aligns predictions, bias profiles,
    observations, and gamma gauges.  Once ``site_diagnostics.parquet`` exists,
    this function is deliberately sufficient to regenerate every table, plot,
    and conclusion from that intermediate product alone.
    """
    if "is_interior" not in sites.columns:
        raise ValueError(
            "Site diagnostics predate the CDS-interior evaluation mask. Re-run "
            "the raw analysis so truth and prediction can be re-gauged on the "
            "same interior coordinates."
        )
    if write_site_parquet:
        sites.to_parquet(output / "site_diagnostics.parquet", index=False)
    all_summary = summarize_bias(sites, ["run", "depth", "mass_condition", "bias_name"])
    all_summary.to_csv(output / "per_bias_summary.tsv", sep="\t", index=False)
    summarize_boundary_diagnostics(
        sites, ["run", "depth", "mass_condition", "bias_name"]
    ).to_csv(
        output / "boundary_out_of_scope_diagnostics.tsv", sep="\t", index=False
    )
    _site_group_summary(sites).to_csv(output / "site_group_summary.tsv", sep="\t", index=False)
    _amplitude_summary(sites).to_csv(output / "alpha_by_amplitude.tsv", sep="\t", index=False)
    _joint_summary(sites).to_csv(output / "joint_compensation_summary.tsv", sep="\t", index=False)

    primary_runs = _choose_primary_runs(run_summary)
    primary_runs.to_csv(output / "primary_runs.tsv", sep="\t", index=False)
    primary_names = set(primary_runs["run"])
    primary_sites = sites[sites["run"].isin(primary_names)].copy()
    condition_summary = all_summary[all_summary["run"].isin(primary_names)].copy()
    # Kept as a compatibility filename, but it now remains explicitly split by
    # depth and mass condition rather than pooling incompatible experiments.
    condition_summary.to_csv(output / "condition_bias_summary.tsv", sep="\t", index=False)
    condition_summary.to_csv(output / "pooled_summary.tsv", sep="\t", index=False)

    matched_runs = _matched_mass_pairs(primary_runs)
    paired_summary = _paired_mass_summary(condition_summary, matched_runs)
    paired_summary.to_csv(output / "mass_conservation_paired_comparison.tsv", sep="\t", index=False)

    by_condition = figures / "by_condition"; by_condition.mkdir(exist_ok=True)
    for (depth, mass_condition), part in condition_summary.groupby(["depth", "mass_condition"], observed=True, sort=True):
        label = f"{depth} — {mass_condition.replace('_', ' ')}"
        safe = f"{depth}__{mass_condition}"
        _plot_condition_overview(part, figures / f"condition_overview_{safe}.png", label)
        _plot_composition_comparison(part, figures / f"composition_vs_dinucleotide_{safe}.png")
    for (depth, mass_condition, bias_name), part in primary_sites.groupby(["depth", "mass_condition", "bias_name"], observed=True, sort=True):
        safe = f"{depth}__{mass_condition}__{bias_name}"
        _plot_bias(part, by_condition / f"{safe}.png", f"{bias_name.replace('artificial_bias_', '')}: {depth}, {mass_condition.replace('_', ' ')}")

    mass_figures = figures / "mass_conservation_comparison"; mass_figures.mkdir(exist_ok=True)
    paired_groups = (
        paired_summary.groupby(
            ["depth", "mass_conserved_run", "mass_free_run"],
            observed=True,
            sort=True,
        )
        if not paired_summary.empty
        else ()
    )
    for pair_index, (_, pair) in enumerate(paired_groups):
        first = pair.iloc[0]
        prefix = f"{first.depth}__datasets{first.dataset_count}__pair{pair_index:02d}"
        _plot_paired_mass_delta(pair, mass_figures / f"{prefix}_delta_heatmap.png")
        conserved = primary_sites[primary_sites["run"] == first.mass_conserved_run]
        free = primary_sites[primary_sites["run"] == first.mass_free_run]
        for bias_name in pair["bias_name"].tolist():
            _plot_paired_mass_bias(
                conserved[conserved["bias_name"] == bias_name],
                free[free["bias_name"] == bias_name],
                mass_figures / f"{prefix}__{bias_name}.png",
                f"{bias_name.replace('artificial_bias_', '')}: matched {first.depth}, {first.dataset_count} datasets",
            )

    # These legacy plots pooled mass modes before this correction. Remove them
    # so an old image cannot be mistaken for a fair comparison.
    for stem in ("all_bias_summary", "composition_vs_dinucleotide"):
        for suffix in (".png", ".svg"):
            (figures / f"{stem}{suffix}").unlink(missing_ok=True)
    _write_report(output, all_summary, condition_summary, paired_summary, skipped)


def _work_frames_by_run(work_dir: Path) -> dict[str, Path]:
    """Map persisted worker frames to run names without loading their payload."""
    result: dict[str, Path] = {}
    for path in sorted(work_dir.glob("*.parquet")):
        if path.name.endswith("_missed.parquet"):
            continue
        table = pq.ParquetFile(path).read_row_group(0, columns=["run"])
        if not table.num_rows:
            continue
        run = str(table.column("run")[0].as_py())
        result[run] = path
    return result


def _read_work_bias(path: Path, bias_name: str) -> pd.DataFrame:
    """Load one bias at a time to keep post-processing memory bounded."""
    pieces: list[pd.DataFrame] = []
    needed = ["bias_name", "programmed_bias", "is_missed_strong", "gamma_underestimate", "L_log_error", "log_alpha", "is_interior"]
    for batch in pq.ParquetFile(path).iter_batches(columns=needed, batch_size=262_144):
        frame = batch.to_pandas()
        part = frame.loc[frame["bias_name"].astype(str) == bias_name]
        if not part.empty:
            pieces.append(part)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _postprocess_worker_intermediates(
    output: Path,
    figures: Path,
    run_summary: pd.DataFrame,
    skipped: pd.DataFrame,
) -> bool:
    """Replot from worker Parquets, avoiding a multi-gigabyte concatenation."""
    work_frames = _work_frames_by_run(output / ".gamma_compensation_work")
    if not work_frames or not set(run_summary["run"]).issubset(work_frames):
        return False
    all_summary_parts: list[pd.DataFrame] = []
    boundary_summary_parts: list[pd.DataFrame] = []
    site_group_parts: list[pd.DataFrame] = []
    amplitude_parts: list[pd.DataFrame] = []
    joint_parts: list[pd.DataFrame] = []
    primary_runs = _choose_primary_runs(run_summary)
    primary_names = set(primary_runs["run"])
    by_condition = figures / "by_condition"; by_condition.mkdir(exist_ok=True)

    for row in tqdm(run_summary.itertuples(index=False), total=len(run_summary), desc="postprocess runs", unit="run", mininterval=.5):
        frame = pd.read_parquet(work_frames[row.run])
        all_summary_parts.append(summarize_bias(frame, ["run", "depth", "mass_condition", "bias_name"]))
        boundary_summary_parts.append(
            summarize_boundary_diagnostics(
                frame, ["run", "depth", "mass_condition", "bias_name"]
            )
        )
        site_group_parts.append(_site_group_summary(frame))
        amplitude_parts.append(_amplitude_summary(frame))
        joint_parts.append(_joint_summary(frame))
        if row.run in primary_names:
            for bias_name, part in frame.groupby("bias_name", observed=True, sort=True):
                safe = f"{row.depth}__{row.mass_condition}__{bias_name}"
                _plot_bias(part, by_condition / f"{safe}.png", f"{str(bias_name).replace('artificial_bias_', '')}: {row.depth}, {str(row.mass_condition).replace('_', ' ')}")
        del frame
        gc.collect()

    all_summary = pd.concat(all_summary_parts, ignore_index=True)
    all_summary.to_csv(output / "per_bias_summary.tsv", sep="\t", index=False)
    pd.concat(boundary_summary_parts, ignore_index=True).to_csv(
        output / "boundary_out_of_scope_diagnostics.tsv", sep="\t", index=False
    )
    pd.concat(site_group_parts, ignore_index=True).to_csv(output / "site_group_summary.tsv", sep="\t", index=False)
    pd.concat(amplitude_parts, ignore_index=True).to_csv(output / "alpha_by_amplitude.tsv", sep="\t", index=False)
    pd.concat(joint_parts, ignore_index=True).to_csv(output / "joint_compensation_summary.tsv", sep="\t", index=False)
    primary_runs.to_csv(output / "primary_runs.tsv", sep="\t", index=False)
    condition_summary = all_summary[all_summary["run"].isin(primary_names)].copy()
    condition_summary.to_csv(output / "condition_bias_summary.tsv", sep="\t", index=False)
    condition_summary.to_csv(output / "pooled_summary.tsv", sep="\t", index=False)
    for (depth, mass_condition), part in condition_summary.groupby(["depth", "mass_condition"], observed=True, sort=True):
        safe = f"{depth}__{mass_condition}"
        _plot_condition_overview(part, figures / f"condition_overview_{safe}.png", f"{depth} — {str(mass_condition).replace('_', ' ')}")
        _plot_composition_comparison(part, figures / f"composition_vs_dinucleotide_{safe}.png")

    paired_summary = _write_paired_mass_figures(condition_summary, primary_runs, work_frames, output, figures)
    for stem in ("all_bias_summary", "composition_vs_dinucleotide"):
        for suffix in (".png", ".svg"):
            (figures / f"{stem}{suffix}").unlink(missing_ok=True)
    _write_report(output, all_summary, condition_summary, paired_summary, skipped)
    return True


def _write_paired_mass_figures(
    condition_summary: pd.DataFrame,
    primary_runs: pd.DataFrame,
    work_frames: dict[str, Path],
    output: Path,
    figures: Path,
) -> pd.DataFrame:
    """Write only fair mass-mode figures from small persisted worker frames."""
    paired_summary = _paired_mass_summary(condition_summary, _matched_mass_pairs(primary_runs))
    paired_summary.to_csv(output / "mass_conservation_paired_comparison.tsv", sep="\t", index=False)
    mass_figures = figures / "mass_conservation_comparison"; mass_figures.mkdir(exist_ok=True)
    paired_groups = (
        paired_summary.groupby(
            ["depth", "mass_conserved_run", "mass_free_run"],
            observed=True,
            sort=True,
        )
        if not paired_summary.empty
        else ()
    )
    for pair_index, (_, pair) in enumerate(paired_groups):
        first = pair.iloc[0]
        prefix = f"{first.depth}__datasets{first.dataset_count}__pair{pair_index:02d}"
        _plot_paired_mass_delta(pair, mass_figures / f"{prefix}_delta_heatmap.png")
        for bias_name in pair["bias_name"].tolist():
            conserved = _read_work_bias(work_frames[first.mass_conserved_run], str(bias_name))
            free = _read_work_bias(work_frames[first.mass_free_run], str(bias_name))
            _plot_paired_mass_bias(
                conserved, free, mass_figures / f"{prefix}__{bias_name}.png",
                f"{str(bias_name).replace('artificial_bias_', '')}: matched {first.depth}, {first.dataset_count} datasets",
            )
            del conserved, free
            gc.collect()
    return paired_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bias-root", type=Path, default=DEFAULT_BIAS_ROOT)
    parser.add_argument("--latent-truth", type=Path, default=DEFAULT_LATENT_TRUTH)
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Independent run-analysis processes. Use 1 for a per-transcript tqdm bar.",
    )
    parser.add_argument(
        "--all-runs", action="store_true",
        help="Analyze every cumulative panel; default analyzes only the largest panel per depth/mass condition.",
    )
    parser.add_argument(
        "--neutral-control-ratio", type=float, default=5.0,
        help="Deterministic neutral controls retained per programmed site (default: 5).",
    )
    parser.add_argument(
        "--neutral-control-min-per-pair", type=int, default=16,
        help="Minimum neutral controls per transcript-dataset pair when not using all neutral sites.",
    )
    parser.add_argument(
        "--all-neutral-sites", action="store_true",
        help="Disable neutral-control sampling and materialize every codon (very large/slow).",
    )
    parser.add_argument(
        "--postprocess-existing", action="store_true",
        help="Regenerate tables, figures, and conclusions from output-dir/site_diagnostics.parquet without re-reading checkpoints or predictions.",
    )
    parser.add_argument(
        "--postprocess-mass-comparison-only", action="store_true",
        help="Regenerate only the fair mass-conserved versus mass-free plots/report from existing summaries and worker Parquets.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail instead of recording unusable runs in skipped_runs.tsv.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.neutral_control_ratio < 0.0 or args.neutral_control_min_per_pair < 0:
        raise ValueError("Neutral-control ratio and minimum must be non-negative.")
    if args.postprocess_existing and args.postprocess_mass_comparison_only:
        raise ValueError("Choose only one post-processing mode.")
    output = args.output_dir.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    if args.postprocess_mass_comparison_only:
        run_path = output / "run_summary.tsv"
        condition_path = output / "condition_bias_summary.tsv"
        summary_path = output / "per_bias_summary.tsv"
        if not all(path.is_file() for path in (run_path, condition_path, summary_path)):
            raise FileNotFoundError(
                "--postprocess-mass-comparison-only requires run_summary.tsv, "
                "condition_bias_summary.tsv, and per_bias_summary.tsv in the output directory."
            )
        run_summary = pd.read_csv(run_path, sep="\t")
        condition_summary = pd.read_csv(condition_path, sep="\t")
        all_summary = pd.read_csv(summary_path, sep="\t")
        skipped_path = output / "skipped_runs.tsv"
        skipped = pd.read_csv(skipped_path, sep="\t") if skipped_path.is_file() else pd.DataFrame(columns=["run", "reason"])
        paired_summary = _write_paired_mass_figures(
            condition_summary, _choose_primary_runs(run_summary),
            _work_frames_by_run(output / ".gamma_compensation_work"), output, figures,
        )
        _write_report(output, all_summary, condition_summary, paired_summary, skipped)
        print(f"Regenerated fair mass-mode comparison figures in {output}")
        return
    if args.postprocess_existing:
        site_path = output / "site_diagnostics.parquet"
        run_path = output / "run_summary.tsv"
        if not run_path.is_file():
            raise FileNotFoundError(
                f"--postprocess-existing requires {run_path}."
            )
        run_summary = pd.read_csv(run_path, sep="\t")
        skipped_path = output / "skipped_runs.tsv"
        skipped = pd.read_csv(skipped_path, sep="\t") if skipped_path.is_file() else pd.DataFrame(columns=["run", "reason"])
        if _postprocess_worker_intermediates(output, figures, run_summary, skipped):
            print(f"Regenerated mass-separated diagnostics from worker intermediates in {output}")
            return
        if not site_path.is_file():
            raise FileNotFoundError(
                "No complete worker intermediates were found and the fallback "
                f"site table is absent: {site_path}."
            )
        print(f"Reusing aligned per-site diagnostics: {site_path}")
        sites = pd.read_parquet(site_path)
        _write_final_diagnostics(
            sites, run_summary, skipped, output, figures, write_site_parquet=False,
        )
        print(f"Regenerated mass-separated diagnostics from existing data in {output}")
        return
    truth = _load_latent_truth(args.latent_truth.expanduser().resolve())
    frames: list[pd.DataFrame] = []
    missed_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []
    run_dirs = discover_run_directories(args.results_root.expanduser().resolve(), args.run_prefix)
    jobs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            config_path = _historical_config(run_dir)
        except ValueError as exc:
            reason = str(exc)
            if args.strict: raise ValueError(f"{run_dir.name}: {reason}")
            skipped_rows.append({"run": run_dir.name, "reason": reason}); continue
        prediction, selection_reason = find_best_val_loss_prediction(run_dir)
        if prediction is None:
            if args.strict: raise ValueError(f"{run_dir.name}: {selection_reason}")
            skipped_rows.append({"run": run_dir.name, "reason": selection_reason}); continue
        try:
            resolved_config = _read_yaml(config_path)
        except (OSError, ValueError) as exc:
            if args.strict: raise
            skipped_rows.append({"run": run_dir.name, "reason": str(exc).replace("\n", " ")}); continue
        _, mass_condition = synthetic_mass_conservation(resolved_config, run_dir.name)
        jobs.append({
            "index": len(jobs), "run_dir": str(run_dir), "run_name": run_dir.name,
            "config_path": str(config_path), "prediction_path": str(prediction),
            "latent_truth": str(args.latent_truth.expanduser().resolve()),
            "bias_root": str(args.bias_root.expanduser().resolve()), "selection": selection_reason,
            "depth": synthetic_depth_label(resolved_config), "mass_condition": mass_condition,
            "dataset_count": len(resolved_config["experiment"]["dataset"]),
            "neutral_control_ratio": args.neutral_control_ratio,
            "neutral_control_min_per_pair": args.neutral_control_min_per_pair,
            "all_neutral_sites": args.all_neutral_sites,
        })

    eligible_before_panel_selection = len(jobs)
    if not args.all_runs:
        # Cumulative panels duplicate the same positions. The maximal completed
        # panel is the relevant one for each depth/mass condition and avoids
        # needlessly recomputing tens of millions of identical codon records.
        selected: dict[tuple[str, str], dict[str, Any]] = {}
        for job in sorted(jobs, key=lambda item: (item["depth"], item["mass_condition"], -item["dataset_count"], item["run_name"])):
            selected.setdefault((job["depth"], job["mass_condition"]), job)
        jobs = list(selected.values())

    print(
        f"Eligible runs: {eligible_before_panel_selection:,}; selected panels: {len(jobs):,}; "
        f"analysis workers: {args.workers}."
    )
    if args.workers == 1:
        # The single-process path offers a precise transcript-level ETA and is
        # useful for profiling one large panel without interleaved child output.
        for job in tqdm(jobs, desc="runs", unit="run", mininterval=0.5):
            try:
                frame, metadata, missed_frame = _analyze_run(
                    Path(job["run_dir"]), _read_yaml(Path(job["config_path"])),
                    Path(job["prediction_path"]), truth, Path(job["bias_root"]), show_progress=True,
                    neutral_control_ratio=args.neutral_control_ratio,
                    neutral_control_min_per_pair=args.neutral_control_min_per_pair,
                    include_all_neutral_sites=args.all_neutral_sites,
                )
            except (FileNotFoundError, KeyError, ValueError, AssertionError) as exc:
                if args.strict: raise
                skipped_rows.append({"run": job["run_name"], "reason": str(exc).replace("\n", " ")})
                continue
            metadata["prediction_selection"] = job["selection"]
            frames.append(frame); missed_frames.append(missed_frame); run_rows.append(metadata)
            tqdm.write(f"[ok] {job['run_name']}: {len(frame):,} sites")
    else:
        # Returning a multi-million-row DataFrame through multiprocessing would
        # dominate runtime and RAM. Workers instead persist their local Parquet
        # frame and return just a path plus small metadata.
        work_dir = output / ".gamma_compensation_work"; work_dir.mkdir(exist_ok=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_analyze_run_worker, {**job, "work_dir": str(work_dir)}) for job in jobs]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="runs", unit="run", mininterval=0.5):
                result = future.result()
                if not result["ok"]:
                    if args.strict: raise RuntimeError(f"{result['run']}: {result['reason']}")
                    skipped_rows.append({"run": result["run"], "reason": result["reason"]})
                    continue
                frame = pd.read_parquet(result["frame_path"])
                missed_frame = pd.read_parquet(result["missed_path"])
                metadata = result["metadata"]; metadata["prediction_selection"] = result["selection"]
                frames.append(frame); missed_frames.append(missed_frame); run_rows.append(metadata)
                tqdm.write(f"[ok] {metadata['run']}: {len(frame):,} sites")

    skipped = pd.DataFrame(skipped_rows, columns=["run", "reason"])
    skipped.to_csv(output / "skipped_runs.tsv", sep="\t", index=False)
    run_summary = pd.DataFrame(run_rows)
    run_summary.to_csv(output / "run_summary.tsv", sep="\t", index=False)
    alpha_calibration = run_summary.reindex(columns=ALPHA_CALIBRATION_COLUMNS)
    alpha_calibration.to_csv(
        output / "alpha_calibration_by_run.tsv", sep="\t", index=False
    )
    if not frames:
        empty = pd.DataFrame(columns=["bias_name", "conservative_interpretation"])
        unavailable = pd.DataFrame({"bias_name": CANONICAL_BIASES})
        unavailable["conservative_interpretation"] = "no_verified_best_val_loss_prediction_available"
        unavailable.to_csv(output / "per_bias_summary.tsv", sep="\t", index=False)
        unavailable.to_csv(output / "pooled_summary.tsv", sep="\t", index=False)
        empty.to_csv(output / "site_group_summary.tsv", sep="\t", index=False)
        empty.to_csv(output / "alpha_by_amplitude.tsv", sep="\t", index=False)
        empty.to_csv(output / "joint_compensation_summary.tsv", sep="\t", index=False)
        empty.to_csv(output / "primary_runs.tsv", sep="\t", index=False)
        _write_report(output, empty, empty, empty, skipped)
        print(f"No eligible best-val-loss artifacts. Wrote selection audit to {output}")
        return
    sites = pd.concat(frames, ignore_index=True)
    missed = pd.concat(missed_frames, ignore_index=True) if missed_frames else pd.DataFrame()
    # Worker-parquet round trips can return nested values as NumPy arrays;
    # normalize them before JSON serialization for the portable TSV artifact.
    missed["replicate_counts_json"] = missed["replicate_counts"].map(
        lambda value: json.dumps(np.asarray(value, dtype=float).reshape(-1).tolist())
    )
    if "replica_ids" in missed:
        missed["replica_ids"] = missed["replica_ids"].map(
            lambda value: json.dumps(np.asarray(value, dtype=object).reshape(-1).astype(str).tolist())
        )
    missed.drop(columns=["replicate_counts"]).to_csv(output / "missed_strong_site_diagnostics.tsv.gz", sep="\t", index=False, compression="gzip")
    _write_final_diagnostics(
        sites, run_summary, skipped, output, figures, write_site_parquet=True,
    )
    print(f"Wrote gamma compensation diagnostics to {output}")


if __name__ == "__main__":
    main()
