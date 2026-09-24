#!/usr/bin/env python3
"""Compare learned gamma with the programmed synthetic bias profiles.

``added_bias`` is stored as multiplier minus one, hence the physical bias is
``b = 1 + added_bias``.  Raw ``b`` is not directly comparable with model
``gamma`` because the model imposes two log-space gauges.  For every
transcript this utility applies the exact same identifiable two-way gauge to
both programmed and learned log profiles:

    c_i   = sum_d pi_d a_di
    m_d   = mean_i a_di
    a_bar = sum_d pi_d m_d
    g_di  = a_di - c_i - m_d + a_bar

where ``a = log(b)`` and ``pi`` are the checkpointed gamma-reference weights.
The terminal stop is absent from the synthetic bias annotations, so learned
gamma is truncated to that same 0-based P-site axis. Primary recovery then
uses only ``5 <= i < L-5`` and re-gauges both truth and prediction over those
identical interior coordinates. Boundary values retain a separately labelled
historical full-CDS gauge and are out-of-scope for fair CDS-only evaluation.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyses.analyze_synthetic_recovery import (
    BOUNDARY_TRIM_CODONS,
    DEFAULT_OUTPUT_DIRECTORY_NAME,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_RUN_PREFIX,
    REPOSITORY_ROOT,
    _encoding_from_config,
    _find_prediction,
    _load_latent_truth,
    _read_yaml,
    _short_case_label,
    _validation_manifest,
    cds_interior_mask,
    discover_run_directories,
    evaluation_domain_metadata,
    synthetic_mass_conservation,
    synthetic_depth_label,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


DEFAULT_BIAS_ROOT = Path("Datasets/Synthetic_data/bias_profile")
DEFAULT_LATENT_TRUTH = Path(
    "Datasets/Synthetic_data/artificial_ground_truth_kinetics_target_mean_one.parquet"
)
DEPTH_SUFFIXES = (
    "_0p25_per_codon",
    "_2_per_codon",
    "_20_per_codon",
)


@dataclass(frozen=True)
class VectorMetrics:
    pcc: float
    rmse: float
    mae: float
    calibration_slope: float
    mean_absolute_relative_error: float
    p95_absolute_log_error: float
    p99_absolute_log_error: float
    max_absolute_log_error: float
    fraction_within_1pct: float
    fraction_within_5pct: float
    fraction_within_10pct: float


def joint_log_gamma_gauge(
    log_profiles: Any,
    reference_weights: Any | None = None,
    position_mask: Any | None = None,
) -> np.ndarray:
    """Apply the joint gauge over one explicitly shared physical domain.

    If ``position_mask`` is supplied it is applied once to the common position
    axis before either gauge term is estimated.  Callers must use this same
    mask for programmed and learned profiles.
    """
    values = np.asarray(log_profiles, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError(
            "log_profiles must have shape [at least 2 datasets, positions]."
        )
    if not np.isfinite(values).all():
        raise ValueError("log_profiles contains a non-finite value.")
    if position_mask is not None:
        selected = np.asarray(position_mask, dtype=bool).reshape(-1)
        if selected.shape != (values.shape[1],):
            raise ValueError(
                f"Gamma position mask has {selected.size} values; expected "
                f"{values.shape[1]}."
            )
        if not bool(selected.any()):
            raise ValueError("Gamma evaluation position mask selects no positions.")
        values = values[:, selected]
    if reference_weights is None:
        weights = np.ones(values.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(reference_weights, dtype=np.float64).reshape(-1)
    if weights.shape != (values.shape[0],):
        raise ValueError(
            f"Expected {values.shape[0]} reference weights, got {weights.shape}."
        )
    if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
        raise ValueError("Reference weights must be finite and strictly positive.")
    pi = weights / weights.sum()
    position_center = np.sum(pi[:, None] * values, axis=0)
    dataset_means = values.mean(axis=1)
    weighted_dataset_mean = float(np.sum(pi * dataset_means))
    return (
        values
        - position_center[None, :]
        - dataset_means[:, None]
        + weighted_dataset_mean
    )


def _pcc(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(
        np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    )
    return (
        float(np.sum(x_centered * y_centered) / denominator)
        if denominator > 0.0
        else float("nan")
    )


def gamma_vector_metrics(predicted: Any, reference: Any) -> VectorMetrics:
    predicted_array = np.asarray(predicted, dtype=np.float64).reshape(-1)
    reference_array = np.asarray(reference, dtype=np.float64).reshape(-1)
    if predicted_array.shape != reference_array.shape or predicted_array.size == 0:
        raise ValueError("Predicted and reference gamma vectors must have one equal shape.")
    if not np.isfinite(predicted_array).all() or not np.isfinite(reference_array).all():
        raise ValueError("Gamma comparison vectors must be finite.")
    error = predicted_array - reference_array
    denominator = float(np.sum(reference_array**2))
    slope = (
        float(np.sum(reference_array * predicted_array) / denominator)
        if denominator > 0.0
        else float("nan")
    )
    relative_error = np.abs(np.expm1(error))
    return VectorMetrics(
        pcc=_pcc(predicted_array, reference_array),
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))),
        calibration_slope=slope,
        mean_absolute_relative_error=float(relative_error.mean()),
        p95_absolute_log_error=float(np.quantile(np.abs(error), 0.95)),
        p99_absolute_log_error=float(np.quantile(np.abs(error), 0.99)),
        max_absolute_log_error=float(np.max(np.abs(error))),
        fraction_within_1pct=float(np.mean(relative_error <= 0.01)),
        fraction_within_5pct=float(np.mean(relative_error <= 0.05)),
        fraction_within_10pct=float(np.mean(relative_error <= 0.10)),
    )


def _masked_gamma_vector_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    selected: np.ndarray,
) -> VectorMetrics:
    """Calculate metrics on a mask, returning NaNs for an empty support set."""
    if bool(np.asarray(selected, dtype=bool).any()):
        return gamma_vector_metrics(predicted[selected], reference[selected])
    return VectorMetrics(*([float("nan")] * len(VectorMetrics.__dataclass_fields__)))


def strong_site_domain_diagnostics(
    *,
    interior_true: Any,
    interior_predicted: Any,
    full_true: Any,
    full_predicted: Any,
    full_interior_mask: Any,
) -> dict[str, float | int]:
    """Separate fair interior misses from out-of-scope boundary misses.

    Interior values must already be jointly re-gauged over the interior.
    Boundary values retain the historical full-CDS gauge because the boundary
    alone is not an identifiable cross-position gauge domain.  The two rates
    are therefore diagnostics for their respective domains and are never
    pooled into the primary score.
    """
    interior_truth = np.asarray(interior_true, dtype=np.float64).reshape(-1)
    interior_learned = np.asarray(interior_predicted, dtype=np.float64).reshape(-1)
    truth_full = np.asarray(full_true, dtype=np.float64).reshape(-1)
    learned_full = np.asarray(full_predicted, dtype=np.float64).reshape(-1)
    interior_mask = np.asarray(full_interior_mask, dtype=bool).reshape(-1)
    if interior_truth.shape != interior_learned.shape:
        raise ValueError("Interior truth and prediction shapes differ.")
    if truth_full.shape != learned_full.shape or truth_full.shape != interior_mask.shape:
        raise ValueError("Full truth, prediction, and position-mask shapes differ.")

    interior_strong = interior_truth >= 1.0
    interior_missed = interior_strong & (interior_learned < 0.5)
    boundary = ~interior_mask
    boundary_strong = boundary & (truth_full >= 1.0)
    boundary_missed = boundary_strong & (learned_full < 0.5)
    full_missed = (truth_full >= 1.0) & (learned_full < 0.5)
    return {
        "interior_strong_site_miss_rate": (
            float(interior_missed.sum() / interior_strong.sum())
            if bool(interior_strong.any())
            else float("nan")
        ),
        "boundary_strong_site_miss_rate": (
            float(boundary_missed.sum() / boundary_strong.sum())
            if bool(boundary_strong.any())
            else float("nan")
        ),
        "n_interior_strong_sites": int(interior_strong.sum()),
        "n_interior_missed_strong_sites": int(interior_missed.sum()),
        "n_boundary_strong_sites": int(boundary_strong.sum()),
        "n_boundary_missed_strong_sites": int(boundary_missed.sum()),
        "fraction_of_all_misses_at_boundary": (
            float(boundary_missed.sum() / full_missed.sum())
            if bool(full_missed.any())
            else float("nan")
        ),
    }


def _base_bias_name(dataset_name: str) -> str:
    for suffix in DEPTH_SUFFIXES:
        if dataset_name.endswith(suffix):
            return dataset_name[: -len(suffix)]
    return dataset_name


def _load_bias_profiles(
    *,
    dataset_name: str,
    bias_root: Path,
    validation_ids: set[str],
    expected_lengths: dict[str, int],
) -> dict[str, np.ndarray]:
    base_name = _base_bias_name(dataset_name)
    if base_name == "artificial_ground_truth":
        return {
            transcript_id: np.zeros(expected_lengths[transcript_id], dtype=np.float32)
            for transcript_id in validation_ids
        }
    path = bias_root / f"{base_name}_compendium_added_bias_only.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Bias profile not found for {dataset_name}: {path}")

    profiles: dict[str, np.ndarray] = {}
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(
        batch_size=256,
        columns=["sample", "transcript_id", "added_bias"],
    ):
        columns = batch.to_pydict()
        for sample, transcript_id, added_bias in zip(
            columns["sample"],
            columns["transcript_id"],
            columns["added_bias"],
        ):
            transcript_id = str(transcript_id)
            if not str(sample).endswith("_mean") or transcript_id not in validation_ids:
                continue
            if transcript_id in profiles:
                raise ValueError(f"Duplicate mean bias profile: {dataset_name}/{transcript_id}")
            profile = np.asarray(added_bias, dtype=np.float64).reshape(-1)
            expected_length = expected_lengths[transcript_id]
            if profile.size != expected_length:
                raise ValueError(
                    f"Bias length mismatch for {dataset_name}/{transcript_id}: "
                    f"bias={profile.size}, expected={expected_length}."
                )
            if not np.isfinite(profile).all() or bool((profile < 0.0).any()):
                raise ValueError(
                    f"Invalid added_bias values for {dataset_name}/{transcript_id}."
                )
            profiles[transcript_id] = profile.astype(np.float32)
    missing = validation_ids - profiles.keys()
    if missing:
        raise ValueError(
            f"Bias file {path} is missing {len(missing)} validation transcripts; "
            f"first={sorted(missing)[0]}."
        )
    return profiles


def _load_learned_gamma(
    prediction_path: Path,
    *,
    expected_lengths: dict[str, int],
    dataset_id_to_name: dict[int, str],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], float], float]:
    required = (
        "transcript_id",
        "dataset_id",
        "log_gamma",
        "gamma",
        "gamma_centering_reliability",
    )
    parquet_file = pq.ParquetFile(prediction_path)
    missing = sorted(set(required) - set(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"{prediction_path} is missing gamma columns: {missing}")
    learned: dict[tuple[str, str], np.ndarray] = {}
    weights: dict[tuple[str, str], float] = {}
    maximum_exp_difference = 0.0
    for batch in parquet_file.iter_batches(batch_size=16, columns=list(required)):
        columns = batch.to_pydict()
        for transcript_id, dataset_id, log_gamma, gamma, reliability in zip(
            columns["transcript_id"],
            columns["dataset_id"],
            columns["log_gamma"],
            columns["gamma"],
            columns["gamma_centering_reliability"],
        ):
            transcript_id = str(transcript_id)
            if transcript_id not in expected_lengths:
                continue
            dataset_name = dataset_id_to_name.get(int(dataset_id))
            if dataset_name is None:
                raise KeyError(f"No dataset name for prediction dataset ID {dataset_id}.")
            key = (transcript_id, dataset_name)
            if key in learned:
                raise ValueError(f"Duplicate prediction row: {dataset_name}/{transcript_id}")
            length = expected_lengths[transcript_id]
            log_values = np.asarray(log_gamma, dtype=np.float64).reshape(-1)[:length]
            gamma_values = np.asarray(gamma, dtype=np.float64).reshape(-1)[:length]
            reliability_values = np.asarray(reliability, dtype=np.float64).reshape(-1)[:length]
            if log_values.size != length or gamma_values.size != length:
                raise ValueError(f"Short gamma profile: {dataset_name}/{transcript_id}")
            if not np.isfinite(log_values).all() or not np.isfinite(gamma_values).all():
                raise ValueError(f"Non-finite learned gamma: {dataset_name}/{transcript_id}")
            maximum_exp_difference = max(
                maximum_exp_difference,
                float(np.max(np.abs(np.exp(log_values) - gamma_values))),
            )
            positive_weights = reliability_values[reliability_values > 0.0]
            if positive_weights.size == 0 or not np.isfinite(positive_weights).all():
                raise ValueError(
                    f"Missing positive gamma reference weight: {dataset_name}/{transcript_id}"
                )
            learned[key] = log_values.astype(np.float32)
            weights[key] = float(np.median(positive_weights))
    return learned, weights, maximum_exp_difference


def _gauge_residuals(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    pi = weights / weights.sum()
    cross = float(np.max(np.abs(np.sum(pi[:, None] * values, axis=0))))
    positional = float(np.max(np.abs(values.mean(axis=1))))
    return cross, positional


def _mean_ci(values: Iterable[float], seed: int) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(2_000, dtype=np.float64)
    for index in range(means.size):
        means[index] = array[rng.integers(0, array.size, array.size)].mean()
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def _summarize_vectors(predicted: list[np.ndarray], truth: list[np.ndarray]) -> VectorMetrics:
    return gamma_vector_metrics(np.concatenate(predicted), np.concatenate(truth))


def _analyze_run(
    *,
    run_dir: Path,
    config: dict[str, Any],
    prediction_path: Path,
    latent_truth: dict[str, np.ndarray],
    bias_root: Path,
    bias_cache: dict[tuple[str, str], dict[str, np.ndarray]],
    repository_root: Path,
    run_index: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = [str(value) for value in config["experiment"]["dataset"]]
    validation_ids_list, validation_hash = _validation_manifest(run_dir)
    validation_ids = set(validation_ids_list)
    expected_lengths = {
        transcript_id: int(latent_truth[transcript_id].size)
        for transcript_id in validation_ids
    }
    depth = synthetic_depth_label(config)
    mass_conservation, mass_condition = synthetic_mass_conservation(
        config, run_dir.name
    )
    id_to_name = _encoding_from_config(config, repository_root)
    # Older completed synthetic runs stored a temporary dataset-encoding path
    # and their encoding file is no longer available after the run.  For these
    # runs the selected experiment dataset list is written in the same order
    # as the compact dataset IDs in the prediction parquet, so recover that
    # local mapping when the loaded mapping does not contain the requested
    # names.  New runs still use the persisted encoding whenever available.
    if not set(datasets).issubset(set(id_to_name.values())):
        id_to_name = {index: dataset for index, dataset in enumerate(datasets)}
    learned, learned_weights, exp_difference = _load_learned_gamma(
        prediction_path,
        expected_lengths=expected_lengths,
        dataset_id_to_name=id_to_name,
    )

    programmed: dict[str, dict[str, np.ndarray]] = {}
    for dataset in datasets:
        cache_key = (_base_bias_name(dataset), validation_hash)
        if cache_key not in bias_cache:
            bias_cache[cache_key] = _load_bias_profiles(
                dataset_name=dataset,
                bias_root=bias_root,
                validation_ids=validation_ids,
                expected_lengths=expected_lengths,
            )
        programmed[dataset] = bias_cache[cache_key]

    detailed_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    pooled_truth: list[np.ndarray] = []
    pooled_predicted: list[np.ndarray] = []
    pooled_programmed_support: list[np.ndarray] = []
    pooled_truth_full: list[np.ndarray] = []
    pooled_predicted_full: list[np.ndarray] = []
    pooled_programmed_support_full: list[np.ndarray] = []
    case_truth: dict[str, list[np.ndarray]] = defaultdict(list)
    case_predicted: dict[str, list[np.ndarray]] = defaultdict(list)
    case_programmed_support: dict[str, list[np.ndarray]] = defaultdict(list)
    case_truth_full: dict[str, list[np.ndarray]] = defaultdict(list)
    case_predicted_full: dict[str, list[np.ndarray]] = defaultdict(list)
    case_programmed_support_full: dict[str, list[np.ndarray]] = defaultdict(list)
    strong_counts = defaultdict(int)
    case_strong_counts: dict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    maximum_true_cross = 0.0
    maximum_true_position = 0.0
    maximum_pred_cross = 0.0
    maximum_pred_position = 0.0
    reference_weight_vectors: list[np.ndarray] = []

    for transcript_index, transcript_id in enumerate(sorted(validation_ids)):
        raw_truth = np.stack(
            [np.log1p(programmed[dataset][transcript_id]) for dataset in datasets]
        ).astype(np.float64)
        raw_predicted = np.stack(
            [learned[(transcript_id, dataset)] for dataset in datasets]
        ).astype(np.float64)
        reference_weights = np.asarray(
            [learned_weights[(transcript_id, dataset)] for dataset in datasets],
            dtype=np.float64,
        )
        reference_weight_vectors.append(reference_weights / reference_weights.sum())
        interior_mask = cds_interior_mask(raw_truth.shape[1])
        true_gamma_full = joint_log_gamma_gauge(raw_truth, reference_weights)
        predicted_gamma_full = joint_log_gamma_gauge(raw_predicted, reference_weights)
        if not bool(interior_mask.any()):
            # No fair CDS-only coordinates exist for this exceptionally short
            # transcript. It contributes neither numerator nor denominator to
            # primary recovery metrics.
            continue
        interior_positions = np.flatnonzero(interior_mask)
        true_gamma = joint_log_gamma_gauge(
            raw_truth, reference_weights, position_mask=interior_mask
        )
        predicted_gamma = joint_log_gamma_gauge(
            raw_predicted, reference_weights, position_mask=interior_mask
        )
        true_cross, true_position = _gauge_residuals(true_gamma, reference_weights)
        pred_cross, pred_position = _gauge_residuals(predicted_gamma, reference_weights)
        maximum_true_cross = max(maximum_true_cross, true_cross)
        maximum_true_position = max(maximum_true_position, true_position)
        maximum_pred_cross = max(maximum_pred_cross, pred_cross)
        maximum_pred_position = max(maximum_pred_position, pred_position)

        for dataset_index, dataset in enumerate(datasets):
            truth_vector = true_gamma[dataset_index]
            predicted_vector = predicted_gamma[dataset_index]
            truth_vector_full = true_gamma_full[dataset_index]
            predicted_vector_full = predicted_gamma_full[dataset_index]
            metrics = gamma_vector_metrics(predicted_vector, truth_vector)
            metrics_full = gamma_vector_metrics(
                predicted_vector_full, truth_vector_full
            )
            original_support_full = programmed[dataset][transcript_id] > 0.0
            original_support = original_support_full[interior_mask]
            support_metrics = _masked_gamma_vector_metrics(
                predicted_vector,
                truth_vector,
                original_support,
            )
            detailed_rows.append(
                {
                    "run": run_dir.name,
                    "depth": depth,
                    "mass_conservation": mass_conservation,
                    "mass_condition": mass_condition,
                    "dataset_count": len(datasets),
                    "transcript_id": transcript_id,
                    "dataset": dataset,
                    "positions": int(truth_vector.size),
                    "full_positions_previous_definition": int(truth_vector_full.size),
                    "programmed_bias_positions": int(original_support.sum()),
                    "programmed_bias_fraction": float(original_support.mean()),
                    "log_gamma_pcc": metrics.pcc,
                    "log_gamma_rmse": metrics.rmse,
                    "log_gamma_mae": metrics.mae,
                    "calibration_slope": metrics.calibration_slope,
                    "mean_absolute_relative_error": metrics.mean_absolute_relative_error,
                    "p95_absolute_log_error": metrics.p95_absolute_log_error,
                    "p99_absolute_log_error": metrics.p99_absolute_log_error,
                    "max_absolute_log_error": metrics.max_absolute_log_error,
                    "fraction_within_1pct": metrics.fraction_within_1pct,
                    "fraction_within_5pct": metrics.fraction_within_5pct,
                    "fraction_within_10pct": metrics.fraction_within_10pct,
                    "programmed_site_log_gamma_rmse": support_metrics.rmse,
                    "programmed_site_mean_absolute_relative_error": (
                        support_metrics.mean_absolute_relative_error
                    ),
                    "programmed_site_fraction_within_5pct": (
                        support_metrics.fraction_within_5pct
                    ),
                    "programmed_site_fraction_within_10pct": (
                        support_metrics.fraction_within_10pct
                    ),
                    "log_gamma_pcc_full_previous_definition": metrics_full.pcc,
                    "log_gamma_rmse_full_previous_definition": metrics_full.rmse,
                    "log_gamma_mae_full_previous_definition": metrics_full.mae,
                    "calibration_slope_full_previous_definition": (
                        metrics_full.calibration_slope
                    ),
                    **evaluation_domain_metadata(),
                }
            )
            pooled_truth.append(truth_vector)
            pooled_predicted.append(predicted_vector)
            pooled_programmed_support.append(original_support)
            pooled_truth_full.append(truth_vector_full)
            pooled_predicted_full.append(predicted_vector_full)
            pooled_programmed_support_full.append(original_support_full)
            case_truth[dataset].append(truth_vector)
            case_predicted[dataset].append(predicted_vector)
            case_programmed_support[dataset].append(original_support)
            case_truth_full[dataset].append(truth_vector_full)
            case_predicted_full[dataset].append(predicted_vector_full)
            case_programmed_support_full[dataset].append(original_support_full)

            domain_diag = strong_site_domain_diagnostics(
                interior_true=truth_vector,
                interior_predicted=predicted_vector,
                full_true=truth_vector_full,
                full_predicted=predicted_vector_full,
                full_interior_mask=interior_mask,
            )
            for target_counts in (strong_counts, case_strong_counts[dataset]):
                target_counts["interior_strong"] += int(
                    domain_diag["n_interior_strong_sites"]
                )
                target_counts["interior_missed"] += int(
                    domain_diag["n_interior_missed_strong_sites"]
                )
                target_counts["boundary_strong"] += int(
                    domain_diag["n_boundary_strong_sites"]
                )
                target_counts["boundary_missed"] += int(
                    domain_diag["n_boundary_missed_strong_sites"]
                )
                target_counts["full_strong"] += int(
                    np.sum(truth_vector_full >= 1.0)
                )
                target_counts["full_missed"] += int(
                    np.sum(
                        (truth_vector_full >= 1.0)
                        & (predicted_vector_full < 0.5)
                    )
                )

            # A bounded deterministic sample supports a readable pooled scatter.
            strongest = np.argsort(np.abs(truth_vector))[-4:]
            evenly_spaced = np.linspace(
                0, truth_vector.size - 1, num=min(4, truth_vector.size), dtype=int
            )
            for local_position in np.unique(
                np.concatenate([strongest, evenly_spaced])
            ):
                position = int(interior_positions[local_position])
                sample_rows.append(
                    {
                        "run": run_dir.name,
                        "depth": depth,
                        "mass_conservation": mass_conservation,
                        "mass_condition": mass_condition,
                        "dataset_count": len(datasets),
                        "transcript_id": transcript_id,
                        "dataset": dataset,
                        "position": position,
                        "true_log_gamma": float(truth_vector[local_position]),
                        "learned_log_gamma": float(predicted_vector[local_position]),
                        **evaluation_domain_metadata(),
                    }
                )

    detailed = pd.DataFrame(detailed_rows)
    pooled = _summarize_vectors(pooled_predicted, pooled_truth)
    pooled_full = _summarize_vectors(pooled_predicted_full, pooled_truth_full)
    pooled_support = np.concatenate(pooled_programmed_support)
    pooled_truth_array = np.concatenate(pooled_truth)
    pooled_predicted_array = np.concatenate(pooled_predicted)
    programmed_sites = gamma_vector_metrics(
        pooled_predicted_array[pooled_support],
        pooled_truth_array[pooled_support],
    )
    pooled_support_full = np.concatenate(pooled_programmed_support_full)
    pooled_truth_full_array = np.concatenate(pooled_truth_full)
    pooled_predicted_full_array = np.concatenate(pooled_predicted_full)
    programmed_sites_full = gamma_vector_metrics(
        pooled_predicted_full_array[pooled_support_full],
        pooled_truth_full_array[pooled_support_full],
    )
    transcript_means = detailed.groupby("transcript_id")["log_gamma_pcc"].mean()
    ci_low, ci_high = _mean_ci(transcript_means, seed=20_000 + run_index)
    normalized_weights = np.stack(reference_weight_vectors)
    if float(np.max(np.ptp(normalized_weights, axis=0))) > 1.0e-6:
        raise AssertionError("Gamma reference weights change across transcripts.")

    summary = {
        "run": run_dir.name,
        "depth": depth,
        "mass_conservation": mass_conservation,
        "mass_condition": mass_condition,
        "dataset_count": len(datasets),
        "datasets": ",".join(datasets),
        "validation_transcripts": len(validation_ids),
        "validation_id_hash": validation_hash,
        "transcript_dataset_pairs": len(detailed),
        "pcc_valid_pairs": int(detailed["log_gamma_pcc"].notna().sum()),
        "mean_pair_log_gamma_pcc": float(detailed["log_gamma_pcc"].mean()),
        "median_pair_log_gamma_pcc": float(detailed["log_gamma_pcc"].median()),
        "mean_pair_log_gamma_pcc_ci_low": ci_low,
        "mean_pair_log_gamma_pcc_ci_high": ci_high,
        "mean_pair_log_gamma_rmse": float(detailed["log_gamma_rmse"].mean()),
        "median_pair_log_gamma_rmse": float(detailed["log_gamma_rmse"].median()),
        "pooled_log_gamma_pcc": pooled.pcc,
        "pooled_log_gamma_rmse": pooled.rmse,
        "pooled_log_gamma_mae": pooled.mae,
        "pooled_calibration_slope": pooled.calibration_slope,
        "pooled_mean_absolute_relative_error": pooled.mean_absolute_relative_error,
        "pooled_p95_absolute_log_error": pooled.p95_absolute_log_error,
        "pooled_p99_absolute_log_error": pooled.p99_absolute_log_error,
        "pooled_max_absolute_log_error": pooled.max_absolute_log_error,
        "pooled_fraction_within_1pct": pooled.fraction_within_1pct,
        "pooled_fraction_within_5pct": pooled.fraction_within_5pct,
        "pooled_fraction_within_10pct": pooled.fraction_within_10pct,
        "pooled_log_gamma_pcc_interior": pooled.pcc,
        "pooled_log_gamma_rmse_interior": pooled.rmse,
        "pooled_log_gamma_mae_interior": pooled.mae,
        "pooled_calibration_slope_interior": pooled.calibration_slope,
        "pooled_log_gamma_pcc_full_previous_definition": pooled_full.pcc,
        "pooled_log_gamma_rmse_full_previous_definition": pooled_full.rmse,
        "pooled_log_gamma_mae_full_previous_definition": pooled_full.mae,
        "pooled_calibration_slope_full_previous_definition": (
            pooled_full.calibration_slope
        ),
        "programmed_site_positions": int(pooled_support.sum()),
        "programmed_site_fraction": float(pooled_support.mean()),
        "programmed_site_log_gamma_pcc": programmed_sites.pcc,
        "programmed_site_log_gamma_rmse": programmed_sites.rmse,
        "programmed_site_mean_absolute_relative_error": (
            programmed_sites.mean_absolute_relative_error
        ),
        "programmed_site_fraction_within_1pct": programmed_sites.fraction_within_1pct,
        "programmed_site_fraction_within_5pct": programmed_sites.fraction_within_5pct,
        "programmed_site_fraction_within_10pct": programmed_sites.fraction_within_10pct,
        "programmed_site_max_absolute_log_error": (
            programmed_sites.max_absolute_log_error
        ),
        "programmed_site_mean_absolute_relative_error_full_previous_definition": (
            programmed_sites_full.mean_absolute_relative_error
        ),
        "programmed_site_fraction_within_5pct_full_previous_definition": (
            programmed_sites_full.fraction_within_5pct
        ),
        "programmed_site_fraction_within_10pct_full_previous_definition": (
            programmed_sites_full.fraction_within_10pct
        ),
        "interior_strong_site_miss_rate": (
            strong_counts["interior_missed"] / strong_counts["interior_strong"]
            if strong_counts["interior_strong"]
            else float("nan")
        ),
        "boundary_strong_site_miss_rate": (
            strong_counts["boundary_missed"] / strong_counts["boundary_strong"]
            if strong_counts["boundary_strong"]
            else float("nan")
        ),
        "n_interior_strong_sites": strong_counts["interior_strong"],
        "n_interior_missed_strong_sites": strong_counts["interior_missed"],
        "n_boundary_strong_sites": strong_counts["boundary_strong"],
        "n_boundary_missed_strong_sites": strong_counts["boundary_missed"],
        "strong_site_miss_rate_full_previous_definition": (
            strong_counts["full_missed"] / strong_counts["full_strong"]
            if strong_counts["full_strong"]
            else float("nan")
        ),
        "fraction_of_all_misses_at_boundary": (
            strong_counts["boundary_missed"]
            / strong_counts["full_missed"]
            if strong_counts["full_missed"]
            else float("nan")
        ),
        "boundary_evaluation_scope": "out_of_scope_for_fair_cds_only_recovery",
        "boundary_gauge_definition": "full_cds_previous_definition",
        **evaluation_domain_metadata(),
        "pooled_rms_multiplicative_factor": math.exp(pooled.rmse),
        "true_cross_dataset_gauge_residual_max": maximum_true_cross,
        "true_positional_gauge_residual_max": maximum_true_position,
        "learned_cross_dataset_gauge_residual_max": maximum_pred_cross,
        "learned_positional_gauge_residual_max": maximum_pred_position,
        "gamma_vs_exp_log_gamma_max_abs_difference": exp_difference,
        "reference_weights": ",".join(f"{value:.8g}" for value in normalized_weights[0]),
        "prediction_path": str(prediction_path),
    }

    case_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        case_frame = detailed[detailed["dataset"] == dataset]
        case_pooled = _summarize_vectors(case_predicted[dataset], case_truth[dataset])
        case_pooled_full = _summarize_vectors(
            case_predicted_full[dataset], case_truth_full[dataset]
        )
        case_support = np.concatenate(case_programmed_support[dataset])
        case_truth_array = np.concatenate(case_truth[dataset])
        case_predicted_array = np.concatenate(case_predicted[dataset])
        case_programmed_sites = gamma_vector_metrics(
            case_predicted_array[case_support],
            case_truth_array[case_support],
        )
        case_support_full = np.concatenate(case_programmed_support_full[dataset])
        case_truth_full_array = np.concatenate(case_truth_full[dataset])
        case_predicted_full_array = np.concatenate(case_predicted_full[dataset])
        case_programmed_sites_full = gamma_vector_metrics(
            case_predicted_full_array[case_support_full],
            case_truth_full_array[case_support_full],
        )
        counts = case_strong_counts[dataset]
        case_rows.append(
            {
                "run": run_dir.name,
                "depth": depth,
                "mass_conservation": mass_conservation,
                "mass_condition": mass_condition,
                "dataset_count": len(datasets),
                "dataset": dataset,
                "transcripts": int(case_frame["transcript_id"].nunique()),
                "pcc_valid_pairs": int(case_frame["log_gamma_pcc"].notna().sum()),
                "mean_pair_log_gamma_pcc": float(case_frame["log_gamma_pcc"].mean()),
                "median_pair_log_gamma_pcc": float(case_frame["log_gamma_pcc"].median()),
                "mean_pair_log_gamma_rmse": float(case_frame["log_gamma_rmse"].mean()),
                "median_pair_log_gamma_rmse": float(case_frame["log_gamma_rmse"].median()),
                "pooled_log_gamma_pcc": case_pooled.pcc,
                "pooled_log_gamma_rmse": case_pooled.rmse,
                "pooled_log_gamma_mae": case_pooled.mae,
                "pooled_calibration_slope": case_pooled.calibration_slope,
                "pooled_mean_absolute_relative_error": (
                    case_pooled.mean_absolute_relative_error
                ),
                "pooled_p95_absolute_log_error": case_pooled.p95_absolute_log_error,
                "pooled_p99_absolute_log_error": case_pooled.p99_absolute_log_error,
                "pooled_max_absolute_log_error": case_pooled.max_absolute_log_error,
                "pooled_fraction_within_1pct": case_pooled.fraction_within_1pct,
                "pooled_fraction_within_5pct": case_pooled.fraction_within_5pct,
                "pooled_fraction_within_10pct": case_pooled.fraction_within_10pct,
                "pooled_log_gamma_pcc_interior": case_pooled.pcc,
                "pooled_log_gamma_rmse_interior": case_pooled.rmse,
                "pooled_log_gamma_pcc_full_previous_definition": (
                    case_pooled_full.pcc
                ),
                "pooled_log_gamma_rmse_full_previous_definition": (
                    case_pooled_full.rmse
                ),
                "programmed_site_positions": int(case_support.sum()),
                "programmed_site_log_gamma_pcc": case_programmed_sites.pcc,
                "programmed_site_log_gamma_rmse": case_programmed_sites.rmse,
                "programmed_site_mean_absolute_relative_error": (
                    case_programmed_sites.mean_absolute_relative_error
                ),
                "programmed_site_fraction_within_1pct": (
                    case_programmed_sites.fraction_within_1pct
                ),
                "programmed_site_fraction_within_5pct": (
                    case_programmed_sites.fraction_within_5pct
                ),
                "programmed_site_fraction_within_10pct": (
                    case_programmed_sites.fraction_within_10pct
                ),
                "programmed_site_max_absolute_log_error": (
                    case_programmed_sites.max_absolute_log_error
                ),
                "programmed_site_mean_absolute_relative_error_full_previous_definition": (
                    case_programmed_sites_full.mean_absolute_relative_error
                ),
                "programmed_site_fraction_within_5pct_full_previous_definition": (
                    case_programmed_sites_full.fraction_within_5pct
                ),
                "programmed_site_fraction_within_10pct_full_previous_definition": (
                    case_programmed_sites_full.fraction_within_10pct
                ),
                "interior_strong_site_miss_rate": (
                    counts["interior_missed"] / counts["interior_strong"]
                    if counts["interior_strong"]
                    else float("nan")
                ),
                "boundary_strong_site_miss_rate": (
                    counts["boundary_missed"] / counts["boundary_strong"]
                    if counts["boundary_strong"]
                    else float("nan")
                ),
                "n_interior_strong_sites": counts["interior_strong"],
                "n_boundary_strong_sites": counts["boundary_strong"],
                "strong_site_miss_rate_full_previous_definition": (
                    counts["full_missed"] / counts["full_strong"]
                    if counts["full_strong"]
                    else float("nan")
                ),
                "fraction_of_all_misses_at_boundary": (
                    counts["boundary_missed"]
                    / counts["full_missed"]
                    if counts["full_missed"]
                    else float("nan")
                ),
                **evaluation_domain_metadata(),
                "mean_programmed_bias_fraction": float(
                    case_frame["programmed_bias_fraction"].mean()
                ),
            }
        )
    return summary, pd.DataFrame(case_rows), pd.DataFrame(sample_rows), detailed


def _make_plot(
    summary: pd.DataFrame,
    cases: pd.DataFrame,
    samples: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.0), constrained_layout=True)
    mass_conditions = summary["mass_condition"].unique()
    if len(mass_conditions) != 1:
        raise ValueError("Each gamma overview must contain exactly one mass condition.")
    mass_condition = str(mass_conditions[0])
    ordered = summary.sort_values(["depth", "dataset_count"]).copy()
    ordered["plot_series"] = ordered["depth"].astype(str)
    palette = ("#6b3fa0", "#16828c", "#c46318", "#bd3d4d")
    colors = {
        series: palette[index % len(palette)]
        for index, series in enumerate(sorted(ordered["plot_series"].unique()))
    }
    for series, frame in ordered.groupby("plot_series", sort=True):
        frame = frame.sort_values("dataset_count")
        color = colors[series]
        axes[0, 0].plot(
            frame["dataset_count"], frame["mean_pair_log_gamma_pcc"],
            marker="o", linewidth=2.2, color=color, label=f"pair mean, {series}",
        )
        axes[0, 0].plot(
            frame["dataset_count"], frame["pooled_log_gamma_pcc"],
            marker="s", linestyle="--", linewidth=1.5, color=color,
            label=f"pooled, {series}",
        )
        axes[0, 0].fill_between(
            frame["dataset_count"].to_numpy(),
            frame["mean_pair_log_gamma_pcc_ci_low"].to_numpy(),
            frame["mean_pair_log_gamma_pcc_ci_high"].to_numpy(),
            color=color, alpha=0.12, linewidth=0,
        )
    axes[0, 0].set_ylim(0.94, 1.001)
    axes[0, 0].set_ylabel("Interior PCC of learned vs programmed log gamma")
    axes[0, 0].set_xlabel("Number of biased datasets")
    axes[0, 0].set_title("A. Gauge-fixed gamma shape recovery", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)

    for series, frame in ordered.groupby("plot_series", sort=True):
        frame = frame.sort_values("dataset_count")
        color = colors[series]
        axes[0, 1].plot(
            frame["dataset_count"], frame["pooled_log_gamma_rmse"],
            marker="o", linewidth=2.2, color=color, label=f"RMSE, {series}",
        )
        axes[0, 1].plot(
            frame["dataset_count"], frame["pooled_p95_absolute_log_error"],
            marker="s", linestyle="--", linewidth=1.5, color=color,
            label=f"p95 |error|, {series}",
        )
    axes[0, 1].set_ylabel("Interior log-gamma error (0 is exact)")
    axes[0, 1].set_xlabel("Number of biased datasets")
    axes[0, 1].set_title("B. Gamma amplitude error", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False)

    representative = ordered.sort_values(["dataset_count", "depth", "run"]).iloc[-1]
    representative_run = str(representative["run"])
    representative_depth = str(representative["depth"])
    largest_count = int(representative["dataset_count"])
    largest_cases = cases[cases["run"] == representative_run].copy()
    largest_cases = largest_cases.sort_values("dataset")
    x = np.arange(len(largest_cases))
    axes[1, 0].bar(
        x,
        largest_cases["mean_pair_log_gamma_pcc"],
        color="#7653a6",
    )
    axes[1, 0].set_xticks(
        x,
        [_short_case_label(value) for value in largest_cases["dataset"]],
        rotation=30,
    )
    axes[1, 0].set_ylim(0.96, 1.001)
    axes[1, 0].set_ylabel("Mean per-transcript log-gamma PCC")
    axes[1, 0].set_title(
        f"C. Individual bias recovery ({representative_depth}, "
        f"{representative['mass_condition']}, N={largest_count})",
        loc="left",
        fontweight="bold",
    )
    for index, (_, row) in enumerate(largest_cases.iterrows()):
        axes[1, 0].text(
            index,
            float(row["mean_pair_log_gamma_pcc"]) + 0.0008,
            "biased-site error\n"
            f"{100.0 * row['programmed_site_mean_absolute_relative_error']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    largest_samples = samples[samples["run"] == representative_run]
    if len(largest_samples) > 80_000:
        largest_samples = largest_samples.sample(80_000, random_state=42)
    axes[1, 1].scatter(
        largest_samples["true_log_gamma"],
        largest_samples["learned_log_gamma"],
        s=4,
        alpha=0.12,
        color="#3d5266",
        linewidths=0,
        rasterized=True,
    )
    limits = np.asarray(
        [
            largest_samples[["true_log_gamma", "learned_log_gamma"]].min().min(),
            largest_samples[["true_log_gamma", "learned_log_gamma"]].max().max(),
        ]
    )
    padding = 0.04 * max(float(limits[1] - limits[0]), 1.0)
    limits = limits + np.asarray([-padding, padding])
    axes[1, 1].plot(limits, limits, linestyle="--", color="#bd3d4d", linewidth=1.5)
    axes[1, 1].set_xlim(limits)
    axes[1, 1].set_ylim(limits)
    axes[1, 1].set_xlabel("Programmed interior-gauged log gamma")
    axes[1, 1].set_ylabel("Learned interior-gauged log gamma")
    axes[1, 1].set_title(
        "D. Position-level calibration "
        f"(pooled r={representative['pooled_log_gamma_pcc']:.4f}, "
        f"slope={representative['pooled_calibration_slope']:.3f})",
        loc="left",
        fontweight="bold",
    )

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_xticks(ordered["dataset_count"])
    axes[0, 1].set_xticks(ordered["dataset_count"])
    fig.suptitle(
        f"Recovery of programmed dataset-specific gamma biases ({mass_condition})",
        fontsize=16,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _make_per_bias_plot(
    summary: pd.DataFrame,
    cases: pd.DataFrame,
    samples: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the largest completed panel for one mass-condition/read-depth pair."""
    if summary["mass_condition"].nunique() != 1 or summary["depth"].nunique() != 1:
        raise ValueError(
            "Each individual-bias figure must contain one mass condition and one depth."
        )
    representative = summary.sort_values(["dataset_count", "depth", "run"]).iloc[-1]
    representative_run = str(representative["run"])
    representative_depth = str(representative["depth"])
    largest_count = int(representative["dataset_count"])
    largest_cases = (
        cases[cases["run"] == representative_run]
        .sort_values("dataset")
        .reset_index(drop=True)
    )
    largest_samples = samples[samples["run"] == representative_run]
    columns = 3
    rows = int(math.ceil((len(largest_cases) + 1) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15.5, 4.8 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    flattened_axes = axes.reshape(-1)
    global_limits = np.asarray(
        [
            largest_samples[["true_log_gamma", "learned_log_gamma"]].min().min(),
            largest_samples[["true_log_gamma", "learned_log_gamma"]].max().max(),
        ],
        dtype=np.float64,
    )
    padding = 0.04 * max(float(global_limits[1] - global_limits[0]), 1.0)
    global_limits += np.asarray([-padding, padding])

    for axis, (_, case) in zip(flattened_axes, largest_cases.iterrows()):
        dataset = str(case["dataset"])
        values = largest_samples[largest_samples["dataset"] == dataset]
        axis.scatter(
            values["true_log_gamma"],
            values["learned_log_gamma"],
            s=4,
            alpha=0.14,
            linewidths=0,
            color="#354f64",
            rasterized=True,
        )
        axis.plot(
            global_limits,
            global_limits,
            linestyle="--",
            color="#bd3d4d",
            linewidth=1.4,
        )
        axis.set_xlim(global_limits)
        axis.set_ylim(global_limits)
        axis.set_title(
            f"{_short_case_label(dataset)}\n"
            f"pooled r={case['pooled_log_gamma_pcc']:.4f}; "
            "biased-site error="
            f"{100.0 * case['programmed_site_mean_absolute_relative_error']:.1f}%",
            loc="left",
            fontweight="bold",
        )
        axis.set_xlabel("Programmed gauge-fixed log gamma")
        axis.set_ylabel("Learned gauge-fixed log gamma")
        axis.grid(alpha=0.18)

    explanation_axis = flattened_axes[len(largest_cases)]
    explanation_axis.axis("off")
    explanation_axis.text(
        0.04,
        0.94,
        "How to read this",
        transform=explanation_axis.transAxes,
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    explanation_axis.text(
        0.04,
        0.80,
        "The dashed line is exact recovery.\n\n"
        "Points are a deterministic diagnostic sample that includes\n"
        "the strongest programmed sites and positions across each\n"
        "held-out transcript.\n\n"
        "All plotted sites satisfy 5 <= i < L-5.\n\n"
        "The comparison removes only the common positional signal\n"
        "and dataset-constant scale that gamma cannot identify.",
        transform=explanation_axis.transAxes,
        va="top",
        fontsize=11,
        linespacing=1.35,
    )
    for axis in flattened_axes[len(largest_cases) + 1 :]:
        axis.axis("off")
    fig.suptitle(
        "Individual programmed gamma biases in the "
        f"{representative_depth}, {representative['mass_condition']}, "
        f"N={largest_count} panel",
        fontsize=16,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(label for _, label, _ in columns) + " |"
    divider = "| " + " | ".join("---" if fmt == "s" else "---:" for _, _, fmt in columns) + " |"
    rows = [header, divider]
    for record in frame.to_dict(orient="records"):
        cells: list[str] = []
        for name, _, fmt in columns:
            value = record[name]
            cells.append(str(value) if fmt == "s" else format(value, fmt))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _write_report(
    summary: pd.DataFrame,
    cases: pd.DataFrame,
    output_path: Path,
) -> None:
    ordered = summary.sort_values(["mass_condition", "depth", "dataset_count"])
    largest = ordered.sort_values(["dataset_count", "depth", "run"]).iloc[-1]
    largest_cases = cases[cases["run"] == str(largest["run"])].copy()
    largest_cases["case"] = largest_cases["dataset"].map(_short_case_label)
    panel_table = _markdown_table(
        ordered,
        [
            ("mass_condition", "mass", "s"),
            ("depth", "depth", "s"),
            ("dataset_count", "N", ".0f"),
            ("validation_transcripts", "transcripts", ".0f"),
            ("mean_pair_log_gamma_pcc", "mean pair PCC", ".4f"),
            ("pooled_log_gamma_pcc", "pooled PCC", ".4f"),
            ("pooled_log_gamma_rmse", "log-RMSE", ".4f"),
            ("pooled_calibration_slope", "slope", ".4f"),
            (
                "programmed_site_mean_absolute_relative_error",
                "biased-site mean relative error",
                ".2%",
            ),
            (
                "programmed_site_fraction_within_10pct",
                "biased sites within 10%",
                ".2%",
            ),
        ],
    )
    case_table = _markdown_table(
        largest_cases,
        [
            ("case", "bias", "s"),
            ("mean_pair_log_gamma_pcc", "mean pair PCC", ".4f"),
            ("pooled_log_gamma_pcc", "pooled PCC", ".4f"),
            ("pooled_log_gamma_rmse", "log-RMSE", ".4f"),
            ("pooled_calibration_slope", "slope", ".4f"),
            (
                "programmed_site_mean_absolute_relative_error",
                "biased-site mean relative error",
                ".2%",
            ),
            (
                "programmed_site_fraction_within_10pct",
                "biased sites within 10%",
                ".2%",
            ),
        ],
    )
    rms_percent = 100.0 * (math.exp(float(largest["pooled_log_gamma_rmse"])) - 1.0)
    overview_links = "\n\n".join(
        f"![Gamma recovery: {condition}]"
        f"(synthetic_gamma_recovery_overview_{condition}.png)"
        for condition in sorted(ordered["mass_condition"].unique())
    )
    bias_links = "\n\n".join(
        f"![Individual gamma biases: {condition}, {depth}]"
        f"(synthetic_gamma_recovery_individual_biases_{condition}_{depth}.png)"
        for condition, depth in sorted(
            set(zip(ordered["mass_condition"], ordered["depth"]))
        )
    )
    output_path.write_text(
        f"""# Synthetic gamma-bias recovery

{overview_links}

{bias_links}

## Answer

The primary numbers below evaluate only the **CDS-observable interior**
`5 <= i < L-5`. Gamma recovers the programmed **identifiable, gauge-fixed individual biases
extremely closely, but not literally to floating-point equality**. In the
largest completed panel (N={int(largest['dataset_count'])}), the equal-pair
mean log-gamma PCC is **{largest['mean_pair_log_gamma_pcc']:.4f}**, pooled PCC
is **{largest['pooled_log_gamma_pcc']:.4f}**, log-RMSE is
**{largest['pooled_log_gamma_rmse']:.4f}**, and the pooled calibration slope is
**{largest['pooled_calibration_slope']:.4f}**. The log-RMSE corresponds to an
RMS multiplicative factor of about **{rms_percent:.2f}%** from one.
Across all codons, **{largest['pooled_fraction_within_5pct']:.2%}** are within
5% multiplicative error. At positions that the raw annotation explicitly
upweighted, the mean multiplicative error is
**{largest['programmed_site_mean_absolute_relative_error']:.2%}** and
**{largest['programmed_site_fraction_within_10pct']:.2%}** are within 10%.
Those stricter site-level numbers are why the result should be described as
near-exact recovery rather than exact equality.

## Why raw bias cannot equal gamma directly

The stored annotation is `added_bias = b - 1`, where `b` multiplies expected
counts. The model fixes two otherwise unidentifiable components. For raw log
bias `a = log(b)` and normalized reference weights `pi`, the comparable truth
is:

```text
c_ti     = sum_d pi_d a_dti
m_dt     = mean_i a_dti
a_bar_t  = sum_d pi_d m_dt
g_true   = a_dti - c_ti - m_dt + a_bar_t
```

The comparison is `log(gamma_learned)` against `g_true`. A bias shared by every
dataset at one position is removed by `c_ti` and belongs to shared `L_bio`;
dataset-constant multipliers are removed by `m_dt`. Neither component is
separately identifiable from normalized profiles. Therefore comparing gamma
directly with `1 + added_bias` would give a mathematically incorrect answer.

## Recovery as the panel grows

{panel_table}

## Individual biases in the largest completed panel

Shown for `{largest['depth']}`, `{largest['mass_condition']}`,
N={int(largest['dataset_count'])}, run
`{largest['run']}`.

{case_table}

## Protocol and checks

- Only held-out validation transcripts are used ({int(largest['validation_transcripts'])}).
- Coordinates are 0-based modeled P-sites. The terminal stop was already
  absent from truth, annotations, and saved modeled profiles before masking.
- The first and last {BOUNDARY_TRIM_CODONS} modeled codons are excluded from
  every primary metric and plot because their programmed 30-nt RPF feature can
  depend on UTR sequence unavailable to the CDS-only model.
- Learned and programmed profiles are independently re-gauged over the same
  interior physical P-site coordinates using checkpointed reference weights.
- Boundary strong-site miss rate is
  **{largest['boundary_strong_site_miss_rate']:.2%}** over
  {int(largest['n_boundary_strong_sites'])} sites and is retained in
  `synthetic_gamma_recovery_boundary_out_of_scope.tsv`; it is explicitly
  out-of-scope for fair CDS-only recovery evaluation.
- Metrics are first calculated per transcript-dataset pair and averaged
  equally; pooled metrics are also shown as a length-weighted diagnostic.
- The scatter plots use a deterministic sample that includes the strongest
  programmed sites and evenly spaced positions; the tables use every codon.
- PCC measures positional shape; log-RMSE measures amplitude; slope one is
  perfect calibration.
- Maximum true/learned gauge residuals are
  {ordered['true_cross_dataset_gauge_residual_max'].max():.2e}/
  {ordered['learned_cross_dataset_gauge_residual_max'].max():.2e} across
  datasets and {ordered['true_positional_gauge_residual_max'].max():.2e}/
  {ordered['learned_positional_gauge_residual_max'].max():.2e} across
  positions.

## Limits

1. Only prefix-matching runs with completed prediction parquets are included;
   incomplete runs are recorded in `synthetic_gamma_recovery_skipped_runs.tsv`.
2. Analyzed read-depth labels: {', '.join(f'`{value}`' for value in sorted(ordered['depth'].unique()))}.
   Missing depths and panels are not inferred.
3. Panels are cumulative, so dataset count and bias composition are confounded.
4. This establishes recovery for the deterministic synthetic mechanisms, not
   uniqueness or correctness of gamma on real biological datasets.
""",
        encoding="utf-8",
    )


def analyze(args: argparse.Namespace) -> dict[str, Path]:
    repository_root = REPOSITORY_ROOT
    results_root = Path(args.results_root).expanduser().resolve()
    bias_root = Path(args.bias_root).expanduser()
    if not bias_root.is_absolute():
        bias_root = repository_root / bias_root
    latent_path = Path(args.latent_ground_truth).expanduser()
    if not latent_path.is_absolute():
        latent_path = repository_root / latent_path
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_truth = _load_latent_truth(latent_path)
    bias_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    summary_rows: list[dict[str, Any]] = []
    case_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    detailed_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, str]] = []

    run_dirs = discover_run_directories(results_root, args.run_prefix)
    if not run_dirs:
        raise RuntimeError(
            f"No run directories beginning {args.run_prefix!r} were found "
            f"directly below {results_root}."
        )
    print(
        f"Found {len(run_dirs)} run directories beginning "
        f"{args.run_prefix!r} below {results_root}."
    )
    checkpoint_variant = str(args.checkpoint_variant)
    for run_index, run_dir in enumerate(run_dirs):
        config_paths = sorted(run_dir.rglob("config.yaml"))
        if len(config_paths) != 1:
            reason = f"expected one config.yaml, found {len(config_paths)}"
            if args.strict:
                raise ValueError(f"{run_dir.name}: {reason}.")
            print(f"[skip] {run_dir.name}: {reason}.")
            skipped_rows.append({"run": run_dir.name, "reason": reason})
            continue
        try:
            # A result tree may be copied while this analysis is running.  In
            # non-strict mode, a YAML, split manifest, prediction parquet, or
            # bias parquet that is not present/readable yet is a skipped run,
            # not a reason to discard metrics from every completed run.
            config = _read_yaml(config_paths[0])
            prediction_path = _find_prediction(
                run_dir,
                checkpoint_variant=checkpoint_variant,
            )
            if prediction_path is None:
                raise FileNotFoundError(
                    f"{checkpoint_variant} prediction parquet is not available yet"
                )
            result = _analyze_run(
                run_dir=run_dir,
                config=config,
                prediction_path=prediction_path,
                latent_truth=latent_truth,
                bias_root=bias_root,
                bias_cache=bias_cache,
                repository_root=repository_root,
                run_index=run_index,
            )
        except Exception as exc:
            if args.strict:
                raise
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[skip] {run_dir.name}: {reason}")
            skipped_rows.append({"run": run_dir.name, "reason": reason})
            continue
        summary, cases, samples, detailed = result
        summary["checkpoint_variant"] = checkpoint_variant
        for frame in (cases, samples, detailed):
            frame["checkpoint_variant"] = checkpoint_variant
        summary_rows.append(summary)
        case_frames.append(cases)
        sample_frames.append(samples)
        detailed_frames.append(detailed)
        gc.collect()

    if not summary_rows:
        raise RuntimeError(
            f"No complete, analyzable {checkpoint_variant} prediction runs were "
            f"found below {results_root}. See the skipped-run messages above."
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mass_condition", "depth", "dataset_count"]
    )
    cases = pd.concat(case_frames, ignore_index=True)
    samples = pd.concat(sample_frames, ignore_index=True)
    detailed = pd.concat(detailed_frames, ignore_index=True)

    comparison_rows: list[dict[str, Any]] = []
    for row in summary.to_dict(orient="records"):
        metric_columns = (
            (
                "gamma_pooled_PCC",
                "pooled_log_gamma_pcc_full_previous_definition",
                "pooled_log_gamma_pcc_interior",
            ),
            (
                "gamma_log_RMSE",
                "pooled_log_gamma_rmse_full_previous_definition",
                "pooled_log_gamma_rmse_interior",
            ),
            (
                "gamma_log_MAE",
                "pooled_log_gamma_mae_full_previous_definition",
                "pooled_log_gamma_mae_interior",
            ),
            (
                "gamma_calibration_slope",
                "pooled_calibration_slope_full_previous_definition",
                "pooled_calibration_slope_interior",
            ),
            (
                "biased_site_multiplicative_error",
                "programmed_site_mean_absolute_relative_error_full_previous_definition",
                "programmed_site_mean_absolute_relative_error",
            ),
            (
                "biased_sites_within_10pct",
                "programmed_site_fraction_within_10pct_full_previous_definition",
                "programmed_site_fraction_within_10pct",
            ),
            (
                "biased_sites_within_5pct",
                "programmed_site_fraction_within_5pct_full_previous_definition",
                "programmed_site_fraction_within_5pct",
            ),
            (
                "strong_site_miss_rate",
                "strong_site_miss_rate_full_previous_definition",
                "interior_strong_site_miss_rate",
            ),
        )
        for metric, full_column, interior_column in metric_columns:
            comparison_rows.append(
                {
                    "run": row["run"],
                    "depth": row["depth"],
                    "mass_condition": row["mass_condition"],
                    "dataset_count": row["dataset_count"],
                    "metric": metric,
                    "original_full_cds": row[full_column],
                    "interior_only": row[interior_column],
                    **evaluation_domain_metadata(),
                }
            )

    case_comparison_rows: list[dict[str, Any]] = []
    for row in cases.to_dict(orient="records"):
        for metric, full_column, interior_column in (
            (
                "gamma_pooled_PCC",
                "pooled_log_gamma_pcc_full_previous_definition",
                "pooled_log_gamma_pcc_interior",
            ),
            (
                "gamma_log_RMSE",
                "pooled_log_gamma_rmse_full_previous_definition",
                "pooled_log_gamma_rmse_interior",
            ),
            (
                "biased_site_multiplicative_error",
                "programmed_site_mean_absolute_relative_error_full_previous_definition",
                "programmed_site_mean_absolute_relative_error",
            ),
            (
                "biased_sites_within_10pct",
                "programmed_site_fraction_within_10pct_full_previous_definition",
                "programmed_site_fraction_within_10pct",
            ),
            (
                "biased_sites_within_5pct",
                "programmed_site_fraction_within_5pct_full_previous_definition",
                "programmed_site_fraction_within_5pct",
            ),
            (
                "strong_site_miss_rate",
                "strong_site_miss_rate_full_previous_definition",
                "interior_strong_site_miss_rate",
            ),
        ):
            case_comparison_rows.append(
                {
                    "run": row["run"],
                    "depth": row["depth"],
                    "mass_condition": row["mass_condition"],
                    "dataset_count": row["dataset_count"],
                    "dataset": row["dataset"],
                    "metric": metric,
                    "original_full_cds": row[full_column],
                    "interior_only": row[interior_column],
                    **evaluation_domain_metadata(),
                }
            )

    paths = {
        "summary": output_dir / "synthetic_gamma_recovery_by_panel.tsv",
        "cases": output_dir / "synthetic_gamma_recovery_by_bias_case.tsv",
        "detailed": output_dir / "synthetic_gamma_recovery_by_transcript_dataset.tsv.gz",
        "samples": output_dir / "synthetic_gamma_recovery_position_sample.tsv.gz",
        "report": output_dir / "GAMMA_RECOVERY.md",
        "skipped": output_dir / "synthetic_gamma_recovery_skipped_runs.tsv",
        "domain_comparison": output_dir
        / "synthetic_gamma_recovery_full_vs_interior.tsv",
        "case_domain_comparison": output_dir
        / "synthetic_gamma_recovery_by_bias_full_vs_interior.tsv",
        "boundary_diagnostics": output_dir
        / "synthetic_gamma_recovery_boundary_out_of_scope.tsv",
    }
    summary.to_csv(paths["summary"], sep="\t", index=False)
    cases.to_csv(paths["cases"], sep="\t", index=False)
    detailed.to_csv(paths["detailed"], sep="\t", index=False, compression="gzip")
    samples.to_csv(paths["samples"], sep="\t", index=False, compression="gzip")
    pd.DataFrame(comparison_rows).to_csv(
        paths["domain_comparison"], sep="\t", index=False
    )
    pd.DataFrame(case_comparison_rows).to_csv(
        paths["case_domain_comparison"], sep="\t", index=False
    )
    summary[
        [
            "run",
            "depth",
            "mass_condition",
            "dataset_count",
            "boundary_strong_site_miss_rate",
            "n_boundary_strong_sites",
            "n_boundary_missed_strong_sites",
            "fraction_of_all_misses_at_boundary",
            "boundary_evaluation_scope",
            "boundary_gauge_definition",
        ]
    ].to_csv(paths["boundary_diagnostics"], sep="\t", index=False)
    pd.DataFrame(skipped_rows, columns=["run", "reason"]).to_csv(
        paths["skipped"], sep="\t", index=False
    )
    for mass_condition, condition_summary in summary.groupby(
        "mass_condition", sort=True
    ):
        condition_cases = cases[cases["mass_condition"] == mass_condition]
        condition_samples = samples[samples["mass_condition"] == mass_condition]
        overview_path = output_dir / (
            f"synthetic_gamma_recovery_overview_{mass_condition}.png"
        )
        _make_plot(
            condition_summary.copy(), condition_cases, condition_samples, overview_path
        )
        paths[f"plot_{mass_condition}"] = overview_path
        for depth, depth_summary in condition_summary.groupby("depth", sort=True):
            depth_cases = condition_cases[condition_cases["depth"] == depth]
            depth_samples = condition_samples[condition_samples["depth"] == depth]
            bias_path = output_dir / (
                "synthetic_gamma_recovery_individual_biases_"
                f"{mass_condition}_{depth}.png"
            )
            _make_per_bias_plot(depth_summary, depth_cases, depth_samples, bias_path)
            paths[f"bias_plot_{mass_condition}_{depth}"] = bias_path
    _write_report(summary, cases, paths["report"])
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--run-prefix",
        default=DEFAULT_RUN_PREFIX,
        help=(
            "Analyze only immediate result directories beginning with this "
            f"prefix (default: {DEFAULT_RUN_PREFIX!r})."
        ),
    )
    parser.add_argument("--bias-root", default=str(DEFAULT_BIAS_ROOT))
    parser.add_argument("--latent-ground-truth", default=str(DEFAULT_LATENT_TRUTH))
    parser.add_argument(
        "--checkpoint-variant",
        choices=("best_pcc", "best_val_loss"),
        default="best_pcc",
        help=(
            "Prediction checkpoint to analyze. Historical unsuffixed prediction "
            "files are accepted only for best_pcc."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of recording and skipping incomplete run directories.",
    )
    return parser


def main() -> None:
    paths = analyze(build_parser().parse_args())
    print("Synthetic gamma recovery report written:")
    for label, path in paths.items():
        print(f"  {label:9s} {path}")


if __name__ == "__main__":
    main()
