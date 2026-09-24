#!/usr/bin/env python3
"""Audit aggregate mass factors in frozen synthetic RiboUnmix predictions.

The primary positional domain is the reported synthetic recovery mask: remove
the synthetic terminal entry and then exclude ten sense codons from each end.
The script additionally audits the full saved model mask because the decoder's
scale, L normalization, and within-profile gamma gauge were defined there.

No checkpoint is loaded, no model is fitted, and no oracle quantity is used to
select a checkpoint. Prediction arrays and observed replicas are streamed from
the frozen prediction exports selected by the configured checkpoint policy and
from the saved dataset files.  The manuscript configuration explicitly uses
best-validation-loss exports.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Iterator, Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import gammaln
from scipy.stats import rankdata, spearmanr
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "analyses") not in sys.path:
    sys.path.insert(0, str(ROOT / "analyses"))

import analyze_synthetic_reference_target_audit as reference  # noqa: E402
from Utils.publication_plot_style import publication_rc  # noqa: E402


DEFAULT_CONFIG = ROOT / "analyses/configs/synthetic_mass_factor.yaml"


@dataclass(frozen=True)
class RunSettings:
    raw_log_gamma_bound: float
    likelihood_mean_floor: float
    mass_conservation: bool
    gamma_gauge: str
    observation_paths: dict[str, Path]
    configured_observation_paths: dict[str, str]


@dataclass(frozen=True)
class Observation:
    consensus: np.ndarray
    replicas: tuple[np.ndarray, ...]
    replica_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-cross-depth", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return result


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def dataset_depth(run: reference.RunSpec, dataset: str) -> str:
    for depth in reference.DEPTH_ORDER:
        suffix = f"_{depth}"
        if dataset.endswith(suffix):
            return depth
    if run.depth not in reference.DEPTH_ORDER:
        raise ValueError(f"Cannot resolve depth for {run.run_id}/{dataset}")
    return run.depth


def observation_key(run: reference.RunSpec, dataset: str) -> str:
    return f"{reference.base_bias_name(dataset)}__{dataset_depth(run, dataset)}"


def inspect_run_settings(
    run: reference.RunSpec, weighted_observation_root: Path
) -> RunSettings:
    config = load_yaml(run.config_path)
    model = config["model"]
    dataset_paths = config["dataset_config"]["dataset_path"]
    resolved: dict[str, Path] = {}
    configured: dict[str, str] = {}
    for dataset in run.datasets:
        if dataset not in dataset_paths:
            raise KeyError(f"{run.run_id}: no configured path for {dataset}")
        configured[dataset] = str(dataset_paths[dataset])
        path = resolve_path(configured[dataset])
        if not path.is_file():
            # Cross-depth launchers wrote node-local /tmp paths into Hydra's
            # frozen config. Resolve the same dataset ID to the repository's
            # durable post-processed observation file; never infer from row
            # order or from predictions.
            depth = dataset_depth(run, dataset)
            bias = reference.base_bias_name(dataset)
            durable_path = weighted_observation_root / depth / f"{bias}.parquet"
            if not durable_path.is_file():
                raise FileNotFoundError(
                    f"Configured observation is unavailable ({path}); "
                    f"durable dataset-ID mapping is also unavailable ({durable_path})"
                )
            path = durable_path
        resolved[dataset] = path
    return RunSettings(
        raw_log_gamma_bound=float(model["dataset_bias_params"]["raw_log_gamma_bound"]),
        likelihood_mean_floor=float(config["loss"]["eps"]),
        mass_conservation=bool(model["mass_conservation"]),
        gamma_gauge=str(model["gamma_centering"]["dataset_constant_scale_gauge"]),
        observation_paths=resolved,
        configured_observation_paths=configured,
    )


def collect_observation_sources(
    runs: Sequence[reference.RunSpec], settings: dict[str, RunSettings]
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for run in runs:
        for dataset in run.datasets:
            key = observation_key(run, dataset)
            path = settings[run.run_id].observation_paths[dataset]
            if key in paths and paths[key].resolve() != path.resolve():
                raise ValueError(f"Conflicting observation paths for {key}")
            paths[key] = path
            depth = dataset_depth(run, dataset)
            bias = reference.base_bias_name(dataset)
            raw_path = (
                ROOT / "Datasets/Synthetic_data" / depth
                / f"{bias}_psite_counts_{depth}.parquet"
            )
            metadata = reference.parquet_metadata(raw_path)
            provenance[key] = {
                "dataset": dataset,
                "bias": bias,
                "depth": depth,
                "weighted_path": str(path.relative_to(ROOT)),
                "weighted_sha256": file_sha256(path),
                "raw_count_path": str(raw_path.relative_to(ROOT)),
                "raw_count_sha256": file_sha256(raw_path),
                "bias_seed": metadata.get("riboart.bias_seed"),
                "observation_sampling_seed": metadata.get(
                    "riboart.observation_sampling_seed"
                ),
                "replicate_ids": metadata.get("riboart.samples"),
            }
    return paths, provenance


def load_observations(
    paths: dict[str, Path], transcript_ids: set[str]
) -> dict[str, dict[str, Observation]]:
    result: dict[str, dict[str, Observation]] = {}
    for index, (key, path) in enumerate(sorted(paths.items()), start=1):
        rows: dict[str, Observation] = {}
        reader = pq.ParquetFile(path)
        for batch in reader.iter_batches(
            columns=["id", "ribo", "ribo_cds_replicas", "replica_ids"],
            batch_size=128,
            use_threads=False,
        ):
            for row in batch.to_pylist():
                transcript_id = str(row["id"])
                if transcript_id not in transcript_ids:
                    continue
                consensus = np.asarray(row["ribo"], dtype=np.float64)
                replicas = tuple(
                    np.asarray(values, dtype=np.float64)
                    for values in row["ribo_cds_replicas"]
                )
                replica_ids = tuple(str(value) for value in row["replica_ids"])
                if len(replicas) != len(replica_ids) or not replicas:
                    raise ValueError(f"{key}/{transcript_id}: invalid replicas")
                if any(values.shape != consensus.shape for values in replicas):
                    raise ValueError(f"{key}/{transcript_id}: replica shape mismatch")
                rows[transcript_id] = Observation(
                    consensus=consensus,
                    replicas=replicas,
                    replica_ids=replica_ids,
                )
        missing = transcript_ids - set(rows)
        if missing:
            raise KeyError(f"{key}: missing observations for {sorted(missing)[:5]}")
        result[key] = rows
        print(f"Loaded observations [{index}/{len(paths)}] {key}", flush=True)
    return result


def iter_prediction_rows(
    run: reference.RunSpec, transcript_ids: set[str], batch_size: int = 64
) -> Iterator[dict[str, Any]]:
    columns = [
        "transcript_id", "dataset_id", "mask", "target", "mu",
        "likelihood_positive_mean", "L_bio", "gamma", "log_gamma",
        "gamma_raw", "log_gamma_raw", "scale_dt", "normalized_shape",
        "log_sigma",
    ]
    reader = pq.ParquetFile(run.prediction_path)
    for batch in reader.iter_batches(columns=columns, batch_size=batch_size, use_threads=False):
        for row in batch.to_pylist():
            if str(row["transcript_id"]) in transcript_ids:
                yield row


def mean_one(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
        raise ValueError("Expected a finite one-dimensional profile")
    if np.any(values <= 0):
        raise ValueError("Mass analysis requires strictly positive factors")
    return values / float(values.mean())


def covariance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(left * right) - np.mean(left) * np.mean(right))


def nb2_nll(y: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    theta = 1.0 / alpha
    log_probability = (
        gammaln(y + theta)
        - gammaln(theta)
        - gammaln(y + 1.0)
        + theta * (np.log(theta) - np.log(theta + mu))
        + y * (np.log(mu) - np.log(theta + mu))
    )
    return float(-np.mean(log_probability))


def regression_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def calibration_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    residual = y - x
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(spearmanr(x, y).statistic)
    intercept, slope = regression_line(x, y)
    return {
        "mass_pearson": pearson,
        "mass_spearman": spearman,
        "mass_mae": float(np.mean(np.abs(residual))),
        "mass_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "log_mass_error_median": float(np.median(residual)),
    }


def oracle_profiles(
    q_full: np.ndarray,
    biases: dict[str, dict[str, np.ndarray]],
    transcript_id: str,
    datasets: Sequence[str],
    pi: dict[str, float],
    take: np.ndarray,
) -> dict[str, Any]:
    q = q_full[take]
    log_g = np.zeros(int(take.sum()), dtype=np.float64)
    log_h_geometric = np.zeros_like(log_g)
    for dataset in datasets:
        bias = biases[reference.base_bias_name(dataset)][transcript_id][take]
        log_g += pi[dataset] * np.log(bias)
        h_star = mean_one(q * bias)
        log_h_geometric += pi[dataset] * np.log(h_star)
    g = np.exp(log_g)
    l_ref_bias = mean_one(q * g)
    l_ref_profile = mean_one(np.exp(log_h_geometric))
    return {
        "Q": mean_one(q),
        "G": g,
        "log_G": log_g,
        "L_ref": l_ref_bias,
        "L_ref_profile": l_ref_profile,
        "L_ref_construction_max_abs": float(
            np.max(np.abs(l_ref_bias - l_ref_profile))
        ),
        "sigma_log_G": float(np.std(log_g, ddof=0)),
    }


def analyze_run(
    run: reference.RunSpec,
    cohort: set[str],
    occupancies: dict[str, np.ndarray],
    biases: dict[str, dict[str, np.ndarray]],
    observations: dict[str, dict[str, Observation]],
    settings: RunSettings,
    pi: dict[str, float],
    trim: int,
    tolerances: dict[str, float],
    profile_cache: dict[tuple[str, str], dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    panel_id = reference.panel_hash(run.datasets, pi)
    dataset_order = {dataset: index + 1 for index, dataset in enumerate(run.datasets)}
    states: dict[str, dict[str, Any]] = {}
    mass_rows: list[dict[str, Any]] = []
    maxima: defaultdict[str, float] = defaultdict(float)

    for row in iter_prediction_rows(run, cohort):
        transcript_id = str(row["transcript_id"])
        dataset_id = int(row["dataset_id"])
        dataset = run.dataset_id_to_name.get(dataset_id)
        if dataset not in run.datasets:
            raise ValueError(f"{run.run_id}/{transcript_id}: invalid dataset ID {dataset_id}")

        state = states.get(transcript_id)
        if state is None:
            q_full = occupancies[transcript_id]
            model_mask = np.asarray(row["mask"], dtype=bool)
            l_full = np.asarray(row["L_bio"], dtype=np.float64)
            if l_full.size != q_full.size + 1:
                raise ValueError(
                    f"{run.run_id}/{transcript_id}: expected one synthetic terminal entry; "
                    f"model={l_full.size}, occupancy={q_full.size}"
                )
            if model_mask.shape != l_full.shape:
                raise ValueError(f"{run.run_id}/{transcript_id}: mask shape mismatch")
            sense_mask = model_mask[:-1].copy()
            sense_positions = np.arange(q_full.size)
            take = (
                sense_mask
                & (sense_positions >= trim)
                & (sense_positions < q_full.size - trim)
            )
            if int(take.sum()) < 3:
                raise ValueError(f"{transcript_id}: fewer than three reported positions")
            evaluation_mask = np.zeros_like(model_mask)
            evaluation_mask[:-1] = take
            l_eval_raw = l_full[:-1][take]
            l_hat = mean_one(l_eval_raw)
            oracle = oracle_profiles(
                q_full, biases, transcript_id, run.datasets, pi, take
            )
            q = np.asarray(oracle["Q"])
            l_ref = np.asarray(oracle["L_ref"])
            profile_l_ref = reference.profile_metrics(l_hat, l_ref)
            profile_ref_q = reference.profile_metrics(l_ref, q)
            profile_l_q = reference.profile_metrics(l_hat, q)
            maxima["learned_L_primary_mean_one"] = max(
                maxima["learned_L_primary_mean_one"], abs(float(l_hat.mean()) - 1.0)
            )
            maxima["learned_L_full_model_mean_one"] = max(
                maxima["learned_L_full_model_mean_one"],
                abs(float(l_full[model_mask].mean()) - 1.0),
            )
            maxima["oracle_L_mean_one"] = max(
                maxima["oracle_L_mean_one"], abs(float(l_ref.mean()) - 1.0)
            )
            maxima["oracle_profile_construction"] = max(
                maxima["oracle_profile_construction"],
                float(oracle["L_ref_construction_max_abs"]),
            )
            state = {
                "seen_datasets": set(),
                "row_indices": [],
                "q_full": q_full,
                "model_mask": model_mask,
                "evaluation_mask": evaluation_mask,
                "take": take,
                "l_full": l_full,
                "l_eval_raw": l_eval_raw,
                "l_hat": l_hat,
                "oracle": oracle,
                "weighted_log_gamma_learned_eval": np.zeros(int(take.sum())),
                "weighted_log_gamma_learned_full": np.zeros(int(model_mask.sum())),
                "weighted_log_gamma_oracle": np.zeros(int(take.sum())),
                "weighted_mass_learned": 0.0,
                "weighted_mass_oracle": 0.0,
                "weighted_log_mass_sq_learned": 0.0,
                "weighted_log_mass_sq_oracle": 0.0,
                "within_center_eval_max": 0.0,
                "within_center_full_max": 0.0,
                "oracle_within_center_max": 0.0,
                "oracle_profile_gamma_max": 0.0,
                "profile_l_ref": profile_l_ref,
                "profile_ref_q": profile_ref_q,
                "profile_l_q": profile_l_q,
            }
            states[transcript_id] = state
            if (
                run.family == "within_depth"
                and run.depth == "20_per_codon"
                and run.n_datasets == 10
                and run.reference_weighting == "equal"
            ):
                profile_cache[(run.run_id, transcript_id)] = {
                    "positions": sense_positions[take],
                    "Q": q,
                    "L_ref": l_ref,
                    "L_hat": l_hat,
                }

        if dataset in state["seen_datasets"]:
            raise ValueError(f"{run.run_id}/{transcript_id}: duplicate {dataset}")
        model_mask = np.asarray(state["model_mask"], dtype=bool)
        evaluation_mask = np.asarray(state["evaluation_mask"], dtype=bool)
        take = np.asarray(state["take"], dtype=bool)
        q_full = np.asarray(state["q_full"])
        l_full = np.asarray(state["l_full"])
        l_hat = np.asarray(state["l_hat"])
        l_ref = np.asarray(state["oracle"]["L_ref"])
        log_g = np.asarray(state["oracle"]["log_G"])

        other_l = np.asarray(row["L_bio"], dtype=np.float64)
        other_mask = np.asarray(row["mask"], dtype=bool)
        if not np.allclose(other_l, l_full, rtol=0.0, atol=1.0e-7):
            raise ValueError(f"{run.run_id}/{transcript_id}: L differs by dataset")
        if not np.array_equal(other_mask, model_mask):
            raise ValueError(f"{run.run_id}/{transcript_id}: mask differs by dataset")

        gamma_full = np.asarray(row["gamma"], dtype=np.float64)
        log_gamma_full = np.asarray(row["log_gamma"], dtype=np.float64)
        log_gamma_raw_full = np.asarray(row["log_gamma_raw"], dtype=np.float64)
        gamma_eval = gamma_full[evaluation_mask]
        log_gamma_eval = log_gamma_full[evaluation_mask]
        learned_within_eval = abs(float(log_gamma_eval.mean()))
        learned_within_full = abs(float(log_gamma_full[model_mask].mean()))
        state["within_center_eval_max"] = max(
            state["within_center_eval_max"], learned_within_eval
        )
        state["within_center_full_max"] = max(
            state["within_center_full_max"], learned_within_full
        )
        state["weighted_log_gamma_learned_eval"] += pi[dataset] * log_gamma_eval
        state["weighted_log_gamma_learned_full"] += (
            pi[dataset] * log_gamma_full[model_mask]
        )

        bias_eval = biases[reference.base_bias_name(dataset)][transcript_id][take]
        log_bias = np.log(bias_eval)
        u = log_bias - log_g
        log_gamma_ref = u - float(u.mean())
        gamma_ref = np.exp(log_gamma_ref)
        h_star = mean_one(q_full[take] * bias_eval)
        log_gamma_ref_profile = np.log(h_star) - np.log(l_ref)
        log_gamma_ref_profile -= float(log_gamma_ref_profile.mean())
        oracle_gamma_construction = float(
            np.max(np.abs(log_gamma_ref - log_gamma_ref_profile))
        )
        oracle_within_center = abs(float(log_gamma_ref.mean()))
        state["oracle_profile_gamma_max"] = max(
            state["oracle_profile_gamma_max"], oracle_gamma_construction
        )
        state["oracle_within_center_max"] = max(
            state["oracle_within_center_max"], oracle_within_center
        )
        state["weighted_log_gamma_oracle"] += pi[dataset] * log_gamma_ref

        learned_gamma_mean = float(gamma_eval.mean())
        learned_covariance = covariance(l_hat, gamma_eval)
        m_learned = float(np.mean(l_hat * gamma_eval))
        oracle_gamma_mean = float(gamma_ref.mean())
        oracle_covariance = covariance(l_ref, gamma_ref)
        m_oracle = float(np.mean(l_ref * gamma_ref))
        learned_decomposition_error = abs(
            m_learned - learned_gamma_mean - learned_covariance
        )
        oracle_decomposition_error = abs(
            m_oracle - oracle_gamma_mean - oracle_covariance
        )
        state["weighted_mass_learned"] += pi[dataset] * m_learned
        state["weighted_mass_oracle"] += pi[dataset] * m_oracle
        state["weighted_log_mass_sq_learned"] += pi[dataset] * math.log(m_learned) ** 2
        state["weighted_log_mass_sq_oracle"] += pi[dataset] * math.log(m_oracle) ** 2

        target = np.asarray(row["target"], dtype=np.float64)
        mu = np.asarray(row["mu"], dtype=np.float64)
        likelihood_mu = np.asarray(row["likelihood_positive_mean"], dtype=np.float64)
        normalized_shape = np.asarray(row["normalized_shape"], dtype=np.float64)
        log_sigma = np.asarray(row["log_sigma"], dtype=np.float64)
        if any(array.shape != l_full.shape for array in (target, mu, likelihood_mu, normalized_shape, log_sigma, gamma_full)):
            raise ValueError(f"{run.run_id}/{transcript_id}/{dataset}: prediction shape mismatch")
        scale = float(row["scale_dt"])
        m_learned_full_raw = float(np.mean(l_full[model_mask] * gamma_full[model_mask]))
        m_learned_eval_raw = float(
            np.mean(l_full[evaluation_mask] * gamma_full[evaluation_mask])
        )
        shape_full_mean = float(normalized_shape[model_mask].mean())
        decoder_product_error = float(
            np.max(
                np.abs(
                    mu[model_mask]
                    - scale * l_full[model_mask] * gamma_full[model_mask]
                )
            )
        )
        decoder_product_scaled_error = decoder_product_error / max(
            1.0, float(np.max(np.abs(mu[model_mask])))
        )
        decoder_shape_error = float(
            np.max(
                np.abs(
                    mu[model_mask]
                    - scale * normalized_shape[model_mask]
                )
            )
        )
        normalized_shape_error = float(
            np.max(
                np.abs(
                    normalized_shape[model_mask]
                    - l_full[model_mask] * gamma_full[model_mask]
                )
            )
        )
        target_sum_full = float(target[model_mask].sum())
        aggregate_ratio_full = (
            float(mu[model_mask].sum() / target_sum_full)
            if target_sum_full > 0.0
            else float("nan")
        )
        target_sum_eval = float(target[evaluation_mask].sum())
        aggregate_ratio_eval = (
            float(mu[evaluation_mask].sum() / target_sum_eval)
            if target_sum_eval > 0.0
            else float("nan")
        )
        scale_raw = float(target[model_mask].mean())
        scale_floor_active = scale_raw < settings.likelihood_mean_floor
        likelihood_floor_active = np.abs(likelihood_mu - mu) > 1.0e-12

        key = observation_key(run, dataset)
        observation = observations[key][transcript_id]
        if observation.consensus.shape != target.shape:
            raise ValueError(f"{key}/{transcript_id}: stored observation shape mismatch")
        replica_mean = np.mean(np.stack(observation.replicas, axis=0), axis=0)
        target_replica_mean_error = float(
            np.max(np.abs(target - replica_mean))
        )
        stored_ribo_replica_mean_error = float(
            np.max(np.abs(observation.consensus - replica_mean))
        )
        alpha_eval = np.exp(log_sigma[evaluation_mask])
        replicate_full_identity_errors: list[float] = []
        replicate_eval_identity_errors: list[float] = []
        replicate_nll: list[float] = []
        replicate_floor_fractions: list[float] = []
        replicate_scale_floor: list[bool] = []
        for replica in observation.replicas:
            raw_scale = float(replica[model_mask].mean())
            replica_scale = max(raw_scale, settings.likelihood_mean_floor)
            raw_mu_replica = replica_scale * normalized_shape
            positive_mu_replica = np.maximum(
                raw_mu_replica, settings.likelihood_mean_floor
            )
            denominator_full = float(replica[model_mask].sum())
            if denominator_full > 0.0:
                ratio_full = float(
                    raw_mu_replica[model_mask].sum() / denominator_full
                )
                replicate_full_identity_errors.append(abs(ratio_full - shape_full_mean))
            denominator_eval = float(replica[evaluation_mask].sum())
            if denominator_eval > 0.0:
                ratio_eval = float(
                    raw_mu_replica[evaluation_mask].sum() / denominator_eval
                )
                replicate_eval_identity_errors.append(abs(ratio_eval - m_learned))
            replicate_nll.append(
                nb2_nll(
                    replica[evaluation_mask],
                    positive_mu_replica[evaluation_mask],
                    alpha_eval,
                )
            )
            replicate_floor_fractions.append(
                float(np.mean(raw_mu_replica[evaluation_mask] < settings.likelihood_mean_floor))
            )
            replicate_scale_floor.append(raw_scale < settings.likelihood_mean_floor)

        raw_bound = settings.raw_log_gamma_bound
        learned_raw_bound_fraction = float(
            np.mean(
                np.abs(log_gamma_raw_full[evaluation_mask])
                >= raw_bound - tolerances["raw_bound_proximity"]
            )
        )
        oracle_raw_bound_fraction = float(np.mean(np.abs(log_bias) > raw_bound))
        learned_correction_magnitude = float(
            np.sqrt(np.mean(np.square(log_gamma_eval)))
        )
        oracle_correction_magnitude = float(
            np.sqrt(np.mean(np.square(log_gamma_ref)))
        )
        row_index = len(mass_rows)
        state["row_indices"].append(row_index)
        mass_rows.append(
            {
                "run_id": run.run_id,
                "family": run.family,
                "panel_depth": run.depth,
                "dataset_depth": dataset_depth(run, dataset),
                "n_datasets": run.n_datasets,
                "n_bias_families": run.n_bias_families,
                "reference_weighting": run.reference_weighting,
                "training_seed": run.training_seed,
                "panel_id": panel_id,
                "transcript_id": transcript_id,
                "dataset": dataset,
                "bias_family": reference.base_bias_name(dataset),
                "bias_order_position": dataset_order[dataset],
                "pi": float(pi[dataset]),
                "observation_key": key,
                "simulator_bias_seed": 20260807,
                "simulator_observation_seed": 20260808,
                "sense_length": int(q_full.size),
                "evaluated_positions": int(evaluation_mask.sum()),
                "boundary_trim_codons": trim,
                "m_learned": m_learned,
                "log_m_learned": math.log(m_learned),
                "m_oracle": m_oracle,
                "log_m_oracle": math.log(m_oracle),
                "log_mass_error": math.log(m_learned) - math.log(m_oracle),
                "abs_log_mass_error": abs(math.log(m_learned) - math.log(m_oracle)),
                "learned_gamma_mean": learned_gamma_mean,
                "learned_covariance": learned_covariance,
                "oracle_gamma_mean": oracle_gamma_mean,
                "oracle_covariance": oracle_covariance,
                "learned_decomposition_error": learned_decomposition_error,
                "oracle_decomposition_error": oracle_decomposition_error,
                "learned_correction_magnitude": learned_correction_magnitude,
                "oracle_correction_magnitude": oracle_correction_magnitude,
                "sigma_log_G": float(state["oracle"]["sigma_log_G"]),
                "learned_log_gamma_mean_eval": float(log_gamma_eval.mean()),
                "learned_log_gamma_mean_full_model_mask": float(
                    log_gamma_full[model_mask].mean()
                ),
                "oracle_log_gamma_mean": float(log_gamma_ref.mean()),
                "oracle_gamma_profile_construction_max_abs": oracle_gamma_construction,
                "raw_log_gamma_bound": raw_bound,
                "learned_raw_bound_fraction": learned_raw_bound_fraction,
                "oracle_raw_candidate_bound_exceeded_fraction": oracle_raw_bound_fraction,
                "oracle_final_log_gamma_max_abs": float(np.max(np.abs(log_gamma_ref))),
                "learned_final_log_gamma_max_abs": float(np.max(np.abs(log_gamma_eval))),
                "m_learned_full_model_mask_raw_L": m_learned_full_raw,
                "m_learned_reported_mask_raw_L": m_learned_eval_raw,
                "L_raw_mean_reported_mask": float(state["l_eval_raw"].mean()),
                "normalized_shape_full_mean": shape_full_mean,
                "aggregate_ratio_full_model_mask": aggregate_ratio_full,
                "aggregate_ratio_reported_mask": aggregate_ratio_eval,
                "aggregate_ratio_full_minus_shape_mean": aggregate_ratio_full - shape_full_mean,
                "aggregate_ratio_reported_minus_primary_m": aggregate_ratio_eval - m_learned,
                "decoder_product_max_abs": decoder_product_error,
                "decoder_product_scaled_max_abs": decoder_product_scaled_error,
                "decoder_shape_max_abs": decoder_shape_error,
                "normalized_shape_product_max_abs": normalized_shape_error,
                "scale_dt": scale,
                "target_full_model_mean": scale_raw,
                "scale_floor_active": scale_floor_active,
                "likelihood_floor_active_fraction": float(
                    likelihood_floor_active[evaluation_mask].mean()
                ),
                "target_replica_mean_max_abs": target_replica_mean_error,
                "stored_ribo_replica_mean_max_abs": stored_ribo_replica_mean_error,
                "replicate_count": len(observation.replicas),
                "replicate_full_identity_max_abs": max(
                    replicate_full_identity_errors, default=float("nan")
                ),
                "replicate_reported_identity_max_abs": max(
                    replicate_eval_identity_errors, default=float("nan")
                ),
                "replicate_likelihood_floor_active_fraction": max(
                    replicate_floor_fractions, default=float("nan")
                ),
                "replicate_scale_floor_active": any(replicate_scale_floor),
                "heldout_replica_nb_nll": float(np.mean(replicate_nll)),
                "learned_cross_center_max_abs_eval": float("nan"),
                "learned_cross_center_max_abs_full": float("nan"),
                "oracle_cross_center_max_abs": float("nan"),
                "weighted_mass_gap_learned": float("nan"),
                "weighted_mass_gap_oracle": float("nan"),
            }
        )
        state["seen_datasets"].add(dataset)

        maxima["learned_decomposition"] = max(
            maxima["learned_decomposition"], learned_decomposition_error
        )
        maxima["oracle_decomposition"] = max(
            maxima["oracle_decomposition"], oracle_decomposition_error
        )
        maxima["decoder_product"] = max(maxima["decoder_product"], decoder_product_error)
        maxima["decoder_product_scaled"] = max(
            maxima["decoder_product_scaled"], decoder_product_scaled_error
        )
        maxima["decoder_shape"] = max(
            maxima["decoder_shape"], decoder_shape_error
        )
        maxima["normalized_shape_product"] = max(
            maxima["normalized_shape_product"], normalized_shape_error
        )
        maxima["target_replica_mean"] = max(
            maxima["target_replica_mean"], target_replica_mean_error
        )
        maxima["stored_ribo_replica_mean"] = max(
            maxima["stored_ribo_replica_mean"], stored_ribo_replica_mean_error
        )
        if np.isfinite(aggregate_ratio_full):
            maxima["aggregate_ratio_full"] = max(
                maxima["aggregate_ratio_full"], abs(aggregate_ratio_full - shape_full_mean)
            )
        if replicate_full_identity_errors:
            maxima["replicate_ratio_full"] = max(
                maxima["replicate_ratio_full"], max(replicate_full_identity_errors)
            )

    missing = cohort - set(states)
    if missing:
        raise KeyError(f"{run.run_id}: missing predictions for {sorted(missing)[:5]}")

    transcript_rows: list[dict[str, Any]] = []
    for transcript_id in sorted(states):
        state = states[transcript_id]
        if state["seen_datasets"] != set(run.datasets):
            missing_datasets = set(run.datasets) - state["seen_datasets"]
            raise ValueError(
                f"{run.run_id}/{transcript_id}: missing {sorted(missing_datasets)}"
            )
        learned_cross_eval = float(
            np.max(np.abs(state["weighted_log_gamma_learned_eval"]))
        )
        learned_cross_full = float(
            np.max(np.abs(state["weighted_log_gamma_learned_full"]))
        )
        oracle_cross = float(np.max(np.abs(state["weighted_log_gamma_oracle"])))
        learned_gap = float(state["weighted_mass_learned"] - 1.0)
        oracle_gap = float(state["weighted_mass_oracle"] - 1.0)
        learned_severity = math.sqrt(state["weighted_log_mass_sq_learned"])
        oracle_severity = math.sqrt(state["weighted_log_mass_sq_oracle"])
        for row_index in state["row_indices"]:
            mass_rows[row_index]["learned_cross_center_max_abs_eval"] = learned_cross_eval
            mass_rows[row_index]["learned_cross_center_max_abs_full"] = learned_cross_full
            mass_rows[row_index]["oracle_cross_center_max_abs"] = oracle_cross
            mass_rows[row_index]["weighted_mass_gap_learned"] = learned_gap
            mass_rows[row_index]["weighted_mass_gap_oracle"] = oracle_gap
        profile_l_ref = state["profile_l_ref"]
        profile_ref_q = state["profile_ref_q"]
        profile_l_q = state["profile_l_q"]
        transcript_rows.append(
            {
                "run_id": run.run_id,
                "family": run.family,
                "panel_depth": run.depth,
                "n_datasets": run.n_datasets,
                "n_bias_families": run.n_bias_families,
                "reference_weighting": run.reference_weighting,
                "training_seed": run.training_seed,
                "panel_id": panel_id,
                "transcript_id": transcript_id,
                "sense_length": int(len(state["q_full"])),
                "evaluated_positions": int(state["evaluation_mask"].sum()),
                "sigma_log_G": float(state["oracle"]["sigma_log_G"]),
                "weighted_mass_learned": float(state["weighted_mass_learned"]),
                "weighted_mass_oracle": float(state["weighted_mass_oracle"]),
                "weighted_mass_gap_learned": learned_gap,
                "weighted_mass_gap_oracle": oracle_gap,
                "mass_severity_learned": learned_severity,
                "mass_severity_oracle": oracle_severity,
                "L_vs_Lref_pearson": profile_l_ref["pearson"],
                "L_vs_Lref_spearman": profile_l_ref["spearman"],
                "L_vs_Lref_clr_rmse": profile_l_ref["clr_rmse"],
                "Lref_vs_Q_pearson": profile_ref_q["pearson"],
                "Lref_vs_Q_spearman": profile_ref_q["spearman"],
                "Lref_vs_Q_clr_rmse": profile_ref_q["clr_rmse"],
                "L_vs_Q_pearson": profile_l_q["pearson"],
                "L_vs_Q_spearman": profile_l_q["spearman"],
                "L_vs_Q_clr_rmse": profile_l_q["clr_rmse"],
                "learned_cross_center_max_abs_eval": learned_cross_eval,
                "learned_cross_center_max_abs_full": learned_cross_full,
                "oracle_cross_center_max_abs": oracle_cross,
                "learned_within_center_max_abs_eval": float(
                    state["within_center_eval_max"]
                ),
                "learned_within_center_max_abs_full": float(
                    state["within_center_full_max"]
                ),
                "oracle_within_center_max_abs": float(
                    state["oracle_within_center_max"]
                ),
                "oracle_L_construction_max_abs": float(
                    state["oracle"]["L_ref_construction_max_abs"]
                ),
                "oracle_gamma_construction_max_abs": float(
                    state["oracle_profile_gamma_max"]
                ),
            }
        )
        maxima["learned_cross_center_eval"] = max(
            maxima["learned_cross_center_eval"], learned_cross_eval
        )
        maxima["learned_cross_center_full"] = max(
            maxima["learned_cross_center_full"], learned_cross_full
        )
        maxima["oracle_cross_center"] = max(
            maxima["oracle_cross_center"], oracle_cross
        )
        maxima["learned_within_center_eval"] = max(
            maxima["learned_within_center_eval"], state["within_center_eval_max"]
        )
        maxima["learned_within_center_full"] = max(
            maxima["learned_within_center_full"], state["within_center_full_max"]
        )
        maxima["oracle_within_center"] = max(
            maxima["oracle_within_center"], state["oracle_within_center_max"]
        )
        maxima["oracle_gamma_construction"] = max(
            maxima["oracle_gamma_construction"], state["oracle_profile_gamma_max"]
        )
        maxima["learned_mass_inequality_violation"] = max(
            maxima["learned_mass_inequality_violation"], max(0.0, -learned_gap)
        )
        maxima["oracle_mass_inequality_violation"] = max(
            maxima["oracle_mass_inequality_violation"], max(0.0, -oracle_gap)
        )

    return pd.DataFrame(mass_rows), pd.DataFrame(transcript_rows), dict(maxima)


def clustered_calibration_bootstrap(
    frame: pd.DataFrame, replicates: int, seed: int
) -> dict[str, tuple[float, float]]:
    ordered = frame.sort_values(["transcript_id", "dataset"], kind="mergesort")
    transcript_ids = sorted(ordered["transcript_id"].unique())
    blocks_x: list[np.ndarray] = []
    blocks_y: list[np.ndarray] = []
    for transcript_id in transcript_ids:
        part = ordered.loc[ordered["transcript_id"] == transcript_id]
        blocks_x.append(part["log_m_oracle"].to_numpy(dtype=np.float64))
        blocks_y.append(part["log_m_learned"].to_numpy(dtype=np.float64))
    sizes = {len(values) for values in blocks_x}
    if len(sizes) != 1:
        raise ValueError("Clustered calibration requires a complete rectangular panel")
    x_matrix = np.stack(blocks_x)
    y_matrix = np.stack(blocks_y)
    rng = np.random.default_rng(seed)
    estimates: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        draw = rng.integers(0, len(transcript_ids), size=len(transcript_ids))
        values = calibration_metrics(x_matrix[draw].reshape(-1), y_matrix[draw].reshape(-1))
        for name, value in values.items():
            estimates[name].append(value)
    return {
        name: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for name, values in estimates.items()
    }


def finite_spearman(left: pd.Series, right: pd.Series) -> float:
    x = left.to_numpy(dtype=np.float64)
    y = right.to_numpy(dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3 or np.std(x[valid]) == 0.0 or np.std(y[valid]) == 0.0:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def build_summary(
    mass: pd.DataFrame,
    transcripts: pd.DataFrame,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    condition_columns = [
        "run_id", "family", "panel_depth", "n_datasets", "n_bias_families",
        "reference_weighting", "training_seed", "panel_id",
    ]
    transcript_by_run = {
        run_id: group for run_id, group in transcripts.groupby("run_id", sort=False)
    }
    for run_index, (keys, group) in enumerate(
        mass.groupby(condition_columns, sort=False)
    ):
        condition = dict(zip(condition_columns, keys, strict=True))
        transcript_group = transcript_by_run[str(condition["run_id"])]
        calibration = calibration_metrics(
            group["log_m_oracle"].to_numpy(float),
            group["log_m_learned"].to_numpy(float),
        )
        intervals = clustered_calibration_bootstrap(
            group,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + run_index,
        )
        log_oracle = group["log_m_oracle"].to_numpy(float)
        log_error = group["log_mass_error"].to_numpy(float)
        row: dict[str, Any] = {
            **condition,
            "summary_level": "run",
            "dataset": "__all__",
            "dataset_depth": condition["panel_depth"],
            "bias_family": "__all__",
            "n_transcripts": int(group["transcript_id"].nunique()),
            "n_transcript_dataset_rows": int(len(group)),
            "oracle_log_mass_median": float(np.median(log_oracle)),
            "oracle_log_mass_q25": float(np.quantile(log_oracle, 0.25)),
            "oracle_log_mass_q75": float(np.quantile(log_oracle, 0.75)),
            "oracle_mass_fraction_above_one": float(np.mean(group["m_oracle"] > 1.0)),
            "oracle_mass_fraction_below_one": float(np.mean(group["m_oracle"] < 1.0)),
            "oracle_weighted_gap_median": float(
                transcript_group["weighted_mass_gap_oracle"].median()
            ),
            "oracle_weighted_gap_q25": float(
                transcript_group["weighted_mass_gap_oracle"].quantile(0.25)
            ),
            "oracle_weighted_gap_q75": float(
                transcript_group["weighted_mass_gap_oracle"].quantile(0.75)
            ),
            "learned_weighted_gap_median": float(
                transcript_group["weighted_mass_gap_learned"].median()
            ),
            "mass_severity_oracle_median": float(
                transcript_group["mass_severity_oracle"].median()
            ),
            "mass_severity_learned_median": float(
                transcript_group["mass_severity_learned"].median()
            ),
            "L_vs_Lref_pearson_median": float(
                transcript_group["L_vs_Lref_pearson"].median()
            ),
            "L_vs_Lref_clr_rmse_median": float(
                transcript_group["L_vs_Lref_clr_rmse"].median()
            ),
            "Lref_vs_Q_pearson_median": float(
                transcript_group["Lref_vs_Q_pearson"].median()
            ),
            "L_vs_Q_pearson_median": float(
                transcript_group["L_vs_Q_pearson"].median()
            ),
            "log_mass_error_q25": float(np.quantile(log_error, 0.25)),
            "log_mass_error_q75": float(np.quantile(log_error, 0.75)),
            "rho_oracle_abs_log_mass_vs_correction_magnitude": finite_spearman(
                group["log_m_oracle"].abs(), group["oracle_correction_magnitude"]
            ),
            "rho_oracle_weighted_gap_vs_sigma_log_G": finite_spearman(
                transcript_group["weighted_mass_gap_oracle"],
                transcript_group["sigma_log_G"],
            ),
            "rho_oracle_severity_vs_sigma_log_G": finite_spearman(
                transcript_group["mass_severity_oracle"],
                transcript_group["sigma_log_G"],
            ),
            "rho_abs_mass_error_vs_correction_magnitude": finite_spearman(
                group["abs_log_mass_error"], group["oracle_correction_magnitude"]
            ),
            "rho_abs_mass_error_vs_transcript_length": finite_spearman(
                group["abs_log_mass_error"], group["sense_length"]
            ),
            "rho_abs_mass_error_vs_heldout_nll": finite_spearman(
                group["abs_log_mass_error"], group["heldout_replica_nb_nll"]
            ),
            "rho_abs_mass_error_vs_learned_bound_fraction": finite_spearman(
                group["abs_log_mass_error"], group["learned_raw_bound_fraction"]
            ),
            "rho_clr_error_vs_learned_mass_severity": finite_spearman(
                transcript_group["L_vs_Lref_clr_rmse"],
                transcript_group["mass_severity_learned"],
            ),
            "rho_clr_error_vs_oracle_mass_severity": finite_spearman(
                transcript_group["L_vs_Lref_clr_rmse"],
                transcript_group["mass_severity_oracle"],
            ),
            "learned_bound_active_row_fraction": float(
                np.mean(group["learned_raw_bound_fraction"] > 0.0)
            ),
            "oracle_bound_infeasible_row_fraction": float(
                np.mean(group["oracle_raw_candidate_bound_exceeded_fraction"] > 0.0)
            ),
            "likelihood_floor_active_row_fraction": float(
                np.mean(group["likelihood_floor_active_fraction"] > 0.0)
            ),
            "replicate_floor_active_row_fraction": float(
                np.mean(group["replicate_likelihood_floor_active_fraction"] > 0.0)
            ),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_cluster": "transcript",
        }
        row.update(calibration)
        for name, (low, high) in intervals.items():
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
        rows.append(row)

        for dataset, dataset_group in group.groupby("dataset", sort=False):
            values = dataset_group["log_m_oracle"].to_numpy(float)
            error = dataset_group["log_mass_error"].to_numpy(float)
            rows.append(
                {
                    **condition,
                    "summary_level": "dataset",
                    "dataset": dataset,
                    "dataset_depth": str(dataset_group["dataset_depth"].iloc[0]),
                    "bias_family": str(dataset_group["bias_family"].iloc[0]),
                    "n_transcripts": int(dataset_group["transcript_id"].nunique()),
                    "n_transcript_dataset_rows": int(len(dataset_group)),
                    "oracle_log_mass_median": float(np.median(values)),
                    "oracle_log_mass_q25": float(np.quantile(values, 0.25)),
                    "oracle_log_mass_q75": float(np.quantile(values, 0.75)),
                    "oracle_mass_fraction_above_one": float(
                        np.mean(dataset_group["m_oracle"] > 1.0)
                    ),
                    "oracle_mass_fraction_below_one": float(
                        np.mean(dataset_group["m_oracle"] < 1.0)
                    ),
                    "mass_pearson": float(
                        np.corrcoef(
                            dataset_group["log_m_oracle"],
                            dataset_group["log_m_learned"],
                        )[0, 1]
                    ),
                    "mass_spearman": finite_spearman(
                        dataset_group["log_m_oracle"],
                        dataset_group["log_m_learned"],
                    ),
                    "mass_mae": float(np.mean(np.abs(error))),
                    "mass_rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "log_mass_error_median": float(np.median(error)),
                    "log_mass_error_q25": float(np.quantile(error, 0.25)),
                    "log_mass_error_q75": float(np.quantile(error, 0.75)),
                }
            )
    return pd.DataFrame(rows)


def build_depth_summary(
    mass: pd.DataFrame,
    run_summary: pd.DataFrame,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Summarize calibration by sequencing depth without treating rows as independent."""
    condition_columns = [
        "run_id", "family", "panel_depth", "n_datasets", "n_bias_families",
        "reference_weighting", "training_seed", "panel_id", "dataset_depth",
    ]
    rows: list[dict[str, Any]] = []
    for group_index, (keys, group) in enumerate(
        mass.groupby(condition_columns, sort=False)
    ):
        condition = dict(zip(condition_columns, keys, strict=True))
        calibration = calibration_metrics(
            group["log_m_oracle"].to_numpy(float),
            group["log_m_learned"].to_numpy(float),
        )
        matching_run = run_summary.loc[
            run_summary["run_id"] == condition["run_id"]
        ]
        if str(condition["panel_depth"]) != "cross_depth":
            intervals = {
                name: (
                    float(matching_run[f"{name}_ci_low"].iloc[0]),
                    float(matching_run[f"{name}_ci_high"].iloc[0]),
                )
                for name in calibration
            }
        else:
            intervals = clustered_calibration_bootstrap(
                group,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 100_000 + group_index,
            )
        per_transcript = []
        for transcript_id, transcript_group in group.groupby(
            "transcript_id", sort=False
        ):
            weights = transcript_group["pi"].to_numpy(float)
            weights = weights / weights.sum()
            oracle_log_mass = transcript_group["log_m_oracle"].to_numpy(float)
            learned_log_mass = transcript_group["log_m_learned"].to_numpy(float)
            per_transcript.append(
                {
                    "transcript_id": transcript_id,
                    "oracle_weighted_gap": float(
                        np.dot(weights, transcript_group["m_oracle"].to_numpy(float))
                        - 1.0
                    ),
                    "learned_weighted_gap": float(
                        np.dot(weights, transcript_group["m_learned"].to_numpy(float))
                        - 1.0
                    ),
                    "oracle_severity": float(
                        np.sqrt(np.dot(weights, np.square(oracle_log_mass)))
                    ),
                    "learned_severity": float(
                        np.sqrt(np.dot(weights, np.square(learned_log_mass)))
                    ),
                }
            )
        transcript_frame = pd.DataFrame(per_transcript)
        pi_by_dataset = group[["dataset", "pi"]].drop_duplicates()
        row: dict[str, Any] = {
            **condition,
            "n_transcripts": int(group["transcript_id"].nunique()),
            "n_datasets_at_depth": int(group["dataset"].nunique()),
            "n_transcript_dataset_rows": int(len(group)),
            "reference_weight_share_at_depth": float(pi_by_dataset["pi"].sum()),
            "oracle_log_mass_median": float(group["log_m_oracle"].median()),
            "oracle_log_mass_q25": float(group["log_m_oracle"].quantile(0.25)),
            "oracle_log_mass_q75": float(group["log_m_oracle"].quantile(0.75)),
            "oracle_mass_fraction_above_one": float(np.mean(group["m_oracle"] > 1.0)),
            "oracle_mass_fraction_below_one": float(np.mean(group["m_oracle"] < 1.0)),
            "oracle_weighted_gap_median": float(
                transcript_frame["oracle_weighted_gap"].median()
            ),
            "learned_weighted_gap_median": float(
                transcript_frame["learned_weighted_gap"].median()
            ),
            "oracle_mass_severity_median": float(
                transcript_frame["oracle_severity"].median()
            ),
            "learned_mass_severity_median": float(
                transcript_frame["learned_severity"].median()
            ),
            "learned_bound_active_row_fraction": float(
                np.mean(group["learned_raw_bound_fraction"] > 0.0)
            ),
            "oracle_bound_infeasible_row_fraction": float(
                np.mean(group["oracle_raw_candidate_bound_exceeded_fraction"] > 0.0)
            ),
            "likelihood_floor_active_row_fraction": float(
                np.mean(group["likelihood_floor_active_fraction"] > 0.0)
            ),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_cluster": "transcript",
        }
        row.update(calibration)
        for name, (low, high) in intervals.items():
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_robust_ols(
    frame: pd.DataFrame,
    predictor: str,
    family: str,
    adjusted: bool,
) -> dict[str, Any]:
    data = frame.loc[frame["family"] == family].copy()
    outcome = data["L_vs_Lref_clr_rmse"].to_numpy(dtype=np.float64)
    if adjusted:
        centered = data[predictor] - data.groupby("run_id")[predictor].transform("mean")
        scales = data.groupby("run_id")[predictor].transform("std").replace(0.0, np.nan)
        severity = (centered / scales).fillna(0.0).to_numpy(dtype=np.float64)
        log_length = np.log(data["sense_length"].to_numpy(dtype=np.float64))
        log_length = (log_length - log_length.mean()) / log_length.std(ddof=0)
        dummies = pd.get_dummies(data["run_id"], drop_first=True, dtype=float)
        design = np.column_stack(
            [np.ones(len(data)), severity, log_length, dummies.to_numpy(dtype=float)]
        )
        coefficient_names = ["intercept", "severity_within_run_sd", "log_length_sd"] + list(dummies.columns)
    else:
        severity = data[predictor].to_numpy(dtype=np.float64)
        severity = (severity - severity.mean()) / severity.std(ddof=0)
        design = np.column_stack([np.ones(len(data)), severity])
        coefficient_names = ["intercept", "severity_global_sd"]
    beta = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residual = outcome - design @ beta
    bread = np.linalg.pinv(design.T @ design)
    meat = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
    clusters = data["transcript_id"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        take = clusters == cluster
        score = design[take].T @ residual[take]
        meat += np.outer(score, score)
    n, p = design.shape
    g = len(unique_clusters)
    correction = (g / (g - 1.0)) * ((n - 1.0) / (n - p))
    covariance_matrix = correction * bread @ meat @ bread
    standard_error = np.sqrt(np.maximum(np.diag(covariance_matrix), 0.0))
    severity_index = 1
    coefficient = float(beta[severity_index])
    se = float(standard_error[severity_index])
    return {
        "family": family,
        "predictor": predictor,
        "model": "run_fixed_effects_plus_log_length" if adjusted else "unadjusted",
        "outcome": "L_vs_Lref_clr_rmse",
        "effect_per_sd": coefficient,
        "cluster_robust_se": se,
        "ci_low": coefficient - 1.96 * se,
        "ci_high": coefficient + 1.96 * se,
        "n_rows": n,
        "n_transcript_clusters": g,
        "n_parameters": p,
        "controls": (
            "run fixed effects (therefore N, depth, panel, pi, seed) and log length"
            if adjusted
            else "none"
        ),
        "coefficient_name": coefficient_names[severity_index],
    }


def build_regressions(transcripts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in sorted(transcripts["family"].unique()):
        for predictor in ("mass_severity_learned", "mass_severity_oracle"):
            rows.append(cluster_robust_ols(transcripts, predictor, family, adjusted=False))
            rows.append(cluster_robust_ols(transcripts, predictor, family, adjusted=True))
    return pd.DataFrame(rows)


def build_sanity_table(
    maxima_by_run: dict[str, dict[str, float]],
    mass: pd.DataFrame,
    tolerances: dict[str, float],
) -> pd.DataFrame:
    maxima: defaultdict[str, float] = defaultdict(float)
    for values in maxima_by_run.values():
        for name, value in values.items():
            maxima[name] = max(maxima[name], float(value))
    reported_ratio_deviation = float(
        mass["aggregate_ratio_reported_minus_primary_m"].abs().max()
    )
    checks = [
        ("learned_L_mean_one_reported_mask", "PASS", maxima["learned_L_primary_mean_one"], tolerances["mean_one"], "L is explicitly normalized on the reported mask"),
        ("learned_L_mean_one_full_model_mask", "PASS", maxima["learned_L_full_model_mean_one"], tolerances["mean_one"], "saved model normalization domain"),
        ("learned_log_gamma_within_center_full_model_mask", "PASS", maxima["learned_within_center_full"], tolerances["within_log_gamma_center_full_model_mask"], "model gauge domain"),
        ("learned_log_gamma_within_center_reported_mask", "MASK_EFFECT", maxima["learned_within_center_eval"], tolerances["within_log_gamma_center_full_model_mask"], "boundary trimming changes the positional mean; gamma is not re-centered"),
        ("learned_cross_dataset_center_reported_mask", "PASS", maxima["learned_cross_center_eval"], tolerances["cross_dataset_log_gamma_center"], "pointwise constraint survives masking"),
        ("learned_cross_dataset_center_full_model_mask", "PASS", maxima["learned_cross_center_full"], tolerances["cross_dataset_log_gamma_center"], "model gauge domain"),
        ("oracle_within_profile_center", "PASS", maxima["oracle_within_center"], tolerances["oracle_centering"], "reported mask"),
        ("oracle_cross_dataset_center", "PASS", maxima["oracle_cross_center"], tolerances["oracle_centering"], "reported mask"),
        ("learned_weighted_mass_inequality", "PASS", maxima["learned_mass_inequality_violation"], tolerances["mass_inequality"], "negative part of weighted mass gap"),
        ("oracle_weighted_mass_inequality", "PASS", maxima["oracle_mass_inequality_violation"], tolerances["mass_inequality"], "negative part of weighted mass gap"),
        ("learned_mass_decomposition", "PASS", maxima["learned_decomposition"], tolerances["mass_decomposition"], "reported mask"),
        ("oracle_mass_decomposition", "PASS", maxima["oracle_decomposition"], tolerances["mass_decomposition"], "reported mask"),
        ("oracle_L_bias_vs_hstar_construction", "PASS", maxima["oracle_profile_construction"], tolerances["h_construction"], "independent algebraic constructions"),
        ("oracle_gamma_bias_vs_hstar_construction", "PASS", maxima["oracle_gamma_construction"], tolerances["h_construction"], "independent algebraic constructions"),
        ("stored_mu_equals_scale_times_L_gamma_absolute", "PASS", maxima["decoder_product"], tolerances["decoder_product"], "full saved model mask; tolerance allows float32 serialization/multiplication roundoff"),
        ("stored_mu_equals_scale_times_L_gamma_scaled", "PASS", maxima["decoder_product_scaled"], tolerances["decoder_product_scaled"], "maximum absolute error divided by max(1, maximum absolute stored mu) within each profile"),
        ("stored_mu_equals_scale_times_normalized_shape", "PASS", maxima["decoder_shape"], tolerances["decoder_product"], "independent saved decoder-shape check on the full model mask"),
        ("normalized_shape_equals_L_gamma", "PASS", maxima["normalized_shape_product"], tolerances["decoder_product"], "full saved model mask"),
        ("aggregate_sum_ratio_identity_full_model_mask", "PASS", maxima["aggregate_ratio_full"], tolerances["decoder_ratio_full_model_mask"], "S is the observed full-mask mean"),
        ("replica_sum_ratio_identity_full_model_mask", "PASS", maxima["replicate_ratio_full"], tolerances["decoder_ratio_full_model_mask"], "replica means reconstructed from frozen normalized shape"),
        ("aggregate_sum_ratio_identity_reported_mask", "MASK_EFFECT", reported_ratio_deviation, tolerances["decoder_ratio_full_model_mask"], "S and model normalization use the full mask; primary mass uses the trimmed reported mask"),
        ("prediction_target_matches_mean_of_raw_replicates", "PASS", maxima["target_replica_mean"], tolerances["decoder_product"], "the datamodule replaces the stored ribo column with the raw-replica arithmetic mean"),
        ("stored_ribo_column_matches_replica_mean", "INFORMATIONAL", maxima["stored_ribo_replica_mean"], float("nan"), "not required: training and prediction use the raw-replica mean, not the legacy stored ribo column"),
    ]
    rows = []
    for name, expected_status, deviation, tolerance, note in checks:
        if expected_status in {"MASK_EFFECT", "INFORMATIONAL"}:
            status = expected_status
        else:
            status = "PASS" if deviation <= tolerance else "FAIL"
        rows.append(
            {
                "check": name,
                "status": status,
                "maximum_deviation": deviation,
                "tolerance": tolerance,
                "note": note,
            }
        )
    rows.extend(
        [
            {
                "check": "oracle_raw_candidate_exceeds_model_bound",
                "status": "FLAG" if (mass["oracle_raw_candidate_bound_exceeded_fraction"] > 0).any() else "PASS",
                "maximum_deviation": float(mass["oracle_raw_candidate_bound_exceeded_fraction"].max()),
                "tolerance": 0.0,
                "note": "fraction of reported positions per row; log(b) is a sufficient raw-score realization",
            },
            {
                "check": "likelihood_mean_floor_active",
                "status": "FLAG" if (mass["likelihood_floor_active_fraction"] > 0).any() else "PASS",
                "maximum_deviation": float(mass["likelihood_floor_active_fraction"].max()),
                "tolerance": 0.0,
                "note": "reported separately and not treated as exact sum-ratio evidence",
            },
        ]
    )
    return pd.DataFrame(rows)


def publication_style() -> dict[str, Any]:
    style = publication_rc()
    style.update(
        {
            "font.size": 16.0,
            "font.weight": "bold",
            "axes.labelsize": 17.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 17.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "legend.fontsize": 13.5,
            "axes.linewidth": 1.2,
        }
    )
    if style.get("text.usetex"):
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}"
            r"\AtBeginDocument{\boldmath}"
        )
    return style


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    # ``dpi`` controls only rasterized artists in the vector PDF.  Keeping it
    # explicit preserves print-quality dense point layers without returning to
    # a slow, multi-megabyte collection of vector markers.
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_oracle_mass_distributions(
    mass: pd.DataFrame, output_dir: Path, dpi: int
) -> None:
    source = mass.loc[mass["family"] == "within_depth"]
    y = source["log_m_oracle"].to_numpy(float)
    y_limits = (float(np.quantile(y, 0.001)) - 0.02, float(np.quantile(y, 0.999)) + 0.02)
    colors = reference.DEPTH_COLORS
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), sharey=True)
        for axis, depth, label in zip(
            axes.flat[:3], reference.DEPTH_ORDER, ("A", "B", "C"), strict=True
        ):
            part = source.loc[source["panel_depth"] == depth]
            values = [
                part.loc[part["n_datasets"] == n, "log_m_oracle"].to_numpy(float)
                for n in range(2, 11)
            ]
            box = axis.boxplot(
                values,
                positions=np.arange(2, 11),
                widths=0.55,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "#222222", "linewidth": 1.2},
                whiskerprops={"color": "#555555"},
                capprops={"color": "#555555"},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(colors[depth])
                patch.set_alpha(0.62)
            axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
            axis.set_title(f"{label}  {reference.DEPTH_LABELS[depth]}", loc="left")
            axis.set_xlabel("Number of datasets $N$")
            axis.set_xticks(range(2, 11))
            axis.set_ylim(*y_limits)
            axis.grid(alpha=0.35)
        axis = axes.flat[3]
        n10 = source.loc[source["n_datasets"] == 10]
        offsets = {"0p25_per_codon": -0.20, "2_per_codon": 0.0, "20_per_codon": 0.20}
        for depth in reference.DEPTH_ORDER:
            part = n10.loc[n10["panel_depth"] == depth]
            grouped = part.groupby("bias_order_position")["log_m_oracle"]
            median = grouped.median()
            q25 = grouped.quantile(0.25)
            q75 = grouped.quantile(0.75)
            x = median.index.to_numpy(float) + offsets[depth]
            axis.errorbar(
                x,
                median,
                yerr=np.vstack([median - q25, q75 - median]),
                color=colors[depth],
                marker="o",
                linewidth=1.2,
                capsize=2.0,
                label=reference.DEPTH_LABELS[depth],
            )
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
        axis.set_title(r"D  Dataset-specific at $N=10$", loc="left")
        axis.set_xlabel("Bias order position")
        axis.set_xticks(range(1, 11))
        axis.set_ylim(*y_limits)
        axis.grid(alpha=0.35)
        axis.legend(loc="best")
        axes[0, 0].set_ylabel(r"Oracle log mass $\log m^{\rm ref}_{t,d}$")
        axes[1, 0].set_ylabel(r"Oracle log mass $\log m^{\rm ref}_{t,d}$")
        fig.subplots_adjust(left=0.08, right=0.995, bottom=0.08, top=0.96, hspace=0.28, wspace=0.10)
        save_figure(fig, output_dir / "oracle_mass_distributions", dpi)


def plot_learned_vs_oracle(
    mass: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    maximum_points: int,
) -> None:
    selections = [
        ("within_depth", "0p25_per_codon", 10, "equal", "A  0.25 reads/codon, $N=10$"),
        ("within_depth", "2_per_codon", 10, "equal", "B  2 reads/codon, $N=10$"),
        ("within_depth", "20_per_codon", 10, "equal", "C  20 reads/codon, $N=10$"),
        ("cross_depth", "cross_depth", 30, "equal", "D  Mixed depth, $N=30$"),
    ]
    selected_frames = []
    for family, depth, n, weighting, _ in selections:
        selected_frames.append(
            mass.loc[
                (mass["family"] == family)
                & (mass["panel_depth"] == depth)
                & (mass["n_datasets"] == n)
                & (mass["reference_weighting"] == weighting)
            ]
        )
    all_values = np.concatenate(
        [
            frame[["log_m_oracle", "log_m_learned"]].to_numpy(float).ravel()
            for frame in selected_frames
        ]
    )
    limit = max(abs(float(np.quantile(all_values, 0.001))), abs(float(np.quantile(all_values, 0.999)))) + 0.03
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.2), sharex=True, sharey=True)
        for axis, selection, frame in zip(axes.flat, selections, selected_frames, strict=True):
            family, depth, n, weighting, title = selection
            if len(frame) > maximum_points:
                frame_plot = frame.sample(maximum_points, random_state=20260918)
            else:
                frame_plot = frame
            axis.scatter(
                frame_plot["log_m_oracle"],
                frame_plot["log_m_learned"],
                s=9,
                alpha=0.28,
                color="#4C78A8",
                edgecolors="none",
                rasterized=True,
            )
            condition = summary.loc[
                (summary["summary_level"] == "run")
                & (summary["family"] == family)
                & (summary["panel_depth"] == depth)
                & (summary["n_datasets"] == n)
                & (summary["reference_weighting"] == weighting)
            ].iloc[0]
            x_line = np.array([-limit, limit])
            axis.plot(x_line, x_line, color="#333333", linestyle="--", linewidth=1.0)
            axis.plot(
                x_line,
                condition["calibration_intercept"]
                + condition["calibration_slope"] * x_line,
                color="#D55E00",
                linewidth=1.8,
            )
            axis.text(
                0.04,
                0.96,
                rf"$r={condition['mass_pearson']:.3f}$; "
                rf"$\beta={condition['calibration_slope']:.2f}$ "
                rf"$[{condition['calibration_slope_ci_low']:.2f},"
                rf"{condition['calibration_slope_ci_high']:.2f}]$",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=13.0,
            )
            axis.set_title(title, loc="left")
            axis.set_xlim(-limit, limit)
            axis.set_ylim(-limit, limit)
            axis.grid(alpha=0.35)
        for axis in axes[-1]:
            axis.set_xlabel(r"Oracle log mass $\log m^{\rm ref}_{t,d}$")
        for axis in axes[:, 0]:
            axis.set_ylabel(r"Learned log mass $\log \widehat m_{t,d}$")
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.97, hspace=0.18, wspace=0.10)
        save_figure(fig, output_dir / "learned_vs_oracle_mass", dpi)


def plot_severity_vs_error(
    transcripts: pd.DataFrame,
    regressions: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    maximum_points: int,
) -> None:
    # The inferential quantity is the adjusted within-run effect, not the raw
    # cloud of repeated transcript/run observations.  A coefficient plot is
    # both more faithful to the reported analysis and dramatically lighter to
    # render than four 100,000-point vector scatter plots.
    selected = regressions.loc[
        regressions["model"] == "run_fixed_effects_plus_log_length"
    ].copy()
    order = [
        ("within_depth", "mass_severity_oracle", "Within-depth: oracle mass"),
        ("within_depth", "mass_severity_learned", "Within-depth: learned mass"),
        ("cross_depth", "mass_severity_oracle", "Mixed-depth: oracle mass"),
        ("cross_depth", "mass_severity_learned", "Mixed-depth: learned mass"),
    ]
    rows = []
    for family, predictor, label in order:
        match = selected.loc[
            (selected["family"] == family) & (selected["predictor"] == predictor)
        ]
        if len(match) != 1:
            raise ValueError(
                f"Expected one adjusted mass-severity coefficient for {family}/{predictor}"
            )
        rows.append((match.iloc[0], label, predictor.endswith("oracle")))
    overall_median_error = float(np.median(transcripts["L_vs_Lref_clr_rmse"]))
    display_scale = 1_000.0
    with plt.rc_context(publication_style()):
        fig, axis = plt.subplots(figsize=(10.8, 5.4), constrained_layout=True)
        y = np.arange(len(rows), dtype=float)
        for index, (result, _, is_oracle) in enumerate(rows):
            effect = float(result["effect_per_sd"])
            low = float(result["ci_low"])
            high = float(result["ci_high"])
            effect_display = display_scale * effect
            low_display = display_scale * low
            high_display = display_scale * high
            color = "#0072B2" if is_oracle else "#D55E00"
            marker = "o" if is_oracle else "s"
            axis.errorbar(
                effect_display,
                y[index],
                xerr=np.array(
                    [[effect_display - low_display], [high_display - effect_display]]
                ),
                color=color,
                marker=marker,
                markersize=8.5,
                linewidth=2.2,
                capsize=4.0,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )
            axis.text(
                high_display + 0.08,
                y[index],
                rf"$\Delta={effect:.4f}$ [{low:.4f}, {high:.4f}]",
                ha="left",
                va="center",
                fontsize=14.0,
            )
        axis.axvline(0.0, color="#555555", linestyle="--", linewidth=1.1)
        axis.set_yticks(y, [label for _, label, _ in rows])
        axis.set_ylim(len(rows) - 0.55, -0.55)
        axis.set_xlabel(
            "Adjusted change in shared-profile CLR RMSE per within-run SD\n"
            r"of log-mass severity ($\times 10^{-3}$)"
        )
        axis.set_title(
            "Adjusted association of mass severity with profile-recovery error",
            loc="left",
        )
        axis.grid(axis="x", alpha=0.30)
        axis.text(
            0.99,
            0.04,
            rf"Overall median CLR RMSE $={overall_median_error:.3f}$",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=14.0,
        )
        save_figure(fig, output_dir / "mass_severity_vs_shared_profile_error", dpi)


def plot_weighted_mass_inequality(
    transcripts: pd.DataFrame, output_dir: Path, dpi: int
) -> None:
    source = transcripts.loc[transcripts["family"] == "within_depth"]
    values = source[["weighted_mass_gap_learned", "weighted_mass_gap_oracle"]].to_numpy(float)
    y_high = float(np.quantile(values, 0.995)) * 1.08
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), sharex=True, sharey=True)
        for axis, depth, label in zip(axes, reference.DEPTH_ORDER, "ABC", strict=True):
            part = source.loc[source["panel_depth"] == depth]
            for column, (field, color, offset, name) in enumerate(
                [
                    ("weighted_mass_gap_learned", "#4C78A8", -0.17, "Learned"),
                    ("weighted_mass_gap_oracle", "#E07B39", 0.17, "Oracle"),
                ]
            ):
                distributions = [
                    part.loc[part["n_datasets"] == n, field].to_numpy(float)
                    for n in range(2, 11)
                ]
                box = axis.boxplot(
                    distributions,
                    positions=np.arange(2, 11) + offset,
                    widths=0.28,
                    showfliers=False,
                    patch_artist=True,
                    medianprops={"color": "#222222", "linewidth": 1.1},
                )
                for patch in box["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.62)
            axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
            axis.set_title(f"{label}  {reference.DEPTH_LABELS[depth]}", loc="left")
            axis.set_xlabel("Number of datasets $N$")
            axis.set_xticks(range(2, 11), labels=[str(n) for n in range(2, 11)])
            axis.set_ylim(-0.002, y_high)
            axis.grid(alpha=0.35)
        axes[0].set_ylabel(r"Weighted mass gap $\sum_d\pi_d m_{t,d}-1$")
        axes[0].legend(
            handles=[
                Line2D([0], [0], color="#4C78A8", linewidth=6, label="Learned"),
                Line2D([0], [0], color="#E07B39", linewidth=6, label="Oracle"),
            ],
            loc="upper right",
        )
        fig.subplots_adjust(left=0.08, right=0.995, bottom=0.16, top=0.92, wspace=0.08)
        save_figure(fig, output_dir / "weighted_mass_inequality", dpi)


def plot_mass_decomposition(mass: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    source = mass.loc[mass["family"] == "within_depth"]
    values = np.concatenate(
        [
            source["learned_gamma_mean"].to_numpy(float) - 1.0,
            source["learned_covariance"].to_numpy(float),
            source["oracle_gamma_mean"].to_numpy(float) - 1.0,
            source["oracle_covariance"].to_numpy(float),
        ]
    )
    limit = max(abs(float(np.quantile(values, 0.005))), abs(float(np.quantile(values, 0.995)))) * 1.12
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(3, 2, figsize=(11.8, 10.0), sharex=True, sharey=True)
        for row, depth in enumerate(reference.DEPTH_ORDER):
            part = source.loc[source["panel_depth"] == depth]
            for column, mode in enumerate(("learned", "oracle")):
                axis = axes[row, column]
                grouped = part.groupby("n_datasets")
                gamma_component = grouped[f"{mode}_gamma_mean"].mean() - 1.0
                covariance_component = grouped[f"{mode}_covariance"].mean()
                total = gamma_component + covariance_component
                axis.plot(gamma_component.index, gamma_component, color="#4C78A8", marker="o", label=r"$\langle\gamma\rangle-1$")
                axis.plot(covariance_component.index, covariance_component, color="#E07B39", marker="s", linestyle="--", label=r"$\mathrm{Cov}(L,\gamma)$")
                axis.plot(total.index, total, color="#333333", linestyle=":", linewidth=1.6, label=r"$m-1$")
                axis.axhline(0.0, color="#777777", linewidth=0.8)
                mode_label = "Learned" if mode == "learned" else "Oracle"
                axis.set_title(
                    f"{chr(65 + row * 2 + column)}  {reference.DEPTH_LABELS[depth]}: {mode_label}",
                    loc="left",
                )
                axis.set_ylim(-limit, limit)
                axis.set_xticks(range(2, 11))
                axis.grid(alpha=0.35)
        axes[0, 0].legend(loc="best")
        for axis in axes[-1]:
            axis.set_xlabel("Number of datasets $N$")
        for axis in axes[:, 0]:
            axis.set_ylabel("Mean contribution")
        fig.subplots_adjust(left=0.08, right=0.995, bottom=0.07, top=0.97, hspace=0.24, wspace=0.10)
        save_figure(fig, output_dir / "mass_decomposition", dpi)


def select_representatives(
    transcripts: pd.DataFrame,
    profile_cache: dict[tuple[str, str], dict[str, np.ndarray]],
    quantiles: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = transcripts.loc[
        (transcripts["family"] == "within_depth")
        & (transcripts["panel_depth"] == "20_per_codon")
        & (transcripts["n_datasets"] == 10)
        & (transcripts["reference_weighting"] == "equal")
    ].copy()
    if source["run_id"].nunique() != 1:
        raise ValueError("Representative selection expected one run")
    chosen = []
    for probability in quantiles:
        target = float(source["mass_severity_oracle"].quantile(probability))
        candidates = source.assign(
            distance=(source["mass_severity_oracle"] - target).abs()
        ).sort_values(["distance", "transcript_id"], kind="mergesort")
        candidates = candidates.loc[
            ~candidates["transcript_id"].isin([row["transcript_id"] for row in chosen])
        ]
        row = candidates.iloc[0].copy()
        row["selection_quantile"] = probability
        row["selection_target"] = target
        chosen.append(row)
    selected = pd.DataFrame(chosen)
    profile_rows = []
    for _, row in selected.iterrows():
        profile = profile_cache[(str(row["run_id"]), str(row["transcript_id"]))]
        for index, position in enumerate(profile["positions"]):
            profile_rows.append(
                {
                    "selection_quantile": float(row["selection_quantile"]),
                    "transcript_id": str(row["transcript_id"]),
                    "codon_position_0based": int(position),
                    "Q": float(profile["Q"][index]),
                    "L_ref": float(profile["L_ref"][index]),
                    "L_hat": float(profile["L_hat"][index]),
                }
            )
    return selected, pd.DataFrame(profile_rows)


def plot_representatives(profiles: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    quantiles = sorted(profiles["selection_quantile"].unique())
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(len(quantiles), 1, figsize=(12.6, 8.4))
        for axis, quantile in zip(np.atleast_1d(axes), quantiles, strict=True):
            part = profiles.loc[profiles["selection_quantile"] == quantile]
            transcript_id = str(part["transcript_id"].iloc[0])
            position = part["codon_position_0based"]
            axis.plot(position, part["Q"], color="#222222", linewidth=1.5, label="$Q$")
            axis.plot(position, part["L_ref"], color="#E69F00", linewidth=1.3, label=r"$L^{\rm ref}$")
            axis.plot(position, part["L_hat"], color="#0072B2", linewidth=1.2, label=r"$\widehat L$")
            title = f"{transcript_id}; oracle mass-severity quantile {quantile:.0%}".replace("%", r"\%")
            axis.set_title(title, loc="left")
            axis.set_ylabel("Mean-one profile")
            axis.grid(alpha=0.35)
        axes[0].legend(ncol=3, loc="upper right")
        axes[-1].set_xlabel("P-site codon position (0-based)")
        fig.subplots_adjust(left=0.08, right=0.995, bottom=0.07, top=0.97, hspace=0.34)
        save_figure(fig, output_dir / "representative_mass_profiles", dpi)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    depth_summary: pd.DataFrame,
    regressions: pd.DataFrame,
    sanity: pd.DataFrame,
    mass: pd.DataFrame,
    transcripts: pd.DataFrame,
    cohorts: dict[str, list[str]],
    cohort_exclusions: pd.DataFrame,
    *,
    checkpoint_variant: str,
    checkpoint_selection_metric: str,
) -> None:
    run_summary = summary.loc[summary["summary_level"] == "run"]
    primary = run_summary.loc[
        (run_summary["family"] == "within_depth")
        & (run_summary["n_datasets"].isin([2, 10]))
    ]
    endpoint_lines = []
    for depth in reference.DEPTH_ORDER:
        for n in (2, 10):
            row = primary.loc[
                (primary["panel_depth"] == depth) & (primary["n_datasets"] == n)
            ].iloc[0]
            endpoint_lines.append(
                f"| {reference.DEPTH_LABELS[depth]} | {n} | "
                f"{row['oracle_log_mass_median']:.4f} "
                f"[{row['oracle_log_mass_q25']:.4f}, {row['oracle_log_mass_q75']:.4f}] | "
                f"{row['oracle_weighted_gap_median']:.4f} | "
                f"{row['mass_pearson']:.3f} | {row['mass_mae']:.4f} | "
                f"{row['L_vs_Lref_pearson_median']:.3f} |"
            )
    regression_lines = []
    adjusted = regressions.loc[
        regressions["model"] == "run_fixed_effects_plus_log_length"
    ]
    for _, row in adjusted.iterrows():
        severity = "learned" if row["predictor"].endswith("learned") else "oracle"
        regression_lines.append(
            f"| {row['family']} | {severity} | {row['effect_per_sd']:.5f} | "
            f"[{row['ci_low']:.5f}, {row['ci_high']:.5f}] | "
            f"{int(row['n_transcript_clusters'])} |"
        )
    mixed_depth_lines = []
    mixed_depth = depth_summary.loc[
        (depth_summary["family"] == "cross_depth")
        & (depth_summary["n_datasets"] == 30)
    ]
    for weighting in ("equal", "quality_rank"):
        for depth in reference.DEPTH_ORDER:
            row = mixed_depth.loc[
                (mixed_depth["reference_weighting"] == weighting)
                & (mixed_depth["dataset_depth"] == depth)
            ].iloc[0]
            mixed_depth_lines.append(
                f"| {weighting} | {reference.DEPTH_LABELS[depth]} | "
                f"{row['reference_weight_share_at_depth']:.3f} | "
                f"{row['mass_pearson']:.3f} "
                f"[{row['mass_pearson_ci_low']:.3f}, {row['mass_pearson_ci_high']:.3f}] | "
                f"{row['mass_mae']:.4f} | {row['calibration_slope']:.3f} |"
            )
    fail_count = int((sanity["status"] == "FAIL").sum())
    mask_count = int((sanity["status"] == "MASK_EFFECT").sum())
    oracle_infeasible = float(
        np.mean(mass["oracle_raw_candidate_bound_exceeded_fraction"] > 0.0)
    )
    learned_bound = float(np.mean(mass["learned_raw_bound_fraction"] > 0.0))
    floor_rows = float(np.mean(mass["likelihood_floor_active_fraction"] > 0.0))
    median_profile_pcc = float(transcripts["L_vs_Lref_pearson"].median())
    median_profile_clr = float(transcripts["L_vs_Lref_clr_rmse"].median())
    median_reference_pcc = float(transcripts["Lref_vs_Q_pearson"].median())
    median_end_to_end_pcc = float(transcripts["L_vs_Q_pearson"].median())
    median_oracle_mass = float(mass["m_oracle"].median())
    median_learned_mass = float(mass["m_learned"].median())
    median_oracle_gap = float(transcripts["weighted_mass_gap_oracle"].median())
    median_learned_gap = float(transcripts["weighted_mass_gap_learned"].median())
    oracle_above = float(np.mean(mass["m_oracle"] > 1.0))
    oracle_below = float(np.mean(mass["m_oracle"] < 1.0))
    median_gamma_contribution = float((mass["oracle_gamma_mean"] - 1.0).median())
    median_covariance_contribution = float(mass["oracle_covariance"].median())
    largest_adjusted_effect = float(adjusted["effect_per_sd"].abs().max())
    largest_effect_fraction = largest_adjusted_effect / median_profile_clr
    significant_positive = adjusted.loc[adjusted["ci_low"] > 0.0]
    intervals_covering_zero = int(
        ((adjusted["ci_low"] <= 0.0) & (adjusted["ci_high"] >= 0.0)).sum()
    )
    if significant_positive.empty:
        severity_interval_text = (
            f"All {len(adjusted)} adjusted intervals include zero."
        )
    else:
        detected = ", ".join(
            f"{row.family}/{row.predictor.replace('mass_severity_', '')}"
            for row in significant_positive.itertuples()
        )
        severity_interval_text = (
            f"{intervals_covering_zero} of {len(adjusted)} adjusted intervals "
            f"include zero; positive intervals occur for {detected}."
        )
    if largest_effect_fraction < 0.01:
        recommendation = (
            "The mass restriction is not harmless for aggregate count calibration, because "
            "it produces an expected weighted mass excess near ten percent when the supplied "
            "scale is the observed positional mean. Within these frozen experiments, however, "
            "it appears largely harmless for recovery of the intended shared positional shape: "
            "the largest adjusted change is below one percent of the median CLR error per "
            "within-run severity SD. There is no evidence that the restriction acts as useful "
            "regularization."
        )
    elif not significant_positive.empty and largest_effect_fraction >= 0.05:
        recommendation = (
            "Aggregate mass coupling materially distorts shared-profile recovery in at "
            "least one tested experiment family: higher mass severity has a positive, "
            "non-negligible adjusted association with CLR error."
        )
    else:
        recommendation = (
            "Aggregate mass coupling has a detectable but small association with positional "
            "error. The present single-seed, single-order evidence is insufficient to call "
            "that association material or to claim useful regularization."
        )
    if cohort_exclusions.empty:
        exclusion_text = "No transcript was excluded for boundary eligibility."
    else:
        exclusion_text = (
            f"The audit excluded {cohort_exclusions['transcript_id'].nunique():,} "
            "unique short transcript(s) for which fewer than three codons remain "
            "after boundary trimming; details are in `cohort_exclusions.csv`."
        )
    report = rf"""# Synthetic aggregate-mass audit

## Scope

This post-hoc audit uses frozen **`{checkpoint_variant}`** prediction exports selected by **`{checkpoint_selection_metric}`**. No checkpoint was changed, no parameter was updated, and no simulator quantity selected a model. The within-depth family contains 27 runs with **{len(cohorts[reference.analysis_cohort_key('within_depth', reference.DEPTH_ORDER[0])]):,}**, **{len(cohorts[reference.analysis_cohort_key('within_depth', reference.DEPTH_ORDER[1])]):,}**, and **{len(cohorts[reference.analysis_cohort_key('within_depth', reference.DEPTH_ORDER[2])]):,}** matched validation transcripts at increasing depth; the mixed-depth family contains 20 runs and **{len(cohorts.get('cross_depth', [])):,}** validation transcripts. Validation sequences were excluded from gradient updates but were reused for checkpoint selection, and no independent synthetic test split exists. {exclusion_text}

The primary positional domain removes the synthetic terminal entry and ten codons from each CDS end, matching the reported recovery analysis. The model itself normalized $L$, centered $\log\gamma$, and calculated $S$ on the full saved mask. Full-mask and reported-mask identities are therefore reported separately. $m_{{t,d}}$ is a calibration quantity, not translation activity or abundance.

## Structural oracle mass deviation

| Depth | $N$ | Median [IQR] $\log m^{{\rm ref}}$ | Median weighted gap | PCC learned/oracle mass | MAE log mass | Median PCC $\widehat L$ vs $L^{{\rm ref}}$ |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(endpoint_lines)}

Individual oracle masses can lie above or below one. The theoretically constrained quantity is the reference-weighted average $\sum_d\pi_d m^{{\rm ref}}_{{t,d}}$, which was non-negative relative to one within numerical tolerance for every transcript. Dataset-resolved results, correction magnitudes, and associations with $\mathrm{{sd}}_i(\log G_i)$ are in `synthetic_mass_summary.csv`.

Across all analyzed transcript--dataset rows, the median oracle mass is **{median_oracle_mass:.3f}** and the median learned mass is **{median_learned_mass:.3f}**. Oracle values are above one in **{oracle_above:.1%}** and below one in **{oracle_below:.1%}** of rows; exact-one cases include panels containing repeated copies of one bias at different depths. At transcript level, the pooled median reference-weighted excess is **{median_oracle_gap:.3f}** for the oracle and **{median_learned_gap:.3f}** for the model. The oracle decomposition has median contributions **{median_gamma_contribution:.3f}** from $\langle\gamma\rangle-1$ and **{median_covariance_contribution:.3f}** from $\operatorname{{Cov}}(L,\gamma)$, so the arithmetic mean of a geometrically centered correction is the larger contribution.

## Learned versus oracle mass

Calibration is evaluated on $\log\widehat m$ versus $\log m^{{\rm ref}}$ with transcript-cluster bootstrap intervals. The table above gives representative endpoints; every run contains Pearson and Spearman correlation, MAE, RMSE, calibration intercept and slope, and the log-error distribution. Oracle raw-score infeasibility occurred in **{oracle_infeasible:.3%}** of transcript--dataset rows. Learned raw scores reached the configured bound in **{learned_bound:.3%}** of rows. A likelihood mean floor affected at least one reported position in **{floor_rows:.3%}** of rows and is recorded separately.

Mixed-depth calibration at $N=30$ is shown explicitly below. Correlation and MAE do not improve monotonically with read depth; mass is a constrained aggregate quantity and should not be used as a substitute for positional-profile recovery.

| Reference weighting | Dataset depth | Total $\pi$ at depth | PCC [cluster 95% CI] | MAE log mass | Calibration slope |
|---|---|---:|---:|---:|---:|
{chr(10).join(mixed_depth_lines)}

## Mass severity and shared-profile recovery

The outcome below is CLR RMSE between $\widehat L$ and $L^{{\rm ref}}$. Effects are changes in CLR RMSE per one within-run standard deviation of severity. Adjusted models include log transcript length and run fixed effects, thereby controlling the available $N$, depth, panel composition, reference weighting, and the single training seed.

| Family | Severity | Adjusted effect per SD | Cluster-robust 95% CI | Transcript clusters |
|---|---|---:|---:|---:|
{chr(10).join(regression_lines)}

The overall median transcript-level PCC between $\widehat L$ and $L^{{\rm ref}}$ is **{median_profile_pcc:.3f}**, with median CLR RMSE **{median_profile_clr:.3f}**. The largest adjusted severity effect is **{largest_adjusted_effect:.5f}**, or **{largest_effect_fraction:.2%}** of that median error per within-run SD. {severity_interval_text} This is association, not a causal decomposition.

Reference contamination is a separate and larger issue: median PCC is **{median_reference_pcc:.3f}** for $L^{{\rm ref}}$ versus $Q$ and **{median_end_to_end_pcc:.3f}** for $\widehat L$ versus $Q$. Disagreement with occupancy is therefore not attributed solely to mass.

## Numerical checks

There are **{fail_count} failed algebraic checks** and **{mask_count} explicitly classified mask-domain effects**. Learned within-profile log-centering and the decoder sum-ratio identity hold on the full model mask. They need not hold after boundary trimming because the frozen correction is not re-centered and the supplied scale remains the full-mask observed mean. Cross-dataset centering survives masking because it is pointwise. The prediction target matches the arithmetic mean of the raw replicas exactly; the legacy stored `ribo` column differs by as much as 0.5 and is not the target used by this datamodule. Exact deviations and floor cases are in `synthetic_mass_sanity_checks.csv`.

Only one optimization seed and one cumulative bias order are present. Training-seed dependence and hierarchical panel/seed uncertainty are therefore not identifiable. Depth comparisons use different validation cohorts and are not paired; comparisons across $N$ within a depth are paired.

## Recommendation

{recommendation}
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    output_dir = args.output_dir or resolve_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    include_cross = bool(
        config["run_selection"]["include_cross_depth_reference_sensitivity"]
    ) and not args.skip_cross_depth
    runs = reference.discover_runs(
        resolve_path(config["paths"]["results_root"]),
        seed=int(config["run_selection"]["training_seed"]),
        checkpoint_variant=str(config["run_selection"]["checkpoint_variant"]),
        include_cross=include_cross,
    )
    by_cohort: defaultdict[str, list[reference.RunSpec]] = defaultdict(list)
    for run in runs:
        by_cohort[reference.analysis_cohort_key(run.family, run.depth)].append(run)
    candidate_cohorts = {
        cohort_key: sorted(
            set.intersection(*(set(run.validation_ids) for run in cohort_runs))
        )
        for cohort_key, cohort_runs in by_cohort.items()
    }
    candidate_union_ids = set().union(
        *(set(values) for values in candidate_cohorts.values())
    )
    print(
        "Candidate matched cohorts: "
        + ", ".join(
            f"{family}={len(values):,}"
            for family, values in candidate_cohorts.items()
        ),
        flush=True,
    )
    trim = int(config["evaluation"]["boundary_trim_codons_each_end"])

    weighted_observation_root = resolve_path(
        config["paths"]["weighted_observation_root"]
    )
    settings = {
        run.run_id: inspect_run_settings(run, weighted_observation_root)
        for run in runs
    }
    if any(value.mass_conservation for value in settings.values()):
        raise ValueError("Mass audit expected mass_conservation=false in every selected run")
    if any(value.gamma_gauge != "geometric_mean_one" for value in settings.values()):
        raise ValueError("Mass audit expected geometric_mean_one gamma gauge")
    observation_paths, observation_provenance = collect_observation_sources(runs, settings)
    observations = load_observations(observation_paths, candidate_union_ids)
    occupancies, occupancy_provenance = reference.load_occupancy_consensus(
        resolve_path(config["paths"]["occupancy"]), candidate_union_ids
    )
    exclusion_rows: list[dict[str, Any]] = []
    cohorts: dict[str, list[str]] = {}
    for cohort_key, transcript_ids in candidate_cohorts.items():
        kept: list[str] = []
        for transcript_id in transcript_ids:
            sense_length = int(occupancies[transcript_id].size)
            retained_positions = max(0, sense_length - 2 * trim)
            if retained_positions < 3:
                exclusion_rows.append(
                    {
                        "cohort_key": cohort_key,
                        "transcript_id": transcript_id,
                        "reason": "fewer_than_three_positions_after_boundary_trim",
                        "sense_length": sense_length,
                        "boundary_trim_codons_each_end": trim,
                        "positions_after_length_based_trim": retained_positions,
                    }
                )
            else:
                kept.append(transcript_id)
        if not kept:
            raise ValueError(
                f"{cohort_key}: no transcripts remain after the boundary trim"
            )
        cohorts[cohort_key] = kept
    cohort_exclusions = pd.DataFrame(
        exclusion_rows,
        columns=[
            "cohort_key",
            "transcript_id",
            "reason",
            "sense_length",
            "boundary_trim_codons_each_end",
            "positions_after_length_based_trim",
        ],
    )
    union_ids = set().union(*(set(values) for values in cohorts.values()))
    print(
        "Analysis cohorts after boundary eligibility: "
        + ", ".join(f"{family}={len(values):,}" for family, values in cohorts.items())
        + f"; exclusions={len(cohort_exclusions):,}",
        flush=True,
    )
    biases, bias_provenance = reference.load_biases(
        resolve_path(config["paths"]["bias_root"]), union_ids
    )

    mass_frames: list[pd.DataFrame] = []
    transcript_frames: list[pd.DataFrame] = []
    maxima_by_run: dict[str, dict[str, float]] = {}
    run_manifest_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    profile_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    tolerances = {name: float(value) for name, value in config["tolerances"].items()}
    for index, run in enumerate(runs, start=1):
        pi, raw_pi_sum = reference.extract_reference_weights(run)
        panel_id = reference.panel_hash(run.datasets, pi)
        for dataset in run.datasets:
            panel_rows.append(
                {
                    "run_id": run.run_id,
                    "family": run.family,
                    "panel_depth": run.depth,
                    "n_datasets": run.n_datasets,
                    "n_bias_families": run.n_bias_families,
                    "reference_weighting": run.reference_weighting,
                    "panel_id": panel_id,
                    "dataset": dataset,
                    "dataset_depth": dataset_depth(run, dataset),
                    "bias_family": reference.base_bias_name(dataset),
                    "bias_order_position": list(run.datasets).index(dataset) + 1,
                    "pi": pi[dataset],
                    "stored_pi_sum_before_float64_renormalization": raw_pi_sum,
                    "observation_path": str(
                        settings[run.run_id].observation_paths[dataset].relative_to(ROOT)
                    ),
                    "configured_observation_path": settings[
                        run.run_id
                    ].configured_observation_paths[dataset],
                    "used_durable_path_fallback": str(
                        settings[run.run_id].configured_observation_paths[dataset]
                    )
                    != str(settings[run.run_id].observation_paths[dataset]),
                }
            )
        run_manifest_rows.append(
            {
                "run_id": run.run_id,
                "family": run.family,
                "panel_depth": run.depth,
                "n_datasets": run.n_datasets,
                "n_bias_families": run.n_bias_families,
                "reference_weighting": run.reference_weighting,
                "training_seed": run.training_seed,
                "panel_id": panel_id,
                "split_id": run.split_id,
                "training_transcripts": len(run.train_ids),
                "validation_transcripts": len(run.validation_ids),
                "analysis_cohort_key": reference.analysis_cohort_key(
                    run.family, run.depth
                ),
                "matched_analysis_transcripts": len(
                    cohorts[reference.analysis_cohort_key(run.family, run.depth)]
                ),
                "prediction_path": str(run.prediction_path.relative_to(ROOT)),
                "prediction_size_bytes": run.prediction_path.stat().st_size,
                "config_path": str(run.config_path.relative_to(ROOT)),
                "split_path": str(run.split_path.relative_to(ROOT)),
                "checkpoint_manifest_path": str(
                    run.checkpoint_manifest_path.relative_to(ROOT)
                ),
                "checkpoint_selection_metric": config["run_selection"][
                    "checkpoint_selection_metric"
                ],
                "raw_log_gamma_bound": settings[run.run_id].raw_log_gamma_bound,
                "likelihood_mean_floor": settings[run.run_id].likelihood_mean_floor,
                "mass_conservation": settings[run.run_id].mass_conservation,
                "gamma_gauge": settings[run.run_id].gamma_gauge,
            }
        )
        print(
            f"[{index}/{len(runs)}] {run.family} {run.depth} "
            f"N={run.n_datasets} {run.reference_weighting}",
            flush=True,
        )
        mass_frame, transcript_frame, maxima = analyze_run(
            run=run,
            cohort=set(cohorts[reference.analysis_cohort_key(run.family, run.depth)]),
            occupancies=occupancies,
            biases=biases,
            observations=observations,
            settings=settings[run.run_id],
            pi=pi,
            trim=trim,
            tolerances=tolerances,
            profile_cache=profile_cache,
        )
        mass_frames.append(mass_frame)
        transcript_frames.append(transcript_frame)
        maxima_by_run[run.run_id] = maxima

    mass = pd.concat(mass_frames, ignore_index=True)
    transcripts = pd.concat(transcript_frames, ignore_index=True)
    bootstrap = config["bootstrap"]
    summary = build_summary(
        mass,
        transcripts,
        bootstrap_replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
    )
    depth_summary = build_depth_summary(
        mass,
        summary.loc[summary["summary_level"] == "run"],
        bootstrap_replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
    )
    regressions = build_regressions(transcripts)
    sanity = build_sanity_table(maxima_by_run, mass, tolerances)
    representative_quantiles = [
        float(value) for value in config["figures"]["representative_quantiles"]
    ]
    selected, representative_profiles = select_representatives(
        transcripts, profile_cache, representative_quantiles
    )

    write_large_csv = bool(config.get("outputs", {}).get("write_large_csv", True))
    if write_large_csv:
        mass.to_csv(
            output_dir / "synthetic_mass_per_transcript_dataset.csv", index=False
        )
    mass.to_parquet(output_dir / "synthetic_mass_per_transcript_dataset.parquet", index=False)
    if write_large_csv:
        transcripts.to_csv(
            output_dir / "synthetic_mass_per_transcript.csv", index=False
        )
    transcripts.to_parquet(output_dir / "synthetic_mass_per_transcript.parquet", index=False)
    summary.to_csv(output_dir / "synthetic_mass_summary.csv", index=False)
    depth_summary.to_csv(
        output_dir / "synthetic_mass_depth_summary.csv", index=False
    )
    regressions.to_csv(output_dir / "synthetic_mass_regressions.csv", index=False)
    sanity.to_csv(output_dir / "synthetic_mass_sanity_checks.csv", index=False)
    pd.DataFrame(run_manifest_rows).to_csv(output_dir / "run_manifest.csv", index=False)
    pd.DataFrame(panel_rows).to_csv(output_dir / "panel_reference_weights.csv", index=False)
    pd.DataFrame(
        [
            {"cohort_key": cohort_key, "transcript_id": transcript_id}
            for cohort_key, values in cohorts.items()
            for transcript_id in values
        ]
    ).to_csv(output_dir / "heldout_transcript_ids.csv", index=False)
    cohort_exclusions.to_csv(output_dir / "cohort_exclusions.csv", index=False)
    selected.to_csv(output_dir / "representative_selection.csv", index=False)
    representative_profiles.to_csv(
        output_dir / "representative_profile_source.csv", index=False
    )

    dpi = int(config["figures"]["png_dpi"])
    maximum_points = int(config["figures"]["maximum_scatter_points"])
    plot_learned_vs_oracle(mass, summary, output_dir, dpi, maximum_points)
    plot_severity_vs_error(
        transcripts, regressions, output_dir, dpi, maximum_points
    )
    if bool(config["figures"].get("generate_extended_diagnostics", False)):
        plot_oracle_mass_distributions(mass, output_dir, dpi)
        plot_weighted_mass_inequality(transcripts, output_dir, dpi)
        plot_mass_decomposition(mass, output_dir, dpi)
        plot_representatives(representative_profiles, output_dir, dpi)
    write_report(
        output_dir,
        summary,
        depth_summary,
        regressions,
        sanity,
        mass,
        transcripts,
        cohorts,
        cohort_exclusions,
        checkpoint_variant=str(config["run_selection"]["checkpoint_variant"]),
        checkpoint_selection_metric=str(
            config["run_selection"]["checkpoint_selection_metric"]
        ),
    )

    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": file_sha256(Path(__file__)),
        "configuration": str(config_path.relative_to(ROOT)),
        "configuration_sha256": file_sha256(config_path),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "checkpoint_selection": (
            f"{config['run_selection']['checkpoint_selection_metric']} on validation "
            "observations; no oracle target"
        ),
        "cohorts": {
            cohort_key: {
                "n_transcripts": len(values),
                "identity_sha256": text_hash(values),
                "excluded_for_boundary_eligibility": int(
                    (cohort_exclusions["cohort_key"] == cohort_key).sum()
                ),
            }
            for cohort_key, values in cohorts.items()
        },
        "occupancy": occupancy_provenance,
        "biases": bias_provenance,
        "observations": observation_provenance,
        "mask_domains": {
            "primary_reported_mask": "remove synthetic terminal and 10 sense codons from each end",
            "model_mask": "saved mask used for L normalization, gamma within-profile gauge, and S",
        },
        "bootstrap": bootstrap,
        "limitations": {
            "optimization_seeds": [42],
            "bias_panel_orders": 1,
            "independent_test_split": False,
            "hierarchical_bootstrap": "not identifiable",
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": __import__("pyarrow").__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    command = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 RIBOUNMIX_PLOT_TEX=1 .venv/bin/python "
        "analyses/analyze_synthetic_mass_factor.py "
        f"--config {shlex.quote(str(config_path))}\n"
    )
    (output_dir / "commands.sh").write_text(command, encoding="utf-8")
    if (sanity["status"] == "FAIL").any():
        failures = sanity.loc[sanity["status"] == "FAIL", "check"].tolist()
        raise RuntimeError(f"Required mass sanity checks failed: {failures}")
    print(f"Wrote synthetic mass audit to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
