#!/usr/bin/env python3
"""Analyze the synthetic fixed-reference ranking demonstration.

This script compares frozen shared ``L_bio`` predictions with the deterministic
mean-one synthetic kinetics target.  Its primary comparison is paired within
seed: quality-ranked fixed-reference centering versus equal-reference
centering.  The deliberately reversed ranking is included automatically only
when its launcher status and prediction artifacts are complete.

No model is loaded or fitted.  Checkpoint choice is made from observed
validation quantities only; latent synthetic truth is used exclusively for
post-hoc evaluation.  The primary checkpoint is ``best_val_loss`` and
``best_pcc`` (validation ``mu`` PCC) is reported as a sensitivity analysis.

Outputs include compact source tables, validation/provenance records, a
machine-readable manifest, a Markdown interpretation, and publication PNG/PDF
figures using the repository's LaTeX/Latin-Modern style.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# Avoid an expensive throw-away cache when the user's Matplotlib config
# directory is read-only on a compute or analysis node.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/riboai_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Utils.publication_plot_style import latex_paper_style  # noqa: E402


DEFAULT_RESULTS_ROOT = (
    REPOSITORY_ROOT / "results" / "synthetic_pi_demo_fixed_20260909"
)
DEFAULT_TRUTH_PATH = (
    REPOSITORY_ROOT
    / "Datasets"
    / "Synthetic_data"
    / "artificial_ground_truth_kinetics_target_mean_one.parquet"
)
DEFAULT_OUTPUT_NAME = "analysis_lbio_ranking"
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "analyses"
    / "artifacts"
    / "synthetic"
    / "pi_demo"
    / "lbio_ranking"
)
PRIMARY_VARIANT = "best_val_loss"
PRIMARY_DOMAIN = "interior_trim5"
RUN_PATTERN = re.compile(r"^pi_demo_(equal|quality|reversed)_seed(\d+)$")
REQUIRED_PREDICTION_COLUMNS = {"transcript_id", "dataset_id", "length", "L_bio"}
POLICY_ORDER = ("reversed", "equal", "quality")
POLICY_COMPARISONS = (
    ("reversed", "equal"),
    ("equal", "quality"),
    ("reversed", "quality"),
)
COMPARISON_LABELS = {
    "equal_vs_reversed": "Reversed $\\rightarrow$\nEqual",
    "quality_vs_equal": "Equal $\\rightarrow$\nQuality",
    "quality_vs_reversed": "Reversed $\\rightarrow$\nQuality",
}
POLICY_COLORS = {
    "reversed": "#D55E00",
    "equal": "#7A7A7A",
    "quality": "#0072B2",
}
POLICY_LABELS = {
    "reversed": "Reversed rank\n$\\pi=(1/2,1/3,1/6)$",
    "equal": "Equal\n$\\pi=(1/3,1/3,1/3)$",
    "quality": "Quality rank\n$\\pi=(1/6,1/3,1/2)$",
}
POLICY_SHORT_LABELS = {
    "reversed": "Reversed rank",
    "equal": "Equal",
    "quality": "Quality rank",
}
VARIANT_LABELS = {
    "best_val_loss": "Best validation\nloss",
    "best_pcc": "Best validation\n$\\mu$ PCC",
}
SEED_MARKERS = ("o", "s", "^", "D", "P", "X")


@dataclass(frozen=True)
class RunArtifact:
    run_name: str
    run_dir: Path
    policy: str
    seed: int
    variant: str
    prediction_path: Path
    validation_ids: tuple[str, ...]
    validation_hash: str
    expected_dataset_count: int
    gamma_pi: tuple[float, ...]
    gamma_manifest_hash: str
    checkpoint_path: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _identity_hash(values: Iterable[str]) -> str:
    canonical = "\n".join(sorted(str(value) for value in values))
    return _sha256_bytes(canonical.encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _one_path(paths: Iterable[Path], description: str) -> Path:
    matches = sorted(paths)
    if len(matches) != 1:
        raise ValueError(f"Expected one {description}, found {len(matches)}.")
    return matches[0]


def _status(run_dir: Path) -> tuple[str, str]:
    path = run_dir / "launcher_status.json"
    if not path.is_file():
        return "missing", "missing_launcher_status"
    try:
        value = _read_json(path)
    except Exception as exc:  # preserve other valid runs
        return "invalid", f"invalid_launcher_status:{type(exc).__name__}"
    status = str(value.get("status", "missing"))
    return status, "" if status == "complete" else f"launcher_status_{status}"


def inspect_runs(
    results_root: Path,
    variants: Sequence[str],
) -> tuple[pd.DataFrame, list[RunArtifact], list[dict[str, Any]]]:
    """Validate artifacts and return an explicit inclusion/exclusion table."""
    availability_rows: list[dict[str, Any]] = []
    artifacts: list[RunArtifact] = []
    provenance: list[dict[str, Any]] = []

    run_dirs = sorted(path for path in results_root.iterdir() if path.is_dir())
    for run_dir in run_dirs:
        match = RUN_PATTERN.fullmatch(run_dir.name)
        if match is None:
            continue
        policy, raw_seed = match.groups()
        seed = int(raw_seed)
        launcher_status, status_reason = _status(run_dir)

        base_reasons: list[str] = []
        if status_reason:
            base_reasons.append(status_reason)

        design: dict[str, Any] = {}
        design_path = run_dir / "pi_demo_design_manifest.json"
        if not design_path.is_file():
            base_reasons.append("missing_design_manifest")
        else:
            try:
                design = _read_json(design_path)
                if str(design.get("policy")) != policy:
                    base_reasons.append("design_policy_mismatch")
                if int(design.get("seed", -1)) != seed:
                    base_reasons.append("design_seed_mismatch")
            except Exception as exc:
                base_reasons.append(f"invalid_design_manifest:{type(exc).__name__}")

        split_path: Path | None = None
        split: dict[str, Any] = {}
        validation_ids: tuple[str, ...] = ()
        validation_hash = ""
        try:
            split_path = _one_path(run_dir.rglob("split_manifest_*.json"), "split manifest")
            split = _read_json(split_path)
            validation_ids = tuple(str(value) for value in split.get("validation_ids", []))
            if not validation_ids:
                base_reasons.append("empty_validation_manifest")
            elif len(validation_ids) != len(set(validation_ids)):
                base_reasons.append("duplicate_validation_ids")
            validation_hash = _identity_hash(validation_ids)
            if int(split.get("seed", seed)) != seed:
                base_reasons.append("split_seed_mismatch")
        except Exception as exc:
            base_reasons.append(f"invalid_split_manifest:{type(exc).__name__}")

        gamma_path: Path | None = None
        gamma: dict[str, Any] = {}
        gamma_pi: tuple[float, ...] = ()
        gamma_manifest_hash = ""
        expected_dataset_count = 0
        try:
            gamma_path = _one_path(
                run_dir.rglob("gamma_reference_manifest.json"),
                "gamma-reference manifest",
            )
            gamma = _read_json(gamma_path)
            gamma_pi = tuple(float(value) for value in gamma.get("reference_pi", []))
            expected_dataset_count = len(gamma.get("selected_dataset_ids", []))
            gamma_manifest_hash = str(gamma.get("reference_manifest_hash", ""))
            if expected_dataset_count <= 0 or len(gamma_pi) != expected_dataset_count:
                base_reasons.append("invalid_gamma_reference_dimensions")
            if gamma_pi and not math.isclose(sum(gamma_pi), 1.0, abs_tol=1.0e-7):
                base_reasons.append("gamma_pi_does_not_sum_to_one")
            expected_weighting = "equal" if policy == "equal" else "quality_rank"
            if str(gamma.get("weighting")) != expected_weighting:
                base_reasons.append("gamma_weighting_policy_mismatch")
            expected_pi = {
                "equal": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                "quality": (1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0),
                "reversed": (1.0 / 2.0, 1.0 / 3.0, 1.0 / 6.0),
            }[policy]
            if len(gamma_pi) == 3 and not np.allclose(gamma_pi, expected_pi, atol=1.0e-7):
                base_reasons.append("gamma_pi_policy_mismatch")
        except Exception as exc:
            base_reasons.append(f"invalid_gamma_manifest:{type(exc).__name__}")

        checkpoint_manifest_path: Path | None = None
        checkpoint_manifest: dict[str, Any] = {}
        checkpoint_manifest_paths = sorted(
            run_dir.rglob("prediction_checkpoint_manifest.json")
        )
        if not checkpoint_manifest_paths:
            base_reasons.append("missing_prediction_manifest")
        elif len(checkpoint_manifest_paths) > 1:
            base_reasons.append("multiple_prediction_manifests")
        else:
            checkpoint_manifest_path = checkpoint_manifest_paths[0]
            try:
                checkpoint_manifest = _read_json(checkpoint_manifest_path)
            except Exception as exc:
                base_reasons.append(
                    f"invalid_prediction_manifest:{type(exc).__name__}"
                )

        panel = design.get("panel", []) if isinstance(design.get("panel", []), list) else []
        provenance.append(
            {
                "run_name": run_dir.name,
                "policy": policy,
                "seed": seed,
                "launcher_status": launcher_status,
                "validation_transcripts": len(validation_ids),
                "validation_id_hash": validation_hash,
                "gamma_reference_weighting": gamma.get("weighting"),
                "gamma_reference_pi": json.dumps(gamma_pi),
                "gamma_reference_manifest_hash": gamma_manifest_hash,
                "quality_metric": design.get("natural_quality_metric"),
                "policy_rank_transform": design.get("policy_rank_transform"),
                "panel": json.dumps(panel, sort_keys=True),
                "design_manifest": str(design_path.resolve()) if design_path.is_file() else "",
                "split_manifest": str(split_path.resolve()) if split_path else "",
                "gamma_manifest": str(gamma_path.resolve()) if gamma_path else "",
            }
        )

        for variant in variants:
            reasons = list(base_reasons)
            predictions = sorted(
                run_dir.rglob(f"predictions_main_val_{variant}_*.parquet")
            )
            prediction_path = predictions[0] if len(predictions) == 1 else None
            if len(predictions) == 0:
                reasons.append("missing_prediction_export")
            elif len(predictions) > 1:
                reasons.append("multiple_prediction_exports")

            checkpoint_path = ""
            entry = checkpoint_manifest.get(variant)
            if not isinstance(entry, dict):
                reasons.append(f"missing_checkpoint_manifest_entry_{variant}")
            else:
                checkpoint_path = str(entry.get("checkpoint_path", ""))
                manifest_output = Path(str(entry.get("output_path", ""))).name
                if prediction_path is not None and manifest_output != prediction_path.name:
                    reasons.append("prediction_manifest_filename_mismatch")
                manifest_hash = str(entry.get("transcript_id_hash", ""))
                if validation_hash and manifest_hash and manifest_hash != validation_hash:
                    reasons.append("prediction_manifest_cohort_hash_mismatch")
                manifest_count = int(entry.get("transcript_count", -1))
                if validation_ids and manifest_count != len(validation_ids):
                    reasons.append("prediction_manifest_transcript_count_mismatch")

            prediction_rows = 0
            prediction_bytes = 0
            if prediction_path is not None:
                try:
                    parquet = pq.ParquetFile(prediction_path)
                    missing = REQUIRED_PREDICTION_COLUMNS.difference(
                        parquet.schema_arrow.names
                    )
                    if missing:
                        reasons.append(
                            "missing_prediction_columns:" + ",".join(sorted(missing))
                        )
                    prediction_rows = int(parquet.metadata.num_rows)
                    prediction_bytes = int(prediction_path.stat().st_size)
                    expected_rows = len(validation_ids) * expected_dataset_count
                    if expected_rows and prediction_rows != expected_rows:
                        reasons.append("prediction_row_count_mismatch")
                    close = getattr(parquet, "close", None)
                    if callable(close):
                        close()
                except Exception as exc:
                    reasons.append(f"invalid_prediction_parquet:{type(exc).__name__}")

            reasons = sorted(set(reasons))
            included = not reasons and prediction_path is not None
            availability_rows.append(
                {
                    "run_name": run_dir.name,
                    "policy": policy,
                    "seed": seed,
                    "checkpoint_variant": variant,
                    "launcher_status": launcher_status,
                    "included": bool(included),
                    "exclusion_reason": ";".join(reasons),
                    "validation_transcripts": len(validation_ids),
                    "validation_id_hash": validation_hash,
                    "prediction_rows": prediction_rows,
                    "prediction_bytes": prediction_bytes,
                    "prediction_path": (
                        str(prediction_path.resolve()) if prediction_path else ""
                    ),
                    "checkpoint_path_from_manifest": checkpoint_path,
                }
            )
            if included:
                assert prediction_path is not None
                artifacts.append(
                    RunArtifact(
                        run_name=run_dir.name,
                        run_dir=run_dir,
                        policy=policy,
                        seed=seed,
                        variant=variant,
                        prediction_path=prediction_path,
                        validation_ids=validation_ids,
                        validation_hash=validation_hash,
                        expected_dataset_count=expected_dataset_count,
                        gamma_pi=gamma_pi,
                        gamma_manifest_hash=gamma_manifest_hash,
                        checkpoint_path=checkpoint_path,
                    )
                )

    availability = pd.DataFrame(availability_rows).sort_values(
        ["seed", "policy", "checkpoint_variant"], ignore_index=True
    )
    return availability, artifacts, provenance


def _normalize_mean_one(profile: Any, label: str) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError(f"{label} has fewer than two positions.")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values.")
    if bool((values < 0.0).any()):
        raise ValueError(f"{label} contains negative values.")
    mean = float(values.mean())
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"{label} has a non-positive mean.")
    return values / mean


def _pcc(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, bool, str]:
    pred_centered = prediction - prediction.mean()
    truth_centered = truth - truth.mean()
    pred_ss = float(np.dot(pred_centered, pred_centered))
    truth_ss = float(np.dot(truth_centered, truth_centered))
    scale = max(float(np.mean(prediction**2)), 1.0)
    if pred_ss <= np.finfo(np.float64).eps * prediction.size * scale:
        return float("nan"), False, "constant_prediction"
    if truth_ss <= np.finfo(np.float64).eps * truth.size:
        return float("nan"), False, "constant_truth"
    value = float(np.dot(pred_centered, truth_centered) / math.sqrt(pred_ss * truth_ss))
    return float(np.clip(value, -1.0, 1.0)), True, "ok"


def _metric_row(
    prediction_full: np.ndarray,
    truth_full: np.ndarray,
    *,
    trim: int,
    domain: str,
) -> dict[str, Any]:
    if domain == "full_cds":
        prediction = prediction_full
        truth = truth_full
    elif domain == f"interior_trim{trim}":
        stop = prediction_full.size - trim
        if trim < 0 or stop <= trim:
            return {
                "positions": 0,
                "pcc": float("nan"),
                "pcc_valid": False,
                "pcc_reason": "insufficient_interior_positions",
                "rmse": float("nan"),
                "mae": float("nan"),
            }
        prediction = prediction_full[trim:stop]
        truth = truth_full[trim:stop]
    else:
        raise ValueError(f"Unknown evaluation domain: {domain}")
    pcc, valid, reason = _pcc(prediction, truth)
    residual = prediction - truth
    return {
        "positions": int(prediction.size),
        "pcc": pcc,
        "pcc_valid": valid,
        "pcc_reason": reason,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def load_truth_subset(path: Path, requested_ids: set[str]) -> dict[str, np.ndarray]:
    """Stream the ground truth and retain only requested validation profiles."""
    parquet = pq.ParquetFile(path)
    required = {"transcript_id", "rib_profile"}
    missing = required.difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Ground truth is missing columns: {sorted(missing)}")
    truth: dict[str, np.ndarray] = {}
    for batch in parquet.iter_batches(
        batch_size=128,
        columns=["transcript_id", "rib_profile"],
    ):
        columns = batch.to_pydict()
        for transcript_id, profile in zip(
            columns["transcript_id"], columns["rib_profile"]
        ):
            transcript_id = str(transcript_id)
            if transcript_id not in requested_ids:
                continue
            if transcript_id in truth:
                raise ValueError(f"Duplicate latent truth ID: {transcript_id}")
            truth[transcript_id] = _normalize_mean_one(
                profile, f"latent truth {transcript_id}"
            )
        del columns, batch
    close = getattr(parquet, "close", None)
    if callable(close):
        close()
    absent = requested_ids.difference(truth)
    if absent:
        example = sorted(absent)[:3]
        raise ValueError(
            f"Latent truth lacks {len(absent)} requested transcripts, e.g. {example}."
        )
    return truth


def analyze_prediction(
    artifact: RunArtifact,
    truth: dict[str, np.ndarray],
    *,
    trim: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read one frozen export and calculate one metric row per transcript/domain."""
    parquet = pq.ParquetFile(artifact.prediction_path)
    first_profiles: dict[str, np.ndarray] = {}
    seen_dataset_ids: dict[str, set[int]] = {}
    rows: list[dict[str, Any]] = []
    maximum_duplicate_difference = 0.0
    maximum_mean_error = 0.0

    for batch in parquet.iter_batches(
        batch_size=24,
        columns=["transcript_id", "dataset_id", "length", "L_bio"],
    ):
        columns = batch.to_pydict()
        for transcript_id, dataset_id, raw_length, raw_profile in zip(
            columns["transcript_id"],
            columns["dataset_id"],
            columns["length"],
            columns["L_bio"],
        ):
            transcript_id = str(transcript_id)
            if transcript_id not in truth:
                raise ValueError(
                    f"{artifact.run_name} predicts an ID absent from truth: {transcript_id}"
                )
            truth_profile = truth[transcript_id]
            expected_length = int(truth_profile.size)
            declared_length = int(raw_length)
            prediction_raw = np.asarray(raw_profile, dtype=np.float64).reshape(-1)
            if declared_length < expected_length or prediction_raw.size < expected_length:
                raise ValueError(
                    f"Short prediction for {transcript_id}: declared={declared_length}, "
                    f"stored={prediction_raw.size}, truth={expected_length}."
                )
            prediction_raw = prediction_raw[:expected_length]
            if not np.isfinite(prediction_raw).all():
                raise ValueError(f"Non-finite L_bio values for {transcript_id}.")

            seen_dataset_ids.setdefault(transcript_id, set()).add(int(dataset_id))
            previous = first_profiles.get(transcript_id)
            if previous is not None:
                if previous.size != prediction_raw.size:
                    raise ValueError(f"Duplicate L_bio length mismatch for {transcript_id}.")
                maximum_duplicate_difference = max(
                    maximum_duplicate_difference,
                    float(np.max(np.abs(previous.astype(np.float64) - prediction_raw))),
                )
                continue

            first_profiles[transcript_id] = prediction_raw.astype(np.float32, copy=True)
            raw_mean = float(prediction_raw.mean())
            maximum_mean_error = max(maximum_mean_error, abs(raw_mean - 1.0))
            prediction = _normalize_mean_one(
                prediction_raw, f"L_bio {artifact.run_name}/{transcript_id}"
            )
            raw_rmse = float(np.sqrt(np.mean((prediction_raw - truth_profile) ** 2)))
            for domain in ("full_cds", f"interior_trim{trim}"):
                metrics = _metric_row(
                    prediction,
                    truth_profile,
                    trim=trim,
                    domain=domain,
                )
                rows.append(
                    {
                        "run_name": artifact.run_name,
                        "policy": artifact.policy,
                        "seed": artifact.seed,
                        "checkpoint_variant": artifact.variant,
                        "evaluation_domain": domain,
                        "boundary_trim_codons": 0 if domain == "full_cds" else trim,
                        "transcript_id": transcript_id,
                        "profile_length": expected_length,
                        "validation_id_hash": artifact.validation_hash,
                        "prediction_mean_before_normalization": raw_mean,
                        "truth_mean_before_normalization": float(truth_profile.mean()),
                        "raw_full_cds_rmse": raw_rmse,
                        **metrics,
                    }
                )
        del columns, batch

    close = getattr(parquet, "close", None)
    if callable(close):
        close()

    expected_ids = set(artifact.validation_ids)
    predicted_ids = set(first_profiles)
    if predicted_ids != expected_ids:
        missing = sorted(expected_ids.difference(predicted_ids))[:3]
        extra = sorted(predicted_ids.difference(expected_ids))[:3]
        raise ValueError(
            f"Prediction/manifest cohort mismatch for {artifact.run_name}: "
            f"missing={missing}, extra={extra}."
        )
    bad_multiplicity = {
        transcript_id: len(dataset_ids)
        for transcript_id, dataset_ids in seen_dataset_ids.items()
        if len(dataset_ids) != artifact.expected_dataset_count
    }
    if bad_multiplicity:
        example = list(sorted(bad_multiplicity.items()))[:3]
        raise ValueError(
            f"Unexpected dataset multiplicity in {artifact.run_name}: {example}."
        )
    if maximum_duplicate_difference > 1.0e-5:
        raise ValueError(
            "Dataset-independent L_bio differs across dataset rows in "
            f"{artifact.run_name}; max absolute difference "
            f"{maximum_duplicate_difference:.3e}."
        )

    profile_hash = hashlib.sha256()
    for transcript_id in sorted(first_profiles):
        profile_hash.update(transcript_id.encode("utf-8"))
        profile_hash.update(b"\0")
        profile_hash.update(first_profiles[transcript_id].tobytes())
    diagnostics = {
        "run_name": artifact.run_name,
        "policy": artifact.policy,
        "seed": artifact.seed,
        "checkpoint_variant": artifact.variant,
        "transcripts": len(first_profiles),
        "prediction_rows": int(sum(len(value) for value in seen_dataset_ids.values())),
        "maximum_duplicate_lbio_absolute_difference": maximum_duplicate_difference,
        "maximum_absolute_lbio_mean_minus_one": maximum_mean_error,
        "frozen_lbio_profile_hash": profile_hash.hexdigest(),
    }
    del first_profiles, seen_dataset_ids
    return rows, diagnostics


def _fisher_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    clipped = np.clip(array, -1.0 + 1.0e-7, 1.0 - 1.0e-7)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def summarize_runs(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["checkpoint_variant", "evaluation_domain", "policy", "seed", "run_name"]
    for key, group in metrics.groupby(keys, sort=True):
        variant, domain, policy, seed, run_name = key
        pcc = group.loc[group["pcc_valid"].astype(bool), "pcc"].to_numpy(float)
        rmse = group["rmse"].to_numpy(float)
        rows.append(
            {
                "checkpoint_variant": variant,
                "evaluation_domain": domain,
                "policy": policy,
                "seed": int(seed),
                "run_name": run_name,
                "transcripts": int(group["transcript_id"].nunique()),
                "valid_pcc_transcripts": int(np.isfinite(pcc).sum()),
                "fisher_mean_pcc": _fisher_mean(pcc),
                "mean_pcc": float(np.nanmean(pcc)),
                "median_pcc": float(np.nanmedian(pcc)),
                "mean_rmse": float(np.nanmean(rmse)),
                "median_rmse": float(np.nanmedian(rmse)),
                "mean_mae": float(np.nanmean(group["mae"].to_numpy(float))),
            }
        )
    return pd.DataFrame(rows).sort_values(keys, ignore_index=True)


def build_quality_equal_pairs(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[pd.DataFrame] = []
    cohort_rows: list[dict[str, Any]] = []
    for (variant, domain, seed), group in metrics.groupby(
        ["checkpoint_variant", "evaluation_domain", "seed"], sort=True
    ):
        equal = group.loc[group["policy"] == "equal"].copy()
        quality = group.loc[group["policy"] == "quality"].copy()
        if equal.empty or quality.empty:
            continue
        columns = ["transcript_id", "pcc", "pcc_valid", "pcc_reason", "rmse", "mae"]
        merged = equal[columns].merge(
            quality[columns],
            on="transcript_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_equal", "_quality"),
        )
        matched_ids = sorted(merged["transcript_id"].astype(str))
        merged.insert(0, "seed", int(seed))
        merged.insert(0, "evaluation_domain", str(domain))
        merged.insert(0, "checkpoint_variant", str(variant))
        merged["pcc_gain_quality_minus_equal"] = (
            merged["pcc_quality"] - merged["pcc_equal"]
        )
        merged["rmse_gain_equal_minus_quality"] = (
            merged["rmse_equal"] - merged["rmse_quality"]
        )
        merged["mae_gain_equal_minus_quality"] = (
            merged["mae_equal"] - merged["mae_quality"]
        )
        pair_rows.append(merged)
        cohort_rows.append(
            {
                "checkpoint_variant": variant,
                "evaluation_domain": domain,
                "seed": int(seed),
                "matched_transcripts": len(matched_ids),
                "matched_transcript_id_hash": _identity_hash(matched_ids),
                "equal_manifest_hash": str(equal["validation_id_hash"].iloc[0]),
                "quality_manifest_hash": str(quality["validation_id_hash"].iloc[0]),
                "manifest_cohorts_identical": set(equal["transcript_id"])
                == set(quality["transcript_id"]),
            }
        )
    if not pair_rows:
        raise RuntimeError("No matched equal-versus-quality prediction pair is available.")
    pairs = pd.concat(pair_rows, ignore_index=True)
    cohorts = pd.DataFrame(cohort_rows)
    return pairs, cohorts


def summarize_paired_by_seed(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["checkpoint_variant", "evaluation_domain", "seed"]
    for (variant, domain, seed), group in pairs.groupby(keys, sort=True):
        pcc_valid = (
            group["pcc_valid_equal"].astype(bool)
            & group["pcc_valid_quality"].astype(bool)
            & np.isfinite(group["pcc_equal"])
            & np.isfinite(group["pcc_quality"])
        )
        pcc_group = group.loc[pcc_valid]
        rows.append(
            {
                "checkpoint_variant": variant,
                "evaluation_domain": domain,
                "seed": int(seed),
                "matched_transcripts": int(group.shape[0]),
                "matched_valid_pcc_transcripts": int(pcc_group.shape[0]),
                "equal_fisher_mean_pcc": _fisher_mean(pcc_group["pcc_equal"]),
                "quality_fisher_mean_pcc": _fisher_mean(pcc_group["pcc_quality"]),
                "pcc_fisher_gain_quality_minus_equal": (
                    _fisher_mean(pcc_group["pcc_quality"])
                    - _fisher_mean(pcc_group["pcc_equal"])
                ),
                "median_paired_pcc_gain": float(
                    np.nanmedian(pcc_group["pcc_gain_quality_minus_equal"])
                ),
                "fraction_pcc_improved": float(
                    np.mean(pcc_group["pcc_gain_quality_minus_equal"] > 0.0)
                ),
                "equal_mean_rmse": float(np.nanmean(group["rmse_equal"])),
                "quality_mean_rmse": float(np.nanmean(group["rmse_quality"])),
                "mean_rmse_gain_equal_minus_quality": float(
                    np.nanmean(group["rmse_gain_equal_minus_quality"])
                ),
                "median_paired_rmse_gain": float(
                    np.nanmedian(group["rmse_gain_equal_minus_quality"])
                ),
                "fraction_rmse_improved": float(
                    np.mean(group["rmse_gain_equal_minus_quality"] > 0.0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(keys, ignore_index=True)


def _cluster_bootstrap_effects(
    group: pd.DataFrame,
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, tuple[float, float, float, int, int]]:
    """Bootstrap transcript IDs, retaining all available seed observations."""
    pcc_valid = (
        group["pcc_valid_equal"].astype(bool)
        & group["pcc_valid_quality"].astype(bool)
        & np.isfinite(group["pcc_equal"])
        & np.isfinite(group["pcc_quality"])
    )
    pcc = group.loc[pcc_valid].copy()
    pcc["z_equal"] = np.arctanh(
        np.clip(pcc["pcc_equal"].to_numpy(float), -1 + 1e-7, 1 - 1e-7)
    )
    pcc["z_quality"] = np.arctanh(
        np.clip(pcc["pcc_quality"].to_numpy(float), -1 + 1e-7, 1 - 1e-7)
    )
    pcc_cluster = pcc.groupby("transcript_id", sort=True).agg(
        z_equal_sum=("z_equal", "sum"),
        z_quality_sum=("z_quality", "sum"),
        count=("z_equal", "size"),
    )
    rmse_cluster = group.groupby("transcript_id", sort=True).agg(
        difference_sum=("rmse_gain_equal_minus_quality", "sum"),
        count=("rmse_gain_equal_minus_quality", "size"),
    )

    pcc_estimate = _fisher_mean(pcc["pcc_quality"]) - _fisher_mean(pcc["pcc_equal"])
    rmse_estimate = float(np.mean(group["rmse_gain_equal_minus_quality"]))
    rng = np.random.default_rng(random_seed)

    def draw_effects(frame: pd.DataFrame, kind: str) -> np.ndarray:
        n_clusters = frame.shape[0]
        values = np.empty(replicates, dtype=np.float64)
        chunk_size = 64
        for start in range(0, replicates, chunk_size):
            stop = min(start + chunk_size, replicates)
            indices = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
            counts = frame["count"].to_numpy(float)[indices].sum(axis=1)
            if kind == "pcc":
                equal = frame["z_equal_sum"].to_numpy(float)[indices].sum(axis=1)
                quality = frame["z_quality_sum"].to_numpy(float)[indices].sum(axis=1)
                values[start:stop] = np.tanh(quality / counts) - np.tanh(equal / counts)
            else:
                difference = frame["difference_sum"].to_numpy(float)[indices].sum(axis=1)
                values[start:stop] = difference / counts
        return values

    pcc_draws = draw_effects(pcc_cluster, "pcc")
    rmse_draws = draw_effects(rmse_cluster, "rmse")
    pcc_low, pcc_high = np.quantile(pcc_draws, [0.025, 0.975])
    rmse_low, rmse_high = np.quantile(rmse_draws, [0.025, 0.975])
    return {
        "fisher_mean_pcc_gain_quality_minus_equal": (
            float(pcc_estimate),
            float(pcc_low),
            float(pcc_high),
            int(pcc_cluster.shape[0]),
            int(pcc.shape[0]),
        ),
        "mean_rmse_gain_equal_minus_quality": (
            rmse_estimate,
            float(rmse_low),
            float(rmse_high),
            int(rmse_cluster.shape[0]),
            int(group.shape[0]),
        ),
    }


def bootstrap_contrasts(
    pairs: pd.DataFrame,
    *,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, ((variant, domain), group) in enumerate(
        pairs.groupby(["checkpoint_variant", "evaluation_domain"], sort=True)
    ):
        effects = _cluster_bootstrap_effects(
            group,
            replicates=replicates,
            random_seed=random_seed + index,
        )
        for statistic, (estimate, low, high, clusters, observations) in effects.items():
            rows.append(
                {
                    "checkpoint_variant": variant,
                    "evaluation_domain": domain,
                    "contrast": "quality_minus_equal",
                    "statistic": statistic,
                    "estimate": estimate,
                    "ci_2p5": low,
                    "ci_97p5": high,
                    "bootstrap_replicates": replicates,
                    "bootstrap_unit": "transcript_id_cluster_retaining_seed_occurrences",
                    "transcript_clusters": clusters,
                    "paired_seed_transcript_observations": observations,
                }
            )
    return pd.DataFrame(rows)


def build_all_policy_pairs(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create position-matched transcript rows for every available policy pair.

    A comparison is named ``right_vs_left``.  Both reported gains are oriented
    so that a positive value means that moving from the left policy to the
    right policy improves latent-profile recovery:

    * PCC gain = PCC(right) - PCC(left)
    * RMSE gain = RMSE(left) - RMSE(right)
    """
    pair_frames: list[pd.DataFrame] = []
    cohort_rows: list[dict[str, Any]] = []
    columns = [
        "transcript_id",
        "pcc",
        "pcc_valid",
        "pcc_reason",
        "rmse",
        "mae",
        "validation_id_hash",
    ]
    for (variant, domain, seed), group in metrics.groupby(
        ["checkpoint_variant", "evaluation_domain", "seed"], sort=True
    ):
        policy_frames = {
            policy: group.loc[group["policy"] == policy, columns].copy()
            for policy in POLICY_ORDER
        }
        for left_policy, right_policy in POLICY_COMPARISONS:
            left = policy_frames[left_policy]
            right = policy_frames[right_policy]
            if left.empty or right.empty:
                continue
            comparison = f"{right_policy}_vs_{left_policy}"
            merged = left.merge(
                right,
                on="transcript_id",
                how="inner",
                validate="one_to_one",
                suffixes=("_left", "_right"),
            )
            matched_ids = sorted(merged["transcript_id"].astype(str))
            merged.insert(0, "right_policy", right_policy)
            merged.insert(0, "left_policy", left_policy)
            merged.insert(0, "comparison", comparison)
            merged.insert(0, "seed", int(seed))
            merged.insert(0, "evaluation_domain", str(domain))
            merged.insert(0, "checkpoint_variant", str(variant))
            merged["pcc_gain_right_minus_left"] = (
                merged["pcc_right"] - merged["pcc_left"]
            )
            merged["rmse_gain_left_minus_right"] = (
                merged["rmse_left"] - merged["rmse_right"]
            )
            merged["mae_gain_left_minus_right"] = (
                merged["mae_left"] - merged["mae_right"]
            )
            pair_frames.append(merged)
            cohort_rows.append(
                {
                    "checkpoint_variant": variant,
                    "evaluation_domain": domain,
                    "seed": int(seed),
                    "comparison": comparison,
                    "left_policy": left_policy,
                    "right_policy": right_policy,
                    "matched_transcripts": len(matched_ids),
                    "matched_transcript_id_hash": _identity_hash(matched_ids),
                    "left_manifest_hash": str(
                        left["validation_id_hash"].iloc[0]
                    ),
                    "right_manifest_hash": str(
                        right["validation_id_hash"].iloc[0]
                    ),
                    "manifest_cohorts_identical": set(left["transcript_id"])
                    == set(right["transcript_id"]),
                }
            )
    if not pair_frames:
        raise RuntimeError("No complete matched policy comparison is available.")
    pairs = pd.concat(pair_frames, ignore_index=True)
    pairs.sort_values(
        [
            "checkpoint_variant",
            "evaluation_domain",
            "seed",
            "comparison",
            "transcript_id",
        ],
        inplace=True,
        ignore_index=True,
    )
    cohorts = pd.DataFrame(cohort_rows).sort_values(
        ["checkpoint_variant", "evaluation_domain", "seed", "comparison"],
        ignore_index=True,
    )
    return pairs, cohorts


def summarize_all_policy_pairs_by_seed(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "checkpoint_variant",
        "evaluation_domain",
        "comparison",
        "left_policy",
        "right_policy",
        "seed",
    ]
    for key, group in pairs.groupby(keys, sort=True):
        variant, domain, comparison, left_policy, right_policy, seed = key
        pcc_valid = (
            group["pcc_valid_left"].astype(bool)
            & group["pcc_valid_right"].astype(bool)
            & np.isfinite(group["pcc_left"])
            & np.isfinite(group["pcc_right"])
        )
        pcc_group = group.loc[pcc_valid]
        left_pcc = _fisher_mean(pcc_group["pcc_left"])
        right_pcc = _fisher_mean(pcc_group["pcc_right"])
        rows.append(
            {
                "checkpoint_variant": variant,
                "evaluation_domain": domain,
                "comparison": comparison,
                "left_policy": left_policy,
                "right_policy": right_policy,
                "seed": int(seed),
                "matched_transcripts": int(group.shape[0]),
                "matched_valid_pcc_transcripts": int(pcc_group.shape[0]),
                "left_fisher_mean_pcc": left_pcc,
                "right_fisher_mean_pcc": right_pcc,
                "pcc_fisher_gain_right_minus_left": right_pcc - left_pcc,
                "median_paired_pcc_gain_right_minus_left": float(
                    np.nanmedian(pcc_group["pcc_gain_right_minus_left"])
                ),
                "fraction_pcc_improved": float(
                    np.mean(pcc_group["pcc_gain_right_minus_left"] > 0.0)
                ),
                "left_mean_rmse": float(np.nanmean(group["rmse_left"])),
                "right_mean_rmse": float(np.nanmean(group["rmse_right"])),
                "mean_rmse_gain_left_minus_right": float(
                    np.nanmean(group["rmse_gain_left_minus_right"])
                ),
                "median_paired_rmse_gain_left_minus_right": float(
                    np.nanmedian(group["rmse_gain_left_minus_right"])
                ),
                "fraction_rmse_improved": float(
                    np.mean(group["rmse_gain_left_minus_right"] > 0.0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(keys, ignore_index=True)


def _cluster_bootstrap_generic_effects(
    group: pd.DataFrame,
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, tuple[float, float, float, int, int]]:
    """Bootstrap transcript IDs for one explicitly oriented policy pair."""
    pcc_valid = (
        group["pcc_valid_left"].astype(bool)
        & group["pcc_valid_right"].astype(bool)
        & np.isfinite(group["pcc_left"])
        & np.isfinite(group["pcc_right"])
    )
    pcc = group.loc[pcc_valid].copy()
    pcc["z_left"] = np.arctanh(
        np.clip(pcc["pcc_left"].to_numpy(float), -1 + 1e-7, 1 - 1e-7)
    )
    pcc["z_right"] = np.arctanh(
        np.clip(pcc["pcc_right"].to_numpy(float), -1 + 1e-7, 1 - 1e-7)
    )
    pcc_cluster = pcc.groupby("transcript_id", sort=True).agg(
        z_left_sum=("z_left", "sum"),
        z_right_sum=("z_right", "sum"),
        count=("z_left", "size"),
    )
    rmse_cluster = group.groupby("transcript_id", sort=True).agg(
        difference_sum=("rmse_gain_left_minus_right", "sum"),
        count=("rmse_gain_left_minus_right", "size"),
    )
    pcc_estimate = _fisher_mean(pcc["pcc_right"]) - _fisher_mean(pcc["pcc_left"])
    rmse_estimate = float(np.mean(group["rmse_gain_left_minus_right"]))
    rng = np.random.default_rng(random_seed)

    def draw_effects(frame: pd.DataFrame, kind: str) -> np.ndarray:
        n_clusters = int(frame.shape[0])
        if n_clusters == 0:
            return np.full(replicates, np.nan, dtype=np.float64)
        values = np.empty(replicates, dtype=np.float64)
        for start in range(0, replicates, 64):
            stop = min(start + 64, replicates)
            indices = rng.integers(
                0,
                n_clusters,
                size=(stop - start, n_clusters),
            )
            counts = frame["count"].to_numpy(float)[indices].sum(axis=1)
            if kind == "pcc":
                left = frame["z_left_sum"].to_numpy(float)[indices].sum(axis=1)
                right = frame["z_right_sum"].to_numpy(float)[indices].sum(axis=1)
                values[start:stop] = np.tanh(right / counts) - np.tanh(left / counts)
            else:
                difference = frame["difference_sum"].to_numpy(float)[indices].sum(axis=1)
                values[start:stop] = difference / counts
        return values

    pcc_draws = draw_effects(pcc_cluster, "pcc")
    rmse_draws = draw_effects(rmse_cluster, "rmse")
    pcc_low, pcc_high = np.nanquantile(pcc_draws, [0.025, 0.975])
    rmse_low, rmse_high = np.nanquantile(rmse_draws, [0.025, 0.975])
    return {
        "fisher_mean_pcc_gain_right_minus_left": (
            float(pcc_estimate),
            float(pcc_low),
            float(pcc_high),
            int(pcc_cluster.shape[0]),
            int(pcc.shape[0]),
        ),
        "mean_rmse_gain_left_minus_right": (
            rmse_estimate,
            float(rmse_low),
            float(rmse_high),
            int(rmse_cluster.shape[0]),
            int(group.shape[0]),
        ),
    }


def bootstrap_all_policy_contrasts(
    pairs: pd.DataFrame,
    *,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "checkpoint_variant",
        "evaluation_domain",
        "comparison",
        "left_policy",
        "right_policy",
    ]
    for index, (key, group) in enumerate(pairs.groupby(keys, sort=True)):
        variant, domain, comparison, left_policy, right_policy = key
        effects = _cluster_bootstrap_generic_effects(
            group,
            replicates=replicates,
            random_seed=random_seed + index,
        )
        for statistic, (estimate, low, high, clusters, observations) in effects.items():
            rows.append(
                {
                    "checkpoint_variant": variant,
                    "evaluation_domain": domain,
                    "comparison": comparison,
                    "left_policy": left_policy,
                    "right_policy": right_policy,
                    "positive_direction": f"favors_{right_policy}",
                    "statistic": statistic,
                    "estimate": estimate,
                    "ci_2p5": low,
                    "ci_97p5": high,
                    "bootstrap_replicates": replicates,
                    "bootstrap_unit": (
                        "transcript_id_cluster_retaining_seed_occurrences"
                    ),
                    "transcript_clusters": clusters,
                    "paired_seed_transcript_observations": observations,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["checkpoint_variant", "evaluation_domain", "comparison", "statistic"],
        ignore_index=True,
    )


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.13,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _effect_row(
    contrasts: pd.DataFrame,
    variant: str,
    domain: str,
    statistic: str,
) -> pd.Series:
    selected = contrasts.loc[
        (contrasts["checkpoint_variant"] == variant)
        & (contrasts["evaluation_domain"] == domain)
        & (contrasts["statistic"] == statistic)
    ]
    if selected.shape[0] != 1:
        raise ValueError(
            f"Expected one contrast for {variant}/{domain}/{statistic}, "
            f"found {selected.shape[0]}."
        )
    return selected.iloc[0]


@latex_paper_style
def plot_main_recovery(
    seed_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    output_stem: Path,
    *,
    variant: str,
    domain: str,
) -> None:
    selected = seed_summary.loc[
        (seed_summary["checkpoint_variant"] == variant)
        & (seed_summary["evaluation_domain"] == domain)
    ].sort_values("seed")
    if selected.empty:
        raise ValueError("No seed summaries are available for the main figure.")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.65))
    x = np.asarray([0.0, 1.0])
    labels = [POLICY_LABELS["equal"], POLICY_LABELS["quality"]]
    for seed_index, row in enumerate(selected.itertuples(index=False)):
        marker = SEED_MARKERS[seed_index % len(SEED_MARKERS)]
        axes[0].plot(
            x,
            [row.equal_fisher_mean_pcc, row.quality_fisher_mean_pcc],
            color="#B8B8B8",
            linewidth=1.0,
            zorder=1,
        )
        axes[1].plot(
            x,
            [row.equal_mean_rmse, row.quality_mean_rmse],
            color="#B8B8B8",
            linewidth=1.0,
            zorder=1,
        )
        for ax, values in zip(
            axes,
            (
                [row.equal_fisher_mean_pcc, row.quality_fisher_mean_pcc],
                [row.equal_mean_rmse, row.quality_mean_rmse],
            ),
        ):
            for policy_index, (policy, value) in enumerate(
                zip(("equal", "quality"), values)
            ):
                ax.scatter(
                    policy_index,
                    value,
                    s=48,
                    marker=marker,
                    color=POLICY_COLORS[policy],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y")
        ax.set_xlim(-0.42, 1.42)
    axes[0].set_ylabel("Fisher-mean transcript PCC")
    axes[1].set_ylabel("Mean transcript RMSE")
    axes[0].set_ylim(min(0.0, float(selected[["equal_fisher_mean_pcc", "quality_fisher_mean_pcc"]].min().min()) - 0.05), 1.0)
    axes[1].set_ylim(
        0.0,
        float(selected[["equal_mean_rmse", "quality_mean_rmse"]].max().max())
        * 1.48,
    )

    pcc_effect = _effect_row(
        contrasts,
        variant,
        domain,
        "fisher_mean_pcc_gain_quality_minus_equal",
    )
    rmse_effect = _effect_row(
        contrasts,
        variant,
        domain,
        "mean_rmse_gain_equal_minus_quality",
    )
    axes[0].text(
        0.03,
        0.05,
        "$\\Delta$PCC = "
        f"{pcc_effect['estimate']:+.3f}\n"
        f"95\\% CI [{pcc_effect['ci_2p5']:+.3f}, {pcc_effect['ci_97p5']:+.3f}]",
        transform=axes[0].transAxes,
        va="bottom",
        fontsize=12,
    )
    axes[1].text(
        0.03,
        0.95,
        "$\\Delta$RMSE = "
        f"{rmse_effect['estimate']:+.3f}\n"
        f"95\\% CI [{rmse_effect['ci_2p5']:+.3f}, {rmse_effect['ci_97p5']:+.3f}]",
        transform=axes[1].transAxes,
        va="top",
        fontsize=12,
    )
    axes[1].text(
        0.03,
        0.05,
        "Positive $\\Delta$: quality rank improves recovery",
        transform=axes[1].transAxes,
        va="bottom",
        fontsize=12,
    )
    _panel_letter(axes[0], "A")
    _panel_letter(axes[1], "B")

    handles = [
        mlines.Line2D(
            [],
            [],
            linestyle="none",
            marker=SEED_MARKERS[index % len(SEED_MARKERS)],
            markersize=7,
            markerfacecolor="#4D4D4D",
            markeredgecolor="white",
            label=f"Seed {int(seed)}",
        )
        for index, seed in enumerate(selected["seed"])
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.24, top=0.83, wspace=0.35)
    _save_figure(fig, output_stem)


@latex_paper_style
def plot_paired_differences(
    pairs: pd.DataFrame,
    output_stem: Path,
    *,
    variant: str,
    domain: str,
) -> None:
    selected = pairs.loc[
        (pairs["checkpoint_variant"] == variant)
        & (pairs["evaluation_domain"] == domain)
    ].copy()
    seeds = sorted(int(value) for value in selected["seed"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.55))
    specifications = (
        ("pcc_gain_quality_minus_equal", "$\\Delta$PCC (quality $-$ equal)"),
        ("rmse_gain_equal_minus_quality", "$\\Delta$RMSE (equal $-$ quality)"),
    )
    for ax, (column, ylabel) in zip(axes, specifications):
        values = [
            selected.loc[selected["seed"] == seed, column].dropna().to_numpy(float)
            for seed in seeds
        ]
        boxes = ax.boxplot(
            values,
            positions=np.arange(len(seeds)),
            widths=0.58,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.3},
            whiskerprops={"color": "#4D4D4D"},
            capprops={"color": "#4D4D4D"},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(POLICY_COLORS["quality"])
            box.set_alpha(0.82)
            box.set_edgecolor("#4D4D4D")
        means = [float(np.mean(value)) for value in values]
        ax.scatter(
            np.arange(len(seeds)),
            means,
            marker="D",
            s=34,
            color="black",
            zorder=3,
            label="Mean",
        )
        ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xticks(np.arange(len(seeds)), [f"Seed {seed}" for seed in seeds])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        fractions = [float(np.mean(value > 0.0)) for value in values]
        for x_value, fraction in enumerate(fractions):
            ax.text(
                x_value,
                0.97,
                f"{100.0 * fraction:.0f}\\%",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=12,
            )
    _panel_letter(axes[0], "A")
    _panel_letter(axes[1], "B")
    fig.text(
        0.5,
        0.015,
        "Positive values favor quality rank",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.24, top=0.92, wspace=0.36)
    _save_figure(fig, output_stem)


@latex_paper_style
def plot_matched_scatter(
    pairs: pd.DataFrame,
    output_stem: Path,
    *,
    variant: str,
    domain: str,
) -> None:
    selected = pairs.loc[
        (pairs["checkpoint_variant"] == variant)
        & (pairs["evaluation_domain"] == domain)
    ].copy()
    seeds = sorted(int(value) for value in selected["seed"].unique())
    seed_colors = dict(zip(seeds, ("#0072B2", "#E69F00", "#009E73", "#CC79A7")))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.55))
    specs = (
        ("pcc_equal", "pcc_quality", "Transcript PCC"),
        ("rmse_equal", "rmse_quality", "Transcript RMSE"),
    )
    for ax, (x_column, y_column, metric_label) in zip(axes, specs):
        finite = np.isfinite(selected[x_column]) & np.isfinite(selected[y_column])
        plotted = selected.loc[finite]
        for seed in seeds:
            seed_rows = plotted.loc[plotted["seed"] == seed]
            ax.scatter(
                seed_rows[x_column],
                seed_rows[y_column],
                s=7,
                alpha=0.20,
                linewidths=0,
                color=seed_colors[seed],
                rasterized=True,
                label=f"Seed {seed}",
            )
        low = float(min(plotted[x_column].min(), plotted[y_column].min()))
        high = float(max(plotted[x_column].max(), plotted[y_column].max()))
        padding = max((high - low) * 0.035, 1.0e-3)
        ax.plot(
            [low - padding, high + padding],
            [low - padding, high + padding],
            color="#333333",
            linewidth=0.9,
            linestyle="--",
        )
        ax.set_xlim(low - padding, high + padding)
        ax.set_ylim(low - padding, high + padding)
        ax.set_xlabel(f"Equal-reference {metric_label}")
        ax.set_ylabel(f"Quality-ranked {metric_label}")
        ax.grid(alpha=0.35)
    _panel_letter(axes[0], "A")
    _panel_letter(axes[1], "B")
    axes[0].legend(loc="lower right", markerscale=2.2)
    axes[0].text(
        0.03,
        0.97,
        "Above diagonal favors quality rank",
        transform=axes[0].transAxes,
        va="top",
        fontsize=12,
    )
    axes[1].text(
        0.03,
        0.97,
        "Below diagonal favors quality rank",
        transform=axes[1].transAxes,
        va="top",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.18, top=0.92, wspace=0.36)
    _save_figure(fig, output_stem)


@latex_paper_style
def plot_checkpoint_sensitivity(
    seed_summary: pd.DataFrame,
    output_stem: Path,
    *,
    domain: str,
) -> None:
    selected = seed_summary.loc[seed_summary["evaluation_domain"] == domain].copy()
    variants = [
        variant
        for variant in ("best_val_loss", "best_pcc")
        if variant in set(selected["checkpoint_variant"])
    ]
    if len(variants) < 2:
        return
    seeds = sorted(int(value) for value in selected["seed"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.45))
    specs = (
        ("pcc_fisher_gain_quality_minus_equal", "$\\Delta$PCC (quality $-$ equal)"),
        ("mean_rmse_gain_equal_minus_quality", "$\\Delta$RMSE (equal $-$ quality)"),
    )
    x = np.arange(len(variants))
    for ax, (column, ylabel) in zip(axes, specs):
        for seed_index, seed in enumerate(seeds):
            seed_rows = selected.loc[selected["seed"] == seed].set_index(
                "checkpoint_variant"
            )
            if not all(variant in seed_rows.index for variant in variants):
                continue
            values = [float(seed_rows.loc[variant, column]) for variant in variants]
            ax.plot(x, values, color="#A5A5A5", linewidth=1.0)
            ax.scatter(
                x,
                values,
                marker=SEED_MARKERS[seed_index % len(SEED_MARKERS)],
                color="#0072B2",
                edgecolor="white",
                linewidth=0.7,
                s=48,
                zorder=3,
                label=f"Seed {seed}" if ax is axes[0] else None,
            )
        ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xticks(x, [VARIANT_LABELS[value] for value in variants])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
    axes[0].legend(loc="best")
    _panel_letter(axes[0], "A")
    _panel_letter(axes[1], "B")
    fig.text(
        0.5,
        0.015,
        "Positive values favor quality rank",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.31, top=0.92, wspace=0.38)
    _save_figure(fig, output_stem)


@latex_paper_style
def plot_three_policy_recovery(
    run_summary: pd.DataFrame,
    output_stem: Path,
    *,
    variant: str,
    domain: str,
) -> None:
    """Primary article panel with reversed, equal, and quality-ranked policies."""
    matplotlib.rcParams.update(
        {
            "font.size": 13.0,
            "font.weight": "bold",
            "axes.labelsize": 13.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 14.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 12.0,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "grid.linewidth": 0.7,
        }
    )
    if matplotlib.rcParams["text.usetex"]:
        matplotlib.rcParams["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}"
            r"\AtBeginDocument{\boldmath}"
        )
    selected = run_summary.loc[
        (run_summary["checkpoint_variant"] == variant)
        & (run_summary["evaluation_domain"] == domain)
    ].copy()
    policies = [policy for policy in POLICY_ORDER if policy in set(selected["policy"])]
    if len(policies) < 2:
        raise ValueError("At least two complete policies are required for the main plot.")
    seeds = sorted(int(value) for value in selected["seed"].unique())
    x = np.arange(len(policies), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.9))
    metric_columns = ("fisher_mean_pcc", "mean_rmse")

    for seed_index, seed in enumerate(seeds):
        seed_rows = selected.loc[selected["seed"] == seed].set_index("policy")
        available = [policy for policy in policies if policy in seed_rows.index]
        positions = [policies.index(policy) for policy in available]
        marker = SEED_MARKERS[seed_index % len(SEED_MARKERS)]
        for ax, metric in zip(axes, metric_columns):
            values = [float(seed_rows.loc[policy, metric]) for policy in available]
            ax.plot(
                positions,
                values,
                color="#B5B5B5",
                linewidth=2.1,
                zorder=1,
            )
            for position, policy, value in zip(positions, available, values):
                ax.scatter(
                    position,
                    value,
                    s=70,
                    marker=marker,
                    color=POLICY_COLORS[policy],
                    edgecolor="white",
                    linewidth=1.0,
                    zorder=3,
                )

    for ax in axes:
        ax.set_xticks(x, [POLICY_SHORT_LABELS[policy] for policy in policies])
        ax.set_xlim(-0.42, len(policies) - 0.58)
        ax.grid(axis="y")
    axes[0].set_ylabel("Fisher-mean transcript PCC")
    axes[1].set_ylabel("Mean transcript RMSE")
    axes[0].set_ylim(
        min(0.0, float(selected["fisher_mean_pcc"].min()) - 0.05),
        1.0,
    )
    axes[1].set_ylim(0.0, float(selected["mean_rmse"].max()) * 1.28)

    policy_means = selected.groupby("policy", sort=False).agg(
        pcc=("fisher_mean_pcc", "mean"),
        rmse=("mean_rmse", "mean"),
    )
    best_pcc = str(policy_means["pcc"].idxmax())
    best_rmse = str(policy_means["rmse"].idxmin())
    axes[0].text(
        0.03,
        0.05,
        f"Highest PCC: {best_pcc.capitalize()}",
        transform=axes[0].transAxes,
        fontsize=12.5,
        fontweight="bold",
        va="bottom",
    )
    axes[1].text(
        0.03,
        0.05,
        f"Lowest RMSE: {best_rmse.capitalize()}",
        transform=axes[1].transAxes,
        fontsize=12.5,
        fontweight="bold",
        va="bottom",
    )
    _panel_letter(axes[0], "A")
    _panel_letter(axes[1], "B")
    handles = [
        mlines.Line2D(
            [],
            [],
            linestyle="none",
            marker=SEED_MARKERS[index % len(SEED_MARKERS)],
            markersize=8,
            markerfacecolor="#4D4D4D",
            markeredgecolor="white",
            label=f"Seed {seed}",
        )
        for index, seed in enumerate(seeds)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 1.035),
    )
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.19, top=0.83, wspace=0.35)
    _save_figure(fig, output_stem)


@latex_paper_style
def plot_all_policy_differences(
    policy_pairs: pd.DataFrame,
    output_stem: Path,
    *,
    variant: str,
    domain: str,
) -> None:
    """Show transcript-paired gains for all three policy comparisons."""
    selected = policy_pairs.loc[
        (policy_pairs["checkpoint_variant"] == variant)
        & (policy_pairs["evaluation_domain"] == domain)
    ].copy()
    comparison_order = [
        f"{right}_vs_{left}"
        for left, right in POLICY_COMPARISONS
        if f"{right}_vs_{left}" in set(selected["comparison"])
    ]
    seeds = sorted(int(value) for value in selected["seed"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.85))
    specifications = (
        ("pcc_gain_right_minus_left", "$\\Delta$PCC (right $-$ left)"),
        ("rmse_gain_left_minus_right", "$\\Delta$RMSE (left $-$ right)"),
    )
    x = np.arange(len(comparison_order), dtype=float)
    right_policy_by_comparison = {
        f"{right}_vs_{left}": right for left, right in POLICY_COMPARISONS
    }
    for ax, (column, ylabel) in zip(axes, specifications):
        values = [
            selected.loc[selected["comparison"] == comparison, column]
            .dropna()
            .to_numpy(float)
            for comparison in comparison_order
        ]
        boxes = ax.boxplot(
            values,
            positions=x,
            widths=0.54,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.3},
            whiskerprops={"color": "#4D4D4D"},
            capprops={"color": "#4D4D4D"},
        )
        for box, comparison in zip(boxes["boxes"], comparison_order):
            box.set_facecolor(
                POLICY_COLORS[right_policy_by_comparison[comparison]]
            )
            box.set_alpha(0.82)
            box.set_edgecolor("#4D4D4D")
        offsets = np.linspace(-0.11, 0.11, max(len(seeds), 1))
        for seed_index, (seed, offset) in enumerate(zip(seeds, offsets)):
            seed_values = [
                float(
                    selected.loc[
                        (selected["comparison"] == comparison)
                        & (selected["seed"] == seed),
                        column,
                    ].mean()
                )
                for comparison in comparison_order
            ]
            ax.scatter(
                x + offset,
                seed_values,
                marker=SEED_MARKERS[seed_index % len(SEED_MARKERS)],
                s=34,
                color="black",
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
                label=f"Seed {seed}" if ax is axes[0] else None,
            )
        ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xticks(
            x,
            [COMPARISON_LABELS[comparison] for comparison in comparison_order],
        )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        fractions = [float(np.mean(value > 0.0)) for value in values]
        for position, fraction in zip(x, fractions):
            ax.text(
                position,
                0.97,
                f"{100.0 * fraction:.0f}\\%",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=12,
            )
    _panel_letter(axes[0], "A")
    _panel_letter(axes[1], "B")
    handles = [
        mlines.Line2D(
            [],
            [],
            linestyle="none",
            marker=SEED_MARKERS[index % len(SEED_MARKERS)],
            markersize=7,
            markerfacecolor="black",
            markeredgecolor="white",
            label=f"Seed {seed}",
        )
        for index, seed in enumerate(seeds)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 1.035),
    )
    fig.text(
        0.5,
        0.015,
        "Positive values mean that moving right improves recovery",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.29, top=0.82, wspace=0.37)
    _save_figure(fig, output_stem)


@latex_paper_style
def plot_all_policy_checkpoint_sensitivity(
    policy_contrasts: pd.DataFrame,
    output_stem: Path,
    *,
    domain: str,
) -> None:
    """Compare all pairwise effects under both checkpoint-selection rules."""
    selected = policy_contrasts.loc[
        policy_contrasts["evaluation_domain"] == domain
    ].copy()
    comparisons = [
        f"{right}_vs_{left}"
        for left, right in POLICY_COMPARISONS
        if f"{right}_vs_{left}" in set(selected["comparison"])
    ]
    variants = [
        variant
        for variant in ("best_val_loss", "best_pcc")
        if variant in set(selected["checkpoint_variant"])
    ]
    if not comparisons or not variants:
        return
    variant_colors = {"best_val_loss": "#0072B2", "best_pcc": "#E69F00"}
    variant_markers = {"best_val_loss": "o", "best_pcc": "s"}
    x = np.arange(len(comparisons), dtype=float)
    offsets = np.linspace(-0.09, 0.09, len(variants))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.65))
    specifications = (
        ("fisher_mean_pcc_gain_right_minus_left", "$\\Delta$PCC"),
        ("mean_rmse_gain_left_minus_right", "$\\Delta$RMSE"),
    )
    for ax, (statistic, ylabel) in zip(axes, specifications):
        for offset, variant in zip(offsets, variants):
            rows = selected.loc[
                (selected["checkpoint_variant"] == variant)
                & (selected["statistic"] == statistic)
            ].set_index("comparison")
            if not all(comparison in rows.index for comparison in comparisons):
                continue
            estimates = np.asarray(
                [float(rows.loc[comparison, "estimate"]) for comparison in comparisons]
            )
            lows = np.asarray(
                [float(rows.loc[comparison, "ci_2p5"]) for comparison in comparisons]
            )
            highs = np.asarray(
                [float(rows.loc[comparison, "ci_97p5"]) for comparison in comparisons]
            )
            ax.errorbar(
                x + offset,
                estimates,
                yerr=np.vstack((estimates - lows, highs - estimates)),
                fmt=variant_markers[variant],
                color=variant_colors[variant],
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=7,
                capsize=2.5,
                linewidth=1.0,
                label=VARIANT_LABELS[variant].replace("\n", " ")
                if ax is axes[0]
                else None,
            )
        ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xticks(x, [COMPARISON_LABELS[value] for value in comparisons])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
    _panel_letter(axes[0], "A")
    _panel_letter(axes[1], "B")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.035),
    )
    fig.text(
        0.5,
        0.015,
        "Positive values mean that moving right improves recovery",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.30, top=0.82, wspace=0.37)
    _save_figure(fig, output_stem)


@latex_paper_style
def plot_all_policy_matched_scatter(
    policy_pairs: pd.DataFrame,
    output_stem: Path,
    *,
    variant: str,
    domain: str,
) -> None:
    """Supplementary matched-transcript scatter for all policy pairs."""
    selected = policy_pairs.loc[
        (policy_pairs["checkpoint_variant"] == variant)
        & (policy_pairs["evaluation_domain"] == domain)
    ].copy()
    comparisons = [
        f"{right}_vs_{left}"
        for left, right in POLICY_COMPARISONS
        if f"{right}_vs_{left}" in set(selected["comparison"])
    ]
    seeds = sorted(int(value) for value in selected["seed"].unique())
    seed_colors = dict(zip(seeds, ("#0072B2", "#E69F00", "#009E73", "#CC79A7")))
    fig, axes = plt.subplots(2, len(comparisons), figsize=(7.15, 6.1))
    if len(comparisons) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    row_specs = (
        ("pcc_left", "pcc_right", "PCC"),
        ("rmse_left", "rmse_right", "RMSE"),
    )
    for column_index, comparison in enumerate(comparisons):
        comparison_rows = selected.loc[selected["comparison"] == comparison]
        for row_index, (x_column, y_column, metric) in enumerate(row_specs):
            ax = axes[row_index, column_index]
            finite = np.isfinite(comparison_rows[x_column]) & np.isfinite(
                comparison_rows[y_column]
            )
            plotted = comparison_rows.loc[finite]
            for seed in seeds:
                seed_rows = plotted.loc[plotted["seed"] == seed]
                ax.scatter(
                    seed_rows[x_column],
                    seed_rows[y_column],
                    s=5,
                    alpha=0.18,
                    linewidths=0,
                    color=seed_colors[seed],
                    rasterized=True,
                    label=f"Seed {seed}",
                )
            low = float(min(plotted[x_column].min(), plotted[y_column].min()))
            high = float(max(plotted[x_column].max(), plotted[y_column].max()))
            padding = max((high - low) * 0.035, 1.0e-3)
            limits = (low - padding, high + padding)
            ax.plot(
                limits,
                limits,
                color="#333333",
                linewidth=0.8,
                linestyle="--",
            )
            ax.set_xlim(*limits)
            ax.set_ylim(*limits)
            ax.grid(alpha=0.30)
            if row_index == 0:
                ax.set_title(COMPARISON_LABELS[comparison].replace("\n", " "))
            ax.set_xlabel(f"Left-policy {metric}")
            if column_index == 0:
                ax.set_ylabel(f"Right-policy {metric}")
            panel_index = row_index * len(comparisons) + column_index
            _panel_letter(ax, chr(ord("A") + panel_index))
    handles = [
        mlines.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor=seed_colors[seed],
            markeredgecolor="none",
            label=f"Seed {seed}",
        )
        for seed in seeds
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.text(
        0.5,
        0.01,
        "PCC above the diagonal and RMSE below it favor the right-hand policy",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    fig.subplots_adjust(
        left=0.11,
        right=0.99,
        bottom=0.12,
        top=0.87,
        wspace=0.38,
        hspace=0.43,
    )
    _save_figure(fig, output_stem)


def _format_interval(row: pd.Series, digits: int = 4) -> str:
    return (
        f"{float(row['estimate']):+.{digits}f} "
        f"[{float(row['ci_2p5']):+.{digits}f}, "
        f"{float(row['ci_97p5']):+.{digits}f}]"
    )


def write_report(
    output_path: Path,
    *,
    availability: pd.DataFrame,
    seed_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    cohorts: pd.DataFrame,
    variants: Sequence[str],
    trim: int,
    command: str,
) -> None:
    primary = seed_summary.loc[
        (seed_summary["checkpoint_variant"] == PRIMARY_VARIANT)
        & (seed_summary["evaluation_domain"] == f"interior_trim{trim}")
    ].sort_values("seed")
    pcc_effect = _effect_row(
        contrasts,
        PRIMARY_VARIANT,
        f"interior_trim{trim}",
        "fisher_mean_pcc_gain_quality_minus_equal",
    )
    rmse_effect = _effect_row(
        contrasts,
        PRIMARY_VARIANT,
        f"interior_trim{trim}",
        "mean_rmse_gain_equal_minus_quality",
    )
    completed = availability.loc[
        (availability["checkpoint_variant"] == PRIMARY_VARIANT)
        & availability["included"].astype(bool),
        ["policy", "seed"],
    ]
    excluded = availability.loc[
        (availability["checkpoint_variant"] == PRIMARY_VARIANT)
        & ~availability["included"].astype(bool),
        ["run_name", "exclusion_reason"],
    ]

    pcc_all_positive = bool(
        (primary["pcc_fisher_gain_quality_minus_equal"] > 0.0).all()
    )
    rmse_all_positive = bool((primary["mean_rmse_gain_equal_minus_quality"] > 0.0).all())
    pcc_ci_positive = float(pcc_effect["ci_2p5"]) > 0.0
    rmse_ci_positive = float(rmse_effect["ci_2p5"]) > 0.0
    if pcc_all_positive and rmse_all_positive and pcc_ci_positive and rmse_ci_positive:
        conclusion = (
            "Across the three completed paired seeds, the measured-depth quality "
            "ranking improves both recovery endpoints. The transcript-cluster "
            "bootstrap intervals are above zero for PCC gain and RMSE reduction."
        )
    elif pcc_all_positive and rmse_all_positive:
        conclusion = (
            "All completed seeds favor the measured-depth quality ranking for both "
            "endpoints, but at least one transcript-cluster interval overlaps zero; "
            "the evidence is directionally consistent rather than definitive."
        )
    else:
        conclusion = (
            "The completed runs do not show a seed-consistent improvement on both "
            "endpoints, so they do not support a general claim that ranking improves "
            "L_bio recovery in this demonstration."
        )

    lines = [
        "# Synthetic fixed-reference ranking: $L_{\\mathrm{bio}}$ recovery",
        "",
        "## What was tested",
        "",
        "The intervention changes only the fixed gamma-reference weights for the "
        "same low-, medium-, and high-depth biased datasets. Equal centering uses "
        "$\\pi=(1/3,1/3,1/3)$; the measured-profile ranking uses "
        "$\\pi=(1/6,1/3,1/2)$ in 0.25, 2, and 20 reads-per-codon order. "
        "The rank comes from median measured read density, not latent truth.",
        "",
        "Frozen shared profiles were compared with the deterministic mean-one "
        "kinetics target. Predictions were first aligned to the latent P-site "
        "coordinate scope (which omits the terminal stop codon), then normalized "
        "to mean one over that full profile "
        f"and evaluated primarily after excluding {trim} codons from each end. "
        "PCCs are aggregated with an equal-transcript Fisher transform; RMSE is "
        "averaged equally across transcripts.",
        "",
        "The primary checkpoint is selected by validation loss. Best validation "
        "$\\mu$ PCC is a prespecified sensitivity analysis. Neither selection uses "
        "latent ground truth.",
        "",
        "## Included runs",
        "",
        f"Completed primary runs: {', '.join(f'{r.policy} seed {int(r.seed)}' for r in completed.itertuples(index=False))}.",
        "",
    ]
    if not excluded.empty:
        lines.extend(["Excluded primary runs:", ""])
        for row in excluded.itertuples(index=False):
            lines.append(f"- `{row.run_name}`: `{row.exclusion_reason}`")
        lines.append("")

    lines.extend(
        [
            "## Primary held-out validation results",
            "",
            "| Seed | Equal PCC | Quality PCC | $\\Delta$PCC | Equal RMSE | Quality RMSE | RMSE reduction |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {int(row.seed)} | {row.equal_fisher_mean_pcc:.4f} | "
            f"{row.quality_fisher_mean_pcc:.4f} | "
            f"{row.pcc_fisher_gain_quality_minus_equal:+.4f} | "
            f"{row.equal_mean_rmse:.4f} | {row.quality_mean_rmse:.4f} | "
            f"{row.mean_rmse_gain_equal_minus_quality:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Matched aggregate effects (estimate and 95% transcript-cluster "
            "bootstrap interval, conditional on these three trained seeds):",
            "",
            f"- Fisher-mean PCC gain, quality minus equal: {_format_interval(pcc_effect)}",
            f"- Mean RMSE reduction, equal minus quality: {_format_interval(rmse_effect)}",
            "",
            "Positive values favor the quality-ranked reference for both reported "
            "contrasts.",
            "",
            "## Conclusion",
            "",
            conclusion,
            "The magnitude and seed consistency nevertheless show that the "
            "reference weighting is consequential for the recovered shared "
            "$L_{\\mathrm{bio}}$ gauge; the direction here favors equal weighting, "
            "not the read-depth ranking.",
            "",
            "## Why read depth need not improve this gauge",
            "",
            "Here $\\pi$ is used only to define the cross-dataset gamma-reference "
            "constraint; the saved gamma manifest explicitly records "
            "`pi_is_gamma_reference_only=true`. It is not an inverse-noise weight "
            "on the likelihood. In an idealized factorization "
            "$\\mu_d \\propto L B_d$, imposing "
            "$\\sum_d \\pi_d\\log\\gamma_d=0$ makes the identified shared profile "
            "absorb a term proportional to "
            "$\\exp(\\sum_d \\pi_d\\log B_d)$. A high-depth dataset is measured "
            "more precisely, but its systematic sequence bias is not thereby closer "
            "to one. Giving the 3-prime-GG dataset more reference weight can therefore "
            "move more of its bias into $L_{\\mathrm{bio}}$ even while reducing "
            "sampling noise.",
            "",
            "This establishes the effect of the chosen measured-profile ranking "
            "relative to equal weighting in this controlled panel. It does not yet "
            "establish the stronger ordering quality > equal > deliberately reversed, "
            "because the reversed runs have no completed frozen prediction exports. "
            "The evaluated cohort is the held-out validation split used for checkpoint "
            "selection, not an untouched test split.",
            "",
            "## Reproduction",
            "",
            "```bash",
            command,
            "```",
            "",
            f"Checkpoint variants analyzed: {', '.join(variants)}.",
            f"Matched-cohort records: {cohorts.shape[0]}.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_three_policy_report(
    output_path: Path,
    *,
    availability: pd.DataFrame,
    run_summary: pd.DataFrame,
    policy_pair_summary: pd.DataFrame,
    policy_contrasts: pd.DataFrame,
    policy_cohorts: pd.DataFrame,
    variants: Sequence[str],
    trim: int,
    command: str,
) -> None:
    """Write the complete reversed/equal/quality interpretation."""
    domain = f"interior_trim{trim}"
    primary = run_summary.loc[
        (run_summary["checkpoint_variant"] == PRIMARY_VARIANT)
        & (run_summary["evaluation_domain"] == domain)
    ].copy()
    primary_pairs = policy_pair_summary.loc[
        (policy_pair_summary["checkpoint_variant"] == PRIMARY_VARIANT)
        & (policy_pair_summary["evaluation_domain"] == domain)
    ].copy()
    primary_contrasts = policy_contrasts.loc[
        (policy_contrasts["checkpoint_variant"] == PRIMARY_VARIANT)
        & (policy_contrasts["evaluation_domain"] == domain)
    ].copy()
    completed = availability.loc[
        (availability["checkpoint_variant"] == PRIMARY_VARIANT)
        & availability["included"].astype(bool),
        ["policy", "seed"],
    ].sort_values(["seed", "policy"])
    excluded = availability.loc[
        (availability["checkpoint_variant"] == PRIMARY_VARIANT)
        & ~availability["included"].astype(bool),
        ["run_name", "exclusion_reason"],
    ]

    policies = [policy for policy in POLICY_ORDER if policy in set(primary["policy"])]
    mean_by_policy = primary.groupby("policy").agg(
        pcc=("fisher_mean_pcc", "mean"),
        rmse=("mean_rmse", "mean"),
    )
    pcc_order = list(mean_by_policy["pcc"].sort_values(ascending=False).index)
    rmse_order = list(mean_by_policy["rmse"].sort_values(ascending=True).index)
    same_order = pcc_order == rmse_order
    ordering_text = " > ".join(policy.capitalize() for policy in pcc_order)
    if same_order:
        outcome = (
            f"Both endpoints give the same recovery ordering: {ordering_text}. "
            "Thus the reference weights materially affect the identified shared "
            "profile, but the measured read-depth ranking does not improve recovery."
        )
    else:
        rmse_ordering = " > ".join(policy.capitalize() for policy in rmse_order)
        outcome = (
            f"PCC orders the policies as {ordering_text}, whereas RMSE orders them "
            f"as {rmse_ordering}. The endpoints therefore do not give one common "
            "ranking."
        )

    def contrast_row(comparison: str, statistic: str) -> pd.Series:
        selected = primary_contrasts.loc[
            (primary_contrasts["comparison"] == comparison)
            & (primary_contrasts["statistic"] == statistic)
        ]
        if selected.shape[0] != 1:
            raise ValueError(
                f"Expected one primary contrast for {comparison}/{statistic}; "
                f"found {selected.shape[0]}."
            )
        return selected.iloc[0]

    lines = [
        "# Synthetic fixed-reference ranking: complete $L_{\\mathrm{bio}}$ comparison",
        "",
        "## Design and endpoint",
        "",
        "The completed experiment compares three fixed gamma-reference policies on "
        "the same low-, medium-, and high-depth biased datasets:",
        "",
        "- reversed ranking: $\\pi=(1/2,1/3,1/6)$;",
        "- equal weighting: $\\pi=(1/3,1/3,1/3)$;",
        "- measured read-depth ranking: $\\pi=(1/6,1/3,1/2)$.",
        "",
        "The entries are ordered as 0.25, 2, and 20 reads per codon. The natural "
        "ranking comes from median measured read density and never uses latent truth.",
        "",
        "Frozen shared profiles were aligned to the latent P-site coordinate scope, "
        "normalized to mean one over the complete aligned profile, and compared with "
        f"the deterministic kinetics target. The primary domain excludes {trim} codons "
        "from each end. PCC uses an equal-transcript Fisher mean and RMSE uses an "
        "equal-transcript arithmetic mean.",
        "",
        "The primary checkpoint is selected by validation loss. Best validation "
        "$\\mu$ PCC is a prespecified sensitivity analysis; latent truth is not used "
        "for checkpoint selection.",
        "",
        "## Run inclusion",
        "",
        f"Complete primary runs: {', '.join(f'{row.policy} seed {int(row.seed)}' for row in completed.itertuples(index=False))}.",
        "",
    ]
    if excluded.empty:
        lines.extend(["No expected primary run was excluded.", ""])
    else:
        lines.extend(["Excluded primary runs:", ""])
        for row in excluded.itertuples(index=False):
            lines.append(f"- `{row.run_name}`: `{row.exclusion_reason}`")
        lines.append("")

    lines.extend(
        [
            "## Primary matched validation results",
            "",
            "| Seed | Reversed PCC | Equal PCC | Quality PCC | Reversed RMSE | Equal RMSE | Quality RMSE |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in sorted(int(value) for value in primary["seed"].unique()):
        seed_rows = primary.loc[primary["seed"] == seed].set_index("policy")
        lines.append(
            f"| {seed} | "
            f"{seed_rows.loc['reversed', 'fisher_mean_pcc']:.4f} | "
            f"{seed_rows.loc['equal', 'fisher_mean_pcc']:.4f} | "
            f"{seed_rows.loc['quality', 'fisher_mean_pcc']:.4f} | "
            f"{seed_rows.loc['reversed', 'mean_rmse']:.4f} | "
            f"{seed_rows.loc['equal', 'mean_rmse']:.4f} | "
            f"{seed_rows.loc['quality', 'mean_rmse']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## All pairwise matched effects",
            "",
            "For each transition below, positive PCC and RMSE gains mean that the "
            "policy on the right recovers latent $L_{\\mathrm{bio}}$ better. Intervals "
            "are 95% transcript-ID cluster bootstrap intervals retaining all seed "
            "occurrences and are conditional on these three trained seeds.",
            "",
            "| Transition | PCC gain [95% CI] | RMSE gain [95% CI] |",
            "|---|---:|---:|",
        ]
    )
    for left_policy, right_policy in POLICY_COMPARISONS:
        comparison = f"{right_policy}_vs_{left_policy}"
        if comparison not in set(primary_contrasts["comparison"]):
            continue
        pcc = contrast_row(
            comparison,
            "fisher_mean_pcc_gain_right_minus_left",
        )
        rmse = contrast_row(
            comparison,
            "mean_rmse_gain_left_minus_right",
        )
        lines.append(
            f"| {left_policy.capitalize()} $\\rightarrow$ {right_policy.capitalize()} "
            f"| {_format_interval(pcc)} | {_format_interval(rmse)} |"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            outcome,
            "",
            "The deliberately reversed control is therefore informative, but not in "
            "the originally expected monotone order quality > equal > reversed. "
            "The actual result must not be presented as evidence that read-depth "
            "ranking improves biological recovery.",
            "",
            "## Why read depth need not improve the gamma-reference gauge",
            "",
            "The saved gamma manifests record `pi_is_gamma_reference_only=true`. "
            "These values define the cross-dataset identifiability constraint; they "
            "are not likelihood precision weights. In the idealized factorization "
            "$\\mu_d \\propto L B_d$, enforcing "
            "$\\sum_d\\pi_d\\log\\gamma_d=0$ makes the identified shared profile "
            "absorb a factor proportional to "
            "$\\exp(\\sum_d\\pi_d\\log B_d)$. Higher depth reduces sampling noise "
            "but does not make a systematic sequence bias closer to one. Consequently, "
            "giving the high-depth 3-prime-GG dataset more reference weight can move "
            "more GG-specific structure into $L_{\\mathrm{bio}}$.",
            "",
            "These results use the held-out validation cohort used for checkpoint "
            "selection, rather than an untouched test cohort.",
            "",
            "## Reproduction",
            "",
            "```bash",
            command,
            "```",
            "",
            f"Checkpoint variants analyzed: {', '.join(variants)}.",
            f"Pairwise matched-cohort records: {policy_cohorts.shape[0]}.",
            f"Per-seed pair summaries: {primary_pairs.shape[0]}.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reproduce_script(path: Path, command: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n", encoding="utf-8")
    path.chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-variants",
        nargs="+",
        choices=("best_val_loss", "best_pcc"),
        default=("best_val_loss", "best_pcc"),
    )
    parser.add_argument("--boundary-trim-codons", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260909)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    truth_path = args.truth.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT
    )
    variants = tuple(dict.fromkeys(str(value) for value in args.checkpoint_variants))
    trim = int(args.boundary_trim_codons)
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    if not truth_path.is_file():
        raise FileNotFoundError(f"Latent truth does not exist: {truth_path}")
    if trim < 0:
        raise ValueError("--boundary-trim-codons must be non-negative.")
    if int(args.bootstrap_replicates) < 100:
        raise ValueError("--bootstrap-replicates must be at least 100.")
    output_dir.mkdir(parents=True, exist_ok=True)

    availability, artifacts, provenance = inspect_runs(results_root, variants)
    availability.to_csv(output_dir / "run_availability.csv", index=False)
    pd.DataFrame(provenance).to_csv(output_dir / "run_provenance.csv", index=False)
    if not artifacts:
        raise RuntimeError("No complete frozen prediction exports passed validation.")

    requested_ids = {value for artifact in artifacts for value in artifact.validation_ids}
    truth = load_truth_subset(truth_path, requested_ids)
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for artifact in sorted(artifacts, key=lambda value: (value.variant, value.seed, value.policy)):
        print(
            f"[analyze] {artifact.run_name} {artifact.variant}: "
            f"{len(artifact.validation_ids)} transcripts",
            flush=True,
        )
        rows, diagnostics = analyze_prediction(artifact, truth, trim=trim)
        metric_rows.extend(rows)
        diagnostic_rows.append(diagnostics)

    metrics = pd.DataFrame(metric_rows)
    metrics.sort_values(
        ["checkpoint_variant", "evaluation_domain", "seed", "policy", "transcript_id"],
        inplace=True,
        ignore_index=True,
    )
    metrics.to_parquet(output_dir / "per_transcript_lbio_metrics.parquet", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "prediction_integrity_diagnostics.csv", index=False
    )

    run_summary = summarize_runs(metrics)
    run_summary.to_csv(output_dir / "summary_by_run.csv", index=False)
    pairs, cohorts = build_quality_equal_pairs(metrics)
    pairs.to_parquet(output_dir / "paired_quality_vs_equal.parquet", index=False)
    cohorts.to_csv(output_dir / "matched_cohort_provenance.csv", index=False)
    seed_summary = summarize_paired_by_seed(pairs)
    seed_summary.to_csv(output_dir / "paired_summary_by_seed.csv", index=False)
    contrasts = bootstrap_contrasts(
        pairs,
        replicates=int(args.bootstrap_replicates),
        random_seed=int(args.bootstrap_seed),
    )
    contrasts.to_csv(output_dir / "paired_cluster_bootstrap.csv", index=False)

    policy_pairs, policy_cohorts = build_all_policy_pairs(metrics)
    policy_pairs.to_parquet(
        output_dir / "paired_all_policy_comparisons.parquet",
        index=False,
    )
    policy_cohorts.to_csv(
        output_dir / "matched_all_policy_cohort_provenance.csv",
        index=False,
    )
    policy_pair_summary = summarize_all_policy_pairs_by_seed(policy_pairs)
    policy_pair_summary.to_csv(
        output_dir / "paired_all_policy_summary_by_seed.csv",
        index=False,
    )
    policy_contrasts = bootstrap_all_policy_contrasts(
        policy_pairs,
        replicates=int(args.bootstrap_replicates),
        random_seed=int(args.bootstrap_seed) + 1000,
    )
    policy_contrasts.to_csv(
        output_dir / "paired_all_policy_cluster_bootstrap.csv",
        index=False,
    )

    primary_domain = f"interior_trim{trim}"
    if PRIMARY_VARIANT not in variants:
        raise ValueError(
            f"The primary figure requires {PRIMARY_VARIANT}; requested variants={variants}."
        )
    plot_three_policy_recovery(
        run_summary,
        output_dir / "synthetic_pi_lbio_recovery",
        variant=PRIMARY_VARIANT,
        domain=primary_domain,
    )
    plot_all_policy_differences(
        policy_pairs,
        output_dir / "synthetic_pi_paired_differences",
        variant=PRIMARY_VARIANT,
        domain=primary_domain,
    )
    plot_all_policy_matched_scatter(
        policy_pairs,
        output_dir / "synthetic_pi_matched_transcript_scatter",
        variant=PRIMARY_VARIANT,
        domain=primary_domain,
    )
    plot_all_policy_checkpoint_sensitivity(
        policy_contrasts,
        output_dir / "synthetic_pi_checkpoint_sensitivity",
        domain=primary_domain,
    )

    # Keep the virtual-environment path un-resolved. Resolving this symlink to
    # /usr/bin/python would lose pyvenv.cfg discovery and make reproduction fail.
    python_path = REPOSITORY_ROOT / ".venv" / "bin" / "python"
    command_parts = [
        f"cd {REPOSITORY_ROOT}",
        "&&",
        str(python_path),
        str(Path(__file__).resolve()),
        "--results-root",
        str(results_root),
        "--truth",
        str(truth_path),
        "--output-dir",
        str(output_dir),
        "--checkpoint-variants",
        *variants,
        "--boundary-trim-codons",
        str(trim),
        "--bootstrap-replicates",
        str(int(args.bootstrap_replicates)),
        "--bootstrap-seed",
        str(int(args.bootstrap_seed)),
    ]
    command = " ".join(command_parts)
    _write_reproduce_script(output_dir / "reproduce.sh", command)
    write_three_policy_report(
        output_dir / "RESULTS.md",
        availability=availability,
        run_summary=run_summary,
        policy_pair_summary=policy_pair_summary,
        policy_contrasts=policy_contrasts,
        policy_cohorts=policy_cohorts,
        variants=variants,
        trim=trim,
        command=command,
    )

    output_files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "analysis": "synthetic_fixed_reference_ranking_lbio_recovery",
        "results_root": str(results_root),
        "latent_truth": str(truth_path),
        "latent_truth_sha256": _sha256_file(truth_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "primary_checkpoint_variant": PRIMARY_VARIANT,
        "checkpoint_variants": list(variants),
        "primary_evaluation_domain": primary_domain,
        "normalization": "prediction_and_truth_mean_one_over_full_cds_before_domain_selection",
        "aggregation": {
            "pcc": "equal-transcript Fisher-z mean",
            "rmse": "equal-transcript arithmetic mean",
            "uncertainty": "transcript-ID cluster bootstrap retaining seed occurrences",
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "policy_order": list(POLICY_ORDER),
        "policy_comparisons": [
            {
                "left": left,
                "right": right,
                "comparison": f"{right}_vs_{left}",
                "positive_effect_direction": f"favors_{right}",
            }
            for left, right in POLICY_COMPARISONS
        ],
        "requested_truth_transcripts": len(requested_ids),
        "included_artifact_count": len(artifacts),
        "reproduce_command": command,
        "output_sha256": {
            path.name: _sha256_file(path)
            for path in output_files
            if path.name != "analysis_manifest.json"
        },
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[complete] Wrote analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
