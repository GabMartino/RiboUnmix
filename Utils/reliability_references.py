"""Fit and apply split-aware transcript-reliability references.

The filtered weighted real-data parquets contain pair-local ``read_density``
and ``coverage`` values.  This module keeps the established SNR/coverage
equation but lets an experiment fit its two dataset-level reference statistics
on an explicit set of training transcript IDs:

    depth_score = sqrt(D) / (sqrt(D) + sqrt(tau_d))
    raw_weight  = 0.70 * depth_score + 0.30 * coverage
    weight      = raw_weight / training_raw_weight_median

The resulting manifest can be consumed by the real-data datamodule.  Held-out
rows are transformed with frozen references and never influence their fit.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


WEIGHTING_MODE = "snr_depth_coverage"
DEPTH_WEIGHT = 0.70
COVERAGE_WEIGHT = 0.30
MANIFEST_VERSION = 1


def transcript_id_hash(transcript_ids: Iterable[str]) -> str:
    """Return a stable SHA-256 digest for a set of transcript IDs."""
    payload = "\n".join(sorted(set(map(str, transcript_ids)))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_pair_statistics(frame: pd.DataFrame, *, dataset_name: str) -> pd.DataFrame:
    required = {"id", "read_density", "coverage"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            f"Dataset {dataset_name!r} is missing reliability columns "
            f"{sorted(missing)}. The independent-panel experiment requires "
            "filtered weighted artifacts from the production preprocessor; "
            "raw profiles are not an accepted fallback."
        )

    result = frame.loc[:, ["id", "read_density", "coverage"]].copy()
    result["id"] = result["id"].astype(str)
    if bool(result["id"].duplicated().any()):
        duplicate = str(result.loc[result["id"].duplicated(), "id"].iloc[0])
        raise ValueError(
            f"Dataset {dataset_name!r} contains duplicate transcript {duplicate!r}."
        )
    result["read_density"] = pd.to_numeric(
        result["read_density"], errors="coerce"
    ).astype("float64")
    result["coverage"] = pd.to_numeric(result["coverage"], errors="coerce").astype(
        "float64"
    )
    density = result["read_density"].to_numpy(dtype=np.float64, copy=False)
    coverage = result["coverage"].to_numpy(dtype=np.float64, copy=False)
    invalid = (
        ~np.isfinite(density)
        | (density <= 0.0)
        | ~np.isfinite(coverage)
        | (coverage <= 0.0)
        | (coverage > 1.0)
    )
    if bool(invalid.any()):
        index = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"Dataset {dataset_name!r}, transcript {result.iloc[index]['id']!r}: "
            "read_density must be finite and positive and coverage must be in "
            f"(0, 1]; got density={density[index]}, coverage={coverage[index]}."
        )
    return result


def materialize_observed_pair_statistics(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
) -> pd.DataFrame:
    """Ensure pair-local density and coverage columns are available.

    Older datamodule revisions imported this compatibility helper when the
    weighted parquet did not yet materialize ``read_density`` and ``coverage``.
    Current production artifacts contain both columns, in which case the frame
    is returned unchanged.  The fallback derives the same quantities from the
    stored consensus ``ribo`` profile and never changes the profile or weight.
    """
    present = {"read_density", "coverage"}.intersection(frame.columns)
    if present == {"read_density", "coverage"}:
        return frame
    if present:
        missing = {"read_density", "coverage"} - present
        raise KeyError(
            f"Dataset {dataset_name!r} has only part of the observed-statistics "
            f"columns; missing {sorted(missing)}."
        )
    if "ribo" not in frame.columns:
        raise KeyError(
            f"Dataset {dataset_name!r} cannot derive read_density/coverage: "
            "the frame has no ribo profile column."
        )

    densities: list[float] = []
    coverages: list[float] = []
    ids = frame["id"].astype(str) if "id" in frame.columns else frame.index.astype(str)
    for transcript_id, profile_cell in zip(ids, frame["ribo"], strict=True):
        profile = np.asarray(profile_cell, dtype=np.float64)
        if (
            profile.ndim != 1
            or profile.size == 0
            or not np.isfinite(profile).all()
            or bool((profile < 0.0).any())
        ):
            raise ValueError(
                f"Dataset {dataset_name!r}, transcript {transcript_id!r}: "
                "invalid consensus profile while deriving reliability statistics."
            )
        total_reads = float(profile.sum(dtype=np.float64))
        coverage = float(np.count_nonzero(profile > 0.0) / profile.size)
        if total_reads <= 0.0 or coverage <= 0.0:
            raise ValueError(
                f"Dataset {dataset_name!r}, transcript {transcript_id!r}: "
                "consensus profile has no positive information while deriving "
                f"reliability statistics (total_reads={total_reads:g}, "
                f"coverage={coverage:g})."
            )
        densities.append(total_reads / float(profile.size))
        coverages.append(coverage)

    result = frame.copy()
    result["read_density"] = np.asarray(densities, dtype=np.float64)
    result["coverage"] = np.asarray(coverages, dtype=np.float64)
    return result


def raw_snr_depth_coverage_weight(
    read_density: np.ndarray | pd.Series,
    coverage: np.ndarray | pd.Series,
    *,
    depth_reference_tau: float,
    depth_weight: float = DEPTH_WEIGHT,
    coverage_weight: float = COVERAGE_WEIGHT,
) -> np.ndarray:
    """Evaluate the established positive raw reliability equation."""
    tau = float(depth_reference_tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("depth_reference_tau must be finite and strictly positive.")
    depth_weight = float(depth_weight)
    coverage_weight = float(coverage_weight)
    if (
        not math.isfinite(depth_weight)
        or not math.isfinite(coverage_weight)
        or depth_weight < 0.0
        or coverage_weight < 0.0
        or not math.isclose(depth_weight + coverage_weight, 1.0, abs_tol=1.0e-12)
    ):
        raise ValueError(
            "Reliability mixture coefficients must be finite, non-negative, "
            "and sum to one."
        )

    density = np.asarray(read_density, dtype=np.float64)
    breadth = np.asarray(coverage, dtype=np.float64)
    if density.shape != breadth.shape:
        raise ValueError("read_density and coverage must have the same shape.")
    if (
        not np.isfinite(density).all()
        or bool((density <= 0.0).any())
        or not np.isfinite(breadth).all()
        or bool((breadth <= 0.0).any())
        or bool((breadth > 1.0).any())
    ):
        raise ValueError("Invalid read-density or coverage value.")
    depth_score = np.sqrt(density) / (np.sqrt(density) + math.sqrt(tau))
    raw = depth_weight * depth_score + coverage_weight * breadth
    if not np.isfinite(raw).all() or bool((raw <= 0.0).any()):
        raise FloatingPointError("Reliability equation produced a non-positive value.")
    return raw


def fit_dataset_reliability_reference(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    training_transcript_ids: Iterable[str],
    depth_weight: float = DEPTH_WEIGHT,
    coverage_weight: float = COVERAGE_WEIGHT,
) -> dict[str, Any]:
    """Fit one dataset's references using only the supplied training IDs."""
    validated = _validated_pair_statistics(frame, dataset_name=dataset_name)
    training_ids = set(map(str, training_transcript_ids))
    if not training_ids:
        raise ValueError(
            f"Dataset {dataset_name!r} received an empty training reference set."
        )
    selected = validated.loc[validated["id"].isin(training_ids)].copy()
    if selected.empty:
        raise ValueError(
            f"Dataset {dataset_name!r} has no rows in the training reference set."
        )
    tau = float(selected["read_density"].median())
    raw = raw_snr_depth_coverage_weight(
        selected["read_density"].to_numpy(dtype=np.float64, copy=False),
        selected["coverage"].to_numpy(dtype=np.float64, copy=False),
        depth_reference_tau=tau,
        depth_weight=depth_weight,
        coverage_weight=coverage_weight,
    )
    normalization_median = float(np.median(raw))
    if not math.isfinite(normalization_median) or normalization_median <= 0.0:
        raise FloatingPointError(
            f"Dataset {dataset_name!r} produced an invalid normalization median."
        )
    selected_ids = selected["id"].astype(str).tolist()
    return {
        "dataset_name": str(dataset_name),
        "weighting_mode": WEIGHTING_MODE,
        "depth_weight": float(depth_weight),
        "coverage_weight": float(coverage_weight),
        "depth_reference_tau": tau,
        "normalization_reference_median": normalization_median,
        "reference_split": "training_only",
        "training_reference_transcript_count": int(len(selected_ids)),
        "training_reference_transcript_id_hash": transcript_id_hash(selected_ids),
    }


def apply_dataset_reliability_reference(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    reference: Mapping[str, Any],
) -> np.ndarray:
    """Apply one frozen dataset reference to every row in ``frame``."""
    validated = _validated_pair_statistics(frame, dataset_name=dataset_name)
    mode = str(reference.get("weighting_mode", ""))
    if mode != WEIGHTING_MODE:
        raise ValueError(
            f"Unsupported frozen weighting mode {mode!r}; expected {WEIGHTING_MODE!r}."
        )
    raw = raw_snr_depth_coverage_weight(
        validated["read_density"].to_numpy(dtype=np.float64, copy=False),
        validated["coverage"].to_numpy(dtype=np.float64, copy=False),
        depth_reference_tau=float(reference["depth_reference_tau"]),
        depth_weight=float(reference.get("depth_weight", DEPTH_WEIGHT)),
        coverage_weight=float(reference.get("coverage_weight", COVERAGE_WEIGHT)),
    )
    denominator = float(reference["normalization_reference_median"])
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(
            f"Dataset {dataset_name!r} has an invalid frozen normalization median."
        )
    weights = raw / denominator
    if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
        raise FloatingPointError(
            f"Dataset {dataset_name!r} produced invalid frozen reliability weights."
        )
    return weights.astype(np.float32, copy=False)


def load_reliability_reference_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("manifest_version", -1)) != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported reliability-reference manifest version in {path}."
        )
    references = payload.get("datasets")
    if not isinstance(references, dict) or not references:
        raise ValueError(f"Reliability-reference manifest {path} has no datasets.")
    return payload
