"""Lightweight transcript-stratification primitives shared by experiments."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd


def css_count(value: Any) -> int:
    """Count conserved-stalling-site entries in the project's accepted forms."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"", "[]", "nan", "None", "null"}:
            return 0
        try:
            return css_count(json.loads(stripped))
        except Exception:
            stripped = stripped.strip("[]()")
            return len([item for item in stripped.split(",") if item.strip()])
    array = np.asarray(value)
    if array.ndim == 0:
        try:
            if pd.isna(array.item()):
                return 0
            return int(bool(array.item()))
        except Exception:
            return 0
    if array.dtype == bool:
        return int(array.sum())
    count = 0
    for item in array.reshape(-1):
        try:
            if pd.isna(item):
                continue
        except Exception:
            pass
        count += 1
    return int(count)


def css_bin(number_of_sites: int) -> str:
    if number_of_sites <= 0:
        return "css_0"
    if number_of_sites == 1:
        return "css_1"
    if number_of_sites <= 3:
        return "css_2_3"
    return "css_4_plus"


def assign_reliability_quantile_bins(
    scores: dict[str, float],
    *,
    number_of_bins: int,
) -> dict[str, int]:
    """Assign deterministic rank-quantile bins without fragile qcut edges."""
    number_of_bins = int(number_of_bins)
    if number_of_bins < 2:
        raise ValueError("number_of_bins must be at least two.")
    if not scores:
        return {}
    ordered = sorted(scores, key=lambda tid: (float(scores[tid]), str(tid)))
    effective_bins = min(number_of_bins, len(ordered))
    return {
        tid: min((rank * effective_bins) // len(ordered), effective_bins - 1)
        for rank, tid in enumerate(ordered)
    }


def sample_stratified_ids(
    *,
    candidate_ids: Sequence[str],
    stratum_by_transcript: dict[str, str],
    target_count: int,
    rng: np.random.Generator,
) -> list[str]:
    """Sample an exact count proportionally across transcript strata."""
    candidates = sorted(set(map(str, candidate_ids)))
    target_count = int(target_count)
    if target_count <= 0:
        return []
    if target_count >= len(candidates):
        raise ValueError(
            f"target_count={target_count} must be below candidates={len(candidates)}."
        )
    strata: dict[str, list[str]] = defaultdict(list)
    for transcript_id in candidates:
        if transcript_id not in stratum_by_transcript:
            raise KeyError(f"Missing stratum for transcript {transcript_id!r}.")
        strata[str(stratum_by_transcript[transcript_id])].append(transcript_id)
    total = len(candidates)
    ideal = {
        stratum: target_count * len(ids) / total for stratum, ids in strata.items()
    }
    quotas = {stratum: int(np.floor(value)) for stratum, value in ideal.items()}
    remaining = target_count - sum(quotas.values())
    order = sorted(
        strata,
        key=lambda stratum: (-(ideal[stratum] - quotas[stratum]), stratum),
    )
    for stratum in order:
        if remaining <= 0:
            break
        if quotas[stratum] < len(strata[stratum]):
            quotas[stratum] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(f"Could not allocate {remaining} stratified IDs.")
    selected: list[str] = []
    for stratum in sorted(strata):
        ids = np.asarray(sorted(strata[stratum]), dtype=object)
        permutation = rng.permutation(len(ids))
        selected.extend(map(str, ids[permutation[: quotas[stratum]]]))
    if len(selected) != target_count:
        raise RuntimeError(
            f"Selected {len(selected)} IDs, expected exactly {target_count}."
        )
    return sorted(selected)
