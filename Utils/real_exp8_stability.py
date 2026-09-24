"""Reusable design primitives for real-data Experiment 8.

Experiment 8 asks whether the sequence-only shared profile becomes stable as
the number of heterogeneous Ribo-seq datasets grows.  Dataset selection is
therefore deliberately *not* a top-quality/rank selection.  Source families
are atomic and candidate subsets are scored by how closely their observed-QC
distribution matches the complete retained dataset collection.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from Utils.real_panel_convergence import (
    infer_source_identifier,
    json_ready,
    load_sequence_metadata,
    load_support_and_stored_weights,
    utc_timestamp,
)
from Utils.reliability_references import transcript_id_hash
from Utils.stratified_transcript_split import (
    assign_reliability_quantile_bins,
    sample_stratified_ids,
)


QUALITY_COLUMNS = (
    "log1p_median_read_density",
    "median_positive_codon_coverage",
    "number_of_eligible_transcripts",
    "median_replica_PCC",
)


def stable_integer_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def prepare_quality_pool(
    quality_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    """Validate eligible QC rows and construct the Experiment-1 source families."""
    frame = quality_table.loc[quality_table["eligible"].astype(bool)].copy()
    frame["dataset_name"] = frame["dataset_name"].astype(str)
    frame = frame.sort_values("dataset_name").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No eligible real datasets are available.")
    if frame["dataset_name"].duplicated().any():
        raise ValueError("The quality table contains duplicate dataset names.")

    if "source_identifier" not in frame.columns:
        frame["source_identifier"] = frame["dataset_name"].map(
            infer_source_identifier
        )
    else:
        frame["source_identifier"] = [
            infer_source_identifier(dataset)
            if pd.isna(source) or not str(source).strip()
            else str(source).strip()
            for dataset, source in zip(
                frame["dataset_name"], frame["source_identifier"], strict=True
            )
        ]

    active_quality_columns: list[str] = []
    imputation: dict[str, float] = {}
    for column in QUALITY_COLUMNS:
        if column not in frame.columns:
            if column == "median_replica_PCC":
                continue
            raise KeyError(f"Quality table is missing required column {column!r}.")
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=np.float64))
        if column == "median_replica_PCC" and int(finite.sum()) < max(2, len(frame) // 2):
            continue
        if not bool(finite.any()):
            raise ValueError(f"Quality variable {column!r} has no finite values.")
        median = float(np.median(values.to_numpy(dtype=np.float64)[finite]))
        frame[column] = values.fillna(median)
        imputation[column] = median
        active_quality_columns.append(column)

    z_columns: list[str] = []
    standardization: dict[str, dict[str, float]] = {}
    for column in active_quality_columns:
        values = frame[column].to_numpy(dtype=np.float64)
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        z_column = f"z__{column}"
        frame[z_column] = (values - mean) / scale
        z_columns.append(z_column)
        standardization[column] = {"mean": mean, "scale": scale}

    families = {
        str(source): sorted(group["dataset_name"].astype(str).tolist())
        for source, group in frame.groupby("source_identifier", sort=True)
    }
    metadata = {
        "quality_columns": active_quality_columns,
        "z_columns": z_columns,
        "missing_value_imputation": imputation,
        "standardization": standardization,
        "source_family_definition": (
            "explicit non-empty source_identifier, otherwise the Experiment-1 "
            "author_year prefix inference"
        ),
        "legacy_scalar_quality_rank_used_for_subset_selection": False,
        "top_quality_selection_used": False,
        "selection_target": "full-pool observed-QC distribution",
    }
    return frame, families, metadata


def feasible_family_sizes(
    family_to_datasets: Mapping[str, Sequence[str]],
    *,
    excluded_families: set[str] | None = None,
) -> list[int]:
    excluded = excluded_families or set()
    reachable = {0}
    for family, datasets in sorted(family_to_datasets.items()):
        if family in excluded:
            continue
        size = len(datasets)
        reachable |= {value + size for value in tuple(reachable)}
    return sorted(reachable)


def nearest_feasible_family_size(
    *,
    requested_size: int,
    family_to_datasets: Mapping[str, Sequence[str]],
    require_disjoint_pair: bool,
    seed: int,
) -> int:
    """Resolve the nearest feasible atomic size after explicit user opt-in."""
    full_count = sum(len(values) for values in family_to_datasets.values())
    candidates = [
        size
        for size in feasible_family_sizes(family_to_datasets)
        if size > 0 and (not require_disjoint_pair or 2 * size <= full_count)
    ]
    candidates.sort(key=lambda size: (abs(size - int(requested_size)), size))
    rng = np.random.default_rng(int(seed))
    for size in candidates:
        if not require_disjoint_pair:
            return int(size)
        for _ in range(200):
            left = _one_exact_family_subset(
                family_to_datasets, target_size=size, rng=rng
            )
            if left is not None and _one_exact_family_subset(
                family_to_datasets,
                target_size=size,
                rng=rng,
                excluded_families=set(left),
            ) is not None:
                return int(size)
    raise ValueError(
        f"No feasible source-family-atomic size near requested N={requested_size}."
    )


def _one_exact_family_subset(
    family_to_datasets: Mapping[str, Sequence[str]],
    *,
    target_size: int,
    rng: np.random.Generator,
    excluded_families: set[str] | None = None,
) -> tuple[str, ...] | None:
    excluded = excluded_families or set()
    families = [name for name in family_to_datasets if name not in excluded]
    rng.shuffle(families)
    # Randomized dynamic programming: the shuffled order gives deterministic
    # diversity across restarts while preserving exact cardinality.
    choices: dict[int, tuple[str, ...]] = {0: ()}
    for family in families:
        size = len(family_to_datasets[family])
        for current, selected in list(choices.items())[::-1]:
            proposed = current + size
            if proposed <= target_size and proposed not in choices:
                choices[proposed] = (*selected, family)
    return choices.get(int(target_size))


def datasets_for_families(
    family_names: Sequence[str], family_to_datasets: Mapping[str, Sequence[str]]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            dataset
            for family in family_names
            for dataset in family_to_datasets[str(family)]
        )
    )


def quality_mismatch(
    selected_datasets: Sequence[str],
    quality_pool: pd.DataFrame,
    *,
    z_columns: Sequence[str],
) -> float:
    """Distance from the full-pool means and quartiles in standardized QC space."""
    indexed = quality_pool.set_index("dataset_name")
    subset = indexed.loc[list(map(str, selected_datasets)), list(z_columns)]
    full = indexed.loc[:, list(z_columns)]
    differences: list[float] = []
    for column in z_columns:
        differences.append(float(subset[column].mean() - full[column].mean()))
        subset_quantiles = subset[column].quantile([0.25, 0.5, 0.75]).to_numpy()
        full_quantiles = full[column].quantile([0.25, 0.5, 0.75]).to_numpy()
        # Means are primary; quartiles are a lower-weight shape diagnostic.
        differences.extend((0.5 * (subset_quantiles - full_quantiles)).tolist())
    return float(np.sqrt(np.mean(np.square(differences))))


def _quality_score_cache(
    quality_pool: pd.DataFrame, z_columns: Sequence[str]
) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    names = quality_pool["dataset_name"].astype(str).tolist()
    values = quality_pool.loc[:, list(z_columns)].to_numpy(dtype=np.float64)
    return (
        {name: index for index, name in enumerate(names)},
        values,
        values.mean(axis=0),
        np.quantile(values, [0.25, 0.5, 0.75], axis=0),
    )


def _cached_quality_mismatch(
    selected_datasets: Sequence[str],
    *,
    index_by_name: Mapping[str, int],
    values: np.ndarray,
    full_mean: np.ndarray,
    full_quantiles: np.ndarray,
) -> float:
    subset = values[[index_by_name[str(name)] for name in selected_datasets]]
    mean_difference = subset.mean(axis=0) - full_mean
    quantile_difference = 0.5 * (
        np.quantile(subset, [0.25, 0.5, 0.75], axis=0) - full_quantiles
    )
    combined = np.concatenate((mean_difference.ravel(), quantile_difference.ravel()))
    return float(np.sqrt(np.mean(np.square(combined))))


def _subset_diversity(
    datasets: Sequence[str], fingerprint_vectors: Mapping[str, np.ndarray] | None
) -> float:
    if fingerprint_vectors is None or len(datasets) < 2:
        return float("nan")
    missing = sorted(set(map(str, datasets)) - set(fingerprint_vectors))
    if missing:
        raise KeyError(f"Fingerprint vectors are missing datasets: {missing[:10]}.")
    distances = [
        float(
            np.linalg.norm(
                fingerprint_vectors[str(left)] - fingerprint_vectors[str(right)]
            )
        )
        for index, left in enumerate(datasets)
        for right in datasets[index + 1 :]
    ]
    return float(np.mean(distances))


def _choose_sampling_candidate(
    candidates: Sequence[tuple], *, sampling_mode: str
) -> tuple:
    if not candidates:
        raise ValueError("No candidate subset was generated.")
    if sampling_mode == "quality_matched":
        # Compare only the observed-QC objective and deterministic dataset/family
        # identifiers.  The last tuple entry is optional fingerprint diversity
        # and is NaN when no fingerprint table is supplied; it must not affect
        # primary quality-matched selection (nor can any legacy rank column).
        return min(candidates, key=lambda candidate: candidate[:-1])
    if sampling_mode not in {"high_diversity", "low_diversity"}:
        raise ValueError(f"Unknown sampling mode {sampling_mode!r}.")
    best_quality = min(float(candidate[0]) for candidate in candidates)
    # The confirmatory modes vary fingerprint diversity only among candidates
    # that remain close to the best observed-QC match.
    quality_limit = best_quality * 1.25 + 0.02
    eligible = [candidate for candidate in candidates if candidate[0] <= quality_limit]
    if any(not math.isfinite(float(candidate[-1])) for candidate in eligible):
        raise ValueError(f"{sampling_mode} requires complete fingerprint vectors.")
    if sampling_mode == "high_diversity":
        return max(eligible, key=lambda candidate: (candidate[-1], -candidate[0]))
    return min(eligible, key=lambda candidate: (candidate[-1], candidate[0]))


def pair_quality_mismatch(
    left: Sequence[str],
    right: Sequence[str],
    quality_pool: pd.DataFrame,
    *,
    z_columns: Sequence[str],
) -> tuple[float, float, float, float]:
    indexed = quality_pool.set_index("dataset_name")
    left_score = quality_mismatch(left, quality_pool, z_columns=z_columns)
    right_score = quality_mismatch(right, quality_pool, z_columns=z_columns)
    mean_delta = (
        indexed.loc[list(left), list(z_columns)].mean()
        - indexed.loc[list(right), list(z_columns)].mean()
    ).to_numpy(dtype=np.float64)
    between = float(np.sqrt(np.mean(np.square(mean_delta))))
    total = float(left_score + right_score + 0.5 * between)
    return total, left_score, right_score, between


def _nearest_sizes_message(
    target: int, family_to_datasets: Mapping[str, Sequence[str]]
) -> str:
    feasible = feasible_family_sizes(family_to_datasets)
    nearest = sorted(feasible, key=lambda value: (abs(value - target), value))[:8]
    return f"target={target}; nearest feasible dataset counts={nearest}"


def select_quality_matched_subset(
    *,
    target_size: int,
    quality_pool: pd.DataFrame,
    family_to_datasets: Mapping[str, Sequence[str]],
    z_columns: Sequence[str],
    seed: int,
    restarts: int = 2000,
    forbidden_subsets: set[tuple[str, ...]] | None = None,
    sampling_mode: str = "quality_matched",
    fingerprint_vectors: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    forbidden = forbidden_subsets or set()
    index_by_name, quality_values, full_mean, full_quantiles = _quality_score_cache(
        quality_pool, z_columns
    )
    candidates: list[tuple[float, tuple[str, ...], tuple[str, ...], float]] = []
    for _ in range(int(restarts)):
        families = _one_exact_family_subset(
            family_to_datasets, target_size=int(target_size), rng=rng
        )
        if families is None:
            break
        datasets = datasets_for_families(families, family_to_datasets)
        if datasets in forbidden:
            continue
        score = _cached_quality_mismatch(
            datasets,
            index_by_name=index_by_name,
            values=quality_values,
            full_mean=full_mean,
            full_quantiles=full_quantiles,
        )
        candidates.append(
            (
                score,
                datasets,
                tuple(sorted(families)),
                _subset_diversity(datasets, fingerprint_vectors),
            )
        )
    if not candidates:
        raise ValueError(
            "No exact source-family-atomic subset could be constructed: "
            + _nearest_sizes_message(int(target_size), family_to_datasets)
        )
    best = _choose_sampling_candidate(candidates, sampling_mode=sampling_mode)
    return {
        "datasets": list(best[1]),
        "source_families": list(best[2]),
        "quality_mismatch": float(best[0]),
        "target_size": int(target_size),
        "actual_size": len(best[1]),
        "selection_fingerprint_diversity_zscored": float(best[3]),
        "sampling_mode": str(sampling_mode),
    }


def select_quality_matched_disjoint_pair(
    *,
    target_size: int,
    quality_pool: pd.DataFrame,
    family_to_datasets: Mapping[str, Sequence[str]],
    z_columns: Sequence[str],
    seed: int,
    restarts: int = 4000,
    sampling_mode: str = "quality_matched",
    fingerprint_vectors: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    index_by_name, quality_values, full_mean, full_quantiles = _quality_score_cache(
        quality_pool, z_columns
    )
    candidates: list[tuple] = []
    for _ in range(int(restarts)):
        left_families = _one_exact_family_subset(
            family_to_datasets, target_size=int(target_size), rng=rng
        )
        if left_families is None:
            break
        right_families = _one_exact_family_subset(
            family_to_datasets,
            target_size=int(target_size),
            rng=rng,
            excluded_families=set(left_families),
        )
        if right_families is None:
            continue
        left = datasets_for_families(left_families, family_to_datasets)
        right = datasets_for_families(right_families, family_to_datasets)
        left_score = _cached_quality_mismatch(
            left,
            index_by_name=index_by_name,
            values=quality_values,
            full_mean=full_mean,
            full_quantiles=full_quantiles,
        )
        right_score = _cached_quality_mismatch(
            right,
            index_by_name=index_by_name,
            values=quality_values,
            full_mean=full_mean,
            full_quantiles=full_quantiles,
        )
        left_mean = quality_values[[index_by_name[name] for name in left]].mean(axis=0)
        right_mean = quality_values[[index_by_name[name] for name in right]].mean(axis=0)
        between = float(np.sqrt(np.mean(np.square(left_mean - right_mean))))
        total = float(left_score + right_score + 0.5 * between)
        pair_diversity = (
            float("nan")
            if fingerprint_vectors is None
            else float(
                np.mean(
                    [
                        _subset_diversity(left, fingerprint_vectors),
                        _subset_diversity(right, fingerprint_vectors),
                    ]
                )
            )
        )
        candidate = (
            total,
            left,
            right,
            tuple(sorted(left_families)),
            tuple(sorted(right_families)),
            left_score,
            right_score,
            between,
            pair_diversity,
        )
        candidates.append(candidate)
    if not candidates:
        raise ValueError(
            "No exact source-family-disjoint A/B pair could be constructed: "
            + _nearest_sizes_message(int(target_size), family_to_datasets)
        )
    best = _choose_sampling_candidate(candidates, sampling_mode=sampling_mode)
    if set(best[1]) & set(best[2]) or set(best[3]) & set(best[4]):
        raise AssertionError("Designated Experiment-8 pair is not disjoint.")
    return {
        "A": {
            "datasets": list(best[1]),
            "source_families": list(best[3]),
            "quality_mismatch": float(best[5]),
        },
        "B": {
            "datasets": list(best[2]),
            "source_families": list(best[4]),
            "quality_mismatch": float(best[6]),
        },
        "joint_quality_objective": float(best[0]),
        "between_subset_quality_mismatch": float(best[7]),
        "selection_pair_fingerprint_diversity_zscored": float(best[8]),
        "sampling_mode": str(sampling_mode),
        "target_size": int(target_size),
    }


def build_experiment_matrix(
    *,
    quality_pool: pd.DataFrame,
    family_to_datasets: Mapping[str, Sequence[str]],
    dataset_sizes: Sequence[int],
    disjoint_pairs: int,
    large_n_subsets: int,
    subset_seed: int,
    candidate_restarts: int = 4000,
    sampling_mode: str = "quality_matched",
    fingerprint_vectors: Mapping[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Construct the default 34-run matrix with overlap-controlled small N."""
    full_count = len(quality_pool)
    z_columns = [column for column in quality_pool if column.startswith("z__")]
    tasks: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for target in map(int, dataset_sizes):
        if target <= 0 or target > full_count:
            raise ValueError(f"Invalid dataset count N={target}; pool size={full_count}.")
        if 2 * target <= full_count:
            for pair_index in range(1, int(disjoint_pairs) + 1):
                pair_id = f"pair{pair_index:02d}"
                selected = select_quality_matched_disjoint_pair(
                    target_size=target,
                    quality_pool=quality_pool,
                    family_to_datasets=family_to_datasets,
                    z_columns=z_columns,
                    seed=stable_integer_seed(subset_seed, target, pair_index),
                    restarts=candidate_restarts,
                    sampling_mode=sampling_mode,
                    fingerprint_vectors=fingerprint_vectors,
                )
                for side in ("A", "B"):
                    run_id = f"real_exp8_N{target:03d}_{pair_id}_{side}"
                    record = {
                        "run_id": run_id,
                        "N": target,
                        "kind": "designated_disjoint_pair",
                        "pair_id": pair_id,
                        "side": side,
                        **selected[side],
                    }
                    tasks.append(record)
                    quality_rows.append(
                        {
                            "run_id": run_id,
                            "N": target,
                            "quality_mismatch": selected[side]["quality_mismatch"],
                            "joint_pair_quality_objective": selected[
                                "joint_quality_objective"
                            ],
                            "between_pair_quality_mismatch": selected[
                                "between_subset_quality_mismatch"
                            ],
                        }
                    )
        elif target < full_count:
            used: set[tuple[str, ...]] = set()
            for subset_index in range(1, int(large_n_subsets) + 1):
                selected = select_quality_matched_subset(
                    target_size=target,
                    quality_pool=quality_pool,
                    family_to_datasets=family_to_datasets,
                    z_columns=z_columns,
                    seed=stable_integer_seed(subset_seed, target, subset_index),
                    restarts=candidate_restarts,
                    forbidden_subsets=used,
                    sampling_mode=sampling_mode,
                    fingerprint_vectors=fingerprint_vectors,
                )
                used.add(tuple(selected["datasets"]))
                run_id = f"real_exp8_N{target:03d}_subset{subset_index:02d}"
                tasks.append(
                    {
                        "run_id": run_id,
                        "N": target,
                        "kind": "large_N_subset",
                        "subset_id": f"subset{subset_index:02d}",
                        **selected,
                    }
                )
                quality_rows.append(
                    {
                        "run_id": run_id,
                        "N": target,
                        "quality_mismatch": selected["quality_mismatch"],
                        "joint_pair_quality_objective": np.nan,
                        "between_pair_quality_mismatch": np.nan,
                    }
                )
        else:
            datasets = tuple(sorted(quality_pool["dataset_name"].astype(str)))
            sources = tuple(sorted(family_to_datasets))
            run_id = f"real_exp8_N{target:03d}_full"
            score = quality_mismatch(datasets, quality_pool, z_columns=z_columns)
            tasks.append(
                {
                    "run_id": run_id,
                    "N": target,
                    "kind": "full_collection",
                    "datasets": list(datasets),
                    "source_families": list(sources),
                    "quality_mismatch": score,
                    "target_size": target,
                    "actual_size": len(datasets),
                }
            )
            quality_rows.append(
                {
                    "run_id": run_id,
                    "N": target,
                    "quality_mismatch": score,
                    "joint_pair_quality_objective": np.nan,
                    "between_pair_quality_mismatch": np.nan,
                }
            )

    for task in tasks:
        if len(task["datasets"]) != int(task["N"]):
            raise AssertionError(f"{task['run_id']} does not have exact target N.")
        observed_sources = {
            str(quality_pool.set_index("dataset_name").at[name, "source_identifier"])
            for name in task["datasets"]
        }
        if observed_sources != set(task["source_families"]):
            raise AssertionError(f"{task['run_id']} violates source-family atomicity.")
    return tasks, pd.DataFrame(quality_rows)


def build_overlap_report(tasks: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full_count = max(int(task["N"]) for task in tasks)
    for left, right in itertools.combinations(tasks, 2):
        left_set = set(map(str, left["datasets"]))
        right_set = set(map(str, right["datasets"]))
        intersection = left_set & right_set
        union = left_set | right_set
        left_sources = set(map(str, left["source_families"]))
        right_sources = set(map(str, right["source_families"]))
        designated = bool(
            left.get("kind") == "designated_disjoint_pair"
            and right.get("kind") == "designated_disjoint_pair"
            and left.get("N") == right.get("N")
            and left.get("pair_id") == right.get("pair_id")
            and left.get("side") != right.get("side")
            and left.get("training_seed") == right.get("training_seed")
        )
        row = {
            "run_a": left["run_id"],
            "run_b": right["run_id"],
            "N_a": len(left_set),
            "N_b": len(right_set),
            "intersection_count": len(intersection),
            "union_count": len(union),
            "overlap_fraction_a": len(intersection) / len(left_set),
            "overlap_fraction_b": len(intersection) / len(right_set),
            "jaccard": len(intersection) / len(union),
            "expected_random_overlap": len(left_set) * len(right_set) / full_count,
            "source_family_intersection_count": len(left_sources & right_sources),
            "is_designated_disjoint_pair": designated,
        }
        if designated and (
            row["intersection_count"] != 0
            or row["source_family_intersection_count"] != 0
        ):
            raise AssertionError("A designated A/B pair overlaps.")
        rows.append(row)
    return pd.DataFrame(rows)


def uniform_reference_weights(dataset_names: Sequence[str]) -> dict[str, float]:
    names = list(map(str, dataset_names))
    if not names or len(names) != len(set(names)):
        raise ValueError("Uniform gamma reference requires distinct dataset names.")
    weight = 1.0 / len(names)
    result = {name: weight for name in names}
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Uniform gamma-reference weights do not sum to one.")
    return result


def build_exp8_transcript_split(
    *,
    experiment_name: str,
    tasks: Sequence[Mapping[str, Any]],
    dataset_mapping: Mapping[str, str],
    sequences_path: str | Path,
    subset_seed: int,
    validation_fraction: float,
    test_fraction: float,
    reliability_bins: int,
    maximum_cds_codons: int | None,
    reused_test_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build one global sequence test set and run-specific measured validation sets."""
    sequence_metadata, length_report = load_sequence_metadata(
        sequences_path, max_cds_codons=maximum_cds_codons
    )
    sequence_ids = set(sequence_metadata.index.astype(str))
    supports, stored_weights, eligibility = load_support_and_stored_weights(
        dataset_mapping, eligible_sequence_ids=sequence_ids
    )
    global_support = {
        transcript_id: sum(transcript_id in supports[name] for name in supports)
        for transcript_id in sequence_ids
    }
    global_candidates = sorted(
        transcript_id
        for transcript_id, support in global_support.items()
        if support >= 2
    )
    if reused_test_ids is None:
        global_reliability = {
            transcript_id: float(
                np.median(
                    [
                        stored_weights[name][transcript_id]
                        for name in supports
                        if transcript_id in stored_weights[name]
                    ]
                )
            )
            for transcript_id in global_candidates
        }
        qbins = assign_reliability_quantile_bins(
            global_reliability, number_of_bins=int(reliability_bins)
        )
        strata = {
            transcript_id: (
                f"qbin_{qbins[transcript_id]:02d}__"
                f"{sequence_metadata.at[transcript_id, 'css_bin']}"
            )
            for transcript_id in global_candidates
        }
        target = int(round(len(global_candidates) * float(test_fraction)))
        if target <= 0 or target >= len(global_candidates):
            raise ValueError("The requested common test fraction is not viable.")
        common_test = sample_stratified_ids(
            candidate_ids=global_candidates,
            stratum_by_transcript=strata,
            target_count=target,
            rng=np.random.default_rng(int(subset_seed)),
        )
        test_source = "global_114_dataset_support_stratified"
    else:
        common_test = sorted(set(map(str, reused_test_ids)))
        missing = sorted(set(common_test) - sequence_ids)
        if missing:
            raise ValueError(
                f"Reused Experiment-1 test IDs fail sequence eligibility: {missing[:10]}."
            )
        test_source = "reused_experiment1_common_test_ids"
    common_test_set = set(common_test)

    panels: dict[str, list[str]] = {}
    train_by_run: dict[str, list[str]] = {}
    validation_by_run: dict[str, list[str]] = {}
    statistics: dict[str, Any] = {}
    for task in tasks:
        run_id = str(task["run_id"])
        selected = list(map(str, task["datasets"]))
        panels[run_id] = selected
        run_support = {
            transcript_id: sum(transcript_id in supports[name] for name in selected)
            for transcript_id in sequence_ids
        }
        measured_candidates = sorted(
            transcript_id
            for transcript_id, support in run_support.items()
            if support >= 2 and transcript_id not in common_test_set
        )
        if len(measured_candidates) < 2:
            raise RuntimeError(f"{run_id} has too few measured split candidates.")
        reliability = {
            transcript_id: float(
                np.median(
                    [
                        stored_weights[name][transcript_id]
                        for name in selected
                        if transcript_id in stored_weights[name]
                    ]
                )
            )
            for transcript_id in measured_candidates
        }
        qbins = assign_reliability_quantile_bins(
            reliability, number_of_bins=int(reliability_bins)
        )
        strata = {
            transcript_id: (
                f"qbin_{qbins[transcript_id]:02d}__"
                f"{sequence_metadata.at[transcript_id, 'css_bin']}"
            )
            for transcript_id in measured_candidates
        }
        validation_target = max(
            1, int(round(len(measured_candidates) * float(validation_fraction)))
        )
        if validation_target >= len(measured_candidates):
            validation_target = len(measured_candidates) - 1
        validation = sample_stratified_ids(
            candidate_ids=measured_candidates,
            stratum_by_transcript=strata,
            target_count=validation_target,
            rng=np.random.default_rng(
                stable_integer_seed(subset_seed, "validation", run_id)
            ),
        )
        validation_set = set(validation)
        training = sorted(set(measured_candidates) - validation_set)
        if set(training) & (validation_set | common_test_set):
            raise AssertionError(f"{run_id} has transcript split leakage.")
        if any(run_support[transcript_id] < 2 for transcript_id in training):
            raise AssertionError(f"{run_id} contains an under-supported training ID.")
        train_by_run[run_id] = training
        validation_by_run[run_id] = validation
        support_values = [run_support[transcript_id] for transcript_id in training]
        statistics[run_id] = {
            "number_of_train_eligible_transcripts": len(training),
            "number_of_validation_transcripts": len(validation),
            "number_of_common_test_sequences": len(common_test),
            "training_support_minimum": min(support_values),
            "training_support_median": float(np.median(support_values)),
            "training_support_maximum": max(support_values),
            "common_test_observed_in_selected_subset": int(
                sum(run_support[transcript_id] > 0 for transcript_id in common_test)
            ),
        }

    return json_ready(
        {
            "manifest_version": 1,
            "experiment_name": str(experiment_name),
            "created_at_utc": utc_timestamp(),
            "random_seed": int(subset_seed),
            "source_sequences_path": str(sequences_path),
            "maximum_cds_codons": maximum_cds_codons,
            "sequence_eligibility_report": length_report,
            "minimum_usable_datasets_per_training_or_validation_transcript": 2,
            "dataset_pair_eligibility_reports": eligibility,
            "panels": panels,
            "panel_train_eligible_ids": train_by_run,
            "panel_validation_ids": validation_by_run,
            "common_validation_ids": validation_by_run[
                sorted(validation_by_run)[0]
            ],
            "common_test_ids": common_test,
            "common_test_source": test_source,
            "common_test_requires_support_in_every_subset": False,
            "common_test_prediction_mode": "sequence_only_shared_profile",
            "panel_support_statistics": statistics,
            "fold_id_hashes": {
                "test": transcript_id_hash(common_test),
                "validation_by_panel": {
                    run_id: transcript_id_hash(ids)
                    for run_id, ids in validation_by_run.items()
                },
            },
            "assertions": {
                "all_training_transcripts_have_at_least_two_selected_datasets": True,
                "all_validation_transcripts_have_at_least_two_selected_datasets": True,
                "common_test_excluded_from_every_training_fold": True,
                "common_test_ids_identical_for_every_run": True,
            },
        }
    )


def summarize_subset_quality(
    tasks: Sequence[Mapping[str, Any]], quality_pool: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        subset = quality_pool.loc[
            quality_pool["dataset_name"].isin(task["datasets"])
        ]
        row: dict[str, Any] = {
            "run_id": task["run_id"],
            "N": int(task["N"]),
            "kind": task["kind"],
            "quality_mismatch": float(task["quality_mismatch"]),
            "number_of_source_families": int(subset["source_identifier"].nunique()),
            "legacy_scalar_quality_rank_used": False,
        }
        for column in QUALITY_COLUMNS:
            if column in subset:
                values = pd.to_numeric(subset[column], errors="coerce")
                row[f"mean_{column}"] = float(values.mean())
                row[f"median_{column}"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows)


def assert_experiment_design(
    *,
    tasks: Sequence[Mapping[str, Any]],
    common_split: Mapping[str, Any],
    overlap_report: pd.DataFrame,
) -> dict[str, bool]:
    checks = {
        "every_selected_subset_has_exact_target_N": all(
            len(task["datasets"]) == int(task["N"]) for task in tasks
        ),
        "every_training_transcript_has_at_least_two_datasets": bool(
            common_split["assertions"][
                "all_training_transcripts_have_at_least_two_selected_datasets"
            ]
        ),
        "common_test_never_enters_training": all(
            not (
                set(common_split["common_test_ids"])
                & set(common_split["panel_train_eligible_ids"][task["run_id"]])
            )
            for task in tasks
        ),
        "designated_pairs_dataset_disjoint": bool(
            (
                overlap_report.loc[
                    overlap_report["is_designated_disjoint_pair"],
                    "intersection_count",
                ]
                == 0
            ).all()
        ),
        "designated_pairs_source_disjoint": bool(
            (
                overlap_report.loc[
                    overlap_report["is_designated_disjoint_pair"],
                    "source_family_intersection_count",
                ]
                == 0
            ).all()
        ),
        "uniform_gamma_pi": all(
            math.isclose(
                sum(uniform_reference_weights(task["datasets"]).values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for task in tasks
        ),
        "legacy_quality_rank_not_used_for_subset_selection": True,
    }
    return checks
