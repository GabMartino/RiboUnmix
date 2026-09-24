"""Reusable design primitives for independent real-dataset panel experiments."""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from Utils.reliability_references import (
    MANIFEST_VERSION as RELIABILITY_MANIFEST_VERSION,
    fit_dataset_reliability_reference,
    transcript_id_hash,
)
from Utils.publication_plot_style import latex_paper_style
from Utils.stratified_transcript_split import (
    assign_reliability_quantile_bins,
    css_bin,
    css_count,
    sample_stratified_ids,
)


PANEL_MANIFEST_VERSION = 1
COMMON_SPLIT_MANIFEST_VERSION = 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_source_identifier(dataset_name: str) -> str:
    """Return the publication/source family encoded by a dataset name.

    Real HEK dataset names consistently begin with ``author_year`` and may
    append condition or replicate labels.  Keeping this prefix atomic prevents
    related conditions/replicates from leaking across nominally independent
    panels.  Names without a year-like second token fall back to their first
    token.  A quality table may still provide a more specific non-empty
    ``source_identifier`` explicitly.
    """
    normalized = str(dataset_name).strip()
    if not normalized:
        raise ValueError("Dataset names must be non-empty.")
    parts = normalized.split("_")
    if len(parts) > 1 and any(character.isdigit() for character in parts[1]):
        return "_".join(parts[:2])
    return parts[0]


def json_ready(value: Any) -> Any:
    """Recursively convert NumPy/Pandas values and non-finite floats for JSON."""
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_dataset_mapping(path: str | Path) -> dict[str, str]:
    """Load the ordered dataset-name to parquet-path mapping from YAML."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mapping = payload.get("dataset_path") if isinstance(payload, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Dataset configuration {config_path} has no dataset_path map.")
    result = {str(name): str(dataset_path) for name, dataset_path in mapping.items()}
    if len(result) != len(mapping):
        raise ValueError(f"Dataset configuration {config_path} has duplicate names.")
    return result


def load_sequence_metadata(
    sequences_path: str | Path,
    *,
    max_cds_codons: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve the same complete-transcript maximum-CDS eligibility rule."""
    available = set(pq.read_schema(sequences_path).names)
    sequence_column = "codons" if "codons" in available else "ref"
    if sequence_column not in available:
        raise KeyError(f"{sequences_path} contains neither codons nor ref.")
    css_column = (
        "conserved_stalling_sites"
        if "conserved_stalling_sites" in available
        else "css"
    )
    if css_column not in available:
        raise KeyError(f"{sequences_path} has no conserved-stalling-site column.")
    frame = pd.read_parquet(
        sequences_path,
        columns=["transcript_id", sequence_column, css_column],
    )
    frame["transcript_id"] = frame["transcript_id"].astype(str)
    if bool(frame["transcript_id"].duplicated().any()):
        duplicate = str(
            frame.loc[frame["transcript_id"].duplicated(), "transcript_id"].iloc[0]
        )
        raise ValueError(f"Sequence table contains duplicate transcript {duplicate!r}.")
    lengths = frame[sequence_column].map(len).astype(int)
    if bool((lengths <= 0).any()):
        raise ValueError("Sequence table contains an empty CDS.")
    retained = pd.Series(True, index=frame.index)
    if max_cds_codons is not None:
        maximum = int(max_cds_codons)
        if maximum < 1:
            raise ValueError("max_cds_codons must be null or positive.")
        retained = lengths <= maximum
    output = pd.DataFrame(
        {
            "transcript_id": frame.loc[retained, "transcript_id"].astype(str),
            "transcript_length": lengths.loc[retained].astype(int),
            "css_count": frame.loc[retained, css_column].map(css_count).astype(int),
        }
    ).set_index("transcript_id", drop=True)
    output["css_bin"] = output["css_count"].map(css_bin)
    report = {
        "max_cds_codons": max_cds_codons,
        "input_transcripts": int(len(frame)),
        "eligible_transcripts": int(len(output)),
        "removed_transcripts": int((~retained).sum()),
        "longest_input_cds_codons": int(lengths.max()),
        "longest_eligible_cds_codons": int(output["transcript_length"].max()),
    }
    return output.sort_index(), report


def _replica_pair_correlations(replica_cell: Any) -> list[float]:
    replicas = [np.asarray(profile, dtype=np.float64) for profile in replica_cell]
    if len(replicas) < 2:
        return []
    if any(profile.ndim != 1 or profile.size == 0 for profile in replicas):
        raise ValueError("Replica profiles must be non-empty one-dimensional arrays.")
    if len({profile.size for profile in replicas}) != 1:
        raise ValueError("Replica profiles have inconsistent lengths.")
    if any(
        not np.isfinite(profile).all() or bool((profile < 0.0).any())
        for profile in replicas
    ):
        raise ValueError("Replica profiles contain non-finite or negative counts.")
    correlations: list[float] = []
    for left, right in itertools.combinations(replicas, 2):
        mask = np.isfinite(left) & np.isfinite(right)
        if int(mask.sum()) < 2:
            continue
        left_valid = left[mask]
        right_valid = right[mask]
        if np.var(left_valid) <= 0.0 or np.var(right_valid) <= 0.0:
            continue
        correlation = float(np.corrcoef(left_valid, right_valid)[0, 1])
        if math.isfinite(correlation):
            correlations.append(correlation)
    return correlations


def _positive_weight_rows(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate stored weights and remove historical zero-weight exclusions.

    A negative or non-finite value is corrupt.  A zero value has a different,
    documented meaning in the compact historical weighted parquets: that
    physical row is not a usable transcript--dataset observation.  Returning
    only positive rows makes the logical retained set identical for panel
    summaries, split support, and the production datamodule.
    """
    required = {"id", "weight"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            f"Dataset {dataset_name!r} is missing weighted-artifact columns "
            f"{sorted(missing)}."
        )
    result = frame.copy()
    result["id"] = result["id"].astype(str)
    if bool(result["id"].duplicated().any()):
        duplicate = str(result.loc[result["id"].duplicated(), "id"].iloc[0])
        raise ValueError(
            f"Dataset {dataset_name!r} contains duplicate transcript {duplicate!r}."
        )
    weights = pd.to_numeric(result["weight"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    invalid = ~np.isfinite(weights) | (weights < 0.0)
    if bool(invalid.any()):
        index = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"Dataset {dataset_name!r}, transcript {result.iloc[index]['id']!r}: "
            "stored weight must be finite and non-negative; got "
            f"{weights[index]}."
        )
    result["weight"] = weights
    positive = weights > 0.0
    report = {
        "physical_rows": int(len(result)),
        "positive_weight_rows": int(np.count_nonzero(positive)),
        "zero_weight_exclusion_rows": int(np.count_nonzero(weights == 0.0)),
    }
    return result.loc[positive].reset_index(drop=True), report


def _positive_information_rows(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove legacy zero-information profiles after strict shape validation."""
    if "ribo" not in frame.columns:
        raise KeyError(f"Dataset {dataset_name!r} is missing the 'ribo' column.")
    usable = np.ones(len(frame), dtype=bool)
    for index, (transcript_id, cell) in enumerate(
        zip(frame["id"], frame["ribo"], strict=True)
    ):
        profile = np.asarray(cell, dtype=np.float64)
        if (
            profile.ndim != 1
            or profile.size == 0
            or not np.isfinite(profile).all()
            or bool((profile < 0.0).any())
        ):
            raise ValueError(
                f"Dataset {dataset_name!r}, transcript {str(transcript_id)!r}: "
                "stored consensus must be a non-empty, finite, non-negative "
                "one-dimensional profile."
            )
        usable[index] = bool(profile.sum(dtype=np.float64) > 0.0)
    report = {
        "positive_weight_rows_checked": int(len(frame)),
        "positive_weight_zero_information_rows": int(np.count_nonzero(~usable)),
        "usable_positive_weight_rows": int(np.count_nonzero(usable)),
    }
    return frame.loc[usable].reset_index(drop=True), report


def _validated_replica_consensus(
    consensus_cell: Any,
    replica_cell: Any,
    *,
    dataset_name: str,
    transcript_id: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Validate a positive-weight pair and return its replica mean profile."""
    stored_consensus = np.asarray(consensus_cell, dtype=np.float64)
    if (
        stored_consensus.ndim != 1
        or stored_consensus.size == 0
        or not np.isfinite(stored_consensus).all()
        or bool((stored_consensus < 0.0).any())
    ):
        raise ValueError(
            f"Dataset {dataset_name!r}, transcript {transcript_id!r}: invalid "
            "stored consensus profile."
        )
    replicas = [np.asarray(profile, dtype=np.float64) for profile in replica_cell]
    if not replicas:
        raise ValueError(
            f"Dataset {dataset_name!r}, transcript {transcript_id!r}: no raw replicas."
        )
    if any(
        profile.ndim != 1
        or profile.size == 0
        or not np.isfinite(profile).all()
        or bool((profile < 0.0).any())
        for profile in replicas
    ):
        raise ValueError(
            f"Dataset {dataset_name!r}, transcript {transcript_id!r}: invalid "
            "raw replica profile."
        )
    lengths = {int(profile.size) for profile in replicas}
    if lengths != {int(stored_consensus.size)}:
        raise ValueError(
            f"Dataset {dataset_name!r}, transcript {transcript_id!r}: stored "
            "consensus and raw-replica lengths disagree."
        )
    consensus = np.mean(np.stack(replicas, axis=0), axis=0, dtype=np.float64)
    total_reads = float(consensus.sum(dtype=np.float64))
    coverage = float(np.count_nonzero(consensus > 0.0) / consensus.size)
    if total_reads <= 0.0 or coverage <= 0.0:
        raise ValueError(
            f"Dataset {dataset_name!r}, transcript {transcript_id!r}: a positive-"
            "weight row must have positive replica-consensus reads and coverage; "
            f"got total_reads={total_reads:g}, coverage={coverage:g}."
        )
    return consensus, replicas


def summarize_dataset_quality(
    *,
    dataset_name: str,
    dataset_path: str | Path,
    eligible_sequence_ids: set[str],
) -> dict[str, Any]:
    """Compute observed-only quality summaries for one real dataset."""
    path = Path(dataset_path)
    required = {"id", "ribo", "ribo_cds_replicas", "weight"}
    available = set(pq.read_schema(path).names)
    missing = required - available
    if missing:
        raise KeyError(
            f"{path}: missing required weighted replica-aware columns "
            f"{sorted(missing)}; raw artifacts are not accepted"
        )
    columns = sorted(required)
    if "replica_ids" in available:
        columns.append("replica_ids")
    frame = pd.read_parquet(path, columns=columns)
    frame, weight_report = _positive_weight_rows(
        frame, dataset_name=str(dataset_name)
    )
    frame, information_report = _positive_information_rows(
        frame, dataset_name=str(dataset_name)
    )
    positive_rows_before_sequence_filter = int(len(frame))
    frame = frame.loc[frame["id"].isin(eligible_sequence_ids)].reset_index(
        drop=True
    )
    if frame.empty:
        raise ValueError(
            "no positive-weight rows overlap the sequence-eligible transcript universe"
        )

    lengths: list[int] = []
    densities: list[float] = []
    coverages: list[float] = []
    zero_fractions: list[float] = []
    total_reads = 0.0
    replica_counts: list[int] = []
    replica_correlations: list[float] = []
    for transcript_id, consensus_cell, replica_cell in zip(
        frame["id"], frame["ribo"], frame["ribo_cds_replicas"], strict=True
    ):
        consensus, replicas = _validated_replica_consensus(
            consensus_cell,
            replica_cell,
            dataset_name=str(dataset_name),
            transcript_id=str(transcript_id),
        )
        density = float(consensus.sum(dtype=np.float64) / consensus.size)
        coverage = float(np.count_nonzero(consensus > 0.0) / consensus.size)
        lengths.append(int(consensus.size))
        densities.append(density)
        coverages.append(coverage)
        zero_fractions.append(float(np.mean(consensus == 0.0)))
        total_reads += float(consensus.sum(dtype=np.float64))
        replica_counts.append(len(replicas))
        replica_correlations.extend(_replica_pair_correlations(replicas))

    source_identifier = infer_source_identifier(str(dataset_name))
    source_stat = path.stat()
    median_density = float(np.median(densities))
    stored_weights = frame["weight"].to_numpy(dtype=np.float64, copy=False)
    return {
        "dataset_name": str(dataset_name),
        "dataset_path": str(path),
        "source_file_size_bytes": int(source_stat.st_size),
        "source_file_mtime_ns": int(source_stat.st_mtime_ns),
        "eligible": True,
        "exclusion_reason": None,
        "observed_quality_statistics_source": (
            "arithmetic_mean_of_weighted_artifact_raw_replicas"
        ),
        "weighted_artifact_schema_variant": (
            "audit_rich_weighted"
            if {"read_density", "coverage"}.issubset(available)
            else "compact_legacy_weighted"
        ),
        "physical_rows_in_weighted_artifact": weight_report["physical_rows"],
        "positive_weight_rows_in_weighted_artifact": weight_report[
            "positive_weight_rows"
        ],
        "zero_weight_exclusion_rows_in_weighted_artifact": weight_report[
            "zero_weight_exclusion_rows"
        ],
        "positive_weight_zero_information_rows_excluded": information_report[
            "positive_weight_zero_information_rows"
        ],
        "positive_weight_rows_removed_by_sequence_eligibility": int(
            positive_rows_before_sequence_filter - len(frame)
        ),
        "number_of_eligible_transcripts": int(len(frame)),
        "median_read_density": median_density,
        "log1p_median_read_density": float(np.log1p(median_density)),
        "median_positive_codon_coverage": float(np.median(coverages)),
        "median_replica_PCC": (
            float(np.median(replica_correlations))
            if replica_correlations
            else None
        ),
        "number_of_replicas": float(np.median(replica_counts)),
        "minimum_number_of_replicas": int(min(replica_counts)),
        "maximum_number_of_replicas": int(max(replica_counts)),
        "transcripts_with_replica_PCC": int(
            sum(len(list(cell)) >= 2 for cell in frame["ribo_cds_replicas"])
        ),
        "median_zero_fraction": float(np.median(zero_fractions)),
        "median_profile_length": float(np.median(lengths)),
        "total_reads": total_reads,
        "median_stored_reliability_weight": float(np.median(stored_weights)),
        "minimum_stored_reliability_weight": float(np.min(stored_weights)),
        "maximum_stored_reliability_weight": float(np.max(stored_weights)),
        "source_identifier": source_identifier,
    }


def build_dataset_quality_table(
    *,
    dataset_mapping: Mapping[str, str],
    sequences_path: str | Path,
    max_cds_codons: int | None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    """Summarize all candidates while explicitly retaining exclusion reasons."""
    sequence_metadata, length_report = load_sequence_metadata(
        sequences_path, max_cds_codons=max_cds_codons
    )
    eligible_sequence_ids = set(sequence_metadata.index.astype(str))
    rows: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    for dataset_name, dataset_path in dataset_mapping.items():
        try:
            path = Path(dataset_path)
            if not path.exists():
                raise FileNotFoundError(f"file does not exist: {path}")
            row = summarize_dataset_quality(
                dataset_name=str(dataset_name),
                dataset_path=path,
                eligible_sequence_ids=eligible_sequence_ids,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            excluded[str(dataset_name)] = reason
            source_path = Path(dataset_path)
            source_stat = source_path.stat() if source_path.exists() else None
            row = {
                "dataset_name": str(dataset_name),
                "dataset_path": str(dataset_path),
                "source_file_size_bytes": (
                    int(source_stat.st_size) if source_stat is not None else None
                ),
                "source_file_mtime_ns": (
                    int(source_stat.st_mtime_ns) if source_stat is not None else None
                ),
                "eligible": False,
                "exclusion_reason": reason,
                "observed_quality_statistics_source": None,
                "number_of_eligible_transcripts": 0,
                "median_read_density": None,
                "log1p_median_read_density": None,
                "median_positive_codon_coverage": None,
                "median_replica_PCC": None,
                "number_of_replicas": None,
                "minimum_number_of_replicas": None,
                "maximum_number_of_replicas": None,
                "transcripts_with_replica_PCC": 0,
                "median_zero_fraction": None,
                "median_profile_length": None,
                "total_reads": None,
                "source_identifier": None,
            }
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table) != len(dataset_mapping):
        raise RuntimeError("Quality table did not retain every candidate dataset.")
    sequence_path = Path(sequences_path).expanduser().resolve()
    sequence_stat = sequence_path.stat()
    table["quality_manifest_version"] = 1
    table["quality_sequences_path"] = str(sequence_path)
    table["quality_sequences_file_size_bytes"] = int(sequence_stat.st_size)
    table["quality_sequences_file_mtime_ns"] = int(sequence_stat.st_mtime_ns)
    table["quality_max_cds_codons"] = (
        None if max_cds_codons is None else int(max_cds_codons)
    )
    return table, excluded, length_report


def _rank_quantile_bin(values: pd.Series, *, number_of_bins: int) -> pd.Series:
    """Rank-based bins that cannot fail on duplicate numerical boundaries."""
    count = len(values)
    effective_bins = max(1, min(int(number_of_bins), count))
    order = sorted(range(count), key=lambda index: (float(values.iloc[index]), index))
    labels = np.empty(count, dtype=np.int64)
    for rank, index in enumerate(order):
        labels[index] = min((rank * effective_bins) // count, effective_bins - 1)
    return pd.Series(labels, index=values.index, dtype="int64")


def target_panel_sizes(number_of_datasets: int, number_of_panels: int) -> list[int]:
    if number_of_panels < 2:
        raise ValueError("number_of_panels must be at least two.")
    if number_of_datasets < number_of_panels:
        raise ValueError("There must be at least one dataset per panel.")
    quotient, remainder = divmod(int(number_of_datasets), int(number_of_panels))
    return [quotient + int(index < remainder) for index in range(number_of_panels)]


def deterministic_stratified_panel_assignment(
    quality_table: pd.DataFrame,
    *,
    number_of_panels: int = 4,
    seed: int = 42,
    number_of_quantile_bins: int = 4,
    additional_balance_objective: Callable[[pd.DataFrame], float] | None = None,
    additional_swap_budget: int = 2000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build exact-size, source-atomic panels with stratified feature balance.

    All datasets sharing ``source_identifier`` are assigned as one indivisible
    unit, so conditions or replicates from one publication cannot occur in two
    panels. Depth and coverage rank bins remain the primary dataset-level
    strata. A deterministic capacity-aware greedy assignment balances those
    strata and the observed quality features while meeting the exact target
    dataset counts. Equal-size source-group swaps then improve balance without
    breaking source atomicity or panel sizes.
    """
    retained = quality_table.loc[quality_table["eligible"].astype(bool)].copy()
    retained = retained.sort_values("dataset_name").reset_index(drop=True)
    if retained.empty:
        raise ValueError("No eligible datasets are available for panel assignment.")
    required = {
        "dataset_name",
        "log1p_median_read_density",
        "median_positive_codon_coverage",
        "number_of_eligible_transcripts",
    }
    missing = required - set(retained.columns)
    if missing:
        raise KeyError(f"Quality table is missing panel features {sorted(missing)}.")
    for column in required - {"dataset_name"}:
        retained[column] = pd.to_numeric(retained[column], errors="coerce")
        if not np.isfinite(retained[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Panel feature {column!r} contains non-finite values.")

    if "source_identifier" not in retained.columns:
        retained["source_identifier"] = retained["dataset_name"].map(
            infer_source_identifier
        )
    else:
        source_values: list[str] = []
        for dataset_name, raw_source in zip(
            retained["dataset_name"], retained["source_identifier"], strict=True
        ):
            if pd.isna(raw_source) or not str(raw_source).strip():
                source_values.append(infer_source_identifier(str(dataset_name)))
            else:
                source_values.append(str(raw_source).strip())
        retained["source_identifier"] = source_values

    targets = target_panel_sizes(len(retained), int(number_of_panels))
    retained["depth_quantile_bin"] = _rank_quantile_bin(
        retained["log1p_median_read_density"],
        number_of_bins=number_of_quantile_bins,
    )
    retained["coverage_quantile_bin"] = _rank_quantile_bin(
        retained["median_positive_codon_coverage"],
        number_of_bins=number_of_quantile_bins,
    )
    retained["eligible_transcript_quantile_bin"] = _rank_quantile_bin(
        retained["number_of_eligible_transcripts"],
        number_of_bins=number_of_quantile_bins,
    )

    replica_numeric = pd.to_numeric(
        retained.get("median_replica_PCC", pd.Series(np.nan, index=retained.index)),
        errors="coerce",
    )
    enough_replica_quality = int(replica_numeric.notna().sum()) >= max(
        2 * int(number_of_panels), int(math.ceil(0.5 * len(retained)))
    )
    if enough_replica_quality:
        observed = replica_numeric.dropna()
        observed_bins = _rank_quantile_bin(
            observed, number_of_bins=number_of_quantile_bins
        )
        retained["replica_category"] = "replica_missing"
        retained.loc[observed_bins.index, "replica_category"] = observed_bins.map(
            lambda value: f"replica_q{int(value)}"
        )
    else:
        retained["replica_category"] = replica_numeric.notna().map(
            {True: "replica_available", False: "replica_missing"}
        )

    retained["joint_stratum"] = (
        "depth_q"
        + retained["depth_quantile_bin"].astype(str)
        + "__coverage_q"
        + retained["coverage_quantile_bin"].astype(str)
        + "__"
        + retained["replica_category"].astype(str)
    )

    feature_columns = [
        "log1p_median_read_density",
        "median_positive_codon_coverage",
        "number_of_eligible_transcripts",
    ]
    if replica_numeric.notna().sum() >= 2:
        retained["median_replica_PCC_balancing"] = replica_numeric.fillna(
            float(replica_numeric.median())
        )
        feature_columns.append("median_replica_PCC_balancing")
    feature_matrix = retained[feature_columns].to_numpy(dtype=np.float64)
    means = feature_matrix.mean(axis=0)
    scales = feature_matrix.std(axis=0, ddof=0)
    scales[scales <= 0.0] = 1.0
    standardized = (feature_matrix - means) / scales

    source_groups: dict[str, np.ndarray] = {
        str(source): group.index.to_numpy(dtype=np.int64)
        for source, group in retained.groupby("source_identifier", sort=True)
    }
    largest_source_size = max(len(indexes) for indexes in source_groups.values())
    if largest_source_size > max(targets):
        largest_sources = sorted(
            (
                (source, len(indexes))
                for source, indexes in source_groups.items()
                if len(indexes) == largest_source_size
            ),
            key=lambda item: item[0],
        )
        raise ValueError(
            "A source group is larger than every target panel and cannot remain "
            f"atomic: sources={largest_sources}, targets={targets}."
        )

    strata = sorted(retained["joint_stratum"].astype(str).unique())
    stratum_index = {stratum: index for index, stratum in enumerate(strata)}
    total_stratum_counts = (
        retained["joint_stratum"]
        .astype(str)
        .value_counts()
        .reindex(strata, fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    expected_stratum_counts = np.outer(
        np.asarray(targets, dtype=np.float64) / float(len(retained)),
        total_stratum_counts,
    )

    rng = np.random.default_rng(int(seed))
    source_tiebreak = dict(
        zip(sorted(source_groups), rng.random(len(source_groups)), strict=True)
    )
    retained["seeded_tiebreak"] = retained["source_identifier"].map(
        source_tiebreak
    )
    group_records: dict[str, dict[str, Any]] = {}
    for source, indexes in source_groups.items():
        stratum_vector = np.zeros(len(strata), dtype=np.int64)
        for stratum, count in (
            retained.loc[indexes, "joint_stratum"].astype(str).value_counts().items()
        ):
            stratum_vector[stratum_index[str(stratum)]] = int(count)
        group_records[source] = {
            "source": source,
            "indexes": indexes,
            "size": int(len(indexes)),
            "feature_sum": standardized[indexes].sum(axis=0),
            "stratum_counts": stratum_vector,
        }

    processing_order = sorted(
        group_records,
        key=lambda source: (
            -int(group_records[source]["size"]),
            -float(
                np.linalg.norm(
                    group_records[source]["feature_sum"]
                    / float(group_records[source]["size"])
                )
            ),
            float(source_tiebreak[source]),
            source,
        ),
    )

    panel_sizes = np.zeros(number_of_panels, dtype=np.int64)
    panel_feature_sums = np.zeros((number_of_panels, standardized.shape[1]))
    panel_stratum_counts = np.zeros(
        (number_of_panels, len(strata)), dtype=np.int64
    )
    source_assignments: dict[str, int] = {}

    def balance_objective(
        feature_sums: np.ndarray,
        stratum_counts: np.ndarray,
    ) -> float:
        feature_component = float(
            np.mean(
                np.square(
                    feature_sums
                    / np.asarray(targets, dtype=np.float64).reshape(-1, 1)
                )
            )
        )
        stratum_component = float(
            np.mean(
                np.square(
                    (stratum_counts - expected_stratum_counts)
                    / np.sqrt(expected_stratum_counts + 1.0)
                )
            )
        )
        return feature_component + 0.35 * stratum_component

    panel_rotation = int(seed) % number_of_panels
    for source in processing_order:
        record = group_records[source]
        group_size = int(record["size"])
        candidates: list[tuple[float, float, float, int, int]] = []
        for panel in range(number_of_panels):
            if panel_sizes[panel] + group_size > targets[panel]:
                continue
            prospective_feature_sums = panel_feature_sums.copy()
            prospective_stratum_counts = panel_stratum_counts.copy()
            prospective_feature_sums[panel] += record["feature_sum"]
            prospective_stratum_counts[panel] += record["stratum_counts"]
            prospective_fill = (
                float(panel_sizes[panel] + group_size) / float(targets[panel])
            )
            maximum_fill = max(
                prospective_fill,
                *(
                    float(panel_sizes[other]) / float(targets[other])
                    for other in range(number_of_panels)
                    if other != panel
                ),
            )
            candidates.append(
                (
                    maximum_fill,
                    balance_objective(
                        prospective_feature_sums,
                        prospective_stratum_counts,
                    ),
                    prospective_fill,
                    (panel - panel_rotation) % number_of_panels,
                    panel,
                )
            )
        if not candidates:
            remaining = (np.asarray(targets) - panel_sizes).tolist()
            raise RuntimeError(
                "Source-atomic panel assignment reached an infeasible capacity "
                f"state at source={source!r}, group_size={group_size}, "
                f"remaining_capacities={remaining}."
            )
        selected_panel = min(candidates)[-1]
        source_assignments[source] = selected_panel
        panel_sizes[selected_panel] += group_size
        panel_feature_sums[selected_panel] += record["feature_sum"]
        panel_stratum_counts[selected_panel] += record["stratum_counts"]
        panel_rotation = (selected_panel + 1) % number_of_panels

    if panel_sizes.tolist() != targets:
        raise RuntimeError(
            f"Panel capacities were not met: observed={panel_sizes.tolist()}, targets={targets}."
        )

    # Swap equal-size source groups only. This preserves exact panel sizes and
    # cannot split a source family while improving the joint balance objective.
    swap_iterations = 0
    maximum_swap_iterations = 500
    while swap_iterations < maximum_swap_iterations:
        current_objective = balance_objective(
            panel_feature_sums, panel_stratum_counts
        )
        best_swap: tuple[float, str, str] | None = None
        sources = sorted(group_records)
        for left_position, left_source in enumerate(sources):
            left_panel = source_assignments[left_source]
            left_record = group_records[left_source]
            for right_source in sources[left_position + 1 :]:
                right_panel = source_assignments[right_source]
                right_record = group_records[right_source]
                if (
                    left_panel == right_panel
                    or left_record["size"] != right_record["size"]
                ):
                    continue
                prospective_feature_sums = panel_feature_sums.copy()
                prospective_stratum_counts = panel_stratum_counts.copy()
                prospective_feature_sums[left_panel] += (
                    right_record["feature_sum"] - left_record["feature_sum"]
                )
                prospective_feature_sums[right_panel] += (
                    left_record["feature_sum"] - right_record["feature_sum"]
                )
                prospective_stratum_counts[left_panel] += (
                    right_record["stratum_counts"] - left_record["stratum_counts"]
                )
                prospective_stratum_counts[right_panel] += (
                    left_record["stratum_counts"] - right_record["stratum_counts"]
                )
                delta = (
                    balance_objective(
                        prospective_feature_sums,
                        prospective_stratum_counts,
                    )
                    - current_objective
                )
                candidate = (delta, left_source, right_source)
                if best_swap is None or candidate < best_swap:
                    best_swap = candidate
        if best_swap is None or best_swap[0] >= -1.0e-12:
            break
        _, left_source, right_source = best_swap
        left_panel = source_assignments[left_source]
        right_panel = source_assignments[right_source]
        left_record = group_records[left_source]
        right_record = group_records[right_source]
        panel_feature_sums[left_panel] += (
            right_record["feature_sum"] - left_record["feature_sum"]
        )
        panel_feature_sums[right_panel] += (
            left_record["feature_sum"] - right_record["feature_sum"]
        )
        panel_stratum_counts[left_panel] += (
            right_record["stratum_counts"] - left_record["stratum_counts"]
        )
        panel_stratum_counts[right_panel] += (
            left_record["stratum_counts"] - right_record["stratum_counts"]
        )
        source_assignments[left_source] = right_panel
        source_assignments[right_source] = left_panel
        swap_iterations += 1

    retained["panel_index"] = (
        retained["source_identifier"].map(source_assignments).astype(int) + 1
    )
    retained["panel"] = retained["panel_index"].map(
        lambda value: f"panel_{int(value):02d}"
    )
    retained = retained.sort_values(["panel_index", "dataset_name"]).reset_index(
        drop=True
    )
    assert_panel_partition(retained, expected_datasets=quality_table.loc[
        quality_table["eligible"].astype(bool), "dataset_name"
    ].astype(str).tolist())
    method = {
        "name": "source_atomic_rank_strata_with_seeded_greedy_balance",
        "seed": int(seed),
        "number_of_quantile_bins": int(number_of_quantile_bins),
        "primary_joint_strata": [
            "log1p_median_read_density rank quantile",
            "median_positive_codon_coverage rank quantile",
            "replica quality/availability category",
        ],
        "greedy_balancing_features": feature_columns,
        "target_panel_sizes": targets,
        "source_identifier_column": "source_identifier",
        "source_identifier_fallback": "author_year prefix inferred from dataset_name",
        "source_groups_are_atomic": True,
        "number_of_source_groups": int(len(source_groups)),
        "largest_source_group_size": int(largest_source_size),
        "source_group_size_histogram": {
            str(size): int(count)
            for size, count in sorted(
                Counter(len(indexes) for indexes in source_groups.values()).items()
            )
        },
        "replica_quality_stratified": bool(enough_replica_quality),
        "duplicate_boundary_handling": "rank bins; numerical qcut is not used",
        "equal_size_source_group_swap_iterations": int(swap_iterations),
    }
    if additional_balance_objective is not None:
        retained, refinement = refine_source_atomic_panel_assignment(
            retained, objective=additional_balance_objective, seed=seed,
            proposal_budget=additional_swap_budget,
        )
        method['additional_qc_refinement'] = refinement
    return retained, method


def refine_source_atomic_panel_assignment(
    assignment: pd.DataFrame, *, objective: Callable[[pd.DataFrame], float],
    seed: int, proposal_budget: int,
    feasible: Callable[[pd.DataFrame], bool] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Budgeted QC-objective extension of the existing equal-size source swaps.

    Uses the frozen source identifiers, never a new source-inference rule.
    Strictly improving swaps preserve exact capacities. This limited search
    does not establish optimality or infeasibility of a better partition.
    """
    if proposal_budget < 0:
        raise ValueError('proposal_budget must be nonnegative.')
    result = assignment.copy().reset_index(drop=True)
    assert_panel_partition(result, expected_datasets=result.dataset_name.tolist())
    if feasible is not None and not feasible(result):
        raise ValueError('Initial source-atomic assignment violates the required support constraint.')
    groups = {str(s):g.index.to_numpy() for s,g in result.groupby('source_identifier',sort=True)}
    pairs = [(a,b) for a,b in itertools.combinations(sorted(groups),2) if len(groups[a])==len(groups[b])]
    initial = current = float(objective(result))
    accepted = 0
    support_rejections = 0
    rng = np.random.default_rng(seed)
    for _ in range(proposal_budget if pairs else 0):
        a,b = pairs[int(rng.integers(len(pairs)))]
        ia,ib = groups[a],groups[b]
        pa,pb = result.loc[ia[0],'panel'],result.loc[ib[0],'panel']
        if pa==pb:
            continue
        result.loc[ia,'panel'],result.loc[ib,'panel'] = pb,pa
        if feasible is not None and not feasible(result):
            support_rejections += 1
            result.loc[ia,'panel'],result.loc[ib,'panel'] = pa,pb
            continue
        score = float(objective(result))
        if score < current-1e-12:
            current=score; accepted+=1
        else:
            result.loc[ia,'panel'],result.loc[ib,'panel'] = pa,pb
    result['panel_index'] = result.panel.map({p:i+1 for i,p in enumerate(sorted(result.panel.unique()))})
    assert_panel_partition(result, expected_datasets=assignment.dataset_name.tolist())
    return result, dict(initial_objective=initial,final_objective=current,accepted_swaps=accepted,
        support_rejected_proposals=support_rejections,
        proposal_budget=proposal_budget,seed=seed,neighborhood='equal-size intact source-family swaps',
        tie_handling='Only improvement > 1e-12 accepted; seeded proposal order; no performance inputs.')


def assert_panel_partition(
    assignment: pd.DataFrame,
    *,
    expected_datasets: Sequence[str],
) -> None:
    names = assignment["dataset_name"].astype(str).tolist()
    expected = list(map(str, expected_datasets))
    if len(names) != len(set(names)):
        raise AssertionError("A dataset occurs in more than one panel.")
    if set(names) != set(expected) or len(names) != len(expected):
        raise AssertionError("Retained dataset union does not equal the intended set.")
    sizes = assignment.groupby("panel", sort=True).size().to_numpy(dtype=int)
    if sizes.size < 2 or int(sizes.max() - sizes.min()) > 1:
        raise AssertionError(f"Panel sizes differ by more than one: {sizes.tolist()}.")
    if "source_identifier" in assignment.columns:
        missing_sources = assignment["source_identifier"].isna() | (
            assignment["source_identifier"].astype(str).str.strip() == ""
        )
        if bool(missing_sources.any()):
            raise AssertionError("Panel assignment contains an empty source identifier.")
        source_panel_counts = assignment.groupby("source_identifier")["panel"].nunique()
        split_sources = source_panel_counts[source_panel_counts > 1]
        if not split_sources.empty:
            raise AssertionError(
                "Related datasets from one source occur in multiple panels: "
                f"{split_sources.index.astype(str).tolist()[:10]}."
            )


def panel_dictionary(assignment: pd.DataFrame) -> dict[str, list[str]]:
    return {
        str(panel): sorted(group["dataset_name"].astype(str).tolist())
        for panel, group in assignment.groupby("panel", sort=True)
    }


def build_panel_balance_report(assignment: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "median_read_density",
        "log1p_median_read_density",
        "median_positive_codon_coverage",
        "number_of_eligible_transcripts",
        "median_replica_PCC",
    ]
    rows: list[dict[str, Any]] = []
    global_means: dict[str, float] = {}
    global_scales: dict[str, float] = {}
    for metric in metrics:
        values = pd.to_numeric(assignment[metric], errors="coerce")
        global_means[metric] = float(values.mean())
        scale = float(values.std(ddof=0))
        global_scales[metric] = scale if math.isfinite(scale) and scale > 0 else 1.0
    panel_means: dict[str, dict[str, float]] = {}
    for panel, group in assignment.groupby("panel", sort=True):
        row: dict[str, Any] = {
            "panel": str(panel),
            "n_datasets": int(len(group)),
            "n_sources": (
                int(group["source_identifier"].astype(str).nunique())
                if "source_identifier" in group.columns
                else int(len(group))
            ),
        }
        panel_means[str(panel)] = {}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            mean_value = float(values.mean()) if values.notna().any() else float("nan")
            median_value = (
                float(values.median()) if values.notna().any() else float("nan")
            )
            panel_means[str(panel)][metric] = mean_value
            row[f"mean_{metric}"] = mean_value
            row[f"median_{metric}"] = median_value
            row[f"standardized_mean_{metric}"] = (
                (mean_value - global_means[metric]) / global_scales[metric]
                if math.isfinite(mean_value)
                else float("nan")
            )
        rows.append(row)
    report = pd.DataFrame(rows)
    for metric in metrics:
        means = np.asarray(
            [panel_means[panel][metric] for panel in sorted(panel_means)],
            dtype=np.float64,
        )
        finite = means[np.isfinite(means)]
        max_difference = (
            float((finite.max() - finite.min()) / global_scales[metric])
            if finite.size >= 2
            else float("nan")
        )
        report[f"max_pairwise_standardized_difference_{metric}"] = max_difference
    return report


@latex_paper_style
def plot_panel_balance(
    assignment: pd.DataFrame,
    *,
    output_directory: str | Path,
) -> list[str]:
    """Save the observed dataset-panel balance figure as PDF and PNG."""
    import matplotlib.pyplot as plt

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    panels = sorted(assignment["panel"].astype(str).unique())
    specifications = [
        ("median_read_density", "Median read density", True),
        ("median_positive_codon_coverage", "Positive-codon coverage", False),
    ]
    replica = pd.to_numeric(assignment["median_replica_PCC"], errors="coerce")
    if replica.notna().any():
        specifications.append(("median_replica_PCC", "Median replica PCC", False))
    figure, axes = plt.subplots(
        1,
        len(specifications),
        figsize=(4.1 * len(specifications), 4.0),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.75, len(panels)))
    rng = np.random.default_rng(0)
    for axis, (metric, label, logarithmic) in zip(
        axes, specifications, strict=True
    ):
        values_by_panel = [
            pd.to_numeric(
                assignment.loc[assignment["panel"] == panel, metric],
                errors="coerce",
            ).dropna().to_numpy(dtype=np.float64)
            for panel in panels
        ]
        box = axis.boxplot(
            values_by_panel,
            tick_labels=[panel.replace("panel_", "P") for panel in panels],
            patch_artist=True,
            showfliers=False,
            widths=0.62,
        )
        for patch, color in zip(box["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.42)
        for position, (values, color) in enumerate(
            zip(values_by_panel, colors, strict=True), start=1
        ):
            jitter = rng.uniform(-0.16, 0.16, size=len(values))
            axis.scatter(
                position + jitter,
                values,
                s=11,
                color=color,
                alpha=0.72,
                linewidths=0,
            )
        if logarithmic:
            axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.set_xlabel("Independent dataset panel")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Observed-data balance across independent panels", fontsize=12)
    stems = [output / "dataset_panel_balance.pdf", output / "dataset_panel_balance.png"]
    figure.savefig(stems[0], bbox_inches="tight")
    figure.savefig(stems[1], dpi=300, bbox_inches="tight")
    plt.close(figure)
    return [str(path) for path in stems]


def load_support_and_stored_weights(
    dataset_mapping: Mapping[str, str],
    *,
    eligible_sequence_ids: set[str],
) -> tuple[
    dict[str, set[str]],
    dict[str, dict[str, float]],
    dict[str, dict[str, int]],
]:
    supports: dict[str, set[str]] = {}
    weights: dict[str, dict[str, float]] = {}
    eligibility_reports: dict[str, dict[str, int]] = {}
    for name, path in dataset_mapping.items():
        frame = pd.read_parquet(path, columns=["id", "weight", "ribo"])
        frame, weight_report = _positive_weight_rows(
            frame, dataset_name=str(name)
        )
        frame, information_report = _positive_information_rows(
            frame, dataset_name=str(name)
        )
        frame = frame.loc[frame["id"].isin(eligible_sequence_ids)]
        numeric = frame["weight"].to_numpy(dtype=np.float64, copy=False)
        ids = frame["id"].astype(str).tolist()
        supports[str(name)] = set(ids)
        weights[str(name)] = dict(zip(ids, numeric.tolist(), strict=True))
        eligibility_reports[str(name)] = {
            **weight_report,
            **information_report,
            "sequence_eligible_usable_rows": int(len(frame)),
        }
    return supports, weights, eligibility_reports


def _support_summary(values: Sequence[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.int64)
    counts = Counter(map(int, array.tolist()))
    return {
        "count": int(array.size),
        "minimum": int(array.min()) if array.size else None,
        "median": float(np.median(array)) if array.size else None,
        "mean": float(array.mean()) if array.size else None,
        "maximum": int(array.max()) if array.size else None,
        "histogram": {str(key): int(value) for key, value in sorted(counts.items())},
    }


def build_common_transcript_split(
    *,
    experiment_name: str,
    dataset_mapping: Mapping[str, str],
    panels: Mapping[str, Sequence[str]],
    sequences_path: str | Path,
    seed: int,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.10,
    reliability_bins: int = 10,
    maximum_cds_codons: int | None = None,
    minimum_panel_support: int = 2,
) -> dict[str, Any]:
    """Create one common held-out split and panel-specific eligible train folds."""
    for fraction_name, fraction in (
        ("validation_fraction", validation_fraction),
        ("test_fraction", test_fraction),
    ):
        if not math.isfinite(float(fraction)) or not 0.0 < float(fraction) < 1.0:
            raise ValueError(f"{fraction_name} must be finite and in (0, 1).")
    sequence_metadata, length_report = load_sequence_metadata(
        sequences_path, max_cds_codons=maximum_cds_codons
    )
    sequence_ids = set(sequence_metadata.index.astype(str))
    selected_names = [name for names in panels.values() for name in names]
    selected_mapping = {name: dataset_mapping[name] for name in selected_names}
    supports, stored_weights, pair_eligibility_reports = load_support_and_stored_weights(
        selected_mapping, eligible_sequence_ids=sequence_ids
    )
    all_data_ids = sorted(set().union(*(supports.values())))

    support_by_panel: dict[str, dict[str, int]] = {}
    for panel, names in panels.items():
        support_by_panel[str(panel)] = {
            transcript_id: int(
                sum(transcript_id in supports[str(name)] for name in names)
            )
            for transcript_id in all_data_ids
        }
    common_eval_ids = sorted(
        transcript_id
        for transcript_id in all_data_ids
        if all(
            support_by_panel[str(panel)][transcript_id] >= minimum_panel_support
            for panel in panels
        )
    )
    if len(common_eval_ids) < 3:
        raise RuntimeError(
            "Fewer than three transcripts satisfy common per-panel evaluation support."
        )

    reliability_scores: dict[str, float] = {}
    for transcript_id in common_eval_ids:
        observed = [
            stored_weights[name][transcript_id]
            for name in selected_names
            if transcript_id in stored_weights[name]
        ]
        reliability_scores[transcript_id] = float(np.median(observed))
    reliability_quantiles = assign_reliability_quantile_bins(
        reliability_scores, number_of_bins=int(reliability_bins)
    )
    strata = {
        transcript_id: (
            f"qbin_{reliability_quantiles[transcript_id]:02d}__"
            f"{sequence_metadata.at[transcript_id, 'css_bin']}"
        )
        for transcript_id in common_eval_ids
    }
    validation_target = int(round(len(common_eval_ids) * float(validation_fraction)))
    test_target = int(round(len(common_eval_ids) * float(test_fraction)))
    if validation_target <= 0 or test_target <= 0:
        raise ValueError("Held-out fractions produce an empty validation or test fold.")
    if validation_target + test_target >= len(common_eval_ids):
        raise ValueError("Held-out folds leave no common-evaluation training candidate.")
    rng = np.random.default_rng(int(seed))
    validation_ids = sample_stratified_ids(
        candidate_ids=common_eval_ids,
        stratum_by_transcript=strata,
        target_count=validation_target,
        rng=rng,
    )
    validation_set = set(validation_ids)
    test_candidates = sorted(set(common_eval_ids) - validation_set)
    test_ids = sample_stratified_ids(
        candidate_ids=test_candidates,
        stratum_by_transcript=strata,
        target_count=test_target,
        rng=rng,
    )
    test_set = set(test_ids)
    heldout = validation_set | test_set
    panel_train_ids: dict[str, list[str]] = {}
    panel_statistics: dict[str, Any] = {}
    for panel, panel_support in support_by_panel.items():
        train_ids = sorted(
            transcript_id
            for transcript_id in all_data_ids
            if transcript_id not in heldout
            and panel_support[transcript_id] >= minimum_panel_support
        )
        if not train_ids:
            raise RuntimeError(f"Panel {panel} has no eligible training transcripts.")
        if any(panel_support[transcript_id] < minimum_panel_support for transcript_id in train_ids):
            raise AssertionError(f"Panel {panel} contains an under-supported train ID.")
        if set(train_ids) & heldout:
            raise AssertionError(f"Panel {panel} training IDs overlap held-out IDs.")
        panel_train_ids[panel] = train_ids
        panel_statistics[panel] = {
            "number_of_train_eligible_transcripts": int(len(train_ids)),
            "train_support": _support_summary(
                [panel_support[transcript_id] for transcript_id in train_ids]
            ),
            "common_eval_support": _support_summary(
                [panel_support[transcript_id] for transcript_id in common_eval_ids]
            ),
            "validation_support": _support_summary(
                [panel_support[transcript_id] for transcript_id in validation_ids]
            ),
            "test_support": _support_summary(
                [panel_support[transcript_id] for transcript_id in test_ids]
            ),
        }

    if validation_set & test_set:
        raise AssertionError("Validation and test IDs overlap.")
    payload = {
        "manifest_version": COMMON_SPLIT_MANIFEST_VERSION,
        "experiment_name": str(experiment_name),
        "created_at_utc": utc_timestamp(),
        "random_seed": int(seed),
        "source_sequences_path": str(sequences_path),
        "maximum_cds_codons": maximum_cds_codons,
        "sequence_eligibility_report": length_report,
        "minimum_usable_datasets_per_panel": int(minimum_panel_support),
        "dataset_pair_eligibility_reports": pair_eligibility_reports,
        "panels": {panel: list(map(str, names)) for panel, names in panels.items()},
        "common_evaluation_criterion": (
            f">={minimum_panel_support} usable selected datasets in every panel"
        ),
        "number_of_common_evaluation_transcripts": int(len(common_eval_ids)),
        "common_evaluation_ids": common_eval_ids,
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(test_fraction),
        "common_validation_ids": validation_ids,
        "common_test_ids": test_ids,
        "train_excluded_ids": sorted(heldout),
        "panel_train_eligible_ids": panel_train_ids,
        "panel_support_statistics": panel_statistics,
        "fold_id_hashes": {
            "validation": transcript_id_hash(validation_ids),
            "test": transcript_id_hash(test_ids),
        },
        "stratification": {
            "method": "established reliability-rank-bin x CSS-bin proportional sampling",
            "reliability_score": (
                "median stored positive pair reliability across available selected datasets; "
                "used only for split stratification"
            ),
            "reliability_bins": int(reliability_bins),
            "css_bins": ["css_0", "css_1", "css_2_3", "css_4_plus"],
            "test_selection": "same strata, sampled from candidates remaining after validation",
        },
        "assertions": {
            "train_validation_overlap": 0,
            "train_test_overlap": 0,
            "validation_test_overlap": 0,
            "all_train_transcripts_meet_panel_support": True,
            "all_heldout_transcripts_meet_every_panel_support": True,
        },
    }
    return json_ready(payload)


def build_panel_stored_weight_manifest(
    *,
    experiment_name: str,
    panel_name: str,
    panel_datasets: Sequence[str],
    dataset_mapping: Mapping[str, str],
    panel_training_ids: Sequence[str],
    validation_ids: Sequence[str],
    test_ids: Sequence[str],
    source_split_manifest: str | Path,
) -> dict[str, Any]:
    """Audit the exact materialized ``w_dt`` values used by one panel.

    This mode deliberately performs no dataset-level reliability refit.  It is
    the faithful baseline for the checked-in/cluster production artifacts,
    whose stored weights may use the historical rank formula and therefore
    must not be silently reinterpreted as the newer SNR formula.
    """
    train_set = set(map(str, panel_training_ids))
    validation_set = set(map(str, validation_ids))
    test_set = set(map(str, test_ids))
    if train_set & (validation_set | test_set):
        raise AssertionError("Stored-weight audit received overlapping folds.")
    datasets: dict[str, Any] = {}
    for dataset_name in panel_datasets:
        dataset_path = Path(dataset_mapping[str(dataset_name)])
        frame = pd.read_parquet(dataset_path, columns=["id", "weight", "ribo"])
        positive, row_report = _positive_weight_rows(
            frame, dataset_name=str(dataset_name)
        )
        positive, information_report = _positive_information_rows(
            positive, dataset_name=str(dataset_name)
        )
        ids = positive["id"].astype(str)
        weights = positive["weight"].to_numpy(dtype=np.float64, copy=False)
        training_mask = ids.isin(train_set).to_numpy(dtype=bool)
        validation_mask = ids.isin(validation_set).to_numpy(dtype=bool)
        test_mask = ids.isin(test_set).to_numpy(dtype=bool)
        training_weights = weights[training_mask]
        if training_weights.size == 0:
            raise ValueError(
                f"Dataset {dataset_name!r} has no positive stored weights among "
                "the panel training IDs."
            )
        datasets[str(dataset_name)] = {
            "source_dataset_path": str(dataset_path),
            "weight_column": "weight",
            "weight_source": "materialized_weighted_artifact",
            "physical_row_count": row_report["physical_rows"],
            "positive_weight_row_count": row_report["positive_weight_rows"],
            "zero_weight_exclusion_row_count": row_report[
                "zero_weight_exclusion_rows"
            ],
            "positive_weight_zero_information_exclusion_row_count": (
                information_report["positive_weight_zero_information_rows"]
            ),
            "training_positive_weight_row_count": int(training_mask.sum()),
            "validation_positive_weight_row_count": int(validation_mask.sum()),
            "test_positive_weight_row_count": int(test_mask.sum()),
            "training_weight_minimum": float(training_weights.min()),
            "training_weight_median": float(np.median(training_weights)),
            "training_weight_mean": float(training_weights.mean()),
            "training_weight_maximum": float(training_weights.max()),
            "training_transcript_id_hash": transcript_id_hash(ids[training_mask]),
        }
    return json_ready(
        {
            "manifest_version": 1,
            "experiment_name": str(experiment_name),
            "panel_name": str(panel_name),
            "created_at_utc": utc_timestamp(),
            "source_split_manifest": str(source_split_manifest),
            "weight_symbol": "w_dt",
            "weight_mode": "stored",
            "weight_source": "input_parquet_weight_column",
            "weight_formula": "preserve_materialized_artifact_values",
            "train_only_reference_fitting": False,
            "reference_split": "precomputed_before_experiment_split",
            "heldout_rows_used_for_fitting": None,
            "zero_weight_policy": (
                "explicit legacy exclusion; omitted from support, split, and loss"
            ),
            "all_loss_weights_strictly_positive": True,
            "panel_training_transcript_count": int(len(train_set)),
            "panel_training_transcript_id_hash": transcript_id_hash(train_set),
            "datasets": datasets,
        }
    )


def fit_panel_reliability_manifest(
    *,
    experiment_name: str,
    panel_name: str,
    panel_datasets: Sequence[str],
    dataset_mapping: Mapping[str, str],
    panel_training_ids: Sequence[str],
    validation_ids: Sequence[str],
    test_ids: Sequence[str],
    source_split_manifest: str | Path,
) -> dict[str, Any]:
    """Fit tau_d and the raw-weight median on actual panel training IDs only."""
    train_set = set(map(str, panel_training_ids))
    heldout = set(map(str, validation_ids)) | set(map(str, test_ids))
    if train_set & heldout:
        raise AssertionError("Cannot fit reliability references with held-out IDs.")
    references: dict[str, Any] = {}
    for dataset_name in panel_datasets:
        dataset_path = Path(dataset_mapping[str(dataset_name)])
        available = set(pq.read_schema(dataset_path).names)
        required = {"id", "read_density", "coverage", "weight"}
        missing = required.difference(available)
        if missing:
            raise KeyError(
                f"Dataset {dataset_name!r} at {dataset_path} is missing required "
                f"production weighted columns {sorted(missing)}; raw/unfiltered "
                "artifacts are not accepted."
            )
        frame = pd.read_parquet(
            dataset_path,
            columns=["id", "read_density", "coverage"],
        )
        reference = fit_dataset_reliability_reference(
            frame,
            dataset_name=str(dataset_name),
            training_transcript_ids=train_set,
        )
        reference["source_dataset_path"] = str(dataset_path)
        reference["observed_pair_statistics_source"] = (
            "materialized_weighted_columns"
        )
        references[str(dataset_name)] = reference
    return json_ready(
        {
            "manifest_version": RELIABILITY_MANIFEST_VERSION,
            "experiment_name": str(experiment_name),
            "panel_name": str(panel_name),
            "created_at_utc": utc_timestamp(),
            "source_split_manifest": str(source_split_manifest),
            "reference_split": "training_only",
            "panel_training_transcript_count": int(len(train_set)),
            "panel_training_transcript_id_hash": transcript_id_hash(train_set),
            "heldout_transcript_count": int(len(heldout)),
            "heldout_rows_used_for_fitting": 0,
            "datasets": references,
        }
    )
