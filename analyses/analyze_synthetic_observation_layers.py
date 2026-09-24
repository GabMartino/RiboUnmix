#!/usr/bin/env python3
"""Audit the synthetic observation hierarchy without loading trained models.

The analysis streams the saved simulator artifacts in transcript order and
keeps the simulator's exact sense-codon coordinate system.  It evaluates

    K_t -> q_t^(r) -> q_t^(r) b_tf -> Y_tf^(r)

at the observation layer, quantifies deterministic bias distortion, extends
the sampled cross-condition matrices with an oracle pre-NB2 matrix, and tests
for common count-sampling randomness.  Count profiles are divided by nominal
depth only; no positional renormalization, smoothing, clipping, interpolation,
or pseudocount is applied.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
from itertools import combinations, zip_longest
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Iterable, Iterator

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import hyp2f1
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utils.publication_plot_style import publication_rc  # noqa: E402
from analyses.analyze_synthetic_input_data import (  # noqa: E402
    iter_grouped_profiles,
    iter_truth,
    parquet_metadata,
    sha256,
)
from analyses.analyze_synthetic_tasep_occupancy_agreement import (  # noqa: E402
    iter_occupancy_replicates,
    normalize_occupancy,
)
from analyses.plot_synthetic_hierarchy import (  # noqa: E402
    dataframe_markdown_table,
    load_sequence_index,
)


DEFAULT_CONFIG = ROOT / "analyses" / "configs" / "synthetic_observation_layers.yaml"
ESTABLISHED_CONDITION_ORDER = (
    "5prime_aa",
    "5prime_cc",
    "5prime_gg",
    "5prime_uu",
    "3prime_aa",
    "3prime_cc",
    "3prime_gg",
    "3prime_uu",
    "au_fraction_gt_0p7",
    "gc_fraction_gt_0p7",
)
ESTABLISHED_DEPTHS = (0.25, 2.0, 20.0)
OBSERVATION_METRICS = (
    "rho_replica_1",
    "rho_replica_2",
    "rho_single",
    "rho_consensus",
    "rho_occupancy",
    "rho_K",
)
FIGURE_OBSERVATION_METRICS = (
    "rho_single",
    "rho_consensus",
    "rho_occupancy",
)
METRIC_LABELS = {
    "rho_replica_1": r"$Y^{(1)}/C$ vs. $q^{(1)}b$",
    "rho_replica_2": r"$Y^{(2)}/C$ vs. $q^{(2)}b$",
    "rho_single": r"Single replica vs. its exact expectation",
    "rho_consensus": r"Count consensus vs. exact consensus expectation",
    "rho_occupancy": r"Count consensus vs. stochastic occupancy",
    "rho_K": r"Count consensus vs. programmed $K_t$",
    "rho_bias": r"Biased expectation vs. occupancy",
    "sampled_cross_condition": "Sampled consensus cross-condition PCC",
    "oracle_cross_condition": "Oracle cross-condition PCC",
}
OBSERVATION_AXIS_LABELS = {
    "rho_single": "Mean single-replica PCC",
    "rho_consensus": "Consensus-to-expectation PCC",
    "rho_occupancy": "Consensus-to-occupancy PCC",
}
REASON_COLUMNS = (
    "too_few_positions",
    "nonfinite_left",
    "nonfinite_right",
    "constant_left",
    "constant_right",
    "constant_both",
    "component_undefined",
)

CROSS_SCHEMA = pa.schema(
    [
        ("depth", pa.string()),
        ("nominal_reads_per_codon", pa.float64()),
        ("transcript_id", pa.string()),
        ("condition_a", pa.string()),
        ("condition_b", pa.string()),
        ("pcc", pa.float64()),
        ("pcc_valid", pa.bool_()),
        ("undefined_reason", pa.string()),
    ]
)
ORACLE_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.string()),
        ("n_positions", pa.int32()),
        ("condition_a", pa.string()),
        ("condition_b", pa.string()),
        ("pcc", pa.float64()),
        ("pcc_valid", pa.bool_()),
        ("undefined_reason", pa.string()),
    ]
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != "ribounmix.synthetic_observation_layer_audit.v1":
        raise ValueError(f"Unsupported configuration schema in {path}")
    condition_order = tuple(item["key"] for item in config["conditions"])
    if condition_order != ESTABLISHED_CONDITION_ORDER:
        raise ValueError(
            "Condition order differs from the existing individual-dataset diagnostic: "
            f"{condition_order}"
        )
    depth_values = tuple(float(item["value"]) for item in config["depths"])
    if depth_values != ESTABLISHED_DEPTHS:
        raise ValueError(f"Depth order must be {ESTABLISHED_DEPTHS}, got {depth_values}")
    if len({item["dataset"] for item in config["conditions"]}) != 10:
        raise ValueError("Exactly ten unique biased observation conditions are required")
    return config


def config_paths(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    conditions = config["conditions"]
    depths = config["depths"]
    bias_paths = {
        item["key"]: resolve_path(
            data["bias_annotation_template"].format(condition=item["key"])
        )
        for item in conditions
    }
    count_paths = {
        depth["slug"]: {
            item["key"]: resolve_path(
                data["count_template"].format(
                    condition=item["key"], depth_slug=depth["slug"]
                )
            )
            for item in conditions
        }
        for depth in depths
    }
    return {
        "kinetic": resolve_path(data["kinetic_profile"]),
        "occupancy": resolve_path(data["occupancy_replicates"]),
        "sequence": resolve_path(data["sequence_annotations"]),
        "bias": bias_paths,
        "counts": count_paths,
    }


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe_json_value(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def pearson_with_reason(left: np.ndarray, right: np.ndarray) -> tuple[float, str]:
    """Pearson correlation with the exact undefined cases required by the audit."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Profile shape mismatch: {left.shape} versus {right.shape}")
    if left.ndim != 1:
        raise ValueError(f"Pearson inputs must be one dimensional, got {left.shape}")
    if left.size < 2:
        return float("nan"), "too_few_positions"
    if not np.isfinite(left).all():
        return float("nan"), "nonfinite_left"
    if not np.isfinite(right).all():
        return float("nan"), "nonfinite_right"
    left_constant = bool(np.all(left == left[0]))
    right_constant = bool(np.all(right == right[0]))
    if left_constant and right_constant:
        return float("nan"), "constant_both"
    if left_constant:
        return float("nan"), "constant_left"
    if right_constant:
        return float("nan"), "constant_right"
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        # This is reachable only through extreme floating-point underflow.
        return float("nan"), "constant_both"
    value = float(np.dot(left_centered, right_centered) / denominator)
    return float(np.clip(value, -1.0, 1.0)), ""


def independent_nb2_equality_probability(
    mean: np.ndarray, *, dispersion_alpha: float
) -> np.ndarray:
    """P(Y_a=Y_b) for two independent, identically distributed NB2 draws.

    With theta=1/alpha and p=theta/(theta+mu), summing the squared NB PMF gives
    p^(2 theta) * 2F1(theta, theta; 1; (1-p)^2).  This supplies the chance
    equality baseline for jointly unaffected positions; no data are regenerated.
    """
    mean = np.asarray(mean, dtype=np.float64)
    if dispersion_alpha <= 0 or not np.isfinite(dispersion_alpha):
        raise ValueError("NB2 dispersion must be finite and positive")
    if not np.isfinite(mean).all() or np.any(mean < 0):
        raise ValueError("NB2 means must be finite and nonnegative")
    theta = 1.0 / float(dispersion_alpha)
    probability = theta / (theta + mean)
    result = np.power(probability, 2.0 * theta) * hyp2f1(
        theta, theta, 1.0, np.square(1.0 - probability)
    )
    if not np.isfinite(result).all():
        raise FloatingPointError("Independent-NB2 exact-equality probability is non-finite")
    return np.clip(result, 0.0, 1.0)


def describe(values: np.ndarray, *, total: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result: dict[str, Any] = {
        "n_total_transcripts": int(total),
        "n_valid_pcc": int(finite.size),
        "n_undefined_pcc": int(total - finite.size),
        "fraction_defined": float(finite.size / total) if total else float("nan"),
    }
    if finite.size:
        p05, p25, median, p75, p95 = np.quantile(
            finite, [0.05, 0.25, 0.50, 0.75, 0.95]
        )
        result.update(
            {
                "median": float(median),
                "q25": float(p25),
                "q75": float(p75),
                "p05": float(p05),
                "p95": float(p95),
                "minimum": float(finite.min()),
                "maximum": float(finite.max()),
            }
        )
    else:
        result.update(
            {
                key: float("nan")
                for key in ("median", "q25", "q75", "p05", "p95", "minimum", "maximum")
            }
        )
    return result


def _reason_record(
    *,
    analysis: str,
    metric: str,
    depth: str,
    condition: str,
    condition_a: str,
    condition_b: str,
    total: int,
    valid: int,
    reasons: Counter[str],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "analysis": analysis,
        "metric": metric,
        "depth": depth,
        "condition": condition,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "n_total_transcripts": int(total),
        "n_valid_pcc": int(valid),
        "n_undefined_pcc": int(total - valid),
        "fraction_defined": float(valid / total) if total else float("nan"),
        "undefined_reason_counts": json.dumps(dict(sorted(reasons.items()))),
    }
    for reason in REASON_COLUMNS:
        record[f"n_{reason}"] = int(reasons.get(reason, 0))
    return record


def _validate_bias_profiles(
    profiles: dict[str, np.ndarray], *, condition: str, transcript_id: str
) -> np.ndarray:
    if set(profiles) != {"rep1", "rep2", "mean"}:
        raise ValueError(f"{condition}/{transcript_id}: incomplete bias roles")
    reference = np.asarray(profiles["rep1"], dtype=np.float64)
    if reference.ndim != 1 or not np.isfinite(reference).all():
        raise ValueError(f"{condition}/{transcript_id}: invalid bias annotation")
    if not (
        np.array_equal(reference, profiles["rep2"])
        and np.array_equal(reference, profiles["mean"])
    ):
        raise ValueError(f"{condition}/{transcript_id}: bias differs across sample roles")
    multiplier = 1.0 + reference
    if np.any(multiplier <= 0.0):
        raise ValueError(f"{condition}/{transcript_id}: non-positive multiplier")
    return multiplier


def _validate_count_profiles(
    profiles: dict[str, np.ndarray], *, condition: str, transcript_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if set(profiles) != {"rep1", "rep2", "mean"}:
        raise ValueError(f"{condition}/{transcript_id}: incomplete count roles")
    rep1 = np.asarray(profiles["rep1"], dtype=np.float64)
    rep2 = np.asarray(profiles["rep2"], dtype=np.float64)
    stored_mean = np.asarray(profiles["mean"], dtype=np.float64)
    if not (rep1.shape == rep2.shape == stored_mean.shape) or rep1.ndim != 1:
        raise ValueError(f"{condition}/{transcript_id}: count arrays are misaligned")
    for role, values in (("rep1", rep1), ("rep2", rep2), ("mean", stored_mean)):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{condition}/{transcript_id}/{role}: invalid counts")
        if not np.array_equal(values, np.rint(values)):
            raise ValueError(f"{condition}/{transcript_id}/{role}: counts are non-integral")
    arithmetic_mean = 0.5 * (rep1 + rep2)
    stored_deviation = float(np.max(np.abs(stored_mean - arithmetic_mean)))
    return rep1, rep2, stored_mean, stored_deviation


def _aligned_items(
    *,
    kinetic_path: Path,
    occupancy_path: Path,
    bias_paths: dict[str, Path],
    count_paths: dict[str, Path] | None,
    condition_keys: list[str],
    batch_size: int,
) -> Iterator[
    tuple[
        str,
        np.ndarray,
        dict[str, np.ndarray],
        dict[str, dict[str, np.ndarray]],
        dict[str, dict[str, np.ndarray]],
    ]
]:
    iterators: list[Iterator[Any]] = [
        iter_truth(kinetic_path, batch_size=batch_size),
        iter_occupancy_replicates(occupancy_path, batch_size=batch_size),
    ]
    iterators.extend(
        iter_grouped_profiles(bias_paths[key], "added_bias", batch_size=batch_size)
        for key in condition_keys
    )
    if count_paths is not None:
        iterators.extend(
            iter_grouped_profiles(count_paths[key], "rib_profile", batch_size=batch_size)
            for key in condition_keys
        )
    for aligned in zip_longest(*iterators):
        if any(item is None for item in aligned):
            raise ValueError("Synthetic source streams have unequal transcript counts")
        identifiers = [str(item[0]) for item in aligned]
        if len(set(identifiers)) != 1:
            raise ValueError(f"Synthetic source streams are misordered: {identifiers}")
        transcript_id = identifiers[0]
        kinetic = np.asarray(aligned[0][1], dtype=np.float64)
        occupancy = aligned[1][1]
        offset = 2
        biases = {
            key: aligned[offset + index][1]
            for index, key in enumerate(condition_keys)
        }
        counts: dict[str, dict[str, np.ndarray]] = {}
        if count_paths is not None:
            count_offset = offset + len(condition_keys)
            counts = {
                key: aligned[count_offset + index][1]
                for index, key in enumerate(condition_keys)
            }
        yield transcript_id, kinetic, occupancy, biases, counts


def validate_source_metadata(
    config: dict[str, Any], paths: dict[str, Any]
) -> dict[str, Any]:
    missing = [
        str(path)
        for path in (
            [paths["kinetic"], paths["occupancy"], paths["sequence"]]
            + list(paths["bias"].values())
            + [path for group in paths["counts"].values() for path in group.values()]
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required synthetic artifacts: {missing}")
    condition_keys = [item["key"] for item in config["conditions"]]
    metadata_records: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    observation_seeds: set[str] = set()
    for condition in condition_keys:
        path = paths["bias"][condition]
        metadata = parquet_metadata(path)
        if metadata.get("riboart.sequence_bias_feature") != condition:
            raise ValueError(f"Bias metadata mismatch: {path}")
        if "terminal boundary excluded" not in metadata.get("riboart.coordinate_scope", ""):
            raise ValueError(f"Ambiguous bias coordinates: {path}")
        fingerprints.add(metadata.get("riboart.source_run_fingerprint", ""))
        metadata_records.append(
            {
                "kind": "bias",
                "condition": condition,
                "path": str(path),
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "metadata": metadata,
            }
        )
    for depth in config["depths"]:
        nominal = float(depth["value"])
        for condition in condition_keys:
            path = paths["counts"][depth["slug"]][condition]
            metadata = parquet_metadata(path)
            if metadata.get("riboart.sequence_bias_feature") != condition:
                raise ValueError(f"Count condition metadata mismatch: {path}")
            if float(metadata.get("riboart.counts_per_codon_unbiased_baseline", "nan")) != nominal:
                raise ValueError(f"Count depth metadata mismatch: {path}")
            if metadata.get("riboart.observation_model") != "negative_binomial_NB2":
                raise ValueError(f"Expected NB2 counts: {path}")
            if metadata.get("riboart.sequence_bias_renormalized") != "false":
                raise ValueError(f"Count profile was unexpectedly renormalized: {path}")
            if "terminal boundary excluded" not in metadata.get("riboart.coordinate_scope", ""):
                raise ValueError(f"Ambiguous count coordinates: {path}")
            fingerprints.add(metadata.get("riboart.source_run_fingerprint", ""))
            observation_seeds.add(metadata.get("riboart.observation_sampling_seed", ""))
            metadata_records.append(
                {
                    "kind": "count",
                    "depth": depth["slug"],
                    "condition": condition,
                    "path": str(path),
                    "rows": pq.ParquetFile(path).metadata.num_rows,
                    "metadata": metadata,
                }
            )
    fingerprints.discard("")
    observation_seeds.discard("")
    if len(fingerprints) != 1:
        raise ValueError(f"Simulator source fingerprints differ: {sorted(fingerprints)}")
    kinetic_metadata = parquet_metadata(paths["kinetic"])
    if kinetic_metadata.get("riboart.source_run_fingerprint") not in fingerprints:
        raise ValueError("Kinetic target fingerprint differs from bias/count artifacts")
    occupancy_metadata = parquet_metadata(paths["occupancy"])
    if "terminal boundary excluded" not in occupancy_metadata.get("positions", ""):
        raise ValueError("Occupancy coordinate metadata does not exclude terminal boundary")
    return {
        "source_run_fingerprint": next(iter(fingerprints)),
        "observation_sampling_seeds": sorted(observation_seeds),
        "all_count_files_declare_same_observation_seed": len(observation_seeds) == 1,
        "kinetic_metadata": kinetic_metadata,
        "occupancy_metadata": occupancy_metadata,
        "artifacts": metadata_records,
    }


def _flush_parquet(
    writer: pq.ParquetWriter, rows: list[dict[str, Any]], schema: pa.Schema
) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def stream_deterministic_and_oracle(
    *,
    config: dict[str, Any],
    paths: dict[str, Any],
    sequence_index: dict[str, Any],
    deterministic_output: Path,
    oracle_output: Path,
    batch_size: int,
    max_transcripts: int | None,
) -> dict[str, Any]:
    condition_keys = [item["key"] for item in config["conditions"]]
    condition_labels = {item["key"]: item["plain_label"] for item in config["conditions"]}
    pair_indices = list(combinations(range(len(condition_keys)), 2))
    target = min(len(sequence_index), max_transcripts or len(sequence_index))
    deterministic_values = np.full((target, len(condition_keys), 2), np.nan, dtype=np.float64)
    oracle_values = np.full((target, len(pair_indices)), np.nan, dtype=np.float64)
    reason_counts: dict[tuple[str, str, str], Counter[str]] = {}
    all_defined = np.ones(target, dtype=bool)
    transcript_ids: list[str] = []
    excluded_missing_sequence: list[str] = []
    max_mean_deviation = {key: 0.0 for key in ("K", "q1", "q2", "qbar")}
    max_identity_deviation = 0.0
    minimum_multiplier = float("inf")
    coordinate_checks = 0
    raw_stream_transcripts = 0
    occupancy_distinct = 0
    csv_fields = [
        "transcript_id",
        "n_positions",
        "condition",
        "condition_label",
        "rho_bias",
        "rho_bias_valid",
        "rho_bias_undefined_reason",
        "expected_total_mass_multiplier",
    ]
    oracle_buffer: list[dict[str, Any]] = []
    with gzip.open(deterministic_output, "wt", encoding="utf-8", newline="") as handle, pq.ParquetWriter(
        oracle_output, ORACLE_SCHEMA, compression="zstd", use_dictionary=True
    ) as oracle_writer:
        csv_writer = csv.DictWriter(handle, fieldnames=csv_fields)
        csv_writer.writeheader()
        for transcript_id, K, occupancy, bias_groups, _ in _aligned_items(
            kinetic_path=paths["kinetic"],
            occupancy_path=paths["occupancy"],
            bias_paths=paths["bias"],
            count_paths=None,
            condition_keys=condition_keys,
            batch_size=batch_size,
        ):
            raw_stream_transcripts += 1
            info = sequence_index.get(transcript_id)
            if info is None:
                excluded_missing_sequence.append(transcript_id)
                continue
            if len(transcript_ids) >= target:
                break
            if info.terminal_codon not in set(config["validation"]["permitted_terminal_codons"]):
                raise ValueError(f"{transcript_id}: unrecognized terminal codon {info.terminal_codon}")
            q1 = normalize_occupancy(occupancy["rep1"])
            q2 = normalize_occupancy(occupancy["rep2"])
            qbar = 0.5 * (q1 + q2)
            K = np.asarray(K, dtype=np.float64)
            n_positions = int(info.sense_codons)
            if not (K.size == q1.size == q2.size == n_positions):
                raise ValueError(
                    f"{transcript_id}: K/q/sequence coordinate mismatch "
                    f"({K.size}, {q1.size}, {q2.size}, {n_positions})"
                )
            if not np.isfinite(K).all() or np.any(K < 0):
                raise ValueError(f"{transcript_id}: invalid K profile")
            row_index = len(transcript_ids)
            transcript_ids.append(transcript_id)
            coordinate_checks += 1
            occupancy_distinct += int(not np.array_equal(q1, q2))
            for name, values in (("K", K), ("q1", q1), ("q2", q2), ("qbar", qbar)):
                max_mean_deviation[name] = max(
                    max_mean_deviation[name], abs(float(values.mean()) - 1.0)
                )
            expected_profiles: list[np.ndarray] = []
            for condition_index, condition in enumerate(condition_keys):
                multiplier = _validate_bias_profiles(
                    bias_groups[condition], condition=condition, transcript_id=transcript_id
                )
                if multiplier.size != n_positions:
                    raise ValueError(f"{condition}/{transcript_id}: bias coordinate mismatch")
                minimum_multiplier = min(minimum_multiplier, float(multiplier.min()))
                expected = qbar * multiplier
                identity_left = 0.5 * (q1 * multiplier + q2 * multiplier)
                identity_deviation = float(np.max(np.abs(identity_left - expected)))
                max_identity_deviation = max(max_identity_deviation, identity_deviation)
                rho_bias, reason = pearson_with_reason(expected, qbar)
                deterministic_values[row_index, condition_index, 0] = rho_bias
                mass_multiplier = float(expected.mean())
                deterministic_values[row_index, condition_index, 1] = mass_multiplier
                if reason:
                    reason_counts.setdefault(("rho_bias", condition, ""), Counter())[reason] += 1
                all_defined[row_index] &= math.isfinite(rho_bias)
                csv_writer.writerow(
                    {
                        "transcript_id": transcript_id,
                        "n_positions": n_positions,
                        "condition": condition,
                        "condition_label": condition_labels[condition],
                        "rho_bias": rho_bias,
                        "rho_bias_valid": math.isfinite(rho_bias),
                        "rho_bias_undefined_reason": reason,
                        "expected_total_mass_multiplier": mass_multiplier,
                    }
                )
                expected_profiles.append(expected)
            for pair_number, (left_index, right_index) in enumerate(pair_indices):
                condition_a = condition_keys[left_index]
                condition_b = condition_keys[right_index]
                value, reason = pearson_with_reason(
                    expected_profiles[left_index], expected_profiles[right_index]
                )
                oracle_values[row_index, pair_number] = value
                if reason:
                    reason_counts.setdefault(
                        ("oracle_cross_condition", condition_a, condition_b), Counter()
                    )[reason] += 1
                all_defined[row_index] &= math.isfinite(value)
                oracle_buffer.append(
                    {
                        "transcript_id": transcript_id,
                        "n_positions": n_positions,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "pcc": value,
                        "pcc_valid": math.isfinite(value),
                        "undefined_reason": reason,
                    }
                )
            if len(oracle_buffer) >= 5000:
                _flush_parquet(oracle_writer, oracle_buffer, ORACLE_SCHEMA)
            if len(transcript_ids) % 2000 == 0:
                print(
                    f"[oracle] {len(transcript_ids):,}/{target:,} aligned transcripts",
                    flush=True,
                )
        _flush_parquet(oracle_writer, oracle_buffer, ORACLE_SCHEMA)
    processed = len(transcript_ids)
    if processed != target:
        raise RuntimeError(f"Oracle pass processed {processed} transcripts, expected {target}")
    summary_rows: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(condition_keys):
        rho_description = describe(deterministic_values[:, condition_index, 0], total=processed)
        mass_values = deterministic_values[:, condition_index, 1]
        mass_description = describe(mass_values, total=processed)
        summary_rows.append(
            {
                "condition": condition,
                "condition_label": condition_labels[condition],
                **{f"rho_bias_{key}": value for key, value in rho_description.items()},
                "mass_n_total_transcripts": processed,
                "mass_n_valid": int(np.isfinite(mass_values).sum()),
                "mass_median": mass_description["median"],
                "mass_q25": mass_description["q25"],
                "mass_q75": mass_description["q75"],
                "mass_p05": mass_description["p05"],
                "mass_p95": mass_description["p95"],
                "mass_minimum": mass_description["minimum"],
                "mass_maximum": mass_description["maximum"],
            }
        )
        reasons = reason_counts.get(("rho_bias", condition, ""), Counter())
        validity_rows.append(
            _reason_record(
                analysis="deterministic_bias",
                metric="rho_bias",
                depth="expected_before_NB2",
                condition=condition,
                condition_a="",
                condition_b="",
                total=processed,
                valid=int(np.isfinite(deterministic_values[:, condition_index, 0]).sum()),
                reasons=reasons,
            )
        )
    oracle_rows: list[dict[str, Any]] = []
    for pair_number, (left_index, right_index) in enumerate(pair_indices):
        condition_a = condition_keys[left_index]
        condition_b = condition_keys[right_index]
        statistics = describe(oracle_values[:, pair_number], total=processed)
        oracle_rows.append(
            {
                "matrix_type": "expected_before_NB2",
                "depth": "expected_before_NB2",
                "nominal_reads_per_codon": np.nan,
                "condition_a": condition_a,
                "condition_b": condition_b,
                **statistics,
                "oracle_median": statistics["median"],
                "difference_from_oracle": 0.0,
                "off_diagonal_rmse_to_oracle": np.nan,
            }
        )
        reasons = reason_counts.get(
            ("oracle_cross_condition", condition_a, condition_b), Counter()
        )
        validity_rows.append(
            _reason_record(
                analysis="oracle_cross_condition",
                metric="oracle_cross_condition",
                depth="expected_before_NB2",
                condition="",
                condition_a=condition_a,
                condition_b=condition_b,
                total=processed,
                valid=int(np.isfinite(oracle_values[:, pair_number]).sum()),
                reasons=reasons,
            )
        )
    tolerance = float(config["validation"]["mean_one_absolute_tolerance"])
    identity_tolerance = float(config["validation"]["identity_absolute_tolerance"])
    if max(max_mean_deviation.values()) > tolerance:
        raise AssertionError(
            f"Mean-one validation failed: {max_mean_deviation}, tolerance={tolerance}"
        )
    if max_identity_deviation > identity_tolerance:
        raise AssertionError(
            f"Bias expectation identity failed: {max_identity_deviation} > {identity_tolerance}"
        )
    if minimum_multiplier <= 0:
        raise AssertionError("Non-positive multiplier encountered")
    return {
        "processed_transcripts": processed,
        "transcript_ids": transcript_ids,
        "all_defined_transcript_ids": [
            transcript_id
            for transcript_id, valid in zip(transcript_ids, all_defined)
            if valid
        ],
        "deterministic_summary": summary_rows,
        "oracle_summary": oracle_rows,
        "validity_rows": validity_rows,
        "oracle_medians": {
            f"{condition_keys[i]}|{condition_keys[j]}": float(
                np.nanmedian(oracle_values[:, pair_number])
            )
            for pair_number, (i, j) in enumerate(pair_indices)
        },
        "validation": {
            "raw_stream_transcripts_seen": raw_stream_transcripts,
            "sequence_aligned_transcripts": processed,
            "excluded_missing_sequence_annotation_count": len(excluded_missing_sequence),
            "excluded_missing_sequence_annotation_ids": excluded_missing_sequence,
            "coordinate_alignment_checks": coordinate_checks,
            "max_absolute_positional_mean_deviation": max_mean_deviation,
            "max_bias_expectation_identity_deviation": max_identity_deviation,
            "minimum_multiplier": minimum_multiplier,
            "normalized_occupancy_trajectories_distinct": occupancy_distinct,
            "mean_one_tolerance": tolerance,
            "identity_tolerance": identity_tolerance,
            "coordinate_convention": (
                "0-based simulator P-site sense-codon positions in saved order; "
                "the sequence terminal codon is excluded and no model padding is introduced"
            ),
        },
    }


def process_depth_worker(
    *,
    config: dict[str, Any],
    path_strings: dict[str, Any],
    depth: dict[str, Any],
    observation_output: str,
    cross_output: str,
    batch_size: int,
    max_transcripts: int | None,
) -> dict[str, Any]:
    paths = {
        "kinetic": Path(path_strings["kinetic"]),
        "occupancy": Path(path_strings["occupancy"]),
        "sequence": Path(path_strings["sequence"]),
        "bias": {key: Path(value) for key, value in path_strings["bias"].items()},
        "counts": {key: Path(value) for key, value in path_strings["counts"].items()},
    }
    sequence_index = load_sequence_index(paths["sequence"])
    condition_keys = [item["key"] for item in config["conditions"]]
    condition_labels = {item["key"]: item["plain_label"] for item in config["conditions"]}
    pair_indices = list(combinations(range(len(condition_keys)), 2))
    nominal_depth = float(depth["value"])
    depth_slug = str(depth["slug"])
    target = min(len(sequence_index), max_transcripts or len(sequence_index))
    metric_values = np.full(
        (target, len(condition_keys), len(OBSERVATION_METRICS)),
        np.nan,
        dtype=np.float64,
    )
    cross_values = np.full((target, len(pair_indices)), np.nan, dtype=np.float64)
    shared_observed_fraction = np.full(
        (target, len(pair_indices), 2), np.nan, dtype=np.float32
    )
    shared_expected_fraction = np.full_like(shared_observed_fraction, np.nan)
    shared_position_totals = np.zeros((len(pair_indices), 2), dtype=np.int64)
    shared_equal_totals = np.zeros((len(pair_indices), 2), dtype=np.int64)
    shared_expected_totals = np.zeros((len(pair_indices), 2), dtype=np.float64)
    reason_counts: dict[tuple[str, str, str], Counter[str]] = {}
    cross_reason_counts: dict[tuple[str, str], Counter[str]] = {}
    all_defined = np.ones(target, dtype=bool)
    transcript_ids: list[str] = []
    excluded_missing_sequence: list[str] = []
    max_q_mean_deviation = 0.0
    max_identity_deviation = 0.0
    max_stored_mean_deviation = 0.0
    minimum_count = float("inf")
    maximum_integrality_deviation = 0.0
    minimum_multiplier = float("inf")
    raw_stream_transcripts = 0
    dispersion_alpha = float(config["validation"]["nb2_dispersion_alpha"])
    observation_fields = [
        "transcript_id",
        "n_positions",
        "depth",
        "nominal_reads_per_codon",
        "condition",
        "condition_label",
    ]
    for metric in OBSERVATION_METRICS:
        observation_fields.extend([metric, f"{metric}_valid", f"{metric}_undefined_reason"])
    cross_buffer: list[dict[str, Any]] = []
    with gzip.open(observation_output, "wt", encoding="utf-8", newline="") as handle, pq.ParquetWriter(
        cross_output, CROSS_SCHEMA, compression="zstd", use_dictionary=True
    ) as cross_writer:
        csv_writer = csv.DictWriter(handle, fieldnames=observation_fields)
        csv_writer.writeheader()
        for transcript_id, K, occupancy, bias_groups, count_groups in _aligned_items(
            kinetic_path=paths["kinetic"],
            occupancy_path=paths["occupancy"],
            bias_paths=paths["bias"],
            count_paths=paths["counts"],
            condition_keys=condition_keys,
            batch_size=batch_size,
        ):
            raw_stream_transcripts += 1
            info = sequence_index.get(transcript_id)
            if info is None:
                excluded_missing_sequence.append(transcript_id)
                continue
            if len(transcript_ids) >= target:
                break
            if info.terminal_codon not in set(config["validation"]["permitted_terminal_codons"]):
                raise ValueError(f"{transcript_id}: unrecognized terminal codon {info.terminal_codon}")
            q1 = normalize_occupancy(occupancy["rep1"])
            q2 = normalize_occupancy(occupancy["rep2"])
            qbar = 0.5 * (q1 + q2)
            K = np.asarray(K, dtype=np.float64)
            n_positions = int(info.sense_codons)
            if not (K.size == q1.size == q2.size == n_positions):
                raise ValueError(f"{depth_slug}/{transcript_id}: K/q/sequence mismatch")
            row_index = len(transcript_ids)
            transcript_ids.append(transcript_id)
            max_q_mean_deviation = max(
                max_q_mean_deviation,
                abs(float(q1.mean()) - 1.0),
                abs(float(q2.mean()) - 1.0),
                abs(float(qbar.mean()) - 1.0),
            )
            count_consensuses: list[np.ndarray] = []
            multipliers: list[np.ndarray] = []
            count_replicates: list[tuple[np.ndarray, np.ndarray]] = []
            for condition_index, condition in enumerate(condition_keys):
                multiplier = _validate_bias_profiles(
                    bias_groups[condition], condition=condition, transcript_id=transcript_id
                )
                rep1, rep2, stored_mean, stored_deviation = _validate_count_profiles(
                    count_groups[condition], condition=condition, transcript_id=transcript_id
                )
                if not (
                    multiplier.size
                    == rep1.size
                    == rep2.size
                    == stored_mean.size
                    == n_positions
                ):
                    raise ValueError(
                        f"{depth_slug}/{condition}/{transcript_id}: positional mismatch"
                    )
                minimum_multiplier = min(minimum_multiplier, float(multiplier.min()))
                minimum_count = min(minimum_count, float(rep1.min()), float(rep2.min()))
                maximum_integrality_deviation = max(
                    maximum_integrality_deviation,
                    float(np.max(np.abs(rep1 - np.rint(rep1)))),
                    float(np.max(np.abs(rep2 - np.rint(rep2)))),
                    float(np.max(np.abs(stored_mean - np.rint(stored_mean)))),
                )
                max_stored_mean_deviation = max(max_stored_mean_deviation, stored_deviation)
                expected_rep1 = q1 * multiplier
                expected_rep2 = q2 * multiplier
                expected_consensus = qbar * multiplier
                identity_deviation = float(
                    np.max(
                        np.abs(
                            0.5 * (expected_rep1 + expected_rep2)
                            - expected_consensus
                        )
                    )
                )
                max_identity_deviation = max(max_identity_deviation, identity_deviation)
                adjusted_rep1 = rep1 / nominal_depth
                adjusted_rep2 = rep2 / nominal_depth
                adjusted_consensus = (rep1 + rep2) / (2.0 * nominal_depth)
                metric_pairs = {
                    "rho_replica_1": (adjusted_rep1, expected_rep1),
                    "rho_replica_2": (adjusted_rep2, expected_rep2),
                    "rho_consensus": (adjusted_consensus, expected_consensus),
                    "rho_occupancy": (adjusted_consensus, qbar),
                    "rho_K": (adjusted_consensus, K),
                }
                metric_results: dict[str, tuple[float, str]] = {
                    metric: pearson_with_reason(left, right)
                    for metric, (left, right) in metric_pairs.items()
                }
                rho1, reason1 = metric_results["rho_replica_1"]
                rho2, reason2 = metric_results["rho_replica_2"]
                if math.isfinite(rho1) and math.isfinite(rho2):
                    metric_results["rho_single"] = (0.5 * (rho1 + rho2), "")
                else:
                    components = []
                    if reason1:
                        components.append(f"replica_1:{reason1}")
                    if reason2:
                        components.append(f"replica_2:{reason2}")
                    metric_results["rho_single"] = (
                        float("nan"),
                        "component_undefined" + (":" + ";".join(components) if components else ""),
                    )
                output_row: dict[str, Any] = {
                    "transcript_id": transcript_id,
                    "n_positions": n_positions,
                    "depth": depth_slug,
                    "nominal_reads_per_codon": nominal_depth,
                    "condition": condition,
                    "condition_label": condition_labels[condition],
                }
                for metric_index, metric in enumerate(OBSERVATION_METRICS):
                    value, reason = metric_results[metric]
                    metric_values[row_index, condition_index, metric_index] = value
                    output_row[metric] = value
                    output_row[f"{metric}_valid"] = math.isfinite(value)
                    output_row[f"{metric}_undefined_reason"] = reason
                    normalized_reason = reason.split(":", 1)[0] if reason else ""
                    if normalized_reason:
                        reason_counts.setdefault((metric, condition, ""), Counter())[
                            normalized_reason
                        ] += 1
                    all_defined[row_index] &= math.isfinite(value)
                csv_writer.writerow(output_row)
                count_consensuses.append(adjusted_consensus)
                multipliers.append(multiplier)
                count_replicates.append((rep1, rep2))
            for pair_number, (left_index, right_index) in enumerate(pair_indices):
                condition_a = condition_keys[left_index]
                condition_b = condition_keys[right_index]
                cross_value, cross_reason = pearson_with_reason(
                    count_consensuses[left_index], count_consensuses[right_index]
                )
                cross_values[row_index, pair_number] = cross_value
                if cross_reason:
                    cross_reason_counts.setdefault((condition_a, condition_b), Counter())[
                        cross_reason
                    ] += 1
                all_defined[row_index] &= math.isfinite(cross_value)
                cross_buffer.append(
                    {
                        "depth": depth_slug,
                        "nominal_reads_per_codon": nominal_depth,
                        "transcript_id": transcript_id,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "pcc": cross_value,
                        "pcc_valid": math.isfinite(cross_value),
                        "undefined_reason": cross_reason,
                    }
                )
                joint_unaffected = (
                    (multipliers[left_index] == 1.0)
                    & (multipliers[right_index] == 1.0)
                )
                n_joint = int(np.count_nonzero(joint_unaffected))
                if n_joint:
                    for replicate_index in range(2):
                        left_counts = count_replicates[left_index][replicate_index]
                        right_counts = count_replicates[right_index][replicate_index]
                        identical = left_counts[joint_unaffected] == right_counts[joint_unaffected]
                        n_identical = int(np.count_nonzero(identical))
                        q_profile = q1 if replicate_index == 0 else q2
                        independent_probability = independent_nb2_equality_probability(
                            nominal_depth * q_profile[joint_unaffected],
                            dispersion_alpha=dispersion_alpha,
                        )
                        expected_identical = float(independent_probability.sum())
                        shared_position_totals[pair_number, replicate_index] += n_joint
                        shared_equal_totals[pair_number, replicate_index] += n_identical
                        shared_expected_totals[pair_number, replicate_index] += expected_identical
                        shared_observed_fraction[
                            row_index, pair_number, replicate_index
                        ] = n_identical / n_joint
                        shared_expected_fraction[
                            row_index, pair_number, replicate_index
                        ] = expected_identical / n_joint
            if len(cross_buffer) >= 5000:
                _flush_parquet(cross_writer, cross_buffer, CROSS_SCHEMA)
            if len(transcript_ids) % 1000 == 0:
                print(
                    f"[{depth_slug}] {len(transcript_ids):,}/{target:,} aligned transcripts",
                    flush=True,
                )
        _flush_parquet(cross_writer, cross_buffer, CROSS_SCHEMA)
    processed = len(transcript_ids)
    if processed != target:
        raise RuntimeError(f"{depth_slug}: processed {processed}, expected {target}")
    observation_summary: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(condition_keys):
        for metric_index, metric in enumerate(OBSERVATION_METRICS):
            values = metric_values[:, condition_index, metric_index]
            statistics = describe(values, total=processed)
            observation_summary.append(
                {
                    "depth": depth_slug,
                    "nominal_reads_per_codon": nominal_depth,
                    "condition": condition,
                    "condition_label": condition_labels[condition],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    **statistics,
                }
            )
            reasons = reason_counts.get((metric, condition, ""), Counter())
            validity_rows.append(
                _reason_record(
                    analysis="observation_layer",
                    metric=metric,
                    depth=depth_slug,
                    condition=condition,
                    condition_a="",
                    condition_b="",
                    total=processed,
                    valid=int(np.isfinite(values).sum()),
                    reasons=reasons,
                )
            )
    sampled_cross_summary: list[dict[str, Any]] = []
    for pair_number, (left_index, right_index) in enumerate(pair_indices):
        condition_a = condition_keys[left_index]
        condition_b = condition_keys[right_index]
        values = cross_values[:, pair_number]
        statistics = describe(values, total=processed)
        sampled_cross_summary.append(
            {
                "matrix_type": "sampled_consensus",
                "depth": depth_slug,
                "nominal_reads_per_codon": nominal_depth,
                "condition_a": condition_a,
                "condition_b": condition_b,
                **statistics,
            }
        )
        validity_rows.append(
            _reason_record(
                analysis="sampled_cross_condition",
                metric="sampled_cross_condition",
                depth=depth_slug,
                condition="",
                condition_a=condition_a,
                condition_b=condition_b,
                total=processed,
                valid=int(np.isfinite(values).sum()),
                reasons=cross_reason_counts.get((condition_a, condition_b), Counter()),
            )
        )
    seeds = {
        condition: parquet_metadata(paths["counts"][condition]).get(
            "riboart.observation_sampling_seed", ""
        )
        for condition in condition_keys
    }
    threshold = float(config["validation"]["common_random_excess_threshold"])
    shared_rows: list[dict[str, Any]] = []
    for pair_number, (left_index, right_index) in enumerate(pair_indices):
        condition_a = condition_keys[left_index]
        condition_b = condition_keys[right_index]
        for replicate_index in range(2):
            positions = int(shared_position_totals[pair_number, replicate_index])
            identical = int(shared_equal_totals[pair_number, replicate_index])
            expected = float(shared_expected_totals[pair_number, replicate_index])
            observed_fraction = identical / positions if positions else float("nan")
            expected_fraction = expected / positions if positions else float("nan")
            excess = observed_fraction - expected_fraction
            transcript_observed = shared_observed_fraction[:, pair_number, replicate_index]
            transcript_expected = shared_expected_fraction[:, pair_number, replicate_index]
            finite_observed = transcript_observed[np.isfinite(transcript_observed)].astype(np.float64)
            finite_expected = transcript_expected[np.isfinite(transcript_expected)].astype(np.float64)
            same_seed = bool(seeds[condition_a] and seeds[condition_a] == seeds[condition_b])
            detected = bool(same_seed and positions > 0 and excess > threshold)
            shared_rows.append(
                {
                    "depth": depth_slug,
                    "nominal_reads_per_codon": nominal_depth,
                    "replicate": replicate_index + 1,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "seed_a": seeds[condition_a],
                    "seed_b": seeds[condition_b],
                    "same_declared_seed": same_seed,
                    "n_total_transcripts": processed,
                    "n_transcripts_with_jointly_unaffected_positions": int(finite_observed.size),
                    "n_jointly_unaffected_positions": positions,
                    "n_exactly_identical_counts": identical,
                    "observed_exact_equality_fraction": observed_fraction,
                    "independent_nb2_expected_equal_count": expected,
                    "independent_nb2_expected_equality_fraction": expected_fraction,
                    "excess_equality_fraction": excess,
                    "observed_to_independent_ratio": (
                        observed_fraction / expected_fraction
                        if expected_fraction > 0
                        else float("nan")
                    ),
                    "median_transcript_observed_equality_fraction": (
                        float(np.median(finite_observed)) if finite_observed.size else np.nan
                    ),
                    "median_transcript_independent_expected_fraction": (
                        float(np.median(finite_expected)) if finite_expected.size else np.nan
                    ),
                    "common_random_numbers_detected": detected,
                    "independent_control": (
                        "analytic NB2 exact-equality probability at the matched q^(r) and C; "
                        "no separately saved independently resampled condition was available"
                    ),
                }
            )
    mean_tolerance = float(config["validation"]["mean_one_absolute_tolerance"])
    identity_tolerance = float(config["validation"]["identity_absolute_tolerance"])
    if max_q_mean_deviation > mean_tolerance:
        raise AssertionError(
            f"{depth_slug}: q mean-one deviation {max_q_mean_deviation} > {mean_tolerance}"
        )
    if max_identity_deviation > identity_tolerance:
        raise AssertionError(
            f"{depth_slug}: expectation identity deviation {max_identity_deviation} > "
            f"{identity_tolerance}"
        )
    if minimum_count < 0 or maximum_integrality_deviation != 0:
        raise AssertionError(f"{depth_slug}: sampled-count validity failed")
    return {
        "depth": depth_slug,
        "nominal_reads_per_codon": nominal_depth,
        "processed_transcripts": processed,
        "transcript_ids": transcript_ids,
        "all_defined_transcript_ids": [
            transcript_id
            for transcript_id, valid in zip(transcript_ids, all_defined)
            if valid
        ],
        "observation_output": observation_output,
        "cross_output": cross_output,
        "observation_summary": observation_summary,
        "sampled_cross_summary": sampled_cross_summary,
        "validity_rows": validity_rows,
        "shared_randomness_rows": shared_rows,
        "validation": {
            "raw_stream_transcripts_seen": raw_stream_transcripts,
            "sequence_aligned_transcripts": processed,
            "excluded_missing_sequence_annotation_count": len(excluded_missing_sequence),
            "excluded_missing_sequence_annotation_ids": excluded_missing_sequence,
            "max_q_mean_one_deviation": max_q_mean_deviation,
            "max_bias_expectation_identity_deviation": max_identity_deviation,
            "minimum_multiplier": minimum_multiplier,
            "minimum_sampled_count": minimum_count,
            "maximum_count_integrality_deviation": maximum_integrality_deviation,
            "maximum_stored_integerized_mean_deviation_from_arithmetic_mean": (
                max_stored_mean_deviation
            ),
        },
    }


def merge_gzip_csvs(partials: list[Path], output: Path) -> None:
    with gzip.open(output, "wt", encoding="utf-8", newline="") as destination:
        wrote_header = False
        for partial in partials:
            with gzip.open(partial, "rt", encoding="utf-8", newline="") as source:
                header = source.readline()
                if not header:
                    raise ValueError(f"Empty partial CSV: {partial}")
                if not wrote_header:
                    destination.write(header)
                    wrote_header = True
                for line in source:
                    destination.write(line)


def _plot_style(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["plot"]
    style = publication_rc()
    style.update(
        {
            "font.size": float(settings["font_size"]),
            "axes.labelsize": float(settings["font_size"]),
            "axes.titlesize": float(settings["font_size"]) + 0.5,
            "figure.titlesize": float(settings["font_size"]) + 1.0,
            "xtick.labelsize": float(settings["tick_font_size"]),
            "ytick.labelsize": float(settings["tick_font_size"]),
            "legend.fontsize": float(settings["tick_font_size"]),
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.grid": False,
            "savefig.dpi": int(settings["dpi"]),
        }
    )
    if style["text.usetex"]:
        style["text.latex.preamble"] += r"\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}"
    return style


def _box_record(row: pd.Series, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "whislo": float(row["p05"]),
        "q1": float(row["q25"]),
        "med": float(row["median"]),
        "q3": float(row["q75"]),
        "whishi": float(row["p95"]),
        "fliers": [],
    }


def _style_boxes(result: dict[str, Any], colors: list[str]) -> None:
    for box, color in zip(result["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.78)
        box.set_edgecolor("#303030")
        box.set_linewidth(0.8)
    for median in result["medians"]:
        median.set_color("#101010")
        median.set_linewidth(1.6)
    for key in ("whiskers", "caps"):
        for artist in result[key]:
            artist.set_color("#555555")
            artist.set_linewidth(0.8)


def _draw_horizontal_interval(
    axis: plt.Axes,
    *,
    y: np.ndarray,
    median: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
    p05: np.ndarray,
    p95: np.ndarray,
    color: str | list[str],
    marker: str = "o",
    label: str | None = None,
    zorder: float = 2.0,
) -> None:
    """Draw a compact distribution summary without implying raw observations.

    The thin capped interval is the 5th--95th percentile range, the thick
    interval is the IQR, and the marker is the median.  All inputs are already
    transcript-level summaries; no codon-level values are pooled here.
    """
    y = np.asarray(y, dtype=np.float64)
    median = np.asarray(median, dtype=np.float64)
    q25 = np.asarray(q25, dtype=np.float64)
    q75 = np.asarray(q75, dtype=np.float64)
    p05 = np.asarray(p05, dtype=np.float64)
    p95 = np.asarray(p95, dtype=np.float64)
    if not isinstance(color, str):
        colors = list(color)
        if len(colors) != len(y):
            raise ValueError("One color is required for every interval")
        for index, item_color in enumerate(colors):
            _draw_horizontal_interval(
                axis,
                y=y[index : index + 1],
                median=median[index : index + 1],
                q25=q25[index : index + 1],
                q75=q75[index : index + 1],
                p05=p05[index : index + 1],
                p95=p95[index : index + 1],
                color=item_color,
                marker=marker,
                label=label if index == 0 else None,
                zorder=zorder,
            )
        return
    axis.errorbar(
        median,
        y,
        xerr=np.vstack([median - p05, p95 - median]),
        fmt="none",
        ecolor=color,
        elinewidth=1.05,
        capsize=2.8,
        capthick=1.05,
        alpha=0.72,
        zorder=zorder,
    )
    axis.hlines(y, q25, q75, color=color, linewidth=4.2, zorder=zorder + 0.2)
    axis.scatter(
        median,
        y,
        color=color,
        marker=marker,
        s=43,
        edgecolor="white",
        linewidth=0.65,
        label=label,
        zorder=zorder + 0.4,
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str, dpi: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = [
        directory / f"{stem}.pdf",
        directory / f"{stem}.png",
        directory / f"{stem}.svg",
    ]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=dpi, bbox_inches="tight")
    fig.savefig(outputs[2], bbox_inches="tight")
    plt.close(fig)
    return outputs


def plot_observation_agreement(
    summary: pd.DataFrame,
    *,
    config: dict[str, Any],
    output_directory: Path,
) -> list[Path]:
    conditions = config["conditions"]
    depths = config["depths"]
    labels = [item["label"] for item in conditions]
    condition_keys = [item["key"] for item in conditions]
    finite_minimum = float(summary["minimum"].replace([np.inf, -np.inf], np.nan).min())
    lower_limit = -0.1 if finite_minimum < 0 else 0.0
    size = tuple(float(value) for value in config["plot"]["observation_size_inches"])
    with matplotlib.rc_context(_plot_style(config)):
        fig, axes = plt.subplots(1, 3, figsize=size, sharex=True, sharey=True)
        depth_colors = ("#0072B2", "#E69F00", "#009E73")
        depth_markers = ("o", "s", "^")
        offsets = (-0.23, 0.0, 0.23)
        base_y = np.arange(len(condition_keys), dtype=np.float64)
        titles = {
            "rho_single": (
                r"A   $\mathbf{Y}_{t,f}^{(r)}/C$ vs."
                "\n" r"$\mathbf{q}_t^{(r)}\odot\mathbf{b}_{t,f}$"
            ),
            "rho_consensus": (
                r"B   $\overline{\mathbf{Y}}_{t,f}/C$ vs."
                "\n" r"$\overline{\mathbf{q}}_t\odot\mathbf{b}_{t,f}$"
            ),
            "rho_occupancy": (
                r"C   $\overline{\mathbf{Y}}_{t,f}/C$ vs."
                "\n" r"$\overline{\mathbf{q}}_t$"
            ),
        }
        for axis, metric in zip(axes, FIGURE_OBSERVATION_METRICS, strict=True):
            for depth_index, depth in enumerate(depths):
                subset = summary.loc[
                    (summary["metric"] == metric)
                    & (summary["depth"] == depth["slug"])
                ].set_index("condition")
                if set(subset.index) != set(condition_keys):
                    raise ValueError(f"Incomplete observation summary for {metric}/{depth['slug']}")
                ordered = subset.loc[condition_keys]
                _draw_horizontal_interval(
                    axis,
                    y=base_y + offsets[depth_index],
                    median=ordered["median"].to_numpy(float),
                    q25=ordered["q25"].to_numpy(float),
                    q75=ordered["q75"].to_numpy(float),
                    p05=ordered["p05"].to_numpy(float),
                    p95=ordered["p95"].to_numpy(float),
                    color=depth_colors[depth_index],
                    marker=depth_markers[depth_index],
                    label=depth["plain_label"],
                    zorder=2.0 + depth_index,
                )
            axis.set_title(titles[metric], loc="left")
            axis.set_xlim(lower_limit, 1.01)
            axis.set_xlabel(
                "Mean of replica PCCs"
                if metric == "rho_single" else "Transcript-level PCC"
            )
            axis.set_xticks(np.arange(math.ceil(lower_limit * 5) / 5, 1.01, 0.2))
            axis.grid(axis="x", alpha=0.28, linewidth=0.7)
            if lower_limit < 0:
                axis.axvline(0.0, color="#777777", linewidth=0.8, linestyle=":")
        axes[0].set_yticks(base_y, labels)
        axes[0].set_ylim(len(condition_keys) - 0.45, -0.55)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.55, 0.005),
            ncol=3,
            borderaxespad=0.0,
            handletextpad=0.45,
            columnspacing=1.2,
        )
        fig.subplots_adjust(left=0.13, right=0.992, top=0.85, bottom=0.19, wspace=0.12)
        return save_figure(
            fig,
            output_directory,
            config["outputs"]["observation_figure_stem"],
            int(config["plot"]["dpi"]),
        )


def plot_deterministic_bias(
    summary: pd.DataFrame,
    *,
    config: dict[str, Any],
    output_directory: Path,
) -> list[Path]:
    conditions = config["conditions"]
    labels = [item["label"] for item in conditions]
    colors = [item["color"] for item in conditions]
    keys = [item["key"] for item in conditions]
    indexed = summary.set_index("condition")
    if set(indexed.index) != set(keys):
        raise ValueError("Deterministic-bias summary is incomplete")
    size = tuple(float(value) for value in config["plot"]["deterministic_size_inches"])
    with matplotlib.rc_context(_plot_style(config)):
        fig, axes = plt.subplots(1, 2, figsize=size)
        y = np.arange(len(keys), dtype=np.float64)
        rho = indexed.loc[keys]
        _draw_horizontal_interval(
            axes[0],
            y=y,
            median=rho["rho_bias_median"].to_numpy(float),
            q25=rho["rho_bias_q25"].to_numpy(float),
            q75=rho["rho_bias_q75"].to_numpy(float),
            p05=rho["rho_bias_p05"].to_numpy(float),
            p95=rho["rho_bias_p95"].to_numpy(float),
            color=colors,
        )
        mass_columns = [
            "mass_median", "mass_q25", "mass_q75", "mass_p05", "mass_p95"
        ]
        if (rho[mass_columns] <= 0).any().any():
            raise ValueError("Expected mass multipliers must be strictly positive")
        log_mass = np.log2(rho[mass_columns].to_numpy(dtype=np.float64))
        _draw_horizontal_interval(
            axes[1],
            y=y,
            median=log_mass[:, 0],
            q25=log_mass[:, 1],
            q75=log_mass[:, 2],
            p05=log_mass[:, 3],
            p95=log_mass[:, 4],
            color=colors,
        )
        for axis in axes:
            axis.grid(axis="x", alpha=0.28, linewidth=0.7)
            axis.set_ylim(len(keys) - 0.45, -0.55)
        axes[0].set_yticks(y, labels)
        axes[1].tick_params(labelleft=False)
        axes[0].set_title("A   Shape distortion", loc="left")
        axes[0].set_xlabel(r"$\mathrm{PCC}(\overline{q}_t b_{t,f},\overline{q}_t)$")
        axes[0].set_xlim(-0.05, 1.01)
        axes[1].set_title("B   Expected mass distortion", loc="left")
        axes[1].set_xlabel(r"$\log_2 M_{t,f}^{\mathrm{bias}}$")
        axes[1].axvline(0.0, color="#777777", linewidth=0.9, linestyle=":")
        fig.text(
            0.995,
            0.985,
            "Point: median; thick: IQR; thin: 5th--95th percentile",
            ha="right",
            va="top",
            fontsize=float(config["plot"]["small_font_size"]),
        )
        fig.subplots_adjust(left=0.14, right=0.99, top=0.90, bottom=0.13, wspace=0.13)
        return save_figure(
            fig,
            output_directory,
            config["outputs"]["deterministic_figure_stem"],
            int(config["plot"]["dpi"]),
        )


def plot_cross_condition_matrices(
    summary: pd.DataFrame,
    *,
    config: dict[str, Any],
    output_directory: Path,
) -> list[Path]:
    conditions = config["conditions"]
    keys = [item["key"] for item in conditions]
    labels = [item["label"] for item in conditions]
    panels = [
        ("sampled_consensus", depth["slug"], f"{depth['plain_label']} (sampled)")
        for depth in config["depths"]
    ] + [("expected_before_NB2", "expected_before_NB2", "Expected before NB2 sampling")]
    matrices: list[np.ndarray] = []
    for matrix_type, depth, _ in panels:
        matrix = np.eye(len(keys), dtype=np.float64)
        subset = summary.loc[
            (summary["matrix_type"] == matrix_type) & (summary["depth"] == depth)
        ]
        if len(subset) != 45:
            raise ValueError(f"Expected 45 matrix entries for {matrix_type}/{depth}, got {len(subset)}")
        lookup = {key: index for index, key in enumerate(keys)}
        for row in subset.itertuples(index=False):
            left = lookup[row.condition_a]
            right = lookup[row.condition_b]
            matrix[left, right] = matrix[right, left] = float(row.median)
        matrices.append(matrix)
    finite_minimum = min(float(np.nanmin(matrix)) for matrix in matrices)
    if finite_minimum < 0:
        vmin, vmax, cmap = -1.0, 1.0, "RdBu_r"
    else:
        vmin, vmax, cmap = 0.0, 1.0, "viridis"
    size = tuple(float(value) for value in config["plot"]["matrix_size_inches"])
    with matplotlib.rc_context(_plot_style(config)):
        fig = plt.figure(figsize=size)
        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=(1.0, 1.0, 0.055),
            left=0.095,
            right=0.94,
            bottom=0.105,
            top=0.955,
            wspace=0.30,
            hspace=0.31,
        )
        axes = np.asarray(
            [
                fig.add_subplot(grid[0, 0]),
                fig.add_subplot(grid[0, 1]),
                fig.add_subplot(grid[1, 0]),
                fig.add_subplot(grid[1, 1]),
            ],
            dtype=object,
        ).reshape(2, 2)
        colorbar_axis = fig.add_subplot(grid[:, 2])
        image = None
        for panel_index, (axis, matrix, (_, _, title)) in enumerate(
            zip(axes.flat, matrices, panels)
        ):
            image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_xticks(range(len(keys)), labels, rotation=50, ha="right")
            axis.set_yticks(range(len(keys)), labels)
            compact_title = (
                f"Sampled, $C={config['depths'][panel_index]['value']:g}$"
                if panel_index < 3
                else "Expected (pre-NB2)"
            )
            axis.set_title(
                f"{'ABCD'[panel_index]}   {compact_title}", fontweight="bold"
            )
            for row_index in range(len(keys)):
                for column_index in range(len(keys)):
                    value = matrix[row_index, column_index]
                    normalized = (value - vmin) / (vmax - vmin)
                    color = "white" if normalized < 0.53 else "#111111"
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=float(config["plot"]["small_font_size"]),
                        color=color,
                    )
        assert image is not None
        colorbar = fig.colorbar(
            image,
            cax=colorbar_axis,
            label="Median transcript-level PCC",
        )
        colorbar.ax.tick_params(labelsize=float(config["plot"]["tick_font_size"]))
        return save_figure(
            fig,
            output_directory,
            config["outputs"]["cross_condition_figure_stem"],
            int(config["plot"]["dpi"]),
        )


def headline_layer_summary(observation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (depth, nominal, metric), group in observation.loc[
        observation["metric"].isin(FIGURE_OBSERVATION_METRICS)
    ].groupby(["depth", "nominal_reads_per_codon", "metric"], sort=False):
        condition_medians = group["median"].to_numpy(dtype=np.float64)
        q25, median, q75 = np.quantile(condition_medians, [0.25, 0.5, 0.75])
        rows.append(
            {
                "depth": depth,
                "nominal_reads_per_codon": nominal,
                "metric": metric,
                "conditions": int(condition_medians.size),
                "median_of_condition_medians": float(median),
                "q25_of_condition_medians": float(q25),
                "q75_of_condition_medians": float(q75),
                "minimum_condition_median": float(condition_medians.min()),
                "maximum_condition_median": float(condition_medians.max()),
                "summary_unit": "ten deliberately selected observation conditions",
            }
        )
    return pd.DataFrame.from_records(rows)


def crosscheck_previous_K_summary(
    observation: pd.DataFrame,
    *,
    config: dict[str, Any],
    previous_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not previous_path.is_file():
        return pd.DataFrame(), {"available": False, "path": str(previous_path)}
    previous = pd.read_csv(previous_path)
    previous = previous.loc[previous["representation"] == "Arithmetic mean"].copy()
    dataset_to_condition = {
        item["dataset"]: item["key"] for item in config["conditions"]
    }
    previous["condition"] = previous["dataset"].map(dataset_to_condition)
    previous = previous.loc[previous["condition"].notna()]
    current = observation.loc[
        observation["metric"] == "rho_K",
        ["depth", "condition", "median", "q25", "q75", "p05", "p95"],
    ].rename(columns={column: f"current_{column}" for column in ("median", "q25", "q75", "p05", "p95")})
    previous = previous.rename(
        columns={
            "median": "previous_median",
            "q25": "previous_q25",
            "q75": "previous_q75",
            "p05": "previous_p05",
            "p95": "previous_p95",
        }
    )
    merged = current.merge(
        previous[
            [
                "depth",
                "condition",
                "previous_median",
                "previous_q25",
                "previous_q75",
                "previous_p05",
                "previous_p95",
            ]
        ],
        on=["depth", "condition"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    for statistic in ("median", "q25", "q75", "p05", "p95"):
        merged[f"absolute_difference_{statistic}"] = np.abs(
            merged[f"current_{statistic}"] - merged[f"previous_{statistic}"]
        )
    complete = bool((merged["_merge"] == "both").all())
    max_difference = float(
        merged[
            [f"absolute_difference_{name}" for name in ("median", "q25", "q75", "p05", "p95")]
        ].max().max()
    )
    return merged, {
        "available": True,
        "path": str(previous_path),
        "matched_all_30_condition_depth_groups": complete and len(merged) == 30,
        "maximum_absolute_summary_statistic_difference": max_difference,
        "interpretation": (
            "PCC is invariant to division by C; exact agreement is expected because the "
            "previous raw-count audit used the same arithmetic two-replica consensus and K_t."
        ),
    }


def write_captions(
    path: Path,
    *,
    headline: pd.DataFrame,
    oracle_rmse: pd.DataFrame,
    shared_detected: bool,
) -> None:
    depth_lines: list[str] = []
    for depth, group in headline.groupby("nominal_reads_per_codon", sort=True):
        lookup = group.set_index("metric")["median_of_condition_medians"]
        depth_lines.append(
            rf"At $C={depth:g}$, the medians across the ten condition-specific medians were "
            rf"{lookup['rho_single']:.3f}, {lookup['rho_consensus']:.3f}, and "
            rf"{lookup['rho_occupancy']:.3f}, respectively."
        )
    rmse_text = ", ".join(
        rf"{row.nominal_reads_per_codon:g}: {row.off_diagonal_rmse_to_oracle:.3f}"
        for row in oracle_rmse.itertuples(index=False)
    )
    warning = (
        "The saved condition files show excess exact count equality over an independent-NB2 "
        "baseline, consistent with common random numbers; the matrices are therefore paired "
        "simulator diagnostics rather than independent experimental replicates."
        if shared_detected
        else "No excess exact equality meeting the prespecified common-random-number threshold was detected."
    )
    text = rf"""% Generated by analyses/analyze_synthetic_observation_layers.py.
% Copy each caption into the corresponding figure environment.

% Suggested label: fig:synthetic_observation_layer_agreement
\caption{{\textbf{{Agreement across the synthetic observation hierarchy.}}
Columns show nominal depths $C\in\{{0.25,2,20\}}$ and boxes summarize 19,283
transcript-level Pearson correlations for each of ten programmed observation
conditions. The first row averages the two separately calculated correlations
$\operatorname{{PCC}}(Y^{{(r)}}/C,q^{{(r)}}b)$; it is defined only when both
replica correlations are defined. The second row compares the arithmetic count
consensus with its exact conditional expectation $\overline q b$, and the third
compares that same consensus with the pre-bias stochastic occupancy $\overline q$.
Boxes show the interquartile range, center lines the median, and whiskers the
5th--95th percentiles; individual outliers are omitted. Dividing counts by $C$
places them on the expected-profile scale but does not change PCC. No positional
mean normalization, smoothing, clipping, interpolation, or pseudocount was used.
{' '.join(depth_lines)}}}
\label{{fig:synthetic_observation_layer_agreement}}

% Suggested label: fig:synthetic_deterministic_bias_effects
\caption{{\textbf{{Deterministic effects of the programmed observation multipliers.}}
\textbf{{(A)}} Transcript-level shape agreement between the expected biased
profile $\overline q_t b_{{t,f}}$ and the stochastic occupancy consensus
$\overline q_t$. \textbf{{(B)}} Expected mass multiplier
$M_{{t,f}}^{{\mathrm{{bias}}}}=\langle\overline q_t b_{{t,f}}\rangle_t$.
The occupancy profiles are mean-one by construction, whereas the biased
profiles are not renormalized. Boxes show the interquartile range, medians, and
5th--95th percentile whiskers over the 19,283 aligned transcripts.}}
\label{{fig:synthetic_deterministic_bias_effects}}

% Suggested label: fig:synthetic_cross_condition_agreement
\caption{{\textbf{{Cross-condition agreement before and after NB2 sampling.}}
The first three matrices give, at each nominal depth, the median across
transcripts of the Pearson correlation between arithmetic two-replica sampled
consensuses from each condition pair. The fourth matrix gives the deterministic
oracle comparison between $\overline q_t b_{{t,f}}$ and
$\overline q_t b_{{t,g}}$ before NB2 count sampling. Every cell is a median of
transcript-level correlations rather than a pooled codon-level correlation, and
all matrices share one color scale. The off-diagonal sampled-to-oracle RMSEs
for $C=0.25,2,20$ were {rmse_text}. The same two stochastic TASEP trajectories
are intentionally reused across conditions and depths. {warning}}}
\label{{fig:synthetic_cross_condition_agreement}}
"""
    path.write_text(text, encoding="utf-8")


def write_report(
    path: Path,
    *,
    validation: dict[str, Any],
    headline: pd.DataFrame,
    oracle_rmse: pd.DataFrame,
    shared: pd.DataFrame,
    K_crosscheck: dict[str, Any],
    command: str,
) -> None:
    headline_table = dataframe_markdown_table(headline, float_digits=4)
    rmse_table = dataframe_markdown_table(oracle_rmse[
        ["nominal_reads_per_codon", "off_diagonal_rmse_to_oracle"]
    ], float_digits=4)
    detected_rows = int(shared["common_random_numbers_detected"].sum())
    text = f"""# Synthetic observation-layer audit

This analysis uses only saved simulator artifacts. It does not load a trained
model, a checkpoint, or model predictions.

## Coordinate and cohort contract

- Sequence-aligned transcripts: **{validation['sequence_aligned_transcripts']:,}**.
- Excluded raw simulator transcripts lacking the sequence annotation: **{validation['excluded_missing_sequence_annotation_count']}**.
- Coordinates: saved 0-based P-site sense-codon order; the terminal codon is
  excluded, and no model-only terminal padding is introduced.
- Maximum mean-one deviation among K/q profiles: **{validation['maximum_mean_one_deviation']:.3e}**.
- Maximum algebraic deviation in
  `(q1*b + q2*b)/2 == qbar*b`: **{validation['maximum_expectation_identity_deviation']:.3e}**.
- Minimum multiplier: **{validation['minimum_multiplier']:.6g}**.
- Minimum sampled count: **{validation['minimum_sampled_count']:.6g}**;
  maximum integrality deviation: **{validation['maximum_count_integrality_deviation']:.3e}**.
- Transcripts with every requested PCC defined: **{validation['transcripts_with_every_requested_pcc_defined']:,}**.

No profile was smoothed, clipped, interpolated, pseudocounted, or normalized
by its observed positional mean. Sampled counts were divided only by nominal
depth C.

## Layer-wise agreement

The table reports the median and spread across the ten deliberately selected
condition-specific medians. It is descriptive; the ten biases are not treated
as independent draws from a population.

{headline_table}

## Oracle cross-condition comparison

{rmse_table}

## Shared count-sampling randomness

The audit compares exact equality at positions where both multipliers equal
one with the analytic probability for two independent NB2 draws at the same
matched q and C. **{detected_rows} pair/depth/replicate rows** exceeded the
prespecified excess-equality threshold. This is separate from the intentional
reuse of the same two TASEP trajectories. No separately saved independent
condition resampling was available. The generator source that wrote these
Parquets is not present in this repository, so the artifacts establish
common-random-number coupling but cannot uniquely distinguish resetting a
generator to the same seed for each condition from another coordinated reuse
of the same random stream.

## Existing count-versus-K cross-check

```json
{json.dumps(_safe_json_value(K_crosscheck), indent=2)}
```

## Reproduction

```bash
{command}
```
"""
    path.write_text(text, encoding="utf-8")


def _commit_files(files: Iterable[Path], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    committed: list[Path] = []
    for source in files:
        target = destination / source.name
        os.replace(source, target)
        committed.append(target)
    return committed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-transcripts", type=int, help="Deterministic smoke-test limit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0 or args.batch_size <= 0:
        parser.error("--workers and --batch-size must be positive")
    if args.max_transcripts is not None and args.max_transcripts <= 0:
        parser.error("--max-transcripts must be positive")
    config_path = args.config.resolve()
    config = load_config(config_path)
    paths = config_paths(config)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_path(config["outputs"]["results_directory"])
    )
    figure_dir = (
        args.figure_dir.resolve()
        if args.figure_dir
        else resolve_path(config["outputs"]["figure_directory"])
    )
    default_output = resolve_path(config["outputs"]["results_directory"])
    default_figures = resolve_path(config["outputs"]["figure_directory"])
    if args.max_transcripts is not None and (
        output_dir == default_output or figure_dir == default_figures
    ):
        parser.error(
            "A smoke test must use explicit non-production --output-dir and --figure-dir paths"
        )
    metadata_audit = validate_source_metadata(config, paths)
    sequence_index = load_sequence_index(paths["sequence"])
    expected_transcripts = int(config["validation"]["expected_transcripts"])
    if args.max_transcripts is None and len(sequence_index) != expected_transcripts:
        raise AssertionError(
            f"Sequence cohort has {len(sequence_index)} transcripts, expected {expected_transcripts}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_names = [
        "observation_layer_agreement_per_transcript.csv.gz",
        "observation_layer_agreement_summary.csv",
        "deterministic_bias_effects_per_transcript.csv.gz",
        "deterministic_bias_effects_summary.csv",
        "oracle_cross_condition_per_transcript.parquet",
        "oracle_cross_condition_correlation_summary.csv",
        "pcc_validity_audit.csv",
        "shared_count_randomness_audit.csv",
        "headline_observation_layer_summary.csv",
        "kinetic_target_summary_crosscheck.csv",
        "observation_layer_figure_captions.tex",
        "synthetic_observation_layer_validation.json",
        "synthetic_observation_layer_provenance.json",
        "synthetic_observation_layer_report.md",
    ]
    result_names.extend(
        f"sampled_cross_condition_per_transcript_{depth['slug']}.parquet"
        for depth in config["depths"]
    )
    figure_stems = [
        config["outputs"]["observation_figure_stem"],
        config["outputs"]["deterministic_figure_stem"],
        config["outputs"]["cross_condition_figure_stem"],
    ]
    targets = [output_dir / name for name in result_names] + [
        figure_dir / f"{stem}.{extension}"
        for stem in figure_stems
        for extension in ("pdf", "png", "svg")
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Requested outputs already exist; use --overwrite: "
            + ", ".join(str(path) for path in existing[:8])
        )
    work_dir = output_dir / ".synthetic_observation_layer_audit_tmp"
    if work_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Staging directory exists: {work_dir}")
        shutil.rmtree(work_dir)
    work_results = work_dir / "results"
    work_figures = work_dir / "figures"
    work_results.mkdir(parents=True)
    work_figures.mkdir(parents=True)
    deterministic_path = work_results / "deterministic_bias_effects_per_transcript.csv.gz"
    oracle_path = work_results / "oracle_cross_condition_per_transcript.parquet"
    deterministic_result = stream_deterministic_and_oracle(
        config=config,
        paths=paths,
        sequence_index=sequence_index,
        deterministic_output=deterministic_path,
        oracle_output=oracle_path,
        batch_size=args.batch_size,
        max_transcripts=args.max_transcripts,
    )
    path_strings = {
        "kinetic": str(paths["kinetic"]),
        "occupancy": str(paths["occupancy"]),
        "sequence": str(paths["sequence"]),
        "bias": {key: str(path) for key, path in paths["bias"].items()},
    }
    futures = []
    worker_count = min(args.workers, len(config["depths"]))
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=get_context("spawn")
    ) as executor:
        for depth in config["depths"]:
            slug = depth["slug"]
            observation_partial = work_results / f"observation_{slug}.csv.gz"
            cross_partial = work_results / f"sampled_cross_condition_per_transcript_{slug}.parquet"
            worker_paths = {
                **path_strings,
                "counts": {
                    key: str(path) for key, path in paths["counts"][slug].items()
                },
            }
            futures.append(
                executor.submit(
                    process_depth_worker,
                    config=config,
                    path_strings=worker_paths,
                    depth=depth,
                    observation_output=str(observation_partial),
                    cross_output=str(cross_partial),
                    batch_size=args.batch_size,
                    max_transcripts=args.max_transcripts,
                )
            )
        depth_results = [future.result() for future in as_completed(futures)]
    depth_order = {item["slug"]: index for index, item in enumerate(config["depths"])}
    depth_results.sort(key=lambda item: depth_order[item["depth"]])
    reference_ids = deterministic_result["transcript_ids"]
    for result in depth_results:
        if result["transcript_ids"] != reference_ids:
            raise AssertionError(f"Transcript order differs for depth {result['depth']}")
    observation_path = work_results / "observation_layer_agreement_per_transcript.csv.gz"
    merge_gzip_csvs(
        [Path(result["observation_output"]) for result in depth_results],
        observation_path,
    )
    observation_summary = pd.DataFrame.from_records(
        row for result in depth_results for row in result["observation_summary"]
    )
    observation_summary.to_csv(
        work_results / "observation_layer_agreement_summary.csv", index=False
    )
    deterministic_summary = pd.DataFrame.from_records(
        deterministic_result["deterministic_summary"]
    )
    deterministic_summary.to_csv(
        work_results / "deterministic_bias_effects_summary.csv", index=False
    )
    oracle_rows = deterministic_result["oracle_summary"]
    oracle_medians = deterministic_result["oracle_medians"]
    sampled_rows: list[dict[str, Any]] = []
    rmse_rows: list[dict[str, Any]] = []
    for result in depth_results:
        squared_differences: list[float] = []
        for row in result["sampled_cross_summary"]:
            key = f"{row['condition_a']}|{row['condition_b']}"
            oracle_median = oracle_medians[key]
            enriched = {
                **row,
                "oracle_median": oracle_median,
                "difference_from_oracle": float(row["median"] - oracle_median),
                "off_diagonal_rmse_to_oracle": np.nan,
            }
            sampled_rows.append(enriched)
            squared_differences.append((float(row["median"]) - oracle_median) ** 2)
        rmse = float(np.sqrt(np.mean(squared_differences)))
        rmse_rows.append(
            {
                "matrix_type": "sampled_vs_oracle_RMSE",
                "depth": result["depth"],
                "nominal_reads_per_codon": result["nominal_reads_per_codon"],
                "condition_a": "",
                "condition_b": "",
                "n_total_transcripts": result["processed_transcripts"],
                "n_valid_pcc": np.nan,
                "n_undefined_pcc": np.nan,
                "fraction_defined": np.nan,
                "median": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "p05": np.nan,
                "p95": np.nan,
                "minimum": np.nan,
                "maximum": np.nan,
                "oracle_median": np.nan,
                "difference_from_oracle": np.nan,
                "off_diagonal_rmse_to_oracle": rmse,
            }
        )
    cross_summary = pd.DataFrame.from_records(oracle_rows + sampled_rows + rmse_rows)
    cross_summary.to_csv(
        work_results / "oracle_cross_condition_correlation_summary.csv", index=False
    )
    validity = pd.DataFrame.from_records(
        deterministic_result["validity_rows"]
        + [row for result in depth_results for row in result["validity_rows"]]
    )
    validity.to_csv(work_results / "pcc_validity_audit.csv", index=False)
    shared = pd.DataFrame.from_records(
        row for result in depth_results for row in result["shared_randomness_rows"]
    )
    shared.to_csv(work_results / "shared_count_randomness_audit.csv", index=False)
    headline = headline_layer_summary(observation_summary)
    headline.to_csv(work_results / "headline_observation_layer_summary.csv", index=False)
    previous_summary = output_dir / "kinetic_target_agreement_summary.csv"
    K_crosscheck_frame, K_crosscheck = crosscheck_previous_K_summary(
        observation_summary,
        config=config,
        previous_path=previous_summary,
    )
    K_crosscheck_frame.to_csv(
        work_results / "kinetic_target_summary_crosscheck.csv", index=False
    )
    oracle_rmse = pd.DataFrame.from_records(rmse_rows)
    shared_detected = bool(shared["common_random_numbers_detected"].any())
    all_defined_sets = [set(deterministic_result["all_defined_transcript_ids"])] + [
        set(result["all_defined_transcript_ids"]) for result in depth_results
    ]
    all_defined_global = set.intersection(*all_defined_sets)
    validation = {
        "sequence_aligned_transcripts": deterministic_result["processed_transcripts"],
        "excluded_missing_sequence_annotation_count": deterministic_result["validation"][
            "excluded_missing_sequence_annotation_count"
        ],
        "excluded_missing_sequence_annotation_ids": deterministic_result["validation"][
            "excluded_missing_sequence_annotation_ids"
        ],
        "maximum_mean_one_deviation": max(
            max(
                deterministic_result["validation"]["max_absolute_positional_mean_deviation"].values()
            ),
            max(result["validation"]["max_q_mean_one_deviation"] for result in depth_results),
        ),
        "maximum_expectation_identity_deviation": max(
            deterministic_result["validation"]["max_bias_expectation_identity_deviation"],
            max(
                result["validation"]["max_bias_expectation_identity_deviation"]
                for result in depth_results
            ),
        ),
        "minimum_multiplier": min(
            deterministic_result["validation"]["minimum_multiplier"],
            min(result["validation"]["minimum_multiplier"] for result in depth_results),
        ),
        "minimum_sampled_count": min(
            result["validation"]["minimum_sampled_count"] for result in depth_results
        ),
        "maximum_count_integrality_deviation": max(
            result["validation"]["maximum_count_integrality_deviation"]
            for result in depth_results
        ),
        "maximum_stored_integerized_mean_deviation_from_arithmetic_mean": max(
            result["validation"][
                "maximum_stored_integerized_mean_deviation_from_arithmetic_mean"
            ]
            for result in depth_results
        ),
        "normalized_occupancy_trajectories_distinct": deterministic_result["validation"][
            "normalized_occupancy_trajectories_distinct"
        ],
        "transcripts_with_every_requested_pcc_defined": len(all_defined_global),
        "per_pass_all_requested_pcc_defined": {
            "deterministic_and_oracle": len(
                deterministic_result["all_defined_transcript_ids"]
            ),
            **{
                result["depth"]: len(result["all_defined_transcript_ids"])
                for result in depth_results
            },
        },
        "coordinate_convention": deterministic_result["validation"][
            "coordinate_convention"
        ],
        "mean_one_tolerance": float(
            config["validation"]["mean_one_absolute_tolerance"]
        ),
        "identity_tolerance": float(
            config["validation"]["identity_absolute_tolerance"]
        ),
        "shared_count_randomness_detected": shared_detected,
        "common_random_excess_threshold": float(
            config["validation"]["common_random_excess_threshold"]
        ),
        "source_metadata": metadata_audit,
        "depth_passes": {
            result["depth"]: result["validation"] for result in depth_results
        },
        "full_run": args.max_transcripts is None,
        "max_transcripts": args.max_transcripts,
    }
    if args.max_transcripts is None and validation["sequence_aligned_transcripts"] != expected_transcripts:
        raise AssertionError("Final cohort does not contain exactly 19,283 transcripts")
    write_json(work_results / "synthetic_observation_layer_validation.json", validation)
    plot_observation_agreement(
        observation_summary, config=config, output_directory=work_figures
    )
    plot_deterministic_bias(
        deterministic_summary, config=config, output_directory=work_figures
    )
    plot_cross_condition_matrices(
        cross_summary, config=config, output_directory=work_figures
    )
    write_captions(
        work_results / "observation_layer_figure_captions.tex",
        headline=headline,
        oracle_rmse=oracle_rmse,
        shared_detected=shared_detected,
    )
    command = (
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        f"{sys.executable} {Path(__file__).resolve()} --config {config_path} "
        f"--workers {worker_count} --batch-size {args.batch_size} --overwrite"
    )
    write_report(
        work_results / "synthetic_observation_layer_report.md",
        validation=validation,
        headline=headline,
        oracle_rmse=oracle_rmse,
        shared=shared,
        K_crosscheck=K_crosscheck,
        command=command,
    )
    provenance = {
        "analysis": "synthetic_observation_layer_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "exact_full_run_command": command,
        "repository_root": str(ROOT),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "matplotlib": matplotlib.__version__,
        "condition_order": [item["key"] for item in config["conditions"]],
        "depth_order": [float(item["value"]) for item in config["depths"]],
        "definitions": {
            "sampled_scale": "raw NB2 counts divided only by nominal C",
            "single": "mean of separately computed replica-1 and replica-2 PCCs, only when both are defined",
            "consensus": "PCC((Y1+Y2)/(2C), qbar*b)",
            "occupancy": "PCC((Y1+Y2)/(2C), qbar)",
            "kinetic": "PCC((Y1+Y2)/(2C), K)",
            "bias_distortion": "PCC(qbar*b, qbar), without renormalization",
            "mass_multiplier": "mean(qbar*b), with mean(qbar)=1",
            "oracle_cross_condition": "median_t PCC(qbar*b_f, qbar*b_g)",
        },
        "forbidden_transformations_confirmed_absent": [
            "smoothing",
            "clipping",
            "interpolation",
            "pseudocounts",
            "positional-mean normalization of sampled or biased profiles",
        ],
        "workers": worker_count,
        "batch_size": args.batch_size,
        "validation": validation,
        "kinetic_target_crosscheck": K_crosscheck,
        "count_rng_interpretation": {
            "generator_source_present_in_repository": False,
            "artifact_level_conclusion": (
                "All count files declare the same observation seed and every one of the "
                "270 condition-pair/depth/replicate audits has excess exact equality over "
                "the independent-NB2 baseline; common random numbers are present."
            ),
            "unresolved_implementation_detail": (
                "Saved artifacts alone do not uniquely distinguish per-condition seed "
                "resetting from another coordinated common-stream implementation."
            ),
        },
    }
    write_json(work_results / "synthetic_observation_layer_provenance.json", provenance)
    # Remove depth CSV fragments only after the merged product and all validations exist.
    for result in depth_results:
        Path(result["observation_output"]).unlink()
    committed_results = _commit_files(
        [path for path in work_results.iterdir() if path.is_file()], output_dir
    )
    committed_figures = _commit_files(
        [path for path in work_figures.iterdir() if path.is_file()], figure_dir
    )
    # Keep the established audit-local figure copy synchronized with the public bundle.
    audit_figure_dir = output_dir / "figures"
    audit_figure_dir.mkdir(exist_ok=True)
    cross_stem = config["outputs"]["cross_condition_figure_stem"]
    if audit_figure_dir.resolve() != figure_dir.resolve():
        for extension in ("pdf", "png", "svg"):
            source = figure_dir / f"{cross_stem}.{extension}"
            shutil.copy2(source, audit_figure_dir / source.name)
    shutil.rmtree(work_dir)
    print("\nValidation summary", flush=True)
    print(f"  aligned transcripts: {validation['sequence_aligned_transcripts']:,}", flush=True)
    print(
        "  transcripts with every requested PCC defined: "
        f"{validation['transcripts_with_every_requested_pcc_defined']:,}",
        flush=True,
    )
    print(
        "  maximum mean-one deviation: "
        f"{validation['maximum_mean_one_deviation']:.3e}",
        flush=True,
    )
    print(
        "  maximum expectation identity deviation: "
        f"{validation['maximum_expectation_identity_deviation']:.3e}",
        flush=True,
    )
    print("\nMedian across the ten condition-specific medians", flush=True)
    for row in headline.itertuples(index=False):
        print(
            f"  C={row.nominal_reads_per_codon:g} {row.metric}: "
            f"{row.median_of_condition_medians:.4f} "
            f"[IQR {row.q25_of_condition_medians:.4f}, {row.q75_of_condition_medians:.4f}]",
            flush=True,
        )
    print("\nSampled-to-oracle off-diagonal matrix RMSE", flush=True)
    for row in oracle_rmse.itertuples(index=False):
        print(
            f"  C={row.nominal_reads_per_codon:g}: "
            f"{row.off_diagonal_rmse_to_oracle:.6f}",
            flush=True,
        )
    if shared_detected:
        print(
            "\nWARNING: shared/common NB2 random numbers detected: jointly unaffected "
            "positions show excess exact equality above the analytic independent-NB2 baseline.",
            flush=True,
        )
    else:
        print("\nNo shared NB2 randomness detected at the configured threshold.", flush=True)
    print(
        "K-target summary cross-check max absolute difference: "
        f"{K_crosscheck.get('maximum_absolute_summary_statistic_difference', float('nan')):.3e}",
        flush=True,
    )
    print(f"Wrote {len(committed_results)} result files and {len(committed_figures)} figure files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
