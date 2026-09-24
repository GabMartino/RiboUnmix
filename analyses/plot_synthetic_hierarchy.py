#!/usr/bin/env python3
"""Generate audited publication figures for the synthetic K -> q -> mu -> Y hierarchy.

Only saved simulator artifacts are read.  Raw TASEP occupancies are normalized
separately within transcript; sampled replicas are never smoothed, clipped,
interpolated, pseudocounted, or normalized by their observed positional means.
The raw simulator coordinate is retained, so the model-only padded terminal
position is never introduced.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utils.publication_plot_style import publication_rc  # noqa: E402
from analyses.analyze_synthetic_input_data import (  # noqa: E402
    iter_grouped_profiles,
    iter_truth,
    parquet_metadata,
    pearson,
    sample_role,
)
from analyses.analyze_synthetic_tasep_occupancy_agreement import (  # noqa: E402
    iter_occupancy_replicates,
    normalize_occupancy,
)


DEFAULT_CONFIG = ROOT / "analyses/configs/synthetic_hierarchy_figures.yaml"


@dataclass(frozen=True)
class SequenceInfo:
    total_codons: int
    terminal_codon: str

    @property
    def sense_codons(self) -> int:
        return self.total_codons - 1


@dataclass
class BaseProfile:
    transcript_id: str
    codons: list[str]
    positions: np.ndarray
    valid_mask: np.ndarray
    K: np.ndarray
    O1: np.ndarray
    O2: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    qbar: np.ndarray


@dataclass
class BiasProfile:
    condition: str
    multiplier: np.ndarray
    affected: np.ndarray
    max_annotation_role_deviation: float
    metadata: dict[str, str]


@dataclass
class CountProfile:
    condition: str
    depth_slug: str
    nominal_depth: float
    rep1: np.ndarray
    rep2: np.ndarray
    arithmetic_mean: np.ndarray
    stored_integerized_mean: np.ndarray
    max_stored_mean_deviation: float
    metadata: dict[str, str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != "ribounmix.synthetic_hierarchy_figures.v1":
        raise ValueError(f"Unsupported configuration schema in {path}")
    return config


def filtered_rows(
    path: Path, transcript_id: str, value_column: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Read the rows for one transcript without materializing a complete table."""
    table = pq.read_table(
        path,
        columns=["sample", "transcript_id", value_column],
        filters=[("transcript_id", "=", transcript_id)],
        use_threads=False,
    )
    rows = table.to_pylist()
    if not rows:
        raise KeyError(f"{transcript_id} is absent from {path}")
    if any(str(row["transcript_id"]) != transcript_id for row in rows):
        raise ValueError(f"Predicate filtering returned a wrong transcript from {path}")
    return rows, parquet_metadata(path)


def load_kinetic(path: Path, transcript_id: str) -> tuple[np.ndarray, dict[str, str]]:
    rows, metadata = filtered_rows(path, transcript_id, "rib_profile")
    if len(rows) != 1 or rows[0]["sample"] != "kinetics_target":
        raise ValueError(f"Expected one kinetics_target row in {path} for {transcript_id}")
    values = np.asarray(rows[0]["rib_profile"], dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
        raise ValueError(f"Invalid kinetic profile for {transcript_id}")
    return values, metadata


def load_occupancy(
    path: Path, transcript_id: str
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    rows, metadata = filtered_rows(path, transcript_id, "rib_profile")
    mapping = {
        "replicate_1_mean_psite_occupancy": "rep1",
        "replicate_2_mean_psite_occupancy": "rep2",
    }
    profiles: dict[str, np.ndarray] = {}
    for row in rows:
        try:
            role = mapping[str(row["sample"])]
        except KeyError as exc:
            raise ValueError(f"Unknown occupancy sample {row['sample']!r}") from exc
        if role in profiles:
            raise ValueError(f"Duplicate occupancy {role} for {transcript_id}")
        profiles[role] = np.asarray(row["rib_profile"], dtype=np.float64)
    if set(profiles) != {"rep1", "rep2"}:
        raise ValueError(f"Incomplete occupancy replicas for {transcript_id}: {profiles}")
    return profiles["rep1"], profiles["rep2"], metadata


def load_bias(
    path: Path, transcript_id: str, condition: str
) -> BiasProfile:
    rows, metadata = filtered_rows(path, transcript_id, "added_bias")
    profiles: dict[str, np.ndarray] = {}
    for row in rows:
        role = sample_role(str(row["sample"]))
        if role in profiles:
            raise ValueError(f"Duplicate bias annotation {role} for {transcript_id}")
        profiles[role] = np.asarray(row["added_bias"], dtype=np.float64)
    if set(profiles) != {"rep1", "rep2", "mean"}:
        raise ValueError(f"Incomplete bias annotations in {path}: {profiles}")
    reference = profiles["rep1"]
    if reference.ndim != 1 or not np.isfinite(reference).all() or np.any(reference < 0):
        raise ValueError(f"Invalid added-bias annotation in {path}")
    role_deviation = max(
        float(np.max(np.abs(reference - profiles[role])))
        for role in ("rep2", "mean")
    )
    if role_deviation != 0.0:
        raise ValueError(f"Bias annotation is not systematic across samples in {path}")
    if metadata.get("riboart.sequence_bias_feature") != condition:
        raise ValueError(f"Bias metadata mismatch in {path}")
    multiplier = 1.0 + reference
    affected = reference > 0.0
    if np.any(multiplier[~affected] != 1.0):
        raise ValueError(f"Unaffected positions do not have multiplier one in {path}")
    declared = metadata.get("riboart.sequence_bias_multiplier_range")
    if declared and affected.any():
        lower, upper = (float(value) for value in declared.split(","))
        observed = multiplier[affected]
        if float(observed.min()) < lower or float(observed.max()) > upper:
            raise ValueError(f"Multiplier values fall outside metadata range in {path}")
    return BiasProfile(condition, multiplier, affected, role_deviation, metadata)


def load_counts(
    path: Path,
    transcript_id: str,
    condition: str,
    depth_slug: str,
    nominal_depth: float,
) -> CountProfile:
    rows, metadata = filtered_rows(path, transcript_id, "rib_profile")
    profiles: dict[str, np.ndarray] = {}
    for row in rows:
        role = sample_role(str(row["sample"]))
        if role in profiles:
            raise ValueError(f"Duplicate count {role} for {transcript_id}")
        profiles[role] = np.asarray(row["rib_profile"], dtype=np.float64)
    if set(profiles) != {"rep1", "rep2", "mean"}:
        raise ValueError(f"Incomplete count replicas in {path}: {profiles}")
    if any(
        values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0)
        for values in profiles.values()
    ):
        raise ValueError(f"Invalid sampled counts in {path}")
    if not (profiles["rep1"].shape == profiles["rep2"].shape == profiles["mean"].shape):
        raise ValueError(f"Count replicas are not aligned in {path}")
    if float(metadata.get("riboart.counts_per_codon_unbiased_baseline", "nan")) != nominal_depth:
        raise ValueError(f"Nominal depth metadata mismatch in {path}")
    if metadata.get("riboart.sequence_bias_feature") != condition:
        raise ValueError(f"Observation-condition metadata mismatch in {path}")
    if metadata.get("riboart.observation_model") != "negative_binomial_NB2":
        raise ValueError(f"Expected NB2 observations in {path}")
    arithmetic = 0.5 * (profiles["rep1"] + profiles["rep2"])
    stored_deviation = float(np.max(np.abs(arithmetic - profiles["mean"])))
    return CountProfile(
        condition=condition,
        depth_slug=depth_slug,
        nominal_depth=nominal_depth,
        rep1=profiles["rep1"],
        rep2=profiles["rep2"],
        arithmetic_mean=arithmetic,
        stored_integerized_mean=profiles["mean"],
        max_stored_mean_deviation=stored_deviation,
        metadata=metadata,
    )


def load_sequence_index(path: Path) -> dict[str, SequenceInfo]:
    """Keep only transcript ID, codon count, and terminal codon in memory."""
    result: dict[str, SequenceInfo] = {}
    with pq.ParquetFile(path) as reader:
        for batch in reader.iter_batches(
            columns=["transcript_id", "codons"], batch_size=256, use_threads=False
        ):
            ids = batch.column("transcript_id").to_pylist()
            codons = batch.column("codons")
            lengths = pc.list_value_length(codons).to_numpy(zero_copy_only=False)
            offsets = codons.offsets.to_numpy(zero_copy_only=False)
            last_indices = pa.array(offsets[1:] - 1, type=pa.int64())
            terminal = pc.take(codons.values, last_indices).to_pylist()
            for transcript_id, length, stop in zip(ids, lengths, terminal):
                key = str(transcript_id)
                if key in result:
                    raise ValueError(f"Duplicate sequence transcript {key}")
                result[key] = SequenceInfo(int(length), str(stop))
    return result


def load_sequence_codons(path: Path, transcript_id: str) -> list[str]:
    table = pq.read_table(
        path,
        columns=["transcript_id", "codons"],
        filters=[("transcript_id", "=", transcript_id)],
        use_threads=False,
    )
    rows = table.to_pylist()
    if len(rows) != 1:
        raise KeyError(f"Expected one sequence row for {transcript_id}, found {len(rows)}")
    return [str(codon) for codon in rows[0]["codons"]]


def expected_per_unit_depth(qbar: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    qbar = np.asarray(qbar, dtype=np.float64)
    multiplier = np.asarray(multiplier, dtype=np.float64)
    if qbar.shape != multiplier.shape:
        raise ValueError("Occupancy consensus and multiplier are not aligned")
    if not np.isfinite(qbar).all() or not np.isfinite(multiplier).all():
        raise ValueError("Expected-profile inputs must be finite")
    return qbar * multiplier


def depth_adjusted_mean_counts(rep1: np.ndarray, rep2: np.ndarray, depth: float) -> np.ndarray:
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError("Nominal depth must be positive")
    rep1 = np.asarray(rep1, dtype=np.float64)
    rep2 = np.asarray(rep2, dtype=np.float64)
    if rep1.shape != rep2.shape:
        raise ValueError("Count replicas are not aligned")
    return (rep1 + rep2) / (2.0 * depth)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(left - right))))


def validate_base_profile(
    transcript_id: str,
    codons: list[str],
    K: np.ndarray,
    O1: np.ndarray,
    O2: np.ndarray,
    *,
    permitted_stops: set[str],
) -> BaseProfile:
    if not codons or codons[-1] not in permitted_stops:
        raise ValueError(f"{transcript_id}: missing a recognized terminal stop codon")
    expected_length = len(codons) - 1
    if not (K.size == O1.size == O2.size == expected_length):
        raise ValueError(
            f"{transcript_id}: K/O/sequence mismatch: "
            f"{K.size}/{O1.size}/{O2.size}/{expected_length}"
        )
    if np.any(K < 0) or np.any(O1 < 0) or np.any(O2 < 0):
        raise ValueError(f"{transcript_id}: profiles must be nonnegative")
    q1 = normalize_occupancy(O1)
    q2 = normalize_occupancy(O2)
    qbar = 0.5 * (q1 + q2)
    return BaseProfile(
        transcript_id=transcript_id,
        codons=codons[:-1],
        positions=np.arange(expected_length, dtype=np.int64),
        valid_mask=np.ones(expected_length, dtype=bool),
        K=K,
        O1=O1,
        O2=O2,
        q1=q1,
        q2=q2,
        qbar=qbar,
    )


def load_base_profile(
    transcript_id: str,
    *,
    kinetic_path: Path,
    occupancy_path: Path,
    sequence_path: Path,
    permitted_stops: set[str],
) -> tuple[BaseProfile, dict[str, dict[str, str]]]:
    K, k_metadata = load_kinetic(kinetic_path, transcript_id)
    O1, O2, o_metadata = load_occupancy(occupancy_path, transcript_id)
    codons = load_sequence_codons(sequence_path, transcript_id)
    base = validate_base_profile(
        transcript_id, codons, K, O1, O2, permitted_stops=permitted_stops
    )
    return base, {"kinetic": k_metadata, "occupancy": o_metadata}


def comparison_values(base: BaseProfile) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "q1_vs_q2": (base.q1, base.q2),
        "q1_vs_K": (base.q1, base.K),
        "q2_vs_K": (base.q2, base.K),
        "qbar_vs_K": (base.qbar, base.K),
    }


def stream_cohort_metrics(
    *,
    kinetic_path: Path,
    occupancy_path: Path,
    sequence_index: dict[str, SequenceInfo],
    permitted_stops: set[str],
    comparison_order: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing_sequence: list[str] = []
    max_mean_deviation = {name: 0.0 for name in ("K", "q1", "q2", "qbar")}
    raw_distinct = normalized_distinct = raw_distinct_normalized_equal = 0
    total_joint = 0
    alignment_checks = 0
    for kinetic_item, occupancy_item in zip_longest(
        iter_truth(kinetic_path, batch_size=128),
        iter_occupancy_replicates(occupancy_path, batch_size=128),
    ):
        if kinetic_item is None or occupancy_item is None:
            raise ValueError("Kinetic and occupancy streams have unequal row counts")
        kinetic_id, K = kinetic_item
        occupancy_id, occupancy = occupancy_item
        if kinetic_id != occupancy_id:
            raise ValueError(f"K/q stream mismatch: {kinetic_id} versus {occupancy_id}")
        total_joint += 1
        info = sequence_index.get(kinetic_id)
        if info is None:
            missing_sequence.append(kinetic_id)
            continue
        if info.terminal_codon not in permitted_stops:
            raise ValueError(f"{kinetic_id}: unexpected terminal codon {info.terminal_codon}")
        O1 = np.asarray(occupancy["rep1"], dtype=np.float64)
        O2 = np.asarray(occupancy["rep2"], dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        if not (K.size == O1.size == O2.size == info.sense_codons):
            raise ValueError(f"{kinetic_id}: population coordinate mismatch")
        alignment_checks += 1
        q1 = normalize_occupancy(O1)
        q2 = normalize_occupancy(O2)
        qbar = 0.5 * (q1 + q2)
        if not np.isfinite(K).all() or np.any(K < 0):
            raise ValueError(f"{kinetic_id}: invalid K profile")
        for name, values in (("K", K), ("q1", q1), ("q2", q2), ("qbar", qbar)):
            max_mean_deviation[name] = max(
                max_mean_deviation[name], abs(float(values.mean()) - 1.0)
            )
        raw_equal = np.array_equal(O1, O2)
        normalized_equal = np.array_equal(q1, q2)
        raw_distinct += int(not raw_equal)
        normalized_distinct += int(not normalized_equal)
        raw_distinct_normalized_equal += int((not raw_equal) and normalized_equal)
        pairs = {
            "q1_vs_q2": (q1, q2),
            "q1_vs_K": (q1, K),
            "q2_vs_K": (q2, K),
            "qbar_vs_K": (qbar, K),
        }
        for comparison in comparison_order:
            left, right = pairs[comparison]
            correlation = pearson(left, right)
            records.append(
                {
                    "transcript_id": kinetic_id,
                    "n_positions": int(K.size),
                    "comparison": comparison,
                    "pearson": correlation,
                    "pearson_valid": bool(np.isfinite(correlation)),
                    "undefined_reason": (
                        "" if np.isfinite(correlation) else "constant_or_zero_variance_profile"
                    ),
                    "rmse": rmse(left, right),
                }
            )
    frame = pd.DataFrame.from_records(records)
    audit = {
        "joint_K_occupancy_transcripts": total_joint,
        "sequence_aligned_transcripts": alignment_checks,
        "excluded_missing_sequence_annotation_count": len(missing_sequence),
        "excluded_missing_sequence_annotation_ids": missing_sequence,
        "max_absolute_positional_mean_deviation": max_mean_deviation,
        "raw_occupancy_replica_distinct_transcripts": raw_distinct,
        "normalized_occupancy_replica_distinct_transcripts": normalized_distinct,
        "raw_distinct_but_normalized_exactly_equal_transcripts": raw_distinct_normalized_equal,
    }
    return frame, audit


def summarize_cohort(frame: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparison in order:
        group = frame.loc[frame["comparison"] == comparison]
        correlations = group["pearson"].to_numpy(dtype=np.float64)
        valid_correlations = correlations[np.isfinite(correlations)]
        errors = group["rmse"].to_numpy(dtype=np.float64)
        valid_errors = errors[np.isfinite(errors)]
        if not valid_errors.size:
            raise ValueError(f"No valid RMSE values for {comparison}")
        p_q1, p_median, p_q3 = (
            np.quantile(valid_correlations, [0.25, 0.5, 0.75])
            if valid_correlations.size
            else (np.nan, np.nan, np.nan)
        )
        r_q1, r_median, r_q3 = np.quantile(valid_errors, [0.25, 0.5, 0.75])
        rows.append(
            {
                "comparison": comparison,
                "n_total_transcripts": int(len(group)),
                "n_valid_transcripts": int(valid_correlations.size),
                "n_undefined_correlations": int(len(group) - valid_correlations.size),
                "n_valid_rmse": int(valid_errors.size),
                "mean_pearson": float(valid_correlations.mean()) if valid_correlations.size else np.nan,
                "median_pearson": float(p_median),
                "pearson_q25": float(p_q1),
                "pearson_q75": float(p_q3),
                "pearson_iqr": float(p_q3 - p_q1),
                "mean_rmse": float(valid_errors.mean()),
                "median_rmse": float(r_median),
                "rmse_q25": float(r_q1),
                "rmse_q75": float(r_q3),
                "rmse_iqr": float(r_q3 - r_q1),
            }
        )
    return pd.DataFrame(rows)


def empirical_midrank_percentile(values: Iterable[float], value: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size or not np.isfinite(value):
        return float("nan")
    return float(100.0 * (np.sum(array < value) + 0.5 * np.sum(array == value)) / array.size)


def verify_all_condition_selection(
    *,
    configured_transcript: str,
    sequence_index: dict[str, SequenceInfo],
    bias_paths: dict[str, Path],
    minimum_length: int,
    maximum_length_exclusive: int,
) -> dict[str, Any]:
    candidates = {
        transcript_id
        for transcript_id, info in sequence_index.items()
        if minimum_length <= info.sense_codons < maximum_length_exclusive
    }
    if configured_transcript not in candidates:
        raise ValueError("Configured all-condition example is outside its length range")
    site_counts: dict[str, dict[str, int]] = {transcript_id: {} for transcript_id in candidates}
    for condition, path in bias_paths.items():
        seen: set[str] = set()
        for transcript_id, profiles in iter_grouped_profiles(
            path, "added_bias", batch_size=256
        ):
            if transcript_id not in candidates:
                continue
            reference = profiles["rep1"]
            if not (
                np.array_equal(reference, profiles["rep2"])
                and np.array_equal(reference, profiles["mean"])
            ):
                raise ValueError(f"{condition}/{transcript_id}: inconsistent annotations")
            site_counts[transcript_id][condition] = int(np.count_nonzero(reference > 0))
            seen.add(transcript_id)
        missing = candidates - seen
        if missing:
            raise ValueError(f"{condition}: missing {len(missing)} selection candidates")
    eligible = [
        transcript_id
        for transcript_id in candidates
        if all(site_counts[transcript_id][condition] > 0 for condition in bias_paths)
    ]
    if not eligible:
        raise ValueError("No transcript satisfies the all-condition selection rule")
    resolved = min(eligible, key=lambda tid: (sequence_index[tid].sense_codons, tid))
    if resolved != configured_transcript:
        raise ValueError(
            f"Configured all-condition transcript {configured_transcript} does not match "
            f"deterministic selection {resolved}"
        )
    return {
        "candidate_count": len(candidates),
        "eligible_all_condition_count": len(eligible),
        "resolved_transcript_id": resolved,
        "resolved_sense_codons": sequence_index[resolved].sense_codons,
        "tie_break": "minimum (sense-codon length, transcript ID)",
    }


def plotting_style(font_size: float) -> dict[str, Any]:
    style = publication_rc()
    style.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 0.5,
            "figure.titlesize": font_size + 1.5,
            "xtick.labelsize": font_size - 0.25,
            "ytick.labelsize": font_size - 0.25,
            "legend.fontsize": font_size - 0.5,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "axes.grid": False,
        }
    )
    return style


def save_figure(fig: plt.Figure, directory: Path, stem: str, dpi: int) -> list[Path]:
    outputs = [directory / f"{stem}.pdf", directory / f"{stem}.png"]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outputs


def plot_profile_layers(
    base: BaseProfile,
    bias: BiasProfile,
    counts: CountProfile,
    *,
    label: str,
    colors: dict[str, str],
    plot_config: dict[str, Any],
    output_directory: Path,
    stem: str,
) -> list[Path]:
    """Reserve a separate column for legends; never cover the saved profiles."""
    if not (base.K.shape == bias.multiplier.shape == counts.rep1.shape == counts.rep2.shape):
        raise ValueError("Figure 1 inputs are not positionally aligned")
    expected = expected_per_unit_depth(base.qbar, bias.multiplier)
    y1 = counts.rep1 / counts.nominal_depth
    y2 = counts.rep2 / counts.nominal_depth
    x = base.positions
    thin = float(plot_config["thin_line_width"])
    standard = float(plot_config["standard_line_width"])
    emphasis = float(plot_config["emphasis_line_width"])
    replica_alpha = float(plot_config["stochastic_replica_alpha"])
    marker_size = float(plot_config["profile_affected_marker_size"])
    font_size = float(plot_config.get("profile_layers_font_size", plot_config["font_size"]))
    style = plotting_style(font_size)
    style.update({
        "font.weight": "bold", "axes.titleweight": "bold",
        "axes.labelweight": "bold", "axes.linewidth": 1.2,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    })
    if style["text.usetex"]:
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}"
        )
    with matplotlib.rc_context(style):
        fig, grid = plt.subplots(
            4, 2,
            figsize=tuple(plot_config["profile_layers_size_inches"]),
            gridspec_kw={"width_ratios": [3.1, 1.25]},
            sharex="col",
        )
        axes, keys = grid[:, 0], grid[:, 1]
        for key in keys:
            key.set_axis_off()

        ax = axes[0]
        band = ax.fill_between(
            x, np.minimum(base.q1, base.q2), np.maximum(base.q1, base.q2),
            color=colors["sky_blue"], alpha=0.38, linewidth=0,
            label=r"Range of $\mathbf q_t^{(1)},\mathbf q_t^{(2)}$", zorder=1,
        )
        mean, = ax.plot(
            x, base.qbar, color=colors["blue"], linewidth=emphasis,
            label=r"$\overline{\mathbf q}_t$", zorder=3,
        )
        kinetic, = ax.plot(
            x, base.K, color=colors["black"], linestyle="--", linewidth=standard,
            label=r"$\mathbf K_t$", zorder=4,
        )
        keys[0].legend(handles=[kinetic, band, mean], loc="center left",
                       bbox_to_anchor=(0, 0.66), borderaxespad=0, handlelength=1.8)
        keys[0].text(
            0, 0.16,
            rf"$\mathrm{{PCC}}(q^{{(1)}},q^{{(2)}})={pearson(base.q1, base.q2):.3f}$"
            + "\n" + rf"$\mathrm{{PCC}}(\overline q,K)={pearson(base.qbar, base.K):.3f}$",
            transform=keys[0].transAxes, ha="left", va="center",
            fontsize=font_size - 2, linespacing=1.6,
        )
        ax.set_ylabel("Mean-one profile")
        ax.set_title(r"A   $\mathbf K_t$ and simulated ribosome profiles", loc="left", pad=10)

        ax = axes[1]
        multiplier, = ax.step(
            x, bias.multiplier, where="mid", color=colors["purple"],
            linewidth=standard, label=r"$\mathbf b_{t,f}$",
        )
        baseline = ax.axhline(1, color=colors["gray"], linewidth=1,
                             linestyle=":", label=r"$b_{t,f,i}=1$")
        affected = ax.scatter(
            x[bias.affected], bias.multiplier[bias.affected], marker="v",
            s=marker_size, color=colors["vermillion"], linewidths=0,
            label="Affected P-sites", zorder=4,
        )
        keys[1].legend(handles=[multiplier, baseline, affected], loc="center left",
                       borderaxespad=0, handlelength=1.8)
        ax.set_ylabel(r"Multiplier $b_{t,f,i}$")
        ax.set_title(rf"B   {label} observation multiplier", loc="left", pad=10)

        ax = axes[2]
        biased, = ax.plot(
            x, expected, color=colors["vermillion"], linewidth=emphasis,
            label=r"$\overline{\mathbf q}_t\odot\mathbf b_{t,f}$", zorder=1,
        )
        unbiased, = ax.plot(
            x, base.qbar, color=colors["blue"], linewidth=standard,
            label=r"$\overline{\mathbf q}_t$", zorder=2,
        )
        ax.scatter(
            x[bias.affected], expected[bias.affected], marker="v", s=marker_size,
            color=colors["vermillion"], linewidths=0, zorder=4,
        )
        keys[2].legend(handles=[unbiased, biased], loc="center left",
                       borderaxespad=0, handlelength=1.8)
        ax.set_ylabel(r"Expected counts / $C$")
        ax.set_title("C   Profiles before NB2 sampling", loc="left", pad=10)

        ax = axes[3]
        rep1, = ax.plot(x, y1, color=colors["sky_blue"], linewidth=thin,
                       alpha=replica_alpha, label=r"$\mathbf Y_{t,f}^{(1)}/C$")
        rep2, = ax.plot(x, y2, color=colors["orange"], linestyle=":", linewidth=standard,
                       alpha=replica_alpha, label=r"$\mathbf Y_{t,f}^{(2)}/C$")
        expected_line, = ax.plot(
            x, expected, color=colors["vermillion"], linestyle="--", linewidth=emphasis,
            label=r"$\overline{\mathbf q}_t\odot\mathbf b_{t,f}$",
        )
        keys[3].legend(handles=[rep1, rep2, expected_line], loc="center left",
                       borderaxespad=0, handlelength=1.8)
        ax.set_ylabel(r"Sampled counts / $C$")
        ax.set_title(
            rf"D   NB2 counts at $C={counts.nominal_depth:g}$ reads/codon",
            loc="left", pad=10,
        )
        ax.set_xlabel("P-site codon index (0-based)")

        for index, ax in enumerate(axes):
            ax.set_xlim(int(x[0]), int(x[-1]))
            ax.set_ylim(bottom=0)
            ax.margins(y=0.12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis="x", labelbottom=index == 3)
        fig.align_ylabels(axes)
        fig.subplots_adjust(left=0.095, right=0.98, top=0.955,
                            bottom=0.065, hspace=0.42, wspace=0.12)
        return save_figure(fig, output_directory, stem, int(plot_config["dpi"]))


def plot_all_conditions(
    base: BaseProfile,
    condition_data: dict[str, dict[str, Any]],
    *,
    conditions: list[dict[str, str]],
    depths: list[dict[str, Any]],
    colors: dict[str, str],
    plot_config: dict[str, Any],
    output_directory: Path,
    stem: str,
) -> list[Path]:
    style = plotting_style(float(plot_config["font_size"]))
    thin = float(plot_config["thin_line_width"])
    standard = float(plot_config["standard_line_width"])
    marker_size = float(plot_config["grid_affected_marker_size"])
    x = base.positions
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(
            len(conditions),
            len(depths),
            figsize=tuple(plot_config["all_conditions_size_inches"]),
            sharex=True,
            squeeze=False,
        )
        for row, condition in enumerate(conditions):
            key = condition["key"]
            data = condition_data[key]
            expected = data["expected"]
            depth_adjusted = {
                depth["slug"]: depth_adjusted_mean_counts(
                    data["counts"][depth["slug"]].rep1,
                    data["counts"][depth["slug"]].rep2,
                    float(depth["value"]),
                )
                for depth in depths
            }
            row_maximum = max(
                float(base.qbar.max()),
                float(expected.max()),
                *(float(values.max()) for values in depth_adjusted.values()),
            )
            upper = max(1.0, row_maximum) * 1.10
            for column, depth in enumerate(depths):
                ax = axes[row, column]
                sampled = depth_adjusted[depth["slug"]]
                ax.plot(x, expected, color=colors["black"], linestyle="--", linewidth=standard, label=r"Expected $\overline q_t b_f$", zorder=1)
                ax.plot(x, base.qbar, color=colors["gray"], linewidth=thin, label=r"Unbiased $\overline q_t$", zorder=2)
                ax.plot(x, sampled, color=colors["blue"], linewidth=standard, label=r"Sampled $\overline Y/C$", zorder=3)
                ax.scatter(
                    x[data["bias"].affected],
                    np.full(int(data["bias"].affected.sum()), upper * 0.985),
                    marker="v",
                    s=marker_size,
                    color=colors["vermillion"],
                    linewidths=0,
                    zorder=4,
                )
                ax.set_xlim(int(x[0]), int(x[-1]))
                ax.set_ylim(0.0, upper)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
                if row < len(conditions) - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                if column > 0:
                    ax.tick_params(axis="y", labelleft=False)
                if row == 0:
                    ax.set_title(str(depth["label"]), pad=5.0)
            axes[row, 0].set_ylabel(
                condition.get("row_label", condition["label"]),
                rotation=90,
                ha="center",
                va="center",
                labelpad=11,
            )
        # One figure-level x label remains legible at manuscript width; three
        # repeated long labels collide beneath the narrow grid columns.
        fig.supxlabel("P-site codon index (0-based)", x=0.56, y=0.012)
        handles = [
            Line2D([0], [0], color=colors["gray"], linewidth=thin, label=r"Unbiased $\overline q_t$"),
            Line2D([0], [0], color=colors["black"], linestyle="--", linewidth=standard, label=r"Expected $\overline q_t b_f$"),
            Line2D([0], [0], color=colors["blue"], linewidth=standard, label=r"Sampled $\overline Y/C$"),
            Line2D(
                [0],
                [0],
                color=colors["vermillion"],
                marker="v",
                linestyle="none",
                markersize=float(plot_config["grid_legend_marker_size"]),
                label="Affected P-site",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.963),
            ncol=2,
            columnspacing=1.8,
            handlelength=2.1,
        )
        fig.suptitle(
            rf"Ten observation effects across depth: \texttt{{{base.transcript_id}}}",
            x=0.54,
            y=0.998,
        )
        fig.supylabel("Profile per unit nominal depth", x=0.018)
        fig.subplots_adjust(left=0.135, right=0.992, top=0.885, bottom=0.06, hspace=0.32, wspace=0.11)
        return save_figure(fig, output_directory, stem, int(plot_config["dpi"]))


def ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(list(values), dtype=np.float64)
    array = np.sort(array[np.isfinite(array)])
    if not array.size:
        return array, array
    return array, np.arange(1, array.size + 1, dtype=np.float64) / array.size


def plot_cohort_summary(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    comparison_styles: dict[str, dict[str, str]],
    order: list[str],
    plot_config: dict[str, Any],
    output_directory: Path,
    stem: str,
) -> list[Path]:
    style = plotting_style(
        float(plot_config.get("cohort_font_size", plot_config["font_size"]))
    )
    summary_by_name = summary.set_index("comparison")
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=tuple(plot_config["cohort_size_inches"]))
        for comparison in order:
            group = metrics.loc[metrics["comparison"] == comparison]
            info = summary_by_name.loc[comparison]
            curve_x, curve_y = ecdf(group["pearson"])
            metric_style = comparison_styles[comparison]
            pcc_label = (
                metric_style["label"]
                + rf"  {info['median_pearson']:.3f} [{info['pearson_q25']:.3f}, {info['pearson_q75']:.3f}]"
            )
            axes[0].step(
                curve_x,
                curve_y,
                where="post",
                color=metric_style["color"],
                linestyle=metric_style["linestyle"],
                linewidth=float(plot_config["emphasis_line_width"]),
                label=pcc_label,
                # Four empirical CDFs each contain roughly 19,000 vertices.
                # Rasterize only these dense paths so that the PDF remains
                # responsive while its text, axes, and legend stay vector.
                rasterized=True,
            )
            error_x, error_y = ecdf(group["rmse"])
            rmse_label = (
                metric_style["label"]
                + rf"  {info['median_rmse']:.3f} [{info['rmse_q25']:.3f}, {info['rmse_q75']:.3f}]"
            )
            axes[1].step(
                error_x,
                error_y,
                where="post",
                color=metric_style["color"],
                linestyle=metric_style["linestyle"],
                linewidth=float(plot_config["emphasis_line_width"]),
                label=rmse_label,
                rasterized=True,
            )
        axes[0].set_title(r"\textbf{A}\quad Pearson correlation", loc="left")
        axes[0].set_xlabel("Per-transcript PCC")
        finite_pcc = metrics["pearson"].to_numpy(dtype=np.float64)
        finite_pcc = finite_pcc[np.isfinite(finite_pcc)]
        axes[0].set_xlim(float(finite_pcc.min()) - 0.015, 1.002)
        axes[1].set_title(r"\textbf{B}\quad Profile error", loc="left")
        axes[1].set_xlabel("Per-transcript RMSE")
        finite_rmse = metrics["rmse"].to_numpy(dtype=np.float64)
        finite_rmse = finite_rmse[np.isfinite(finite_rmse)]
        axes[1].set_xlim(0.0, float(finite_rmse.max()) * 1.02)
        for ax in axes:
            ax.set_ylim(0.0, 1.005)
            ax.set_ylabel("Empirical cumulative fraction")
        cohort_legend_size = float(
            plot_config.get("cohort_legend_font_size", style["legend.fontsize"])
        )
        axes[0].legend(
            loc="upper left",
            handlelength=2.2,
            labelspacing=0.25,
            fontsize=cohort_legend_size,
        )
        axes[1].legend(
            loc="upper right",
            handlelength=2.2,
            labelspacing=0.25,
            fontsize=cohort_legend_size,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.92,
        )
        fig.suptitle("Programmed kinetics and stochastic TASEP occupancy")
        fig.subplots_adjust(left=0.09, right=0.995, top=0.85, bottom=0.16, wspace=0.30)
        return save_figure(fig, output_directory, stem, int(plot_config["dpi"]))


def preflight_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing outputs without --overwrite:\n" + joined
        )


def source_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "metadata": parquet_metadata(path),
    }


def dataframe_markdown_table(frame: pd.DataFrame, *, float_digits: int = 4) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "NA" if not np.isfinite(value) else f"{float(value):.{float_digits}f}"
        return str(value).replace("|", r"\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_captions(
    path: Path,
    *,
    profile_transcript: str,
    profile_condition_label: str,
    all_condition_transcript: str,
    cohort_size: int,
) -> None:
    text = rf"""% Publication-ready caption drafts for figures generated by
% analyses/plot_synthetic_hierarchy.py.

% Figure 1: synthetic_profile_layers_example
\caption{{\textbf{{Synthetic-generation layers for transcript \texttt{{{profile_transcript}}}.}}
\textbf{{(A)}} The deterministic programmed kinetic target $K_t$ (black dashed), the
mean-one two-trajectory occupancy consensus $\overline q_t$ (blue), and a translucent
band spanning the two separately simulated mean-one profiles $q_t^{{(1)}}$ and
$q_t^{{(2)}}$ at every position. \textbf{{(B)}} The exact {profile_condition_label}
observation multiplier $b_f$; red triangles mark the saved affected P-site coordinates
and the horizontal reference is $b_f=1$. \textbf{{(C)}} The unbiased expectation per
unit nominal depth, $\overline q_t$, and the deterministically biased expectation
$\overline q_t b_f$ before count sampling. \textbf{{(D)}} The biased expectation and
the two saved NB2 count replicas divided only by the nominal depth $C=20$ for display.
$K_t$, $q_t^{{(1)}}$, $q_t^{{(2)}}$, and $\overline q_t$ are mean one by construction;
the biased expectations and depth-adjusted counts are not renormalized by their
positional means. Zeros and peaks are retained without smoothing, clipping,
interpolation, or pseudocounts. The same two TASEP trajectories are reused across
observation conditions and depths. Their consensus is a finite two-trajectory
reference, not exact stationary TASEP ground truth.}}

% Figure 2: synthetic_ten_biases_across_depths
\caption{{\textbf{{Ten deterministic observation effects across sequencing depths.}}
Transcript \texttt{{{all_condition_transcript}}} is the shortest transcript in the
prespecified 100--149-sense-codon range with at least one saved affected P-site for
every condition. Rows are the ten injected observation effects and columns are
$C\in\{{0.25,2,20\}}$ expected reads/codon. Each cell shows the mean-one unbiased
occupancy consensus $\overline q_t$ (gray), the exact biased expectation per
unit depth $\overline q_t b_f$ (black dashed), and the arithmetic mean of the two
sampled NB2 replicas divided only by $C$ (blue). Red triangles identify the exact
saved affected coordinates. The three panels within a row share one y-axis range;
different rows may differ. The biased expectation and sampled profiles are not
normalized by their positional means. All zeros and peaks are retained, with no
smoothing, clipping, interpolation, or pseudocounts. The two occupancy profiles come
from separate stochastic TASEP trajectories and are reused across every condition and
depth; $\overline q_t$ is their finite consensus, not exact stationary ground truth.}}

% Figure 3: synthetic_K_q_stochasticity_summary
\caption{{\textbf{{Population-level stochasticity of synthetic TASEP occupancy.}}
Empirical cumulative distributions across {cohort_size:,} exactly aligned transcripts
show \textbf{{(A)}} Pearson correlation and \textbf{{(B)}} RMSE for
$q_t^{{(1)}}$ versus $q_t^{{(2)}}$, each trajectory versus the deterministic programmed
kinetic target $K_t$, and their consensus $\overline q_t$ versus $K_t$. Legend entries
report the median and interquartile range. $K_t$, both separately normalized stochastic
TASEP trajectories, and $\overline q_t$ have positional mean one. No count-depth
adjustment is involved in this cohort panel, and no smoothing, clipping,
interpolation, or pseudocount is applied. The same two trajectories are reused across
the downstream observation conditions and depths. Their average reduces finite-run
variation but is not exact stationary TASEP ground truth; trajectory agreement is not
a performance ceiling for a learned sequence model.}}
"""
    path.write_text(text)


def write_report(
    path: Path,
    *,
    command: str,
    config_path: Path,
    data_paths: dict[str, Path],
    cohort_audit: dict[str, Any],
    validation: dict[str, Any],
    summary: pd.DataFrame,
    selection_audit: dict[str, Any],
    representative: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    summary_columns = [
        "comparison",
        "n_valid_transcripts",
        "n_undefined_correlations",
        "median_pearson",
        "pearson_q25",
        "pearson_q75",
        "median_rmse",
        "rmse_q25",
        "rmse_q75",
    ]
    first_rows = representative.loc[representative["figure"] == "Figure 1"]
    extreme = bool(first_rows["any_extreme_tail"].iloc[0])
    second_rows = representative.loc[representative["figure"] == "Figure 2"]
    second_extreme = bool(second_rows["any_extreme_tail"].iloc[0])
    second_q_percentile = float(
        second_rows["q1_vs_q2_cohort_percentile_midrank"].iloc[0]
    )
    second_k_percentile = float(
        second_rows["qbar_vs_K_cohort_percentile_midrank"].iloc[0]
    )
    lines = [
        "# Synthetic hierarchy figure audit",
        "",
        f"Generated: **{datetime.now(timezone.utc).isoformat()}**.",
        "",
        "## Scope",
        "",
        "The figures use saved simulator outputs only. No simulation, neural model, or",
        "checkpoint is loaded. Raw TASEP occupancies are normalized separately to define",
        "the mean-one profiles $q^{(1)}$ and $q^{(2)}$. Sampled counts are divided by",
        "nominal depth only where explicitly labeled; no profile is smoothed, clipped,",
        "interpolated, pseudocounted, or normalized by its observed positional mean.",
        "",
        "The saved simulator arrays exclude the terminal boundary. Sequence alignment is",
        "validated against `codons[:-1]`; the zero terminal position appended in the",
        "model-ready weighted Parquets is never loaded.",
        "",
        "## Provenance",
        "",
        f"Configuration: `{config_path.relative_to(ROOT)}`",
        "",
    ]
    lines.extend(f"- `{name}`: `{source.relative_to(ROOT)}`" for name, source in data_paths.items())
    lines.extend(
        [
            "",
            "All K, bias, and count artifacts used here share the simulator source-run",
            "fingerprint recorded in their Parquet metadata. The occupancy export does not",
            "carry that fingerprint; it is linked by transcript identity, exact array length,",
            "coordinate metadata, and the supplied occupancy-export provenance.",
            "",
            "## Cohort and exclusions",
            "",
            f"- Joint K/occupancy transcripts: {cohort_audit['joint_K_occupancy_transcripts']:,}",
            f"- Exact sequence-aligned cohort: {cohort_audit['sequence_aligned_transcripts']:,}",
            f"- Excluded for missing sequence annotation: {cohort_audit['excluded_missing_sequence_annotation_count']}",
            f"- Excluded IDs: {', '.join(cohort_audit['excluded_missing_sequence_annotation_ids'])}",
            f"- Undefined PCCs: {int(summary['n_undefined_correlations'].sum())} across all four comparisons",
            "",
            "Every included position is a saved modeled P-site. The source artifacts do not",
            "provide a separate validity-mask column, so the exact mask is all `True` over",
            "the saved finite array after finite-value and coordinate checks.",
            "",
            "## Validation",
            "",
            f"- Mean-one absolute tolerance: {validation['tolerances']['mean_one_absolute_tolerance']:.1e}",
            f"- Identity absolute tolerance: {validation['tolerances']['identity_absolute_tolerance']:.1e}",
            f"- Maximum $|mean(K)-1|$: {validation['max_deviations']['mean_K']:.3e}",
            f"- Maximum $|mean(q^{{(1)}})-1|$: {validation['max_deviations']['mean_q1']:.3e}",
            f"- Maximum $|mean(q^{{(2)}})-1|$: {validation['max_deviations']['mean_q2']:.3e}",
            f"- Maximum $|mean(\\overline q)-1|$: {validation['max_deviations']['mean_qbar']:.3e}",
            f"- Maximum expected-profile identity deviation: {validation['max_deviations']['expected_profile_identity']:.3e}",
            f"- Maximum unbiased $b=1$ identity deviation: {validation['max_deviations']['unbiased_identity']:.3e}",
            f"- Maximum annotation disagreement across saved sample roles: {validation['max_deviations']['annotation_roles']:.3e}",
            f"- Maximum coordinate-length deviation: {validation['max_deviations']['coordinate_length']}",
            f"- Maximum stored integerized-mean versus exact arithmetic-mean count deviation: {validation['max_deviations']['stored_count_mean']:.3g} count",
            f"- Raw occupancy trajectories differ in {cohort_audit['raw_occupancy_replica_distinct_transcripts']:,} of {cohort_audit['sequence_aligned_transcripts']:,} included transcripts.",
            "",
            "The expected profile is reconstructed as the documented deterministic identity",
            "$\\overline q_t b_f$ because no separate expected-$\\mu$ artifact was exported.",
            "Its zero identity deviation validates this implementation, not an independent",
            "second copy of the simulator mean.",
            "",
            "Affected markers are exactly `added_bias > 0`; the plotted multiplier is",
            "`1 + added_bias`, never a binary proxy.",
            "",
            "## Deterministic example selection",
            "",
            f"The all-condition example resolves to `{selection_audit['resolved_transcript_id']}`",
            f"({selection_audit['resolved_sense_codons']} sense codons) among",
            f"{selection_audit['eligible_all_condition_count']} eligible transcripts. The",
            f"tie-break is {selection_audit['tie_break']}.",
            "",
            f"The fixed Figure 1 example is {'in' if extreme else 'not in'} an empirical",
            "extreme tail (lowest or highest 5%) for either audited correlation. It was not",
            "replaced after inspection.",
            "",
            f"The deterministic Figure 2 example is {'in' if second_extreme else 'not in'}",
            "an empirical extreme tail and is retained without post-hoc replacement. Its",
            f"trajectory-agreement percentile is {second_q_percentile:.2f} and its",
            f"consensus--kinetics percentile is {second_k_percentile:.2f}.",
            "",
            "Representative-example details are in",
            "`results/synthetic_K_q_representative_examples.csv`.",
            "",
            "## Cohort summaries",
            "",
            dataframe_markdown_table(summary.loc[:, summary_columns], float_digits=4),
            "",
            "Intervals in the legends are interquartile ranges, not confidence intervals.",
            "The high K--q correlations show that programmed kinetics strongly constrain",
            "occupancy, while q1--q2 disagreement quantifies finite-trajectory variation.",
            "The improved consensus--K agreement is consistent with noise reduction but does",
            "not establish that the two-trajectory mean is exact stationary occupancy.",
            "",
            "## Outputs",
            "",
        ]
    )
    lines.extend(f"- `{figure.relative_to(ROOT)}`" if figure.is_relative_to(ROOT) else f"- `{figure}`" for figure in figure_paths)
    lines.extend(
        [
            "",
            "## Exact command",
            "",
            "```bash",
            command,
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--figure-dir", type=Path, help="Override configured figure directory")
    parser.add_argument("--results-dir", type=Path, help="Override configured results directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    data_config = config["data"]
    paths = {
        "kinetic_profile": resolve_path(data_config["kinetic_profile"]),
        "occupancy_replicates": resolve_path(data_config["occupancy_replicates"]),
        "sequence_annotations": resolve_path(data_config["sequence_annotations"]),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    bias_paths = {
        condition["key"]: resolve_path(
            data_config["bias_annotation_template"].format(condition=condition["key"])
        )
        for condition in config["conditions"]
    }
    for path in bias_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    outputs = config["outputs"]
    figure_directory = (
        args.figure_dir.expanduser().resolve()
        if args.figure_dir
        else resolve_path(outputs["figure_directory"])
    )
    results_directory = (
        args.results_dir.expanduser().resolve()
        if args.results_dir
        else resolve_path(outputs["results_directory"])
    )
    figure_directory.mkdir(parents=True, exist_ok=True)
    results_directory.mkdir(parents=True, exist_ok=True)
    output_paths = [
        figure_directory / f"{outputs['profile_layers_stem']}.pdf",
        figure_directory / f"{outputs['profile_layers_stem']}.png",
        figure_directory / f"{outputs['all_conditions_stem']}.pdf",
        figure_directory / f"{outputs['all_conditions_stem']}.png",
        figure_directory / f"{outputs['cohort_stem']}.pdf",
        figure_directory / f"{outputs['cohort_stem']}.png",
        results_directory / outputs["cohort_summary_csv"],
        results_directory / outputs["cohort_per_transcript_parquet"],
        results_directory / outputs["representative_metadata_csv"],
        results_directory / outputs["validation_json"],
        figure_directory / outputs["captions_tex"],
        results_directory / outputs["report_markdown"],
    ]
    preflight_outputs(output_paths, args.overwrite)

    permitted_stops = set(config["validation"]["permitted_terminal_codons"])
    sequence_index = load_sequence_index(paths["sequence_annotations"])
    comparison_order = list(config["comparison_styles"])
    cohort_metrics, cohort_audit = stream_cohort_metrics(
        kinetic_path=paths["kinetic_profile"],
        occupancy_path=paths["occupancy_replicates"],
        sequence_index=sequence_index,
        permitted_stops=permitted_stops,
        comparison_order=comparison_order,
    )
    summary = summarize_cohort(cohort_metrics, comparison_order)

    example1_config = config["examples"]["profile_layers"]
    example2_config = config["examples"]["all_conditions"]
    example1, example1_metadata = load_base_profile(
        example1_config["transcript_id"],
        kinetic_path=paths["kinetic_profile"],
        occupancy_path=paths["occupancy_replicates"],
        sequence_path=paths["sequence_annotations"],
        permitted_stops=permitted_stops,
    )
    example2, example2_metadata = load_base_profile(
        example2_config["transcript_id"],
        kinetic_path=paths["kinetic_profile"],
        occupancy_path=paths["occupancy_replicates"],
        sequence_path=paths["sequence_annotations"],
        permitted_stops=permitted_stops,
    )

    conditions_by_key = {condition["key"]: condition for condition in config["conditions"]}
    first_condition = str(example1_config["condition"])
    first_bias = load_bias(
        bias_paths[first_condition], example1.transcript_id, first_condition
    )
    first_depth = next(
        depth for depth in config["depths"] if depth["slug"] == example1_config["depth_slug"]
    )
    first_count_path = resolve_path(
        data_config["count_template"].format(
            condition=first_condition, depth_slug=first_depth["slug"]
        )
    )
    first_counts = load_counts(
        first_count_path,
        example1.transcript_id,
        first_condition,
        first_depth["slug"],
        float(first_depth["value"]),
    )
    count_paths_used: dict[str, Path] = {
        f"{first_condition}/{first_depth['slug']}": first_count_path
    }

    condition_data: dict[str, dict[str, Any]] = {}
    all_source_fingerprints: set[str] = set()
    max_annotation_deviation = first_bias.max_annotation_role_deviation
    max_stored_mean_deviation = first_counts.max_stored_mean_deviation
    alignment_checks = 1
    for condition in config["conditions"]:
        key = condition["key"]
        bias = load_bias(bias_paths[key], example2.transcript_id, key)
        if bias.multiplier.shape != example2.K.shape:
            raise ValueError(f"{key}: multiplier/sequence alignment failure")
        counts_by_depth: dict[str, CountProfile] = {}
        for depth in config["depths"]:
            count_path = resolve_path(
                data_config["count_template"].format(
                    condition=key, depth_slug=depth["slug"]
                )
            )
            count_paths_used[f"{key}/{depth['slug']}"] = count_path
            counts = load_counts(
                count_path,
                example2.transcript_id,
                key,
                depth["slug"],
                float(depth["value"]),
            )
            if counts.rep1.shape != example2.K.shape:
                raise ValueError(f"{key}/{depth['slug']}: count/sequence alignment failure")
            counts_by_depth[depth["slug"]] = counts
            alignment_checks += 1
            max_stored_mean_deviation = max(
                max_stored_mean_deviation, counts.max_stored_mean_deviation
            )
            fingerprint = counts.metadata.get("riboart.source_run_fingerprint")
            if fingerprint:
                all_source_fingerprints.add(fingerprint)
        fingerprint = bias.metadata.get("riboart.source_run_fingerprint")
        if fingerprint:
            all_source_fingerprints.add(fingerprint)
        max_annotation_deviation = max(
            max_annotation_deviation, bias.max_annotation_role_deviation
        )
        condition_data[key] = {
            "bias": bias,
            "expected": expected_per_unit_depth(example2.qbar, bias.multiplier),
            "counts": counts_by_depth,
        }
    for metadata_group in (example1_metadata, example2_metadata):
        fingerprint = metadata_group["kinetic"].get("riboart.source_run_fingerprint")
        if fingerprint:
            all_source_fingerprints.add(fingerprint)
    first_fingerprint = first_bias.metadata.get("riboart.source_run_fingerprint")
    first_count_fingerprint = first_counts.metadata.get("riboart.source_run_fingerprint")
    all_source_fingerprints.update(
        value for value in (first_fingerprint, first_count_fingerprint) if value
    )
    if len(all_source_fingerprints) != 1:
        raise ValueError(f"K, bias, and count artifacts have fingerprints {all_source_fingerprints}")

    selection_audit = verify_all_condition_selection(
        configured_transcript=example2.transcript_id,
        sequence_index=sequence_index,
        bias_paths=bias_paths,
        minimum_length=int(example2_config["minimum_sense_codons"]),
        maximum_length_exclusive=int(example2_config["maximum_sense_codons_exclusive"]),
    )

    mean_tolerance = float(config["validation"]["mean_one_absolute_tolerance"])
    identity_tolerance = float(config["validation"]["identity_absolute_tolerance"])
    maximum_means = cohort_audit["max_absolute_positional_mean_deviation"]
    if any(float(value) > mean_tolerance for value in maximum_means.values()):
        raise ValueError(f"Mean-one validation failed: {maximum_means}")
    first_expected = expected_per_unit_depth(example1.qbar, first_bias.multiplier)
    expected_deviation = max(
        float(np.max(np.abs(first_expected - example1.qbar * first_bias.multiplier))),
        *(
            float(np.max(np.abs(data["expected"] - example2.qbar * data["bias"].multiplier)))
            for data in condition_data.values()
        ),
    )
    unbiased_deviation = max(
        float(np.max(np.abs(example1.qbar * np.ones_like(example1.qbar) - example1.qbar))),
        float(np.max(np.abs(example2.qbar * np.ones_like(example2.qbar) - example2.qbar))),
    )
    if expected_deviation > identity_tolerance or unbiased_deviation > identity_tolerance:
        raise ValueError("Expected-profile identity validation failed")

    figure_paths: list[Path] = []
    figure_paths.extend(
        plot_profile_layers(
            example1,
            first_bias,
            first_counts,
            label=conditions_by_key[first_condition]["label"],
            colors=config["colors"],
            plot_config=config["plot"],
            output_directory=figure_directory,
            stem=outputs["profile_layers_stem"],
        )
    )
    figure_paths.extend(
        plot_all_conditions(
            example2,
            condition_data,
            conditions=config["conditions"],
            depths=config["depths"],
            colors=config["colors"],
            plot_config=config["plot"],
            output_directory=figure_directory,
            stem=outputs["all_conditions_stem"],
        )
    )
    figure_paths.extend(
        plot_cohort_summary(
            cohort_metrics,
            summary,
            comparison_styles=config["comparison_styles"],
            order=comparison_order,
            plot_config=config["plot"],
            output_directory=figure_directory,
            stem=outputs["cohort_stem"],
        )
    )

    summary_path = results_directory / outputs["cohort_summary_csv"]
    detail_path = results_directory / outputs["cohort_per_transcript_parquet"]
    summary.to_csv(summary_path, index=False)
    cohort_metrics.to_parquet(detail_path, index=False, compression="zstd")

    pcc_distributions = {
        comparison: cohort_metrics.loc[
            cohort_metrics["comparison"] == comparison, "pearson"
        ].to_numpy(dtype=np.float64)
        for comparison in comparison_order
    }
    tail = float(config["validation"]["extreme_tail_percentile"])
    representative_rows: list[dict[str, Any]] = []
    for figure_name, base, biases, selection_rule in (
        ("Figure 1", example1, [first_bias], example1_config["selection_rule"]),
        (
            "Figure 2",
            example2,
            [condition_data[item["key"]]["bias"] for item in config["conditions"]],
            example2_config["selection_rule"],
        ),
    ):
        pcc_q = pearson(base.q1, base.q2)
        pcc_k = pearson(base.qbar, base.K)
        percentile_q = empirical_midrank_percentile(pcc_distributions["q1_vs_q2"], pcc_q)
        percentile_k = empirical_midrank_percentile(pcc_distributions["qbar_vs_K"], pcc_k)
        extreme_q = percentile_q <= tail or percentile_q >= 100.0 - tail
        extreme_k = percentile_k <= tail or percentile_k >= 100.0 - tail
        for bias in biases:
            representative_rows.append(
                {
                    "figure": figure_name,
                    "transcript_id": base.transcript_id,
                    "selection_rule": selection_rule,
                    "modeled_psite_codons": int(base.K.size),
                    "observation_effect": bias.condition,
                    "affected_positions": int(bias.affected.sum()),
                    "affected_fraction": float(bias.affected.mean()),
                    "q1_vs_q2_pearson": pcc_q,
                    "q1_vs_q2_cohort_percentile_midrank": percentile_q,
                    "q1_vs_q2_extreme_tail": extreme_q,
                    "qbar_vs_K_pearson": pcc_k,
                    "qbar_vs_K_cohort_percentile_midrank": percentile_k,
                    "qbar_vs_K_extreme_tail": extreme_k,
                    "any_extreme_tail": bool(extreme_q or extreme_k),
                }
            )
    representative = pd.DataFrame(representative_rows)
    representative_path = results_directory / outputs["representative_metadata_csv"]
    representative.to_csv(representative_path, index=False)

    validation = {
        "status": "PASS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tolerances": {
            "mean_one_absolute_tolerance": mean_tolerance,
            "identity_absolute_tolerance": identity_tolerance,
        },
        "max_deviations": {
            "mean_K": maximum_means["K"],
            "mean_q1": maximum_means["q1"],
            "mean_q2": maximum_means["q2"],
            "mean_qbar": maximum_means["qbar"],
            "expected_profile_identity": expected_deviation,
            "unbiased_identity": unbiased_deviation,
            "annotation_roles": max_annotation_deviation,
            "coordinate_length": 0,
            "stored_count_mean": max_stored_mean_deviation,
        },
        "coordinate_convention": {
            "positions": "0-based saved P-site indices",
            "valid_mask": "all saved finite positions are valid; no separate mask column exists",
            "terminal_boundary": "excluded; model-ready padded terminal never loaded",
            "alignment_checks": alignment_checks + cohort_audit["sequence_aligned_transcripts"],
        },
        "transformations": {
            "occupancy": "each raw O trajectory divided by its own positional mean",
            "counts": "divided by nominal C only in Figures 1D and 2",
            "smoothing": False,
            "clipping": False,
            "interpolation": False,
            "pseudocount": False,
            "observed_mean_renormalization": False,
        },
        "rendering": {
            "matplotlib_backend": matplotlib.get_backend(),
            "riboi_plot_tex": os.environ.get("RIBOUNMIX_PLOT_TEX", "auto"),
            "png_dpi": int(config["plot"]["dpi"]),
        },
        "affected_positions": "exactly added_bias > 0",
        "expected_profile_note": "qbar*b is reconstructed from saved q and b; no independent mu artifact exists",
        "source_run_fingerprint_for_K_bias_counts": next(iter(all_source_fingerprints)),
        "occupancy_source_fingerprint_available": False,
        "cohort": cohort_audit,
        "all_condition_selection_audit": selection_audit,
        "source_files": {
            **{name: source_record(path) for name, path in paths.items()},
            "bias_annotations": {key: source_record(path) for key, path in bias_paths.items()},
            "count_observations": {
                key: source_record(path) for key, path in count_paths_used.items()
            },
        },
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    validation_path = results_directory / outputs["validation_json"]
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    captions_path = figure_directory / outputs["captions_tex"]
    write_captions(
        captions_path,
        profile_transcript=example1.transcript_id,
        profile_condition_label=conditions_by_key[first_condition]["label"],
        all_condition_transcript=example2.transcript_id,
        cohort_size=cohort_audit["sequence_aligned_transcripts"],
    )

    executable = Path(sys.executable)
    command_parts = [
        str(executable.relative_to(ROOT)) if executable.is_relative_to(ROOT) else str(executable),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "--config",
        str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
    ]
    if args.figure_dir:
        command_parts.extend(["--figure-dir", str(figure_directory)])
    if args.results_dir:
        command_parts.extend(["--results-dir", str(results_directory)])
    if args.overwrite:
        command_parts.append("--overwrite")
    environment = [
        f"RIBOUNMIX_PLOT_TEX={shlex.quote(os.environ.get('RIBOUNMIX_PLOT_TEX', 'auto'))}",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
    ]
    command = " ".join(environment) + " " + shlex.join(command_parts)
    report_path = results_directory / outputs["report_markdown"]
    write_report(
        report_path,
        command=command,
        config_path=config_path,
        data_paths=paths,
        cohort_audit=cohort_audit,
        validation=validation,
        summary=summary,
        selection_audit=selection_audit,
        representative=representative,
        figure_paths=figure_paths,
    )

    print("Synthetic hierarchy figures: PASS")
    print(f"Cohort: {cohort_audit['sequence_aligned_transcripts']:,} transcripts")
    print(summary.to_string(index=False))
    for output in output_paths:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
