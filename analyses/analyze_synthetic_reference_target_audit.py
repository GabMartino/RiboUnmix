#!/usr/bin/env python3
"""Post-hoc estimand audit for the frozen synthetic RiboUnmix experiments.

The analysis separates recovery of the model-defined reference target H,
distortion of H from the saved two-trajectory TASEP occupancy consensus Q,
and end-to-end agreement of the learned shared profile with Q.  It never loads
a checkpoint, changes a model, or uses an oracle target for model selection.

All profile calculations are transcript-local and use the identical saved
model mask, terminal convention, and fixed boundary exclusion.  Codon-level
arrays are streamed; only scalar transcript metrics and a few reproducibly
selected example profiles are retained.
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
import re
import shlex
import sys
from typing import Any, Iterable, Iterator, Sequence

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utils.publication_plot_style import publication_rc  # noqa: E402


DEFAULT_CONFIG = ROOT / "analyses/configs/synthetic_reference_target.yaml"
BIAS_ORDER = (
    "artificial_bias_3prime_aa",
    "artificial_bias_3prime_cc",
    "artificial_bias_3prime_gg",
    "artificial_bias_3prime_uu",
    "artificial_bias_5prime_aa",
    "artificial_bias_5prime_cc",
    "artificial_bias_5prime_gg",
    "artificial_bias_5prime_uu",
    "artificial_bias_gc_fraction_gt_0p7",
    "artificial_bias_au_fraction_gt_0p7",
)
DEPTH_ORDER = ("0p25_per_codon", "2_per_codon", "20_per_codon")
DEPTH_LABELS = {
    "0p25_per_codon": "0.25 reads/codon",
    "2_per_codon": "2 reads/codon",
    "20_per_codon": "20 reads/codon",
    "cross_depth": "mixed depths",
}
DEPTH_COLORS = {
    "0p25_per_codon": "#0072B2",
    "2_per_codon": "#E69F00",
    "20_per_codon": "#009E73",
}
COMPARISONS = {
    "analysis1_L_vs_H": ("L", "H", r"A  $\widehat L_t$ vs. $H_t$"),
    "analysis2_H_vs_Q": ("H", "Q", r"B  $H_t$ vs. $\bar q_t$"),
    "analysis3_L_vs_Q": ("L", "Q", r"C  $\widehat L_t$ vs. $\bar q_t$"),
}
METRICS = ("pearson", "spearman", "mae", "clr_rmse", "aitchison_distance")
WITHIN_RE = re.compile(
    r"^riboai_synthetic_within_(0p25|2|20)_per_codon_panel(\d+)_gammaequal_seed(\d+)_"
)
CROSS_RE = re.compile(
    r"^riboai_synthetic_inter_artificial_bias_cumulative_biases(\d+)_datasets(\d+)_"
    r"gamma(equal|quality_rank)_seed(\d+)_"
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    family: str
    depth: str
    n_datasets: int
    n_bias_families: int
    reference_weighting: str
    training_seed: int
    run_dir: Path
    config_path: Path
    prediction_path: Path
    checkpoint_manifest_path: Path
    split_path: Path
    datasets: tuple[str, ...]
    dataset_id_to_name: dict[int, str]
    validation_ids: frozenset[str]
    train_ids: frozenset[str]
    split_id: str


def analysis_cohort_key(family: str, depth: str) -> str:
    """Return the scientifically matched transcript cohort for one run.

    Within-depth validation identities are fixed across cumulative panel sizes
    but differ between depths.  Matching all three depths would therefore
    discard almost the entire validation cohort for no inferential benefit:
    the depth curves are not treated as paired observations.  Mixed-depth runs
    share one split and remain one matched cohort.
    """
    if family == "within_depth":
        if depth not in DEPTH_ORDER:
            raise ValueError(f"Unknown within-depth cohort: {depth}")
        return f"within_depth::{depth}"
    if family == "cross_depth":
        return "cross_depth"
    raise ValueError(f"Unknown experiment family: {family}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-cross-depth", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def encoding_from_config(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("paths", {}).get("encodings", {}).get("datasets")
    path = None if raw is None else resolve_path(str(raw))
    if path is not None and path.is_file():
        mapping = load_yaml(path)
        return {int(dataset_id): str(name) for name, dataset_id in mapping.items()}
    universe = config.get("split", {}).get("master_dataset_universe")
    if isinstance(universe, list) and universe:
        names = [str(value) for value in universe]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate names in split.master_dataset_universe")
        return {index: name for index, name in enumerate(names)}
    fallback = ROOT / "Datasets/encodings/synthetic_dataset_encoding.yaml"
    mapping = load_yaml(fallback)
    return {int(dataset_id): str(name) for name, dataset_id in mapping.items()}


def exactly_one(paths: Sequence[Path], label: str) -> Path:
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {label}; found {len(paths)}")
    return paths[0]


def discover_runs(
    results_root: Path,
    *,
    seed: int,
    checkpoint_variant: str,
    include_cross: bool,
) -> list[RunSpec]:
    runs: list[RunSpec] = []
    prediction_pattern = f"predictions_main_val_{checkpoint_variant}_*.parquet"
    for run_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        within = WITHIN_RE.match(run_dir.name)
        cross = CROSS_RE.match(run_dir.name) if include_cross else None
        if within is None and cross is None:
            continue
        prediction_paths = sorted(run_dir.glob(f"results/**/{prediction_pattern}"))
        if not prediction_paths:
            continue
        config_path = exactly_one(sorted(run_dir.glob("logs/**/config.yaml")), "config")
        split_path = exactly_one(
            sorted(run_dir.glob("results/**/split_manifest*.json")), "split manifest"
        )
        checkpoint_manifest = exactly_one(
            sorted(run_dir.glob("results/**/prediction_checkpoint_manifest.json")),
            "checkpoint manifest",
        )
        prediction_path = exactly_one(prediction_paths, "prediction export")
        config = load_yaml(config_path)
        split = json.loads(split_path.read_text(encoding="utf-8"))
        datasets = tuple(str(value) for value in config["experiment"]["dataset"])
        run_seed = int(config["experiment"]["seed"])
        if run_seed != seed or int(split["seed"]) != seed:
            continue
        if within is not None:
            depth = f"{within.group(1)}_per_codon"
            n_datasets = int(within.group(2))
            n_bias = n_datasets
            family = "within_depth"
            weighting = "equal"
        else:
            assert cross is not None
            n_bias = int(cross.group(1))
            n_datasets = int(cross.group(2))
            depth = "cross_depth"
            family = "cross_depth"
            weighting = "equal" if cross.group(3) == "equal" else "quality_rank"
        if len(datasets) != n_datasets:
            raise ValueError(f"{run_dir.name}: panel size/name mismatch")
        checkpoint_data = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
        if checkpoint_variant not in checkpoint_data:
            raise KeyError(f"{checkpoint_manifest}: missing {checkpoint_variant}")
        if Path(checkpoint_data[checkpoint_variant]["output_path"]).name != prediction_path.name:
            raise ValueError(f"{run_dir.name}: prediction/checkpoint manifest mismatch")
        train_ids = frozenset(str(value) for value in split["train_ids"])
        validation_ids = frozenset(str(value) for value in split["validation_ids"])
        if train_ids & validation_ids:
            raise ValueError(f"{run_dir.name}: overlapping train/validation IDs")
        split_id = hashlib.sha256(
            (text_hash(train_ids) + text_hash(validation_ids)).encode("utf-8")
        ).hexdigest()[:16]
        runs.append(
            RunSpec(
                run_id=run_dir.name,
                family=family,
                depth=depth,
                n_datasets=n_datasets,
                n_bias_families=n_bias,
                reference_weighting=weighting,
                training_seed=run_seed,
                run_dir=run_dir,
                config_path=config_path,
                prediction_path=prediction_path,
                checkpoint_manifest_path=checkpoint_manifest,
                split_path=split_path,
                datasets=datasets,
                dataset_id_to_name=encoding_from_config(config),
                validation_ids=validation_ids,
                train_ids=train_ids,
                split_id=split_id,
            )
        )
    expected_within = {(depth, n) for depth in DEPTH_ORDER for n in range(2, 11)}
    observed_within = {
        (run.depth, run.n_datasets) for run in runs if run.family == "within_depth"
    }
    if observed_within != expected_within:
        raise ValueError(
            f"Incomplete within-depth grid: missing={sorted(expected_within-observed_within)}"
        )
    if include_cross:
        expected_cross = {
            (n, weighting)
            for n in range(1, 11)
            for weighting in ("equal", "quality_rank")
        }
        observed_cross = {
            (run.n_bias_families, run.reference_weighting)
            for run in runs
            if run.family == "cross_depth"
        }
        if observed_cross != expected_cross:
            raise ValueError(
                f"Incomplete cross-depth grid: missing={sorted(expected_cross-observed_cross)}"
            )
    return sorted(
        runs,
        key=lambda run: (
            run.family,
            DEPTH_ORDER.index(run.depth) if run.depth in DEPTH_ORDER else 99,
            run.n_bias_families,
            run.reference_weighting,
        ),
    )


def base_bias_name(dataset: str) -> str:
    for suffix in ("_0p25_per_codon", "_2_per_codon", "_20_per_codon"):
        if dataset.endswith(suffix):
            return dataset[: -len(suffix)]
    return dataset


def load_occupancy_consensus(
    path: Path, transcript_ids: set[str]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    profiles: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(
        columns=["sample", "transcript_id", "rib_profile"],
        batch_size=256,
        use_threads=False,
    ):
        for row in batch.to_pylist():
            transcript_id = str(row["transcript_id"])
            if transcript_id not in transcript_ids:
                continue
            sample = str(row["sample"])
            if sample.startswith("replicate_1"):
                key = "rep1"
            elif sample.startswith("replicate_2"):
                key = "rep2"
            else:
                raise ValueError(f"Unexpected occupancy sample {sample!r}")
            values = np.asarray(row["rib_profile"], dtype=np.float64)
            if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"Invalid occupancy profile for {transcript_id}/{sample}")
            profiles[transcript_id][key] = values / values.mean()
    missing = sorted(transcript_ids - set(profiles))
    if missing:
        raise KeyError(f"Missing occupancy profiles for {missing[:5]}")
    consensus: dict[str, np.ndarray] = {}
    for transcript_id, replicas in profiles.items():
        if set(replicas) != {"rep1", "rep2"}:
            raise ValueError(f"Incomplete occupancy replicas for {transcript_id}")
        if replicas["rep1"].shape != replicas["rep2"].shape:
            raise ValueError(f"Occupancy replica length mismatch for {transcript_id}")
        consensus[transcript_id] = 0.5 * (replicas["rep1"] + replicas["rep2"])
    metadata = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "replicate_ids": [
            "replicate_1_mean_psite_occupancy",
            "replicate_2_mean_psite_occupancy",
        ],
        "trajectory_rng_seeds": "not exported in parquet metadata",
        "definition": "normalize each raw finite-trajectory occupancy to full-sense mean one, then arithmetic mean",
    }
    return consensus, metadata


def parquet_metadata(path: Path) -> dict[str, str]:
    raw = pq.ParquetFile(path).schema_arrow.metadata or {}
    return {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in raw.items()
        if key not in {b"ARROW:schema", b"pandas"}
    }


def load_biases(
    bias_root: Path, transcript_ids: set[str]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    provenance: dict[str, Any] = {}
    for bias in BIAS_ORDER:
        path = bias_root / f"{bias}_compendium_added_bias_only.parquet"
        rows: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        reader = pq.ParquetFile(path)
        for batch in reader.iter_batches(
            columns=["sample", "transcript_id", "added_bias"],
            batch_size=256,
            use_threads=False,
        ):
            for row in batch.to_pylist():
                transcript_id = str(row["transcript_id"])
                if transcript_id not in transcript_ids:
                    continue
                sample = str(row["sample"])
                role = "mean" if sample.endswith("_mean") else (
                    "rep1" if sample.endswith("_rep1") else "rep2"
                )
                added = np.asarray(row["added_bias"], dtype=np.float64)
                if added.ndim != 1 or not np.isfinite(added).all() or np.any(added < 0):
                    raise ValueError(f"Invalid bias profile for {bias}/{transcript_id}")
                rows[transcript_id][role] = added
        missing = sorted(transcript_ids - set(rows))
        if missing:
            raise KeyError(f"{bias}: missing bias profiles for {missing[:5]}")
        result[bias] = {}
        for transcript_id, replicas in rows.items():
            if set(replicas) != {"rep1", "rep2", "mean"}:
                raise ValueError(f"{bias}/{transcript_id}: incomplete bias replicas")
            if not (
                np.array_equal(replicas["rep1"], replicas["mean"])
                and np.array_equal(replicas["rep2"], replicas["mean"])
            ):
                raise ValueError(f"{bias}/{transcript_id}: bias differs across replicas")
            result[bias][transcript_id] = 1.0 + replicas["mean"]
        metadata = parquet_metadata(path)
        provenance[bias] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "bias_seed": metadata.get("riboart.bias_seed"),
            "source_run_fingerprint": metadata.get("riboart.source_run_fingerprint"),
            "replicate_identity_verified": True,
        }
    return result, provenance


def mean_one(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
        raise ValueError("Mean-one input must be a finite one-dimensional profile")
    if np.any(values <= 0):
        raise ValueError("Log-shape analysis requires strictly positive profiles")
    mean = float(values.mean())
    if mean <= 0:
        raise ValueError("Profile mean must be positive")
    return values / mean


def clr(values: np.ndarray) -> np.ndarray:
    logs = np.log(np.asarray(values, dtype=np.float64))
    return logs - logs.mean()


def profile_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = mean_one(left)
    right = mean_one(right)
    if left.shape != right.shape:
        raise ValueError("Metric profile length mismatch")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    pearson = (
        float(np.dot(left_centered, right_centered) / denominator)
        if denominator > np.finfo(np.float64).eps * left.size
        else float("nan")
    )
    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    rank_left = left_rank - left_rank.mean()
    rank_right = right_rank - right_rank.mean()
    rank_denominator = float(np.linalg.norm(rank_left) * np.linalg.norm(rank_right))
    spearman = (
        float(np.dot(rank_left, rank_right) / rank_denominator)
        if rank_denominator > np.finfo(np.float64).eps * left.size
        else float("nan")
    )
    clr_delta = clr(left) - clr(right)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mae": float(np.mean(np.abs(left - right))),
        "clr_rmse": float(np.sqrt(np.mean(np.square(clr_delta)))),
        "aitchison_distance": float(np.linalg.norm(clr_delta)),
    }


def extract_reference_weights(run: RunSpec) -> tuple[dict[str, float], float]:
    found: dict[str, float] = {}
    panel = set(run.datasets)
    reader = pq.ParquetFile(run.prediction_path)
    for batch in reader.iter_batches(
        columns=["dataset_id", "gamma_centering_reliability"],
        batch_size=max(64, 2 * run.n_datasets),
        use_threads=False,
    ):
        for row in batch.to_pylist():
            dataset_id = int(row["dataset_id"])
            name = run.dataset_id_to_name.get(dataset_id)
            if name not in panel or name in found:
                continue
            values = np.asarray(row["gamma_centering_reliability"], dtype=np.float64)
            unique = np.unique(values)
            if unique.size != 1 or unique[0] <= 0:
                raise ValueError(f"{run.run_id}/{name}: nonconstant stored reference weight")
            found[name] = float(unique[0])
        if set(found) == panel:
            break
    if set(found) != panel:
        raise KeyError(f"{run.run_id}: failed to recover all stored reference weights")
    raw_sum = float(sum(found.values()))
    normalized = {name: value / raw_sum for name, value in found.items()}
    return normalized, raw_sum


def panel_hash(datasets: Sequence[str], pi: dict[str, float]) -> str:
    payload = {
        "datasets": list(datasets),
        "pi": {name: pi[name] for name in datasets},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def iter_prediction_rows(
    run: RunSpec,
    transcript_ids: set[str],
    *,
    batch_size: int = 64,
) -> Iterator[dict[str, Any]]:
    """Stream selected rows without assuming transcript-contiguous Parquet order."""
    columns = [
        "transcript_id",
        "dataset_id",
        "mask",
        "L_bio",
        "gamma",
        "log_gamma",
        "scale_dt",
    ]
    reader = pq.ParquetFile(run.prediction_path)
    for batch in reader.iter_batches(
        columns=columns, batch_size=batch_size, use_threads=False
    ):
        for row in batch.to_pylist():
            transcript_id = str(row["transcript_id"])
            if transcript_id in transcript_ids:
                yield row


def target_profiles(
    q_full: np.ndarray,
    bias_by_name: dict[str, np.ndarray],
    datasets: Sequence[str],
    pi: dict[str, float],
    take: np.ndarray,
) -> dict[str, np.ndarray | float]:
    q = mean_one(q_full[take])
    log_g = np.zeros(q.size, dtype=np.float64)
    log_h = np.zeros(q.size, dtype=np.float64)
    for dataset in datasets:
        bias = bias_by_name[base_bias_name(dataset)]
        if bias.shape != q_full.shape:
            raise ValueError(f"Bias/occupancy alignment mismatch for {dataset}")
        b = bias[take]
        weight = float(pi[dataset])
        log_g += weight * np.log(b)
        h_star = mean_one(q_full[take] * b)
        log_h += weight * np.log(h_star)
    g = np.exp(log_g)
    h_qg = mean_one(q * g)
    h_hstar = mean_one(np.exp(log_h))
    return {
        "Q": q,
        "G": g,
        "clr_G": clr(g),
        "H_qG": h_qg,
        "H_hstar": h_hstar,
        "H_construction_max_abs": float(np.max(np.abs(h_qg - h_hstar))),
        "sigma_log_G": float(np.std(log_g, ddof=0)),
    }


def analyze_run(
    run: RunSpec,
    cohort: set[str],
    occupancies: dict[str, np.ndarray],
    biases: dict[str, dict[str, np.ndarray]],
    pi: dict[str, float],
    trim: int,
    tolerances: dict[str, float],
    profile_cache: dict[tuple[str, str], dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    metric_rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    maxima = defaultdict(float)
    panel_id = panel_hash(run.datasets, pi)
    states: dict[str, dict[str, Any]] = {}
    for row in iter_prediction_rows(run, cohort):
        transcript_id = str(row["transcript_id"])
        dataset_id = int(row["dataset_id"])
        dataset = run.dataset_id_to_name.get(dataset_id)
        if dataset not in run.datasets:
            raise ValueError(f"{run.run_id}/{transcript_id}: invalid dataset row {dataset_id}")

        state = states.get(transcript_id)
        if state is None:
            q_full = occupancies[transcript_id]
            model_mask = np.asarray(row["mask"], dtype=bool)
            l_full = np.asarray(row["L_bio"], dtype=np.float64)
            if l_full.size == q_full.size + 1:
                l_sense = l_full[:-1]
                mask_sense = model_mask[:-1]
                terminal_removed = True
            elif l_full.size == q_full.size:
                l_sense = l_full
                mask_sense = model_mask
                terminal_removed = False
            else:
                raise ValueError(
                    f"{run.run_id}/{transcript_id}: model={l_full.size}, occupancy={q_full.size}"
                )
            positions = np.arange(q_full.size)
            take = mask_sense & (positions >= trim) & (positions < q_full.size - trim)
            if int(take.sum()) < 3:
                raise ValueError(f"{transcript_id}: fewer than three evaluation positions")
            mask_rows.append(
                {
                    "run_id": run.run_id,
                    "transcript_id": transcript_id,
                    "model_length": l_full.size,
                    "sense_length": q_full.size,
                    "terminal_entry_removed": terminal_removed,
                    "boundary_trim_codons": trim,
                    "evaluated_positions": int(take.sum()),
                    "first_evaluated_position": int(np.flatnonzero(take)[0]),
                    "last_evaluated_position": int(np.flatnonzero(take)[-1]),
                    "mask_sha256": hashlib.sha256(np.packbits(take).tobytes()).hexdigest(),
                }
            )
            target = target_profiles(
                q_full,
                {bias: biases[bias][transcript_id] for bias in BIAS_ORDER},
                run.datasets,
                pi,
                take,
            )
            q = np.asarray(target["Q"])
            h = np.asarray(target["H_hstar"])
            g = np.asarray(target["G"])
            l = mean_one(l_sense[take])
            maxima["mean_one"] = max(
                maxima["mean_one"],
                abs(float(q.mean()) - 1.0),
                abs(float(h.mean()) - 1.0),
                abs(float(l.mean()) - 1.0),
            )
            maxima["h_construction"] = max(
                maxima["h_construction"], float(target["H_construction_max_abs"])
            )
            model_error = clr(l) - clr(h)
            reference_gap = clr(g)
            total_error = clr(l) - clr(q)
            vector_residual = total_error - (model_error + reference_gap)
            square_cross = 2.0 * float(np.dot(model_error, reference_gap))
            squared_residual = (
                float(np.dot(total_error, total_error))
                - float(np.dot(model_error, model_error))
                - float(np.dot(reference_gap, reference_gap))
                - square_cross
            )
            maxima["clr_identity"] = max(
                maxima["clr_identity"], float(np.max(np.abs(vector_residual)))
            )
            maxima["clr_square_identity"] = max(
                maxima["clr_square_identity"], abs(squared_residual)
            )
            base_row: dict[str, Any] = {
                "run_id": run.run_id,
                "family": run.family,
                "depth": run.depth,
                "n_datasets": run.n_datasets,
                "n_bias_families": run.n_bias_families,
                "reference_weighting": run.reference_weighting,
                "training_seed": run.training_seed,
                "panel_id": panel_id,
                "transcript_id": transcript_id,
                "evaluated_positions": int(take.sum()),
                "sigma_log_G": float(target["sigma_log_G"]),
                "H_construction_max_abs": float(target["H_construction_max_abs"]),
                "model_error_l2": float(np.linalg.norm(model_error)),
                "reference_gap_l2": float(np.linalg.norm(reference_gap)),
                "total_error_l2": float(np.linalg.norm(total_error)),
                "model_error_squared": float(np.dot(model_error, model_error)),
                "reference_gap_squared": float(np.dot(reference_gap, reference_gap)),
                "total_error_squared": float(np.dot(total_error, total_error)),
                "model_reference_cross_term": square_cross,
                "clr_vector_identity_max_abs": float(np.max(np.abs(vector_residual))),
                "clr_squared_identity_abs_residual": abs(squared_residual),
            }
            values = {"L": l, "H": h, "Q": q}
            for comparison, (left_name, right_name, _) in COMPARISONS.items():
                for metric, value in profile_metrics(
                    values[left_name], values[right_name]
                ).items():
                    base_row[f"{comparison}__{metric}"] = value
            base_row["analysis2_clr_rmse_minus_sigma_log_G"] = (
                base_row["analysis2_H_vs_Q__clr_rmse"] - base_row["sigma_log_G"]
            )
            maxima["a2_clr_equals_sigma"] = max(
                maxima["a2_clr_equals_sigma"],
                abs(base_row["analysis2_clr_rmse_minus_sigma_log_G"]),
            )
            state = {
                "seen_datasets": set(),
                "l_full": l_full,
                "mask_full": model_mask,
                "l_eval_raw": l_sense[take],
                "take": take,
                "q_length": q_full.size,
                "weighted_log_gamma": np.zeros(int(take.sum()), dtype=np.float64),
                "decoder_m": [],
                "base_row": base_row,
            }
            states[transcript_id] = state
            if (
                run.family == "within_depth"
                and run.depth == "20_per_codon"
                and run.n_datasets == 10
                and run.reference_weighting == "equal"
            ):
                profile_cache[(run.run_id, transcript_id)] = {
                    "Q": q,
                    "H": h,
                    "L": l,
                    "clr_G": np.asarray(target["clr_G"]),
                    "positions": positions[take],
                }
        if dataset in state["seen_datasets"]:
            raise ValueError(f"{run.run_id}/{transcript_id}: duplicate {dataset} row")
        other_l = np.asarray(row["L_bio"], dtype=np.float64)
        other_mask = np.asarray(row["mask"], dtype=bool)
        if not np.allclose(other_l, state["l_full"], rtol=0.0, atol=1e-7):
            raise ValueError(f"{run.run_id}/{transcript_id}: L differs by dataset")
        if not np.array_equal(other_mask, state["mask_full"]):
            raise ValueError(f"{run.run_id}/{transcript_id}: mask differs by dataset")
        q_length = int(state["q_length"])
        take = np.asarray(state["take"], dtype=bool)
        gamma_eval = np.asarray(row["gamma"], dtype=np.float64)[:q_length][take]
        log_gamma_eval = np.asarray(row["log_gamma"], dtype=np.float64)[:q_length][take]
        if np.any(gamma_eval <= 0) or not np.isfinite(log_gamma_eval).all():
            raise ValueError(f"{run.run_id}/{transcript_id}/{dataset}: invalid gamma")
        state["weighted_log_gamma"] += float(pi[dataset]) * log_gamma_eval
        m_value = float(np.mean(np.asarray(state["l_eval_raw"]) * gamma_eval))
        state["decoder_m"].append(m_value)
        state["seen_datasets"].add(dataset)
        decoder_rows.append(
            {
                "run_id": run.run_id,
                "family": run.family,
                "depth": run.depth,
                "n_datasets": run.n_datasets,
                "n_bias_families": run.n_bias_families,
                "reference_weighting": run.reference_weighting,
                "training_seed": run.training_seed,
                "panel_id": panel_id,
                "transcript_id": transcript_id,
                "dataset": dataset,
                "pi": float(pi[dataset]),
                "scale_dt": float(row["scale_dt"]),
                "m_dt": m_value,
                "abs_m_minus_one": abs(m_value - 1.0),
                "arithmetic_scale_compatible": abs(m_value - 1.0)
                <= tolerances["decoder_mean_compatibility"],
            }
        )

    missing = sorted(cohort - set(states))
    if missing:
        raise KeyError(f"{run.run_id}: prediction export misses {missing[:5]}")
    for transcript_id in sorted(states):
        state = states[transcript_id]
        if state["seen_datasets"] != set(run.datasets):
            missing_datasets = sorted(set(run.datasets) - state["seen_datasets"])
            raise ValueError(
                f"{run.run_id}/{transcript_id}: incomplete panel rows; missing {missing_datasets}"
            )
        weighted_log_gamma = np.asarray(state["weighted_log_gamma"])
        gamma_constraint = float(np.max(np.abs(weighted_log_gamma)))
        maxima["gamma_centering"] = max(maxima["gamma_centering"], gamma_constraint)
        decoder_m = np.asarray(state["decoder_m"], dtype=np.float64)
        base_row = state["base_row"]
        base_row.update(
            {
                "gamma_weighted_log_center_max_abs": gamma_constraint,
                "decoder_m_mean": float(np.mean(decoder_m)),
                "decoder_m_min": float(np.min(decoder_m)),
                "decoder_m_max": float(np.max(decoder_m)),
                "decoder_m_max_abs_deviation": float(np.max(np.abs(decoder_m - 1.0))),
                "decoder_incompatible_fraction": float(
                    np.mean(
                        np.abs(decoder_m - 1.0)
                        > tolerances["decoder_mean_compatibility"]
                    )
                ),
            }
        )
        metric_rows.append(base_row)
    return metric_rows, decoder_rows, mask_rows, dict(maxima)


def bootstrap_draws(n: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(replicates, n), dtype=np.int32)


def bootstrap_interval(values: np.ndarray, draws: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size != draws.shape[1] or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be complete and aligned")
    estimates = np.median(values[draws], axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def aggregate_metrics(
    metrics: pd.DataFrame,
    cohort_ids: dict[str, list[str]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    draws_by_cohort = {
        cohort_key: bootstrap_draws(
            len(ids), bootstrap_replicates, bootstrap_seed + index
        )
        for index, (cohort_key, ids) in enumerate(sorted(cohort_ids.items()))
    }
    rows: list[dict[str, Any]] = []
    condition_columns = [
        "run_id",
        "family",
        "depth",
        "n_datasets",
        "n_bias_families",
        "reference_weighting",
        "training_seed",
        "panel_id",
    ]
    for keys, group in metrics.groupby(condition_columns, sort=False):
        condition = dict(zip(condition_columns, keys, strict=True))
        cohort_key = analysis_cohort_key(
            str(condition["family"]), str(condition["depth"])
        )
        ordered_ids = cohort_ids[cohort_key]
        indexed = group.set_index("transcript_id")
        if set(indexed.index) != set(ordered_ids):
            raise ValueError(f"{condition['run_id']}: incomplete aggregation cohort")
        indexed = indexed.loc[ordered_ids]
        draws = draws_by_cohort[cohort_key]
        for comparison in COMPARISONS:
            for metric in METRICS:
                column = f"{comparison}__{metric}"
                values = indexed[column].to_numpy(dtype=np.float64)
                finite = np.isfinite(values)
                if not finite.all():
                    raise ValueError(f"{condition['run_id']}/{column}: undefined metric")
                low, high = bootstrap_interval(values, draws)
                rows.append(
                    {
                        **condition,
                        "comparison": comparison,
                        "metric": metric,
                        "n_transcripts": int(values.size),
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                        "mean": float(np.mean(values)),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "bootstrap_replicates": bootstrap_replicates,
                        "bootstrap_cluster": "transcript",
                    }
                )
    return pd.DataFrame(rows), draws_by_cohort


def aggregate_reference_only(
    metrics: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Summarize H-versus-Q on the union of depth-specific validation IDs.

    H and Q do not depend on read depth.  The same transcript/N value is
    therefore duplicated when an ID occurs in more than one depth split.  We
    verify that identity, deduplicate it, and use the union cohort.  This avoids
    drawing three visually different H--Q curves whose only difference is the
    composition of the depth-specific validation splits.
    """
    source = metrics.loc[metrics["family"] == "within_depth", [
        "n_datasets",
        "transcript_id",
        "analysis2_H_vs_Q__pearson",
    ]].copy()
    duplicate_spread = source.groupby(
        ["n_datasets", "transcript_id"], sort=False
    )["analysis2_H_vs_Q__pearson"].agg(lambda values: float(values.max() - values.min()))
    maximum_spread = float(duplicate_spread.max())
    if maximum_spread > 2.0e-12:
        raise ValueError(
            "H-versus-Q differs across depth copies for an identical transcript/N: "
            f"maximum absolute spread={maximum_spread:.3e}"
        )
    deduplicated = source.drop_duplicates(["n_datasets", "transcript_id"])
    ids_by_n = {
        int(n): set(group["transcript_id"].astype(str))
        for n, group in deduplicated.groupby("n_datasets", sort=True)
    }
    common_ids = sorted(set.intersection(*ids_by_n.values()))
    if not common_ids:
        raise ValueError("The reference-only union cohort is empty")
    draws = bootstrap_draws(
        len(common_ids), bootstrap_replicates, bootstrap_seed
    )
    rows: list[dict[str, Any]] = []
    for n in sorted(ids_by_n):
        indexed = deduplicated.loc[deduplicated["n_datasets"] == n].set_index(
            "transcript_id"
        )
        values = indexed.loc[common_ids, "analysis2_H_vs_Q__pearson"].to_numpy(float)
        low, high = bootstrap_interval(values, draws)
        rows.append(
            {
                "n_datasets": n,
                "n_transcripts": len(values),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "mean": float(np.mean(values)),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_cluster": "transcript",
                "cohort": "union_of_depth_specific_validation_ids",
                "maximum_duplicate_spread": maximum_spread,
            }
        )
    return pd.DataFrame(rows)


def paired_contrasts(
    metrics: pd.DataFrame,
    cohort_ids: dict[str, list[str]],
    draws_by_cohort: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    within = metrics.loc[metrics["family"] == "within_depth"]
    for depth in DEPTH_ORDER:
        low = within.loc[(within["depth"] == depth) & (within["n_datasets"] == 2)]
        high = within.loc[(within["depth"] == depth) & (within["n_datasets"] == 10)]
        cohort_key = analysis_cohort_key("within_depth", depth)
        ids = cohort_ids[cohort_key]
        low = low.set_index("transcript_id").loc[ids]
        high = high.set_index("transcript_id").loc[ids]
        for comparison in COMPARISONS:
            for metric in METRICS:
                column = f"{comparison}__{metric}"
                difference = high[column].to_numpy(float) - low[column].to_numpy(float)
                ci = bootstrap_interval(difference, draws_by_cohort[cohort_key])
                rows.append(
                    {
                        "contrast": "N10_minus_N2",
                        "family": "within_depth",
                        "depth": depth,
                        "reference_weighting": "equal",
                        "comparison": comparison,
                        "metric": metric,
                        "n_transcripts": len(ids),
                        "median_paired_difference": float(np.median(difference)),
                        "bootstrap_ci_low": ci[0],
                        "bootstrap_ci_high": ci[1],
                    }
                )
    cross = metrics.loc[metrics["family"] == "cross_depth"]
    if not cross.empty:
        ids = cohort_ids["cross_depth"]
        for n_bias in range(1, 11):
            equal = cross.loc[
                (cross["n_bias_families"] == n_bias)
                & (cross["reference_weighting"] == "equal")
            ].set_index("transcript_id").loc[ids]
            quality = cross.loc[
                (cross["n_bias_families"] == n_bias)
                & (cross["reference_weighting"] == "quality_rank")
            ].set_index("transcript_id").loc[ids]
            for comparison in COMPARISONS:
                for metric in METRICS:
                    column = f"{comparison}__{metric}"
                    difference = (
                        quality[column].to_numpy(float) - equal[column].to_numpy(float)
                    )
                    ci = bootstrap_interval(difference, draws_by_cohort["cross_depth"])
                    rows.append(
                        {
                            "contrast": "quality_rank_minus_equal",
                            "family": "cross_depth",
                            "depth": "cross_depth",
                            "reference_weighting": "paired",
                            "n_bias_families": n_bias,
                            "comparison": comparison,
                            "metric": metric,
                            "n_transcripts": len(ids),
                            "median_paired_difference": float(np.median(difference)),
                            "bootstrap_ci_low": ci[0],
                            "bootstrap_ci_high": ci[1],
                        }
                    )
    return pd.DataFrame(rows)


def reference_error_associations(metrics: pd.DataFrame) -> pd.DataFrame:
    source = metrics.loc[
        (metrics["family"] == "within_depth")
        & (metrics["depth"] == "20_per_codon")
    ].copy()
    source["one_minus_analysis2_pcc"] = 1.0 - source[
        "analysis2_H_vs_Q__pearson"
    ]
    rows: list[dict[str, Any]] = []
    for n, group in source.groupby("n_datasets", sort=True):
        rho, pvalue = spearmanr(group["sigma_log_G"], group["one_minus_analysis2_pcc"])
        rows.append(
            {
                "scope": f"N={int(n)}",
                "n_datasets": int(n),
                "n_transcripts": len(group),
                "x": "sd_i(log G_i)",
                "y": "1-PCC(H,Q)",
                "spearman_rho": float(rho),
                "asymptotic_pvalue_descriptive_only": float(pvalue),
                "note": "positions are not treated as observations; p-value is descriptive and not multiplicity-adjusted",
            }
        )
    rho, pvalue = spearmanr(source["sigma_log_G"], source["one_minus_analysis2_pcc"])
    rows.append(
        {
            "scope": "pooled N (descriptive)",
            "n_datasets": np.nan,
            "n_transcripts": len(source),
            "x": "sd_i(log G_i)",
            "y": "1-PCC(H,Q)",
            "spearman_rho": float(rho),
            "asymptotic_pvalue_descriptive_only": float(pvalue),
            "note": "repeated transcripts across N make the pooled p-value invalid for inference",
        }
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
            "xtick.labelsize": 14.5,
            "ytick.labelsize": 14.5,
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
    # ``dpi`` applies only to artists explicitly rasterized inside the PDF;
    # axes and typography remain vector while dense point clouds stay light.
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_within_curve(
    axis: plt.Axes,
    summary: pd.DataFrame,
    comparison: str,
) -> None:
    part = summary.loc[
        (summary["family"] == "within_depth")
        & (summary["metric"] == "pearson")
        & (summary["comparison"] == comparison)
    ]
    line_styles = {
        "0p25_per_codon": ("--", "o", 3),
        "2_per_codon": (":", "s", 2),
        "20_per_codon": ("-", "^", 1),
    }
    for depth in DEPTH_ORDER:
        curve = part.loc[part["depth"] == depth].sort_values("n_datasets")
        y = curve["median"].to_numpy(float)
        low = curve["bootstrap_ci_low"].to_numpy(float)
        high = curve["bootstrap_ci_high"].to_numpy(float)
        line_style, marker, zorder = line_styles[depth]
        axis.errorbar(
            curve["n_datasets"],
            y,
            yerr=np.vstack([y - low, high - y]),
            color=DEPTH_COLORS[depth],
            marker=marker,
            markersize=6.2,
            linestyle=line_style,
            linewidth=2.2,
            capsize=3.0,
            label=DEPTH_LABELS[depth],
            zorder=zorder,
        )


def _plot_reference_curve(axis: plt.Axes, reference_summary: pd.DataFrame) -> None:
    curve = reference_summary.sort_values("n_datasets")
    y = curve["median"].to_numpy(float)
    low = curve["bootstrap_ci_low"].to_numpy(float)
    high = curve["bootstrap_ci_high"].to_numpy(float)
    axis.errorbar(
        curve["n_datasets"],
        y,
        yerr=np.vstack([y - low, high - y]),
        color="#333333",
        marker="D",
        markersize=5.8,
        linewidth=2.3,
        capsize=3.0,
        label="Depth-independent target",
        zorder=4,
    )


def plot_main(
    summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    source = summary.loc[
        (summary["family"] == "within_depth") & (summary["metric"] == "pearson")
    ]
    y_low = max(-1.0, float(source["q25"].min()) - 0.03)
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.3), sharex=True, sharey=True)
        for axis, (comparison, (_, _, title)) in zip(
            axes, COMPARISONS.items(), strict=True
        ):
            if comparison == "analysis2_H_vs_Q":
                _plot_reference_curve(axis, reference_summary)
            else:
                _plot_within_curve(axis, summary, comparison)
            axis.set_title(title, loc="left")
            axis.set_xlabel("Number of datasets $N$")
            axis.set_xticks(range(2, 11))
            axis.set_ylim(y_low, 1.005)
            axis.grid(alpha=0.45)
        axes[1].text(
            0.04,
            0.06,
            "Independent of read depth by construction",
            transform=axes[1].transAxes,
            ha="left",
            va="bottom",
            fontsize=13.5,
        )
        axes[0].set_ylabel("Median transcript-level PCC")
        axes[0].legend(loc="lower right")
        fig.subplots_adjust(left=0.07, right=0.995, bottom=0.16, top=0.91, wspace=0.08)
        save_figure(fig, output_dir / "reference_target_audit_main", dpi)


def plot_manuscript_overview(
    summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Four-panel main-text view of estimand and mixed-depth recovery."""
    source = summary.loc[summary["metric"] == "pearson"]
    y_low = max(0.0, float(source["q25"].min()) - 0.035)
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), sharey=True)
        for axis, comparison, title in (
            (axes[0, 0], "analysis1_L_vs_H", r"A  Estimand recovery: $\widehat L$ vs. $H$"),
            (axes[0, 1], "analysis2_H_vs_Q", r"B  Estimand fidelity: $H$ vs. $Q$"),
            (axes[1, 0], "analysis3_L_vs_Q", r"C  Occupancy recovery: $\widehat L$ vs. $Q$"),
        ):
            if comparison == "analysis2_H_vs_Q":
                _plot_reference_curve(axis, reference_summary)
                axis.text(
                    0.04,
                    0.07,
                    "Simulator-only; independent of read depth",
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=13.0,
                )
            else:
                _plot_within_curve(axis, summary, comparison)
            axis.set_title(title, loc="left")
            axis.set_xlabel("Datasets at one depth, $N$")
            axis.set_xticks(range(2, 11))
            axis.grid(alpha=0.30)
        mixed = source.loc[
            (source["family"] == "cross_depth")
            & (source["comparison"] == "analysis3_L_vs_Q")
        ]
        styles = {
            "equal": ("#4C78A8", "o", "-", "Equal reference"),
            "quality_rank": (
                "#D55E00",
                "s",
                "--",
                "Depth-ranked reference",
            ),
        }
        axis = axes[1, 1]
        for weighting, (color, marker, line_style, label) in styles.items():
            curve = mixed.loc[mixed["reference_weighting"] == weighting].sort_values(
                "n_datasets"
            )
            y = curve["median"].to_numpy(float)
            low = curve["bootstrap_ci_low"].to_numpy(float)
            high = curve["bootstrap_ci_high"].to_numpy(float)
            axis.errorbar(
                curve["n_datasets"],
                y,
                yerr=np.vstack([y - low, high - y]),
                color=color,
                marker=marker,
                markersize=6.2,
                markerfacecolor="none" if weighting == "quality_rank" else color,
                markeredgewidth=1.3,
                linestyle=line_style,
                linewidth=2.2,
                capsize=3.0,
                label=label,
            )
        axis.set_title(r"D  Mixed-depth occupancy recovery: $\widehat L$ vs. $Q$", loc="left")
        axis.set_xlabel("Datasets across three depths, $N$")
        axis.set_xticks(range(3, 31, 3))
        axis.grid(alpha=0.30)
        axis.legend(loc="lower right")
        for axis in axes.flat:
            axis.set_ylim(y_low, 1.005)
        for axis in axes[:, 0]:
            axis.set_ylabel("Median transcript-level PCC")
        axes[0, 0].legend(loc="lower right")
        fig.subplots_adjust(
            left=0.085,
            right=0.995,
            bottom=0.085,
            top=0.965,
            hspace=0.30,
            wspace=0.12,
        )
        save_figure(fig, output_dir / "synthetic_shared_profile_recovery_overview", dpi)


def plot_distributions(metrics: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    source = metrics.loc[metrics["family"] == "within_depth"]
    colors = ["#4C78A8", "#A0A0A0", "#E07B39"]
    offsets = [-0.24, 0.0, 0.24]
    all_values = [
        source[f"{comparison}__pearson"].to_numpy(float) for comparison in COMPARISONS
    ]
    y_low = max(-1.0, min(float(np.min(value)) for value in all_values) - 0.03)
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(3, 1, figsize=(12.8, 9.0), sharex=True, sharey=True)
        for axis, depth in zip(axes, DEPTH_ORDER, strict=True):
            part = source.loc[source["depth"] == depth]
            for color, offset, comparison in zip(colors, offsets, COMPARISONS, strict=True):
                values = [
                    part.loc[part["n_datasets"] == n, f"{comparison}__pearson"].to_numpy(float)
                    for n in range(2, 11)
                ]
                boxes = axis.boxplot(
                    values,
                    positions=np.arange(2, 11) + offset,
                    widths=0.20,
                    patch_artist=True,
                    showfliers=False,
                    whis=(5, 95),
                    manage_ticks=False,
                )
                for box in boxes["boxes"]:
                    box.set(facecolor=color, edgecolor=color, alpha=0.70)
                for key in ("whiskers", "caps", "medians"):
                    for artist in boxes[key]:
                        artist.set(color="#222222", linewidth=0.8)
            axis.set_title(DEPTH_LABELS[depth], loc="left")
            axis.set_ylabel("Transcript PCC")
            axis.set_ylim(y_low, 1.005)
            axis.grid(axis="y", alpha=0.45)
        axes[-1].set_xlabel("Number of datasets $N$")
        axes[-1].set_xticks(range(2, 11))
        handles = [
            Line2D([], [], color=color, linewidth=7, label=COMPARISONS[name][2])
            for color, name in zip(colors, COMPARISONS, strict=True)
        ]
        fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995))
        fig.subplots_adjust(left=0.08, right=0.995, bottom=0.08, top=0.93, hspace=0.22)
        save_figure(fig, output_dir / "reference_target_per_transcript_distributions", dpi)


def plot_reference_scatter(
    metrics: pd.DataFrame,
    associations: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    source = metrics.loc[
        (metrics["family"] == "within_depth")
        & (metrics["depth"] == "20_per_codon")
    ].copy()
    source["reference_error"] = 1.0 - source["analysis2_H_vs_Q__pearson"]
    with plt.rc_context(publication_style()):
        fig, axis = plt.subplots(figsize=(7.2, 5.7), constrained_layout=True)
        scatter = axis.scatter(
            source["sigma_log_G"],
            source["reference_error"],
            c=source["n_datasets"],
            cmap="viridis",
            s=24,
            alpha=0.72,
            edgecolors="none",
            rasterized=True,
        )
        overall = associations.loc[associations["scope"] == "pooled N (descriptive)"].iloc[0]
        axis.text(
            0.03,
            0.97,
            rf"Pooled descriptive $\rho={overall['spearman_rho']:.3f}$",
            transform=axis.transAxes,
            ha="left",
            va="top",
        )
        axis.set_xlabel(r"Positional variation $\mathrm{sd}_i[\log G_{ti}^{(N)}]$")
        axis.set_ylabel(r"Reference-target error $1-\mathrm{PCC}(H_t,\bar q_t)$")
        axis.set_title("Panel-average bias variation and reference-target error")
        axis.grid(alpha=0.40)
        colorbar = fig.colorbar(scatter, ax=axis, pad=0.02)
        colorbar.set_label("Number of datasets $N$", fontweight="bold")
        save_figure(fig, output_dir / "reference_error_vs_logG_variation", dpi)


def select_representatives(
    metrics: pd.DataFrame,
    profile_cache: dict[tuple[str, str], dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = metrics.loc[
        (metrics["family"] == "within_depth")
        & (metrics["depth"] == "20_per_codon")
        & (metrics["n_datasets"] == 10)
        & (metrics["reference_weighting"] == "equal")
    ].sort_values("transcript_id")
    if source["run_id"].nunique() != 1:
        raise ValueError("Representative rule expected one N=10, 20-depth run")
    value_column = "analysis2_H_vs_Q__clr_rmse"
    chosen: list[pd.Series] = []
    for probability in (0.10, 0.50, 0.90):
        target = float(source[value_column].quantile(probability))
        ranked = source.assign(distance=(source[value_column] - target).abs()).sort_values(
            ["distance", "transcript_id"], kind="mergesort"
        )
        ranked = ranked.loc[~ranked["transcript_id"].isin([row["transcript_id"] for row in chosen])]
        row = ranked.iloc[0].copy()
        row["selection_quantile"] = probability
        row["selection_target"] = target
        chosen.append(row)
    chosen_frame = pd.DataFrame(chosen)
    profile_rows: list[dict[str, Any]] = []
    for _, row in chosen_frame.iterrows():
        profile = profile_cache[(str(row["run_id"]), str(row["transcript_id"]))]
        for index, position in enumerate(profile["positions"]):
            profile_rows.append(
                {
                    "selection_quantile": row["selection_quantile"],
                    "transcript_id": row["transcript_id"],
                    "codon_position_0based": int(position),
                    "Q": float(profile["Q"][index]),
                    "H": float(profile["H"][index]),
                    "L_hat": float(profile["L"][index]),
                    "clr_G": float(profile["clr_G"][index]),
                }
            )
    return chosen_frame, pd.DataFrame(profile_rows)


def plot_representatives(profiles: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    selections = sorted(profiles["selection_quantile"].unique())
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(
            len(selections), 2, figsize=(13.5, 8.4),
            gridspec_kw={"width_ratios": [3.2, 1.35]},
        )
        for row_index, quantile in enumerate(selections):
            part = profiles.loc[profiles["selection_quantile"] == quantile]
            transcript_id = str(part["transcript_id"].iloc[0])
            position = part["codon_position_0based"]
            left, right = axes[row_index]
            left.plot(position, part["Q"], color="#222222", linewidth=1.5, label=r"$\bar q_t$")
            left.plot(position, part["H"], color="#E69F00", linewidth=1.3, label="$H$")
            left.plot(position, part["L_hat"], color="#0072B2", linewidth=1.2, label=r"$\widehat L$")
            left.set_ylabel("Mean-one profile")
            left.set_title(
                f"{transcript_id}; reference-gap quantile {quantile:.0%}".replace("%", r"\%"),
                loc="left",
            )
            left.grid(alpha=0.35)
            right.plot(position, part["clr_G"], color="#A23B3B", linewidth=1.2)
            right.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
            right.set_ylabel(r"$\mathrm{clr}(G)$")
            right.grid(alpha=0.35)
        axes[0, 0].legend(ncol=3, loc="upper right")
        axes[-1, 0].set_xlabel("P-site codon position (0-based)")
        axes[-1, 1].set_xlabel("P-site codon position (0-based)")
        fig.subplots_adjust(left=0.07, right=0.995, bottom=0.07, top=0.97, hspace=0.35, wspace=0.16)
        save_figure(fig, output_dir / "reference_target_representative_profiles", dpi)


def plot_reference_weight_sensitivity(
    summary: pd.DataFrame, output_dir: Path, dpi: int
) -> None:
    source = summary.loc[
        (summary["family"] == "cross_depth") & (summary["metric"] == "pearson")
    ]
    if source.empty:
        return
    y_low = max(-1.0, float(source["q25"].min()) - 0.03)
    styles = {
        "equal": ("#4C78A8", "o", "-", "Equal reference"),
        "quality_rank": ("#E07B39", "s", "--", "Depth-ranked reference"),
    }
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharex=True, sharey=True)
        for axis, (comparison, (_, _, title)) in zip(axes, COMPARISONS.items(), strict=True):
            part = source.loc[source["comparison"] == comparison]
            for weighting, (color, marker, line_style, label) in styles.items():
                curve = part.loc[part["reference_weighting"] == weighting].sort_values(
                    "n_bias_families"
                )
                y = curve["median"].to_numpy(float)
                low = curve["bootstrap_ci_low"].to_numpy(float)
                high = curve["bootstrap_ci_high"].to_numpy(float)
                axis.errorbar(
                    curve["n_bias_families"], y,
                    yerr=np.vstack([y - low, high - y]),
                    color=color, marker=marker, markersize=4.5,
                    markerfacecolor="none" if weighting == "quality_rank" else color,
                    markeredgewidth=1.2, linestyle=line_style,
                    linewidth=1.8, capsize=2.3, label=label,
                )
            axis.set_title(title, loc="left")
            axis.set_xlabel("Bias families in panel")
            axis.set_xticks(range(1, 11))
            axis.set_ylim(y_low, 1.005)
            axis.grid(alpha=0.45)
        axes[1].text(
            0.04, 0.06, r"Targets coincide: $G_{\rm equal}=G_{\rm ranked}$",
            transform=axes[1].transAxes, ha="left", va="bottom", fontsize=13.0,
        )
        axes[0].set_ylabel("Median transcript-level PCC")
        axes[0].legend(loc="lower right")
        fig.subplots_adjust(left=0.07, right=0.995, bottom=0.16, top=0.91, wspace=0.08)
        save_figure(fig, output_dir / "cross_depth_reference_weight_sensitivity", dpi)


def build_sanity_table(
    maxima_by_run: dict[str, dict[str, float]],
    panel_table: pd.DataFrame,
    metrics: pd.DataFrame,
    decoder: pd.DataFrame,
    tolerances: dict[str, float],
    oracle_selection_ok: bool,
) -> pd.DataFrame:
    global_max = defaultdict(float)
    for values in maxima_by_run.values():
        for key, value in values.items():
            global_max[key] = max(global_max[key], float(value))
    sample = metrics.iloc[0]
    # Algebraic unit checks use a real positive occupancy vector length but do
    # not alter or score any fitted output.
    q = np.exp(np.linspace(-0.7, 0.9, 101))
    q = mean_one(q)
    constant_g = np.full(q.size, 2.7)
    constant_deviation = float(np.max(np.abs(mean_one(q * constant_g) - q)))
    pi = np.array([0.2, 0.3, 0.5])
    raw_log_bias = np.vstack(
        [np.sin(np.linspace(0, 3, q.size)), np.cos(np.linspace(0, 2, q.size)), np.linspace(-1, 1, q.size)]
    )
    centered = raw_log_bias - np.sum(pi[:, None] * raw_log_bias, axis=0)[None, :]
    centered_g = np.exp(np.sum(pi[:, None] * centered, axis=0))
    centered_deviation = float(np.max(np.abs(mean_one(q * centered_g) - q)))
    # If L=H, Analysis 1 is exact and Analysis 3 equals Analysis 2.
    artificial_h = mean_one(q * np.exp(np.sin(np.linspace(0, 2, q.size)) * 0.2))
    a1 = profile_metrics(artificial_h, artificial_h)
    a2 = profile_metrics(artificial_h, q)
    a3 = profile_metrics(artificial_h, q)
    artificial_deviation = max(
        abs(a1["pearson"] - 1.0),
        abs(a1["mae"]),
        *(abs(a2[key] - a3[key]) for key in METRICS),
    )
    pi_deviation = float(
        panel_table.groupby("run_id")["pi"].sum().sub(1.0).abs().max()
    )
    decoder_max = float(decoder["abs_m_minus_one"].max())
    decoder_prevalence = float((~decoder["arithmetic_scale_compatible"]).mean())
    definitions = [
        ("reference_weights_sum_to_one", pi_deviation, tolerances["reference_weight_sum"], "required"),
        ("Q_H_L_mean_one_on_evaluation_mask", global_max["mean_one"], tolerances["mean_one"], "required"),
        ("H_qG_equals_geometric_mean_hstar", global_max["h_construction"], tolerances["h_construction_max_abs"], "required"),
        ("constant_G_implies_H_equals_Q", constant_deviation, tolerances["constant_g_h_equals_q"], "constructed algebra check"),
        ("weighted_log_centered_bias_implies_H_equals_Q", centered_deviation, tolerances["centered_bias_h_equals_q"], "constructed algebra check"),
        ("artificial_L_equals_H_check", artificial_deviation, tolerances["artificial_l_equals_h"], "constructed algebra check"),
        ("clr_vector_identity", global_max["clr_identity"], tolerances["clr_identity_max_abs"], "required"),
        ("analysis2_clr_rmse_equals_sigma_logG", global_max["a2_clr_equals_sigma"], tolerances["clr_identity_max_abs"], "exact identity; association would be tautological"),
        ("learned_gamma_weighted_log_center", global_max["gamma_centering"], tolerances["gamma_weighted_log_center"], "required"),
        (
            "no_oracle_checkpoint_selection",
            0.0 if oracle_selection_ok else 1.0,
            0.0,
            "checkpoint selected by the configured validation metric; oracle metrics were diagnostic only",
        ),
        ("decoder_arithmetic_scale_compatibility", decoder_max, tolerances["decoder_mean_compatibility"], f"diagnostic flag; incompatible fraction={decoder_prevalence:.6f}"),
    ]
    rows = []
    for name, deviation, tolerance, note in definitions:
        if name == "decoder_arithmetic_scale_compatibility":
            status = "FLAG" if deviation > tolerance else "PASS"
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
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    associations: pd.DataFrame,
    sanity: pd.DataFrame,
    metrics: pd.DataFrame,
    decoder: pd.DataFrame,
    cohorts: dict[str, list[str]],
    cohort_exclusions: pd.DataFrame,
    *,
    checkpoint_variant: str,
    checkpoint_selection_metric: str,
) -> None:
    primary = summary.loc[
        (summary["family"] == "within_depth")
        & (summary["metric"] == "pearson")
        & (summary["n_datasets"].isin([2, 10]))
    ]
    lines: list[str] = []
    for depth in DEPTH_ORDER:
        for n in (2, 10):
            part = primary.loc[(primary["depth"] == depth) & (primary["n_datasets"] == n)]
            values = {
                comparison: float(part.loc[part["comparison"] == comparison, "median"].iloc[0])
                for comparison in COMPARISONS
            }
            lines.append(
                f"| {DEPTH_LABELS[depth]} | {n} | {values['analysis1_L_vs_H']:.4f} | "
                f"{values['analysis2_H_vs_Q']:.4f} | {values['analysis3_L_vs_Q']:.4f} |"
            )
    decoder_prevalence = float((~decoder["arithmetic_scale_compatible"]).mean())
    decoder_median = float(decoder["m_dt"].median())
    decoder_q = decoder["m_dt"].quantile([0.05, 0.95])
    pooled = associations.loc[associations["scope"] == "pooled N (descriptive)"].iloc[0]
    cross = summary.loc[
        (summary["family"] == "cross_depth")
        & (summary["metric"] == "pearson")
        & (summary["n_datasets"].isin([3, 30]))
    ]
    cross_lines: list[str] = []
    for n in (3, 30):
        for weighting in ("equal", "quality_rank"):
            part = cross.loc[
                (cross["n_datasets"] == n)
                & (cross["reference_weighting"] == weighting)
            ]
            if part.empty:
                continue
            values = {
                comparison: float(
                    part.loc[part["comparison"] == comparison, "median"].iloc[0]
                )
                for comparison in COMPARISONS
            }
            cross_lines.append(
                f"| {n} | {weighting} | {values['analysis1_L_vs_H']:.4f} | "
                f"{values['analysis2_H_vs_Q']:.4f} | {values['analysis3_L_vs_Q']:.4f} |"
            )
    cross_contrasts = contrasts.loc[
        (contrasts["contrast"] == "quality_rank_minus_equal")
        & (contrasts["metric"] == "pearson")
    ]
    finite_training_contrasts = cross_contrasts.loc[
        cross_contrasts["comparison"].isin(
            ["analysis1_L_vs_H", "analysis3_L_vs_Q"]
        ),
        "median_paired_difference",
    ].to_numpy(dtype=float)
    maximum_rank_effect = (
        float(np.max(np.abs(finite_training_contrasts)))
        if finite_training_contrasts.size
        else float("nan")
    )
    contrast_text: list[str] = []
    for n_bias in (1, 10):
        part = cross_contrasts.loc[cross_contrasts["n_bias_families"] == n_bias]
        if part.empty:
            continue
        values = {
            comparison: part.loc[part["comparison"] == comparison].iloc[0]
            for comparison in COMPARISONS
        }
        n = 3 * n_bias
        contrast_text.append(
            f"At $N={n}$, the ranked-minus-equal paired median PCC differences were "
            f"{values['analysis1_L_vs_H']['median_paired_difference']:.6f} for Analysis 1 "
            f"([{values['analysis1_L_vs_H']['bootstrap_ci_low']:.6f}, "
            f"{values['analysis1_L_vs_H']['bootstrap_ci_high']:.6f}]), "
            f"{values['analysis2_H_vs_Q']['median_paired_difference']:.6f} for Analysis 2, "
            f"and {values['analysis3_L_vs_Q']['median_paired_difference']:.6f} for Analysis 3 "
            f"([{values['analysis3_L_vs_Q']['bootstrap_ci_low']:.6f}, "
            f"{values['analysis3_L_vs_Q']['bootstrap_ci_high']:.6f}])."
        )
    if cross_lines:
        cross_result_block = f"""

The common mixed-depth cohort contains **{len(cohorts['cross_depth']):,} validation transcripts**.

| Datasets | Reference weights | $\\widehat L$ vs $H$ | $H$ vs $Q$ | $\\widehat L$ vs $Q$ |
|---:|---|---:|---:|---:|
{chr(10).join(cross_lines)}

{chr(10).join(contrast_text)}
"""
    else:
        cross_result_block = "\n\nThe mixed-depth sensitivity family was not included in this execution.\n"
    within_contrasts = contrasts.loc[
        (contrasts["contrast"] == "N10_minus_N2")
        & (contrasts["metric"] == "pearson")
    ]
    within_change_lines: list[str] = []
    for depth in DEPTH_ORDER:
        part = within_contrasts.loc[within_contrasts["depth"] == depth]
        values = {
            comparison: part.loc[part["comparison"] == comparison].iloc[0]
            for comparison in COMPARISONS
        }
        within_change_lines.append(
            f"For {DEPTH_LABELS[depth]}, the paired median $N=10$ minus $N=2$ "
            f"PCC changes were {values['analysis1_L_vs_H']['median_paired_difference']:.4f} "
            f"([{values['analysis1_L_vs_H']['bootstrap_ci_low']:.4f}, "
            f"{values['analysis1_L_vs_H']['bootstrap_ci_high']:.4f}]) for Analysis 1, "
            f"{values['analysis2_H_vs_Q']['median_paired_difference']:.4f} "
            f"([{values['analysis2_H_vs_Q']['bootstrap_ci_low']:.4f}, "
            f"{values['analysis2_H_vs_Q']['bootstrap_ci_high']:.4f}]) for Analysis 2, "
            f"and {values['analysis3_L_vs_Q']['median_paired_difference']:.4f} "
            f"([{values['analysis3_L_vs_Q']['bootstrap_ci_low']:.4f}, "
            f"{values['analysis3_L_vs_Q']['bootstrap_ci_high']:.4f}]) for Analysis 3."
        )
    if cohort_exclusions.empty:
        exclusion_text = (
            "No transcript was excluded after applying the prespecified "
            "boundary trim."
        )
    else:
        exclusion_counts = cohort_exclusions.groupby("cohort_key").size().to_dict()
        detail = ", ".join(
            f"{key}: {int(exclusion_counts.get(key, 0))}" for key in sorted(cohorts)
        )
        exclusion_text = (
            f"A total of {cohort_exclusions['transcript_id'].nunique():,} unique "
            "short transcript(s) were excluded because fewer than three codons "
            "remain after the prespecified boundary trim. Cohort-specific "
            f"counts are {detail}; full details are in `cohort_exclusions.csv`."
        )
    report = f"""# Synthetic reference-target audit

## Scope and estimands

{exclusion_text}

This is a post-hoc analysis of frozen **`{checkpoint_variant}`** prediction exports, selected by **`{checkpoint_selection_metric}`**.  No model was retrained and no oracle profile selected a checkpoint.  The primary experiment family contains 27 cumulative within-depth fits ($N=2,\\ldots,10$ at 0.25, 2, and 20 reads/codon), all with training seed 42 and equal reference weights.  The matched cohorts contain **{len(cohorts[analysis_cohort_key('within_depth', DEPTH_ORDER[0])]):,}**, **{len(cohorts[analysis_cohort_key('within_depth', DEPTH_ORDER[1])]):,}**, and **{len(cohorts[analysis_cohort_key('within_depth', DEPTH_ORDER[2])]):,}** validation transcripts at the three depths.  The curves are paired across $N$ within a depth, but not across depths.  These transcripts were excluded from gradient updates but were used for checkpoint selection; no independent synthetic test split exists.

The proximal occupancy target $Q_t$ is the normalized arithmetic consensus of the two saved finite TASEP trajectories.  It is simulator-defined and traffic-aware, but it is not an exact stationary occupancy or experimental biological truth.  A separate noiseless-$h^\\star$ export was not present; $h^\\star_{{t,d,i}}=(q_{{t,i}}b_{{t,d,i}})/(\\ell_t^{{-1}}\\sum_{{j=1}}^{{\\ell_t}}q_{{t,j}}b_{{t,d,j}})$ was therefore reconstructed deterministically from the saved occupancy and injected-bias profiles, never from realized counts.  The reference-defined target $H_t^{{(N)}}$ is constructed primarily from the geometric mean of these noiseless expected relative profiles, with the $qG$ construction used as a numerical identity check.

## Primary PCC results

| Depth | N | $\\widehat L$ vs $H$ | $H$ vs $Q$ | $\\widehat L$ vs $Q$ |
|---|---:|---:|---:|---:|
{chr(10).join(lines)}

{chr(10).join(within_change_lines)}

Thus the large end-to-end improvement is accompanied by a large improvement in reference-target fidelity, whereas recovery of the run-specific reference target is already high and does not improve in PCC.  This supports panel-average bias flattening as the dominant explanation for the $N$ trend in these runs.  It is not an additive PCC decomposition, and the single panel order prevents a causal separation of dataset count from panel composition.

The three PCCs must not be subtracted or interpreted as an additive error decomposition.  The exact additive statement is in CLR space and the exported table retains the model-error norm, reference-gap norm, their cross-term, and the total-error norm.

## Positional panel-average bias

The requested CLR reference error is algebraically identical to $\\mathrm{{sd}}_i(\\log G_i)$, so correlating those quantities would be circular.  The nontrivial diagnostic instead compares $\\mathrm{{sd}}_i(\\log G_i)$ with $1-\\mathrm{{PCC}}(H,Q)$.  Its pooled descriptive Spearman correlation is **{pooled['spearman_rho']:.3f}** across repeated transcript--panel rows.  The pooled p-value is not inferential because transcripts recur across $N$; condition-resolved associations are in `reference_error_associations.csv`.

## Decoder scale audit

All analyzed fits use the mass-free decoder $\\mu_{{dti}}=S_{{dt}}L_{{ti}}\\gamma_{{dti}}$ with $S_{{dt}}$ fixed to the observed arithmetic mean.  Across saved transcript--dataset rows, $m_{{dt}}=\\langle L_t\\gamma_{{dt}}\\rangle_{{I_t}}$ has median **{decoder_median:.4f}** and 5th--95th percentiles **{decoder_q.loc[0.05]:.4f}--{decoder_q.loc[0.95]:.4f}**.  Using tolerance $10^{{-3}}$, the incompatible fraction is **{decoder_prevalence:.3%}**.  Therefore the fitted mean generally does not preserve the observed arithmetic scale.  This is flagged diagnostically; the audit does not modify or retrain the decoder.

## Cross-depth reference weighting

The supplementary mixed-depth experiments compare equal and depth-ranked reference weights.  They are not independent bias-panel orderings: each newly added bias family contributes the same three depths, and the injected bias itself is depth-independent.  Consequently this is a reference-weight sensitivity analysis, not the requested fixed-$N$ panel-composition sensitivity.  No alternative cumulative bias ordering is present, so a genuine panel-order sensitivity figure is unavailable rather than invented.
{cross_result_block}
For every included bias family, the three depth weights sum to the same family-level mass under both conventions.  Therefore $G$, $H$, and Analysis 2 are exactly identical between equal and depth-ranked references in this design.  The only estimable ranking effect is a finite-training effect on $\\widehat L$; the largest absolute paired median PCC difference across learned-profile comparisons and panel sizes is **{maximum_rank_effect:.6f}**.  These experiments therefore do **not** provide evidence that depth ranking improves occupancy recovery.

## Sanity checks and limitations

All required algebraic checks pass except any row explicitly marked `FLAG` in `sanity_checks.csv`; the decoder incompatibility is intentionally a flag rather than a failed reconstruction check.  Bias seed 20260807 and count-sampling seed 20260808 are available in simulator metadata.  RNG seeds for the two exported TASEP trajectories were not saved, but exact regeneration is unnecessary because their occupancy profiles were exported directly.

Only one optimization seed and one cumulative bias order are available.  Confidence intervals therefore use paired transcript-cluster bootstrap sampling; a hierarchical panel/seed bootstrap is not identifiable.  Positions and the two TASEP trajectories are not treated as independent biological observations.

## Recommendation

The matched three-panel PCC figure belongs in the main paper because it distinguishes model recovery of its estimand, fidelity of that estimand to occupancy, and end-to-end occupancy recovery.  The CLR norm identity, $\\sigma_G$ scatter, per-transcript distributions, representative profiles, reference-weight sensitivity, gamma-centering check, and decoder-scale incompatibility belong in the appendix.  Claims should use “reference-defined target” for $H$ and “two-trajectory TASEP occupancy consensus” for $Q$; neither should be called experimentally established biological ground truth.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    paths = config["paths"]
    results_root = resolve_path(paths["results_root"])
    output_dir = args.output_dir or resolve_path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_variant = str(config["run_selection"]["checkpoint_variant"])
    include_cross = bool(
        config["run_selection"]["include_cross_depth_reference_sensitivity"]
    ) and not args.skip_cross_depth
    runs = discover_runs(
        results_root,
        seed=int(config["run_selection"]["training_seed"]),
        checkpoint_variant=checkpoint_variant,
        include_cross=include_cross,
    )
    by_cohort: dict[str, list[RunSpec]] = defaultdict(list)
    for run in runs:
        by_cohort[analysis_cohort_key(run.family, run.depth)].append(run)
    candidate_cohorts = {
        cohort_key: sorted(
            set.intersection(*(set(run.validation_ids) for run in cohort_runs))
        )
        for cohort_key, cohort_runs in by_cohort.items()
    }
    candidate_union_ids = set().union(
        *(set(ids) for ids in candidate_cohorts.values())
    )
    print(
        "Candidate matched cohorts: "
        + ", ".join(
            f"{family}={len(ids):,}" for family, ids in candidate_cohorts.items()
        ),
        flush=True,
    )
    trim = int(config["evaluation"]["boundary_trim_codons"])
    occupancies, occupancy_provenance = load_occupancy_consensus(
        resolve_path(paths["occupancy"]), candidate_union_ids
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
    union_ids = set().union(*(set(ids) for ids in cohorts.values()))
    print(
        "Analysis cohorts after boundary eligibility: "
        + ", ".join(f"{family}={len(ids):,}" for family, ids in cohorts.items())
        + f"; exclusions={len(cohort_exclusions):,}",
        flush=True,
    )
    biases, bias_provenance = load_biases(resolve_path(paths["bias_root"]), union_ids)

    panel_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    maxima_by_run: dict[str, dict[str, float]] = {}
    profile_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    pi_by_run: dict[str, dict[str, float]] = {}
    tolerances = {key: float(value) for key, value in config["tolerances"].items()}
    for index, run in enumerate(runs, start=1):
        pi, raw_pi_sum = extract_reference_weights(run)
        pi_by_run[run.run_id] = pi
        p_hash = panel_hash(run.datasets, pi)
        for dataset in run.datasets:
            panel_rows.append(
                {
                    "run_id": run.run_id,
                    "family": run.family,
                    "depth": run.depth,
                    "n_datasets": run.n_datasets,
                    "n_bias_families": run.n_bias_families,
                    "reference_weighting": run.reference_weighting,
                    "panel_id": p_hash,
                    "dataset": dataset,
                    "base_bias": base_bias_name(dataset),
                    "pi": pi[dataset],
                    "stored_pi_sum_before_float64_renormalization": raw_pi_sum,
                }
            )
        run_rows.append(
            {
                "run_id": run.run_id,
                "family": run.family,
                "depth": run.depth,
                "n_datasets": run.n_datasets,
                "n_bias_families": run.n_bias_families,
                "reference_weighting": run.reference_weighting,
                "training_seed": run.training_seed,
                "panel_id": p_hash,
                "split_id": run.split_id,
                "training_transcripts": len(run.train_ids),
                "validation_transcripts": len(run.validation_ids),
                "analysis_cohort_key": analysis_cohort_key(run.family, run.depth),
                "matched_analysis_transcripts": len(
                    cohorts[analysis_cohort_key(run.family, run.depth)]
                ),
                "config_path": str(run.config_path.relative_to(ROOT)),
                "config_sha256": sha256(run.config_path),
                "prediction_path": str(run.prediction_path.relative_to(ROOT)),
                "prediction_size_bytes": run.prediction_path.stat().st_size,
                "prediction_mtime_ns": run.prediction_path.stat().st_mtime_ns,
                "checkpoint_manifest_path": str(
                    run.checkpoint_manifest_path.relative_to(ROOT)
                ),
                "checkpoint_manifest_sha256": sha256(run.checkpoint_manifest_path),
                "split_path": str(run.split_path.relative_to(ROOT)),
                "split_sha256": sha256(run.split_path),
                "checkpoint_selection_metric": config["run_selection"][
                    "checkpoint_selection_metric"
                ],
            }
        )
        print(
            f"[{index}/{len(runs)}] {run.family} {run.depth} "
            f"N={run.n_datasets} {run.reference_weighting}",
            flush=True,
        )
        m_rows, d_rows, mk_rows, maxima = analyze_run(
            run,
            set(cohorts[analysis_cohort_key(run.family, run.depth)]),
            occupancies,
            biases,
            pi,
            trim,
            tolerances,
            profile_cache,
        )
        metric_rows.extend(m_rows)
        decoder_rows.extend(d_rows)
        mask_rows.extend(mk_rows)
        maxima_by_run[run.run_id] = maxima

    metrics = pd.DataFrame(metric_rows)
    decoder = pd.DataFrame(decoder_rows)
    masks = pd.DataFrame(mask_rows).drop_duplicates(
        ["transcript_id", "sense_length", "boundary_trim_codons", "mask_sha256"]
    )
    run_manifest = pd.DataFrame(run_rows)
    panel_table = pd.DataFrame(panel_rows)
    bootstrap_cfg = config["bootstrap"]
    summary, draws_by_cohort = aggregate_metrics(
        metrics,
        cohorts,
        bootstrap_replicates=int(bootstrap_cfg["replicates"]),
        bootstrap_seed=int(bootstrap_cfg["seed"]),
    )
    reference_only_summary = aggregate_reference_only(
        metrics,
        bootstrap_replicates=int(bootstrap_cfg["replicates"]),
        bootstrap_seed=int(bootstrap_cfg["seed"]) + 10_000,
    )
    contrasts = paired_contrasts(metrics, cohorts, draws_by_cohort)
    associations = reference_error_associations(metrics)

    oracle_selection_ok = all(
        row["checkpoint_selection_metric"]
        == config["run_selection"]["checkpoint_selection_metric"]
        for row in run_rows
    )
    sanity = build_sanity_table(
        maxima_by_run,
        panel_table,
        metrics,
        decoder,
        tolerances,
        oracle_selection_ok,
    )
    if bool((sanity.loc[sanity["status"] == "FAIL"]).shape[0]):
        failed = sanity.loc[sanity["status"] == "FAIL", "check"].tolist()
        raise RuntimeError(f"Required sanity checks failed: {failed}")

    selected, representative_profiles = select_representatives(metrics, profile_cache)
    run_manifest.to_csv(output_dir / "run_manifest.csv", index=False)
    panel_table.to_csv(output_dir / "panel_reference_weights.csv", index=False)
    pd.concat(
        [
            pd.DataFrame({"cohort_key": cohort_key, "transcript_id": ids})
            for cohort_key, ids in cohorts.items()
        ],
        ignore_index=True,
    ).to_csv(output_dir / "matched_transcript_ids.csv", index=False)
    cohort_exclusions.to_csv(output_dir / "cohort_exclusions.csv", index=False)
    masks.to_csv(output_dir / "evaluation_masks.csv", index=False)
    metrics.to_parquet(output_dir / "per_transcript_metrics.parquet", index=False)
    metrics.to_csv(output_dir / "per_transcript_metrics.csv.gz", index=False)
    decoder.to_parquet(output_dir / "decoder_scale_diagnostics.parquet", index=False)
    summary.to_csv(output_dir / "aggregate_summary.csv", index=False)
    reference_only_summary.to_csv(
        output_dir / "reference_only_union_cohort_summary.csv", index=False
    )
    contrasts.to_csv(output_dir / "paired_contrasts.csv", index=False)
    summary.loc[summary["family"] == "cross_depth"].to_csv(
        output_dir / "mixed_depth_reference_weight_summary.csv", index=False
    )
    contrasts.loc[
        contrasts["contrast"] == "quality_rank_minus_equal"
    ].to_csv(
        output_dir / "mixed_depth_ranked_minus_equal_contrasts.csv", index=False
    )
    associations.to_csv(output_dir / "reference_error_associations.csv", index=False)
    sanity.to_csv(output_dir / "sanity_checks.csv", index=False)
    selected.to_csv(output_dir / "representative_selection.csv", index=False)
    representative_profiles.to_csv(output_dir / "representative_profile_source.csv", index=False)

    dpi = int(config["figures"]["png_dpi"])
    plot_main(summary, reference_only_summary, output_dir, dpi)
    plot_manuscript_overview(summary, reference_only_summary, output_dir, dpi)
    plot_distributions(metrics, output_dir, dpi)
    plot_reference_scatter(metrics, associations, output_dir, dpi)
    plot_representatives(representative_profiles, output_dir, dpi)
    plot_reference_weight_sensitivity(summary, output_dir, dpi)
    write_report(
        output_dir,
        summary,
        contrasts,
        associations,
        sanity,
        metrics,
        decoder,
        cohorts,
        cohort_exclusions,
        checkpoint_variant=checkpoint_variant,
        checkpoint_selection_metric=str(
            config["run_selection"]["checkpoint_selection_metric"]
        ),
    )

    split_tables: dict[str, dict[str, Any]] = {}
    assignment_rows: list[dict[str, Any]] = []
    for run in runs:
        if run.split_id in split_tables:
            continue
        split_tables[run.split_id] = {
            "split_id": run.split_id,
            "train_identity_sha256": text_hash(run.train_ids),
            "validation_identity_sha256": text_hash(run.validation_ids),
            "training_transcripts": len(run.train_ids),
            "validation_transcripts": len(run.validation_ids),
        }
        assignment_rows.extend(
            {"split_id": run.split_id, "transcript_id": transcript_id, "assignment": "train"}
            for transcript_id in sorted(run.train_ids)
        )
        assignment_rows.extend(
            {"split_id": run.split_id, "transcript_id": transcript_id, "assignment": "validation"}
            for transcript_id in sorted(run.validation_ids)
        )
    pd.DataFrame(assignment_rows).to_csv(
        output_dir / "transcript_split_assignments.csv.gz", index=False
    )

    # Sampling seeds are common across the three depth exports; retain the
    # exact metadata and explicitly record the unavailable trajectory RNG seed.
    observation_metadata: dict[str, Any] = {}
    for depth in DEPTH_ORDER:
        raw_path = (
            ROOT
            / "Datasets/Synthetic_data"
            / depth
            / f"artificial_bias_3prime_aa_psite_counts_{depth}.parquet"
        )
        md = parquet_metadata(raw_path)
        observation_metadata[depth] = {
            "path": str(raw_path.relative_to(ROOT)),
            "bias_seed": md.get("riboart.bias_seed"),
            "observation_sampling_seed": md.get("riboart.observation_sampling_seed"),
            "nominal_reads_per_codon": md.get(
                "riboart.counts_per_codon_unbiased_baseline"
            ),
            "replicate_ids": md.get("riboart.samples"),
        }
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "configuration": str(config_path.relative_to(ROOT)),
        "configuration_sha256": sha256(config_path),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "checkpoint_variant": checkpoint_variant,
        "checkpoint_selection": (
            f"{config['run_selection']['checkpoint_selection_metric']} on validation "
            "observations; no oracle target"
        ),
        "training_and_evaluation_split_note": "validation excluded from gradient updates but reused for checkpoint selection; no test split exists",
        "occupancy": occupancy_provenance,
        "biases": bias_provenance,
        "simulator_observation_metadata": observation_metadata,
        "cohorts": {
            cohort_key: {
                "n_transcripts": len(ids),
                "identity_sha256": text_hash(ids),
                "excluded_for_boundary_eligibility": int(
                    (cohort_exclusions["cohort_key"] == cohort_key).sum()
                ),
            }
            for cohort_key, ids in cohorts.items()
        },
        "splits": list(split_tables.values()),
        "evaluation_mask": {
            "saved_model_mask": True,
            "terminal_entry_removed": True,
            "boundary_trim_codons_each_end": trim,
            "observed_zero_filtering": False,
            "renormalize_Q_H_L_on_exact_mask": True,
        },
        "uncertainty": config["bootstrap"],
        "panel_order_sensitivity": "unavailable: one cumulative bias-family ordering",
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
        "RIBOUNMIX_PLOT_TEX=1 .venv/bin/python "
        "analyses/analyze_synthetic_reference_target_audit.py "
        f"--config {shlex.quote(str(config_path.relative_to(ROOT)))}\n"
    )
    (output_dir / "commands.sh").write_text(command, encoding="utf-8")
    print(f"Wrote reference-target audit to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
