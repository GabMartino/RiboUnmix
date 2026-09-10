from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import hydra
import lightning as pl
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata

try:
    from omegaconf.nodes import AnyNode
except ImportError:
    AnyNode = None

from Dataloaders.RiboAIQueuingMultiDataset.RiboAIQueuingDatamoduleMultiDataset import (
    GroupedBatchStatistics,
    GroupedOptimizerBatchPlan,
    RiboAIQueuingDatamoduleMultiDataset,
    filter_sequences_by_max_cds_codons,
    load_dataset_quality_ranking,
    resolve_grouped_optimizer_batch_plan,
)
from Models.RiboQueuingModel import RiboQueuingModel
from Models.RiboQueuingModelLighningModule import (
    RiboQueuingModelLightningModule,
    resolve_sample_reduction_mode,
)
from Utils.checkpoints import find_checkpoint
from Utils.external_transcript_split import load_external_transcript_split
from Utils.reliability_references import transcript_id_hash
from Utils.stratified_transcript_split import (
    assign_reliability_quantile_bins as shared_assign_reliability_quantile_bins,
    css_bin as shared_css_bin,
    css_count as shared_css_count,
    sample_stratified_ids as shared_sample_stratified_ids,
)


# ============================================================
# Torch checkpoint safety for OmegaConf / NumPy objects
# ============================================================

safe_globals = [
    np.dtype,
    DictConfig,
    ListConfig,
    ContainerMetadata,
    Any,
]

if AnyNode is not None:
    safe_globals.append(AnyNode)

try:
    safe_globals.append(np._core.multiarray.scalar)
except AttributeError:
    safe_globals.append(np.core.multiarray.scalar)

try:
    safe_globals.append(np.dtypes.StrDType)
except AttributeError:
    pass

torch.serialization.add_safe_globals(safe_globals)


# ============================================================
# Generic helpers
# ============================================================

def cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Safe nested getter for DictConfig/dict/object configs."""
    cur = cfg

    for key in path.split("."):
        if cur is None:
            return default

        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
        else:
            if not hasattr(cur, key):
                return default
            cur = getattr(cur, key)

    return cur


def cfg_bool(cfg: Any, path: str, default: bool = False) -> bool:
    """Read a config value as bool without treating the string "false" as True."""
    value = cfg_get(cfg, path, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "none", "null", ""}:
            return False

    return bool(value)


def cfg_optional_positive_int(cfg: Any, path: str) -> int | None:
    """Resolve an optional positive integer configuration value."""
    value = cfg_get(cfg, path, None)
    if value is None:
        return None
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{path} must be null or at least one, got {value!r}.")
    return resolved


def open_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return {} if data is None else data


def dataset_paths_for(
    *,
    mapping: dict[str, str],
    datasets: Sequence[str],
    source_label: str,
) -> list[str]:
    missing = [dataset for dataset in datasets if dataset not in mapping]
    if missing:
        raise KeyError(
            f"Dataset(s) {missing} missing from dataset path source {source_label}."
        )

    return [str(mapping[dataset]) for dataset in datasets]


def normalize_dataset_list(raw: Any) -> list[str]:
    if raw is None:
        return []

    if isinstance(raw, str):
        return [raw]

    return [str(x) for x in list(raw)]


def get_datasets(cfg: DictConfig) -> list[str]:
    raw = cfg.experiment.dataset

    if isinstance(raw, str) and raw.lower() == "all":
        datasets = list(cfg.dataset_config.dataset_path.keys())

        OmegaConf.set_struct(cfg, False)
        cfg.experiment.dataset = datasets
        OmegaConf.set_struct(cfg, True)

        print(f"Command 'all' detected. Loading all {len(datasets)} datasets.")
        return datasets

    return normalize_dataset_list(raw)


def get_split_universe_datasets(
    cfg: DictConfig,
    experiment_datasets: list[str],
) -> list[str]:
    """
    Dataset universe used for both the transcript pool and fixed validation.

    Example:
        experiment.dataset: ["grimson_2019"]
        split.master_dataset_universe: ["grimson_2019", "kutay_2021"]

    The overall pool is generated from Grimson ∪ Kutay and validation
    candidates must be present in both. The resulting validation IDs stay fixed
    when an experiment later selects only Grimson.
    """
    raw = cfg_get(cfg, "split.master_dataset_universe", None)

    if raw is None:
        print(
            "\n[split] No split.master_dataset_universe provided. "
            "Using experiment.dataset as split universe; validation will not "
            "be fixed across runs with different experiment datasets.\n"
        )
        return list(experiment_datasets)

    if isinstance(raw, str) and raw.lower() == "all":
        return list(cfg.dataset_config.dataset_path.keys())

    split_universe = normalize_dataset_list(raw)

    if len(split_universe) == 0:
        raise ValueError("split.master_dataset_universe was provided but is empty.")

    return split_universe


def make_dataset_signature(datasets: list[str]) -> str:
    raw = "_".join(sorted(datasets))

    if len(raw) <= 100:
        return raw

    short_hash = hashlib.md5(raw.encode()).hexdigest()[:6]
    return f"{len(datasets)}_datasets_mix_{short_hash}"


def format_run_tag_value(value: Any) -> str:
    return (
        str(value)
        .replace("/", "-")
        .replace(".", "p")
        .replace("-", "m")
        .replace(" ", "")
    )


def make_dataset_selection_run_tag(cfg: DictConfig) -> str:
    """Identify the actual dataset subset in every run artifact path."""
    datasets = sorted(normalize_dataset_list(cfg_get(cfg, "experiment.dataset", [])))
    if not datasets:
        return "DataN0"

    if len(datasets) == 1:
        dataset_token = format_run_tag_value(datasets[0])
        return f"DataN1_{dataset_token}"

    subset_hash = hashlib.md5("\n".join(datasets).encode()).hexdigest()[:6]
    return f"DataN{len(datasets)}_h{subset_hash}"


def make_loss_run_tag(cfg: DictConfig) -> str:
    nb_weight = format_run_tag_value(cfg_get(cfg, "loss.replica_nb_weight"))
    raw_weight = format_run_tag_value(
        cfg_get(cfg, "loss.consensus_raw_pcc_weight")
    )
    vst_weight = format_run_tag_value(
        cfg_get(cfg, "loss.consensus_nb_vst_pcc_weight")
    )
    gamma_weight = format_run_tag_value(cfg_get(cfg, "loss.gamma_reg_weight"))
    reduction = format_run_tag_value(resolve_sample_reduction_mode(cfg.loss))
    experiment_mode = format_run_tag_value(
        cfg_get(cfg, "loss.experiment_mode", "mean_gradient_reweighted_nb")
    )
    alpha_mode = format_run_tag_value(cfg_get(cfg, "model.alpha_mode", "learned"))
    beta = format_run_tag_value(cfg_get(cfg, "loss.nb_mean_gradient_beta", 0.0))
    return (
        f"Loss-rNB{nb_weight}-cPCC{raw_weight}"
        f"-cVSTPCC{vst_weight}-Gamma{gamma_weight}-Reduce{reduction}"
        f"-NBMode{experiment_mode}-Alpha{alpha_mode}-Beta{beta}"
    )


def make_sampling_run_tag(cfg: DictConfig) -> str | None:
    sampling = str(cfg_get(cfg, "data.train_sampling_strategy", "default")).strip()
    if sampling in {"", "default", "None", "none"}:
        return None
    return f"Sampling_{format_run_tag_value(sampling)}"


def make_numerical_eligibility_run_tag(cfg: DictConfig) -> str | None:
    """Identify gamma support and sequence eligibility that change a run."""
    parts: list[str] = []
    raw_bound = cfg_get(
        cfg,
        "model.dataset_bias_params.raw_log_gamma_bound",
        None,
    )
    if raw_bound is not None:
        parts.append(f"GammaRawB{format_run_tag_value(raw_bound)}")
    max_cds_codons = cfg_get(cfg, "data.max_cds_codons", None)
    if max_cds_codons is not None:
        parts.append(f"CDSMax{format_run_tag_value(max_cds_codons)}")
    return "-".join(parts) if parts else None


def make_gamma_centering_run_tag(cfg: DictConfig) -> str:
    mode = str(
        cfg_get(
            cfg,
            "model.gamma_centering.mode",
            "disabled",
        )
    ).lower()
    if mode == "disabled":
        return "GammaCtrOff"
    weighting = str(
        cfg_get(
            cfg,
            "model.gamma_centering.reference.weighting",
            "equal",
        )
    ).lower()
    mode_tag = "FixedRef" if mode == "fixed_reference" else "Batch"
    if weighting == "quality_rank":
        power = format_run_tag_value(
            cfg_get(
                cfg,
                "model.gamma_centering.reference.quality_rank_power",
                1.0,
            )
        )
        return f"GammaCtr{mode_tag}QRankP{power}"
    return f"GammaCtr{mode_tag}Equal"


def _sequence_feature_routes(cfg: DictConfig) -> dict[str, str]:
    raw_config = cfg_get(cfg, "model.additional_sequence_features", {}) or {}
    return {
        str(name): str(dict(spec or {}).get("route", "none")).lower()
        for name, spec in raw_config.items()
    }


def infer_sequence_features_preset(cfg: DictConfig) -> str:
    """Infer the Slurm feature preset from the fully resolved Hydra routes."""
    routes = _sequence_feature_routes(cfg)
    useful_features = ("exo", "gmp", "tmp", "openen", "tAI_profile_codon")
    useful_routes = {name: routes.get(name, "none") for name in useful_features}

    if all(route == "none" for route in useful_routes.values()):
        return "Baseline"
    for route, label in (
        ("biological", "Biological"),
        ("dataset_bias", "DatasetBias"),
        ("both", "Both"),
    ):
        if all(value == route for value in useful_routes.values()):
            return label

    enabled = {name: route for name, route in useful_routes.items() if route != "none"}
    if enabled == {"openen": "biological", "tAI_profile_codon": "biological"}:
        return "CoreBio"

    single_bio_labels = {
        "exo": "ExoBio",
        "gmp": "GmpBio",
        "tmp": "TmpBio",
        "openen": "OpenenBio",
        "tAI_profile_codon": "TaiBio",
    }
    if len(enabled) == 1:
        name, route = next(iter(enabled.items()))
        if route == "biological" and name in single_bio_labels:
            return single_bio_labels[name]
    return "Custom"


def make_sequence_features_run_tag(cfg: DictConfig) -> str:
    """Encode the readable preset and exact feature routing in output paths."""
    raw_config = cfg_get(cfg, "model.additional_sequence_features", {}) or {}
    aliases = {
        "openen": "open",
        "tai_profile_codon": "tai",
    }
    grouped: dict[str, list[str]] = {
        "biological": [],
        "dataset_bias": [],
        "both": [],
    }

    for raw_name, raw_spec in raw_config.items():
        spec = dict(raw_spec or {})
        route = str(spec.get("route", "none")).lower()
        if route == "none":
            continue
        if route not in grouped:
            raise ValueError(
                f"Invalid additional-sequence-feature route {route!r} for "
                f"{raw_name!r}."
            )

        name = aliases.get(str(raw_name).lower(), str(raw_name).lower())
        token = format_run_tag_value(name)
        missing_values = list(spec.get("missing_values", []) or [])
        fill_value = float(spec.get("fill_value", 0.0))
        if missing_values:
            token += f"Fill{format_run_tag_value(fill_value)}"
        scale = float(spec.get("scale", 1.0))
        if not np.isclose(scale, 1.0):
            token += f"s{format_run_tag_value(scale)}"
        grouped[route].append(token)

    preset = infer_sequence_features_preset(cfg)
    if not any(grouped.values()):
        return f"FeatPreset{preset}_SeqFeatBase"

    labels = {
        "biological": "Bio",
        "dataset_bias": "Bias",
        "both": "Both",
    }
    parts = []
    for route in ("biological", "dataset_bias", "both"):
        if grouped[route]:
            parts.append(f"{labels[route]}-{'-'.join(sorted(grouped[route]))}")
    return f"FeatPreset{preset}_SeqFeat" + "_".join(parts)


# Linux limits a single path component to 255 bytes, and the run tag is used
# verbatim as one directory name under checkpoints/, logs/ and results/.
RUN_TAG_MAX_COMPONENT_BYTES = 255


def abbreviate_run_tag(
    tag: str, *, max_bytes: int = RUN_TAG_MAX_COMPONENT_BYTES
) -> str:
    """Keep a run tag usable as a single filesystem path component.

    Single-dataset runs inline the dataset name instead of hashing a subset, so
    a descriptive name such as ``artificial_bias_gc_fraction_gt_0p7`` pushes the
    tag past the component limit and every artifact directory fails to be
    created. Truncating with a digest of the full tag keeps the readable head
    while remaining unique per configuration. Tags within the limit are returned
    unchanged, so existing artifact paths are untouched.
    """
    encoded = tag.encode("utf-8")
    if len(encoded) <= max_bytes:
        return tag
    suffix = f"_h{hashlib.md5(encoded).hexdigest()[:8]}"
    keep = max_bytes - len(suffix)
    if keep <= 0:
        raise ValueError(f"max_bytes={max_bytes} is too small for a run tag.")
    return encoded[:keep].decode("utf-8", "ignore") + suffix


def make_run_tag(cfg: DictConfig) -> str:
    parts = [
        "queueNB",
        make_dataset_selection_run_tag(cfg),
        make_loss_run_tag(cfg),
    ]

    sampling_tag = make_sampling_run_tag(cfg)
    if sampling_tag is not None:
        parts.append(sampling_tag)

    numerical_eligibility_tag = make_numerical_eligibility_run_tag(cfg)
    if numerical_eligibility_tag is not None:
        parts.append(numerical_eligibility_tag)

    parts.append(make_gamma_centering_run_tag(cfg))
    parts.append(make_sequence_features_run_tag(cfg))

    return abbreviate_run_tag("_".join(parts))


def dataset_name_from_path(path: str | Path) -> str:
    return os.path.basename(str(path)).split(".")[0]


def sanitize_metric_name_for_filename(metric_name: str) -> str:
    return str(metric_name).replace("/", "__")


def shared_logger_version_from_environment() -> str | None:
    """
    Return a deterministic logger version shared by all externally launched DDP
    ranks. With Slurm + srun every rank executes this script independently before
    Lightning has a Trainer/rank-zero guard, so TensorBoardLogger's auto version
    discovery can race and create one version_* directory per rank. Local
    torchrun uses ``RIBOAI_LOGGER_VERSION`` for the same purpose.
    """
    # Local torchrun launches do not provide SLURM identifiers, but all ranks
    # still need one deterministic TensorBoard version directory. The local
    # launcher supplies this explicit value before falling back to Slurm's
    # job/array identifiers.
    explicit_version = os.environ.get("RIBOAI_LOGGER_VERSION")
    if explicit_version:
        return explicit_version

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if array_task_id and array_task_id not in {"", "NO_VAL", "4294967294"}:
            return f"slurm_{slurm_job_id}_{array_task_id}"
        return f"slurm_{slurm_job_id}"

    return None


def env_global_rank() -> int:
    for key in ("RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return 0


# ============================================================
# Fixed common-transcript, reliability/CSS-stratified split logic
# ============================================================

def css_count(x: Any) -> int:
    """Compatibility wrapper around the shared split-stratification helper."""
    return shared_css_count(x)


def css_bin(n_css: int) -> str:
    return shared_css_bin(n_css)


def build_transcript_metadata(
    *,
    sequences_path: str | Path,
    datasets_paths: Sequence[str | Path],
    max_cds_codons: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Build transcript-level metadata from:
        1. master sequence/CSS parquet
        2. selected dataset-specific ribo parquets

    Returned metadata is keyed by transcript_id and contains:
        - datasets
        - availability
        - css_count
        - css_bin
        - has_css

    Important:
        The supplied datasets_paths define the split universe.
    """
    # Ordinarily only the CSS column is needed. When a maximum CDS length is
    # configured, read the compact codon-token sequence as well so eligibility
    # is resolved before validation sampling rather than after a split is made.
    _seq_available = set(pq.read_schema(sequences_path).names)
    _seq_css_col = (
        "conserved_stalling_sites"
        if "conserved_stalling_sites" in _seq_available
        else "css"
    )
    sequence_column: str | None = None
    if max_cds_codons is not None:
        sequence_column = "codons" if "codons" in _seq_available else "ref"
        if sequence_column not in _seq_available:
            raise KeyError(
                f"{sequences_path} contains neither 'codons' nor 'ref', so "
                "data.max_cds_codons cannot be enforced."
            )
    _seq_columns = list(
        dict.fromkeys(
            c
            for c in ("transcript_id", _seq_css_col, sequence_column)
            if c is not None and c in _seq_available
        )
    )
    seq_df = pd.read_parquet(sequences_path, columns=_seq_columns or None)

    if "transcript_id" in seq_df.columns:
        seq_df = seq_df.set_index("transcript_id")

    seq_df.index = seq_df.index.astype(str)

    length_by_transcript: dict[str, int] = {}
    if sequence_column is not None:
        seq_df, length_filter = filter_sequences_by_max_cds_codons(
            seq_df,
            sequence_column=sequence_column,
            max_cds_codons=max_cds_codons,
        )
        length_by_transcript = {
            str(transcript_id): int(len(sequence))
            for transcript_id, sequence in seq_df[sequence_column].items()
        }
        print(
            "[split] CDS-length eligibility: "
            f"max={int(max_cds_codons):,} codons, "
            f"removed={length_filter['removed_transcripts']:,}/"
            f"{length_filter['input_transcripts']:,}, "
            "longest retained="
            f"{length_filter['longest_retained_cds_codons']:,} codons."
        )
        seq_df = seq_df.drop(columns=[sequence_column])

    css_col = "conserved_stalling_sites" if "conserved_stalling_sites" in seq_df.columns else "css"

    if css_col not in seq_df.columns:
        raise KeyError(
            "Could not find CSS column. Expected 'conserved_stalling_sites' or 'css'."
        )

    tid_to_datasets: dict[str, set[str]] = defaultdict(set)

    for path in datasets_paths:
        dataset_name = dataset_name_from_path(path)
        # Only the `id` column is needed to record dataset availability.
        df = pd.read_parquet(path, columns=["id"])

        if "id" not in df.columns:
            raise KeyError(f"'id' column missing in dataset parquet: {path}")

        for tid in df["id"].astype(str).values:
            tid_to_datasets[str(tid)].add(dataset_name)

    valid_ids = sorted(set(seq_df.index.astype(str)).intersection(tid_to_datasets.keys()))

    if len(valid_ids) == 0:
        raise RuntimeError("No transcript IDs overlap between sequence table and ribo datasets.")

    metadata: dict[str, dict[str, Any]] = {}

    for tid in valid_ids:
        datasets = sorted(tid_to_datasets[tid])
        n_css = css_count(seq_df.loc[tid, css_col])

        if len(datasets) == 1:
            availability = f"{datasets[0]}_only"
        else:
            availability = "paired_" + "__".join(datasets)

        metadata[tid] = {
            "transcript_id": tid,
            "datasets": datasets,
            "availability": availability,
            "cds_codon_length": length_by_transcript.get(tid),
            "css_count": int(n_css),
            "css_bin": css_bin(int(n_css)),
            "has_css": int(n_css) > 0,
        }

    return metadata


def load_common_transcript_reliability(
    *,
    datasets_paths: Sequence[str | Path],
    eligible_ids: Sequence[str],
) -> tuple[list[str], dict[str, float], list[str]]:
    """Load positive pair weights and aggregate them for common transcripts.

    A validation candidate must have one retained row in every supplied dataset.
    Its scalar reliability score is the ordinary median of its dataset-specific,
    median-normalized transcript weights.  The score is used only to stratify
    validation selection; the original pair weights remain unchanged and are
    later used by the loss.
    """
    if not datasets_paths:
        raise ValueError("At least one validation dataset path is required.")

    eligible = set(map(str, eligible_ids))
    common_ids: set[str] | None = None
    weights_by_dataset: list[dict[str, float]] = []
    dataset_names: list[str] = []

    for path in datasets_paths:
        path = Path(path)
        dataset_name = dataset_name_from_path(path)
        if dataset_name in dataset_names:
            raise ValueError(
                f"Duplicate validation dataset name {dataset_name!r} from {path}."
            )

        available_columns = set(pq.read_schema(path).names)
        missing = {"id", "weight"} - available_columns
        if missing:
            raise KeyError(
                f"Validation split requires columns {sorted(missing)} in "
                f"dataset={dataset_name}, path={path}."
            )

        frame = pd.read_parquet(path, columns=["id", "weight"])
        transcript_ids = frame["id"].astype(str)
        if bool(transcript_ids.duplicated().any()):
            duplicate = str(transcript_ids[transcript_ids.duplicated()].iloc[0])
            raise ValueError(
                "Duplicate transcript ID while building the validation split: "
                f"dataset={dataset_name}, transcript={duplicate}."
            )

        weights = pd.to_numeric(frame["weight"], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=False,
        )
        invalid_finite = ~np.isfinite(weights)
        if bool(invalid_finite.any()):
            index = int(np.flatnonzero(invalid_finite)[0])
            raise ValueError(
                "Non-finite transcript weight while building the validation "
                f"split: dataset={dataset_name}, "
                f"transcript={transcript_ids.iloc[index]}, weight={weights[index]}."
            )
        nonpositive = weights <= 0.0
        if bool(nonpositive.any()):
            index = int(np.flatnonzero(nonpositive)[0])
            raise ValueError(
                "Validation splitting requires strictly positive transcript "
                f"weights: dataset={dataset_name}, "
                f"transcript={transcript_ids.iloc[index]}, weight={weights[index]}."
            )

        weight_map = dict(zip(transcript_ids.tolist(), weights.tolist(), strict=True))
        ids_here = set(weight_map).intersection(eligible)
        common_ids = ids_here if common_ids is None else common_ids.intersection(ids_here)
        weights_by_dataset.append(weight_map)
        dataset_names.append(dataset_name)

    common = sorted(common_ids or set())
    if not common:
        raise RuntimeError(
            "No transcript with a valid sequence and positive weight is common to "
            f"all selected validation datasets: {dataset_names}."
        )

    aggregate_scores = {
        tid: float(np.median([weight_map[tid] for weight_map in weights_by_dataset]))
        for tid in common
    }
    if not all(np.isfinite(score) and score > 0.0 for score in aggregate_scores.values()):
        raise RuntimeError("Common-transcript reliability aggregation produced an invalid score.")

    return common, aggregate_scores, dataset_names


def assign_reliability_quantile_bins(
    scores: dict[str, float],
    *,
    number_of_bins: int,
) -> dict[str, int]:
    """Compatibility wrapper around the shared rank-quantile implementation."""
    return shared_assign_reliability_quantile_bins(
        scores, number_of_bins=number_of_bins
    )


def sample_stratified_validation_ids(
    *,
    candidate_ids: Sequence[str],
    stratum_by_transcript: dict[str, str],
    target_count: int,
    rng: np.random.Generator,
) -> list[str]:
    """Compatibility wrapper around the shared proportional stratum sampler."""
    return shared_sample_stratified_ids(
        candidate_ids=candidate_ids,
        stratum_by_transcript=stratum_by_transcript,
        target_count=target_count,
        rng=rng,
    )


def print_split_summary(
    *,
    name: str,
    ids: Sequence[str],
    metadata: dict[str, dict[str, Any]],
) -> None:
    ids = list(map(str, ids))

    support_counts: dict[int, int] = defaultdict(int)
    css_bin_counts: dict[str, int] = defaultdict(int)

    css_total = 0
    css_positive = 0

    for tid in ids:
        m = metadata[tid]

        support_counts[len(m["datasets"])] += 1
        css_bin_counts[str(m["css_bin"])] += 1

        css_total += int(m["css_count"])
        css_positive += int(bool(m["has_css"]))

    print(f"\n=== {name} split summary ===")
    print(f"transcripts: {len(ids)}")
    print(f"CSS-positive transcripts: {css_positive}")
    print(f"total CSS sites: {css_total}")

    print("transcripts by retained dataset count:")
    for dataset_count, value in sorted(support_counts.items()):
        print(f"  datasets={dataset_count:3d}: {value}")

    print("CSS bins:")
    for key, value in sorted(css_bin_counts.items()):
        print(f"  {key:12s} {value}")

    reliability_scores = [
        float(metadata[tid]["validation_reliability_score"])
        for tid in ids
        if metadata[tid].get("validation_reliability_score") is not None
    ]
    if reliability_scores:
        score_array = np.asarray(reliability_scores, dtype=np.float64)
        print(
            "common-transcript reliability: "
            f"n={len(score_array)}, min={score_array.min():.4f}, "
            f"median={np.median(score_array):.4f}, "
            f"max={score_array.max():.4f}"
        )

        bin_counts: dict[int, int] = defaultdict(int)
        for tid in ids:
            bin_id = metadata[tid].get("validation_reliability_bin")
            if bin_id is not None:
                bin_counts[int(bin_id)] += 1
        print("validation reliability bins:")
        for bin_id, value in sorted(bin_counts.items()):
            print(f"  qbin_{bin_id:02d}: {value}")


def fixed_common_validation_split(
    *,
    sequences_path: str | Path,
    split_universe_dataset_paths: Sequence[str | Path],
    validation_frac: float = 0.10,
    random_seed: int = 42,
    validation_weight_bins: int = 10,
    max_cds_codons: int | None = None,
) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    """Build one fixed common validation panel and use all other IDs for training.

    Validation candidates have a retained positive-weight row in every dataset
    of the master split universe. Their split-only reliability score is

        median_d(weight[d, transcript]).

    Candidates are stratified jointly by reliability rank-quantile and CSS-count
    bin. CSS is therefore a sampling signal, not a separately held-out split and
    not a fixed validation quota. The same master universe, seed, and input data
    always produce the same validation IDs, independent of experiment.dataset.

    Every other transcript in the master union is assigned to training. Each
    selected dataset later contributes the subset of those training IDs for
    which it has a retained row, so training sizes may differ by dataset.
    """
    validation_frac = float(validation_frac)
    if not math.isfinite(validation_frac) or not 0.0 < validation_frac < 1.0:
        raise ValueError("split.validation_frac must be finite and in (0, 1).")

    rng = np.random.default_rng(int(random_seed))

    metadata = build_transcript_metadata(
        sequences_path=sequences_path,
        datasets_paths=split_universe_dataset_paths,
        max_cds_codons=max_cds_codons,
    )

    all_ids = sorted(metadata.keys())
    common_ids, reliability_scores, validation_dataset_names = (
        load_common_transcript_reliability(
            datasets_paths=split_universe_dataset_paths,
            eligible_ids=all_ids,
        )
    )
    common_id_set = set(common_ids)
    for tid, transcript_metadata in metadata.items():
        is_common = tid in common_id_set
        transcript_metadata["common_to_validation_datasets"] = is_common
        transcript_metadata["validation_reliability_score"] = (
            float(reliability_scores[tid]) if is_common else None
        )
        transcript_metadata["validation_reliability_bin"] = None
        transcript_metadata["validation_stratum"] = None

    validation_target = int(round(len(common_ids) * validation_frac))
    if validation_target <= 0:
        raise ValueError(
            "validation_frac produces an empty validation target; increase "
            "split.validation_frac."
        )
    if validation_target >= len(common_ids):
        raise ValueError(
            "The fixed validation target must leave at least one common "
            f"transcript for training; requested={validation_target}, "
            f"common={len(common_ids)}."
        )

    reliability_bins = assign_reliability_quantile_bins(
        reliability_scores,
        number_of_bins=validation_weight_bins,
    )
    validation_strata: dict[str, str] = {}
    for tid, bin_id in reliability_bins.items():
        metadata[tid]["validation_reliability_bin"] = int(bin_id)
        stratum = f"qbin_{int(bin_id):02d}__{metadata[tid]['css_bin']}"
        metadata[tid]["validation_stratum"] = stratum
        validation_strata[tid] = stratum

    validation_ids = sample_stratified_validation_ids(
        candidate_ids=common_ids,
        stratum_by_transcript=validation_strata,
        target_count=validation_target,
        rng=rng,
    )
    validation_set = set(validation_ids)
    train_ids = sorted(set(all_ids) - validation_set)

    train_set = set(train_ids)
    if train_set & validation_set:
        raise RuntimeError("Overlap between train and validation splits.")

    covered = train_set | validation_set

    if covered != set(all_ids):
        missing = set(all_ids) - covered
        extra = covered - set(all_ids)

        raise RuntimeError(
            f"Split coverage error. missing={len(missing)}, extra={len(extra)}"
        )

    train_ids = sorted(train_set)
    validation_ids = sorted(validation_set)

    for tid in train_ids:
        metadata[tid]["split_role"] = "train"
    for tid in validation_ids:
        metadata[tid]["split_role"] = "validation"

    if not all(metadata[tid]["common_to_validation_datasets"] for tid in validation_ids):
        raise RuntimeError(
            "Internal split error: validation contains a transcript that is not "
            "common to every master-universe dataset."
        )

    print_split_summary(name="Train", ids=train_ids, metadata=metadata)
    print_split_summary(name="Validation", ids=validation_ids, metadata=metadata)
    print(
        "\n[split] Fixed validation panel is common to all master datasets: "
        f"datasets={len(validation_dataset_names)}, candidates={len(common_ids)}, "
        f"selected={len(validation_ids)}, "
        f"weight_bins={min(int(validation_weight_bins), len(common_ids))}."
    )

    return train_ids, validation_ids, metadata


def save_split_manifest(
    *,
    out_file: Path,
    experiment_datasets: list[str],
    split_universe_datasets: list[str],
    split_dataset_source: str | None = None,
    split_dataset_paths: Sequence[str] | None = None,
    training_dataset_paths: Sequence[str] | None = None,
    validation_weight_bins: int = 10,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    metadata: dict[str, dict[str, Any]],
    seed: int,
    validation_frac: float,
    max_cds_codons: int | None = None,
) -> None:
    """Save the fixed validation panel and master-universe provenance."""
    out_file.parent.mkdir(parents=True, exist_ok=True)

    def metadata_value_counts(ids: Sequence[str], field: str) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for tid in map(str, ids):
            value = metadata[tid].get(field)
            if value is not None:
                counts[str(value)] += 1
        return dict(sorted(counts.items()))

    def reliability_bin_counts(ids: Sequence[str]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for tid in map(str, ids):
            bin_id = metadata[tid].get("validation_reliability_bin")
            if bin_id is not None:
                counts[f"qbin_{int(bin_id):02d}"] += 1
        return dict(sorted(counts.items()))

    def reliability_summary(ids: Sequence[str]) -> dict[str, float | int | None]:
        values = np.asarray(
            [
                float(metadata[tid]["validation_reliability_score"])
                for tid in map(str, ids)
                if metadata[tid].get("validation_reliability_score") is not None
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
        return {
            "count": int(values.size),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }

    common_ids = [
        tid
        for tid, transcript_metadata in metadata.items()
        if bool(transcript_metadata.get("common_to_validation_datasets", False))
    ]
    universe_count = len(metadata)

    manifest = {
        "seed": int(seed),
        "experiment_datasets": list(experiment_datasets),
        "split_universe_datasets": list(split_universe_datasets),
        "split_dataset_source": split_dataset_source,
        "split_dataset_paths": list(map(str, split_dataset_paths or [])),
        "training_dataset_paths": list(map(str, training_dataset_paths or [])),
        "sequence_eligibility": {
            "max_cds_codons": (
                None if max_cds_codons is None else int(max_cds_codons)
            ),
            "rule": (
                "retain the complete transcript when CDS codon length is at "
                "most max_cds_codons; never truncate"
            ),
        },
        "validation_selection": {
            "strategy": "fixed_master_common_weight_css_stratified",
            "reference_datasets": list(map(str, split_universe_datasets)),
            "candidate_rule": (
                "retained positive-weight row in every master-universe dataset"
            ),
            "aggregate_reliability": "median_dataset_specific_weight",
            "weight_bins": int(validation_weight_bins),
            "css_role": "joint stratification signal; no separate split or quota",
            "configured_fraction_of_common_candidates": float(validation_frac),
            "fixed_across_experiment_dataset_subsets": True,
            "all_validation_ids_are_common": all(
                bool(metadata[tid].get("common_to_validation_datasets", False))
                for tid in map(str, validation_ids)
            ),
        },
        "realized_fractions": {
            "train_of_master_union": len(train_ids) / max(universe_count, 1),
            "validation_of_master_union": len(validation_ids) / max(universe_count, 1),
            "validation_of_common_candidates": len(validation_ids) / max(len(common_ids), 1),
        },
        "counts": {
            "master_union": universe_count,
            "common_validation_candidates": len(common_ids),
            "train": len(train_ids),
            "validation": len(validation_ids),
        },
        "css_bin_counts": {
            "common_candidates": metadata_value_counts(common_ids, "css_bin"),
            "train": metadata_value_counts(train_ids, "css_bin"),
            "validation": metadata_value_counts(validation_ids, "css_bin"),
        },
        "validation_stratum_counts": {
            "common_candidates": metadata_value_counts(
                common_ids,
                "validation_stratum",
            ),
            "validation": metadata_value_counts(
                validation_ids,
                "validation_stratum",
            ),
        },
        "validation_reliability_bin_counts": {
            "common_candidates": reliability_bin_counts(common_ids),
            "train": reliability_bin_counts(train_ids),
            "validation": reliability_bin_counts(validation_ids),
        },
        "validation_reliability_summary": {
            "common_candidates": reliability_summary(common_ids),
            "train": reliability_summary(train_ids),
            "validation": reliability_summary(validation_ids),
        },
        "train_ids": list(map(str, train_ids)),
        "validation_ids": list(map(str, validation_ids)),
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved split manifest: {out_file}")


# ============================================================
# Checkpoints
# ============================================================

def load_weights_only(
    lit_model: pl.LightningModule,
    ckpt_path: str | Path,
) -> None:
    """
    Loads only model weights.

    Correct for checkpoints saved with save_weights_only=True.
    It intentionally does not restore optimizer/scheduler state, but the model
    state itself must match exactly. Partial legacy loads are not supported.
    """
    ckpt_path = Path(ckpt_path)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    has_fixed_reference_provenance = any(
        key.endswith("gamma_reference_dataset_ids") for key in state_dict
    )
    torch_model = getattr(lit_model, "model", None)
    if (
        torch_model is not None
        and getattr(torch_model, "gamma_centering_mode", "disabled")
        == "fixed_reference"
        and not has_fixed_reference_provenance
    ):
        raise RuntimeError(
            "Checkpoint is incompatible with fixed-reference gamma centering: "
            "it has no checkpointed gamma reference-panel provenance. Load it "
            "with model.gamma_centering.mode=batch_grouped to reproduce its "
            "historical function. Applying a new fixed-reference gauge to old "
            "weights is not supported implicitly."
        )

    try:
        lit_model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} is incompatible with the current model "
            "state. Partial or legacy state-dict loading is disabled; use a "
            "checkpoint produced by the current configuration."
        ) from exc

    print(f"Loaded weights from: {ckpt_path}")


def resolve_gamma_reference_panel(
    *,
    cfg: DictConfig,
    experiment_datasets: Sequence[str],
    dataset_encoding: dict[str, int],
) -> dict[str, list]:
    """Resolve a fixed panel once, before model construction.

    By default the panel is every experiment dataset. Explicit reference names
    override that default, but must remain a duplicate-free subset of the
    datasets actually selected for this experiment.
    """
    selected_names = [str(name) for name in experiment_datasets]
    selected_set = set(selected_names)
    explicit_names = cfg_get(
        cfg,
        "model.gamma_centering.reference.dataset_names",
        None,
    )
    if explicit_names is not None:
        reference_names = [str(name) for name in explicit_names]
    else:
        reference_names = list(selected_names)

    if len(reference_names) != len(set(reference_names)):
        raise ValueError("Duplicate gamma reference dataset names are not allowed.")
    inactive = [name for name in reference_names if name not in selected_set]
    if inactive:
        raise ValueError(
            "Gamma reference datasets must be selected and trained in this "
            f"experiment; inactive names: {inactive}."
        )
    missing = [name for name in reference_names if name not in dataset_encoding]
    if missing:
        raise KeyError(
            f"Gamma reference dataset(s) missing from dataset encoding: {missing}."
        )

    ranking_path = str(cfg_get(cfg, "data.dataset_quality_ranking.path", ""))
    if not ranking_path:
        quality_by_name = {name: 1.0 for name in selected_names}
    else:
        _, quality_by_name = load_dataset_quality_ranking(
            ranking_path,
            dataset_column=str(
                cfg_get(cfg, "data.dataset_quality_ranking.dataset_column", "dataset")
            ),
            rank_column=str(
                cfg_get(
                    cfg,
                    "data.dataset_quality_ranking.rank_column",
                    "quality_rank",
                )
            ),
        )
    strict = cfg_bool(cfg, "data.dataset_quality_ranking.strict", True)
    missing_quality = [name for name in selected_names if name not in quality_by_name]
    if missing_quality and strict:
        raise KeyError(
            "Selected dataset(s) missing from the dataset-quality table: "
            f"{missing_quality}."
        )

    selected_ids = [int(dataset_encoding[name]) for name in selected_names]
    reference_ids = [int(dataset_encoding[name]) for name in reference_names]
    reference_quality = [
        float(quality_by_name.get(name, 1.0)) for name in reference_names
    ]
    return {
        "selected_names": selected_names,
        "selected_ids": selected_ids,
        "reference_names": reference_names,
        "reference_ids": reference_ids,
        "reference_quality": reference_quality,
    }


def choose_checkpoint(
    *,
    checkpoint_callback: ModelCheckpoint,
    ckpt_dir: Path,
    run_checkpoint_root: Path,
    prefer: str = "best",
) -> str | None:
    if checkpoint_callback.best_model_path:
        return checkpoint_callback.best_model_path

    ckpt_found = find_checkpoint(str(ckpt_dir), prefer=prefer)
    if ckpt_found is not None:
        return str(ckpt_found)

    ckpt_found = find_checkpoint(str(run_checkpoint_root), prefer=prefer)
    if ckpt_found is not None:
        return str(ckpt_found)

    return None


PREDICTION_CHECKPOINT_VARIANTS = ("best_val_loss", "best_pcc")


def resolve_prediction_checkpoint_variants(cfg: DictConfig) -> tuple[str, ...]:
    """Return the ordered, unique checkpoint variants requested for prediction."""
    raw_variants = cfg_get(
        cfg,
        "prediction.checkpoint_variants",
        list(PREDICTION_CHECKPOINT_VARIANTS),
    )
    if isinstance(raw_variants, str):
        variants = (raw_variants,)
    else:
        variants = tuple(str(value) for value in raw_variants)
    if not variants:
        raise ValueError("prediction.checkpoint_variants must not be empty.")
    unknown = sorted(set(variants).difference(PREDICTION_CHECKPOINT_VARIANTS))
    if unknown:
        raise ValueError(
            "prediction.checkpoint_variants supports only "
            f"{PREDICTION_CHECKPOINT_VARIANTS}, got {unknown}."
        )
    if len(set(variants)) != len(variants):
        raise ValueError("prediction.checkpoint_variants must not contain duplicates.")
    return variants


def find_prediction_checkpoint(
    checkpoint_root: str | Path,
    variant: str,
) -> str | None:
    """Find one metric-specific checkpoint without mixing PCC and loss files."""
    if variant not in PREDICTION_CHECKPOINT_VARIANTS:
        raise ValueError(f"Unknown prediction checkpoint variant: {variant!r}.")

    root = Path(checkpoint_root)
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.rglob("*.ckpt")
        if path.is_file() and path.name != "last.ckpt"
    ]
    if variant == "best_pcc":
        candidates = [path for path in candidates if path.name.startswith("pcc-")]
        metric_name = "val_mu_pcc"
        select = max
    else:
        candidates = [
            path
            for path in candidates
            if not path.name.startswith("pcc-") and "val_loss" in path.name
        ]
        metric_name = "val_loss"
        select = min
    if not candidates:
        return None

    metric_pattern = re.compile(
        rf"{re.escape(metric_name)}=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    )
    scored: list[tuple[float, Path]] = []
    for path in candidates:
        match = metric_pattern.search(path.name)
        if match is not None:
            scored.append((float(match.group(1)), path))
    if scored:
        return str(select(scored, key=lambda item: item[0])[1])
    return str(max(candidates, key=lambda path: path.stat().st_mtime_ns))


# ============================================================
# Predictions
# ============================================================

def predictions_to_parquet(
    *,
    predictions: Any,
    out_file: Path,
) -> int:
    rows = []

    sequence_keys = {
        "target",
        "likelihood_positive_mean",
        "mu",
        "rho_bio",
        "L_bio",
        "gamma",
        "log_gamma",
        "gamma_raw",
        "log_gamma_raw",
        "gamma_cross_dataset_log_center",
        "gamma_centering_reliability",
        "gamma_centering_eligible",
        "gamma_centering_applied",
        "gamma_num_distinct_datasets",
        "gamma_total_reliability",
        "gamma_centering_constraint_error",
        "normalized_shape",
        "log_sigma",
        "mask",
        "codon_ids",
    }
    bool_sequence_keys = {
        "mask",
        "gamma_centering_eligible",
        "gamma_centering_applied",
    }

    def to_numpy(x):
        if x is None:
            return None

        if torch.is_tensor(x):
            return x.detach().cpu().numpy()

        return x

    def to_python_list(x):
        x = to_numpy(x)

        if x is None:
            return None

        if isinstance(x, np.ndarray):
            return x.tolist()

        if torch.is_tensor(x):
            return x.detach().cpu().tolist()

        return x

    def get_batch_item(x, i: int):
        x = to_numpy(x)

        if x is None:
            return None

        if isinstance(x, (list, tuple)):
            return x[i]

        arr = np.asarray(x)

        if arr.ndim == 0:
            return arr.item()

        return arr[i]

    def slice_sequence(x, key: str, i: int, valid_len: int):
        x = to_numpy(x)

        if x is None:
            return None

        arr = np.asarray(x)

        if arr.ndim == 0:
            return arr.item()

        if arr.ndim == 1:
            if arr.shape[0] == valid_len:
                sliced = arr[:valid_len]
            else:
                item = arr[i]
                return item.item() if np.ndim(item) == 0 else np.asarray(item).tolist()
        else:
            sliced = arr[i, :valid_len]

        if key in bool_sequence_keys:
            return np.asarray(sliced).astype(np.bool_, copy=False).tolist()

        if key == "codon_ids":
            return np.asarray(sliced).astype(np.int64, copy=False).tolist()

        return np.asarray(sliced).astype(np.float32, copy=False).tolist()

    for batch in flatten_prediction_batches(predictions):
        batch_size = len(batch["ids"])

        for i in range(batch_size):
            transcript_id = str(batch["ids"][i])
            dataset_id = int(get_batch_item(batch["dataset_id"], i))
            valid_len = int(get_batch_item(batch["lengths"], i))

            row = {
                "transcript_id": transcript_id,
                "ids": transcript_id,
                "dataset_id": dataset_id,
                "length": valid_len,
                "lengths": valid_len,
                "css": to_python_list(batch["css"][i]) if "css" in batch else None,
            }

            for key, val in batch.items():
                if key in {"ids", "dataset_id", "lengths", "css"}:
                    continue

                val_np = to_numpy(val)

                if val_np is None:
                    continue

                if key in sequence_keys:
                    row[key] = slice_sequence(val_np, key, i, valid_len)
                    continue

                arr = np.asarray(val_np)

                if arr.ndim == 0:
                    row[key] = arr.item()

                elif arr.ndim == 1:
                    if arr.shape[0] == batch_size:
                        item = arr[i]
                        row[key] = item.item() if np.ndim(item) == 0 else np.asarray(item).tolist()
                    else:
                        row[key] = arr.astype(np.float32, copy=False).tolist()

                else:
                    item = np.asarray(arr[i])

                    if item.size == 1:
                        row[key] = item.reshape(-1)[0].item()
                    else:
                        row[key] = item.astype(np.float32, copy=False).tolist()

            rows.append(row)

    df_predictions = pd.DataFrame(rows)

    print(f"Saving {len(df_predictions)} prediction rows to {out_file}...")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_predictions.to_parquet(out_file, engine="pyarrow", index=False)
    return len(df_predictions)


def export_sequence_only_shared_profiles(
    *,
    prediction_path: Path,
    output_path: Path,
    expected_transcript_ids: Sequence[str],
    run_id: str,
    dataset_count: int | None,
    subset_identifier: str | None,
    mean_one_tolerance: float = 1.0e-4,
) -> int:
    """Export exactly one directly predicted L_t vector per held-out sequence."""
    frame = pd.read_parquet(
        prediction_path,
        columns=["transcript_id", "length", "mask", "L_bio"],
    )
    frame["transcript_id"] = frame["transcript_id"].astype(str)
    expected = list(map(str, expected_transcript_ids))
    if len(expected) != len(set(expected)):
        raise ValueError("Sequence-only prediction IDs contain duplicates.")
    observed = frame["transcript_id"].tolist()
    if len(observed) != len(set(observed)):
        raise ValueError(
            "Sequence-only shared-profile prediction must contain exactly one "
            "row per transcript."
        )
    if set(observed) != set(expected):
        raise ValueError(
            "Sequence-only shared-profile transcript mismatch: "
            f"missing={sorted(set(expected) - set(observed))[:10]}, "
            f"extra={sorted(set(observed) - set(expected))[:10]}."
        )
    indexed = frame.set_index("transcript_id")
    rows: list[dict[str, Any]] = []
    for transcript_id in expected:
        row = indexed.loc[transcript_id]
        values = np.asarray(row["L_bio"], dtype=np.float64)
        mask = np.asarray(row["mask"], dtype=bool)
        length = int(row["length"])
        if values.ndim != 1 or mask.ndim != 1 or values.shape != mask.shape:
            raise ValueError(f"Invalid L_t/mask shape for {transcript_id}.")
        if length != int(mask.sum()) or length <= 0:
            raise ValueError(f"Invalid length/mask for {transcript_id}.")
        valid = values[mask]
        if not np.isfinite(valid).all() or np.any(valid <= 0.0):
            raise ValueError(f"Non-finite or non-positive L_t for {transcript_id}.")
        mean = float(valid.mean())
        if abs(mean - 1.0) > float(mean_one_tolerance):
            raise ValueError(
                f"L_t is not mean-one for {transcript_id}: mean={mean:.8g}."
            )
        rows.append(
            {
                "transcript_id": transcript_id,
                "transcript_length": length,
                "L_t": values.astype(np.float32),
                "valid_position_mask": mask,
                "run_id": str(run_id),
                "N": dataset_count,
                "subset_identifier": subset_identifier,
                "L_mean": mean,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, engine="pyarrow", index=False)
    return len(rows)


def flatten_prediction_batches(predictions: Any) -> list[dict[str, Any]]:
    """Flatten Lightning prediction outputs from single-process or DDP strategies."""
    flat: list[dict[str, Any]] = []

    def visit(obj: Any) -> None:
        if obj is None:
            return

        if isinstance(obj, dict):
            flat.append(obj)
            return

        if isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
            return

        raise TypeError(
            "Unexpected prediction output type "
            f"{type(obj).__name__}; expected dict/list/tuple/None."
        )

    visit(predictions)
    return flat


def trainer_barrier(trainer: pl.Trainer, name: str) -> None:
    barrier = getattr(getattr(trainer, "strategy", None), "barrier", None)
    if callable(barrier):
        barrier(name)


def save_predictions_for_trainer(
    *,
    predictions: Any,
    out_file: Path,
    trainer: pl.Trainer,
) -> int:
    """Save predictions safely for both single-process and DDP prediction."""
    world_size = int(getattr(trainer, "world_size", 1) or 1)
    rank = int(getattr(trainer, "global_rank", 0) or 0)
    is_global_zero = bool(getattr(trainer, "is_global_zero", True))

    if world_size <= 1:
        return predictions_to_parquet(predictions=predictions, out_file=out_file)

    tmp_dir = out_file.parent / f".{out_file.stem}_ddp_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rank_file = tmp_dir / f"rank_{rank:05d}.parquet"

    local_rows = predictions_to_parquet(predictions=predictions, out_file=rank_file)
    trainer_barrier(trainer, f"prediction-write-{out_file.stem}")

    total_rows = 0
    if is_global_zero:
        part_files = sorted(tmp_dir.glob("rank_*.parquet"))
        frames = [
            pd.read_parquet(part_file)
            for part_file in part_files
            if part_file.exists()
        ]

        if frames:
            df_predictions = pd.concat(frames, ignore_index=True)
        else:
            df_predictions = pd.DataFrame()

        if {"transcript_id", "dataset_id"}.issubset(df_predictions.columns):
            before = len(df_predictions)
            df_predictions = df_predictions.drop_duplicates(
                subset=["transcript_id", "dataset_id"],
                keep="first",
            )
            dropped = before - len(df_predictions)
            if dropped > 0:
                print(f"Dropped {dropped} duplicate prediction rows while merging DDP shards.")

        total_rows = len(df_predictions)
        print(f"Saving {total_rows} merged prediction rows to {out_file}...")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df_predictions.to_parquet(out_file, engine="pyarrow", index=False)

        for part_file in part_files:
            part_file.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    trainer_barrier(trainer, f"prediction-merge-{out_file.stem}")
    return total_rows if is_global_zero else local_rows


# ============================================================
# Datamodule factory
# ============================================================

def make_datamodule(
    *,
    cfg: DictConfig,
    datasets_paths: list[str],
    train_fold: list[str],
    val_fold: list[str],
    seed: int,
    predict_fold: list[str] | None = None,
) -> RiboAIQueuingDatamoduleMultiDataset:
    feature_cfg = cfg_get(cfg, "model.additional_sequence_features", {})
    if OmegaConf.is_config(feature_cfg):
        feature_cfg = OmegaConf.to_container(feature_cfg, resolve=True)
    else:
        feature_cfg = dict(feature_cfg or {})
    execution_enabled = cfg_bool(
        cfg, "training.execution_microbatching.enabled", False
    )
    raw_execution_groups = cfg_get(
        cfg,
        "training.execution_microbatching.max_transcript_groups_per_forward",
        None,
    )
    raw_execution_pairs = cfg_get(
        cfg,
        "training.execution_microbatching.max_pair_rows_per_forward",
        None,
    )
    raw_execution_tokens = cfg_get(
        cfg,
        "training.execution_microbatching.max_padded_codon_tokens_per_forward",
        None,
    )
    execution_groups = (
        int(raw_execution_groups)
        if execution_enabled and raw_execution_groups is not None
        else None
    )
    execution_pairs = (
        int(raw_execution_pairs)
        if execution_enabled and raw_execution_pairs is not None
        else None
    )
    execution_tokens = (
        int(raw_execution_tokens)
        if execution_enabled and raw_execution_tokens is not None
        else None
    )
    if (
        execution_enabled
        and execution_groups is None
        and execution_pairs is None
        and execution_tokens is None
    ):
        raise ValueError(
            "training.execution_microbatching.enabled=true requires at least one "
            "execution-forward limit."
        )
    return RiboAIQueuingDatamoduleMultiDataset(
        sequences_path=cfg.paths.sequences_path,
        datasets_paths=datasets_paths,
        batch_size=int(cfg.data.batch_size),
        split=(
            (train_fold, val_fold)
            if predict_fold is None
            else (train_fold, val_fold, predict_fold)
        ),
        num_workers=int(cfg.data.num_workers),
        predict_num_workers=int(cfg_get(cfg, "data.predict_num_workers", 0)),
        seed=seed,
        nt_encoding_path=cfg.paths.encodings.nt,
        codon_encoding_path=cfg.paths.encodings.codon,
        codon_to_aa_encoding_path=cfg.paths.encodings.codon_to_aa,
        aa_encoding_path=cfg.paths.encodings.aa,
        datasets_encoding_path=cfg.paths.encodings.datasets,
        train_sampling_strategy=cfg_get(
            cfg,
            "data.train_sampling_strategy",
            "transcript_grouped_multidataset_pairs",
        ),
        minimum_positive_datasets_per_transcript=int(
            cfg_get(cfg, "data.minimum_positive_datasets_per_transcript", 2)
        ),
        max_cds_codons=cfg_optional_positive_int(cfg, "data.max_cds_codons"),
        pin_memory=cfg_bool(cfg, "data.pin_memory", True),
        prefetch_factor=cfg_get(cfg, "data.prefetch_factor", 4),
        multiprocessing_context=cfg_get(
            cfg, "data.multiprocessing_context", "spawn"
        ),
        additional_sequence_features=feature_cfg,
        dataset_quality_ranking_path=cfg_get(
            cfg, "data.dataset_quality_ranking.path", None
        ),
        dataset_quality_dataset_column=str(
            cfg_get(cfg, "data.dataset_quality_ranking.dataset_column", "dataset")
        ),
        dataset_quality_rank_column=str(
            cfg_get(cfg, "data.dataset_quality_ranking.rank_column", "quality_rank")
        ),
        dataset_quality_strict=cfg_bool(
            cfg, "data.dataset_quality_ranking.strict", True
        ),
        reliability_reference_manifest_path=cfg_get(
            cfg, "data.reliability_reference_manifest", None
        ),
        sequence_only_shared_profile_prediction=cfg_bool(
            cfg, "prediction.sequence_only_shared_profile", False
        ),
        execution_microbatch_max_transcript_groups=execution_groups,
        execution_microbatch_max_pair_rows=execution_pairs,
        execution_microbatch_max_padded_codon_tokens=execution_tokens,
    )


def configured_trainer_world_size(cfg: DictConfig) -> int:
    """Infer the common pre-Trainer DDP world size from resolved config."""
    devices = cfg_get(cfg, "trainer.devices", 1)
    if isinstance(devices, (list, tuple, ListConfig)):
        devices_per_node = max(len(devices), 1)
    else:
        try:
            devices_per_node = max(int(devices), 1)
        except (TypeError, ValueError):
            devices_per_node = 1
    num_nodes = max(int(cfg_get(cfg, "trainer.num_nodes", 1)), 1)
    return devices_per_node * num_nodes


def resolve_training_grouped_optimizer_batching(
    *,
    cfg: DictConfig,
    datamodule: RiboAIQueuingDatamoduleMultiDataset,
) -> tuple[GroupedBatchStatistics | None, GroupedOptimizerBatchPlan | None]:
    """Resolve accumulation before Trainer/optimizer/scheduler construction."""
    grouped_cfg_path = "training.grouped_optimizer_batch"
    enabled = cfg_bool(cfg, f"{grouped_cfg_path}.enabled", False)
    strategy = str(cfg_get(cfg, "data.train_sampling_strategy", "")).lower()
    grouped_strategies = {
        "transcript_grouped_pairs",
        "transcript_grouped_multidataset_pairs",
    }
    if not enabled or strategy not in grouped_strategies:
        return None, None

    datamodule.setup("fit")
    statistics = datamodule.preview_train_grouped_batch_statistics(iteration_index=0)
    world_size = configured_trainer_world_size(cfg)
    auto = cfg_bool(
        cfg,
        f"{grouped_cfg_path}.auto_accumulate_grad_batches",
        True,
    )
    configured_accumulation = int(
        cfg_get(cfg, "trainer.accumulate_grad_batches", 1)
    )
    if auto and configured_accumulation != 1:
        raise ValueError(
            "Automatic grouped optimizer batching conflicts with explicit "
            f"trainer.accumulate_grad_batches={configured_accumulation}. Set it "
            "to 1, or set training.grouped_optimizer_batch."
            "auto_accumulate_grad_batches=false to keep the explicit value."
        )

    plan = resolve_grouped_optimizer_batch_plan(
        statistics,
        target_unique_transcripts_per_optimizer_step=int(
            cfg_get(
                cfg,
                f"{grouped_cfg_path}.target_unique_transcripts_per_optimizer_step",
                32,
            )
        ),
        accumulation_statistic=str(
            cfg_get(cfg, f"{grouped_cfg_path}.accumulation_statistic", "median")
        ),
        max_accumulate_grad_batches=int(
            cfg_get(cfg, f"{grouped_cfg_path}.max_accumulate_grad_batches", 32)
        ),
        target_scope=str(
            cfg_get(cfg, f"{grouped_cfg_path}.target_scope", "per_rank")
        ),
        world_size=world_size,
        forced_accumulate_grad_batches=(None if auto else configured_accumulation),
    )
    execution_microbatching = cfg_bool(
        cfg,
        "training.execution_microbatching.enabled",
        False,
    )
    if execution_microbatching and world_size != 1:
        raise ValueError(
            "training.execution_microbatching is single-process only. Launch one "
            "independent experiment per GPU with trainer.devices=[0]."
        )
    OmegaConf.update(
        cfg,
        "trainer.accumulate_grad_batches",
        1 if execution_microbatching else int(plan.resolved_accumulate_grad_batches),
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        f"{grouped_cfg_path}.resolved",
        {
            **plan.to_dict(),
            "batch_statistics": statistics.to_dict(),
        },
        merge=False,
        force_add=True,
    )
    return statistics, plan


def print_grouped_optimizer_batch_plan(
    *,
    selected_datasets: Sequence[str],
    statistics: GroupedBatchStatistics,
    plan: GroupedOptimizerBatchPlan,
) -> None:
    pair_rows = np.asarray(statistics.pair_rows_per_microbatch, dtype=float)
    unique_transcripts = np.asarray(
        statistics.unique_transcripts_per_microbatch,
        dtype=float,
    )
    unique_transcript_cv = (
        float(unique_transcripts.std() / unique_transcripts.mean())
        if unique_transcripts.size and unique_transcripts.mean() > 0.0
        else 0.0
    )

    def range_text(values: np.ndarray) -> str:
        if values.size == 0:
            return "n/a"
        return (
            f"{values.min():.0f} / {np.median(values):.1f} / "
            f"{values.mean():.2f} / {values.max():.0f}"
        )

    print("\n=== Group-aware optimizer batch plan ===")
    rows = (
        ("selected datasets", str(len(selected_datasets))),
        ("transcripts considered", str(statistics.transcripts_considered)),
        ("positive-support K=0", str(statistics.transcripts_with_positive_k0)),
        ("positive-support K=1", str(statistics.transcripts_with_positive_k1)),
        (
            "positive-support K>=2",
            str(statistics.transcripts_with_positive_k2_or_more),
        ),
        ("transcripts admitted", str(statistics.transcripts_admitted)),
        (
            "transcripts excluded for support",
            str(statistics.transcripts_excluded_for_insufficient_support),
        ),
        ("admitted positive pair rows", str(statistics.positive_pair_rows)),
        (
            "per-dataset pair capacity",
            str(statistics.per_dataset_pair_capacity),
        ),
        (
            "nominal total pair capacity/rank",
            str(statistics.physical_pair_capacity),
        ),
        (
            "group-size min / median / mean / max",
            f"{statistics.minimum_group_size} / {statistics.median_group_size:.1f} / "
            f"{statistics.mean_group_size:.2f} / {statistics.maximum_group_size}",
        ),
        ("pair rows/logical batch min/median/mean/max", range_text(pair_rows)),
        (
            "unique transcripts/logical batch min/median/mean/max",
            range_text(unique_transcripts),
        ),
        ("unique transcripts/logical batch CV", f"{unique_transcript_cv:.6f}"),
        (
            "configured target transcripts/update",
            str(plan.configured_target_unique_transcripts),
        ),
        (
            "effective local target transcripts/update",
            str(plan.effective_local_target_unique_transcripts),
        ),
        ("target scope / world size", f"{plan.target_scope} / {plan.world_size}"),
        ("resolved accumulation factor", str(plan.resolved_accumulate_grad_batches)),
        (
            "estimated local transcripts/update",
            f"{plan.estimated_unique_transcripts_per_optimizer_step:.2f}",
        ),
        (
            "estimated local pair rows/update",
            f"{plan.estimated_pair_rows_per_optimizer_step:.2f}",
        ),
        (
            "estimated global pair rows/update",
            f"{plan.estimated_global_pair_rows_per_optimizer_step:.2f}",
        ),
        (
            "estimated logical batches/epoch/rank",
            str(plan.microbatches_per_epoch_per_rank),
        ),
        (
            "estimated optimizer steps/epoch",
            str(plan.estimated_optimizer_steps_per_epoch),
        ),
        (
            "steps vs transcript-target expectation",
            f"{plan.estimated_optimizer_steps_per_epoch} vs "
            f"{plan.expected_optimizer_steps_from_transcript_target} "
            f"(ratio={plan.optimizer_step_expectation_ratio:.3f})",
        ),
    )
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}} : {value}")
    print(
        "The accumulation factor counts logical grouped batches, not execution "
        "chunks. Execution chunks reconstruct each transcript-balanced logical-"
        "batch mean with group-count scaling. When the factor is greater than "
        "one, logical-batch means are averaged; exact equal weighting across the "
        "whole optimizer window additionally requires equal logical group counts."
    )


def save_grouped_optimizer_batch_manifest(
    *,
    output_path: Path,
    statistics: GroupedBatchStatistics,
    plan: GroupedOptimizerBatchPlan,
    selected_datasets: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_datasets": list(map(str, selected_datasets)),
        "plan": plan.to_dict(),
        "batch_statistics": statistics.to_dict(),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config_riboai_queuing_multidataset",
)
def main(cfg: DictConfig) -> None:
    sample_reduction = resolve_sample_reduction_mode(cfg.loss)
    seed = int(cfg.experiment.seed)

    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)

    # ------------------------------------------------------------
    # Experiment datasets vs split-universe datasets
    # ------------------------------------------------------------
    # experiment_datasets:
    #     actual datasets used for training/validation/prediction.
    #
    # split_universe_datasets:
    #     master universe used both for the complete transcript ID pool and for
    #     the common-transcript validation candidate set. This makes validation
    #     IDs fixed across experiment.dataset subset ablations.
    # ------------------------------------------------------------
    experiment_datasets = get_datasets(cfg)
    split_dataset_mapping = dict(cfg.dataset_config.dataset_path)
    external_split_manifest = cfg_get(cfg, "split.external_manifest", None)
    external_split_manifest = (
        None
        if external_split_manifest is None or not str(external_split_manifest).strip()
        else str(external_split_manifest)
    )
    external_panel_name = cfg_get(cfg, "split.external_panel_name", None)
    if external_split_manifest is not None:
        if external_panel_name is None or not str(external_panel_name).strip():
            raise ValueError(
                "split.external_panel_name is required when "
                "split.external_manifest is set."
            )
        external_panel_name = str(external_panel_name)
        # The external manifest already fixed the cross-panel transcript
        # universe. Avoid rebuilding the legacy all-dataset common split.
        split_universe_datasets = list(experiment_datasets)
        split_dataset_source_label = (
            f"external manifest panel {external_panel_name}"
        )
    else:
        split_dataset_source_label = "active dataset_config"
        split_universe_datasets = get_split_universe_datasets(
            cfg,
            experiment_datasets,
        )

    experiment_dataset_paths = dataset_paths_for(
        mapping=dict(cfg.dataset_config.dataset_path),
        datasets=experiment_datasets,
        source_label="active dataset_config",
    )

    split_universe_dataset_paths = dataset_paths_for(
        mapping=split_dataset_mapping,
        datasets=split_universe_datasets,
        source_label=split_dataset_source_label,
    )

    datasets_outside_split_universe = sorted(
        set(experiment_datasets) - set(split_universe_datasets)
    )
    if datasets_outside_split_universe:
        raise ValueError(
            "Every experiment dataset must belong to split.master_dataset_universe "
            "so the fixed validation panel is guaranteed to be present. Missing: "
            f"{datasets_outside_split_universe}."
        )

    print("\n=== Dataset configuration ===")
    print(f"Experiment datasets:     {experiment_datasets}")
    print(f"Split universe datasets: {split_universe_datasets}")
    print(f"Training dataset config: active dataset_config")
    print(f"Split dataset source:    {split_dataset_source_label}")

    # ------------------------------------------------------------
    # Split strategy
    # ------------------------------------------------------------
    # Validation is one deterministic panel drawn from transcripts retained in
    # every master-universe dataset. Reliability rank and CSS-count bin jointly
    # stratify the sample; CSS has no reserved quota and no separate split.
    # Every other master-union transcript is assigned to training. Individual
    # datasets naturally contribute only the training IDs that they contain.
    # ------------------------------------------------------------
    validation_frac = float(cfg_get(cfg, "split.validation_frac", 0.10))
    validation_weight_bins = int(cfg_get(cfg, "split.validation_weight_bins", 10))
    max_cds_codons = cfg_optional_positive_int(cfg, "data.max_cds_codons")
    external_split_payload: dict[str, Any] | None = None
    if external_split_manifest is not None:
        print("\n=== External transcript split configuration ===")
        print(f"manifest:                 {external_split_manifest}")
        print(f"panel:                    {external_panel_name}")
        print(f"maximum eligible CDS:     {max_cds_codons}")
        (
            train_fold,
            validation_fold,
            test_fold,
            external_split_payload,
        ) = load_external_transcript_split(
            external_split_manifest,
            panel_name=str(external_panel_name),
            experiment_datasets=experiment_datasets,
        )
        split_metadata = None
        print(f"training transcripts:     {len(train_fold):,}")
        print(f"common validation IDs:    {len(validation_fold):,}")
        print(f"common test IDs:          {len(test_fold):,}")
    else:
        print("\n=== Fixed split configuration ===")
        print(f"validation fraction of common candidates: {validation_frac}")
        print(f"validation reliability bins:              {validation_weight_bins}")
        print(f"maximum eligible CDS codons:              {max_cds_codons}")
        train_fold, validation_fold, split_metadata = fixed_common_validation_split(
            sequences_path=cfg.paths.sequences_path,
            split_universe_dataset_paths=split_universe_dataset_paths,
            validation_frac=validation_frac,
            random_seed=seed,
            validation_weight_bins=validation_weight_bins,
            max_cds_codons=max_cds_codons,
        )
        test_fold = []
    if len(validation_fold) == 0:
        raise RuntimeError("Validation split is empty.")

    dataset_str = make_dataset_signature(experiment_datasets)
    split_universe_str = make_dataset_signature(split_universe_datasets)
    run_tag = make_run_tag(cfg)

    print(f"\nTracking dataset signature: {dataset_str}")
    print(f"Split universe signature:   {split_universe_str}")
    print(f"Run tag:                    {run_tag}")

    paths_logs = Path(cfg.paths.logs) / dataset_str / run_tag
    paths_results = Path(cfg.paths.results) / dataset_str / run_tag
    paths_checkpoints = Path(cfg.paths.checkpoints) / dataset_str / run_tag

    if env_global_rank() == 0:
        paths_results.mkdir(parents=True, exist_ok=True)
        (paths_results / "loss_reduction_manifest.json").write_text(
            json.dumps(
                {
                    "experiment_mode": str(
                        cfg_get(
                            cfg,
                            "loss.experiment_mode",
                            "mean_gradient_reweighted_nb",
                        )
                    ),
                    "alpha_mode": str(cfg_get(cfg, "model.alpha_mode", "learned")),
                    "fixed_alpha": float(cfg_get(cfg, "model.fixed_alpha", 0.1)),
                    "nb_mean_gradient_beta": float(
                        cfg_get(cfg, "loss.nb_mean_gradient_beta", 0.0)
                    ),
                    "sample_reduction": sample_reduction,
                    "available_sample_reductions": [
                        "global_weighted",
                        "dataset_balanced",
                        "transcript_balanced",
                    ],
                    "dataset_quality_rank_in_loss": False,
                    "pair_objective": {
                        "replica_nb": {
                            "target": "raw_replicas",
                            "within_pair_reduction": "arithmetic_mean",
                            "weight": float(cfg.loss.replica_nb_weight),
                        },
                        "raw_pcc": {
                            "target": "arithmetic_replica_consensus",
                            "within_pair_reduction": "once_per_pair",
                            "weight": float(cfg.loss.consensus_raw_pcc_weight),
                        },
                        "nb_vst_pcc": {
                            "target": "arithmetic_replica_consensus",
                            "within_pair_reduction": "once_per_pair",
                            "weight": float(
                                cfg.loss.consensus_nb_vst_pcc_weight
                            ),
                        },
                        "gamma_regularization_weight": float(
                            cfg.loss.gamma_reg_weight
                        ),
                    },
                    "checkpoint_monitors": {
                        "best_val_loss": {"metric": "val_loss", "mode": "min"},
                        **(
                            {
                                "best_pcc": {
                                    "metric": "val_mu_pcc",
                                    "mode": "max",
                                }
                            }
                            if cfg_bool(
                                cfg,
                                "callbacks.save_best_pcc_checkpoint",
                                True,
                            )
                            else {}
                        ),
                    },
                    "prediction_checkpoint_variants": list(
                        resolve_prediction_checkpoint_variants(cfg)
                    ),
                    "val_loss_reduction": sample_reduction,
                    "train_sampling_strategy": str(
                        cfg_get(cfg, "data.train_sampling_strategy", "unknown")
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    # Save split provenance once. Under Slurm+srun, every rank executes this
    # script before the Lightning Trainer exists, so use the environment rank
    # guard rather than trainer.is_global_zero.
    if env_global_rank() == 0:
        if external_split_manifest is not None:
            source_bytes = Path(external_split_manifest).read_bytes()
            panel_support_statistics = dict(
                (external_split_payload or {}).get(
                    "panel_support_statistics", {}
                )
            ).get(str(external_panel_name), {})
            external_run_manifest = {
                "split_strategy": "external_common_validation_and_test",
                "source_manifest": str(Path(external_split_manifest).resolve()),
                "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "panel_name": str(external_panel_name),
                "experiment_datasets": list(map(str, experiment_datasets)),
                "training_dataset_paths": list(map(str, experiment_dataset_paths)),
                "seed": seed,
                "maximum_cds_codons": max_cds_codons,
                "minimum_positive_datasets_per_training_transcript": int(
                    cfg_get(cfg, "data.minimum_positive_datasets_per_transcript", 2)
                ),
                "train_ids": list(map(str, train_fold)),
                "validation_ids": list(map(str, validation_fold)),
                "test_ids": list(map(str, test_fold)),
                "fold_id_hashes": {
                    "train": transcript_id_hash(train_fold),
                    "validation": transcript_id_hash(validation_fold),
                    "test": transcript_id_hash(test_fold),
                },
                "fold_sizes": {
                    "train": len(train_fold),
                    "validation": len(validation_fold),
                    "test": len(test_fold),
                },
                "panel_support_statistics": panel_support_statistics,
                "heldout_train_overlap": 0,
            }
            (paths_results / "split_manifest.json").write_text(
                json.dumps(external_run_manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            save_split_manifest(
                out_file=paths_results / f"split_manifest_experiment_{dataset_str}_universe_{split_universe_str}.json",
                experiment_datasets=experiment_datasets,
                split_universe_datasets=split_universe_datasets,
                split_dataset_source=split_dataset_source_label,
                split_dataset_paths=split_universe_dataset_paths,
                training_dataset_paths=experiment_dataset_paths,
                validation_weight_bins=validation_weight_bins,
                train_ids=train_fold,
                validation_ids=validation_fold,
                metadata=split_metadata,
                seed=seed,
                validation_frac=validation_frac,
                max_cds_codons=max_cds_codons,
            )

    dataset_encoding = open_file(cfg.paths.encodings.datasets)
    missing_dataset_encodings = [
        dataset for dataset in experiment_datasets if dataset not in dataset_encoding
    ]
    if missing_dataset_encodings:
        raise KeyError(
            "Experiment dataset(s) missing from dataset encoding: "
            f"{missing_dataset_encodings}"
        )
    gamma_reference_panel = resolve_gamma_reference_panel(
        cfg=cfg,
        experiment_datasets=experiment_datasets,
        dataset_encoding=dataset_encoding,
    )
    nt_encoding = open_file(cfg.paths.encodings.nt)
    codon_encoding = open_file(cfg.paths.encodings.codon)
    codon_to_aa_encoding = open_file(cfg.paths.encodings.codon_to_aa)
    aa_encoding = open_file(cfg.paths.encodings.aa)
    torch_model = RiboQueuingModel(
        model_configs=cfg.model,
        eps=float(cfg.model.get("eps", 1e-8)),
        selected_dataset_names=gamma_reference_panel["selected_names"],
        selected_dataset_ids=gamma_reference_panel["selected_ids"],
        reference_dataset_names=gamma_reference_panel["reference_names"],
        reference_dataset_ids=gamma_reference_panel["reference_ids"],
        reference_dataset_quality_weights=gamma_reference_panel["reference_quality"],
        nt_encoding=nt_encoding,
        codon_to_aa_encoding=codon_to_aa_encoding,
        codon_encoding=codon_encoding,
        aa_encoding=aa_encoding,
    )
    print(
        "Gamma centering: "
        f"mode={torch_model.gamma_centering_mode}, "
        f"reference_count={torch_model.gamma_reference_dataset_ids.numel()}, "
        f"manifest={torch_model.gamma_reference_manifest_hash[:12]}"
    )
    print(
        "Bias context GRU precision: "
        f"{torch_model.dataset_bias_model.local_context_gru.precision}; "
        f"Trainer precision: {cfg.trainer.precision}. "
        "FP32 context, when selected, includes embeddings and LayerNorm; "
        "observation/alpha heads keep Trainer precision."
    )
    if env_global_rank() == 0:
        gamma_raw_weights = (
            torch_model.gamma_reference_weights.detach().cpu().to(torch.float64)
        )
        gamma_pi = gamma_raw_weights / gamma_raw_weights.sum()
        gamma_manifest = {
            "centering_mode": str(torch_model.gamma_centering_mode),
            "dataset_constant_scale_gauge": str(
                torch_model.gamma_dataset_constant_scale_gauge
            ),
            "selected_dataset_names": list(
                map(str, gamma_reference_panel["selected_names"])
            ),
            "selected_dataset_ids": list(
                map(int, gamma_reference_panel["selected_ids"])
            ),
            "reference_dataset_names": list(
                map(str, gamma_reference_panel["reference_names"])
            ),
            "reference_dataset_ids": list(
                map(int, gamma_reference_panel["reference_ids"])
            ),
            "reference_raw_weights": [
                float(value) for value in gamma_raw_weights.tolist()
            ],
            "reference_pi": [float(value) for value in gamma_pi.tolist()],
            "weighting": str(torch_model.gamma_centering_weighting),
            "quality_rank_power": float(
                torch_model.gamma_centering_quality_rank_power
            ),
            "reference_manifest_hash": str(
                torch_model.gamma_reference_manifest_hash
            ),
            "pi_is_gamma_reference_only": True,
            "w_dt_source": (
                str(cfg_get(cfg, "data.reliability_reference_manifest", None))
                if cfg_get(cfg, "data.reliability_reference_manifest", None)
                else "input_parquet_weight"
            ),
        }
        (paths_results / "gamma_reference_manifest.json").write_text(
            json.dumps(gamma_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    lit_model = RiboQueuingModelLightningModule(
        torch_model,
        config=cfg,
        dataset_encoding=dataset_encoding,
    )

    # Actual datamodules use the experiment datasets but receive the fixed
    # master-universe split IDs. Validation IDs are present in every selected
    # dataset. Training is the remaining ID pool, intersected independently
    # with the rows available in each selected dataset by the dataloader.
    datamodule = make_datamodule(
        cfg=cfg,
        datasets_paths=experiment_dataset_paths,
        train_fold=train_fold,
        val_fold=validation_fold,
        seed=seed,
        predict_fold=(test_fold if test_fold else None),
    )

    grouped_batch_statistics, grouped_optimizer_plan = (
        resolve_training_grouped_optimizer_batching(
            cfg=cfg,
            datamodule=datamodule,
        )
    )
    if grouped_batch_statistics is not None and grouped_optimizer_plan is not None:
        if (
            external_split_manifest is not None
            and grouped_batch_statistics.transcripts_excluded_for_insufficient_support
            != 0
        ):
            raise RuntimeError(
                "External panel manifest violated its training-support contract: "
                f"{grouped_batch_statistics.transcripts_excluded_for_insufficient_support} "
                "provided training transcripts have fewer than the configured "
                "number of usable selected datasets."
            )
        print_grouped_optimizer_batch_plan(
            selected_datasets=experiment_datasets,
            statistics=grouped_batch_statistics,
            plan=grouped_optimizer_plan,
        )
        if env_global_rank() == 0:
            save_grouped_optimizer_batch_manifest(
                output_path=paths_results / "grouped_optimizer_batch_plan.json",
                statistics=grouped_batch_statistics,
                plan=grouped_optimizer_plan,
                selected_datasets=experiment_datasets,
            )
        lit_model.configure_grouped_optimizer_batch_logging(
            plan={
                **grouped_optimizer_plan.to_dict(),
                "batch_support_statistics": {
                    "transcripts_considered": (
                        grouped_batch_statistics.transcripts_considered
                    ),
                    "transcripts_with_positive_k0": (
                        grouped_batch_statistics.transcripts_with_positive_k0
                    ),
                    "transcripts_with_positive_k1": (
                        grouped_batch_statistics.transcripts_with_positive_k1
                    ),
                    "transcripts_with_positive_k2_or_more": (
                        grouped_batch_statistics.transcripts_with_positive_k2_or_more
                    ),
                    "transcripts_admitted": (
                        grouped_batch_statistics.transcripts_admitted
                    ),
                    "transcripts_excluded_for_insufficient_support": (
                        grouped_batch_statistics.transcripts_excluded_for_insufficient_support
                    ),
                    "positive_pair_rows": grouped_batch_statistics.positive_pair_rows,
                    "group_size_min": grouped_batch_statistics.minimum_group_size,
                    "group_size_median": grouped_batch_statistics.median_group_size,
                    "group_size_mean": grouped_batch_statistics.mean_group_size,
                    "group_size_max": grouped_batch_statistics.maximum_group_size,
                },
            },
            enabled=cfg_bool(
                cfg,
                "training.grouped_optimizer_batch.log_batch_structure",
                True,
            ),
        )

    logger_version = shared_logger_version_from_environment()
    tb_logger = TensorBoardLogger(
        save_dir=str(paths_logs),
        name="",
        version=logger_version,
    )
    exp_name = Path(tb_logger.log_dir or tb_logger.save_dir).name

    # Save full resolved config next to the TensorBoard events file so each
    # version_N directory is self-contained and traceable without Hydra outputs.
    tb_log_dir = Path(tb_logger.log_dir)
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    if env_global_rank() == 0:
        OmegaConf.save(cfg, tb_log_dir / "config.yaml")

    ckpt_dir = paths_checkpoints / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    monitor = str(cfg.optim.scheduler.monitor)
    metric_mode = str(cfg.optim.scheduler.mode)

    val_loss_checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="val-loss-{epoch}-{val_loss:.4f}",
        save_top_k=1,
        save_last=True,
        # Full-state checkpoints are required to continue long jobs across
        # scheduler wall-time limits without resetting Adam/scheduler state.
        # Prediction still loads only the state_dict from these files.
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
    )

    # Keep a second, independently selected checkpoint. Validation loss follows
    # the configured optimized objective; val_mu_pcc is the unweighted
    # consensus-profile PCC diagnostic.
    save_best_pcc_checkpoint = cfg_bool(
        cfg,
        "callbacks.save_best_pcc_checkpoint",
        True,
    )
    pcc_checkpoint_callback = (
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="pcc-{epoch}-{val_mu_pcc:.4f}",
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
            monitor="val_mu_pcc",
            mode="max",
        )
        if save_best_pcc_checkpoint
        else None
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=int(cfg.callbacks.early_stopping_patience),
        mode=metric_mode,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    devices_cfg = cfg.trainer.devices
    if isinstance(devices_cfg, (list, tuple, ListConfig)):
        n_devices = len(devices_cfg)
    else:
        try:
            n_devices = int(devices_cfg)
        except (TypeError, ValueError):
            n_devices = 1

    trainer_kwargs = {
        "accelerator": cfg.trainer.accelerator,
        "devices": devices_cfg,
        "num_nodes": int(cfg_get(cfg, "trainer.num_nodes", 1)),
        "precision": cfg.trainer.precision,
        "detect_anomaly": cfg_bool(cfg, "trainer.detect_anomaly", False),
        "max_epochs": int(cfg.trainer.max_epochs),
        "num_sanity_val_steps": int(
            cfg_get(cfg, "trainer.num_sanity_val_steps", 2)
        ),
        "logger": tb_logger,
        "log_every_n_steps": int(cfg.trainer.log_every_n_steps),
        "accumulate_grad_batches": int(cfg_get(cfg, "trainer.accumulate_grad_batches", 1)),
        "callbacks": [
            callback
            for callback in (
                val_loss_checkpoint_callback,
                pcc_checkpoint_callback,
                early_stopping,
                lr_monitor,
            )
            if callback is not None
        ],
        "use_distributed_sampler": cfg_bool(
            cfg,
            "trainer.use_distributed_sampler",
            False,
        ),
    }
    if n_devices > 1:
        trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"

    execution_microbatching = cfg_bool(
        cfg,
        "training.execution_microbatching.enabled",
        False,
    )
    if execution_microbatching:
        trainer_kwargs["accumulate_grad_batches"] = 1
    # Manual optimization performs clipping once per logical optimizer step.
    # Lightning's automatic clipping must be disabled for execution chunks.
    trainer_kwargs["gradient_clip_val"] = (
        None
        if execution_microbatching
        else cfg_get(cfg, "trainer.gradient_clip_val", 0.0)
    )
    trainer_kwargs["gradient_clip_algorithm"] = cfg_get(
        cfg, "trainer.gradient_clip_algorithm", "norm"
    )

    trainer = pl.Trainer(**trainer_kwargs)

    do_train = cfg_bool(cfg, "experiment.train", True)
    do_predict = cfg_bool(cfg, "experiment.predict", False)
    from_checkpoint = cfg_bool(cfg, "experiment.from_checkpoint", False)
    resume_training_state = cfg_bool(
        cfg, "experiment.resume_training_state", False
    )
    allow_weights_only_resume = cfg_bool(
        cfg, "experiment.allow_weights_only_resume", False
    )

    selected_ckpt = None
    trainer_resume_ckpt = None

    if resume_training_state and not do_train:
        raise ValueError("resume_training_state=true requires experiment.train=true.")

    if (from_checkpoint or resume_training_state) and do_train:
        configured_resume_path = cfg_get(
            cfg, "experiment.resume_checkpoint_path", None
        )
        if configured_resume_path:
            selected_ckpt = str(Path(str(configured_resume_path)).expanduser().resolve())
            if not Path(selected_ckpt).is_file():
                raise FileNotFoundError(
                    f"Configured resume checkpoint does not exist: {selected_ckpt}"
                )
        else:
            selected_ckpt = choose_checkpoint(
                checkpoint_callback=val_loss_checkpoint_callback,
                ckpt_dir=ckpt_dir,
                run_checkpoint_root=paths_checkpoints,
                prefer="last" if resume_training_state else "latest",
            )

        if selected_ckpt is None:
            raise FileNotFoundError(
                "Checkpoint continuation was requested but no checkpoint was "
                f"found under: {paths_checkpoints}"
            )

        print(f"Checkpoint selected from disk: {selected_ckpt}")
    elif from_checkpoint:
        print(
            "Prediction-only checkpoint loading will resolve best_val_loss and "
            "best_pcc independently."
        )

    # ------------------------------------------------------------
    # Training on every non-validation ID, validating on the fixed common panel
    # ------------------------------------------------------------
    if do_train:
        if selected_ckpt is not None:
            if resume_training_state:
                checkpoint_payload = torch.load(
                    selected_ckpt, map_location="cpu", weights_only=False
                )
                has_optimizer_state = bool(checkpoint_payload.get("optimizer_states"))
                del checkpoint_payload
                if has_optimizer_state:
                    trainer_resume_ckpt = selected_ckpt
                    print(
                        "Resuming complete Trainer state (model, optimizer, "
                        "scheduler, epoch, callbacks, and loops)."
                    )
                elif allow_weights_only_resume:
                    print(
                        "WARNING: historical checkpoint is weights-only. Loading "
                        "model weights, but Adam/scheduler/callback state will restart; "
                        "this is a warm continuation, not an exact Trainer resume."
                    )
                    load_weights_only(lit_model=lit_model, ckpt_path=selected_ckpt)
                else:
                    raise RuntimeError(
                        "Exact training resume requested, but the checkpoint has no "
                        "optimizer state (it was saved with save_weights_only=True). "
                        "Set experiment.allow_weights_only_resume=true only if an "
                        "explicitly labelled warm continuation is acceptable."
                    )
            else:
                print("Loading checkpoint weights before training.")
                load_weights_only(lit_model=lit_model, ckpt_path=selected_ckpt)

        trainer.fit(
            lit_model,
            datamodule=datamodule,
            ckpt_path=trainer_resume_ckpt,
        )

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    if do_predict:
        paths_results.mkdir(parents=True, exist_ok=True)
        prediction_split_name = "test" if test_fold else "val"
        prediction_split_ids = test_fold if test_fold else validation_fold
        prediction_variants = resolve_prediction_checkpoint_variants(cfg)
        callback_paths = {
            "best_val_loss": val_loss_checkpoint_callback.best_model_path,
            "best_pcc": (
                pcc_checkpoint_callback.best_model_path
                if pcc_checkpoint_callback is not None
                else ""
            ),
        }
        prediction_manifest: dict[str, dict[str, Any]] = {}

        for variant in prediction_variants:
            # Search the complete task checkpoint tree first. This matters for
            # a historical weights-only warm continuation: the newly created
            # callback cannot restore its old best-k bookkeeping, but the
            # scientifically selected checkpoint must still be the minimum
            # val-loss file across both the original and continuation segments.
            disk_selected = find_prediction_checkpoint(paths_checkpoints, variant)
            ckpt_to_use = disk_selected or callback_paths[variant]
            if ckpt_to_use is None:
                raise FileNotFoundError(
                    f"Prediction requested {variant!r}, but no matching checkpoint "
                    f"was found below {paths_checkpoints}."
                )

            print(f"Loading {variant} checkpoint for prediction: {ckpt_to_use}")
            load_weights_only(lit_model=lit_model, ckpt_path=ckpt_to_use)
            print(
                f"Predicting on the fixed common {prediction_split_name} set with "
                f"{variant}..."
            )
            main_predictions = trainer.predict(
                model=lit_model,
                datamodule=datamodule,
                ckpt_path=None,
            )

            out_file = paths_results / (
                f"predictions_main_{prediction_split_name}_{variant}_{dataset_str}.parquet"
            )
            main_prediction_rows = save_predictions_for_trainer(
                predictions=main_predictions,
                out_file=out_file,
                trainer=trainer,
            )
            shared_profile_path: Path | None = None
            if (
                trainer.is_global_zero
                and cfg_bool(cfg, "prediction.sequence_only_shared_profile", False)
            ):
                shared_profile_path = paths_results / (
                    "common_test_L_profiles.parquet"
                    if variant == "best_val_loss"
                    else f"common_test_L_profiles_{variant}.parquet"
                )
                raw_dataset_count = cfg_get(cfg, "orchestrator.N", None)
                export_sequence_only_shared_profiles(
                    prediction_path=out_file,
                    output_path=shared_profile_path,
                    expected_transcript_ids=prediction_split_ids,
                    run_id=str(cfg_get(cfg, "orchestrator.run_id", cfg.name)),
                    dataset_count=(
                        None
                        if raw_dataset_count is None
                        else int(raw_dataset_count)
                    ),
                    subset_identifier=cfg_get(
                        cfg, "orchestrator.subset_identifier", None
                    ),
                )
            prediction_manifest[variant] = {
                "checkpoint_path": str(ckpt_to_use),
                "output_path": str(out_file),
                "shared_profile_output_path": (
                    str(shared_profile_path)
                    if shared_profile_path is not None
                    else None
                ),
                "prediction_rows": int(main_prediction_rows),
                "split_name": prediction_split_name,
                "transcript_count": int(len(prediction_split_ids)),
                "transcript_id_hash": transcript_id_hash(prediction_split_ids),
                "sequence_only_shared_profile_prediction": cfg_bool(
                    cfg, "prediction.sequence_only_shared_profile", False
                ),
                "observation_dependent_dummy_outputs_are_scientific": False,
            }

            if trainer.is_global_zero and main_prediction_rows > 0:
                print(
                    f"{variant} {prediction_split_name} prediction complete: "
                    f"{out_file}"
                )
            elif trainer.is_global_zero:
                print(
                    f"No {variant} {prediction_split_name} predictions were returned."
                )

        if trainer.is_global_zero:
            (paths_results / "prediction_checkpoint_manifest.json").write_text(
                json.dumps(prediction_manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
