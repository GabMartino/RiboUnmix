from __future__ import annotations

import hashlib
import json
import os
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
    RiboAIQueuingDatamoduleMultiDataset,
)
from Models.RiboQueuingModel import RiboQueuingModel
from Models.RiboQueuingModelLighningModule import RiboQueuingModelLightningModule
from Utils.checkpoints import find_checkpoint


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


def open_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return {} if data is None else data


def resolve_dataset_config_path(name_or_path: str | Path) -> Path:
    path = Path(str(name_or_path))

    if path.suffix in {".yaml", ".yml"} or len(path.parts) > 1:
        return path

    return Path(__file__).resolve().parent / "config" / "dataset_config" / f"{path}.yaml"


def dataset_path_mapping_from_source(
    cfg: DictConfig,
    source: Any,
) -> tuple[dict[str, str], str]:
    """
    Resolve a dataset_path mapping.

    source:
        None / "active" / "dataset_config"
            Use the active Hydra dataset_config.
        "datasets_paths"
            Load config/dataset_config/datasets_paths.yaml.
        path/to/file.yaml
            Load an explicit YAML config with a dataset_path mapping.
    """
    if source is None:
        return dict(cfg.dataset_config.dataset_path), "active dataset_config"

    source_str = str(source).strip()
    if source_str.lower() in {"", "active", "dataset_config", "same"}:
        return dict(cfg.dataset_config.dataset_path), "active dataset_config"

    source_path = resolve_dataset_config_path(source_str)
    source_cfg = open_file(source_path)

    if "dataset_path" not in source_cfg:
        raise KeyError(f"dataset_path missing in split source config: {source_path}")

    return dict(source_cfg["dataset_path"]), str(source_path)


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


def get_split_universe_datasets(cfg: DictConfig, experiment_datasets: list[str]) -> list[str]:
    """
    Dataset universe used ONLY to build the transcript split.

    This is the important fairness fix.

    Example:
        experiment.dataset: ["grimson_2019"]
        split.master_dataset_universe: ["grimson_2019", "kutay_2021"]

    Then the split is generated from Grimson ∪ Kutay, but the datamodule
    later filters the train/val transcript IDs to Grimson-available pairs.
    """
    raw = cfg_get(cfg, "split.master_dataset_universe", None)

    if raw is None:
        print(
            "\n[split] No split.master_dataset_universe provided. "
            "Using experiment.dataset as split universe. "
            "This is NOT ideal for single-vs-multi comparisons.\n"
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


def make_dataset_balance_run_tag(cfg: DictConfig) -> str:
    if cfg_bool(cfg, "loss.dataset_balanced_loss", False):
        return "DBLossOn"
    return "DBLossOff"


def make_replica_objective_run_tag(cfg: DictConfig) -> str:
    objective = str(cfg_get(cfg, "loss.replica_objective", "replica")).lower()

    if objective == "consensus":
        return "ConsensusOnly"

    if objective == "consensus_plus_replica":
        replica_nll_weight = format_run_tag_value(
            cfg_get(cfg, "loss.replica_nll_weight", 0.0)
        )
        return f"ConsensusReplicas_nll{replica_nll_weight}"

    if objective == "replica":
        return "ReplicasOnly"

    return f"ReplicaObj{format_run_tag_value(objective)}"


def make_sampling_run_tag(cfg: DictConfig) -> str | None:
    sampling = str(cfg_get(cfg, "data.train_sampling_strategy", "default")).strip()
    if sampling in {"", "default", "None", "none"}:
        return None
    return f"Sampling_{format_run_tag_value(sampling)}"


def make_pcc_run_tag(cfg: DictConfig) -> str:
    if not cfg_bool(cfg, "loss.pcc_loss_enabled", False):
        return "PCCOff"

    mode = str(cfg_get(cfg, "loss.pcc_loss_mode", "raw")).lower()
    total_weight = format_run_tag_value(cfg_get(cfg, "loss.pcc_loss_weight", 1.0))

    if mode == "raw":
        return f"PCCraw_w{total_weight}"

    if mode == "log1p":
        return f"PCClog1p_w{total_weight}"

    if mode.startswith("hybrid_raw_nb_vst"):
        suffix = ""
        if "weighted" in mode:
            suffix += "Weighted"
        if "mean_ratio_gated" in mode:
            suffix += "Gated"

        raw_weight = format_run_tag_value(
            cfg_get(cfg, "loss.pcc_raw_component_weight", 0.0)
        )
        nb_vst_weight = format_run_tag_value(
            cfg_get(cfg, "loss.pcc_nb_vst_component_weight", 0.0)
        )
        return (
            f"PCCrawVarAdj{suffix}_w{total_weight}"
            f"_raw{raw_weight}_var{nb_vst_weight}"
        )

    if mode.startswith("nb_vst"):
        suffix = ""
        if "weighted" in mode:
            suffix += "Weighted"
        if "mean_ratio_gated" in mode:
            suffix += "Gated"
        return f"PCCvarAdj{suffix}_w{total_weight}"

    return f"PCC{format_run_tag_value(mode)}_w{total_weight}"


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


def make_run_tag(cfg: DictConfig) -> str:
    parts = [
        "queueNB",
        make_dataset_balance_run_tag(cfg),
        make_replica_objective_run_tag(cfg),
        make_pcc_run_tag(cfg),
    ]

    sampling_tag = make_sampling_run_tag(cfg)
    if sampling_tag is not None:
        parts.append(sampling_tag)

    parts.append(make_sequence_features_run_tag(cfg))

    return "_".join(parts)


def dataset_name_from_path(path: str | Path) -> str:
    return os.path.basename(str(path)).split(".")[0]


def sanitize_metric_name_for_filename(metric_name: str) -> str:
    return str(metric_name).replace("/", "__")


def shared_logger_version_from_environment() -> str | None:
    """
    Return a deterministic logger version shared by all externally launched DDP
    ranks. With Slurm + srun every rank executes this script independently before
    Lightning has a Trainer/rank-zero guard, so TensorBoardLogger's auto version
    discovery can race and create one version_* directory per rank.
    """
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
# CSS + dataset-availability-aware split logic
# ============================================================

def css_count(x: Any) -> int:
    """
    Robust CSS counter.

    Handles:
        - list/array of CSS positions
        - boolean masks
        - None / NaN
        - stringified lists
    """
    if x is None:
        return 0

    if isinstance(x, float) and np.isnan(x):
        return 0

    if isinstance(x, str):
        s = x.strip()

        if s in {"", "[]", "nan", "None", "null"}:
            return 0

        try:
            parsed = json.loads(s)
            return css_count(parsed)
        except Exception:
            s = s.strip("[]()")
            if not s:
                return 0
            return len([v for v in s.split(",") if v.strip()])

    arr = np.asarray(x)

    if arr.ndim == 0:
        try:
            if pd.isna(arr.item()):
                return 0
        except Exception:
            pass

        try:
            return int(bool(arr.item()))
        except Exception:
            return 0

    if arr.dtype == bool:
        return int(arr.sum())

    count = 0

    for v in arr.reshape(-1):
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass

        count += 1

    return int(count)


def css_bin(n_css: int) -> str:
    if n_css <= 0:
        return "css_0"
    if n_css == 1:
        return "css_1"
    if n_css <= 3:
        return "css_2_3"
    return "css_4_plus"


def split_ids(
    ids: Sequence[str],
    *,
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    ids = list(map(str, ids))

    if len(ids) == 0:
        return [], []

    if len(ids) == 1:
        return ids, []

    n_val = int(round(len(ids) * float(val_frac)))
    n_val = max(1, min(n_val, len(ids) - 1))

    perm = rng.permutation(ids)

    val_ids = list(map(str, perm[:n_val]))
    train_ids = list(map(str, perm[n_val:]))

    return train_ids, val_ids


def build_transcript_metadata(
    *,
    sequences_path: str | Path,
    datasets_paths: Sequence[str | Path],
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
    # Only the CSS column (keyed by transcript_id) is read from the sequence
    # parquet here; the large unused sequence/structure columns are skipped.
    _seq_available = set(pq.read_schema(sequences_path).names)
    _seq_css_col = (
        "conserved_stalling_sites"
        if "conserved_stalling_sites" in _seq_available
        else "css"
    )
    _seq_columns = [c for c in ("transcript_id", _seq_css_col) if c in _seq_available]
    seq_df = pd.read_parquet(sequences_path, columns=_seq_columns or None)

    if "transcript_id" in seq_df.columns:
        seq_df = seq_df.set_index("transcript_id")

    seq_df.index = seq_df.index.astype(str)

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
            "css_count": int(n_css),
            "css_bin": css_bin(int(n_css)),
            "has_css": int(n_css) > 0,
        }

    return metadata


def print_split_summary(
    *,
    name: str,
    ids: Sequence[str],
    metadata: dict[str, dict[str, Any]],
) -> None:
    ids = list(map(str, ids))

    availability_counts: dict[str, int] = defaultdict(int)
    css_bin_counts: dict[str, int] = defaultdict(int)

    css_total = 0
    css_positive = 0

    for tid in ids:
        m = metadata[tid]

        availability_counts[str(m["availability"])] += 1
        css_bin_counts[str(m["css_bin"])] += 1

        css_total += int(m["css_count"])
        css_positive += int(bool(m["has_css"]))

    print(f"\n=== {name} split summary ===")
    print(f"transcripts: {len(ids)}")
    print(f"CSS-positive transcripts: {css_positive}")
    print(f"total CSS sites: {css_total}")

    print("availability:")
    for key, value in sorted(availability_counts.items()):
        print(f"  {key:45s} {value}")

    print("CSS bins:")
    for key, value in sorted(css_bin_counts.items()):
        print(f"  {key:12s} {value}")


def css_and_availability_aware_splits(
    *,
    sequences_path: str | Path,
    datasets_paths: Sequence[str | Path],
    css_split_path: str | Path,
    train_frac: float = 0.85,
    main_val_frac: float = 0.10,
    css_benchmark_frac: float = 0.05,
    random_seed: int = 42,
) -> tuple[list[str], list[str], list[str], dict[str, dict[str, Any]]]:
    """
    Build three transcript-level splits:

        train_ids
            Used for training.

        main_val_ids
            Representative validation set for profile loss/PCC, early stopping,
            and LR scheduling.

        css_benchmark_ids
            CSS-enriched held-out set for biological-branch CSS/peak recall.
            Do not use this set for early stopping.

    No transcript appears in more than one split.
    """
    train_frac = float(train_frac)
    main_val_frac = float(main_val_frac)
    css_benchmark_frac = float(css_benchmark_frac)

    if not np.isclose(train_frac + main_val_frac + css_benchmark_frac, 1.0):
        raise ValueError("train_frac + main_val_frac + css_benchmark_frac must sum to 1.")

    rng = np.random.default_rng(int(random_seed))

    metadata = build_transcript_metadata(
        sequences_path=sequences_path,
        datasets_paths=datasets_paths,
    )

    all_ids = sorted(metadata.keys())

    with Path(css_split_path).open("r", encoding="utf-8") as f:
        css_split = json.load(f)

    old_css_val = set(map(str, css_split.get("validation_set", [])))

    css_positive = [tid for tid in all_ids if metadata[tid]["has_css"]]
    css_positive_old_val = [tid for tid in css_positive if tid in old_css_val]
    css_positive_other = [tid for tid in css_positive if tid not in old_css_val]

    n_css_benchmark = int(round(len(all_ids) * css_benchmark_frac))
    n_css_benchmark = min(n_css_benchmark, len(css_positive))

    rng.shuffle(css_positive_old_val)
    rng.shuffle(css_positive_other)

    css_benchmark_ids = css_positive_old_val[:n_css_benchmark]

    if len(css_benchmark_ids) < n_css_benchmark:
        need = n_css_benchmark - len(css_benchmark_ids)
        css_benchmark_ids += css_positive_other[:need]

    css_benchmark_set = set(map(str, css_benchmark_ids))

    remaining = [tid for tid in all_ids if tid not in css_benchmark_set]

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)

    for tid in remaining:
        key = (str(metadata[tid]["availability"]), str(metadata[tid]["css_bin"]))
        strata[key].append(tid)

    main_val_target = int(round(len(all_ids) * main_val_frac))
    remaining_val_frac = main_val_target / max(len(remaining), 1)

    train_ids: list[str] = []
    main_val_ids: list[str] = []

    for _, ids in sorted(strata.items(), key=lambda kv: str(kv[0])):
        tr, va = split_ids(ids, val_frac=remaining_val_frac, rng=rng)

        train_ids.extend(tr)
        main_val_ids.extend(va)

    train_set = set(train_ids)
    main_val_set = set(main_val_ids)
    css_set = set(css_benchmark_ids)

    if train_set & main_val_set:
        raise RuntimeError("Overlap between train and main validation splits.")
    if train_set & css_set:
        raise RuntimeError("Overlap between train and CSS benchmark splits.")
    if main_val_set & css_set:
        raise RuntimeError("Overlap between main validation and CSS benchmark splits.")

    covered = train_set | main_val_set | css_set

    if covered != set(all_ids):
        missing = set(all_ids) - covered
        extra = covered - set(all_ids)

        raise RuntimeError(
            f"Split coverage error. missing={len(missing)}, extra={len(extra)}"
        )

    train_ids = sorted(train_set)
    main_val_ids = sorted(main_val_set)
    css_benchmark_ids = sorted(css_set)

    print_split_summary(name="Train", ids=train_ids, metadata=metadata)
    print_split_summary(name="Main validation", ids=main_val_ids, metadata=metadata)
    print_split_summary(name="CSS biological benchmark", ids=css_benchmark_ids, metadata=metadata)

    return train_ids, main_val_ids, css_benchmark_ids, metadata


def save_split_manifest(
    *,
    out_file: Path,
    experiment_datasets: list[str],
    split_universe_datasets: list[str],
    split_dataset_source: str | None = None,
    split_dataset_paths: Sequence[str] | None = None,
    training_dataset_paths: Sequence[str] | None = None,
    extra_train_ids: Sequence[str] | None = None,
    train_ids: Sequence[str],
    main_val_ids: Sequence[str],
    css_benchmark_ids: Sequence[str],
    metadata: dict[str, dict[str, Any]],
    seed: int,
    train_frac: float,
    main_val_frac: float,
    css_benchmark_frac: float,
) -> None:
    """
    Save split provenance so you can verify that single-dataset and multi-dataset
    experiments used the same master transcript split.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)

    def availability_counts(ids: Sequence[str]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)

        for tid in map(str, ids):
            counts[str(metadata[tid]["availability"])] += 1

        return dict(sorted(counts.items()))

    manifest = {
        "seed": int(seed),
        "experiment_datasets": list(experiment_datasets),
        "split_universe_datasets": list(split_universe_datasets),
        "split_dataset_source": split_dataset_source,
        "split_dataset_paths": list(map(str, split_dataset_paths or [])),
        "training_dataset_paths": list(map(str, training_dataset_paths or [])),
        "extra_train_ids_from_training_datasets": list(map(str, extra_train_ids or [])),
        "fractions": {
            "train_frac": float(train_frac),
            "main_val_frac": float(main_val_frac),
            "css_benchmark_frac": float(css_benchmark_frac),
        },
        "counts": {
            "train": len(train_ids),
            "main_val": len(main_val_ids),
            "css_benchmark": len(css_benchmark_ids),
        },
        "availability_counts": {
            "train": availability_counts(train_ids),
            "main_val": availability_counts(main_val_ids),
            "css_benchmark": availability_counts(css_benchmark_ids),
        },
        "train_ids": list(map(str, train_ids)),
        "main_val_ids": list(map(str, main_val_ids)),
        "css_benchmark_ids": list(map(str, css_benchmark_ids)),
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved split manifest: {out_file}")


def filter_ids_available_in_experiment(
    *,
    ids: Sequence[str],
    metadata: dict[str, dict[str, Any]],
    experiment_datasets: Sequence[str],
) -> list[str]:
    experiment_dataset_set = set(map(str, experiment_datasets))
    filtered = []

    for tid in map(str, ids):
        dataset_names = set(map(str, metadata.get(tid, {}).get("datasets", [])))
        if dataset_names & experiment_dataset_set:
            filtered.append(tid)

    return filtered


def dataset_names_by_transcript(
    *,
    ids: Sequence[str],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    allowed: dict[str, list[str]] = {}

    for tid in map(str, ids):
        if tid not in metadata:
            continue
        allowed[tid] = list(map(str, metadata[tid].get("datasets", [])))

    return allowed


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
    It intentionally does not restore optimizer/scheduler state.
    """
    ckpt_path = Path(ckpt_path)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    incompatible = lit_model.load_state_dict(state_dict, strict=False)

    print(f"Loaded weights from: {ckpt_path}")

    if len(incompatible.missing_keys) > 0:
        print(f"Missing keys: {len(incompatible.missing_keys)}")

    if len(incompatible.unexpected_keys) > 0:
        print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")


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
        "additive_bias",
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
    train_allowed_dataset_names_by_transcript: dict[str, list[str]] | None = None,
    val_allowed_dataset_names_by_transcript: dict[str, list[str]] | None = None,
    split_size: float,
    seed: int,
) -> RiboAIQueuingDatamoduleMultiDataset:
    feature_cfg = cfg_get(cfg, "model.additional_sequence_features", {})
    if OmegaConf.is_config(feature_cfg):
        feature_cfg = OmegaConf.to_container(feature_cfg, resolve=True)
    else:
        feature_cfg = dict(feature_cfg or {})
    return RiboAIQueuingDatamoduleMultiDataset(
        sequences_path=cfg.paths.sequences_path,
        datasets_paths=datasets_paths,
        batch_size=int(cfg.data.batch_size),
        split=(train_fold, val_fold),
        split_p=split_size,
        num_workers=int(cfg.data.num_workers),
        seed=seed,
        nt_encoding_path=cfg.paths.encodings.nt,
        codon_encoding_path=cfg.paths.encodings.codon,
        codon_to_aa_encoding_path=cfg.paths.encodings.codon_to_aa,
        aa_encoding_path=cfg.paths.encodings.aa,
        datasets_encoding_path=cfg.paths.encodings.datasets,
        balanced_train_sampling=cfg_bool(cfg, "data.balanced_train_sampling", False),
        dataset_balance_gamma=float(cfg_get(cfg, "data.dataset_balance_gamma", 0.0)),
        train_samples_per_epoch=cfg_get(cfg, "data.train_samples_per_epoch", None),
        dataset_aware_batching=cfg_bool(cfg, "data.dataset_aware_batching", False),
        datasets_per_batch=int(cfg_get(cfg, "data.datasets_per_batch", 2)),
        train_sampling_strategy=cfg_get(cfg, "data.train_sampling_strategy", "random_dataset_per_transcript"),
        train_allowed_dataset_names_by_transcript=train_allowed_dataset_names_by_transcript,
        val_allowed_dataset_names_by_transcript=val_allowed_dataset_names_by_transcript,
        pin_memory=cfg_bool(cfg, "data.pin_memory", True),
        prefetch_factor=cfg_get(cfg, "data.prefetch_factor", 4),
        use_ribo_replicas=cfg_bool(cfg, "data.use_ribo_replicas", False),
        ribo_replicas_column=str(
            cfg_get(cfg, "data.ribo_replicas_column", "ribo_cds_replicas")
        ),
        additional_sequence_features=feature_cfg,
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
    #     fixed master universe used to generate transcript IDs.
    #
    # For fair single-vs-multi comparison:
    #
    #   experiment.dataset: ["grimson_2019"]
    #   split.master_dataset_universe: ["grimson_2019", "kutay_2021"]
    #
    #   experiment.dataset: ["kutay_2021"]
    #   split.master_dataset_universe: ["grimson_2019", "kutay_2021"]
    #
    #   experiment.dataset: ["grimson_2019", "kutay_2021"]
    #   split.master_dataset_universe: ["grimson_2019", "kutay_2021"]
    # ------------------------------------------------------------
    experiment_datasets = get_datasets(cfg)
    split_universe_datasets = get_split_universe_datasets(cfg, experiment_datasets)

    experiment_dataset_paths = dataset_paths_for(
        mapping=dict(cfg.dataset_config.dataset_path),
        datasets=experiment_datasets,
        source_label="active dataset_config",
    )

    split_source = cfg_get(cfg, "split.source_dataset_config", None)
    split_dataset_mapping, split_dataset_source_label = dataset_path_mapping_from_source(
        cfg,
        split_source,
    )
    split_universe_dataset_paths = dataset_paths_for(
        mapping=split_dataset_mapping,
        datasets=split_universe_datasets,
        source_label=split_dataset_source_label,
    )

    split_size = float(
        cfg_get(
            cfg,
            "experiment.split_size",
            cfg_get(cfg, "split.train_frac", 0.90),
        )
    )

    print("\n=== Dataset configuration ===")
    print(f"Experiment datasets:     {experiment_datasets}")
    print(f"Split universe datasets: {split_universe_datasets}")
    print(f"Training dataset config: active dataset_config")
    print(f"Split dataset source:    {split_dataset_source_label}")

    # ------------------------------------------------------------
    # Split strategy
    # ------------------------------------------------------------
    # main_val_fold is representative and is used for early stopping.
    # css_benchmark_fold is CSS-enriched and is used only for biological-branch
    # peak/CSS analysis after training.
    #
    # Important:
    #     The split is generated from split_universe_dataset_paths, not from
    #     experiment_dataset_paths.
    # ------------------------------------------------------------
    train_frac = float(cfg_get(cfg, "split.train_frac", split_size))
    main_val_frac = float(cfg_get(cfg, "split.main_val_frac", 1.0 - split_size))

    # Use the actual key. Keep backward-compatible fallback to default.
    css_benchmark_frac = float(cfg_get(cfg, "split.css_benchmark_frac", -1.0))

    if css_benchmark_frac < 0.0:
        css_benchmark_frac = float(cfg_get(cfg, "split.default_css_benchmark_frac", 0.05))
        main_val_frac = max(1.0 - train_frac - css_benchmark_frac, 0.0)

    total_frac = train_frac + main_val_frac + css_benchmark_frac

    if not np.isclose(total_frac, 1.0):
        main_val_frac = 1.0 - train_frac - css_benchmark_frac

        if main_val_frac <= 0.0:
            raise ValueError(
                "Invalid split fractions. Need train_frac + css_benchmark_frac < 1."
            )

    print("\n=== Split fractions ===")
    print(f"train_frac:         {train_frac}")
    print(f"main_val_frac:      {main_val_frac}")
    print(f"css_benchmark_frac: {css_benchmark_frac}")

    train_fold, main_val_fold, css_benchmark_fold, split_metadata = css_and_availability_aware_splits(
        sequences_path=cfg.paths.sequences_path,
        datasets_paths=split_universe_dataset_paths,
        css_split_path=cfg.paths.css_split,
        train_frac=train_frac,
        main_val_frac=main_val_frac,
        css_benchmark_frac=css_benchmark_frac,
        random_seed=seed,
    )

    extra_train_ids: list[str] = []
    if cfg_bool(cfg, "split.include_experiment_only_train_ids", False):
        experiment_metadata = build_transcript_metadata(
            sequences_path=cfg.paths.sequences_path,
            datasets_paths=experiment_dataset_paths,
        )
        split_id_set = set(map(str, split_metadata.keys()))
        extra_train_ids = sorted(set(experiment_metadata.keys()) - split_id_set)

        if extra_train_ids:
            print(
                "\n[split] Adding training-only transcript IDs available in the "
                "active experiment datasets but absent from the split source: "
                f"{len(extra_train_ids)}"
            )
            train_fold = sorted(set(map(str, train_fold)).union(extra_train_ids))
            for tid in extra_train_ids:
                split_metadata[tid] = {
                    **experiment_metadata[tid],
                    "split_role": "extra_train_from_training_datasets",
                }
        else:
            print("\n[split] No extra training-only IDs found outside the split source.")

    main_val_fold_for_experiment = filter_ids_available_in_experiment(
        ids=main_val_fold,
        metadata=split_metadata,
        experiment_datasets=experiment_datasets,
    )
    removed_main_val_ids = len(main_val_fold) - len(main_val_fold_for_experiment)
    if removed_main_val_ids > 0:
        print(
            "\n[split] Filtering main validation IDs to transcripts available "
            "in the active experiment datasets: "
            f"{len(main_val_fold_for_experiment)} kept, {removed_main_val_ids} removed."
        )
    if len(main_val_fold_for_experiment) == 0:
        raise RuntimeError(
            "Main validation split is empty after filtering to active experiment "
            f"datasets {experiment_datasets}. Check split.master_dataset_universe "
            "or experiment.dataset."
        )

    dataset_str = make_dataset_signature(experiment_datasets)
    split_universe_str = make_dataset_signature(split_universe_datasets)
    run_tag = make_run_tag(cfg)

    print(f"\nTracking dataset signature: {dataset_str}")
    print(f"Split universe signature:   {split_universe_str}")
    print(f"Run tag:                    {run_tag}")

    paths_logs = Path(cfg.paths.logs) / dataset_str / run_tag
    paths_results = Path(cfg.paths.results) / dataset_str / run_tag
    paths_checkpoints = Path(cfg.paths.checkpoints) / dataset_str / run_tag

    # Save split provenance once. Under Slurm+srun, every rank executes this
    # script before the Lightning Trainer exists, so use the environment rank
    # guard rather than trainer.is_global_zero.
    if env_global_rank() == 0:
        save_split_manifest(
            out_file=paths_results / f"split_manifest_experiment_{dataset_str}_universe_{split_universe_str}.json",
            experiment_datasets=experiment_datasets,
            split_universe_datasets=split_universe_datasets,
            split_dataset_source=split_dataset_source_label,
            split_dataset_paths=split_universe_dataset_paths,
            training_dataset_paths=experiment_dataset_paths,
            extra_train_ids=extra_train_ids,
            train_ids=train_fold,
            main_val_ids=main_val_fold,
            css_benchmark_ids=css_benchmark_fold,
            metadata=split_metadata,
            seed=seed,
            train_frac=train_frac,
            main_val_frac=main_val_frac,
            css_benchmark_frac=css_benchmark_frac,
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
    torch_model = RiboQueuingModel(
        model_configs=cfg.model,
        eps=float(cfg.model.get("eps", 1e-8)),
    )

    lit_model = RiboQueuingModelLightningModule(
        torch_model,
        config=cfg,
        dataset_encoding=dataset_encoding,
    )

    restrict_train_pairs_to_split_source = cfg_bool(
        cfg, "split.restrict_train_pairs_to_split_source", False
    )
    restrict_val_pairs_to_split_source = cfg_bool(
        cfg, "split.restrict_val_pairs_to_split_source", False
    )
    train_allowed_datasets = (
        dataset_names_by_transcript(ids=train_fold, metadata=split_metadata)
        if restrict_train_pairs_to_split_source
        else None
    )
    val_allowed_datasets = (
        dataset_names_by_transcript(
            ids=main_val_fold_for_experiment,
            metadata=split_metadata,
        )
        if restrict_val_pairs_to_split_source
        else None
    )

    # Actual datamodules use the experiment datasets, but receive the split IDs.
    # When configured, validation flat pairs are restricted to the filtered split
    # source availability even though training/prediction can load raw profiles.
    datamodule = make_datamodule(
        cfg=cfg,
        datasets_paths=experiment_dataset_paths,
        train_fold=train_fold,
        val_fold=main_val_fold_for_experiment,
        train_allowed_dataset_names_by_transcript=train_allowed_datasets,
        val_allowed_dataset_names_by_transcript=val_allowed_datasets,
        split_size=split_size,
        seed=seed,
    )

    css_benchmark_fold_for_experiment = filter_ids_available_in_experiment(
        ids=css_benchmark_fold,
        metadata=split_metadata,
        experiment_datasets=experiment_datasets,
    )

    css_datamodule = None
    if len(css_benchmark_fold_for_experiment) > 0:
        css_allowed_datasets = (
            dataset_names_by_transcript(
                ids=css_benchmark_fold_for_experiment,
                metadata=split_metadata,
            )
            if restrict_val_pairs_to_split_source
            else None
        )
        css_datamodule = make_datamodule(
            cfg=cfg,
            datasets_paths=experiment_dataset_paths,
            train_fold=train_fold,
            val_fold=css_benchmark_fold_for_experiment,
            train_allowed_dataset_names_by_transcript=train_allowed_datasets,
            val_allowed_dataset_names_by_transcript=css_allowed_datasets,
            split_size=split_size,
            seed=seed,
        )
    else:
        print(
            "\nNo CSS benchmark transcripts are available for the selected "
            f"experiment datasets {experiment_datasets}. CSS prediction will be skipped.\n"
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

    safe_monitor = sanitize_metric_name_for_filename(monitor)
    filename = "{epoch}-{" + monitor + ":.4f}"

    # If monitor contains "/", Lightning may create nested paths from filename.
    # This fallback avoids that.
    if "/" in monitor:
        filename = "{epoch}-" + safe_monitor + "-{" + monitor + ":.4f}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=filename,
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        monitor=monitor,
        mode=metric_mode,
    )

    # Keep a second, independently selected checkpoint for the metric the
    # current ablation is explicitly trying to improve. The likelihood-best
    # checkpoint remains available for calibration/NLL comparisons.
    pcc_checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="pcc-{epoch}-{val_mu_pcc:.4f}",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        monitor="val_mu_pcc",
        mode="max",
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
        "precision": cfg.trainer.precision,
        "max_epochs": int(cfg.trainer.max_epochs),
        "logger": tb_logger,
        "log_every_n_steps": int(cfg.trainer.log_every_n_steps),
        "accumulate_grad_batches": int(cfg_get(cfg, "trainer.accumulate_grad_batches", 1)),
        "callbacks": [
            checkpoint_callback,
            pcc_checkpoint_callback,
            early_stopping,
            lr_monitor,
        ],
        "use_distributed_sampler": cfg_bool(
            cfg,
            "trainer.use_distributed_sampler",
            False,
        ),
    }
    if n_devices > 1:
        trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"

    # CAGrad has its own biological-gradient override; keep Trainer clipping for
    # non-CAGrad runs only.
    if not cfg_bool(cfg, "cagrad.enabled", cfg_bool(cfg, "optim.use_cagrad", False)):
        trainer_kwargs["gradient_clip_val"] = cfg_get(cfg, "trainer.gradient_clip_val", 0.0)
        trainer_kwargs["gradient_clip_algorithm"] = cfg_get(cfg, "trainer.gradient_clip_algorithm", "norm")

    trainer = pl.Trainer(**trainer_kwargs)

    do_train = cfg_bool(cfg, "experiment.train", True)
    do_predict = cfg_bool(cfg, "experiment.predict", False)
    from_checkpoint = cfg_bool(cfg, "experiment.from_checkpoint", False)

    selected_ckpt = None

    if from_checkpoint:
        prefer = "latest" if do_train else "best"

        selected_ckpt = choose_checkpoint(
            checkpoint_callback=checkpoint_callback,
            ckpt_dir=ckpt_dir,
            run_checkpoint_root=paths_checkpoints,
            prefer=prefer,
        )

        if selected_ckpt is None:
            raise FileNotFoundError(
                f"from_checkpoint=True but no checkpoint found under: {paths_checkpoints}"
            )

        print(f"Checkpoint selected from disk: {selected_ckpt}")

    # ------------------------------------------------------------
    # Training on train_fold, validating on representative main_val_fold
    # ------------------------------------------------------------
    if do_train:
        if selected_ckpt is not None:
            print("Loading checkpoint weights before training.")
            load_weights_only(lit_model=lit_model, ckpt_path=selected_ckpt)

        trainer.fit(lit_model, datamodule=datamodule)

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    if do_predict:
        paths_results.mkdir(parents=True, exist_ok=True)

        ckpt_to_use = (
            pcc_checkpoint_callback.best_model_path
            or checkpoint_callback.best_model_path
            or selected_ckpt
        )

        if ckpt_to_use is None:
            ckpt_to_use = choose_checkpoint(
                checkpoint_callback=checkpoint_callback,
                ckpt_dir=ckpt_dir,
                run_checkpoint_root=paths_checkpoints,
                prefer="best",
            )

        if ckpt_to_use is not None:
            print(f"Loading checkpoint weights for prediction: {ckpt_to_use}")
            load_weights_only(lit_model=lit_model, ckpt_path=ckpt_to_use)
        else:
            print("No checkpoint found. Predicting with current model weights.")

        # Predict representative main validation.
        print("Predicting on representative main validation set...")
        main_predictions = trainer.predict(
            model=lit_model,
            datamodule=datamodule,
            ckpt_path=None,
        )

        out_file = paths_results / f"predictions_main_val_{dataset_str}.parquet"
        main_prediction_rows = save_predictions_for_trainer(
            predictions=main_predictions,
            out_file=out_file,
            trainer=trainer,
        )

        if trainer.is_global_zero and main_prediction_rows > 0:
            print(f"Main validation prediction complete: {out_file}")
        elif trainer.is_global_zero:
            print("No main validation predictions were returned.")

        # Predict CSS-enriched biological benchmark.
        if css_datamodule is not None:
            print("Predicting on CSS biological benchmark set...")
            css_predictions = trainer.predict(
                model=lit_model,
                datamodule=css_datamodule,
                ckpt_path=None,
            )

            out_file = paths_results / f"predictions_css_benchmark_{dataset_str}.parquet"
            css_prediction_rows = save_predictions_for_trainer(
                predictions=css_predictions,
                out_file=out_file,
                trainer=trainer,
            )

            if trainer.is_global_zero and css_prediction_rows > 0:
                print(f"CSS benchmark prediction complete: {out_file}")
            elif trainer.is_global_zero:
                print("No CSS benchmark predictions were returned.")
        elif trainer.is_global_zero:
            print("Skipping CSS benchmark prediction because the CSS benchmark split is empty.")


if __name__ == "__main__":
    main()
