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


def open_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return {} if data is None else data


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


def make_run_tag(cfg: DictConfig) -> str:
    tag = "PCGrad" if bool(cfg.optim.use_pcgrad) else "NOPCGrad"

    sampling = str(cfg_get(cfg, "data.train_sampling_strategy", "default"))
    dataset_balanced_loss = bool(cfg_get(cfg, "loss.dataset_balanced_loss", False))

    if sampling not in {"", "default", "None", "none"}:
        tag += f"_{sampling}"

    tag += "_DBLoss" if dataset_balanced_loss else "_SampleMeanLoss"

    return tag


def dataset_name_from_path(path: str | Path) -> str:
    return os.path.basename(str(path)).split(".")[0]


def sanitize_metric_name_for_filename(metric_name: str) -> str:
    return str(metric_name).replace("/", "__")


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
    seq_df = pd.read_parquet(sequences_path)

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
        df = pd.read_parquet(path)

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
    predictions: list[dict[str, Any]],
    out_file: Path,
) -> None:
    rows = []

    sequence_keys = {
        # Core target/prediction
        "y",
        "target",
        "mu",
        "mu_obs",
        "mu_total",
        "mu_base",

        # Mean profiles
        "mu_L_only",
        "mu_L_bio",
        "mu_L_obs",
        "mu_bio_only",
        "mu_bio_smooth",

        # Biological branch
        "w_logits",
        "w_bio",
        "w_prob",
        "h_bio",
        "rho_bio",
        "L_bio",
        "L_queue",
        "bio_q_base",

        # Observed branch
        "w_obs",
        "h_obs",
        "rho_obs",
        "L_obs",
        "L_queue_obs",
        "bio_q",
        "q",
        "q_for_profile",
        "profile_prob",

        # Multiplicative observation bias
        "obs_bias_raw",
        "obs_bias_effective",
        "obs_bias_amp",
        "obs_bias_amp_logits",
        "obs_bias_keep_prob",
        "obs_bias_keep_gate",
        "obs_bias_keep_hard",
        "obs_bias_gate_logits",

        # Backward-compatible old names
        "L_effective",
        "bio_q_smooth",
        "h_bio_smooth",
        "rho",
        "rho_diag",
        "exp_b",
        "b",
        "b_shape",
        "log_b",
        "lambda_frac",
        "lambda_bg",
        "additive_noise",
        "additive_support_eff",
        "additive_frac_eff",
        "keep_gate",

        # Batch arrays
        "mask",
        "codon_ids",
        "profile_kappa",
        "kappa_input",
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

        if key == "mask":
            return np.asarray(sliced).astype(np.bool_, copy=False).tolist()

        if key == "codon_ids":
            return np.asarray(sliced).astype(np.int64, copy=False).tolist()

        return np.asarray(sliced).astype(np.float32, copy=False).tolist()

    for batch in predictions:
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


# ============================================================
# Datamodule factory
# ============================================================

def make_datamodule(
    *,
    cfg: DictConfig,
    datasets_paths: list[str],
    train_fold: list[str],
    val_fold: list[str],
    split_size: float,
    seed: int,
) -> RiboAIQueuingDatamoduleMultiDataset:
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
        balanced_train_sampling=bool(cfg_get(cfg, "data.balanced_train_sampling", False)),
        dataset_balance_gamma=float(cfg_get(cfg, "data.dataset_balance_gamma", 0.0)),
        train_samples_per_epoch=cfg_get(cfg, "data.train_samples_per_epoch", None),
        dataset_aware_batching=bool(cfg_get(cfg, "data.dataset_aware_batching", False)),
        datasets_per_batch=int(cfg_get(cfg, "data.datasets_per_batch", 2)),
        train_sampling_strategy=cfg_get(cfg, "data.train_sampling_strategy", "random_dataset_per_transcript"),
        pin_memory=bool(cfg_get(cfg, "data.pin_memory", True)),
        prefetch_factor=cfg_get(cfg, "data.prefetch_factor", 4),
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

    experiment_dataset_paths = [
        cfg.dataset_config.dataset_path[dataset]
        for dataset in experiment_datasets
    ]

    split_universe_dataset_paths = [
        cfg.dataset_config.dataset_path[dataset]
        for dataset in split_universe_datasets
    ]

    split_size = float(cfg.experiment.split_size)

    print("\n=== Dataset configuration ===")
    print(f"Experiment datasets:     {experiment_datasets}")
    print(f"Split universe datasets: {split_universe_datasets}")

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

    dataset_str = make_dataset_signature(experiment_datasets)
    split_universe_str = make_dataset_signature(split_universe_datasets)
    run_tag = make_run_tag(cfg)

    print(f"\nTracking dataset signature: {dataset_str}")
    print(f"Split universe signature:   {split_universe_str}")
    print(f"Run tag:                    {run_tag}")

    paths_logs = Path(cfg.paths.logs) / dataset_str / run_tag
    paths_results = Path(cfg.paths.results) / dataset_str / run_tag
    paths_checkpoints = Path(cfg.paths.checkpoints) / dataset_str / run_tag

    # Save split provenance. This is essential for checking fairness across runs.
    save_split_manifest(
        out_file=paths_results / f"split_manifest_experiment_{dataset_str}_universe_{split_universe_str}.json",
        experiment_datasets=experiment_datasets,
        split_universe_datasets=split_universe_datasets,
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

    torch_model = RiboQueuingModel(
        model_configs=cfg.model,
        eps=float(cfg.model.get("eps", 1e-8)),
        mu_max=float(cfg.model.get("mu_max", 1e8)),
    )

    lit_model = RiboQueuingModelLightningModule(
        torch_model,
        config=cfg,
        dataset_encoding=dataset_encoding,
    )

    # Actual datamodules use the experiment datasets, but receive the master split IDs.
    # They should filter transcript IDs according to dataset availability internally.
    datamodule = make_datamodule(
        cfg=cfg,
        datasets_paths=experiment_dataset_paths,
        train_fold=train_fold,
        val_fold=main_val_fold,
        split_size=split_size,
        seed=seed,
    )

    css_datamodule = make_datamodule(
        cfg=cfg,
        datasets_paths=experiment_dataset_paths,
        train_fold=train_fold,
        val_fold=css_benchmark_fold,
        split_size=split_size,
        seed=seed,
    )

    tb_logger = TensorBoardLogger(save_dir=str(paths_logs), name="")
    exp_name = Path(tb_logger.log_dir or tb_logger.save_dir).name

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

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=int(cfg.callbacks.early_stopping_patience),
        mode=metric_mode,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer_kwargs = {
        "accelerator": cfg.trainer.accelerator,
        "devices": cfg.trainer.devices,
        "precision": cfg.trainer.precision,
        "max_epochs": int(cfg.trainer.max_epochs),
        "logger": tb_logger,
        "log_every_n_steps": int(cfg.trainer.log_every_n_steps),
        "accumulate_grad_batches": int(cfg_get(cfg, "trainer.accumulate_grad_batches", 1)),
        "callbacks": [checkpoint_callback, early_stopping, lr_monitor],
    }

    # Automatic optimization can use Lightning clipping.
    # PCGrad/manual optimization clips inside the LightningModule.
    if not bool(cfg.optim.use_pcgrad):
        trainer_kwargs["gradient_clip_val"] = float(cfg.trainer.gradient_clip_val)
        trainer_kwargs["gradient_clip_algorithm"] = str(cfg.trainer.gradient_clip_algorithm)

    trainer = pl.Trainer(**trainer_kwargs)

    do_train = bool(cfg.experiment.train)
    do_predict = bool(cfg.experiment.predict)
    from_checkpoint = bool(cfg.experiment.from_checkpoint)

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

        ckpt_to_use = checkpoint_callback.best_model_path or selected_ckpt

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

        if main_predictions is not None and len(main_predictions) > 0:
            out_file = paths_results / f"predictions_main_val_{dataset_str}.parquet"

            predictions_to_parquet(
                predictions=main_predictions,
                out_file=out_file,
            )

            print(f"Main validation prediction complete: {out_file}")
        else:
            print("No main validation predictions were returned.")

        # Predict CSS-enriched biological benchmark.
        print("Predicting on CSS biological benchmark set...")
        css_predictions = trainer.predict(
            model=lit_model,
            datamodule=css_datamodule,
            ckpt_path=None,
        )

        if css_predictions is not None and len(css_predictions) > 0:
            out_file = paths_results / f"predictions_css_benchmark_{dataset_str}.parquet"

            predictions_to_parquet(
                predictions=css_predictions,
                out_file=out_file,
            )

            print(f"CSS benchmark prediction complete: {out_file}")
        else:
            print("No CSS benchmark predictions were returned.")


if __name__ == "__main__":
    main()