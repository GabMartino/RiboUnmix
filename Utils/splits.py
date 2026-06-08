from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def conserved_stalling_sites_aware_split(
    css_split_path: str | Path,
    split_size: float = 0.8,
    random_seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Reproduce the split logic based on a sidecar JSON.

    The dataset_path points to a .parquet file; we expect a .json next to it
    with keys: "training_set" and "validation_set".

    split_size is the fraction kept for training.
    """
    with open(css_split_path, "rb") as f:
        data = json.load(f)

    train_ids = list(data["training_set"])
    val_ids = list(data["validation_set"])

    total_size = len(train_ids) + len(val_ids)
    proposed_val = int(total_size * (1 - float(split_size)))

    rng = np.random.default_rng(int(random_seed))

    if proposed_val <= len(val_ids):
        new_val = rng.choice(val_ids, size=proposed_val, replace=False)
        new_train = train_ids + list(set(val_ids) - set(new_val))
        new_val = list(new_val)
    else:
        need = proposed_val - len(val_ids)
        add = rng.choice(train_ids, size=need, replace=False)
        new_val = val_ids + list(add)
        new_train = list(set(train_ids) - set(add))

    assert len(new_train) + len(new_val) == total_size
    return new_train, new_val


def _dataset_name_from_path(path: str | Path) -> str:
    return os.path.basename(str(path)).split(".")[0]


def _css_count(x) -> int:
    """
    Robust CSS counter.

    Handles:
      - list/array of CSS positions
      - boolean mask
      - None / NaN
      - stringified lists, if present
    """
    if x is None:
        return 0

    if isinstance(x, float) and np.isnan(x):
        return 0

    if isinstance(x, str):
        s = x.strip()
        if s in {"", "[]", "nan", "None"}:
            return 0

        try:
            parsed = json.loads(s)
            return _css_count(parsed)
        except Exception:
            s = s.strip("[]()")
            if not s:
                return 0
            return len([v for v in s.split(",") if v.strip()])

    arr = np.asarray(x)

    if arr.ndim == 0:
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


def _css_bin(c: int) -> str:
    if c <= 0:
        return "css_0"
    if c == 1:
        return "css_1"
    if c <= 3:
        return "css_2_3"
    return "css_4_plus"


def _split_ids(
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

    n_val = int(round(len(ids) * val_frac))
    n_val = max(1, min(n_val, len(ids) - 1))

    perm = rng.permutation(ids)
    val = list(map(str, perm[:n_val]))
    train = list(map(str, perm[n_val:]))

    return train, val


def build_transcript_metadata(
    *,
    sequences_path: str | Path,
    datasets_paths: Sequence[str | Path],
) -> dict[str, dict]:
    """
    Build transcript-level metadata:
      - CSS count from the sequence/CSS parquet
      - dataset availability from dataset-specific ribo parquets
    """
    seq_df = pd.read_parquet(sequences_path)

    if "transcript_id" in seq_df.columns:
        seq_df = seq_df.set_index("transcript_id")

    seq_df.index = seq_df.index.astype(str)

    css_col = (
        "conserved_stalling_sites"
        if "conserved_stalling_sites" in seq_df.columns
        else "css"
    )

    if css_col not in seq_df.columns:
        raise KeyError(
            "Could not find CSS column. Expected 'conserved_stalling_sites' or 'css'."
        )

    tid_to_datasets: dict[str, set[str]] = defaultdict(set)

    for path in datasets_paths:
        dataset_name = _dataset_name_from_path(path)
        df = pd.read_parquet(path)

        if "id" not in df.columns:
            raise KeyError(f"'id' column missing in {path}")

        ids = df["id"].astype(str).values
        for tid in ids:
            tid_to_datasets[str(tid)].add(dataset_name)

    valid_ids = sorted(set(seq_df.index.astype(str)).intersection(tid_to_datasets.keys()))

    metadata = {}

    for tid in valid_ids:
        datasets = sorted(tid_to_datasets[tid])
        css_n = _css_count(seq_df.loc[tid, css_col])

        if len(datasets) == 1:
            availability = f"{datasets[0]}_only"
        else:
            availability = "paired_" + "__".join(datasets)

        metadata[tid] = {
            "transcript_id": tid,
            "datasets": datasets,
            "availability": availability,
            "css_count": css_n,
            "css_bin": _css_bin(css_n),
            "has_css": css_n > 0,
        }

    return metadata


def css_and_availability_aware_splits(
    *,
    sequences_path: str | Path,
    datasets_paths: Sequence[str | Path],
    css_split_path: str | Path,
    train_frac: float = 0.85,
    main_val_frac: float = 0.10,
    css_benchmark_frac: float = 0.05,
    random_seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Returns:
        train_ids
        main_val_ids
        css_benchmark_ids

    Logic:
      1. Split at transcript level.
      2. Main validation is representative:
            stratified by availability + CSS bin.
      3. CSS benchmark is CSS-enriched:
            preferentially drawn from CSS-positive transcripts.
      4. No transcript can appear in more than one split.
    """
    if not np.isclose(train_frac + main_val_frac + css_benchmark_frac, 1.0):
        raise ValueError("train_frac + main_val_frac + css_benchmark_frac must sum to 1.")

    rng = np.random.default_rng(int(random_seed))

    metadata = build_transcript_metadata(
        sequences_path=sequences_path,
        datasets_paths=datasets_paths,
    )

    all_ids = sorted(metadata.keys())

    with open(css_split_path, "r", encoding="utf-8") as f:
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

    css_benchmark_ids = list(map(str, css_benchmark_ids))
    css_benchmark_set = set(css_benchmark_ids)

    remaining = [tid for tid in all_ids if tid not in css_benchmark_set]

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for tid in remaining:
        key = (metadata[tid]["availability"], metadata[tid]["css_bin"])
        strata[key].append(tid)

    main_val_ids = []
    train_ids = []

    main_val_target = int(round(len(all_ids) * main_val_frac))
    remaining_val_frac = main_val_target / max(len(remaining), 1)

    for key, ids in strata.items():
        tr, va = _split_ids(ids, val_frac=remaining_val_frac, rng=rng)
        train_ids.extend(tr)
        main_val_ids.extend(va)

    train_set = set(train_ids)
    main_val_set = set(main_val_ids)
    css_set = set(css_benchmark_ids)

    if train_set & main_val_set:
        raise RuntimeError("Overlap between train and main_val.")
    if train_set & css_set:
        raise RuntimeError("Overlap between train and css_benchmark.")
    if main_val_set & css_set:
        raise RuntimeError("Overlap between main_val and css_benchmark.")

    total = len(train_set | main_val_set | css_set)

    if total != len(all_ids):
        missing = set(all_ids) - (train_set | main_val_set | css_set)
        raise RuntimeError(f"Split does not cover all valid IDs. Missing={len(missing)}")

    return (
        sorted(train_set),
        sorted(main_val_set),
        sorted(css_set),
    )
