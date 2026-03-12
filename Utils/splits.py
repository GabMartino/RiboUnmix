from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


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
