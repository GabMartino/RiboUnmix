from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional, Iterator, Sequence

import lightning as pl
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Sampler,
    SequentialSampler,
    WeightedRandomSampler,
)
from tqdm import tqdm

from Dataloaders.RiboAIQueuingMultiDataset.RiboAIQueuingMultiDataset import (
    RiboAIQueuingDatasetMultiDataset,
)


def open_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Utilities
# ============================================================


def _as_numpy_int(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)


def _as_numpy_str(x) -> np.ndarray:
    return np.asarray(x).astype(str)


def _dataset_pair_weights_from_ids(
    dataset_ids: Sequence[int] | np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Per-pair dataset balancing weights.

    Let N_d be the number of flat transcript-dataset pairs in dataset d.

        gamma = 0.0 -> all pairs have equal weight
        gamma = 1.0 -> total expected dataset mass is equalized

    Pair weight:

        w_{t,d} = N_d^{-gamma}
    """
    dataset_ids = _as_numpy_int(dataset_ids)
    gamma = float(gamma)

    unique_ids, counts = np.unique(dataset_ids, return_counts=True)
    count_map = {int(ds): float(n) for ds, n in zip(unique_ids, counts)}

    weights = np.asarray(
        [count_map[int(ds)] ** (-gamma) for ds in dataset_ids],
        dtype=np.float64,
    )

    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights, dtype=np.float64)

    return weights


def _transcript_pair_weights(
    *,
    flat_transcript_ids: Sequence[str] | np.ndarray,
    flat_dataset_ids: Sequence[int] | np.ndarray,
    dataset_balance_gamma: float = 0.0,
) -> np.ndarray:
    """
    Per-pair weights for a transcript-balanced flat-pair objective.

    For transcript t measured in k_t datasets, each observed pair gets weight

        1 / k_t

    so the total contribution of one transcript is approximately one unit,
    regardless of how many datasets measured it.

    Optional mild dataset balancing can be added through N_d^{-gamma}:

        w_{t,d} = (1 / k_t) * N_d^{-gamma}

    Recommended for your current setting:

        dataset_balance_gamma = 0.0 or 0.25

    Avoid gamma=1.0 unless you explicitly want equal dataset importance.
    """
    flat_transcript_ids = _as_numpy_str(flat_transcript_ids)
    flat_dataset_ids = _as_numpy_int(flat_dataset_ids)

    if len(flat_transcript_ids) != len(flat_dataset_ids):
        raise ValueError("flat_transcript_ids and flat_dataset_ids must have same length.")

    transcript_counts: dict[str, int] = defaultdict(int)
    for tid in flat_transcript_ids:
        transcript_counts[str(tid)] += 1

    transcript_factor = np.asarray(
        [1.0 / max(transcript_counts[str(tid)], 1) for tid in flat_transcript_ids],
        dtype=np.float64,
    )

    dataset_factor = _dataset_pair_weights_from_ids(
        flat_dataset_ids,
        gamma=float(dataset_balance_gamma),
    )

    weights = transcript_factor * dataset_factor
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    if weights.sum() <= 0:
        weights = np.ones_like(weights, dtype=np.float64)

    return weights


# ============================================================
# Samplers
# ============================================================


class RandomDatasetPerTranscriptSampler(Sampler[int]):
    """
    Main recommended sampler for partially overlapping datasets.

    It operates on a deterministic flat-pair dataset, where each item is a
    transcript-dataset pair.

    For each transcript, one available dataset-pair is sampled per epoch.

    If transcript t appears in k_t datasets, the sampler selects one of those
    k_t flat-pair indices. Therefore each transcript contributes approximately
    one training example per epoch, not k_t examples.

    This implements the objective:

        L = mean_t mean_{d in D(t)} loss(t, d)

    stochastically, without double-weighting paired transcripts.

    dataset_balance_gamma optionally changes the probability of choosing a
    dataset for transcripts with multiple dataset observations:

        p(d | t) proportional to N_d^{-gamma}

    where N_d is the number of flat pairs in dataset d.

        gamma = 0.0 -> uniform over available datasets for that transcript
        gamma = 0.5 -> mild preference for smaller datasets
        gamma = 1.0 -> strong equalizing pressure
    """

    def __init__(
        self,
        *,
        flat_transcript_ids: Sequence[str] | np.ndarray,
        flat_dataset_ids: Sequence[int] | np.ndarray,
        num_samples: Optional[int] = None,
        dataset_balance_gamma: float = 0.0,
        seed: int = 42,
        shuffle_transcripts: bool = True,
    ):
        self.flat_transcript_ids = _as_numpy_str(flat_transcript_ids)
        self.flat_dataset_ids = _as_numpy_int(flat_dataset_ids)
        self.dataset_balance_gamma = float(dataset_balance_gamma)
        self.seed = int(seed)
        self.shuffle_transcripts = bool(shuffle_transcripts)
        self._iter_count = 0

        if len(self.flat_transcript_ids) != len(self.flat_dataset_ids):
            raise ValueError("flat_transcript_ids and flat_dataset_ids must have same length.")

        groups: dict[str, list[int]] = defaultdict(list)
        for idx, tid in enumerate(self.flat_transcript_ids):
            groups[str(tid)].append(int(idx))

        self.transcript_ids = np.asarray(sorted(groups.keys()), dtype=object)
        self.groups = [np.asarray(groups[str(tid)], dtype=np.int64) for tid in self.transcript_ids]

        dataset_counts: dict[int, int] = defaultdict(int)
        for ds in self.flat_dataset_ids:
            dataset_counts[int(ds)] += 1
        self.dataset_counts = {int(k): int(v) for k, v in dataset_counts.items()}

        self.num_transcripts = len(self.groups)
        self.num_samples = int(num_samples) if num_samples is not None else self.num_transcripts

        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self._iter_count)
        self._iter_count += 1

        n_groups = self.num_transcripts

        if self.num_samples == n_groups:
            group_order = np.arange(n_groups, dtype=np.int64)
            if self.shuffle_transcripts:
                rng.shuffle(group_order)
        else:
            group_order = rng.integers(
                low=0,
                high=n_groups,
                size=self.num_samples,
                endpoint=False,
                dtype=np.int64,
            )

        for group_idx in group_order:
            pair_indices = self.groups[int(group_idx)]

            if pair_indices.size == 1:
                yield int(pair_indices[0])
                continue

            if self.dataset_balance_gamma <= 0.0:
                probs = None
            else:
                ds_ids = self.flat_dataset_ids[pair_indices]
                raw = np.asarray(
                    [
                        float(self.dataset_counts[int(ds)]) ** (-self.dataset_balance_gamma)
                        for ds in ds_ids
                    ],
                    dtype=np.float64,
                )
                raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                probs = raw / raw.sum() if raw.sum() > 0 else None

            chosen = rng.choice(pair_indices, size=1, replace=False, p=probs)
            yield int(chosen[0])

    def __len__(self) -> int:
        return self.num_samples


class DatasetAwareBatchSampler:
    """
    Legacy / explicit dataset-aware batch sampler.

    Each batch contains a controlled number of datasets.

    Example:
        batch_size = 8
        datasets_per_batch = 4

    gives approximately:
        2 samples from dataset A
        2 samples from dataset B
        2 samples from dataset C
        2 samples from dataset D

    This is useful when you explicitly want balanced multi-dataset batches.
    For your current setting, this can over-equalize Grimson/Kutay and may hurt
    the larger/cleaner dataset.
    """

    def __init__(
        self,
        sampler=None,   # accepted for Lightning DDP compat; not used internally
        *,
        dataset_ids,
        lengths,
        batch_size: int,
        datasets_per_batch: int,
        num_batches: int,
        gamma: float = 1.0,
        seed: int = 42,
        drop_last: bool = False,
        sort_by_length: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        self.dataset_ids = np.asarray(dataset_ids, dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)

        self.batch_size = int(batch_size)
        self.datasets_per_batch = int(datasets_per_batch)
        self.num_batches = int(num_batches)
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.sort_by_length = bool(sort_by_length)
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank) % self.num_replicas

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.datasets_per_batch <= 0:
            raise ValueError("datasets_per_batch must be positive.")
        if self.datasets_per_batch > self.batch_size:
            raise ValueError(
                f"datasets_per_batch={self.datasets_per_batch} cannot exceed "
                f"batch_size={self.batch_size}."
            )
        if len(self.dataset_ids) != len(self.lengths):
            raise ValueError("dataset_ids and lengths must have the same length.")

        self.unique_dataset_ids = np.unique(self.dataset_ids)
        self.indices_by_dataset = {
            int(ds): np.flatnonzero(self.dataset_ids == ds)
            for ds in self.unique_dataset_ids
        }

        counts = np.asarray(
            [len(self.indices_by_dataset[int(ds)]) for ds in self.unique_dataset_ids],
            dtype=np.float64,
        )

        # gamma = 0.0 -> sample datasets proportional to pair counts.
        # gamma = 1.0 -> sample datasets uniformly.
        probs = counts ** (1.0 - self.gamma)
        self.dataset_probs = probs / probs.sum()
        self._iter_count = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._iter_count)
        self._iter_count += 1

        n_datasets = len(self.unique_dataset_ids)
        d_per_batch = min(self.datasets_per_batch, n_datasets)

        base_k = self.batch_size // d_per_batch
        remainder = self.batch_size % d_per_batch

        batch_idx = 0
        for _ in range(self.num_batches):
            chosen_datasets = rng.choice(
                self.unique_dataset_ids,
                size=d_per_batch,
                replace=False,
                p=self.dataset_probs,
            )

            batch = []
            for j, ds in enumerate(chosen_datasets):
                ds = int(ds)
                k = base_k + (1 if j < remainder else 0)
                pool = self.indices_by_dataset[ds]
                sampled = rng.choice(pool, size=k, replace=True)
                batch.extend(sampled.tolist())

            if self.sort_by_length:
                batch = sorted(
                    batch,
                    key=lambda idx: int(self.lengths[idx]),
                    reverse=True,
                )

            if len(batch) == self.batch_size or not self.drop_last:
                batch_idx += 1
                if (batch_idx - 1) % self.num_replicas == self.rank:
                    yield batch

    def __len__(self):
        return (self.num_batches + self.num_replicas - 1 - self.rank) // self.num_replicas


class SortedLengthBatchSampler(BatchSampler):
    """
    Batch sampler that groups sampled indices by sequence length.

    This is important for variable-length transcripts because memory depends on
    the token budget, not only on batch_size.

    It sorts all sampled indices by length, chunks into batches, and optionally
    shuffles the order of length-homogeneous batches.
    """

    def __init__(
        self,
        sampler,
        batch_size: int,
        drop_last: bool = False,
        lengths=None,
        seed: int = 42,
        shuffle: bool = True,
        descending: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.descending = bool(descending)
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank) % self.num_replicas

        if lengths is None:
            raise ValueError("SortedLengthBatchSampler requires lengths.")

        self.lengths = np.asarray(lengths)
        self._iter_count = 0

    def __iter__(self):
        idx = np.fromiter(iter(self.sampler), dtype=np.int64)

        if idx.size == 0:
            return
            yield

        if idx.max(initial=0) >= len(self.lengths):
            raise IndexError(
                f"Sampler yielded index {idx.max()} but lengths has size {len(self.lengths)}."
            )

        order = np.argsort(self.lengths[idx], kind="stable")
        if self.descending:
            order = order[::-1]
        idx = idx[order]

        batches = [
            idx[i: i + self.batch_size].tolist()
            for i in range(0, len(idx), self.batch_size)
        ]

        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        if self.shuffle and len(batches) > 1:
            rng = np.random.default_rng(self.seed + self._iter_count)
            self._iter_count += 1
            batch_order = rng.permutation(len(batches))
            batches = [batches[int(j)] for j in batch_order]

        for i, batch in enumerate(batches):
            if i % self.num_replicas == self.rank:
                yield batch

    def __len__(self):
        n = len(self.sampler)
        total = n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
        return (total + self.num_replicas - 1 - self.rank) // self.num_replicas


class TranscriptGroupedMultiDatasetBatchSampler:
    """
    Batch sampler that keeps transcript-dataset pairs for the same transcript
    together.

    This is intended for multi-dataset biological-gradient methods such as
    CAGrad, where comparing dataset gradients is cleaner when each dataset sees
    matched transcript content inside the same batch.

    Each emitted batch contains flat pair indices. For a transcript t measured
    in datasets D(t), the sampler emits the group:

        [(t, d) for d in D(t)]

    as an atomic unit. Groups are packed into batches without splitting them.
    If require_multidataset is true, transcripts with fewer than two available
    dataset observations are skipped.
    """

    def __init__(
        self,
        *,
        flat_transcript_ids: Sequence[str] | np.ndarray,
        flat_dataset_ids: Sequence[int] | np.ndarray,
        lengths,
        batch_size: int,
        num_samples: Optional[int] = None,
        gamma: float = 0.0,
        seed: int = 42,
        drop_last: bool = False,
        sort_by_length: bool = True,
        require_multidataset: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        self.flat_transcript_ids = _as_numpy_str(flat_transcript_ids)
        self.flat_dataset_ids = _as_numpy_int(flat_dataset_ids)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples) if num_samples is not None else None
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.sort_by_length = bool(sort_by_length)
        self.require_multidataset = bool(require_multidataset)
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank) % self.num_replicas
        self._iter_count = 0

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(self.flat_transcript_ids) != len(self.flat_dataset_ids):
            raise ValueError("flat_transcript_ids and flat_dataset_ids must have same length.")
        if len(self.flat_transcript_ids) != len(self.lengths):
            raise ValueError("flat_transcript_ids and lengths must have same length.")

        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, tid in enumerate(self.flat_transcript_ids):
            grouped[str(tid)].append(int(idx))

        groups: list[np.ndarray] = []
        group_lengths: list[int] = []
        for tid in sorted(grouped):
            indices = np.asarray(grouped[tid], dtype=np.int64)
            dataset_count = len(np.unique(self.flat_dataset_ids[indices]))
            if self.require_multidataset and dataset_count < 2:
                continue
            groups.append(indices)
            group_lengths.append(int(self.lengths[indices[0]]))

        if len(groups) == 0:
            raise RuntimeError(
                "No transcript groups available for transcript-grouped sampling. "
                "If this split has no transcripts measured in multiple datasets, use "
                "train_sampling_strategy=transcript_grouped_pairs or another sampler."
            )

        self.groups = groups
        self.group_lengths = np.asarray(group_lengths, dtype=np.int64)
        self.group_pair_counts = np.asarray(
            [min(len(g), self.batch_size) for g in self.groups],
            dtype=np.int64,
        )

        dataset_counts: dict[int, int] = defaultdict(int)
        for ds in self.flat_dataset_ids:
            dataset_counts[int(ds)] += 1
        self.dataset_counts = {int(k): int(v) for k, v in dataset_counts.items()}

        # Per-epoch target number of (transcript, dataset) pairs when num_samples
        # is not given. Used by the group-selection-balancing path below.
        self._epoch_pairs = int(self.group_pair_counts.sum())

        # Group-SELECTION balancing weights. When gamma > 0, groups are drawn (with
        # replacement) in proportion to the rarity of the datasets they contain, so
        # transcripts measured in a rare dataset (e.g. the small one) are upsampled
        # instead of being flooded by single-dataset groups of the abundant dataset.
        # Weight of a group = sum over its member pairs of N_d^{-gamma}, reusing the
        # same N_d^{-gamma} scheme as _sample_group_indices and
        # _dataset_pair_weights_from_ids. gamma <= 0 -> None -> uniform selection
        # (unchanged legacy behavior).
        self.group_select_weights: Optional[np.ndarray] = None
        if self.gamma > 0.0:
            raw = np.asarray(
                [
                    sum(
                        float(self.dataset_counts[int(self.flat_dataset_ids[idx])]) ** (-self.gamma)
                        for idx in group
                    )
                    for group in self.groups
                ],
                dtype=np.float64,
            )
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            total = raw.sum()
            if total > 0:
                self.group_select_weights = raw / total

    def _sample_group_indices(
        self,
        group: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if group.size <= self.batch_size:
            return group.copy()

        if self.gamma <= 0.0:
            probs = None
        else:
            raw = np.asarray(
                [
                    float(self.dataset_counts[int(self.flat_dataset_ids[idx])]) ** (-self.gamma)
                    for idx in group
                ],
                dtype=np.float64,
            )
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            probs = raw / raw.sum() if raw.sum() > 0 else None

        sampled = rng.choice(group, size=self.batch_size, replace=False, p=probs)
        return np.asarray(sampled, dtype=np.int64)

    def _select_groups(
        self,
        rng: np.random.Generator,
    ) -> list[tuple[int, np.ndarray]]:
        if self.group_select_weights is not None:
            # Dataset-rarity-weighted selection with replacement. Fill until the
            # target pair count is reached; groups containing rare datasets are
            # upsampled so per-epoch per-dataset representation is balanced.
            target_pairs = (
                int(self.num_samples)
                if self.num_samples is not None
                else self._epoch_pairs
            )
            order: list[int] = []
            n_pairs = 0
            num_groups = len(self.groups)
            # Sample in chunks to avoid one rng call per group.
            chunk = max(num_groups, 1)
            while n_pairs < target_pairs:
                draws = rng.choice(
                    num_groups, size=chunk, replace=True, p=self.group_select_weights
                )
                for group_idx in draws:
                    order.append(int(group_idx))
                    n_pairs += int(self.group_pair_counts[int(group_idx)])
                    if n_pairs >= target_pairs:
                        break
            group_order = np.asarray(order, dtype=np.int64)
        elif self.num_samples is None:
            group_order = rng.permutation(len(self.groups))
        else:
            order = []
            n_pairs = 0
            while n_pairs < self.num_samples:
                group_idx = int(rng.integers(0, len(self.groups)))
                order.append(group_idx)
                n_pairs += int(self.group_pair_counts[group_idx])
            group_order = np.asarray(order, dtype=np.int64)

        selected = []
        for group_idx in group_order:
            indices = self._sample_group_indices(self.groups[int(group_idx)], rng)
            length = int(self.group_lengths[int(group_idx)])
            selected.append((length, indices))

        if self.sort_by_length:
            selected.sort(key=lambda item: item[0], reverse=True)

        return selected

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._iter_count)
        self._iter_count += 1

        selected_groups = self._select_groups(rng)

        batches: list[list[int]] = []
        batch: list[int] = []

        for _, group in selected_groups:
            group_list = group.tolist()

            if batch and len(batch) + len(group_list) > self.batch_size:
                batches.append(batch)
                batch = []

            batch.extend(group_list)

        if batch and (len(batch) == self.batch_size or not self.drop_last):
            batches.append(batch)

        if len(batches) > 1:
            batch_order = rng.permutation(len(batches))
            batches = [batches[int(i)] for i in batch_order]

        for i, batch in enumerate(batches):
            if i % self.num_replicas == self.rank:
                yield batch

    def __len__(self):
        if self.group_select_weights is not None:
            # Weighted-with-replacement selection draws ~target_pairs pairs/epoch.
            target_pairs = (
                int(self.num_samples)
                if self.num_samples is not None
                else self._epoch_pairs
            )
            n_batches = target_pairs // self.batch_size if self.drop_last else (
                target_pairs + self.batch_size - 1
            ) // self.batch_size
        elif self.num_samples is None:
            order = np.arange(len(self.groups), dtype=np.int64)
            if self.sort_by_length:
                order = order[np.argsort(self.group_lengths[order], kind="stable")[::-1]]

            n_batches = 0
            batch_len = 0
            for group_idx in order:
                group_len = int(self.group_pair_counts[int(group_idx)])
                if batch_len and batch_len + group_len > self.batch_size:
                    n_batches += 1
                    batch_len = 0
                batch_len += group_len

            if batch_len and (batch_len == self.batch_size or not self.drop_last):
                n_batches += 1
        else:
            n_batches = self.num_samples // self.batch_size if self.drop_last else (
                self.num_samples + self.batch_size - 1
            ) // self.batch_size

        return (n_batches + self.num_replicas - 1 - self.rank) // self.num_replicas


# ============================================================
# DataModule
# ============================================================


class RiboAIQueuingDatamoduleMultiDataset(pl.LightningDataModule):
    """
    DataModule for multi-dataset ribo-seq profile training.

    Recommended current strategy for partially overlapping datasets:

        train_sampling_strategy: random_dataset_per_transcript
        dataset_balance_gamma: 0.0 or 0.25
        dataset_aware_batching: false

    Why:
        - Each transcript contributes approximately once per epoch.
        - Common transcripts are not double-weighted just because they appear in
          two datasets.
        - Dataset choice for common transcripts is stochastic across epochs.
        - Optional gamma gives mild preference to smaller datasets without fully
          equalizing them.

    Available train_sampling_strategy values:

        random_dataset_per_transcript
            Recommended. Uses deterministic flat pairs internally, but samples
            one available dataset-pair per transcript per epoch.

        transcript_balanced_pairs
            Samples flat pairs with replacement using weight 1/k_t for each
            transcript-dataset pair. Similar objective, less exact per epoch.

        flat_pairs
            Full deterministic flat-pair pass. Common transcripts count once per
            dataset observation.

        dataset_balanced_pairs
            Weighted flat-pair sampling with N_d^{-gamma}. This is the previous
            balanced_train_sampling=True behavior without dataset-aware batches.

        dataset_aware_balanced_pairs
            Explicitly balanced multi-dataset batches. Strong equalization.

        transcript_grouped_pairs
            Groups all available dataset observations for each sampled transcript
            into the same batch. Single-dataset transcripts are included.

        transcript_grouped_multidataset_pairs
            Same as transcript_grouped_pairs, but only uses transcripts measured
            in at least two datasets. This gives matched multi-dataset ground
            truth per transcript inside each batch, which is useful for CAGrad.

        legacy_random_dataset
            Old behavior: dataset_choice_mode='random' inside __getitem__.
            Kept for comparison, but less reproducible with persistent workers.

    Backward compatibility:
        If train_sampling_strategy is None:
            - balanced_train_sampling=False -> random_dataset_per_transcript
            - balanced_train_sampling=True and dataset_aware_batching=False
                -> dataset_balanced_pairs
            - balanced_train_sampling=True and dataset_aware_batching=True
                -> dataset_aware_balanced_pairs
    """

    def __init__(
        self,
        sequences_path: str,
        datasets_paths: list[str],
        batch_size: int,
        split: tuple | None,
        nt_encoding_path: str,
        codon_to_aa_encoding_path: str,
        codon_encoding_path: str,
        aa_encoding_path: str,
        datasets_encoding_path: str,
        split_p: float = 0.9,
        num_workers: int = 4,
        seed: int = 42,
        balanced_train_sampling: bool = False,
        dataset_balance_gamma: float = 0.0,
        train_samples_per_epoch: Optional[int] = None,
        dataset_aware_batching: bool = False,
        datasets_per_batch: int = 4,
        train_sampling_strategy: Optional[str] = None,
        pin_memory: bool = True,
        prefetch_factor: Optional[int] = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.sequences_path = sequences_path
        self.datasets_paths = list(datasets_paths)
        self.batch_size = int(batch_size)
        self.split = split
        self.split_p = float(split_p)
        self.num_workers = int(num_workers)
        self.seed = int(seed)

        self.balanced_train_sampling = bool(balanced_train_sampling)
        self.dataset_balance_gamma = float(dataset_balance_gamma)
        self.train_samples_per_epoch = train_samples_per_epoch

        self.dataset_aware_batching = bool(dataset_aware_batching)
        self.datasets_per_batch = int(datasets_per_batch)
        self.train_sampling_strategy = train_sampling_strategy

        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = prefetch_factor

        self.nt_enc = open_file(nt_encoding_path)
        self.c2aa_enc = open_file(codon_to_aa_encoding_path)
        self.c_enc = open_file(codon_encoding_path)
        self.aa_enc = open_file(aa_encoding_path)
        self.datasets_enc = open_file(datasets_encoding_path)

        self.train_dataset_obj = None
        self.val_dataset_obj = None

        self.train_lengths = None
        self.val_lengths = None

        self.train_flat_dataset_ids = None
        self.train_flat_transcript_ids = None
        self.train_pair_weights = None

        self._has_loaded_data = False

    # ------------------------------------------------------------
    # Strategy resolution / flat metadata
    # ------------------------------------------------------------

    def _resolve_train_sampling_strategy(self) -> str:
        if self.train_sampling_strategy is not None:
            return str(self.train_sampling_strategy).lower()

        if self.balanced_train_sampling:
            if self.dataset_aware_batching:
                return "dataset_aware_balanced_pairs"
            return "dataset_balanced_pairs"

        # New default replacing the old random __getitem__ dataset selection.
        return "random_dataset_per_transcript"

    def _training_dataset_choice_mode(self) -> str:
        strategy = self._resolve_train_sampling_strategy()
        if strategy == "legacy_random_dataset":
            return "random"
        return "deterministic"

    def _get_flat_dataset_ids(self, dataset_obj) -> np.ndarray:
        if hasattr(dataset_obj, "flat_dataset_ids"):
            return _as_numpy_int(dataset_obj.flat_dataset_ids)
        raise AttributeError(
            "RiboAIQueuingDatasetMultiDataset must expose flat_dataset_ids in "
            "deterministic mode. Your existing code already appears to use this "
            "attribute for balanced sampling."
        )

    def _get_flat_lengths(self, dataset_obj) -> np.ndarray:
        if hasattr(dataset_obj, "flat_lengths"):
            return np.asarray(dataset_obj.flat_lengths, dtype=np.int32)
        raise AttributeError(
            "RiboAIQueuingDatasetMultiDataset must expose flat_lengths in deterministic mode."
        )

    def _get_flat_transcript_ids(self, dataset_obj) -> np.ndarray:
        """
        Tries to recover one transcript ID per flat pair.

        Best fix in the dataset class:
            self.flat_transcript_ids = [...]

        This helper includes fallbacks for common attribute names, but if none
        exist, add flat_transcript_ids to RiboAIQueuingDatasetMultiDataset when
        building flat pairs.
        """
        candidate_attrs = [
            "flat_transcript_ids",
            "flat_ids",
            "flat_transcripts",
            "flat_transcript_id",
        ]

        for attr in candidate_attrs:
            if hasattr(dataset_obj, attr):
                val = getattr(dataset_obj, attr)
                arr = _as_numpy_str(val)
                if len(arr) == len(dataset_obj):
                    return arr

        # Fallback: infer from flat_indices / flat_global_indices plus transcript_ids.
        index_attrs = [
            "flat_transcript_indices",
            "flat_global_indices",
            "flat_ref_indices",
            "flat_indices",
        ]
        transcript_id_attrs = ["transcripts_ids", "transcript_ids", "ids"]

        for idx_attr in index_attrs:
            if not hasattr(dataset_obj, idx_attr):
                continue
            flat_idx = _as_numpy_int(getattr(dataset_obj, idx_attr))
            if len(flat_idx) != len(dataset_obj):
                continue

            for tid_attr in transcript_id_attrs:
                if hasattr(dataset_obj, tid_attr):
                    tids = _as_numpy_str(getattr(dataset_obj, tid_attr))
                    if flat_idx.max(initial=0) < len(tids):
                        return tids[flat_idx]

        # Fallback for flat_pairs list/dict.
        if hasattr(dataset_obj, "flat_pairs"):
            pairs = getattr(dataset_obj, "flat_pairs")
            tids = []
            for p in pairs:
                if isinstance(p, dict):
                    for key in ["transcript_id", "id", "tid"]:
                        if key in p:
                            tids.append(str(p[key]))
                            break
                    else:
                        raise AttributeError("Could not find transcript id key in flat_pairs dict.")
                elif isinstance(p, (tuple, list)) and len(p) >= 1:
                    # Prefer first element if it is string-like; otherwise second.
                    if isinstance(p[0], str):
                        tids.append(str(p[0]))
                    elif len(p) >= 2 and isinstance(p[1], str):
                        tids.append(str(p[1]))
                    else:
                        raise AttributeError(
                            "flat_pairs exists, but tuple layout is not recognized. "
                            "Add dataset_obj.flat_transcript_ids explicitly."
                        )
                else:
                    raise AttributeError("flat_pairs layout is not recognized.")
            arr = _as_numpy_str(tids)
            if len(arr) == len(dataset_obj):
                return arr

        raise AttributeError(
            "Could not infer flat_transcript_ids. Add this attribute to "
            "RiboAIQueuingDatasetMultiDataset when building deterministic flat pairs:\n\n"
            "    self.flat_transcript_ids = np.asarray([...], dtype=str)\n\n"
            "It must have length == len(dataset_obj), one transcript ID per flat pair."
        )

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    def setup(self, stage=None):
        if self._has_loaded_data:
            print(f"Data already in memory. Skipping load for stage: {stage}")
            return

        print(f"Loading data from disk for stage: {stage}")
        self._has_loaded_data = True

        print(f"Unioning master sequences with {len(self.datasets_paths)} datasets...")

        seq_df = pd.read_parquet(self.sequences_path)
        if "transcript_id" in seq_df.columns:
            seq_df = seq_df.set_index("transcript_id")
        seq_df.index = seq_df.index.astype(str)

        print("Length of the main sequence:", len(seq_df.index))

        loaded_datasets = {}
        union_index = pd.Index([], dtype=seq_df.index.dtype)

        for path in tqdm(self.datasets_paths, desc="Loading ribo datasets"):
            df = pd.read_parquet(path)

            if "id" not in df.columns:
                raise KeyError(f"'id' column missing in {path}")
            if "ribo" not in df.columns:
                raise KeyError(f"'ribo' column missing in {path}")

            df = df.set_index("id")
            df.index = df.index.astype(str)

            dataset_name = os.path.basename(path).split(".")[0]
            loaded_datasets[dataset_name] = df[["ribo"]]
            union_index = union_index.union(df.index, sort=False)

        valid_index = seq_df.index.intersection(union_index, sort=False)
        if len(valid_index) == 0:
            raise RuntimeError(
                "No transcript IDs overlap between sequence table and ribo datasets."
            )

        seq_df_union = seq_df.loc[valid_index]
        ref_arrays = seq_df_union["ref"].values

        css_col = (
            "conserved_stalling_sites"
            if "conserved_stalling_sites" in seq_df_union.columns
            else "css"
        )
        if css_col not in seq_df_union.columns:
            raise KeyError(
                "Could not find CSS column. Expected either 'conserved_stalling_sites' or 'css'."
            )

        css = seq_df_union[css_col].values
        lengths = np.asarray([len(x) for x in ref_arrays], dtype=np.int32)

        shared_data = {
            "transcript_id": valid_index.values.astype(str),
            "ref": ref_arrays,
            "css": css,
            "ribo_profiles": defaultdict(dict),
            "lengths": lengths,
            "datasets_names": list(loaded_datasets.keys()),
        }

        valid_ids = set(valid_index.astype(str))
        for dataset_name, df in loaded_datasets.items():
            for t_id, ribo_profile in zip(df.index.astype(str), df["ribo"].values):
                if t_id in valid_ids:
                    shared_data["ribo_profiles"][t_id][dataset_name] = ribo_profile

        if self.split is None:
            raise NotImplementedError(
                "Random split is not implemented here. Pass a transcript-level split."
            )

        print("Using provided split transcript IDs.")

        all_ids = np.asarray(shared_data["transcript_id"]).astype(str)
        train_id_set = set(map(str, self.split[0]))
        val_id_set = set(map(str, self.split[1]))

        train_indices = [i for i, tid in enumerate(all_ids) if tid in train_id_set]
        val_indices = [i for i, tid in enumerate(all_ids) if tid in val_id_set]

        train_ids = all_ids[train_indices].tolist()
        val_ids = all_ids[val_indices].tolist()

        if len(train_ids) == 0:
            raise RuntimeError("Training split is empty after intersecting with available transcripts.")
        if len(val_ids) == 0:
            raise RuntimeError("Validation split is empty after intersecting with available transcripts.")

        print(f"Train transcripts: {len(train_ids)}")
        print(f"Validation transcripts: {len(val_ids)}")

        strategy = self._resolve_train_sampling_strategy()
        train_choice_mode = self._training_dataset_choice_mode()

        print("\n=== Training sampling strategy ===")
        print(f"strategy: {strategy}")
        print(f"dataset_choice_mode: {train_choice_mode}")
        print(f"dataset_balance_gamma: {self.dataset_balance_gamma}")
        print(f"train_samples_per_epoch: {self.train_samples_per_epoch}")

        self.train_dataset_obj = RiboAIQueuingDatasetMultiDataset(
            data=shared_data,
            lengths=shared_data["lengths"],
            nt_encoding=self.nt_enc,
            codon_to_aa_encoding=self.c2aa_enc,
            codon_encoding=self.c_enc,
            aa_encoding=self.aa_enc,
            datasets_encoding=self.datasets_enc,
            transcripts_ids=train_ids,
            dataset_choice_mode=train_choice_mode,
            seed=self.seed,
        )

        if train_choice_mode == "deterministic":
            self.train_lengths = self._get_flat_lengths(self.train_dataset_obj)
            self.train_flat_dataset_ids = self._get_flat_dataset_ids(self.train_dataset_obj)
            self.train_flat_transcript_ids = self._get_flat_transcript_ids(self.train_dataset_obj)

            self.train_pair_weights = _transcript_pair_weights(
                flat_transcript_ids=self.train_flat_transcript_ids,
                flat_dataset_ids=self.train_flat_dataset_ids,
                dataset_balance_gamma=self.dataset_balance_gamma,
            )

            self._print_flat_pair_summary(
                dataset_obj=self.train_dataset_obj,
                flat_transcript_ids=self.train_flat_transcript_ids,
                flat_dataset_ids=self.train_flat_dataset_ids,
                split_name="train",
            )
        else:
            # Legacy random mode: one item per transcript, dataset selected in __getitem__.
            self.train_lengths = lengths[train_indices]
            self.train_flat_dataset_ids = None
            self.train_flat_transcript_ids = None
            self.train_pair_weights = None

            print("\n=== Legacy random dataset mode ===")
            print("Each transcript appears once per epoch; dataset is randomly chosen inside __getitem__. ")
            print("For reproducibility, prefer random_dataset_per_transcript instead.")

        self.val_dataset_obj = RiboAIQueuingDatasetMultiDataset(
            data=shared_data,
            lengths=shared_data["lengths"],
            nt_encoding=self.nt_enc,
            codon_to_aa_encoding=self.c2aa_enc,
            codon_encoding=self.c_enc,
            aa_encoding=self.aa_enc,
            datasets_encoding=self.datasets_enc,
            transcripts_ids=val_ids,
            dataset_choice_mode="deterministic",
            seed=self.seed,
        )

        self.val_lengths = self._get_flat_lengths(self.val_dataset_obj)

        val_flat_dataset_ids = self._get_flat_dataset_ids(self.val_dataset_obj)
        val_flat_transcript_ids = self._get_flat_transcript_ids(self.val_dataset_obj)
        self._print_flat_pair_summary(
            dataset_obj=self.val_dataset_obj,
            flat_transcript_ids=val_flat_transcript_ids,
            flat_dataset_ids=val_flat_dataset_ids,
            split_name="validation",
        )

    def _print_flat_pair_summary(
        self,
        *,
        dataset_obj,
        flat_transcript_ids: np.ndarray,
        flat_dataset_ids: np.ndarray,
        split_name: str,
    ) -> None:
        print(f"\n=== {split_name.capitalize()} flat-pair summary ===")
        print(f"flat transcript-dataset pairs: {len(dataset_obj)}")
        print(f"unique transcripts: {len(np.unique(flat_transcript_ids))}")

        counts_by_ds = defaultdict(int)
        for ds in flat_dataset_ids:
            counts_by_ds[int(ds)] += 1

        print("dataset pair counts:")
        for ds_id, n in sorted(counts_by_ds.items()):
            ds_name = getattr(dataset_obj, "idx_to_dataset", {}).get(ds_id, str(ds_id))
            print(f"  {ds_name:35s} id={ds_id:3d} pairs={n}")

        k_by_t = defaultdict(int)
        for tid in flat_transcript_ids:
            k_by_t[str(tid)] += 1

        k_values = np.asarray(list(k_by_t.values()), dtype=np.int64)
        unique_k, k_counts = np.unique(k_values, return_counts=True)
        print("transcripts by number of available datasets:")
        for k, n in zip(unique_k, k_counts):
            print(f"  k={int(k):2d}: transcripts={int(n)}")

    # ------------------------------------------------------------
    # Workers / DataLoaders
    # ------------------------------------------------------------

    def worker_init_fn(self, worker_id: int):
        # With persistent_workers=True this is called when workers are created,
        # not necessarily every epoch. The new recommended sampler makes dataset
        # choice in the main process, so this seed is mainly for dataset-level
        # augmentation/caching randomness if any remains.
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        worker_seed = self.seed + worker_id + int(epoch) * 1000
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)

    def _dist_info(self) -> tuple[int, int]:
        """Return (num_replicas, rank) for the current distributed context."""
        if self.trainer is not None and self.trainer.world_size > 1:
            return self.trainer.world_size, self.trainer.global_rank
        return 1, 0

    def _dataloader_kwargs(self):
        kwargs = {
            "num_workers": self.num_workers,
            "persistent_workers": (self.num_workers > 0),
            "worker_init_fn": self.worker_init_fn,
            "pin_memory": self.pin_memory,
        }

        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.prefetch_factor)

        return kwargs

    def train_dataloader(self):
        if self.train_dataset_obj is None:
            raise RuntimeError("setup() must be called before train_dataloader().")

        strategy = self._resolve_train_sampling_strategy()
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        seed = self.seed + int(epoch)
        num_replicas, rank = self._dist_info()

        if strategy == "legacy_random_dataset":
            base_sampler = SequentialSampler(self.train_dataset_obj)

            batch_sampler = SortedLengthBatchSampler(
                sampler=base_sampler,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=True,
                lengths=self.train_lengths,
                seed=seed,
                descending=True,
                num_replicas=num_replicas,
                rank=rank,
            )

        elif strategy == "random_dataset_per_transcript":
            if self.train_flat_transcript_ids is None or self.train_flat_dataset_ids is None:
                raise RuntimeError("random_dataset_per_transcript requires deterministic flat metadata.")

            base_sampler = RandomDatasetPerTranscriptSampler(
                flat_transcript_ids=self.train_flat_transcript_ids,
                flat_dataset_ids=self.train_flat_dataset_ids,
                num_samples=self.train_samples_per_epoch,
                dataset_balance_gamma=self.dataset_balance_gamma,
                seed=seed,
                shuffle_transcripts=True,
            )

            batch_sampler = SortedLengthBatchSampler(
                sampler=base_sampler,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=True,
                lengths=self.train_lengths,
                seed=seed,
                descending=True,
                num_replicas=num_replicas,
                rank=rank,
            )

        elif strategy == "transcript_balanced_pairs":
            if self.train_pair_weights is None:
                raise RuntimeError("transcript_balanced_pairs requires deterministic flat metadata.")

            num_samples = (
                int(self.train_samples_per_epoch)
                if self.train_samples_per_epoch is not None
                else len(self.train_dataset_obj)
            )

            generator = torch.Generator().manual_seed(seed)
            base_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(self.train_pair_weights, dtype=torch.double),
                num_samples=num_samples,
                replacement=True,
                generator=generator,
            )

            batch_sampler = SortedLengthBatchSampler(
                sampler=base_sampler,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=True,
                lengths=self.train_lengths,
                seed=seed,
                descending=True,
                num_replicas=num_replicas,
                rank=rank,
            )

        elif strategy == "flat_pairs":
            if self.train_samples_per_epoch is None:
                base_sampler = SequentialSampler(self.train_dataset_obj)
            else:
                generator = torch.Generator().manual_seed(seed)
                base_sampler = WeightedRandomSampler(
                    weights=torch.ones(len(self.train_dataset_obj), dtype=torch.double),
                    num_samples=int(self.train_samples_per_epoch),
                    replacement=True,
                    generator=generator,
                )

            batch_sampler = SortedLengthBatchSampler(
                sampler=base_sampler,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=True,
                lengths=self.train_lengths,
                seed=seed,
                descending=True,
                num_replicas=num_replicas,
                rank=rank,
            )

        elif strategy == "dataset_balanced_pairs":
            if self.train_flat_dataset_ids is None:
                raise RuntimeError("dataset_balanced_pairs requires deterministic flat metadata.")

            num_samples = (
                int(self.train_samples_per_epoch)
                if self.train_samples_per_epoch is not None
                else len(self.train_dataset_obj)
            )

            weights = _dataset_pair_weights_from_ids(
                self.train_flat_dataset_ids,
                gamma=self.dataset_balance_gamma,
            )

            generator = torch.Generator().manual_seed(seed)
            base_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=num_samples,
                replacement=True,
                generator=generator,
            )

            batch_sampler = SortedLengthBatchSampler(
                sampler=base_sampler,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=True,
                lengths=self.train_lengths,
                seed=seed,
                descending=True,
                num_replicas=num_replicas,
                rank=rank,
            )

        elif strategy == "dataset_aware_balanced_pairs":
            if self.train_flat_dataset_ids is None:
                raise RuntimeError("dataset_aware_balanced_pairs requires deterministic flat metadata.")

            num_samples = (
                int(self.train_samples_per_epoch)
                if self.train_samples_per_epoch is not None
                else len(self.train_dataset_obj)
            )
            num_batches = (num_samples + self.batch_size - 1) // self.batch_size

            batch_sampler = DatasetAwareBatchSampler(
                dataset_ids=self.train_flat_dataset_ids,
                lengths=self.train_lengths,
                batch_size=self.batch_size,
                datasets_per_batch=self.datasets_per_batch,
                num_batches=num_batches,
                gamma=self.dataset_balance_gamma,
                seed=seed,
                drop_last=False,
                sort_by_length=True,
                num_replicas=num_replicas,
                rank=rank,
            )

        elif strategy in {"transcript_grouped_pairs", "transcript_grouped_multidataset_pairs"}:
            if self.train_flat_transcript_ids is None or self.train_flat_dataset_ids is None:
                raise RuntimeError(f"{strategy} requires deterministic flat metadata.")

            batch_sampler = TranscriptGroupedMultiDatasetBatchSampler(
                flat_transcript_ids=self.train_flat_transcript_ids,
                flat_dataset_ids=self.train_flat_dataset_ids,
                lengths=self.train_lengths,
                batch_size=self.batch_size,
                num_samples=self.train_samples_per_epoch,
                gamma=self.dataset_balance_gamma,
                seed=seed,
                drop_last=False,
                sort_by_length=True,
                require_multidataset=(strategy == "transcript_grouped_multidataset_pairs"),
                num_replicas=num_replicas,
                rank=rank,
            )

        else:
            raise ValueError(
                f"Unknown train_sampling_strategy={strategy!r}. Supported: "
                "random_dataset_per_transcript, transcript_balanced_pairs, flat_pairs, "
                "dataset_balanced_pairs, dataset_aware_balanced_pairs, "
                "transcript_grouped_pairs, transcript_grouped_multidataset_pairs, "
                "legacy_random_dataset."
            )

        return DataLoader(
            self.train_dataset_obj,
            batch_sampler=batch_sampler,
            collate_fn=self.train_dataset_obj.collate_fn,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self):
        if self.val_dataset_obj is None:
            raise RuntimeError("setup() must be called before val_dataloader().")

        batch_sampler = SortedLengthBatchSampler(
            sampler=SequentialSampler(self.val_dataset_obj),
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            lengths=self.val_lengths,
            seed=self.seed,
            descending=True,
        )

        return DataLoader(
            self.val_dataset_obj,
            batch_sampler=batch_sampler,
            collate_fn=self.val_dataset_obj.collate_fn,
            **self._dataloader_kwargs(),
        )

    def predict_dataloader(self):
        if self.val_dataset_obj is None:
            raise RuntimeError("setup() must be called before predict_dataloader().")

        num_replicas, rank = self._dist_info()

        batch_sampler = SortedLengthBatchSampler(
            sampler=SequentialSampler(self.val_dataset_obj),
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            lengths=self.val_lengths,
            seed=self.seed,
            descending=True,
            num_replicas=num_replicas,
            rank=rank,
        )

        return DataLoader(
            self.val_dataset_obj,
            batch_sampler=batch_sampler,
            collate_fn=self.val_dataset_obj.collate_fn,
            **self._dataloader_kwargs(),
        )
