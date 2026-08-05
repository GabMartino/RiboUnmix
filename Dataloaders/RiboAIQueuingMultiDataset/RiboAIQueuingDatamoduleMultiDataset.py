from __future__ import annotations

import math
import os
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import lightning as pl
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Sampler,
    SequentialSampler,
)
from tqdm import tqdm

from Dataloaders.RiboAIQueuingMultiDataset.RiboAIQueuingMultiDataset import (
    RiboAIQueuingDatasetMultiDataset,
)


RIBO_REPLICAS_COLUMN = "ribo_cds_replicas"


@dataclass(frozen=True)
class GroupedBatchStatistics:
    """Non-mutating summary of one transcript-grouped sampler plan."""

    iteration_index: int
    number_of_microbatches: int
    eligible_transcript_groups: int
    physical_pair_capacity: int
    pair_rows_per_microbatch: tuple[int, ...]
    unique_transcripts_per_microbatch: tuple[int, ...]
    datasets_per_transcript_group: tuple[int, ...]
    minimum_group_size: int
    median_group_size: float
    mean_group_size: float
    maximum_group_size: int
    minimum_unique_transcripts_per_microbatch: int
    median_unique_transcripts_per_microbatch: float
    mean_unique_transcripts_per_microbatch: float
    maximum_unique_transcripts_per_microbatch: int

    @property
    def median_pair_rows_per_microbatch(self) -> float:
        if not self.pair_rows_per_microbatch:
            return 0.0
        return float(np.median(np.asarray(self.pair_rows_per_microbatch, dtype=float)))

    @property
    def mean_pair_rows_per_microbatch(self) -> float:
        if not self.pair_rows_per_microbatch:
            return 0.0
        return float(np.mean(np.asarray(self.pair_rows_per_microbatch, dtype=float)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroupedOptimizerBatchPlan:
    """Resolved Lightning gradient-accumulation plan for grouped batches."""

    enabled: bool
    configured_target_unique_transcripts: int
    effective_local_target_unique_transcripts: int
    target_scope: str
    world_size: int
    accumulation_statistic: str
    estimated_unique_transcripts_per_microbatch: float
    resolved_accumulate_grad_batches: int
    estimated_unique_transcripts_per_optimizer_step: float
    estimated_global_unique_transcripts_per_optimizer_step: float
    estimated_pair_rows_per_optimizer_step: float
    microbatches_per_epoch_per_rank: int
    estimated_optimizer_steps_per_epoch: int
    expected_optimizer_steps_from_transcript_target: int
    optimizer_step_expectation_ratio: float
    accumulation_clamped_at_maximum: bool
    physical_pair_microbatch_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _select_grouped_accumulation_statistic(
    statistics: GroupedBatchStatistics,
    statistic: str,
) -> float:
    values = np.asarray(
        statistics.unique_transcripts_per_microbatch,
        dtype=np.float64,
    )
    if values.size == 0:
        raise RuntimeError("Cannot resolve accumulation from an empty batch plan.")
    name = str(statistic).strip().lower()
    if name == "median":
        return float(np.median(values))
    if name == "mean":
        return float(np.mean(values))
    if name == "p25":
        return float(np.percentile(values, 25.0))
    if name == "minimum":
        return float(np.min(values))
    raise ValueError(
        "accumulation_statistic must be one of median, mean, p25, minimum; "
        f"got {statistic!r}."
    )


def resolve_grouped_optimizer_batch_plan(
    statistics: GroupedBatchStatistics,
    *,
    target_unique_transcripts_per_optimizer_step: int = 32,
    accumulation_statistic: str = "median",
    min_accumulate_grad_batches: int = 1,
    max_accumulate_grad_batches: int = 32,
    target_scope: str = "per_rank",
    world_size: int = 1,
    forced_accumulate_grad_batches: int | None = None,
) -> GroupedOptimizerBatchPlan:
    """Resolve a common accumulation factor from an actual grouped epoch plan."""

    configured_target = int(target_unique_transcripts_per_optimizer_step)
    if configured_target <= 0:
        raise ValueError("target_unique_transcripts_per_optimizer_step must be positive.")
    world_size = max(int(world_size), 1)
    scope = str(target_scope).strip().lower()
    if scope == "per_rank":
        local_target = configured_target
    elif scope == "global":
        local_target = max(1, math.ceil(configured_target / world_size))
    else:
        raise ValueError("target_scope must be 'per_rank' or 'global'.")

    minimum = int(min_accumulate_grad_batches)
    maximum = int(max_accumulate_grad_batches)
    if minimum <= 0 or maximum < minimum:
        raise ValueError(
            "Need 1 <= min_accumulate_grad_batches <= max_accumulate_grad_batches."
        )

    groups_per_microbatch = _select_grouped_accumulation_statistic(
        statistics,
        accumulation_statistic,
    )
    raw_accumulation = math.ceil(local_target / max(groups_per_microbatch, 1.0))
    if forced_accumulate_grad_batches is None:
        accumulation = min(max(raw_accumulation, minimum), maximum)
        clamped_at_maximum = raw_accumulation > maximum
    else:
        accumulation = int(forced_accumulate_grad_batches)
        if accumulation <= 0:
            raise ValueError("forced_accumulate_grad_batches must be positive.")
        clamped_at_maximum = False
    if clamped_at_maximum:
        warnings.warn(
            "Grouped optimizer accumulation hit max_accumulate_grad_batches="
            f"{maximum}; the estimated {accumulation * groups_per_microbatch:.2f} "
            "unique transcripts per local optimizer step is below the requested "
            f"target {local_target}.",
            RuntimeWarning,
            stacklevel=2,
        )

    global_microbatches = int(statistics.number_of_microbatches)
    local_microbatches = math.ceil(global_microbatches / world_size)
    optimizer_steps = math.ceil(local_microbatches / accumulation)
    selected_group_occurrences = int(
        sum(statistics.unique_transcripts_per_microbatch)
    )
    local_group_occurrences = math.ceil(selected_group_occurrences / world_size)
    expected_steps = math.ceil(local_group_occurrences / max(local_target, 1))
    ratio = optimizer_steps / max(expected_steps, 1)
    estimated_local_groups = accumulation * groups_per_microbatch

    return GroupedOptimizerBatchPlan(
        enabled=True,
        configured_target_unique_transcripts=configured_target,
        effective_local_target_unique_transcripts=local_target,
        target_scope=scope,
        world_size=world_size,
        accumulation_statistic=str(accumulation_statistic).strip().lower(),
        estimated_unique_transcripts_per_microbatch=groups_per_microbatch,
        resolved_accumulate_grad_batches=accumulation,
        estimated_unique_transcripts_per_optimizer_step=estimated_local_groups,
        estimated_global_unique_transcripts_per_optimizer_step=(
            estimated_local_groups * world_size
        ),
        estimated_pair_rows_per_optimizer_step=(
            accumulation * statistics.median_pair_rows_per_microbatch
        ),
        microbatches_per_epoch_per_rank=local_microbatches,
        estimated_optimizer_steps_per_epoch=optimizer_steps,
        expected_optimizer_steps_from_transcript_target=expected_steps,
        optimizer_step_expectation_ratio=float(ratio),
        accumulation_clamped_at_maximum=clamped_at_maximum,
        physical_pair_microbatch_size=int(statistics.physical_pair_capacity),
    )


def open_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed NumPy from PyTorch's per-worker seed without capturing CUDA state."""
    del worker_id  # The worker id is already incorporated into initial_seed().
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def load_dataset_quality_ranking(
    path: str,
    *,
    dataset_column: str = "dataset",
    rank_column: str = "quality_rank",
) -> tuple[dict[str, float], dict[str, float]]:
    """Load dataset ranks and convert them to strictly positive quality weights.

    Rank 1 is best.  With ``R`` equal to the largest rank in the complete table,
    the deterministic conversion is ``quality_weight = (R - rank + 1) / R``.
    Consequently the best dataset has weight 1 and even the lowest-ranked
    dataset retains a positive weight.  The model may optionally raise these
    weights to a configurable power during gamma centering.
    """
    ranking = pd.read_csv(path, sep="\t")
    missing_columns = {dataset_column, rank_column} - set(ranking.columns)
    if missing_columns:
        raise KeyError(
            f"Dataset-quality table {path!r} is missing columns "
            f"{sorted(missing_columns)}."
        )

    names = ranking[dataset_column].astype(str)
    if names.duplicated().any():
        duplicates = sorted(names[names.duplicated(keep=False)].unique().tolist())
        raise ValueError(
            f"Dataset-quality table {path!r} contains duplicate datasets: "
            f"{duplicates}."
        )

    ranks = pd.to_numeric(ranking[rank_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(ranks).all() or np.any(ranks <= 0.0):
        raise ValueError(
            f"Column {rank_column!r} in {path!r} must contain finite positive ranks."
        )
    max_rank = float(ranks.max())
    quality = (max_rank - ranks + 1.0) / max_rank

    rank_by_dataset = dict(zip(names, ranks.astype(float)))
    quality_by_dataset = dict(zip(names, quality.astype(float)))
    return rank_by_dataset, quality_by_dataset


# ============================================================
# Utilities
# ============================================================


def _as_numpy_int(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)


def _as_numpy_str(x) -> np.ndarray:
    return np.asarray(x).astype(str)


# ============================================================
# Samplers
# ============================================================

def _shard_batches_padded(batches, num_replicas: int, rank: int):
    """Return this rank's shard of an already-built global batch list.

    Every rank builds the SAME `batches` list (identical sampler seed), so the
    global ordering is consistent and batch `i` deterministically belongs to
    rank `i % W`. Under DDP each rank MUST yield the same number of batches:
    unequal counts desynchronize the per-step gradient all-reduce and the
    epoch-end collectives (e.g. EarlyStopping's boolean reduce), which deadlocks
    NCCL until the watchdog timeout. We therefore pad the global list up to a
    multiple of `W` by wrapping around before striding, so each rank yields
    exactly ceil(n / W) batches. Padding (rather than dropping the remainder)
    keeps every sample in the epoch; the <= W-1 duplicated batches are re-drawn
    on the next epoch (training) or de-duplicated at prediction merge.
    """
    W = max(int(num_replicas), 1)
    n = len(batches)
    if W == 1 or n == 0:
        return list(batches)
    rem = n % W
    if rem != 0:
        batches = list(batches) + [batches[j % n] for j in range(W - rem)]
    return batches[rank::W]


class TranscriptGroupedMultiDatasetBatchSampler(BatchSampler):
    """
    Batch sampler that keeps transcript-dataset pairs for the same transcript
    together.

    This is intended for losses that need matched transcript content across
    datasets, including cross-dataset gamma centering for shared transcripts.

    Each emitted batch contains flat pair indices. For a transcript t measured
    in datasets D(t), the sampler emits the group:

        [(t, d) for d in D(t)]

    as an atomic unit. Groups are packed into batches without splitting them.
    If require_multidataset is true, transcripts with fewer than two available
    dataset observations are skipped.
    """

    def __init__(
        self,
        sampler: Optional[Sampler] = None,
        *,
        flat_transcript_ids: Sequence[str] | np.ndarray,
        flat_dataset_ids: Sequence[int] | np.ndarray,
        lengths,
        batch_size: int,
        seed: int = 42,
        drop_last: bool = False,
        sort_by_length: bool = True,
        require_multidataset: bool = True,
        shuffle_batches: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        self.flat_transcript_ids = _as_numpy_str(flat_transcript_ids)
        self.flat_dataset_ids = _as_numpy_int(flat_dataset_ids)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        if sampler is None:
            sampler = SequentialSampler(range(len(self.flat_transcript_ids)))
        # Lightning reconstructs BatchSampler subclasses during prediction to
        # wrap them for index tracking. Exposing and initializing ``sampler``
        # makes that reconstruction compatible. Group selection and distributed
        # sharding remain owned by this sampler (Trainer disables automatic
        # distributed-sampler injection in the project config).
        super().__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.sort_by_length = bool(sort_by_length)
        self.require_multidataset = bool(require_multidataset)
        self.shuffle_batches = bool(shuffle_batches)
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank) % self.num_replicas
        self._iter_count = 0
        self.last_epoch_statistics: GroupedBatchStatistics | None = None

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
        group_transcript_ids: list[str] = []
        group_dataset_counts: list[int] = []
        group_lengths: list[int] = []
        for tid in sorted(grouped):
            indices = np.asarray(grouped[tid], dtype=np.int64)
            dataset_count = len(np.unique(self.flat_dataset_ids[indices]))
            if self.require_multidataset and dataset_count < 2:
                continue
            groups.append(indices)
            group_transcript_ids.append(str(tid))
            group_dataset_counts.append(int(dataset_count))
            group_lengths.append(int(self.lengths[indices[0]]))

        if len(groups) == 0:
            raise RuntimeError(
                "No transcript groups available for transcript-grouped sampling. "
                "If this split has no transcripts measured in multiple datasets, use "
                "train_sampling_strategy=transcript_grouped_pairs or another sampler."
            )

        self.groups = groups
        self.group_transcript_ids = tuple(group_transcript_ids)
        self.group_dataset_counts = np.asarray(group_dataset_counts, dtype=np.int64)
        self.group_lengths = np.asarray(group_lengths, dtype=np.int64)
        self.group_full_pair_counts = np.asarray(
            [len(group) for group in self.groups], dtype=np.int64
        )
        oversized_indices = np.flatnonzero(
            self.group_full_pair_counts > self.batch_size
        )
        if oversized_indices.size:
            group_index = int(oversized_indices[0])
            transcript_id = self.group_transcript_ids[group_index]
            group_size = int(self.group_full_pair_counts[group_index])
            dataset_count = int(self.group_dataset_counts[group_index])
            raise ValueError(
                "Complete transcript group does not fit in the physical pair "
                f"microbatch: transcript_id={transcript_id!r}, group_size={group_size}, "
                f"physical_pair_capacity={self.batch_size}, "
                f"distinct_datasets={dataset_count}, suggested_minimum_batch_size="
                f"{group_size}. Increasing gradient accumulation does not solve an "
                "oversized individual group; data.batch_size must fit at least one complete "
                "transcript group."
            )

    def _sample_group_indices(
        self,
        group_index: int,
    ) -> np.ndarray:
        group = self.groups[int(group_index)]
        return group.copy()

    def _select_groups(
        self,
        rng: np.random.Generator,
    ) -> list[tuple[int, int, np.ndarray]]:
        group_order = rng.permutation(len(self.groups))

        selected = []
        for group_idx in group_order:
            group_idx = int(group_idx)
            indices = self._sample_group_indices(group_idx)
            length = int(self.group_lengths[group_idx])
            selected.append((length, group_idx, indices))

        if self.sort_by_length:
            selected.sort(key=lambda item: item[0], reverse=True)

        return selected

    def _build_epoch_plan(
        self,
        iteration_index: int,
    ) -> tuple[list[list[int]], GroupedBatchStatistics]:
        """Build one deterministic plan without touching sampler state."""
        rng = np.random.default_rng(self.seed + int(iteration_index))
        selected_groups = self._select_groups(rng)

        # Each batch carries its selected group records until statistics are
        # computed. The emitted DataLoader batch remains a plain index list.
        packed: list[tuple[list[int], list[tuple[str, int]]]] = []
        batch: list[int] = []
        batch_groups: list[tuple[str, int]] = []

        for _, group_index, group in selected_groups:
            group_list = group.tolist()

            if batch and len(batch) + len(group_list) > self.batch_size:
                packed.append((batch, batch_groups))
                batch = []
                batch_groups = []

            batch.extend(group_list)
            selected_dataset_count = int(
                np.unique(self.flat_dataset_ids[group]).size
            )
            batch_groups.append(
                (
                    self.group_transcript_ids[group_index],
                    selected_dataset_count,
                )
            )

        if batch and (len(batch) == self.batch_size or not self.drop_last):
            packed.append((batch, batch_groups))

        if self.shuffle_batches and len(packed) > 1:
            batch_order = rng.permutation(len(packed))
            packed = [packed[int(i)] for i in batch_order]

        packed = _shard_batches_padded(packed, self.num_replicas, self.rank)
        batches = [indices for indices, _ in packed]
        pair_rows = tuple(len(indices) for indices in batches)
        unique_transcripts = tuple(
            len(set(self.flat_transcript_ids[np.asarray(indices, dtype=np.int64)]))
            for indices in batches
        )
        group_records = [record for _, records in packed for record in records]
        datasets_per_group = tuple(int(record[1]) for record in group_records)
        def _minimum(values: tuple[int, ...]) -> int:
            return int(min(values)) if values else 0

        def _maximum(values: tuple[int, ...]) -> int:
            return int(max(values)) if values else 0

        def _median(values: tuple[int, ...]) -> float:
            return float(np.median(values)) if values else 0.0

        def _mean(values: tuple[int, ...]) -> float:
            return float(np.mean(values)) if values else 0.0

        statistics = GroupedBatchStatistics(
            iteration_index=int(iteration_index),
            number_of_microbatches=len(batches),
            eligible_transcript_groups=len(self.groups),
            physical_pair_capacity=self.batch_size,
            pair_rows_per_microbatch=pair_rows,
            unique_transcripts_per_microbatch=unique_transcripts,
            datasets_per_transcript_group=datasets_per_group,
            minimum_group_size=_minimum(datasets_per_group),
            median_group_size=_median(datasets_per_group),
            mean_group_size=_mean(datasets_per_group),
            maximum_group_size=_maximum(datasets_per_group),
            minimum_unique_transcripts_per_microbatch=_minimum(unique_transcripts),
            median_unique_transcripts_per_microbatch=_median(unique_transcripts),
            mean_unique_transcripts_per_microbatch=_mean(unique_transcripts),
            maximum_unique_transcripts_per_microbatch=_maximum(unique_transcripts),
        )
        return batches, statistics

    def preview_epoch_batch_statistics(
        self,
        iteration_index: int = 0,
    ) -> GroupedBatchStatistics:
        """Preview an epoch without incrementing `_iter_count` or consuming RNG."""
        _, statistics = self._build_epoch_plan(iteration_index=int(iteration_index))
        return statistics

    def __iter__(self):
        iteration_index = self._iter_count
        batches, statistics = self._build_epoch_plan(iteration_index)
        self._iter_count += 1
        self.last_epoch_statistics = statistics
        yield from batches

    def __len__(self):
        return self.preview_epoch_batch_statistics(
            iteration_index=self._iter_count
        ).number_of_microbatches


# ============================================================
# DataModule
# ============================================================


class RiboAIQueuingDatamoduleMultiDataset(pl.LightningDataModule):
    """
    DataModule for multi-dataset ribo-seq profile training.

    Training uses one of the transcript-grouped strategies. Groups are atomic,
    so every retained dataset observation for a transcript is packed together.
    The two supported strategies are ``transcript_grouped_pairs`` (including
    single-dataset transcripts) and ``transcript_grouped_multidataset_pairs``
    (requiring at least two dataset observations).
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
        num_workers: int = 4,
        predict_num_workers: int = 0,
        seed: int = 42,
        train_sampling_strategy: Optional[str] = None,
        train_allowed_dataset_names_by_transcript: Optional[dict[str, Sequence[str]]] = None,
        val_allowed_dataset_names_by_transcript: Optional[dict[str, Sequence[str]]] = None,
        pin_memory: bool = True,
        prefetch_factor: Optional[int] = 4,
        multiprocessing_context: Optional[str] = "spawn",
        additional_sequence_features: Optional[dict] = None,
        dataset_quality_ranking_path: Optional[str] = None,
        dataset_quality_dataset_column: str = "dataset",
        dataset_quality_rank_column: str = "quality_rank",
        dataset_quality_strict: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "train_allowed_dataset_names_by_transcript",
                "val_allowed_dataset_names_by_transcript",
            ]
        )

        self.sequences_path = sequences_path
        self.datasets_paths = list(datasets_paths)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("Physical pair-row microbatch capacity must be positive.")
        self.split = split
        self.num_workers = int(num_workers)
        self.predict_num_workers = int(predict_num_workers)
        if self.num_workers < 0 or self.predict_num_workers < 0:
            raise ValueError("DataLoader worker counts must be non-negative.")
        self.seed = int(seed)

        self.train_sampling_strategy = train_sampling_strategy
        self.train_allowed_dataset_names_by_transcript = train_allowed_dataset_names_by_transcript
        self.val_allowed_dataset_names_by_transcript = val_allowed_dataset_names_by_transcript

        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = prefetch_factor
        if multiprocessing_context is None:
            self.multiprocessing_context = None
        else:
            context = str(multiprocessing_context).strip().lower()
            if context not in {"spawn", "forkserver", "fork"}:
                raise ValueError(
                    "multiprocessing_context must be spawn, forkserver, fork, "
                    f"or null; got {multiprocessing_context!r}."
                )
            self.multiprocessing_context = context

        self.additional_sequence_features = dict(additional_sequence_features or {})
        self.dataset_quality_ranking_path = dataset_quality_ranking_path
        self.dataset_quality_dataset_column = str(dataset_quality_dataset_column)
        self.dataset_quality_rank_column = str(dataset_quality_rank_column)
        self.dataset_quality_strict = bool(dataset_quality_strict)

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
        self.val_flat_dataset_ids = None
        self.val_flat_transcript_ids = None

        self._has_loaded_data = False

    # ------------------------------------------------------------
    # Strategy resolution / flat metadata
    # ------------------------------------------------------------

    def _resolve_train_sampling_strategy(self) -> str:
        strategy = str(
            self.train_sampling_strategy or "transcript_grouped_pairs"
        ).lower()
        supported = {
            "transcript_grouped_pairs",
            "transcript_grouped_multidataset_pairs",
        }
        if strategy not in supported:
            raise ValueError(
                "Only transcript-grouped sampling is supported; choose one of "
                f"{sorted(supported)}, got {strategy!r}."
            )
        return strategy

    def _get_flat_dataset_ids(self, dataset_obj) -> np.ndarray:
        if hasattr(dataset_obj, "flat_dataset_ids"):
            return _as_numpy_int(dataset_obj.flat_dataset_ids)
        raise AttributeError(
            "RiboAIQueuingDatasetMultiDataset must expose flat_dataset_ids in "
            "deterministic transcript-grouped mode."
        )

    def _get_flat_lengths(self, dataset_obj) -> np.ndarray:
        if hasattr(dataset_obj, "flat_lengths"):
            return np.asarray(dataset_obj.flat_lengths, dtype=np.int32)
        raise AttributeError(
            "RiboAIQueuingDatasetMultiDataset must expose flat_lengths in deterministic mode."
        )

    def _get_flat_transcript_ids(self, dataset_obj) -> np.ndarray:
        """Return the explicit transcript ID stored for every flat pair."""
        if not hasattr(dataset_obj, "flat_transcript_ids"):
            raise AttributeError(
                "Deterministic datasets must expose flat_transcript_ids."
            )
        transcript_ids = _as_numpy_str(dataset_obj.flat_transcript_ids)
        if len(transcript_ids) != len(dataset_obj):
            raise ValueError(
                "flat_transcript_ids must contain one ID per flat pair; got "
                f"{len(transcript_ids)} IDs for a dataset of length {len(dataset_obj)}."
            )
        return transcript_ids

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

        # Read the base sequence/CSS columns plus only the optional per-codon
        # features that are routed to at least one model branch.
        seq_available = set(pq.read_schema(self.sequences_path).names)
        seq_css_col = (
            "conserved_stalling_sites"
            if "conserved_stalling_sites" in seq_available
            else "css"
        )
        active_sequence_feature_names = [
            str(name)
            for name, raw_spec in self.additional_sequence_features.items()
            if str(dict(raw_spec or {}).get("route", "none")).lower() != "none"
        ]
        missing_sequence_features = sorted(
            set(active_sequence_feature_names) - seq_available
        )
        if missing_sequence_features:
            raise KeyError(
                "Configured additional sequence feature columns are missing from "
                f"{self.sequences_path}: {missing_sequence_features}."
            )
        seq_columns = [
            c for c in ("transcript_id", "ref", seq_css_col) if c in seq_available
        ]
        seq_columns.extend(active_sequence_feature_names)
        seq_df = pd.read_parquet(self.sequences_path, columns=seq_columns or None)
        if "transcript_id" in seq_df.columns:
            seq_df = seq_df.set_index("transcript_id")
        seq_df.index = seq_df.index.astype(str)

        print("Length of the main sequence:", len(seq_df.index))

        loaded_datasets = {}
        weighted_dataset_names = []
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
            columns = ["ribo"]
            if "weight" in df.columns:
                columns.append("weight")
                weighted_dataset_names.append(dataset_name)

            if RIBO_REPLICAS_COLUMN not in df.columns:
                raise KeyError(
                    f"Required replica column {RIBO_REPLICAS_COLUMN!r} is missing "
                    f"in {path}. Point dataset_config at replica-aware parquets."
                )
            columns.append(RIBO_REPLICAS_COLUMN)

            loaded_datasets[dataset_name] = df[columns]
            union_index = union_index.union(df.index, sort=False)

        if weighted_dataset_names:
            print(
                "Using transcript weights from "
                f"{len(weighted_dataset_names)}/{len(loaded_datasets)} loaded datasets."
            )
        else:
            print("No transcript weight column found; using unit sample weights.")

        dataset_quality_ranks = {name: float("nan") for name in loaded_datasets}
        dataset_quality_weights = {name: 1.0 for name in loaded_datasets}
        if self.dataset_quality_ranking_path:
            rank_lookup, quality_lookup = load_dataset_quality_ranking(
                self.dataset_quality_ranking_path,
                dataset_column=self.dataset_quality_dataset_column,
                rank_column=self.dataset_quality_rank_column,
            )
            missing_rankings = sorted(set(loaded_datasets) - set(rank_lookup))
            if missing_rankings and self.dataset_quality_strict:
                raise KeyError(
                    "The dataset-quality ranking is missing active datasets: "
                    f"{missing_rankings}."
                )
            for name in loaded_datasets:
                if name in rank_lookup:
                    dataset_quality_ranks[name] = rank_lookup[name]
                    dataset_quality_weights[name] = quality_lookup[name]
            print(
                "Loaded dataset-quality metadata for "
                f"{len(loaded_datasets) - len(missing_rankings)}/{len(loaded_datasets)} "
                f"active datasets from {self.dataset_quality_ranking_path}."
            )
            if missing_rankings:
                print(
                    "Missing rankings use neutral quality weight 1.0 because "
                    f"data.dataset_quality_ranking.strict=false: {missing_rankings}"
                )

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
            "ribo_replicas": defaultdict(dict),
            "sample_weights": defaultdict(dict),
            "dataset_quality_ranks": dataset_quality_ranks,
            "dataset_quality_weights": dataset_quality_weights,
            "lengths": lengths,
            "datasets_names": list(loaded_datasets.keys()),
            "sequence_features": {
                name: seq_df_union[name].values
                for name in active_sequence_feature_names
            },
        }

        def _stack_replicas(cell) -> np.ndarray:
            """Stack a per-transcript replica cell into a [n_replicas, L] array."""
            reps = [np.asarray(rep, dtype=np.float32) for rep in cell]
            if len(reps) == 0:
                raise ValueError("Encountered a transcript with zero replicas.")
            rep_lengths = {rep.shape[0] for rep in reps}
            if len(rep_lengths) != 1:
                raise ValueError(
                    f"Replica length mismatch within a transcript: {sorted(rep_lengths)}."
                )
            return np.ascontiguousarray(np.stack(reps, axis=0))

        valid_ids = set(valid_index.astype(str))
        for dataset_name, df in loaded_datasets.items():
            has_weight = "weight" in df.columns
            ids = df.index.astype(str)
            weight_values = df["weight"].values if has_weight else None
            replica_values = df[RIBO_REPLICAS_COLUMN].values
            for i, t_id in enumerate(ids):
                if t_id not in valid_ids:
                    continue
                reps = _stack_replicas(replica_values[i])  # [R, L], CDS-aligned
                shared_data["ribo_replicas"][t_id][dataset_name] = reps
                # Consensus is retained for PCC and diagnostics. The NB count
                # likelihood is always evaluated on the raw replica profiles.
                shared_data["ribo_profiles"][t_id][dataset_name] = reps.mean(axis=0)
                if has_weight:
                    sample_weight = float(weight_values[i])
                    if not np.isfinite(sample_weight):
                        raise ValueError(
                            "Non-finite transcript sample weight in "
                            f"dataset={dataset_name}, transcript={t_id}: "
                            f"{sample_weight}."
                        )
                    if sample_weight < 0.0:
                        raise ValueError(
                            "Negative transcript sample weight in "
                            f"dataset={dataset_name}, transcript={t_id}: "
                            f"{sample_weight}."
                        )
                    shared_data["sample_weights"][t_id][dataset_name] = sample_weight

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
        print("dataset pair mode: deterministic flat pairs")

        print("\n=== Training sampling strategy ===")
        print(f"strategy: {strategy}")

        self.train_dataset_obj = RiboAIQueuingDatasetMultiDataset(
            data=shared_data,
            lengths=shared_data["lengths"],
            nt_encoding=self.nt_enc,
            codon_to_aa_encoding=self.c2aa_enc,
            codon_encoding=self.c_enc,
            aa_encoding=self.aa_enc,
            datasets_encoding=self.datasets_enc,
            transcripts_ids=train_ids,
            allowed_dataset_names_by_transcript=self.train_allowed_dataset_names_by_transcript,
            additional_sequence_features=self.additional_sequence_features,
        )

        self.train_lengths = self._get_flat_lengths(self.train_dataset_obj)
        self.train_flat_dataset_ids = self._get_flat_dataset_ids(self.train_dataset_obj)
        self.train_flat_transcript_ids = self._get_flat_transcript_ids(self.train_dataset_obj)

        self._print_flat_pair_summary(
            dataset_obj=self.train_dataset_obj,
            flat_transcript_ids=self.train_flat_transcript_ids,
            flat_dataset_ids=self.train_flat_dataset_ids,
            split_name="train",
        )
        self.val_dataset_obj = RiboAIQueuingDatasetMultiDataset(
            data=shared_data,
            lengths=shared_data["lengths"],
            nt_encoding=self.nt_enc,
            codon_to_aa_encoding=self.c2aa_enc,
            codon_encoding=self.c_enc,
            aa_encoding=self.aa_enc,
            datasets_encoding=self.datasets_enc,
            transcripts_ids=val_ids,
            allowed_dataset_names_by_transcript=self.val_allowed_dataset_names_by_transcript,
            additional_sequence_features=self.additional_sequence_features,
        )

        self.val_lengths = self._get_flat_lengths(self.val_dataset_obj)

        val_flat_dataset_ids = self._get_flat_dataset_ids(self.val_dataset_obj)
        val_flat_transcript_ids = self._get_flat_transcript_ids(self.val_dataset_obj)
        self.val_flat_dataset_ids = val_flat_dataset_ids
        self.val_flat_transcript_ids = val_flat_transcript_ids
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

        sample_weights = getattr(dataset_obj, "flat_sample_weights", None)
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=np.float32)
            print(
                "sample weights: "
                f"mean={sample_weights.mean():.4f}, "
                f"min={sample_weights.min():.4f}, "
                f"max={sample_weights.max():.4f}, "
                f"nonzero={(sample_weights > 0.0).mean():.3f}"
            )

    # ------------------------------------------------------------
    # Workers / DataLoaders
    # ------------------------------------------------------------

    def _dist_info(self) -> tuple[int, int]:
        """Return (num_replicas, rank) for the current distributed context."""
        if self.trainer is not None and self.trainer.world_size > 1:
            return self.trainer.world_size, self.trainer.global_rank
        return 1, 0

    def _dataloader_kwargs(self, *, num_workers: Optional[int] = None):
        workers = self.num_workers if num_workers is None else int(num_workers)
        kwargs = {
            "num_workers": workers,
            "persistent_workers": (workers > 0),
            "worker_init_fn": seed_dataloader_worker,
            "pin_memory": self.pin_memory,
        }

        if workers > 0:
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)
            if self.multiprocessing_context is not None:
                kwargs["multiprocessing_context"] = self.multiprocessing_context

        return kwargs

    def _make_train_grouped_batch_sampler(
        self,
        *,
        num_replicas: int,
        rank: int,
        iteration_seed: Optional[int] = None,
    ) -> TranscriptGroupedMultiDatasetBatchSampler:
        strategy = self._resolve_train_sampling_strategy()
        if strategy not in {
            "transcript_grouped_pairs",
            "transcript_grouped_multidataset_pairs",
        }:
            raise RuntimeError(
                "Grouped sampler planning requires a transcript-grouped training "
                f"strategy, got {strategy!r}."
            )
        if self.train_flat_transcript_ids is None or self.train_flat_dataset_ids is None:
            raise RuntimeError(f"{strategy} requires deterministic flat metadata.")
        return TranscriptGroupedMultiDatasetBatchSampler(
            flat_transcript_ids=self.train_flat_transcript_ids,
            flat_dataset_ids=self.train_flat_dataset_ids,
            lengths=self.train_lengths,
            batch_size=self.batch_size,
            seed=self.seed if iteration_seed is None else int(iteration_seed),
            drop_last=False,
            sort_by_length=True,
            require_multidataset=(
                strategy == "transcript_grouped_multidataset_pairs"
            ),
            num_replicas=max(int(num_replicas), 1),
            rank=int(rank),
        )

    def preview_train_grouped_batch_statistics(
        self,
        iteration_index: int = 0,
    ) -> GroupedBatchStatistics:
        """Preview the unsharded global plan used to resolve DDP accumulation.

        The accumulation factor must be identical on every rank. We therefore
        preview the deterministic global plan once and let the resolver convert
        its microbatch count to a per-rank count using world size. Actual
        DataLoader samplers still shard and pad that same plan per rank.
        """
        if self.train_dataset_obj is None:
            raise RuntimeError("setup('fit') must run before grouped batch preview.")
        sampler = self._make_train_grouped_batch_sampler(
            num_replicas=1,
            rank=0,
            iteration_seed=self.seed,
        )
        return sampler.preview_epoch_batch_statistics(
            iteration_index=int(iteration_index)
        )

    def train_dataloader(self):
        if self.train_dataset_obj is None:
            raise RuntimeError("setup() must be called before train_dataloader().")

        self._resolve_train_sampling_strategy()
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        seed = self.seed + int(epoch)
        num_replicas, rank = self._dist_info()
        batch_sampler = self._make_train_grouped_batch_sampler(
            num_replicas=num_replicas,
            rank=rank,
            iteration_seed=seed,
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

        if self.val_flat_transcript_ids is None or self.val_flat_dataset_ids is None:
            raise RuntimeError("Validation dataloader requires deterministic flat metadata.")

        num_replicas, rank = self._dist_info()
        batch_sampler = TranscriptGroupedMultiDatasetBatchSampler(
            flat_transcript_ids=self.val_flat_transcript_ids,
            flat_dataset_ids=self.val_flat_dataset_ids,
            lengths=self.val_lengths,
            batch_size=self.batch_size,
            seed=self.seed,
            drop_last=False,
            sort_by_length=True,
            require_multidataset=False,
            shuffle_batches=False,
            num_replicas=num_replicas,
            rank=rank,
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

        if self.val_flat_transcript_ids is None or self.val_flat_dataset_ids is None:
            raise RuntimeError("Prediction dataloader requires deterministic flat metadata.")

        num_replicas, rank = self._dist_info()

        batch_sampler = TranscriptGroupedMultiDatasetBatchSampler(
            flat_transcript_ids=self.val_flat_transcript_ids,
            flat_dataset_ids=self.val_flat_dataset_ids,
            lengths=self.val_lengths,
            batch_size=self.batch_size,
            drop_last=False,
            seed=self.seed,
            sort_by_length=True,
            require_multidataset=False,
            shuffle_batches=False,
            num_replicas=num_replicas,
            rank=rank,
        )

        return DataLoader(
            self.val_dataset_obj,
            batch_sampler=batch_sampler,
            collate_fn=self.val_dataset_obj.collate_fn,
            **self._dataloader_kwargs(num_workers=self.predict_num_workers),
        )
