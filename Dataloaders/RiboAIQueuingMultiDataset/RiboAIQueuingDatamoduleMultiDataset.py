from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import lightning as pl
import numpy as np
import pandas as pd
import torch
import yaml

from torch.utils.data import (
    DataLoader,
    BatchSampler,
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


class SortedLengthBatchSampler(BatchSampler):
    """
    Batch sampler that groups sampled indices by sequence length.

    Important:
      - sampler yields dataset indices.
      - lengths must be aligned to the dataset indexing space.
      - supports SequentialSampler and WeightedRandomSampler.
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
    ):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.descending = bool(descending)

        if lengths is None:
            raise ValueError("SortedLengthBatchSampler requires lengths.")

        self.lengths = np.asarray(lengths)
        self._rng = np.random.default_rng(self.seed)

    def __iter__(self):
        idx = np.fromiter(iter(self.sampler), dtype=np.int64)

        if idx.size == 0:
            return
            yield  # pragma: no cover

        if idx.max(initial=0) >= len(self.lengths):
            raise IndexError(
                f"Sampler yielded index {idx.max()} but lengths has size {len(self.lengths)}."
            )

        # Sort only the sampled indices by length.
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
            batch_order = self._rng.permutation(len(batches))
            for j in batch_order:
                yield batches[j]
        else:
            for batch in batches:
                yield batch

    def __len__(self):
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class RiboAIQueuingDatamoduleMultiDataset(pl.LightningDataModule):
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
        dataset_balance_gamma: float = 1.0,
        train_samples_per_epoch: Optional[int] = None,
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

        self.nt_enc = open_file(nt_encoding_path)
        self.c2aa_enc = open_file(codon_to_aa_encoding_path)
        self.c_enc = open_file(codon_encoding_path)
        self.aa_enc = open_file(aa_encoding_path)
        self.datasets_enc = open_file(datasets_encoding_path)

        self.train_dataset_obj = None
        self.val_dataset_obj = None

        self.train_lengths = None
        self.val_lengths = None
        self.train_sample_weights = None

        self._has_loaded_data = False

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

            df = df.set_index("id")
            df.index = df.index.astype(str)

            if "ribo" not in df.columns:
                raise KeyError(f"'ribo' column missing in {path}")

            df = df[["ribo"]]

            dataset_name = os.path.basename(path).split(".")[0]
            loaded_datasets[dataset_name] = df

            union_index = union_index.union(df.index, sort=False)

        valid_index = seq_df.index.intersection(union_index, sort=False)

        if len(valid_index) == 0:
            raise RuntimeError("No transcript IDs overlap between sequence table and ribo datasets.")

        seq_df_union = seq_df.loc[valid_index]

        ref_arrays = seq_df_union["ref"].values

        css_col = (
            "conserved_stalling_sites"
            if "conserved_stalling_sites" in seq_df_union.columns
            else "css"
        )

        if css_col not in seq_df_union.columns:
            raise KeyError(
                "Could not find CSS column. Expected either "
                "'conserved_stalling_sites' or 'css'."
            )

        css = seq_df_union[css_col].values
        lengths = np.array([len(x) for x in ref_arrays], dtype=np.int32)

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
            raise NotImplementedError("Random split is not implemented here. Pass a transcript-level split.")

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

        # ============================================================
        # Training dataset
        # ============================================================
        #
        # Unbalanced mode:
        #   __len__ = number of transcripts.
        #   __getitem__ chooses a random dataset for that transcript.
        #
        # Balanced mode:
        #   __len__ = number of transcript-dataset pairs.
        #   sampler controls dataset balance over flat pairs.
        #
        train_choice_mode = "deterministic" if self.balanced_train_sampling else "random"

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

        if self.balanced_train_sampling:
            if not hasattr(self.train_dataset_obj, "flat_lengths"):
                raise AttributeError(
                    "balanced_train_sampling=True requires the dataset to expose flat_lengths."
                )

            if not hasattr(self.train_dataset_obj, "make_dataset_balanced_weights"):
                raise AttributeError(
                    "balanced_train_sampling=True requires the dataset method "
                    "make_dataset_balanced_weights(gamma)."
                )

            self.train_lengths = np.asarray(self.train_dataset_obj.flat_lengths, dtype=np.int32)

            self.train_sample_weights = self.train_dataset_obj.make_dataset_balanced_weights(
                gamma=self.dataset_balance_gamma,
            )

            if len(self.train_sample_weights) != len(self.train_dataset_obj):
                raise RuntimeError(
                    f"train_sample_weights length {len(self.train_sample_weights)} "
                    f"does not match train dataset length {len(self.train_dataset_obj)}."
                )

            print("\n=== Balanced training sampling enabled ===")
            print(f"dataset_balance_gamma: {self.dataset_balance_gamma}")
            print(f"train dataset flat pairs: {len(self.train_dataset_obj)}")

            if self.train_samples_per_epoch is None:
                print(f"train_samples_per_epoch: {len(self.train_dataset_obj)}")
            else:
                print(f"train_samples_per_epoch: {self.train_samples_per_epoch}")

            if hasattr(self.train_dataset_obj, "dataset_pair_counts"):
                counts = self.train_dataset_obj.dataset_pair_counts()
                print("\n=== Training dataset pair counts ===")
                for ds_id, n in sorted(counts.items()):
                    ds_name = self.train_dataset_obj.idx_to_dataset.get(ds_id, str(ds_id))
                    print(f"  {ds_name:35s} id={ds_id:3d} pairs={n}")

        else:
            self.train_lengths = lengths[train_indices]
            self.train_sample_weights = None

            print("\n=== Unbalanced/random training sampling enabled ===")
            print("Each transcript appears once per epoch; dataset is randomly chosen inside __getitem__.")

        # ============================================================
        # Validation dataset
        # ============================================================
        #
        # Always deterministic so validation covers all transcript-dataset pairs.
        #
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

        if hasattr(self.val_dataset_obj, "flat_lengths"):
            self.val_lengths = np.asarray(self.val_dataset_obj.flat_lengths, dtype=np.int32)
        else:
            self.val_lengths = np.repeat(
                lengths[val_indices],
                [len(shared_data["ribo_profiles"][tid]) for tid in val_ids],
            )

        print(f"Validation flat transcript-dataset pairs: {len(self.val_dataset_obj)}")

    def worker_init_fn(self, worker_id: int):
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        worker_seed = self.seed + worker_id + int(epoch) * 1000

        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)

    def train_dataloader(self):
        if self.train_dataset_obj is None:
            raise RuntimeError("setup() must be called before train_dataloader().")

        if self.balanced_train_sampling:
            if self.train_sample_weights is None:
                raise RuntimeError(
                    "balanced_train_sampling=True but train_sample_weights is None."
                )

            epoch = self.trainer.current_epoch if self.trainer is not None else 0
            generator = torch.Generator().manual_seed(self.seed + int(epoch))

            num_samples = (
                int(self.train_samples_per_epoch)
                if self.train_samples_per_epoch is not None
                else len(self.train_dataset_obj)
            )

            base_sampler = WeightedRandomSampler(
                weights=self.train_sample_weights,
                num_samples=num_samples,
                replacement=True,
                generator=generator,
            )

            shuffle_batches = True

        else:
            base_sampler = SequentialSampler(self.train_dataset_obj)
            shuffle_batches = True

        batch_sampler = SortedLengthBatchSampler(
            sampler=base_sampler,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=shuffle_batches,
            lengths=self.train_lengths,
            seed=self.seed,
            descending=True,
        )

        return DataLoader(
            self.train_dataset_obj,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset_obj.collate_fn,
            persistent_workers=(self.num_workers > 0),
            worker_init_fn=self.worker_init_fn,
            pin_memory=False,
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
            num_workers=self.num_workers,
            collate_fn=self.val_dataset_obj.collate_fn,
            persistent_workers=(self.num_workers > 0),
            worker_init_fn=self.worker_init_fn,
            pin_memory=False,
        )

    def predict_dataloader(self):
        if self.val_dataset_obj is None:
            raise RuntimeError("setup() must be called before predict_dataloader().")

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
            num_workers=self.num_workers,
            collate_fn=self.val_dataset_obj.collate_fn,
            persistent_workers=(self.num_workers > 0),
            worker_init_fn=self.worker_init_fn,
            pin_memory=True,
        )