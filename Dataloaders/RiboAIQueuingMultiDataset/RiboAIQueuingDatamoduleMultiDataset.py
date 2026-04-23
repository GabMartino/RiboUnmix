import os
from collections import defaultdict

import lightning as pl
import pandas as pd
import torch
import numpy as np
import yaml

from torch import Generator
from torch.utils.data import DataLoader, Subset, Sampler, BatchSampler, SequentialSampler
from tqdm import tqdm

from Dataloaders.RiboAIQueuingMultiDataset.RiboAIQueuingMultiDataset import (
    RiboAIQueuingDatasetMultiDataset
)


def open_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class SortedLengthBatchSampler(BatchSampler):
    """
    Batch sampler that groups subset-space indices by sequence length.
    Compatible with Lightning sampler injection because it exposes:
      - sampler
      - batch_size
      - drop_last
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
        self.lengths = np.asarray(lengths)
        self._rng = np.random.default_rng(self.seed)

    def __iter__(self):
        # sampler yields indices in subset-space: 0..len(subset)-1
        idx = np.fromiter(iter(self.sampler), dtype=np.int64)

        if idx.size == 0:
            return
            yield  # pragma: no cover

        # sort only the indices assigned by the current sampler
        order = np.argsort(self.lengths[idx], kind="stable")
        if self.descending:
            order = order[::-1]
        idx = idx[order]

        batches = [
            idx[i:i + self.batch_size].tolist()
            for i in range(0, len(idx), self.batch_size)
        ]

        if self.drop_last and len(batches) > 0 and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        if self.shuffle and len(batches) > 1:
            batch_order = self._rng.permutation(len(batches))
            for j in batch_order:
                yield batches[j]
        else:
            for b in batches:
                yield b

    def __len__(self):
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class RiboAIQueuingDatamoduleMultiDataset(pl.LightningDataModule):
    def __init__(
        self,
        sequences_path: str,
        datasets_paths: list,
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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.sequences_path = sequences_path
        self.datasets_paths = datasets_paths
        self.batch_size = int(batch_size)
        self.split = split
        self.split_p = float(split_p)
        self.num_workers = int(num_workers)
        self.seed = int(seed)

        self.nt_enc = open_file(nt_encoding_path)
        self.c2aa_enc = open_file(codon_to_aa_encoding_path)
        self.c_enc = open_file(codon_encoding_path)
        self.aa_enc = open_file(aa_encoding_path)
        self.datasets_enc = open_file(datasets_encoding_path)
        self.train_dataset_obj = None
        self.val_dataset_obj = None

        self.train_lengths = None
        self.val_lengths = None
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

        # Optional but often important: normalize ID dtype
        seq_df.index = seq_df.index.astype(str)

        print("Length of the main sequence:", len(seq_df.index))

        loaded_datasets = {}
        union_index = pd.Index([], dtype=seq_df.index.dtype)

        for path in tqdm(self.datasets_paths, desc="Loading Data"):
            df = pd.read_parquet(path)
            df = df.set_index("id")

            if "ribo" not in df.columns:
                raise KeyError(f"'ribo' column missing in {path}")

            df = df[["ribo"]]
            df.index = df.index.astype(str)

            dataset_name = path.split("/")[-1].split(".")[0]
            loaded_datasets[dataset_name] = df

            union_index = union_index.union(df.index, sort=False)

        valid_index = seq_df.index.intersection(union_index, sort=False)
        seq_df_union = seq_df.loc[valid_index]

        ref_arrays = seq_df_union["ref"].values
        css_col = "conserved_stalling_sites" if "conserved_stalling_sites" in seq_df_union.columns else "css"
        css = seq_df_union[css_col].values
        lengths = np.array([len(x) for x in ref_arrays], dtype=np.int32)

        shared_data = {
            "transcript_id": valid_index.values,
            "ref": ref_arrays,
            "css": css,
            "ribo_profiles": defaultdict(dict),
            "lengths": lengths,
            "datasets_names": list(loaded_datasets.keys())
        }

        valid_ids = set(valid_index)

        for dataset_name, df in loaded_datasets.items():
            for t_id, ribo_profile in zip(df.index, df["ribo"].values):
                if t_id in valid_ids:
                    shared_data["ribo_profiles"][t_id][dataset_name] = ribo_profile


        # ---- 5) build transcript-level train/val split ----
        if self.split is not None:
            print("Using provided split (transcript IDs).")
            all_ids = np.asarray(shared_data["transcript_id"]).astype(str)

            train_id_set = set(map(str, self.split[0]))
            val_id_set = set(map(str, self.split[1]))

            train_indices = [i for i, t in enumerate(all_ids) if t in train_id_set]
            val_indices = [i for i, t in enumerate(all_ids) if t in val_id_set]

            train_ids = all_ids[train_indices].tolist()
            val_ids = all_ids[val_indices].tolist()
        else:
            raise NotImplementedError("This has not implemented yet")
            # print(f"Random split with split_p={self.split_p:.3f}")
            # g = Generator().manual_seed(self.seed)
            # perm = torch.randperm(nT, generator=g).tolist()
            # train_len = int(nT * self.split_p)
            # train_indices = perm[:train_len]
            # val_indices = perm[train_len:]
            # if len(val_indices) == 0:
            #     raise RuntimeError("Validation split is empty; decrease split_p.")

        '''
            Training set: 1 epoch of full transcripts list ( with subset) randomly sampling the datasets
            Validation set: 1 epoch of transcripts list ( with subset) randomly sampling the datasets
            Prediction set: full datasets
        '''
        self.train_dataset_obj = RiboAIQueuingDatasetMultiDataset(
            data=shared_data,
            lengths=shared_data["lengths"],
            nt_encoding=self.nt_enc,
            codon_to_aa_encoding=self.c2aa_enc,
            codon_encoding=self.c_enc,
            aa_encoding=self.aa_enc,
            datasets_encoding=self.datasets_enc,
            transcripts_ids=train_ids,
            dataset_choice_mode="random",
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
            dataset_choice_mode="deterministic",
        )

        self.train_lengths = lengths[train_indices]
        self.val_lengths = np.repeat(
            lengths[val_indices],
            [len(shared_data["ribo_profiles"][tid]) for tid in val_ids]
        )


    def worker_init_fn(self, worker_id):
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        worker_seed = self.seed + worker_id + epoch
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)

    def train_dataloader(self):
        batch_sampler = SortedLengthBatchSampler(
            sampler=SequentialSampler(self.train_dataset_obj),
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
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
            pin_memory=True,
        )

    def val_dataloader(self):
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

    def predict_dataloader(self):
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