import os
import lightning as pl
import pandas as pd
import torch
import numpy as np
import yaml

from torch import Generator
from torch.utils.data import DataLoader, Subset, Sampler, BatchSampler, SequentialSampler
from tqdm import tqdm

from Dataloaders.RiboAIQueuingMultiDatasetMultiEmbeddings.RiboAIQueuingMultiDatasetMultiEmbeddings import \
    RiboAIQueuingDatasetMultiDatasetMultiEmbeddings


def open_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SortedLengthBatchSampler(BatchSampler):
    """
    Batch sampler that groups subset-space indices by sequence length.
    Compatible with Lightning sampler injection.
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
        idx = np.fromiter(iter(self.sampler), dtype=np.int64)

        if idx.size == 0:
            return
            yield  # pragma: no cover

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


class RiboAIQueuingDatamoduleMultiDatasetMultiEmbeddings(pl.LightningDataModule):
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
            embeddings: list,  # FIX 2: Added the embeddings list argument
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

        # Save the embeddings list so setup() can use it
        self.embeddings = embeddings if embeddings is not None else []

        self.nt_enc = open_file(nt_encoding_path)
        self.c2aa_enc = open_file(codon_to_aa_encoding_path)
        self.c_enc = open_file(codon_encoding_path)
        self.aa_enc = open_file(aa_encoding_path)
        self.datasets_enc = open_file(datasets_encoding_path)

        self.train_set = None
        self.val_set = None

        self.train_dataset_obj = None
        self.val_dataset_obj = None

        self.train_lengths = None
        self.val_lengths = None

    def setup(self, stage=None):
        if self.train_set is not None:
            return

        print(f"Intersecting master sequences with {len(self.datasets_paths)} datasets...")

        # ---- 1) master sequences ----
        seq_df = pd.read_parquet(self.sequences_path)
        if "transcript_id" in seq_df.columns:
            seq_df = seq_df.set_index("transcript_id")

        common_index = seq_df.index
        loaded_datasets: dict[str, pd.DataFrame] = {}

        # ---- 2) load datasets + compute intersection ----
        for path in tqdm(self.datasets_paths, desc="Loading and Intersecting"):
            df = pd.read_parquet(path)
            dataset_name = path.split("/")[-1].split(".")[0]
            if "id" in df.columns:
                df = df.set_index("id")

            common_index = common_index.intersection(df.index, sort=False)
            loaded_datasets[dataset_name] = df

        dataset_names = list(loaded_datasets.keys())
        nD = len(dataset_names)

        print(f"Original sequences: {len(seq_df)}")
        print(f"Sequences surviving the Inner Join: {len(common_index)}")
        if len(common_index) == 0:
            raise RuntimeError("Empty intersection between master sequences and datasets.")

        # ---- 3) filter master to common transcripts ----
        seq_df_common = seq_df.loc[common_index]

        ref_arrays = seq_df_common["ref"].values

        # Handle column naming variations safely
        css_col = "conserved_stalling_sites" if "conserved_stalling_sites" in seq_df_common.columns else "css"
        css = seq_df_common[css_col].values

        lengths = np.array([len(x) for x in ref_arrays], dtype=np.int32)

        nT = len(common_index)

        shared_data = {
            "transcript_id": common_index.values,
            "ref": ref_arrays,
            "css": css,
            "ribo_profiles": {},
            "lengths": lengths,
            "datasets_names": dataset_names,
        }

        # FIX 3: Dynamically extract the requested embeddings
        for emb_name in self.embeddings:
            if emb_name not in seq_df_common.columns:
                raise ValueError(f"Requested embedding '{emb_name}' is missing from the master sequences parquet file!")
            shared_data[emb_name] = seq_df_common[emb_name].values

        # ---- 4) extract ribo profiles aligned to common_index ----
        for dataset_name, df in loaded_datasets.items():
            aligned_df = df.loc[common_index]
            shared_data["ribo_profiles"][dataset_name] = [
                np.asarray(arr, dtype=np.float32) for arr in aligned_df["ribo"].values
            ]

        # ---- 5) build transcript-level train/val split ----
        if self.split is not None:
            print("Using provided split (transcript IDs).")
            train_ids = set(self.split[0])
            val_ids = set(self.split[1])

            train_t = [i for i, t in enumerate(shared_data["transcript_id"]) if t in train_ids]
            val_t = [i for i, t in enumerate(shared_data["transcript_id"]) if t in val_ids]

            if len(train_t) == 0 or len(val_t) == 0:
                raise RuntimeError(f"Split produced empty train/val: train={len(train_t)} val={len(val_t)}")
        else:
            print(f"Random split with split_p={self.split_p:.3f}")
            g = Generator().manual_seed(self.seed)
            perm = torch.randperm(nT, generator=g).tolist()
            train_len = int(nT * self.split_p)
            train_t = perm[:train_len]
            val_t = perm[train_len:]
            if len(val_t) == 0:
                raise RuntimeError("Validation split is empty; decrease split_p.")

        # ---- 6) expand transcript indices to global multi-dataset indices ----
        def expand_indices(t_indices: list[int]) -> list[int]:
            return [d * nT + t for d in range(nD) for t in t_indices]

        train_indices = expand_indices(train_t)
        val_indices = expand_indices(val_t)

        all_lengths = np.tile(lengths, nD)
        train_lengths = all_lengths[train_indices]
        val_lengths = all_lengths[val_indices]

        # ---- 7) build datasets (Using the NEW Dataset Class) ----
        self.train_dataset_obj = RiboAIQueuingDatasetMultiDatasetMultiEmbeddings(
            data=shared_data,
            lengths=shared_data["lengths"],
            embeddings=self.embeddings,  # <-- Pass the list here
            nt_encoding=self.nt_enc,
            codon_to_aa_encoding=self.c2aa_enc,
            codon_encoding=self.c_enc,
            aa_encoding=self.aa_enc,
            datasets_encoding=self.datasets_enc,
        )
        self.val_dataset_obj = RiboAIQueuingDatasetMultiDatasetMultiEmbeddings(
            data=shared_data,
            lengths=shared_data["lengths"],
            embeddings=self.embeddings,  # <-- Pass the list here
            nt_encoding=self.nt_enc,
            codon_to_aa_encoding=self.c2aa_enc,
            codon_encoding=self.c_enc,
            aa_encoding=self.aa_enc,
            datasets_encoding=self.datasets_enc,
        )

        # ---- 8) subsets + per-subset lengths for sampler ----
        self.train_set = Subset(self.train_dataset_obj, train_indices)
        self.val_set = Subset(self.val_dataset_obj, val_indices)

        self.train_lengths = train_lengths
        self.val_lengths = val_lengths

    def worker_init_fn(self, worker_id):
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        worker_seed = self.seed + worker_id + epoch
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)

    def train_dataloader(self):
        batch_sampler = SortedLengthBatchSampler(
            sampler=SequentialSampler(self.train_set),
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
            lengths=self.train_lengths,
            seed=self.seed,
            descending=True,
        )
        return DataLoader(
            self.train_set,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset_obj.collate_fn,
            persistent_workers=(self.num_workers > 0),
            worker_init_fn=self.worker_init_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        batch_sampler = SortedLengthBatchSampler(
            sampler=SequentialSampler(self.val_set),
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            lengths=self.val_lengths,
            seed=self.seed,
            descending=True,
        )
        return DataLoader(
            self.val_set,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset_obj.collate_fn,
            persistent_workers=(self.num_workers > 0),
            worker_init_fn=self.worker_init_fn,
            pin_memory=True,
        )

    def predict_dataloader(self):
        batch_sampler = SortedLengthBatchSampler(
            sampler=SequentialSampler(self.val_set),
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            lengths=self.val_lengths,
            seed=self.seed,
            descending=True,
        )
        return DataLoader(
            self.val_set,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset_obj.collate_fn,
            persistent_workers=(self.num_workers > 0),
            worker_init_fn=self.worker_init_fn,
            pin_memory=True,
        )