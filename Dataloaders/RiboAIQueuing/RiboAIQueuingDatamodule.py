import lightning as pl
import pandas as pd
import torch
import numpy as np
import yaml
from torch import Generator
from torch.utils.data import DataLoader, Subset, Sampler

from Dataloaders.RiboAIQueuing.RiboAIQueuingDataset import RiboAIQueuingDataset


def open_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class SortedLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size: int, seed: int = 42, shuffle: bool = True, descending: bool = True):
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)

        if isinstance(data_source, Subset):
            lengths = np.array([data_source.dataset.lengths[i] for i in data_source.indices])
        else:
            lengths = np.asarray(data_source.lengths)

        # cache sorted indices once
        sorted_idx = np.argsort(lengths)
        self.sorted_idx = sorted_idx[::-1] if descending else sorted_idx

        # pre-split into batches once (arrays of indices in subset-space)
        self.batches = [
            self.sorted_idx[i:i + self.batch_size]
            for i in range(0, len(self.sorted_idx), self.batch_size)
        ]

        self._rng = np.random.default_rng(self.seed)

    def __iter__(self):
        if self.shuffle:
            order = self._rng.permutation(len(self.batches))
            for j in order:
                yield self.batches[j].tolist()
        else:
            for b in self.batches:
                yield b.tolist()

    def __len__(self):
        return len(self.batches)


# --- 2. Lightning DataModule ---
class RiboAIQueuingDatamodule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path: str,
                 batch_size: int,
                 split: list,
                 nt_encoding_path: str,
                 codon_to_aa_encoding_path: str,
                 codon_encoding_path: str,
                 aa_encoding_path: str,
                 direction: str = "forward",
                 split_p: float = 0.9,
                 num_workers: int = 4,
                 seed: int = 42):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.split = split
        self.split_p = split_p
        self.num_workers = num_workers
        self.seed = seed
        self.direction = direction

        self.nt_enc = open_file(nt_encoding_path)
        self.c2aa_enc = open_file(codon_to_aa_encoding_path)
        self.c_enc = open_file(codon_encoding_path)
        self.aa_enc = open_file(aa_encoding_path)

        self.train_set = None
        self.val_set = None

    def setup(self, stage=None):
        if self.train_set is not None:
            return

        print(f"Loading dataset from {self.dataset_path} ...")
        df = pd.read_parquet(self.dataset_path)

        shared_data = df.to_dict('records')
        shared_lengths = df['sequence'].map(len).values
        total_len = len(shared_data)

        self.train_dataset_obj = RiboAIQueuingDataset(
            data=shared_data,
            lengths=shared_lengths,
            nt_encoding=self.nt_enc,
            codon_to_aa_encoding=self.c2aa_enc,
            codon_encoding=self.c_enc,
            aa_encoding=self.aa_enc
        )

        self.val_dataset_obj = RiboAIQueuingDataset(
            data=shared_data,
            lengths=shared_lengths,
            nt_encoding=self.nt_enc,
            codon_to_aa_encoding=self.c2aa_enc,
            codon_encoding=self.c_enc,
            aa_encoding=self.aa_enc,
        )

        if self.split is not None:
            train_indices = self.split[0]
            val_indices = self.split[1]
        else:
            g = Generator().manual_seed(self.seed)
            train_len = int(total_len * self.split_p)
            indices = torch.randperm(total_len, generator=g).tolist()
            train_indices = indices[:train_len]
            val_indices = indices[train_len:]

        self.train_set = Subset(self.train_dataset_obj, train_indices)

        # Validation sorting
        val_sorted_args = np.argsort([self.val_dataset_obj.lengths[i] for i in val_indices])[::-1]
        sorted_val_indices = [val_indices[i] for i in val_sorted_args]
        self.val_set = Subset(self.val_dataset_obj, sorted_val_indices)

    def worker_init_fn(self, worker_id):
        # We need to add the current epoch to the worker seed to ensure new augmentations per epoch
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        worker_seed = self.seed + worker_id + epoch
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)

    def train_dataloader(self):
        batch_sampler = SortedLengthBatchSampler(
            data_source=self.train_set,
            batch_size=self.batch_size,
            shuffle=True
        )

        return DataLoader(
            self.train_set,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset_obj.collate_fn,
            persistent_workers=True,
            worker_init_fn=self.worker_init_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset_obj.collate_fn,
            persistent_workers=True,
            worker_init_fn=self.worker_init_fn
        )