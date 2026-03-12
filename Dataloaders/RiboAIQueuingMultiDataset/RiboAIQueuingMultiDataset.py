import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

import pyarrow.parquet as pq

class RiboAIQueuingDatasetMultiDataset(Dataset):
    """
    Returns per-sample:
      (transcript_id, encoded_sequence[T,F], ribo[T], classes[T], mask[T], total_reads)

    Collate returns:
      (ids_sorted, packed_sequence, ribo_padded, classes_padded, lengths_sorted, mask_padded_bool, weights)
    """

    def __init__(
        self,
        nt_encoding: dict,
        codon_to_aa_encoding: dict,
        codon_encoding: dict,
        aa_encoding: dict,
        dataset_path: str | None = None,
        data: dict | None = None,
        lengths: np.ndarray | None = None,
        seed: int = 42
    ):
        super().__init__()

        self.data_records = data
        self.lengths = np.asarray(lengths)

        # Encodings / maps
        self.codon_map = codon_encoding
        self.aa_map = aa_encoding
        self.codon2aa_map = codon_to_aa_encoding
        self.nt_encoding = nt_encoding
        self.num_codons = 64
        self.n_codons = len(self.codon_map)
        self.n_aa = len(self.aa_map)
        self.onehot2nt = {np.argmax(v).item(): k for  k, v in self.nt_encoding.items()}
        self.codon_idx_to_aa_idx = {v: self.aa_map[self.codon2aa_map[k]] for k, v in self.codon_map.items()}
        self.num_datasets = len(self.data_records["ribo_profiles"].keys())
        self.datasets_names = list(self.data_records["ribo_profiles"].keys())

        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self._encoded_cache = [None] * len(self.data_records["ref"])
    def __len__(self) -> int:
        return len(self.data_records["ref"]) * self.num_datasets


    def _extract_features(self, nucleotide_sequence_per_codon) -> np.ndarray:
        # --- ref ---

        raw_nt_sequence = np.stack([np.stack(c) for c in nucleotide_sequence_per_codon])
        batch_size = raw_nt_sequence.shape[0]
        nt_sequence = raw_nt_sequence.reshape(batch_size, -1)

        codon_sequence = np.stack([np.stack([self.onehot2nt[np.argmax(n).item()] for n in c]) for c in raw_nt_sequence])
        codon_sequence = np.char.add(np.char.add(codon_sequence[:, 0], codon_sequence[:, 1]), codon_sequence[:, 2])
        codon_sequence = np.stack([self.codon_map[c] for c in codon_sequence])

        aa_sequence = np.stack([self.codon_idx_to_aa_idx[c] for c in codon_sequence])
        aa_sequence = np.eye(self.n_aa, dtype=np.int32)[aa_sequence]
        codon_sequence = np.eye(self.num_codons, dtype=np.int32)[codon_sequence]


        concatenated_sequence = np.concatenate([nt_sequence, codon_sequence, aa_sequence], axis=1)
        return concatenated_sequence

    def __getitem__(self, index: int):
        nT = len(self.data_records["ref"])
        nD = self.num_datasets
        if index < 0 or index >= nT * nD:
            raise IndexError(index)

        idx_dataset = index // nT  # 0..nD-1
        idx_transcript = index % nT  # 0..nT-1

        transcript_id = self.data_records["transcript_id"][idx_transcript]
        ref = self.data_records["ref"][idx_transcript]
        encoded = self._encoded_cache[idx_transcript]
        if encoded is None:
            encoded = self._extract_features(ref).astype(np.float32, copy=False)
            self._encoded_cache[idx_transcript] = encoded

        dataset_name = self.datasets_names[idx_dataset]
        ribo = self.data_records["ribo_profiles"][dataset_name][idx_transcript]
        css = self.data_records["css"][idx_transcript]

        return idx_dataset, transcript_id, encoded, ribo, css

    def collate_fn(self, batch):
        idx_datasets, ids, sequences, profiles, css_s = zip(*batch)

        lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
        lengths_sorted, order = lengths.sort(descending=True)
        order = order.tolist()  # <-- critical

        ids_datasets_sorted = torch.tensor([idx_datasets[i] for i in order], dtype=torch.long)
        ids_sorted = [ids[i] for i in order]
        seq_sorted = [torch.as_tensor(sequences[i], dtype=torch.float32) for i in order]
        prof_sorted = [torch.as_tensor(profiles[i], dtype=torch.float32) for i in order]
        css_sorted = [css_s[i] for i in order]

        seq_pad = pad_sequence(seq_sorted, batch_first=True, padding_value=0.0)
        prof_pad = pad_sequence(prof_sorted, batch_first=True, padding_value=0.0)

        Tmax = prof_pad.size(1)
        mask_pad = (torch.arange(Tmax).unsqueeze(0) < lengths_sorted.unsqueeze(1)).bool()

        seq_packed = pack_padded_sequence(seq_pad, lengths_sorted, batch_first=True, enforce_sorted=True)

        return ids_datasets_sorted, ids_sorted, seq_packed, prof_pad, lengths_sorted, mask_pad, css_sorted

def main():


    aa_encoding = yaml.safe_load(open("../../Datasets/encodings/aa_encoding.yaml"))
    codon2aa = yaml.safe_load(open("../../Datasets/encodings/codon2aa.yaml"))
    codon_encoding = yaml.safe_load(open("../../Datasets/encodings/codon_encoding.yaml"))
    nt_encoding = yaml.safe_load(open("../../Datasets/encodings/nt_encoding.yaml"))

    dataset_path = "../../Datasets/data/raw_datasets_with_css/grimson_2019.parquet"

    dataset = RiboAIQueuingDatasetMultiDataset(nt_encoding = nt_encoding,
                                   codon_to_aa_encoding = codon2aa,
                                   codon_encoding=codon_encoding,
                                   aa_encoding=aa_encoding,
                                   dataset_path=dataset_path)

    transcript_id, encoded, ribo, css = dataset.__getitem__(0)

    dataloader = DataLoader(dataset=dataset, batch_size=3, collate_fn=dataset.collate_fn)

    batch = next(iter(dataloader))
    print(batch)



if __name__ == "__main__":
    main()