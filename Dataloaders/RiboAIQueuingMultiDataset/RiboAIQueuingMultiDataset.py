import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset


class RiboAIQueuingDatasetMultiDataset(Dataset):
    """
    Returns per-sample:
      (real_idx_dataset, transcript_id, encoded_sequence[T,F], ribo[T], classes[T], mask[T], total_reads)

    Collate returns:
      (ids_datasets_sorted, ids_sorted, packed_sequence, ribo_padded, lengths_sorted, mask_padded_bool, css_sorted)
    """

    def __init__(
        self,
        nt_encoding: dict,
        codon_to_aa_encoding: dict,
        codon_encoding: dict,
        aa_encoding: dict,
        datasets_encoding: dict,
        transcripts_ids: list,
        data: dict | None = None,
        lengths: np.ndarray | None = None,
        seed: int = 42,
        dataset_choice_mode: str = "random", ##"random", "deterministic"ì

    ):
        super().__init__()

        self.data_records = data
        self.lengths = np.asarray(lengths)

        self.codon_map = codon_encoding
        self.aa_map = aa_encoding
        self.codon2aa_map = codon_to_aa_encoding
        self.nt_encoding = nt_encoding
        self.datasets_encoding = datasets_encoding
        self.idx_to_dataset = {v: k for k, v in self.datasets_encoding.items()}
        self.transcripts_ids = list(transcripts_ids)
        self.num_codons = 64
        self.n_codons = len(self.codon_map)
        self.n_aa = len(self.aa_map)
        self.onehot2nt = {np.argmax(v).item(): k for k, v in self.nt_encoding.items()}
        self.codon_idx_to_aa_idx = {
            v: self.aa_map[self.codon2aa_map[k]] for k, v in self.codon_map.items()
        }
        self.dataset_choice_mode = dataset_choice_mode
        self.seed = seed
        self.global_idx_by_tid = {
            tid: i for i, tid in enumerate(self.data_records["transcript_id"])
        }

        self._encoded_cache = [None] * len(self.data_records["ref"])

        self.linearized_indexing = [len(self.data_records["ribo_profiles"][t_id].keys()) for t_id in self.transcripts_ids]
        self.cum_sum_starts = np.cumsum(np.r_[0, self.linearized_indexing[:-1]])
        self.cum_sum_ends = np.cumsum(self.linearized_indexing)
        self.total_length = self.cum_sum_ends[-1]

    def __len__(self) -> int:
        if self.dataset_choice_mode == "random":
            return len(self.transcripts_ids)
        elif self.dataset_choice_mode == "deterministic":
            return self.total_length

    def _flat_to_logical(self, j):
        i = np.searchsorted(self.cum_sum_ends, j, side="right")
        offset = j - self.cum_sum_starts[i]
        return i, offset

    def _extract_features(self, nucleotide_sequence_per_codon) -> np.ndarray:
        raw_nt_sequence = np.stack([np.stack(c) for c in nucleotide_sequence_per_codon])
        batch_size = raw_nt_sequence.shape[0]
        nt_sequence = raw_nt_sequence.reshape(batch_size, -1)

        codon_sequence = np.stack(
            [np.stack([self.onehot2nt[np.argmax(n).item()] for n in c]) for c in raw_nt_sequence]
        )
        codon_sequence = np.char.add(
            np.char.add(codon_sequence[:, 0], codon_sequence[:, 1]),
            codon_sequence[:, 2]
        )
        codon_sequence = np.stack([self.codon_map[c] for c in codon_sequence])

        aa_sequence = np.stack([self.codon_idx_to_aa_idx[c] for c in codon_sequence])
        aa_sequence = np.eye(self.n_aa, dtype=np.int32)[aa_sequence]
        codon_sequence = np.eye(self.num_codons, dtype=np.int32)[codon_sequence]

        concatenated_sequence = np.concatenate([nt_sequence, codon_sequence, aa_sequence], axis=1)
        return concatenated_sequence

    def __getitem__(self, index: int):
        if self.dataset_choice_mode == "random":
            local_idx = index
            offset = None
        else:
            local_idx, offset = self._flat_to_logical(index)

        transcript_id = self.transcripts_ids[local_idx]
        global_idx = self.global_idx_by_tid[transcript_id]

        ref = self.data_records["ref"][global_idx]
        css = self.data_records["css"][global_idx]

        encoded = self._encoded_cache[global_idx]
        if encoded is None:
            encoded = self._extract_features(ref).astype(np.float32, copy=False)
            self._encoded_cache[global_idx] = encoded

        available_map = self.data_records["ribo_profiles"][transcript_id]
        available_datasets = list(available_map.keys())

        if self.dataset_choice_mode == "random":
            dataset_name = np.random.choice(available_datasets)
        elif self.dataset_choice_mode == "deterministic":
            dataset_name = available_datasets[offset]
        else:
            raise ValueError(f"Unknown dataset_choice_mode={self.dataset_choice_mode}")

        ribo = available_map[dataset_name]

        if len(ribo) != encoded.shape[0]:
            raise ValueError(
                f"Length mismatch for transcript_id={transcript_id}, dataset={dataset_name}: "
                f"seq_len={encoded.shape[0]}, ribo_len={len(ribo)}, css_len={len(css)}"
            )

        real_idx_dataset = self.datasets_encoding[dataset_name]
        return real_idx_dataset, transcript_id, encoded, ribo, css

    def collate_fn(self, batch):
        idx_datasets, ids, sequences, profiles, css_s = zip(*batch)

        lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
        lengths_sorted, order = lengths.sort(descending=True)
        order = order.tolist()

        ids_datasets_sorted = torch.as_tensor([idx_datasets[i] for i in order], dtype=torch.long)
        ids_sorted = [ids[i] for i in order]
        seq_sorted = [
            torch.tensor(np.array(sequences[i], copy=True), dtype=torch.float32)
            for i in order
        ]
        prof_sorted = [
            torch.tensor(np.array(profiles[i], copy=True), dtype=torch.float32)
            for i in order
        ]
        css_sorted = [css_s[i] for i in order]

        seq_pad = pad_sequence(seq_sorted, batch_first=True, padding_value=0.0)
        prof_pad = pad_sequence(prof_sorted, batch_first=True, padding_value=0.0)

        Tmax = prof_pad.size(1)
        mask_pad = (torch.arange(Tmax).unsqueeze(0) < lengths_sorted.unsqueeze(1)).bool()

        seq_packed = pack_padded_sequence(
            seq_pad,
            lengths_sorted,
            batch_first=True,
            enforce_sorted=True,
        )

        return ids_datasets_sorted, ids_sorted, seq_packed, prof_pad, lengths_sorted, mask_pad, css_sorted