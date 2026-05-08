import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset


class RiboAIQueuingDatasetMultiDataset(Dataset):
    """
    Returns per-sample:
      (real_idx_dataset, transcript_id, encoded_sequence[T,F], ribo[T], css)

    Collate returns:
      (ids_datasets_sorted, ids_sorted, packed_sequence, ribo_padded,
       lengths_sorted, mask_padded_bool, css_sorted)
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
        dataset_choice_mode: str = "random",  # "random" or "deterministic"
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

        if self.dataset_choice_mode not in {"random", "deterministic"}:
            raise ValueError(f"Unknown dataset_choice_mode={self.dataset_choice_mode}")

        self.global_idx_by_tid = {
            tid: i for i, tid in enumerate(self.data_records["transcript_id"])
        }

        self._feature_cache = [None] * len(self.data_records["ref"])
        self._codon_id_cache = [None] * len(self.data_records["ref"])

        # ------------------------------------------------------------
        # Build deterministic flat pair index:
        #   flat index j -> (local transcript index, dataset name)
        # ------------------------------------------------------------
        self.flat_local_indices: list[int] = []
        self.flat_dataset_names: list[str] = []
        self.flat_dataset_ids: list[int] = []
        self.flat_lengths: list[int] = []

        for local_idx, tid in enumerate(self.transcripts_ids):
            available_map = self.data_records["ribo_profiles"][tid]
            available_datasets = list(available_map.keys())

            global_idx = self.global_idx_by_tid[tid]
            transcript_length = int(self.lengths[global_idx])

            for dataset_name in available_datasets:
                self.flat_local_indices.append(local_idx)
                self.flat_dataset_names.append(dataset_name)
                self.flat_dataset_ids.append(int(self.datasets_encoding[dataset_name]))
                self.flat_lengths.append(transcript_length)

        self.flat_local_indices = np.asarray(self.flat_local_indices, dtype=np.int64)
        self.flat_dataset_ids = np.asarray(self.flat_dataset_ids, dtype=np.int64)
        self.flat_lengths = np.asarray(self.flat_lengths, dtype=np.int32)

        self.total_length = int(len(self.flat_local_indices))

        if self.total_length == 0:
            raise RuntimeError("Dataset has zero transcript-dataset pairs.")

    def __len__(self) -> int:
        if self.dataset_choice_mode == "random":
            return len(self.transcripts_ids)

        if self.dataset_choice_mode == "deterministic":
            return self.total_length

        raise ValueError(f"Unknown dataset_choice_mode={self.dataset_choice_mode}")

    def make_dataset_balanced_weights(self, gamma: float = 1.0) -> torch.Tensor:
        """
        Returns one sampling weight per flat transcript-dataset pair.

        For pair j from dataset d:

            weight_j = N_d^(-gamma)

        gamma = 0.0 -> no balancing
        gamma = 1.0 -> equal expected dataset sampling
        gamma = 0.5 -> softened balancing
        """
        if self.dataset_choice_mode != "deterministic":
            raise RuntimeError(
                "Balanced sampling requires dataset_choice_mode='deterministic'."
            )

        gamma = float(gamma)

        dataset_ids = self.flat_dataset_ids
        unique_ids, counts = np.unique(dataset_ids, return_counts=True)
        count_map = {int(ds): int(c) for ds, c in zip(unique_ids, counts)}

        weights = np.asarray(
            [count_map[int(ds)] ** (-gamma) for ds in dataset_ids],
            dtype=np.float64,
        )

        # Normalization is not required by WeightedRandomSampler, but it keeps
        # the numbers easier to inspect.
        weights = weights / np.mean(weights)

        return torch.as_tensor(weights, dtype=torch.double)

    def dataset_pair_counts(self) -> dict[int, int]:
        unique_ids, counts = np.unique(self.flat_dataset_ids, return_counts=True)
        return {int(ds): int(c) for ds, c in zip(unique_ids, counts)}

    def _extract_features_and_codon_ids(
            self,
            nucleotide_sequence_per_codon,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Builds sequence features and codon IDs.

        Parameters
        ----------
        nucleotide_sequence_per_codon:
            Iterable with length T, where each item contains the 3 nucleotide
            one-hot vectors of one codon.

        Returns
        -------
        concatenated_sequence:
            [T, F] float/int feature matrix:
                nucleotide features + codon one-hot + amino-acid one-hot

        codon_ids:
            [T] integer codon IDs in [0, num_codons - 1]
        """
        raw_nt_sequence = np.stack(
            [np.stack(c) for c in nucleotide_sequence_per_codon]
        )

        T = raw_nt_sequence.shape[0]

        # Flatten 3 nucleotide one-hots per codon into one vector per codon.
        nt_sequence = raw_nt_sequence.reshape(T, -1)

        # Convert nucleotide one-hot triplets back to nucleotide symbols.
        codon_letters = np.stack(
            [
                np.stack(
                    [
                        self.onehot2nt[np.argmax(n).item()]
                        for n in codon_nts
                    ]
                )
                for codon_nts in raw_nt_sequence
            ]
        )

        codon_strings = np.char.add(
            np.char.add(codon_letters[:, 0], codon_letters[:, 1]),
            codon_letters[:, 2],
        )

        # Integer codon IDs.
        codon_ids = np.asarray(
            [self.codon_map[codon] for codon in codon_strings],
            dtype=np.int64,
        )

        # Amino-acid IDs from codon IDs.
        aa_ids = np.asarray(
            [self.codon_idx_to_aa_idx[int(codon_id)] for codon_id in codon_ids],
            dtype=np.int64,
        )

        aa_onehot = np.eye(self.n_aa, dtype=np.float32)[aa_ids]
        codon_onehot = np.eye(self.num_codons, dtype=np.float32)[codon_ids]

        concatenated_sequence = np.concatenate(
            [
                nt_sequence.astype(np.float32, copy=False),
                codon_onehot,
                aa_onehot,
            ],
            axis=1,
        )

        return concatenated_sequence.astype(np.float32, copy=False), codon_ids

    def __getitem__(self, index: int):
        if self.dataset_choice_mode == "random":
            local_idx = int(index)
            transcript_id = self.transcripts_ids[local_idx]
            available_map = self.data_records["ribo_profiles"][transcript_id]
            available_datasets = list(available_map.keys())
            dataset_name = np.random.choice(available_datasets)

        elif self.dataset_choice_mode == "deterministic":
            local_idx = int(self.flat_local_indices[index])
            transcript_id = self.transcripts_ids[local_idx]
            dataset_name = self.flat_dataset_names[index]
            available_map = self.data_records["ribo_profiles"][transcript_id]

        else:
            raise ValueError(f"Unknown dataset_choice_mode={self.dataset_choice_mode}")

        global_idx = self.global_idx_by_tid[transcript_id]

        ref = self.data_records["ref"][global_idx]
        css = self.data_records["css"][global_idx]

        encoded = self._feature_cache[global_idx]
        codon_ids = self._codon_id_cache[global_idx]

        if encoded is None or codon_ids is None:
            encoded, codon_ids = self._extract_features_and_codon_ids(ref)
            self._feature_cache[global_idx] = encoded
            self._codon_id_cache[global_idx] = codon_ids

        ribo = available_map[dataset_name]

        if len(ribo) != encoded.shape[0]:
            raise ValueError(
                f"Length mismatch for transcript_id={transcript_id}, dataset={dataset_name}: "
                f"seq_len={encoded.shape[0]}, ribo_len={len(ribo)}, css_len={len(css)}"
            )

        if len(codon_ids) != encoded.shape[0]:
            raise ValueError(
                f"Codon ID length mismatch for transcript_id={transcript_id}: "
                f"seq_len={encoded.shape[0]}, codon_ids_len={len(codon_ids)}"
            )

        real_idx_dataset = int(self.datasets_encoding[dataset_name])

        return real_idx_dataset, transcript_id, encoded, codon_ids, ribo, css

    def collate_fn(self, batch):
        idx_datasets, ids, sequences, codon_ids, profiles, css_s = zip(*batch)

        lengths = torch.tensor(
            [s.shape[0] for s in sequences],
            dtype=torch.long,
        )

        lengths_sorted, order = lengths.sort(descending=True)
        order = order.tolist()

        ids_datasets_sorted = torch.as_tensor(
            [idx_datasets[i] for i in order],
            dtype=torch.long,
        )

        ids_sorted = [ids[i] for i in order]

        seq_sorted = [
            torch.from_numpy(
                np.array(sequences[i], dtype=np.float32, copy=True)
            )
            for i in order
        ]

        codon_ids_sorted = [
            torch.from_numpy(
                np.array(codon_ids[i], dtype=np.int64, copy=True)
            )
            for i in order
        ]

        prof_sorted = [
            torch.from_numpy(
                np.array(profiles[i], dtype=np.float32, copy=True)
            )
            for i in order
        ]

        css_sorted = [css_s[i] for i in order]

        seq_pad = pad_sequence(
            seq_sorted,
            batch_first=True,
            padding_value=0.0,
        )

        prof_pad = pad_sequence(
            prof_sorted,
            batch_first=True,
            padding_value=0.0,
        )

        codon_ids_pad = pad_sequence(
            codon_ids_sorted,
            batch_first=True,
            padding_value=0,
        )

        Tmax = prof_pad.size(1)

        mask_pad = (
                torch.arange(Tmax).unsqueeze(0)
                < lengths_sorted.unsqueeze(1)
        ).bool()

        # Ensure codon_ids has exactly the same T as profile/mask.
        if codon_ids_pad.size(1) > Tmax:
            codon_ids_pad = codon_ids_pad[:, :Tmax]
        elif codon_ids_pad.size(1) < Tmax:
            pad_T = Tmax - codon_ids_pad.size(1)
            codon_ids_pad = torch.cat(
                [
                    codon_ids_pad,
                    torch.zeros(
                        codon_ids_pad.size(0),
                        pad_T,
                        dtype=codon_ids_pad.dtype,
                    ),
                ],
                dim=1,
            )

        codon_ids_pad = codon_ids_pad.masked_fill(~mask_pad, 0)

        seq_packed = pack_padded_sequence(
            seq_pad,
            lengths_sorted,
            batch_first=True,
            enforce_sorted=True,
        )

        return (
            ids_datasets_sorted,  # 0: [B]
            ids_sorted,  # 1: list[str]
            seq_packed,  # 2: PackedSequence
            prof_pad,  # 3: [B, T]
            lengths_sorted,  # 4: [B]
            mask_pad,  # 5: [B, T]
            codon_ids_pad,  # 6: [B, T]
            css_sorted,  # 7: list
        )