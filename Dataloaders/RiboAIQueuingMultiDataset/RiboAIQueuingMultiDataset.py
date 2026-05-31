from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset


class RiboAIQueuingDatasetMultiDataset(Dataset):
    """
    Dataset for ribo-seq transcript-dataset pairs.

    Two access modes are supported:

    1. dataset_choice_mode="random"
        __len__ returns the number of unique transcripts.
        __getitem__(i) selects one available dataset randomly for transcript i.

    2. dataset_choice_mode="deterministic"
        __len__ returns the number of flat transcript-dataset pairs.
        __getitem__(j) returns the exact pair j = (transcript_id, dataset_name).

    The deterministic mode exposes flat-pair metadata:

        self.flat_transcript_ids     # [num_pairs], transcript id per pair
        self.flat_local_indices      # [num_pairs], local transcript index per pair
        self.flat_global_indices     # [num_pairs], global sequence-table index per pair
        self.flat_dataset_names      # [num_pairs], dataset name per pair
        self.flat_dataset_ids        # [num_pairs], integer dataset id per pair
        self.flat_lengths            # [num_pairs], transcript length per pair

    These arrays are required by transcript-balanced samplers in the datamodule.

    Returns per sample:
        (
            real_idx_dataset,
            transcript_id,
            encoded_sequence[T, F],
            codon_ids[T],
            ribo_profile[T],
            css,
        )

    Collate returns:
        (
            ids_datasets_sorted,   # [B]
            ids_sorted,            # list[str]
            seq_packed,            # PackedSequence
            prof_pad,              # [B, T_max]
            lengths_sorted,        # [B]
            mask_pad,              # [B, T_max]
            codon_ids_pad,         # [B, T_max]
            css_sorted,            # list
        )

    Speed-oriented details:
        1. Avoids string reconstruction of codons during feature extraction.
        2. Uses a triplet-index lookup table for codon IDs.
        3. Precomputes sequence features/codon IDs by default.
        4. Caches ribo profiles as contiguous float32 arrays.
        5. Avoids unnecessary copy=True conversions in collate_fn.
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
        precompute_features: bool = True,
        precompute_ribo: bool = True,
    ):
        super().__init__()

        if data is None:
            raise ValueError("data cannot be None.")

        if lengths is None:
            raise ValueError("lengths cannot be None.")

        self.data_records = data
        self.lengths = np.asarray(lengths, dtype=np.int64)

        self.codon_map = codon_encoding
        self.aa_map = aa_encoding
        self.codon2aa_map = codon_to_aa_encoding
        self.nt_encoding = nt_encoding
        self.datasets_encoding = datasets_encoding
        self.idx_to_dataset = {int(v): str(k) for k, v in self.datasets_encoding.items()}

        self.transcripts_ids = list(map(str, transcripts_ids))
        self.num_codons = 64
        self.n_codons = len(self.codon_map)
        self.n_aa = len(self.aa_map)

        self.dataset_choice_mode = str(dataset_choice_mode)
        self.seed = int(seed)
        self.precompute_features = bool(precompute_features)
        self.precompute_ribo = bool(precompute_ribo)

        if self.dataset_choice_mode not in {"random", "deterministic"}:
            raise ValueError(f"Unknown dataset_choice_mode={self.dataset_choice_mode}")

        # Global transcript id -> global sequence-table index.
        self.global_idx_by_tid = {
            str(tid): i
            for i, tid in enumerate(self.data_records["transcript_id"])
        }

        missing = [tid for tid in self.transcripts_ids if tid not in self.global_idx_by_tid]
        if missing:
            raise KeyError(
                "Some requested transcripts are missing from shared data. "
                f"First missing IDs: {missing[:10]}"
            )

        # Local transcript index -> transcript id / global sequence index.
        self.local_to_tid = np.asarray(self.transcripts_ids, dtype=object)
        self.local_to_global_idx = np.asarray(
            [self.global_idx_by_tid[tid] for tid in self.transcripts_ids],
            dtype=np.int64,
        )

        # ------------------------------------------------------------
        # Lookup tables for fast sequence feature extraction.
        # ------------------------------------------------------------
        self._build_encoding_lookup_tables()

        # One cache slot per global transcript in shared_data.
        self._feature_cache: list[np.ndarray | None] = [None] * len(self.data_records["ref"])
        self._codon_id_cache: list[np.ndarray | None] = [None] * len(self.data_records["ref"])

        # Ribo cache keyed by (transcript_id, dataset_name).
        self._ribo_cache: dict[tuple[str, str], np.ndarray] = {}

        # ------------------------------------------------------------
        # Precompute available datasets per local transcript.
        # ------------------------------------------------------------
        self.available_dataset_names_by_local: list[list[str]] = []
        self.available_dataset_ids_by_local: list[np.ndarray] = []

        for tid in self.transcripts_ids:
            available_map = self.data_records["ribo_profiles"][tid]
            available_names = list(available_map.keys())

            if len(available_names) == 0:
                raise RuntimeError(f"Transcript {tid} has no available ribo profiles.")

            missing_dataset_names = [
                name for name in available_names if name not in self.datasets_encoding
            ]
            if missing_dataset_names:
                raise KeyError(
                    f"Transcript {tid} has dataset names missing from datasets_encoding: "
                    f"{missing_dataset_names[:10]}"
                )

            available_ids = np.asarray(
                [int(self.datasets_encoding[name]) for name in available_names],
                dtype=np.int64,
            )

            self.available_dataset_names_by_local.append(available_names)
            self.available_dataset_ids_by_local.append(available_ids)

        # ------------------------------------------------------------
        # Build deterministic flat pair index:
        #   flat index j -> (transcript_id, local transcript index, dataset name)
        # ------------------------------------------------------------
        self.flat_transcript_ids: list[str] = []
        self.flat_local_indices: list[int] = []
        self.flat_global_indices: list[int] = []
        self.flat_dataset_names: list[str] = []
        self.flat_dataset_ids: list[int] = []
        self.flat_lengths: list[int] = []

        for local_idx, tid in enumerate(self.transcripts_ids):
            tid = str(tid)
            available_names = self.available_dataset_names_by_local[local_idx]
            global_idx = int(self.local_to_global_idx[local_idx])
            transcript_length = int(self.lengths[global_idx])

            for dataset_name in available_names:
                dataset_name = str(dataset_name)

                self.flat_transcript_ids.append(tid)
                self.flat_local_indices.append(local_idx)
                self.flat_global_indices.append(global_idx)
                self.flat_dataset_names.append(dataset_name)
                self.flat_dataset_ids.append(int(self.datasets_encoding[dataset_name]))
                self.flat_lengths.append(transcript_length)

        self.flat_transcript_ids = np.asarray(self.flat_transcript_ids, dtype=str)
        self.flat_local_indices = np.asarray(self.flat_local_indices, dtype=np.int64)
        self.flat_global_indices = np.asarray(self.flat_global_indices, dtype=np.int64)
        self.flat_dataset_names = np.asarray(self.flat_dataset_names, dtype=object)
        self.flat_dataset_ids = np.asarray(self.flat_dataset_ids, dtype=np.int64)
        self.flat_lengths = np.asarray(self.flat_lengths, dtype=np.int32)

        self.total_length = int(len(self.flat_local_indices))

        if self.total_length == 0:
            raise RuntimeError("Dataset has zero transcript-dataset pairs.")

        if not (
            len(self.flat_transcript_ids)
            == len(self.flat_local_indices)
            == len(self.flat_global_indices)
            == len(self.flat_dataset_names)
            == len(self.flat_dataset_ids)
            == len(self.flat_lengths)
        ):
            raise RuntimeError("Flat-pair metadata arrays have inconsistent lengths.")

        # Number of available datasets per transcript.
        self.num_datasets_by_transcript: dict[str, int] = defaultdict(int)
        for tid in self.flat_transcript_ids:
            self.num_datasets_by_transcript[str(tid)] += 1

        if self.precompute_features:
            self._precompute_sequence_features()

        if self.precompute_ribo:
            self._precompute_ribo_profiles()

    # ============================================================
    # Basic dataset API
    # ============================================================

    def __len__(self) -> int:
        if self.dataset_choice_mode == "random":
            return len(self.transcripts_ids)

        if self.dataset_choice_mode == "deterministic":
            return self.total_length

        raise ValueError(f"Unknown dataset_choice_mode={self.dataset_choice_mode}")

    def __getitem__(self, index: int):
        if self.dataset_choice_mode == "random":
            local_idx = int(index)
            transcript_id = str(self.local_to_tid[local_idx])

            available_names = self.available_dataset_names_by_local[local_idx]
            dataset_name = str(np.random.choice(available_names))

        elif self.dataset_choice_mode == "deterministic":
            local_idx = int(self.flat_local_indices[index])
            transcript_id = str(self.flat_transcript_ids[index])
            dataset_name = str(self.flat_dataset_names[index])

        else:
            raise ValueError(f"Unknown dataset_choice_mode={self.dataset_choice_mode}")

        global_idx = int(self.local_to_global_idx[local_idx])

        encoded, codon_ids = self._get_encoded_and_codon_ids(global_idx)
        ribo = self._get_ribo_profile(transcript_id, dataset_name)
        css = self.data_records["css"][global_idx]

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

    # ============================================================
    # Balancing / metadata helpers
    # ============================================================

    def make_dataset_balanced_weights(self, gamma: float = 1.0) -> torch.Tensor:
        """
        Returns one sampling weight per flat transcript-dataset pair.

        For pair j from dataset d:

            weight_j = N_d^(-gamma)

        gamma = 0.0 -> no dataset balancing
        gamma = 1.0 -> equal expected dataset sampling
        gamma = 0.5 -> softened dataset balancing
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

        weights = weights / np.mean(weights)
        return torch.as_tensor(weights, dtype=torch.double)

    def make_transcript_balanced_weights(
        self,
        dataset_balance_gamma: float = 0.0,
    ) -> torch.Tensor:
        """
        Returns one sampling weight per flat transcript-dataset pair.

        For transcript t measured in k_t datasets, each pair gets 1/k_t.
        Optional dataset balancing multiplies by N_d^(-dataset_balance_gamma).

            weight_{t,d} = (1 / k_t) * N_d^(-gamma)

        Recommended:
            dataset_balance_gamma = 0.0 or 0.25
        """
        if self.dataset_choice_mode != "deterministic":
            raise RuntimeError(
                "Transcript-balanced sampling requires dataset_choice_mode='deterministic'."
            )

        gamma = float(dataset_balance_gamma)

        transcript_counts: dict[str, int] = defaultdict(int)
        for tid in self.flat_transcript_ids:
            transcript_counts[str(tid)] += 1

        unique_ds, ds_counts = np.unique(self.flat_dataset_ids, return_counts=True)
        ds_count_map = {int(ds): float(n) for ds, n in zip(unique_ds, ds_counts)}

        weights = []
        for tid, ds_id in zip(self.flat_transcript_ids, self.flat_dataset_ids):
            k_t = max(transcript_counts[str(tid)], 1)
            n_d = max(ds_count_map[int(ds_id)], 1.0)
            w = (1.0 / float(k_t)) * (n_d ** (-gamma))
            weights.append(w)

        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / np.mean(weights)
        return torch.as_tensor(weights, dtype=torch.double)

    def dataset_pair_counts(self) -> dict[int, int]:
        unique_ids, counts = np.unique(self.flat_dataset_ids, return_counts=True)
        return {int(ds): int(c) for ds, c in zip(unique_ids, counts)}

    def transcript_dataset_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for tid in self.flat_transcript_ids:
            counts[str(tid)] += 1
        return dict(counts)

    def flat_pair_summary(self) -> dict:
        """
        Small metadata summary useful for debugging datamodule sampling.
        """
        k_values = np.asarray(list(self.transcript_dataset_counts().values()), dtype=np.int64)
        unique_k, k_counts = np.unique(k_values, return_counts=True)

        return {
            "num_flat_pairs": int(self.total_length),
            "num_unique_transcripts": int(len(np.unique(self.flat_transcript_ids))),
            "dataset_pair_counts": self.dataset_pair_counts(),
            "transcripts_by_num_datasets": {
                int(k): int(n) for k, n in zip(unique_k, k_counts)
            },
        }

    # ============================================================
    # Encoding / caching
    # ============================================================

    def _build_encoding_lookup_tables(self) -> None:
        """
        Builds fast lookup tables:

            nucleotide one-hot index triplet -> codon id
            codon id -> amino-acid id
            codon id -> codon one-hot
            amino-acid id -> amino-acid one-hot
        """
        nt_items = list(self.nt_encoding.items())

        if len(nt_items) == 0:
            raise ValueError("nt_encoding is empty.")

        nt_dim = len(np.asarray(nt_items[0][1]))
        self.nt_dim = int(nt_dim)

        self.nt_base_to_idx: dict[str, int] = {}
        self.nt_idx_to_base: dict[int, str] = {}

        for base, vec in nt_items:
            idx = int(np.argmax(np.asarray(vec)))
            base = str(base)
            self.nt_base_to_idx[base] = idx
            self.nt_idx_to_base[idx] = base

        # codon_triplet_lookup[a, b, c] = codon_id.
        codon_triplet_lookup = np.full(
            (self.nt_dim, self.nt_dim, self.nt_dim),
            fill_value=-1,
            dtype=np.int64,
        )

        for codon, codon_id in self.codon_map.items():
            codon = str(codon)

            if len(codon) != 3:
                raise ValueError(f"Invalid codon key {codon!r}; expected length 3.")

            try:
                a = self.nt_base_to_idx[codon[0]]
                b = self.nt_base_to_idx[codon[1]]
                c = self.nt_base_to_idx[codon[2]]
            except KeyError as exc:
                raise KeyError(
                    f"Codon {codon!r} contains nucleotide not found in nt_encoding."
                ) from exc

            codon_triplet_lookup[a, b, c] = int(codon_id)

        self.codon_triplet_lookup = codon_triplet_lookup

        codon_idx_to_aa_idx = np.zeros(self.num_codons, dtype=np.int64)

        for codon, codon_id in self.codon_map.items():
            codon = str(codon)
            codon_id = int(codon_id)

            if codon not in self.codon2aa_map:
                raise KeyError(f"Codon {codon!r} missing from codon_to_aa_encoding.")

            aa_name = self.codon2aa_map[codon]

            if aa_name not in self.aa_map:
                raise KeyError(f"Amino acid {aa_name!r} missing from aa_encoding.")

            codon_idx_to_aa_idx[codon_id] = int(self.aa_map[aa_name])

        self.codon_idx_to_aa_idx = codon_idx_to_aa_idx

        self.codon_onehot_lut = np.eye(self.num_codons, dtype=np.float32)
        self.aa_onehot_lut = np.eye(self.n_aa, dtype=np.float32)

    def _precompute_sequence_features(self) -> None:
        """
        Precompute encoded features and codon IDs for the transcripts used by this split.

        This is usually faster than lazy per-worker caching, because with multiple
        DataLoader workers each worker has its own dataset/cache copy.
        """
        for global_idx in np.unique(self.local_to_global_idx):
            global_idx = int(global_idx)
            ref = self.data_records["ref"][global_idx]
            encoded, codon_ids = self._extract_features_and_codon_ids(ref)
            self._feature_cache[global_idx] = encoded
            self._codon_id_cache[global_idx] = codon_ids

    def _precompute_ribo_profiles(self) -> None:
        """
        Cache all ribo profiles used by this split as contiguous float32 arrays.
        """
        for tid in self.transcripts_ids:
            available_map = self.data_records["ribo_profiles"][tid]

            for dataset_name, profile in available_map.items():
                key = (str(tid), str(dataset_name))
                self._ribo_cache[key] = np.ascontiguousarray(
                    np.asarray(profile, dtype=np.float32)
                )

    def _get_encoded_and_codon_ids(self, global_idx: int) -> tuple[np.ndarray, np.ndarray]:
        encoded = self._feature_cache[global_idx]
        codon_ids = self._codon_id_cache[global_idx]

        if encoded is None or codon_ids is None:
            ref = self.data_records["ref"][global_idx]
            encoded, codon_ids = self._extract_features_and_codon_ids(ref)
            self._feature_cache[global_idx] = encoded
            self._codon_id_cache[global_idx] = codon_ids

        return encoded, codon_ids

    def _get_ribo_profile(self, transcript_id: str, dataset_name: str) -> np.ndarray:
        key = (str(transcript_id), str(dataset_name))
        ribo = self._ribo_cache.get(key)

        if ribo is None:
            ribo_raw = self.data_records["ribo_profiles"][transcript_id][dataset_name]
            ribo = np.ascontiguousarray(np.asarray(ribo_raw, dtype=np.float32))
            self._ribo_cache[key] = ribo

        return ribo

    def _extract_features_and_codon_ids(
        self,
        nucleotide_sequence_per_codon,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Builds sequence features and codon IDs.

        Robust to both:
            1. rectangular arrays with shape [T, 3, nt_dim]
            2. object/ragged arrays where each codon is a list/array of 3 nt one-hots
        """
        # Fast path: works if the parquet cell is already rectangular.
        try:
            raw_nt_sequence = np.asarray(
                nucleotide_sequence_per_codon,
                dtype=np.float32,
            )
        except (ValueError, TypeError):
            # Robust fallback: this matches the logic of your original implementation.
            raw_nt_sequence = np.stack(
                [
                    np.stack(codon, axis=0).astype(np.float32, copy=False)
                    for codon in nucleotide_sequence_per_codon
                ],
                axis=0,
            )

        # If NumPy still made an object array, force the robust path.
        if raw_nt_sequence.dtype == object:
            raw_nt_sequence = np.stack(
                [
                    np.stack(codon, axis=0).astype(np.float32, copy=False)
                    for codon in nucleotide_sequence_per_codon
                ],
                axis=0,
            )

        raw_nt_sequence = np.ascontiguousarray(raw_nt_sequence, dtype=np.float32)

        if raw_nt_sequence.ndim != 3:
            raise ValueError(
                "Expected nucleotide_sequence_per_codon with shape [T, 3, nt_dim], "
                f"got shape {raw_nt_sequence.shape}."
            )

        if raw_nt_sequence.shape[1] != 3:
            raise ValueError(
                "Expected exactly 3 nucleotide one-hot vectors per codon, "
                f"got shape {raw_nt_sequence.shape}."
            )

        if raw_nt_sequence.shape[2] != self.nt_dim:
            raise ValueError(
                f"Expected nucleotide one-hot dim {self.nt_dim}, "
                f"got {raw_nt_sequence.shape[2]}."
            )

        T = raw_nt_sequence.shape[0]

        # Flatten 3 nucleotide one-hots per codon into one vector per codon.
        nt_sequence = raw_nt_sequence.reshape(T, -1)

        # Fast one-hot triplet -> codon id lookup.
        nt_idx = np.argmax(raw_nt_sequence, axis=-1).astype(np.int64, copy=False)

        codon_ids = self.codon_triplet_lookup[
            nt_idx[:, 0],
            nt_idx[:, 1],
            nt_idx[:, 2],
        ]

        if np.any(codon_ids < 0):
            bad = np.flatnonzero(codon_ids < 0)[:10]
            bad_triplets = nt_idx[bad].tolist()
            raise ValueError(
                "Found nucleotide triplet(s) not present in codon encoding. "
                f"First bad positions: {bad.tolist()}, triplets={bad_triplets}."
            )

        codon_ids = np.ascontiguousarray(codon_ids.astype(np.int64, copy=False))
        aa_ids = self.codon_idx_to_aa_idx[codon_ids]

        aa_onehot = self.aa_onehot_lut[aa_ids]
        codon_onehot = self.codon_onehot_lut[codon_ids]

        concatenated_sequence = np.concatenate(
            [
                nt_sequence.astype(np.float32, copy=False),
                codon_onehot,
                aa_onehot,
            ],
            axis=1,
        )

        concatenated_sequence = np.ascontiguousarray(
            concatenated_sequence,
            dtype=np.float32,
        )

        return concatenated_sequence, codon_ids

    # ============================================================
    # Collate
    # ============================================================

    def collate_fn(self, batch):
        idx_datasets, ids, sequences, codon_ids, profiles, css_s = zip(*batch)

        lengths = torch.as_tensor(
            [s.shape[0] for s in sequences],
            dtype=torch.long,
        )

        lengths_sorted, order = lengths.sort(descending=True)
        order_list = order.tolist()

        ids_datasets_sorted = torch.as_tensor(
            [idx_datasets[i] for i in order_list],
            dtype=torch.long,
        )

        ids_sorted = [ids[i] for i in order_list]

        # Arrays should already be contiguous and correctly typed from cache.
        # np.asarray(..., dtype=...) avoids copies when already correct.
        seq_sorted = [
            torch.from_numpy(np.asarray(sequences[i], dtype=np.float32))
            for i in order_list
        ]

        codon_ids_sorted = [
            torch.from_numpy(np.asarray(codon_ids[i], dtype=np.int64))
            for i in order_list
        ]

        prof_sorted = [
            torch.from_numpy(np.asarray(profiles[i], dtype=np.float32))
            for i in order_list
        ]

        css_sorted = [css_s[i] for i in order_list]

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
            ids_datasets_sorted,
            ids_sorted,
            seq_packed,
            prof_pad,
            lengths_sorted,
            mask_pad,
            codon_ids_pad,
            css_sorted,
        )
