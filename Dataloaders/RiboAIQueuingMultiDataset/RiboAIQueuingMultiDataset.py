from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

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
            sample_weight,
            dataset_bias_features[T, F_bias],  # when F_bias > 0
            ribo_replicas[R, T],  # only when use_ribo_replicas=True
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
            sample_weights_sorted, # [B]
            bias_features_pad,     # [B, T_max, F_bias], optional
            replica_pad,           # [B, R_max, T_max], optional
            replica_mask,          # [B, R_max], optional
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
        allowed_dataset_names_by_transcript: Mapping[str, Sequence[str]] | None = None,
        precompute_features: bool = True,
        precompute_ribo: bool = True,
        use_ribo_replicas: bool = False,
        additional_sequence_features: Mapping[str, Mapping] | None = None,
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
        self.use_ribo_replicas = bool(use_ribo_replicas)
        self.additional_sequence_features = self._parse_additional_sequence_features(
            additional_sequence_features
        )
        self.biological_feature_specs = [
            spec
            for spec in self.additional_sequence_features
            if spec["route"] in {"biological", "both"}
        ]
        self.dataset_bias_feature_specs = [
            spec
            for spec in self.additional_sequence_features
            if spec["route"] in {"dataset_bias", "both"}
        ]
        self.biological_extra_dim = sum(
            spec["dimension"] for spec in self.biological_feature_specs
        )
        self.dataset_bias_extra_dim = sum(
            spec["dimension"] for spec in self.dataset_bias_feature_specs
        )
        self.ribo_replicas_records = self.data_records.get("ribo_replicas")
        if self.use_ribo_replicas and not self.ribo_replicas_records:
            raise ValueError(
                "use_ribo_replicas=True but data['ribo_replicas'] is empty. The "
                "datamodule must populate per-replica profiles before constructing "
                "the dataset."
            )
        # Replica cache keyed by (transcript_id, dataset_name) -> [n_replicas, L].
        self._ribo_replicas_cache: dict[tuple[str, str], np.ndarray] = {}
        self.allowed_dataset_names_by_transcript = (
            {
                str(tid): set(map(str, dataset_names))
                for tid, dataset_names in allowed_dataset_names_by_transcript.items()
            }
            if allowed_dataset_names_by_transcript is not None
            else None
        )

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
        self._dataset_bias_feature_cache: list[np.ndarray | None] = [
            None
        ] * len(self.data_records["ref"])

        # Ribo cache keyed by (transcript_id, dataset_name).
        self._ribo_cache: dict[tuple[str, str], np.ndarray] = {}
        self._sample_weight_cache: dict[tuple[str, str], float] = {}

        # ------------------------------------------------------------
        # Precompute available datasets per local transcript.
        # ------------------------------------------------------------
        self.available_dataset_names_by_local: list[list[str]] = []
        self.available_dataset_ids_by_local: list[np.ndarray] = []

        for tid in self.transcripts_ids:
            available_map = self.data_records["ribo_profiles"][tid]
            available_names = list(available_map.keys())
            if self.allowed_dataset_names_by_transcript is not None:
                allowed = self.allowed_dataset_names_by_transcript.get(str(tid))
                if allowed is not None:
                    available_names = [
                        name for name in available_names if str(name) in allowed
                    ]

            if len(available_names) == 0:
                raise RuntimeError(
                    f"Transcript {tid} has no available ribo profiles after "
                    "applying allowed dataset filtering."
                )

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
        self.flat_sample_weights: list[float] = []

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
                self.flat_sample_weights.append(
                    self._get_sample_weight(tid, dataset_name)
                )

        self.flat_transcript_ids = np.asarray(self.flat_transcript_ids, dtype=str)
        self.flat_local_indices = np.asarray(self.flat_local_indices, dtype=np.int64)
        self.flat_global_indices = np.asarray(self.flat_global_indices, dtype=np.int64)
        self.flat_dataset_names = np.asarray(self.flat_dataset_names, dtype=object)
        self.flat_dataset_ids = np.asarray(self.flat_dataset_ids, dtype=np.int64)
        self.flat_lengths = np.asarray(self.flat_lengths, dtype=np.int32)
        self.flat_sample_weights = np.asarray(
            self.flat_sample_weights,
            dtype=np.float32,
        )

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
            == len(self.flat_sample_weights)
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

        if self.precompute_features and self.precompute_ribo:
            self._validate_lengths()

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

        encoded, codon_ids, dataset_bias_features = self._get_sequence_inputs(global_idx)
        ribo = self._get_ribo_profile(transcript_id, dataset_name)
        replicas = (
            self._get_ribo_replicas(transcript_id, dataset_name)
            if self.use_ribo_replicas
            else None
        )
        css = self.data_records["css"][global_idx]
        sample_weight = self._get_sample_weight(transcript_id, dataset_name)

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

        if replicas is not None and replicas.shape[1] != encoded.shape[0]:
            raise ValueError(
                f"Replica length mismatch for transcript_id={transcript_id}, "
                f"dataset={dataset_name}: seq_len={encoded.shape[0]}, "
                f"replica_shape={replicas.shape}"
            )

        real_idx_dataset = int(self.datasets_encoding[dataset_name])

        sample = (
            real_idx_dataset,
            transcript_id,
            encoded,
            codon_ids,
            ribo,
            css,
            sample_weight,
        )
        if self.dataset_bias_extra_dim > 0:
            sample = (*sample, dataset_bias_features)
        if replicas is not None:
            sample = (*sample, replicas)
        return sample

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

    @staticmethod
    def _parse_additional_sequence_features(
        feature_config: Mapping[str, Mapping] | None,
    ) -> list[dict]:
        if not feature_config:
            return []

        allowed_routes = {"none", "biological", "dataset_bias", "both"}
        specs = []
        for name, raw_spec in feature_config.items():
            spec = dict(raw_spec or {})
            route = str(spec.get("route", "none")).lower()
            if route not in allowed_routes:
                raise ValueError(
                    f"Invalid route {route!r} for sequence feature {name!r}; "
                    f"expected one of {sorted(allowed_routes)}."
                )
            if route == "none":
                continue
            dimension = int(spec.get("dimension", 1))
            if dimension <= 0:
                raise ValueError(
                    f"Feature {name!r} must have a positive dimension, got {dimension}."
                )
            specs.append(
                {
                    "name": str(name),
                    "route": route,
                    "dimension": dimension,
                    "scale": float(spec.get("scale", 1.0)),
                    "missing_values": tuple(
                        float(value) for value in spec.get("missing_values", [])
                    ),
                    "fill_value": float(spec.get("fill_value", 0.0)),
                }
            )
        return specs

    def _extract_additional_feature(
        self,
        *,
        global_idx: int,
        spec: Mapping,
        sequence_length: int,
    ) -> np.ndarray:
        name = str(spec["name"])
        feature_records = self.data_records.get("sequence_features", {})
        if name not in feature_records:
            raise KeyError(
                f"Configured sequence feature {name!r} was not loaded by the datamodule."
            )

        cell = feature_records[name][global_idx]
        try:
            values = np.asarray(cell, dtype=np.float32)
        except (TypeError, ValueError):
            values = np.stack(
                [np.asarray(row, dtype=np.float32) for row in cell], axis=0
            )
        if values.dtype == object:
            values = np.stack(
                [np.asarray(row, dtype=np.float32) for row in cell], axis=0
            )
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError(
                f"Sequence feature {name!r} must have shape [T] or [T, C], "
                f"got {values.shape}."
            )
        expected_dim = int(spec["dimension"])
        if values.shape != (sequence_length, expected_dim):
            raise ValueError(
                f"Sequence feature {name!r} has shape {values.shape}; expected "
                f"({sequence_length}, {expected_dim}) to match ref."
            )
        fill_value = float(spec["fill_value"])
        missing_values = tuple(spec.get("missing_values", ()))
        if missing_values:
            values = np.where(
                np.isin(values, np.asarray(missing_values, dtype=np.float32)),
                fill_value,
                values,
            )
        values = np.nan_to_num(
            values,
            nan=fill_value,
            posinf=fill_value,
            neginf=fill_value,
        )
        values = values * float(spec["scale"])
        return np.ascontiguousarray(values, dtype=np.float32)

    def _additional_features_for_route(
        self,
        *,
        global_idx: int,
        specs: Sequence[Mapping],
        sequence_length: int,
    ) -> np.ndarray:
        if not specs:
            return np.empty((sequence_length, 0), dtype=np.float32)
        return np.ascontiguousarray(
            np.concatenate(
                [
                    self._extract_additional_feature(
                        global_idx=global_idx,
                        spec=spec,
                        sequence_length=sequence_length,
                    )
                    for spec in specs
                ],
                axis=1,
            ),
            dtype=np.float32,
        )

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
            bio_extra = self._additional_features_for_route(
                global_idx=global_idx,
                specs=self.biological_feature_specs,
                sequence_length=encoded.shape[0],
            )
            if bio_extra.shape[1] > 0:
                encoded = np.ascontiguousarray(
                    np.concatenate((encoded, bio_extra), axis=1), dtype=np.float32
                )
            bias_extra = self._additional_features_for_route(
                global_idx=global_idx,
                specs=self.dataset_bias_feature_specs,
                sequence_length=encoded.shape[0],
            )
            self._feature_cache[global_idx] = encoded
            self._codon_id_cache[global_idx] = codon_ids
            self._dataset_bias_feature_cache[global_idx] = bias_extra

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

    def _validate_lengths(self) -> None:
        """
        Validate that ribo profile lengths match sequence lengths for all precomputed pairs.
        Runs at setup time so bad transcripts raise immediately rather than mid-epoch.
        """
        mismatches = []
        for i in range(len(self.flat_transcript_ids)):
            tid = str(self.flat_transcript_ids[i])
            dataset_name = str(self.flat_dataset_names[i])
            global_idx = int(self.flat_global_indices[i])

            encoded = self._feature_cache[global_idx]
            ribo = self._ribo_cache.get((tid, dataset_name))

            if encoded is None or ribo is None:
                continue

            if len(ribo) != encoded.shape[0]:
                mismatches.append(
                    f"  transcript_id={tid}, dataset={dataset_name}: "
                    f"seq_len={encoded.shape[0]}, ribo_len={len(ribo)}"
                )

        if mismatches:
            detail = "\n".join(mismatches[:20])
            raise RuntimeError(
                f"Length mismatches found in {len(mismatches)} transcript-dataset pair(s):\n{detail}"
            )

    def _get_sequence_inputs(
        self, global_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        encoded = self._feature_cache[global_idx]
        codon_ids = self._codon_id_cache[global_idx]
        bias_features = self._dataset_bias_feature_cache[global_idx]

        if encoded is None or codon_ids is None or bias_features is None:
            ref = self.data_records["ref"][global_idx]
            encoded, codon_ids = self._extract_features_and_codon_ids(ref)
            bio_extra = self._additional_features_for_route(
                global_idx=global_idx,
                specs=self.biological_feature_specs,
                sequence_length=encoded.shape[0],
            )
            if bio_extra.shape[1] > 0:
                encoded = np.ascontiguousarray(
                    np.concatenate((encoded, bio_extra), axis=1), dtype=np.float32
                )
            bias_features = self._additional_features_for_route(
                global_idx=global_idx,
                specs=self.dataset_bias_feature_specs,
                sequence_length=encoded.shape[0],
            )
            self._feature_cache[global_idx] = encoded
            self._codon_id_cache[global_idx] = codon_ids
            self._dataset_bias_feature_cache[global_idx] = bias_features

        return encoded, codon_ids, bias_features

    def _get_ribo_profile(self, transcript_id: str, dataset_name: str) -> np.ndarray:
        key = (str(transcript_id), str(dataset_name))
        ribo = self._ribo_cache.get(key)

        if ribo is None:
            ribo_raw = self.data_records["ribo_profiles"][transcript_id][dataset_name]
            ribo = np.ascontiguousarray(np.asarray(ribo_raw, dtype=np.float32))
            self._ribo_cache[key] = ribo

        return ribo

    def _get_ribo_replicas(self, transcript_id: str, dataset_name: str) -> np.ndarray:
        """Return the per-replica profiles as a contiguous [n_replicas, L] array."""
        key = (str(transcript_id), str(dataset_name))
        replicas = self._ribo_replicas_cache.get(key)

        if replicas is None:
            replicas_raw = self.ribo_replicas_records[transcript_id][dataset_name]
            replicas = np.ascontiguousarray(np.asarray(replicas_raw, dtype=np.float32))
            if replicas.ndim != 2:
                raise ValueError(
                    f"Expected replicas with shape [n_replicas, L] for "
                    f"transcript={transcript_id}, dataset={dataset_name}, got "
                    f"shape {replicas.shape}."
                )
            self._ribo_replicas_cache[key] = replicas

        return replicas

    def _get_sample_weight(self, transcript_id: str, dataset_name: str) -> float:
        key = (str(transcript_id), str(dataset_name))
        cached = self._sample_weight_cache.get(key)
        if cached is not None:
            return cached

        value = 1.0
        sample_weights = self.data_records.get("sample_weights")
        if sample_weights is not None:
            transcript_weights = sample_weights.get(str(transcript_id))
            if transcript_weights is not None:
                value = transcript_weights.get(str(dataset_name), 1.0)

        value = float(value)
        if not np.isfinite(value):
            value = 1.0

        value = float(np.clip(value, 0.0, 1.0))
        self._sample_weight_cache[key] = value
        return value

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
        has_bias_features = self.dataset_bias_extra_dim > 0
        has_replicas = len(batch[0]) == 8 + int(has_bias_features)
        (
            idx_datasets,
            ids,
            sequences,
            codon_ids,
            profiles,
            css_s,
            sample_weights,
            *optional_values,
        ) = zip(*batch)
        bias_features = optional_values[0] if has_bias_features else None
        replicas = optional_values[int(has_bias_features)] if has_replicas else None

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

        bias_features_sorted = None
        if bias_features is not None:
            bias_features_sorted = [
                torch.from_numpy(np.asarray(bias_features[i], dtype=np.float32))
                for i in order_list
            ]

        prof_sorted = [
            torch.from_numpy(np.asarray(profiles[i], dtype=np.float32))
            for i in order_list
        ]

        replicas_sorted = None
        if replicas is not None:
            replicas_sorted = [
                torch.from_numpy(np.asarray(replicas[i], dtype=np.float32))
                for i in order_list
            ]

        css_sorted = [css_s[i] for i in order_list]
        sample_weights_sorted = torch.as_tensor(
            [sample_weights[i] for i in order_list],
            dtype=torch.float32,
        )

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

        bias_features_pad = None
        if bias_features_sorted is not None:
            bias_features_pad = pad_sequence(
                bias_features_sorted,
                batch_first=True,
                padding_value=0.0,
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

        replica_pad = None
        replica_mask = None
        if replicas_sorted is not None:
            Rmax = max(int(rep.shape[0]) for rep in replicas_sorted)
            replica_pad = torch.zeros(
                len(replicas_sorted),
                Rmax,
                Tmax,
                dtype=torch.float32,
            )
            replica_mask = torch.zeros(
                len(replicas_sorted),
                Rmax,
                dtype=torch.bool,
            )
            for row_idx, rep in enumerate(replicas_sorted):
                if rep.ndim != 2:
                    raise ValueError(
                        f"Expected replica tensor with shape [n_replicas, T], "
                        f"got {tuple(rep.shape)}."
                    )
                n_rep, rep_len = int(rep.shape[0]), int(rep.shape[1])
                if rep_len != int(lengths_sorted[row_idx].item()):
                    raise ValueError(
                        "Replica length mismatch after sorting: "
                        f"rep_len={rep_len}, length={int(lengths_sorted[row_idx].item())}."
                    )
                replica_pad[row_idx, :n_rep, :rep_len] = rep
                replica_mask[row_idx, :n_rep] = True

        seq_packed = pack_padded_sequence(
            seq_pad,
            lengths_sorted,
            batch_first=True,
            enforce_sorted=True,
        )

        collated = (
            ids_datasets_sorted,
            ids_sorted,
            seq_packed,
            prof_pad,
            lengths_sorted,
            mask_pad,
            codon_ids_pad,
            css_sorted,
            sample_weights_sorted,
        )
        if bias_features_pad is not None:
            collated = (*collated, bias_features_pad)
        if replica_pad is not None and replica_mask is not None:
            collated = (*collated, replica_pad, replica_mask)
        return collated
