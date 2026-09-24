from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset

from Utils.transcript_batch_metadata import ValidatedTranscriptMetadata


def transcript_group_indices_from_ids(
    transcript_ids: Sequence[str],
    expected_pair_rows_by_transcript: Mapping[str, int] | None = None,
) -> torch.LongTensor:
    """Assign deterministic, microbatch-local integer IDs to transcripts.

    The first distinct transcript receives group zero, the next unseen
    transcript group one, and so on. The tensor is consumed directly by the
    vectorized sample-loss reducer. Gamma centering receives the same sorted
    transcript-ID rows, so the two mechanisms cannot disagree about row
    identity. The debug assertion below verifies the one-to-one mapping once
    in collate rather than regrouping Python strings in every training step.
    """
    group_by_transcript: dict[str, int] = {}
    group_indices: list[int] = []
    for raw_transcript_id in transcript_ids:
        transcript_id = str(raw_transcript_id)
        group_index = group_by_transcript.get(transcript_id)
        if group_index is None:
            group_index = len(group_by_transcript)
            group_by_transcript[transcript_id] = group_index
        group_indices.append(group_index)

    result = torch.tensor(group_indices, dtype=torch.long)
    if __debug__:
        transcript_by_group: dict[int, str] = {}
        observed_rows_by_transcript: dict[str, int] = defaultdict(int)
        for transcript_id, group_index in zip(
            map(str, transcript_ids),
            group_indices,
            strict=True,
        ):
            observed_rows_by_transcript[transcript_id] += 1
            previous = transcript_by_group.setdefault(group_index, transcript_id)
            if previous != transcript_id:
                raise AssertionError(
                    "One transcript_group_index was assigned to different "
                    f"transcripts: {previous!r} and {transcript_id!r}."
                )
        if expected_pair_rows_by_transcript is not None:
            for transcript_id, observed_rows in observed_rows_by_transcript.items():
                expected_rows = int(
                    expected_pair_rows_by_transcript.get(transcript_id, 0)
                )
                if observed_rows != expected_rows:
                    raise RuntimeError(
                        "Incomplete transcript group reached collate: "
                        f"transcript={transcript_id}, observed_pair_rows="
                        f"{observed_rows}, expected_pair_rows={expected_rows}. "
                        "Transcript-balanced reduction requires atomic groups."
                    )
    return result


class RiboUnmixMultiDataset(Dataset):
    """
    Dataset for ribo-seq transcript-dataset pairs.

    The dataset exposes deterministic transcript-dataset pairs. The grouped
    samplers rely on ``__len__`` and ``__getitem__`` referring to the flat pair
    index, so there is no random per-transcript dataset-selection mode.

    It exposes flat-pair metadata:

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
            biological_extra_features[T, F_extra],
            codon_ids[T],
            ribo_profile[T],
            css,
            sample_weight,
            dataset_quality_rank,
            dataset_quality_weight,
            dataset_bias_features[T, F_bias],  # when F_bias > 0
            ribo_replicas[R, T],
            execution_microbatch_metadata,     # dict or None
        )

    Collate returns:
        (
            ids_datasets_sorted,   # [B]
            ids_sorted,            # list[str]
            seq_packed,            # one dense biological sequence per transcript
            prof_pad,              # [B, T_max]
            lengths_sorted,        # [B]
            mask_pad,              # [B, T_max]
            codon_ids_pad,         # [B, T_max]
            css_sorted,            # list
            sample_weights_sorted, # [B]
            dataset_quality_ranks,  # [B], rank 1 is best
            dataset_quality_weights,# [B], positive rank-derived weight
            transcript_group_index, # [B], local IDs from ids_sorted
            bias_features_pad,     # [B, T_max, F_bias], optional
            replica_pad,           # [B, R_max, T_max]
            replica_mask,          # [B, R_max]
            transcript_metadata,   # validated host grouping and lengths
            execution_microbatch_metadata, # dict or None
        )

    Speed-oriented details:
        1. Avoids string reconstruction of codons during feature extraction.
        2. Uses a triplet-index lookup table for codon IDs.
        3. Caches compact uint8 codon IDs, not dense base one-hots.
        4. Caches ribo profiles as contiguous float32 arrays.
        5. Builds dense biological inputs once per transcript group in collate.
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
        precompute_features: bool = True,
        precompute_ribo: bool = True,
        additional_sequence_features: Mapping[str, Mapping] | None = None,
    ):
        super().__init__()

        if data is None:
            raise ValueError("data cannot be None.")

        if lengths is None:
            raise ValueError("lengths cannot be None.")

        self.data_records = data
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.sequence_representation = str(
            self.data_records.get("sequence_representation", "nucleotide_onehot")
        )
        if self.sequence_representation not in {
            "nucleotide_onehot",
            "codon_tokens",
        }:
            raise ValueError(
                "Unknown sequence representation: "
                f"{self.sequence_representation!r}."
            )

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

        self.precompute_features = bool(precompute_features)
        self.precompute_ribo = bool(precompute_ribo)
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
        if not self.ribo_replicas_records:
            raise ValueError(
                "data['ribo_replicas'] is required and cannot be empty. Every "
                "sample must carry at least one replica profile."
            )
        # Replica cache keyed by (transcript_id, dataset_name) -> [n_replicas, L].
        self._ribo_replicas_cache: dict[tuple[str, str], np.ndarray] = {}
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

        # Keep only compact codon IDs plus genuinely transcript-specific extra
        # features. The former dense 97-float base representation consumed
        # roughly 4 GiB for MANE and duplicated identical sequence rows across
        # dataset observations during collate.
        self._biological_extra_feature_cache: list[np.ndarray | None] = [
            None
        ] * len(self.data_records["ref"])
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

            if len(available_names) == 0:
                raise RuntimeError(
                    f"Transcript {tid} has no available ribo profiles."
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
        return self.total_length

    def __getitem__(self, index: int):
        execution_metadata = None
        if isinstance(index, (tuple, list)):
            if len(index) != 8 or str(index[0]) != "execution_microbatch_v1":
                raise ValueError(f"Invalid execution-microbatch dataset index: {index!r}")
            (
                _,
                index,
                logical_batch_index,
                execution_chunk_index,
                execution_chunk_count,
                logical_group_count,
                execution_group_count,
                logical_pair_count,
            ) = index
            execution_metadata = {
                "logical_batch_index": int(logical_batch_index),
                "execution_chunk_index": int(execution_chunk_index),
                "execution_chunk_count": int(execution_chunk_count),
                "logical_group_count": int(logical_group_count),
                "execution_group_count": int(execution_group_count),
                "logical_pair_count": int(logical_pair_count),
            }
            if int(index) == -1 and int(execution_group_count) == 0:
                return {'empty_execution': execution_metadata}
            if int(index) < 0 or int(execution_group_count) < 1:
                raise ValueError('Invalid active execution-microbatch index.')
        index = int(index)
        local_idx = int(self.flat_local_indices[index])
        transcript_id = str(self.flat_transcript_ids[index])
        dataset_name = str(self.flat_dataset_names[index])

        global_idx = int(self.local_to_global_idx[local_idx])

        biological_extra_features, codon_ids, dataset_bias_features = (
            self._get_sequence_inputs(global_idx)
        )
        ribo = self._get_ribo_profile(transcript_id, dataset_name)
        replicas = self._get_ribo_replicas(transcript_id, dataset_name)
        css = self.data_records["css"][global_idx]
        sample_weight = self._get_sample_weight(transcript_id, dataset_name)
        dataset_quality_rank, dataset_quality_weight = self._get_dataset_quality(
            dataset_name
        )

        sequence_length = int(codon_ids.shape[0])
        if len(ribo) != sequence_length:
            raise ValueError(
                f"Length mismatch for transcript_id={transcript_id}, dataset={dataset_name}: "
                f"seq_len={sequence_length}, ribo_len={len(ribo)}, css_len={len(css)}"
            )

        if biological_extra_features.shape != (
            sequence_length,
            self.biological_extra_dim,
        ):
            raise ValueError(
                "Biological extra-feature shape mismatch for "
                f"transcript_id={transcript_id}: got "
                f"{biological_extra_features.shape}, expected "
                f"({sequence_length}, {self.biological_extra_dim})."
            )

        if replicas.shape[1] != sequence_length:
            raise ValueError(
                f"Replica length mismatch for transcript_id={transcript_id}, "
                f"dataset={dataset_name}: seq_len={sequence_length}, "
                f"replica_shape={replicas.shape}"
            )

        real_idx_dataset = int(self.datasets_encoding[dataset_name])

        sample = (
            real_idx_dataset,
            transcript_id,
            biological_extra_features,
            codon_ids,
            ribo,
            css,
            sample_weight,
            dataset_quality_rank,
            dataset_quality_weight,
        )
        if self.dataset_bias_extra_dim > 0:
            sample = (*sample, dataset_bias_features)
        return (*sample, replicas, execution_metadata)

    def flat_pair_summary(self) -> dict:
        """
        Small metadata summary useful for debugging datamodule sampling.
        """
        _, k_values = np.unique(self.flat_transcript_ids, return_counts=True)
        unique_k, k_counts = np.unique(k_values, return_counts=True)

        dataset_pair_counts: dict[int, int] = {}
        for dataset_id in self.flat_dataset_ids:
            dataset_id = int(dataset_id)
            dataset_pair_counts[dataset_id] = dataset_pair_counts.get(dataset_id, 0) + 1

        return {
            "num_flat_pairs": int(self.total_length),
            "num_unique_transcripts": int(len(np.unique(self.flat_transcript_ids))),
            "dataset_pair_counts": dataset_pair_counts,
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

        # A codon completely determines the original 97-dimensional base
        # representation: 3 nucleotide one-hots, codon one-hot, and amino-acid
        # one-hot. Materialize this tiny table once, then index it only for the
        # unique transcripts present in the current collated batch.
        base_feature_lut = np.empty(
            (self.num_codons, 3 * self.nt_dim + self.num_codons + self.n_aa),
            dtype=np.float32,
        )
        seen_ids: set[int] = set()
        for raw_codon, raw_codon_id in self.codon_map.items():
            codon = str(raw_codon)
            codon_id = int(raw_codon_id)
            if codon_id < 0 or codon_id >= self.num_codons:
                raise ValueError(f"Codon ID out of range: {codon!r} -> {codon_id}.")
            nucleotide_features = np.concatenate(
                [
                    np.asarray(self.nt_encoding[base], dtype=np.float32)
                    for base in codon
                ]
            )
            base_feature_lut[codon_id] = np.concatenate(
                [
                    nucleotide_features,
                    self.codon_onehot_lut[codon_id],
                    self.aa_onehot_lut[codon_idx_to_aa_idx[codon_id]],
                ]
            )
            seen_ids.add(codon_id)
        if seen_ids != set(range(self.num_codons)):
            raise ValueError(
                "Codon encoding must define every contiguous ID from 0 to "
                f"{self.num_codons - 1}."
            )
        self.base_biological_feature_lut = np.ascontiguousarray(base_feature_lut)

    def _precompute_sequence_features(self) -> None:
        """
        Precompute compact codon IDs and optional transcript-specific features.

        The dense base one-hots are deliberately not cached. They are recovered
        from ``base_biological_feature_lut`` for unique transcripts in collate.
        """
        for global_idx in np.unique(self.local_to_global_idx):
            global_idx = int(global_idx)
            ref = self.data_records["ref"][global_idx]
            codon_ids = self._sequence_cell_to_codon_ids(ref)
            bio_extra = self._additional_features_for_route(
                global_idx=global_idx,
                specs=self.biological_feature_specs,
                sequence_length=codon_ids.shape[0],
            )
            bias_extra = self._additional_features_for_route(
                global_idx=global_idx,
                specs=self.dataset_bias_feature_specs,
                sequence_length=codon_ids.shape[0],
            )
            self._biological_extra_feature_cache[global_idx] = bio_extra
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

            codon_ids = self._codon_id_cache[global_idx]
            ribo = self._ribo_cache.get((tid, dataset_name))

            if codon_ids is None or ribo is None:
                continue

            if len(ribo) != codon_ids.shape[0]:
                mismatches.append(
                    f"  transcript_id={tid}, dataset={dataset_name}: "
                    f"seq_len={codon_ids.shape[0]}, ribo_len={len(ribo)}"
                )

        if mismatches:
            detail = "\n".join(mismatches[:20])
            raise RuntimeError(
                f"Length mismatches found in {len(mismatches)} transcript-dataset pair(s):\n{detail}"
            )

    def _get_sequence_inputs(
        self, global_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        biological_extra = self._biological_extra_feature_cache[global_idx]
        codon_ids = self._codon_id_cache[global_idx]
        bias_features = self._dataset_bias_feature_cache[global_idx]

        if biological_extra is None or codon_ids is None or bias_features is None:
            ref = self.data_records["ref"][global_idx]
            codon_ids = self._sequence_cell_to_codon_ids(ref)
            biological_extra = self._additional_features_for_route(
                global_idx=global_idx,
                specs=self.biological_feature_specs,
                sequence_length=codon_ids.shape[0],
            )
            bias_features = self._additional_features_for_route(
                global_idx=global_idx,
                specs=self.dataset_bias_feature_specs,
                sequence_length=codon_ids.shape[0],
            )
            self._biological_extra_feature_cache[global_idx] = biological_extra
            self._codon_id_cache[global_idx] = codon_ids
            self._dataset_bias_feature_cache[global_idx] = bias_features

        return biological_extra, codon_ids, bias_features

    def _get_ribo_profile(self, transcript_id: str, dataset_name: str) -> np.ndarray:
        key = (str(transcript_id), str(dataset_name))
        ribo = self._ribo_cache.get(key)

        if ribo is None:
            ribo_raw = self.data_records["ribo_profiles"][transcript_id][dataset_name]
            ribo = np.ascontiguousarray(np.asarray(ribo_raw, dtype=np.float32))
            if ribo.ndim != 1:
                raise ValueError(
                    "Expected consensus profile with shape [positions] for "
                    f"transcript={transcript_id}, dataset={dataset_name}, got "
                    f"shape {ribo.shape}."
                )
            if not np.isfinite(ribo).all() or bool((ribo < 0.0).any()):
                raise ValueError(
                    "Consensus targets must be finite and non-negative for "
                    f"transcript={transcript_id}, dataset={dataset_name}."
                )
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
            if replicas.shape[0] == 0:
                raise ValueError(
                    f"No replicas for transcript={transcript_id}, "
                    f"dataset={dataset_name}."
                )
            if not np.isfinite(replicas).all() or bool((replicas < 0.0).any()):
                raise ValueError(
                    "Replica targets must be finite and non-negative for "
                    f"transcript={transcript_id}, dataset={dataset_name}."
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
            raise ValueError(
                "Transcript sample weight must be finite for "
                f"transcript={transcript_id}, dataset={dataset_name}; got {value}."
            )
        if value <= 0.0:
            raise ValueError(
                "Transcript sample weight must be strictly positive for "
                f"transcript={transcript_id}, dataset={dataset_name}; got {value}."
            )

        # Median-normalized weights, including values above one, are preserved.
        self._sample_weight_cache[key] = value
        return value

    def _get_dataset_quality(self, dataset_name: str) -> tuple[float, float]:
        """Return batch metadata independent of transcript/sample weights."""
        name = str(dataset_name)
        rank = float(self.data_records.get("dataset_quality_ranks", {}).get(name, np.nan))
        weight = float(
            self.data_records.get("dataset_quality_weights", {}).get(name, 1.0)
        )
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"Dataset quality weight for {name!r} must be finite and positive, "
                f"got {weight}."
            )
        return rank, weight

    def _sequence_cell_to_codon_ids(self, sequence_cell) -> np.ndarray:
        if self.sequence_representation == "codon_tokens":
            tokens = np.asarray(sequence_cell).reshape(-1)
            try:
                values = np.fromiter(
                    (self.codon_map[str(token).upper()] for token in tokens),
                    dtype=np.uint8,
                    count=int(tokens.size),
                )
            except KeyError as exc:
                raise ValueError(
                    f"Sequence contains a codon absent from codon_encoding: {exc.args[0]!r}."
                ) from exc
            return np.ascontiguousarray(values, dtype=np.uint8)
        return self._extract_codon_ids(sequence_cell)

    def _extract_codon_ids(self, nucleotide_sequence_per_codon) -> np.ndarray:
        """Convert the stored nucleotide one-hots to compact codon IDs."""
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

        return np.ascontiguousarray(codon_ids.astype(np.uint8, copy=False))

    def _extract_features_and_codon_ids(
        self,
        nucleotide_sequence_per_codon,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility helper that reconstructs the former dense features."""
        codon_ids = self._extract_codon_ids(nucleotide_sequence_per_codon)
        encoded = np.ascontiguousarray(
            self.base_biological_feature_lut[codon_ids],
            dtype=np.float32,
        )
        return encoded, codon_ids

    # ============================================================
    # Collate
    # ============================================================

    def collate_fn(self, batch):
        from Utils.global_batch import empty_execution_metadata
        if len(batch) == 1 and empty_execution_metadata(batch[0]) is not None:
            return batch[0]
        has_bias_features = self.dataset_bias_extra_dim > 0
        expected_fields = 11 + int(has_bias_features)
        if any(len(sample) != expected_fields for sample in batch):
            raise ValueError(
                "Every sample must contain the mandatory replica tensor; expected "
                f"{expected_fields} fields per sample."
            )
        (
            idx_datasets,
            ids,
            sequences,
            codon_ids,
            profiles,
            css_s,
            sample_weights,
            dataset_quality_ranks,
            dataset_quality_weights,
            *optional_values,
        ) = zip(*batch)
        bias_features = optional_values[0] if has_bias_features else None
        replicas = optional_values[int(has_bias_features)]
        execution_metadata_values = optional_values[int(has_bias_features) + 1]
        if all(value is None for value in execution_metadata_values):
            execution_metadata = None
        elif any(value is None for value in execution_metadata_values):
            raise ValueError(
                "An execution microbatch cannot mix indexed and ordinary samples."
            )
        else:
            first_metadata = dict(execution_metadata_values[0])
            if any(dict(value) != first_metadata for value in execution_metadata_values[1:]):
                raise ValueError(
                    "All rows in one execution microbatch must carry identical metadata."
                )
            execution_metadata = first_metadata

        lengths = torch.as_tensor(
            [value.shape[0] for value in codon_ids],
            dtype=torch.long,
        )

        lengths_sorted, order = lengths.sort(descending=True)
        order_list = order.tolist()

        ids_datasets_sorted = torch.as_tensor(
            [idx_datasets[i] for i in order_list],
            dtype=torch.long,
        )

        ids_sorted = [ids[i] for i in order_list]

        biological_extra_sorted = [
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

        replicas_sorted = [
            torch.from_numpy(np.asarray(replicas[i], dtype=np.float32))
            for i in order_list
        ]

        css_sorted = [css_s[i] for i in order_list]
        sample_weights_sorted = torch.as_tensor(
            [sample_weights[i] for i in order_list],
            dtype=torch.float32,
        )
        dataset_quality_ranks_sorted = torch.as_tensor(
            [dataset_quality_ranks[i] for i in order_list],
            dtype=torch.float32,
        )
        dataset_quality_weights_sorted = torch.as_tensor(
            [dataset_quality_weights[i] for i in order_list],
            dtype=torch.float32,
        )
        transcript_group_indices_sorted = transcript_group_indices_from_ids(
            ids_sorted,
            expected_pair_rows_by_transcript=self.num_datasets_by_transcript,
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

        # The biological encoder is dataset-blind. Pack one canonical sequence
        # per transcript group, while retaining pair-row codon IDs for the
        # dataset-specific branch. Group IDs are contiguous and assigned by
        # first occurrence, so canonical order remains length-sorted.
        group_indices = tuple(transcript_group_indices_sorted.tolist())
        rows_by_transcript: list[list[int]] = []
        for pair_row, group in enumerate(group_indices):
            if group == len(rows_by_transcript):
                rows_by_transcript.append([])
            rows_by_transcript[group].append(pair_row)

        # Validate immutable sequence inputs here, before device transfer and
        # before canonical packing could hide a conflicting optional feature.
        # Equal codons imply equal lengths/masks and generated position inputs.
        # The base biological channels are derived from the shared codon LUT.
        invariant_fields = [("codon IDs", codon_ids_sorted)]
        if self.biological_extra_dim > 0:
            invariant_fields.append(
                ("biological sequence features", biological_extra_sorted)
            )
        if bias_features_sorted is not None:
            invariant_fields.append(
                ("dataset-bias optional sequence features", bias_features_sorted)
            )
        for rows in rows_by_transcript:
            for name, values in invariant_fields:
                reference = values[rows[0]]
                for row in rows[1:]:
                    candidate = values[row]
                    same = candidate.shape == reference.shape and (
                        torch.allclose(
                            reference, candidate, rtol=0.0, atol=0.0, equal_nan=True
                        )
                        if reference.is_floating_point()
                        else torch.equal(reference, candidate)
                    )
                    if not same:
                        raise ValueError(
                            "Dataset rows for one transcript have different "
                            f"{name} (rows {rows[0]} and {row})."
                        )

        transcript_metadata = ValidatedTranscriptMetadata(
            group_indices=group_indices,
            rows_by_transcript=tuple(tuple(rows) for rows in rows_by_transcript),
            lengths=tuple(lengths_sorted.tolist()),
        )
        canonical_pair_rows = transcript_metadata.canonical_rows

        canonical_index = torch.as_tensor(canonical_pair_rows, dtype=torch.long)
        unique_lengths = lengths_sorted.index_select(0, canonical_index)
        unique_codon_ids_pad = pad_sequence(
            [codon_ids_sorted[index] for index in canonical_pair_rows],
            batch_first=True,
            padding_value=0,
        )
        base_feature_lut = torch.from_numpy(self.base_biological_feature_lut)
        unique_sequence_pad = base_feature_lut[unique_codon_ids_pad]
        unique_mask = (
            torch.arange(unique_sequence_pad.shape[1]).unsqueeze(0)
            < unique_lengths.unsqueeze(1)
        )
        unique_sequence_pad = unique_sequence_pad.masked_fill(
            ~unique_mask.unsqueeze(-1),
            0.0,
        )
        if self.biological_extra_dim > 0:
            unique_biological_extra_pad = pad_sequence(
                [biological_extra_sorted[index] for index in canonical_pair_rows],
                batch_first=True,
                padding_value=0.0,
            )
            unique_sequence_pad = torch.cat(
                (unique_sequence_pad, unique_biological_extra_pad),
                dim=-1,
            )

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
            if n_rep == 0:
                raise ValueError("Every sample must contain at least one replica.")
            if rep_len != int(lengths_sorted[row_idx].item()):
                raise ValueError(
                    "Replica length mismatch after sorting: "
                    f"rep_len={rep_len}, length={int(lengths_sorted[row_idx].item())}."
                )
            replica_pad[row_idx, :n_rep, :rep_len] = rep
            replica_mask[row_idx, :n_rep] = True

        seq_packed = pack_padded_sequence(
            unique_sequence_pad,
            unique_lengths,
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
            dataset_quality_ranks_sorted,
            dataset_quality_weights_sorted,
            transcript_group_indices_sorted,
        )
        if bias_features_pad is not None:
            collated = (*collated, bias_features_pad)
        return (*collated, replica_pad, replica_mask, transcript_metadata, execution_metadata)
