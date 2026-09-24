from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import DataLoader

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import (
    RiboUnmixMultiDatasetDataModule,
    TranscriptGroupedMultiDatasetBatchSampler,
)
from Dataloaders.RiboUnmixBenchmarking.RiboUnmixBenchmarkingDataset import (
    RiboUnmixBenchmarkingDataset,
)


def _normalize_dataset_specs(
    dataset_specs: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for name, raw_spec in dataset_specs.items():
        spec = dict(raw_spec)
        missing = [key for key in ("ribo_path", "cds_path") if not spec.get(key)]
        if missing:
            raise KeyError(f"Benchmarking dataset {name!r} is missing {missing}.")
        normalized[str(name)] = {
            "ribo_path": str(spec["ribo_path"]),
            "cds_path": str(spec["cds_path"]),
        }
    return normalized


def available_ids_for_dataset(
    dataset_specs: Mapping[str, Mapping[str, str]],
    dataset_name: str,
) -> list[str]:
    """Read the profile/CDS ID intersections without loading array columns."""
    specs = _normalize_dataset_specs(dataset_specs)
    dataset_name = str(dataset_name)
    if dataset_name not in specs:
        raise KeyError(
            f"Unknown benchmarking dataset {dataset_name!r}. "
            f"Available datasets: {sorted(specs)}"
        )

    spec = specs[dataset_name]
    required_profile_columns = {
        "id",
        "ribo",
        "ribo_cds_replicas",
        "replica_ids",
        "weight",
    }
    profile_columns = set(pq.read_schema(spec["ribo_path"]).names)
    missing_profile_columns = required_profile_columns.difference(profile_columns)
    if missing_profile_columns:
        raise KeyError(
            f"Benchmark profile {spec['ribo_path']} is not preprocessed; missing "
            f"{sorted(missing_profile_columns)}. Run "
            "Datasets/benchmarking_data/weight_benchmarking_datasets.py first."
        )

    ribo_table = pd.read_parquet(spec["ribo_path"], columns=["id", "weight"])
    ribo_ids = ribo_table["id"].astype(str)
    weights = ribo_table["weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError(
            f"Benchmark profile {spec['ribo_path']} contains a non-finite or "
            "non-positive reliability weight."
        )
    cds_ids = pd.read_parquet(spec["cds_path"], columns=["id"])["id"].astype(str)

    if ribo_ids.duplicated().any():
        raise ValueError(f"Duplicate profile IDs in {spec['ribo_path']}.")
    if cds_ids.duplicated().any():
        raise ValueError(f"Duplicate CDS IDs in {spec['cds_path']}.")

    cds_id_set = set(cds_ids)
    common_ids = [tid for tid in ribo_ids if tid in cds_id_set]
    if not common_ids:
        raise RuntimeError(
            f"No matching profile/CDS IDs for benchmarking dataset {dataset_name}."
        )
    return common_ids


class RiboUnmixBenchmarkingDataModule(RiboUnmixMultiDatasetDataModule):
    """Lightning datamodule for the four organism benchmarking datasets.

    The selected dataset is defined by a preprocessed replica-aware profile
    parquet and its aligned CDS parquet (``id``, ``cds_seq``). The inherited
    sampling and dataloader implementation is reused after these files are
    normalized to the shared in-memory structure expected by RiboUnmix.
    """

    def __init__(
        self,
        *,
        dataset_specs: Mapping[str, Mapping[str, str]],
        dataset_name: str,
        batch_size: int,
        split: tuple[Sequence[str], Sequence[str]],
        predict_ids: Optional[Sequence[str]],
        nt_encoding_path: str,
        codon_to_aa_encoding_path: str,
        codon_encoding_path: str,
        aa_encoding_path: str,
        datasets_encoding_path: str,
        num_workers: int = 4,
        predict_num_workers: int = 0,
        seed: int = 42,
        train_sampling_strategy: Optional[str] = None,
        pin_memory: bool = True,
        prefetch_factor: Optional[int] = 4,
        multiprocessing_context: Optional[str] = "spawn",
        precompute_features: bool = False,
        additional_sequence_features: Optional[dict] = None,
    ):
        self.dataset_specs = _normalize_dataset_specs(dataset_specs)
        self.dataset_name = str(dataset_name)
        self.predict_ids = None if predict_ids is None else list(map(str, predict_ids))

        if self.dataset_name.lower() == "all":
            raise ValueError(
                "Benchmarking only supports one dataset per run; 'all' is not allowed."
            )
        if self.dataset_name not in self.dataset_specs:
            raise KeyError(
                f"Unknown benchmarking dataset {self.dataset_name!r}. "
                f"Available datasets: {sorted(self.dataset_specs)}"
            )

        active_features = [
            str(name)
            for name, raw_spec in dict(additional_sequence_features or {}).items()
            if str(dict(raw_spec or {}).get("route", "none")).lower() != "none"
        ]
        if active_features:
            raise ValueError(
                "Benchmarking CDS files do not contain additional per-codon feature "
                f"columns. Set their routes to 'none': {active_features}."
            )

        super().__init__(
            sequences_path="",
            datasets_paths=[self.dataset_specs[self.dataset_name]["ribo_path"]],
            batch_size=batch_size,
            split=split,
            nt_encoding_path=nt_encoding_path,
            codon_to_aa_encoding_path=codon_to_aa_encoding_path,
            codon_encoding_path=codon_encoding_path,
            aa_encoding_path=aa_encoding_path,
            datasets_encoding_path=datasets_encoding_path,
            num_workers=num_workers,
            predict_num_workers=predict_num_workers,
            seed=seed,
            train_sampling_strategy=train_sampling_strategy,
            minimum_positive_datasets_per_transcript=1,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
            additional_sequence_features=additional_sequence_features,
        )
        self.benchmarking_precompute_features = bool(precompute_features)

        max_codon_id = max(int(value) for value in self.c_enc.values())
        if max_codon_id > np.iinfo(np.uint8).max:
            raise ValueError("Benchmarking codon IDs do not fit in uint8 storage.")
        self._benchmark_codon_to_id = {
            str(codon).upper().replace("U", "T"): int(codon_id)
            for codon, codon_id in self.c_enc.items()
        }

        if self.dataset_name not in self.datasets_enc:
            raise KeyError(
                "Benchmarking dataset missing from dataset encoding: "
                f"{self.dataset_name}"
            )

        self.predict_dataset_obj = None
        self.predict_lengths = None
        self.predict_flat_dataset_ids = None
        self.predict_flat_transcript_ids = None

    @staticmethod
    def _required_columns(path: str, required: Sequence[str]) -> None:
        available = set(pq.read_schema(path).names)
        missing = [column for column in required if column not in available]
        if missing:
            raise KeyError(f"Missing columns {missing} in benchmarking parquet {path}.")

    def _make_dataset(
        self,
        *,
        shared_data: dict,
        transcript_ids: Sequence[str],
    ) -> RiboUnmixBenchmarkingDataset:
        return RiboUnmixBenchmarkingDataset(
            data=shared_data,
            lengths=shared_data["lengths"],
            nt_encoding=self.nt_enc,
            codon_to_aa_encoding=self.c2aa_enc,
            codon_encoding=self.c_enc,
            aa_encoding=self.aa_enc,
            datasets_encoding=self.datasets_enc,
            transcripts_ids=list(map(str, transcript_ids)),
            precompute_features=self.benchmarking_precompute_features,
            additional_sequence_features=self.additional_sequence_features,
        )

    def setup(self, stage=None):
        if self._has_loaded_data:
            print(f"Benchmarking data already in memory. Skipping load for stage: {stage}")
            return

        print(f"Loading benchmarking dataset: {self.dataset_name}")

        transcript_ids: list[str] = []
        ref_arrays: list[np.ndarray] = []
        css_arrays: list[np.ndarray] = []
        lengths: list[int] = []
        ribo_profiles: defaultdict[str, dict[str, np.ndarray]] = defaultdict(dict)
        ribo_replicas: defaultdict[str, dict[str, np.ndarray]] = defaultdict(dict)
        sample_weights: defaultdict[str, dict[str, float]] = defaultdict(dict)
        for dataset_name in [self.dataset_name]:
            spec = self.dataset_specs[dataset_name]
            self._required_columns(
                spec["ribo_path"],
                ("id", "ribo", "ribo_cds_replicas", "replica_ids", "weight"),
            )
            self._required_columns(spec["cds_path"], ("id", "cds_seq"))

            ribo_columns = [
                "id",
                "ribo",
                "ribo_cds_replicas",
                "replica_ids",
                "weight",
            ]
            ribo_df = pd.read_parquet(spec["ribo_path"], columns=ribo_columns)
            cds_df = pd.read_parquet(spec["cds_path"], columns=["id", "cds_seq"])
            ribo_df["id"] = ribo_df["id"].astype(str)
            cds_df["id"] = cds_df["id"].astype(str)

            if ribo_df["id"].duplicated().any():
                raise ValueError(f"Duplicate profile IDs in {spec['ribo_path']}.")
            if cds_df["id"].duplicated().any():
                raise ValueError(f"Duplicate CDS IDs in {spec['cds_path']}.")

            ribo_df = ribo_df.set_index("id")
            cds_df = cds_df.set_index("id")
            common_ids = [tid for tid in ribo_df.index if tid in cds_df.index]
            if not common_ids:
                raise RuntimeError(f"No profile/CDS overlap for {dataset_name}.")

            print(
                f"  {dataset_name}: {len(common_ids)} aligned transcripts "
                f"({len(ribo_df) - len(common_ids)} profiles without CDS)"
            )

            for tid in common_ids:
                cds_strings = np.char.replace(
                    np.char.upper(np.asarray(cds_df.at[tid, "cds_seq"]).astype(str)),
                    "U",
                    "T",
                )
                ribo = np.ascontiguousarray(
                    np.asarray(ribo_df.at[tid, "ribo"], dtype=np.float32)
                )
                if cds_strings.ndim != 1 or ribo.ndim != 1:
                    raise ValueError(
                        f"Expected one-dimensional CDS/ribo arrays for {tid}, got "
                        f"cds={cds_strings.shape}, ribo={ribo.shape}."
                    )
                if len(cds_strings) != len(ribo):
                    raise ValueError(
                        f"Length mismatch for {tid} in {dataset_name}: "
                        f"CDS={len(cds_strings)}, ribo={len(ribo)}."
                    )
                if len(cds_strings) == 0:
                    raise ValueError(f"Empty CDS/profile for {tid} in {dataset_name}.")
                if not np.isfinite(ribo).all() or np.any(ribo < 0):
                    raise ValueError(
                        f"Ribosome profile for {tid} contains non-finite or negative values."
                    )
                if float(ribo.sum(dtype=np.float64)) <= 0.0 or not bool(
                    np.any(ribo > 0.0)
                ):
                    raise ValueError(
                        f"Zero-information profile survived preprocessing for {tid} "
                        f"in {dataset_name}."
                    )

                raw_replicas = np.asarray(
                    ribo_df.at[tid, "ribo_cds_replicas"], dtype=object
                )
                if raw_replicas.ndim != 1 or len(raw_replicas) < 1:
                    raise ValueError(
                        f"Expected ribo_cds_replicas [R,L] for {tid}, got "
                        f"outer shape {raw_replicas.shape}."
                    )
                replicas = np.ascontiguousarray(
                    np.stack(
                        [np.asarray(replica, dtype=np.float32) for replica in raw_replicas],
                        axis=0,
                    ),
                    dtype=np.float32,
                )
                replica_ids = np.asarray(ribo_df.at[tid, "replica_ids"], dtype=str)
                if replicas.ndim != 2 or replicas.shape[0] < 1:
                    raise ValueError(
                        f"Expected ribo_cds_replicas [R,L] for {tid}, got "
                        f"{replicas.shape}."
                    )
                if replicas.shape[1] != len(ribo):
                    raise ValueError(
                        f"Replica/profile length mismatch for {tid} in {dataset_name}: "
                        f"replicas={replicas.shape}, consensus={len(ribo)}."
                    )
                if replica_ids.ndim != 1 or len(replica_ids) != replicas.shape[0]:
                    raise ValueError(
                        f"replica_ids do not match replicas for {tid} in "
                        f"{dataset_name}."
                    )
                if not np.isfinite(replicas).all() or np.any(replicas < 0.0):
                    raise ValueError(
                        f"Replicas for {tid} in {dataset_name} contain non-finite "
                        "or negative values."
                    )

                invalid_codons = sorted(
                    {
                        str(codon)
                        for codon in cds_strings
                        if str(codon) not in self._benchmark_codon_to_id
                    }
                )
                if invalid_codons:
                    raise ValueError(
                        f"CDS for {tid} contains invalid codons: {invalid_codons[:10]}"
                    )
                cds = np.fromiter(
                    (
                        self._benchmark_codon_to_id[str(codon)]
                        for codon in cds_strings
                    ),
                    dtype=np.uint8,
                    count=len(cds_strings),
                )

                transcript_ids.append(tid)
                ref_arrays.append(cds)
                css_arrays.append(np.empty(0, dtype=np.int64))
                lengths.append(len(cds_strings))
                ribo_profiles[tid][dataset_name] = ribo
                ribo_replicas[tid][dataset_name] = replicas
                sample_weight = float(ribo_df.at[tid, "weight"])
                if not np.isfinite(sample_weight) or sample_weight <= 0.0:
                    raise ValueError(
                        "Benchmark transcript weights must be finite and strictly "
                        f"positive; dataset={dataset_name}, transcript={tid}, "
                        f"weight={sample_weight}."
                    )
                sample_weights[tid][dataset_name] = sample_weight

        shared_data = {
            "transcript_id": np.asarray(transcript_ids, dtype=str),
            "ref": np.asarray(ref_arrays, dtype=object),
            "css": np.asarray(css_arrays, dtype=object),
            "ribo_profiles": ribo_profiles,
            "ribo_replicas": ribo_replicas,
            "sample_weights": sample_weights,
            "lengths": np.asarray(lengths, dtype=np.int32),
            "datasets_names": [self.dataset_name],
            "sequence_features": {},
        }

        all_id_set = set(transcript_ids)
        train_requested = set(map(str, self.split[0]))
        val_requested = set(map(str, self.split[1]))
        predict_requested = set(map(str, self.predict_ids or []))

        if train_requested & val_requested or train_requested & predict_requested or val_requested & predict_requested:
            raise ValueError("Benchmarking train/validation/prediction splits overlap.")

        train_ids = [tid for tid in transcript_ids if tid in train_requested]
        val_ids = [tid for tid in transcript_ids if tid in val_requested]
        predict_ids = [tid for tid in transcript_ids if tid in predict_requested]

        missing_split_ids = (
            train_requested | val_requested | predict_requested
        ) - all_id_set
        if missing_split_ids:
            raise KeyError(
                "Split contains IDs absent from selected benchmarking datasets: "
                f"{sorted(missing_split_ids)[:10]}"
            )
        if not train_ids:
            raise RuntimeError("Benchmarking training split is empty.")
        if not val_ids:
            raise RuntimeError("Benchmarking validation split is empty.")

        strategy = self._resolve_train_sampling_strategy()
        print(
            f"Benchmarking split sizes: train={len(train_ids)}, val={len(val_ids)}, "
            f"predict={len(predict_ids)}"
        )
        print(f"Training sampling: {strategy}; dataset pair mode: deterministic flat pairs")

        self.train_dataset_obj = self._make_dataset(
            shared_data=shared_data,
            transcript_ids=train_ids,
        )
        self.train_lengths = self._get_flat_lengths(self.train_dataset_obj)
        self.train_flat_dataset_ids = self._get_flat_dataset_ids(self.train_dataset_obj)
        self.train_flat_transcript_ids = self._get_flat_transcript_ids(
            self.train_dataset_obj
        )
        self._print_flat_pair_summary(
            dataset_obj=self.train_dataset_obj,
            flat_transcript_ids=self.train_flat_transcript_ids,
            flat_dataset_ids=self.train_flat_dataset_ids,
            considered_transcript_ids=train_ids,
            split_name="train",
        )

        self.val_dataset_obj = self._make_dataset(
            shared_data=shared_data,
            transcript_ids=val_ids,
        )
        self.val_lengths = self._get_flat_lengths(self.val_dataset_obj)
        self.val_flat_dataset_ids = self._get_flat_dataset_ids(self.val_dataset_obj)
        self.val_flat_transcript_ids = self._get_flat_transcript_ids(self.val_dataset_obj)
        self._print_flat_pair_summary(
            dataset_obj=self.val_dataset_obj,
            flat_transcript_ids=self.val_flat_transcript_ids,
            flat_dataset_ids=self.val_flat_dataset_ids,
            considered_transcript_ids=val_ids,
            split_name="validation",
        )

        if predict_ids:
            self.predict_dataset_obj = self._make_dataset(
                shared_data=shared_data,
                transcript_ids=predict_ids,
            )
            self.predict_lengths = self._get_flat_lengths(self.predict_dataset_obj)
            self.predict_flat_dataset_ids = self._get_flat_dataset_ids(
                self.predict_dataset_obj
            )
            self.predict_flat_transcript_ids = self._get_flat_transcript_ids(
                self.predict_dataset_obj
            )
            self._print_flat_pair_summary(
                dataset_obj=self.predict_dataset_obj,
                flat_transcript_ids=self.predict_flat_transcript_ids,
                flat_dataset_ids=self.predict_flat_dataset_ids,
                considered_transcript_ids=predict_ids,
                split_name="prediction",
            )

        self._has_loaded_data = True

    def predict_dataloader(self):
        if self.predict_dataset_obj is None:
            raise RuntimeError(
                "Prediction split is empty or setup() has not been called."
            )

        num_replicas, rank = self._dist_info()
        batch_sampler = TranscriptGroupedMultiDatasetBatchSampler(
            flat_transcript_ids=self.predict_flat_transcript_ids,
            flat_dataset_ids=self.predict_flat_dataset_ids,
            lengths=self.predict_lengths,
            batch_size=self.batch_size,
            seed=self.seed,
            drop_last=False,
            sort_by_length=True,
            require_multidataset=False,
            shuffle_batches=False,
            num_replicas=num_replicas,
            rank=rank,
        )
        return DataLoader(
            self.predict_dataset_obj,
            batch_sampler=batch_sampler,
            collate_fn=self.predict_dataset_obj.collate_fn,
            **self._dataloader_kwargs(),
        )
