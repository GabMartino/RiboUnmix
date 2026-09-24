#!/usr/bin/env python3
"""Strictly bounded-memory codon/position residual diagnostic.

The script fits two panel-specific weighted least-squares baselines to frozen
RiboUnmix shared profiles using only streaming sufficient statistics.  It does
not train RiboUnmix, alter gamma/pi, retain training profiles, or construct a
codon-level pandas table.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import shlex
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

# Set numerical thread limits before importing NumPy/PyTorch.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lightning.pytorch.utilities import move_data_to_device
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.publication_plot_style import latex_paper_style
from analyses.paths import artifact_directory

DEFAULT_RUN_ROOT = (
    PROJECT_ROOT / "results" / "my_panels_a100_b32_20260906_114323"
)
DEFAULT_OUTPUT_NAME = "posthoc_robustness_streaming"
SPLINE_KNOTS = np.asarray((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), dtype=np.float64)
BASELINES = ("position_only", "codon_plus_position")
REGIONS = ("full_cds", "interior_minus_20")
PROFILE_VARIANTS = (
    "original_profile",
    "position_only_residual",
    "codon_plus_position_residual",
)


class MemoryBudgetExceeded(RuntimeError):
    """Raised before the process reaches the requested RSS ceiling."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-rss-gb", type=float, default=4.0)
    parser.add_argument(
        "--rss-stop-fraction",
        type=float,
        default=0.90,
        help="Stop at this fraction of --max-rss-gb to retain headroom.",
    )
    parser.add_argument("--fit-max-transcripts", type=int, default=None)
    parser.add_argument("--test-max-transcripts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-padded-codon-tokens", type=int, default=20_000)
    parser.add_argument("--max-design-positions", type=int, default=25_000)
    parser.add_argument("--boundary-trim-codons", type=int, default=20)
    parser.add_argument("--mean-one-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--constant-variance-floor", type=float, default=1.0e-12)
    parser.add_argument(
        "--near-constant-variance-ratio", type=float, default=1.0e-6
    )
    parser.add_argument("--verification-transcripts", type=int, default=8)
    parser.add_argument("--minimum-replay-pcc", type=float, default=0.99)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260908)
    parser.add_argument("--scalar-write-batch-size", type=int, default=1_000)
    parser.add_argument(
        "--training-cache-dir",
        type=Path,
        default=None,
        help="Optional directory containing compatible compact training L_t caches.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.max_rss_gb) or args.max_rss_gb <= 0.0:
        raise ValueError("--max-rss-gb must be finite and positive.")
    if not 0.0 < args.rss_stop_fraction < 1.0:
        raise ValueError("--rss-stop-fraction must lie strictly between 0 and 1.")
    for name in (
        "fit_max_transcripts",
        "test_max_transcripts",
        "verification_transcripts",
        "bootstrap_replicates",
        "scalar_write_batch_size",
        "batch_size",
        "max_padded_codon_tokens",
        "max_design_positions",
    ):
        value = getattr(args, name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.batch_size > 4:
        raise ValueError("--batch-size may not exceed the bounded-memory default of 4.")
    if args.max_padded_codon_tokens > 20_000:
        raise ValueError("--max-padded-codon-tokens may not exceed 20,000.")
    if args.max_design_positions > 25_000:
        raise ValueError("--max-design-positions may not exceed 25,000.")
    if args.scalar_write_batch_size > 1_000:
        raise ValueError("--scalar-write-batch-size may not exceed 1,000.")
    if args.boundary_trim_codons < 0:
        raise ValueError("--boundary-trim-codons must be non-negative.")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _id_hash(ids: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(map(str, ids))).encode("utf-8")
    ).hexdigest()


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_ready(row.get(field)) for field in fields})


def _current_rss_gb() -> float:
    with Path("/proc/self/statm").open("r", encoding="ascii") as handle:
        resident_pages = int(handle.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / float(1024**3)


class MemoryLogger:
    FIELDS = (
        "elapsed_seconds",
        "phase",
        "panel",
        "index",
        "transcripts_processed",
        "codon_positions_processed",
        "rss_gb",
        "max_rss_gb",
        "stop_rss_gb",
        "device",
        "device_allocated_gb",
        "device_reserved_gb",
        "device_peak_allocated_gb",
        "event",
    )

    def __init__(self, path: Path, *, max_rss_gb: float, stop_fraction: float, device: torch.device):
        self.path = path
        self.max_rss_gb = float(max_rss_gb)
        self.stop_rss_gb = self.max_rss_gb * float(stop_fraction)
        self.device = device
        self.started = time.monotonic()
        write_header = not path.is_file() or path.stat().st_size == 0
        self._handle = path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDS)
        if write_header:
            self._writer.writeheader()
        self._handle.flush()

    def measure(self) -> dict[str, float]:
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / float(1024**3)
            reserved = torch.cuda.memory_reserved(self.device) / float(1024**3)
            peak = torch.cuda.max_memory_allocated(self.device) / float(1024**3)
        else:
            allocated = reserved = peak = 0.0
        return {
            "rss_gb": _current_rss_gb(),
            "device_allocated_gb": allocated,
            "device_reserved_gb": reserved,
            "device_peak_allocated_gb": peak,
        }

    def record(
        self,
        phase: str,
        *,
        panel: str = "",
        index: int = 0,
        transcripts_processed: int = 0,
        codon_positions_processed: int = 0,
        event: str = "periodic",
        enforce: bool = True,
    ) -> float:
        memory = self.measure()
        self._writer.writerow(
            {
                "elapsed_seconds": time.monotonic() - self.started,
                "phase": phase,
                "panel": panel,
                "index": int(index),
                "transcripts_processed": int(transcripts_processed),
                "codon_positions_processed": int(codon_positions_processed),
                "rss_gb": memory["rss_gb"],
                "max_rss_gb": self.max_rss_gb,
                "stop_rss_gb": self.stop_rss_gb,
                "device": str(self.device),
                "device_allocated_gb": memory["device_allocated_gb"],
                "device_reserved_gb": memory["device_reserved_gb"],
                "device_peak_allocated_gb": memory["device_peak_allocated_gb"],
                "event": event,
            }
        )
        self._handle.flush()
        if enforce and memory["rss_gb"] >= self.stop_rss_gb:
            raise MemoryBudgetExceeded(
                f"RSS {memory['rss_gb']:.3f} GiB reached the pre-exhaustion stop "
                f"threshold {self.stop_rss_gb:.3f} GiB "
                f"({self.max_rss_gb:.3f} GiB ceiling)."
            )
        return float(memory["rss_gb"])

    def guard(self, phase: str, *, panel: str, index: int) -> float:
        memory = self.measure()
        if memory["rss_gb"] >= self.stop_rss_gb:
            self.record(phase, panel=panel, index=index, event="pre_exhaustion_stop")
        return float(memory["rss_gb"])

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()


def _panel_names(run_root: Path) -> list[str]:
    panels = sorted(map(str, _read_json(run_root / "panel_manifest.json").get("panels", {})))
    if len(panels) != 4:
        raise ValueError(f"Expected exactly four panels, found {panels}.")
    return panels


def _load_folds(
    run_root: Path, panels: Sequence[str]
) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    common = _read_json(run_root / "common_split_manifest.json")
    common_test = sorted(map(str, common.get("common_test_ids", [])))
    if len(common_test) != 1_593:
        raise ValueError(f"Expected 1,593 common test IDs, found {len(common_test)}.")
    folds: dict[str, dict[str, list[str]]] = {}
    for panel in panels:
        manifest = _read_json(run_root / panel / "split_manifest.json")
        panel_folds = {
            fold: list(map(str, manifest.get(f"{fold}_ids", [])))
            for fold in ("train", "validation", "test")
        }
        if set(panel_folds["test"]) != set(common_test):
            raise ValueError(f"{panel}: test manifest differs from common test cohort.")
        for left, right in itertools.combinations(panel_folds, 2):
            overlap = set(panel_folds[left]) & set(panel_folds[right])
            if overlap:
                raise ValueError(f"{panel}: {left}/{right} folds overlap.")
        folds[panel] = panel_folds
    return folds, common_test


def _construct_fitting_cohort(
    folds: Mapping[str, Mapping[str, Sequence[str]]], panels: Sequence[str]
) -> tuple[list[str], dict[str, Any]]:
    train_sets = [set(map(str, folds[panel]["train"])) for panel in panels]
    validation = set().union(
        *(set(map(str, folds[panel]["validation"])) for panel in panels)
    )
    test = set().union(*(set(map(str, folds[panel]["test"])) for panel in panels))
    intersection = set.intersection(*train_sets)
    pre_exclusion_count = len(intersection)
    fitting = sorted(intersection - validation - test)
    if not fitting or set(fitting) & (validation | test):
        raise RuntimeError("Invalid four-panel fitting cohort after held-out exclusions.")
    return fitting, {
        "construction": (
            "intersection_of_four_training_manifests_minus_union_of_all_"
            "validation_and_test_transcripts"
        ),
        "pre_exclusion_intersection_count": pre_exclusion_count,
        "fitting_transcript_count": len(fitting),
        "fitting_transcript_id_hash": _id_hash(fitting),
        "validation_union_count": len(validation),
        "test_union_count": len(test),
        "overlap_with_validation": len(set(fitting) & validation),
        "overlap_with_test": len(set(fitting) & test),
        "panel_training_counts": {
            panel: len(folds[panel]["train"]) for panel in panels
        },
        "panel_training_id_hashes": {
            panel: _id_hash(folds[panel]["train"]) for panel in panels
        },
    }


def _locate_test_profile(path: Path) -> Path:
    required = {"transcript_id", "transcript_length", "L_t"}
    candidates = []
    for candidate in (path / "predictions").rglob("common_test_L_profiles.parquet"):
        if candidate.is_file() and required <= set(pq.read_schema(candidate).names):
            candidates.append(candidate.resolve())
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one compact test profile below {path}; found {candidates}.")
    return candidates[0]


def _local_sequence_path(cfg: DictConfig) -> Path:
    configured = Path(str(cfg.paths.sequences_path))
    candidates = (
        configured,
        PROJECT_ROOT / "Datasets" / "data" / "sequence" / configured.name,
        PROJECT_ROOT
        / "Datasets"
        / "data"
        / "sequence"
        / "MANE.selection.sequence_embeddings_with_css.parquet",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve sequence parquet from {configured}.")


def _repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_config(panel_dir: Path) -> DictConfig:
    path = panel_dir / "resolved_config.yaml"
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Resolved configuration is not a mapping: {path}")
    OmegaConf.set_struct(cfg, False)
    cfg.paths.sequences_path = str(_local_sequence_path(cfg))
    cfg.data.num_workers = 0
    cfg.data.predict_num_workers = 0
    cfg.data.pin_memory = False
    cfg.prediction.sequence_only_shared_profile = True
    OmegaConf.set_struct(cfg, True)
    return cfg


def _checkpoint_path(panel_dir: Path) -> Path:
    manifest_path = panel_dir / "scientific_checkpoint_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("checkpoint_variant") != "best_val_loss":
        raise ValueError(f"{manifest_path} does not select best_val_loss.")
    configured = Path(str(manifest.get("checkpoint_path", "")))
    matches = sorted((panel_dir / "checkpoints").rglob(configured.name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not uniquely resolve {configured.name!r} below {panel_dir / 'checkpoints'}."
        )
    return matches[0].resolve()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _create_sequence_index(
    *,
    sqlite_path: Path,
    manifest_path: Path,
    sequence_path: Path,
    fitting_ids: Sequence[str],
    test_ids: Sequence[str],
    memory: MemoryLogger,
) -> dict[str, Any]:
    expected = {
        "sequence_sha256": _sha256_file(sequence_path),
        "fitting_transcript_count": len(fitting_ids),
        "fitting_transcript_id_hash": _id_hash(fitting_ids),
        "test_transcript_count": len(test_ids),
        "test_transcript_id_hash": _id_hash(test_ids),
    }
    if sqlite_path.is_file() and manifest_path.is_file():
        existing = _read_json(manifest_path)
        if all(existing.get(key) == value for key, value in expected.items()):
            connection = sqlite3.connect(sqlite_path)
            try:
                count = int(connection.execute("SELECT COUNT(*) FROM sequences").fetchone()[0])
            finally:
                connection.close()
            if count == len(fitting_ids) + len(test_ids):
                existing["cache_status"] = "reused"
                return existing
        raise RuntimeError(
            f"Existing sequence index is incompatible: {sqlite_path}. "
            "Use a fresh --output-dir rather than silently replacing it."
        )

    requested_fit = set(map(str, fitting_ids))
    requested_test = set(map(str, test_ids))
    if requested_fit & requested_test:
        raise RuntimeError("Fitting and test sequence-index IDs overlap.")
    requested = requested_fit | requested_test
    seen: set[str] = set()
    content_digest = hashlib.sha256()
    connection = sqlite3.connect(sqlite_path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            "CREATE TABLE sequences ("
            "transcript_id TEXT PRIMARY KEY, split TEXT NOT NULL, "
            "length INTEGER NOT NULL, codons BLOB NOT NULL)"
        )
        parquet_file = pq.ParquetFile(sequence_path)
        try:
            for batch_index, batch in enumerate(
                parquet_file.iter_batches(
                    batch_size=128, columns=["transcript_id", "codons"]
                )
            ):
                columns = batch.to_pydict()
                inserts: list[tuple[str, str, int, bytes]] = []
                for raw_id, raw_codons in zip(
                    columns["transcript_id"], columns["codons"], strict=True
                ):
                    transcript_id = str(raw_id)
                    if transcript_id not in requested:
                        continue
                    codons = tuple(str(codon).upper() for codon in raw_codons)
                    if not codons or any(len(codon) != 3 for codon in codons):
                        raise ValueError(f"Invalid codon sequence for {transcript_id}.")
                    payload = "".join(codons).encode("ascii")
                    split = "fit" if transcript_id in requested_fit else "test"
                    inserts.append((transcript_id, split, len(codons), payload))
                    seen.add(transcript_id)
                    content_digest.update(transcript_id.encode("utf-8") + b"\0" + payload)
                if inserts:
                    connection.executemany(
                        "INSERT INTO sequences(transcript_id, split, length, codons) "
                        "VALUES (?, ?, ?, ?)",
                        inserts,
                    )
                    connection.commit()
                del columns, inserts, batch
                memory.guard("sequence_index", panel="", index=batch_index + 1)
                if (batch_index + 1) % 100 == 0:
                    memory.record(
                        "sequence_index",
                        index=batch_index + 1,
                        transcripts_processed=len(seen),
                    )
        finally:
            parquet_file.close()
    finally:
        connection.close()
    missing = sorted(requested - seen)
    if missing:
        raise KeyError(f"Sequence parquet is missing {len(missing)} selected IDs, e.g. {missing[:5]}.")
    manifest = {
        **expected,
        "cache_status": "generated",
        "sqlite_path": str(sqlite_path),
        "sqlite_sha256": _sha256_file(sqlite_path),
        "indexed_transcript_count": len(seen),
        "sequence_record_content_sha256": content_digest.hexdigest(),
        "storage": "disk-backed SQLite; codon sequences are not retained in process memory",
    }
    _write_json(manifest_path, manifest)
    return manifest


def _decode_codon_blob(blob: bytes, expected_length: int) -> tuple[str, ...]:
    text = bytes(blob).decode("ascii")
    if len(text) != 3 * expected_length:
        raise ValueError("Corrupt codon payload in disk-backed sequence index.")
    return tuple(text[index : index + 3] for index in range(0, len(text), 3))


def _lookup_sequence(
    connection: sqlite3.Connection, transcript_id: str
) -> tuple[str, ...]:
    row = connection.execute(
        "SELECT length, codons FROM sequences WHERE transcript_id = ?",
        (str(transcript_id),),
    ).fetchone()
    if row is None:
        raise KeyError(f"Sequence index has no transcript {transcript_id}.")
    return _decode_codon_blob(row[1], int(row[0]))


def _iter_bounded_sequence_batches(
    connection: sqlite3.Connection,
    transcript_ids: Sequence[str],
    *,
    batch_size: int,
    max_padded_tokens: int,
    memory: MemoryLogger,
    panel: str,
) -> Iterator[list[tuple[str, tuple[str, ...]]]]:
    pending: list[tuple[str, tuple[str, ...]]] = []
    maximum_length = 0
    effective_batch_size = int(batch_size)
    for transcript_index, transcript_id in enumerate(transcript_ids):
        rss = memory.guard("fit", panel=panel, index=transcript_index)
        if rss >= 0.75 * memory.max_rss_gb and effective_batch_size > 1:
            effective_batch_size = max(1, effective_batch_size // 2)
            memory.record(
                "fit",
                panel=panel,
                index=transcript_index,
                transcripts_processed=transcript_index,
                event=f"adaptive_batch_size_lowered_to_{effective_batch_size}",
            )
        codons = _lookup_sequence(connection, transcript_id)
        length = len(codons)
        if length > max_padded_tokens:
            raise MemoryBudgetExceeded(
                f"{transcript_id} has {length} codons, exceeding the hard "
                f"{max_padded_tokens:,}-padded-token limit."
            )
        candidate_maximum = max(maximum_length, length)
        candidate_count = len(pending) + 1
        candidate_tokens = candidate_count * candidate_maximum
        if pending and (
            candidate_count > effective_batch_size
            or candidate_tokens > max_padded_tokens
        ):
            yield pending
            pending = []
            maximum_length = 0
        pending.append((str(transcript_id), codons))
        maximum_length = max(maximum_length, length)
    if pending:
        yield pending


@dataclass
class SequenceBatchEncoder:
    nt_encoding: Mapping[str, Any]
    codon_to_aa_encoding: Mapping[str, Any]
    codon_encoding: Mapping[str, Any]
    aa_encoding: Mapping[str, Any]
    dataset_encoding: Mapping[str, Any]
    additional_sequence_features: Mapping[str, Any]
    dummy_dataset_name: str

    def collate(
        self, rows: Sequence[tuple[str, tuple[str, ...]]]
    ) -> tuple[Any, ...]:
        from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDataset import (
            RiboUnmixMultiDataset,
        )

        ids = [str(row[0]) for row in rows]
        codons = [tuple(row[1]) for row in rows]
        lengths = np.asarray([len(value) for value in codons], dtype=np.int32)
        references = np.empty(len(codons), dtype=object)
        references[:] = codons
        profiles = {
            transcript_id: {
                self.dummy_dataset_name: np.ones(length, dtype=np.float32)
            }
            for transcript_id, length in zip(ids, lengths, strict=True)
        }
        replicas = {
            transcript_id: {
                self.dummy_dataset_name: np.ones((1, length), dtype=np.float32)
            }
            for transcript_id, length in zip(ids, lengths, strict=True)
        }
        data = {
            "transcript_id": np.asarray(ids, dtype=str),
            "ref": references,
            "sequence_representation": "codon_tokens",
            "css": [np.zeros(length, dtype=np.int8) for length in lengths],
            "ribo_profiles": profiles,
            "ribo_replicas": replicas,
            "sample_weights": {
                transcript_id: {self.dummy_dataset_name: 1.0}
                for transcript_id in ids
            },
            "dataset_quality_ranks": {self.dummy_dataset_name: float("nan")},
            "dataset_quality_weights": {self.dummy_dataset_name: 1.0},
        }
        dataset = RiboUnmixMultiDataset(
            nt_encoding=dict(self.nt_encoding),
            codon_to_aa_encoding=dict(self.codon_to_aa_encoding),
            codon_encoding=dict(self.codon_encoding),
            aa_encoding=dict(self.aa_encoding),
            datasets_encoding=dict(self.dataset_encoding),
            transcripts_ids=ids,
            data=data,
            lengths=lengths,
            precompute_features=True,
            precompute_ribo=True,
            additional_sequence_features=dict(self.additional_sequence_features),
        )
        samples = [dataset[index] for index in range(len(dataset))]
        collated = dataset.collate_fn(samples)
        del samples, dataset, data, profiles, replicas, references
        return collated


def _make_frozen_model_and_encoder(
    cfg: DictConfig,
    *,
    checkpoint: Path,
    device: torch.device,
) -> tuple[Any, SequenceBatchEncoder, list[str]]:
    from main_ribounmix_multidataset import (
        get_datasets,
        load_weights_only,
        open_file,
        resolve_gamma_reference_panel,
    )
    from Models.RiboUnmixModel import RiboUnmixModel
    from Models.RiboUnmixLightningModule import RiboUnmixLightningModule

    datasets = list(map(str, get_datasets(cfg)))
    nt_encoding = open_file(_repo_path(cfg.paths.encodings.nt))
    codon_to_aa = open_file(_repo_path(cfg.paths.encodings.codon_to_aa))
    codon_encoding = open_file(_repo_path(cfg.paths.encodings.codon))
    aa_encoding = open_file(_repo_path(cfg.paths.encodings.aa))
    dataset_encoding = open_file(_repo_path(cfg.paths.encodings.datasets))
    gamma_panel = resolve_gamma_reference_panel(
        cfg=cfg,
        experiment_datasets=datasets,
        dataset_encoding=dataset_encoding,
    )
    torch_model = RiboUnmixModel(
        model_configs=cfg.model,
        eps=float(cfg.model.get("eps", 1.0e-8)),
        selected_dataset_names=gamma_panel["selected_names"],
        selected_dataset_ids=gamma_panel["selected_ids"],
        reference_dataset_names=gamma_panel["reference_names"],
        reference_dataset_ids=gamma_panel["reference_ids"],
        reference_dataset_quality_weights=gamma_panel["reference_quality"],
        nt_encoding=nt_encoding,
        codon_to_aa_encoding=codon_to_aa,
        codon_encoding=codon_encoding,
        aa_encoding=aa_encoding,
    )
    module = RiboUnmixLightningModule(
        torch_model, config=cfg, dataset_encoding=dataset_encoding
    )
    load_weights_only(module, checkpoint)
    module.to(device)
    module.eval()
    feature_cfg = OmegaConf.to_container(
        cfg.model.get("additional_sequence_features", {}), resolve=True
    )
    active_biological = {
        name: spec
        for name, spec in dict(feature_cfg or {}).items()
        if str(dict(spec or {}).get("route", "none")).lower()
        in {"biological", "both"}
    }
    if active_biological:
        raise RuntimeError(
            "The compact streaming encoder needs routed biological feature columns, "
            f"but this resolved configuration enables {sorted(active_biological)}."
        )
    encoder = SequenceBatchEncoder(
        nt_encoding=nt_encoding,
        codon_to_aa_encoding=codon_to_aa,
        codon_encoding=codon_encoding,
        aa_encoding=aa_encoding,
        dataset_encoding=dataset_encoding,
        additional_sequence_features=dict(feature_cfg or {}),
        dummy_dataset_name=datasets[0],
    )
    return module, encoder, datasets


def _autocast_context(cfg: DictConfig, device: torch.device):
    precision = str(getattr(cfg.trainer, "precision", "")).lower()
    if device.type == "cuda" and "bf16" in precision:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and "16" in precision:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _predict_shared_batch(
    *,
    module: Any,
    encoder: SequenceBatchEncoder,
    rows: Sequence[tuple[str, tuple[str, ...]]],
    cfg: DictConfig,
    device: torch.device,
    mean_one_tolerance: float,
) -> dict[str, np.ndarray]:
    batch = encoder.collate(rows)
    padded_tokens = int(batch[5].numel())
    moved = move_data_to_device(batch, device)
    with torch.inference_mode(), _autocast_context(cfg, device):
        shared = module.model._forward_biological_unique_transcripts(
            x_packed=moved[2],
            mask_b=moved[5].bool(),
            transcript_group_index=moved[11],
        )[0]["L_bio"]
    values = shared.detach().float().cpu().numpy()
    masks = moved[5].detach().bool().cpu().numpy()
    lengths = moved[4].detach().cpu().numpy()
    output: dict[str, np.ndarray] = {}
    for index, raw_id in enumerate(moved[1]):
        transcript_id = str(raw_id)
        profile = np.asarray(values[index][masks[index]], dtype=np.float64)
        if profile.size != int(lengths[index]) or not np.isfinite(profile).all():
            raise ValueError(f"Invalid frozen shared profile for {transcript_id}.")
        if np.any(profile <= 0.0):
            raise ValueError(f"Non-positive frozen shared profile for {transcript_id}.")
        if abs(float(profile.mean()) - 1.0) > mean_one_tolerance:
            raise ValueError(
                f"{transcript_id}: frozen L_t mean {profile.mean():.8g} is outside tolerance."
            )
        output[transcript_id] = profile
    del shared, values, masks, lengths, moved, batch
    output["__padded_tokens__"] = np.asarray((padded_tokens,), dtype=np.int64)
    return output


def _relative_position(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Transcript length must be positive.")
    if length == 1:
        return np.asarray((0.5,), dtype=np.float64)
    return np.arange(length, dtype=np.float64) / float(length - 1)


def _natural_spline_basis(u: np.ndarray) -> np.ndarray:
    values = np.asarray(u, dtype=np.float64).reshape(-1)
    terminal = float(SPLINE_KNOTS[-1])

    def d(index: int) -> np.ndarray:
        knot = float(SPLINE_KNOTS[index])
        return (
            np.maximum(values - knot, 0.0) ** 3
            - np.maximum(values - terminal, 0.0) ** 3
        ) / (terminal - knot)

    final_interior = d(len(SPLINE_KNOTS) - 2)
    nonlinear = [
        d(index) - final_interior for index in range(len(SPLINE_KNOTS) - 2)
    ]
    return np.column_stack((values, *nonlinear))


def _design_matrices(
    codons: Sequence[str], codon_order: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    length = len(codons)
    spline = _natural_spline_basis(_relative_position(length))
    position = np.empty((length, 6), dtype=np.float64)
    position[:, 0] = 1.0
    position[:, 1:] = spline
    full = np.zeros((length, 69), dtype=np.float64)
    full[:, 0] = 1.0
    reference = str(codon_order[0])
    contrast = {str(codon): index + 1 for index, codon in enumerate(codon_order[1:])}
    for row_index, raw_codon in enumerate(codons):
        codon = str(raw_codon)
        if codon != reference:
            try:
                full[row_index, contrast[codon]] = 1.0
            except KeyError as exc:
                raise KeyError(f"Unknown codon {codon!r} in diagnostic design.") from exc
    full[:, 64:] = spline
    return position, full


def _term_names(codon_order: Sequence[str], baseline: str) -> list[str]:
    spline = [
        "spline_linear_u",
        "spline_natural_d0_minus_d4",
        "spline_natural_d1_minus_d4",
        "spline_natural_d2_minus_d4",
        "spline_natural_d3_minus_d4",
    ]
    if baseline == "position_only":
        return ["intercept", *spline]
    reference = str(codon_order[0])
    return [
        "intercept",
        *(f"codon[{codon}]_vs_{reference}" for codon in codon_order[1:]),
        *spline,
    ]


@dataclass
class SufficientStatistics:
    dimension: int

    def __post_init__(self) -> None:
        self.A = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        self.b = np.zeros(self.dimension, dtype=np.float64)
        self.weighted_y2 = 0.0
        self.total_weight = 0.0
        self.transcripts = 0
        self.positions = 0
        self.blocks = 0

    def update(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        if X.shape[0] > 25_000:
            raise MemoryBudgetExceeded(
                f"Temporary design block has {X.shape[0]:,} positions (>25,000)."
            )
        weighted_X = weights[:, None] * X
        self.A += X.T @ weighted_X
        self.b += X.T @ (weights * y)
        self.weighted_y2 += float(np.dot(weights, y * y))
        self.total_weight += float(weights.sum())
        self.positions += int(y.size)
        self.blocks += 1
        del weighted_X


def _update_sufficient_statistics(
    *,
    rows: Sequence[tuple[str, tuple[str, ...]]],
    predictions: Mapping[str, np.ndarray],
    codon_order: Sequence[str],
    stats: Mapping[str, SufficientStatistics],
    max_design_positions: int,
) -> int:
    position_blocks: list[np.ndarray] = []
    full_blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    positions = 0
    for transcript_id, codons in rows:
        y = np.asarray(predictions[transcript_id], dtype=np.float64)
        if y.size != len(codons):
            raise ValueError(f"{transcript_id}: frozen L_t/codon length mismatch.")
        position, full = _design_matrices(codons, codon_order)
        position_blocks.append(position)
        full_blocks.append(full)
        targets.append(y)
        weights.append(np.full(y.size, 1.0 / float(y.size), dtype=np.float64))
        positions += int(y.size)
    if positions > max_design_positions:
        raise MemoryBudgetExceeded(
            f"Temporary design block has {positions:,} positions, exceeding "
            f"--max-design-positions={max_design_positions:,}."
        )
    X_position = np.concatenate(position_blocks, axis=0)
    X_full = np.concatenate(full_blocks, axis=0)
    y_block = np.concatenate(targets)
    w_block = np.concatenate(weights)
    stats["position_only"].update(X_position, y_block, w_block)
    stats["codon_plus_position"].update(X_full, y_block, w_block)
    for accumulator in stats.values():
        accumulator.transcripts += len(rows)
    del X_position, X_full, y_block, w_block
    del position_blocks, full_blocks, targets, weights
    return positions


def _solve_statistics(
    accumulator: SufficientStatistics, *, rcond: float = 1.0e-12
) -> tuple[np.ndarray, dict[str, Any]]:
    singular_values = np.linalg.svd(accumulator.A, compute_uv=False)
    tolerance = rcond * singular_values[0]
    effective_rank = int(np.count_nonzero(singular_values > tolerance))
    coefficient, _, lstsq_rank, _ = np.linalg.lstsq(
        accumulator.A, accumulator.b, rcond=rcond
    )
    weighted_sse = float(
        accumulator.weighted_y2
        - 2.0 * coefficient @ accumulator.b
        + coefficient @ accumulator.A @ coefficient
    )
    diagnostics = {
        "dimension": accumulator.dimension,
        "effective_rank": effective_rank,
        "lstsq_rank": int(lstsq_rank),
        "condition_number_A": float(np.linalg.cond(accumulator.A)),
        "largest_singular_value_A": float(singular_values[0]),
        "smallest_singular_value_A": float(singular_values[-1]),
        "rcond": rcond,
        "transcript_count": accumulator.transcripts,
        "position_count": accumulator.positions,
        "design_block_count": accumulator.blocks,
        "total_wls_weight": accumulator.total_weight,
        "weighted_y2": accumulator.weighted_y2,
        "weighted_sse": max(weighted_sse, 0.0),
        "weighted_residual_mean_square": max(weighted_sse, 0.0)
        / max(accumulator.total_weight, np.finfo(float).eps),
        "A_sha256": _sha256_array(accumulator.A),
        "b_sha256": _sha256_array(accumulator.b),
    }
    return coefficient, diagnostics


def _fit_progress_paths(output_dir: Path, panel: str) -> tuple[Path, Path]:
    directory = output_dir / "fit_progress"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{panel}.npz", directory / f"{panel}.json"


def _save_fit_progress(
    *,
    output_dir: Path,
    panel: str,
    fitting_ids: Sequence[str],
    checkpoint_sha256: str,
    stats: Mapping[str, SufficientStatistics],
    batch_count: int,
) -> None:
    processed = int(stats["position_only"].transcripts)
    if stats["codon_plus_position"].transcripts != processed:
        raise RuntimeError("Baseline sufficient-statistic progress diverged.")
    stats_path, manifest_path = _fit_progress_paths(output_dir, panel)
    temporary_path = stats_path.with_name(stats_path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary_path,
        A_position_only=stats["position_only"].A,
        b_position_only=stats["position_only"].b,
        A_codon_plus_position=stats["codon_plus_position"].A,
        b_codon_plus_position=stats["codon_plus_position"].b,
        weighted_y2=np.asarray(
            [stats[name].weighted_y2 for name in BASELINES], dtype=np.float64
        ),
        total_weight=np.asarray(
            [stats[name].total_weight for name in BASELINES], dtype=np.float64
        ),
        positions=np.asarray(
            [stats[name].positions for name in BASELINES], dtype=np.int64
        ),
        blocks=np.asarray([stats[name].blocks for name in BASELINES], dtype=np.int64),
    )
    os.replace(temporary_path, stats_path)
    _write_json(
        manifest_path,
        {
            "panel": panel,
            "selected_fitting_transcript_count": len(fitting_ids),
            "selected_fitting_transcript_id_hash": _id_hash(fitting_ids),
            "processed_transcript_count": processed,
            "processed_prefix_id_hash": _id_hash(fitting_ids[:processed]),
            "checkpoint_sha256": checkpoint_sha256,
            "batch_count": int(batch_count),
            "position_count": int(stats["position_only"].positions),
            "statistics_path": str(stats_path),
            "statistics_sha256": _sha256_file(stats_path),
        },
    )


def _load_fit_progress(
    *,
    output_dir: Path,
    panel: str,
    fitting_ids: Sequence[str],
    checkpoint_sha256: str,
) -> tuple[dict[str, SufficientStatistics], int]:
    empty = {
        "position_only": SufficientStatistics(6),
        "codon_plus_position": SufficientStatistics(69),
    }
    stats_path, manifest_path = _fit_progress_paths(output_dir, panel)
    if not stats_path.is_file() or not manifest_path.is_file():
        return empty, 0
    manifest = _read_json(manifest_path)
    processed = int(manifest.get("processed_transcript_count", -1))
    compatible = (
        manifest.get("selected_fitting_transcript_count") == len(fitting_ids)
        and manifest.get("selected_fitting_transcript_id_hash") == _id_hash(fitting_ids)
        and manifest.get("checkpoint_sha256") == checkpoint_sha256
        and 0 <= processed <= len(fitting_ids)
        and manifest.get("processed_prefix_id_hash") == _id_hash(fitting_ids[:processed])
        and manifest.get("statistics_sha256") == _sha256_file(stats_path)
    )
    if not compatible:
        raise RuntimeError(
            f"Incompatible fit-progress checkpoint for {panel}: {manifest_path}. "
            "Use a fresh --output-dir rather than mixing fitting cohorts or checkpoints."
        )
    with np.load(stats_path) as saved:
        for index, baseline in enumerate(BASELINES):
            accumulator = empty[baseline]
            suffix = baseline
            accumulator.A[...] = saved[f"A_{suffix}"]
            accumulator.b[...] = saved[f"b_{suffix}"]
            accumulator.weighted_y2 = float(saved["weighted_y2"][index])
            accumulator.total_weight = float(saved["total_weight"][index])
            accumulator.positions = int(saved["positions"][index])
            accumulator.blocks = int(saved["blocks"][index])
            accumulator.transcripts = processed
    return empty, int(manifest.get("batch_count", 0))


def _find_compatible_training_cache(
    *,
    run_root: Path,
    explicit_dir: Path | None,
    panel: str,
    full_fitting_hash: str,
    checkpoint_sha256: str,
) -> tuple[Path | None, dict[str, Any]]:
    candidates: list[Path] = []
    if explicit_dir is not None:
        candidates.append(
            explicit_dir / f"{panel}_fitting_cohort_L_profiles.parquet"
        )
    candidates.extend(
        sorted(
            artifact_directory("real_data", run_root).glob(
                f"**/{panel}_fitting_cohort_L_profiles.parquet"
            )
        )
    )
    rejected: list[dict[str, str]] = []
    for cache_path in dict.fromkeys(path.resolve() for path in candidates if path.is_file()):
        provenance_path = cache_path.with_name(
            f"{panel}_fitting_cohort_inference_provenance.json"
        )
        if not provenance_path.is_file():
            rejected.append(
                {"path": str(cache_path), "reason": "missing inference provenance"}
            )
            continue
        provenance = _read_json(provenance_path)
        if str(provenance.get("cache_transcript_id_hash")) != full_fitting_hash:
            rejected.append(
                {"path": str(cache_path), "reason": "fitting cohort hash mismatch"}
            )
            continue
        if str(provenance.get("checkpoint_sha256")) != checkpoint_sha256:
            rejected.append(
                {"path": str(cache_path), "reason": "checkpoint hash mismatch"}
            )
            continue
        return cache_path, {
            "status": "compatible_cache_selected",
            "path": str(cache_path),
            "sha256": _sha256_file(cache_path),
            "provenance_path": str(provenance_path),
            "rejected_candidates": rejected,
        }
    return None, {"status": "no_compatible_cache", "rejected_candidates": rejected}


def _iter_profile_cache_batches(
    *,
    cache_path: Path,
    selected_ids: Sequence[str],
    sequence_connection: sqlite3.Connection,
    batch_size: int,
    max_padded_tokens: int,
    mean_one_tolerance: float,
) -> Iterator[
    tuple[list[tuple[str, tuple[str, ...]]], dict[str, np.ndarray]]
]:
    selected = set(map(str, selected_ids))
    observed: set[str] = set()
    pending_rows: list[tuple[str, tuple[str, ...]]] = []
    pending_predictions: dict[str, np.ndarray] = {}
    pending_maximum = 0
    parquet_file = pq.ParquetFile(cache_path)
    try:
        for batch in parquet_file.iter_batches(
            batch_size=max(1, batch_size),
            columns=["transcript_id", "transcript_length", "L_t"],
        ):
            columns = batch.to_pydict()
            for raw_id, raw_length, raw_profile in zip(
                columns["transcript_id"],
                columns["transcript_length"],
                columns["L_t"],
                strict=True,
            ):
                transcript_id = str(raw_id)
                if transcript_id not in selected:
                    continue
                if transcript_id in observed:
                    raise ValueError(f"Duplicate cached training profile {transcript_id}.")
                profile = np.asarray(raw_profile, dtype=np.float64).reshape(-1)
                codons = _lookup_sequence(sequence_connection, transcript_id)
                if profile.size != int(raw_length) or profile.size != len(codons):
                    raise ValueError(f"Cached profile length mismatch for {transcript_id}.")
                if not np.isfinite(profile).all() or np.any(profile <= 0.0):
                    raise ValueError(f"Invalid cached profile for {transcript_id}.")
                if abs(float(profile.mean()) - 1.0) > mean_one_tolerance:
                    raise ValueError(f"Cached profile mean-one check failed for {transcript_id}.")
                candidate_maximum = max(pending_maximum, profile.size)
                candidate_count = len(pending_rows) + 1
                if pending_rows and (
                    candidate_count > batch_size
                    or candidate_count * candidate_maximum > max_padded_tokens
                ):
                    yield pending_rows, pending_predictions
                    pending_rows = []
                    pending_predictions = {}
                    pending_maximum = 0
                pending_rows.append((transcript_id, codons))
                pending_predictions[transcript_id] = profile
                pending_maximum = max(pending_maximum, profile.size)
                observed.add(transcript_id)
            del columns, batch
        if pending_rows:
            yield pending_rows, pending_predictions
    finally:
        parquet_file.close()
    if observed != selected:
        missing = sorted(selected - observed)
        raise KeyError(
            f"Training cache lacks {len(missing)} selected fitting IDs, e.g. {missing[:5]}."
        )


def _read_selected_test_profiles(
    path: Path, selected_ids: set[str]
) -> dict[str, np.ndarray]:
    profiles: dict[str, np.ndarray] = {}
    parquet_file = pq.ParquetFile(path)
    try:
        for batch in parquet_file.iter_batches(
            batch_size=1, columns=["transcript_id", "transcript_length", "L_t"]
        ):
            row = batch.to_pydict()
            transcript_id = str(row["transcript_id"][0])
            if transcript_id in selected_ids:
                profile = np.asarray(row["L_t"][0], dtype=np.float64).reshape(-1)
                if profile.size != int(row["transcript_length"][0]):
                    raise ValueError(f"Stored test profile length mismatch for {transcript_id}.")
                profiles[transcript_id] = profile
                if len(profiles) == len(selected_ids):
                    break
            del row, batch
    finally:
        parquet_file.close()
    if set(profiles) != selected_ids:
        raise KeyError(f"Compact test profile {path} lacks verification transcripts.")
    return profiles


def _pearson(left: np.ndarray, right: np.ndarray, variance_floor: float) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or y.size != x.size:
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_ss = float(np.dot(x_centered, x_centered))
    y_ss = float(np.dot(y_centered, y_centered))
    if x_ss / x.size <= variance_floor or y_ss / y.size <= variance_floor:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / math.sqrt(x_ss * y_ss))


def _verify_frozen_replay(
    *,
    panel: str,
    module: Any,
    encoder: SequenceBatchEncoder,
    cfg: DictConfig,
    device: torch.device,
    test_profile_path: Path,
    verification_ids: Sequence[str],
    sequence_connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    requested = set(map(str, verification_ids))
    stored = _read_selected_test_profiles(test_profile_path, requested)
    generated: dict[str, np.ndarray] = {}
    for rows in _iter_bounded_sequence_batches(
        sequence_connection,
        list(verification_ids),
        batch_size=args.batch_size,
        max_padded_tokens=args.max_padded_codon_tokens,
        memory=_NoopFitMemoryAdapter(),
        panel=panel,
    ):
        batch_profiles = _predict_shared_batch(
            module=module,
            encoder=encoder,
            rows=rows,
            cfg=cfg,
            device=device,
            mean_one_tolerance=args.mean_one_tolerance,
        )
        batch_profiles.pop("__padded_tokens__")
        generated.update(batch_profiles)
    rows_out: list[dict[str, Any]] = []
    for transcript_id in verification_ids:
        replay = generated[transcript_id]
        reference = stored[transcript_id]
        if replay.shape != reference.shape:
            raise ValueError(f"{panel}/{transcript_id}: replay/stored shape mismatch.")
        pcc = _pearson(replay, reference, args.constant_variance_floor)
        rows_out.append(
            {
                "panel": panel,
                "transcript_id": transcript_id,
                "transcript_length": replay.size,
                "replay_vs_stored_pcc": pcc,
                "replay_vs_stored_rmse": float(
                    np.sqrt(np.mean((replay - reference) ** 2))
                ),
                "replay_vs_stored_max_abs_difference": float(
                    np.max(np.abs(replay - reference))
                ),
            }
        )
    finite = [row["replay_vs_stored_pcc"] for row in rows_out if math.isfinite(row["replay_vs_stored_pcc"])]
    if not finite or min(finite) < args.minimum_replay_pcc:
        raise RuntimeError(
            f"{panel}: frozen streaming replay verification failed; "
            f"minimum PCC={min(finite) if finite else float('nan'):.6g}, "
            f"required={args.minimum_replay_pcc}."
        )
    del generated, stored
    return rows_out


class _NoopFitMemoryAdapter:
    """Adapter for tiny held-out verification batches; caller guards the panel."""

    max_rss_gb = float("inf")

    @staticmethod
    def guard(phase: str, *, panel: str, index: int) -> float:
        del phase, panel, index
        return 0.0

    @staticmethod
    def record(*args: Any, **kwargs: Any) -> float:
        del args, kwargs
        return 0.0


def _fit_panel(
    *,
    panel: str,
    panel_dir: Path,
    run_root: Path,
    fitting_ids: Sequence[str],
    full_fitting_hash: str,
    test_profile_path: Path,
    verification_ids: Sequence[str],
    sequence_connection: sqlite3.Connection,
    codon_order: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
    memory: MemoryLogger,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    checkpoint = _checkpoint_path(panel_dir)
    checkpoint_hash = _sha256_file(checkpoint)
    cache_path, cache_audit = _find_compatible_training_cache(
        run_root=run_root,
        explicit_dir=(
            args.training_cache_dir.expanduser().resolve()
            if args.training_cache_dir is not None
            else None
        ),
        panel=panel,
        full_fitting_hash=full_fitting_hash,
        checkpoint_sha256=checkpoint_hash,
    )
    stats, batches = _load_fit_progress(
        output_dir=output_dir,
        panel=panel,
        fitting_ids=fitting_ids,
        checkpoint_sha256=checkpoint_hash,
    )
    processed_before_resume = int(stats["position_only"].transcripts)
    remaining_fitting_ids = fitting_ids[processed_before_resume:]
    positions = int(stats["position_only"].positions)
    verification_rows: list[dict[str, Any]] = []
    module = encoder = cfg = None
    started = time.monotonic()
    memory.record("fit_panel_start", panel=panel, event="start")
    if cache_path is not None:
        for batch_rows, predictions in _iter_profile_cache_batches(
            cache_path=cache_path,
            selected_ids=remaining_fitting_ids,
            sequence_connection=sequence_connection,
            batch_size=args.batch_size,
            max_padded_tokens=args.max_padded_codon_tokens,
            mean_one_tolerance=args.mean_one_tolerance,
        ):
            positions += _update_sufficient_statistics(
                rows=batch_rows,
                predictions=predictions,
                codon_order=codon_order,
                stats=stats,
                max_design_positions=args.max_design_positions,
            )
            batches += 1
            del batch_rows, predictions
            memory.guard("fit", panel=panel, index=batches)
            if batches % 100 == 0:
                _save_fit_progress(
                    output_dir=output_dir,
                    panel=panel,
                    fitting_ids=fitting_ids,
                    checkpoint_sha256=checkpoint_hash,
                    stats=stats,
                    batch_count=batches,
                )
                memory.record(
                    "fit",
                    panel=panel,
                    index=batches,
                    transcripts_processed=stats["position_only"].transcripts,
                    codon_positions_processed=positions,
                )
        source = "compatible_compact_training_cache_stream"
    else:
        cfg = _load_config(panel_dir)
        module, encoder, datasets = _make_frozen_model_and_encoder(
            cfg, checkpoint=checkpoint, device=device
        )
        memory.record(
            "fit_checkpoint_loaded", panel=panel, event="checkpoint_loaded"
        )
        verification_rows = _verify_frozen_replay(
            panel=panel,
            module=module,
            encoder=encoder,
            cfg=cfg,
            device=device,
            test_profile_path=test_profile_path,
            verification_ids=verification_ids,
            sequence_connection=sequence_connection,
            args=args,
        )
        for batch_rows in _iter_bounded_sequence_batches(
            sequence_connection,
            remaining_fitting_ids,
            batch_size=args.batch_size,
            max_padded_tokens=args.max_padded_codon_tokens,
            memory=memory,
            panel=panel,
        ):
            predictions = _predict_shared_batch(
                module=module,
                encoder=encoder,
                rows=batch_rows,
                cfg=cfg,
                device=device,
                mean_one_tolerance=args.mean_one_tolerance,
            )
            padded_tokens = int(predictions.pop("__padded_tokens__")[0])
            if padded_tokens > args.max_padded_codon_tokens:
                raise MemoryBudgetExceeded(
                    f"{panel}: batch has {padded_tokens:,} padded codon tokens."
                )
            positions += _update_sufficient_statistics(
                rows=batch_rows,
                predictions=predictions,
                codon_order=codon_order,
                stats=stats,
                max_design_positions=args.max_design_positions,
            )
            batches += 1
            del predictions, batch_rows
            memory.guard("fit", panel=panel, index=batches)
            if batches % 100 == 0:
                _save_fit_progress(
                    output_dir=output_dir,
                    panel=panel,
                    fitting_ids=fitting_ids,
                    checkpoint_sha256=checkpoint_hash,
                    stats=stats,
                    batch_count=batches,
                )
                memory.record(
                    "fit",
                    panel=panel,
                    index=batches,
                    transcripts_processed=stats["position_only"].transcripts,
                    codon_positions_processed=positions,
                )
        source = "frozen_best_checkpoint_streaming_sequence_inference"

    if stats["position_only"].transcripts != len(fitting_ids):
        raise RuntimeError(
            f"{panel}: sufficient statistics cover "
            f"{stats['position_only'].transcripts:,}/{len(fitting_ids):,} fitting transcripts."
        )
    _save_fit_progress(
        output_dir=output_dir,
        panel=panel,
        fitting_ids=fitting_ids,
        checkpoint_sha256=checkpoint_hash,
        stats=stats,
        batch_count=batches,
    )

    coefficients: dict[str, np.ndarray] = {}
    fit_diagnostics: dict[str, Any] = {}
    coefficient_rows: list[dict[str, Any]] = []
    stats_path = output_dir / "sufficient_statistics" / f"{panel}.npz"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stats_path,
        A_position_only=stats["position_only"].A,
        b_position_only=stats["position_only"].b,
        A_codon_plus_position=stats["codon_plus_position"].A,
        b_codon_plus_position=stats["codon_plus_position"].b,
        weighted_y2=np.asarray(
            (
                stats["position_only"].weighted_y2,
                stats["codon_plus_position"].weighted_y2,
            ),
            dtype=np.float64,
        ),
    )
    for baseline in BASELINES:
        coefficient, diagnostics = _solve_statistics(stats[baseline])
        if diagnostics["effective_rank"] != diagnostics["dimension"]:
            raise np.linalg.LinAlgError(
                f"{panel}/{baseline}: design rank "
                f"{diagnostics['effective_rank']}/{diagnostics['dimension']}."
            )
        coefficients[baseline] = coefficient
        fit_diagnostics[baseline] = diagnostics
        for term_index, (term, value) in enumerate(
            zip(_term_names(codon_order, baseline), coefficient, strict=True)
        ):
            coefficient_rows.append(
                {
                    "panel": panel,
                    "baseline": baseline,
                    "term_index": term_index,
                    "term": term,
                    "coefficient": float(value),
                }
            )
    _write_csv(
        output_dir / "coefficients" / f"{panel}.csv",
        coefficient_rows,
        ("panel", "baseline", "term_index", "term", "coefficient"),
    )
    provenance = {
        "panel": panel,
        "source": source,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "resolved_config_path": str(panel_dir / "resolved_config.yaml"),
        "resolved_config_sha256": _sha256_file(panel_dir / "resolved_config.yaml"),
        "cache_audit": cache_audit,
        "evaluation_mode": True,
        "torch_inference_mode": True,
        "neural_parameter_updates": False,
        "gamma_or_pi_modified": False,
        "training_profiles_saved_or_retained": False,
        "fitting_transcript_count": stats["position_only"].transcripts,
        "fitting_transcript_id_hash": _id_hash(fitting_ids),
        "fitting_position_count": positions,
        "streaming_batch_count": batches,
        "resumed_from_processed_transcript_count": processed_before_resume,
        "configured_max_batch_size": args.batch_size,
        "maximum_padded_codon_tokens": args.max_padded_codon_tokens,
        "maximum_design_positions": args.max_design_positions,
        "sufficient_statistics_path": str(stats_path),
        "sufficient_statistics_file_sha256": _sha256_file(stats_path),
        "fit_diagnostics": fit_diagnostics,
        "stored_test_replay_verification": verification_rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    if cfg is not None:
        provenance.update(
            {
                "device": str(device),
                "resolved_precision": str(getattr(cfg.trainer, "precision", None)),
                "configured_dataset_count": len(datasets),
                "sequence_input_convention": (
                    "RiboUnmixMultiDataset codon_tokens + production "
                    "collate_fn; exact shared biological branch"
                ),
            }
        )
    _write_json(output_dir / "fit_provenance" / f"{panel}.json", provenance)

    # Release the only live checkpoint before the caller opens the next one.
    del module, encoder, cfg, stats
    gc.collect()
    pa.default_memory_pool().release_unused()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    memory.record(
        "fit_checkpoint_unloaded", panel=panel, event="checkpoint_unloaded"
    )
    return coefficients, provenance, verification_rows


class BufferedParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema, max_rows: int):
        self.path = path
        self.incomplete_path = path.with_suffix(path.suffix + ".incomplete")
        self.schema = schema
        self.max_rows = int(max_rows)
        self.buffer: list[dict[str, Any]] = []
        self.writer = pq.ParquetWriter(
            self.incomplete_path,
            schema=schema,
            compression="zstd",
        )
        self.closed = False

    def append(self, row: Mapping[str, Any]) -> None:
        self.buffer.append(dict(row))
        if len(self.buffer) >= self.max_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        if len(self.buffer) > self.max_rows:
            raise MemoryBudgetExceeded("Scalar Parquet buffer exceeded its hard row limit.")
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        self.writer.write_table(table, row_group_size=self.max_rows)
        self.buffer.clear()
        del table

    def close(self, *, complete: bool) -> None:
        if self.closed:
            return
        if complete:
            self.flush()
        else:
            self.buffer.clear()
        self.writer.close()
        self.closed = True
        if complete:
            os.replace(self.incomplete_path, self.path)


PCC_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.string()),
        ("transcript_length", pa.int32()),
        ("region", pa.string()),
        ("panel_a", pa.string()),
        ("panel_b", pa.string()),
        ("panel_pair", pa.string()),
        ("profile_variant", pa.string()),
        ("n_positions", pa.int32()),
        ("pcc", pa.float64()),
        ("valid", pa.bool_()),
        ("reason_code", pa.string()),
    ]
)


VARIANCE_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.string()),
        ("transcript_length", pa.int32()),
        ("panel", pa.string()),
        ("baseline", pa.string()),
        ("region", pa.string()),
        ("n_positions", pa.int32()),
        ("original_variance", pa.float64()),
        ("baseline_variance", pa.float64()),
        ("residual_variance", pa.float64()),
        ("residual_to_original_variance_ratio", pa.float64()),
        ("constant_residual", pa.bool_()),
        ("nearly_constant_residual", pa.bool_()),
        ("valid", pa.bool_()),
        ("reason_code", pa.string()),
    ]
)


def _region_slice(length: int, region: str, trim: int) -> slice:
    if region == "full_cds":
        return slice(0, length)
    if region != "interior_minus_20":
        raise ValueError(f"Unknown region {region}.")
    return slice(trim, length - trim) if length > 2 * trim else slice(0, 0)


def _variance_diagnostic(
    original: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
    *,
    floor: float,
    near_ratio: float,
) -> dict[str, Any]:
    if original.size < 2:
        return {
            "original_variance": float("nan"),
            "baseline_variance": float("nan"),
            "residual_variance": float("nan"),
            "residual_to_original_variance_ratio": float("nan"),
            "constant_residual": False,
            "nearly_constant_residual": False,
            "valid": False,
            "reason_code": "too_short_for_region",
        }
    original_variance = float(np.var(original))
    baseline_variance = float(np.var(baseline))
    residual_variance = float(np.var(residual))
    ratio = (
        residual_variance / original_variance
        if original_variance > floor
        else float("nan")
    )
    constant = residual_variance <= floor
    nearly_constant = (
        not constant and math.isfinite(ratio) and ratio <= near_ratio
    )
    if original_variance <= floor:
        reason = "constant_original"
    elif constant:
        reason = "constant_residual"
    elif nearly_constant:
        reason = "nearly_constant_residual"
    else:
        reason = "valid"
    return {
        "original_variance": original_variance,
        "baseline_variance": baseline_variance,
        "residual_variance": residual_variance,
        "residual_to_original_variance_ratio": ratio,
        "constant_residual": constant,
        "nearly_constant_residual": nearly_constant,
        "valid": reason == "valid",
        "reason_code": reason,
    }


def _iter_compact_profile_rows(path: Path) -> Iterator[dict[str, Any]]:
    parquet_file = pq.ParquetFile(path)
    try:
        for batch in parquet_file.iter_batches(
            batch_size=1,
            columns=["transcript_id", "transcript_length", "L_t"],
        ):
            values = batch.to_pydict()
            yield {
                "transcript_id": str(values["transcript_id"][0]),
                "transcript_length": int(values["transcript_length"][0]),
                "L_t": np.asarray(values["L_t"][0], dtype=np.float64).reshape(-1),
            }
            del values, batch
    finally:
        parquet_file.close()


def _evaluate_test_stream(
    *,
    panels: Sequence[str],
    test_profile_paths: Mapping[str, Path],
    test_ids: Sequence[str],
    sequence_connection: sqlite3.Connection,
    coefficients: Mapping[str, Mapping[str, np.ndarray]],
    codon_order: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
    memory: MemoryLogger,
) -> tuple[Path, Path]:
    pcc_path = output_dir / "heldout_pairwise_pcc.parquet"
    variance_path = output_dir / "heldout_variance_diagnostics.parquet"
    pcc_writer = BufferedParquetWriter(
        pcc_path, PCC_SCHEMA, args.scalar_write_batch_size
    )
    variance_writer = BufferedParquetWriter(
        variance_path, VARIANCE_SCHEMA, args.scalar_write_batch_size
    )
    iterators = {
        panel: _iter_compact_profile_rows(test_profile_paths[panel])
        for panel in panels
    }
    complete = False
    try:
        for transcript_index, expected_id in enumerate(test_ids, start=1):
            memory.guard("heldout_test", panel="all", index=transcript_index)
            current = {panel: next(iterators[panel]) for panel in panels}
            observed_ids = {row["transcript_id"] for row in current.values()}
            if observed_ids != {expected_id}:
                raise ValueError(
                    "Compact test profile exports are not synchronized with the "
                    f"sorted common test cohort at row {transcript_index}: "
                    f"expected {expected_id}, observed {sorted(observed_ids)}."
                )
            codons = _lookup_sequence(sequence_connection, expected_id)
            length = len(codons)
            original: dict[str, np.ndarray] = {}
            for panel in panels:
                values = current[panel]["L_t"]
                if values.size != length or current[panel]["transcript_length"] != length:
                    raise ValueError(f"{panel}/{expected_id}: test profile/codon mismatch.")
                if not np.isfinite(values).all():
                    raise ValueError(f"{panel}/{expected_id}: non-finite test profile.")
                original[panel] = values

            X_position, X_full = _design_matrices(codons, codon_order)
            baseline_predictions: dict[tuple[str, str], np.ndarray] = {}
            residuals: dict[tuple[str, str], np.ndarray] = {}
            validity: dict[tuple[str, str, str], tuple[bool, str]] = {}
            for panel in panels:
                for baseline, X in (
                    ("position_only", X_position),
                    ("codon_plus_position", X_full),
                ):
                    prediction = X @ coefficients[panel][baseline]
                    residual = original[panel] - prediction
                    # Preserve the raw signed residual exactly: no transform,
                    # smoothing, rectification, or renormalization.
                    baseline_predictions[(panel, baseline)] = prediction
                    residuals[(panel, baseline)] = residual
                    for region in REGIONS:
                        region_slice = _region_slice(
                            length, region, args.boundary_trim_codons
                        )
                        diagnostic = _variance_diagnostic(
                            original[panel][region_slice],
                            prediction[region_slice],
                            residual[region_slice],
                            floor=args.constant_variance_floor,
                            near_ratio=args.near_constant_variance_ratio,
                        )
                        validity[(panel, baseline, region)] = (
                            bool(diagnostic["valid"]),
                            str(diagnostic["reason_code"]),
                        )
                        variance_writer.append(
                            {
                                "transcript_id": expected_id,
                                "transcript_length": length,
                                "panel": panel,
                                "baseline": baseline,
                                "region": region,
                                "n_positions": int(
                                    original[panel][region_slice].size
                                ),
                                **diagnostic,
                            }
                        )

            for panel_a, panel_b in itertools.combinations(panels, 2):
                pair = (
                    f"{int(panel_a.removeprefix('panel_'))}-"
                    f"{int(panel_b.removeprefix('panel_'))}"
                )
                for region in REGIONS:
                    region_slice = _region_slice(
                        length, region, args.boundary_trim_codons
                    )
                    original_a = original[panel_a][region_slice]
                    original_b = original[panel_b][region_slice]
                    original_valid = (
                        original_a.size >= 2
                        and float(np.var(original_a)) > args.constant_variance_floor
                        and float(np.var(original_b)) > args.constant_variance_floor
                    )
                    required_flags = [
                        validity[(panel, baseline, region)]
                        for panel in (panel_a, panel_b)
                        for baseline in BASELINES
                    ]
                    matched_valid = original_valid and all(
                        flag for flag, _ in required_flags
                    )
                    if matched_valid:
                        reason = "valid"
                    elif not original_valid:
                        reason = "constant_original_or_too_short"
                    else:
                        reason = "|".join(
                            f"{panel}:{baseline}:{validity[(panel, baseline, region)][1]}"
                            for panel in (panel_a, panel_b)
                            for baseline in BASELINES
                            if not validity[(panel, baseline, region)][0]
                        )
                    variants = (
                        ("original_profile", original_a, original_b),
                        (
                            "position_only_residual",
                            residuals[(panel_a, "position_only")][region_slice],
                            residuals[(panel_b, "position_only")][region_slice],
                        ),
                        (
                            "codon_plus_position_residual",
                            residuals[(panel_a, "codon_plus_position")][region_slice],
                            residuals[(panel_b, "codon_plus_position")][region_slice],
                        ),
                    )
                    for variant, left, right in variants:
                        pcc_writer.append(
                            {
                                "transcript_id": expected_id,
                                "transcript_length": length,
                                "region": region,
                                "panel_a": panel_a,
                                "panel_b": panel_b,
                                "panel_pair": pair,
                                "profile_variant": variant,
                                "n_positions": int(left.size),
                                "pcc": (
                                    _pearson(
                                        left,
                                        right,
                                        args.constant_variance_floor,
                                    )
                                    if matched_valid
                                    else float("nan")
                                ),
                                "valid": matched_valid,
                                "reason_code": reason,
                            }
                        )

            del current, codons, original, X_position, X_full
            del baseline_predictions, residuals, validity
            if transcript_index % 100 == 0:
                memory.record(
                    "heldout_test",
                    panel="all",
                    index=transcript_index,
                    transcripts_processed=transcript_index,
                )
        # Detect extra rows when the complete test cohort is requested.
        if args.test_max_transcripts is None:
            for panel in panels:
                try:
                    extra = next(iterators[panel])
                except StopIteration:
                    continue
                raise ValueError(
                    f"{panel} compact test profile contains extra row {extra['transcript_id']}."
                )
        complete = True
    finally:
        for iterator in iterators.values():
            iterator.close()
        pcc_writer.close(complete=complete)
        variance_writer.close(complete=complete)
    return pcc_path, variance_path


def _parquet_to_csv_streaming(path: Path, output_path: Path, batch_size: int = 1_000) -> None:
    parquet_file = pq.ParquetFile(path)
    handle = output_path.open("w", encoding="utf-8", newline="")
    try:
        fields = parquet_file.schema_arrow.names
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                writer.writerow(row)
            del batch
    finally:
        handle.flush()
        handle.close()
        parquet_file.close()


def _stable_seed(base: int, label: str) -> int:
    salt = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little")
    return int((int(base) + salt) % (2**63 - 1))


def _bootstrap(values: np.ndarray, *, replicates: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    medians = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sample = values[rng.integers(0, values.size, size=values.size)]
        means[replicate] = float(sample.mean())
        medians[replicate] = float(np.median(sample))
    return means, medians


def _load_valid_pcc_values(
    pcc_path: Path,
) -> dict[tuple[str, str, str], dict[str, float]]:
    values: dict[tuple[str, str, str], dict[str, float]] = {}
    parquet_file = pq.ParquetFile(pcc_path)
    try:
        for batch in parquet_file.iter_batches(
            batch_size=1_000,
            columns=[
                "transcript_id",
                "region",
                "panel_pair",
                "profile_variant",
                "pcc",
                "valid",
            ],
        ):
            for row in batch.to_pylist():
                if not row["valid"] or row["pcc"] is None:
                    continue
                pcc = float(row["pcc"])
                if not math.isfinite(pcc):
                    continue
                key = (
                    str(row["region"]),
                    str(row["panel_pair"]),
                    str(row["profile_variant"]),
                )
                values.setdefault(key, {})[str(row["transcript_id"])] = pcc
            del batch
    finally:
        parquet_file.close()
    return values


def _summarize_pcc(
    *,
    pcc_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = _load_valid_pcc_values(pcc_path)
    pairs = sorted({key[1] for key in raw})
    summaries: list[dict[str, Any]] = []
    bootstrap_schema = pa.schema(
        [
            ("region", pa.string()),
            ("panel_pair", pa.string()),
            ("profile_variant", pa.string()),
            ("statistic", pa.string()),
            ("replicate", pa.int32()),
            ("value", pa.float64()),
        ]
    )
    bootstrap_writer = BufferedParquetWriter(
        output_dir / "heldout_cluster_bootstrap_replicates.parquet",
        bootstrap_schema,
        args.scalar_write_batch_size,
    )
    complete = False
    try:
        for region in REGIONS:
            for pair in pairs:
                variant_maps = {
                    variant: raw.get((region, pair, variant), {})
                    for variant in PROFILE_VARIANTS
                }
                matched_ids = set.intersection(
                    *(set(mapping) for mapping in variant_maps.values())
                )
                for variant in PROFILE_VARIANTS:
                    ordered_ids = sorted(matched_ids)
                    data = np.asarray(
                        [variant_maps[variant][transcript_id] for transcript_id in ordered_ids],
                        dtype=np.float64,
                    )
                    if not data.size:
                        continue
                    label = f"{region}|{pair}|{variant}"
                    means, medians = _bootstrap(
                        data,
                        replicates=args.bootstrap_replicates,
                        seed=_stable_seed(args.bootstrap_seed, label),
                    )
                    for statistic, point, distribution in (
                        ("mean_pcc", float(data.mean()), means),
                        ("median_pcc", float(np.median(data)), medians),
                    ):
                        summaries.append(
                            {
                                "region": region,
                                "panel_pair": pair,
                                "profile_variant": variant,
                                "statistic": statistic,
                                "point_estimate": point,
                                "bootstrap_ci95_low": float(np.percentile(distribution, 2.5)),
                                "bootstrap_ci95_high": float(np.percentile(distribution, 97.5)),
                                "n_matched_transcripts": int(data.size),
                                "bootstrap_cluster": "transcript",
                                "bootstrap_replicates": args.bootstrap_replicates,
                            }
                        )
                        for replicate, value in enumerate(distribution):
                            bootstrap_writer.append(
                                {
                                    "region": region,
                                    "panel_pair": pair,
                                    "profile_variant": variant,
                                    "statistic": statistic,
                                    "replicate": replicate,
                                    "value": float(value),
                                }
                            )

            # Pool dependent panel pairs inside each transcript first, then
            # bootstrap transcript clusters.
            pooled_maps: dict[str, dict[str, float]] = {}
            for variant in PROFILE_VARIANTS:
                transcript_to_values: dict[str, list[float]] = {}
                for pair in pairs:
                    for transcript_id, value in raw.get((region, pair, variant), {}).items():
                        transcript_to_values.setdefault(transcript_id, []).append(value)
                pooled_maps[variant] = {
                    transcript_id: float(np.median(values))
                    for transcript_id, values in transcript_to_values.items()
                    if len(values) == len(pairs)
                }
            pooled_matched = set.intersection(
                *(set(mapping) for mapping in pooled_maps.values())
            )
            for variant in PROFILE_VARIANTS:
                data = np.asarray(
                    [pooled_maps[variant][transcript_id] for transcript_id in sorted(pooled_matched)],
                    dtype=np.float64,
                )
                if not data.size:
                    continue
                pair_label = "pooled_transcript_median_across_six_pairs"
                label = f"{region}|{pair_label}|{variant}"
                means, medians = _bootstrap(
                    data,
                    replicates=args.bootstrap_replicates,
                    seed=_stable_seed(args.bootstrap_seed, label),
                )
                for statistic, point, distribution in (
                    ("mean_pcc", float(data.mean()), means),
                    ("median_pcc", float(np.median(data)), medians),
                ):
                    summaries.append(
                        {
                            "region": region,
                            "panel_pair": pair_label,
                            "profile_variant": variant,
                            "statistic": statistic,
                            "point_estimate": point,
                            "bootstrap_ci95_low": float(np.percentile(distribution, 2.5)),
                            "bootstrap_ci95_high": float(np.percentile(distribution, 97.5)),
                            "n_matched_transcripts": int(data.size),
                            "bootstrap_cluster": "transcript_after_within_transcript_pair_median",
                            "bootstrap_replicates": args.bootstrap_replicates,
                        }
                    )
                    for replicate, value in enumerate(distribution):
                        bootstrap_writer.append(
                            {
                                "region": region,
                                "panel_pair": pair_label,
                                "profile_variant": variant,
                                "statistic": statistic,
                                "replicate": replicate,
                                "value": float(value),
                            }
                        )
        complete = True
    finally:
        bootstrap_writer.close(complete=complete)

    summary_fields = (
        "region",
        "panel_pair",
        "profile_variant",
        "statistic",
        "point_estimate",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "n_matched_transcripts",
        "bootstrap_cluster",
        "bootstrap_replicates",
    )
    _write_csv(output_dir / "heldout_agreement_summary.csv", summaries, summary_fields)

    deltas: list[dict[str, Any]] = []
    for region in REGIONS:
        for pair in [*pairs, "pooled_transcript_median_across_six_pairs"]:
            if pair.startswith("pooled"):
                # Recover the values represented by the pooled summary.
                variant_maps: dict[str, dict[str, float]] = {}
                for variant in PROFILE_VARIANTS:
                    transcript_values: dict[str, list[float]] = {}
                    for source_pair in pairs:
                        for transcript_id, value in raw.get(
                            (region, source_pair, variant), {}
                        ).items():
                            transcript_values.setdefault(transcript_id, []).append(value)
                    variant_maps[variant] = {
                        transcript_id: float(np.median(values))
                        for transcript_id, values in transcript_values.items()
                        if len(values) == len(pairs)
                    }
            else:
                variant_maps = {
                    variant: raw.get((region, pair, variant), {})
                    for variant in PROFILE_VARIANTS
                }
            matched = set.intersection(*(set(mapping) for mapping in variant_maps.values()))
            ordered = sorted(matched)
            original = np.asarray(
                [variant_maps["original_profile"][item] for item in ordered], dtype=np.float64
            )
            for variant in PROFILE_VARIANTS[1:]:
                adjusted = np.asarray(
                    [variant_maps[variant][item] for item in ordered], dtype=np.float64
                )
                if not adjusted.size:
                    continue
                delta = adjusted - original
                means, medians = _bootstrap(
                    delta,
                    replicates=args.bootstrap_replicates,
                    seed=_stable_seed(
                        args.bootstrap_seed, f"delta|{region}|{pair}|{variant}"
                    ),
                )
                for statistic, point, distribution in (
                    ("mean_delta_pcc", float(delta.mean()), means),
                    ("median_delta_pcc", float(np.median(delta)), medians),
                ):
                    deltas.append(
                        {
                            "region": region,
                            "panel_pair": pair,
                            "profile_variant": variant,
                            "statistic": statistic,
                            "point_estimate": point,
                            "bootstrap_ci95_low": float(np.percentile(distribution, 2.5)),
                            "bootstrap_ci95_high": float(np.percentile(distribution, 97.5)),
                            "n_matched_transcripts": int(delta.size),
                        }
                    )
    _write_csv(
        output_dir / "heldout_agreement_paired_deltas.csv",
        deltas,
        (
            "region",
            "panel_pair",
            "profile_variant",
            "statistic",
            "point_estimate",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "n_matched_transcripts",
        ),
    )
    return summaries, deltas


def _summarize_variance(variance_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    parquet_file = pq.ParquetFile(variance_path)
    try:
        for batch in parquet_file.iter_batches(batch_size=1_000):
            for row in batch.to_pylist():
                key = (str(row["panel"]), str(row["baseline"]), str(row["region"]))
                group = groups.setdefault(
                    key,
                    {
                        "original": [],
                        "baseline": [],
                        "residual": [],
                        "ratio": [],
                        "n": 0,
                        "valid": 0,
                        "constant": 0,
                        "near": 0,
                        "too_short": 0,
                    },
                )
                group["n"] += 1
                group["valid"] += int(bool(row["valid"]))
                group["constant"] += int(bool(row["constant_residual"]))
                group["near"] += int(bool(row["nearly_constant_residual"]))
                group["too_short"] += int(row["reason_code"] == "too_short_for_region")
                if row["valid"]:
                    group["original"].append(float(row["original_variance"]))
                    group["baseline"].append(float(row["baseline_variance"]))
                    group["residual"].append(float(row["residual_variance"]))
                    group["ratio"].append(
                        float(row["residual_to_original_variance_ratio"])
                    )
            del batch
    finally:
        parquet_file.close()
    rows_out: list[dict[str, Any]] = []
    for (panel, baseline, region), group in sorted(groups.items()):
        median = lambda name: (
            float(np.median(np.asarray(group[name], dtype=np.float64)))
            if group[name]
            else float("nan")
        )
        rows_out.append(
            {
                "panel": panel,
                "baseline": baseline,
                "region": region,
                "n_transcripts": group["n"],
                "n_valid": group["valid"],
                "n_constant_residuals": group["constant"],
                "n_nearly_constant_residuals": group["near"],
                "n_too_short": group["too_short"],
                "median_original_variance": median("original"),
                "median_baseline_variance": median("baseline"),
                "median_residual_variance": median("residual"),
                "median_residual_to_original_variance_ratio": median("ratio"),
            }
        )
    _write_csv(
        output_dir / "heldout_variance_summary.csv",
        rows_out,
        tuple(rows_out[0]) if rows_out else (),
    )
    return rows_out


@latex_paper_style
def _plot_summary(summaries: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    pairs = sorted(
        {
            str(row["panel_pair"])
            for row in summaries
            if not str(row["panel_pair"]).startswith("pooled")
        }
    )
    labels = {
        "original_profile": "Original profile",
        "position_only_residual": "Position-only residual",
        "codon_plus_position_residual": "Codon + position residual",
    }
    colors = {
        "original_profile": "#4575b4",
        "position_only_residual": "#e08214",
        "codon_plus_position_residual": "#1b9e77",
    }
    figure, axes = plt.subplots(
        2, 1, figsize=(10.2, 7.0), sharex=True, constrained_layout=True
    )
    plotted_rows = [
        row
        for row in summaries
        if not str(row["panel_pair"]).startswith("pooled")
        and row["statistic"] == "median_pcc"
    ]
    ci_min = min(float(row["bootstrap_ci95_low"]) for row in plotted_rows)
    ci_max = max(float(row["bootstrap_ci95_high"]) for row in plotted_rows)
    # Use one honest, shared zoom for both regions.  The padding and 0.05
    # rounding keep the axis legible while exposing changes that disappear on
    # a mechanically fixed [-1, 1] PCC scale.
    plot_lower = max(-1.0, math.floor((ci_min - 0.02) / 0.05) * 0.05)
    plot_upper = min(1.0, math.ceil((ci_max + 0.02) / 0.05) * 0.05)
    offsets = (-0.22, 0.0, 0.22)
    for axis, region in zip(axes, REGIONS, strict=True):
        for offset, variant in zip(offsets, PROFILE_VARIANTS, strict=True):
            lookup = {
                str(row["panel_pair"]): row
                for row in summaries
                if row["region"] == region
                and row["profile_variant"] == variant
                and row["statistic"] == "median_pcc"
            }
            selected = [lookup[pair] for pair in pairs]
            points = np.asarray([row["point_estimate"] for row in selected])
            lower = np.asarray([row["bootstrap_ci95_low"] for row in selected])
            upper = np.asarray([row["bootstrap_ci95_high"] for row in selected])
            x = np.arange(len(pairs), dtype=float) + offset
            axis.errorbar(
                x,
                points,
                yerr=np.vstack((points - lower, upper - points)),
                fmt="o",
                color=colors[variant],
                label=labels[variant],
                capsize=2.5,
                linewidth=1.2,
            )
        axis.set_ylim(plot_lower, plot_upper)
        axis.grid(axis="y", alpha=0.22)
        axis.set_ylabel("Median transcript PCC\n(95% cluster-bootstrap CI)")
        axis.set_title(
            "Full CDS"
            if region == "full_cds"
            else "Interior: exclude 20 codons at each end"
        )
    axes[-1].set_xticks(np.arange(len(pairs)), pairs)
    axes[-1].set_xlabel("Independent panel pair")
    axes[0].legend(ncol=3, frameon=False, loc="lower center")
    figure.suptitle(
        "Cross-panel agreement beyond fixed panel-specific additive baselines",
        fontsize=12,
        fontweight="semibold",
    )
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "cross_panel_agreement_adjusted.pdf", bbox_inches="tight")
    figure.savefig(
        figures / "cross_panel_agreement_adjusted.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    del figure, axes


def _summary_row(
    summaries: Sequence[Mapping[str, Any]], region: str, variant: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in summaries
        if row["region"] == region
        and row["panel_pair"] == "pooled_transcript_median_across_six_pairs"
        and row["profile_variant"] == variant
        and row["statistic"] == "median_pcc"
    ]
    if len(matches) != 1:
        raise KeyError(f"Missing pooled summary for {region}/{variant}.")
    return matches[0]


def _write_interpretation(
    *,
    output_dir: Path,
    cohort: Mapping[str, Any],
    selected_fit_ids: Sequence[str],
    selected_test_ids: Sequence[str],
    summaries: Sequence[Mapping[str, Any]],
    variance_summary: Sequence[Mapping[str, Any]],
    max_observed_rss: float,
) -> None:
    lines = [
        "# Frozen codon/position residual analysis",
        "",
        "This is a panel-specific post-hoc least-squares diagnostic on frozen RiboUnmix shared profiles. It is not a shared-only ablation, does not update neural parameters, does not change gamma or pi, and does not identify a biological-versus-technical decomposition.",
        "",
        "## Cohorts and execution",
        "",
        f"- Full four-panel training intersection after every validation/test exclusion: **{cohort['fitting_transcript_count']:,} transcripts**, hash `{cohort['fitting_transcript_id_hash']}`.",
        f"- Transcripts used in this fit: **{len(selected_fit_ids):,}**, hash `{_id_hash(selected_fit_ids)}`.",
        f"- Held-out common-test transcripts evaluated: **{len(selected_test_ids):,}**, hash `{_id_hash(selected_test_ids)}`.",
        f"- Maximum logged RSS: **{max_observed_rss:.3f} GiB**.",
        "- WLS gives each transcript total weight one (`1/n_t` per position). Coefficients are fit independently to each panel's own frozen L_t.",
        "",
        "## Matched held-out agreement",
        "",
        "Each pooled value is the median PCC across six panel pairs within a transcript, followed by the median across transcripts. Intervals resample transcript clusters.",
        "",
        "| Region | Original | Position residual | Codon + position residual |",
        "|---|---:|---:|---:|",
    ]
    for region, label in (("full_cds", "Full CDS"), ("interior_minus_20", "Interior −20")):
        rows = [_summary_row(summaries, region, variant) for variant in PROFILE_VARIANTS]
        values = [
            f"{row['point_estimate']:.3f} [{row['bootstrap_ci95_low']:.3f}, {row['bootstrap_ci95_high']:.3f}]"
            for row in rows
        ]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(("", "## Variance validity", ""))
    for baseline in BASELINES:
        full_rows = [
            row
            for row in variance_summary
            if row["baseline"] == baseline and row["region"] == "full_cds"
        ]
        ratios = np.asarray(
            [row["median_residual_to_original_variance_ratio"] for row in full_rows],
            dtype=np.float64,
        )
        invalid = sum(
            int(row["n_constant_residuals"])
            + int(row["n_nearly_constant_residuals"])
            for row in full_rows
        )
        total = sum(int(row["n_transcripts"]) for row in full_rows)
        lines.append(
            f"- `{baseline}`: median panel-level residual/original variance ratio "
            f"{np.median(ratios):.4g}; constant or nearly constant residuals "
            f"{invalid:,}/{total:,} panel-transcript profiles."
        )
    full = [_summary_row(summaries, "full_cds", variant) for variant in PROFILE_VARIANTS]
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            f"On the full CDS, pooled median agreement changes from {full[0]['point_estimate']:.3f} for original profiles to {full[1]['point_estimate']:.3f} after the position-only subtraction and {full[2]['point_estimate']:.3f} after the codon-plus-position subtraction.",
            "",
            "Persistent residual agreement supports reproducible shared structure beyond these fixed additive baselines. A decrease means that the specified positional and/or codon effects account for part of the common signal. Those effects may themselves be biological. This diagnostic does not assign the remaining or removed signal to biological versus technical sources.",
            "",
        )
    )
    (output_dir / "RESULTS_INTERPRETATION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_design_specification(
    output_dir: Path, codon_order: Sequence[str], args: argparse.Namespace
) -> None:
    _write_json(
        output_dir / "design_specification.json",
        {
            "position": "u=(i-1)/(n_t-1); u=0.5 only for the degenerate n_t=1 case",
            "baselines": {
                "position_only": "beta_0,k + s_k(u)",
                "codon_plus_position": "beta_0,k + beta_c,k + s_k(u)",
            },
            "natural_cubic_spline": {
                "boundary_knots": [0.0, 1.0],
                "interior_knots": [0.2, 0.4, 0.6, 0.8],
                "basis_excluding_intercept": [
                    "u",
                    "d_0(u)-d_4(u)",
                    "d_1(u)-d_4(u)",
                    "d_2(u)-d_4(u)",
                    "d_3(u)-d_4(u)",
                ],
                "d_j": "((u-xi_j)_+^3-(u-xi_5)_+^3)/(xi_5-xi_j)",
                "fixed_complexity_no_tuning": True,
            },
            "codon_effect": {
                "coding": "treatment contrasts plus intercept",
                "reference_codon": codon_order[0],
                "ordered_codon_identities": list(codon_order),
                "contrast_count": 63,
            },
            "dimensions": {"position_only": 6, "codon_plus_position": 69},
            "wls": "each transcript has total weight 1; every position weight is 1/n_t",
            "streaming_update": [
                "A <- A + X.T @ (w[:,None] * X)",
                "b <- b + X.T @ (w * y)",
            ],
            "maximum_design_positions_per_block": args.max_design_positions,
            "test_interior_uses_same_coefficients": True,
            "residual_transform": "none; signed L_t-b_k retained without normalization",
        },
    )


def _write_reproduce_command(
    output_dir: Path, run_root: Path, args: argparse.Namespace
) -> None:
    command = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        "-u",
        "analyses/analyze_real_panel_posthoc_robustness_streaming.py",
        "--run-root",
        str(run_root),
        "--output-dir",
        str(output_dir),
        "--device",
        str(args.device),
        "--max-rss-gb",
        str(args.max_rss_gb),
        "--rss-stop-fraction",
        str(args.rss_stop_fraction),
        "--batch-size",
        str(args.batch_size),
        "--max-padded-codon-tokens",
        str(args.max_padded_codon_tokens),
        "--max-design-positions",
        str(args.max_design_positions),
        "--boundary-trim-codons",
        str(args.boundary_trim_codons),
        "--mean-one-tolerance",
        str(args.mean_one_tolerance),
        "--constant-variance-floor",
        str(args.constant_variance_floor),
        "--near-constant-variance-ratio",
        str(args.near_constant_variance_ratio),
        "--verification-transcripts",
        str(args.verification_transcripts),
        "--minimum-replay-pcc",
        str(args.minimum_replay_pcc),
        "--bootstrap-replicates",
        str(args.bootstrap_replicates),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--scalar-write-batch-size",
        str(args.scalar_write_batch_size),
    ]
    if args.fit_max_transcripts is not None:
        command.extend(("--fit-max-transcripts", str(args.fit_max_transcripts)))
    if args.test_max_transcripts is not None:
        command.extend(("--test-max-transcripts", str(args.test_max_transcripts)))
    if args.training_cache_dir is not None:
        command.extend(("--training-cache-dir", str(args.training_cache_dir)))
    log_path = output_dir / "execution.log"
    shell = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        'export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/posthoc_streaming_mpl}"\n'
        + " ".join(shlex.quote(part) for part in command)
        + f" > {shlex.quote(str(log_path))} 2>&1\n"
    )
    path = output_dir / "reproduce_command.sh"
    path.write_text(shell, encoding="utf-8")
    path.chmod(0o755)


def _read_max_rss(memory_log: Path) -> float:
    maximum = 0.0
    with memory_log.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            maximum = max(maximum, float(row["rss_gb"]))
    return maximum


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    run_root = args.run_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else artifact_directory("real_data", run_root, DEFAULT_OUTPUT_NAME)
    )
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coefficients").mkdir(exist_ok=True)
    (output_dir / "fit_provenance").mkdir(exist_ok=True)
    device = _resolve_device(args.device)
    memory_path = output_dir / "memory_log.csv"
    memory = MemoryLogger(
        memory_path,
        max_rss_gb=args.max_rss_gb,
        stop_fraction=args.rss_stop_fraction,
        device=device,
    )
    status: dict[str, Any] = {
        "status": "running",
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "neural_parameter_updates": False,
        "gamma_or_pi_changes": False,
    }
    _write_json(output_dir / "execution_status.json", status)
    sequence_connection: sqlite3.Connection | None = None
    try:
        memory.record("startup", event="start")
        panels = _panel_names(run_root)
        folds, full_test_ids = _load_folds(run_root, panels)
        full_fitting_ids, cohort = _construct_fitting_cohort(folds, panels)
        selected_fitting_ids = (
            full_fitting_ids[: args.fit_max_transcripts]
            if args.fit_max_transcripts is not None
            else full_fitting_ids
        )
        selected_test_ids = (
            full_test_ids[: args.test_max_transcripts]
            if args.test_max_transcripts is not None
            else full_test_ids
        )
        cohort.update(
            {
                "selected_fitting_transcript_count": len(selected_fitting_ids),
                "selected_fitting_transcript_id_hash": _id_hash(selected_fitting_ids),
                "fit_max_transcripts": args.fit_max_transcripts,
                "full_common_test_count": len(full_test_ids),
                "full_common_test_id_hash": _id_hash(full_test_ids),
                "selected_test_transcript_count": len(selected_test_ids),
                "selected_test_transcript_id_hash": _id_hash(selected_test_ids),
                "test_max_transcripts": args.test_max_transcripts,
            }
        )
        _write_json(output_dir / "cohort_provenance.json", cohort)
        _write_csv(
            output_dir / "fitting_cohort_transcripts.csv",
            ({"transcript_id": value} for value in selected_fitting_ids),
            ("transcript_id",),
        )
        _write_csv(
            output_dir / "heldout_test_transcripts.csv",
            ({"transcript_id": value} for value in selected_test_ids),
            ("transcript_id",),
        )
        test_profile_paths = {
            panel: _locate_test_profile(run_root / panel) for panel in panels
        }
        first_cfg = _load_config(run_root / panels[0])
        sequence_path = _local_sequence_path(first_cfg)
        from main_ribounmix_multidataset import open_file

        codon_encoding = open_file(_repo_path(first_cfg.paths.encodings.codon))
        codon_order = tuple(sorted(map(str, codon_encoding)))
        if len(codon_order) != 64 or codon_order[0] != "AAA":
            raise ValueError(
                f"Expected 64 codons with lexicographic reference AAA; got {codon_order[:3]}."
            )
        _write_design_specification(output_dir, codon_order, args)
        del first_cfg, codon_encoding

        sequence_manifest = _create_sequence_index(
            sqlite_path=output_dir / "sequence_lookup.sqlite",
            manifest_path=output_dir / "sequence_lookup_manifest.json",
            sequence_path=sequence_path,
            fitting_ids=selected_fitting_ids,
            test_ids=selected_test_ids,
            memory=memory,
        )
        sequence_connection = sqlite3.connect(output_dir / "sequence_lookup.sqlite")
        memory.record("sequence_index_ready", event="complete")

        all_coefficients: dict[str, dict[str, np.ndarray]] = {}
        panel_provenance: dict[str, Any] = {}
        replay_verification: list[dict[str, Any]] = []
        verification_ids = selected_test_ids[
            : min(args.verification_transcripts, len(selected_test_ids))
        ]
        for panel in panels:
            coefficients, provenance, verification = _fit_panel(
                panel=panel,
                panel_dir=run_root / panel,
                run_root=run_root,
                fitting_ids=selected_fitting_ids,
                full_fitting_hash=cohort["fitting_transcript_id_hash"],
                test_profile_path=test_profile_paths[panel],
                verification_ids=verification_ids,
                sequence_connection=sequence_connection,
                codon_order=codon_order,
                output_dir=output_dir,
                args=args,
                memory=memory,
                device=device,
            )
            all_coefficients[panel] = coefficients
            panel_provenance[panel] = provenance
            replay_verification.extend(verification)

        coefficient_rows: list[dict[str, Any]] = []
        for panel in panels:
            with (output_dir / "coefficients" / f"{panel}.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                coefficient_rows.extend(csv.DictReader(handle))
        _write_csv(
            output_dir / "diagnostic_regression_coefficients.csv",
            coefficient_rows,
            ("panel", "baseline", "term_index", "term", "coefficient"),
        )
        if replay_verification:
            _write_csv(
                output_dir / "frozen_replay_verification.csv",
                replay_verification,
                (
                    "panel",
                    "transcript_id",
                    "transcript_length",
                    "replay_vs_stored_pcc",
                    "replay_vs_stored_rmse",
                    "replay_vs_stored_max_abs_difference",
                ),
            )
        _write_json(
            output_dir / "fit_provenance.json",
            {
                "cohort": cohort,
                "sequence_index": sequence_manifest,
                "panels": panel_provenance,
                "fit_method": "streaming panel-specific WLS sufficient statistics",
                "neural_parameter_updates": False,
                "gamma_or_pi_changes": False,
                "more_than_one_checkpoint_loaded_at_once": False,
            },
        )

        pcc_path, variance_path = _evaluate_test_stream(
            panels=panels,
            test_profile_paths=test_profile_paths,
            test_ids=selected_test_ids,
            sequence_connection=sequence_connection,
            coefficients=all_coefficients,
            codon_order=codon_order,
            output_dir=output_dir,
            args=args,
            memory=memory,
        )
        sequence_connection.close()
        sequence_connection = None
        _parquet_to_csv_streaming(
            pcc_path, output_dir / "heldout_pairwise_pcc.csv"
        )
        _parquet_to_csv_streaming(
            variance_path, output_dir / "heldout_variance_diagnostics.csv"
        )
        summaries, deltas = _summarize_pcc(
            pcc_path=pcc_path, output_dir=output_dir, args=args
        )
        variance_summary = _summarize_variance(variance_path, output_dir)
        memory.record("plot_start", event="start")
        _plot_summary(summaries, output_dir)
        memory.record("complete", event="complete")
        _write_reproduce_command(output_dir, run_root, args)
        maximum_rss = _read_max_rss(memory_path)
        _write_interpretation(
            output_dir=output_dir,
            cohort=cohort,
            selected_fit_ids=selected_fitting_ids,
            selected_test_ids=selected_test_ids,
            summaries=summaries,
            variance_summary=variance_summary,
            max_observed_rss=maximum_rss,
        )
        status.update(
            {
                "status": "complete",
                "fitting_transcript_count": len(selected_fitting_ids),
                "test_transcript_count": len(selected_test_ids),
                "maximum_logged_rss_gb": maximum_rss,
                "device": str(device),
                "cpu_numerical_threads": 1,
            }
        )
        _write_json(output_dir / "execution_status.json", status)
        print(
            f"Complete: fit={len(selected_fitting_ids):,}, "
            f"test={len(selected_test_ids):,}, max RSS={maximum_rss:.3f} GiB"
        )
        print(output_dir)
        return 0
    except Exception as exc:
        status.update(
            {
                "status": "stopped_cleanly_memory_guard"
                if isinstance(exc, MemoryBudgetExceeded)
                else "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(output_dir / "execution_status.json", status)
        raise
    finally:
        if sequence_connection is not None:
            sequence_connection.close()
        memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
