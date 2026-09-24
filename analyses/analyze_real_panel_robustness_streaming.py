#!/usr/bin/env python3
"""Low-memory positional robustness analysis for the four real-data panels.

The script deliberately consumes only the compact, best-validation-loss
``common_test_L_profiles.parquet`` exports.  Prediction arrays are read with
one-row PyArrow batches and only the four arrays for the current transcript
are retained.  All position-level masks are temporary; only scalar summaries
are written, incrementally, to Parquet.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# Keep implicit BLAS/OpenMP fan-out under control even when the caller forgets
# the recommended shell prefix.  The exact recorded commands set these too.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ribounmix-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr

try:
    import psutil
except ImportError as exc:  # pragma: no cover - exercised by environment setup
    # Managed workstations can have an ABI3 build in their local Conda package
    # cache even when the active venv cannot reach PyPI. Import that genuine
    # psutil package if available; otherwise do not weaken the requested guard.
    cache_candidates = sorted(
        (Path.home() / ".local/share/miniforge3/pkgs").glob(
            "psutil-*/lib/python*/site-packages"
        )
    )
    for candidate in reversed(cache_candidates):
        sys.path.insert(0, str(candidate))
        try:
            import psutil  # type: ignore[no-redef]

            break
        except ImportError:
            sys.path.pop(0)
    else:
        raise SystemExit(
            "psutil is required for the requested RSS hard limit. Install it "
            "with `python -m pip install psutil` and rerun."
        ) from exc


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
RESULTS_DIR = SCRIPT_PATH.parent
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT / "results" / "my_panels_a100_b32_20260906_114323"
)
DEFAULT_OUTPUT_NAME = "streaming_position_sensitivity"
PANEL_NAMES = ("panel_01", "panel_02", "panel_03", "panel_04")
PANEL_PAIRS = tuple(itertools.combinations(PANEL_NAMES, 2))
EXPECTED_PAIR_MEDIANS = {
    "panel_01--panel_02": 0.718,
    "panel_01--panel_03": 0.778,
    "panel_01--panel_04": 0.736,
    "panel_02--panel_03": 0.751,
    "panel_02--panel_04": 0.755,
    "panel_03--panel_04": 0.775,
}
EXPECTED_POOLED_MEDIAN = 0.752
STOP_CODONS = {"TAA", "TAG", "TGA"}

if str(RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Utils.publication_plot_style import latex_paper_style  # noqa: E402
from analyses.paths import artifact_directory  # noqa: E402
from analyze_real_panel_convergence import (  # noqa: E402
    _locate_panel_prediction,
    _pearson,
    _read_json,
    _spearman,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream four compact held-out L_t exports and measure positional "
            "and boundary robustness without loading model checkpoints."
        )
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Base output directory. Default: RUN_ROOT/posthoc_robustness/"
            f"{DEFAULT_OUTPUT_NAME}. A full/ or smoke_XXXXX/ child is created."
        ),
    )
    parser.add_argument("--max-rss-gb", type=float, default=4.0)
    parser.add_argument(
        "--max-transcripts",
        type=int,
        default=None,
        help="Analyze only the first N manifest transcripts (smoke-test mode).",
    )
    parser.add_argument("--random-draws", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=93017)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=77123)
    parser.add_argument("--mean-one-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this script's selected full/smoke output directory.",
    )
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


class MemoryGuard:
    """Track current RSS through psutil and enforce a hard user-supplied cap."""

    def __init__(self, path: Path, limit_gb: float) -> None:
        if not math.isfinite(limit_gb) or limit_gb <= 0:
            raise ValueError("--max-rss-gb must be positive and finite.")
        self.path = path
        self.limit_bytes = float(limit_gb) * (1024**3)
        self.process = psutil.Process(os.getpid())
        self.handle = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=[
                "timestamp_utc",
                "stage",
                "transcripts_processed",
                "rss_bytes",
                "rss_gib",
                "peak_observed_rss_bytes",
                "peak_observed_rss_gib",
                "limit_gib",
            ],
        )
        self.writer.writeheader()
        self.peak_bytes = 0
        self._last_logged: tuple[str, int] | None = None
        self.check("startup", 0, log=True)

    def check(self, stage: str, processed: int, *, log: bool = False) -> int:
        rss = int(self.process.memory_info().rss)
        self.peak_bytes = max(self.peak_bytes, rss)
        periodic = processed >= 0 and processed % 100 == 0
        should_log = log or (
            periodic and self._last_logged != (stage, int(processed))
        )
        if should_log:
            self.writer.writerow(
                {
                    "timestamp_utc": _utc_now(),
                    "stage": stage,
                    "transcripts_processed": int(processed),
                    "rss_bytes": rss,
                    "rss_gib": rss / (1024**3),
                    "peak_observed_rss_bytes": self.peak_bytes,
                    "peak_observed_rss_gib": self.peak_bytes / (1024**3),
                    "limit_gib": self.limit_bytes / (1024**3),
                }
            )
            self.handle.flush()
            print(
                f"[memory] {stage}: transcripts={processed}, "
                f"RSS={rss / (1024**3):.3f} GiB, "
                f"peak={self.peak_bytes / (1024**3):.3f} GiB",
                flush=True,
            )
            self._last_logged = (stage, int(processed))
        if rss > self.limit_bytes:
            raise MemoryError(
                f"RSS hard limit exceeded during {stage} after transcript "
                f"{processed}: {rss / (1024**3):.3f} GiB > "
                f"{self.limit_bytes / (1024**3):.3f} GiB. Partial Parquet "
                "files were closed; lower the workload or raise --max-rss-gb."
            )
        return rss

    def close(self) -> None:
        if not self.handle.closed:
            self.check("shutdown", -1, log=True)
            self.handle.close()


class BufferedParquetWriter:
    """Incremental scalar writer with a strict <=1000-row write batch."""

    def __init__(
        self,
        final_path: Path,
        schema: pa.Schema,
        *,
        batch_size: int = 1000,
    ) -> None:
        if not 1 <= batch_size <= 1000:
            raise ValueError("Parquet scalar write batch must be in [1, 1000].")
        self.final_path = final_path
        # PID-qualified staging prevents two accidentally overlapping shell
        # invocations from renaming or appending to one another's open file.
        self.partial_path = final_path.with_name(
            final_path.name + f".partial.{os.getpid()}"
        )
        self.schema = schema
        self.batch_size = batch_size
        self.buffer: list[dict[str, Any]] = []
        self.writer = pq.ParquetWriter(
            self.partial_path,
            schema,
            compression="zstd",
            use_dictionary=True,
        )
        self.rows_written = 0
        self.closed = False

    def append(self, row: Mapping[str, Any]) -> None:
        self.buffer.append(dict(row))
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def extend(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            self.append(row)

    def flush(self) -> None:
        while self.buffer:
            chunk = self.buffer[: self.batch_size]
            del self.buffer[: self.batch_size]
            table = pa.Table.from_pylist(chunk, schema=self.schema)
            self.writer.write_table(table, row_group_size=len(chunk))
            self.rows_written += len(chunk)
            del table

    def close(self, *, accept: bool) -> None:
        if self.closed:
            return
        self.flush()
        self.writer.close()
        self.closed = True
        if accept:
            os.replace(self.partial_path, self.final_path)


REFERENCE_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.string()),
        ("gene_id", pa.string()),
        ("panel_a", pa.string()),
        ("panel_b", pa.string()),
        ("panel_pair", pa.string()),
        ("transcript_length", pa.int32()),
        ("PCC", pa.float64()),
        ("Spearman", pa.float64()),
        ("RMSE", pa.float64()),
        ("valid", pa.bool_()),
        ("reason_code", pa.string()),
    ]
)

VALIDATION_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.string()),
        ("gene_id", pa.string()),
        ("panel", pa.string()),
        ("run_identifier", pa.string()),
        ("transcript_length", pa.int32()),
        ("profile_array_length", pa.int32()),
        ("implicit_valid_mask_count", pa.int32()),
        ("first_modeled_coordinate", pa.int32()),
        ("last_modeled_coordinate", pa.int32()),
        ("first_codon", pa.string()),
        ("terminal_codon", pa.string()),
        ("terminal_stop_included", pa.bool_()),
        ("codon_sequence_sha256", pa.string()),
        ("all_finite", pa.bool_()),
        ("all_positive", pa.bool_()),
        ("L_mean", pa.float64()),
        ("absolute_mean_one_deviation", pa.float64()),
    ]
)

METRIC_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.string()),
        ("gene_id", pa.string()),
        ("panel_a", pa.string()),
        ("panel_b", pa.string()),
        ("panel_pair", pa.string()),
        ("analysis_family", pa.string()),
        ("condition", pa.string()),
        ("transcript_length", pa.int32()),
        ("n_removed", pa.int32()),
        ("n_retained", pa.int32()),
        ("removed_fraction", pa.float64()),
        ("PCC", pa.float64()),
        ("Spearman", pa.float64()),
        ("RMSE", pa.float64()),
        ("retained_variance_a", pa.float64()),
        ("retained_variance_b", pa.float64()),
        ("reference_full_PCC", pa.float64()),
        ("paired_PCC_change", pa.float64()),
        ("valid", pa.bool_()),
        ("reason_code", pa.string()),
        ("random_draws_requested", pa.int16()),
        ("random_draws_valid", pa.int16()),
        ("PCC_sd_across_random_draws", pa.float64()),
        ("PCC_q025_across_random_draws", pa.float64()),
        ("PCC_q50_across_random_draws", pa.float64()),
        ("PCC_q975_across_random_draws", pa.float64()),
        ("Spearman_sd_across_random_draws", pa.float64()),
        ("Spearman_q025_across_random_draws", pa.float64()),
        ("Spearman_q50_across_random_draws", pa.float64()),
        ("Spearman_q975_across_random_draws", pa.float64()),
        ("RMSE_sd_across_random_draws", pa.float64()),
        ("RMSE_q025_across_random_draws", pa.float64()),
        ("RMSE_q50_across_random_draws", pa.float64()),
        ("RMSE_q975_across_random_draws", pa.float64()),
        ("in_boundary_fixed_cohort", pa.bool_()),
        ("in_combined_fixed_cohort", pa.bool_()),
    ]
)

MASK_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.string()),
        ("gene_id", pa.string()),
        ("mask_family", pa.string()),
        ("target_fraction_per_panel", pa.float64()),
        ("selection_domain_start", pa.int32()),
        ("selection_domain_stop_exclusive", pa.int32()),
        ("selection_domain_size", pa.int32()),
        ("ceil_target_count", pa.int32()),
        ("panel", pa.string()),
        ("cutoff", pa.float64()),
        ("panel_selected_count_with_ties", pa.int32()),
        ("panel_actual_selected_fraction", pa.float64()),
        ("four_panel_union_count", pa.int32()),
        ("four_panel_union_fraction_of_domain", pa.float64()),
        ("retained_count_after_union", pa.int32()),
        ("coordinate_convention", pa.string()),
    ]
)

JACCARD_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.string()),
        ("gene_id", pa.string()),
        ("panel_a", pa.string()),
        ("panel_b", pa.string()),
        ("panel_pair", pa.string()),
        ("transcript_length", pa.int32()),
        ("ceil_target_count", pa.int32()),
        ("selected_count_a_with_ties", pa.int32()),
        ("selected_count_b_with_ties", pa.int32()),
        ("actual_selected_fraction_a", pa.float64()),
        ("actual_selected_fraction_b", pa.float64()),
        ("intersection_count", pa.int32()),
        ("union_count", pa.int32()),
        ("jaccard", pa.float64()),
        ("full_profile_PCC", pa.float64()),
        ("valid", pa.bool_()),
        ("reason_code", pa.string()),
    ]
)

COHORT_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.string()),
        ("gene_id", pa.string()),
        ("transcript_length", pa.int32()),
        ("n_at_least_150", pa.bool_()),
        ("valid_all_six_pairs_at_k0_k5_k20_k50", pa.bool_()),
        ("boundary_fixed_cohort", pa.bool_()),
        ("valid_all_six_pairs_combined_condition", pa.bool_()),
        ("combined_fixed_cohort", pa.bool_()),
        ("exclusion_reason", pa.string()),
    ]
)

BOOTSTRAP_SCHEMA = pa.schema(
    [
        ("bootstrap_index", pa.int32()),
        ("estimator_type", pa.string()),
        ("condition", pa.string()),
        ("panel_pair", pa.string()),
        ("estimate", pa.float64()),
        ("cluster_unit", pa.string()),
        ("number_of_sampled_clusters", pa.int32()),
    ]
)


@dataclass(frozen=True)
class SequenceMetadata:
    gene_id: str
    length: int
    first_codon: str
    terminal_codon: str
    codon_sha256: str


@dataclass
class ProfileRecord:
    transcript_id: str
    panel: str
    run_identifier: str
    length: int
    values: np.ndarray


@dataclass
class MetricResult:
    pcc: float
    spearman: float
    rmse: float
    variance_a: float
    variance_b: float
    n: int
    valid: bool
    reason: str


def _resolve_sequence_path(run_root: Path, split_manifest: Mapping[str, Any]) -> Path:
    candidates = [
        PROJECT_ROOT / "Datasets/data/sequence/MANE.selection.cds_codons.parquet",
        PROJECT_ROOT
        / "Datasets/data/sequence/MANE.selection.sequence_embeddings_with_css.parquet",
    ]
    source = split_manifest.get("source_sequences_path")
    if source:
        raw = Path(str(source))
        if raw.exists():
            candidates.append(raw)
        candidates.append(PROJECT_ROOT / "Datasets/data/sequence" / raw.name)
    for candidate in candidates:
        if candidate.exists() and "codons" in pq.read_schema(candidate).names:
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve a local sequence artifact with codons for {run_root}."
    )


def _load_sequence_metadata(
    sequence_path: Path,
    expected_ids: Sequence[str],
    memory: MemoryGuard,
) -> dict[str, SequenceMetadata]:
    wanted = set(expected_ids)
    schema_names = set(pq.read_schema(sequence_path).names)
    columns = ["transcript_id", "codons"]
    has_gene = "gene_id" in schema_names
    if has_gene:
        columns.append("gene_id")
    metadata: dict[str, SequenceMetadata] = {}
    source = pa.memory_map(str(sequence_path), "r")
    try:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=128, columns=columns):
            names = batch.schema.names
            tid_col = batch.column(names.index("transcript_id"))
            codon_col = batch.column(names.index("codons"))
            gene_col = batch.column(names.index("gene_id")) if has_gene else None
            for index in range(batch.num_rows):
                transcript_id = str(tid_col[index].as_py())
                if transcript_id not in wanted:
                    continue
                if transcript_id in metadata:
                    raise ValueError(
                        f"Duplicate sequence row for held-out transcript {transcript_id}."
                    )
                codons = codon_col[index].as_py()
                if not isinstance(codons, list) or not codons:
                    raise ValueError(f"Missing codon coordinates for {transcript_id}.")
                normalized = [str(value).upper() for value in codons]
                digest = hashlib.sha256("\0".join(normalized).encode()).hexdigest()
                gene = (
                    str(gene_col[index].as_py())
                    if gene_col is not None and gene_col[index].as_py() is not None
                    else transcript_id
                )
                metadata[transcript_id] = SequenceMetadata(
                    gene_id=gene,
                    length=len(normalized),
                    first_codon=normalized[0],
                    terminal_codon=normalized[-1],
                    codon_sha256=digest,
                )
                del codons, normalized
            memory.check("sequence_metadata_scan", len(metadata))
    finally:
        source.close()
    missing = wanted - set(metadata)
    if missing:
        raise ValueError(
            f"Sequence artifact is missing {len(missing)} common-test IDs; "
            f"examples={sorted(missing)[:10]}."
        )
    non_stops = [
        transcript_id
        for transcript_id, record in metadata.items()
        if record.terminal_codon not in STOP_CODONS
    ]
    if non_stops:
        raise ValueError(
            "The compact-coordinate convention expects the terminal stop codon "
            f"to be included, but {len(non_stops)} transcripts fail; "
            f"examples={non_stops[:10]}."
        )
    return metadata


def _read_id_stream(path: Path) -> list[str]:
    identifiers: list[str] = []
    source = pa.memory_map(str(path), "r")
    try:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(
            batch_size=1000, columns=["transcript_id"]
        ):
            identifiers.extend(str(value) for value in batch.column(0).to_pylist())
    finally:
        source.close()
    return identifiers


def _iter_profile_file(path: Path) -> Iterator[ProfileRecord]:
    source = pa.memory_map(str(path), "r")
    try:
        parquet = pq.ParquetFile(source)
        columns = [
            "transcript_id",
            "panel",
            "run_identifier",
            "transcript_length",
            "L_t",
        ]
        for batch in parquet.iter_batches(batch_size=1, columns=columns):
            values32 = batch.column(4)[0].values.to_numpy(zero_copy_only=False)
            yield ProfileRecord(
                transcript_id=str(batch.column(0)[0].as_py()),
                panel=str(batch.column(1)[0].as_py()),
                run_identifier=str(batch.column(2)[0].as_py()),
                length=int(batch.column(3)[0].as_py()),
                values=np.asarray(values32, dtype=np.float64),
            )
    finally:
        source.close()


class AlignedProfileSource:
    """Four-way one-row streaming alignment with an on-disk SQLite fallback."""

    def __init__(
        self,
        panel_paths: Mapping[str, Path],
        expected_ids: Sequence[str],
        work_dir: Path,
    ) -> None:
        self.panel_paths = dict(panel_paths)
        self.expected_ids = list(expected_ids)
        self.work_dir = work_dir
        self.id_orders = {
            panel: _read_id_stream(path) for panel, path in self.panel_paths.items()
        }
        expected_set = set(self.expected_ids)
        if len(expected_set) != len(self.expected_ids):
            raise ValueError("common_test_ids contains duplicates.")
        for panel, identifiers in self.id_orders.items():
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{panel} compact prediction contains duplicate IDs.")
            observed = set(identifiers)
            if not expected_set <= observed:
                raise ValueError(
                    f"{panel} transcript identity mismatch: "
                    f"missing={sorted(expected_set-observed)[:10]}, "
                    f"file_rows={len(identifiers)}."
                )
        self.direct = all(
            identifiers[: len(self.expected_ids)] == self.expected_ids
            for identifiers in self.id_orders.values()
        )
        self.sqlite_path = self.work_dir / "disk_backed_profile_alignment.sqlite"
        _json_dump(
            self.work_dir / "stream_alignment.json",
            {
                "strategy": (
                    "four_way_pyarrow_one_row_stream"
                    if self.direct
                    else "sqlite_disk_backed_join"
                ),
                "prediction_arrays_indexed_in_memory": False,
                "small_identifier_lists_held_in_memory": True,
                "identical_order_to_common_test_manifest": {
                    panel: order == self.expected_ids
                    for panel, order in self.id_orders.items()
                },
            },
        )
        if not self.direct:
            self._materialize_sqlite()

    def _materialize_sqlite(self) -> None:
        if self.sqlite_path.exists():
            self.sqlite_path.unlink()
        connection = sqlite3.connect(self.sqlite_path)
        try:
            connection.execute(
                "CREATE TABLE profiles (panel TEXT, transcript_id TEXT, "
                "run_identifier TEXT, length INTEGER, values BLOB, "
                "PRIMARY KEY(panel, transcript_id))"
            )
            for panel, path in self.panel_paths.items():
                for row in _iter_profile_file(path):
                    connection.execute(
                        "INSERT INTO profiles VALUES (?, ?, ?, ?, ?)",
                        (
                            panel,
                            row.transcript_id,
                            row.run_identifier,
                            row.length,
                            sqlite3.Binary(
                                np.asarray(row.values, dtype=np.float32).tobytes()
                            ),
                        ),
                    )
                connection.commit()
        finally:
            connection.close()

    def __iter__(self) -> Iterator[tuple[str, dict[str, ProfileRecord]]]:
        if self.direct:
            iterators = {
                panel: iter(_iter_profile_file(path))
                for panel, path in self.panel_paths.items()
            }
            for expected_id in self.expected_ids:
                current: dict[str, ProfileRecord] = {}
                for panel in PANEL_NAMES:
                    try:
                        record = next(iterators[panel])
                    except StopIteration as exc:
                        raise ValueError(f"{panel} ended before {expected_id}.") from exc
                    if record.transcript_id != expected_id:
                        raise ValueError(
                            f"Streaming alignment error: {panel} yielded "
                            f"{record.transcript_id}, expected {expected_id}."
                        )
                    current[panel] = record
                yield expected_id, current
            # Smoke mode intentionally stops after a verified manifest prefix.
            # Full mode has selected/file lengths equal and therefore checks EOF.
            if all(
                len(order) == len(self.expected_ids)
                for order in self.id_orders.values()
            ):
                for panel, iterator in iterators.items():
                    try:
                        extra = next(iterator)
                    except StopIteration:
                        continue
                    raise ValueError(
                        f"{panel} contains an unexpected trailing row "
                        f"{extra.transcript_id}."
                    )
            return

        connection = sqlite3.connect(self.sqlite_path)
        try:
            for transcript_id in self.expected_ids:
                current = {}
                for panel in PANEL_NAMES:
                    row = connection.execute(
                        "SELECT run_identifier, length, values FROM profiles "
                        "WHERE panel=? AND transcript_id=?",
                        (panel, transcript_id),
                    ).fetchone()
                    if row is None:
                        raise ValueError(f"Disk-backed join missed {panel}/{transcript_id}.")
                    run_identifier, length, blob = row
                    values = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
                    current[panel] = ProfileRecord(
                        transcript_id=transcript_id,
                        panel=panel,
                        run_identifier=str(run_identifier),
                        length=int(length),
                        values=values,
                    )
                yield transcript_id, current
        finally:
            connection.close()


def _validate_current_profiles(
    transcript_id: str,
    records: Mapping[str, ProfileRecord],
    metadata: SequenceMetadata,
    tolerance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel in PANEL_NAMES:
        record = records[panel]
        values = record.values
        if record.transcript_id != transcript_id or record.panel != panel:
            raise ValueError(
                f"Compact identity fields disagree for {panel}/{transcript_id}."
            )
        if values.ndim != 1 or record.length <= 0:
            raise ValueError(f"Invalid one-dimensional profile for {panel}/{transcript_id}.")
        if values.size != record.length or record.length != metadata.length:
            raise ValueError(
                f"Length/coordinate disagreement for {panel}/{transcript_id}: "
                f"array={values.size}, export={record.length}, "
                f"codons={metadata.length}."
            )
        finite = bool(np.isfinite(values).all())
        positive = bool(np.all(values > 0.0))
        if not finite or not positive:
            raise ValueError(
                f"Non-finite or non-positive L_t for {panel}/{transcript_id}."
            )
        mean = float(values.mean())
        deviation = abs(mean - 1.0)
        if deviation > tolerance:
            raise ValueError(
                f"Mean-one check failed for {panel}/{transcript_id}: "
                f"mean={mean:.9g}, deviation={deviation:.3g}, tolerance={tolerance}. "
                "No analysis renormalization was applied."
            )
        rows.append(
            {
                "transcript_id": transcript_id,
                "gene_id": metadata.gene_id,
                "panel": panel,
                "run_identifier": record.run_identifier,
                "transcript_length": record.length,
                "profile_array_length": int(values.size),
                "implicit_valid_mask_count": int(values.size),
                "first_modeled_coordinate": 0,
                "last_modeled_coordinate": int(values.size - 1),
                "first_codon": metadata.first_codon,
                "terminal_codon": metadata.terminal_codon,
                "terminal_stop_included": metadata.terminal_codon in STOP_CODONS,
                "codon_sequence_sha256": metadata.codon_sha256,
                "all_finite": finite,
                "all_positive": positive,
                "L_mean": mean,
                "absolute_mean_one_deviation": deviation,
            }
        )
    lengths = {record.length for record in records.values()}
    if len(lengths) != 1:
        raise ValueError(f"Panel lengths disagree for {transcript_id}: {lengths}.")
    return rows


def _pair_label(panel_a: str, panel_b: str) -> str:
    return f"{panel_a}--{panel_b}"


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right), dtype=np.float64)))


def _metric_result(
    left: np.ndarray,
    right: np.ndarray,
    keep: np.ndarray,
    *,
    minimum_positions: int,
) -> MetricResult:
    n = int(keep.sum())
    if n < minimum_positions:
        return MetricResult(
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            n,
            False,
            "insufficient_retained_positions",
        )
    x = left[keep]
    y = right[keep]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return MetricResult(
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            n,
            False,
            "non_finite_retained_values",
        )
    variance_a = float(np.var(x))
    variance_b = float(np.var(y))
    rmse = _rmse(x, y)
    if variance_a <= 0.0 and variance_b <= 0.0:
        reason = "constant_profiles_both"
    elif variance_a <= 0.0:
        reason = "constant_profile_a"
    elif variance_b <= 0.0:
        reason = "constant_profile_b"
    else:
        reason = "ok"
    if reason != "ok":
        return MetricResult(
            math.nan,
            math.nan,
            rmse,
            variance_a,
            variance_b,
            n,
            False,
            reason,
        )
    return MetricResult(
        _pearson(x, y),
        _spearman(x, y),
        rmse,
        variance_a,
        variance_b,
        n,
        True,
        "ok",
    )


def _all_pair_results(
    profiles: Mapping[str, np.ndarray],
    keep: np.ndarray,
    *,
    minimum_positions: int,
) -> dict[str, MetricResult]:
    return {
        _pair_label(panel_a, panel_b): _metric_result(
            profiles[panel_a],
            profiles[panel_b],
            keep,
            minimum_positions=minimum_positions,
        )
        for panel_a, panel_b in PANEL_PAIRS
    }


def _top_mask(
    values: np.ndarray,
    fraction: float,
    domain: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float]:
    if domain is None:
        domain = np.ones(values.size, dtype=bool)
    domain_values = values[domain]
    if domain_values.size == 0:
        raise ValueError("Cannot define a top-position set on an empty domain.")
    target = int(math.ceil(float(fraction) * int(domain_values.size)))
    cutoff = float(np.partition(domain_values, domain_values.size - target)[-target])
    selected = domain & (values >= cutoff)
    return selected, target, cutoff


def _stable_rng(seed: int, transcript_id: str, label: str) -> np.random.Generator:
    payload = f"{seed}|{transcript_id}|{label}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)


def _fast_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator <= 0.0:
        return math.nan
    return float(np.dot(left_centered, right_centered) / denominator)


def _random_removal_summaries(
    *,
    transcript_id: str,
    profiles: Mapping[str, np.ndarray],
    remove_count: int,
    draws: int,
    random_seed: int,
    label: str,
    minimum_positions: int = 50,
) -> dict[str, dict[str, Any]]:
    n = next(iter(profiles.values())).size
    results = np.full((draws, len(PANEL_PAIRS), 3), np.nan, dtype=np.float64)
    if n - remove_count < minimum_positions:
        return {
            _pair_label(*pair): {
                "valid_draws": 0,
                "reason": "insufficient_retained_positions",
            }
            for pair in PANEL_PAIRS
        }
    rng = _stable_rng(random_seed, transcript_id, label)
    all_indices = np.arange(n, dtype=np.int32)
    for draw_index in range(draws):
        removed = rng.choice(all_indices, size=remove_count, replace=False)
        keep = np.ones(n, dtype=bool)
        keep[removed] = False
        selected = {panel: profiles[panel][keep] for panel in PANEL_NAMES}
        ranks = {
            panel: rankdata(selected[panel], method="average")
            for panel in PANEL_NAMES
        }
        for pair_index, (panel_a, panel_b) in enumerate(PANEL_PAIRS):
            left = selected[panel_a]
            right = selected[panel_b]
            results[draw_index, pair_index, 0] = _fast_pearson(left, right)
            results[draw_index, pair_index, 1] = _fast_pearson(
                ranks[panel_a], ranks[panel_b]
            )
            results[draw_index, pair_index, 2] = _rmse(left, right)
        del removed, keep, selected, ranks

    summaries: dict[str, dict[str, Any]] = {}
    for pair_index, pair in enumerate(PANEL_PAIRS):
        values = results[:, pair_index, :]
        complete = np.isfinite(values).all(axis=1)
        valid = values[complete]
        pair_name = _pair_label(*pair)
        if valid.size == 0:
            summaries[pair_name] = {
                "valid_draws": 0,
                "reason": "no_valid_random_draws",
            }
            continue
        entry: dict[str, Any] = {
            "valid_draws": int(valid.shape[0]),
            "reason": "ok" if valid.shape[0] == draws else "some_random_draws_invalid",
        }
        for metric_index, metric in enumerate(("PCC", "Spearman", "RMSE")):
            vector = valid[:, metric_index]
            quantiles = np.quantile(vector, [0.025, 0.5, 0.975])
            entry[metric] = float(vector.mean())
            entry[f"{metric}_sd"] = (
                float(vector.std(ddof=1)) if vector.size > 1 else 0.0
            )
            entry[f"{metric}_q025"] = float(quantiles[0])
            entry[f"{metric}_q50"] = float(quantiles[1])
            entry[f"{metric}_q975"] = float(quantiles[2])
        summaries[pair_name] = entry
    del results
    return summaries


def _base_metric_row(
    *,
    transcript_id: str,
    gene_id: str,
    pair: tuple[str, str],
    family: str,
    condition: str,
    n: int,
    removed: int,
    result: MetricResult,
    full_pcc: float,
    boundary_fixed: bool,
    combined_fixed: bool,
) -> dict[str, Any]:
    panel_a, panel_b = pair
    pcc = result.pcc if math.isfinite(result.pcc) else None
    reference = full_pcc if math.isfinite(full_pcc) else None
    delta = (
        float(result.pcc - full_pcc)
        if math.isfinite(result.pcc) and math.isfinite(full_pcc)
        else None
    )
    return {
        "transcript_id": transcript_id,
        "gene_id": gene_id,
        "panel_a": panel_a,
        "panel_b": panel_b,
        "panel_pair": _pair_label(panel_a, panel_b),
        "analysis_family": family,
        "condition": condition,
        "transcript_length": n,
        "n_removed": int(removed),
        "n_retained": result.n,
        "removed_fraction": float(removed / n),
        "PCC": pcc,
        "Spearman": result.spearman if math.isfinite(result.spearman) else None,
        "RMSE": result.rmse if math.isfinite(result.rmse) else None,
        "retained_variance_a": (
            result.variance_a if math.isfinite(result.variance_a) else None
        ),
        "retained_variance_b": (
            result.variance_b if math.isfinite(result.variance_b) else None
        ),
        "reference_full_PCC": reference,
        "paired_PCC_change": delta,
        "valid": result.valid,
        "reason_code": result.reason,
        "random_draws_requested": 0,
        "random_draws_valid": 0,
        "in_boundary_fixed_cohort": boundary_fixed,
        "in_combined_fixed_cohort": combined_fixed,
    }


def _random_metric_row(
    *,
    transcript_id: str,
    gene_id: str,
    pair: tuple[str, str],
    condition: str,
    n: int,
    removed: int,
    summary: Mapping[str, Any],
    full_pcc: float,
    draws: int,
) -> dict[str, Any]:
    panel_a, panel_b = pair
    valid_draws = int(summary.get("valid_draws", 0))
    valid = valid_draws > 0
    pcc = float(summary["PCC"]) if valid else None
    return {
        "transcript_id": transcript_id,
        "gene_id": gene_id,
        "panel_a": panel_a,
        "panel_b": panel_b,
        "panel_pair": _pair_label(panel_a, panel_b),
        "analysis_family": "matched_random_removal_control",
        "condition": condition,
        "transcript_length": n,
        "n_removed": removed,
        "n_retained": n - removed,
        "removed_fraction": float(removed / n),
        "PCC": pcc,
        "Spearman": float(summary["Spearman"]) if valid else None,
        "RMSE": float(summary["RMSE"]) if valid else None,
        "reference_full_PCC": full_pcc if math.isfinite(full_pcc) else None,
        "paired_PCC_change": (
            pcc - full_pcc if valid and math.isfinite(full_pcc) else None
        ),
        "valid": valid,
        "reason_code": str(summary.get("reason", "no_valid_random_draws")),
        "random_draws_requested": draws,
        "random_draws_valid": valid_draws,
        "PCC_sd_across_random_draws": summary.get("PCC_sd"),
        "PCC_q025_across_random_draws": summary.get("PCC_q025"),
        "PCC_q50_across_random_draws": summary.get("PCC_q50"),
        "PCC_q975_across_random_draws": summary.get("PCC_q975"),
        "Spearman_sd_across_random_draws": summary.get("Spearman_sd"),
        "Spearman_q025_across_random_draws": summary.get("Spearman_q025"),
        "Spearman_q50_across_random_draws": summary.get("Spearman_q50"),
        "Spearman_q975_across_random_draws": summary.get("Spearman_q975"),
        "RMSE_sd_across_random_draws": summary.get("RMSE_sd"),
        "RMSE_q025_across_random_draws": summary.get("RMSE_q025"),
        "RMSE_q50_across_random_draws": summary.get("RMSE_q50"),
        "RMSE_q975_across_random_draws": summary.get("RMSE_q975"),
        "in_boundary_fixed_cohort": False,
        "in_combined_fixed_cohort": False,
    }


def _run_reference_precheck(
    *,
    source: AlignedProfileSource,
    metadata: Mapping[str, SequenceMetadata],
    output_dir: Path,
    mean_one_tolerance: float,
    memory: MemoryGuard,
) -> tuple[Path, Path]:
    reference_path = output_dir / "reference_full_profile_metrics.parquet"
    validation_path = output_dir / "streamed_input_validation.parquet"
    reference_writer = BufferedParquetWriter(reference_path, REFERENCE_SCHEMA)
    validation_writer = BufferedParquetWriter(validation_path, VALIDATION_SCHEMA)
    accepted = False
    try:
        for processed, (transcript_id, records) in enumerate(source, start=1):
            validation_writer.extend(
                _validate_current_profiles(
                    transcript_id,
                    records,
                    metadata[transcript_id],
                    mean_one_tolerance,
                )
            )
            profiles = {panel: records[panel].values for panel in PANEL_NAMES}
            keep = np.ones(records[PANEL_NAMES[0]].length, dtype=bool)
            for panel_a, panel_b in PANEL_PAIRS:
                result = _metric_result(
                    profiles[panel_a], profiles[panel_b], keep, minimum_positions=2
                )
                reference_writer.append(
                    {
                        "transcript_id": transcript_id,
                        "gene_id": metadata[transcript_id].gene_id,
                        "panel_a": panel_a,
                        "panel_b": panel_b,
                        "panel_pair": _pair_label(panel_a, panel_b),
                        "transcript_length": keep.size,
                        "PCC": result.pcc if math.isfinite(result.pcc) else None,
                        "Spearman": (
                            result.spearman if math.isfinite(result.spearman) else None
                        ),
                        "RMSE": result.rmse if math.isfinite(result.rmse) else None,
                        "valid": result.valid,
                        "reason_code": result.reason,
                    }
                )
            memory.check("reference_precheck", processed)
            del profiles, keep, records
        accepted = True
    finally:
        reference_writer.close(accept=accepted)
        validation_writer.close(accept=accepted)
    return reference_path, validation_path


def _validate_reference_summary(
    reference_path: Path,
    transcript_count: int,
    *,
    is_smoke: bool,
) -> pd.DataFrame:
    frame = pd.read_parquet(reference_path)
    expected_rows = transcript_count * len(PANEL_PAIRS)
    if len(frame) != expected_rows:
        raise ValueError(
            f"Reference precheck row count {len(frame)} != {expected_rows}."
        )
    if not bool(frame["valid"].all()) or frame["PCC"].isna().any():
        invalid = frame.loc[~frame["valid"] | frame["PCC"].isna()]
        raise ValueError(
            f"Reference precheck contains {len(invalid)} invalid rows; refusing "
            "new positional analyses."
        )
    rows: list[dict[str, Any]] = []
    for pair, group in frame.groupby("panel_pair", sort=True):
        rows.append(
            {
                "panel_pair": pair,
                "n": len(group),
                "median_PCC": float(group["PCC"].median()),
                "reported_target": EXPECTED_PAIR_MEDIANS.get(str(pair)),
                "matches_reported_rounding": (
                    None
                    if is_smoke
                    else round(float(group["PCC"].median()), 3)
                    == EXPECTED_PAIR_MEDIANS[str(pair)]
                ),
            }
        )
    pooled = float(frame["PCC"].median())
    rows.append(
        {
            "panel_pair": "pooled_all_pairs",
            "n": len(frame),
            "median_PCC": pooled,
            "reported_target": EXPECTED_POOLED_MEDIAN,
            "matches_reported_rounding": (
                None
                if is_smoke
                else round(pooled, 3) == EXPECTED_POOLED_MEDIAN
            ),
        }
    )
    summary = pd.DataFrame(rows)
    if not is_smoke and not bool(summary["matches_reported_rounding"].all()):
        raise RuntimeError(
            "Full-profile reference medians did not reproduce the accepted "
            "analysis to three decimals. Refusing to interpret robustness "
            f"results. Observed:\n{summary.to_string(index=False)}"
        )
    return summary


def _run_streaming_analysis(
    *,
    source: AlignedProfileSource,
    metadata: Mapping[str, SequenceMetadata],
    output_dir: Path,
    random_draws: int,
    random_seed: int,
    memory: MemoryGuard,
) -> dict[str, int]:
    metrics_path = output_dir / "per_transcript_pair_scalar_metrics.parquet"
    masks_path = output_dir / "mask_scalar_definitions.parquet"
    jaccard_path = output_dir / "high_profile_position_overlap.parquet"
    cohort_path = output_dir / "boundary_cohort_flags.parquet"
    metric_writer = BufferedParquetWriter(metrics_path, METRIC_SCHEMA)
    mask_writer = BufferedParquetWriter(masks_path, MASK_SCHEMA)
    jaccard_writer = BufferedParquetWriter(jaccard_path, JACCARD_SCHEMA)
    cohort_writer = BufferedParquetWriter(cohort_path, COHORT_SCHEMA)
    accepted = False
    processed = 0
    try:
        for processed, (transcript_id, records) in enumerate(source, start=1):
            meta = metadata[transcript_id]
            profiles = {panel: records[panel].values for panel in PANEL_NAMES}
            n = meta.length
            full_keep = np.ones(n, dtype=bool)
            full_results = _all_pair_results(
                profiles, full_keep, minimum_positions=2
            )

            top_data: dict[float, dict[str, Any]] = {}
            for fraction in (0.01, 0.05):
                masks: dict[str, np.ndarray] = {}
                cutoffs: dict[str, float] = {}
                targets: dict[str, int] = {}
                for panel in PANEL_NAMES:
                    masks[panel], targets[panel], cutoffs[panel] = _top_mask(
                        profiles[panel], fraction
                    )
                union = np.logical_or.reduce([masks[p] for p in PANEL_NAMES])
                keep = ~union
                top_data[fraction] = {
                    "masks": masks,
                    "cutoffs": cutoffs,
                    "target": next(iter(targets.values())),
                    "union": union,
                    "keep": keep,
                    "results": _all_pair_results(
                        profiles, keep, minimum_positions=50
                    ),
                }
                union_count = int(union.sum())
                for panel in PANEL_NAMES:
                    selected_count = int(masks[panel].sum())
                    mask_writer.append(
                        {
                            "transcript_id": transcript_id,
                            "gene_id": meta.gene_id,
                            "mask_family": f"top_{int(fraction*100):02d}pct_union",
                            "target_fraction_per_panel": fraction,
                            "selection_domain_start": 0,
                            "selection_domain_stop_exclusive": n,
                            "selection_domain_size": n,
                            "ceil_target_count": targets[panel],
                            "panel": panel,
                            "cutoff": cutoffs[panel],
                            "panel_selected_count_with_ties": selected_count,
                            "panel_actual_selected_fraction": selected_count / n,
                            "four_panel_union_count": union_count,
                            "four_panel_union_fraction_of_domain": union_count / n,
                            "retained_count_after_union": int(keep.sum()),
                            "coordinate_convention": (
                                "zero_based_modeled_codon_index; index n-1 is "
                                "the encoded terminal stop codon"
                            ),
                        }
                    )

            # Pairwise overlap of independently selected top-5% sets.
            top5_masks = top_data[0.05]["masks"]
            target5 = int(top_data[0.05]["target"])
            for pair in PANEL_PAIRS:
                panel_a, panel_b = pair
                mask_a = top5_masks[panel_a]
                mask_b = top5_masks[panel_b]
                intersection = int(np.logical_and(mask_a, mask_b).sum())
                union_count = int(np.logical_or(mask_a, mask_b).sum())
                jaccard = intersection / union_count if union_count else math.nan
                full = full_results[_pair_label(*pair)]
                jaccard_writer.append(
                    {
                        "transcript_id": transcript_id,
                        "gene_id": meta.gene_id,
                        "panel_a": panel_a,
                        "panel_b": panel_b,
                        "panel_pair": _pair_label(*pair),
                        "transcript_length": n,
                        "ceil_target_count": target5,
                        "selected_count_a_with_ties": int(mask_a.sum()),
                        "selected_count_b_with_ties": int(mask_b.sum()),
                        "actual_selected_fraction_a": float(mask_a.mean()),
                        "actual_selected_fraction_b": float(mask_b.mean()),
                        "intersection_count": intersection,
                        "union_count": union_count,
                        "jaccard": jaccard if math.isfinite(jaccard) else None,
                        "full_profile_PCC": (
                            full.pcc if math.isfinite(full.pcc) else None
                        ),
                        "valid": math.isfinite(jaccard) and full.valid,
                        "reason_code": (
                            "ok" if union_count and full.valid else "undefined_union_or_PCC"
                        ),
                    }
                )

            # Boundary windows.  The full fixed cohort is determined only after
            # all six pairs in all four windows have been checked.
            boundary_results: dict[int, dict[str, MetricResult]] = {}
            boundary_keeps: dict[int, np.ndarray] = {}
            for k in (0, 5, 20, 50):
                keep = np.zeros(n, dtype=bool)
                if n > 2 * k:
                    keep[k : n - k if k else n] = True
                boundary_keeps[k] = keep
                boundary_results[k] = _all_pair_results(
                    profiles, keep, minimum_positions=50
                )
            length_ok = n >= 150
            boundary_metrics_ok = all(
                result.valid
                for results in boundary_results.values()
                for result in results.values()
            )
            boundary_fixed = length_ok and boundary_metrics_ok

            interior20 = boundary_keeps[20]
            combined_masks: dict[str, np.ndarray] = {}
            combined_cutoffs: dict[str, float] = {}
            combined_targets: dict[str, int] = {}
            for panel in PANEL_NAMES:
                (
                    combined_masks[panel],
                    combined_targets[panel],
                    combined_cutoffs[panel],
                ) = _top_mask(profiles[panel], 0.01, interior20)
            combined_union = np.logical_or.reduce(
                [combined_masks[p] for p in PANEL_NAMES]
            )
            combined_keep = interior20 & ~combined_union
            combined_results = _all_pair_results(
                profiles, combined_keep, minimum_positions=50
            )
            combined_metrics_ok = all(r.valid for r in combined_results.values())
            combined_fixed = boundary_fixed and combined_metrics_ok
            combined_union_count = int((combined_union & interior20).sum())
            domain_size = int(interior20.sum())
            for panel in PANEL_NAMES:
                selected = int(combined_masks[panel].sum())
                mask_writer.append(
                    {
                        "transcript_id": transcript_id,
                        "gene_id": meta.gene_id,
                        "mask_family": "interior20_then_top_01pct_union",
                        "target_fraction_per_panel": 0.01,
                        "selection_domain_start": 20,
                        "selection_domain_stop_exclusive": max(20, n - 20),
                        "selection_domain_size": domain_size,
                        "ceil_target_count": combined_targets[panel],
                        "panel": panel,
                        "cutoff": combined_cutoffs[panel],
                        "panel_selected_count_with_ties": selected,
                        "panel_actual_selected_fraction": (
                            selected / domain_size if domain_size else None
                        ),
                        "four_panel_union_count": combined_union_count,
                        "four_panel_union_fraction_of_domain": (
                            combined_union_count / domain_size if domain_size else None
                        ),
                        "retained_count_after_union": int(combined_keep.sum()),
                        "coordinate_convention": (
                            "zero_based_modeled_codon_index; remove indices "
                            "[0,20) and [n-20,n), including terminal stop at n-1"
                        ),
                    }
                )

            exclusion = "ok"
            if not length_ok:
                exclusion = "transcript_length_below_150"
            elif not boundary_metrics_ok:
                exclusion = "undefined_metric_in_boundary_window"
            elif not combined_metrics_ok:
                exclusion = "undefined_metric_in_combined_condition"
            cohort_writer.append(
                {
                    "transcript_id": transcript_id,
                    "gene_id": meta.gene_id,
                    "transcript_length": n,
                    "n_at_least_150": length_ok,
                    "valid_all_six_pairs_at_k0_k5_k20_k50": boundary_metrics_ok,
                    "boundary_fixed_cohort": boundary_fixed,
                    "valid_all_six_pairs_combined_condition": combined_metrics_ok,
                    "combined_fixed_cohort": combined_fixed,
                    "exclusion_reason": exclusion,
                }
            )

            # Full-profile rows are kept as a distinct all-test-set reference.
            for pair in PANEL_PAIRS:
                label = _pair_label(*pair)
                result = full_results[label]
                metric_writer.append(
                    _base_metric_row(
                        transcript_id=transcript_id,
                        gene_id=meta.gene_id,
                        pair=pair,
                        family="full_test_reference",
                        condition="full_profile_all_test_transcripts",
                        n=n,
                        removed=0,
                        result=result,
                        full_pcc=result.pcc,
                        boundary_fixed=boundary_fixed,
                        combined_fixed=combined_fixed,
                    )
                )

            # Extreme masks and independently generated matched-count random
            # controls.  Only 100x6x3 scalar values for this transcript exist.
            for fraction in (0.01, 0.05):
                percent = int(fraction * 100)
                data = top_data[fraction]
                removed = int(data["union"].sum())
                condition = f"top_{percent:02d}pct_four_panel_union_removed"
                random_condition = f"matched_random_to_top_{percent:02d}pct_union"
                random_summary = _random_removal_summaries(
                    transcript_id=transcript_id,
                    profiles=profiles,
                    remove_count=removed,
                    draws=random_draws,
                    random_seed=random_seed,
                    label=f"top_{percent:02d}pct_union",
                )
                for pair in PANEL_PAIRS:
                    label = _pair_label(*pair)
                    metric_writer.append(
                        _base_metric_row(
                            transcript_id=transcript_id,
                            gene_id=meta.gene_id,
                            pair=pair,
                            family="extreme_position_sensitivity",
                            condition=condition,
                            n=n,
                            removed=removed,
                            result=data["results"][label],
                            full_pcc=full_results[label].pcc,
                            boundary_fixed=boundary_fixed,
                            combined_fixed=combined_fixed,
                        )
                    )
                    metric_writer.append(
                        _random_metric_row(
                            transcript_id=transcript_id,
                            gene_id=meta.gene_id,
                            pair=pair,
                            condition=random_condition,
                            n=n,
                            removed=removed,
                            summary=random_summary[label],
                            full_pcc=full_results[label].pcc,
                            draws=random_draws,
                        )
                    )
                del random_summary

            for k in (0, 5, 20, 50):
                removed = n - int(boundary_keeps[k].sum())
                for pair in PANEL_PAIRS:
                    label = _pair_label(*pair)
                    metric_writer.append(
                        _base_metric_row(
                            transcript_id=transcript_id,
                            gene_id=meta.gene_id,
                            pair=pair,
                            family="boundary_sensitivity",
                            condition=f"boundary_k{k:03d}",
                            n=n,
                            removed=removed,
                            result=boundary_results[k][label],
                            full_pcc=full_results[label].pcc,
                            boundary_fixed=boundary_fixed,
                            combined_fixed=combined_fixed,
                        )
                    )
            for pair in PANEL_PAIRS:
                label = _pair_label(*pair)
                metric_writer.append(
                    _base_metric_row(
                        transcript_id=transcript_id,
                        gene_id=meta.gene_id,
                        pair=pair,
                        family="boundary_and_extreme_combined",
                        condition="boundary_k020_then_top_01pct_union_removed",
                        n=n,
                        removed=n - int(combined_keep.sum()),
                        result=combined_results[label],
                        full_pcc=full_results[label].pcc,
                        boundary_fixed=boundary_fixed,
                        combined_fixed=combined_fixed,
                    )
                )

            memory.check("positional_analysis", processed)
            del (
                records,
                profiles,
                full_keep,
                full_results,
                top_data,
                top5_masks,
                boundary_results,
                boundary_keeps,
                combined_masks,
                combined_union,
                combined_keep,
                combined_results,
            )
        accepted = True
    finally:
        metric_writer.close(accept=accepted)
        mask_writer.close(accept=accepted)
        jaccard_writer.close(accept=accepted)
        cohort_writer.close(accept=accepted)
    return {
        "transcripts": processed,
        "metric_rows": metric_writer.rows_written,
        "mask_rows": mask_writer.rows_written,
        "jaccard_rows": jaccard_writer.rows_written,
        "cohort_rows": cohort_writer.rows_written,
    }


def _numeric_summary(values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(np.float64)
    if clean.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "p75": math.nan,
            "p95": math.nan,
        }
    q = np.quantile(clean, [0.05, 0.25, 0.75, 0.95])
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(np.median(clean)),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "p75": float(q[2]),
        "p95": float(q[3]),
    }


def _mark_summary_scope(metrics: pd.DataFrame) -> pd.Series:
    scope = metrics["valid"].fillna(False).astype(bool)
    boundary = metrics["condition"].isin(
        ["boundary_k000", "boundary_k005", "boundary_k020", "boundary_k050"]
    )
    scope &= (~boundary) | metrics["in_boundary_fixed_cohort"].fillna(False)
    combined = metrics["condition"].eq(
        "boundary_k020_then_top_01pct_union_removed"
    )
    scope &= (~combined) | metrics["in_combined_fixed_cohort"].fillna(False)
    return scope


def _scoped_metrics_with_combined_aliases(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return scalar rows in scope plus the exact combined matched cohort.

    The combined comparison must put the original, interior-20, and
    interior-20-plus-top-1%-union values on the same transcript/pair rows.
    Aliases are created only in this compact scalar summary layer; the
    incrementally written primary metric artifact remains nonduplicated.
    """

    scoped = metrics.loc[_mark_summary_scope(metrics)].copy()
    aliases = {
        "full_profile_all_test_transcripts": "combined_matched_full_profile",
        "boundary_k020": "combined_matched_interior20",
        "boundary_k020_then_top_01pct_union_removed": (
            "combined_matched_interior20_top1_union"
        ),
    }
    matched = metrics.loc[
        metrics["condition"].isin(aliases)
        & metrics["in_combined_fixed_cohort"].fillna(False)
        & metrics["valid"].fillna(False)
    ].copy()
    matched["condition"] = matched["condition"].map(aliases)
    matched["analysis_family"] = "combined_matched_comparison"
    return pd.concat([scoped, matched], ignore_index=True)


def _make_compact_summaries(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # This is the first pandas load of the main result and it contains scalar
    # rows only--never profile arrays or codon-level values.
    metrics = pd.read_parquet(output_dir / "per_transcript_pair_scalar_metrics.parquet")
    metrics["in_summary_scope"] = _mark_summary_scope(metrics)
    scoped = _scoped_metrics_with_combined_aliases(metrics)

    pair_rows: list[dict[str, Any]] = []
    for (family, condition, pair), group in scoped.groupby(
        ["analysis_family", "condition", "panel_pair"], sort=True
    ):
        row: dict[str, Any] = {
            "analysis_family": family,
            "condition": condition,
            "panel_pair": pair,
            "cohort_scope": (
                "n>=150_and_all_boundary_metrics_valid"
                if str(condition).startswith("boundary_k")
                and "then" not in str(condition)
                else (
                    "boundary_fixed_plus_combined_valid"
                    if "then_top" in str(condition)
                    else "condition_eligible_pair_rows"
                )
            ),
        }
        for metric in ("PCC", "Spearman", "RMSE"):
            for name, value in _numeric_summary(group[metric]).items():
                row[f"{metric}_{name}"] = value
        delta = _numeric_summary(group["paired_PCC_change"])
        row.update({f"paired_PCC_change_{k}": v for k, v in delta.items()})
        row["median_removed_fraction"] = float(group["removed_fraction"].median())
        row["median_n_retained"] = float(group["n_retained"].median())
        pair_rows.append(row)
    pair_summary = pd.DataFrame(pair_rows)
    pair_summary.to_csv(output_dir / "pairwise_condition_summary.csv", index=False)

    transcript_rows: list[dict[str, Any]] = []
    for (transcript_id, gene_id, family, condition), group in scoped.groupby(
        ["transcript_id", "gene_id", "analysis_family", "condition"], sort=True
    ):
        valid = group.loc[group["valid"]]
        count = int(valid["panel_pair"].nunique())
        transcript_rows.append(
            {
                "transcript_id": transcript_id,
                "gene_id": gene_id,
                "analysis_family": family,
                "condition": condition,
                "valid_pair_count": count,
                "all_six_pairs_valid": count == 6,
                "median_PCC_across_six_pairs": (
                    float(valid["PCC"].median()) if count == 6 else math.nan
                ),
                "median_Spearman_across_six_pairs": (
                    float(valid["Spearman"].median()) if count == 6 else math.nan
                ),
                "median_RMSE_across_six_pairs": (
                    float(valid["RMSE"].median()) if count == 6 else math.nan
                ),
                "median_paired_PCC_change_across_six_pairs": (
                    float(valid["paired_PCC_change"].median())
                    if count == 6
                    else math.nan
                ),
                "removed_fraction": float(group["removed_fraction"].iloc[0]),
                "n_retained": int(group["n_retained"].iloc[0]),
            }
        )
    transcript_summary = pd.DataFrame(transcript_rows)
    transcript_summary.to_parquet(
        output_dir / "transcript_condition_summary.parquet",
        engine="pyarrow",
        index=False,
        row_group_size=1000,
    )
    combined_source = transcript_summary.loc[
        transcript_summary["analysis_family"].eq("combined_matched_comparison")
    ]
    combined_source.to_csv(
        output_dir / "combined_condition_matched_transcript_summary.csv",
        index=False,
    )
    pair_summary.loc[
        pair_summary["analysis_family"].eq("combined_matched_comparison")
    ].to_csv(
        output_dir / "combined_condition_matched_pairwise_summary.csv",
        index=False,
    )

    aggregate_rows: list[dict[str, Any]] = []
    complete = transcript_summary.loc[transcript_summary["all_six_pairs_valid"]]
    for (family, condition), group in complete.groupby(
        ["analysis_family", "condition"], sort=True
    ):
        row = {"analysis_family": family, "condition": condition}
        for metric in (
            "median_PCC_across_six_pairs",
            "median_Spearman_across_six_pairs",
            "median_RMSE_across_six_pairs",
            "median_paired_PCC_change_across_six_pairs",
        ):
            for name, value in _numeric_summary(group[metric]).items():
                row[f"{metric}_{name}"] = value
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(
        output_dir / "transcript_level_condition_summary.csv", index=False
    )
    return metrics, transcript_summary, pair_summary


def _bootstrap_intervals(
    *,
    metrics: pd.DataFrame,
    transcript_summary: pd.DataFrame,
    metadata: Mapping[str, SequenceMetadata],
    output_dir: Path,
    resamples: int,
    seed: int,
    memory: MemoryGuard,
) -> pd.DataFrame:
    scoped = _scoped_metrics_with_combined_aliases(metrics)
    scoped = scoped.loc[scoped["valid"] & scoped["PCC"].notna()]
    pair_pivot = scoped.pivot_table(
        index="transcript_id",
        columns=["condition", "panel_pair"],
        values="PCC",
        aggfunc="first",
    )
    pair_pivot.columns = [
        f"pair|||{condition}|||{pair}" for condition, pair in pair_pivot.columns
    ]
    pair_delta_pivot = scoped.pivot_table(
        index="transcript_id",
        columns=["condition", "panel_pair"],
        values="paired_PCC_change",
        aggfunc="first",
    )
    pair_delta_pivot.columns = [
        f"pair_delta|||{condition}|||{pair}"
        for condition, pair in pair_delta_pivot.columns
    ]
    complete = transcript_summary.loc[
        transcript_summary["all_six_pairs_valid"]
        & transcript_summary["median_PCC_across_six_pairs"].notna()
    ]
    transcript_pivot = complete.pivot_table(
        index="transcript_id",
        columns="condition",
        values="median_PCC_across_six_pairs",
        aggfunc="first",
    )
    transcript_pivot.columns = [
        f"transcript|||{condition}|||all_six_pairs"
        for condition in transcript_pivot.columns
    ]
    transcript_delta_pivot = complete.pivot_table(
        index="transcript_id",
        columns="condition",
        values="median_paired_PCC_change_across_six_pairs",
        aggfunc="first",
    )
    transcript_delta_pivot.columns = [
        f"transcript_delta|||{condition}|||all_six_pairs"
        for condition in transcript_delta_pivot.columns
    ]
    identifiers = sorted(
        set(pair_pivot.index)
        | set(pair_delta_pivot.index)
        | set(transcript_pivot.index)
        | set(transcript_delta_pivot.index)
    )
    pair_pivot = pair_pivot.reindex(identifiers)
    pair_delta_pivot = pair_delta_pivot.reindex(identifiers)
    transcript_pivot = transcript_pivot.reindex(identifiers)
    transcript_delta_pivot = transcript_delta_pivot.reindex(identifiers)
    matrix_frame = pd.concat(
        [
            pair_pivot,
            pair_delta_pivot,
            transcript_pivot,
            transcript_delta_pivot,
        ],
        axis=1,
    )
    keys = list(matrix_frame.columns)
    matrix = matrix_frame.to_numpy(dtype=np.float64)

    gene_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, transcript_id in enumerate(identifiers):
        gene_to_indices[metadata[transcript_id].gene_id].append(index)
    genes = sorted(gene_to_indices)
    cluster_indices = [np.asarray(gene_to_indices[gene], dtype=np.int32) for gene in genes]
    cluster_unit = "gene_id" if len(genes) < len(identifiers) else "transcript_id_gene_unique"
    eligible_cluster_counts = np.asarray(
        [
            sum(
                bool(np.isfinite(matrix[indices, column_index]).any())
                for indices in cluster_indices
            )
            for column_index in range(matrix.shape[1])
        ],
        dtype=np.int32,
    )
    replicates = np.full((resamples, len(keys)), np.nan, dtype=np.float64)
    bootstrap_writer = BufferedParquetWriter(
        output_dir / "cluster_bootstrap_replicates.parquet", BOOTSTRAP_SCHEMA
    )
    rng = np.random.default_rng(seed)
    accepted = False
    try:
        for bootstrap_index in range(resamples):
            sampled = rng.integers(0, len(genes), size=len(genes))
            row_indices = np.concatenate([cluster_indices[i] for i in sampled])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                estimate = np.nanmedian(matrix[row_indices, :], axis=0)
            replicates[bootstrap_index] = estimate
            for key, value in zip(keys, estimate, strict=True):
                estimator, condition, panel_pair = key.split("|||", 2)
                estimator_name = {
                    "pair": "pairwise_median_PCC",
                    "pair_delta": "pairwise_median_paired_PCC_change",
                    "transcript": "median_of_transcript_six_pair_medians",
                    "transcript_delta": (
                        "median_of_transcript_six_pair_median_PCC_changes"
                    ),
                }[estimator]
                bootstrap_writer.append(
                    {
                        "bootstrap_index": bootstrap_index,
                        "estimator_type": estimator_name,
                        "condition": condition,
                        "panel_pair": panel_pair,
                        "estimate": float(value) if math.isfinite(value) else None,
                        "cluster_unit": cluster_unit,
                        "number_of_sampled_clusters": len(genes),
                    }
                )
            memory.check("cluster_bootstrap", bootstrap_index + 1)
        accepted = True
    finally:
        bootstrap_writer.close(accept=accepted)

    points = np.nanmedian(matrix, axis=0)
    interval_rows = []
    for column_index, key in enumerate(keys):
        estimator, condition, panel_pair = key.split("|||", 2)
        estimator_name = {
            "pair": "pairwise_median_PCC",
            "pair_delta": "pairwise_median_paired_PCC_change",
            "transcript": "median_of_transcript_six_pair_medians",
            "transcript_delta": (
                "median_of_transcript_six_pair_median_PCC_changes"
            ),
        }[estimator]
        valid = replicates[:, column_index]
        valid = valid[np.isfinite(valid)]
        quantiles = np.quantile(valid, [0.025, 0.975])
        interval_rows.append(
            {
                "estimator_type": estimator_name,
                "condition": condition,
                "panel_pair": panel_pair,
                "point_estimate": float(points[column_index]),
                "conditional_CI95_low": float(quantiles[0]),
                "conditional_CI95_high": float(quantiles[1]),
                "bootstrap_resamples": resamples,
                "bootstrap_seed": seed,
                "cluster_unit": cluster_unit,
                "number_of_sampled_clusters": len(genes),
                "number_of_eligible_clusters_for_estimate": int(
                    eligible_cluster_counts[column_index]
                ),
                "interpretation": (
                    "Conditional on these four fitted models, the fixed panel "
                    "partition, and the held-out transcript collection."
                ),
            }
        )
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(output_dir / "cluster_bootstrap_intervals.csv", index=False)
    del (
        matrix,
        replicates,
        matrix_frame,
        pair_pivot,
        pair_delta_pivot,
        transcript_pivot,
        transcript_delta_pivot,
    )
    return intervals


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


@latex_paper_style
def _make_figures(
    *,
    output_dir: Path,
    metrics: pd.DataFrame,
    transcript_summary: pd.DataFrame,
    intervals: pd.DataFrame,
) -> dict[str, Any]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    condition_labels = {
        "top_01pct_four_panel_union_removed": "Top-1% union",
        "matched_random_to_top_01pct_union": "Matched random\n(top-1% count)",
        "top_05pct_four_panel_union_removed": "Top-5% union",
        "matched_random_to_top_05pct_union": "Matched random\n(top-5% count)",
    }
    extreme = transcript_summary.loc[
        transcript_summary["condition"].isin(condition_labels)
        & transcript_summary["all_six_pairs_valid"]
    ].copy()
    extreme["plot_label"] = extreme["condition"].map(condition_labels)
    extreme_source = extreme[
        [
            "transcript_id",
            "gene_id",
            "condition",
            "plot_label",
            "median_paired_PCC_change_across_six_pairs",
            "removed_fraction",
            "n_retained",
        ]
    ]
    extreme_source.to_csv(
        figure_dir / "source_extreme_vs_matched_random.csv", index=False
    )
    ordered = list(condition_labels)
    data = [
        extreme.loc[
            extreme["condition"] == condition,
            "median_paired_PCC_change_across_six_pairs",
        ].dropna()
        for condition in ordered
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.boxplot(
        data,
        tick_labels=[condition_labels[c] for c in ordered],
        showfliers=False,
        medianprops={"color": "#b22222", "linewidth": 1.6},
        patch_artist=True,
        boxprops={"facecolor": "#8db8d8", "alpha": 0.55},
    )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Paired change in transcript-median PCC vs full profile")
    ax.set_title("Extreme-position removal versus matched random removal")
    ax.grid(axis="y", alpha=0.22)
    _save_figure(fig, figure_dir / "extreme_removal_vs_matched_random")

    jaccard = pd.read_parquet(output_dir / "high_profile_position_overlap.parquet")
    jaccard.to_csv(figure_dir / "source_high_profile_position_overlap.csv", index=False)
    finite = jaccard.loc[
        jaccard["valid"] & jaccard["jaccard"].notna() & jaccard["full_profile_PCC"].notna()
    ]
    association = spearmanr(
        finite["jaccard"].to_numpy(), finite["full_profile_PCC"].to_numpy()
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    hexbin = ax.hexbin(
        finite["jaccard"],
        finite["full_profile_PCC"],
        gridsize=42,
        mincnt=1,
        cmap="viridis",
    )
    fig.colorbar(hexbin, ax=ax, label="Transcript–panel-pair count")
    ax.set_xlabel("Jaccard overlap of independently selected high-profile positions")
    ax.set_ylabel("Full-profile PCC")
    ax.set_title(
        "High-profile-position overlap versus profile agreement\n"
        f"Spearman rho={association.statistic:.3f}"
    )
    ax.grid(alpha=0.15)
    _save_figure(fig, figure_dir / "high_profile_overlap_vs_full_PCC")

    boundary_conditions = [
        "boundary_k000",
        "boundary_k005",
        "boundary_k020",
        "boundary_k050",
    ]
    boundary = transcript_summary.loc[
        transcript_summary["condition"].isin(boundary_conditions)
        & transcript_summary["all_six_pairs_valid"]
    ].copy()
    boundary["k"] = boundary["condition"].str.extract(r"(\d+)$").astype(int)
    boundary.to_csv(figure_dir / "source_boundary_fixed_cohort.csv", index=False)
    interval_lookup = intervals.loc[
        intervals["estimator_type"].eq(
            "median_of_transcript_six_pair_medians"
        )
        & intervals["condition"].isin(boundary_conditions)
    ].set_index("condition")
    curve_rows = []
    for condition in boundary_conditions:
        group = boundary.loc[boundary["condition"] == condition]
        interval = interval_lookup.loc[condition]
        curve_rows.append(
            {
                "condition": condition,
                "k": int(condition[-3:]),
                "n_transcripts": len(group),
                "median_transcript_median_PCC": float(
                    group["median_PCC_across_six_pairs"].median()
                ),
                "CI95_low": float(interval["conditional_CI95_low"]),
                "CI95_high": float(interval["conditional_CI95_high"]),
            }
        )
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(figure_dir / "source_boundary_curve_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.7), constrained_layout=True)
    ax.errorbar(
        curve["k"],
        curve["median_transcript_median_PCC"],
        yerr=np.vstack(
            [
                curve["median_transcript_median_PCC"] - curve["CI95_low"],
                curve["CI95_high"] - curve["median_transcript_median_PCC"],
            ]
        ),
        marker="o",
        linewidth=1.8,
        capsize=3,
        color="#3b6f8f",
    )
    ax.set_xticks([0, 5, 20, 50])
    ax.set_xlabel("Modeled codons removed from each end (k)")
    ax.set_ylabel("Median transcript-level PCC across six pairs")
    ax.set_title(r"Boundary sensitivity on the fixed $n\geq150$ cohort")
    ax.grid(alpha=0.22)
    _save_figure(fig, figure_dir / "boundary_sensitivity_fixed_cohort")

    combined_conditions = [
        "combined_matched_full_profile",
        "combined_matched_interior20",
        "combined_matched_interior20_top1_union",
    ]
    combined_labels = ["Full", "Interior k=20", "Interior k=20\n+ top-1% union removed"]
    combined = transcript_summary.loc[
        transcript_summary["condition"].isin(combined_conditions)
        & transcript_summary["all_six_pairs_valid"]
    ].copy()
    combined.to_csv(
        figure_dir / "source_combined_condition_matched_cohort.csv", index=False
    )
    combined_intervals = intervals.loc[
        intervals["estimator_type"].eq(
            "median_of_transcript_six_pair_medians"
        )
        & intervals["condition"].isin(combined_conditions)
    ].set_index("condition")
    combined_rows = []
    for condition, label in zip(combined_conditions, combined_labels, strict=True):
        group = combined.loc[combined["condition"] == condition]
        interval = combined_intervals.loc[condition]
        combined_rows.append(
            {
                "condition": condition,
                "label": label.replace("\n", " "),
                "n_transcripts": len(group),
                "median_transcript_median_PCC": float(
                    group["median_PCC_across_six_pairs"].median()
                ),
                "CI95_low": float(interval["conditional_CI95_low"]),
                "CI95_high": float(interval["conditional_CI95_high"]),
            }
        )
    combined_curve = pd.DataFrame(combined_rows)
    combined_curve.to_csv(
        figure_dir / "source_combined_condition_matched_summary.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(6.7, 4.7), constrained_layout=True)
    x = np.arange(len(combined_curve))
    ax.errorbar(
        x,
        combined_curve["median_transcript_median_PCC"],
        yerr=np.vstack(
            [
                combined_curve["median_transcript_median_PCC"]
                - combined_curve["CI95_low"],
                combined_curve["CI95_high"]
                - combined_curve["median_transcript_median_PCC"],
            ]
        ),
        marker="o",
        linewidth=1.8,
        capsize=3,
        color="#7b4f9d",
    )
    ax.set_xticks(x, combined_labels)
    ax.set_ylabel("Median transcript-level PCC across six pairs")
    ax.set_title("Combined sensitivity on one identical eligible cohort")
    ax.grid(axis="y", alpha=0.22)
    _save_figure(fig, figure_dir / "combined_condition_matched_comparison")
    plt.close("all")
    return {
        "high_profile_overlap_spearman_rho": float(association.statistic),
        "high_profile_overlap_rows": len(finite),
    }


def _write_interpretation(
    *,
    output_dir: Path,
    reference_summary: pd.DataFrame,
    transcript_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    figure_stats: Mapping[str, Any],
    is_smoke: bool,
) -> None:
    aggregate = {}
    complete = transcript_summary.loc[transcript_summary["all_six_pairs_valid"]]
    for condition, group in complete.groupby("condition"):
        aggregate[str(condition)] = {
            "n": int(len(group)),
            "median_PCC": float(group["median_PCC_across_six_pairs"].median()),
            "median_paired_change": float(
                group["median_paired_PCC_change_across_six_pairs"].median()
            ),
        }
    masks = pd.read_parquet(output_dir / "mask_scalar_definitions.parquet")
    removal = (
        masks.loc[masks["panel"].eq("panel_01")]
        .groupby("mask_family")["four_panel_union_fraction_of_domain"]
        .agg(["count", "median", "mean", "min", "max"])
        .reset_index()
    )
    removal.to_csv(output_dir / "actual_union_removed_fraction_summary.csv", index=False)
    cohorts = pd.read_parquet(output_dir / "boundary_cohort_flags.parquet")
    boundary_n = int(cohorts["boundary_fixed_cohort"].sum())
    combined_n = int(cohorts["combined_fixed_cohort"].sum())
    full = aggregate.get("full_profile_all_test_transcripts", {})
    top1 = aggregate.get("top_01pct_four_panel_union_removed", {})
    rand1 = aggregate.get("matched_random_to_top_01pct_union", {})
    top5 = aggregate.get("top_05pct_four_panel_union_removed", {})
    rand5 = aggregate.get("matched_random_to_top_05pct_union", {})
    combined_full = aggregate.get("combined_matched_full_profile", {})
    combined_interior = aggregate.get("combined_matched_interior20", {})
    combined_removed = aggregate.get(
        "combined_matched_interior20_top1_union", {}
    )
    lines = [
        "# Streaming positional robustness analysis",
        "",
        (
            "This is a 25-transcript smoke test, not a scientific result."
            if is_smoke
            else "This report uses all four fitted uniform-reference panel models."
        ),
        "No model or checkpoint was loaded, retrained, or altered.",
        "",
        "## Reference reproduction",
        "",
        "```",
        reference_summary.to_string(index=False),
        "```",
        "",
        "## Main directional results",
        "",
        f"- Full profile: transcript-median-over-six-pairs PCC = {full.get('median_PCC', math.nan):.4f}.",
        f"- Top-1% four-panel union removed: PCC = {top1.get('median_PCC', math.nan):.4f}; median paired change = {top1.get('median_paired_change', math.nan):+.4f}.",
        f"- Matched random removal for the top-1% union count: PCC = {rand1.get('median_PCC', math.nan):.4f}; median paired change = {rand1.get('median_paired_change', math.nan):+.4f}.",
        f"- Top-5% four-panel union removed: PCC = {top5.get('median_PCC', math.nan):.4f}; median paired change = {top5.get('median_paired_change', math.nan):+.4f}.",
        f"- Matched random removal for the top-5% union count: PCC = {rand5.get('median_PCC', math.nan):.4f}; median paired change = {rand5.get('median_paired_change', math.nan):+.4f}.",
        f"- Boundary fixed cohort: {boundary_n} transcripts; combined-condition fixed cohort: {combined_n} transcripts.",
        f"- On that identical combined cohort: full={combined_full.get('median_PCC', math.nan):.4f}, interior-20={combined_interior.get('median_PCC', math.nan):.4f}, interior-20 plus top-1% union removal={combined_removed.get('median_PCC', math.nan):.4f}.",
        f"- Jaccard/full-PCC descriptive Spearman association: rho={figure_stats['high_profile_overlap_spearman_rho']:.4f}.",
        "",
        "The union removed fraction is reported from the realized masks and is not assumed to equal 1% or 5%.",
        "A decrease after removal means that the removed positions contributed to agreement; it does not show that the original agreement was spurious.",
        "High-profile positions are model predictions, not experimentally validated stalls and not independently detected local maxima.",
        "The 100 random masks per transcript quantify sensitivity to removing positions; they are not independent experiments or fitted-model replicates.",
        "Bootstrap intervals are conditional on these four fitted models, this one panel partition, and this held-out collection; the six panel pairs are not treated as independent fitted-model replicates.",
        "The terminal stop is included at modeled index n-1. Boundary k removes [0,k) and [n-k,n), so k>0 removes the terminal stop.",
        "",
    ]
    (output_dir / "interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def _validate_output_schemas_and_counts(
    output_dir: Path,
    transcript_count: int,
    random_draws: int,
) -> dict[str, Any]:
    expected = {
        "reference_full_profile_metrics.parquet": transcript_count * 6,
        "streamed_input_validation.parquet": transcript_count * 4,
        # full 6 + extreme/random 24 + boundary 24 + combined 6 = 60
        "per_transcript_pair_scalar_metrics.parquet": transcript_count * 60,
        # top1/top5/combined, four panels each
        "mask_scalar_definitions.parquet": transcript_count * 12,
        "high_profile_position_overlap.parquet": transcript_count * 6,
        "boundary_cohort_flags.parquet": transcript_count,
    }
    checks: dict[str, Any] = {}
    for filename, expected_rows in expected.items():
        path = output_dir / filename
        parquet = pq.ParquetFile(path)
        observed = parquet.metadata.num_rows
        if observed != expected_rows:
            raise ValueError(
                f"{filename}: observed {observed} rows, expected {expected_rows}."
            )
        max_row_group = max(
            parquet.metadata.row_group(i).num_rows
            for i in range(parquet.metadata.num_row_groups)
        )
        if max_row_group > 1000:
            raise ValueError(
                f"{filename}: row group {max_row_group} exceeds 1000-row contract."
            )
        checks[filename] = {
            "rows": observed,
            "expected_rows": expected_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "maximum_row_group_rows": max_row_group,
            "schema": str(parquet.schema_arrow),
        }
    random_frame = pd.read_parquet(
        output_dir / "per_transcript_pair_scalar_metrics.parquet",
        columns=[
            "analysis_family",
            "random_draws_requested",
            "random_draws_valid",
        ],
    )
    random_rows = random_frame.loc[
        random_frame["analysis_family"].eq("matched_random_removal_control")
    ]
    if not bool((random_rows["random_draws_requested"] == random_draws).all()):
        raise ValueError("Random-control draw-count schema check failed.")
    checks["random_control"] = {
        "rows": len(random_rows),
        "draws_requested_per_row": random_draws,
        "minimum_valid_draws": int(random_rows["random_draws_valid"].min()),
        "maximum_valid_draws": int(random_rows["random_draws_valid"].max()),
        "draw_level_results_saved": False,
    }
    return checks


def _validate_reference_rowwise(
    *, run_root: Path, reference_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Compare scalar reference metrics with the accepted prior analysis."""

    existing_path = run_root / "analysis/cross_panel_L_agreement_long.parquet"
    if not existing_path.exists():
        result = {
            "status": "prior_scalar_analysis_not_available",
            "prior_path": str(existing_path),
        }
        _json_dump(output_dir / "reference_rowwise_reproduction.json", result)
        return result
    existing = pd.read_parquet(existing_path)
    streamed = pd.read_parquet(reference_path)
    identifiers = set(streamed["transcript_id"])
    existing = existing.loc[existing["transcript_id"].isin(identifiers)]
    keys = ["transcript_id", "panel_a", "panel_b"]
    joined = existing.merge(
        streamed,
        on=keys,
        suffixes=("_existing", "_streaming"),
        validate="one_to_one",
    )
    if len(joined) != len(streamed):
        raise ValueError(
            f"Rowwise reference join retained {len(joined)}/{len(streamed)} rows."
        )
    result = {"status": "exact_match", "rows": len(joined)}
    for metric in ("PCC", "Spearman", "RMSE"):
        difference = np.abs(
            joined[f"{metric}_existing"] - joined[f"{metric}_streaming"]
        )
        maximum = float(difference.max())
        result[f"{metric}_max_abs_difference"] = maximum
        if maximum > 1.0e-12:
            raise RuntimeError(
                f"Streaming {metric} differs from accepted analysis by {maximum}."
            )
    _json_dump(output_dir / "reference_rowwise_reproduction.json", result)
    return result


def _validate_scalar_logic(output_dir: Path) -> dict[str, Any]:
    """Validate algebraic mask, cohort, metric, and scalar-schema invariants."""

    metrics = pd.read_parquet(
        output_dir / "per_transcript_pair_scalar_metrics.parquet"
    )
    masks = pd.read_parquet(output_dir / "mask_scalar_definitions.parquet")
    jaccard = pd.read_parquet(
        output_dir / "high_profile_position_overlap.parquet"
    )
    cohorts = pd.read_parquet(output_dir / "boundary_cohort_flags.parquet")
    validation = pd.read_parquet(output_dir / "streamed_input_validation.parquet")
    finite = metrics["PCC"].notna() & metrics["reference_full_PCC"].notna()
    checks: dict[str, Any] = {
        "all_panel_identity_counts_equal": bool(
            validation.groupby("panel")["transcript_id"].nunique().nunique() == 1
        ),
        "profile_length_and_coordinate_identity": bool(
            (
                validation["profile_array_length"].eq(
                    validation["transcript_length"]
                )
                & validation["implicit_valid_mask_count"].eq(
                    validation["transcript_length"]
                )
                & validation["last_modeled_coordinate"].eq(
                    validation["transcript_length"] - 1
                )
            ).all()
        ),
        "all_profiles_finite_and_positive": bool(
            validation["all_finite"].all() and validation["all_positive"].all()
        ),
        "maximum_mean_one_deviation": float(
            validation["absolute_mean_one_deviation"].max()
        ),
        "all_terminal_codons_are_stops": bool(
            validation["terminal_stop_included"].all()
        ),
        "tie_inclusive_set_never_smaller_than_ceil_target": bool(
            (
                masks["panel_selected_count_with_ties"]
                >= masks["ceil_target_count"]
            ).all()
        ),
        "union_count_bounds_hold": bool(
            (
                masks["four_panel_union_count"]
                >= masks["panel_selected_count_with_ties"]
            ).all()
            and (
                masks["four_panel_union_count"]
                <= masks["selection_domain_size"]
            ).all()
        ),
        "mask_retained_plus_union_equals_domain": bool(
            (
                masks["retained_count_after_union"]
                + masks["four_panel_union_count"]
                == masks["selection_domain_size"]
            ).all()
        ),
        "metric_retained_plus_removed_equals_length": bool(
            (
                metrics["n_removed"] + metrics["n_retained"]
                == metrics["transcript_length"]
            ).all()
        ),
        "maximum_paired_delta_algebra_error": float(
            np.max(
                np.abs(
                    metrics.loc[finite, "paired_PCC_change"]
                    - (
                        metrics.loc[finite, "PCC"]
                        - metrics.loc[finite, "reference_full_PCC"]
                    )
                )
            )
        ),
        "jaccard_count_identity": bool(
            (
                jaccard["union_count"]
                == jaccard["selected_count_a_with_ties"]
                + jaccard["selected_count_b_with_ties"]
                - jaccard["intersection_count"]
            ).all()
        ),
        "maximum_jaccard_algebra_error": float(
            np.max(
                np.abs(
                    jaccard["jaccard"]
                    - jaccard["intersection_count"] / jaccard["union_count"]
                )
            )
        ),
        "boundary_fixed_cohort_definition": bool(
            cohorts["boundary_fixed_cohort"].eq(
                cohorts["n_at_least_150"]
                & cohorts["valid_all_six_pairs_at_k0_k5_k20_k50"]
            ).all()
        ),
        "combined_fixed_cohort_definition": bool(
            cohorts["combined_fixed_cohort"].eq(
                cohorts["boundary_fixed_cohort"]
                & cohorts["valid_all_six_pairs_combined_condition"]
            ).all()
        ),
        "position_array_columns_absent_from_scalar_outputs": True,
    }
    for filename in (
        "per_transcript_pair_scalar_metrics.parquet",
        "mask_scalar_definitions.parquet",
        "high_profile_position_overlap.parquet",
    ):
        forbidden = {"L_t", "L_bio", "mask"} & set(
            pq.read_schema(output_dir / filename).names
        )
        if forbidden:
            checks["position_array_columns_absent_from_scalar_outputs"] = False
    failed = [
        key
        for key, value in checks.items()
        if isinstance(value, bool) and not value
    ]
    if checks["maximum_paired_delta_algebra_error"] > 1.0e-12:
        failed.append("maximum_paired_delta_algebra_error")
    if checks["maximum_jaccard_algebra_error"] > 1.0e-12:
        failed.append("maximum_jaccard_algebra_error")
    if failed:
        raise RuntimeError(f"Scalar logic validation failed: {failed}.")
    checks["status"] = "PASS"
    _json_dump(output_dir / "logic_validation.json", checks)
    return checks


def _command_text(
    *,
    run_root: Path,
    output_root: Path,
    max_rss_gb: float,
    max_transcripts: int | None,
    random_draws: int,
    random_seed: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    log_path: Path,
) -> str:
    python = PROJECT_ROOT / ".venv/bin/python"
    command = [
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        f"MPLCONFIGDIR={output_root / 'matplotlib_cache'}",
        str(python),
        "-u",
        str(SCRIPT_PATH),
        "--run-root",
        str(run_root),
        "--output-root",
        str(output_root),
        "--max-rss-gb",
        str(max_rss_gb),
        "--random-draws",
        str(random_draws),
        "--random-seed",
        str(random_seed),
        "--bootstrap-resamples",
        str(bootstrap_resamples),
        "--bootstrap-seed",
        str(bootstrap_seed),
    ]
    if max_transcripts is not None:
        command.extend(["--max-transcripts", str(max_transcripts)])
    return " \\\n  ".join(command) + f" > {log_path} 2>&1\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    if not run_root.exists():
        raise FileNotFoundError(run_root)
    if args.max_transcripts is not None and args.max_transcripts <= 0:
        raise ValueError("--max-transcripts must be positive.")
    if args.random_draws <= 0 or args.bootstrap_resamples <= 0:
        raise ValueError("Random draws and bootstrap resamples must be positive.")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else artifact_directory("real_data", run_root, DEFAULT_OUTPUT_NAME)
    )
    mode_name = (
        f"smoke_{args.max_transcripts:05d}"
        if args.max_transcripts is not None
        else "full"
    )
    output_dir = output_root / mode_name
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_dir}. Use --overwrite to "
                "replace only this streaming-analysis directory."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_root / "matplotlib_cache").mkdir(exist_ok=True)

    common_manifest_path = run_root / "common_split_manifest.json"
    if not common_manifest_path.exists():
        raise FileNotFoundError(common_manifest_path)
    common_manifest = _read_json(common_manifest_path)
    all_test_ids = [str(value) for value in common_manifest["common_test_ids"]]
    if len(all_test_ids) != 1593:
        raise ValueError(
            f"Expected the completed run's 1,593 test transcripts, found "
            f"{len(all_test_ids)}."
        )
    selected_ids = (
        all_test_ids[: args.max_transcripts]
        if args.max_transcripts is not None
        else all_test_ids
    )
    if not selected_ids:
        raise ValueError("No transcripts selected.")

    panel_paths: dict[str, Path] = {}
    manifest_paths: dict[str, Path] = {}
    manifest_kinds: dict[str, str] = {}
    for panel in PANEL_NAMES:
        prediction, manifest, kind = _locate_panel_prediction(run_root / panel)
        if prediction is None or manifest is None:
            raise FileNotFoundError(f"No usable {panel} prediction: {kind}")
        if prediction.name != "common_test_L_profiles.parquet":
            raise ValueError(
                f"{panel} resolved {prediction.name}; this low-memory analysis "
                "accepts only compact common_test_L_profiles.parquet exports."
            )
        schema = set(pq.read_schema(prediction).names)
        required = {
            "transcript_id",
            "panel",
            "run_identifier",
            "transcript_length",
            "L_t",
        }
        if not required <= schema:
            raise KeyError(f"{panel} compact schema missing {sorted(required-schema)}.")
        panel_paths[panel] = prediction.resolve()
        manifest_paths[panel] = manifest.resolve()
        manifest_kinds[panel] = kind

    smoke_log = output_root / "smoke_00025.log"
    full_log = output_root / "full_run.log"
    smoke_command = _command_text(
        run_root=run_root,
        output_root=output_root,
        max_rss_gb=args.max_rss_gb,
        max_transcripts=25,
        random_draws=args.random_draws,
        random_seed=args.random_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        log_path=smoke_log,
    )
    full_command = _command_text(
        run_root=run_root,
        output_root=output_root,
        max_rss_gb=args.max_rss_gb,
        max_transcripts=None,
        random_draws=args.random_draws,
        random_seed=args.random_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        log_path=full_log,
    )
    (output_root / "exact_smoke_test_command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + smoke_command,
        encoding="utf-8",
    )
    (output_root / "exact_full_run_command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + full_command,
        encoding="utf-8",
    )
    print("Exact invocation:\n" + (smoke_command if args.max_transcripts else full_command))

    memory = MemoryGuard(output_dir / "peak_memory_log.csv", args.max_rss_gb)
    status = "failed"
    manifest: dict[str, Any] = {
        "analysis": "real_panel_positional_robustness_streaming",
        "analysis_version": 1,
        "started_at_utc": _utc_now(),
        "status": status,
        "run_root": str(run_root),
        "output_directory": str(output_dir),
        "mode": "smoke" if args.max_transcripts is not None else "full",
        "number_of_manifest_test_transcripts": len(all_test_ids),
        "number_of_selected_transcripts": len(selected_ids),
        "settings": {
            "max_rss_gb": args.max_rss_gb,
            "max_transcripts": args.max_transcripts,
            "random_draws_per_transcript_and_fraction": args.random_draws,
            "random_seed": args.random_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "mean_one_tolerance": args.mean_one_tolerance,
            "arrow_prediction_batch_rows": 1,
            "maximum_scalar_writer_batch_rows": 1000,
        },
        "implementation_guards": {
            "models_or_checkpoints_loaded": False,
            "prediction_files_read_with_pandas": False,
            "prediction_profile_dictionary_constructed": False,
            "codon_positions_exploded": False,
            "draw_level_random_results_saved": False,
            "amplitudes_renormalized_after_masking": False,
        },
        "common_split_manifest": {
            "path": str(common_manifest_path),
            "sha256": _sha256_file(common_manifest_path),
            "test_id_hash_from_manifest": common_manifest.get("fold_id_hashes", {}).get(
                "test"
            ),
        },
        "prediction_inputs": {
            panel: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "source_manifest": str(manifest_paths[panel]),
                "source_manifest_kind": manifest_kinds[panel],
                "checkpoint_variant": _read_json(manifest_paths[panel]).get(
                    "checkpoint_variant", "best_val_loss"
                ),
                "schema": str(pq.read_schema(path)),
            }
            for panel, path in panel_paths.items()
        },
        "coordinate_and_mask_convention": {
            "compact_export_mask": (
                "Implicit all-valid mask over the compact vector. The compact "
                "export was materialized only after the production analyzer "
                "validated raw masks; this analysis additionally requires "
                "len(L_t)==transcript_length==MANE codon count."
            ),
            "coordinate": "zero-based modeled codon index",
            "first_index": 0,
            "last_index": "n-1",
            "terminal_stop": "included at n-1",
            "boundary_k": "remove [0,k) and [n-k,n)",
        },
        "git": _git_state(),
        "script": {
            "path": str(SCRIPT_PATH),
            "sha256_at_start": _sha256_file(SCRIPT_PATH),
        },
        "exact_command": smoke_command if args.max_transcripts else full_command,
    }
    _json_dump(output_dir / "analysis_manifest.json", manifest)
    try:
        sequence_path = _resolve_sequence_path(run_root, common_manifest)
        metadata = _load_sequence_metadata(sequence_path, selected_ids, memory)
        manifest["sequence_coordinate_source"] = {
            "path": str(sequence_path),
            "sha256": _sha256_file(sequence_path),
            "held_out_rows": len(metadata),
            "unique_gene_ids": len({record.gene_id for record in metadata.values()}),
            "all_terminal_codons_are_stops": all(
                record.terminal_codon in STOP_CODONS for record in metadata.values()
            ),
        }

        aligned = AlignedProfileSource(panel_paths, selected_ids, output_dir)
        reference_path, validation_path = _run_reference_precheck(
            source=aligned,
            metadata=metadata,
            output_dir=output_dir,
            mean_one_tolerance=args.mean_one_tolerance,
            memory=memory,
        )
        reference_summary = _validate_reference_summary(
            reference_path,
            len(selected_ids),
            is_smoke=args.max_transcripts is not None,
        )
        reference_summary.to_csv(
            output_dir / "reference_reproduction_summary.csv", index=False
        )
        print("Reference reproduction:\n" + reference_summary.to_string(index=False))

        # A fresh iterator guarantees no prediction profiles survive the
        # reference pass.  The ID-only index is small and reused.
        aligned_analysis = AlignedProfileSource(panel_paths, selected_ids, output_dir)
        counts = _run_streaming_analysis(
            source=aligned_analysis,
            metadata=metadata,
            output_dir=output_dir,
            random_draws=args.random_draws,
            random_seed=args.random_seed,
            memory=memory,
        )
        output_checks = _validate_output_schemas_and_counts(
            output_dir, len(selected_ids), args.random_draws
        )
        _json_dump(output_dir / "output_validation.json", output_checks)
        reference_rowwise = _validate_reference_rowwise(
            run_root=run_root,
            reference_path=reference_path,
            output_dir=output_dir,
        )
        logic_checks = _validate_scalar_logic(output_dir)

        metrics, transcript_summary, pair_summary = _make_compact_summaries(output_dir)
        intervals = _bootstrap_intervals(
            metrics=metrics,
            transcript_summary=transcript_summary,
            metadata=metadata,
            output_dir=output_dir,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
            memory=memory,
        )
        figure_stats = _make_figures(
            output_dir=output_dir,
            metrics=metrics,
            transcript_summary=transcript_summary,
            intervals=intervals,
        )
        _write_interpretation(
            output_dir=output_dir,
            reference_summary=reference_summary,
            transcript_summary=transcript_summary,
            pair_summary=pair_summary,
            figure_stats=figure_stats,
            is_smoke=args.max_transcripts is not None,
        )
        memory.check("completed_analysis", len(selected_ids), log=True)
        status = "complete"
        manifest.update(
            {
                "status": status,
                "completed_at_utc": _utc_now(),
                "reference_reproduction": reference_summary.to_dict("records"),
                "streaming_output_counts": counts,
                "output_validation": output_checks,
                "reference_rowwise_reproduction": reference_rowwise,
                "logic_validation": logic_checks,
                "figure_statistics": figure_stats,
                "peak_observed_rss_bytes": memory.peak_bytes,
                "peak_observed_rss_gib": memory.peak_bytes / (1024**3),
                "inference_scope": (
                    "All intervals are conditional on these four fitted models, "
                    "this panel partition, and this held-out collection."
                ),
            }
        )
        _json_dump(output_dir / "analysis_manifest.json", manifest)
        print(
            f"Completed {mode_name}: {len(selected_ids)} transcripts; "
            f"peak observed RSS={memory.peak_bytes / (1024**3):.3f} GiB; "
            f"outputs={output_dir}",
            flush=True,
        )
        del metrics, transcript_summary, pair_summary, intervals
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "peak_observed_rss_bytes": memory.peak_bytes,
                "peak_observed_rss_gib": memory.peak_bytes / (1024**3),
            }
        )
        _json_dump(output_dir / "analysis_manifest.json", manifest)
        raise
    finally:
        memory.close()
        plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())
