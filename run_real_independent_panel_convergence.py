#!/usr/bin/env python3
"""Design and launch independent real-dataset-panel convergence experiments.

The script owns experiment design and process orchestration only. Every panel
still trains through ``main_ribounmix_multidataset.py`` and its
production model, loss, sampler, checkpoint, and prediction implementations.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from Utils.real_panel_convergence import (
    assert_panel_partition,
    build_common_transcript_split,
    build_dataset_quality_table,
    build_panel_balance_report,
    build_panel_stored_weight_manifest,
    deterministic_stratified_panel_assignment,
    fit_panel_reliability_manifest,
    json_ready,
    load_dataset_mapping,
    panel_dictionary,
    plot_panel_balance,
    utc_timestamp,
    write_json,
)
from Utils.reliability_references import transcript_id_hash


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config/config_ribounmix_multidataset.yaml"
DEFAULT_DATASET_CONFIG = (
    PROJECT_ROOT
    / "config/dataset_config/weighted_hek_riboseq_codon_replicas.yaml"
)
DEFAULT_ENTRYPOINT = PROJECT_ROOT / "main_ribounmix_multidataset.py"
DEFAULT_ANALYSIS = PROJECT_ROOT / "analyses/analyze_real_panel_convergence.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/real_independent_panel_convergence"
REQUIRED_WEIGHTED_DATASET_COLUMNS = frozenset(
    {"id", "ribo", "ribo_cds_replicas", "weight"}
)
OPTIONAL_WEIGHT_AUDIT_COLUMNS = frozenset({"read_density", "coverage"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct balanced disjoint real-data panels, one common held-out "
            "transcript split, and optionally launch one independent model per GPU."
        )
    )
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument("--dry-run", action="store_true", help="Write the complete design but do not launch training.")
    preparation.add_argument("--prepare-only", action="store_true", help="Prepare production run paths without launching training (no _dry_run suffix).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic panel/split seed (default: 42).")
    parser.add_argument("--num-panels", type=int, default=4, help="Number of independent panels (paper default: 4).")
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU IDs; panels queue round-robin (default: 0,1).")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None, help="Outer run identifier (default: panel_convergence_seed<seed>).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Production real-data training YAML.")
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--entrypoint", type=Path, default=DEFAULT_ENTRYPOINT)
    parser.add_argument("--analysis-script", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--sequences-path", type=Path, default=None, help="Override the sequence parquet resolved from the base config.")
    parser.add_argument("--quality-table", type=Path, default=None, help="Load a previously computed observed-only quality CSV.")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--reliability-bins", type=int, default=10)
    parser.add_argument(
        "--reliability-weight-mode",
        choices=("stored", "train-only-snr"),
        default="train-only-snr",
        help=(
            "Source of local loss weights w_dt. 'train-only-snr' (default) fits "
            "SNR/coverage reference statistics on training IDs only and freezes "
            "them for validation/test. 'stored' is the explicit compatibility "
            "mode for already materialized historical weights."
        ),
    )
    parser.add_argument("--max-cds-codons", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Per-dataset logical grouped-batch quota (default: 32). This is "
            "separate from the execution-forward memory limits."
        ),
    )
    parser.add_argument(
        "--max-pair-rows-per-forward",
        type=int,
        default=512,
        help="Execution-only GPU forward limit for pair rows (default: 512).",
    )
    parser.add_argument(
        "--max-padded-codon-tokens-per-forward",
        type=int,
        default=256000,
        help="Execution-only padded-codon-token limit (default: 256000).",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=25,
        help="TensorBoard logging interval in execution chunks (default: 25).",
    )
    parser.add_argument(
        "--reference-chunk-size",
        type=int,
        default=16,
        help="Fixed-reference dataset identities evaluated per internal chunk (default: 16).",
    )
    parser.add_argument(
        "--raw-log-gamma-bound",
        type=float,
        default=8.0,
        help="Symmetric numerical bound applied before gamma centering (default: 8).",
    )
    parser.add_argument("--expected-dataset-count", type=int, default=114)
    parser.add_argument("--allow-excluded-datasets", action="store_true", help="Proceed after recording unusable candidates; disabled for the paper run.")
    parser.add_argument("--overwrite-design", action="store_true", help="Overwrite design files in an existing run directory without deleting checkpoints.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--predict-num-workers", type=int, default=0)
    parser.add_argument(
        "--multiprocessing-context",
        choices=("spawn", "forkserver", "fork"),
        default="spawn",
    )
    default_python = (
        PROJECT_ROOT / ".venv/bin/python"
        if (PROJECT_ROOT / ".venv/bin/python").exists()
        else Path(sys.executable)
    )
    parser.add_argument("--python-executable", type=Path, default=default_python)
    return parser.parse_args(argv)


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def _set_nested(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = payload
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _git_metadata() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            arguments,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {"commit": commit, "worktree_dirty": bool(status) if status is not None else None}


def _parse_gpus(raw: str) -> list[str]:
    if not re.fullmatch(r"\d+(,\d+)*", str(raw).strip()):
        raise ValueError("--gpus must be a comma-separated list such as 0,1,2,3.")
    values = str(raw).split(",")
    if len(values) != len(set(values)):
        raise ValueError("--gpus contains a duplicate physical GPU ID.")
    return values


def _validate_requested_gpus(gpus: Sequence[str]) -> None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi was not found; refusing to launch GPU training.")
    result = subprocess.run(
        [executable, "--query-gpu=index", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "nvidia-smi could not enumerate GPUs; refusing to launch: "
            f"{result.stderr.strip()}"
        )
    available = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    unavailable = sorted(set(map(str, gpus)) - available, key=int)
    if unavailable:
        raise RuntimeError(
            f"Requested unavailable physical GPU IDs {unavailable}; available IDs are "
            f"{sorted(available, key=int)}."
        )


def _inspect_weighted_dataset_sources(
    dataset_mapping: Mapping[str, str],
    *,
    dataset_config_path: Path,
) -> dict[str, Any]:
    """Resolve and validate both compact and audit-rich weighted artifacts.

    Historical production parquets contain the four core columns and may use a
    zero stored weight to mark a row as excluded.  Newer regenerated artifacts
    additionally materialize ``read_density`` and ``coverage``.  Those two
    audit columns are not needed to prove that the configured source is a
    weighted, replica-aware artifact.
    """
    resolved_counts: dict[str, int] = {}
    for raw_path in dataset_mapping.values():
        resolved = str(Path(raw_path).expanduser().resolve())
        resolved_counts[resolved] = resolved_counts.get(resolved, 0) + 1

    records: list[dict[str, Any]] = []
    for dataset_name, raw_path in dataset_mapping.items():
        path = Path(raw_path).expanduser().resolve()
        record: dict[str, Any] = {
            "dataset_name": str(dataset_name),
            "configured_path": str(raw_path),
            "resolved_path": str(path),
            "exists": path.is_file(),
            "path_stem_matches_dataset_name": path.stem == str(dataset_name),
            "resolved_path_is_unique": resolved_counts[str(path)] == 1,
            "required_columns": sorted(REQUIRED_WEIGHTED_DATASET_COLUMNS),
            "optional_audit_columns": sorted(OPTIONAL_WEIGHT_AUDIT_COLUMNS),
            "schema_columns": [],
            "missing_required_columns": [],
            "missing_optional_audit_columns": [],
            "schema_variant": None,
            "physical_row_count": None,
            "positive_weight_row_count": None,
            "zero_weight_exclusion_row_count": None,
            "invalid_weight_row_count": None,
            "file_size_bytes": None,
            "file_mtime_ns": None,
            "error": None,
        }
        if path.is_file():
            stat = path.stat()
            record["file_size_bytes"] = int(stat.st_size)
            record["file_mtime_ns"] = int(stat.st_mtime_ns)
            try:
                schema_columns = list(pq.read_schema(path).names)
                record["schema_columns"] = schema_columns
                record["missing_required_columns"] = sorted(
                    REQUIRED_WEIGHTED_DATASET_COLUMNS.difference(schema_columns)
                )
                record["missing_optional_audit_columns"] = sorted(
                    OPTIONAL_WEIGHT_AUDIT_COLUMNS.difference(schema_columns)
                )
                if not record["missing_required_columns"]:
                    record["schema_variant"] = (
                        "audit_rich_weighted"
                        if not record["missing_optional_audit_columns"]
                        else "compact_legacy_weighted"
                    )
                    weight_frame = pd.read_parquet(path, columns=["weight"])
                    weights = pd.to_numeric(
                        weight_frame["weight"], errors="coerce"
                    ).to_numpy(dtype=np.float64)
                    invalid = ~np.isfinite(weights) | (weights < 0.0)
                    record["physical_row_count"] = int(weights.size)
                    record["positive_weight_row_count"] = int(
                        np.count_nonzero(weights > 0.0)
                    )
                    record["zero_weight_exclusion_row_count"] = int(
                        np.count_nonzero(weights == 0.0)
                    )
                    record["invalid_weight_row_count"] = int(
                        np.count_nonzero(invalid)
                    )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
        else:
            record["error"] = "configured parquet does not exist"
        record["contract_passed"] = bool(
            record["exists"]
            and record["path_stem_matches_dataset_name"]
            and record["resolved_path_is_unique"]
            and not record["missing_required_columns"]
            and record["invalid_weight_row_count"] == 0
            and int(record["positive_weight_row_count"] or 0) > 0
            and record["error"] is None
        )
        records.append(record)

    failures = [record for record in records if not record["contract_passed"]]
    return {
        "dataset_config_path": str(dataset_config_path),
        "required_artifact_type": "filtered_weighted_replica_aware_parquet",
        "required_columns": sorted(REQUIRED_WEIGHTED_DATASET_COLUMNS),
        "optional_audit_columns": sorted(OPTIONAL_WEIGHT_AUDIT_COLUMNS),
        "zero_weight_policy": (
            "legacy zero-weight rows are explicit exclusions and never enter "
            "support counts, splits, or model losses"
        ),
        "candidate_dataset_count": int(len(records)),
        "passing_dataset_count": int(len(records) - len(failures)),
        "status": "PASS" if not failures else "FAIL",
        "dataset_directories": sorted(
            {str(Path(record["resolved_path"]).parent) for record in records}
        ),
        "failures": failures,
        "datasets": records,
    }


def _load_or_compute_quality(
    *,
    args: argparse.Namespace,
    dataset_mapping: Mapping[str, str],
    sequences_path: Path,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    if args.quality_table is None:
        return build_dataset_quality_table(
            dataset_mapping=dataset_mapping,
            sequences_path=sequences_path,
            max_cds_codons=args.max_cds_codons,
        )
    source = _absolute(args.quality_table)
    quality = pd.read_csv(source)
    required_table_columns = {
        "dataset_name",
        "dataset_path",
        "eligible",
        "source_file_size_bytes",
        "source_file_mtime_ns",
        "quality_manifest_version",
        "quality_sequences_path",
        "quality_sequences_file_size_bytes",
        "quality_sequences_file_mtime_ns",
        "quality_max_cds_codons",
    }
    missing_table_columns = required_table_columns.difference(quality.columns)
    if missing_table_columns:
        raise KeyError(
            "Loaded quality table is missing provenance columns "
            f"{sorted(missing_table_columns)}. Recompute it from the configured "
            "weighted artifacts."
        )
    quality["dataset_name"] = quality["dataset_name"].astype(str)
    if bool(quality["dataset_name"].duplicated().any()):
        raise ValueError("Loaded quality table contains duplicate dataset names.")
    expected = set(dataset_mapping)
    observed = set(quality["dataset_name"])
    if expected != observed:
        raise ValueError(
            "Loaded quality-table dataset identity mismatch: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}."
        )
    table_paths = {
        str(row.dataset_name): str(
            (
                Path(str(row.dataset_path)).expanduser()
                if Path(str(row.dataset_path)).is_absolute()
                else PROJECT_ROOT / Path(str(row.dataset_path)).expanduser()
            ).resolve()
        )
        for row in quality.itertuples()
    }
    path_mismatches = {
        name: {
            "quality_table_path": table_paths[name],
            "configured_weighted_path": str(Path(dataset_mapping[name]).resolve()),
        }
        for name in dataset_mapping
        if table_paths[name] != str(Path(dataset_mapping[name]).resolve())
    }
    if path_mismatches:
        examples = dict(list(path_mismatches.items())[:5])
        raise ValueError(
            "Loaded quality table was computed from different dataset paths. "
            "Refusing stale/raw provenance; recompute without --quality-table. "
            f"Examples: {examples}"
        )

    stale_dataset_files: dict[str, dict[str, int]] = {}
    indexed_quality = quality.set_index("dataset_name", drop=False)
    for dataset_name, configured_path in dataset_mapping.items():
        current_stat = Path(configured_path).stat()
        row = indexed_quality.loc[str(dataset_name)]
        cached_size = int(row["source_file_size_bytes"])
        cached_mtime = int(row["source_file_mtime_ns"])
        if (
            cached_size != int(current_stat.st_size)
            or cached_mtime != int(current_stat.st_mtime_ns)
        ):
            stale_dataset_files[str(dataset_name)] = {
                "cached_size": cached_size,
                "current_size": int(current_stat.st_size),
                "cached_mtime_ns": cached_mtime,
                "current_mtime_ns": int(current_stat.st_mtime_ns),
            }
    if stale_dataset_files:
        examples = dict(list(stale_dataset_files.items())[:5])
        raise ValueError(
            "Loaded quality table is stale relative to one or more weighted "
            "dataset files. Recompute without --quality-table. "
            f"Examples: {examples}"
        )

    def _one_cached_value(column: str) -> Any:
        values = quality[column].drop_duplicates()
        if len(values) != 1:
            raise ValueError(
                f"Loaded quality table has inconsistent {column!r} provenance. "
                "Recompute without --quality-table."
            )
        return values.iloc[0]

    manifest_version = int(_one_cached_value("quality_manifest_version"))
    if manifest_version != 1:
        raise ValueError(
            f"Unsupported quality manifest version {manifest_version}; recompute "
            "without --quality-table."
        )
    cached_sequences_path = Path(
        str(_one_cached_value("quality_sequences_path"))
    ).expanduser().resolve()
    current_sequences_path = sequences_path.expanduser().resolve()
    current_sequence_stat = current_sequences_path.stat()
    cached_sequence_size = int(
        _one_cached_value("quality_sequences_file_size_bytes")
    )
    cached_sequence_mtime = int(
        _one_cached_value("quality_sequences_file_mtime_ns")
    )
    if (
        cached_sequences_path != current_sequences_path
        or cached_sequence_size != int(current_sequence_stat.st_size)
        or cached_sequence_mtime != int(current_sequence_stat.st_mtime_ns)
    ):
        raise ValueError(
            "Loaded quality table was computed from a different or modified "
            "sequence table. Recompute without --quality-table."
        )
    cached_max_cds = _one_cached_value("quality_max_cds_codons")
    cached_max_cds = None if pd.isna(cached_max_cds) else int(cached_max_cds)
    requested_max_cds = (
        None if args.max_cds_codons is None else int(args.max_cds_codons)
    )
    if cached_max_cds != requested_max_cds:
        raise ValueError(
            "Loaded quality table used a different maximum-CDS eligibility "
            f"rule ({cached_max_cds!r} versus {requested_max_cds!r}). Recompute "
            "without --quality-table."
        )
    quality["eligible"] = quality["eligible"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )
    quality = quality.set_index("dataset_name").loc[list(dataset_mapping)].reset_index()
    exclusions = {
        str(row.dataset_name): str(row.exclusion_reason)
        for row in quality.loc[~quality["eligible"]].itertuples()
    }
    return quality, exclusions, {
        "loaded_quality_table": str(source),
        "quality_manifest_version": manifest_version,
        "sequences_path": str(current_sequences_path),
        "max_cds_codons": args.max_cds_codons,
    }


def _panel_name(index: int) -> str:
    return f"panel_{index:02d}"


def _resolved_panel_config(
    *,
    base: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    run_name: str,
    panel_name: str,
    panel_datasets: Sequence[str],
    seed: int,
    split_manifest: Path,
    reliability_manifest: Path | None,
    sequences_path: Path,
    panel_directory: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(base))
    resolved["dataset_config"] = copy.deepcopy(dict(dataset_config))
    resolved["name"] = run_name
    overrides = {
        "experiment.from_checkpoint": False,
        "experiment.train": True,
        "experiment.predict": True,
        "experiment.dataset": list(map(str, panel_datasets)),
        "experiment.seed": int(seed),
        "prediction.checkpoint_variants": ["best_val_loss"],
        "split.master_dataset_universe": list(map(str, panel_datasets)),
        "split.external_manifest": str(split_manifest),
        "split.external_panel_name": panel_name,
        "data.train_sampling_strategy": "transcript_grouped_multidataset_pairs",
        "data.minimum_positive_datasets_per_transcript": 2,
        "data.batch_size": int(args.batch_size),
        "data.reliability_reference_manifest": (
            str(reliability_manifest) if reliability_manifest is not None else None
        ),
        "data.num_workers": int(args.num_workers),
        "data.predict_num_workers": int(args.predict_num_workers),
        "data.multiprocessing_context": str(args.multiprocessing_context),
        "model.mass_conservation": False,
        "model.alpha_mode": "learned",
        "model.gamma_centering.mode": "fixed_reference",
        "model.gamma_centering.reference.dataset_names": None,
        "model.gamma_centering.reference.weighting": "equal",
        "model.gamma_centering.reference.quality_rank_power": 0.0,
        "model.gamma_centering.reference.chunk_size": int(args.reference_chunk_size),
        "training.execution_microbatching.max_pair_rows_per_forward": int(
            args.max_pair_rows_per_forward
        ),
        "training.execution_microbatching.max_padded_codon_tokens_per_forward": int(
            args.max_padded_codon_tokens_per_forward
        ),
        "training.grouped_optimizer_batch.target_unique_transcripts_per_optimizer_step": 32,
        "training.grouped_optimizer_batch.auto_accumulate_grad_batches": True,
        "training.grouped_optimizer_batch.max_accumulate_grad_batches": 32,
        "trainer.log_every_n_steps": int(args.log_every_n_steps),
        "metrics.log_validation_transcript_mu_pcc_distribution": False,
        "metrics.log_example_plot": False,
        "model.dataset_bias_params.raw_log_gamma_bound": float(
            args.raw_log_gamma_bound
        ),
        "loss.experiment_mode": "standard_nb",
        "loss.nb_mean_gradient_beta": 0.0,
        "loss.sample_reduction": "transcript_balanced",
        "callbacks.save_best_pcc_checkpoint": False,
        "trainer.devices": [0],
        "trainer.use_distributed_sampler": False,
        "paths.sequences_path": str(sequences_path),
        "paths.checkpoints": str(panel_directory / "checkpoints"),
        "paths.logs": str(panel_directory / "logs"),
        "paths.results": str(panel_directory / "predictions"),
    }
    if args.max_cds_codons is not None:
        overrides["data.max_cds_codons"] = int(args.max_cds_codons)
    for dotted_path, value in overrides.items():
        _set_nested(resolved, dotted_path, value)
    resolved["orchestrator"] = {
        "experiment": "independent_dataset_panel_convergence",
        "panel": panel_name,
        "run_name": run_name,
        "created_at_utc": utc_timestamp(),
        "controlled_overrides": overrides,
        "reliability_weight_mode": str(args.reliability_weight_mode),
    }
    return resolved


def _hydra_list(values: Sequence[str]) -> str:
    return json.dumps(list(map(str, values)), separators=(",", ":"))


def _training_command(
    *,
    args: argparse.Namespace,
    run_name: str,
    panel_name: str,
    panel_datasets: Sequence[str],
    split_manifest: Path,
    reliability_manifest: Path | None,
    sequences_path: Path,
    panel_directory: Path,
) -> list[str]:
    command = [
        str(args.python_executable),
        "-u",
        str(_absolute(args.entrypoint)),
        f"--config-path={_absolute(args.config).parent}",
        f"--config-name={_absolute(args.config).stem}",
        f"dataset_config={_absolute(args.dataset_config).stem}",
        f"name={run_name}",
        f"experiment.dataset={_hydra_list(panel_datasets)}",
        f"experiment.seed={int(args.seed)}",
        "experiment.from_checkpoint=false",
        "experiment.train=true",
        "experiment.predict=true",
        "prediction.checkpoint_variants=[best_val_loss]",
        f"split.master_dataset_universe={_hydra_list(panel_datasets)}",
        f"split.external_manifest={split_manifest}",
        f"split.external_panel_name={panel_name}",
        "data.train_sampling_strategy=transcript_grouped_multidataset_pairs",
        "data.minimum_positive_datasets_per_transcript=2",
        f"data.batch_size={int(args.batch_size)}",
        (
            f"data.reliability_reference_manifest={reliability_manifest}"
            if reliability_manifest is not None
            else "data.reliability_reference_manifest=null"
        ),
        f"data.num_workers={int(args.num_workers)}",
        f"data.predict_num_workers={int(args.predict_num_workers)}",
        f"data.multiprocessing_context={args.multiprocessing_context}",
        "model.mass_conservation=false",
        "model.alpha_mode=learned",
        "model.gamma_centering.mode=fixed_reference",
        "model.gamma_centering.reference.dataset_names=null",
        "model.gamma_centering.reference.weighting=equal",
        "model.gamma_centering.reference.quality_rank_power=0.0",
        f"model.gamma_centering.reference.chunk_size={int(args.reference_chunk_size)}",
        (
            "training.execution_microbatching.max_pair_rows_per_forward="
            f"{int(args.max_pair_rows_per_forward)}"
        ),
        (
            "training.execution_microbatching.max_padded_codon_tokens_per_forward="
            f"{int(args.max_padded_codon_tokens_per_forward)}"
        ),
        "training.grouped_optimizer_batch.target_unique_transcripts_per_optimizer_step=32",
        "training.grouped_optimizer_batch.auto_accumulate_grad_batches=true",
        "training.grouped_optimizer_batch.max_accumulate_grad_batches=32",
        f"trainer.log_every_n_steps={int(args.log_every_n_steps)}",
        "metrics.log_validation_transcript_mu_pcc_distribution=false",
        "metrics.log_example_plot=false",
        f"model.dataset_bias_params.raw_log_gamma_bound={float(args.raw_log_gamma_bound)}",
        "loss.experiment_mode=standard_nb",
        "loss.nb_mean_gradient_beta=0.0",
        "loss.sample_reduction=transcript_balanced",
        "callbacks.save_best_pcc_checkpoint=false",
        "trainer.devices=[0]",
        "trainer.use_distributed_sampler=false",
        f"paths.sequences_path={sequences_path}",
        f"paths.checkpoints={panel_directory / 'checkpoints'}",
        f"paths.logs={panel_directory / 'logs'}",
        f"paths.results={panel_directory / 'predictions'}",
        f"hydra.run.dir={panel_directory / 'hydra'}",
        "hydra.job.chdir=false",
    ]
    if args.max_cds_codons is not None:
        command.append(f"+data.max_cds_codons={int(args.max_cds_codons)}")
    return command


def _print_balance_report(report: pd.DataFrame) -> None:
    columns = [
        "panel",
        "n_datasets",
        "n_sources",
        "mean_median_read_density",
        "median_median_read_density",
        "mean_median_positive_codon_coverage",
        "median_median_positive_codon_coverage",
        "mean_median_replica_PCC",
        "mean_number_of_eligible_transcripts",
    ]
    available = [column for column in columns if column in report.columns]
    print("\n=== Dataset-panel balance report ===")
    print(report[available].to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    standardized = [
        column
        for column in report.columns
        if column.startswith("max_pairwise_standardized_difference_")
    ]
    if standardized:
        print("\nMaximum pairwise standardized panel-mean differences:")
        for column in standardized:
            print(f"  {column.removeprefix('max_pairwise_standardized_difference_')}: {report[column].iloc[0]:.4f}")


def _contract_rows(
    *,
    assignment: pd.DataFrame,
    candidates: Sequence[str],
    excluded: Mapping[str, str],
    common_split: Mapping[str, Any],
    panel_directories: Sequence[Path],
    reliability_manifests: Mapping[str, Mapping[str, Any]],
    expected_dataset_count: int,
) -> list[dict[str, Any]]:
    retained = assignment["dataset_name"].astype(str).tolist()
    panels = panel_dictionary(assignment)
    heldout = set(common_split["common_validation_ids"]) | set(
        common_split["common_test_ids"]
    )
    train_overlap = any(
        bool(set(ids) & heldout)
        for ids in common_split["panel_train_eligible_ids"].values()
    )
    size_values = [len(values) for values in panels.values()]
    source_panel_counts = assignment.groupby("source_identifier")["panel"].nunique()
    source_families_disjoint = bool((source_panel_counts == 1).all())
    run_manifests = {
        directory.name: json.loads(
            (directory / "run_manifest.json").read_text(encoding="utf-8")
        )
        for directory in panel_directories
    }
    resolved_configs = {
        directory.name: yaml.safe_load(
            (directory / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        for directory in panel_directories
    }
    exact_reference_sets = all(
        set(run_manifests[panel]["fixed_gamma_reference"]["dataset_names"])
        == set(panel_names)
        for panel, panel_names in panels.items()
    )
    uniform_pi = all(
        set(run_manifests[panel]["fixed_gamma_reference"]["pi"])
        == set(panel_names)
        and np.allclose(
            list(run_manifests[panel]["fixed_gamma_reference"]["pi"].values()),
            1.0 / len(panel_names),
        )
        and math.isclose(
            sum(run_manifests[panel]["fixed_gamma_reference"]["pi"].values()),
            1.0,
            abs_tol=1.0e-12,
        )
        for panel, panel_names in panels.items()
    )
    mass_free = all(
        config["model"]["mass_conservation"] is False
        for config in resolved_configs.values()
    )
    fixed_centering = all(
        config["model"]["gamma_centering"]["mode"] == "fixed_reference"
        for config in resolved_configs.values()
    )
    bounded_gamma = all(
        float(config["model"]["dataset_bias_params"]["raw_log_gamma_bound"])
        > 0.0
        for config in resolved_configs.values()
    )
    chunked_reference = all(
        int(config["model"]["gamma_centering"]["reference"]["chunk_size"])
        > 0
        for config in resolved_configs.values()
    )
    classical_nb = all(
        config["loss"]["experiment_mode"] == "standard_nb"
        and float(config["loss"]["nb_mean_gradient_beta"]) == 0.0
        for config in resolved_configs.values()
    )
    best_loss_only = all(
        config["prediction"]["checkpoint_variants"] == ["best_val_loss"]
        and config["callbacks"]["save_best_pcc_checkpoint"] is False
        for config in resolved_configs.values()
    )
    distinct_reliability = all(
        run_manifests[panel]["reliability_weight"]["separate_from_gamma_pi"]
        and (
            (
                manifest.get("weight_mode") == "stored"
                and manifest.get("weight_source")
                == "input_parquet_weight_column"
                and manifest.get("all_loss_weights_strictly_positive") is True
            )
            or manifest.get("heldout_rows_used_for_fitting") == 0
        )
        for panel, manifest in reliability_manifests.items()
    )
    rows = [
        ("dataset panels mutually exclusive", len(retained) == len(set(retained)), "required"),
        (
            "source/publication families mutually exclusive",
            source_families_disjoint,
            "required",
        ),
        ("retained union equals intended selected set", set(retained) == set(candidates) - set(excluded), "required"),
        ("paper dataset count is retained", len(retained) == expected_dataset_count, "required" if not excluded else "warning"),
        ("panel sizes differ by <= 1", max(size_values) - min(size_values) <= 1, "required"),
        ("one common validation ID list", bool(common_split["common_validation_ids"]), "required"),
        ("one common test ID list", bool(common_split["common_test_ids"]), "required"),
        ("no held-out ID is used for training", not train_overlap, "required"),
        ("every training transcript has >=2 panel datasets", bool(common_split["assertions"]["all_train_transcripts_meet_panel_support"]), "required"),
        ("fixed-reference set equals each panel", exact_reference_sets, "required"),
        ("gamma pi is uniform inside each panel", uniform_pi, "required"),
        ("w_dt source is resolved and separate from pi", distinct_reliability, "required"),
        ("model.mass_conservation=false", mass_free, "required"),
        ("gamma centering mode=fixed_reference", fixed_centering, "required"),
        ("raw log-gamma has finite numerical support", bounded_gamma, "required"),
        ("fixed reference is evaluated in bounded chunks", chunked_reference, "required"),
        ("classical NB2 objective (beta=0)", classical_nb, "required"),
        ("checkpoint selection=best_val_loss", best_loss_only, "required"),
        ("output directories are unique", len(panel_directories) == len(set(panel_directories)), "required"),
        ("random seed is recorded", common_split.get("random_seed") is not None, "required"),
    ]
    return [
        {
            "contract": label,
            "status": "PASS" if passed else ("WARN" if severity == "warning" else "FAIL"),
            "severity": severity,
        }
        for label, passed, severity in rows
    ]


def _print_contract(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n=== Experimental contract ===")
    width = max(len(str(row["contract"])) for row in rows)
    for row in rows:
        print(f"[{row['status']:^4}] {str(row['contract']):<{width}}")


def _write_panel_files(
    *,
    args: argparse.Namespace,
    run_root: Path,
    panels: Mapping[str, Sequence[str]],
    panel_sources: Mapping[str, Sequence[str]],
    dataset_mapping: Mapping[str, str],
    base_config: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    common_split: Mapping[str, Any],
    common_split_path: Path,
    sequences_path: Path,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]], list[Path]]:
    commands: dict[str, list[str]] = {}
    reliability_manifests: dict[str, dict[str, Any]] = {}
    panel_directories: list[Path] = []
    for panel_index, (panel_name, panel_datasets) in enumerate(
        sorted(panels.items()), start=1
    ):
        panel_directory = run_root / panel_name
        panel_directory.mkdir(parents=True, exist_ok=True)
        for child in ("checkpoints", "predictions", "logs"):
            (panel_directory / child).mkdir(parents=True, exist_ok=True)
        panel_directories.append(panel_directory)
        selected_path = panel_directory / "selected_datasets.txt"
        selected_path.write_text("\n".join(panel_datasets) + "\n", encoding="utf-8")

        if args.reliability_weight_mode == "train-only-snr":
            reliability_path: Path | None = (
                panel_directory / "reliability_reference_manifest.json"
            )
            reliability_manifest = fit_panel_reliability_manifest(
                experiment_name="independent_dataset_panel_convergence",
                panel_name=panel_name,
                panel_datasets=panel_datasets,
                dataset_mapping=dataset_mapping,
                panel_training_ids=common_split["panel_train_eligible_ids"][panel_name],
                validation_ids=common_split["common_validation_ids"],
                test_ids=common_split["common_test_ids"],
                source_split_manifest=common_split_path,
            )
            reliability_audit_path = reliability_path
            reliability_run_record = {
                "symbol": "w_dt",
                "mode": "train-only-snr",
                "manifest": str(reliability_path),
                "fitting_split": "training_only",
                "source": "frozen_snr_coverage_reference",
                "separate_from_gamma_pi": True,
            }
        else:
            reliability_path = None
            reliability_audit_path = (
                panel_directory / "stored_reliability_weight_manifest.json"
            )
            reliability_manifest = build_panel_stored_weight_manifest(
                experiment_name="independent_dataset_panel_convergence",
                panel_name=panel_name,
                panel_datasets=panel_datasets,
                dataset_mapping=dataset_mapping,
                panel_training_ids=common_split["panel_train_eligible_ids"][panel_name],
                validation_ids=common_split["common_validation_ids"],
                test_ids=common_split["common_test_ids"],
                source_split_manifest=common_split_path,
            )
            reliability_run_record = {
                "symbol": "w_dt",
                "mode": "stored",
                "manifest": str(reliability_audit_path),
                "fitting_split": "precomputed_artifact; no experiment-time refit",
                "source": "input_parquet_weight_column",
                "zero_weight_policy": "excluded before support and model loading",
                "separate_from_gamma_pi": True,
            }
        write_json(reliability_audit_path, reliability_manifest)
        reliability_manifests[panel_name] = reliability_manifest

        per_panel_split = {
            "source_common_split_manifest": str(common_split_path),
            "panel_name": panel_name,
            "train_ids": common_split["panel_train_eligible_ids"][panel_name],
            "validation_ids": common_split["common_validation_ids"],
            "test_ids": common_split["common_test_ids"],
            "fold_id_hashes": {
                "train": transcript_id_hash(common_split["panel_train_eligible_ids"][panel_name]),
                "validation": transcript_id_hash(common_split["common_validation_ids"]),
                "test": transcript_id_hash(common_split["common_test_ids"]),
            },
            "support_statistics": common_split["panel_support_statistics"][panel_name],
            "heldout_train_overlap": 0,
        }
        write_json(panel_directory / "split_manifest.json", per_panel_split)

        run_name = f"real_panel_convergence_panel{panel_index:02d}"
        resolved = _resolved_panel_config(
            base=base_config,
            dataset_config=dataset_config,
            run_name=run_name,
            panel_name=panel_name,
            panel_datasets=panel_datasets,
            seed=args.seed,
            split_manifest=common_split_path,
            reliability_manifest=reliability_path,
            sequences_path=sequences_path,
            panel_directory=panel_directory,
            args=args,
        )
        (panel_directory / "resolved_config.yaml").write_text(
            yaml.safe_dump(json_ready(resolved), sort_keys=False), encoding="utf-8"
        )
        equal_pi = 1.0 / len(panel_datasets)
        write_json(
            panel_directory / "run_manifest.json",
            {
                "experiment_name": run_name,
                "outer_run_identifier": run_root.name,
                "panel_name": panel_name,
                "random_seed": args.seed,
                "selected_datasets": list(panel_datasets),
                "selected_source_families": list(panel_sources[panel_name]),
                "source_families_are_atomic_across_panels": True,
                "fixed_gamma_reference": {
                    "dataset_names": list(panel_datasets),
                    "weighting": "equal",
                    "pi": {name: equal_pi for name in panel_datasets},
                },
                "reliability_weight": reliability_run_record,
                "checkpoint_selection": "best_val_loss",
                "prediction_split": "common_test",
                "common_test_id_hash": transcript_id_hash(common_split["common_test_ids"]),
            },
        )
        command = _training_command(
            args=args,
            run_name=run_name,
            panel_name=panel_name,
            panel_datasets=panel_datasets,
            split_manifest=common_split_path,
            reliability_manifest=reliability_path,
            sequences_path=sequences_path,
            panel_directory=panel_directory,
        )
        commands[panel_name] = command
        (panel_directory / "launch_command.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + shlex.join(command)
            + "\n",
            encoding="utf-8",
        )
    return commands, reliability_manifests, panel_directories


def _launch_one(
    *,
    panel_name: str,
    physical_gpu: str,
    command: Sequence[str],
    panel_directory: Path,
) -> dict[str, Any]:
    temporary = Path(tempfile.mkdtemp(prefix=f"riboai_{panel_name}_", dir="/tmp"))
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "TMPDIR": str(temporary),
            "MPLCONFIGDIR": str(temporary / "matplotlib"),
            "TRITON_CACHE_DIR": str(temporary / "triton"),
            "HYDRA_FULL_ERROR": "1",
            "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            "PYTORCH_ALLOC_CONF": environment.get(
                "PYTORCH_ALLOC_CONF", "expandable_segments:True"
            ),
            "RIBOUNMIX_LOGGER_VERSION": panel_name,
        }
    )
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(environment["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    log_path = panel_directory / "logs" / "launcher.log"
    print(f"[{panel_name}] starting on physical GPU {physical_gpu}", flush=True)
    return_code = 1
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
            process = subprocess.Popen(
                list(command),
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
                print(f"[{panel_name}] {line}", end="", flush=True)
            return_code = int(process.wait())
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\nLauncher exception: {type(exc).__name__}: {exc}\n")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    status = {
        "panel": panel_name,
        "physical_gpu": str(physical_gpu),
        "return_code": return_code,
        "completed": return_code == 0,
        "log_path": str(log_path),
    }
    write_json(panel_directory / "training_status.json", status)
    print(
        f"[{panel_name}] {'completed' if return_code == 0 else 'FAILED'} "
        f"(status {return_code})",
        flush=True,
    )
    return status


def _launch_queued_panels(
    *,
    commands: Mapping[str, Sequence[str]],
    gpus: Sequence[str],
    run_root: Path,
) -> dict[str, dict[str, Any]]:
    queues: dict[str, list[str]] = {gpu: [] for gpu in gpus}
    for index, panel_name in enumerate(sorted(commands)):
        queues[gpus[index % len(gpus)]].append(panel_name)

    def worker(gpu: str, panel_names: Sequence[str]) -> list[dict[str, Any]]:
        results = []
        for panel_name in panel_names:
            try:
                status = _launch_one(
                    panel_name=panel_name,
                    physical_gpu=gpu,
                    command=commands[panel_name],
                    panel_directory=run_root / panel_name,
                )
            except Exception as exc:
                status = {
                    "panel": panel_name,
                    "physical_gpu": str(gpu),
                    "return_code": 1,
                    "completed": False,
                    "launcher_exception": f"{type(exc).__name__}: {exc}",
                }
                write_json(run_root / panel_name / "training_status.json", status)
                print(f"[{panel_name}] launcher FAILED: {exc}", flush=True)
            results.append(status)
        return results

    statuses: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(worker, gpu, panel_names): gpu
            for gpu, panel_names in queues.items()
            if panel_names
        }
        for future in as_completed(futures):
            for status in future.result():
                statuses[str(status["panel"])] = status
    return statuses


def _consolidate_scientific_checkpoints(
    *,
    run_root: Path,
    panels: Mapping[str, Sequence[str]],
    common_test_ids: Sequence[str],
) -> None:
    for panel_name, datasets in panels.items():
        panel_directory = run_root / panel_name
        manifests = sorted(
            (panel_directory / "predictions").rglob(
                "prediction_checkpoint_manifest.json"
            )
        )
        if len(manifests) != 1:
            raise RuntimeError(
                f"Expected one prediction checkpoint manifest for {panel_name}, "
                f"found {len(manifests)}: {manifests}."
            )
        source = json.loads(manifests[0].read_text(encoding="utf-8"))
        if set(source) != {"best_val_loss"}:
            raise RuntimeError(
                f"{panel_name} did not produce only best_val_loss predictions."
            )
        record = source["best_val_loss"]
        if record.get("split_name") != "test":
            raise RuntimeError(f"{panel_name} prediction artifact is not the test split.")
        if record.get("transcript_id_hash") != transcript_id_hash(common_test_ids):
            raise RuntimeError(f"{panel_name} prediction transcript hash mismatch.")
        write_json(
            panel_directory / "scientific_checkpoint_manifest.json",
            {
                "panel_name": panel_name,
                "selected_datasets": list(datasets),
                "selection_rule": "minimum validation loss",
                "checkpoint_variant": "best_val_loss",
                "checkpoint_path": record["checkpoint_path"],
                "prediction_path": record["output_path"],
                "prediction_split": "common_test",
                "test_transcript_id_hash": record["transcript_id_hash"],
                "source_runtime_manifest": str(manifests[0]),
            },
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.config = _absolute(args.config)
    args.dataset_config = _absolute(args.dataset_config)
    args.entrypoint = _absolute(args.entrypoint)
    args.analysis_script = _absolute(args.analysis_script)
    args.output_root = _absolute(args.output_root)
    # Preserve a virtual-environment symlink. Path.resolve() would replace
    # ``.venv/bin/python`` with the base interpreter and defeat venv discovery.
    args.python_executable = args.python_executable.expanduser().absolute()
    if args.num_panels < 2:
        raise ValueError("--num-panels must be at least two.")
    if args.num_workers < 0 or args.predict_num_workers < 0:
        raise ValueError("Worker counts must be non-negative.")
    positive_runtime_options = {
        "--batch-size": args.batch_size,
        "--max-pair-rows-per-forward": args.max_pair_rows_per_forward,
        "--max-padded-codon-tokens-per-forward": (
            args.max_padded_codon_tokens_per_forward
        ),
        "--log-every-n-steps": args.log_every_n_steps,
        "--reference-chunk-size": args.reference_chunk_size,
    }
    invalid_runtime_options = {
        name: value for name, value in positive_runtime_options.items() if value <= 0
    }
    if invalid_runtime_options:
        raise ValueError(
            "Runtime sizing options must be positive; got "
            f"{invalid_runtime_options}."
        )
    if not math.isfinite(args.raw_log_gamma_bound) or args.raw_log_gamma_bound <= 0.0:
        raise ValueError("--raw-log-gamma-bound must be finite and positive.")
    gpus = _parse_gpus(args.gpus)
    base_config = _load_yaml(args.config)
    dataset_config = _load_yaml(args.dataset_config)
    dataset_mapping = load_dataset_mapping(args.dataset_config)
    dataset_mapping = {
        name: str((_absolute(Path(path)) if Path(path).is_absolute() else _absolute(PROJECT_ROOT / path)))
        for name, path in dataset_mapping.items()
    }
    configured_sequence_path = Path(
        args.sequences_path
        if args.sequences_path is not None
        else base_config["paths"]["sequences_path"]
    )
    sequences_path = (
        _absolute(configured_sequence_path)
        if configured_sequence_path.is_absolute()
        else _absolute(PROJECT_ROOT / configured_sequence_path)
    )
    run_id = args.run_id or f"panel_convergence_seed{args.seed}"
    directory_name = f"{run_id}_dry_run" if args.dry_run else run_id
    run_root = args.output_root / directory_name
    if run_root.exists() and not args.overwrite_design:
        raise FileExistsError(
            f"Run directory already exists: {run_root}. Use a new --run-id or "
            "--overwrite-design (which never deletes checkpoints)."
        )
    if (
        run_root.exists()
        and not args.dry_run
        and args.overwrite_design
        and any(run_root.rglob("*.ckpt"))
    ):
        raise FileExistsError(
            "Refusing to overwrite a non-dry design that already contains checkpoints."
        )
    run_root.mkdir(parents=True, exist_ok=True)

    source_contract = _inspect_weighted_dataset_sources(
        dataset_mapping,
        dataset_config_path=args.dataset_config,
    )
    source_contract_path = run_root / "dataset_source_preflight.json"
    write_json(source_contract_path, source_contract)
    print(f"Dataset configuration: {args.dataset_config}")
    print(
        "Required dataset artifact: weighted replica-aware parquet with core "
        f"columns {sorted(REQUIRED_WEIGHTED_DATASET_COLUMNS)}"
    )
    print(
        "Optional audit columns (quality is derived from raw-replica consensus "
        f"when absent): {sorted(OPTIONAL_WEIGHT_AUDIT_COLUMNS)}"
    )
    print(
        "Resolved dataset directories: "
        f"{source_contract['dataset_directories']}"
    )
    print(
        "Weighted dataset source preflight: "
        f"{source_contract['status']} "
        f"({source_contract['passing_dataset_count']}/"
        f"{source_contract['candidate_dataset_count']})"
    )
    schema_variant_counts: dict[str, int] = {}
    for record in source_contract["datasets"]:
        variant = str(record.get("schema_variant") or "invalid")
        schema_variant_counts[variant] = schema_variant_counts.get(variant, 0) + 1
    zero_exclusion_rows = int(
        sum(
            int(record.get("zero_weight_exclusion_row_count") or 0)
            for record in source_contract["datasets"]
        )
    )
    print(f"Weighted artifact schema variants: {schema_variant_counts}")
    print(
        "Legacy zero-weight exclusion rows (never used for support or loss): "
        f"{zero_exclusion_rows}"
    )
    if source_contract["failures"]:
        compact_failures = [
            {
                "dataset_name": record["dataset_name"],
                "resolved_path": record["resolved_path"],
                "missing_required_columns": record["missing_required_columns"],
                "invalid_weight_row_count": record["invalid_weight_row_count"],
                "positive_weight_row_count": record["positive_weight_row_count"],
                "error": record["error"],
            }
            for record in source_contract["failures"][:10]
        ]
        raise RuntimeError(
            "Dataset-source preflight rejected non-production artifacts. "
            f"Details: {compact_failures}. Full report: {source_contract_path}"
        )
    if args.reliability_weight_mode == "train-only-snr":
        compact_schema_datasets = [
            record["dataset_name"]
            for record in source_contract["datasets"]
            if record["missing_optional_audit_columns"]
        ]
        if compact_schema_datasets:
            raise RuntimeError(
                "--reliability-weight-mode=train-only-snr requires materialized "
                "read_density and coverage audit columns. The configured compact "
                "weighted artifacts preserve a historical stored-weight formula, "
                "so the orchestrator will not silently reinterpret them. Regenerate "
                "auditable weights or explicitly request the compatibility mode "
                "--reliability-weight-mode=stored. Affected datasets: "
                f"{compact_schema_datasets[:10]}"
            )

    print(f"Candidate real datasets: {len(dataset_mapping)}")
    quality, exclusions, sequence_report = _load_or_compute_quality(
        args=args,
        dataset_mapping=dataset_mapping,
        sequences_path=sequences_path,
    )
    quality.to_csv(run_root / "dataset_quality_table.csv", index=False)
    if exclusions:
        print("\nExcluded datasets:")
        for name, reason in exclusions.items():
            print(f"  {name}: {reason}")
        write_json(
            run_root / "dataset_exclusions.json",
            {"excluded_datasets": exclusions, "created_at_utc": utc_timestamp()},
        )
        if not args.allow_excluded_datasets:
            write_json(
                run_root / "panel_manifest.json",
                {
                    "experiment_name": "independent_dataset_panel_convergence",
                    "status": "design_failed_before_partition",
                    "all_candidate_datasets": list(dataset_mapping),
                    "excluded_datasets": exclusions,
                    "reason": "Explicit --allow-excluded-datasets was not supplied.",
                },
            )
            raise RuntimeError(
                "One or more datasets are unusable. Reasons were recorded; no "
                "dataset was silently dropped."
            )
    retained_count = int(quality["eligible"].astype(bool).sum())
    if not exclusions and retained_count != int(args.expected_dataset_count):
        raise RuntimeError(
            f"Expected {args.expected_dataset_count} real datasets, found {retained_count}."
        )

    assignment, assignment_method = deterministic_stratified_panel_assignment(
        quality,
        number_of_panels=args.num_panels,
        seed=args.seed,
    )
    assert_panel_partition(
        assignment,
        expected_datasets=quality.loc[quality["eligible"].astype(bool), "dataset_name"].astype(str).tolist(),
    )
    assignment.to_csv(run_root / "panel_assignment.csv", index=False)
    panels = panel_dictionary(assignment)
    panel_sources = {
        str(panel): sorted(group["source_identifier"].astype(str).unique().tolist())
        for panel, group in assignment.groupby("panel", sort=True)
    }
    balance_report = build_panel_balance_report(assignment)
    balance_report.to_csv(run_root / "panel_balance_report.csv", index=False)
    _print_balance_report(balance_report)
    plot_panel_balance(assignment, output_directory=run_root / "figures")

    git = _git_metadata()
    panel_manifest = {
        "manifest_version": 1,
        "experiment_name": "independent_dataset_panel_convergence",
        "outer_run_identifier": run_id,
        "dry_run": bool(args.dry_run),
        "created_at_utc": utc_timestamp(),
        "random_seed": int(args.seed),
        "number_of_panels": int(args.num_panels),
        "all_candidate_datasets": list(dataset_mapping),
        "excluded_datasets": exclusions,
        "retained_datasets": assignment["dataset_name"].astype(str).tolist(),
        "panels": panels,
        "panel_source_families": panel_sources,
        "panel_sizes": {panel: len(names) for panel, names in panels.items()},
        "panel_source_family_counts": {
            panel: len(sources) for panel, sources in panel_sources.items()
        },
        "source_family_partition": {
            "atomic": True,
            "identifier_column": "source_identifier",
            "fallback": "author_year prefix inferred from dataset_name",
            "number_of_source_families": int(
                assignment["source_identifier"].astype(str).nunique()
            ),
            "largest_source_family_dataset_count": int(
                assignment.groupby("source_identifier").size().max()
            ),
        },
        "dataset_quality_summaries": quality.to_dict(orient="records"),
        "stratification_variables": [
            "log1p(median_read_density)",
            "median_positive_codon_coverage",
            "number_of_eligible_transcripts",
            "median_replica_PCC when sufficiently available",
        ],
        "stratification_method": assignment_method,
        "source_configuration_path": str(args.config),
        "source_dataset_configuration_path": str(args.dataset_config),
        "dataset_source_preflight_path": str(source_contract_path),
        "dataset_source_preflight_status": source_contract["status"],
        "reliability_weight_mode": str(args.reliability_weight_mode),
        "training_execution_profile": {
            "logical_per_dataset_batch_size": args.batch_size,
            "target_unique_transcripts_per_optimizer_step": 32,
            "automatic_accumulation": True,
            "max_accumulate_grad_batches": 32,
            "max_pair_rows_per_forward": args.max_pair_rows_per_forward,
            "max_padded_codon_tokens_per_forward": (
                args.max_padded_codon_tokens_per_forward
            ),
            "reference_chunk_size": args.reference_chunk_size,
            "log_every_n_steps": args.log_every_n_steps,
            "heavy_tensorboard_diagnostics": False,
            "intended_gpu": "A100 64GB",
        },
        "source_training_entrypoint": str(args.entrypoint),
        "source_sequences_path": str(sequences_path),
        "sequence_eligibility_report": sequence_report,
        "git": git,
    }
    for index, panel in enumerate(sorted(panels), start=1):
        panel_manifest[f"panel_{index}"] = panels[panel]
    write_json(run_root / "panel_manifest.json", panel_manifest)

    common_split = build_common_transcript_split(
        experiment_name="independent_dataset_panel_convergence",
        dataset_mapping=dataset_mapping,
        panels=panels,
        sequences_path=sequences_path,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        reliability_bins=args.reliability_bins,
        maximum_cds_codons=args.max_cds_codons,
        minimum_panel_support=2,
    )
    common_split_path = run_root / "common_split_manifest.json"
    write_json(common_split_path, common_split)

    commands, reliability_manifests, panel_directories = _write_panel_files(
        args=args,
        run_root=run_root,
        panels=panels,
        panel_sources=panel_sources,
        dataset_mapping=dataset_mapping,
        base_config=base_config,
        dataset_config=dataset_config,
        common_split=common_split,
        common_split_path=common_split_path,
        sequences_path=sequences_path,
    )
    write_json(
        run_root / "launch_manifest.json",
        {
            "dry_run": bool(args.dry_run),
            "gpus": gpus,
            "queue_policy": "panels sorted by name, round-robin to physical GPUs",
            "commands": {panel: shlex.join(command) for panel, command in commands.items()},
        },
    )
    contract = _contract_rows(
        assignment=assignment,
        candidates=list(dataset_mapping),
        excluded=exclusions,
        common_split=common_split,
        panel_directories=panel_directories,
        reliability_manifests=reliability_manifests,
        expected_dataset_count=args.expected_dataset_count,
    )
    pd.DataFrame(contract).to_csv(run_root / "experimental_contract.csv", index=False)
    _print_contract(contract)
    failures = [row for row in contract if row["status"] == "FAIL"]
    if failures:
        raise RuntimeError(f"Experimental contract failed: {failures}")

    print("\n=== Planned panel commands ===")
    for index, panel_name in enumerate(sorted(commands)):
        gpu = gpus[index % len(gpus)]
        print(f"\n{panel_name} -> physical GPU {gpu}\n{shlex.join(commands[panel_name])}")
    if args.dry_run or args.prepare_only:
        mode = "Preparation" if args.prepare_only else "Dry run"
        print(f"\n{mode} complete. Design artifacts: {run_root}")
        print("No training process was launched.")
        return 0

    _validate_requested_gpus(gpus)
    statuses = _launch_queued_panels(commands=commands, gpus=gpus, run_root=run_root)
    write_json(run_root / "training_summary.json", {"panels": statuses})
    failed = sorted(
        panel for panel, status in statuses.items() if not status.get("completed")
    )
    completed = sorted(
        panel for panel, status in statuses.items() if status.get("completed")
    )
    print(f"\nCompleted panels: {completed}")
    print(f"Failed panels:    {failed}")
    if failed or len(statuses) != len(commands):
        return 1

    _consolidate_scientific_checkpoints(
        run_root=run_root,
        panels=panels,
        common_test_ids=common_split["common_test_ids"],
    )
    analysis_command = [
        str(args.python_executable),
        str(args.analysis_script),
        "--run-root",
        str(run_root),
    ]
    print(f"Launching convergence analysis: {shlex.join(analysis_command)}")
    analysis_status = subprocess.run(analysis_command, cwd=PROJECT_ROOT, check=False)
    return int(analysis_status.returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
