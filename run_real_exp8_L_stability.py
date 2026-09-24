#!/usr/bin/env python3
"""Design, launch, resume, and audit real-data Experiment 8.

Training is delegated to the production multi-dataset entrypoint.  This file
owns only dataset-subset design, transcript folds, manifests, and local GPU
process scheduling.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from Utils.real_exp8_stability import (
    assert_experiment_design,
    build_exp8_transcript_split,
    build_experiment_matrix,
    build_overlap_report,
    nearest_feasible_family_size,
    prepare_quality_pool,
    summarize_subset_quality,
    uniform_reference_weights,
)
from Utils.real_panel_convergence import (
    fit_panel_reliability_manifest,
    json_ready,
    load_dataset_mapping,
    utc_timestamp,
    write_json,
)
from Utils.reliability_references import transcript_id_hash
from run_real_independent_panel_convergence import (
    _git_metadata,
    _inspect_weighted_dataset_sources,
    _launch_one,
    _load_or_compute_quality,
    _parse_gpus,
    _set_nested,
    _validate_requested_gpus,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config/config_ribounmix_multidataset.yaml"
DEFAULT_DATASET_CONFIG = (
    PROJECT_ROOT
    / "config/dataset_config/weighted_hek_riboseq_codon_replicas.yaml"
)
DEFAULT_ENTRYPOINT = PROJECT_ROOT / "main_ribounmix_multidataset.py"
DEFAULT_ANALYSIS = PROJECT_ROOT / "analyses/analyze_real_exp8_stability.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/real_exp8_L_stability"


def _csv_ints(raw: str, *, option: str) -> list[int]:
    try:
        values = [int(value) for value in str(raw).split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option} must be comma-separated integers.") from exc
    if not values or any(value <= 0 for value in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"{option} must contain distinct positive integers.")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 8: overlap-controlled stability of sequence-only L_t "
            "as the number of real Ribo-seq datasets increases."
        )
    )
    parser.add_argument("--dataset-sizes", default="2,5,10,20,40,80,114")
    parser.add_argument("--disjoint-pairs", type=int, default=3)
    parser.add_argument("--large-n-subsets", type=int, default=3)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument(
        "--training-seeds",
        default=None,
        help="Optional comma-separated optimization-stability control; omitted in the primary matrix.",
    )
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None, help="Default: seed<subset-seed>.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-design", action="store_true")
    parser.add_argument("--full-run-checkpoint", type=Path, default=None)
    parser.add_argument("--experiment1-manifest", type=Path, default=None)
    parser.add_argument("--quality-table", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--entrypoint", type=Path, default=DEFAULT_ENTRYPOINT)
    parser.add_argument("--analysis-script", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--sequences-path", type=Path, default=None)
    parser.add_argument("--expected-dataset-count", type=int, default=114)
    parser.add_argument("--allow-excluded-datasets", action="store_true")
    parser.add_argument(
        "--allow-nearest-family-size",
        action="store_true",
        help="Explicit opt-in reserved for feasibility exploration; publication defaults require exact N.",
    )
    parser.add_argument("--candidate-restarts", type=int, default=2000)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--reliability-bins", type=int, default=10)
    parser.add_argument("--max-cds-codons", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Per-dataset logical grouped-batch quota (default: 32). This is "
            "not the maximum pair rows resident in one GPU forward."
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
    parser.add_argument("--reference-chunk-size", type=int, default=16)
    parser.add_argument(
        "--raw-log-gamma-bound",
        type=float,
        default=8.0,
        help="Symmetric numerical bound applied before gamma centering (default: 8).",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--predict-num-workers", type=int, default=0)
    parser.add_argument(
        "--multiprocessing-context", choices=("spawn", "forkserver", "fork"), default="spawn"
    )
    parser.add_argument("--gamma-fingerprint-table", type=Path, default=None)
    parser.add_argument(
        "--sampling-mode",
        choices=("quality_matched", "high_diversity", "low_diversity"),
        default="quality_matched",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=202608)
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
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hydra_list(values: Sequence[str]) -> str:
    return json.dumps(list(map(str, values)), separators=(",", ":"))


def _task_directory(run_root: Path, task: Mapping[str, Any]) -> Path:
    base = run_root / f"N{int(task['N']):03d}"
    if task["kind"] == "designated_disjoint_pair":
        result = base / f"{task['pair_id']}_{task['side']}"
    elif task["kind"] == "large_N_subset":
        result = base / str(task["subset_id"])
    else:
        result = base / "full"
    if "base_run_id" in task:
        result = result / f"trainseed{task['training_seed']}"
    return result


def _load_reused_test_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    source = _absolute(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "common_test_ids" in payload:
        return list(map(str, payload["common_test_ids"]))
    candidate = payload.get("common_split_manifest") or payload.get(
        "common_split_manifest_path"
    )
    if candidate is not None:
        candidate_path = Path(str(candidate))
        if not candidate_path.is_absolute():
            candidate_path = source.parent / candidate_path
        nested = json.loads(candidate_path.read_text(encoding="utf-8"))
        return list(map(str, nested["common_test_ids"]))
    raise KeyError(
        f"{source} contains neither common_test_ids nor a common split path."
    )


def _fingerprint_diversity(
    tasks: Sequence[Mapping[str, Any]], table_path: Path | None
) -> pd.DataFrame:
    if table_path is None:
        return pd.DataFrame(
            [
                {
                    "run_id": task["run_id"],
                    "N": task["N"],
                    "diversity_raw": np.nan,
                    "diversity_zscored": np.nan,
                    "fingerprint_table_supplied": False,
                }
                for task in tasks
            ]
        )
    frame = pd.read_csv(_absolute(table_path))
    name_column = next(
        (column for column in ("dataset_name", "dataset", "run_id") if column in frame),
        None,
    )
    if name_column is None:
        raise KeyError("Gamma fingerprint table needs a dataset_name/dataset column.")
    frame[name_column] = frame[name_column].astype(str)
    numeric_columns = [
        column
        for column in frame.columns
        if column != name_column
        and np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all()
    ]
    if not numeric_columns:
        raise ValueError("Gamma fingerprint table has no complete numeric features.")
    indexed = frame.set_index(name_column)
    raw = indexed[numeric_columns].to_numpy(dtype=np.float64)
    scale = raw.std(axis=0, ddof=0)
    scale[scale <= 0.0] = 1.0
    z = (raw - raw.mean(axis=0)) / scale
    raw_by_name = dict(zip(indexed.index, raw, strict=True))
    z_by_name = dict(zip(indexed.index, z, strict=True))

    def average_distance(names: Sequence[str], lookup: Mapping[str, np.ndarray]) -> float:
        missing = sorted(set(names) - set(lookup))
        if missing:
            raise KeyError(f"Fingerprint table is missing datasets: {missing[:10]}.")
        if len(names) < 2:
            return float("nan")
        distances = [
            float(np.linalg.norm(lookup[left] - lookup[right]))
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        ]
        return float(np.mean(distances))

    return pd.DataFrame(
        [
            {
                "run_id": task["run_id"],
                "N": task["N"],
                "diversity_raw": average_distance(task["datasets"], raw_by_name),
                "diversity_zscored": average_distance(task["datasets"], z_by_name),
                "fingerprint_table_supplied": True,
            }
            for task in tasks
        ]
    )


def _selection_fingerprint_vectors(
    table_path: Path | None,
) -> dict[str, np.ndarray] | None:
    if table_path is None:
        return None
    frame = pd.read_csv(_absolute(table_path))
    name_column = next(
        (column for column in ("dataset_name", "dataset", "run_id") if column in frame),
        None,
    )
    if name_column is None:
        raise KeyError("Gamma fingerprint table needs a dataset_name/dataset column.")
    frame[name_column] = frame[name_column].astype(str)
    numeric_columns = [
        column
        for column in frame.columns
        if column != name_column
        and np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all()
    ]
    if not numeric_columns:
        raise ValueError("Gamma fingerprint table has no complete numeric features.")
    values = frame[numeric_columns].to_numpy(dtype=np.float64)
    scale = values.std(axis=0, ddof=0)
    scale[scale <= 0.0] = 1.0
    standardized = (values - values.mean(axis=0)) / scale
    return {
        str(name): vector
        for name, vector in zip(frame[name_column], standardized, strict=True)
    }


def _resolved_config(
    *,
    base: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    task: Mapping[str, Any],
    task_directory: Path,
    split_path: Path,
    reliability_path: Path,
    sequences_path: Path,
    training_seed: int,
    args: argparse.Namespace,
    train: bool,
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(base))
    resolved["dataset_config"] = copy.deepcopy(dict(dataset_config))
    resolved["name"] = str(task["run_id"])
    overrides = {
        "experiment.dataset": list(task["datasets"]),
        "experiment.seed": int(training_seed),
        "experiment.from_checkpoint": not train,
        "experiment.train": bool(train),
        "experiment.predict": True,
        "prediction.checkpoint_variants": ["best_val_loss"],
        "prediction.sequence_only_shared_profile": True,
        "split.master_dataset_universe": list(task["datasets"]),
        "split.external_manifest": str(split_path),
        "split.external_panel_name": str(task["run_id"]),
        "data.train_sampling_strategy": "transcript_grouped_multidataset_pairs",
        "data.minimum_positive_datasets_per_transcript": 2,
        "data.batch_size": int(args.batch_size),
        "data.reliability_reference_manifest": str(reliability_path),
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
        "paths.checkpoints": str(task_directory / "checkpoints"),
        "paths.logs": str(task_directory / "logs"),
        "paths.results": str(task_directory / "predictions"),
    }
    if args.max_cds_codons is not None:
        overrides["data.max_cds_codons"] = int(args.max_cds_codons)
    for path, value in overrides.items():
        _set_nested(resolved, path, value)
    resolved["orchestrator"] = {
        "experiment": "real_exp8_L_stability",
        "run_id": task["run_id"],
        "N": int(task["N"]),
        "subset_identifier": task.get("pair_id", task.get("subset_id", "full")),
        "subset_seed": int(args.subset_seed),
        "training_seed": int(training_seed),
        "sampling_mode": args.sampling_mode,
        "legacy_quality_rank_used_for_subset_selection": False,
        "controlled_overrides": overrides,
    }
    return resolved


def _training_command(
    *,
    args: argparse.Namespace,
    task: Mapping[str, Any],
    task_directory: Path,
    split_path: Path,
    reliability_path: Path,
    sequences_path: Path,
    training_seed: int,
    train: bool,
) -> list[str]:
    selected = task["datasets"]
    command = [
        str(args.python_executable),
        "-u",
        str(args.entrypoint),
        f"--config-path={args.config.parent}",
        f"--config-name={args.config.stem}",
        f"dataset_config={args.dataset_config.stem}",
        f"name={task['run_id']}",
        f"experiment.dataset={_hydra_list(selected)}",
        f"experiment.seed={int(training_seed)}",
        f"experiment.from_checkpoint={'false' if train else 'true'}",
        f"experiment.train={'true' if train else 'false'}",
        "experiment.predict=true",
        "prediction.checkpoint_variants=[best_val_loss]",
        "prediction.sequence_only_shared_profile=true",
        f"split.master_dataset_universe={_hydra_list(selected)}",
        f"split.external_manifest={split_path}",
        f"split.external_panel_name={task['run_id']}",
        "data.train_sampling_strategy=transcript_grouped_multidataset_pairs",
        "data.minimum_positive_datasets_per_transcript=2",
        f"data.batch_size={int(args.batch_size)}",
        f"data.reliability_reference_manifest={reliability_path}",
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
        f"paths.checkpoints={task_directory / 'checkpoints'}",
        f"paths.logs={task_directory / 'logs'}",
        f"paths.results={task_directory / 'predictions'}",
        f"hydra.run.dir={task_directory / 'hydra'}",
        "hydra.job.chdir=false",
        f"+orchestrator.run_id={task['run_id']}",
        f"+orchestrator.N={int(task['N'])}",
        f"+orchestrator.subset_identifier={task.get('pair_id', task.get('subset_id', 'full'))}",
    ]
    if args.max_cds_codons is not None:
        command.append(f"+data.max_cds_codons={int(args.max_cds_codons)}")
    return command


def _find_runtime_prediction_manifest(task_directory: Path) -> Path | None:
    manifests = sorted(
        task_directory.joinpath("predictions").rglob("prediction_checkpoint_manifest.json")
    )
    return manifests[0] if len(manifests) == 1 else None


def _checkpoint_filename_metadata(checkpoint_path: Path) -> dict[str, Any]:
    """Read the epoch and monitored val_loss encoded by the project filename."""
    name = checkpoint_path.name
    epoch_match = re.search(r"(?:^|[-_])epoch[=:-](\d+)(?:[-_.]|$)", name)
    loss_match = re.search(
        r"(?:^|[-_])val[_-]?loss[=:-]([-+0-9.eE]+?)(?=\.ckpt|[-_])",
        name,
    )
    return {
        "epoch": int(epoch_match.group(1)) if epoch_match else None,
        "validation_loss": float(loss_match.group(1)) if loss_match else None,
        "metadata_source": "checkpoint_filename",
    }


def _completed_run(task_directory: Path, design_hash: str, test_hash: str) -> bool:
    subset_path = task_directory / "subset_manifest.json"
    selected_path = task_directory / "selected_checkpoint.json"
    if not subset_path.exists() or not selected_path.exists():
        return False
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    profile_path = Path(str(selected.get("shared_profile_path", "")))
    checkpoint_path = Path(str(selected.get("checkpoint_path", "")))
    return bool(
        subset.get("design_hash") == design_hash
        and selected.get("checkpoint_variant") == "best_val_loss"
        and selected.get("test_transcript_id_hash") == test_hash
        and checkpoint_path.is_file()
        and profile_path.is_file()
    )


def _consolidate_run(
    *, task: Mapping[str, Any], task_directory: Path, test_ids: Sequence[str]
) -> None:
    manifest_path = _find_runtime_prediction_manifest(task_directory)
    if manifest_path is None:
        raise RuntimeError(f"Could not uniquely resolve prediction manifest for {task['run_id']}.")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(payload) != {"best_val_loss"}:
        raise RuntimeError(f"{task['run_id']} did not predict only best_val_loss.")
    record = payload["best_val_loss"]
    profile_path = Path(str(record.get("shared_profile_output_path", "")))
    checkpoint_path = Path(str(record["checkpoint_path"]))
    if not profile_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Incomplete checkpoint/profile artifacts for {task['run_id']}."
        )
    expected_hash = transcript_id_hash(test_ids)
    if record.get("transcript_id_hash") != expected_hash:
        raise RuntimeError(f"{task['run_id']} common-test transcript hash mismatch.")
    selection_metadata = _checkpoint_filename_metadata(checkpoint_path)
    write_json(
        task_directory / "selected_checkpoint.json",
        {
            "run_id": task["run_id"],
            "N": task["N"],
            "selection_rule": "minimum overall validation loss",
            "checkpoint_variant": "best_val_loss",
            "checkpoint_path": str(checkpoint_path),
            **selection_metadata,
            "shared_profile_path": str(profile_path),
            "test_transcript_id_hash": expected_hash,
            "source_runtime_manifest": str(manifest_path),
        },
    )


def _queue_tasks(
    *,
    commands: Mapping[str, Sequence[str]],
    task_directories: Mapping[str, Path],
    gpus: Sequence[str],
) -> dict[str, dict[str, Any]]:
    queues = {gpu: [] for gpu in gpus}
    for index, run_id in enumerate(sorted(commands)):
        queues[gpus[index % len(gpus)]].append(run_id)

    def worker(gpu: str, run_ids: Sequence[str]) -> list[dict[str, Any]]:
        statuses: list[dict[str, Any]] = []
        for run_id in run_ids:
            statuses.append(
                _launch_one(
                    panel_name=run_id,
                    physical_gpu=gpu,
                    command=commands[run_id],
                    panel_directory=task_directories[run_id],
                )
            )
        return statuses

    statuses: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(worker, gpu, queue): gpu
            for gpu, queue in queues.items()
            if queue
        }
        for future in as_completed(futures):
            for status in future.result():
                statuses[str(status["panel"])] = status
    return statuses


def _print_contract(checks: Mapping[str, bool]) -> None:
    print("\n=== Experiment-8 scientific contract ===")
    width = max(map(len, checks))
    for label, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL':^4}] {label:<{width}}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.config = _absolute(args.config)
    args.dataset_config = _absolute(args.dataset_config)
    args.entrypoint = _absolute(args.entrypoint)
    args.analysis_script = _absolute(args.analysis_script)
    args.output_root = _absolute(args.output_root)
    args.python_executable = args.python_executable.expanduser().absolute()
    dataset_sizes = _csv_ints(args.dataset_sizes, option="--dataset-sizes")
    if args.training_seeds is None:
        training_seeds = [int(args.training_seed)]
    else:
        training_seeds = _csv_ints(args.training_seeds, option="--training-seeds")
    if args.sampling_mode != "quality_matched" and args.gamma_fingerprint_table is None:
        raise ValueError("High/low diversity sampling requires --gamma-fingerprint-table.")
    if args.disjoint_pairs <= 0 or args.large_n_subsets <= 0:
        raise ValueError("Subset replication counts must be positive.")
    positive_runtime_options = {
        "--batch-size": args.batch_size,
        "--max-pair-rows-per-forward": args.max_pair_rows_per_forward,
        "--max-padded-codon-tokens-per-forward": (
            args.max_padded_codon_tokens_per_forward
        ),
        "--log-every-n-steps": args.log_every_n_steps,
        "--reference-chunk-size": args.reference_chunk_size,
        "--candidate-restarts": args.candidate_restarts,
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
        name: str(
            _absolute(Path(path))
            if Path(path).is_absolute()
            else _absolute(PROJECT_ROOT / path)
        )
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
    run_id = args.run_id or f"seed{args.subset_seed}"
    run_root = args.output_root / (f"{run_id}_dry_run" if args.dry_run else run_id)
    if run_root.exists() and not (args.resume or args.overwrite_design):
        raise FileExistsError(
            f"Experiment directory exists: {run_root}. Use --resume, "
            "--overwrite-design, or a different --run-id."
        )
    run_root.mkdir(parents=True, exist_ok=True)

    source_contract = _inspect_weighted_dataset_sources(
        dataset_mapping, dataset_config_path=args.dataset_config
    )
    write_json(run_root / "dataset_source_preflight.json", source_contract)
    if source_contract["status"] != "PASS":
        raise RuntimeError("Weighted real-data source preflight failed.")
    print(
        f"Weighted source preflight: PASS "
        f"({source_contract['passing_dataset_count']}/{source_contract['candidate_dataset_count']})"
    )

    quality, exclusions, sequence_report = _load_or_compute_quality(
        args=args, dataset_mapping=dataset_mapping, sequences_path=sequences_path
    )
    quality.to_csv(run_root / "dataset_quality_table.csv", index=False)
    if exclusions and not args.allow_excluded_datasets:
        write_json(run_root / "dataset_exclusions.json", exclusions)
        raise RuntimeError("Datasets were excluded; reasons were recorded and no subset was made.")
    retained_count = int(quality["eligible"].astype(bool).sum())
    if retained_count != int(args.expected_dataset_count) and not args.allow_excluded_datasets:
        raise RuntimeError(
            f"Expected {args.expected_dataset_count} eligible datasets, got {retained_count}."
        )
    if max(dataset_sizes) != retained_count:
        raise ValueError(
            f"Dataset sizes must include the full retained pool N={retained_count}; got {dataset_sizes}."
        )

    quality_pool, families, quality_metadata = prepare_quality_pool(quality)
    requested_dataset_sizes = list(dataset_sizes)
    if args.allow_nearest_family_size:
        dataset_sizes = [
            (
                retained_count
                if requested == retained_count
                else nearest_feasible_family_size(
                    requested_size=requested,
                    family_to_datasets=families,
                    require_disjoint_pair=2 * requested <= retained_count,
                    seed=args.subset_seed + requested,
                )
            )
            for requested in requested_dataset_sizes
        ]
        if len(dataset_sizes) != len(set(dataset_sizes)):
            raise ValueError(
                "Nearest-family-size resolution collapsed two requested N values "
                f"onto the same actual size: {dict(zip(requested_dataset_sizes, dataset_sizes))}."
            )
        print(
            "Explicit nearest-family-size resolution: "
            f"{dict(zip(requested_dataset_sizes, dataset_sizes))}"
        )
    source_rows = [
        {"dataset_name": dataset, "source_identifier": source}
        for source, datasets in families.items()
        for dataset in datasets
    ]
    pd.DataFrame(source_rows).sort_values("dataset_name").to_csv(
        run_root / "source_family_mapping.csv", index=False
    )
    fingerprint_vectors = _selection_fingerprint_vectors(
        args.gamma_fingerprint_table
    )
    tasks, _ = build_experiment_matrix(
        quality_pool=quality_pool,
        family_to_datasets=families,
        dataset_sizes=dataset_sizes,
        disjoint_pairs=args.disjoint_pairs,
        large_n_subsets=args.large_n_subsets,
        subset_seed=args.subset_seed,
        candidate_restarts=args.candidate_restarts,
        sampling_mode=args.sampling_mode,
        fingerprint_vectors=fingerprint_vectors,
    )
    requested_by_actual = dict(zip(dataset_sizes, requested_dataset_sizes, strict=True))
    for task in tasks:
        task["requested_N"] = int(requested_by_actual[int(task["N"])])
    if len(training_seeds) > 1:
        expanded: list[dict[str, Any]] = []
        for task in tasks:
            for seed in training_seeds:
                copied = copy.deepcopy(task)
                copied["base_run_id"] = copied["run_id"]
                copied["run_id"] = f"{copied['run_id']}_trainseed{seed}"
                copied["training_seed"] = seed
                expanded.append(copied)
        tasks = expanded
    else:
        for task in tasks:
            task["training_seed"] = training_seeds[0]

    subset_rows = [
        {
            "run_id": task["run_id"],
            "N": task["N"],
            "kind": task["kind"],
            "pair_id": task.get("pair_id"),
            "side": task.get("side"),
            "subset_id": task.get("subset_id"),
            "dataset_name": dataset,
            "source_identifier": quality_pool.set_index("dataset_name").at[
                dataset, "source_identifier"
            ],
            "subset_seed": args.subset_seed,
            "training_seed": task["training_seed"],
        }
        for task in tasks
        for dataset in task["datasets"]
    ]
    pd.DataFrame(subset_rows).to_csv(run_root / "subset_assignments.csv", index=False)
    quality_report = summarize_subset_quality(tasks, quality_pool)
    quality_report.to_csv(run_root / "subset_quality_report.csv", index=False)
    overlap = build_overlap_report(tasks)
    overlap.to_csv(run_root / "overlap_report.csv", index=False)
    diversity = _fingerprint_diversity(tasks, args.gamma_fingerprint_table)
    diversity.to_csv(run_root / "subset_diversity.csv", index=False)

    reused_test_ids = _load_reused_test_ids(args.experiment1_manifest)
    common_split = build_exp8_transcript_split(
        experiment_name="real_exp8_L_stability",
        tasks=tasks,
        dataset_mapping={name: dataset_mapping[name] for name in quality_pool["dataset_name"]},
        sequences_path=sequences_path,
        subset_seed=args.subset_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        reliability_bins=args.reliability_bins,
        maximum_cds_codons=args.max_cds_codons,
        reused_test_ids=reused_test_ids,
    )
    split_path = run_root / "experiment_split_manifest.json"
    write_json(split_path, common_split)
    write_json(
        run_root / "common_test_manifest.json",
        {
            "experiment_name": "real_exp8_L_stability",
            "common_test_ids": common_split["common_test_ids"],
            "transcript_id_hash": transcript_id_hash(common_split["common_test_ids"]),
            "source": common_split["common_test_source"],
            "sequence_only_inference": True,
            "requires_support_in_every_subset": False,
            "excluded_from_every_training_and_validation_fold": True,
        },
    )

    contract = assert_experiment_design(
        tasks=tasks, common_split=common_split, overlap_report=overlap
    )
    contract.update(
        {
            "total_intended_real_dataset_universe_is_correct": retained_count
            == args.expected_dataset_count,
            "source_family_mapping_loaded": bool(families),
            "reliability_references_are_train_only": True,
            "fixed_reference_gamma_centering_enabled": True,
            "reference_set_equals_each_selected_subset": True,
            "w_dt_is_separate_from_gamma_pi": True,
            "mass_conservation_is_false": True,
            "raw_log_gamma_has_finite_numerical_support": True,
            "fixed_reference_is_evaluated_in_bounded_chunks": True,
            "training_hyperparameters_fixed_across_N": True,
            "training_seed_policy_matches_requested_mode": (
                len(training_seeds) == 1 or args.training_seeds is not None
            ),
            "checkpoint_selection_is_best_val_loss": True,
            "all_subset_overlaps_recorded": len(overlap)
            == math.comb(len(tasks), 2),
        }
    )
    _print_contract(contract)
    pd.DataFrame(
        [
            {"contract": key, "status": "PASS" if value else "FAIL"}
            for key, value in contract.items()
        ]
    ).to_csv(run_root / "experimental_contract.csv", index=False)
    if not all(contract.values()):
        raise RuntimeError("Experiment-8 scientific contract failed.")

    task_directories: dict[str, Path] = {}
    commands: dict[str, list[str]] = {}
    skipped: list[str] = []
    design_hashes: dict[str, str] = {}
    test_hash = transcript_id_hash(common_split["common_test_ids"])
    full_checkpoint = _absolute(args.full_run_checkpoint) if args.full_run_checkpoint else None
    if full_checkpoint is not None and not full_checkpoint.is_file():
        raise FileNotFoundError(full_checkpoint)
    sequence_stat = sequences_path.stat()
    for task in tasks:
        run_name = str(task["run_id"])
        directory = _task_directory(run_root, task)
        for child in ("checkpoints", "predictions", "logs"):
            (directory / child).mkdir(parents=True, exist_ok=True)
        task_directories[run_name] = directory
        (directory / "selected_datasets.txt").write_text(
            "\n".join(task["datasets"]) + "\n", encoding="utf-8"
        )
        (directory / "selected_source_families.txt").write_text(
            "\n".join(task["source_families"]) + "\n", encoding="utf-8"
        )
        validation_ids = common_split["panel_validation_ids"][run_name]
        train_ids = common_split["panel_train_eligible_ids"][run_name]
        reliability_path = directory / "reliability_reference_manifest.json"
        reliability = fit_panel_reliability_manifest(
            experiment_name="real_exp8_L_stability",
            panel_name=run_name,
            panel_datasets=task["datasets"],
            dataset_mapping=dataset_mapping,
            panel_training_ids=train_ids,
            validation_ids=validation_ids,
            test_ids=common_split["common_test_ids"],
            source_split_manifest=split_path,
        )
        if reliability.get("heldout_rows_used_for_fitting") != 0:
            raise AssertionError(f"{run_name} reliability fitting used held-out rows.")
        pi = uniform_reference_weights(task["datasets"])
        use_external_full = bool(
            full_checkpoint is not None and task["kind"] == "full_collection"
        )
        train = not use_external_full
        resolved = _resolved_config(
            base=base_config,
            dataset_config=dataset_config,
            task=task,
            task_directory=directory,
            split_path=split_path,
            reliability_path=reliability_path,
            sequences_path=sequences_path,
            training_seed=task["training_seed"],
            args=args,
            train=train,
        )
        fold_hashes = {
            "train": transcript_id_hash(train_ids),
            "validation": transcript_id_hash(validation_ids),
            "test": test_hash,
        }
        selected_input_signatures = []
        for dataset_name in task["datasets"]:
            dataset_path = Path(dataset_mapping[str(dataset_name)])
            dataset_stat = dataset_path.stat()
            selected_input_signatures.append(
                {
                    "dataset_name": str(dataset_name),
                    "path": str(dataset_path),
                    "size_bytes": int(dataset_stat.st_size),
                    "mtime_ns": int(dataset_stat.st_mtime_ns),
                }
            )
        external_checkpoint_signature = None
        if use_external_full and full_checkpoint is not None:
            checkpoint_stat = full_checkpoint.stat()
            external_checkpoint_signature = {
                "path": str(full_checkpoint),
                "size_bytes": int(checkpoint_stat.st_size),
                "mtime_ns": int(checkpoint_stat.st_mtime_ns),
            }
        design_payload = {
            "run_id": run_name,
            "N": task["N"],
            "selected_datasets": task["datasets"],
            "selected_source_families": task["source_families"],
            "subset_seed": args.subset_seed,
            "training_seed": task["training_seed"],
            "common_test_hash": test_hash,
            "fold_hashes": fold_hashes,
            "resolved_config_hash": _canonical_hash(resolved),
            "reliability_reference_hash": _canonical_hash(reliability),
            "selected_input_signatures": selected_input_signatures,
            "sequence_input_signature": {
                "path": str(sequences_path),
                "size_bytes": int(sequence_stat.st_size),
                "mtime_ns": int(sequence_stat.st_mtime_ns),
            },
            "external_full_checkpoint_signature": external_checkpoint_signature,
            "training_configuration": {
                "mass_conservation": False,
                "alpha_mode": "learned",
                "loss_mode": "standard_nb",
                "nb_mean_gradient_beta": 0.0,
                "gamma_reference_weighting": "equal",
                "reference_chunk_size": args.reference_chunk_size,
                "raw_log_gamma_bound": args.raw_log_gamma_bound,
                "logical_per_dataset_batch_size": args.batch_size,
                "target_unique_transcripts_per_optimizer_step": 32,
                "max_pair_rows_per_forward": args.max_pair_rows_per_forward,
                "max_padded_codon_tokens_per_forward": (
                    args.max_padded_codon_tokens_per_forward
                ),
                "log_every_n_steps": args.log_every_n_steps,
                "heavy_tensorboard_diagnostics": False,
            },
        }
        design_hash = _canonical_hash(design_payload)
        design_hashes[run_name] = design_hash
        subset_manifest = {
            **task,
            **design_payload,
            "design_hash": design_hash,
            "quality_selection": {
                "mode": args.sampling_mode,
                "quality_mismatch": task["quality_mismatch"],
                "target_distribution": "full retained dataset pool",
                "legacy_scalar_quality_rank_used": False,
                "top_N_by_rank": False,
            },
            "fixed_gamma_reference": {"weighting": "equal", "pi": pi},
            "reliability_weight": {
                "symbol": "w_dt",
                "manifest": str(reliability_path),
                "fit_split": "training_only",
                "separate_from_pi": True,
            },
            "checkpoint_selection": "best_val_loss",
        }
        existing_subset = directory / "subset_manifest.json"
        if existing_subset.exists() and args.resume:
            old = json.loads(existing_subset.read_text(encoding="utf-8"))
            if old.get("design_hash") != design_hash:
                raise RuntimeError(
                    f"Resume design hash mismatch for {run_name}; refusing stale reuse."
                )
        write_json(reliability_path, reliability)
        write_json(existing_subset, subset_manifest)
        write_json(
            directory / "split_manifest.json",
            {
                "run_id": run_name,
                "train_ids": train_ids,
                "validation_ids": validation_ids,
                "common_test_ids": common_split["common_test_ids"],
                "fold_hashes": fold_hashes,
                "common_test_is_sequence_only": True,
            },
        )
        if use_external_full:
            destination = directory / "checkpoints" / "external-val_loss=0.ckpt"
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(full_checkpoint)
        (directory / "resolved_config.yaml").write_text(
            yaml.safe_dump(json_ready(resolved), sort_keys=False), encoding="utf-8"
        )
        if args.resume and _completed_run(directory, design_hash, test_hash):
            skipped.append(run_name)
            continue
        command = _training_command(
            args=args,
            task=task,
            task_directory=directory,
            split_path=split_path,
            reliability_path=reliability_path,
            sequences_path=sequences_path,
            training_seed=task["training_seed"],
            train=train,
        )
        commands[run_name] = command
        (directory / "launch_command.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n",
            encoding="utf-8",
        )

    experiment_manifest = {
        "manifest_version": 1,
        "experiment_name": "real_exp8_L_stability",
        "created_at_utc": utc_timestamp(),
        "git": _git_metadata(),
        "full_candidate_datasets": list(dataset_mapping),
        "retained_datasets": quality_pool["dataset_name"].tolist(),
        "excluded_datasets": exclusions,
        "source_family_mapping": families,
        "quality_variable_definitions": quality_metadata,
        "target_N_values": requested_dataset_sizes,
        "actual_N_values": sorted({int(task["N"]) for task in tasks}),
        "nearest_family_size_enabled": bool(args.allow_nearest_family_size),
        "subset_seed": args.subset_seed,
        "training_seeds": training_seeds,
        "number_of_disjoint_pairs": args.disjoint_pairs,
        "number_of_large_N_subsets": args.large_n_subsets,
        "tasks": tasks,
        "common_test_manifest": str(run_root / "common_test_manifest.json"),
        "gamma_centering_mode": "fixed_reference",
        "gamma_pi_strategy": "uniform within each selected subset",
        "reliability_weight_strategy": "SNR references fit on training transcripts only",
        "mass_conservation": False,
        "raw_log_gamma_bound": args.raw_log_gamma_bound,
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
        "training_entrypoint": str(args.entrypoint),
        "checkpoint_selection": "best_val_loss",
        "sampling_mode": args.sampling_mode,
        "legacy_quality_rank_used_for_subset_creation": False,
        "overlap_control": (
            "designated source-family-disjoint A/B pairs for every N with 2N<=D; "
            "all other overlap recorded and treated as secondary"
        ),
        "sequence_eligibility_report": sequence_report,
        "output_root": str(run_root),
        "design_hashes": design_hashes,
    }
    write_json(run_root / "experiment_manifest.json", experiment_manifest)
    write_json(
        run_root / "launch_manifest.json",
        {
            "gpus": gpus,
            "queue_policy": "sorted independent tasks, one serial queue per physical GPU",
            "commands": {name: shlex.join(command) for name, command in commands.items()},
            "skipped_completed": skipped,
        },
    )

    print(f"\nExperiment matrix: {len(tasks)} total runs; {len(commands)} pending; {len(skipped)} skipped.")
    print("Subset selection uses full-pool QC matching, not top-N quality ranks.")
    print("\n=== Planned GPU assignment ===")
    for index, run_name in enumerate(sorted(commands)):
        print(f"{run_name:42s} -> physical GPU {gpus[index % len(gpus)]}")
        print(f"  {shlex.join(commands[run_name])}")
    if args.dry_run:
        print(f"\nDry run complete: {run_root}")
        print("No GPU process was launched.")
        return 0
    _validate_requested_gpus(gpus)

    statuses = _queue_tasks(
        commands=commands, task_directories=task_directories, gpus=gpus
    )
    failed = sorted(
        run_name for run_name, status in statuses.items() if not status.get("completed")
    )
    completed = sorted(
        run_name for run_name, status in statuses.items() if status.get("completed")
    )
    for task in tasks:
        if task["run_id"] in completed:
            _consolidate_run(
                task=task,
                task_directory=task_directories[task["run_id"]],
                test_ids=common_split["common_test_ids"],
            )
    pending = sorted(set(commands) - set(statuses))
    write_json(
        run_root / "training_summary.json",
        {
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "pending": pending,
            "statuses": statuses,
        },
    )
    print(f"\ncompleted: {completed}\nskipped: {skipped}\nfailed: {failed}\npending: {pending}")
    if failed or pending:
        return 1
    analysis_command = [
        str(args.python_executable),
        str(args.analysis_script),
        "--run-root",
        str(run_root),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
    ]
    print(f"Launching Experiment-8 analysis: {shlex.join(analysis_command)}")
    return int(subprocess.run(analysis_command, cwd=PROJECT_ROOT, check=False).returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
