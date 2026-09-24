#!/usr/bin/env python3
"""Resume incomplete Experiment-1/Experiment-8 tasks in an existing run tree.

The script reuses each task's saved ``launch_command.sh`` and never rebuilds
dataset panels, transcript splits, or reliability references. Completed
best-validation-loss prediction artifacts are skipped. Interrupted tasks use
their most advanced usable checkpoint; corrupt files are rejected, full-state
checkpoints resume exactly, and historical weights-only checkpoints require
the explicit warm-resume opt-in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import copy
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from Models.utils.gru_precision import GRU_COMPUTE_POLICY


PROJECT_ROOT = Path(__file__).resolve().parent
RESUME_LAUNCHER_VERSION = "2026-09-11.exp8-runtime-v14-global-batch"


def _csv_strings(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("comma-separated values must be unique")
    return values


def _csv_positive_ints(raw: str) -> tuple[int, ...]:
    strings = _csv_strings(raw)
    try:
        values = tuple(int(value) for value in strings)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "dataset sizes must be comma-separated positive integers"
        ) from exc
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("dataset sizes must be positive")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("dataset sizes must be unique")
    return values


def _nonnegative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument(
        "--gpus", default="0,1",
        help=(
            "Comma-separated numeric GPU IDs, or 'inherit' for a single worker "
            "that preserves Slurm's CUDA_VISIBLE_DEVICES mask verbatim."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument('--global-batch-gpus', type=int, default=None,
                   help='Opt in to one synchronized model across this many visible GPUs; preserve global logical batches. Requires --gpus inherit and unchanged runtime profiles.')
    p.add_argument('--exp8-runtime-profile', choices=('unchanged', 'auto', 'aggressive'),
                   default='unchanged', help='Per-task execution settings from N and visible GPU memory.')
    p.add_argument('--profile-gpu-memory-gib', type=float, default=None,
                   help='GPU memory override for CPU-only dry-run profile inspection.')
    p.add_argument(
        "--use-saved-resolved-config", action="store_true",
        help="Use each task's saved resolved_config.yaml, not today's base-config defaults.",
    )
    p.add_argument(
        "--summary-path", type=Path, default=None,
        help=(
            "Optional per-job summary path. Use distinct paths for concurrent "
            "array elements instead of overwriting the run-root summary."
        ),
    )
    p.add_argument(
        "--schedule-order",
        choices=("manifest", "largest-first", "explicit-first-largest"),
        default="manifest",
        help=(
            "manifest preserves path order; largest-first starts expensive "
            "large-N runs early; explicit-first-largest first schedules exact "
            "--include-run-ids and then the remaining tasks by descending N."
        ),
    )
    p.add_argument(
        "--dataset-sizes",
        type=_csv_positive_ints,
        default=None,
        help=(
            "Run only tasks whose orchestrator N is in this comma-separated "
            "list. When combined with --include-run-ids, selection is their union."
        ),
    )
    p.add_argument(
        "--include-run-ids",
        type=_csv_strings,
        default=None,
        help=(
            "Also run these exact comma-separated run IDs. When combined with "
            "--dataset-sizes, selection is their union."
        ),
    )
    p.add_argument(
        "--allow-weights-only-warm-resume",
        action="store_true",
        help="Required for historical checkpoints lacking Adam/scheduler state.",
    )
    p.add_argument(
        "--throughput-profile", choices=("unchanged", "safe-faster"),
        default="unchanged",
        help=(
            "unchanged preserves the recorded execution limits; safe-faster "
            "only changes execution partitioning and diagnostic logging."
        ),
    )
    p.add_argument("--max-pair-rows-per-forward", type=int, default=None)
    p.add_argument("--max-padded-codon-tokens-per-forward", type=int, default=None)
    p.add_argument("--reference-chunk-size", type=int, default=None)
    p.add_argument("--log-every-n-steps", type=int, default=None)
    p.add_argument(
        "--bias-gru-precision",
        choices=("inherit", "float32"),
        default=None,
        help=(
            "float32 keeps the bias embeddings/GRU/LayerNorm in FP32 under AMP; "
            "heads remain mixed precision. Retains optimizer state and the "
            "objective, not bitwise trajectory equivalence. If omitted, keep "
            "the previous resume's explicit choice or the original run setting."
            " Both GRUs are protected in FP32 under CUDA AMP even with legacy inherit."
        ),
    )
    p.add_argument(
        "--bias-gru-tbptt-window", type=_nonnegative_int, default=None,
        help=(
            "Opt-in biased training gradient: carry bias-GRU hidden states but "
            "detach every K codons per layer/direction (suggested first trial: 1024). "
            "0 restores full BPTT. Keeps the full-CDS loss and optimizer state. "
            "If omitted, retain the previous resume/checkpoint policy."
        ),
    )
    p.add_argument(
        "--capture-gru-failure",
        action="store_true",
        help="Capture the actual failed fused CUDA GRU invocation for isolated replay; no gradient repair.",
    )
    p.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Enable autograd anomaly tracing for a diagnostic rerun; keeps precision unchanged.",
    )
    p.add_argument(
        "--data-num-workers",
        type=_nonnegative_int,
        default=None,
        help=(
            "Override data.num_workers for every selected task. Setting zero "
            "avoids spawn/pickle duplication for very large in-memory datasets."
        ),
    )
    return p.parse_args(argv)


def _gpus(raw: str) -> list[str]:
    if raw.strip() == "inherit":
        return ["inherit"]
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values or any(not x.isdigit() for x in values):
        raise ValueError("--gpus must be comma-separated numeric GPU IDs or 'inherit'.")
    if len(set(values)) != len(values):
        raise ValueError("--gpus must not contain duplicate GPU IDs.")
    return values


def _gpu_environment(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    # Slurm can assign e.g. physical GPU 2, a remapped ordinal 0, or a UUID.
    # Do not replace that allocation with a guessed physical GPU 0.
    if gpu != "inherit":
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def _read_launch_command(path: Path) -> list[str]:
    """Return argv from a saved, shell-formatted training command.

    ``launch_command.sh`` is intentionally stored as a shell command so it is
    easy to inspect and run manually.  ``subprocess.Popen(..., shell=False)``
    instead requires an argv list.  In particular, it must *not* receive one
    giant string: operating systems then interpret the entire command line as
    the name of the executable (yielding ENOENT or ENAMETOOLONG for N=80/114).

    The nested-token pass also accepts a command line accidentally serialized
    as one quoted string by an older launcher version.  We still reject any
    unsupported shell program rather than trying to execute it ambiguously.
    """
    commands: list[str] = []
    pending = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("set "):
            continue
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        commands.append(pending + line)
        pending = ""
    if pending:
        raise ValueError(f"Unterminated line continuation in {path}.")
    if len(commands) != 1:
        raise ValueError(f"Expected one command in {path}, found {len(commands)}.")

    tokens = shlex.split(commands[0], posix=True)
    # A historical bug could save the whole command as one shell-quoted token.
    # Unwrap it once (or repeatedly for a doubly quoted file), but do not use a
    # shell for execution afterwards.
    while len(tokens) == 1 and any(char.isspace() for char in tokens[0]):
        nested = shlex.split(tokens[0], posix=True)
        if nested == tokens:
            break
        tokens = nested
    _validate_command_argv(tokens, source=path, executable_must_exist=False)
    return tokens


def _validate_command_argv(
    command: Sequence[str], *, source: Path, executable_must_exist: bool
) -> None:
    """Fail before scheduling GPUs if a command is not a safe argv list."""
    if isinstance(command, str) or not isinstance(command, Sequence):
        raise TypeError(f"{source}: training command must be an argv sequence, not a string.")
    if len(command) < 3:
        raise ValueError(f"{source}: expected Python, '-u', and an entrypoint; got {list(command)!r}.")
    if not all(isinstance(token, str) and token for token in command):
        raise ValueError(f"{source}: command contains a non-string or empty argv token.")
    executable = command[0]
    if any(char.isspace() for char in executable) or "\x00" in executable:
        raise ValueError(
            f"{source}: executable token is not tokenized: {executable!r}. "
            "Refusing to pass a whole shell command to subprocess."
        )
    if not any(Path(token).name == "main_ribounmix_multidataset.py" for token in command):
        raise ValueError(f"{source}: RiboUnmix training entrypoint is absent from the saved command.")
    if executable_must_exist:
        executable_path = Path(executable)
        if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
            raise FileNotFoundError(
                f"{source}: resolved Python executable is not runnable: {executable_path}"
            )


def _set_override(command: list[str], key: str, value: str) -> None:
    canonical_key = key.lstrip("+")
    matches = [
        i for i, token in enumerate(command)
        if token.partition("=")[0].lstrip("+") == canonical_key
    ]
    if len(matches) > 1:
        raise ValueError(f"Command contains duplicate override {key}.")
    token = f"{key}={value}"
    if matches:
        command[matches[0]] = token
    else:
        command.append(token)


def _make_command_portable(command: list[str], root: Path, task_dir: Path) -> None:
    """Retarget saved absolute paths when a run tree moved to another HPC."""
    # Keep the venv executable path itself; resolve() may replace its symlink
    # with the base interpreter and lose the environment's site-packages.
    command[0] = str(Path(sys.executable).absolute())
    for i, token in enumerate(command):
        if Path(token).name == "main_ribounmix_multidataset.py":
            command[i] = str(PROJECT_ROOT / "main_ribounmix_multidataset.py")
        elif token.startswith("--config-path="):
            command[i] = f"--config-path={PROJECT_ROOT / 'config'}"
    split = (
        root / "experiment_split_manifest.json"
        if (root / "experiment_split_manifest.json").is_file()
        else root / "common_split_manifest.json"
    )
    _set_override(command, "split.external_manifest", str(split))
    reliability = task_dir / "reliability_reference_manifest.json"
    if reliability.is_file():
        _set_override(command, "data.reliability_reference_manifest", str(reliability))
    _set_override(command, "paths.checkpoints", str(task_dir / "checkpoints"))
    _set_override(command, "paths.logs", str(task_dir / "logs"))
    _set_override(command, "paths.results", str(task_dir / "predictions"))
    _set_override(command, "hydra.run.dir", str(task_dir / "hydra"))
    sequences = PROJECT_ROOT / "Datasets/data/sequence/MANE.selection.sequence_embeddings_with_css.parquet"
    if sequences.is_file():
        _set_override(command, "paths.sequences_path", str(sequences))
    _validate_command_argv(
        command,
        source=task_dir / "launch_command.sh",
        executable_must_exist=True,
    )


def _completed_prediction(task_dir: Path) -> Path | None:
    profiles = sorted(task_dir.joinpath("predictions").rglob("common_test_L_profiles.parquet"))
    if profiles and all(path.stat().st_size > 0 for path in profiles):
        return profiles[-1]
    manifests = sorted(task_dir.joinpath("predictions").rglob("prediction_checkpoint_manifest.json"))
    for path in reversed(manifests):
        try:
            record = json.loads(path.read_text(encoding="utf-8")).get("best_val_loss", {})
        except Exception:
            continue
        output = Path(str(record.get("output_path", "")))
        if output.is_file() and output.stat().st_size > 0:
            return output
    predictions = sorted(task_dir.joinpath("predictions").rglob("predictions_main_test_best_val_loss_*.parquet"))
    return predictions[-1] if predictions else None


def _use_saved_resolved_config(command: list[str], task_dir: Path) -> dict[str, Any]:
    """Freeze saved hyperparameters and embedded dataset mapping for continuation.

    The saved config is never overwritten. Its defaults list is removed because
    dataset_config is already embedded; this prevents recomposition from today's
    YAML. Original launch and explicit resume overrides remain authoritative.
    Only a relocated project prefix is translated in the snapshot and argv.
    """
    import yaml

    source = task_dir / "resolved_config.yaml"
    raw = source.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or not isinstance(config.get("dataset_config"), dict):
        raise ValueError(f"{source}: expected a resolved config with embedded dataset_config.")
    original = _read_launch_command(task_dir / "launch_command.sh")
    old_entrypoint = next(Path(token) for token in original
                          if Path(token).name == "main_ribounmix_multidataset.py")
    old_prefix = str(old_entrypoint.parent) + "/"
    new_prefix = str(PROJECT_ROOT) + "/"

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and value.startswith(old_prefix):
            return new_prefix + value[len(old_prefix):]
        return value

    config = relocate(config)
    config.pop("defaults", None)
    target = task_dir / "resume_inputs" / "frozen_config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    frozen_command = []
    for token in command:
        if token.startswith("dataset_config="):
            continue
        if token.startswith("--config-path="):
            token = f"--config-path={target.parent}"
        elif token.startswith("--config-name="):
            token = "--config-name=frozen_config"
        elif "=" in token:
            key, value = token.split("=", 1)
            # The saved YAML is fully resolved, so keys originally introduced
            # with Hydra's single-plus append syntax may already exist in it.
            # Double-plus preserves the intended value while accepting either
            # an existing or an absent key in the frozen snapshot.
            if key.startswith("+") and not key.startswith("++"):
                key = "++" + key[1:]
            token = key + "=" + relocate(value)
        frozen_command.append(token)
    command[:] = frozen_command
    # Compose without importing the training entrypoint, opening data, or
    # initializing a GPU. Catch Hydra override/schema errors before submission.
    from hydra import compose, initialize_config_dir
    overrides = [token for token in command if "=" in token and not token.startswith("--")]
    with initialize_config_dir(version_base=None, config_dir=str(target.parent)):
        compose(config_name="frozen_config", overrides=overrides)
    return {
        "source": str(source), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "snapshot": str(target),
        "snapshot_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "hydra_composition_checked": True,
        "note": "Saved hyperparameters; current repository code. Not a code-version guarantee.",
    }


def _localize_record_path(task_dir: Path, raw: Any, search_root: str) -> Path | None:
    path = Path(str(raw or ""))
    if path.is_file():
        return path
    if path.name:
        matches = sorted(task_dir.joinpath(search_root).rglob(path.name))
        if matches:
            return matches[-1]
    return None


def _consolidate_task(task_dir: Path, run_id: str) -> None:
    """Materialize the scientific selection manifest after a successful retry."""
    manifests = sorted(
        task_dir.joinpath("predictions").rglob("prediction_checkpoint_manifest.json"),
        key=lambda p: p.stat().st_mtime_ns,
    )
    if not manifests:
        return
    payload = json.loads(manifests[-1].read_text(encoding="utf-8"))
    if set(payload) != {"best_val_loss"}:
        return
    record = payload["best_val_loss"]
    checkpoint = _localize_record_path(task_dir, record.get("checkpoint_path"), "checkpoints")
    prediction = _localize_record_path(task_dir, record.get("output_path"), "predictions")
    shared = _localize_record_path(
        task_dir, record.get("shared_profile_output_path"), "predictions"
    )
    if checkpoint is None or prediction is None:
        return
    subset_manifest = task_dir / "subset_manifest.json"
    if subset_manifest.is_file() and shared is not None:
        subset = json.loads(subset_manifest.read_text(encoding="utf-8"))
        selected = {
            "run_id": run_id,
            "N": subset.get("N"),
            "selection_rule": "minimum overall validation loss",
            "checkpoint_variant": "best_val_loss",
            "checkpoint_path": str(checkpoint),
            "shared_profile_path": str(shared),
            "test_transcript_id_hash": record.get("transcript_id_hash"),
            "source_runtime_manifest": str(manifests[-1]),
        }
        (task_dir / "selected_checkpoint.json").write_text(
            json.dumps(selected, indent=2) + "\n", encoding="utf-8"
        )
    run_manifest = task_dir / "run_manifest.json"
    if run_manifest.is_file():
        run = json.loads(run_manifest.read_text(encoding="utf-8"))
        scientific = {
            "panel_name": run.get("panel_name", task_dir.name),
            "selected_datasets": run.get("selected_datasets", []),
            "selection_rule": "minimum validation loss",
            "checkpoint_variant": "best_val_loss",
            "checkpoint_path": str(checkpoint),
            "prediction_path": str(prediction),
            "prediction_split": "common_test",
            "test_transcript_id_hash": record.get("transcript_id_hash"),
            "source_runtime_manifest": str(manifests[-1]),
        }
        (task_dir / "scientific_checkpoint_manifest.json").write_text(
            json.dumps(scientific, indent=2) + "\n", encoding="utf-8"
        )


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hyper_parameters = payload.get("hyper_parameters", {})
    result = {
        "path": str(path),
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", -1)),
        "has_optimizer_state": bool(payload.get("optimizer_states")),
        "has_scheduler_state": bool(payload.get("lr_schedulers")),
        "bias_gru_tbptt_window": hyper_parameters.get("bias_gru_tbptt_window"),
        "campaign_task_contract_sha256": hyper_parameters.get(
            "campaign_task_contract_sha256"
        ),
    }
    del payload
    return result


def _select_resume_checkpoint(
    task_dir: Path,
    expected_task_contract_sha256: str | None = None,
) -> tuple[Path | None, dict[str, Any] | None, list[dict[str, str]]]:
    """Return the most advanced usable checkpoint and report corrupt files.

    Prefer a full-state checkpoint even when a later weights-only diagnostic
    checkpoint exists.  If Slurm terminates a process during checkpoint I/O,
    a zero-byte or truncated ``last.ckpt`` must not abort preparation of every
    other task in the experiment matrix.
    """

    valid: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(task_dir.joinpath("checkpoints").rglob("*.ckpt")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size <= 0:
                raise EOFError("checkpoint file is empty")
            metadata = _checkpoint_metadata(path)
            if (
                expected_task_contract_sha256 is not None
                and metadata["campaign_task_contract_sha256"]
                != expected_task_contract_sha256
            ):
                rejected.append(
                    {
                        "path": str(path),
                        "reason": "checkpoint belongs to a different task contract",
                    }
                )
                continue
        except Exception as exc:
            rejected.append(
                {
                    "path": str(path),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        valid.append((path, metadata))

    if not valid:
        return None, None, rejected

    full_state = [item for item in valid if item[1]["has_optimizer_state"]]
    pool = full_state or valid
    checkpoint, metadata = max(
        pool,
        key=lambda item: (
            int(item[1]["global_step"]),
            int(item[1]["epoch"]),
            item[0].name == "last.ckpt",
            item[0].stat().st_mtime_ns,
        ),
    )
    return checkpoint, metadata, rejected


def _task_id(command: Sequence[str], task_dir: Path) -> str:
    for token in command:
        if token.startswith("name="):
            return token.split("=", 1)[1]
    return task_dir.name


def _command_override(command: Sequence[str], key: str) -> str | None:
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=")
    matches = [
        token.split("=", 1)[1]
        for token in command
        if token.startswith(prefixes)
    ]
    if len(matches) > 1:
        raise ValueError(f"Command contains duplicate override {key}.")
    return matches[0] if matches else None


def _task_dataset_size(
    command: Sequence[str], task_dir: Path, run_id: str
) -> int | None:
    raw = _command_override(command, "orchestrator.N")
    if raw is not None:
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{task_dir}: invalid orchestrator.N override {raw!r}."
            ) from exc

    subset_manifest = task_dir / "subset_manifest.json"
    if subset_manifest.is_file():
        payload = json.loads(subset_manifest.read_text(encoding="utf-8"))
        if payload.get("N") is not None:
            return int(payload["N"])

    match = re.search(r"(?:^|_)N(\d+)(?:_|$)", run_id)
    return int(match.group(1)) if match else None


def _matches_task_filter(
    *,
    run_id: str,
    dataset_size: int | None,
    dataset_sizes: Sequence[int] | None,
    include_run_ids: Sequence[str] | None,
) -> bool:
    """Return true when a task matches either requested inclusion filter."""
    if dataset_sizes is None and include_run_ids is None:
        return True
    return bool(
        (dataset_sizes is not None and dataset_size in set(dataset_sizes))
        or (include_run_ids is not None and run_id in set(include_run_ids))
    )


def _check_runtime_gpu_precision(command: list[str], task_dir: Path) -> None:
    """Do not silently emulate or change saved BF16 training on an older GPU."""
    precision = _command_override(command, 'trainer.precision')
    source = task_dir / 'resolved_config.yaml'
    if precision is None and source.is_file():
        import yaml
        precision = yaml.safe_load(source.read_text()).get('trainer', {}).get('precision')
    if 'bf16' in str(precision).lower() and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError(
            'Saved trainer.precision requires native BF16 support, but the allocated GPU '
            'does not provide it. Request a native-BF16-capable GPU using your cluster\'s '
            'documented GRES/constraint. No automatic precision change is made; reducing '
            'the execution budget or using the auto profile does not fix this incompatibility.'
        )


def _prepare_task(args: argparse.Namespace, root: Path, launch_path: Path) -> dict[str, Any]:
    task_dir = launch_path.parent
    command = _read_launch_command(launch_path)
    _make_command_portable(command, root, task_dir)
    run_id = _task_id(command, task_dir)
    completed = _completed_prediction(task_dir)
    if completed is not None:
        _consolidate_task(task_dir, run_id)
        return {"run_id": run_id, "task_dir": task_dir, "status": "skipped_complete", "artifact": str(completed)}

    runtime_profile = None
    if getattr(args, 'exp8_runtime_profile', 'unchanged') != 'unchanged':
        from Utils.exp8_runtime_profile import execution_profile
        if args.gpus != 'inherit' or args.throughput_profile != 'unchanged':
            raise ValueError('Exp8 runtime profiles require --gpus inherit and --throughput-profile unchanged.')
        n = _task_dataset_size(command, task_dir, run_id)
        if n is None:
            raise ValueError('Cannot resolve N for the Exp8 runtime profile.')
        memory = args.profile_gpu_memory_gib
        if memory is not None and not args.dry_run:
            raise ValueError('--profile-gpu-memory-gib is only allowed with --dry-run.')
        gpu_name = 'dry-run memory override'
        if memory is None:
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ValueError('Runtime profile requires exactly one visible GPU; for CPU dry runs set --profile-gpu-memory-gib.')
            properties = torch.cuda.get_device_properties(0)
            memory = properties.total_memory / 2**30
            gpu_name = properties.name
            _check_runtime_gpu_precision(command, task_dir)
        selected = execution_profile(n, memory, args.exp8_runtime_profile)
        args = copy.copy(args)
        # Explicit execution flags win, without changing the shared design.
        for key, value in selected.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        runtime_profile = dict(profile=args.exp8_runtime_profile, N=n,
                               gpu_name=gpu_name, gpu_memory_gib=memory,
                               settings={key: getattr(args, key) for key in selected},
                               measured_optimum=False)
        print('Exp8 runtime settings: ' + json.dumps(runtime_profile), flush=True)

    checkpoint, metadata, corrupt_checkpoints = _select_resume_checkpoint(task_dir)
    resume_mode = "fresh_start"
    if checkpoint is None and corrupt_checkpoints:
        resume_mode = "fresh_start_no_usable_checkpoint"
    if checkpoint is not None:
        assert metadata is not None
        if metadata["has_optimizer_state"]:
            resume_mode = "exact_full_state"
        else:
            resume_mode = "weights_only_warm_start"
            if not args.allow_weights_only_warm_resume:
                return {
                    "run_id": run_id, "task_dir": task_dir,
                    "status": "blocked_weights_only", "checkpoint": metadata,
                }
        _set_override(command, "experiment.from_checkpoint", "true")
        _set_override(command, "experiment.resume_training_state", "true")
        _set_override(command, "experiment.resume_checkpoint_path", str(checkpoint.resolve()))
        _set_override(
            command, "experiment.allow_weights_only_resume",
            "true" if args.allow_weights_only_warm_resume else "false",
        )
        # The first continuation from a historical weights-only file cannot
        # restore epoch state. Limit that new segment to the originally
        # remaining epochs instead of silently granting another full 200.
        chain_path = task_dir / "resume_chain.json"
        if chain_path.exists():
            chain = json.loads(chain_path.read_text(encoding="utf-8"))
            segment_max_epochs = int(chain["segment_max_epochs"])
        elif resume_mode == "weights_only_warm_start":
            original_max_epochs = 200
            segment_max_epochs = max(1, original_max_epochs - metadata["epoch"] - 1)
            chain = {
                "historical_checkpoint": metadata,
                "original_max_epochs": original_max_epochs,
                "segment_max_epochs": segment_max_epochs,
                "limitation": "Adam, scheduler, and early-stopping states were absent.",
            }
            chain_path.write_text(json.dumps(chain, indent=2) + "\n", encoding="utf-8")
        else:
            segment_max_epochs = None
        if segment_max_epochs is not None:
            _set_override(command, "trainer.max_epochs", str(segment_max_epochs))

    if args.throughput_profile == "safe-faster":
        pair_rows = args.max_pair_rows_per_forward or 256
        padded_tokens = args.max_padded_codon_tokens_per_forward or 128000
        reference_chunk = args.reference_chunk_size or 8
        log_steps = args.log_every_n_steps or 25
        _set_override(command, "training.execution_microbatching.max_pair_rows_per_forward", str(pair_rows))
        _set_override(command, "training.execution_microbatching.max_padded_codon_tokens_per_forward", str(padded_tokens))
        _set_override(command, "model.gamma_centering.reference.chunk_size", str(reference_chunk))
        _set_override(command, "trainer.log_every_n_steps", str(log_steps))
        _set_override(command, "trainer.num_sanity_val_steps", "0")
        _set_override(command, "metrics.log_example_plot", "false")
        _set_override(
            command, "metrics.log_validation_transcript_mu_pcc_distribution", "false"
        )
    else:
        if args.max_pair_rows_per_forward is not None:
            _set_override(command, "training.execution_microbatching.max_pair_rows_per_forward", str(args.max_pair_rows_per_forward))
        if args.max_padded_codon_tokens_per_forward is not None:
            _set_override(command, "training.execution_microbatching.max_padded_codon_tokens_per_forward", str(args.max_padded_codon_tokens_per_forward))
        if args.reference_chunk_size is not None:
            _set_override(command, "model.gamma_centering.reference.chunk_size", str(args.reference_chunk_size))
        if args.log_every_n_steps is not None:
            _set_override(command, "trainer.log_every_n_steps", str(args.log_every_n_steps))

    if args.reference_chunk_size is not None or args.throughput_profile == 'safe-faster':
        # The checkpoint stores the old chunk size in model extra state. Make
        # the execution override survive both weight and full-state loading.
        _set_override(command, '++model.gamma_centering.reference.chunk_size_override_on_load', 'true')
    if args.data_num_workers is not None:
        _set_override(command, "data.num_workers", str(args.data_num_workers))
    if getattr(args, "detect_anomaly", False):
        _set_override(command, "trainer.detect_anomaly", "true")
    if getattr(args, "capture_gru_failure", False):
        _set_override(command, "++model.dataset_bias_params.context_gru_failure_capture_dir",
                      json.dumps(str(task_dir / "diagnostics" / "gru_failures")))

    # Repeated submissions must not silently revert a numerical repair simply
    # because launch_command.sh intentionally remains the original command.
    bias_gru_precision = getattr(args, "bias_gru_precision", None)
    precision_source = "cli" if bias_gru_precision is not None else "original_launch"
    previous_resume = task_dir / "resume_manifest.json"
    previous = json.loads(previous_resume.read_text(encoding="utf-8")) if previous_resume.is_file() else {}
    if bias_gru_precision is None:
        bias_gru_precision = previous.get("bias_gru_precision_override")
        if bias_gru_precision is not None:
            precision_source = "previous_resume_manifest"
    if bias_gru_precision is not None:
        if bias_gru_precision not in {"inherit", "float32"}:
            raise ValueError(f"Invalid saved bias GRU precision: {bias_gru_precision!r}")
        _set_override(
            command, "++model.dataset_bias_params.context_gru_precision", bias_gru_precision
        )

    tbptt_window = getattr(args, "bias_gru_tbptt_window", None)
    tbptt_source = "cli" if tbptt_window is not None else "original_launch_or_config"
    for source, saved_window in (
        ("previous_resume_manifest", previous.get("bias_gru_tbptt_window_override")),
        ("checkpoint", (metadata or {}).get("bias_gru_tbptt_window")),
    ):
        if tbptt_window is None and saved_window is not None:
            tbptt_window, tbptt_source = saved_window, source
    if tbptt_window is not None:
        if not isinstance(tbptt_window, int) or tbptt_window < 0:
            raise ValueError("Bias GRU TBPTT window must be a non-negative integer.")
        if tbptt_window and getattr(args, "capture_gru_failure", False):
            raise ValueError("Whole-GRU failure capture is not compatible with TBPTT window states.")
        _set_override(command, "++model.dataset_bias_params.context_gru_tbptt_window", str(tbptt_window))

    global_batch_gpus = getattr(args, 'global_batch_gpus', None)
    if global_batch_gpus is not None:
        if global_batch_gpus < 1 or args.gpus != 'inherit':
            raise ValueError('--global-batch-gpus requires a positive count and --gpus inherit.')
        if args.throughput_profile != 'unchanged' or args.exp8_runtime_profile != 'unchanged':
            raise ValueError('global_batch uses the saved global plan; use unchanged runtime profiles and explicit per-GPU execution limits if needed.')
        if not args.dry_run:
            if torch.cuda.device_count() != global_batch_gpus:
                raise ValueError('The visible CUDA device count must match --global-batch-gpus.')
            for device in range(global_batch_gpus):
                with torch.cuda.device(device):
                    _check_runtime_gpu_precision(command, task_dir)
        _set_override(command, 'trainer.devices', json.dumps(list(range(global_batch_gpus))))
        _set_override(command, 'trainer.use_distributed_sampler', 'false')
        _set_override(command, 'training.execution_microbatching.enabled', 'true')
        _set_override(command, '++training.execution_microbatching.distributed_mode', 'global_batch')

    saved_config_audit = None
    if getattr(args, "use_saved_resolved_config", False):
        saved_config_audit = _use_saved_resolved_config(command, task_dir)

    audit = {
        "run_id": run_id,
        "numerical_formulation_version": "log-space-nb2-v1",
        "gru_compute_policy": GRU_COMPUTE_POLICY,
        "global_batch_gpus_override": global_batch_gpus,
        "distributed_execution_note": (
            "global_batch preserves logical batch membership and accumulation, using SUM gradients before clipping/Adam. "
            "Changing GPU count can change dropout draws and floating-point reduction order; not a bitwise continuation. "
            "An incomplete final accumulation window is applied before end-of-epoch validation."
            if global_batch_gpus is not None else None
        ),
        "resume_mode": resume_mode,
        "checkpoint": metadata,
        "rejected_checkpoints": corrupt_checkpoints,
        "throughput_profile": args.throughput_profile,
        "exp8_runtime_profile": runtime_profile,
        "data_num_workers_override": args.data_num_workers,
        "bias_gru_precision_override": bias_gru_precision,
        "bias_gru_precision_source": precision_source,
        "bias_gru_tbptt_window_override": tbptt_window,
        "bias_gru_tbptt_window_source": tbptt_source,
        "training_gradient_note": (
            "TBPTT, when enabled, preserves forward hidden-state propagation and "
            "the full-CDS loss but omits temporal derivatives across window boundaries "
            "in each bias-GRU layer/direction. It is a biased training gradient, "
            "not an exact continuation of full-BPTT optimization. Model/optimizer "
            "state restoration does not imply an unchanged training method."
        ),
        "numerical_resume_note": (
            "Full-state resume restores the checkpoint's optimizer/trainer state. "
            "An explicit precision change preserves the logical objective but "
            "does not promise the original bitwise training trajectory."
            " Current code protects both GRUs in FP32 under CUDA AMP, including "
            "legacy inherit configurations; other neural operations retain AMP."
        ),
        "command": shlex.join(command),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "saved_config": saved_config_audit,
    }
    (task_dir / "resume_command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n",
        encoding="utf-8",
    )
    (task_dir / "resume_manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    # Keep the human-readable shell string in resume_manifest.json, but hand
    # Popen an argv list. The ordering here is intentional: ``audit`` contains
    # a string-valued ``command`` for JSON, so the executable list must win in
    # the in-memory task record.
    return {
        **audit,
        "run_id": run_id,
        "task_dir": task_dir,
        "status": "pending",
        "command": command,
    }


def _launch(task: dict[str, Any], gpu: str) -> dict[str, Any]:
    task_dir = Path(task["task_dir"])
    # Defence in depth: a malformed task must fail here with an actionable
    # message, never be handed to Popen as one enormous executable filename.
    _validate_command_argv(
        task["command"],
        source=task_dir / "resume_command.sh",
        executable_must_exist=True,
    )
    temporary = Path(tempfile.mkdtemp(prefix=f"riboai_resume_{task['run_id']}_", dir="/tmp"))
    env = _gpu_environment(gpu)
    env.update({
        "TMPDIR": str(temporary),
        "MPLCONFIGDIR": str(temporary / "matplotlib"),
        "TRITON_CACHE_DIR": str(temporary / "triton"),
        "HYDRA_FULL_ERROR": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
        "PYTORCH_ALLOC_CONF": env.get("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
        "RIBOUNMIX_LOGGER_VERSION": task["run_id"],
    })
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    log = task_dir / "logs" / "resume_launcher.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    gpu_label = (
        f"inherited CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        if gpu == "inherit" else f"physical GPU {gpu}"
    )
    print(f"[{task['run_id']}] resume on {gpu_label}", flush=True)
    code = 1
    try:
        with log.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(f"\n=== resume {datetime.now(timezone.utc).isoformat()} {gpu_label} ===\n")
            process = subprocess.Popen(
                task["command"], cwd=PROJECT_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                print(f"[{task['run_id']}] {line}", end="", flush=True)
            code = int(process.wait())
    except Exception as exc:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"Launcher exception: {type(exc).__name__}: {exc}\n")
        print(f"[{task['run_id']}] launcher exception: {exc}", flush=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    if code == 0:
        _consolidate_task(task_dir, task["run_id"])
    return {"run_id": task["run_id"], "gpu": gpu, "return_code": code, "completed": code == 0}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.run_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    gpus = _gpus(args.gpus)
    print(f"Resume launcher version: {RESUME_LAUNCHER_VERSION}")
    print(f"Resolved Python executable: {Path(sys.executable).absolute()}")
    launchers = sorted(root.rglob("launch_command.sh"))
    if not launchers:
        raise FileNotFoundError(f"No launch_command.sh files below {root}.")
    descriptors: list[dict[str, Any]] = []
    for path in launchers:
        saved_command = _read_launch_command(path)
        run_id = _task_id(saved_command, path.parent)
        descriptors.append(
            {
                "launch_path": path,
                "run_id": run_id,
                "dataset_size": _task_dataset_size(
                    saved_command, path.parent, run_id
                ),
            }
        )

    discovered_run_ids = {str(item["run_id"]) for item in descriptors}
    discovered_sizes = {
        int(item["dataset_size"])
        for item in descriptors
        if item["dataset_size"] is not None
    }
    requested_run_ids = set(args.include_run_ids or ())
    unknown_run_ids = requested_run_ids - discovered_run_ids
    if unknown_run_ids:
        raise ValueError(
            "Requested --include-run-ids were not found: "
            f"{sorted(unknown_run_ids)}."
        )
    requested_sizes = set(args.dataset_sizes or ())
    unknown_sizes = requested_sizes - discovered_sizes
    if unknown_sizes:
        raise ValueError(
            "Requested --dataset-sizes were not found: "
            f"{sorted(unknown_sizes)}; available={sorted(discovered_sizes)}."
        )

    selected_descriptors = [
        item
        for item in descriptors
        if _matches_task_filter(
            run_id=str(item["run_id"]),
            dataset_size=item["dataset_size"],
            dataset_sizes=args.dataset_sizes,
            include_run_ids=args.include_run_ids,
        )
    ]
    excluded_descriptors = [
        item for item in descriptors if item not in selected_descriptors
    ]
    if not selected_descriptors:
        raise ValueError("Task filters selected zero runs.")

    tasks = [
        _prepare_task(args, root, Path(item["launch_path"]))
        for item in selected_descriptors
    ]
    pending = [task for task in tasks if task["status"] == "pending"]
    blocked = [task for task in tasks if task["status"].startswith("blocked")]
    skipped = [task for task in tasks if task["status"] == "skipped_complete"]
    size_by_run_id = {
        str(item["run_id"]): item["dataset_size"] for item in descriptors
    }
    if args.schedule_order in {"largest-first", "explicit-first-largest"}:
        explicit_priority = set(args.include_run_ids or ())
        pending.sort(
            key=lambda task: (
                (
                    0
                    if args.schedule_order == "explicit-first-largest"
                    and str(task["run_id"]) in explicit_priority
                    else 1
                ),
                -int(size_by_run_id.get(str(task["run_id"])) or 0),
                str(task["run_id"]),
            )
        )
    modes: dict[str, int] = {}
    for task in pending:
        modes[task["resume_mode"]] = modes.get(task["resume_mode"], 0) + 1
    print(f"Run root: {root}")
    if args.dataset_sizes is not None or args.include_run_ids is not None:
        print(
            "Task filter: "
            f"dataset_sizes={list(args.dataset_sizes or ())}, "
            f"include_run_ids={list(args.include_run_ids or ())}; "
            f"selected={len(selected_descriptors)}, "
            f"excluded={len(excluded_descriptors)}."
        )
        print("Selected run IDs:")
        for item in selected_descriptors:
            print(
                f"  {item['run_id']} (N={item['dataset_size']})"
            )
    print(f"Schedule order: {args.schedule_order}")
    print(f"Tasks: {len(tasks)} total; {len(skipped)} complete; {len(pending)} pending; {len(blocked)} blocked.")
    print(f"Pending modes: {modes}")
    for i, task in enumerate(pending):
        print(f"  {task['run_id']} -> GPU {gpus[i % len(gpus)]} ({task['resume_mode']})")
        if task.get("bias_gru_precision_override") is not None:
            print(
                "    Bias GRU precision override: "
                f"{task['bias_gru_precision_override']} "
                f"({task['bias_gru_precision_source']}); "
                "Trainer precision and logical objective are unchanged."
            )
        if task.get("bias_gru_tbptt_window_override") is not None:
            print(
                f"    Bias GRU TBPTT window: {task['bias_gru_tbptt_window_override']} "
                f"({task['bias_gru_tbptt_window_source']}); "
                "0=full BPTT; >0=biased truncated training gradient."
            )
        for rejected in task.get("rejected_checkpoints", []):
            print(
                "    WARNING rejected checkpoint: "
                f"{rejected['path']} ({rejected['reason']})"
            )
    if blocked:
        print("Historical weights-only checkpoints were found. Re-run with --allow-weights-only-warm-resume after accepting the documented limitation.")
    if args.dry_run or blocked:
        return 2 if blocked and not args.dry_run else 0

    queues = {gpu: [] for gpu in gpus}
    for i, task in enumerate(pending):
        queues[gpus[i % len(gpus)]].append(task)
    statuses = []
    def worker(gpu: str, queue: list[dict[str, Any]]):
        return [_launch(task, gpu) for task in queue]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, queue) for gpu, queue in queues.items() if queue]
        for future in as_completed(futures):
            statuses.extend(future.result())
    summary = {
        "run_root": str(root), "throughput_profile": args.throughput_profile,
        "schedule_order": args.schedule_order,
        "task_filter": {
            "dataset_sizes": list(args.dataset_sizes or ()),
            "include_run_ids": list(args.include_run_ids or ()),
            "excluded_run_ids": [
                str(item["run_id"]) for item in excluded_descriptors
            ],
        },
        "skipped_complete": [t["run_id"] for t in skipped],
        "statuses": statuses,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = (
        args.summary_path.expanduser().resolve()
        if args.summary_path is not None else root / "resume_training_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    failed = [s["run_id"] for s in statuses if not s["completed"]]
    print(f"Resume complete; failed tasks: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
