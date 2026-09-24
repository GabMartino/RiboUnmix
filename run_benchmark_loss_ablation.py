#!/usr/bin/env python3
"""Run one task from the matched four-dataset loss-ablation matrix.

The task mapping is deliberately small and transparent:

    training seed -> loss arm -> benchmark dataset

One invocation launches exactly one production benchmarking model.  The SLURM
wrapper maps one array element to one invocation and one scheduler-assigned GPU.
No data preparation, split search, or post-hoc alteration of predictions occurs
here.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DESIGN = (
    REPO_ROOT / "config" / "experiment_designs" / "benchmark_loss_ablation.yaml"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "riboai_benchmarking_experiments" / "loss_ablation_v1"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_training_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError("--training-seeds must be comma-separated integers.") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--training-seeds must contain distinct integer seeds.")
    return seeds


def load_design(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Loss-ablation design not found: {path}")
    design = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    datasets = design.get("datasets")
    arms = design.get("arms")
    if not isinstance(datasets, list) or not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Design datasets must be a non-empty unique list.")
    if not isinstance(arms, dict) or not arms:
        raise ValueError("Design arms must be a non-empty mapping.")
    required = {
        "replica_nb_weight",
        "consensus_raw_pcc_weight",
        "consensus_nb_vst_pcc_weight",
    }
    for name, arm in arms.items():
        if not isinstance(arm, dict) or not required.issubset(arm):
            raise ValueError(f"Loss arm {name!r} is missing {sorted(required)}.")
        weights = [float(arm[key]) for key in required]
        if any(value < 0.0 for value in weights):
            raise ValueError(f"Loss arm {name!r} contains a negative coefficient.")
        if float(arm["replica_nb_weight"]) != 1.0:
            raise ValueError(f"Loss arm {name!r} must retain replica_nb_weight=1.")
    if "full" not in arms:
        raise ValueError("The contemporaneous full-loss control arm is mandatory.")
    return design


@dataclass(frozen=True)
class AblationTask:
    task_index: int
    task_id: str
    training_seed: int
    split_seed: int
    arm: str
    arm_label: str
    dataset: str
    replica_nb_weight: float
    consensus_raw_pcc_weight: float
    consensus_nb_vst_pcc_weight: float
    gamma_reg_weight: float


def build_tasks(design: dict[str, Any], training_seeds: Iterable[int]) -> list[AblationTask]:
    fixed = dict(design.get("fixed") or {})
    split_seed = int(fixed.get("split_seed", 42))
    gamma_reg_weight = float(fixed.get("gamma_reg_weight", 1.0e-4))
    tasks: list[AblationTask] = []
    for training_seed in training_seeds:
        for arm_name, arm in design["arms"].items():
            for dataset in design["datasets"]:
                index = len(tasks)
                tasks.append(
                    AblationTask(
                        task_index=index,
                        task_id=(
                            f"seed{int(training_seed)}__{arm_name}__{dataset}"
                        ),
                        training_seed=int(training_seed),
                        split_seed=split_seed,
                        arm=str(arm_name),
                        arm_label=str(arm.get("label", arm_name)),
                        dataset=str(dataset),
                        replica_nb_weight=float(arm["replica_nb_weight"]),
                        consensus_raw_pcc_weight=float(
                            arm["consensus_raw_pcc_weight"]
                        ),
                        consensus_nb_vst_pcc_weight=float(
                            arm["consensus_nb_vst_pcc_weight"]
                        ),
                        gamma_reg_weight=gamma_reg_weight,
                    )
                )
    return tasks


def task_directory(output_root: Path, task: AblationTask) -> Path:
    return (
        output_root
        / "runs"
        / f"seed{task.training_seed}"
        / task.arm
        / task.dataset
    )


def _prediction_is_present(
    manifest_path: Path,
    variant: str,
) -> bool:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest[variant]
        recorded = Path(str(entry["output_path"]))
        candidates = (recorded, manifest_path.parent / recorded.name)
        return int(entry["prediction_rows"]) > 0 and any(
            path.is_file() and path.stat().st_size > 0 for path in candidates
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def verify_attempt_outputs(
    attempt_dir: Path,
    dataset: str,
    checkpoint_variants: Iterable[str],
) -> Path:
    manifests = sorted(
        (attempt_dir / "predictions" / dataset).rglob(
            "prediction_checkpoint_manifest.json"
        )
    )
    valid = [
        manifest
        for manifest in manifests
        if all(_prediction_is_present(manifest, variant) for variant in checkpoint_variants)
    ]
    if len(valid) != 1:
        raise RuntimeError(
            f"Expected exactly one complete prediction manifest in {attempt_dir}; "
            f"found {len(valid)}."
        )
    return valid[0]


def completed_attempt(
    task_dir: Path,
    *,
    task: AblationTask,
    design_sha256: str,
    checkpoint_variants: tuple[str, ...],
) -> Path | None:
    for status_path in sorted(task_dir.glob("attempts/*/task_status.json")):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                status.get("state") != "completed"
                or status.get("task_id") != task.task_id
                or status.get("design_sha256") != design_sha256
            ):
                continue
            verify_attempt_outputs(
                status_path.parent,
                task.dataset,
                checkpoint_variants,
            )
            return status_path.parent
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
    return None


def resolve_python(raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        selected = candidate
    else:
        found = shutil.which(raw)
        if found is None:
            raise FileNotFoundError(f"Python executable not found: {raw}")
        selected = Path(found)

    # Do not use Path.resolve() here. A virtual environment's ``bin/python`` is
    # normally a symlink to the base interpreter; resolving that final symlink
    # changes ``sys.prefix`` and discards the venv's site-packages when the
    # returned path is executed. ``abspath`` normalizes a relative spelling
    # while deliberately preserving the environment entry point.
    selected = Path(os.path.abspath(os.fspath(selected)))
    if not selected.is_file() or not os.access(selected, os.X_OK):
        raise FileNotFoundError(f"Python executable is not usable: {selected}")
    return selected


def build_training_command(
    *,
    python_executable: Path,
    task: AblationTask,
    attempt_dir: Path,
    design: dict[str, Any],
    num_workers: int,
    predict_num_workers: int,
) -> list[str]:
    fixed = dict(design.get("fixed") or {})
    variants = ",".join(map(str, fixed["checkpoint_variants"]))
    return [
        str(python_executable),
        "-u",
        str(REPO_ROOT / "main_ribounmix_benchmarking.py"),
        f"name=benchmark_loss_ablation_{task.arm}_{task.dataset}_seed{task.training_seed}",
        f"experiment.dataset={task.dataset}",
        f"experiment.seed={task.training_seed}",
        "experiment.from_checkpoint=false",
        "experiment.train=true",
        "experiment.predict=true",
        f"split.seed={task.split_seed}",
        f"prediction.checkpoint_variants=[{variants}]",
        f"loss.replica_nb_weight={task.replica_nb_weight}",
        f"loss.consensus_raw_pcc_weight={task.consensus_raw_pcc_weight}",
        f"loss.consensus_nb_vst_pcc_weight={task.consensus_nb_vst_pcc_weight}",
        f"loss.gamma_reg_weight={task.gamma_reg_weight}",
        f"data.batch_size={int(fixed.get('batch_size', 32))}",
        f"data.num_workers={num_workers}",
        f"data.predict_num_workers={predict_num_workers}",
        "trainer.devices=[0]",
        "trainer.use_distributed_sampler=false",
        f"paths.checkpoints={attempt_dir / 'checkpoints'}",
        f"paths.logs={attempt_dir / 'logs'}",
        f"paths.results={attempt_dir / 'predictions'}",
        f"hydra.run.dir={attempt_dir / 'hydra'}",
        "hydra.job.chdir=false",
    ]


def attempt_identifier() -> str:
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    array = os.environ.get("SLURM_ARRAY_TASK_ID")
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")
    if job:
        suffix = f"_{array}" if array not in (None, "") else ""
        return f"slurm_{job}{suffix}_r{restart}"
    return f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def snapshot_design(output_root: Path, design_path: Path, design_sha256: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "benchmark_loss_ablation_design.yaml"
    if destination.exists():
        if sha256_file(destination) != design_sha256:
            raise RuntimeError(
                f"Output root is frozen to another design: {destination}"
            )
        return
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_bytes(design_path.read_bytes())
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if sha256_file(destination) != design_sha256:
            raise RuntimeError(
                f"Concurrent worker froze a different design at {destination}."
            )
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--training-seeds", default="42")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--require-single-visible-gpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--predict-num-workers", type=int, default=0)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    design_path = args.design.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    design = load_design(design_path)
    seeds = parse_training_seeds(args.training_seeds)
    tasks = build_tasks(design, seeds)
    design_sha256 = sha256_file(design_path)
    variants = tuple(map(str, design["fixed"]["checkpoint_variants"]))

    if args.list_tasks:
        print("task_index\ttask_id\ttraining_seed\tarm\tdataset")
        for task in tasks:
            print(
                f"{task.task_index}\t{task.task_id}\t{task.training_seed}\t"
                f"{task.arm}\t{task.dataset}"
            )
        return 0
    if args.task_index is None:
        raise ValueError("Provide --task-index or --list-tasks.")
    if not 0 <= args.task_index < len(tasks):
        raise IndexError(
            f"Task index {args.task_index} is outside 0..{len(tasks) - 1} "
            f"for seeds {seeds}."
        )
    if args.num_workers < 0 or args.predict_num_workers < 0:
        raise ValueError("Worker counts must be non-negative.")

    task = tasks[args.task_index]
    preview_attempt = task_directory(output_root, task) / "attempts" / "DRY_RUN"
    python_executable = resolve_python(args.python_executable)
    preview_command = build_training_command(
        python_executable=python_executable,
        task=task,
        attempt_dir=preview_attempt,
        design=design,
        num_workers=args.num_workers,
        predict_num_workers=args.predict_num_workers,
    )
    print(json.dumps(asdict(task), indent=2, sort_keys=True))
    print(f"Design SHA256: {design_sha256}")
    print(f"Command: {shlex.join(preview_command)}")
    if args.dry_run:
        return 0

    if args.require_single_visible_gpu:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or len([item for item in visible.split(",") if item]) != 1:
            raise RuntimeError(
                "Expected exactly one scheduler-assigned visible GPU; "
                f"CUDA_VISIBLE_DEVICES={visible!r}."
            )

    task_dir = task_directory(output_root, task)
    task_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (task_dir / ".execution.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Task is already running: {task.task_id}") from exc

    snapshot_design(output_root, design_path, design_sha256)
    if not args.rerun_completed:
        completed = completed_attempt(
            task_dir,
            task=task,
            design_sha256=design_sha256,
            checkpoint_variants=variants,
        )
        if completed is not None:
            print(f"SKIP completed task {task.task_id}: {completed}")
            return 0

    attempt_dir = task_dir / "attempts" / attempt_identifier()
    if attempt_dir.exists():
        raise FileExistsError(f"Attempt directory already exists: {attempt_dir}")
    attempt_dir.mkdir(parents=True)
    command = build_training_command(
        python_executable=python_executable,
        task=task,
        attempt_dir=attempt_dir,
        design=design,
        num_workers=args.num_workers,
        predict_num_workers=args.predict_num_workers,
    )
    base_config_path = REPO_ROOT / "config" / "config_ribounmix_benchmarking.yaml"
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    dataset_inputs: dict[str, str] = {}
    for input_name, raw_path in base_config["benchmarking"]["datasets"][task.dataset].items():
        input_path = Path(str(raw_path))
        if not input_path.is_absolute():
            input_path = (REPO_ROOT / input_path).resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Benchmark {input_name} input is missing: {input_path}")
        dataset_inputs[str(input_path)] = sha256_file(input_path)
    provenance = {
        **asdict(task),
        "study_id": str(design["study_id"]),
        "design_path": str(design_path),
        "design_sha256": design_sha256,
        "command_argv": command,
        "command_shell_escaped": shlex.join(command),
        "created_at_utc": utc_now(),
        "repository_root": str(REPO_ROOT),
        "source_sha256": {
            "run_benchmark_loss_ablation.py": sha256_file(
                REPO_ROOT / "run_benchmark_loss_ablation.py"
            ),
            "main_ribounmix_benchmarking.py": sha256_file(
                REPO_ROOT / "main_ribounmix_benchmarking.py"
            ),
            "config/config_ribounmix_benchmarking.yaml": sha256_file(
                REPO_ROOT / "config" / "config_ribounmix_benchmarking.yaml"
            ),
            "Models/RiboUnmixLightningModule.py": sha256_file(
                REPO_ROOT / "Models" / "RiboUnmixLightningModule.py"
            ),
            "Models/RiboUnmixModel/RiboUnmixModel.py": sha256_file(
                REPO_ROOT / "Models" / "RiboUnmixModel" / "RiboUnmixModel.py"
            ),
            "Dataloaders/RiboUnmixBenchmarking/RiboUnmixBenchmarkingDataModule.py": sha256_file(
                REPO_ROOT
                / "Dataloaders"
                / "RiboUnmixBenchmarking"
                / "RiboUnmixBenchmarkingDataModule.py"
            ),
        },
        "input_sha256": dataset_inputs,
        "runtime": {
            "hostname": os.uname().nodename,
            "python": str(python_executable),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    atomic_write_json(attempt_dir / "task_spec.json", provenance)
    (attempt_dir / "launch_command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n",
        encoding="utf-8",
    )
    status = {
        "state": "running",
        "task_id": task.task_id,
        "design_sha256": design_sha256,
        "started_at_utc": utc_now(),
    }
    atomic_write_json(attempt_dir / "task_status.json", status)

    environment = os.environ.copy()
    environment["RIBOUNMIX_LOGGER_VERSION"] = (
        f"loss_ablation_{task.task_id}_{attempt_dir.name}"
    )
    try:
        process = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
        status["return_code"] = int(process.returncode)
        if process.returncode != 0:
            status["state"] = "failed"
            status["finished_at_utc"] = utc_now()
            atomic_write_json(attempt_dir / "task_status.json", status)
            return int(process.returncode)
        manifest = verify_attempt_outputs(attempt_dir, task.dataset, variants)
        status.update(
            {
                "state": "completed",
                "finished_at_utc": utc_now(),
                "prediction_manifest": str(manifest),
            }
        )
        atomic_write_json(attempt_dir / "task_status.json", status)
        print(f"Completed {task.task_id}: {manifest}")
        return 0
    except BaseException as exc:
        status.update(
            {
                "state": "failed",
                "finished_at_utc": utc_now(),
                "exception": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(attempt_dir / "task_status.json", status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
