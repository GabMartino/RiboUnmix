#!/usr/bin/env python3
"""Prepare matched shared-only synthetic controls; never train by default.

Clone the archived full-model configs, verify the native train/validation split,
and reuse model.mean_correction=unity. --arm learned optionally provides fresh
full-model comparators under the same current code. The design file declares
the panel size and depth for each task. One task runs on one GPU.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_DESIGN = ROOT / "config/experiment_designs/synthetic_gamma_unity.yaml"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_file(root, pattern):
    paths = sorted(root.rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} in {root}, found {len(paths)}.")
    return paths[0]


def input_path(value):
    """Use repository-relative inputs; relocate only exact Datasets suffixes."""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    elif not path.is_file() and "Datasets" in path.parts:
        path = ROOT.joinpath(*path.parts[path.parts.index("Datasets"):])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def make_config(source, *, arm, task_root, split_path, experiment_label="C2"):
    """Preserve scientific settings; change only the intervention and I/O."""
    cfg = copy.deepcopy(source)
    cfg.pop("defaults", None)  # Archived config is already fully composed.
    cfg["training"]["grouped_optimizer_batch"].pop("resolved", None)
    cfg["name"] = (
        f"synthetic_{experiment_label}_N{len(cfg['experiment']['dataset']):03}_"
        f"{arm}_seed{cfg['experiment']['seed']}"
    )
    cfg["model"]["mean_correction"] = arm
    cfg["experiment"].update(from_checkpoint=False, train=True, predict=True)
    cfg["prediction"]["checkpoint_variants"] = ["best_val_loss"]
    cfg["callbacks"]["save_best_pcc_checkpoint"] = False
    cfg["split"]["expected_manifest"] = str(split_path)
    for key in ("checkpoints", "logs"):
        cfg["paths"][key] = str(task_root / key)
    cfg["paths"]["results"] = str(task_root / "predictions")
    cfg["trainer"]["devices"] = [0]  # Logical device inside the Slurm allocation.
    return cfg


def validate_source(
    cfg,
    split,
    n,
    *,
    expected_dataset_config="synthetic_2_per_codon",
):
    checks = {
        "dataset_count": len(cfg["experiment"]["dataset"]) == n,
        "seed42": cfg["experiment"]["seed"] == 42,
        "learned_gamma": cfg["model"].get("mean_correction", "learned") == "learned",
        "learned_alpha": cfg["model"]["alpha_mode"] == "learned",
        "mass_free": cfg["model"]["mass_conservation"] is False,
        "standard_nb": cfg["loss"]["experiment_mode"] == "standard_nb",
        "beta_zero": cfg["loss"]["nb_mean_gradient_beta"] == 0,
        "transcript_balanced": cfg["loss"]["sample_reduction"] == "transcript_balanced",
        "equal_reference": cfg["model"]["gamma_centering"]["reference"]["weighting"] == "equal",
        "expected_depth_config": (
            cfg["dataset_config"]["_name_"] == expected_dataset_config
        ),
        "matching_dataset_order": cfg["experiment"]["dataset"] == split["experiment_datasets"],
        "no_train_validation_overlap": not set(split["train_ids"]) & set(split["validation_ids"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Archived comparator violates design: {checks}")
    return checks


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prepare(design_path, output_root, arm):
    design = yaml.safe_load(design_path.read_text())
    output_root.mkdir(parents=True, exist_ok=True)
    # Array elements may prepare simultaneously. They must share one immutable plan.
    with (output_root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        tasks, file_hashes, generated = [], {}, {}
        file_hashes[str(design_path)] = sha256(design_path)
        task_subdirs = [
            entry.get("task_subdir", f"N{int(entry['N']):03}")
            for entry in design["experiments"]
        ]
        if len(task_subdirs) != len(set(task_subdirs)):
            raise ValueError("Every design entry must resolve to a unique task_subdir.")
        for entry, task_subdir in zip(design["experiments"], task_subdirs):
            source_run = (ROOT / entry["source_run"]).resolve()
            source_config = unique_file(source_run / "logs", "config.yaml")
            source_split = unique_file(source_run / "results", "split_manifest*.json")
            source = yaml.safe_load(source_config.read_text())
            split = json.loads(source_split.read_text())
            expected_dataset_config = entry.get(
                "expected_dataset_config", "synthetic_2_per_codon"
            )
            checks = validate_source(
                source,
                split,
                int(entry["N"]),
                expected_dataset_config=expected_dataset_config,
            )
            experiment_label = entry.get("experiment_label", "C2")
            task_root = output_root / task_subdir / arm
            split_path = task_root / "historical_split.json"
            cfg = make_config(
                source,
                arm=arm,
                task_root=task_root,
                split_path=split_path,
                experiment_label=experiment_label,
            )
            # No dataset/QC/reliability recalculation: retain saved weighted inputs.
            mapping = cfg["dataset_config"]["dataset_path"]
            for name, value in mapping.items():
                mapping[name] = str(input_path(value))
            cfg["paths"]["sequences_path"] = str(input_path(cfg["paths"]["sequences_path"]))
            encodings = cfg["paths"]["encodings"]
            for name, value in encodings.items():
                encodings[name] = str(input_path(value))
            for key in ("latent_path", "observed_path"):
                cfg["synthetic_ground_truth"][key] = str(input_path(cfg["synthetic_ground_truth"][key]))
            inputs = [source_config, source_split, *map(Path, mapping.values()),
                      Path(cfg["paths"]["sequences_path"]), *map(Path, encodings.values()),
                      *[Path(cfg["synthetic_ground_truth"][k]) for k in ("latent_path", "observed_path")]]
            for path in inputs:
                if str(path) not in file_hashes:
                    file_hashes[str(path)] = sha256(path)
            config_path = task_root / "config.yaml"
            generated[config_path] = yaml.safe_dump(cfg, sort_keys=False)
            generated[split_path] = source_split.read_text()
            command = [sys.executable, "-u", str(ROOT / "main_ribounmix_synthetic.py"),
                       f"--config-path={task_root}", "--config-name=config",
                       f"hydra.run.dir={task_root / 'hydra'}", "hydra.job.chdir=false"]
            tasks.append(dict(task_index=len(tasks), N=entry["N"], arm=arm,
                              depth_reads_per_codon=entry.get("depth_reads_per_codon"),
                              experiment_label=experiment_label,
                              expected_dataset_config=expected_dataset_config,
                              task_id=cfg["name"], root=str(task_root), config=str(config_path),
                              config_sha256=hashlib.sha256(generated[config_path].encode()).hexdigest(),
                              source_run=str(source_run), source_config=str(source_config),
                              source_split=str(source_split), command=command,
                              train_count=len(split["train_ids"]), validation_count=len(split["validation_ids"]),
                              source_checks=checks))
        # Current-code identity is recorded; equality with historical source is NOT asserted.
        code = [ROOT / "main_ribounmix_synthetic.py", ROOT / "main_ribounmix_multidataset.py", Path(__file__)]
        for folder in ("Models", "Dataloaders", "Utils"):
            code.extend(sorted((ROOT / folder).rglob("*.py")))
        code_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in code}
        plan = dict(design=design, tasks=tasks, input_sha256=file_hashes,
                    code_sha256=code_hashes, python=platform.python_version(),
                    evaluation="Historical validation cohort, not an independent synthetic test",
                    primary_metrics=["PCC(L,q_bar)", "RMSE(L,q_bar)"],
                    limitations=["Current code may differ from the historical full-model code; use --arm learned for paired reruns.",
                                 "Alpha head learns from detached context; context stops learning when gamma=1.",
                                 "Same simulator counts, single seed, and bias ordering; not a diversity-versus-read-count test."])
        manifest = output_root / "experiment_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != plan:
            raise ValueError("Existing design/config/inputs/code differ; use a new --output-root.")
        for path, text in generated.items():
            if path.exists() and path.read_text() != text:
                raise ValueError(f"Prepared artifact changed: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(text)
        if not manifest.exists():
            write_json(manifest, plan)
        return plan


def execute(task):
    root = Path(task["root"])
    with (root / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = root / "execution_status.json"
        if status_path.exists() or list((root / "checkpoints").rglob("*.ckpt")):
            raise RuntimeError("An attempt already exists. Refusing to silently restart or overwrite it.")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one allocated CUDA GPU per task; do not use DDP.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The matched run requires BF16 support; do not silently change precision.")
        state = dict(status="running", command=task["command"], config_sha256=task["config_sha256"],
                     started_at=datetime.now(timezone.utc).isoformat(),
                     gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__)
        write_json(status_path, state)
        try:
            with (root / "train.log").open("w") as log:
                result = subprocess.run(task["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state.update(exit_code=result.returncode,
                         status="process_finished" if result.returncode == 0 else "failed")
            if result.returncode:
                raise RuntimeError(f"Training exited {result.returncode}; see {root / 'train.log'}")
        except BaseException as exc:
            state.update(status="failed_or_interrupted", error=str(exc))
            raise
        finally:
            state["ended_at"] = datetime.now(timezone.utc).isoformat()
            write_json(status_path, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/synthetic_gamma_unity_C2_seed42")
    parser.add_argument("--arm", choices=("unity", "learned"), default="unity",
                        help="learned prepares contemporaneous full-model comparators; use a separate output root")
    parser.add_argument(
        "--task-index",
        type=int,
        help="Zero-based task index in the selected design file.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default: prepare configs and print commands, no training")
    mode.add_argument("--run", action="store_true", help="train exactly one --task-index on one visible GPU")
    args = parser.parse_args(argv)
    if args.run and args.task_index is None:
        parser.error("--run requires --task-index")
    plan = prepare(args.design.resolve(), args.output_root.resolve(), args.arm)
    if args.task_index is not None and not 0 <= args.task_index < len(plan["tasks"]):
        parser.error(
            f"--task-index must be in 0..{len(plan['tasks']) - 1} for this design"
        )
    tasks = plan["tasks"] if args.task_index is None else [plan["tasks"][args.task_index]]
    for task in tasks:
        depth = task.get("depth_reads_per_codon")
        depth_text = f", C={depth:g}" if depth is not None else ""
        print(
            f"[{task['task_index']}] {task['task_id']}: N={task['N']}{depth_text}; "
            f"{task['train_count']} train / {task['validation_count']} validation",
            flush=True,
        )
        print(shlex.join(task["command"]), flush=True)
    for warning in plan["limitations"]:
        print(f"SCIENTIFIC NOTE: {warning}", flush=True)
    if args.run:
        execute(tasks[0])
    else:
        print("Prepared only; no training was launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
