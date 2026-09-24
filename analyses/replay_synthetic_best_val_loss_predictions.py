#!/usr/bin/env python3
"""Regenerate validation predictions from historical synthetic val-loss checkpoints.

Each replay loads that run's persisted ``config.yaml`` rather than the current
synthetic default.  This is essential for historical checkpoints: for example,
the stored dataset-bias context GRU uses ``context_gru_hidden_size: 128``.
Only the minimum-``val_loss`` checkpoint is predicted; PCC checkpoints are not
loaded by this utility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_ribounmix_multidataset import (  # noqa: E402
    find_prediction_checkpoint,
    main as pipeline_main,
)
from analyses.analyze_synthetic_recovery import (  # noqa: E402
    DEFAULT_RESULTS_ROOT,
    DEFAULT_RUN_PREFIX,
)


def _one_config(run_dir: Path) -> Path:
    paths = sorted(run_dir.rglob("config.yaml"))
    # Replay itself creates a fresh Lightning ``version_N/config.yaml``. The
    # immutable training configuration is stored under the historical direct
    # logger directory and must win on subsequent invocations.
    historical = [path for path in paths if path.parent.name.startswith("direct_")]
    if len(historical) == 1:
        return historical[0]
    if len(paths) != 1:
        raise ValueError(f"expected one saved config.yaml, found {len(paths)}")
    return paths[0]


def _prepare_config(run_dir: Path, config_path: Path, *, device: int | None) -> DictConfig:
    """Make a prediction-only copy while retaining every model architecture field."""
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Saved configuration is not a mapping: {config_path}")
    OmegaConf.set_struct(cfg, False)
    cfg.experiment.train = False
    cfg.experiment.predict = True
    cfg.experiment.from_checkpoint = True
    cfg.prediction = {"checkpoint_variants": ["best_val_loss"]}
    # Saved configs often contain the absolute path from the training server.
    # The run directory itself is canonical and has the expected three roots.
    cfg.paths.checkpoints = str(run_dir / "checkpoints")
    cfg.paths.logs = str(run_dir / "logs")
    cfg.paths.results = str(run_dir / "results")
    # The historical resolved config intentionally stored repository-relative
    # inputs. Make them absolute so invoking this utility from a results/log
    # directory cannot turn ``./Datasets/...`` into a missing path.
    def repository_path(value: Any) -> str:
        path = Path(str(value)).expanduser()
        return str(path if path.is_absolute() else (ROOT / path).resolve())

    if cfg_get := cfg.get("paths"):
        for key in ("sequences_path", "css_split"):
            if cfg_get.get(key) is not None:
                cfg.paths[key] = repository_path(cfg_get[key])
        for key, value in dict(cfg_get.get("encodings", {})).items():
            cfg.paths.encodings[key] = repository_path(value)
    for dataset, value in dict(cfg.dataset_config.dataset_path).items():
        cfg.dataset_config.dataset_path[dataset] = repository_path(value)
    synthetic_truth = cfg.get("synthetic_ground_truth")
    if synthetic_truth is not None and synthetic_truth.get("observed_path"):
        synthetic_truth.observed_path = repository_path(synthetic_truth.observed_path)
    if device is not None:
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.devices = [int(device)]
    return cfg


def _output_already_valid(run_dir: Path) -> bool:
    manifests = sorted(run_dir.rglob("prediction_checkpoint_manifest.json"))
    if len(manifests) != 1:
        return False
    try:
        record = json.loads(manifests[0].read_text(encoding="utf-8")).get("best_val_loss")
        if not isinstance(record, dict):
            return False
        output = Path(str(record.get("output_path", "")))
        if not output.is_absolute():
            output = (manifests[0].parent / output).resolve()
        checkpoint = Path(str(record.get("checkpoint_path", ""))).name
        return output.is_file() and "predictions_main_val_best_val_loss_" in output.name and "val_loss" in checkpoint and not checkpoint.startswith("pcc-")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _replay(run_dir: Path, *, device: int | None, dry_run: bool) -> str:
    config_path = _one_config(run_dir)
    cfg = _prepare_config(run_dir, config_path, device=device)
    checkpoint = find_prediction_checkpoint(Path(str(cfg.paths.checkpoints)), "best_val_loss")
    if checkpoint is None:
        raise FileNotFoundError("no non-PCC val_loss checkpoint found")
    context_hidden = cfg.model.dataset_bias_params.context_gru_hidden_size
    if dry_run:
        return f"would predict {Path(checkpoint).name}; context_gru_hidden_size={context_hidden}"
    print(
        f"[replay] {run_dir.name}: {Path(checkpoint).name}; "
        f"context_gru_hidden_size={context_hidden}; config={config_path}"
    )
    # Calling Hydra's undecorated function avoids re-composing the current
    # default config and therefore preserves the historical 128-wide GRU.
    pipeline_main.__wrapped__(cfg)
    return "predicted"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--run", action="append", default=[], help="Exact run directory name; may be passed repeatedly.")
    parser.add_argument("--device", type=int, default=None, help="Override the saved GPU index for this replay.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even when a valid best-val-loss prediction already exists.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Some direct invocations from a run directory use the caller's CWD for
    # unqualified data paths. Keep the whole replay in repository context too.
    os.chdir(ROOT)
    root = args.results_root.expanduser().resolve()
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith(args.run_prefix))
    if args.run:
        requested = set(args.run)
        candidates = [path for path in candidates if path.name in requested]
        missing = requested - {path.name for path in candidates}
        if missing:
            raise FileNotFoundError(f"Requested runs not found: {sorted(missing)}")
    if not candidates:
        raise RuntimeError("No matching synthetic run directories found.")
    for run_dir in candidates:
        if not args.overwrite and _output_already_valid(run_dir):
            print(f"[skip] {run_dir.name}: verified best-val-loss prediction already exists")
            continue
        try:
            status = _replay(run_dir, device=args.device, dry_run=args.dry_run)
            print(f"[ok] {run_dir.name}: {status}")
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            print(f"[skip] {run_dir.name}: {exc}")


if __name__ == "__main__":
    main()
