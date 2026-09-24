#!/usr/bin/env python3
"""Train four contiguous best-to-worst dataset-quality panels."""
from __future__ import annotations

import argparse
from pathlib import Path

from run_cumulative_stability import ROOT, run_task
from Utils.quality_selection_experiments import (
    FOUR_PANEL_DESIGN, prepare_quality_selection_experiment,
)

DEFAULT_OUTPUT = ROOT / "results/four_quality_strata_seed42"
TASK_COUNT = 8


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/config_ribounmix_multidataset.yaml")
    parser.add_argument("--datasets", type=Path, default=ROOT / "config/dataset_config/weighted_hek_riboseq_codon_replicas.yaml")
    parser.add_argument("--ranking", type=Path, default=ROOT / "Datasets/data/HEK_riboseq_profile_quality_rank_components.tsv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--task-index", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for key in ("config", "datasets", "ranking", "output_root"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.task_index is not None and not 0 <= args.task_index < TASK_COUNT:
        parser.error(f"--task-index must be in 0..{TASK_COUNT - 1}")
    plan = prepare_quality_selection_experiment(
        root=args.output_root, project_root=ROOT, config=args.config, datasets=args.datasets,
        ranking_path=args.ranking, seed=args.seed, design=FOUR_PANEL_DESIGN)
    if len(plan["tasks"]) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} frozen tasks, found {len(plan['tasks'])}.")
    return 0 if args.prepare_only else run_task(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
