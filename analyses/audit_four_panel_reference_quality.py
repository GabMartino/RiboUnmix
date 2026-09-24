#!/usr/bin/env python3
"""CPU-only audit / preparation of dataset-QC gamma references; never trains."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# --gpus describes future launch commands, never devices used by this program.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['audit', 'prepare-existing', 'prepare-rank-balanced'], default='audit')
    parser.add_argument('--resume-preparation', type=Path,
                        help='Finish final validation/freezing of this saved preparation directory; no audit/search/refit repeated.')
    parser.add_argument('--panel-manifest', type=Path, default=ROOT/'results/my_panels_a100_b32_20260906_114323/panel_manifest.json')
    parser.add_argument('--common-split-manifest', type=Path)
    parser.add_argument('--ranking-table', type=Path, default=ROOT/'Datasets/data/HEK_riboseq_profile_quality_rank.tsv')
    parser.add_argument('--ranking-dataset-column', default='dataset')
    parser.add_argument('--ranking-rank-column', default='quality_rank')
    parser.add_argument('--ranking-provenance', type=Path, help='Optional documentary JSON; claims are recorded, not assumed verified.')
    parser.add_argument('--dataset-quality-table', type=Path)
    parser.add_argument('--source-family-mapping', type=Path, help='CSV with dataset_id,source_family; must agree with frozen mappings.')
    parser.add_argument('--reference-weights', type=Path, help='CSV with policy,panel,dataset,pi (or canonical *_id columns).')
    parser.add_argument('--duplicate-weight-policy', choices=['error', 'collapse-identical'], default='error')
    parser.add_argument('--alias-map', type=Path, help='JSON alias-to-canonical-ID mapping; no fuzzy matching.')
    parser.add_argument('--config-root', type=Path, help='Default: panel-manifest parent; <panel>/hydra/.hydra/config.yaml then resolved_config.yaml.')
    parser.add_argument('--reliability-root', type=Path, help='Default: panel-manifest parent.')
    parser.add_argument('--historical-ranked-root', type=Path, help='Optional: read ONLY panel/run manifests and configurations for provenance.')
    parser.add_argument('--output-root', type=Path, default=ROOT/'results/four_panel_reference_quality_audit')
    parser.add_argument('--partition-seed', type=int, default=42)
    parser.add_argument('--training-seed', type=int, help='Default: saved seed. Existing-panel preparation requires the saved seed.')
    parser.add_argument('--gpus', default='0', help='Future one-process-per-GPU commands only; inherit preserves Slurm visibility.')
    parser.add_argument('--dry-run', action='store_true', help='All modes are preparation-only; records intent, never enables training.')
    parser.add_argument('--dataset-config', type=Path, help='Optional relocated dataset_path YAML; same canonical universe required.')
    parser.add_argument('--sequences-path', type=Path, help='Optional relocated copy of the same sequence artifact.')
    parser.add_argument('--design-config', type=Path, help='JSON overriding prespecified search block weights/budget; frozen before search.')
    parser.add_argument('--require-validation-support', action='store_true',
                        help='Constrain every proposed swap to preserve all frozen validation support; use feasible original-panel starts.')
    parser.add_argument('--source-mass-warning', type=float, help='Optional descriptive heuristic, never a partition-selection constraint.')
    parser.add_argument('--jitter-seed', type=int, default=1701)
    parser.add_argument('--no-tex', action='store_true', help='Use Matplotlib serif fonts if LaTeX is unavailable; still vector output.')
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    base = args.panel_manifest.parent
    for name, relative in [('common_split_manifest', 'common_split_manifest.json'),
                           ('dataset_quality_table', 'dataset_quality_table.csv')]:
        if getattr(args, name) is None and (base/relative).is_file():
            setattr(args, name, base/relative)
    args.config_root = args.config_root or base
    args.reliability_root = args.reliability_root or base
    if args.source_mass_warning is not None and not 0 < args.source_mass_warning <= 1:
        parser.error('--source-mass-warning must lie in (0,1].')
    return args


def main(argv=None):
    from Utils.panel_reference_audit import run_audit
    args = parse_args(argv)
    try:
        if args.resume_preparation is not None:
            from Utils.panel_reference_preparation import finish_preparation
            finish_preparation(args.resume_preparation, gpus=args.gpus)
            return 0
        audit = run_audit(args)
        if audit['manifest']['hard_errors']:
            print('Audit contains hard errors; no training design prepared.', file=sys.stderr)
            for detail in audit['manifest']['hard_errors']:
                print(f'  - {detail}', file=sys.stderr)
            return 2
        if args.mode != 'audit':
            from Utils.panel_reference_preparation import prepare_design
            prepare_design(args, audit)
        return 0
    except (ValueError, KeyError, FileNotFoundError, RuntimeError, AssertionError, TypeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
