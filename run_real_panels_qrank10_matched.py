#!/usr/bin/env python3
"""Fresh equal / ten-component-ranked arms on the historical four-panel design.

Preparation is serialized across Slurm array elements. Only design metadata
from the historical run is used: no historical model or optimizer is loaded.
A bundled reference verifies the historical design without its results folder.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import pandas as pd
import torch
import yaml
from importlib.metadata import version

from Models.utils.gru_precision import GRU_COMPUTE_POLICY
from run_real_exp8_L_stability_quality_rank import inspect_ranking_components

ROOT = Path(__file__).resolve().parent
ARM_NAMES = ('equal', 'qrank10components_p1')
DEFAULT_REFERENCE_DESIGN_MANIFEST = (
    ROOT / 'config/experiment_designs/panels_equal_seed42_20260906_114323.json'
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execution_identity():
    files = {ROOT / name for name in (
        'main_ribounmix_multidataset.py',
        'resume_real_experiment_from_checkpoints.py',
        'run_real_independent_panel_convergence.py',
        'run_real_independent_panel_convergence_quality_rank.py',
        'run_real_exp8_L_stability_quality_rank.py',
        'run_real_panels_qrank10_matched.py',
    )}
    for folder in ('Models', 'Dataloaders', 'Utils'):
        files.update((ROOT / folder).rglob('*.py'))
    return dict(
        sources={str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
        python=platform.python_version(),
        packages={p: version(p) for p in ('torch', 'lightning', 'numpy', 'pandas', 'hydra-core')},
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gru_compute_policy=GRU_COMPUTE_POLICY,
    )


def task_spec(index):
    if not 0 <= index < 8:
        raise ValueError('Task index must be 0..7.')
    panel = index % 4 + 1
    arm = ARM_NAMES[index // 4]
    prefix = 'real_panel_convergence' if arm == 'equal' else 'real_panel_qrank_convergence'
    return arm, f'panel_{panel:02d}', f'{prefix}_panel{panel:02d}'


def read_json(path):
    return json.loads(path.read_text())


def design_identity(root):
    """Portable equivalent of the historical assignment and ordered-ID checks."""
    assignment = pd.read_csv(root / 'panel_assignment.csv')
    columns = ['dataset_name', 'panel', 'source_identifier']
    rows = assignment[columns].astype(str).sort_values('dataset_name')
    split = read_json(root / 'common_split_manifest.json')

    def fingerprint(ids):
        ids = list(map(str, ids))
        # Preserve ordering and duplicates, just like the original list comparison.
        payload = json.dumps(ids, ensure_ascii=True, separators=(',', ':')).encode('utf-8')
        return dict(count=len(ids), ordered_ids_sha256=hashlib.sha256(payload).hexdigest())

    return dict(
        panel_assignment=rows.to_dict(orient='records'),
        common_validation_ids=fingerprint(split['common_validation_ids']),
        common_test_ids=fingerprint(split['common_test_ids']),
        panel_train_eligible_ids={panel: fingerprint(ids)
                                 for panel, ids in split['panel_train_eligible_ids'].items()},
    )


def load_reference_design(args):
    """Use an explicit historical root, or the portable reference shipped with code."""
    if args.reference_design_root is not None:
        reference = args.reference_design_root
        names = ('panel_assignment.csv', 'common_split_manifest.json')
        missing = [name for name in names if not (reference / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f'Incomplete explicit reference design at {reference}: missing {missing}. '
                'Omit --reference-design-root (unset REFERENCE_DESIGN_ROOT in Slurm) '
                'to use the bundled historical design. No historical checkpoints are needed.'
            )
        return dict(schema_version=1, source_run=str(reference),
                    source_files_sha256={name: sha(reference / name) for name in names},
                    design_identity=design_identity(reference))
    path = getattr(args, 'reference_design_manifest', None) or DEFAULT_REFERENCE_DESIGN_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f'Portable historical design is missing: {path}. Deploy this JSON with '
            'the launcher, or provide --reference-design-root with the two design metadata files.'
        )
    reference = read_json(path)
    if reference.get('schema_version') != 1 or 'design_identity' not in reference:
        raise ValueError(f'Unsupported historical design reference: {path}')
    return reference


def validate_historical_design(new_root, reference):
    actual = design_identity(new_root)
    if actual != reference['design_identity']:
        changed = [key for key in actual if actual[key] != reference['design_identity'].get(key)]
        raise RuntimeError(
            f'{new_root.name}: historical panel/split identity differs ({changed}). '
            'Refusing a confounded comparison; no training was launched.'
        )


def frozen_files(root):
    paths = [root / 'frozen_config.yaml', root / 'frozen_dataset_config.yaml',
             root / 'frozen_quality_rank_10components.tsv', root / 'frozen_reference_design.json']
    for arm in ARM_NAMES:
        paths.extend(root / arm / name for name in ('panel_assignment.csv', 'common_split_manifest.json', 'panel_manifest.json'))
        for n in range(1, 5):
            paths.extend(root / arm / f'panel_{n:02d}' / name for name in (
                'launch_command.sh', 'run_manifest.json', 'split_manifest.json',
                'reliability_reference_manifest.json',
            ))
    # resolved_config.yaml is not hashed: the training entrypoint rewrites that
    # audit file. The immutable launch/config snapshots above define preparation.
    return {str(p.relative_to(root)): sha(p) for p in paths}


def input_signatures(root):
    """Check source-file identity cheaply, without re-reading all count arrays."""
    dataset_cfg = yaml.safe_load((root / 'frozen_dataset_config.yaml').read_text())
    cfg = yaml.safe_load((root / 'frozen_config.yaml').read_text())
    paths = [*dataset_cfg['dataset_path'].values(), cfg['paths']['sequences_path']]
    result = {}
    for raw in paths:
        path = (ROOT / raw).resolve()
        info = path.stat()
        result[str(path)] = dict(size=info.st_size, mtime_ns=info.st_mtime_ns)
    return result


def prepare(args):
    root = args.output_root
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f'.{root.name}.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = root / 'matched_experiment.json'
        identity = execution_identity()
        if marker.exists():
            saved = read_json(marker)
            if saved['execution_identity'] != identity:
                raise RuntimeError('Training source/environment changed. Restore the pinned version or use a new output root; do not mix implementations within this comparison.')
            if saved['frozen_files'] != frozen_files(root):
                raise RuntimeError('Frozen design inputs changed; resume refused.')
            if saved['input_signatures'] != input_signatures(root):
                raise RuntimeError('Count/sequence file signatures changed; resume refused.')
            print(f'Reusing frozen matched experiment: {root}', flush=True)
            return
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f'Incomplete preparation at {root}. Inspect preparation logs; use a new output root after correcting the cause. Nothing overwritten.')
        metadata = inspect_ranking_components(args.ranking_table, expected_count=10)
        reference = load_reference_design(args)
        root.mkdir(exist_ok=True)
        (root / 'frozen_reference_design.json').write_text(json.dumps(reference, indent=2) + '\n')
        print(f'Historical design reference: {reference["source_run"]} (metadata only)', flush=True)
        ranking = root / 'frozen_quality_rank_10components.tsv'
        ranking.write_bytes(args.ranking_table.read_bytes())
        cfg = yaml.safe_load(args.config.read_text())
        cfg['trainer']['precision'] = 'bf16-mixed'
        cfg['model']['dataset_bias_params']['context_gru_precision'] = 'float32'
        cfg['model']['dataset_bias_params']['context_gru_tbptt_window'] = 0
        cfg['data']['dataset_quality_ranking']['path'] = str(ranking)
        config = root / 'frozen_config.yaml'
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        dataset_config = root / 'frozen_dataset_config.yaml'
        dataset_config.write_bytes(args.dataset_config.read_bytes())
        signatures = input_signatures(root)
        shared = ['--prepare-only', '--seed', '42', '--gpus', '0',
                  '--output-root', str(root), '--config', str(config),
                  '--dataset-config', str(dataset_config),
                  '--python-executable', sys.executable,
                  '--batch-size', '32', '--num-workers', '0', '--predict-num-workers', '0',
                  '--max-pair-rows-per-forward', '512',
                  '--max-padded-codon-tokens-per-forward', '256000',
                  '--reference-chunk-size', '16', '--log-every-n-steps', '100']
        for arm, runner in zip(ARM_NAMES, ('run_real_independent_panel_convergence.py',
                                         'run_real_independent_panel_convergence_quality_rank.py')):
            command = [sys.executable, '-u', str(ROOT / runner), *shared, '--run-id', arm]
            if arm != 'equal':
                command += ['--reference-design-root', str(root / 'equal'),
                            '--quality-ranking-table', str(ranking), '--quality-rank-power', '1.0',
                            '--quality-table', str(root / 'equal/dataset_quality_table.csv')]
            print(f'Preparing {arm}; details: {root / (arm + "_prepare.log")}', flush=True)
            with (root / f'{arm}_prepare.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            validate_historical_design(root / arm, reference)
        if signatures != input_signatures(root):
            raise RuntimeError('Source inputs changed during preparation; refusing to launch.')
        marker.write_text(json.dumps(dict(
            experiment='matched_equal_vs_qrank10_fixed_historical_panels',
            reference_design_root=str(args.reference_design_root) if args.reference_design_root is not None else None,
            reference_design_manifest=str(root / 'frozen_reference_design.json'),
            historical_source_run=reference['source_run'],
            initialization='Fresh training in both arms; historical design only, no historical checkpoints.',
            ranking_source=str(args.ranking_table), ranking_metadata=metadata,
            ranking_sha256=sha(ranking), execution_identity=identity,
            frozen_files=frozen_files(root), input_signatures=signatures,
            task_mapping={str(i): task_spec(i) for i in range(8)},
            numerical_policy='BF16 mixed; both CUDA GRUs FP32; full BPTT (window=0).',
        ), indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/panels_matched_equal_vs_qrank10_seed42_v1')
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument('--reference-design-root', type=Path,
                           help='Optional historical run containing panel_assignment.csv and common_split_manifest.json. No checkpoints needed.')
    reference.add_argument('--reference-design-manifest', type=Path,
                           help=f'Portable design reference. Default: {DEFAULT_REFERENCE_DESIGN_MANIFEST.relative_to(ROOT)}')
    parser.add_argument('--ranking-table', type=Path, default=ROOT / 'Datasets/data/HEK_riboseq_profile_quality_rank_components.tsv')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/config_ribounmix_multidataset.yaml')
    parser.add_argument('--dataset-config', type=Path, default=ROOT / 'config/dataset_config/weighted_hek_riboseq_codon_replicas.yaml')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare-only', action='store_true')
    action.add_argument('--task-index', type=int, choices=range(8))
    parser.add_argument('--dry-run', action='store_true', help='Prepare and audit selected resume command, without training.')
    args = parser.parse_args(argv)
    for name in ('output_root', 'reference_design_root', 'reference_design_manifest', 'ranking_table', 'config', 'dataset_config'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    prepare(args)
    if args.prepare_only:
        return 0
    arm, panel, run_id = task_spec(args.task_index)
    task = args.output_root / arm / panel
    with (task / '.array_training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [sys.executable, '-u', str(ROOT / 'resume_real_experiment_from_checkpoints.py'),
                   '--run-root', str(args.output_root / arm), '--include-run-ids', run_id,
                   '--gpus', 'inherit', '--use-saved-resolved-config',
                   '--throughput-profile', 'unchanged', '--bias-gru-precision', 'float32',
                   '--bias-gru-tbptt-window', '0', '--summary-path', str(task / 'matched_resume_summary.json')]
        if args.dry_run:
            command.append('--dry-run')
        return subprocess.run(command, cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
