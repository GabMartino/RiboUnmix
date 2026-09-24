#!/usr/bin/env python3
"""Cumulative top-N Exp8 with frozen global quality-ranked gamma references.

One independently initialized model is fitted per N and training seed. Dataset
panels are nested, so convergence measures are dependent and not disjoint-panel
reproducibility. Reuses Exp8's production training commands, held-out split,
train-only reliability fitting, checkpoint selection and GPU queue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import run_real_exp8_L_stability as base
from Utils.real_panel_convergence import infer_source_identifier
from run_real_independent_panel_convergence_quality_rank import _quality_rank_reference

PROJECT_ROOT = base.PROJECT_ROOT
DEFAULT_RANKING = PROJECT_ROOT / 'Datasets/data/HEK_riboseq_profile_quality_rank_components.tsv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'results/real_exp8_L_stability_quality_rank_10components'
DEFAULT_ANALYSIS = PROJECT_ROOT / 'analyses/analyze_real_exp8_cumulative_quality_rank.py'
EXPERIMENT = 'real_exp8_L_stability_cumulative_quality_rank'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--quality-ranking-table', type=Path, default=DEFAULT_RANKING)
    parser.add_argument('--quality-rank-power', type=float, default=1.0)
    parser.add_argument('--expected-rank-components', type=int, default=10)
    parser.add_argument('--design-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    raw = list(sys.argv[1:] if argv is None else argv)
    extra, remaining = parser.parse_known_args(raw)
    if '--help' in raw or '-h' in raw:
        print(__doc__)
        print('Additional options: --quality-ranking-table TSV, --quality-rank-power P '
              '(default 1), --expected-rank-components N (default 10), '
              '--design-only (ranking/subsets only; no profile reads), '
              '--prepare-only (write the complete runnable design without training).')
    args = base.parse_args(remaining)
    args.quality_ranking_table = extra.quality_ranking_table.resolve()
    args.quality_rank_power = extra.quality_rank_power
    args.expected_rank_components = extra.expected_rank_components
    args.design_only = extra.design_only
    args.prepare_only = extra.prepare_only
    if args.output_root == base.DEFAULT_OUTPUT_ROOT:
        args.output_root = DEFAULT_OUTPUT
    if args.analysis_script == base.DEFAULT_ANALYSIS:
        args.analysis_script = DEFAULT_ANALYSIS
    if args.full_run_checkpoint or args.allow_nearest_family_size or args.overwrite_design:
        raise ValueError('Cumulative mode requires exact N and its own full ranked model; '
                         'checkpoint reuse, nearest-family sizes and design overwrites are unsupported.')
    if args.sampling_mode != 'quality_matched' or args.gamma_fingerprint_table:
        raise ValueError('Cumulative mode selects by frozen global rank, not diversity sampling.')
    if not math.isfinite(args.quality_rank_power) or args.quality_rank_power <= 0:
        raise ValueError('--quality-rank-power must be finite and strictly positive.')
    if args.prepare_only and (args.design_only or args.dry_run):
        raise ValueError('--prepare-only cannot be combined with --design-only or --dry-run.')
    if args.expected_rank_components <= 0:
        raise ValueError('--expected-rank-components must be positive.')
    for key in ('batch_size', 'reference_chunk_size', 'max_pair_rows_per_forward',
                'max_padded_codon_tokens_per_forward', 'log_every_n_steps'):
        if getattr(args, key) <= 0:
            raise ValueError(f'{key} must be positive.')
    if not math.isfinite(args.raw_log_gamma_bound) or args.raw_log_gamma_bound <= 0:
        raise ValueError('--raw-log-gamma-bound must be finite and positive.')
    return args


def inspect_ranking_components(ranking_path, expected_count=None):
    """Validate and describe the component ranks recorded in a ranking TSV."""
    ranking = pd.read_csv(ranking_path, sep='\t')
    component_columns = sorted(
        column for column in ranking.columns
        if column.startswith('rank_') and column != 'rank_component_count')
    if 'rank_component_count' not in ranking:
        raise ValueError('Ranking must contain rank_component_count.')
    declared = pd.to_numeric(ranking['rank_component_count'], errors='raise')
    unique_declared = sorted(declared.dropna().unique().tolist())
    actual_count = len(component_columns)
    if unique_declared != [actual_count]:
        raise ValueError(
            f'Ranking declares component counts {unique_declared}, but contains '
            f'{actual_count} rank component columns.')
    if ranking[component_columns].isna().any().any():
        raise ValueError('Ranking contains missing component ranks.')
    if expected_count is not None and actual_count != expected_count:
        raise ValueError(
            f'Expected a {expected_count}-component ranking, but {ranking_path} '
            f'contains {actual_count} components.')
    return {'count': actual_count, 'columns': component_columns}


def build_cumulative_tasks(dataset_names, ranking_path, sizes, training_seeds):
    """Exact top-N prefixes; frozen ranks, with dataset-name tie breaking."""
    ranking = pd.read_csv(ranking_path, sep='\t')
    if not {'dataset', 'quality_rank'} <= set(ranking):
        raise ValueError('Ranking requires dataset and quality_rank columns.')
    if ranking['dataset'].isna().any() or ranking['dataset'].duplicated().any():
        raise ValueError('Ranking contains missing or duplicate dataset identities.')
    ranking['quality_rank'] = pd.to_numeric(ranking['quality_rank'], errors='raise')
    if not np.isfinite(ranking.quality_rank).all() or (ranking.quality_rank <= 0).any():
        raise ValueError('Ranks must be finite and positive.')
    missing = set(dataset_names) - set(ranking.dataset)
    if missing:
        raise ValueError(f'Active datasets absent from ranking: {sorted(missing)}')
    ordered = ranking.loc[ranking.dataset.isin(dataset_names)].sort_values(
        ['quality_rank', 'dataset'], kind='stable').reset_index(drop=True)
    sizes = sorted(sizes)
    if len(sizes) < 2 or len(sizes) != len(set(sizes)) or min(sizes) < 2 or max(sizes) != len(ordered):
        raise ValueError(f'Use at least two distinct sizes >=2 including full active pool N={len(ordered)}.')
    if not training_seeds or len(training_seeds) != len(set(training_seeds)):
        raise ValueError('Training seeds must be nonempty and distinct.')
    ordered['cumulative_position'] = np.arange(1, len(ordered) + 1)
    ordered['source_identifier'] = ordered.dataset.map(infer_source_identifier)
    tasks = []
    for seed in training_seeds:
        for n in sizes:
            prefix = ordered.iloc[:n]
            stem = f'exp8_qrank_N{n:03d}'
            tasks.append(dict(
                run_id=f'{stem}_seed{seed}', N=n, requested_N=n,
                kind='full_collection' if n == len(ordered) else 'cumulative_top_quality',
                subset_id='top_quality', base_run_id=stem, training_seed=seed,
                datasets=prefix.dataset.tolist(),
                source_families=sorted(prefix.source_identifier.unique()),
                target_size=n, actual_size=n,
            ))
    return tasks, ordered


def task_directory(root, task):
    return root / f"N{task['N']:03d}" / f"trainseed{task['training_seed']}"


def ranked_overrides(args, frozen_ranking):
    return {
        'model.gamma_centering.reference.weighting': 'quality_rank',
        'model.gamma_centering.reference.quality_rank_power': args.quality_rank_power,
        'data.dataset_quality_ranking.path': str(frozen_ranking),
        'data.dataset_quality_ranking.dataset_column': 'dataset',
        'data.dataset_quality_ranking.rank_column': 'quality_rank',
        'data.dataset_quality_ranking.strict': True,
    }


def configure_task(args, task, directory, split_path, reliability_path, sequences_path,
                   base_config, dataset_config, frozen_ranking):
    shared = dict(args=args, task=task, task_directory=directory, split_path=split_path,
                  reliability_path=reliability_path, sequences_path=sequences_path,
                  training_seed=task['training_seed'], train=True)
    resolved = base._resolved_config(base=base_config, dataset_config=dataset_config, **shared)
    changes = ranked_overrides(args, frozen_ranking)
    for key, value in changes.items():
        base._set_nested(resolved, key, value)
    resolved['orchestrator'].update(
        experiment=EXPERIMENT, sampling_mode='cumulative_top_quality',
        ranking_source_filename=args.quality_ranking_table.name,
        ranking_component_count=args.ranking_component_count,
        ranking_component_columns=args.ranking_component_columns,
        legacy_quality_rank_used_for_subset_selection=True,
        source_family_atomicity_enforced=False)
    resolved['orchestrator']['controlled_overrides'].update(changes)
    command = base._training_command(**shared)
    command = [part for part in command if part.split('=', 1)[0] not in changes]
    command.extend(f'{key}={json.dumps(value)}' for key, value in changes.items())
    reference = _quality_rank_reference(panel_datasets=task['datasets'],
        ranking_path=frozen_ranking, base_config=resolved, power=args.quality_rank_power)
    return resolved, command, reference


def _signature(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def stable_hash(payload):
    """Ignore audit timestamps, but retain scientific inputs and file signatures."""
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k != 'created_at_utc'}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    return base._canonical_hash(clean(payload))


def main(argv=None):
    args = parse_args(argv)
    for key in ('config', 'dataset_config', 'entrypoint', 'analysis_script', 'output_root'):
        setattr(args, key, getattr(args, key).resolve())
    args.python_executable = args.python_executable.expanduser().absolute()
    seeds = ([args.training_seed] if args.training_seeds is None else
             base._csv_ints(args.training_seeds, option='--training-seeds'))
    sizes = base._csv_ints(args.dataset_sizes, option='--dataset-sizes')
    mapping = base.load_dataset_mapping(args.dataset_config)
    mapping = {name: str((PROJECT_ROOT / path).resolve()) for name, path in mapping.items()}
    cfg = base._load_yaml(args.config)
    dataset_cfg = base._load_yaml(args.dataset_config)
    component_metadata = inspect_ranking_components(
        args.quality_ranking_table, expected_count=args.expected_rank_components)
    args.ranking_component_count = component_metadata['count']
    args.ranking_component_columns = component_metadata['columns']
    tasks, ordered = build_cumulative_tasks(mapping, args.quality_ranking_table, sizes, seeds)
    if len(ordered) != args.expected_dataset_count and not args.allow_excluded_datasets:
        raise ValueError(f'Expected {args.expected_dataset_count} datasets; found {len(ordered)}.')
    run_id = args.run_id or (f'cumulative_qrank10components_p{args.quality_rank_power:g}'
                             f'_seed{args.subset_seed}')
    suffix = '_design_only' if args.design_only else '_dry_run' if args.dry_run else ''
    root = args.output_root / (run_id + suffix)
    if root.exists() and not args.resume:
        raise FileExistsError(f'{root} exists; use a new run ID or --resume.')
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / f'frozen_{args.quality_ranking_table.name}'
    ranking_bytes = args.quality_ranking_table.read_bytes()
    if frozen.exists() and frozen.read_bytes() != ranking_bytes:
        raise ValueError('Ranking differs from the frozen run table; resume refused.')
    frozen.write_bytes(ranking_bytes)
    ordered.to_csv(root / 'cumulative_dataset_order.csv', index=False)
    overlap = base.build_overlap_report(tasks)
    overlap.to_csv(root / 'overlap_report.csv', index=False)
    manifest = dict(experiment_name=EXPERIMENT, tasks=tasks,
        ranking_source=str(args.quality_ranking_table), frozen_ranking_table=str(frozen),
        ranking_sha256=hashlib.sha256(ranking_bytes).hexdigest(),
        ranking_component_count=args.ranking_component_count,
        ranking_component_columns=args.ranking_component_columns,
        quality_rank_power=args.quality_rank_power, gamma_reference_weighting='quality_rank',
        selection_rule='ascending frozen global quality_rank, then dataset name; exact top-N prefixes',
        source_family_atomicity_enforced=False, training_seeds=seeds, subset_seed=args.subset_seed,
        independence='Nested datasets and common held-out transcripts make comparisons dependent.',
        initialization='Every N is trained independently from scratch; no warm starts.',
        rank_scope='Full frozen table; no re-ranking within active datasets or prefixes.',
        experiment_design='cumulative_top_quality', git=base._git_metadata())
    for task in tasks:
        task['directory'] = str(task_directory(root, task).relative_to(root))
    if args.design_only:
        for task in tasks:
            task['fixed_gamma_reference'] = _quality_rank_reference(
                panel_datasets=task['datasets'], ranking_path=frozen,
                base_config={'data': {'dataset_quality_ranking': {
                    'dataset_column': 'dataset', 'rank_column': 'quality_rank'}}},
                power=args.quality_rank_power)
        base.write_json(root / 'cumulative_design.json', manifest)
        print(f'Ranking-only design: {len(tasks)} tasks, N={sorted(sizes)}\n{root}')
        return 0

    sequences = (PROJECT_ROOT / Path(args.sequences_path or cfg['paths']['sequences_path'])).resolve()
    source = base._inspect_weighted_dataset_sources(mapping, dataset_config_path=args.dataset_config)
    base.write_json(root / 'dataset_source_preflight.json', source)
    if source['status'] != 'PASS':
        raise RuntimeError('Weighted input source preflight failed; see dataset_source_preflight.json.')
    print(f'Weighted source preflight passed for {len(mapping)} datasets. '
          'Computing quality diagnostics (or validating --quality-table cache)...', flush=True)
    if args.resume and args.quality_table is None and (root / 'dataset_quality_table.csv').is_file():
        # The shared loader validates paths, sizes, mtimes and sequence eligibility.
        args.quality_table = root / 'dataset_quality_table.csv'
    quality, exclusions, report = base._load_or_compute_quality(
        args=args, dataset_mapping=mapping, sequences_path=sequences)
    quality.to_csv(root / 'dataset_quality_table.csv', index=False)
    if exclusions:
        raise ValueError('Cumulative design requires every configured dataset eligible; '
                         'create an explicit filtered dataset config and adjust sizes/count.')
    pool, families, _ = base.prepare_quality_pool(quality)
    source_by_name = pool.set_index('dataset_name')['source_identifier'].to_dict()
    for task in tasks:
        task['source_families'] = sorted({source_by_name[name] for name in task['datasets']})
    base.build_overlap_report(tasks).to_csv(root / 'overlap_report.csv', index=False)
    ordered['source_identifier'] = ordered.dataset.map(source_by_name)
    ordered.to_csv(root / 'cumulative_dataset_order.csv', index=False)
    from Utils.real_exp8_stability import quality_mismatch
    for task in tasks:
        task['quality_mismatch'] = quality_mismatch(task['datasets'], pool,
            z_columns=[key for key in pool if key.startswith('z__')])
    quality_report = base.summarize_subset_quality(tasks, pool)
    quality_report['legacy_scalar_quality_rank_used'] = True
    quality_report.to_csv(root / 'subset_quality_report.csv', index=False)
    print('Building common held-out transcript set and per-task folds...', flush=True)
    split = base.build_exp8_transcript_split(experiment_name=EXPERIMENT, tasks=tasks,
        dataset_mapping=mapping, sequences_path=sequences, subset_seed=args.subset_seed,
        validation_fraction=args.validation_fraction, test_fraction=args.test_fraction,
        reliability_bins=args.reliability_bins, maximum_cds_codons=args.max_cds_codons,
        reused_test_ids=base._load_reused_test_ids(args.experiment1_manifest))
    split_path = root / 'experiment_split_manifest.json'
    if args.resume and split_path.exists():
        if stable_hash(json.loads(split_path.read_text())) != stable_hash(split):
            raise ValueError('Transcript split changed; resume refused.')
    base.write_json(split_path, split)
    test_ids = split['common_test_ids']
    test_hash = base.transcript_id_hash(test_ids)
    base.write_json(root / 'common_test_manifest.json', dict(common_test_ids=test_ids,
        transcript_id_hash=test_hash, sequence_only_inference=True))
    commands, directories, skipped = {}, {}, []
    for task in tasks:
        name = task['run_id']
        directory = task_directory(root, task)
        directories[name] = directory
        directory.mkdir(parents=True, exist_ok=True)
        for child in ('checkpoints', 'predictions', 'logs'):
            (directory / child).mkdir(parents=True, exist_ok=True)
        reliability_path = directory / 'reliability_reference_manifest.json'
        print(f"Preparing {name}: N={task['N']}, train-only reliability and ranked reference...", flush=True)
        reliability = base.fit_panel_reliability_manifest(experiment_name=EXPERIMENT,
            panel_name=name, panel_datasets=task['datasets'], dataset_mapping=mapping,
            panel_training_ids=split['panel_train_eligible_ids'][name],
            validation_ids=split['panel_validation_ids'][name], test_ids=test_ids,
            source_split_manifest=split_path)
        if reliability.get('heldout_rows_used_for_fitting') != 0:
            raise AssertionError('Reliability fitting used held-out observations.')
        resolved, command, reference = configure_task(args, task, directory, split_path,
            reliability_path, sequences, cfg, dataset_cfg, frozen)
        design_hash = stable_hash(dict(config=resolved, reference=reference,
            reliability=reliability, split=split, inputs=[_signature(mapping[n]) for n in task['datasets']],
            sequences=_signature(sequences)))
        previous = directory / 'subset_manifest.json'
        if previous.exists() and json.loads(previous.read_text()).get('design_hash') != design_hash:
            raise ValueError(f'{name}: design/input hash changed; resume refused.')
        if args.resume and base._completed_run(directory, design_hash, test_hash):
            skipped.append(name)
            continue
        base.write_json(reliability_path, reliability)
        base.write_json(previous, dict(**task, design_hash=design_hash,
            fixed_gamma_reference=reference, reliability_weight=dict(symbol='w_dt',
                fit_split='training_only', separate_from_pi=True),
            checkpoint_selection='best_val_loss', source_family_atomicity_enforced=False))
        (directory / 'resolved_config.yaml').write_text(yaml.safe_dump(base.json_ready(resolved), sort_keys=False))
        (directory / 'launch_command.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + shlex.join(command) + '\n')
        commands[name] = command
    manifest.update(common_test_manifest=str(root / 'common_test_manifest.json'),
                    sequence_eligibility_report=report, source_family_mapping=families)
    base.write_json(root / 'experiment_manifest.json', manifest)
    base.write_json(root / 'launch_manifest.json', dict(commands=commands, skipped_completed=skipped))
    print(f'{len(tasks)} cumulative ranked runs; {len(commands)} pending; {len(skipped)} complete.\n{root}')
    if args.prepare_only:
        print('Preparation complete; no training launched.')
        return 0
    if args.dry_run:
        print('Dry run complete; no training launched.')
        return 0
    gpus = base._parse_gpus(args.gpus)
    base._validate_requested_gpus(gpus)
    statuses = base._queue_tasks(commands=commands, task_directories=directories, gpus=gpus)
    base.write_json(root / 'training_summary.json', dict(statuses=statuses, skipped=skipped))
    for task in tasks:
        if statuses.get(task['run_id'], {}).get('completed'):
            base._consolidate_run(task=task, task_directory=directories[task['run_id']], test_ids=test_ids)
    if set(statuses) != set(commands) or any(not s.get('completed') for s in statuses.values()):
        return 1
    return subprocess.run([str(args.python_executable), str(args.analysis_script),
        '--run-root', str(root), '--bootstrap-seed', str(args.bootstrap_seed)], check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
