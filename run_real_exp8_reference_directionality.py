#!/usr/bin/env python3
"""Matched cumulative Exp8: equal, ranked and reversed references, three seeds.

Reuse the completed ten-component cumulative design's ordered prefixes, folds
and numerical training-only reliability references. Nothing is repartitioned
or refitted. The first invocation freezes the new configurations and production
code under a lock; subsequent array elements execute exactly one frozen task.
No separate preparation job is required. --dry-run freezes/inspects the plan
on CPU without training; execution requires --authorize-training.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml
from hydra import compose, initialize_config_dir

from prepare_rank_balanced_reference_directionality import (
    ARMS, _assert_arm_match, _reference_policy, initialization_audit,
    object_sha256, parse_training_seeds, read_json, sha256, write_json,
)
from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import (
    load_dataset_quality_ranking,
)
from Utils.external_transcript_split import load_external_transcript_split
from Utils.panel_reference_preparation import assert_training_contract, relocated_input, snapshot_code
from Utils.reliability_references import load_reliability_reference_manifest, transcript_id_hash
from run_real_exp8_L_stability import _consolidate_run
from run_real_exp8_L_stability_quality_rank import inspect_ranking_components


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / 'results/real_exp8_L_stability_quality_rank_10components/cumulative_qrank10components_p1.0_seed42'
DEFAULT_OUTPUT = ROOT / 'results/real_exp8_cumulative_qrank10_directionality'
SIZES = (2, 5, 10, 20, 40, 80, 114)


def environment():
    return dict(python=platform.python_version(), packages={
        name: importlib.metadata.version(name)
        for name in ('torch', 'lightning', 'numpy', 'pandas', 'pyarrow', 'hydra-core', 'omegaconf')
    })


def task_matrix(sizes, seeds):
    """Stable indices: seed, then N, then arm; adjacent jobs are matched arms."""
    return [dict(array_index=i, N=n, training_seed=seed, arm=arm,
                 run_id=f'exp8_direction_N{n:03d}_{arm}_seed{seed}',
                 directory=f'runs/seed{seed}/{arm}/N{n:03d}')
            for i, (seed, n, arm) in enumerate(
                (seed, n, arm) for seed in seeds for n in sizes for arm in ARMS)]


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if not Path(name).is_file() or sha256(Path(name)) != expected:
            raise ValueError(f'Frozen input/configuration/code changed: {name}')


def load_source(source, sizes):
    """Load design inputs only; historical predictions/checkpoints are not read."""
    manifest_path = source / 'experiment_manifest.json'
    manifest = read_json(manifest_path)
    if manifest.get('experiment_design') != 'cumulative_top_quality':
        raise ValueError('The source must be the original cumulative top-quality design.')
    ranking = source / Path(manifest['frozen_ranking_table']).name
    components = inspect_ranking_components(ranking, expected_count=10)
    if sha256(ranking) != manifest['ranking_sha256']:
        raise ValueError('Frozen source ranking checksum mismatch.')
    ranks, quality = load_dataset_quality_ranking(str(ranking))
    source_tasks = [t for t in manifest['tasks'] if int(t['training_seed']) == 42]
    by_n = {int(t['N']): t for t in source_tasks}
    if len(by_n) != len(source_tasks) or set(sizes) - set(by_n):
        raise ValueError('Source has duplicate/missing seed-42 prefix templates.')
    full = by_n[max(by_n)]['datasets']
    if len(full) != len(set(full)) or set(full) - set(ranks):
        raise ValueError('Duplicated source datasets or missing global ranks.')
    if full != sorted(full, key=lambda d: (ranks[d], d)):
        raise ValueError('Source order is not ascending frozen global rank, then ID.')
    split_path = source / 'experiment_split_manifest.json'
    split = read_json(split_path)
    templates, references, folds = {}, {}, {}
    inputs = [manifest_path, ranking, split_path]
    for n in sizes:
        task = by_n[n]
        names = task['datasets']
        if names != full[:n] or len(names) != n:
            raise ValueError(f'N={n}: not the frozen exact top-N prefix.')
        directory = source / task['directory']
        config_path = directory / 'resolved_config.yaml'
        cfg = yaml.safe_load(config_path.read_text())
        if cfg['experiment']['dataset'] != names or cfg['experiment']['seed'] != 42:
            raise ValueError(f'N={n}: source configuration membership/seed mismatch.')
        train, validation, test, _ = load_external_transcript_split(
            split_path, panel_name=task['run_id'], experiment_datasets=names)
        # Exp8 has a common TEST set, but validation is subset-specific. The
        # legacy common_validation_ids field aliases the first subset only.
        # The production split loader above checks the authoritative per-N
        # validation IDs and excludes them and the common test IDs from train.
        ref_path = directory / 'reliability_reference_manifest.json'
        ref = load_reliability_reference_manifest(ref_path)
        if (ref.get('reference_split') != 'training_only'
                or ref.get('heldout_rows_used_for_fitting') != 0
                or ref['panel_training_transcript_id_hash'] != transcript_id_hash(train)
                or ref['panel_training_transcript_count'] != len(train)
                or set(ref['datasets']) != set(names)):
            raise ValueError(f'N={n}: reliability references do not match the frozen training list.')
        templates[n], references[n] = cfg, ref_path
        folds[n] = dict(source_panel=task['run_id'], train_ids=train,
                        validation_ids=validation, test_ids=test)
        inputs.extend([config_path, ref_path])
    return dict(manifest=manifest, ranking=ranking, components=components,
                ranks=ranks, quality=quality, templates=templates, references=references,
                folds=folds, split_path=split_path, inputs=inputs)


def task_config(template, task, root, source_panel, implementation):
    cfg = copy.deepcopy(template)
    # The source's resolved YAML already embeds dataset_config: do not import a
    # potentially changed base YAML through its obsolete Hydra defaults list.
    cfg.pop('defaults', None)
    cfg['name'] = task['run_id']
    cfg['experiment'].update(seed=task['training_seed'], from_checkpoint=False,
        resume_training_state=False, resume_checkpoint_path=None,
        allow_weights_only_resume=False, train=True, predict=True)
    cfg['prediction'].update(checkpoint_variants=['best_val_loss'], sequence_only_shared_profile=True)
    cfg['split'].update(external_manifest=str(root / 'frozen_inputs/experiment_split_manifest.json'),
                        external_panel_name=source_panel)
    cfg['data']['reliability_reference_manifest'] = str(
        root / f'frozen_inputs/N{task["N"]:03d}_reliability_reference_manifest.json')
    cfg['data']['dataset_quality_ranking'].update(
        path=str(root / 'frozen_inputs/HEK_riboseq_profile_quality_rank_components.tsv'),
        dataset_column='dataset', rank_column='quality_rank', strict=True)
    # Match the safe production execution used in the four-panel rerun. These
    # are identical in all arms: no TBPTT or policy-specific hyperparameters.
    cfg['model']['mean_correction'] = 'learned'
    cfg['model']['dataset_bias_params'].update(context_gru_precision='float32', context_gru_tbptt_window=0)
    reference = cfg['model']['gamma_centering']['reference']
    reference.pop('explicit_weights', None)
    reference.update(dataset_names=None, **implementation)
    cfg['trainer'].update(devices=[0], num_nodes=1, precision='bf16-mixed', use_distributed_sampler=False)
    cfg['data']['num_workers'] = 0  # Avoid copying the full in-memory compendium into spawn workers.
    cfg['data']['predict_num_workers'] = 0
    task_root = root / task['directory']
    cfg['paths'].update(checkpoints=str(task_root / 'checkpoints'), logs=str(task_root / 'logs'),
                        results=str(task_root / 'predictions'))
    cfg['orchestrator'] = dict(experiment='exp8_cumulative_reference_directionality',
        N=task['N'], arm=task['arm'], training_seed=task['training_seed'], source_panel=source_panel)
    assert_training_contract(cfg)
    return cfg


def freeze_plan(args):
    """One locked, automatic freeze, not a panel-search/preparation-job pipeline."""
    root = args.output_root
    if root == args.source_root or root.is_relative_to(args.source_root):
        raise ValueError('Use a distinct output root outside the historical experiment.')
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f'.{root.name}.freeze.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = root / 'experiment_manifest.json'
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if (manifest['training_seeds'] != list(args.training_seeds)
                    or manifest['sizes'] != list(args.sizes)
                    or manifest['source_root'] != str(args.source_root)):
                raise ValueError('Requested design differs from the frozen experiment; use a new output root.')
            if object_sha256(manifest['tasks']) != manifest['tasks_sha256']:
                raise ValueError('Frozen task mapping changed.')
            return manifest
        if (root / 'runs').exists() and any((root / 'runs').rglob('execution_status.json')):
            raise ValueError('Existing training without an experiment manifest; refusing to overwrite.')
        source = load_source(args.source_root, args.sizes)
        frozen, configs = root / 'frozen_inputs', root / 'resolved_configs'
        frozen.mkdir(parents=True, exist_ok=True)
        configs.mkdir(exist_ok=True)
        copies = [(source['ranking'], frozen / 'HEK_riboseq_profile_quality_rank_components.tsv'),
                  (source['split_path'], frozen / 'experiment_split_manifest.json')]
        copies += [(path, frozen / f'N{n:03d}_reliability_reference_manifest.json')
                   for n, path in source['references'].items()]
        for src, dest in copies:
            shutil.copy2(src, dest)  # Numerical references and folds are copied byte for byte.
        source_hashes = {str(p): sha256(p) for p in source['inputs']}
        code_hashes = snapshot_code(root / 'code_snapshot')
        assets, relocations = {}, []
        for cfg in source['templates'].values():
            cfg['paths']['sequences_path'] = str(relocated_input(cfg['paths']['sequences_path']))
            for mapping in (cfg['paths']['encodings'], cfg['dataset_config']['dataset_path']):
                for key, value in mapping.items():
                    actual = relocated_input(value)
                    if str(actual) != value:
                        relocations.append(dict(recorded=value, resolved=str(actual)))
                    mapping[key] = str(actual)
            for path in [cfg['paths']['sequences_path'], *cfg['paths']['encodings'].values(),
                         *cfg['dataset_config']['dataset_path'].values()]:
                if path not in assets:
                    assets[path] = sha256(Path(path))
        tasks = task_matrix(args.sizes, args.training_seeds)
        weights, initializations, differences, configuration_hashes = [], [], [], {}
        for task in tasks:
            n, seed, arm = task['N'], task['training_seed'], task['arm']
            template = source['templates'][n]
            names = template['experiment']['dataset']
            implementation, weight_rows = _reference_policy(arm, names, source['ranks'], source['quality'])
            if seed == args.training_seeds[0]:
                weights.extend(dict(N=n, **row) for row in weight_rows)
            cfg = task_config(template, task, root, source['folds'][n]['source_panel'], implementation)
            initial = initialization_audit(cfg)
            expected_buffer = hashlib.sha256(np.asarray(
                [row['assigned_q'] for row in weight_rows], dtype=np.float32).tobytes()).hexdigest()
            if initial['reference_buffer_sha256'] != expected_buffer:
                raise ValueError(f'{task["run_id"]}: production reference buffer differs from the planned q values.')
            initializations.append(dict(N=n, training_seed=seed, arm=arm, **initial))
            cfg['orchestrator']['initialization_sha256'] = initial['parameter_sha256']
            cfg['orchestrator']['task_contract_sha256'] = object_sha256(cfg)
            path = configs / f'{task["run_id"]}.yaml'
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            with initialize_config_dir(version_base=None, config_dir=str(configs)):
                compose(config_name=path.stem)
            configuration_hashes[str(path)] = sha256(path)
            task.update(datasets=names, source_panel=source['folds'][n]['source_panel'],
                config_path=str(path), config_sha256=configuration_hashes[str(path)],
                initialization_sha256=initial['parameter_sha256'],
                task_contract_sha256=cfg['orchestrator']['task_contract_sha256'])
            task_root = root / task['directory']
            task_root.mkdir(parents=True, exist_ok=True)
            task['command'] = [sys.executable, '-u', str(root / 'code_snapshot/main_ribounmix_multidataset.py'),
                f'--config-path={configs}', f'--config-name={path.stem}',
                f'hydra.run.dir={task_root / "hydra"}', 'hydra.job.chdir=false']
            print(f'Frozen {task["run_id"]}', flush=True)
        for seed in args.training_seeds:
            for n in args.sizes:
                matched = {t['arm']: yaml.safe_load(Path(t['config_path']).read_text())
                           for t in tasks if t['N'] == n and t['training_seed'] == seed}
                checks = [r for r in initializations if r['N'] == n and r['training_seed'] == seed]
                if len({r['parameter_sha256'] for r in checks}) != 1:
                    raise ValueError(f'N={n}/seed{seed}: unmatched trainable initializations.')
                for arm in ('ranked', 'reverse'):
                    differences.extend(dict(N=n, training_seed=seed, arm=arm, **row)
                                       for row in _assert_arm_match(matched['equal'], matched[arm]))
        weight_table = pd.DataFrame(weights)
        weight_table.to_csv(root / 'reference_weights.csv', index=False)
        summaries = []
        for (n, arm), frame in weight_table.groupby(['N', 'arm']):
            pi = frame.pi.to_numpy(float)
            summaries.append(dict(N=n, arm=arm, N_ref=1 / (pi @ pi),
                min_pi=pi.min(), max_pi=pi.max(), weighted_mean_rank=pi @ frame.global_rank,
                total_variation_from_equal=0.5 * np.abs(pi - 1 / n).sum()))
        pd.DataFrame(summaries).to_csv(root / 'reference_concentration.csv', index=False)
        pd.DataFrame(tasks).drop(columns=['command','datasets']).to_csv(root / 'task_matrix.csv', index=False)
        pd.DataFrame(initializations).to_csv(root / 'initialization_audit.csv', index=False)
        pd.DataFrame(differences).to_csv(root / 'paired_configuration_differences.csv', index=False)
        manifest = dict(schema_version=1, experiment_design='matched_cumulative_reference_directionality',
            status='frozen_not_launched', source_root=str(args.source_root), source_file_sha256=source_hashes,
            output_root=str(root), sizes=list(args.sizes), training_seeds=list(args.training_seeds), arms=list(ARMS),
            number_of_tasks=len(tasks), tasks=tasks, tasks_sha256=object_sha256(tasks),
            source_folds=source['folds'], ranking=dict(sha256=sha256(source['ranking']),
                frozen_copy=str(copies[0][1]), R=max(source['ranks'].values()),
                components=source['components']['columns'], direction='1 = best',
                transcript_scope='not verified; no transfer of train-only w_dt provenance to the QC ranking'),
            software=environment(), asset_sha256=assets, code_sha256=code_hashes,
            frozen_file_sha256={**configuration_hashes, **{str(p): sha256(p) for p in frozen.iterdir()}},
            path_relocations=relocations,
            historical_changes=['fresh initialization for every task, including ranked seed42',
                'one current frozen code snapshot; historical implementation equality is not verified',
                'BF16 mixed with FP32 recurrent kernels, full BPTT', 'num_workers=0; one logical GPU',
                'resolved embedded dataset_config rather than recomposing old Hydra defaults'],
            historical_data_identity='Historical byte-level data provenance not verified; current input bytes frozen identically across all new arms.',
            source_family_atomicity_enforced=False,
            interpretation='Nested top-quality prefixes are dependent and add progressively poorer-ranked datasets; this is not source-disjoint reproducibility.')
        contract = dict(primary='PCC between adjacent N models within policy and seed on a common finite test cohort',
            size_pairs=[list(p) for p in zip(args.sizes[:-1], args.sizes[1:])],
            secondary=['same-N cross-policy sensitivity', 'same-N same-policy cross-seed variability',
                       'RMSE/amplitude and observation-fit diagnostics'],
            reference_policies=dict(equal='uniform', ranked='global q=(R-r+1)/R; power one',
                reverse='same prefix q multiset reassigned in reverse global-rank/ID order'),
            comparison='Differences of matched summaries; no comparison with representative or disjoint subsets',
            bootstrap=dict(draws=5000, seed=20260910, cluster='transcript, carrying all N comparisons and policies together',
                           scope='conditional on fitted models; show each training seed'),
            controls='Within N, all arms/seeds reuse identical folds and numerical w_dt references. Across N only the test list is common; preserve the original subset-specific validation folds.',
            limitations='At small N the top-ranked q values are almost equal; do not increase the exponent after inspecting results to force a difference.')
        write_json(root / 'comparison_contract.json', contract)
        # Written last: an interrupted freeze can be retried, but no task may
        # train against an incomplete set of configurations.
        write_json(manifest_path, manifest)
        return manifest


def validate_outputs(root, task, test_ids):
    from analyses.analyze_real_panel_convergence import _extract_panel_profiles
    _consolidate_run(task=task, task_directory=root, test_ids=test_ids)
    selected = read_json(root / 'selected_checkpoint.json')
    profiles, _ = _extract_panel_profiles(panel_name=f'N{task["N"]:03d}', run_identifier=task['run_id'],
        prediction_path=Path(selected['shared_profile_path']), expected_ids=set(test_ids), mean_one_tolerance=1e-4)
    return dict(n_test=len(profiles), prediction_sha256=sha256(Path(selected['shared_profile_path'])),
                checkpoint_sha256=sha256(Path(selected['checkpoint_path'])), **selected)


def run_one(args, manifest):
    task = manifest['tasks'][args.task_index]
    if not args.authorize_training:
        raise ValueError('Training requires --authorize-training; --dry-run never trains.')
    cfg = yaml.safe_load(Path(task['config_path']).read_text())
    if environment() != manifest['software']:
        raise ValueError('Runtime Python/library versions differ from this experiment freeze.')
    verify_hashes(manifest['code_sha256'])
    verify_hashes(manifest['frozen_file_sha256'])
    paths = [cfg['paths']['sequences_path'], *cfg['paths']['encodings'].values(),
             *(cfg['dataset_config']['dataset_path'][d] for d in task['datasets'])]
    verify_hashes({p: manifest['asset_sha256'][p] for p in paths})
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one scheduler-visible CUDA GPU is required.')
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError('Native BF16 is required; allocate a supported GPU (exclude dgx1).')
    root = args.output_root / task['directory']
    state_path = root / 'execution_status.json'
    with (root / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read_json(state_path) if state_path.exists() else None
        test = manifest['source_folds'][str(task['N'])]['test_ids']
        if previous and previous['config_sha256'] != task['config_sha256']:
            raise ValueError('Earlier attempt has a different frozen configuration.')
        if previous and previous['status'] == 'completed':
            if validate_outputs(root, task, test) != previous['outputs']:
                raise ValueError('Previously validated outputs changed.')
            print(f'Already completed and validated: {task["run_id"]}')
            return 0
        if previous and not args.resume:
            raise ValueError('Earlier attempt exists; use --resume for full-state continuation.')
        command = list(task['command'])
        command[0] = sys.executable
        resume_checkpoint = None
        existing = list((root / 'checkpoints').rglob('*.ckpt'))
        if existing:
            if not previous or not args.resume:
                raise ValueError('Existing checkpoints cannot initialize a fresh task.')
            from resume_real_experiment_from_checkpoints import _select_resume_checkpoint
            checkpoint, metadata, rejected = _select_resume_checkpoint(root)
            if (checkpoint is None or not metadata['has_optimizer_state'] or not metadata['has_scheduler_state']):
                raise ValueError(f'No usable full-state checkpoint. Rejected: {rejected}')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved.get('hyper_parameters', {}).get('campaign_task_contract_sha256') != task['task_contract_sha256']:
                raise ValueError('Resume checkpoint does not belong to the frozen task.')
            del saved
            resume_checkpoint = dict(**metadata, sha256=sha256(checkpoint))
            command += ['experiment.from_checkpoint=true', 'experiment.resume_training_state=true',
                        f'experiment.resume_checkpoint_path={checkpoint}', 'experiment.allow_weights_only_resume=false']
        attempt = dict(started_unix=time.time(), hostname=platform.node(), python=sys.executable,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'), slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=torch.cuda.get_device_name(0),
            resume_checkpoint=resume_checkpoint, command=command)
        state = dict(task_id=task['run_id'], config_sha256=task['config_sha256'], status='running',
                     attempts=[*(previous['attempts'] if previous else []), attempt])
        write_json(state_path, state)
        try:
            result = subprocess.run(command, cwd=args.output_root / 'code_snapshot', check=False)
            attempt.update(exit_code=result.returncode, finished_unix=time.time())
            if result.returncode:
                state['status'] = 'failed'
                write_json(state_path, state)
                return result.returncode
            state['outputs'] = validate_outputs(root, task, test)
            state['status'] = 'completed'
        except Exception as exc:
            state.update(status='failed', reason=str(exc))
            write_json(state_path, state)
            raise
        write_json(state_path, state)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--training-seeds', type=parse_training_seeds, default=(42,43,44))
    parser.add_argument('--sizes', type=parse_training_seeds, default=SIZES)
    parser.add_argument('--task-index', type=int)
    parser.add_argument('--dry-run', action='store_true', help='Freeze/inspect configurations on CPU; no training.')
    parser.add_argument('--authorize-training', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if tuple(sorted(args.sizes)) != args.sizes or min(args.sizes) < 2 or len(args.sizes) < 2:
        parser.error('--sizes must contain at least two increasing values >=2.')
    if args.task_index is not None and not 0 <= args.task_index < len(task_matrix(args.sizes, args.training_seeds)):
        parser.error('--task-index is outside the selected matrix.')
    if not args.dry_run and (args.task_index is None or not args.authorize_training):
        parser.error('Use --dry-run or --task-index INDEX --authorize-training.')
    try:
        manifest = freeze_plan(args)
        print(f'{manifest["number_of_tasks"]} fresh trainings; {len(args.sizes)*3} per seed; {args.output_root}', flush=True)
        if args.dry_run:
            rows = manifest['tasks'] if args.task_index is None else [manifest['tasks'][args.task_index]]
            print(pd.DataFrame(rows)[['array_index','run_id','N','arm','training_seed']].to_string(index=False))
            print('CPU freeze/inspection only. No training launched.')
            return 0
        # JSON serialization normalizes integer dictionary keys in the same way
        # on the first worker as on subsequent workers.
        return run_one(args, read_json(args.output_root / 'experiment_manifest.json'))
    except (ValueError, FileNotFoundError, KeyError, RuntimeError, BlockingIOError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
