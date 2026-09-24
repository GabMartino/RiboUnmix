#!/usr/bin/env python3
"""Train four fixed panels under five reference policies and a gamma=1 baseline.

Read panel membership from a small repository design file; rebuild transcript
splits and training-only reliability references directly from current data.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

from run_cumulative_stability import (
    ROOT, POLICIES, make_config, object_sha256, run_task, sha256, write_json,
)
from Utils.real_panel_convergence import (
    assert_panel_partition, build_common_transcript_split,
    fit_panel_reliability_manifest, infer_source_identifier,
)

DESIGN = 'four_panel_stability_from_data_v1'
DEFAULT_OUTPUT = ROOT / 'results/four_panel_stability_seed42'
DEFAULT_PANELS = ROOT / 'config/experiment_designs/panels_equal_seed42_20260906_114323.json'
TASK_COUNT = 4 * (len(POLICIES) + 1)


def add_shared_only(root, plan):
    """Append four baselines by cloning the exact equal-arm configurations.

    Existing indices 0..19, folds and model configs stay intact. This also
    extends an already prepared experiment without refitting any inputs.
    """
    if any(task['arm'] == 'shared_only' for task in plan['tasks']):
        return plan
    for equal in [task for task in plan['tasks'] if task['arm'] == 'equal']:
        if sha256(equal['config_path']) != equal['config_sha256']:
            raise ValueError('The equal-arm configuration changed after preparation.')
        cfg = yaml.safe_load(Path(equal['config_path']).read_text())
        task = copy.deepcopy(equal)
        panel, seed = task['panel_id'], task['training_seed']
        task.update(array_index=len(plan['tasks']), arm='shared_only',
                    run_id=f'four_{panel}_shared_only_seed{seed}',
                    directory=f'runs/seed{seed}/shared_only/{panel}')
        cfg['name'] = task['run_id']
        cfg['model']['mean_correction'] = 'unity'
        directory = root / task['directory']
        cfg['paths'].update(checkpoints=str(directory / 'checkpoints'),
                            logs=str(directory / 'logs'), results=str(directory / 'predictions'))
        cfg['orchestrator']['arm'] = 'shared_only'
        cfg['orchestrator'].pop('task_contract_sha256')
        cfg['orchestrator']['task_contract_sha256'] = object_sha256(cfg)
        path = root / 'configs' / f'{task["run_id"]}.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        directory.mkdir(parents=True, exist_ok=True)
        task.update(config_path=str(path), config_sha256=sha256(path),
                    task_contract_sha256=cfg['orchestrator']['task_contract_sha256'])
        task['command'] = [sys.executable, '-u', str(ROOT / 'main_ribounmix_multidataset.py'),
                           f'--config-path={path.parent}', f'--config-name={path.stem}',
                           f'hydra.run.dir={directory}/hydra', 'hydra.job.chdir=false']
        plan['tasks'].append(task)
        plan['frozen_file_sha256'][str(path)] = sha256(path)
    weight_path = root / 'reference_weights.csv'
    weights = pd.read_csv(weight_path)
    # Drop an uncommitted append if a previous setup was interrupted.
    weights = weights[weights.arm != 'shared_only']
    new_weights = weights[weights.arm == 'equal'].assign(arm='shared_only')
    weights = pd.concat([weights, new_weights], ignore_index=True)
    weights.to_csv(weight_path, index=False)
    weights.groupby(['panel_id', 'arm', 'source_family'], as_index=False).pi.sum().to_csv(
        root / 'source_family_reference_mass.csv', index=False)
    pd.DataFrame(plan['tasks']).drop(columns=['command', 'datasets']).to_csv(root / 'task_matrix.csv', index=False)
    plan['tasks_sha256'] = object_sha256(plan['tasks'])
    plan['frozen_file_sha256'][str(weight_path)] = sha256(weight_path)
    plan['shared_only_ablation'] = dict(
        task_indices=[task['array_index'] for task in plan['tasks'] if task['arm'] == 'shared_only'],
        primary_comparison='full equal reference versus shared_only; all six panel pairs',
        mean_correction='gamma=1 during fresh training and prediction; learned alpha retained',
        limitation='The alpha head still learns, but its detached dataset-context encoder receives no gradient in shared_only; this is a whole-system ablation, not a dispersion-controlled gamma effect.')
    write_json(root / 'experiment_manifest.json', plan)
    print('Added four shared-only baselines at indices 20..23; equal-arm inputs reused exactly.', flush=True)
    return plan


def load_panels(path, mapping, ranking):
    """Keep declared membership; order each panel by the unchanged global rank."""
    if path.suffix == '.json':
        rows = json.loads(path.read_text())['design_identity']['panel_assignment']
        assignment = pd.DataFrame(rows)
    else:
        assignment = pd.read_csv(path).rename(columns={
            'dataset_id': 'dataset_name', 'panel_id': 'panel',
            'source_family': 'source_identifier'})
    if 'source_identifier' not in assignment:
        assignment['source_identifier'] = assignment.dataset_name.map(infer_source_identifier)
    assert_panel_partition(assignment, expected_datasets=list(mapping))
    if assignment.panel.nunique() != 4:
        raise ValueError('The design must contain exactly four panels.')
    ranks = ranking.set_index('dataset').quality_rank
    if (ranks.index.has_duplicates or not np.isfinite(ranks).all()
            or (ranks <= 0).any() or not set(mapping).issubset(ranks.index)):
        raise ValueError('Every selected dataset needs a finite positive global rank, with unique dataset rows.')
    assignment['global_rank'] = assignment.dataset_name.map(ranks)
    assignment = assignment.sort_values(['panel', 'global_rank', 'dataset_name'])
    panels = {panel: group.dataset_name.tolist()
              for panel, group in assignment.groupby('panel', sort=True)}
    return panels, assignment, ranks


def prepare(args):
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.setup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request = dict(seed=args.seed, config_sha256=sha256(args.config),
                       dataset_yaml_sha256=sha256(args.datasets),
                       ranking_sha256=sha256(args.ranking), panels_sha256=sha256(args.panels))
        plan_path = root / 'experiment_manifest.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if (plan.get('output_root') != str(root) or plan.get('setup_request') != request
                    or plan.get('experiment_design') != DESIGN):
                raise ValueError('This directory has a different setup or machine path; use a fresh --output-root.')
            return add_shared_only(root, plan)

        base = yaml.safe_load(args.config.read_text())
        mapping = yaml.safe_load(args.datasets.read_text())['dataset_path']
        mapping = {name: str((ROOT / path).resolve()) for name, path in mapping.items()}
        base['paths']['sequences_path'] = str((ROOT / base['paths']['sequences_path']).resolve())
        base['paths']['encodings'] = {
            name: str((ROOT / path).resolve()) for name, path in base['paths']['encodings'].items()}
        panels, assignment, ranks = load_panels(args.panels, mapping, pd.read_csv(args.ranking, sep='\t'))
        inputs, configs = root / 'inputs', root / 'configs'
        inputs.mkdir(exist_ok=True)
        configs.mkdir(exist_ok=True)
        shutil.copy2(args.ranking, inputs / 'ranking.tsv')
        assignment.to_csv(inputs / 'panel_assignment.csv', index=False)

        print('Building common validation/test transcripts from the four panels...', flush=True)
        split = build_common_transcript_split(
            experiment_name=DESIGN, dataset_mapping=mapping, panels=panels,
            sequences_path=base['paths']['sequences_path'], seed=args.seed,
            validation_fraction=.1, test_fraction=.1, reliability_bins=10,
            maximum_cds_codons=base['data'].get('max_cds_codons'), minimum_panel_support=2)
        write_json(inputs / 'split.json', split)
        folds = {}
        for panel, names in panels.items():
            folds[panel] = dict(source_panel=panel,
                                train_ids=split['panel_train_eligible_ids'][panel],
                                validation_ids=split['common_validation_ids'],
                                test_ids=split['common_test_ids'])
            print(f'{panel}: {len(names)} datasets; fitting training-only reliability references...', flush=True)
            reference = fit_panel_reliability_manifest(
                experiment_name=DESIGN, panel_name=panel, panel_datasets=names,
                dataset_mapping=mapping, panel_training_ids=folds[panel]['train_ids'],
                validation_ids=folds[panel]['validation_ids'], test_ids=folds[panel]['test_ids'],
                source_split_manifest=inputs / 'split.json')
            write_json(inputs / f'reliability_{panel}.json', reference)

        tasks, weight_rows = [], []
        R = float(ranks.max())
        sources = assignment.set_index('dataset_name').source_identifier
        for panel, names in panels.items():
            q = (R - ranks.loc[names].to_numpy(float) + 1) / R
            for arm, power, reverse in POLICIES:
                raw = q ** power
                if reverse:
                    raw = raw[::-1]
                pi = raw / raw.sum()
                task = dict(array_index=len(tasks), panel_id=panel, N=len(names),
                            arm=arm, power=power, reverse=reverse, training_seed=args.seed,
                            datasets=names, source_panel=panel, experiment_design=DESIGN,
                            run_id=f'four_{panel}_{arm}_seed{args.seed}',
                            directory=f'runs/seed{args.seed}/{arm}/{panel}')
                cfg = make_config(base, mapping, task, root, dict(zip(names, map(float, raw))))
                path = configs / f'{task["run_id"]}.yaml'
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                task.update(config_path=str(path), config_sha256=sha256(path),
                            task_contract_sha256=cfg['orchestrator']['task_contract_sha256'])
                (root / task['directory']).mkdir(parents=True, exist_ok=True)
                task['command'] = [sys.executable, '-u', str(ROOT / 'main_ribounmix_multidataset.py'),
                                   f'--config-path={configs}', f'--config-name={path.stem}',
                                   f'hydra.run.dir={root / task["directory"]}/hydra', 'hydra.job.chdir=false']
                tasks.append(task)
                weight_rows.extend(dict(panel_id=panel, N=len(names), arm=arm, dataset_id=d,
                                        global_rank=float(ranks[d]), original_q=float(original),
                                        assigned_q=float(value), pi=float(weight), source_family=sources[d])
                                   for d, original, value, weight in zip(names, q, raw, pi))
        weights = pd.DataFrame(weight_rows)
        weights.to_csv(root / 'reference_weights.csv', index=False)
        weights.groupby(['panel_id', 'arm', 'source_family'], as_index=False).pi.sum().to_csv(
            root / 'source_family_reference_mass.csv', index=False)
        pd.DataFrame(tasks).drop(columns=['command', 'datasets']).to_csv(root / 'task_matrix.csv', index=False)
        plan = dict(experiment_design=DESIGN, output_root=str(root), setup_request=request,
                    training_seeds=[args.seed], panels=panels, source_folds=folds,
                    tasks=tasks, tasks_sha256=object_sha256(tasks),
                    frozen_file_sha256={str(p): sha256(p) for p in [
                        *inputs.iterdir(), *configs.iterdir(), root / 'reference_weights.csv']})
        plan = add_shared_only(root, plan)
        print(f'Ready: {len(tasks)} tasks; {len(split["common_test_ids"])} common test transcripts. {root}', flush=True)
        return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/config_ribounmix_multidataset.yaml')
    parser.add_argument('--datasets', type=Path, default=ROOT / 'config/dataset_config/weighted_hek_riboseq_codon_replicas.yaml')
    parser.add_argument('--ranking', type=Path, default=ROOT / 'Datasets/data/HEK_riboseq_profile_quality_rank_components.tsv')
    parser.add_argument('--panels', type=Path, default=DEFAULT_PANELS)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=42)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare-only', action='store_true')
    action.add_argument('--task-index', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    for key in ('config', 'datasets', 'ranking', 'panels', 'output_root'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.task_index is not None and not 0 <= args.task_index < TASK_COUNT:
        parser.error('--task-index must be in 0..23')
    plan = prepare(args)
    return 0 if args.prepare_only else run_task(args, plan)


if __name__ == '__main__':
    raise SystemExit(main())
