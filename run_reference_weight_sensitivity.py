#!/usr/bin/env python3
"""Freeze/run a five-model, single-seed reference-weight sensitivity experiment.

Reuses all seven prefixes of the frozen cumulative directionality design. Membership, transcript
folds, local reliability weights, initialization and optimization are paired.
Only fixed-reference gamma weights vary: uniform, q/q^3, and their reversals.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml
from hydra import compose, initialize_config_dir

from prepare_rank_balanced_reference_directionality import (
    _assert_arm_match, initialization_audit, normalized,
    object_sha256, read_json, reverse_weight_assignment, sha256, write_json, parse_training_seeds,
)
from run_real_exp8_reference_directionality import environment, run_one, task_config, verify_hashes
from Utils.external_transcript_split import load_external_transcript_split
from Utils.panel_reference_preparation import relocated_input, snapshot_code
from Utils.reliability_references import load_reliability_reference_manifest, transcript_id_hash
from Utils.real_panel_convergence import infer_source_identifier
from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import load_dataset_quality_ranking
from run_real_exp8_L_stability_quality_rank import inspect_ranking_components

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / 'results/real_exp8_cumulative_qrank10_directionality'
DEFAULT_OUTPUT = ROOT / 'results/cumulative_reference_weight_stability_seed42'
DESIGN = 'full_collection_reference_weight_sensitivity_v2'
CUMULATIVE_DESIGN = 'cumulative_reference_weight_stability_v3'
SIZES = (2,5,10,20,40,80,114)
POLICIES = (('equal', 0, 'equal'), ('ranked_p1', 1, 'ranked'),
            ('reverse_p1', 1, 'reverse'), ('ranked_p3', 3, 'ranked'),
            ('reverse_p3', 3, 'reverse'))
CONTRASTS = (('ranked_p1', 'reverse_p1'), ('ranked_p3', 'reverse_p3'),
             ('equal', 'ranked_p1'), ('equal', 'reverse_p1'),
             ('equal', 'ranked_p3'), ('equal', 'reverse_p3'),
             ('ranked_p1', 'ranked_p3'), ('reverse_p1', 'reverse_p3'))


def reference_policy(names, ranks, quality, power, direction):
    """Apply the exponent before reversal; use one explicit resolver for all arms."""
    if direction not in {'equal', 'ranked', 'reverse'}:
        raise ValueError(f'Unknown direction: {direction}')
    if power not in {0, 1, 3} or (direction == 'equal') != (power == 0):
        raise ValueError('Use equal/p=0 or ranked/reverse with p=1 or p=3.')
    powered = {name: float(quality[name]) ** power for name in names}
    reverse, donors = reverse_weight_assignment(names, ranks, powered)
    raw = reverse if direction == 'reverse' else powered
    pi = normalized(raw[name] for name in names)
    rows = [dict(dataset_id=name, global_rank=float(ranks[name]),
                 original_q=float(quality[name]), power=power, direction=direction,
                 assigned_q=raw[name], pi=float(weight),
                 assigned_from_dataset=(donors[name] if direction == 'reverse' else name)
                 if direction != 'equal' else None)
            for name, weight in zip(names, pi)]
    # Explicit weights already contain the exponent; the model must not reapply it.
    return dict(weighting='explicit', quality_rank_power=1.0, explicit_weights=raw), rows


def task_matrix(names, seed, source_panel='exp8_qrank_N114_seed42'):
    return [dict(array_index=i, N=len(names), arm=arm, power=power, direction=direction,
                 training_seed=seed, source_panel=source_panel, datasets=list(names),
                 run_id=f'ref_sensitivity_N{len(names):03d}_{arm}_seed{seed}',
                 directory=f'runs/seed{seed}/{arm}/N{len(names):03d}')
            for i, (arm, power, direction) in enumerate(POLICIES)]


def load_cumulative_source(source, n=114):
    """Reuse the exact full prefix and its frozen folds; no dataset selection/refit."""
    manifest_path = source / 'experiment_manifest.json'
    manifest = read_json(manifest_path)
    if manifest.get('experiment_design') != 'matched_cumulative_reference_directionality':
        raise ValueError('Expected the frozen cumulative directionality experiment.')
    if object_sha256(manifest['tasks']) != manifest['tasks_sha256']:
        raise ValueError('Source task mapping changed.')
    selected = [t for t in manifest['tasks'] if t['N'] == n and t['arm'] == 'equal' and t['training_seed'] == 42]
    if len(selected) != 1:
        raise ValueError(f'Expected exactly one source N={n}/equal/seed42 template.')
    task = selected[0]
    recorded = Path(manifest['output_root'])

    def frozen_path(raw):
        path = Path(raw)
        local = source / path.relative_to(recorded) if path.is_absolute() else source / path
        expected = manifest['frozen_file_sha256'][str(path)]
        if sha256(local) != expected:
            raise ValueError(f'Source frozen artifact changed: {local}')
        return local

    template_path = frozen_path(task['config_path'])
    if sha256(template_path) != task['config_sha256']:
        raise ValueError('Source task/configuration checksum mismatch.')
    template = yaml.safe_load(template_path.read_text())
    ranking_path = frozen_path(template['data']['dataset_quality_ranking']['path'])
    inspect_ranking_components(ranking_path, expected_count=10)
    ranks, quality = load_dataset_quality_ranking(str(ranking_path))
    if len(ranks) != 115 or max(ranks.values()) != 115 or sha256(ranking_path) != manifest['ranking']['sha256']:
        raise ValueError('Expected the unchanged 115-dataset global ranking.')
    names = task['datasets']
    full = next(t['datasets'] for t in manifest['tasks'] if t['N']==114)
    if (len(full)!=114 or len(set(full))!=114 or set(full)-set(ranks)
            or full!=sorted(full,key=lambda name:(ranks[name],name))
            or len(names) != n or len(set(names)) != n or set(names) - set(ranks)
            or names != sorted(names, key=lambda name:(ranks[name],name))
            or template['experiment']['dataset'] != names):
        raise ValueError(f'Source N={n} membership/order differs from the frozen prefix.')
    if any(t['datasets'] != full[:t['N']] for t in manifest['tasks']):
        raise ValueError('Source tasks do not share exact cumulative prefixes.')
    split_path = frozen_path(template['split']['external_manifest'])
    source_panel = task['source_panel']
    if template['split']['external_panel_name'] != source_panel:
        raise ValueError('Source split panel differs from the task.')
    train, validation, test, _ = load_external_transcript_split(
        split_path, panel_name=source_panel, experiment_datasets=names)
    folds = manifest['source_folds'][str(n)]
    if folds != dict(source_panel=source_panel, train_ids=train, validation_ids=validation, test_ids=test):
        raise ValueError('Source fold IDs differ between the manifest and frozen split.')
    reliability_path = frozen_path(template['data']['reliability_reference_manifest'])
    ref = load_reliability_reference_manifest(reliability_path)
    if (ref['reference_split'] != 'training_only' or ref['heldout_rows_used_for_fitting'] != 0
            or ref['panel_training_transcript_id_hash'] != transcript_id_hash(train)
            or ref['panel_training_transcript_count'] != len(train) or set(ref['datasets']) != set(names)):
        raise ValueError(f'Frozen reliability references differ from N={n} training membership.')
    return dict(template=template, template_path=template_path, names=names, folds=folds,
                split_path=split_path, reliability_path=reliability_path, ranking_path=ranking_path,
                ranks=ranks, quality=quality, rank_universe=115., ranking_sha256=sha256(ranking_path),
                manifest_path=manifest_path)


def write_design_report(root, manifest, weights, concentration, distances):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    for ax, power in zip(axes, (1, 3)):
        for arm, color in [('equal', '#64748b'), (f'ranked_p{power}', '#267eab'),
                           (f'reverse_p{power}', '#9a519b')]:
            part = weights[weights.arm == arm].sort_values(['global_rank', 'dataset_id'])
            ax.plot(part.global_rank, part.pi * 100, '.-', label=arm, color=color)
        ax.set(title=f'Weighting strength p = {power}', xlabel='Frozen global rank (1 = best)',
               ylim=(0, weights.pi.max() * 100 * 1.08))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Gamma-reference weight (%)')
    fig.tight_layout()
    fig.savefig(root / 'planned_reference_weights.svg')
    plt.close(fig)
    selected = weights[weights.arm == 'equal'][['dataset_id', 'global_rank', 'source_family']]
    folds = manifest['source_folds'][str(len(selected))]
    content = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Single-seed reference-weight sensitivity</title>
<style>body{{max-width:1080px;margin:35px auto;padding:0 22px;font:16px/1.6 system-ui;color:#203448}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px 12px;border-bottom:1px solid #dce5ec;text-align:right}}img{{width:100%}}.note{{padding:15px;border-left:4px solid #267eab;background:#f3f7fb}}.table{{overflow:auto}}code,pre{{background:#f3f3f3;padding:4px}}pre{{overflow:auto}}</style></head><body>
<h1>Does reference-weight direction matter when the contrast is stronger?</h1>
<p class="note">Five fresh trainings · full cumulative collection · seed {manifest['training_seeds'][0]} · design prepared, no training results implied.</p>
<h2>What changes?</h2>
<p>All runs train on the same {len(selected)} datasets from the frozen N = 114 cumulative endpoint, spanning global ranks {int(selected.global_rank.min())}–{int(selected.global_rank.max())}. The exact source dataset order, transcript folds and training-only reliability references are reused. This follow-up holds N = 114 fixed while changing reference direction and strength; it does not rerun the smaller prefixes. The active collection excludes zhu_2025 at global rank 110, and R remains 115.</p>
<ol><li><b>equal:</b> uniform reference weights.</li><li><b>ranked_p1 / reverse_p1:</b> current global score q = (116 − rank)/115, assigned forward or in reverse rank order.</li><li><b>ranked_p3 / reverse_p3:</b> cube that same q, then assign the resulting weight multiset forward or in reverse rank order.</li></ol>
<p>In every case normalize over the same full collection: π = assigned weight / sum of assigned weights. Cubing amplifies the ratio of large to small scores; reverse preserves the complete weight multiset, so each ranked/reverse pair has identical dataset-weight concentration. All five configurations use the same explicit-weight implementation.</p>
<img src="planned_reference_weights.svg" alt="Planned equal, ranked and reverse weights at exponents one and three">
<p class="note">The broad rank range creates a substantial direction contrast already at p = 1, and p = 3 provides a prespecified stronger perturbation without adding datasets.</p>
<h2>Quantitative contrast, before training</h2><div class="table">{concentration.to_html(index=False, float_format=lambda x:f'{x:.5f}', border=0)}</div>
<p>N_ref = 1/Σπ² summarizes weight concentration, not independent sample size; family_N_ref uses the aggregate weight of each source family.</p>
<div class="table">{distances.to_html(index=False, float_format=lambda x:f'{x:.5f}', border=0)}</div><p>TV = ½Σ|π_A − π_B| is the fraction of total reference weight reassigned.</p>
<h2>What stays fixed?</h2><p>Dataset identities and ordering, {len(folds['train_ids'])} training / {len(folds['validation_ids'])} validation / {len(folds['test_ids'])} test transcript IDs, training-only w_dt references, initial trainable parameters, batch construction, architecture, objective, optimizer and stopping/checkpoint rules. Every run starts fresh and exports the best-validation-loss checkpoint. The production initialization audit verifies identical parameters and the planned reference buffers.</p>
<h2>How to read the result</h2><p>The primary comparison is ranked_p3 versus reverse_p3, with p = 1 providing the current-strength comparison. The analysis reports per-transcript PCC, RMSE and profile variance on the same complete test cohort, plus the change in sensitivity from p = 1 to p = 3. A larger discrepancy at p = 3 supports sensitivity to reference direction in this collection and seed; similar profiles at both strengths indicate robustness over these tested perturbations. Neither outcome establishes that the higher-quality reference is biologically more accurate.</p>
<p>The L_bio profiles are normalized to mean one, so RMSE and variance measure differences in normalized profile shape and amplitude, not absolute ribosome abundance. Sequence-only test exports contain no valid observation-reconstruction target. Inspect convergence before interpreting differences; best-checkpoint losses may be affected by the reference parameterization and are not an independent biological validation.</p>
<p><b>Single-seed scope:</b> seed {manifest['training_seeds'][0]} is shared across all five runs; this controls initialization but does not measure variability across seeds or guarantee bitwise deterministic GPU training. No across-seed confidence interval or superiority claim is made. A null result does not prove general invariance, and hyperparameters must not be changed after examining this experiment's test results.</p>
<p>The source cumulative design did not impose source-family atomicity on prefix boundaries; family identities here are inferred from dataset names for diagnostics, and reversing dataset weights can change aggregate family concentration; inspect <a href="source_family_reference_mass.csv">family masses</a>. The QC ranking's transcript scope is unverified. Current data bytes and code are frozen for the five new runs; historical byte-level equality is not assumed.</p>
<h2>Run and analyze</h2><pre>bash submit_reference_weight_sensitivity_univie.sh
python analyses/analyze_reference_weight_sensitivity.py --experiment-root {html.escape(str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root))}</pre>
<p>Prepare on the training machine; the generated configurations and hashes refer to that machine's paths and environment. Preparation is CPU-only. See <a href="task_matrix.csv">task mapping</a>, <a href="comparison_contract.json">comparison contract</a>, and <a href="analysis/analysis_report.html">analysis report (after analysis)</a>.</p>
<details><summary>All selected datasets</summary><div class="table">{selected.to_html(index=False,border=0)}</div></details></body></html>'''
    (root / 'design_report.html').write_text(content)


def freeze_plan(args):
    root = args.output_root
    source = args.prepared_root
    sizes = tuple(args.sizes)
    design = CUMULATIVE_DESIGN if len(sizes)>1 else DESIGN
    if root == source or root.is_relative_to(source):
        raise ValueError('Use an output root outside the source preparation.')
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f'.{root.name}.freeze.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        saved = root / 'experiment_manifest.json'
        if saved.exists():
            manifest = read_json(saved)
            if (manifest['source_root'] != str(source) or manifest['training_seeds'] != [args.seed]
                    or manifest['experiment_design'] != design or manifest['sizes'] != list(sizes)):
                raise ValueError('Existing frozen design differs; choose a new output root.')
            if object_sha256(manifest['tasks']) != manifest['tasks_sha256']:
                raise ValueError('Frozen task mapping changed.')
            verify_hashes(manifest['frozen_file_sha256'])
            return manifest
        if list(root.glob('runs/**/execution_status.json')):
            raise ValueError('Existing training without a completed design manifest.')
        bases = {n:load_cumulative_source(source,n) for n in sizes}
        base = bases[max(sizes)]
        test_ids = base['folds']['test_ids']
        if any(b['folds']['test_ids'] != test_ids for b in bases.values()):
            raise ValueError('Cumulative comparisons require identical test IDs across N.')
        frozen = root / 'frozen_inputs'
        configs = root / 'resolved_configs'
        frozen.mkdir(parents=True, exist_ok=True)
        configs.mkdir(exist_ok=True)
        copies = [(base['ranking_path'], frozen / 'HEK_riboseq_profile_quality_rank_components.tsv'),
                  (base['split_path'], frozen / 'experiment_split_manifest.json'),
                  (base['manifest_path'], frozen / 'source_experiment_manifest.json')]
        for n,b in bases.items():
            copies.extend([(b['reliability_path'], frozen / f'N{n:03d}_reliability_reference_manifest.json'),
                           (b['template_path'], frozen / f'source_N{n:03d}_template.yaml')])
        for src, dst in copies:
            shutil.copy2(src, dst)
        code_hashes = snapshot_code(root / 'code_snapshot')
        assets, tasks = {}, []
        for n,b in bases.items():
            template,names = b['template'],b['names']
            template['paths']['sequences_path'] = str(relocated_input(template['paths']['sequences_path']))
            for key, raw in template['paths']['encodings'].items():
                local = relocated_input(raw)
                template['paths']['encodings'][key] = str(root / 'code_snapshot' / local.relative_to(ROOT))
            paths = template['dataset_config']['dataset_path']
            template['dataset_config']['dataset_path'] = {name: str(relocated_input(paths[name])) for name in names}
            for raw in [template['paths']['sequences_path'], *template['paths']['encodings'].values(),
                        *template['dataset_config']['dataset_path'].values()]:
                if raw not in assets:
                    assets[raw] = sha256(Path(raw))
            tasks.extend(task_matrix(names,args.seed,b['folds']['source_panel']))
        for index,task in enumerate(tasks):
            task['array_index']=index
        family = {name:infer_source_identifier(name) for name in base['names']}
        weights, audits, differences, paired = [], [], [], {}
        for task in tasks:
            n=task['N']
            selected=bases[n]
            template,names=selected['template'],selected['names']
            source_panel=selected['folds']['source_panel']
            implementation, rows = reference_policy(names, base['ranks'], base['quality'], task['power'], task['direction'])
            weights.extend(dict(N=len(names), arm=task['arm'], source_family=family[row['dataset_id']], **row) for row in rows)
            cfg = task_config(template, task, root, source_panel, implementation)
            cfg['orchestrator'].update(experiment=design, power=task['power'])
            audit = initialization_audit(cfg)
            expected_buffer = hashlib.sha256(np.asarray([row['assigned_q'] for row in rows], dtype=np.float32).tobytes()).hexdigest()
            if audit['reference_buffer_sha256'] != expected_buffer:
                raise ValueError(f'{task["arm"]}: production reference buffer differs from planned weights.')
            cfg['orchestrator']['initialization_sha256'] = audit['parameter_sha256']
            cfg['orchestrator']['task_contract_sha256'] = object_sha256(cfg)
            path = configs / f'{task["run_id"]}.yaml'
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            with initialize_config_dir(version_base=None, config_dir=str(configs)):
                compose(config_name=path.stem)
            task_root = root / task['directory']
            task_root.mkdir(parents=True, exist_ok=True)
            task.update(config_path=str(path), config_sha256=sha256(path),
                        initialization_sha256=audit['parameter_sha256'],
                        task_contract_sha256=cfg['orchestrator']['task_contract_sha256'],
                        command=[sys.executable, '-u', str(root / 'code_snapshot/main_ribounmix_multidataset.py'),
                                 f'--config-path={configs}', f'--config-name={path.stem}',
                                 f'hydra.run.dir={task_root / "hydra"}', 'hydra.job.chdir=false'])
            audits.append(dict(N=n,arm=task['arm'], training_seed=args.seed, **audit))
            paired[n,task['arm']] = cfg
            print(f'Frozen {task["run_id"]}', flush=True)
        for n in sizes:
            if len({audit['parameter_sha256'] for audit in audits if audit['N']==n}) != 1:
                raise ValueError(f'N={n}: initial trainable parameters differ across policies.')
            for arm,_,_ in POLICIES:
                differences.extend(dict(N=n,arm=arm, **row) for row in _assert_arm_match(paired[n,'equal'], paired[n,arm]))
        weights = pd.DataFrame(weights)
        weights.to_csv(root / 'reference_weights.csv', index=False)
        family_mass = weights.groupby(['N', 'arm', 'source_family'], as_index=False).pi.sum()
        family_mass.to_csv(root / 'source_family_reference_mass.csv', index=False)
        summaries = []
        for (n,arm), group in weights.groupby(['N','arm'], sort=False):
            pi = group.pi.to_numpy()
            mass = family_mass[(family_mass.N==n)&(family_mass.arm == arm)].pi.to_numpy()
            summaries.append(dict(N=n,arm=arm, N_ref=1/(pi@pi), family_N_ref=1/(mass@mass),
                                  max_pi=pi.max(), min_pi=pi.min(), weighted_mean_rank=pi@group.global_rank.to_numpy()))
        concentration = pd.DataFrame(summaries)
        concentration.to_csv(root / 'reference_concentration.csv', index=False)
        distance_rows=[]
        for n,group in weights.groupby('N'):
            pivot = group.pivot(index='dataset_id', columns='arm', values='pi')
            distance_rows.extend(dict(N=n,arm_a=a,arm_b=b,total_variation=.5*abs(pivot[a]-pivot[b]).sum()) for a,b in CONTRASTS)
            for power in (1,3):
                np.testing.assert_allclose(sorted(pivot[f'ranked_p{power}']), sorted(pivot[f'reverse_p{power}']), rtol=0, atol=1e-15)
        distances = pd.DataFrame(distance_rows)
        distances.to_csv(root / 'policy_distances.csv', index=False)
        pd.DataFrame(audits).to_csv(root / 'initialization_audit.csv', index=False)
        pd.DataFrame(differences).to_csv(root / 'paired_configuration_differences.csv', index=False)
        pd.DataFrame(tasks).drop(columns=['command','datasets']).to_csv(root / 'task_matrix.csv', index=False)
        contract = dict(primary_contrast=['ranked_p3','reverse_p3'], current_strength_contrast=['ranked_p1','reverse_p1'],
                        contrasts=CONTRASTS, metrics=['PCC', 'RMSE', 'variance_a', 'variance_b'],
                        cohort='same frozen complete sequence-only test transcript set',
                        scope='The frozen N=114 cumulative endpoint and one training seed; descriptive sensitivity, no across-seed inference or biological superiority claim.',
                        exponent_rule='p=1 and p=3 fixed before this experiment; no held-out tuning',
                        source_family_limitation='Dataset-level reversal preserves dataset concentration but can change aggregate source-family concentration.')
        if len(sizes)>1:
            contract.update(primary='Paired difference in adjacent-size stability: ranked minus equal PCC; equal minus ranked RMSE',
                            primary_contrast=['equal','ranked_p1'],secondary_contrasts=['ranked_p3 versus equal','ranked versus reverse at matched power','N=2 anchor drift','profile variance / collapse'],
                            size_pairs=list(zip(sizes[:-1],sizes[1:])),scope='Cumulative stability conditional on one seed; no required monotonic decline of equal weighting, no biological superiority claim.',
                            split_limitation='Within N folds match across policies; across N train/validation membership and w_dt fitting references can vary, while test IDs are fixed.')
        write_json(root / 'comparison_contract.json', contract)
        manifest = dict(experiment_design=design, output_root=str(root),
                        source_root=str(source), training_seeds=[args.seed], sizes=list(sizes),
                        arms=[a for a,_,_ in POLICIES], number_of_tasks=len(tasks), tasks=tasks, tasks_sha256=object_sha256(tasks),
                        source_folds={str(n):b['folds'] for n,b in bases.items()},
                        ranking=dict(R=base['rank_universe'], sha256=base['ranking_sha256']),
                        source_file_sha256={str(src):sha256(src) for src,_ in copies},
                        software=environment(), code_sha256=code_hashes, asset_sha256=assets)
        if len(sizes)>1:
            from analyses.analyze_cumulative_reference_weight_stability import write_design_report as write_cumulative_design
            write_cumulative_design(root,manifest,weights,concentration)
        else:
            write_design_report(root, manifest, weights, concentration, distances)
        manifest['frozen_file_sha256'] = {str(f):sha256(f) for f in [*frozen.iterdir(), *configs.iterdir(), *root.glob('*.csv'), root/'comparison_contract.json', root/'design_report.html', *root.glob('*.svg')] if f.is_file()}
        write_json(saved, manifest)  # Last: no task can run with an incomplete plan.
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', '--prepared-root', dest='prepared_root', type=Path, default=DEFAULT_SOURCE,
                        help='Frozen cumulative directionality experiment; --prepared-root remains an alias.')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sizes',type=parse_training_seeds,default=SIZES,help='Frozen cumulative sizes; default 2,5,10,20,40,80,114.')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare-only', action='store_true')
    action.add_argument('--list-tasks', action='store_true')
    action.add_argument('--task-index', type=int)
    parser.add_argument('--authorize-training', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Inspect one frozen task without training.')
    parser.add_argument('--resume', action='store_true', help='Resume a failed task with full optimizer/scheduler state.')
    args = parser.parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    args.prepared_root = args.prepared_root.expanduser().resolve()
    if args.seed < 0 or args.seed > 2**32-1:
        parser.error('--seed must be in 0..2**32-1')
    if tuple(sorted(args.sizes))!=tuple(args.sizes) or set(args.sizes)-set(SIZES):
        parser.error('--sizes must be increasing source sizes from 2,5,10,20,40,80,114')
    if args.task_index is not None and not 0 <= args.task_index < len(POLICIES)*len(args.sizes):
        parser.error(f'--task-index must be in 0..{len(POLICIES)*len(args.sizes)-1}')
    try:
        manifest = freeze_plan(args)
        if args.prepare_only or args.list_tasks:
            print(pd.read_csv(args.output_root / 'task_matrix.csv')[['array_index','arm','power','training_seed','N']].to_string(index=False))
            print(f'Prepared design: {args.output_root / "design_report.html"}; no training launched.')
            return 0
        if args.dry_run:
            print(manifest['tasks'][args.task_index]['command'])
            return 0
        return run_one(args, manifest)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError, AssertionError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
