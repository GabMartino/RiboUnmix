#!/usr/bin/env python3
"""Analyze available equal/ranked/reverse models; never train or alter inputs.

Partial downloads are normal: missing arms are listed, not imputed. Cross-panel
reproducibility and same-panel policy sensitivity are different estimands.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import itertools
import json
from pathlib import Path
import re
import shlex
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import load_dataset_quality_ranking
from prepare_rank_balanced_reference_directionality import _assert_arm_match, _reference_policy
from analyses.analyze_real_panel_convergence import _locate_panel_prediction
from analyses.compare_real_panel_weighting import metrics, paired_bootstrap
from run_real_independent_panel_convergence_quality_rank import _flatten_config
from Utils.panel_reference_audit import global_quality_groups
from Utils.publication_plot_style import publication_rc
from Utils.reliability_references import transcript_id_hash
from Utils.tensorboard_scalars import find_event_runs, load_scalars

ARMS = ('equal', 'ranked', 'reverse')
COLORS = dict(equal='#0072B2', ranked='#D55E00', reverse='#8A5A9E')
LABELS = dict(equal='Equal', ranked='Ranked', reverse='Reverse')
CONTRASTS = (('equal', 'ranked'), ('reverse', 'ranked'), ('equal', 'reverse'))
DOMAINS = {'full_cds': slice(None), 'interior_20': slice(20, -20)}
DEFAULT_ROOT = ROOT/'results/four_panel_rank_balanced_qrank10_directionality_seed42'
TAGS = ('train_loss', 'val_loss', 'val_replica_nll', 'val_mu_pcc',
        'val_mean_ratio', 'val_nb_alpha_mean', 'val_L_bio_max')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def locate_experiment(root):
    root = Path(root).expanduser().resolve()
    if (root/'experiment_manifest.json').is_file():
        return root
    candidate = root/'equal_ranked_reverse_qrank10'
    if (candidate/'experiment_manifest.json').is_file():
        return candidate
    raise FileNotFoundError(f'No directionality experiment at {root} or {candidate}.')


def relocate(path, root, recorded_root):
    """Relocate only the exact experiment-root prefix, never fuzzy basenames."""
    path = Path(path)
    if path.is_absolute():
        return root/path.relative_to(recorded_root)
    return root/path


def audit_inputs(root):
    manifest = read_json(root/'experiment_manifest.json')
    tasks = read_json(root/'tasks.json')
    recorded_root = Path(manifest['output_root'])
    hashes, limitations = [], []
    for original, expected in manifest['frozen_file_sha256'].items():
        try:
            local = relocate(original, root, recorded_root)
        except ValueError:
            limitations.append(f'External training snapshot not reverified locally: {original} (recorded SHA256 {expected}).')
            continue
        observed = sha256(local)
        if observed != expected:
            raise ValueError(f'Frozen input/configuration checksum mismatch: {local}')
        hashes.append(dict(file=str(local), sha256=observed, recorded_path=original))
    rank_path = root/'frozen_inputs/HEK_riboseq_profile_quality_rank_components.tsv'
    ranks, q = load_dataset_quality_ranking(str(rank_path))
    rank_frame = pd.read_csv(rank_path, sep='\t')
    components = [c for c in rank_frame if c.startswith('rank_') and c != 'rank_component_count']
    if (len(components) != 10 or set(rank_frame.rank_component_count) != {10}
            or sha256(rank_path) != manifest['ranking']['sha256']):
        raise ValueError('The saved input is not the declared ten-component ranking.')
    R = float(rank_frame.quality_rank.max())
    if R != manifest['ranking']['rank_universe']:
        raise ValueError('Complete-table rank universe differs from the experiment manifest.')
    split = read_json(root/'frozen_inputs/common_split_manifest.json')
    assignment = pd.read_csv(root/'frozen_inputs/panel_assignment.csv')
    panels = sorted(split['panels'])
    if assignment.dataset_id.duplicated().any() or (assignment.groupby('source_family').panel_id.nunique() > 1).any():
        raise ValueError('Datasets are duplicated or a source family crosses panels.')
    if set(assignment.dataset_id) != set(itertools.chain.from_iterable(split['panels'].values())):
        raise ValueError('Split and assignment dataset universes differ.')
    if assignment.groupby('panel_id').size().tolist() != manifest['partition']['capacities']:
        raise ValueError('Panel capacities differ from the frozen experiment.')
    test_ids = sorted(split['common_test_ids'])
    heldout = set(test_ids) | set(split['common_validation_ids'])
    if len(test_ids) != len(set(test_ids)) or set(test_ids) & set(split['common_validation_ids']):
        raise ValueError('Duplicate or overlapping held-out identities.')
    for panel in panels:
        if set(split['panel_train_eligible_ids'][panel]) & heldout:
            raise ValueError(f'{panel}: held-out leakage.')
    initialization = pd.read_csv(root/'initialization_and_gradient_audit.csv').set_index('task_id')
    weights = pd.read_csv(root/'reference_weights.csv')
    if weights.duplicated(['panel_id', 'arm', 'dataset_id']).any():
        raise ValueError('Duplicate policy-level reference weights.')
    configs = {}
    for task in tasks:
        key = (int(task['training_seed']), task['arm'], task['panel_id'])
        if key in configs:
            raise ValueError(f'Duplicate scientific task: {key}')
        cfg_path = relocate(task['config_path'], root, recorded_root)
        if sha256(cfg_path) != task['config_sha256']:
            raise ValueError(f'{task["task_id"]}: configuration checksum mismatch.')
        cfg = yaml.safe_load(cfg_path.read_text())
        configs[key] = cfg
        names = split['panels'][key[2]]
        if cfg['experiment']['dataset'] != names or int(cfg['experiment']['seed']) != key[0]:
            raise ValueError(f'{key}: dataset order or seed differs.')
        ref_path = root/f'frozen_inputs/{key[2]}_reliability_reference_manifest.json'
        ref = read_json(ref_path)
        if (ref.get('reference_split') != 'training_only' or ref.get('heldout_rows_used_for_fitting') != 0
                or ref.get('panel_training_transcript_id_hash') != transcript_id_hash(split['panel_train_eligible_ids'][key[2]])):
            raise ValueError(f'{key}: training-only reliability identity not verified.')
        if relocate(cfg['data']['reliability_reference_manifest'], root, recorded_root) != ref_path:
            raise ValueError(f'{key}: does not use its frozen reliability reference.')
        if relocate(cfg['split']['external_manifest'], root, recorded_root) != root/'frozen_inputs/common_split_manifest.json':
            raise ValueError(f'{key}: does not use the frozen split.')
        policy, expected_rows = _reference_policy(key[1], names, ranks, q)
        reference = cfg['model']['gamma_centering']['reference']
        if any(reference.get(k) != v for k, v in policy.items()):
            raise ValueError(f'{key}: wrong reference policy.')
        selected = weights[(weights.panel_id == key[2]) & (weights.arm == key[1])].set_index('dataset_id').loc[names]
        np.testing.assert_allclose(selected.pi, [row['pi'] for row in expected_rows], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(selected.assigned_q, [row['assigned_q'] for row in expected_rows], rtol=1e-12, atol=1e-14)
    for (seed, arm, panel), cfg in configs.items():
        _assert_arm_match(configs[seed, 'equal', panel], cfg)
        peer_ids = [t['task_id'] for t in tasks if t['training_seed'] == seed and t['panel_id'] == panel]
        if initialization.loc[peer_ids].parameter_sha256.nunique() != 1:
            raise ValueError(f'{seed}/{panel}: unmatched initial trainable parameters.')
    unique = assignment[['dataset_id']].copy()
    unique['global_rank'] = unique.dataset_id.map(ranks)
    labels, grouping = global_quality_groups(unique.global_rank)
    group_by_dataset = dict(zip(unique.dataset_id, labels))
    weights['quality_group'] = weights.dataset_id.map(group_by_dataset)
    limitations.append('Global QC ranking transcript scope (training-only versus full dataset) is not verified; this differs from the verified train-only w_dt references.')
    return manifest, tasks, configs, split, test_ids, weights, grouping, hashes, limitations


def aligned_metrics(left, right, domain):
    if left['coordinate_hash'] != right['coordinate_hash']:
        raise ValueError('Different codon sequences or valid-position coordinates; no truncation allowed.')
    x, y = left['values'][DOMAINS[domain]], right['values'][DOMAINS[domain]]
    return dict(n_positions=len(x), **metrics(x, y))


def read_profiles(task_dir, task, cfg, ids, expected_weights, root, manifest):
    compact, source, detail = _locate_panel_prediction(task_dir)
    if compact is None:
        return None, dict(detail=detail)
    runtime = read_json(source)['best_val_loss']
    if runtime['transcript_id_hash'] != transcript_id_hash(ids) or runtime['transcript_count'] != len(ids):
        raise ValueError('Runtime held-out transcript identity differs.')
    run_manifest = read_json(task_dir/'run_manifest.json')
    for field, expected in [('task_id', task['task_id']), ('config_sha256', task['config_sha256']),
                            ('split_sha256', sha256(root/'frozen_inputs/common_split_manifest.json')),
                            ('reliability_sha256', sha256(root/f'frozen_inputs/{task["panel_id"]}_reliability_reference_manifest.json'))]:
        if run_manifest.get(field) != expected:
            raise ValueError(f'Runtime {field} differs from prepared task.')
    runtime_cfg_path = task_dir/'hydra/.hydra/config.yaml'
    runtime_cfg = yaml.safe_load(runtime_cfg_path.read_text())
    if _flatten_config(cfg) != _flatten_config(runtime_cfg):
        raise ValueError('Actual Hydra training configuration differs from the frozen configuration.')
    environment = read_json(task_dir/'execution_environment.json')
    if environment['config_sha256'] != task['config_sha256'] or environment['experiment_manifest_sha256'] != sha256(root/'experiment_manifest.json'):
        raise ValueError('Execution identity differs from the frozen task/experiment.')
    gamma = read_json(source.parent/'gamma_reference_manifest.json')
    names = cfg['experiment']['dataset']
    expected = expected_weights.set_index('dataset_id').loc[names].pi.to_numpy(float)
    if gamma['reference_dataset_names'] != names or gamma['centering_mode'] != 'fixed_reference':
        raise ValueError('Actual fixed reference does not use the complete selected panel in order.')
    if gamma['weighting'] != cfg['model']['gamma_centering']['reference']['weighting']:
        raise ValueError('Actual gamma-reference policy differs from the prepared arm.')
    np.testing.assert_allclose(gamma['reference_pi'], expected, rtol=1e-6, atol=1e-9)
    frame = pd.read_parquet(compact)
    if not {'L_t', 'transcript_length', 'valid_position_mask'} <= set(frame):
        raise ValueError('Missing compact L or positional mask; retain the production compact export.')
    if frame.transcript_id.duplicated().any() or set(frame.transcript_id) != set(ids):
        raise ValueError('Compact export does not contain exactly the common test IDs.')
    profiles = {}
    for row in frame.itertuples(index=False):
        x = np.asarray(row.L_t, dtype=np.float64)
        mask = np.asarray(row.valid_position_mask, dtype=bool)
        if x.ndim != 1 or x.size != row.transcript_length or mask.shape != x.shape or not mask.all():
            raise ValueError(f'{row.transcript_id}: compact export is not a complete valid CDS.')
        if not np.isfinite(x).all() or (x <= 0).any() or abs(x.mean()-1) > 1e-4:
            raise ValueError(f'{row.transcript_id}: nonfinite, nonpositive or non-mean-one L. No repairs applied.')
        if hasattr(row, 'run_id') and row.run_id != task['task_id']:
            raise ValueError('Compact profile run identity differs.')
        profiles[str(row.transcript_id)] = dict(values=x, length=len(x))
    raw = source.parent/Path(runtime['output_path']).name
    # Read coordinates and shared L only, not dummy mu/targets from sequence-only export.
    seen = set()
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=16, columns=['transcript_id', 'length', 'mask', 'codon_ids', 'L_bio']):
        for row in batch.to_pylist():
            tid = str(row['transcript_id'])
            if tid not in profiles:
                raise ValueError(f'Unexpected raw prediction ID: {tid}')
            mask = np.asarray(row['mask'], bool)
            x = np.asarray(row['L_bio'], float)
            codons = np.asarray(row['codon_ids'], dtype='<i8')
            if mask.shape != x.shape or mask.shape != codons.shape or int(mask.sum()) != row['length']:
                raise ValueError(f'{tid}: inconsistent raw coordinates.')
            np.testing.assert_allclose(x[mask], profiles[tid]['values'], rtol=2e-5, atol=2e-6)
            digest = hashlib.sha256(codons[mask].tobytes()+np.flatnonzero(mask).astype('<i8').tobytes()).hexdigest()
            if 'coordinate_hash' in profiles[tid] and profiles[tid]['coordinate_hash'] != digest:
                raise ValueError(f'{tid}: coordinate disagreement across dataset rows.')
            profiles[tid]['coordinate_hash'] = digest
            seen.add(tid)
    if seen != set(ids):
        raise ValueError('Raw and compact transcript cohorts differ.')
    epoch_match = re.search(r'epoch=(\d+)', Path(runtime['checkpoint_path']).name)
    info = dict(detail=detail, prediction_path=str(compact), prediction_sha256=sha256(compact),
                raw_alignment_source=str(raw), raw_sha256=sha256(raw),
                runtime_manifest=str(source), runtime_manifest_sha256=sha256(source),
                runtime_config_sha256=sha256(runtime_cfg_path), checkpoint=runtime['checkpoint_path'],
                selected_epoch=int(epoch_match.group(1)) if epoch_match else None,
                sequence_only=bool(runtime.get('sequence_only_shared_profile_prediction')),
                observation_dependent_dummy_outputs_are_scientific=runtime.get('observation_dependent_dummy_outputs_are_scientific'),
                hostname=environment.get('hostname'), gpu=environment.get('visible_gpu'),
                n_test_transcripts=len(ids), max_mean_one_error=max(abs(v['values'].mean()-1) for v in profiles.values()))
    return profiles, info


def collect(root, manifest, tasks, configs, ids, weights):
    profiles, availability = {}, []
    for task in tasks:
        key = (int(task['training_seed']), task['arm'], task['panel_id'])
        task_dir = root/'runs'/f'seed{key[0]}'/key[1]/key[2]
        row = {k:task[k] for k in ('task_id', 'array_index', 'training_seed', 'arm', 'panel_id')}
        row.update(task_directory=str(task_dir), status='no_local_outputs')
        if list(task_dir.rglob('*.ckpt')):
            row['status'] = 'checkpoint_without_validated_export'
        elif list(task_dir.rglob('events.out.tfevents*')):
            row['status'] = 'logs_only'
        try:
            selected = weights[(weights.panel_id == key[2]) & (weights.arm == key[1])]
            values, info = read_profiles(task_dir, task, configs[key], ids, selected, root, manifest)
            row.update(info)
            if values is not None:
                profiles[key] = values
                row['status'] = 'validated_predictions'
        except (ValueError, KeyError, AssertionError, FileNotFoundError, RuntimeError, OSError) as exc:
            row.update(status='invalid_artifacts', detail=f'{type(exc).__name__}: {exc}')
        availability.append(row)
    return profiles, pd.DataFrame(availability)


def evaluation_tables(profiles, ids):
    cross, sensitivity, diagnostics = [], [], []
    for (seed, arm, panel), values in sorted(profiles.items()):
        for tid in ids:
            for domain, section in DOMAINS.items():
                x = values[tid]['values'][section]
                diagnostics.append(dict(training_seed=seed, arm=arm, panel_id=panel, transcript_id=tid, domain=domain,
                    n_positions=len(x), variance=float(x.var()) if len(x) else np.nan,
                    maximum=float(x.max()) if len(x) else np.nan, q99=float(np.quantile(x,.99)) if len(x) else np.nan,
                    near_constant=bool(x.var() <= 1e-12) if len(x) else True))
    for seed in sorted({k[0] for k in profiles}):
        for arm in ARMS:
            panels = sorted(k[2] for k in profiles if k[:2] == (seed, arm))
            for a, b in itertools.combinations(panels, 2):
                for tid in ids:
                    for domain in DOMAINS:
                        cross.append(dict(training_seed=seed, arm=arm, panel_a=a, panel_b=b,
                            pair=f'{a}__{b}', transcript_id=tid, domain=domain,
                            **aligned_metrics(profiles[seed,arm,a][tid], profiles[seed,arm,b][tid], domain)))
        for a, b in CONTRASTS:
            panels = sorted({k[2] for k in profiles if k[:2] == (seed,a)} & {k[2] for k in profiles if k[:2] == (seed,b)})
            for panel in panels:
                for tid in ids:
                    for domain in DOMAINS:
                        sensitivity.append(dict(training_seed=seed, arm_a=a, arm_b=b, panel_id=panel,
                            contrast=f'{b}_vs_{a}', transcript_id=tid, domain=domain,
                            **aligned_metrics(profiles[seed,a,panel][tid], profiles[seed,b,panel][tid], domain)))
    common_columns = ['training_seed','transcript_id','domain','n_positions','PCC','Spearman','RMSE','reason']
    return (pd.DataFrame(cross) if cross else pd.DataFrame(columns=common_columns+['arm','pair','panel_a','panel_b']),
            pd.DataFrame(sensitivity) if sensitivity else pd.DataFrame(columns=common_columns+['arm_a','arm_b','panel_id','contrast','variance_a','variance_b']),
            pd.DataFrame(diagnostics))


def cross_policy_effects(frame, ids, n_boot, seed):
    """Matched difference of medians, not median of transcript differences.

    A common finite cohort across every matched pair is selected first. Calls
    share the RNG seed/cohort, so all pairs carry the same transcript resamples.
    """
    rows, cohorts = [], []
    for training_seed in sorted(frame.training_seed.unique()):
        for a, b in CONTRASTS:
            for domain in DOMAINS:
                f = frame[(frame.training_seed == training_seed) & (frame.domain == domain)]
                pairs = sorted(set(f.loc[f.arm == a,'pair']) & set(f.loc[f.arm == b,'pair']))
                if not pairs:
                    continue
                for metric in ('PCC','Spearman','RMSE'):
                    matrices = [f[f.arm == arm].pivot(index='transcript_id',columns='pair',values=metric)
                                .reindex(index=ids,columns=pairs).to_numpy(float) for arm in (a,b)]
                    valid = np.isfinite(matrices[0]).all(axis=1) & np.isfinite(matrices[1]).all(axis=1)
                    cohort = np.asarray(ids)[valid]
                    for tid, ok in zip(ids, valid):
                        cohorts.append(dict(training_seed=training_seed,contrast=f'{b}_minus_{a}',domain=domain,
                            metric=metric,transcript_id=tid,included=bool(ok),reason='ok' if ok else 'undefined_in_at_least_one_matched_pair_or_policy'))
                    if not len(cohort):
                        continue
                    matrices = [m[valid] for m in matrices]
                    for group, columns in [('pooled_matched_pairs',list(range(len(pairs))))]+[(p,[i]) for i,p in enumerate(pairs)]:
                        point, ci, _ = paired_bootstrap(matrices[0][:,columns],matrices[1][:,columns],n_boot,seed)
                        rows.append(dict(training_seed=training_seed,contrast=f'{b}_minus_{a}',arm_a=a,arm_b=b,
                            domain=domain,metric=metric,group=group,estimate=point[2],ci_low=ci[0,2],ci_high=ci[1,2],
                            median_a=point[0],median_b=point[1],n_transcripts=len(cohort),cohort_hash=transcript_id_hash(cohort),
                            matched_pairs=';'.join(pairs),n_matched_pairs=len(pairs),all_six_pairs=len(pairs)==6,
                            bootstrap_replicates=n_boot,bootstrap_seed=seed,statistic='difference_of_medians'))
    return (pd.DataFrame(rows) if rows else pd.DataFrame(columns=['training_seed','contrast','domain','metric','group','estimate','ci_low','ci_high']),
            pd.DataFrame(cohorts,columns=['training_seed','contrast','domain','metric','transcript_id','included','reason']))


def summarize_metrics(frame, groups):
    rows = []
    for key, group in frame.groupby(groups):
        for metric in ('PCC','Spearman','RMSE'):
            x = group[metric].to_numpy(float); valid = x[np.isfinite(x)]
            rows.append(dict(zip(groups,key),metric=metric,n_valid=len(valid),n_excluded=len(x)-len(valid),
                median=float(np.median(valid)) if len(valid) else np.nan,
                mean=float(np.mean(valid)) if len(valid) else np.nan,
                q25=float(np.quantile(valid,.25)) if len(valid) else np.nan,
                q75=float(np.quantile(valid,.75)) if len(valid) else np.nan))
    return pd.DataFrame(rows)


def collect_logs(availability, *, extra_tags=(), tag_prefixes=()):
    """Read standard progress metrics and any requested diagnostic tag families."""
    records = []
    for task in availability.to_dict('records'):
        for run in find_event_runs(Path(task['task_directory'])/'logs'):
            scalars = load_scalars(run)
            epochs = {e.step:int(e.value) for e in scalars.get('epoch',[])}
            for tag in scalars:
                if tag not in TAGS and tag not in extra_tags and not tag.startswith(tag_prefixes):
                    continue
                for event in scalars.get(tag,[]):
                    records.append(dict(task_id=task['task_id'],training_seed=task['training_seed'],arm=task['arm'],
                        panel_id=task['panel_id'],tag=tag,step=event.step,epoch=epochs.get(event.step,np.nan),
                        value=event.value,wall_time=event.wall_time,source=str(run)))
    frame = pd.DataFrame(records,columns=['task_id','training_seed','arm','panel_id','tag','step','epoch','value','wall_time','source'])
    # If logging restarted, retain the last recorded value at the same optimizer step.
    if len(frame):
        frame = frame.sort_values('wall_time').drop_duplicates(['task_id','tag','step'],keep='last')
    return frame


def plot_label(text):
    return str(text).replace('_',r'\_').replace('%',r'\%') if plt.rcParams['text.usetex'] else str(text)


def save_figure(fig, stem, dpi):
    for ext in ('pdf','svg','png'):
        fig.savefig(stem.with_suffix('.'+ext),dpi=dpi)
    plt.close(fig)
    return stem


def ecdf(ax, values, **kwargs):
    x = np.sort(np.asarray(values,dtype=float))
    x = x[np.isfinite(x)]
    if len(x):
        ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',**kwargs)


def plot_availability(availability, out, dpi):
    from matplotlib.colors import ListedColormap
    seeds = sorted(availability.training_seed.unique())
    fig, axes = plt.subplots(1,len(seeds),figsize=(7.15,2.65),squeeze=False)
    mapping = {'no_local_outputs':0,'logs_only':1,'checkpoint_without_validated_export':1,'invalid_artifacts':2,'validated_predictions':3}
    for ax, seed in zip(axes[0],seeds):
        f = availability[availability.training_seed==seed]
        panels = sorted(f.panel_id.unique())
        matrix = f.pivot(index='panel_id',columns='arm',values='status').reindex(index=panels,columns=ARMS)
        a = np.array([[mapping[value] for value in row] for row in matrix.to_numpy()])
        ax.pcolormesh(np.arange(4)-.5,np.arange(len(panels)+1)-.5,a,
                      cmap=ListedColormap(['#eeeeee','#F1CF87','#C86464','#80B6A8']),
                      vmin=0,vmax=3,edgecolors='white',linewidth=.5)
        ax.invert_yaxis()
        for i in range(len(panels)):
            for j in range(3):
                ax.text(j,i,['--','Logs/ckpt','Invalid','L ready'][a[i,j]],ha='center',va='center',fontsize=8)
        ax.set_xticks(range(3),[LABELS[a] for a in ARMS]); ax.set_yticks(range(len(panels)),[p.replace('panel_','P') for p in panels])
        ax.set_title(f'Seed {seed}: {int((a==3).sum())}/12',loc='left')
        ax.tick_params(length=0)
    fig.text(.5,.03,'Local artifact availability; missing downloads are not evidence of failed training.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.06,right=.98,bottom=.18,top=.84,wspace=.3)
    return [save_figure(fig,out/'availability',dpi)]


def reference_tables(weights):
    mass = weights.groupby(['panel_id','arm','quality_group'],as_index=False).pi.sum().rename(columns={'pi':'reference_mass'})
    sources = weights.groupby(['panel_id','arm','source_family'],as_index=False).pi.sum().rename(columns={'pi':'source_mass'})
    rows = []
    for (panel,arm),g in weights.groupby(['panel_id','arm']):
        s = sources[(sources.panel_id==panel)&(sources.arm==arm)].sort_values(['source_mass','source_family'],ascending=[False,True])
        largest = g.sort_values(['pi','dataset_id'],ascending=[False,True]).iloc[0]
        rows.append(dict(panel_id=panel,arm=arm,N_ref=1/np.square(g.pi).sum(),max_dataset_weight=largest.pi,
            largest_dataset=largest.dataset_id,weighted_mean_global_rank=float(np.dot(g.pi,g.global_rank)),
            largest_source=s.iloc[0].source_family,max_source_mass=s.iloc[0].source_mass))
    return mass,sources,pd.DataFrame(rows)


def plot_reference_design(mass, concentration, grouping, out, dpi):
    fig, axes = plt.subplots(1,2,figsize=(7.15,3.25))
    panels = sorted(mass.panel_id.unique()); group_colors = ['#2166AC','#92C5DE','#F4A582','#B2182B']
    for i,panel in enumerate(panels):
        for j,arm in enumerate(ARMS):
            x = i+(j-1)*.23; bottom=0
            for group,color in enumerate(group_colors,1):
                value = mass.loc[(mass.panel_id==panel)&(mass.arm==arm)&(mass.quality_group==group),'reference_mass'].sum()
                axes[0].bar(x,value,bottom=bottom,width=.2,color=color,linewidth=.25,edgecolor='white')
                bottom += value
            axes[0].text(x,-.025,'EQR'[j],ha='center',va='top',fontsize=8)
        for j,arm in enumerate(ARMS):
            g = concentration[(concentration.panel_id==panel)&(concentration.arm==arm)].iloc[0]
            axes[1].plot(i+(j-1)*.16,g.max_source_mass,'o',color=COLORS[arm],ms=4)
    names=[p.replace('panel_','P') for p in panels]
    axes[0].set_xticks(range(4),names); axes[0].tick_params(axis='x',pad=14,length=0)
    axes[0].set_ylim(0,1); axes[0].set_ylabel('Reference mass'); axes[0].set_title('A. Quality composition',loc='left')
    axes[1].set_xticks(range(4),names); axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel('Largest source-family share'); axes[1].set_title('B. Source concentration',loc='left')
    axes[1].legend(handles=[Line2D([],[],color=COLORS[a],marker='o',lw=0,label=LABELS[a]) for a in ARMS],fontsize=8,loc='upper right')
    axes[1].grid(axis='y',alpha=.2)
    cuts=grouping['boundaries']
    definitions=[rf'$r\leq {cuts[0]:g}$ (best group)',rf'${cuts[0]:g}<r\leq {cuts[1]:g}$',
                 rf'${cuts[1]:g}<r\leq {cuts[2]:g}$',rf'$r>{cuts[2]:g}$ (worst group)']
    fig.legend(handles=[Patch(color=c,label=label) for c,label in zip(group_colors,definitions)],loc='lower center',ncol=2,fontsize=8,bbox_to_anchor=(.5,0))
    fig.text(.5,.98,'Frozen reference design (not model performance); E = equal, Q = ranked, R = reverse',ha='center',va='top',fontsize=8)
    fig.subplots_adjust(left=.085,right=.99,top=.83,bottom=.32,wspace=.43)
    return [save_figure(fig,out/'reference_design',dpi)]


def plot_cross_panel(frame, effects, out, dpi):
    stems=[]
    for seed in sorted(frame.training_seed.unique()):
        f = frame[(frame.training_seed==seed)&(frame.domain=='full_cds')]
        pairs = sorted(f.pair.unique()); arms = [a for a in ARMS if a in set(f.arm)]
        fig,ax=plt.subplots(figsize=(7.15,3.3))
        for i,pair in enumerate(pairs):
            for j,arm in enumerate(arms):
                x=f.loc[(f.pair==pair)&(f.arm==arm),'PCC'].dropna().to_numpy()
                if not len(x): continue
                y=i+(j-(len(arms)-1)/2)*.23
                if len(x)>1 and np.ptp(x)>0:
                    parts=ax.violinplot(x,positions=[y],vert=False,widths=.21 if len(arms)>1 else .5,showextrema=False)
                    for body in parts['bodies']: body.set_facecolor(COLORS[arm]);body.set_edgecolor(COLORS[arm]);body.set_alpha(.23)
                q1,med,q3=np.quantile(x,[.25,.5,.75])
                ax.plot([q1,q3],[y,y],color=COLORS[arm],lw=3,solid_capstyle='butt')
                ax.plot(med,y,'|',ms=8,color='black')
        ax.set_yticks(range(len(pairs)),[p.replace('panel_','P').replace('__',' -- ') for p in pairs]);ax.invert_yaxis()
        x=f.PCC.dropna(); ax.set_xlim(min(-.02,float(x.min())-.025) if len(x) else -1,1.02)
        ax.set_xlabel('Full-CDS per-transcript PCC');ax.set_title('Cross-panel reproducibility',loc='left')
        ax.legend(handles=[Patch(facecolor=COLORS[a],alpha=.6,label=LABELS[a]) for a in arms],loc='lower left')
        counts=', '.join(f'{LABELS[a]} {f[f.arm==a].pair.nunique()}/6 pairs' for a in ARMS)
        fig.text(.98,.95,f'Seed {seed}',ha='right',fontsize=9)
        fig.text(.5,.025,counts+'; unmatched availability is descriptive only.',ha='center',fontsize=8)
        fig.subplots_adjust(left=.16,right=.98,bottom=.2,top=.87)
        stems.append(save_figure(fig,out/f'cross_panel_reproducibility_seed{seed}',dpi))
    for seed in sorted(effects.training_seed.unique()):
        f=effects[(effects.training_seed==seed)&(effects.domain=='full_cds')&(effects.group!='pooled_matched_pairs')]
        if f.empty:continue
        fig,axes=plt.subplots(1,2,figsize=(7.15,3.4))
        groups=sorted(f.group.unique())
        for ax,metric in zip(axes,('PCC','RMSE')):
            for j,(a,b) in enumerate(CONTRASTS):
                c=f[f.contrast==f'{b}_minus_{a}']; c=c[c.metric==metric]
                for row in c.itertuples():
                    y=groups.index(row.group)+(j-1)*.2
                    ax.plot([row.ci_low,row.ci_high],[y,y],color=COLORS[b],ls='--' if a=='reverse' else '-',lw=1)
                    ax.plot(row.estimate,y,['o','s','^'][j],color=COLORS[b],ms=4)
            ax.axvline(0,ls='--',color='.45',lw=.7)
            ax.set_yticks(range(len(groups)),[p.replace('panel_','P').replace('__','--') for p in groups]);ax.invert_yaxis()
            ax.set_xlabel(f'Change in median {metric}');ax.set_title(metric,loc='left');ax.grid(axis='x',alpha=.2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        fig.legend(handles=[Line2D([],[],marker=['o','s','^'][j],color=COLORS[b],lw=1,
                   label=f'{LABELS[b]} minus {LABELS[a]} ({f[f.contrast==f"{b}_minus_{a}"].group.nunique()}/6 pairs)')
                   for j,(a,b) in enumerate(CONTRASTS) if f'{b}_minus_{a}' in set(f.contrast)],loc='lower center',ncol=2,fontsize=8)
        prefix='Partial: ' if not f.all_six_pairs.all() else ''
        fig.suptitle(f'{prefix}matched cross-panel policy effects, seed {seed}',fontsize=12)
        fig.subplots_adjust(left=.15,right=.99,bottom=.23,top=.84,wspace=.63)
        stems.append(save_figure(fig,out/f'paired_reproducibility_effects_seed{seed}',dpi))
    return stems


def plot_sensitivity(sensitivity,out,dpi):
    stems=[]
    for (seed,panel),f in sensitivity.groupby(['training_seed','panel_id']):
        # Each variance scatter must have exactly one definition for each axis.
        if f.contrast.nunique()>1:
            for contrast,sub in f.groupby('contrast'):
                stems.extend(plot_sensitivity(sub,out/contrast,dpi))
            continue
        fig,axes=plt.subplots(1,2,figsize=(7.15,2.9))
        for j,(a,b) in enumerate(CONTRASTS):
            selected=f[f.contrast==f'{b}_vs_{a}']
            if selected.empty:continue
            for domain,style in [('full_cds','-'),('interior_20','--')]:
                g=selected[selected.domain==domain]
                ecdf(axes[0],g.PCC,color=COLORS[b],ls=style,lw=1.2,
                     label=f'{LABELS[b]} vs {LABELS[a]}, '+('full' if domain=='full_cds' else 'interior'))
            g=selected[selected.domain=='full_cds']
            mask=(g.variance_a>0)&(g.variance_b>0)
            axes[1].scatter(g.loc[mask,'variance_a'],g.loc[mask,'variance_b'],s=5,alpha=.2,color=COLORS[b],rasterized=False)
            axes[1].set_xlabel(f'{LABELS[a]}: variance of $L_t$')
            axes[1].set_ylabel(f'{LABELS[b]}: variance of $L_t$')
        axes[0].set_xlabel('Same-panel cross-policy PCC');axes[0].set_ylabel('Fraction of transcripts')
        axes[0].set_ylim(0,1.02);axes[0].set_title('A. Shared-profile sensitivity',loc='left')
        if axes[0].get_legend_handles_labels()[0]:axes[0].legend(loc='upper left',fontsize=8)
        positive=f.loc[f.domain=='full_cds',['variance_a','variance_b']].to_numpy().ravel();positive=positive[positive>0]
        if len(positive):
            low,high=positive.min()*.8,positive.max()*1.2
            axes[1].plot([low,high],[low,high],'--',color='.4',lw=.8)
            axes[1].set(xscale='log',yscale='log',xlim=(low,high),ylim=(low,high))
        else:
            axes[1].text(.5,.5,'All profiles have zero positional variance',ha='center',transform=axes[1].transAxes,fontsize=8)
        axes[1].set_title('B. Profile variance',loc='left')
        fig.text(.5,.985,f'{panel.replace("panel_","P")}, seed {seed}: {f.transcript_id.nunique():,} held-out transcripts',ha='center',va='top',fontsize=9)
        fig.subplots_adjust(left=.085,right=.99,bottom=.21,top=.8,wspace=.43)
        out.mkdir(parents=True,exist_ok=True)
        stems.append(save_figure(fig,out/f'policy_sensitivity_{panel}_seed{seed}',dpi))
    return stems


def plot_learning(logs,availability,out,dpi):
    stems=[]
    if logs.empty:return stems
    for seed in sorted(logs.training_seed.unique()):
        fig,axes=plt.subplots(2,2,figsize=(7.15,4.6),sharex=True,sharey=True)
        for ax,panel in zip(axes.flat,sorted(availability.panel_id.unique())):
            for arm in ARMS:
                g=logs[(logs.training_seed==seed)&(logs.panel_id==panel)&(logs.arm==arm)&(logs.tag=='val_loss')].sort_values('epoch')
                if g.empty:continue
                ax.plot(g.epoch+1,g.value,color=COLORS[arm],lw=1.2,label=LABELS[arm])
                a=availability[(availability.training_seed==seed)&(availability.panel_id==panel)&(availability.arm==arm)]
                epoch=a.iloc[0].get('selected_epoch',np.nan)
                chosen=g[g.epoch==epoch]
                if len(chosen):ax.plot(chosen.epoch+1,chosen.value,'o',ms=4,color=COLORS[arm])
            ax.set_title(panel.replace('panel_','P'),loc='left');ax.grid(axis='y',alpha=.2)
        for ax in axes[1]:ax.set_xlabel('Epoch (one-based)')
        for ax in axes[:,0]:ax.set_ylabel('Validation objective')
        fig.legend(handles=[Line2D([],[],color=COLORS[a],label=LABELS[a]) for a in ARMS if a in set(logs.arm)],loc='lower center',ncol=3)
        fig.suptitle(f'Observation-fit context, seed {seed}; dots mark exported checkpoints',fontsize=11)
        fig.subplots_adjust(left=.085,right=.98,bottom=.19,top=.87,wspace=.2,hspace=.32)
        stems.append(save_figure(fig,out/f'validation_learning_curves_seed{seed}',dpi))
    return stems


def example_selections(sensitivity, saved_path):
    """Freeze examples on first analysis; never pick the largest ranking benefit."""
    if saved_path.is_file():
        return pd.read_csv(saved_path)
    rows=[]
    # Fixed panel and seed, not whichever fitted model looks most attractive.
    f=sensitivity[(sensitivity.training_seed==42)&(sensitivity.panel_id=='panel_01')&
                  (sensitivity.contrast=='ranked_vs_equal')&(sensitivity.domain=='full_cds')]
    f=f[np.isfinite(f.PCC)].copy()
    if len(f):
        for quantile in (.5,.1):
            target=float(np.quantile(f.PCC,quantile,method='linear'))
            selected=f.assign(distance=(f.PCC-target).abs()).sort_values(['distance','transcript_id']).iloc[0]
            rows.append(dict(training_seed=42,panel_id='panel_01',transcript_id=selected.transcript_id,
                quantile=quantile,target_PCC=target,selection_PCC=selected.PCC,
                selection_rule='Closest to empirical percentile of same-panel equal/ranked full-CDS PCC; lexicographic ID tie-break; frozen on first analysis; descriptive test-set examples, not validation-selected.'))
    return pd.DataFrame(rows,columns=['training_seed','panel_id','transcript_id','quantile','target_PCC','selection_PCC','selection_rule'])


def plot_examples(profiles,selection,out,dpi):
    if selection.empty:return []
    fig,axes=plt.subplots(len(selection),1,figsize=(7.15,2.05*len(selection)),squeeze=False)
    for ax,row in zip(axes[:,0],selection.itertuples()):
        for arm in ARMS:
            key=(int(row.training_seed),arm,row.panel_id)
            if key not in profiles:continue
            x=profiles[key][row.transcript_id]['values']
            ax.plot(np.arange(1,len(x)+1),x,color=COLORS[arm],lw=.7,alpha=.85,label=LABELS[arm])
        ax.set_xlim(1,len(x));ax.set_ylim(bottom=0)
        ax.set_ylabel('$L_t$ (mean one)');ax.set_xlabel('Codon position')
        ax.set_title(f'{row.transcript_id} (selection percentile {100*row.quantile:g})',loc='left',fontsize=10)
    axes[0,0].legend(ncol=3,loc='upper right',fontsize=8)
    fig.subplots_adjust(left=.085,right=.99,bottom=.13,top=.92,hspace=.65)
    return [save_figure(fig,out/'fixed_transcript_examples',dpi)]


def write_report(out,manifest,availability,sensitivity_summary,effects,diagnostic_summary,selected_validation,limitations,command,figures):
    ready=availability[availability.status=='validated_predictions']
    counts=availability.assign(ready=availability.status.eq('validated_predictions')).pivot_table(index='training_seed',columns='arm',values='ready',aggfunc='sum').reindex(columns=ARMS)
    html_parts=['<!doctype html><html><head><meta charset="utf-8"><title>Reference directionality analysis</title>',
        '<style>body{font:16px/1.55 system-ui;max-width:1080px;margin:35px auto;padding:0 22px;color:#222}table{border-collapse:collapse;font-size:13px}td,th{padding:6px 12px;border-bottom:1px solid #ddd;text-align:right}img{width:100%;height:auto}code,pre{background:#f3f3f3;overflow:auto;padding:8px}.note{border-left:4px solid #D55E00;padding:12px;background:#fff8ef}</style></head><body>',
        '<h1>Equal, ranked and reverse reference weights</h1>',
        f'<p class="note">Partial local results: {len(ready)}/{len(availability)} validated exports. Missing downloads are not labelled failed training. The prepared manifest is a design record, not a completion record.</p>',
        '<h2>What the plots answer</h2><ol><li>Cross-panel PCC distributions and matched changes: does reference policy alter reproducibility across source-disjoint panels?</li><li>Same-panel cross-policy PCC and profile variance: how much does the native shared branch change? A larger same-panel PCC means less sensitivity, not improved reproducibility.</li><li>Validation curves: is agreement accompanied by comparable observation fit?</li><li>Reference mass and source concentration: what intervention do the three policies actually implement?</li></ol>',counts.to_html(),
        '<p>Cross-panel policy effects need at least two matching panels in each policy; a full comparison needs all four. Missing arms are never replaced, and seeds are never pooled into model replicates.</p>',
        '<h2>Ranking and matching</h2>',
        f'<p>Ten-component frozen ranking; R={manifest["ranking"]["rank_universe"]:g}; SHA256 <code>{manifest["ranking"]["sha256"]}</code>. Rank 1 is best. Ranked weights use q=(R-r+1)/R normalized over the whole panel. Reverse assigns that exact panel weight multiset in reverse global-rank order. Equal is uniform. Source-family shares can still differ between ranked and reverse.</p>',
        '<p>Frozen configurations, dataset order, seed, split and training-only reliability references are checked. Actual Hydra configurations, runtime gamma weights and execution manifest identities are checked for every included export. Initial trainable parameter hashes match within panel/seed across arms.</p>',
        '<h2>Methods and interpretation</h2><p>Native, positive, finite, mean-one shared profiles are evaluated at identical codon sequences and coordinates. No smoothing, clipping, offsets, interior renormalization or imputation. Interior evaluation removes 20 codons per end. PCC and Spearman are undefined for positional variance ≤10⁻¹²; RMSE remains interpretable for constant profiles.</p>',
        '<p>Reproducibility effects are differences of policy medians, including a separately named pooled transcript–pair median. Positive PCC and negative RMSE changes favour agreement. Paired transcript-cluster bootstrap intervals resample each transcript with both policies and every matched pair attached. All pairs in a contrast use one common finite cohort and the same resamples. Intervals are conditional on the fitted models and selected collections; panel pairs sharing models/transcripts are dependent. Similar effective reference counts N_ref describe weight concentration, not the number of independent experimental sources.</p>',
        '<p>Violin KDEs stop at observed support. Thick bars span the 25th–75th percentiles and black ticks show medians; no whiskers. All lower tails are visible. ECDFs use actual unsmoothed transcript metrics. Variance scatter uses original amplitudes. Example transcripts are frozen on first analysis: P01/seed42 examples closest to the empirical 50th and 10th percentiles of equal/ranked full-CDS PCC, with lexicographic ID tie-breaking. They are descriptive test-selected examples, not validation-selected or evidence of a ranking benefit.</p>',
        '<p><strong>Sequence-only exports contain dummy observation-dependent outputs.</strong> These are not used for test reconstruction, gamma/alpha estimates or supplied-mass diagnostics. Observation-fit context comes only from logged validation metrics at the exported best-validation-loss epoch. Validation is used for checkpoint selection, so it is not independent test evidence. Lower alpha is not automatically better. QC rank is not biological ground truth; agreement is not biological accuracy.</p>',
        '<h2>Available same-panel sensitivity</h2>',sensitivity_summary.to_html(index=False,float_format=lambda x:f'{x:.5g}'),
        '<h2>Matched cross-panel policy effects</h2>',effects.to_html(index=False,float_format=lambda x:f'{x:.5g}') if len(effects) else '<p>Not yet estimable from the downloaded models. No placeholder effects or confidence intervals were created.</p>',
        '<h2>Profile-collapse diagnostics</h2>',diagnostic_summary.to_html(index=False,float_format=lambda x:f'{x:.5g}'),
        '<p>See <a href="paired_amplitude_summary.csv">paired amplitude ratios</a>. The fraction below one-half variance is a descriptive, fixed heuristic, not a biological acceptability threshold. Median ratios must not conceal extreme reductions in a minority of transcripts.</p>',
        '<h2>Validation at the exported checkpoint</h2>',selected_validation.to_html(index=False,float_format=lambda x:f'{x:.5g}'),
        '<h2>Figures</h2>']
    for stem in figures:
        relative=stem.relative_to(out)
        html_parts += [f'<h3>{html.escape(stem.name.replace("_"," "))}</h3>',f'<p><a href="{relative}.pdf">PDF</a> · <a href="{relative}.svg">SVG</a></p><img src="{relative}.png" alt="{html.escape(stem.name)}">']
    html_parts += ['<h2>Provenance limitations</h2><ul>']+[f'<li>{html.escape(x)}</li>' for x in limitations]
    html_parts += ['</ul><h2>All task states</h2>',availability[['training_seed','arm','panel_id','status','detail']].to_html(index=False),'<h2>Regenerate</h2>',f'<pre>{html.escape(command)}</pre></body></html>']
    (out/'analysis_report.html').write_text('\n'.join(html_parts))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--bootstrap-replicates',type=int,default=5000)
    parser.add_argument('--bootstrap-seed',type=int,default=20260910)
    parser.add_argument('--dpi',type=int,default=300)
    parser.add_argument('--no-tex',action='store_true')
    parser.add_argument('--skip-training-logs',action='store_true')
    args=parser.parse_args(argv)
    if args.bootstrap_replicates<1:parser.error('Bootstrap replicates must be positive.')
    root=locate_experiment(args.run_root)
    out=(args.output_dir or root/'analysis_directionality').resolve();out.mkdir(parents=True,exist_ok=True)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    manifest,tasks,configs,split,ids,weights,grouping,hashes,limitations=audit_inputs(root)
    print('Audited frozen ten-component experiment; loading available test exports.',flush=True)
    profiles,availability=collect(root,manifest,tasks,configs,ids,weights)
    print(availability.groupby(['training_seed','arm','status']).size().to_string(),flush=True)
    cross,sensitivity,diagnostics=evaluation_tables(profiles,ids)
    effects,cohorts=cross_policy_effects(cross,ids,args.bootstrap_replicates,args.bootstrap_seed)
    cross_summary=summarize_metrics(cross,['training_seed','arm','domain','pair'])
    sensitivity_summary=summarize_metrics(sensitivity,['training_seed','contrast','panel_id','domain'])
    if len(diagnostics):
        diagnostic_summary=diagnostics.groupby(['training_seed','arm','panel_id','domain']).agg(
            n_transcripts=('transcript_id','size'),median_variance=('variance','median'),near_constant_fraction=('near_constant','mean'),
            median_profile_maximum=('maximum','median'),maximum_profile_value=('maximum','max')).reset_index()
    else:diagnostic_summary=pd.DataFrame()
    logs=pd.DataFrame() if args.skip_training_logs else collect_logs(availability)
    if args.skip_training_logs:limitations.append('Validation logs skipped by CLI request.')
    selected=[]
    for row in availability[availability.status=='validated_predictions'].to_dict('records'):
        record={k:row[k] for k in ['training_seed','arm','panel_id','selected_epoch']}
        if len(logs):
            for tag in TAGS:
                f=logs[(logs.task_id==row['task_id'])&(logs.epoch==row.get('selected_epoch'))&(logs.tag==tag)]
                record[tag]=float(f.sort_values('wall_time').iloc[-1].value) if len(f) else np.nan
        selected.append(record)
    selected_validation=pd.DataFrame(selected)
    mass,sources,concentration=reference_tables(weights)
    amplitude=[]
    for key,f in sensitivity.groupby(['training_seed','contrast','panel_id','domain']):
        valid=(f.variance_a>0)&np.isfinite(f.variance_b)
        ratio=(f.loc[valid,'variance_b']/f.loc[valid,'variance_a']).to_numpy()
        amplitude.append(dict(zip(['training_seed','contrast','panel_id','domain'],key),n_valid=len(ratio),
            n_excluded=int((~valid).sum()),median_variance_ratio=float(np.median(ratio)) if len(ratio) else np.nan,
            fraction_variance_ratio_below_half=float((ratio<.5).mean()) if len(ratio) else np.nan))
    tables=dict(training_availability=availability,reference_weights=weights,reference_quality_group_mass=mass,
        source_family_reference_mass=sources,reference_concentration=concentration,
        cross_panel_transcript_metrics=cross,same_panel_policy_sensitivity=sensitivity,profile_diagnostics=diagnostics,
        cross_panel_summary=cross_summary,same_panel_sensitivity_summary=sensitivity_summary,
        paired_reproducibility_effects=effects,matched_bootstrap_cohorts=cohorts,
        paired_amplitude_summary=pd.DataFrame(amplitude),
        profile_diagnostic_summary=diagnostic_summary,training_scalars=logs,selected_checkpoint_validation=selected_validation,
        verified_input_hashes=pd.DataFrame(hashes))
    for name,frame in tables.items():frame.to_csv(out/f'{name}.csv',index=False)
    (out/'global_quality_groups.json').write_text(json.dumps(grouping,indent=2)+'\n')
    if args.no_tex:
        from Utils.publication_plot_style import LATEX_PAPER_RC
        style=dict(LATEX_PAPER_RC)
    else:
        style=publication_rc()
    style.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':12,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9})
    if args.no_tex:style.update({'text.usetex':False,'font.serif':['DejaVu Serif'],'mathtext.fontset':'cm'})
    with plt.rc_context(style):
        current_figures=plot_availability(availability,figures,args.dpi)
        current_figures+=plot_reference_design(mass,concentration,grouping,figures,args.dpi)
        current_figures+=plot_cross_panel(cross,effects,figures,args.dpi)
        current_figures+=plot_sensitivity(sensitivity,figures,args.dpi)
        current_figures+=plot_learning(logs,availability,figures,args.dpi)
    selection=example_selections(sensitivity,out/'example_selection.csv')
    selection.to_csv(out/'example_selection.csv',index=False)
    example_rows=[]
    for row in selection.itertuples():
        for arm in ARMS:
            key=(int(row.training_seed),arm,row.panel_id)
            if key not in profiles:continue
            example_rows.extend(dict(training_seed=row.training_seed,panel_id=row.panel_id,arm=arm,
                                     transcript_id=row.transcript_id,codon_position=i+1,L_t=value)
                                for i,value in enumerate(profiles[key][row.transcript_id]['values']))
    pd.DataFrame(example_rows,columns=['training_seed','panel_id','arm','transcript_id','codon_position','L_t']).to_csv(out/'example_profiles.csv',index=False)
    with plt.rc_context(style):
        current_figures+=plot_examples(profiles,selection,figures,args.dpi)
    command=shlex.join([sys.executable,str(Path(__file__).resolve()),'--run-root',str(root),'--output-dir',str(out),
        '--bootstrap-replicates',str(args.bootstrap_replicates),'--bootstrap-seed',str(args.bootstrap_seed),'--dpi',str(args.dpi)]
        +(['--no-tex'] if args.no_tex else [])+(['--skip-training-logs'] if args.skip_training_logs else []))
    write_report(out,manifest,availability,sensitivity_summary,effects,diagnostic_summary,selected_validation,limitations,command,current_figures)
    (out/'README.md').write_text(
        '# Reference-directionality analysis\n\nOpen [the report](analysis_report.html) for methods, current availability and figures.\n\n'
        f'Validated local models: {int(availability.status.eq("validated_predictions").sum())}/{len(tasks)}. '
        'Figures represent downloaded artifacts only, not scheduler status. Re-run after downloading more exports.\n\n'
        'PDF/SVG figures use native vector artists; PNG previews use the requested DPI. Each figure has numeric CSV sources in this directory. '
        'Cross-panel contrasts are differences of medians with paired transcript-cluster intervals; same-panel PCC is sensitivity, not a reproducibility effect.\n\n'
        f'```bash\n{command}\n```\n')
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='complete' if availability.status.eq('validated_predictions').all() else 'partial',
        experiment_root=str(root),experiment_manifest_sha256=sha256(root/'experiment_manifest.json'),script_sha256=sha256(Path(__file__)),
        validated_models=int(availability.status.eq('validated_predictions').sum()),planned_models=len(tasks),
        ranking=manifest['ranking'],common_test_count=len(ids),common_test_hash=transcript_id_hash(ids),
        bootstrap_replicates=args.bootstrap_replicates,bootstrap_seed=args.bootstrap_seed,tex_used=style['text.usetex'],
        current_figures=[str(stem.relative_to(out)) for stem in current_figures],
        input_mutations=False,training_or_inference_launched=False,limitations=limitations,command=command,
        output_sha256={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='analysis_manifest.json'})
    (out/'analysis_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Wrote actual-data figures and tables: {out}/analysis_report.html',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
