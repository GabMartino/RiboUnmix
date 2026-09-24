#!/usr/bin/env python3
"""Read-only analysis of partial cumulative equal/ranked/reverse experiments.

The primary endpoint is mean transcript PCC between adjacent sizes, not
agreement with N=114. Missing exports leave gaps. Policy effects require both
endpoints under both policies at the same seed. A common finite cohort and
joint transcript bootstrap are recomputed on each invocation; seeds are never
pooled as independent transcript observations. No inference/training is run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import re
import shlex
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyses.analyze_rank_balanced_reference_directionality import (
    ARMS, COLORS, LABELS, CONTRASTS, DOMAINS, aligned_metrics, collect_logs,
    read_json, relocate, save_figure, sha256, summarize_metrics, plot_sensitivity,
)
from analyses.analyze_real_panel_convergence import _locate_panel_prediction, _extract_panel_profiles
from prepare_rank_balanced_reference_directionality import _reference_policy, _assert_arm_match, object_sha256
from run_real_independent_panel_convergence_quality_rank import _flatten_config
from run_real_exp8_L_stability_quality_rank import inspect_ranking_components
from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import load_dataset_quality_ranking
from Utils.external_transcript_split import load_external_transcript_split
from Utils.reliability_references import transcript_id_hash
from Utils.publication_plot_style import publication_rc

DEFAULT_ROOT = ROOT / 'results/real_exp8_cumulative_qrank10_directionality'
METRICS = ('PCC', 'Spearman', 'RMSE')
SLOT_COLUMNS = ['training_seed', 'arm', 'transition', 'N_a', 'N_b']
METRIC_COLUMNS = ['transcript_id', 'domain', 'n_positions', 'PCC', 'Spearman', 'RMSE',
                  'variance_a', 'variance_b', 'reason']


def audit_design(root):
    manifest = read_json(root / 'experiment_manifest.json')
    if manifest['experiment_design'] != 'matched_cumulative_reference_directionality':
        raise ValueError('Expected the new matched cumulative directionality experiment.')
    recorded = Path(manifest['output_root'])
    tasks = manifest['tasks']
    if object_sha256(tasks) != manifest['tasks_sha256']:
        raise ValueError('Frozen task mapping checksum differs.')
    verified, limitations = [], []
    for category in ('frozen_file_sha256', 'code_sha256'):
        for original, expected in manifest[category].items():
            local = relocate(original, root, recorded)
            if not local.is_file() and category == 'code_sha256':
                limitations.append(f'Code snapshot not downloaded: {original}; recorded SHA256 {expected}.')
                continue
            actual = sha256(local)
            if actual != expected:
                raise ValueError(f'{category} mismatch: {local}')
            verified.append(dict(category=category, file=str(local), recorded_path=original, sha256=actual))
    ranking = root / 'frozen_inputs/HEK_riboseq_profile_quality_rank_components.tsv'
    components = inspect_ranking_components(ranking, expected_count=10)
    ranks, q = load_dataset_quality_ranking(str(ranking))
    if (sha256(ranking) != manifest['ranking']['sha256']
            or max(ranks.values()) != manifest['ranking']['R']
            or components['columns'] != manifest['ranking']['components']):
        raise ValueError('Complete global ranking differs from the frozen definition.')
    sizes, seeds = manifest['sizes'], manifest['training_seeds']
    if sizes != sorted(set(sizes)) or len(sizes) < 2:
        raise ValueError('Invalid cumulative size order.')
    expected = set(itertools.product(seeds, ARMS, sizes))
    keys = [(int(t['training_seed']), t['arm'], int(t['N'])) for t in tasks]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Task matrix is not the declared seed/policy/size product.')
    split_path = root / 'frozen_inputs/experiment_split_manifest.json'
    split = read_json(split_path)
    ids = sorted(map(str, split['common_test_ids']))
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate frozen test transcripts.')
    initial = pd.read_csv(root / 'initialization_audit.csv').set_index(['training_seed', 'arm', 'N'])
    saved_weights = pd.read_csv(root / 'reference_weights.csv')
    if saved_weights.duplicated(['N', 'arm', 'dataset_id']).any():
        raise ValueError('Duplicate policy-level reference rows.')
    configs, weight_rows, fold_rows, names_by_n = {}, [], [], {}
    for task, key in zip(tasks, keys):
        cfg_path = relocate(task['config_path'], root, recorded)
        if sha256(cfg_path) != task['config_sha256']:
            raise ValueError(f'{task["run_id"]}: configuration checksum mismatch.')
        cfg = yaml.safe_load(cfg_path.read_text()); configs[key] = cfg
        names = task['datasets']; n = key[2]
        if len(names) != n or len(set(names)) != n or set(names) - set(ranks):
            raise ValueError(f'N={n}: duplicate/missing dataset ranks or wrong capacity.')
        if names != cfg['experiment']['dataset'] or cfg['experiment']['seed'] != key[0]:
            raise ValueError(f'{key}: configuration dataset order or seed differs.')
        if n in names_by_n and names_by_n[n] != names:
            raise ValueError(f'N={n}: policy/seed membership mismatch.')
        names_by_n[n] = names
        train, validation, test, _ = load_external_transcript_split(
            split_path, panel_name=task['source_panel'], experiment_datasets=names)
        frozen_folds = manifest['source_folds'][str(n)]
        if train != frozen_folds['train_ids'] or validation != frozen_folds['validation_ids'] or test != frozen_folds['test_ids']:
            raise ValueError(f'{key}: per-N folds differ from the source design.')
        ref_path = root / f'frozen_inputs/N{n:03d}_reliability_reference_manifest.json'
        ref = read_json(ref_path)
        if (ref['reference_split'] != 'training_only' or ref['heldout_rows_used_for_fitting'] != 0
                or ref['panel_training_transcript_id_hash'] != transcript_id_hash(train)
                or set(ref['datasets']) != set(names)):
            raise ValueError(f'{key}: unmatched training-only reliability reference.')
        if (relocate(cfg['data']['reliability_reference_manifest'], root, recorded) != ref_path
                or relocate(cfg['split']['external_manifest'], root, recorded) != split_path
                or cfg['split']['external_panel_name'] != task['source_panel']
                or relocate(cfg['data']['dataset_quality_ranking']['path'], root, recorded) != ranking):
            raise ValueError(f'{key}: configuration does not use the frozen inputs.')
        policy, rows = _reference_policy(key[1], names, ranks, q)
        reference = cfg['model']['gamma_centering']['reference']
        if reference.get('dataset_names') not in (None, names) or any(reference.get(k) != v for k,v in policy.items()):
            raise ValueError(f'{key}: incorrect full-prefix gamma reference.')
        selected = saved_weights[(saved_weights.N == n) & (saved_weights.arm == key[1])]
        if set(selected.dataset_id) != set(names):
            raise ValueError(f'{key}: reference-weight table has missing/extra datasets.')
        selected = selected.set_index('dataset_id').loc[names]
        for field in ('global_rank', 'assigned_q', 'pi'):
            np.testing.assert_allclose(selected[field], [r[field] for r in rows], rtol=1e-12, atol=1e-14)
        if key[0] == seeds[0]:
            weight_rows.extend(dict(N=n, **row) for row in rows)
            if key[1] == 'equal':
                fold_rows.append(dict(N=n, n_train=len(train), n_validation=len(validation), n_test=len(test),
                    train_hash=transcript_id_hash(train), validation_hash=transcript_id_hash(validation),
                    test_hash=transcript_id_hash(test), reliability_sha256=sha256(ref_path)))
        if initial.loc[key, 'parameter_sha256'] != task['initialization_sha256']:
            raise ValueError(f'{key}: initial-parameter hash mismatch.')
    full = names_by_n[max(sizes)]
    if full != sorted(full, key=lambda d: (ranks[d], d)) or any(names_by_n[n] != full[:n] for n in sizes):
        raise ValueError('Collections are not the fixed global-rank prefixes.')
    for (seed, arm, n), cfg in configs.items():
        _assert_arm_match(configs[seed, 'equal', n], cfg)
        if initial.loc[(seed, slice(None), n), 'parameter_sha256'].nunique() != 1:
            raise ValueError(f'N={n}/seed{seed}: unmatched initializations.')
    limitations.extend([
        'Training-data bytes are not re-read by this analysis; their recorded hashes are retained in the experiment manifest.',
        'Global QC ranking transcript scope is not verified as training-only; w_dt has a separate train-only provenance.',
        'The common test list is frozen, but validation/training lists and fitted w_dt references differ across N.',
        'Downloaded running states are snapshots, not a live scheduler query. A missing export is not proof of failed training.',
    ])
    return manifest, configs, ids, pd.DataFrame(weight_rows), pd.DataFrame(fold_rows), verified, limitations


def verify_runtime_config(prepared, actual, state):
    a, b = _flatten_config(prepared), _flatten_config(actual)
    changes = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    allowed = {'experiment.from_checkpoint', 'experiment.resume_training_state', 'experiment.resume_checkpoint_path'}
    if not changes:
        return
    resumes = [s.get('resume_checkpoint') for s in state.get('attempts', []) if s.get('resume_checkpoint')]
    if (changes - allowed or not resumes or not b.get('experiment.resume_training_state')
            or not b.get('experiment.from_checkpoint') or b.get('experiment.allow_weights_only_resume')
            or b.get('experiment.resume_checkpoint_path') not in [s['path'] for s in resumes]):
        raise ValueError(f'Unapproved runtime configuration differences: {sorted(changes)}')


def read_export(root, task, cfg, ids, weights, state):
    directory = root / task['directory']
    compact, source, reason = _locate_panel_prediction(directory)
    if compact is None:
        return None, dict(detail=reason)
    runtime = read_json(source)['best_val_loss']
    if (runtime['transcript_id_hash'] != transcript_id_hash(ids) or runtime['transcript_count'] != len(ids)
            or runtime.get('split_name') != 'test' or not runtime.get('sequence_only_shared_profile_prediction')):
        raise ValueError('Export is not the frozen sequence-only common test set.')
    if state.get('task_id') != task['run_id'] or state.get('config_sha256') != task['config_sha256']:
        raise ValueError('Execution record does not match the frozen task.')
    gamma = read_json(source.parent / 'gamma_reference_manifest.json')
    names = task['datasets']
    if (gamma['reference_dataset_names'] != names or gamma['centering_mode'] != 'fixed_reference'
            or gamma['weighting'] != cfg['model']['gamma_centering']['reference']['weighting']
            or not gamma.get('pi_is_gamma_reference_only')):
        raise ValueError('Actual gamma reference differs from the complete selected collection/policy.')
    selected = weights[(weights.N == task['N']) & (weights.arm == task['arm'])].set_index('dataset_id').loc[names]
    np.testing.assert_allclose(gamma['reference_pi'], selected.pi, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(gamma['reference_raw_weights'], selected.assigned_q, rtol=1e-6, atol=1e-9)
    profiles, checks = _extract_panel_profiles(panel_name=f'N{task["N"]:03d}', run_identifier=task['run_id'],
        prediction_path=compact, expected_ids=set(ids), mean_one_tolerance=1e-4)
    frame = pd.read_parquet(compact, columns=['transcript_id', 'valid_position_mask', 'run_id', 'N'])
    if frame.run_id.ne(task['run_id']).any() or frame.N.ne(task['N']).any():
        raise ValueError('Compact profile identity differs from the task.')
    for row in frame.itertuples(index=False):
        mask = np.asarray(row.valid_position_mask, bool)
        if mask.shape != profiles[row.transcript_id]['values'].shape or not mask.all():
            raise ValueError('Compact export does not span the complete valid CDS.')
    raw = source.parent / Path(runtime['output_path']).name
    seen = set()
    # Read coordinates and L only: mu/target in sequence-only exports are dummy
    # observation-dependent fields, not held-out reconstruction measurements.
    unity = cfg['model'].get('mean_correction', 'learned') == 'unity'
    columns = ['transcript_id', 'length', 'mask', 'codon_ids', 'L_bio']
    if unity:
        columns.append('gamma')
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=16, use_threads=False, columns=columns):
        for row in batch.to_pylist():
            tid = str(row['transcript_id'])
            mask, values = np.asarray(row['mask'], bool), np.asarray(row['L_bio'], float)
            codons = np.asarray(row['codon_ids'], dtype='<i8')
            if tid not in profiles or values.shape != mask.shape or codons.shape != mask.shape or int(mask.sum()) != row['length']:
                raise ValueError(f'{tid}: raw coordinate/length mismatch.')
            np.testing.assert_allclose(values[mask], profiles[tid]['values'], rtol=2e-5, atol=2e-6)
            if unity:
                correction = np.asarray(row['gamma'], float)
                if correction.shape != mask.shape or not np.equal(correction[mask], 1.).all():
                    raise ValueError('Shared-only prediction contains a non-unit gamma.')
            digest = hashlib.sha256(codons[mask].tobytes() + np.flatnonzero(mask).astype('<i8').tobytes()).hexdigest()
            if 'coordinate_hash' in profiles[tid] and profiles[tid]['coordinate_hash'] != digest:
                raise ValueError(f'{tid}: coordinate mismatch across duplicate dataset rows.')
            profiles[tid]['coordinate_hash'] = digest
            seen.add(tid)
    if seen != set(ids):
        raise ValueError('Raw and compact transcript cohorts differ.')
    digest = sha256(compact)
    saved = state.get('outputs', {})
    if saved and (saved['prediction_sha256'] != digest or saved['checkpoint_path'] != runtime['checkpoint_path']):
        raise ValueError('Completed output record no longer matches the saved export.')
    original_root = Path(read_json(root / 'experiment_manifest.json')['output_root'])
    checkpoint = relocate(runtime['checkpoint_path'], root, original_root)
    checkpoint_sha = sha256(checkpoint) if checkpoint.is_file() else None
    if checkpoint_sha and saved.get('checkpoint_sha256') and checkpoint_sha != saved['checkpoint_sha256']:
        raise ValueError('Selected checkpoint checksum mismatch.')
    epoch = re.search(r'epoch=(\d+)', Path(runtime['checkpoint_path']).name)
    return profiles, dict(prediction_path=str(compact), prediction_sha256=digest,
        raw_prediction_path=str(raw), raw_sha256=sha256(raw), runtime_manifest=str(source),
        runtime_manifest_sha256=sha256(source), checkpoint=runtime['checkpoint_path'], checkpoint_sha256=checkpoint_sha,
        checkpoint_locally_verified=checkpoint.is_file(), selected_epoch=int(epoch.group(1)) if epoch else np.nan,
        n_test_transcripts=len(ids), max_mean_one_error=float(checks.absolute_mean_one_deviation.max()), detail='verified best_val_loss export')


def collect(root, manifest, configs, ids, weights):
    profiles, availability = {}, []
    for task in manifest['tasks']:
        directory = root / task['directory']; key = (task['training_seed'], task['arm'], task.get('panel_id',task['N']))
        row = dict(task_id=task['run_id'], array_index=task['array_index'], training_seed=key[0], arm=key[1],
                   N=task['N'], panel_id=task.get('panel_id',f'N{task["N"]:03d}'), task_directory=str(directory), status='no_local_outputs',
                   recorded_status='absent', runtime_config_verified=False, selected_epoch=np.nan)
        state_path = directory / 'execution_status.json'
        try:
            state = read_json(state_path) if state_path.exists() else {}
            row['recorded_status'] = state.get('status', 'absent')
            if state.get('status') == 'failed':
                row.update(status='recorded_failure', detail=state.get('reason', 'See execution_status.json'))
            if list(directory.rglob('events.out.tfevents*')):
                row['status'] = 'logs_only'
            if list((directory / 'checkpoints').rglob('*.ckpt')):
                row['status'] = 'checkpoint_without_export'
            runtime_cfg = directory / 'hydra/.hydra/config.yaml'
            if runtime_cfg.is_file():
                verify_runtime_config(configs[key], yaml.safe_load(runtime_cfg.read_text()), state)
                row.update(runtime_config_verified=True, runtime_config_sha256=sha256(runtime_cfg))
            if list((directory / 'predictions').rglob('prediction_checkpoint_manifest.json')) and state.get('status') == 'completed':
                if not row['runtime_config_verified']:
                    raise ValueError('Actual Hydra training configuration has not been downloaded.')
                values, info = read_export(root, task, configs[key], ids, weights, state)
                row.update(info)
                if values is not None:
                    profiles[key] = values
                    row['status'] = 'validated_predictions'
            if state.get('status') == 'failed':
                row['status'] = 'recorded_failure'
        except (ValueError, KeyError, AssertionError, FileNotFoundError, RuntimeError, OSError) as exc:
            row.update(status='invalid_artifacts', detail=f'{type(exc).__name__}: {exc}')
        availability.append(row)
    return profiles, pd.DataFrame(availability)


def pair_record(left, right, domain):
    if left['coordinate_hash'] != right['coordinate_hash'] or left['length'] != right['length']:
        return dict(n_positions=0, PCC=np.nan, Spearman=np.nan, RMSE=np.nan,
                    variance_a=np.nan, variance_b=np.nan, reason='coordinate_or_length_mismatch')
    return aligned_metrics(left, right, domain)


def evaluation_tables(profiles, ids, sizes):
    adjacent, sensitivity, optimization, diagnostics = [], [], [], []
    seeds = sorted({k[0] for k in profiles})
    for seed in seeds:
        for arm in ARMS:
            for j, (a,b) in enumerate(zip(sizes[:-1], sizes[1:])):
                if (seed,arm,a) not in profiles or (seed,arm,b) not in profiles:
                    continue  # Never bridge over a missing planned size.
                for tid in ids:
                    for domain in DOMAINS:
                        adjacent.append(dict(training_seed=seed, arm=arm, transition=j, N_a=a, N_b=b,
                            transcript_id=tid, domain=domain,
                            **pair_record(profiles[seed,arm,a][tid], profiles[seed,arm,b][tid], domain)))
        for n in sizes:
            for a,b in CONTRASTS:
                if (seed,a,n) not in profiles or (seed,b,n) not in profiles: continue
                for tid in ids:
                    for domain in DOMAINS:
                        sensitivity.append(dict(training_seed=seed, N=n, panel_id=f'N{n:03d}',
                            arm_a=a, arm_b=b, contrast=f'{b}_vs_{a}', transcript_id=tid, domain=domain,
                            **pair_record(profiles[seed,a,n][tid], profiles[seed,b,n][tid], domain)))
    for arm,n in itertools.product(ARMS,sizes):
        ready = sorted(s for s in seeds if (s,arm,n) in profiles)
        for a,b in itertools.combinations(ready,2):
            for tid in ids:
                for domain in DOMAINS:
                    optimization.append(dict(arm=arm,N=n,seed_a=a,seed_b=b,transcript_id=tid,domain=domain,
                        **pair_record(profiles[a,arm,n][tid],profiles[b,arm,n][tid],domain)))
    for (seed,arm,n), values in sorted(profiles.items()):
        for tid in ids:
            for domain,section in DOMAINS.items():
                x=values[tid]['values'][section]
                diagnostics.append(dict(training_seed=seed,arm=arm,N=n,transcript_id=tid,domain=domain,
                    n_positions=len(x), variance=float(x.var()) if len(x) else np.nan,
                    maximum=float(x.max()) if len(x) else np.nan,
                    q99=float(np.quantile(x,.99)) if len(x) else np.nan,
                    near_constant=bool(x.var()<=1e-12) if len(x) else True))
    return (pd.DataFrame(adjacent, columns=SLOT_COLUMNS+METRIC_COLUMNS),
            pd.DataFrame(sensitivity, columns=['training_seed','N','panel_id','arm_a','arm_b','contrast']+METRIC_COLUMNS),
            pd.DataFrame(optimization, columns=['arm','N','seed_a','seed_b']+METRIC_COLUMNS),
            pd.DataFrame(diagnostics, columns=['training_seed','arm','N','transcript_id','domain','n_positions',
                                             'variance','maximum','q99','near_constant']))


def bootstrap_means(matrix, draws, seed):
    """One sampled transcript carries ALL columns (N, policies and seeds)."""
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError('Expected a nonempty finite transcript-by-comparison matrix.')
    rng=np.random.default_rng(seed); samples=np.empty((draws,matrix.shape[1])); digest=hashlib.sha256()
    for start in range(0,draws,32):
        index=rng.integers(0,len(matrix),size=(min(32,draws-start),len(matrix)))
        digest.update(index.astype('<i8').tobytes())
        samples[start:start+len(index)]=matrix[index].mean(axis=1)
    return matrix.mean(axis=0), samples, digest.hexdigest()


def descriptive_summary(frame, groups):
    """Keep an explicit CSV schema when no matched models are available yet."""
    return summarize_metrics(frame, groups).reindex(
        columns=groups+['metric','n_valid','n_excluded','median','mean','q25','q75'])


def adjacent_summaries(frame, ids, n_transitions, draws=5000, seed=20260910):
    summaries, effects, cohorts = [], [], []
    slots = [tuple(row) for row in frame[SLOT_COLUMNS].drop_duplicates().sort_values(SLOT_COLUMNS).itertuples(index=False,name=None)]
    for domain in DOMAINS:
        f=frame[frame.domain==domain]
        if f.empty: continue
        for metric in METRICS:
            pivot=f.pivot(index='transcript_id',columns=SLOT_COLUMNS,values=metric).reindex(index=ids)
            matrix=pivot.reindex(columns=pd.MultiIndex.from_tuples(slots,names=SLOT_COLUMNS)).to_numpy(float)
            finite=np.isfinite(matrix).all(axis=1); cohort=np.asarray(ids)[finite]
            for tid,ok in zip(ids,finite):
                cohorts.append(dict(domain=domain,metric=metric,transcript_id=tid,included=bool(ok),
                    reason='ok' if ok else 'undefined_in_at_least_one_available_N_policy_seed',n_available_slots=len(slots)))
            if not len(cohort): continue
            point, samples, digest=bootstrap_means(matrix[finite],draws,seed)
            common=dict(domain=domain,metric=metric,n_transcripts=len(cohort),n_excluded=len(ids)-len(cohort),
                cohort_hash=transcript_id_hash(cohort),n_available_slots=len(slots),bootstrap_replicates=draws,
                bootstrap_seed=seed,bootstrap_index_sha256=digest)
            low,high=np.quantile(samples,[.025,.975],axis=0,method='linear')
            for j,slot in enumerate(slots):
                summaries.append(dict(zip(SLOT_COLUMNS,slot),**common,estimate=point[j],ci_low=low[j],ci_high=high[j],
                    median_descriptive=float(np.median(matrix[finite,j])),statistic='mean_transcript_metric'))
            indices={slot:j for j,slot in enumerate(slots)}
            for training_seed in sorted({s[0] for s in slots}):
                for a,b in CONTRASTS:
                    matched=[s for s in slots if s[:2]==(training_seed,a) and (s[0],b,*s[2:]) in indices]
                    for sa in matched:
                        sb=(sa[0],b,*sa[2:]);ia,ib=indices[sa],indices[sb]
                        lo,hi=np.quantile(samples[:,ib]-samples[:,ia],[.025,.975],method='linear')
                        effects.append(dict(training_seed=training_seed,arm_a=a,arm_b=b,contrast=f'{b}_minus_{a}',
                            transition=sa[2],N_a=sa[3],N_b=sa[4],**common,
                            estimate=point[ib]-point[ia],ci_low=lo,ci_high=hi,
                            mean_a=point[ia],mean_b=point[ib],matched_transitions=len(matched),
                            all_transitions_matched=len(matched)==n_transitions,statistic='difference_of_means'))
    base=['domain','metric','n_transcripts','n_excluded','cohort_hash','n_available_slots',
          'bootstrap_replicates','bootstrap_seed','bootstrap_index_sha256','estimate','ci_low','ci_high','statistic']
    return (pd.DataFrame(summaries,columns=SLOT_COLUMNS+base+['median_descriptive']),
            pd.DataFrame(effects,columns=['training_seed','arm_a','arm_b','contrast','transition','N_a','N_b']+base+
                         ['mean_a','mean_b','matched_transitions','all_transitions_matched']),
            pd.DataFrame(cohorts,columns=['domain','metric','transcript_id','included','reason','n_available_slots']))


def comparison_availability(availability, sizes, seeds):
    ready=set(availability.loc[availability.status=='validated_predictions',['training_seed','arm','N']].itertuples(index=False,name=None))
    rows=[]
    for seed in seeds:
        for j,(n,m) in enumerate(zip(sizes[:-1],sizes[1:])):
            for a,b in CONTRASTS:
                missing=[f'seed{seed}/{arm}/N{k:03d}' for arm in (a,b) for k in (n,m) if (seed,arm,k) not in ready]
                rows.append(dict(training_seed=seed,transition=j,N_a=n,N_b=m,contrast=f'{b}_minus_{a}',
                    ready=not missing,missing_models=';'.join(missing)))
    return pd.DataFrame(rows)


def training_summary(logs, availability):
    rows=[]
    for task in availability.to_dict('records'):
        f=logs[logs.task_id==task['task_id']]; loss=f[f.tag=='val_loss'].sort_values('epoch')
        row={k:task[k] for k in ('task_id','training_seed','arm','N','status','recorded_status','selected_epoch')}
        row['n_logged_validation_epochs']=int(loss.epoch.nunique())
        if len(loss):
            row.update(last_validation_epoch=float(loss.epoch.max()),
                latest_val_loss=float(loss.iloc[-1].value),
                best_logged_val_loss=float(loss.value.min()))
        for tag in ('val_loss','val_mu_pcc','val_replica_nll','val_mean_ratio','val_nb_alpha_mean'):
            chosen=f[(f.tag==tag)&(f.epoch==task.get('selected_epoch'))].sort_values('wall_time')
            row['selected_'+tag]=float(chosen.iloc[-1].value) if len(chosen) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_availability(table, sizes, seeds, out, dpi):
    mapping={'no_local_outputs':0,'logs_only':1,'checkpoint_without_export':1,'recorded_failure':2,
             'invalid_artifacts':2,'validated_predictions':3}
    colors=['#eeeeee','#F1CF87','#C86464','#80B6A8']
    fig,axes=plt.subplots(1,len(seeds),figsize=(7.15,3.7),squeeze=False)
    for ax,seed in zip(axes.flat,seeds):
        f=table[table.training_seed==seed]
        matrix=f.pivot(index='N',columns='arm',values='status').reindex(index=sizes,columns=ARMS)
        values=np.array([[mapping[x] for x in row] for row in matrix.to_numpy()])
        ax.pcolormesh(np.arange(4)-.5,np.arange(len(sizes)+1)-.5,values,vmin=0,vmax=3,
            cmap=ListedColormap(colors),edgecolors='white',linewidth=.5)
        for i in range(len(sizes)):
            for j in range(3):ax.text(j,i,['--','Logs','Check','Ready'][values[i,j]],ha='center',va='center',fontsize=8)
        ax.set_xticks(range(3),[LABELS[a] for a in ARMS]);ax.set_yticks(range(len(sizes)),sizes);ax.invert_yaxis()
        ax.tick_params(length=0);ax.set_title(f'Seed {seed}: {(values==3).sum()}/{len(sizes)*3}',loc='left')
    axes[0,0].set_ylabel('Datasets per model')
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,
        ['No local outputs','Logs/checkpoints only','Failed/invalid','Validated export'])],loc='lower center',ncol=2,fontsize=8)
    fig.subplots_adjust(left=.08,right=.99,top=.86,bottom=.21,wspace=.3)
    return [save_figure(fig,out/'availability',dpi)]


def format_transition_axis(ax, sizes):
    ax.set_xscale('log')
    ax.set_xticks(sizes[:-1],[rf'${a}\!\to\!{b}$' for a,b in zip(sizes[:-1],sizes[1:])])
    plt.setp(ax.get_xticklabels(),rotation=30,ha='right',rotation_mode='anchor')
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(sizes[0]/1.25,sizes[-2]*1.25)
    ax.set_xlabel('Adjacent dataset counts per model')
    ax.grid(axis='y',alpha=.3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def plot_stability(summary,effects,sizes,out,dpi):
    stems=[]
    for seed in sorted(summary.training_seed.unique()):
        f=summary[(summary.training_seed==seed)&(summary.domain=='full_cds')]
        fig,axes=plt.subplots(1,2,figsize=(7.15,2.95))
        for ax,metric in zip(axes,('PCC','RMSE')):
            for arm in ARMS:
                g=f[(f.arm==arm)&(f.metric==metric)].set_index('transition').reindex(range(len(sizes)-1))
                if g.estimate.notna().any():
                    ax.errorbar(sizes[:-1],g.estimate,yerr=[g.estimate-g.ci_low,g.ci_high-g.estimate],
                        fmt='o-',color=COLORS[arm],ms=4,lw=.9,capsize=2,label=LABELS[arm])
            format_transition_axis(ax,sizes)
            ax.set_title('Adjacent-size stability' if metric=='PCC' else 'Amplitude disagreement',loc='left')
            ax.set_ylabel(f'Mean transcript {metric}')
            if metric=='PCC':ax.set_ylim(min(-.02,float(f[f.metric==metric].ci_low.min())-.03),1.02)
        axes[0].legend(loc='best')
        fig.suptitle(f'Available adjacent comparisons — seed {seed}',fontsize=11)
        counts='; '.join(f'{metric}: n={int(f.loc[f.metric==metric,"n_transcripts"].min())}'
                        for metric in ('PCC','RMSE') if (f.metric==metric).any())
        fig.text(.5,.01,counts+'; missing transitions are not interpolated.',ha='center',fontsize=8)
        fig.subplots_adjust(left=.09,right=.99,bottom=.32,top=.76,wspace=.4)
        stems.append(save_figure(fig,out/f'adjacent_stability_seed{seed}',dpi))
    for seed in sorted(effects.training_seed.unique()):
        f=effects[(effects.training_seed==seed)&(effects.domain=='full_cds')]
        fig,axes=plt.subplots(1,2,figsize=(7.15,3.05))
        for ax,metric in zip(axes,('PCC','RMSE')):
            for j,(a,b) in enumerate(CONTRASTS):
                g=f[(f.contrast==f'{b}_minus_{a}')&(f.metric==metric)].set_index('transition').reindex(range(len(sizes)-1))
                if g.estimate.notna().any():
                    ax.errorbar(np.asarray(sizes[:-1])*np.exp((j-1)*.035),g.estimate,
                        yerr=[g.estimate-g.ci_low,g.ci_high-g.estimate],fmt=['o-','s-','^-'][j],
                        color=COLORS[b],ms=4,lw=.8,capsize=2,label=f'{LABELS[b]} minus {LABELS[a]}')
            format_transition_axis(ax,sizes);ax.axhline(0,color='.4',ls='--',lw=.8)
            ax.set_title(f'Change in mean {metric}',loc='left')
            ax.set_ylabel('Policy difference')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
        fig.suptitle(f'Matched policy effects, seed {seed}',fontsize=11)
        fig.subplots_adjust(left=.09,right=.99,bottom=.36,top=.77,wspace=.4)
        stems.append(save_figure(fig,out/f'adjacent_policy_effects_seed{seed}',dpi))
    return stems


def plot_learning(logs,availability,out,dpi):
    stems=[]
    for seed in sorted(logs.training_seed.unique()):
        for tag,label,stem in [('val_loss','Validation objective','validation_objective'),
                               ('val_mu_pcc','Observed-profile PCC','validation_mu_pcc')]:
            f=logs[(logs.training_seed==seed)&(logs.tag==tag)&logs.epoch.notna()]
            if f.empty:continue
            panels=sorted(f.panel_id.unique());ncols=min(2,len(panels));nrows=math.ceil(len(panels)/ncols)
            fig,axes=plt.subplots(nrows,ncols,figsize=(7.15,1.8*nrows+1.05),squeeze=False)
            for ax,panel in zip(axes.flat,panels):
                for arm in ARMS:
                    g=f[(f.panel_id==panel)&(f.arm==arm)].sort_values('epoch')
                    if g.empty:continue
                    ax.plot(g.epoch+1,g.value,'o-',ms=2.3,color=COLORS[arm],lw=1,label=LABELS[arm])
                    task=availability[(availability.training_seed==seed)&(availability.arm==arm)&(availability.panel_id==panel)].iloc[0]
                    chosen=g[g.epoch==task.selected_epoch]
                    if len(chosen):ax.plot(chosen.epoch+1,chosen.value,'o',ms=6,mfc='white',mec=COLORS[arm])
                ax.set_title(f'N={int(panel[1:])}',loc='left');ax.set_xlabel('Epoch (one-based)');ax.set_ylabel(label)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True));ax.grid(alpha=.25)
            for ax in list(axes.flat)[len(panels):]:ax.set_visible(False)
            fig.legend(handles=[Line2D([],[],color=COLORS[a],lw=1,label=LABELS[a]) for a in ARMS if a in set(f.arm)],
                       loc='lower center',ncol=3)
            fig.suptitle(f'Partial training logs, seed {seed}',fontsize=12)
            fig.subplots_adjust(left=.095,right=.99,bottom=.29 if nrows==1 else .14,top=.80 if nrows==1 else .9,
                                hspace=.65,wspace=.3)
            stems.append(save_figure(fig,out/f'{stem}_seed{seed}',dpi))
    return stems


def plot_reference(weights,out,dpi):
    rows=[]
    for (n,arm),f in weights.groupby(['N','arm']):
        p=f.pi.to_numpy(float)
        rows.append(dict(N=n,arm=arm,N_ref=1/(p@p),max_pi=p.max(),min_pi=p.min(),
            weighted_mean_rank=p@f.global_rank,total_variation_from_equal=.5*np.abs(p-1/n).sum()))
    table=pd.DataFrame(rows);table.to_csv(out.parent/'reference_concentration.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.85))
    for arm in ARMS:
        g=table[table.arm==arm].sort_values('N')
        axes[0].plot(g.N,g.N_ref/g.N,['-','--',':'][ARMS.index(arm)],color=COLORS[arm],label=LABELS[arm],lw=1.4)
        axes[1].plot(g.N,g.weighted_mean_rank,'o-',ms=3,color=COLORS[arm],lw=1)
    for ax in axes:
        ax.set_xscale('log');ax.set_xticks(sorted(table.N.unique()),sorted(table.N.unique()));ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Datasets per model');ax.grid(alpha=.25)
    axes[0].set_title('Weight concentration',loc='left');axes[0].set_ylabel('$N_{ref}/N$');axes[0].legend(loc='lower left')
    axes[1].set_title('Reference quality composition',loc='left');axes[1].set_ylabel('Weighted global rank (1 = best)')
    fig.text(.5,.02,'Frozen design, not performance. Ranked and reversed concentration is identical.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.085,right=.99,bottom=.28,top=.85,wspace=.45)
    return table,[save_figure(fig,out/'reference_design',dpi)]


def write_report(out,manifest,tables,limitations,command,figures,draws,seed):
    a=tables['training_availability'];ready=a[a.status=='validated_predictions']
    parts=['<!doctype html><html><head><meta charset="utf-8"><title>Cumulative reference directionality</title>',
        '<style>body{font:16px/1.55 system-ui;max-width:1100px;margin:35px auto;padding:0 22px;color:#222}table{border-collapse:collapse;font-size:13px}td,th{padding:5px 9px;border-bottom:1px solid #ddd;text-align:right}.table{overflow:auto}img{max-width:100%}code,pre{background:#f3f3f3;padding:6px;overflow:auto}.note{border-left:4px solid #D55E00;background:#fff8ef;padding:12px}</style></head><body>',
        '<h1>Cumulative equal / ranked / reverse experiment</h1>',
        f'<p class="note">{len(ready)}/{len(a)} validated completed exports locally. Running/completed states describe downloaded records, not a live scheduler query. Checkpoints or directories alone are not completed test results.</p>',
        '<h2>What can currently be compared?</h2>',
        '<p>Adjacent-size agreement is computed only when both models have valid best-val-loss exports. A policy effect requires the same two sizes under both policies at the same training seed. Missing models are never replaced by another seed, another subset design, an intermediate checkpoint, or a zero effect.</p>']
    for title,key in [('Available exports','training_availability'),('Adjacent-size estimates','adjacent_summary'),
                      ('Matched policy effects','adjacent_policy_effects'),('Same-N cross-policy sensitivity','same_N_policy_summary'),
                      ('Same-N optimization variability','same_N_seed_summary'),('Profile amplitudes / collapse','profile_diagnostic_summary'),
                      ('Training progress and exported checkpoint context','training_progress'),('Per-N split identity','split_summary')]:
        table=tables[key]
        if key=='training_availability':table=table[['training_seed','arm','N','status','recorded_status']]
        if key=='adjacent_summary':table=table[(table.domain=='full_cds')&(table.metric=='PCC')]
        if key=='adjacent_policy_effects':table=table[(table.domain=='full_cds')&(table.metric=='PCC')]
        parts.extend([f'<h2>{title}</h2>',f'<p><a href="{key}.csv">Exact table</a></p>',
            '<div class="table">'+(table.to_html(index=False,float_format=lambda x:f'{x:.6g}') if len(table) else '<p>Not estimable from the currently available matched exports.</p>')+'</div>'])
    parts.extend(['<h2>Interpretation and uncertainty</h2>',
        '<p>The primary statistic is the mean full-CDS transcript PCC for adjacent models (2→5, 5→10, etc.), not convergence to the N=114 model. RMSE and Spearman are saved separately. Same-N cross-policy PCC measures reference sensitivity, not improved cross-size agreement.</p>',
        f'<p>For each domain/metric, the primary finite cohort is common across ALL currently available adjacent comparisons, policies and training seeds. It is recomputed as exports arrive; a partial curve may therefore change cohort. Each of {draws:,} bootstrap draws (seed {seed}) samples transcripts with every available N/policy/seed entry attached. Policy differences are recomputed in the same draw. Intervals are pointwise, conditional on fitted models and selected collections, not uncertainty over retraining or independent compendia. Training seeds remain separate; no primary ensemble or seed-pooling is performed.</p>',
        '<p>The complete valid CDS and interior-20 domain retain original amplitudes. No smoothing, position selection by observed zeros, clipping of peaks, length truncation, or interior renormalization is performed. Undefined correlations are excluded explicitly, not replaced by zero. Exclusion records and cohort membership are saved.</p>',
        '<p>Learning curves come from real validation logs, not the dummy mu/target fields of sequence-only exports. Hollow markers identify exported checkpoints; other points are unfinished-training context. Best-so-far validation minima from different training durations are not a matched final-policy comparison. Validation sets differ across N, so their loss levels are not directly comparable across N.</p>',
        f'<p>Ten-component ranking SHA256: <code>{html.escape(manifest["ranking"]["sha256"])}</code>; complete-table R={manifest["ranking"]["R"]:g}, rank 1 best. Ranked q=(R-r+1)/R is normalized over each complete prefix. Reverse reassigns the same prefix q multiset in reverse global-rank order; equal is uniform. At small N the selected q values are almost equal. Pi affects gamma centering, not the local loss weights w_dt.</p>',
        '<p>Nested prefixes add progressively poorer-ranked datasets and can split source families. They are dependent stability comparisons, not source-disjoint panel replications. Higher reproducibility does not establish biological accuracy; null or negative policy effects must remain visible.</p>',
        '<h2>Availability, exclusions and provenance</h2><p><a href="comparison_availability.csv">Exact missing models per policy contrast</a> · <a href="metric_exclusions.csv">Metric exclusions</a> · <a href="bootstrap_cohorts.csv">Common finite cohorts</a> · <a href="verified_input_hashes.csv">Verified input/code hashes</a></p>',
        '<ul>'+''.join(f'<li>{html.escape(s)}</li>' for s in limitations)+'</ul><h2>Figures</h2>'])
    for stem in figures:
        rel=stem.relative_to(out)
        parts.append(f'<p><a href="{rel}.pdf">PDF</a> · <a href="{rel}.svg">SVG</a></p><img src="{rel}.png" alt="{html.escape(stem.name)}">')
    parts.extend(['<h2>Regenerate</h2><pre>'+html.escape(command)+'</pre></body></html>'])
    (out/'analysis_report.html').write_text('\n'.join(parts))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--bootstrap-replicates',type=int,default=5000)
    parser.add_argument('--bootstrap-seed',type=int,default=20260910)
    parser.add_argument('--dpi',type=int,default=600)
    parser.add_argument('--no-tex',action='store_true')
    parser.add_argument('--skip-training-logs',action='store_true')
    args=parser.parse_args(argv)
    if args.bootstrap_replicates<2 or args.dpi<50:parser.error('Use at least two bootstrap draws and dpi >=50.')
    root=args.run_root.expanduser().resolve();out=(args.output_dir or root/'analysis_directionality').resolve()
    if out==root or out in root.parents or any(out.is_relative_to(root/p) for p in ('runs','frozen_inputs','code_snapshot','resolved_configs')):
        parser.error('Analysis output must be a separate directory, not a training/input directory.')
    out.mkdir(parents=True,exist_ok=True);figures=out/'figures';figures.mkdir(exist_ok=True)
    manifest,configs,ids,weights,folds,hashes,limitations=audit_design(root)
    print('Frozen design verified; inspecting available exports.',flush=True)
    profiles,availability=collect(root,manifest,configs,ids,weights)
    print(availability.groupby(['training_seed','arm','status']).size().to_string(),flush=True)
    adjacent,sensitivity,optimization,diagnostics=evaluation_tables(profiles,ids,manifest['sizes'])
    summary,effects,cohorts=adjacent_summaries(adjacent,ids,len(manifest['sizes'])-1,args.bootstrap_replicates,args.bootstrap_seed)
    logs=collect_logs(availability[availability.runtime_config_verified & availability.status.ne('invalid_artifacts')]) if not args.skip_training_logs else collect_logs(availability.iloc[:0])
    progress=training_summary(logs,availability)
    diagnostic_summary=(diagnostics.groupby(['training_seed','arm','N','domain'],as_index=False).agg(
        n_transcripts=('transcript_id','size'),median_variance=('variance','median'),
        near_constant_fraction=('near_constant','mean'),median_maximum=('maximum','median'),
        maximum_profile_value=('maximum','max')))
    tables=dict(training_availability=availability,reference_weights=weights,split_summary=folds,
        adjacent_transcript_metrics=adjacent,adjacent_summary=summary,adjacent_policy_effects=effects,
        adjacent_descriptive_summary=descriptive_summary(adjacent,SLOT_COLUMNS+['domain']),
        bootstrap_cohorts=cohorts,comparison_availability=comparison_availability(availability,manifest['sizes'],manifest['training_seeds']),
        same_N_policy_transcript_metrics=sensitivity,
        same_N_policy_summary=descriptive_summary(sensitivity,['training_seed','N','contrast','domain']),
        same_N_seed_transcript_metrics=optimization,
        same_N_seed_summary=descriptive_summary(optimization,['arm','N','seed_a','seed_b','domain']),
        profile_diagnostics=diagnostics,profile_diagnostic_summary=diagnostic_summary,
        training_scalars=logs,training_progress=progress,verified_input_hashes=pd.DataFrame(hashes))
    exclusions=pd.concat([f.assign(comparison_family=name) for name,f in
        [('adjacent',adjacent),('same_N_policy',sensitivity),('same_N_seed',optimization)]],ignore_index=True)
    tables['metric_exclusions']=exclusions[exclusions.reason.ne('ok')]
    for name,frame in tables.items():frame.to_csv(out/f'{name}.csv',index=False)
    style=publication_rc()
    if args.no_tex:style.update({'text.usetex':False,'font.serif':['DejaVu Serif'],'mathtext.fontset':'cm'})
    style.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':12,'xtick.labelsize':9,
                  'ytick.labelsize':9,'legend.fontsize':9})
    current=[]
    with plt.rc_context(style):
        current+=plot_availability(availability,manifest['sizes'],manifest['training_seeds'],figures,args.dpi)
        concentration,stems=plot_reference(weights,figures,args.dpi);current+=stems
        current+=plot_stability(summary,effects,manifest['sizes'],figures,args.dpi)
        current+=plot_learning(logs,availability,figures,args.dpi)
        current+=plot_sensitivity(sensitivity,figures,args.dpi)
    command=shlex.join([sys.executable,str(Path(__file__).resolve()),'--run-root',str(root),'--output-dir',str(out),
        '--bootstrap-replicates',str(args.bootstrap_replicates),'--bootstrap-seed',str(args.bootstrap_seed),'--dpi',str(args.dpi)]
        +(['--no-tex'] if args.no_tex else [])+(['--skip-training-logs'] if args.skip_training_logs else []))
    write_report(out,manifest,tables,limitations,command,current,args.bootstrap_replicates,args.bootstrap_seed)
    (out/'regenerate.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+command+'\n')
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='complete_local_exports' if len(profiles)==len(manifest['tasks']) else 'partial_local_exports',
        experiment_root=str(root),experiment_manifest_sha256=sha256(root/'experiment_manifest.json'),script_sha256=sha256(Path(__file__)),
        validated_models=len(profiles),planned_models=len(manifest['tasks']),common_test_transcripts=len(ids),
        common_test_hash=transcript_id_hash(ids),bootstrap_replicates=args.bootstrap_replicates,bootstrap_seed=args.bootstrap_seed,
        adjacent_policy_effect_rows=len(effects),training_or_inference_launched=False,input_mutations=False,
        limitations=limitations,tex_used=bool(style['text.usetex']),current_figures=[str(p.relative_to(out)) for p in current],
        command=command,output_sha256={str(p.relative_to(out)):sha256(p) for p in out.rglob('*')
            if p.is_file() and p.name!='analysis_manifest.json'})
    (out/'analysis_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Analysis: {out / "analysis_report.html"}\nValidated {len(profiles)}/{len(manifest["tasks"])} models; no training/inference launched.',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
