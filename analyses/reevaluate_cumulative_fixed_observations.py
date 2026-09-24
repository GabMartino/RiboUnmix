#!/usr/bin/env python3
"""Re-evaluate frozen cumulative checkpoints on identical real test observations.

Uses the original test IDs intersected with usable observations in ALL configured
datasets, including datasets whose training runs have not finished. This fixes
the cohort before looking at model performance and as new results arrive.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import html
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
from torch.utils.data import DataLoader
import yaml

from run_cumulative_stability import DEFAULT_OUTPUT, object_sha256, sha256, write_json
from Utils.cumulative_dataset_pcc_report import write_dataset_mu_report
from Utils.reliability_references import transcript_id_hash
from analyses.analyze_cumulative_reference_weight_stability import ARMS, COLORS, STYLE, table_html
from analyses.analyze_real_panel_posthoc_robustness_streaming import _make_frozen_model_and_encoder, _autocast_context
from analyses.paths import artifact_directory
from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDataset import RiboUnmixMultiDataset
from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import TranscriptGroupedMultiDatasetBatchSampler


def local_path(value, root, original_root):
    path = Path(value)
    if path.is_relative_to(original_root):
        return root / path.relative_to(original_root)
    original_project = original_root.parent.parent
    if path.is_relative_to(original_project):
        return ROOT / path.relative_to(original_project)
    return path if path.is_absolute() else ROOT / path


def local_config(task, root, manifest):
    original = Path(manifest['output_root'])
    path = local_path(task['config_path'], root, original)
    if sha256(path) != task['config_sha256']:
        raise ValueError(f'Frozen training config changed: {path}')
    cfg = yaml.safe_load(path.read_text())
    for key in ('sequences_path',):
        cfg['paths'][key] = str(local_path(cfg['paths'][key], root, original))
    for key, value in cfg['paths']['encodings'].items():
        cfg['paths']['encodings'][key] = str(local_path(value, root, original))
    cfg['dataset_config']['dataset_path'] = {d: str(local_path(p, root, original))
        for d, p in cfg['dataset_config']['dataset_path'].items()}
    cfg['data']['dataset_quality_ranking']['path'] = str(local_path(cfg['data']['dataset_quality_ranking']['path'], root, original))
    cfg['data']['reliability_reference_manifest'] = str(local_path(cfg['data']['reliability_reference_manifest'], root, original))
    return OmegaConf.create(cfg)


def audit_folds(manifest):
    rows, transitions = [], []
    sizes = manifest['sizes']
    common_test = set(manifest['source_folds'][str(sizes[0])]['test_ids'])
    for n in sizes:
        fold = manifest['source_folds'][str(n)]
        train, validation, test = (set(fold[k]) for k in ('train_ids', 'validation_ids', 'test_ids'))
        if train & validation or train & test or validation & test or test != common_test:
            raise ValueError(f'N={n} violates within-model disjointness or common-test identity.')
        rows.append(dict(N=n, n_train=len(train), n_validation=len(validation), n_test=len(test),
            train_validation_overlap=0, train_test_overlap=0, validation_test_overlap=0))
    for a, b in zip(sizes[:-1], sizes[1:]):
        fa, fb = manifest['source_folds'][str(a)], manifest['source_folds'][str(b)]
        ta, tb = set(fa['train_ids']), set(fb['train_ids'])
        va, vb = set(fa['validation_ids']), set(fb['validation_ids'])
        transitions.append(dict(N_a=a, N_b=b, retained_training=len(ta & tb),
            training_removed=len(ta - tb), old_train_now_validation=len(ta & vb),
            old_validation_now_train=len(va & tb), common_validation=len(va & vb)))
    return pd.DataFrame(rows), pd.DataFrame(transitions)


def prepare_cohort(root, manifest, out):
    widest = max(manifest['tasks'], key=lambda task: task['N'])
    cfg = local_config(widest, root, manifest)
    dataset_paths = dict(cfg.dataset_config.dataset_path)
    original_test = set(manifest['source_folds'][str(manifest['sizes'][0])]['test_ids'])
    cohort = original_test.copy()
    coverage = []
    for name, path in dataset_paths.items():
        frame = pd.read_parquet(path, columns=['id', 'weight', 'read_density', 'coverage'])
        positive = (frame.weight > 0) & (frame.read_density > 0) & (frame.coverage > 0)
        supported = set(frame.loc[positive, 'id'].astype(str))
        cohort &= supported
        coverage.append(dict(dataset_id=name, original_test_observations=len(original_test & supported)))
    ids = sorted(cohort)
    if not ids:
        raise ValueError('No common observed test transcripts across the configured dataset universe.')
    payload = dict(experiment_manifest_sha256=sha256(root / 'experiment_manifest.json'),
        transcript_ids=ids, transcript_id_hash=transcript_id_hash(ids), dataset_universe=list(dataset_paths),
        n_transcripts=len(ids), original_test_count=len(original_test),
        cohort_rule='Original common test IDs intersected with positive-weight, positive-density, positive-coverage observations in ALL configured datasets.',
        checkpoint_selection='Original best validation loss; no selection or tuning on re-evaluation results.',
        evaluation_weighting='Equal transcript means within each dataset; equal dataset means for aggregate curves.',
        limitation=('Complete observation coverage selects broadly observed transcripts; train/validation/test membership is fixed across N.'
            if manifest['setup_request'].get('split_protocol') == 'fixed_complete_transcripts' else
            'Complete observation coverage selects broadly observed transcripts; existing training-fold differences remain.'))
    path = out / 'cohort_manifest.json'
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError('The fixed evaluation cohort or source experiment changed; use a new output directory.')
    write_json(path, payload)
    pd.DataFrame(coverage).to_csv(out / 'original_test_coverage.csv', index=False)
    return cfg, ids, payload


def load_observations(cfg, ids, datasets):
    sequence = pd.read_parquet(cfg.paths.sequences_path,
        columns=['transcript_id', 'codons', 'conserved_stalling_sites'], filters=[('transcript_id', 'in', ids)])
    sequence = sequence.set_index('transcript_id').loc[ids]
    replicas, profiles = defaultdict(dict), defaultdict(dict)
    digests = {}
    for name in datasets:
        frame = pd.read_parquet(cfg.dataset_config.dataset_path[name], columns=['id', 'ribo_cds_replicas'], filters=[('id', 'in', ids)])
        frame = frame.set_index('id').loc[ids]
        digest = hashlib.sha256()
        for tid, cell in frame.ribo_cds_replicas.items():
            array = np.asarray(np.stack(cell), dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != len(sequence.at[tid, 'codons']) or not np.isfinite(array).all() or (array < 0).any() or array.sum() <= 0:
                raise ValueError(f'Invalid observed replicas for {name}/{tid}.')
            replicas[tid][name] = array
            profiles[tid][name] = array.mean(axis=0)
            digest.update(tid.encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
        digests[name] = digest.hexdigest()
    return sequence, profiles, replicas, digests


def observed_dataset(sequence, profiles, replicas, datasets, encoder):
    ids = sequence.index.tolist()
    data = dict(transcript_id=np.asarray(ids, dtype=str), ref=sequence.codons.to_numpy(), sequence_representation='codon_tokens',
        css=sequence.conserved_stalling_sites.tolist(),
        ribo_profiles={t: {d: profiles[t][d] for d in datasets} for t in ids},
        ribo_replicas={t: {d: replicas[t][d] for d in datasets} for t in ids},
        sample_weights={t: {d: 1. for d in datasets} for t in ids},
        dataset_quality_ranks={d: np.nan for d in datasets}, dataset_quality_weights={d: 1. for d in datasets})
    return RiboUnmixMultiDataset(data=data, lengths=sequence.codons.map(len).to_numpy(), transcripts_ids=ids,
        nt_encoding=encoder.nt_encoding, codon_encoding=encoder.codon_encoding, codon_to_aa_encoding=encoder.codon_to_aa_encoding,
        aa_encoding=encoder.aa_encoding, datasets_encoding=encoder.dataset_encoding,
        additional_sequence_features=encoder.additional_sequence_features)


def evaluate_task(task, cfg, checkpoint, sequence, profiles, replicas, device, max_tokens):
    module, encoder, names = _make_frozen_model_and_encoder(cfg, checkpoint=checkpoint, device=device)
    dataset = observed_dataset(sequence, profiles, replicas, names, encoder)
    sampler = TranscriptGroupedMultiDatasetBatchSampler(flat_transcript_ids=dataset.flat_transcript_ids,
        flat_dataset_ids=dataset.flat_dataset_ids, considered_transcript_ids=sequence.index.tolist(), lengths=dataset.flat_lengths,
        batch_size=8, seed=task['training_seed'], drop_last=False, sort_by_length=True, require_multidataset=False,
        minimum_distinct_datasets=1, shuffle_batches=False, num_replicas=1, rank=0,
        execution_microbatch_max_transcript_groups=8, execution_microbatch_max_pair_rows=256,
        execution_microbatch_max_padded_codon_tokens=max_tokens)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collate_fn, num_workers=0)
    inverse = {int(v): k for k, v in encoder.dataset_encoding.items()}
    rows = []
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            with _autocast_context(cfg, device):
                result = module.forward_batch(move_data_to_device(batch, device))
            mask, target = result['mask'], result['target'].float()
            mu, bio = result['mu'].float(), result['extras']['L_bio'].float()
            mu_pcc = module._pearson_per_sample(module._apply_pcc_floor(mu), target, mask).cpu().numpy()
            bio_pcc = module._pearson_per_sample(bio, target, mask).cpu().numpy()
            rmse = (((mu - target).square() * mask).sum(1) / mask.sum(1)).sqrt().cpu().numpy()
            lengths = result['lengths'].tolist()
            for index, (tid, did) in enumerate(zip(result['ids'], result['dataset_ids'].tolist())):
                rows.append(dict(transcript_id=str(tid), dataset_id=inverse[did], mu_pcc=float(mu_pcc[index]),
                    L_bio_pcc=float(bio_pcc[index]), correction_gain=float(mu_pcc[index] - bio_pcc[index]),
                    mu_rmse=float(rmse[index]), n_codons=int(lengths[index])))
            if (batch_index + 1) % 50 == 0:
                print(f'  {task["run_id"]}: {len(rows):,}/{len(dataset):,} pairs, {time.monotonic()-started:.0f}s', flush=True)
            del result, mu, bio, target, mask
    frame = pd.DataFrame(rows)
    expected = pd.MultiIndex.from_product([sequence.index, names])
    actual = pd.MultiIndex.from_frame(frame[['transcript_id', 'dataset_id']])
    if actual.has_duplicates or set(actual) != set(expected):
        raise ValueError('Re-evaluation did not cover the exact common transcript × dataset grid.')
    del module, dataset, loader
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return frame


def evaluation_signature():
    """Invalidate numerical caches when inference changes, independently of HTML edits."""
    functions = (local_config, load_observations, observed_dataset, evaluate_task)
    model_sources = [
        'Models/RiboUnmixLightningModule.py',
        'Models/RiboUnmixModel/RiboUnmixModel.py',
        'Models/RiboUnmixModel/DatasetBiasSubmodel.py',
        'Dataloaders/RiboUnmixMultiDataset/RiboUnmixMultiDataset.py',
        'Dataloaders/RiboUnmixMultiDataset/RiboUnmixMultiDatasetDataModule.py',
        'analyses/analyze_real_panel_posthoc_robustness_streaming.py',
    ]
    return object_sha256(dict(functions=[inspect.getsource(f) for f in functions],
        model_sources={name: sha256(ROOT / name) for name in model_sources}))


def write_report(root, out, manifest, cohort, task_results, fold_table, transition_table):
    per_dataset = []
    summaries = []
    for task in manifest['tasks']:
        frame = task_results.get(task['run_id'])
        means = frame.groupby('dataset_id')[['mu_pcc', 'L_bio_pcc', 'correction_gain']].mean() if frame is not None else None
        for name in task['datasets']:
            values = means.loc[name].to_dict() if means is not None else dict(mu_pcc=np.nan, L_bio_pcc=np.nan, correction_gain=np.nan)
            per_dataset.append(dict(task_id=task['run_id'], N=task['N'], arm=task['arm'], dataset_id=name,
                selected_epoch=frame.attrs['epoch'] if frame is not None else np.nan,
                status='validated_predictions' if frame is not None else 'checkpoint_without_export',
                n_transcripts=cohort['n_transcripts'] if frame is not None else 0, **values))
        if means is not None:
            for scope, names in [('all_selected', task['datasets']), ('best2_datasets', manifest['tasks'][0]['datasets'])]:
                summaries.append(dict(N=task['N'], arm=task['arm'], scope=scope, **means.loc[names].mean().to_dict()))
    per_dataset = pd.DataFrame(per_dataset)
    per_dataset.to_csv(out / 'observed_fit_per_dataset.csv', index=False)
    summary = pd.DataFrame(summaries, columns=['N', 'arm', 'scope', 'mu_pcc', 'L_bio_pcc', 'correction_gain'])
    summary.to_csv(out / 'observed_fit_summary.csv', index=False)
    note = f'the same {cohort["n_transcripts"]} held-out transcripts and real observations are used across every dataset, N and policy.'
    individual = write_dataset_mu_report(per_dataset, pd.read_csv(root / 'reference_weights.csv'), manifest['sizes'],
        ARMS, COLORS, out, evaluation_label='Held-out test', cohort_note=note)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, scope in enumerate(['all_selected', 'best2_datasets']):
        for col, metric in enumerate(['mu_pcc', 'L_bio_pcc']):
            ax = axes[row, col]
            for arm in ARMS:
                values = summary[(summary.arm == arm) & (summary.scope == scope)].set_index('N')[metric].reindex(manifest['sizes'])
                ax.plot(range(len(values)), values, 'o--' if 'reverse' in arm else 'o-', color=COLORS[arm], label=arm)
            ax.set(xticks=range(len(manifest['sizes'])), xticklabels=manifest['sizes'], xlabel='Training datasets N',
                ylabel='PCC against observed profiles', title=f'{scope}: {"μ" if col == 0 else "L_bio"}', ylim=(0, 1))
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'{cohort["n_transcripts"]} fixed held-out transcripts · same real targets and masks at every evaluation')
    fig.tight_layout()
    for suffix in ('svg', 'pdf'):
        fig.savefig(out / f'fixed_observed_fit.{suffix}')
    plt.close(fig)
    equal = summary[(summary.arm == 'equal') & (summary.scope == 'best2_datasets')].sort_values('N')
    interpretation = 'At least two completed sizes are needed to assess an across-N trend.'
    if len(equal) >= 2:
        first, last = equal.iloc[0], equal.iloc[-1]
        interpretation = (f'On the fixed best-two observations under equal weighting, from N={int(first.N)} to N={int(last.N)}, '
            f'μ PCC changes {first.mu_pcc:.3f}→{last.mu_pcc:.3f} while L_bio PCC changes {first.L_bio_pcc:.3f}→{last.L_bio_pcc:.3f}, '
            'showing that preserved observation fit can coexist with changing agreement of the shared component.')
    fold_values = list(manifest['source_folds'].values())
    fixed_training = all(fold['train_ids'] == fold_values[0]['train_ids'] and
                         fold['validation_ids'] == fold_values[0]['validation_ids'] for fold in fold_values)
    audit_text = ('Training, validation and test memberships are identical at every N. Every fold is disjoint, and each dataset has the same complete transcript cohort. This design controls transcript membership while adding datasets.' if fixed_training else
        'Within each model, train/validation/test are disjoint; policies at a given N reuse the same inputs. The legacy cumulative setup nevertheless reassigned training and validation membership across N. These test metrics repair the evaluation comparison; they cannot remove that training-design confound from existing checkpoints. Fresh fixed-cohort training is required to isolate adding datasets while holding transcript membership constant.')
    audit_link = html.escape(os.path.relpath(ROOT / 'Docs/cumulative_cohort_audit_and_correction.html', out), quote=True)
    stability_link = html.escape(os.path.relpath(root / 'analysis/analysis_report.html', out), quote=True)
    refresh_output = f' --output-dir {html.escape(str(out))}' if out != root / 'fixed_observed_evaluation' else ''
    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Fixed-observation cumulative evaluation</title><style>{STYLE}</style></head><body>
<h1>Cumulative evaluation on identical held-out observations</h1>
<p>Snapshot generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} · training seed {manifest['training_seeds'][0]}</p>
<p class="note"><b>{len(task_results)}/{len(manifest['tasks'])} checkpoints re-evaluated</b> on {cohort['n_transcripts']} transcripts observed in all {len(cohort['dataset_universe'])} configured datasets; the cohort is fixed using the full dataset universe, including unfinished models.</p>
<p>Every point uses the original best-validation-loss checkpoint. No model is trained or selected using these test results. Targets are arithmetic means of real replicas; μ and L_bio use identical CDS masks and the saved PCC convention (zero for near-constant profiles). Each trained dataset has exactly the same transcript IDs at every N and policy.</p>
<h2>Observed fit on the fixed cohort</h2><img src="fixed_observed_fit.svg" alt="Mu and Lbio PCC on a common observed test cohort">
<p class="note">{interpretation}</p>
<p>The bottom row fixes both transcripts and the best-two dataset identities; the top row fixes transcripts but adds dataset targets as N grows, so its mean still changes composition.</p>
<p>PCC removes transcript-wide scale, so the μ-versus-L_bio gap measures the fitted correction's contribution to shape agreement with these observations; it does not establish recovery of purely technical bias or biological truth. One training seed and unfinished larger sizes limit the ranking claim.</p>
{individual}
<h2>Training and validation audit of these existing models</h2>{table_html(fold_table)}
{table_html(transition_table)}
<p>{audit_text}</p>
<p><a href="{audit_link}">Why the old comparison was confounded and how the corrected experiment works</a>.</p>
<p>The complete observation intersection favors broadly observed transcripts; report it alongside coverage, rather than claiming representative performance over every transcript.</p>
<p><a href="cohort_manifest.json">Exact cohort and evaluation rules</a> · <a href="original_test_coverage.csv">Original test coverage per dataset</a> · <a href="observed_fit_summary.csv">Mean metrics</a> · <a href="analysis_manifest.json">Snapshot provenance</a> · <a href="{stability_link}">Training diagnostics and L_bio stability</a></p>
<h2>Refresh</h2><pre>python analyses/reevaluate_cumulative_fixed_observations.py --experiment-root {html.escape(str(root))}{refresh_output}</pre>
</body></html>'''
    (out / 'analysis_report.html').write_text(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--output-dir', type=Path, help='Optional new directory for a separately frozen evaluation cohort.')
    parser.add_argument('--task-index', type=int, action='append', help='Optional subset of array indices; default evaluates all completed checkpoints.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-padded-tokens', type=int, default=32768)
    args = parser.parse_args(argv)
    root = args.experiment_root.expanduser().resolve()
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    out = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else artifact_directory("real_data", root, "fixed_observed_evaluation")
    )
    out.mkdir(parents=True, exist_ok=True)
    fold_table, transitions = audit_folds(manifest)
    fold_table.to_csv(out / 'split_audit.csv', index=False); transitions.to_csv(out / 'training_role_changes.csv', index=False)
    cfg, ids, cohort = prepare_cohort(root, manifest, out)
    print(f'Fixed observed test cohort: {len(ids)} transcripts × {len(cohort["dataset_universe"])} datasets.', flush=True)
    tasks, states = [], {}
    for task in manifest['tasks']:
        path = root / task['directory'] / 'execution_status.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get('status') == 'completed':
            states[task['run_id']] = state
            if args.task_index is None or task['array_index'] in args.task_index:
                tasks.append(task)
    if not tasks:
        raise ValueError('No selected completed checkpoints are available for re-evaluation.')
    needed = sorted({d for t in tasks for d in t['datasets']})
    sequence, profiles, replicas, observation_hashes = load_observations(cfg, ids, needed)
    signature = evaluation_signature()
    results = {}
    for task in manifest['tasks']:
        cache = out / f'{task["run_id"]}.parquet'
        meta = out / f'{task["run_id"]}.json'
        if cache.exists() and meta.exists():
            record = json.loads(meta.read_text())
            if record['cohort_hash'] != cohort['transcript_id_hash'] or record['metrics_sha256'] != sha256(cache):
                raise ValueError(f'Invalid cached evaluation: {cache}')
            results[task['run_id']] = pd.read_parquet(cache)
            results[task['run_id']].attrs['epoch'] = record['epoch']
    for task in tasks:
        state = states[task['run_id']]['outputs']
        checkpoint = local_path(state['checkpoint_path'], root, Path(manifest['output_root']))
        if state['checkpoint_variant'] != 'best_val_loss' or sha256(checkpoint) != state['checkpoint_sha256']:
            raise ValueError(f'Selected checkpoint mismatch: {checkpoint}')
        expected = dict(checkpoint_sha256=state['checkpoint_sha256'], cohort_hash=cohort['transcript_id_hash'],
            config_sha256=task['config_sha256'], observation_hashes={d: observation_hashes[d] for d in task['datasets']},
            epoch=state['epoch'], evaluation_signature=signature)
        meta = out / f'{task["run_id"]}.json'
        if task['run_id'] in results:
            previous = json.loads(meta.read_text())
            if any(previous.get(k) != v for k, v in expected.items()):
                raise ValueError(f'Cached result has different scientific inputs: {meta}')
            print(f'Skip completed evaluation: {task["run_id"]}', flush=True)
            continue
        print(f'Evaluating {task["run_id"]}: {len(ids) * task["N"]:,} real transcript–dataset pairs', flush=True)
        frame = evaluate_task(task, local_config(task, root, manifest), checkpoint, sequence, profiles, replicas,
                              torch.device(args.device), args.max_padded_tokens)
        path = out / f'{task["run_id"]}.parquet'; temporary = path.with_suffix('.tmp.parquet')
        frame.to_parquet(temporary, index=False); temporary.replace(path)
        write_json(meta, dict(**expected, evaluation_code_sha256=sha256(Path(__file__)), metrics_sha256=sha256(path), n_pairs=len(frame)))
        frame.attrs['epoch'] = state['epoch']; results[task['run_id']] = frame
    write_report(root, out, manifest, cohort, results, fold_table, transitions)
    write_json(out / 'analysis_manifest.json', dict(
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        experiment_manifest_sha256=sha256(root / 'experiment_manifest.json'),
        evaluation_cohort_sha256=sha256(out / 'cohort_manifest.json'),
        validated_models=len(results), planned_models=len(manifest['tasks']),
        n_transcripts=len(ids), n_evaluated_pairs=sum(len(frame) for frame in results.values()),
        evaluation_signature=signature,
        source_code_sha256={str(p): sha256(p) for p in [Path(__file__), ROOT / 'Utils/cumulative_dataset_pcc_report.py']},
        outputs={p.name: sha256(p) for p in out.iterdir() if p.is_file() and p.name != 'analysis_manifest.json'}))
    print(f'Fixed-observation report: {out / "analysis_report.html"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
