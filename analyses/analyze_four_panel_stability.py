#!/usr/bin/env python3
"""Compare L_bio across six panel pairs on common held-out test sequences.

Logged validation observation fit remains a training diagnostic: each dataset
can contribute a different observed transcript subset. No inference is run here.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_four_panel_stability import DEFAULT_OUTPUT, DESIGN
from run_cumulative_stability import object_sha256, sha256, write_json
from analyses.analyze_cumulative_reference_weight_stability import (
    ARMS as POLICY_ARMS, COLORS as POLICY_COLORS, COMPARISONS,
    STYLE, summarize_stability, table_html,
)
from analyses.analyze_real_exp8_reference_directionality import collect, pair_record, training_summary
from analyses.analyze_rank_balanced_reference_directionality import collect_logs, relocate
from Utils.partial_stability_report import (
    audit_transcript_folds, comparison_readiness, effect_sentence, plot_availability, plot_validation_history,
    profile_variation_text, report_timestamp, snapshot_text,
)

ARMS = [*POLICY_ARMS, 'shared_only']
COLORS = {**POLICY_COLORS, 'shared_only': '#bd3b3b'}
ABLATION_ARMS = ['shared_only', 'equal', 'ranked_p1', 'ranked_p3']
ABLATION_COMPARISONS = [('shared_only', arm) for arm in ABLATION_ARMS[1:]]


def peak_overlap(left, right):
    """Jaccard overlap of positions strictly above each profile's 90th percentile."""
    if left['coordinate_hash'] != right['coordinate_hash'] or left['length'] != right['length']:
        return np.nan
    x, y = left['values'], right['values']
    if not np.isfinite(x).all() or not np.isfinite(y).all() or min(np.var(x), np.var(y)) <= 1e-12:
        return np.nan
    a, b = x > np.quantile(x, .9), y > np.quantile(y, .9)
    if not a.any() or not b.any():
        return np.nan
    return float((a & b).sum() / (a | b).sum())


def cross_panel_rows(profiles, ids, panels, seed):
    rows = []
    for a, b in itertools.combinations(panels, 2):
        for arm in ARMS:
            left, right = (seed, arm, a), (seed, arm, b)
            if left not in profiles or right not in profiles:
                continue
            for tid in ids:
                rows.append(dict(kind='cross_panel', panel_a=a, panel_b=b, arm=arm,
                                 transcript_id=tid,
                                 peak_Jaccard=peak_overlap(profiles[left][tid], profiles[right][tid]),
                                 **pair_record(profiles[left][tid], profiles[right][tid], 'full_cds')))
    return pd.DataFrame(rows, columns=['kind', 'panel_a', 'panel_b', 'arm', 'transcript_id',
                                      'PCC', 'RMSE', 'peak_Jaccard', 'variance_a', 'variance_b', 'reason'])


def summarize_pairs(rows, ids):
    # Reuse the same paired estimator and common-finite-cohort rule as cumulative analysis.
    names = {'panel_a': 'N_a', 'panel_b': 'N_b'}
    return tuple(table.rename(columns={v: k for k, v in names.items()})
                 for table in summarize_stability(rows.rename(columns=names), ids,
                    comparisons=[*COMPARISONS, *ABLATION_COMPARISONS],
                    metrics=('PCC', 'RMSE', 'peak_Jaccard')))


def plot_design(weights, panels, out):
    concentration = pd.DataFrame([
        dict(panel_id=panel, arm=arm, N=len(group), N_ref=1 / np.square(group.pi).sum(),
             weighted_mean_rank=np.dot(group.pi, group.global_rank))
        for (panel, arm), group in weights[weights.arm != 'shared_only'].groupby(['panel_id', 'arm'], sort=False)])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    membership = weights[weights.arm == 'equal']
    for i, panel in enumerate(panels):
        ranks = membership[membership.panel_id == panel].global_rank
        axes[0].scatter(ranks, np.full(len(ranks), i), s=24, alpha=.7, color='#267eab')
    axes[0].set(yticks=range(len(panels)), yticklabels=panels, xlabel='Global rank (1 = best)',
                title='Fixed, disjoint panel membership')
    for arm in POLICY_ARMS:
        frame = concentration[concentration.arm == arm].set_index('panel_id').loc[panels]
        style = 'o--' if 'reverse' in arm else 'o-'
        axes[1].plot(range(len(panels)), frame.N_ref / frame.N, style, color=COLORS[arm], label=arm)
        axes[2].plot(range(len(panels)), frame.weighted_mean_rank, style, color=COLORS[arm])
    axes[1].set(title='Weight concentration', ylabel='Effective reference fraction', ylim=(0, 1.06))
    axes[2].set(title='Reference quality composition', ylabel='Weighted global rank (1 = best)')
    axes[1].legend(fontsize=8)
    for ax in axes[1:]:
        ax.set_xticks(range(len(panels)), panels, rotation=25)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'panel_design.svg')
    plt.close(fig)
    concentration.to_csv(out / 'reference_concentration.csv', index=False)


def plot_agreement(summary, effects, panels, out, *, ablation=False):
    pairs = list(itertools.combinations(panels, 2))
    index = pd.MultiIndex.from_tuples(pairs)
    x = np.arange(len(pairs))
    arms = ABLATION_ARMS if ablation else POLICY_ARMS
    baseline = 'shared_only' if ablation else 'equal'
    full_arms = ABLATION_ARMS[1:] if ablation else ['ranked_p1', 'ranked_p3']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, metric in enumerate(('PCC', 'RMSE')):
        for arm in arms:
            frame = summary[(summary.metric == metric) & (summary.arm == arm)]
            values = frame.set_index(['panel_a', 'panel_b'])['mean'].reindex(index)
            axes[0, column].plot(x, values, 'o--' if 'reverse' in arm else 'o-', color=COLORS[arm], label=arm)
        for arm in full_arms:
            frame = effects[(effects.metric == metric) & (effects.baseline == baseline) & (effects.policy == arm)]
            values = frame.set_index(['panel_a', 'panel_b']).mean_improvement.reindex(index)
            axes[1, column].plot(x, values, 'o-', color=COLORS[arm], label=arm)
        axes[0, column].set(title=f'Cross-panel {metric}', ylabel=f'Mean transcript {metric}')
        axes[1, column].set(title='Positive = full model improves agreement' if ablation else 'Positive = ranking improves agreement',
                            ylabel=('PCC_full − PCC_shared' if metric == 'PCC' else 'RMSE_shared − RMSE_full') if ablation
                            else ('PCC_ranked − PCC_equal' if metric == 'PCC' else 'RMSE_equal − RMSE_ranked'))
        axes[1, column].axhline(0, color='#64748b', linewidth=1)
    for ax in axes.flat:
        ax.set_xticks(x, [f'{a[-2:]} vs {b[-2:]}' for a, b in pairs], rotation=25)
        ax.set_xlabel('Panel pair')
        ax.grid(alpha=.2)
        if summary[summary.arm.isin(arms)].empty:
            ax.text(.5, .5, 'No completed panel pairs yet', ha='center', transform=ax.transAxes)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    stem = 'gamma_ablation' if ablation else 'cross_panel_agreement'
    for extension in ('svg', 'pdf', 'png'):
        fig.savefig(out / f'{stem}.{extension}', dpi=180)
    plt.close(fig)


def plot_shape_diagnostics(summary, variance, panels, out):
    pairs = list(itertools.combinations(panels, 2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for arm in ABLATION_ARMS:
        values = variance[variance.arm == arm].groupby('panel_id').variance.mean().reindex(panels)
        axes[0].plot(range(len(panels)), values, 'o-', color=COLORS[arm], label=arm)
        frame = summary[(summary.metric == 'peak_Jaccard') & (summary.arm == arm)]
        peaks = frame.set_index(['panel_a', 'panel_b'])['mean'].reindex(pd.MultiIndex.from_tuples(pairs))
        axes[1].plot(range(len(pairs)), peaks, 'o-', color=COLORS[arm], label=arm)
    axes[0].set(title='Normalized profile variation', ylabel='Mean within-transcript variance')
    axes[0].set_xticks(range(len(panels)), panels, rotation=25)
    axes[1].set(title='Peak-position agreement', ylabel='Mean peak Jaccard overlap', ylim=(0, 1.05))
    axes[1].set_xticks(range(len(pairs)), [f'{a[-2:]} vs {b[-2:]}' for a, b in pairs], rotation=25)
    for ax in axes:
        ax.grid(alpha=.2)
        if variance.empty:
            ax.text(.5, .5, 'No completed profiles yet', ha='center', transform=ax.transAxes)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'profile_structure.svg')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    root = args.experiment_root.expanduser().resolve()
    print(f'Analyzing available four-panel exports in {root}', flush=True)
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    if manifest['experiment_design'] != DESIGN:
        raise ValueError('Expected the standalone four-panel experiment.')
    if object_sha256(manifest['tasks']) != manifest['tasks_sha256']:
        raise ValueError('The saved task mapping changed.')
    recorded = Path(manifest['output_root'])
    for original, expected in manifest['frozen_file_sha256'].items():
        if sha256(relocate(original, root, recorded)) != expected:
            raise ValueError(f'Saved input or configuration changed: {original}')
    panels, seed = list(manifest['panels']), manifest['training_seeds'][0]
    folds = manifest['source_folds']
    fold_controls, fold_counts, fold_pairs = audit_transcript_folds(folds, require_common_validation=True)
    ids = folds[panels[0]]['test_ids']
    n_validation = len(folds[panels[0]]['validation_ids'])
    print(f'Fold audit: common validation={n_validation:,}, common test={len(ids):,}; train fixed={fold_controls["identical_train_ids"]}.', flush=True)
    configs = {(task['training_seed'], task['arm'], task['panel_id']):
               yaml.safe_load(relocate(task['config_path'], root, recorded).read_text())
               for task in manifest['tasks']}
    weights = pd.read_csv(root / 'reference_weights.csv')
    profiles, availability = collect(root, manifest, configs, ids, weights)
    print(f'Validated {len(profiles)}/{len(manifest["tasks"])} models; computing matched transcript metrics.', flush=True)
    readiness = comparison_readiness(availability, 'panel_id', list(itertools.combinations(panels, 2)), ARMS)
    logs = collect_logs(availability)
    validation_fit = training_summary(logs, availability)
    validation_fit['panel_id'] = validation_fit.task_id.map(availability.set_index('task_id').panel_id)
    rows = cross_panel_rows(profiles, ids, panels, seed)
    summary, effects, cohorts = summarize_pairs(rows, ids)
    variance = pd.DataFrame([
        dict(panel_id=panel, arm=arm, transcript_id=tid,
             variance=float(np.var(profile['values'])), near_constant=bool(np.var(profile['values']) <= 1e-12))
        for (_, arm, panel), values in profiles.items() for tid, profile in values.items()],
        columns=['panel_id', 'arm', 'transcript_id', 'variance', 'near_constant'])
    counts = pd.DataFrame([dict(panel_id=panel, datasets=len(manifest['panels'][panel]),
                               train=len(folds[panel]['train_ids']), validation=len(folds[panel]['validation_ids']),
                               test=len(ids)) for panel in panels])
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    for name, table in [('availability', availability), ('transcript_agreement', rows),
                        ('agreement_summary', summary), ('paired_policy_improvements', effects),
                        ('metric_cohorts', cohorts), ('profile_variance', variance), ('split_counts', counts),
                        ('gamma_ablation_effects', effects[effects.baseline == 'shared_only']),
                        ('validation_fit', validation_fit), ('training_scalars', logs),
                        ('comparison_readiness', readiness), ('fold_audit', fold_counts),
                        ('fold_pair_audit', fold_pairs)]:
        table.to_csv(out / f'{name}.csv', index=False)
    plot_design(weights, panels, out)
    plot_agreement(summary, effects, panels, out)
    plot_agreement(summary, effects, panels, out, ablation=True)
    plot_shape_diagnostics(summary, variance, panels, out)
    plot_availability(availability, 'panel_id', panels, ARMS, out)
    plot_validation_history(logs, availability, 'panel_id', panels, ARMS, COLORS, out)
    complete = int(availability.status.eq('validated_predictions').sum())
    created = report_timestamp()
    ablation_comment = effect_sentence(effects, 'shared_only', 'equal', unit='panel pairs')
    ranking_comment = ' '.join(effect_sentence(effects, 'equal', arm, unit='panel pairs')
                               for arm in ('ranked_p1', 'ranked_p3'))
    ablation_pairs = int(((effects.baseline == 'shared_only') & (effects.policy == 'equal') & (effects.metric == 'PCC')).sum())
    report = f'''<!doctype html><html><head><meta charset="utf-8"><title>Four-panel reference stability</title>
<style>{STYLE}</style></head><body><h1>Do dataset-specific corrections improve cross-panel L_bio agreement?</h1>
<p>Partial results · generated {created} · training seed {seed}</p>
{snapshot_text(availability,len(ids))}
<p class="note"><b>Verified cohort scope:</b> all panels share {n_validation:,} validation and {len(ids):,} test transcript IDs, and each training fold excludes both held-out sets. No training/validation role swapping occurs across panels. Training membership is panel-specific, and usable validation observations can differ between datasets.</p>
<details><summary><b>What is matched in this report?</b></summary>
{table_html(fold_counts)}
<p><b>L_bio PCC, RMSE and peak overlap:</b> compare predictions on common held-out sequences, using an explicitly recorded common finite cohort for each metric. These calculations do not require observed targets and remain valid descriptions of cross-panel reproducibility.</p>
<p><b>Logged μ PCC and observation loss:</b> describe validation fit. The nominal validation pool is common, but each dataset contributes its own eligible observed rows, and the four panels contain different dataset identities. Their aggregate validation scores are not a comparison against identical observations.</p>
<p><b>Training:</b> policies and the gamma=1 baseline within a panel reuse its folds. Across panels, transcript eligibility changes with observation coverage; a strict experiment that changes only dataset composition would also need a common training cohort.</p>
<p><a href="fold_audit.csv">Fold counts</a> · <a href="fold_pair_audit.csv">Pairwise training overlap and held-out role audit</a>.</p></details>
<h2>Current interpretation</h2><p>{ablation_comment}</p><p>{ranking_comment}</p>
<p>The primary full-equal versus gamma=1 comparison currently covers <b>{ablation_pairs}/6 panel pairs</b>. Compare the same pair across models: pooling all available shared-only pairs against fewer full-model pairs would confound the comparison with panel composition. PCC, RMSE and peak overlap measure different aspects of agreement and can give different conclusions.</p>
<h2>Which models are available?</h2><img src="availability.svg" alt="Available model exports by panel and arm">
<p>Green cells have validated predictions; a panel-pair effect needs both endpoints for both models, and additional shared-only endpoints do not fill gaps in a full-model comparison.</p>
<p><a href="comparison_readiness.csv">Exact available and missing panel-pair endpoints</a> · <a href="availability.csv">Artifact availability and validation details</a></p>
<h2>Primary experiment: full model versus gamma = 1</h2>
<p>Each shared_only model is trained from scratch with <code>model.mean_correction=unity</code>, using its panel's exact equal-arm configuration, splits, reliability references and seed. The existing 20 full-model tasks keep their indices; four baselines occupy indices 20–23. Gamma is one during training and prediction. Setting it to one only after full-model training would answer a different question.</p>
<img src="gamma_ablation.svg" alt="Cross-panel agreement for full and gamma-one models, with paired PCC and RMSE improvements">
<p class="note">{ablation_comment}</p>
<p>Positive full-minus-shared PCC and shared-minus-full RMSE differences support improved reproducibility from the full system; the equal-reference comparison is primary, while the ranked models additionally change reference weighting.</p>
<p class="note"><b>Dispersion-context limitation:</b> the learned alpha head is retained, but gamma=1 removes the training gradients to its detached dataset-context encoder. The full and shared-only models therefore learn different dispersion representations. This is a whole-system ablation; it is not the separate confirmation with identical fixed dispersion in both arms.</p>
<p>The gamma=1 baseline retains uniform reference metadata for matching, but those weights have no effect on its final mean correction. The analysis verifies unit gamma in its exported predictions.</p>
<h2>Does agreement preserve profile structure?</h2>
<img src="profile_structure.svg" alt="Within-transcript profile variance and cross-panel peak overlap for full and shared-only models">
<p>{profile_variation_text(variance)}</p>
<p>Peak overlap is the intersection divided by the union of positions strictly above each profile's 90th percentile. Ties at the threshold are excluded; flat profiles or empty peak sets give an undefined score, not perfect overlap. PCC, RMSE and peak overlap each use a common finite cohort across available panel pairs and models.</p>
<h2>Observation fit on validation transcripts</h2>
<img src="validation_history.svg" alt="Per-panel validation loss histories with selected checkpoint markers">
<p>Curves include unfinished training as progress diagnostics; only checkpoint-selected metrics from validated completed models enter the selected-fit table.</p>
<p>The table reports logged metrics at the selected best-validation-loss epoch. Validation was used for checkpoint selection, so these are descriptive fit diagnostics. The common-test exports contain sequence-only predictions with dummy observation targets and are never used to compute count-reconstruction accuracy. Independent test-count evaluation would require a separate observation-based prediction pass.</p>
<p>The same {n_validation:,} nominal validation IDs are shared across panels; effective observed subsets vary by dataset. Full versus gamma=1 within the same panel uses matched available observations, while cross-panel validation means also change dataset composition. The cumulative experiment's 704-transcript re-evaluation does not re-evaluate these panel models.</p>
{table_html(validation_fit[['panel_id','arm','selected_epoch','selected_val_loss','selected_val_mu_pcc','selected_val_replica_nll']])}
<h2>Four fixed dataset collections</h2><img src="panel_design.svg" alt="Panel rank membership, weight concentration and weighted reference rank">
<p>Ranked and reverse use the same dataset-weight multiset within each panel; their reference quality differs because the large weights are assigned to opposite ends of the global ranking.</p>
{table_html(counts)}
<p>The default uses the original repository panel assignment: all 114 datasets appear once across panels of 29/29/28/28, with related datasets from each source kept together. A supplied alternative assignment is recorded in <a href="../inputs/panel_assignment.csv">the exact membership table</a>. Splits are rebuilt from current data: validation and test IDs are common across all panels, each held-out transcript has at least two usable datasets in every panel, and panel-specific reliability references use training transcripts only.</p>
<h2>All six panel pairs</h2><img src="cross_panel_agreement.svg" alt="Cross-panel PCC and RMSE and paired ranked-minus-equal improvements for all six pairs">
<p class="note">{ranking_comment}</p>
<p>For each transcript, compare its mean-one full-CDS L_bio across two independently trained panels under the same policy, then compare that agreement between policies. The analysis uses one common finite transcript cohort across available pairs and policies for each metric; undefined correlations are excluded explicitly. This estimates reproducibility across dataset collections, not cumulative resistance to adding lower-ranked datasets.</p>
<p>π changes gamma centering; it does not directly weight dataset observation losses. The local w_dt reliability weights remain a separate mechanism. The p=1 policies reproduce the original score mapping; p=3 is a prespecified stronger-reference sensitivity test. Ranked-versus-reverse comparisons isolate dataset-weight concentration, but source-family weight concentration can still differ.</p>
<p class="note">Greater agreement is not biological ground-truth accuracy: inspect <a href="profile_variance.csv">profile variance and near-constant flags</a>. One seed and one panel partition give conditional evidence; the six pairs share models and transcripts and are not six independent training repetitions. No across-seed confidence intervals are reported.</p>
<h2>Paired improvements</h2>{table_html(effects)}<h2>Agreement per policy and panel pair</h2>{table_html(summary)}
<h2>Availability</h2>{table_html(availability[['panel_id','arm','status','selected_epoch']])}
<h2>Refresh this partial analysis</h2><pre>python analyses/analyze_four_panel_stability.py</pre>
<p>Re-run after downloading more outputs. The analysis reads model artifacts and writes reports; it does not launch training or change the experiment inputs. Cohorts are recomputed from the available valid results, so the saved timestamp and cohort CSV belong to this snapshot.</p></body></html>'''
    (out / 'analysis_report.html').write_text(report)
    write_json(out / 'analysis_manifest.json', dict(
        created_utc=created, planned_models=len(manifest['tasks']),
        snapshot_status='partial' if complete < len(manifest['tasks']) else 'complete',
        experiment_manifest_sha256=sha256(root / 'experiment_manifest.json'),
        analysis_code_sha256=sha256(Path(__file__)), validated_models=complete,
        fold_controls=fold_controls,
        report_helpers_sha256=sha256(ROOT / 'Utils/partial_stability_report.py'),
        outputs={p.name: sha256(p) for p in out.iterdir() if p.is_file() and p.name != 'analysis_manifest.json'}))
    print(f'{complete}/{len(manifest["tasks"])} validated exports; report: {out / "analysis_report.html"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
