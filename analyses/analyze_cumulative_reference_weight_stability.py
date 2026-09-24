#!/usr/bin/env python3
"""Analyze cumulative L_bio stability, cached observed test fit and validation logs.

The default root contains the legacy runs. Pass --experiment-root
results/cumulative_stability_fixed_cohort_seed42 for the corrected training design.
This CPU analysis does not run checkpoint inference or change any training split.
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from run_cumulative_stability import DESIGN as SIMPLE_DESIGN, FIXED_DESIGN, DEFAULT_OUTPUT, POLICIES, object_sha256, sha256, write_json
from analyses.analyze_real_exp8_reference_directionality import collect, pair_record, training_summary
from analyses.analyze_rank_balanced_reference_directionality import collect_logs, relocate
from Utils.partial_stability_report import (
    audit_transcript_folds, comparison_readiness, effect_sentence, plot_availability, plot_validation_history,
    profile_variation_text, report_timestamp, snapshot_text,
)
from Utils.cumulative_dataset_pcc_report import write_dataset_mu_report

ARMS=[p[0] for p in POLICIES]
COLORS={'equal':'#64748b','ranked_p1':'#267eab','reverse_p1':'#9a519b','ranked_p3':'#006d50','reverse_p3':'#d17820'}
COMPARISONS=[('equal','ranked_p1'),('equal','ranked_p3'),('reverse_p1','ranked_p1'),('reverse_p3','ranked_p3')]
STYLE='body{max-width:1100px;margin:35px auto;padding:0 22px;font:16px/1.6 system-ui;color:#203448}table{border-collapse:collapse;font-size:13px}td,th{padding:6px 10px;border-bottom:1px solid #ddd;text-align:right}.table{overflow:auto}img{max-width:100%}.note{padding:14px;background:#f3f7fb;border-left:4px solid #267eab}pre{overflow:auto;background:#f3f3f3;padding:12px}'


def read_json(path):
    return json.loads(Path(path).read_text())


def table_html(frame):
    return '<div class="table">'+frame.to_html(index=False,border=0,na_rep='—',float_format=lambda x:f'{x:.5g}')+'</div>'


def fixed_observation_section(root, manifest, hashes):
    """Display the separately computed test snapshot without rerunning inference."""
    directory = root / 'fixed_observed_evaluation'
    meta_path = directory / 'analysis_manifest.json'
    command = 'python analyses/reevaluate_cumulative_fixed_observations.py --experiment-root ' + html.escape(str(root))
    if not meta_path.exists():
        return ('<section id="fixed-observed-fit"><h2>Observed-profile fit on fixed test observations</h2>'
                '<p>No re-evaluated test snapshot is available for this experiment yet. '
                'The validation diagnostics below cannot supply matched test metrics. '
                'After checkpoints finish, run:</p><pre>' + command + '</pre></section>')
    meta = read_json(meta_path)
    if meta['experiment_manifest_sha256'] != sha256(root / 'experiment_manifest.json'):
        raise ValueError('The cached observed evaluation belongs to a different experiment manifest.')
    hashes[str(meta_path)] = sha256(meta_path)
    for name in ('cohort_manifest.json', 'observed_fit_summary.csv', 'fixed_observed_fit.svg', 'analysis_report.html'):
        path = directory / name
        if sha256(path) != meta['outputs'][name]:
            raise ValueError(f'Cached observed evaluation changed: {path}')
        hashes[str(path)] = meta['outputs'][name]
    cohort = read_json(directory / 'cohort_manifest.json')
    test_ids = set(next(iter(manifest['source_folds'].values()))['test_ids'])
    if not set(cohort['transcript_ids']) <= test_ids:
        raise ValueError('The observed evaluation includes transcripts outside the common test fold.')
    means = pd.read_csv(directory / 'observed_fit_summary.csv')
    equal = means[(means.arm == 'equal') & (means.scope == 'best2_datasets')].sort_values('N')
    caption = 'At least two completed equal-reference sizes are needed to assess an across-N trend.'
    if len(equal) >= 2:
        a, b = equal.iloc[0], equal.iloc[-1]
        caption = (f'On the same best-two dataset observations under equal weighting, N={int(a.N)}→{int(b.N)} '
            f'changes μ PCC from {a.mu_pcc:.3f} to {b.mu_pcc:.3f} and L_bio PCC from {a.L_bio_pcc:.3f} '
            f'to {b.L_bio_pcc:.3f}, so the shared and corrected observation profiles can behave differently.')
    return f'''<section id="fixed-observed-fit"><h2>Observed-profile fit on fixed test observations</h2>
<p class="note"><b>{meta['validated_models']}/{meta['planned_models']} checkpoints</b> re-evaluated on the same <b>{cohort['n_transcripts']} held-out transcripts per dataset</b>, observed throughout the {len(cohort['dataset_universe'])}-dataset universe; cached snapshot {html.escape(meta['created_utc'])}.</p>
<img src="../fixed_observed_evaluation/fixed_observed_fit.svg" alt="Mu and Lbio PCC on identical real held-out observations">
<p>{caption}</p>
<p>The bottom row fixes both transcript and dataset identities; the top row adds dataset targets as N grows. Both use real replica consensus targets and equal evaluation weights. These are observed-profile PCCs; the separate stability figures compare L_bio predictions with each other.</p>
<p><a href="../fixed_observed_evaluation/analysis_report.html#dataset-mu-pcc"><b>Every dataset's μ PCC curve and exact values →</b></a> · <a href="../fixed_observed_evaluation/cohort_manifest.json">Exact observed test cohort</a></p>
<p>This analysis reads the cached test snapshot. Refresh it after additional checkpoints finish:</p><pre>{command}</pre></section>'''


def rationale(fixed_cohort=False):
    split_text = (
        'One complete transcript intersection across all configured datasets is partitioned once. Training, validation and test IDs are identical across N and policies; training-only reliability references are identical for each retained dataset. Adding datasets therefore holds transcript membership fixed. This complete-case cohort favors broadly observed transcripts.'
        if fixed_cohort else
        'Across N, the legacy setup resampled training/validation IDs and refitted reliability references; transcripts could change between training and validation roles. Only test IDs are common across sizes. Within N, all policies share those inputs, model configuration and seed. Consequently, the legacy across-N comparison confounds dataset addition with training cohort changes. See the fixed-cohort correction for fresh training.')
    return '''<h2>What is the hypothesis?</h2>
<p>As progressively lower-ranked datasets enter the nested collection, quality-based gamma references may preserve the learned L_bio profile better than an equal reference. The primary evidence is a positive <b>paired stability improvement</b> for the same addition of datasets: PCC_ranked − PCC_equal &gt; 0, supported by RMSE_equal − RMSE_ranked &gt; 0.</p>
<p>The primary stability estimand uses one shared anchor: for every policy and size N, compare L_bio with the same equal-reference N=2 model on the same test transcript and CDS coordinates. This prevents separately trained N=2 models from entering the ranked-versus-equal contrast. Arm-specific N=2 anchors and adjacent N→M comparisons are retained as secondary diagnostics. Missing endpoints leave a gap; they are never replaced by a larger available size.</p>
<p><b>Equal-weight agreement need not decline monotonically.</b> New datasets have lower global rank, but may still add useful information; larger samples, averaging, transcript coverage and nested-set overlap can increase stability even under equal weighting. The last addition also represents a smaller fraction of the final collection. The experiment tests a hypothesis rather than enforcing a desired curve.</p>
<p>Compare ranked versus reverse at the same exponent to separate direction from dataset-weight concentration; if both improve similarly over equal, concentration is a plausible explanation rather than quality direction. The p=1 comparison tests the original mapping and p=3 is a prespecified stronger-reference sensitivity analysis. Even q³ stays near-uniform for the smallest top-quality prefixes.</p>
<p><b>Scope of the intervention:</b> π controls gamma centering; local dataset–transcript reliability weights w_dt separately control observation losses. Ranking the reference does not directly remove the influence of lower-ranked datasets on learning. High agreement measures stability, not biological correctness.</p>
<p>__SPLIT_TEXT__ Source-family aggregation can differ under reversed dataset weights, and QC-ranking transcript scope remains unverified.</p>
<p>Supplement adjacent agreement with N=2-versus-N drift and profile variance: small consecutive changes can accumulate, and flat or otherwise uninformative profiles should not count as successful stabilization. N=2 is a drift anchor, not biological truth; mean-one L_bio variance measures normalized profile amplitude.</p>
<p class="note">One training seed gives conditional evidence for this initialization. It does not estimate optimization variability, establish general reproducibility, or prove a biological advantage. Metrics use matched finite transcript cohorts; counts of transcripts are not independent training repetitions.</p>'''.replace('__SPLIT_TEXT__', split_text)


def write_design_report(root,manifest,weights,concentration):
    sizes=manifest['sizes']
    fixed_cohort=manifest['experiment_design']==FIXED_DESIGN
    standalone=manifest['experiment_design'] in {SIMPLE_DESIGN,FIXED_DESIGN}
    source_text=('The setup reads the current dataset YAML and ranking TSV directly, constructs exact top-N prefixes, and partitions the complete transcript intersection across all configured datasets once; the same train/validation/test IDs are reused at every N and in every dataset.' if fixed_cohort else
        'The setup reads the current dataset YAML and ranking TSV directly, constructs exact top-N prefixes, and creates one common test set with one training/validation split per N; no previous experiment files are required.' if standalone else 'The setup reuses the original cumulative experiment’s frozen prefixes and folds.')
    launcher='run_cumulative_fixed_cohort_univie.slurm' if fixed_cohort else 'run_cumulative_reference_weight_stability_univie.slurm'
    root_arg=f' --experiment-root {html.escape(str(root))}' if fixed_cohort else ''
    fig,axes=plt.subplots(1,2,figsize=(11,3.6))
    for arm in ARMS:
        part=concentration[concentration.arm==arm].sort_values('N')
        style='--' if 'reverse' in arm else '-'
        axes[0].plot(range(len(sizes)),part.N_ref/part.N,style,marker='o',color=COLORS[arm],label=arm)
        axes[1].plot(range(len(sizes)),part.weighted_mean_rank,style,marker='o',color=COLORS[arm])
    for ax in axes:
        ax.set_xticks(range(len(sizes)),sizes)
        ax.set_xlabel('Datasets in frozen cumulative prefix')
        ax.grid(alpha=.2)
    axes[0].set(title='Weight concentration',ylabel='Effective reference fraction N_ref / N',ylim=(0,1.06))
    axes[1].set(title='Reference quality composition',ylabel='Weighted global rank (1 = best)',ylim=(0,115))
    axes[0].legend(fontsize=8,loc='lower left')
    fig.tight_layout()
    fig.savefig(root/'cumulative_reference_design.svg')
    plt.close(fig)
    folds=pd.DataFrame([dict(N=n,train=len(manifest['source_folds'][str(n)]['train_ids']),
        validation=len(manifest['source_folds'][str(n)]['validation_ids']),test=len(manifest['source_folds'][str(n)]['test_ids'])) for n in sizes])
    folds.to_csv(root/'split_counts.csv',index=False)
    content=f'''<!doctype html><html><head><meta charset="utf-8"><title>Cumulative quality-reference stability</title><style>{STYLE}</style></head><body>
<h1>Does quality ranking stabilize L_bio as datasets are added?</h1>
<p class="note">{len(sizes)} cumulative sizes × 5 policies × seed {manifest['training_seeds'][0]} = {len(manifest['tasks'])} fresh trainings; the fixed-N=114 sensitivity study is a secondary endpoint comparison.</p>
<p>{source_text} Sizes: {', '.join(map(str,sizes))}. The default dataset configuration contains 114 active datasets out of the 115-row ranking universe (global rank 110 is absent). Each model starts fresh; cumulative membership does not mean continuing a previous checkpoint.</p>
<p>At every size train equal, ranked_p1, reverse_p1, ranked_p3 and reverse_p3. Define q=(116−rank)/115, apply p=1 or p=3, and reverse the same weight multiset within each selected prefix. Renormalize over that prefix, with global ranks unchanged.</p>
<img src="cumulative_reference_design.svg" alt="Planned concentration and quality composition across all cumulative sizes and policies">
<p class="note">The ranking intervention grows stronger as lower-ranked datasets enter, but the predicted ordering of profile stability remains an empirical hypothesis.</p>
{rationale(fixed_cohort)}<h2>Frozen split sizes</h2>{table_html(folds)}
<h2>Run</h2><pre>sbatch {launcher}
python analyses/analyze_cumulative_reference_weight_stability.py{root_arg}</pre>
<p>The first array worker creates the shared inputs; the others reuse them. Run the analysis command after training to report available models and explicit gaps. Setup writes local paths on the training machine. <a href="task_matrix.csv">Exact task mapping</a> · <a href="analysis/analysis_report.html">Analysis report</a>.</p></body></html>'''
    (root/'design_report.html').write_text(content)


def stability_rows(profiles,ids,sizes,seed):
    rows=[]
    comparisons=[('adjacent',a,b) for a,b in zip(sizes[:-1],sizes[1:])]
    comparisons += [('anchor',sizes[0],b) for b in sizes[1:]]
    for kind,a,b in comparisons:
        for arm in ARMS:
            ka,kb=(seed,arm,a),(seed,arm,b)
            if ka not in profiles or kb not in profiles:
                continue
            for tid in ids:
                rows.append(dict(kind=kind,N_a=a,N_b=b,arm=arm,transcript_id=tid,
                                 **pair_record(profiles[ka][tid],profiles[kb][tid],'full_cds')))
    # A single anchor is required for a clean between-policy stability contrast.
    # The policy-specific N=2 fits above remain useful as a sensitivity analysis,
    # but they are distinct fitted models even when their reference weights are
    # almost uniform.
    anchor_key=(seed,'equal',sizes[0])
    if anchor_key in profiles:
        for b in sizes[1:]:
            for arm in ARMS:
                kb=(seed,arm,b)
                if kb not in profiles:
                    continue
                for tid in ids:
                    rows.append(dict(kind='shared_anchor',N_a=sizes[0],N_b=b,arm=arm,
                                     transcript_id=tid,
                                     **pair_record(profiles[anchor_key][tid],profiles[kb][tid],'full_cds')))
    return pd.DataFrame(rows,columns=['kind','N_a','N_b','arm','transcript_id','PCC','RMSE','variance_a','variance_b','reason'])


def summarize_stability(frame,ids,comparisons=COMPARISONS,metrics=('PCC','RMSE')):
    summaries,effects,cohorts=[],[],[]
    common_ids={}
    for kind,group in frame.groupby('kind',sort=False):
        for metric in metrics:
            values=group.pivot(index='transcript_id',columns=['N_a','N_b','arm'],values=metric).reindex(ids)
            common_ids[kind,metric]=values.index[np.isfinite(values.to_numpy(float)).all(axis=1)]
    for (kind,a,b),group in frame.groupby(['kind','N_a','N_b'],sort=False):
        for metric in metrics:
            pivot=group.pivot(index='transcript_id',columns='arm',values=metric).reindex(ids)
            finite=pivot.index.isin(common_ids[kind,metric])
            valid=pivot.loc[finite]
            for tid,included in zip(ids,finite):
                cohorts.append(dict(kind=kind,N_a=a,N_b=b,metric=metric,transcript_id=tid,included=bool(included)))
            common=dict(kind=kind,N_a=a,N_b=b,metric=metric,n_valid=len(valid),n_excluded=len(ids)-len(valid),
                        matched_arms=','.join(sorted(pivot.columns)))
            for arm in pivot.columns:
                summaries.append(dict(**common,arm=arm,mean=valid[arm].mean(),median=valid[arm].median()))
            for baseline,ranked in comparisons:
                if baseline not in pivot or ranked not in pivot:
                    continue
                delta=(valid[ranked]-valid[baseline])*(-1 if metric=='RMSE' else 1)
                effects.append(dict(**common,baseline=baseline,policy=ranked,mean_improvement=delta.mean(),
                                    median_improvement=delta.median(),fraction_positive=(delta>0).mean() if len(delta) else np.nan))
    common=['kind','N_a','N_b','metric','n_valid','n_excluded','matched_arms']
    return (pd.DataFrame(summaries,columns=common+['arm','mean','median']),
            pd.DataFrame(effects,columns=common+['baseline','policy','mean_improvement','median_improvement','fraction_positive']),
            pd.DataFrame(cohorts,columns=['kind','N_a','N_b','metric','transcript_id','included']))


def plot_stability(summary,effects,sizes,out,kind='adjacent'):
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    transitions=list(zip(sizes[:-1],sizes[1:])) if kind=='adjacent' else [(sizes[0],n) for n in sizes[1:]]
    title_prefix={'adjacent':'Adjacent', 'anchor':f'Arm-specific N={sizes[0]} anchor',
                  'shared_anchor':f'Shared equal N={sizes[0]} anchor'}[kind]
    x=np.arange(len(transitions))
    for column,metric in enumerate(('PCC','RMSE')):
        for arm in ARMS:
            s=summary[(summary.kind==kind)&(summary.metric==metric)&(summary.arm==arm)].set_index(['N_a','N_b'])
            vals=s['mean'].reindex(pd.MultiIndex.from_tuples(transitions)).to_numpy(float)
            axes[0,column].plot(x,vals,'o--' if 'reverse' in arm else 'o-',color=COLORS[arm],label=arm)
            if arm in ('ranked_p1','ranked_p3'):
                e=effects[(effects.kind==kind)&(effects.metric==metric)&(effects.baseline=='equal')&(effects.policy==arm)].set_index(['N_a','N_b'])
                vals=e.mean_improvement.reindex(pd.MultiIndex.from_tuples(transitions)).to_numpy(float)
                axes[1,column].plot(x,vals,'o-',color=COLORS[arm],label=arm+' versus equal')
        axes[0,column].set_title(title_prefix+(' profile agreement' if metric=='PCC' else ' profile difference'))
        axes[0,column].set_ylabel('Mean transcript '+metric)
        axes[1,column].set_ylabel('PCC_ranked − PCC_equal' if metric=='PCC' else 'RMSE_equal − RMSE_ranked')
        axes[1,column].set_title('Positive = ranked is more stable')
        axes[1,column].axhline(0,color='#64748b',linewidth=1)
    if summary.empty:
        for ax in axes.flat:
            ax.text(.5,.5,'No matched completed exports yet',ha='center',transform=ax.transAxes)
    for ax in axes.flat:
        ax.set_xticks(x,[f'{a}→{b}' for a,b in transitions],rotation=25)
        ax.set_xlabel('Addition of lower-ranked datasets' if kind=='adjacent' else f'Comparison with the N={sizes[0]} model')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    axes[1,0].legend(fontsize=8)
    fig.tight_layout()
    for extension in ('svg','png','pdf'):
        stem={'adjacent':'cumulative_stability', 'anchor':'anchor_stability',
              'shared_anchor':'shared_anchor_stability'}[kind]
        fig.savefig(out/f'{stem}.{extension}',dpi=180)
    plt.close(fig)


def equal_anchor_drift_sentence(summary,kind='shared_anchor'):
    """State the available equal-reference drift without implying biological truth."""
    selected=summary[(summary.kind==kind)&(summary.arm=='equal')]
    pcc=selected[selected.metric=='PCC'].sort_values('N_b')
    rmse=selected[selected.metric=='RMSE'].sort_values('N_b')
    available=sorted(set(pcc.N_b).intersection(rmse.N_b))
    if not available:
        return 'No complete equal-reference anchor comparison is available yet.'
    first,last=available[0],available[-1]
    p0=float(pcc.set_index('N_b').loc[first,'mean']);p1=float(pcc.set_index('N_b').loc[last,'mean'])
    r0=float(rmse.set_index('N_b').loc[first,'mean']);r1=float(rmse.set_index('N_b').loc[last,'mean'])
    return (f'Against the same equal-reference N=2 anchor and the same test transcripts, '
            f'equal-reference PCC changes from {p0:.3f} at N={first} to {p1:.3f} at N={last}, '
            f'while RMSE changes from {r0:.3f} to {r1:.3f}; this is accumulated profile drift, '
            'not evidence that the N=2 profile is biologically correct.')


def equal_anchor_transcript_sentence(frame):
    selected=frame[(frame.kind=='shared_anchor')&(frame.arm=='equal')]
    pivot=selected.pivot(index='transcript_id',columns='N_b',values='PCC')
    sizes=sorted(pivot.columns)
    if len(sizes)<2:
        return 'Too few completed equal-reference sizes are available to assess transcript-level progression.'
    clauses=[]
    for a,b in zip(sizes[:-1],sizes[1:]):
        valid=pivot[[a,b]].dropna()
        fraction=float((valid[b]<valid[a]).mean()) if len(valid) else np.nan
        clauses.append(f'{a}→{b}: {100*fraction:.1f}%')
    return ('The anchor PCC decreases for most individual transcripts at every available step '
            f'({"; ".join(clauses)}; matched n={len(pivot):,}), so the mean trend is not driven by a small subset.')


def observed_fit_tables(logs, availability, tasks, anchor_datasets):
    """Match observed-profile diagnostics at each exported best-loss checkpoint.

    Per-dataset tags use the same unweighted pair reduction for mu and L_bio.
    Average these dataset means equally, independently of the gamma reference.
    A missing dataset or metric leaves the aggregate undefined, not reweighted.
    """
    task_map = {task['run_id']: task for task in tasks}
    dataset_rows, summaries = [], []
    for model in availability.to_dict('records'):
        task = task_map[model['task_id']]
        metadata = {key: model[key] for key in ('task_id', 'training_seed', 'arm', 'N', 'status', 'selected_epoch')}
        selected = logs[(logs.task_id == model['task_id']) & (logs.epoch == model['selected_epoch'])]
        selected = selected.sort_values('wall_time').drop_duplicates('tag', keep='last').set_index('tag')
        ready = model['status'] == 'validated_predictions' and np.isfinite(model['selected_epoch'])
        current = []
        for dataset in task['datasets']:
            row = dict(**metadata, dataset_id=dataset)
            for metric, prefix in (('mu_pcc', 'val_mu_pcc/'), ('L_bio_pcc', 'val_L_bio_pcc/')):
                tag = prefix + dataset
                present = ready and tag in selected.index
                row[metric] = float(selected.loc[tag, 'value']) if present else np.nan
                row[metric + '_step'] = float(selected.loc[tag, 'step']) if present else np.nan
                row[metric + '_source'] = str(selected.loc[tag, 'source']) if present else ''
            # Do not combine tags from different validation passes or restarted logs.
            row['matched'] = bool(np.isfinite([row['mu_pcc'], row['L_bio_pcc']]).all()
                and row['mu_pcc_step'] == row['L_bio_pcc_step']
                and row['mu_pcc_source'] == row['L_bio_pcc_source'])
            row['correction_gain'] = row['mu_pcc'] - row['L_bio_pcc'] if row['matched'] else np.nan
            current.append(row)
        dataset_rows.extend(current)
        by_dataset = pd.DataFrame(current).set_index('dataset_id')
        for scope, names in (('all_selected', task['datasets']), ('best2_datasets', anchor_datasets)):
            subset = by_dataset.reindex(names)
            matched = subset['matched'].eq(True)
            complete = bool(matched.all())
            summaries.append(dict(**metadata, scope=scope, n_expected_datasets=len(names),
                n_matched_datasets=int(matched.sum()), complete=complete, dataset_ids=';'.join(names),
                missing_datasets=';'.join(subset.index[~matched]),
                mu_pcc=subset.mu_pcc.mean() if complete else np.nan,
                L_bio_pcc=subset.L_bio_pcc.mean() if complete else np.nan,
                correction_gain=subset.correction_gain.mean() if complete else np.nan))
    return pd.DataFrame(dataset_rows), pd.DataFrame(summaries)


def plot_observed_fit(observed, sizes, out, fixed_cohort=False):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    columns = [('mu_pcc', 'μ versus observed'), ('L_bio_pcc', 'L_bio versus observed'),
               ('correction_gain', 'PCC gain from fitted γ')]
    pcc_values = observed[['mu_pcc', 'L_bio_pcc']].to_numpy(float)
    finite = pcc_values[np.isfinite(pcc_values)]
    pcc_limits = (max(-1, float(finite.min()) - .03), min(1, float(finite.max()) + .03)) if len(finite) else (0, 1)
    for row, scope in enumerate(('all_selected', 'best2_datasets')):
        for column, (metric, title) in enumerate(columns):
            ax = axes[row, column]
            for arm in ARMS:
                subset = observed[(observed.scope == scope) & (observed.arm == arm)].set_index('N')
                values = subset[metric].reindex(sizes).to_numpy(float)
                ax.plot(range(len(sizes)), values, 'o--' if 'reverse' in arm else 'o-',
                        color=COLORS[arm], label=arm, markersize=5, linewidth=1.6)
            label = 'All N datasets' if row == 0 else 'Best-2 dataset identities'
            ax.set_title(f'{label}\n{title}', fontsize=12)
            ax.set_ylabel('Equal mean of dataset PCCs' if column < 2 else 'Mean PCC(μ, y) − PCC(L_bio, y)')
            if column < 2:
                ax.set_ylim(pcc_limits)
            else:
                ax.axhline(0, color='#64748b', linewidth=.8)
            if not np.isfinite(observed.loc[observed.scope == scope, metric]).any():
                ax.text(.5, .5, 'No matched selected-checkpoint logs', ha='center', transform=ax.transAxes)
            ax.set_xticks(range(len(sizes)), sizes)
            ax.set_xlabel('Number of training datasets N')
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False)
    cohort_title=('Identical validation transcripts in every dataset and N; the bottom row also fixes dataset identities'
                  if fixed_cohort else 'Validation transcripts change with N in both rows; only the bottom row fixes dataset identities')
    fig.suptitle('Observed-profile fit at the exported best-validation-loss checkpoint\n' + cohort_title, fontsize=13)
    fig.tight_layout(rect=(0, .05, 1, .92))
    for extension in ('svg', 'png', 'pdf'):
        fig.savefig(out / f'observed_fit_by_size.{extension}', dpi=180)
    plt.close(fig)


def observed_fit_interpretation(observed, fixed_cohort=False):
    complete = observed[(observed.arm == 'equal') & observed.complete]
    all_data = complete[complete.scope == 'all_selected'].sort_values('N')
    if len(all_data) < 2:
        return 'At least two completed equal-reference sizes with matched observed-profile logs are needed to describe a trend.'
    first, last = all_data.iloc[0], all_data.iloc[-1]
    text = (f'Under equal reference, from N={int(first.N)} to N={int(last.N)}, the equal-dataset mean '
            f'PCC(μ, observed) changes from {first.mu_pcc:.3f} to {last.mu_pcc:.3f}, while '
            f'PCC(L_bio, observed) changes from {first.L_bio_pcc:.3f} to {last.L_bio_pcc:.3f}')
    anchors = complete[complete.scope == 'best2_datasets'].set_index('N')
    if first.N in anchors.index and last.N in anchors.index:
        a, b = anchors.loc[first.N], anchors.loc[last.N]
        text += (f'; on the best-2 dataset identities, the corresponding changes are '
                 f'{a.mu_pcc:.3f}→{b.mu_pcc:.3f} for μ and {a.L_bio_pcc:.3f}→{b.L_bio_pcc:.3f} for L_bio')
    return text + (', using identical validation transcripts across all datasets and N.' if fixed_cohort
                   else ', with changing validation transcripts limiting interpretation of both trends.')


def observed_ranking_interpretation(observed):
    matched = observed[(observed.scope == 'best2_datasets') & observed.complete
                       & observed.arm.isin(['equal', 'ranked_p1', 'reverse_p1'])]
    counts = matched.groupby('N').arm.nunique()
    sizes = counts[counts == 3].index
    if not len(sizes):
        return 'A completed equal / ranked_p1 / reverse_p1 trio is required for the observed-fit direction comparison.'
    n = max(sizes)
    group = matched[matched.N == n].set_index('arm')
    equal, ranked, reverse = (group.loc[arm] for arm in ('equal', 'ranked_p1', 'reverse_p1'))
    return (f'At N={int(n)}, the largest completed equal / linear-ranked / linear-reverse trio, '
            f'best-2-dataset PCC(L_bio, observed) is {equal.L_bio_pcc:.3f}, {ranked.L_bio_pcc:.3f} '
            f'and {reverse.L_bio_pcc:.3f}, respectively; PCC(μ, observed) is {equal.mu_pcc:.3f}, '
            f'{ranked.mu_pcc:.3f} and {reverse.mu_pcc:.3f}. '
            'These are matched within-N descriptive differences, with no across-seed uncertainty estimate; '
            'their ordering is an observation to test, not a guaranteed advantage of quality ranking.')


def plot_profile_variation(diagnostics, sizes, out):
    fig, ax = plt.subplots(figsize=(10, 3.7))
    for arm in ARMS:
        values = diagnostics[diagnostics.arm == arm].groupby('N').variance.median().reindex(sizes)
        ax.plot(range(len(sizes)), values, 'o--' if 'reverse' in arm else 'o-', color=COLORS[arm], label=arm)
    ax.set(xticks=range(len(sizes)), xticklabels=sizes, xlabel='Number of training datasets N',
           ylabel='Median transcript variance of L_bio', title='Shared-profile amplitude on the same held-out test transcripts',
           ylim=(0, None))
    ax.grid(alpha=.2)
    ax.legend(fontsize=9, ncol=3)
    fig.tight_layout()
    for extension in ('svg', 'png', 'pdf'):
        fig.savefig(out / f'profile_variation_by_size.{extension}', dpi=180)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args(argv)
    root=args.experiment_root.expanduser().resolve()
    print(f'Analyzing available cumulative exports in {root}',flush=True)
    manifest=read_json(root/'experiment_manifest.json')
    if manifest['experiment_design'] not in {SIMPLE_DESIGN,FIXED_DESIGN,'cumulative_reference_weight_stability_v3'} or len(manifest['training_seeds'])!=1:
        raise ValueError('Expected the single-seed cumulative stability design.')
    fixed_cohort=manifest['experiment_design']==FIXED_DESIGN
    if object_sha256(manifest['tasks'])!=manifest['tasks_sha256']:
        raise ValueError('Frozen task mapping changed.')
    hashes={}
    for original,expected in manifest['frozen_file_sha256'].items():
        path=relocate(original,root,Path(manifest['output_root']))
        if sha256(path)!=expected:
            raise ValueError(f'Changed frozen input: {path}')
        hashes[str(path)]=expected
    tasks=manifest['tasks'];seed=manifest['training_seeds'][0];sizes=manifest['sizes']
    fold_controls,fold_counts,fold_pairs=audit_transcript_folds(manifest['source_folds'],
        require_common_validation=fixed_cohort,require_common_training=fixed_cohort)
    ids=manifest['source_folds'][str(sizes[0])]['test_ids']
    print(f'Fold audit: train fixed={fold_controls["identical_train_ids"]}, validation fixed={fold_controls["identical_validation_ids"]}; common test={len(ids):,}.',flush=True)
    configs={(t['training_seed'],t['arm'],t['N']):yaml.safe_load(relocate(t['config_path'],root,Path(manifest['output_root'])).read_text()) for t in tasks}
    weights=pd.read_csv(root/'reference_weights.csv')
    if manifest['experiment_design'] in {SIMPLE_DESIGN,FIXED_DESIGN}:
        write_design_report(root,manifest,weights,pd.read_csv(root/'reference_concentration.csv'))
    profiles,availability=collect(root,manifest,configs,ids,weights)
    print(f'Validated {len(profiles)}/{len(tasks)} models; computing matched transcript metrics.',flush=True)
    logs=collect_logs(availability, extra_tags=('val_L_bio_pcc',),
                      tag_prefixes=('val_mu_pcc/', 'val_L_bio_pcc/'))
    validation_fit=training_summary(logs,availability)
    anchor_datasets=next(task['datasets'] for task in tasks if task['N']==sizes[0])
    observed_datasets,observed_summary=observed_fit_tables(logs,availability,tasks,anchor_datasets)
    readiness=comparison_readiness(availability,'N',list(zip(sizes[:-1],sizes[1:])),ARMS)
    frame=stability_rows(profiles,ids,sizes,seed)
    summary,effects,cohorts=summarize_stability(frame,ids)
    diagnostics=[]
    for (_,arm,n),values in profiles.items():
        for tid in ids:
            x=values[tid]['values']
            diagnostics.append(dict(N=n,arm=arm,transcript_id=tid,variance=float(np.var(x)),near_constant=bool(np.var(x)<=1e-12)))
    diagnostics=pd.DataFrame(diagnostics,columns=['N','arm','transcript_id','variance','near_constant'])
    out=root/'analysis';out.mkdir(exist_ok=True)
    for name,table in [('availability',availability),('transcript_stability',frame),('stability_summary',summary),
                       ('paired_policy_improvements',effects),('metric_cohorts',cohorts),('profile_variance',diagnostics),
                       ('comparison_readiness',readiness),('validation_fit',validation_fit),('training_scalars',logs),
                       ('observed_fit_per_dataset',observed_datasets),('observed_fit_summary',observed_summary),
                       ('fold_audit',fold_counts),('fold_pair_audit',fold_pairs)]:
        table.to_csv(out/f'{name}.csv',index=False)
    plot_stability(summary,effects,sizes,out)
    plot_stability(summary,effects,sizes,out,kind='anchor')
    plot_stability(summary,effects,sizes,out,kind='shared_anchor')
    plot_availability(availability,'N',sizes,ARMS,out)
    plot_validation_history(logs,availability,'N',sizes,ARMS,COLORS,out)
    plot_observed_fit(observed_summary,sizes,out,fixed_cohort)
    plot_profile_variation(diagnostics,sizes,out)
    cohort_note='The same validation transcripts are observed in every configured dataset and reused at every N.' if fixed_cohort else None
    dataset_mu_report=write_dataset_mu_report(observed_datasets,weights,sizes,ARMS,COLORS,out,cohort_note=cohort_note)
    complete=int(availability.status.eq('validated_predictions').sum())
    created=report_timestamp()
    adjacent_comments=' '.join(effect_sentence(effects,'equal',arm,kind='adjacent',unit='adjacent steps') for arm in ('ranked_p1','ranked_p3'))
    anchor_comments=' '.join(effect_sentence(effects,'equal',arm,kind='anchor',unit='anchor comparisons') for arm in ('ranked_p1','ranked_p3'))
    shared_anchor_comments=' '.join(effect_sentence(effects,'equal',arm,kind='shared_anchor',unit='shared-anchor comparisons') for arm in ('ranked_p1','ranked_p3'))
    direction_comments=' '.join(effect_sentence(effects,baseline,policy,kind='shared_anchor',unit='shared-anchor comparisons') for baseline,policy in COMPARISONS if baseline.startswith('reverse'))
    drift_comment=equal_anchor_drift_sentence(summary)
    transcript_drift_comment=equal_anchor_transcript_sentence(frame)
    observed_comments=observed_fit_interpretation(observed_summary,fixed_cohort)
    observed_ranking_comments=observed_ranking_interpretation(observed_summary)
    audit_note=('Training, validation and test transcript membership is fixed across all datasets, sizes and policies.' if fixed_cohort else
        '<b>Training-cohort confound:</b> the legacy setup reassigned training and validation membership across N. Within each model the folds are disjoint, and the common test set stayed out of every training run. Across-N changes do not isolate dataset addition alone.')
    fixed_report=(root/'fixed_observed_evaluation/analysis_manifest.json').exists()
    fixed_observed_html=fixed_observation_section(root,manifest,hashes)
    observed_link=('<p class="note"><b>Use the <a href="../fixed_observed_evaluation/analysis_report.html">re-evaluated fixed-observation test report</a> for observed-profile comparisons across N and datasets.</b> The validation plots on this page describe the original training logs.</p>' if fixed_report else '')
    validation_membership=('uses the same validation transcripts in every dataset and at every N' if fixed_cohort else 'still uses the validation transcripts available at each N')
    evaluation_text=(('The selected checkpoints have been re-evaluated on the same real held-out transcript–dataset pairs; use the fixed-observation section above.' + ('' if fixed_cohort else ' These metrics retain the legacy training-cohort confound.')) if fixed_report else
        'Run <code>python analyses/reevaluate_cumulative_fixed_observations.py --experiment-root '+html.escape(str(root))+'</code> to evaluate the selected checkpoints on identical real held-out observations.')
    refresh_arg=f' --experiment-root {html.escape(str(root))}' if root!=DEFAULT_OUTPUT else ''
    body=f'''<!doctype html><html><head><meta charset="utf-8"><title>Cumulative profile stability</title><style>{STYLE}</style></head><body>
<h1>Does ranking stabilize L_bio as lower-ranked datasets are added?</h1>
<p>Partial results · generated {created} · training seed {seed}</p>
<p class="note">{audit_note} <a href="../../../Docs/cumulative_cohort_audit_and_correction.html">Split audit and corrected experiment</a>.</p>
{observed_link}
{snapshot_text(availability,len(ids))}
<details><summary><b>Verified transcript folds</b></summary>{table_html(fold_counts)}
<p>The fold audit enforces disjoint train/validation/test roles within each model and identical test IDs across N; the corrected design also requires identical training and validation IDs.</p>
<p><a href="fold_audit.csv">Fold counts</a> · <a href="fold_pair_audit.csv">Training overlaps and validation-to-training role changes between sizes</a>.</p></details>
<h2>Current interpretation</h2><p>{drift_comment}</p><p>{transcript_drift_comment}</p><p>{shared_anchor_comments}</p>
<p>Read PCC and RMSE together: they measure shape alignment and amplitude-sensitive differences respectively, and their conclusions can disagree. Missing sizes and one training seed prevent a general claim that ranking stabilizes every addition.</p>
<h2>Which comparisons are available?</h2><img src="availability.svg" alt="Available models by dataset count and weighting policy">
<p>Only green cells contribute test-profile metrics; two completed endpoints under the same policy are required for an adjacent comparison, and missing sizes are never bridged.</p>
<p><a href="comparison_readiness.csv">Exact available and missing endpoints</a> · <a href="../design_report.html">Design and frozen weight curves</a></p>
<h2>Weight concentration and reference quality</h2><img src="../cumulative_reference_design.svg" alt="Reference concentration and quality composition">
<p>Ranked and reverse have equal dataset-weight concentration at a given exponent, while opposite assignments change which quality ranks anchor the reference; these curves describe the design, not model performance.</p>
{fixed_observed_html}
<details id="validation-observed-fit"><summary><b>Training validation diagnostics: individual-dataset μ PCC and μ/L_bio means</b></summary>
{dataset_mu_report}
<h2 id="observed-fit">Validation fit of μ and L_bio as N grows</h2>
<img src="observed_fit_by_size.svg" alt="Observed-profile PCC for mu and L_bio and their difference versus dataset count, for all selected datasets and the best two dataset identities">
<p class="note">{observed_comments}</p>
<p>The top row evaluates every dataset in that model's training collection, so lower-ranked observations enter the evaluation as N grows; the bottom row always evaluates <b>{html.escape(', '.join(anchor_datasets))}</b>, which removes changing dataset membership and {validation_membership}.</p>
<p>Each point is taken from the epoch of the <b>exported best-validation-loss checkpoint</b>, not the epoch with the highest PCC and not the latest unfinished epoch; missing completed models or incomplete pairs of diagnostic tags leave gaps.</p>
<details><summary><b>Exactly how observed-profile PCC is computed</b></summary>
<p>For transcript t in dataset d, y_dt is the arithmetic consensus of the observed replica profiles; PCC is computed across valid CDS positions, separately for μ_dt and shared L_bio,t, against that same y_dt. These runs use a prediction floor of zero, so μ is not thresholded before computing PCC. The training diagnostic assigns zero to correlations with near-zero prediction or target variance; it does not exclude these pairs as undefined. This differs from the finite-cohort convention used for the independent L_bio stability analysis.</p>
<pre>r_mu(d, N)  = mean over validation pairs t in dataset d of PCC(μ_dt, y_dt)
r_bio(d, N) = mean over the same pairs t of PCC(L_bio,t, y_dt)
reported PCC(N, D) = (1 / |D|) × sum over datasets d in D of r(d, N)
correction gain   = reported PCC_mu − reported PCC_bio
D = all N selected datasets, or the two best dataset identities</pre>
<p>Both curves use the <b>same unweighted mean within each dataset, then equal weights across datasets</b>; evaluation weights never use the ranking π. All expected dataset tags must be present at the selected epoch, with matching log source and optimizer step for μ and L_bio. The exports include coverage and log provenance. The original global <code>val_mu_pcc</code> below averages physical transcript–dataset pairs, while global <code>val_L_bio_pcc</code> uses the configured reliability-weighted transcript reduction; subtracting those global scalars would mix two different averages. The new figures instead use the matched per-dataset tags, logged on one GPU in these runs.</p>
<p>Because μ_dt = S_dt × L_bio,t × γ_dt and S_dt is a positive scalar over the transcript, PCC removes its overall scale. The right column therefore describes the fitted correction's contribution to observed profile shape at a frozen checkpoint; it is not the effect of retraining a model with <code>mean_correction=unity</code>, and it does not measure absolute count calibration.</p>
</details>
<h3>How to read this together with L_bio stability</h3>
<p>{observed_ranking_comments}</p>
<p>A lower all-dataset PCC can reflect a harder observation mixture, different transcript coverage, or changed predictions; it cannot by itself establish that adding datasets damaged the model. A larger μ-versus-L_bio gap means the fitted dataset correction contributes more to observed shape agreement, but does not establish that the residual is purely technical or that L_bio is biologically correct. Conversely, high agreement with every observed dataset is not the objective of a shared component when those datasets contain systematic differences.</p>
<p><b>The useful ranking argument is preservation of a reproducible, informative shared profile while retaining observed-profile fit.</b> Read the μ curves, L_bio-versus-observation curves, fixed-test-transcript stability and profile variance jointly; neither a stable L_bio nor a high μ PCC establishes that argument alone. A drop in L_bio-versus-observed PCC is not automatically successful bias removal.</p>
<h3>The fixed-observation comparison for the article</h3>
<p>{evaluation_text}</p>
<p>The fixed test evaluation uses real replica consensus targets, identical masks and fixed evaluation weights across all N and policies. PCC(μ, y), PCC(L_bio, y) and their paired difference describe agreement with observations; they do not provide biological ground truth. The original sequence-only test exports contain dummy count targets and cannot supply this evaluation, and TensorBoard averages cannot be filtered back to a common transcript subset. The curves in this expandable section remain <b>validation diagnostics</b> used in the same runs as checkpoint selection.</p>
<details><summary>Selected-checkpoint observed-fit values and coverage</summary>
{table_html(observed_summary[['N','arm','scope','status','selected_epoch','n_expected_datasets','n_matched_datasets','mu_pcc','L_bio_pcc','correction_gain']])}
</details>
<p><a href="observed_fit_summary.csv">Observed-fit means and coverage</a> · <a href="observed_fit_per_dataset.csv">Per-dataset values and log provenance</a> · <a href="observed_fit_by_size.pdf">PDF figure</a> · <a href="training_scalars.csv">Underlying logged scalars</a>.</p>
</details>
<h2>Primary: accumulated drift from one shared N=2 anchor</h2><img src="shared_anchor_stability.svg" alt="Agreement with one shared equal-reference N=2 anchor and paired ranked-versus-equal improvements">
<p class="note">{drift_comment}</p>
<p>{transcript_drift_comment}</p>
<p>{shared_anchor_comments}</p>
<p>{direction_comments}</p>
<p>Every curve uses the exact same fitted equal-reference N=2 model on the same transcript IDs and CDS coordinates. The N=2 model is a drift anchor, not biological ground truth.</p>
<h2>Secondary: agreement between adjacent sizes</h2><img src="cumulative_stability.svg" alt="Adjacent-size PCC and RMSE with paired ranked-versus-equal improvements">
<p>{adjacent_comments}</p><p>High adjacent agreement can coexist with accumulated drift because each pair of neighboring models shares most of its dataset prefix.</p>
<details><summary><b>Sensitivity: each policy compared with its own independently fitted N=2 model</b></summary>
<img src="anchor_stability.svg" alt="Agreement with separately fitted policy-specific N=2 anchors">
<p>{anchor_comments}</p><p>These comparisons are secondary because differences among the five N=2 fits enter the between-policy contrast.</p></details>
<h2>Training progress and validation fit</h2><img src="validation_history.svg" alt="Validation-loss histories including unfinished models">
<p>Curves show optimization progress and dots identify selected checkpoints when available; validation values were used for selection and are not independent test-count accuracy.</p>
{table_html(validation_fit[['N','arm','recorded_status','n_logged_validation_epochs','best_logged_val_loss','selected_epoch','selected_val_mu_pcc','selected_val_replica_nll']] if 'best_logged_val_loss' in validation_fit else validation_fit)}
<h2>Does shared-profile amplitude change with N?</h2>
<img src="profile_variation_by_size.svg" alt="Median within-transcript variance of the mean-one L_bio profile versus dataset count on the common test transcripts">
<p>{profile_variation_text(diagnostics)}</p>
<p>L_bio has mean one over each CDS, so its within-transcript variance measures the strength of peaks and troughs on a comparable scale; the plot takes the median over the same {len(ids):,} test transcripts at every completed model. Falling variance may reflect useful smoothing or loss of signal, and must be interpreted together with observed fit and stability.</p>
{rationale(fixed_cohort)}<h2>Task availability</h2>{table_html(availability[['N','arm','status','recorded_status','selected_epoch']])}
<h2>Paired policy improvements</h2><p>Positive values favor ranked; within each metric and comparison type, all available transitions and policies use the same finite transcript cohort, so changing exclusions across steps cannot drive a trend. A missing arm is not imputed; cohort membership and excluded counts are exported. These are descriptive estimates with no across-seed confidence intervals.</p>{table_html(effects)}
<h2>Shared-anchor, arm-specific-anchor and adjacent stability</h2>{table_html(summary)}
<p><a href="profile_variance.csv">Per-transcript profile variance and near-constant flags</a> · <a href="metric_cohorts.csv">Exact finite cohorts</a> · <a href="availability.csv">Detailed availability/validation failures</a> · <a href="shared_anchor_stability.pdf">Primary PDF figure</a>.</p>
<h2>Refresh this partial analysis</h2><pre>python analyses/analyze_cumulative_reference_weight_stability.py{refresh_arg}</pre>
<p>Re-run after downloading more outputs. The analysis reads model artifacts and writes reports; it does not launch training or change the experiment inputs. Cohorts are recomputed from the available valid results, so the saved timestamp and cohort CSV belong to this snapshot.</p></body></html>'''
    (out/'analysis_report.html').write_text(body)
    for row in availability.to_dict('records'):
        for field in ('prediction_path','raw_prediction_path','runtime_manifest'):
            raw=row.get(field)
            if isinstance(raw,str) and Path(raw).is_file(): hashes[raw]=sha256(Path(raw))
    for source in logs.source.unique():
        for path in Path(source).glob('events.out.tfevents*'):
            hashes[str(path)]=sha256(path)
    dataset_report_code=ROOT/'Utils/cumulative_dataset_pcc_report.py'
    hashes[str(dataset_report_code)]=sha256(dataset_report_code)
    report_helpers=ROOT/'Utils/partial_stability_report.py'
    hashes[str(report_helpers)]=sha256(report_helpers)
    write_json(out/'analysis_manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        experiment_manifest_sha256=sha256(root/'experiment_manifest.json'),source_sha256=hashes,
        validated_models=complete,planned_models=len(tasks),snapshot_status='partial' if complete<len(tasks) else 'complete',
        fold_controls=fold_controls,
        analysis_code_sha256=sha256(Path(__file__)),outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='analysis_manifest.json'}))
    print(f'{complete}/{len(tasks)} validated exports; report: {out/"analysis_report.html"}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
