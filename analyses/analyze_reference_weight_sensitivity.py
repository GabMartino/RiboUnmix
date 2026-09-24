#!/usr/bin/env python3
"""Descriptive, single-seed policy sensitivity on frozen sequence-only exports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_reference_weight_sensitivity import CONTRASTS, DEFAULT_OUTPUT, DESIGN, object_sha256, read_json, sha256, write_json
from analyses.analyze_real_exp8_reference_directionality import collect, pair_record
from analyses.analyze_rank_balanced_reference_directionality import relocate


def reference_design_table(weights, source):
    """Compare frozen old prefixes with new weights, without training outcomes."""
    lookup=weights[weights.arm=='equal'].set_index('dataset_id')
    rows=[]
    def append(design,n,arm,power,names,pi):
        ranks=lookup.loc[names,'global_rank'].to_numpy(float)
        pi=np.asarray(pi,dtype=float)
        rows.append(dict(design=design,N=n,arm=arm,power=power,
                         N_ref=1/(pi@pi),reference_fraction=1/(n*(pi@pi)),
                         weighted_mean_rank=pi@ranks))
    for task in source['tasks']:
        if task['training_seed']!=42 or task['arm']!='equal':
            continue
        names=task['datasets']
        q=lookup.loc[names,'original_q'].to_numpy(float)
        # Source names retain frozen ascending (global rank, dataset ID) order.
        rank_order=lookup.loc[names].reset_index().sort_values(['global_rank','dataset_id']).dataset_id.tolist()
        if names!=rank_order:
            raise ValueError('Source prefix is not in frozen rank order.')
        for arm,raw in [('equal',np.ones(len(names))),('ranked',q),('reverse',q[::-1])]:
            append('previous',len(names),arm,0 if arm=='equal' else 1,names,raw/raw.sum())
    for arm,group in weights.groupby('arm',sort=False):
        append('new',len(group),arm,int(group.power.iloc[0]),group.dataset_id.tolist(),group.pi)
    return pd.DataFrame(rows)


def plot_reference_comparison(weights, source_path, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    comparison=reference_design_table(weights,read_json(source_path))
    comparison.to_csv(out/'reference_design_comparison.csv',index=False)
    previous=comparison[comparison.design=='previous']
    current=comparison[comparison.design=='new'].set_index('arm')
    sizes=sorted(previous.N.unique())
    colors={'equal':'#64748b','ranked':'#267eab','reverse':'#9a519b'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    metrics=[('reference_fraction','Weight concentration','Effective reference fraction  N_ref / N'),
             ('weighted_mean_rank','Reference quality composition','Weighted global rank (1 = best)')]
    for ax,(metric,title,ylabel) in zip(axes,metrics):
        ax.axvspan(len(sizes)-1.4,len(sizes)+.4,color='#edf4fa',zorder=0)
        for arm,color in colors.items():
            part=previous[previous.arm==arm].set_index('N').loc[sizes]
            ax.plot(range(len(sizes)),part[metric],color=color,
                    linestyle='--' if arm=='reverse' else '-',marker='o',markersize=4,label=arm.capitalize())
            new_arm=arm if arm=='equal' else arm+'_p3'
            final=float(current.loc[new_arm,metric])
            # This final segment changes exponent at fixed N, not collection size.
            ax.plot([len(sizes)-1,len(sizes)],[part[metric].iloc[-1],final],color=color,
                    linestyle=':',linewidth=1.4)
            ax.plot(len(sizes),final,marker='D',markersize=8,color=color,
                    markerfacecolor='none' if arm=='reverse' else color,zorder=4)
        ax.set(title=title,ylabel=ylabel,xlim=(-.25,len(sizes)+.4))
        ax.set_xticks(range(len(sizes)+1),[str(n) for n in sizes]+[f'{sizes[-1]}\nq³'])
        ax.set_xlabel('Previous cumulative sizes (q¹) → stronger weights at N = 114')
        ax.grid(axis='y',alpha=.2)
        ax.text(.98,.96,'N = 114 fixed',transform=ax.transAxes,ha='right',va='top',fontsize=9,color='#365874')
    axes[0].set_ylim(0,1.12)
    axes[0].text(.03,.10,'Lower fraction = more concentrated\nRanked / reverse coincide',transform=axes[0].transAxes,fontsize=9)
    fraction_label=f"{current.loc['ranked_p1','reference_fraction']:.3f} → {current.loc['ranked_p3','reference_fraction']:.3f}"
    axes[0].annotate(fraction_label,xy=(len(sizes),current.loc['ranked_p3','reference_fraction']),
                     xytext=(len(sizes)-3,.29),fontsize=10,arrowprops=dict(arrowstyle='->',color='#52667a'))
    axes[1].set_ylim(0,120)
    axes[1].legend(loc='upper left',frameon=False,fontsize=9)
    for arm,offset in [('ranked',-16),('reverse',-15)]:
        y=current.loc[arm+'_p3','weighted_mean_rank']
        axes[1].annotate(f'{y:.2f}',(len(sizes),y),xytext=(-6,offset),textcoords='offset points',ha='right',color=colors[arm],fontsize=10)
    fig.suptitle('Previous cumulative design versus the new N = 114 weighting-strength test',fontsize=13)
    fig.text(.5,.015,'Previous N = 114 q¹ and new N = 114 q¹ are identical; diamonds add q³. These are design weights, not training results.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    for extension in ('svg','png','pdf'):
        fig.savefig(out/f'reference_design_comparison.{extension}',dpi=180)
    plt.close(fig)
    summary=current.reset_index()[['arm','N_ref','reference_fraction','weighted_mean_rank']].rename(
        columns={'arm':'Policy at N = 114','N_ref':'Effective reference count',
                 'reference_fraction':'N_ref / N','weighted_mean_rank':'Weighted global rank'})
    return ('''<section id="reference-design-comparison"><h2>Weight concentration and reference quality: previous versus new</h2>
<img src="reference_design_comparison.svg" alt="Previous cumulative reference concentration and weighted global rank, followed by the new cubic weighting at fixed N=114">
<p class="note">At fixed N = 114, q³ concentrates reference weight more strongly and separates ranked from reverse quality composition further, while the new q¹ policies exactly reproduce the previous N = 114 reference weights.</p>
<p>The previous cumulative points all use the linear score mapping; the final diamond changes only the exponent at the same 114 datasets, and the two shaded positions isolate that comparison. Ranked and reverse always have identical dataset-weight concentration. N_ref / N = 1 means uniform weights, while smaller values mean more concentration; weighted global rank summarizes which datasets anchor the reference, with smaller ranks indicating higher QC ranking.</p>
<p><b>Should the model results differ more?</b> The stronger q³ perturbation gives more opportunity to reveal sensitivity, but neither these curves nor the larger dataset count guarantee a larger profile difference. At N = 114 the q¹ policy itself is unchanged, and any rerun differences there can involve optimization or implementation changes. Compare q³ ranked versus reverse within the new matched runs, then compare that discrepancy with their q¹ counterparts.</p>
<!--REFERENCE_SUMMARY-->
<p><a href="reference_design_comparison.csv">Exact comparison table</a> · <a href="reference_design_comparison.pdf">PDF figure</a> · <a href="reference_design_comparison.png">PNG figure</a>; all cumulative curves are reconstructed from the frozen source membership and global scores, including planned sizes without completed training.</p></section>'''
        .replace('<!--REFERENCE_SUMMARY-->', '<div class="table">'+summary.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')+'</div>'))


def compare_profiles(profiles, tasks, ids):
    """Retain undefined correlations and their reasons instead of imputing them."""
    task_by_arm = {task['arm']:task for task in tasks}
    rows = []
    for a,b in CONTRASTS:
        ta,tb = task_by_arm[a],task_by_arm[b]
        ka,kb = (ta['training_seed'],a,ta['N']), (tb['training_seed'],b,tb['N'])
        if ka not in profiles or kb not in profiles:
            continue
        for tid in ids:
            rows.append(dict(arm_a=a,arm_b=b,transcript_id=tid,
                             **pair_record(profiles[ka][tid],profiles[kb][tid],'full_cds')))
    return pd.DataFrame(rows, columns=['arm_a','arm_b','transcript_id','n_positions','PCC','Spearman','RMSE','variance_a','variance_b','reason'])


def summarize(frame):
    rows=[]
    for (a,b),group in frame.groupby(['arm_a','arm_b'],sort=False):
        for metric in ('PCC','RMSE','variance_a','variance_b'):
            values=group.loc[np.isfinite(group[metric]),metric]
            rows.append(dict(arm_a=a,arm_b=b,metric=metric,n_valid=len(values),n_excluded=len(group)-len(values),
                             median=values.median(),mean=values.mean()))
    return pd.DataFrame(rows,columns=['arm_a','arm_b','metric','n_valid','n_excluded','median','mean'])


def strength_effect(frame):
    """Paired change in discrepancy, with the same transcript in both strengths."""
    pairs=[]
    for power in (1,3):
        pairs.append(frame[(frame.arm_a==f'ranked_p{power}') & (frame.arm_b==f'reverse_p{power}')].set_index('transcript_id'))
    a,b=pairs
    common=a.index.intersection(b.index)
    delta=pd.DataFrame(index=common)
    delta['delta_one_minus_PCC']=(a.loc[common,'PCC']-b.loc[common,'PCC']).astype(float)
    delta['delta_RMSE']=(b.loc[common,'RMSE']-a.loc[common,'RMSE']).astype(float)
    return delta.reset_index()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root',type=Path,default=ROOT/'results/reference_weight_sensitivity_N114_seed42')
    args=parser.parse_args(argv)
    root=args.experiment_root.expanduser().resolve()
    manifest=read_json(root/'experiment_manifest.json')
    if manifest['experiment_design'] not in {DESIGN, 'fixed_panel_reference_weight_sensitivity_v1'}:
        raise ValueError('Expected a single-seed reference-weight sensitivity experiment.')
    if object_sha256(manifest['tasks'])!=manifest['tasks_sha256']:
        raise ValueError('Frozen task mapping changed.')
    source_hashes={}
    for original,expected in manifest['frozen_file_sha256'].items():
        local=relocate(original,root,Path(manifest['output_root']))
        if sha256(local)!=expected:
            raise ValueError(f'Frozen design artifact changed: {local}')
        source_hashes[str(local)]=expected
    tasks=manifest['tasks']
    configs={}
    for task in tasks:
        path=relocate(task['config_path'],root,Path(manifest['output_root']))
        configs[task['training_seed'],task['arm'],task['N']]=yaml.safe_load(path.read_text())
    ids=manifest['source_folds'][str(tasks[0]['N'])]['test_ids']
    weights=pd.read_csv(root/'reference_weights.csv')
    profiles,availability=collect(root,manifest,configs,ids,weights)
    details=compare_profiles(profiles,tasks,ids)
    summary=summarize(details)
    delta=strength_effect(details)
    out=root/'analysis'
    out.mkdir(exist_ok=True)
    reference_html=''
    source_path=root/'frozen_inputs/source_experiment_manifest.json'
    if manifest['experiment_design']==DESIGN and source_path.is_file():
        reference_html=plot_reference_comparison(weights,source_path,out)
    tables={'availability':availability,'transcript_policy_sensitivity':details,
            'policy_sensitivity_summary':summary,'paired_strength_effect':delta}
    for name,table in tables.items():
        table.to_csv(out/f'{name}.csv',index=False)
    plot_html=''
    if not details.empty:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,metric in zip(axes,('PCC','RMSE')):
            values,labels=[],[]
            for power in (1,3):
                group=details[(details.arm_a==f'ranked_p{power}') & (details.arm_b==f'reverse_p{power}')]
                finite=group.loc[np.isfinite(group[metric]),metric].to_numpy()
                if finite.size:
                    values.append(1-finite if metric=='PCC' else finite)
                    labels.append(f'p={power}\nn={len(finite)}')
            if values:
                ax.boxplot(values,tick_labels=labels,showfliers=False)
            ax.set(ylabel='1 − PCC' if metric=='PCC' else 'RMSE',title='Ranked versus reverse')
            ax.grid(axis='y',alpha=.2)
        fig.tight_layout()
        fig.savefig(out/'direction_sensitivity.svg')
        plt.close(fig)
        plot_html='<img src="direction_sensitivity.svg" alt="Per-transcript discrepancies between ranked and reverse profiles at p=1 and p=3"><p>Higher values indicate greater disagreement between the two reference directions; boxes summarize transcripts within this single trained seed, with outliers omitted from the drawing but retained in the tables.</p>'
    delta_summary=[]
    for metric in ('delta_one_minus_PCC','delta_RMSE'):
        finite=delta.loc[np.isfinite(delta[metric]),metric]
        delta_summary.append(dict(metric=metric,n_paired=len(finite),n_excluded=len(delta)-len(finite),
                                  median=finite.median(),mean=finite.mean()))
    delta_table=pd.DataFrame(delta_summary)
    delta_table.to_csv(out/'strength_effect_summary.csv',index=False)
    completed=int(availability.status.eq('validated_predictions').sum())
    def table(frame):
        return '<div class="table">'+frame.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div>'
    report=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reference-weight sensitivity results</title>
<style>body{{max-width:1100px;margin:35px auto;padding:0 22px;font:16px/1.6 system-ui;color:#203448}}.table{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:6px 10px;border-bottom:1px solid #ddd;text-align:right}}img{{max-width:100%}}.note{{padding:14px;background:#f3f7fb;border-left:4px solid #267eab}}</style></head><body>
<h1>Reference-weight sensitivity · N = {tasks[0]['N']}</h1><p class="note">{completed}/5 validated exports · seed {manifest['training_seeds'][0]} · {len(ids)} frozen test transcripts.</p>
<p><a href="../design_report.html">Experiment design and planned weight contrast</a></p>
{reference_html}
<h2>Availability</h2>{table(availability[['arm','status','recorded_status','selected_epoch']])}
<p>Only completed, validated best-validation-loss exports enter comparisons; missing or invalid policies are not imputed. Detailed validation failures are recorded in <a href="availability.csv">availability.csv</a>.</p>
<h2>Direction sensitivity</h2>{plot_html}<p>The primary contrast is ranked_p3 versus reverse_p3; p=1 is the prespecified current-strength comparison. PCC measures shape agreement, while RMSE and variance retain normalized profile amplitude differences. These are comparisons between models, not accuracy against biological truth.</p>
{table(summary)}
<h2>Does the stronger weighting increase the difference?</h2><p>For each transcript, subtract its p=1 discrepancy from its p=3 discrepancy; positive Δ(1 − PCC) or ΔRMSE indicates greater disagreement at p=3. Both strengths must be available, and each metric uses only paired finite values.</p>{table(delta_table)}
<p class="note">One seed cannot establish reproducibility across initializations; transcript counts do not create independent training replicates. Differences may also reflect optimization trajectories or convergence. No confidence intervals across training runs are estimated.</p>
<p>L_bio is mean-one normalized: its variance and RMSE concern profile shape/amplitude, not absolute abundance; observation-dependent fields in sequence-only exports must not be treated as held-out reconstruction targets. Aggregate source-family weight concentration can differ between ranked and reverse despite identical dataset-weight concentration.</p>
<p>Read the full <a href="transcript_policy_sensitivity.csv">per-transcript metrics</a> and <a href="paired_strength_effect.csv">paired strength effects</a>; regenerate with <code>python analyses/analyze_reference_weight_sensitivity.py --experiment-root {html.escape(str(root))}</code>.</p></body></html>'''
    (out/'analysis_report.html').write_text(report)
    for row in availability.to_dict('records'):
        for key in ('prediction_path','raw_prediction_path','runtime_manifest'):
            raw=row.get(key)
            if isinstance(raw,str) and Path(raw).is_file():
                source_hashes[raw]=sha256(Path(raw))
    write_json(out/'analysis_manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        experiment_manifest_sha256=sha256(root/'experiment_manifest.json'), source_sha256=source_hashes,
        analysis_code_sha256=sha256(Path(__file__)),
        outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='analysis_manifest.json'}))
    print(f'{completed}/5 validated exports; report: {out / "analysis_report.html"}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
