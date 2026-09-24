"""Observed-QC design audit. No prediction, checkpoint, or performance readers."""
from __future__ import annotations

import hashlib
import html
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import shlex
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import yaml

from Utils.real_panel_convergence import assert_panel_partition, infer_source_identifier, write_json, utc_timestamp
from Utils.external_transcript_split import load_external_transcript_split
from Utils.publication_plot_style import LATEX_PAPER_RC

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_QC = ['log1p_median_read_density', 'median_positive_codon_coverage',
               'number_of_eligible_transcripts', 'median_replica_PCC']
RAW_COMPONENTS = {'rank_periodicity': 'frame0_cds_fraction_total',
    'rank_cds_enrichment': 'cds_psite_fraction_total', 'rank_depth': 'log10_cds_psite_reads',
    'rank_transcript_support': 'profile_n_detected_transcripts',
    'rank_rpf_length_center': 'rpf_length_center_distance_nt',
    'rank_rpf_length_spread': 'rpf_length_iqr_median'}
COLORS = ['#0072B2', '#56B4E9', '#E69F00', '#D55E00']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def software_provenance():
    return dict(python=platform.python_version(), platform=platform.platform(),
        packages={p: importlib.metadata.version(p) for p in
                  ('numpy', 'pandas', 'scipy', 'matplotlib', 'torch', 'lightning', 'hydra-core')},
        cpu_only=True)


def aliases_from(path):
    aliases = read_json(path) if path else {}
    if not isinstance(aliases, dict) or not all(isinstance(k, str) and isinstance(v, str) for k,v in aliases.items()):
        raise ValueError('Alias map must be a JSON object of alias -> canonical dataset ID.')
    if any(v in aliases and aliases[v] != v for v in aliases.values()):
        raise ValueError('Chained/ambiguous aliases are not supported; specify canonical destinations directly.')
    return aliases


def canonical_ids(values, aliases, label):
    values = list(values)
    if any(pd.isna(x) or not str(x).strip() for x in values):
        raise ValueError(f'{label}: empty dataset identifiers.')
    names = [aliases.get(str(x), str(x)) for x in values]
    duplicates = pd.Series(names)[pd.Series(names).duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise ValueError(f'{label}: duplicate/ambiguous canonical IDs: {duplicates}')
    return names


def global_quality_groups(ranks):
    values = np.asarray(ranks, float)
    if not np.isfinite(values).all():
        raise ValueError('Global groups require all retained ranks; missing ranks cannot be imputed.')
    boundaries = np.quantile(values, [.25, .5, .75], method='linear')
    labels = np.searchsorted(boundaries, values, side='left') + 1
    groups = []
    for group in range(1,5):
        selected = values[labels == group]
        groups.append(dict(group=group, count=int(len(selected)),
            observed_min=float(selected.min()) if len(selected) else None,
            observed_max=float(selected.max()) if len(selected) else None))
    return labels, dict(quantiles=[.25,.5,.75], boundaries=boundaries.tolist(),
        convention='NumPy linear empirical quantiles over all retained global ranks; (-inf,b1], (b1,b2], (b2,b3], (b3,inf). Ties stay together in the lower group at a boundary.',
        ranks_recomputed=False, groups=groups, unequal_group_sizes=len({g['count'] for g in groups}) > 1)


def load_inputs(args, tracked, issues):
    def record(path):
        path = Path(path)
        tracked[str(path)] = sha256(path)
        return path
    panel_manifest = read_json(record(args.panel_manifest))
    aliases = aliases_from(record(args.alias_map) if args.alias_map else None)
    panels = {str(p): canonical_ids(names, aliases, f'panel {p}')
              for p,names in panel_manifest['panels'].items()}
    if len(panels) != 4:
        raise ValueError(f'Expected four experimental panels, found {len(panels)}.')
    universe = canonical_ids(panel_manifest.get('retained_datasets', sum(list(panels.values()), [])), aliases, 'retained universe')
    names = sum(list(panels.values()), [])
    canonical_ids(names, {}, 'panel union')
    if set(names) != set(universe):
        raise ValueError(f'Panel universe mismatch: missing={sorted(set(universe)-set(names))}, extra={sorted(set(names)-set(universe))}.')
    sources = {}
    def add_sources(frame, id_col, source_col, label):
        ids = canonical_ids(frame[id_col], aliases, label)
        for name, source in zip(ids, frame[source_col]):
            if pd.isna(source) or not str(source).strip():
                continue
            if name in sources and sources[name] != str(source):
                raise ValueError(f'Ambiguous source family for {name}: {sources[name]} vs {source} ({label}).')
            sources[name] = str(source)
    frozen_quality = pd.DataFrame(panel_manifest.get('dataset_quality_summaries', []))
    if {'dataset_name','source_identifier'} <= set(frozen_quality):
        add_sources(frozen_quality, 'dataset_name', 'source_identifier', 'frozen manifest')
    quality = pd.read_csv(record(args.dataset_quality_table)) if args.dataset_quality_table else frozen_quality
    if {'dataset_name','source_identifier'} <= set(quality):
        add_sources(quality, 'dataset_name', 'source_identifier', 'quality table')
    if args.source_family_mapping:
        mapping = pd.read_csv(record(args.source_family_mapping))
        add_sources(mapping, 'dataset_id', 'source_family', 'explicit source mapping')
    missing_sources = sorted(set(names)-set(sources))
    if missing_sources and panel_manifest.get('source_family_partition', {}).get('fallback') == 'author_year prefix inferred from dataset_name':
        sources.update({d: infer_source_identifier(d) for d in missing_sources})
        issues.append(dict(severity='limitation', detail=f'Frozen source fallback rule reused for {missing_sources}.'))
    elif missing_sources:
        raise ValueError(f'Missing source mappings: {missing_sources}; supply --source-family-mapping.')
    table = pd.DataFrame([dict(dataset_id=d, panel_id=p, source_family=sources[d], dataset_order=i)
                         for p,ds in panels.items() for i,d in enumerate(ds)])
    assignment = table.rename(columns={'dataset_id':'dataset_name', 'panel_id':'panel','source_family':'source_identifier'})
    assert_panel_partition(assignment, expected_datasets=universe)
    frozen_sources = panel_manifest.get('panel_source_families')
    if frozen_sources and any(set(frozen_sources[p]) != set(table.loc[table.panel_id==p, 'source_family']) for p in panels):
        raise ValueError('Source mapping disagrees with frozen panel_source_families.')

    ranking = pd.read_csv(record(args.ranking_table), sep='\t')
    rcol, dcol = args.ranking_rank_column, args.ranking_dataset_column
    ranking_ids = canonical_ids(ranking[dcol], aliases, 'global ranking')
    numeric_ranks = pd.to_numeric(ranking[rcol], errors='coerce').to_numpy(float)
    invalid = ~np.isfinite(numeric_ranks) | (numeric_ranks <= 0)
    if invalid.any():
        raise ValueError(f'Unusable global ranks (no imputation): {np.asarray(ranking_ids)[invalid].tolist()}.')
    # Reuse the production loader before any retained-universe filtering.
    from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import load_dataset_quality_ranking
    old_ranks, old_q = load_dataset_quality_ranking(str(args.ranking_table), dataset_column=dcol, rank_column=rcol)
    ranks = {new: old_ranks[str(old)] for old,new in zip(ranking[dcol], ranking_ids)}
    q = {new: old_q[str(old)] for old,new in zip(ranking[dcol], ranking_ids)}
    ranking = ranking.assign(**{dcol:ranking_ids})
    missing = sorted(set(names)-set(ranks))
    unmatched = sorted(set(ranks)-set(names))
    table['global_rank'] = table.dataset_id.map(ranks)
    table['raw_reference_score_q'] = table.dataset_id.map(q)
    table['rank_missing'] = table.global_rank.isna()
    table['ranking_sha256'] = tracked[str(args.ranking_table)]
    components = [c for c in ranking if c.startswith('rank_') and c not in ('rank_component_count',rcol)]
    extra_rank_columns = [c for c in ranking if c not in (dcol, rcol)]
    table = table.merge(ranking[[dcol,*extra_rank_columns]].rename(columns={dcol:'dataset_id'}), on='dataset_id', how='left', validate='one_to_one')
    if not quality.empty:
        quality = quality.copy()
        quality['dataset_id'] = canonical_ids(quality['dataset_name'], aliases, 'observed quality')
        extra = [c for c in quality if c not in table and c not in ('dataset_name','source_identifier')]
        table = table.merge(quality[['dataset_id',*extra]], on='dataset_id', how='left', validate='one_to_one')
        table['observed_qc_missing'] = ~table.dataset_id.isin(quality.dataset_id)
    else:
        table['observed_qc_missing'] = True
    if missing:
        issues.append(dict(severity='hard_error', detail=f'Missing global ranks (no imputation): {missing}'))
    metadata = dict(path=str(args.ranking_table), sha256=tracked[str(args.ranking_table)],
        dataset_column=dcol, rank_column=rcol, direction='1 = best',
        component_columns=components, component_count=len(components),
        declared_component_counts=sorted(ranking.rank_component_count.dropna().unique().tolist()) if 'rank_component_count' in ranking else [],
        global_table_rows=len(ranking), global_rank_max=max(ranks.values()),
        missing_retained_ids=missing, global_ids_not_retained=unmatched,
        alias_mapping=aliases, raw_qc_columns=[c for c in RAW_COMPONENTS.values() if c in ranking],
        conversion='Production load_dataset_quality_ranking: q=(R-r+1)/R using max rank of complete input; pi=q/sum_panel(q); p=1.',
        transcript_scope='not verified', component_input_data='not verified',
        frozen_before_historical_training='not verified',
        rerun_freeze='Preparation copies and hashes the complete ranking before generating commands.')
    if components and 'quality_rank_score' in ranking:
        component_sum=ranking[components].apply(pd.to_numeric,errors='coerce').sum(axis=1,min_count=len(components))
        metadata['component_rank_sum_matches_saved_score']=bool(np.allclose(component_sum,pd.to_numeric(ranking.quality_rank_score,errors='coerce'),equal_nan=False))
        metadata['component_aggregation_note']='Component-rank sum checked against saved score; the supplied global rank remains authoritative, never recalculated.'
        if not metadata['component_rank_sum_matches_saved_score']:
            issues.append(dict(severity='discrepancy',detail='Available component-rank sum does not reproduce the saved composite score; global ranks were not changed. Verify the documented aggregation/missing-component rule.'))
    if args.ranking_provenance:
        metadata['supplied_provenance'] = read_json(record(args.ranking_provenance))
        metadata['supplied_provenance_status'] = 'User-supplied documentary claims; not independently verified.'
    split = read_json(record(args.common_split_manifest)) if args.common_split_manifest else None
    if split is not None:
        for panel,ds in panels.items():
            load_external_transcript_split(args.common_split_manifest, panel_name=panel, experiment_datasets=ds)
    return table, panels, panel_manifest, split, metadata, components


def add_weights(table):
    table = table.copy()
    q = table.raw_reference_score_q.to_numpy(float)
    if not np.isfinite(q).all() or (q <= 0).any():
        raise ValueError('q must be positive and finite for every retained dataset.')
    table['pi_equal'] = 1/table.groupby('panel_id').dataset_id.transform('size')
    table['pi_ranked'] = table.raw_reference_score_q/table.groupby('panel_id').raw_reference_score_q.transform('sum')
    for policy in ('equal','ranked'):
        if not (table[f'pi_{policy}'] > 0).all():
            raise ValueError('Reference weights are not strictly positive.')
        np.testing.assert_allclose(table.groupby('panel_id')[f'pi_{policy}'].sum(), 1, atol=1e-12)
    return table


def check_saved_weights(saved, table, aliases, duplicate_policy='error', origin='provided CSV'):
    saved = saved.rename(columns={'dataset':'dataset_id','panel':'panel_id'}).copy()
    keys = ['policy','panel_id','dataset_id']
    if not set([*keys,'pi']) <= set(saved):
        raise ValueError('Saved weight table needs policy,panel,dataset,pi (or panel_id,dataset_id).')
    saved.dataset_id = saved.dataset_id.map(lambda x: aliases.get(str(x), str(x)))
    saved.policy = saved.policy.replace({'quality_rank':'ranked'})
    if not set(saved.policy) <= {'equal','ranked'}:
        raise ValueError(f'Unknown weight policies: {sorted(set(saved.policy))}')
    duplicate_rows = int(saved.duplicated(keys, keep=False).sum())
    if duplicate_rows:
        conflicting = saved.groupby(keys).pi.nunique(dropna=False).gt(1).any()
        if duplicate_policy == 'error' or conflicting:
            raise ValueError(f'Duplicate policy-level weights ({duplicate_rows} rows, conflicting={conflicting}); use collapse-identical only for exact duplicate values.')
        saved = saved.drop_duplicates(keys)
    checks = []
    for policy in sorted(saved.policy.unique()):
        expected = table[['dataset_id','panel_id',f'pi_{policy}']].rename(columns={f'pi_{policy}':'expected_pi'})
        merged = expected.merge(saved.loc[saved.policy==policy, [*keys[1:],'pi']], on=keys[1:], how='outer', indicator=True)
        for row in merged.to_dict('records'):
            observed, expected_pi = row['pi'], row['expected_pi']
            ok = row['_merge']=='both' and np.isfinite(observed) and observed>0 and np.isclose(observed,expected_pi,rtol=1e-10,atol=1e-12)
            checks.append(dict(source=origin,policy=policy,panel_id=row['panel_id'],dataset_id=row['dataset_id'],
                saved_pi=observed,reconstructed_pi=expected_pi,difference=observed-expected_pi,
                status='match' if ok else 'mismatch',duplicate_rows=duplicate_rows,
                duplicate_policy=duplicate_policy,identity_status=row['_merge']))
    return pd.DataFrame(checks)


def summarize(table, features):
    ranks, counts, mass, balance, concentration, sources, curves = [], [], [], [], [], [], []
    for panel, frame in table.groupby('panel_id', sort=True):
        values = frame.global_rank.to_numpy(float)
        quantiles = np.quantile(values, [0,.25,.5,.75,1], method='linear')
        ranks.append(dict(panel_id=panel,n_datasets=len(frame),n_sources=frame.source_family.nunique(),
            rank_min=quantiles[0],rank_q25=quantiles[1],rank_median=quantiles[2],rank_q75=quantiles[3],
            rank_max=quantiles[4],rank_mean=values.mean(),mean_q=frame.raw_reference_score_q.mean()))
        for group in range(1,5):
            n = int((frame.global_quality_group==group).sum())
            counts.append(dict(panel_id=panel,group=group,n=n,proportion=n/len(frame)))
        for policy in ('equal','ranked'):
            weight = frame[f'pi_{policy}'].to_numpy(float)
            for group in range(1,5):
                mass.append(dict(panel_id=panel,policy=policy,group=group,
                    reference_mass=weight[frame.global_quality_group.to_numpy()==group].sum()))
            sm = frame.groupby('source_family')[f'pi_{policy}'].sum().sort_values(ascending=False)
            top = frame.sort_values([f'pi_{policy}','dataset_id'],ascending=[False,True])
            concentration.append(dict(panel_id=panel,policy=policy,N_ref=1/np.square(weight).sum(),
                maximum_dataset_weight=weight.max(),largest_datasets=';'.join(sorted(frame.loc[np.isclose(weight,weight.max(),rtol=1e-12,atol=0),'dataset_id'])),
                maximum_source_mass=sm.max(),largest_sources=';'.join(sorted(sm.index[np.isclose(sm,sm.max(),rtol=1e-12,atol=0)])),
                reference_weighted_mean_rank=float(weight@values),
                reference_weighted_mean_q=float(weight@frame.raw_reference_score_q.to_numpy(float))))
            for source, value in sm.items():
                sources.append(dict(panel_id=panel,policy=policy,source_family=source,source_mass=value,
                    dataset_count=int((frame.source_family==source).sum())))
            for i, row in enumerate(top.to_dict('records')):
                curves.append(dict(panel_id=panel,policy=policy,position=i+1,dataset_id=row['dataset_id'],
                    pi=row[f'pi_{policy}'],cumulative_mass=float(top[f'pi_{policy}'].iloc[:i+1].sum())))
            full_weights = np.ones(len(table)) if policy=='equal' else table.raw_reference_score_q.to_numpy(float)
            for feature in features:
                full = pd.to_numeric(table[feature],errors='coerce').to_numpy(float)
                x = pd.to_numeric(frame[feature],errors='coerce').to_numpy(float)
                valid, full_valid = np.isfinite(x), np.isfinite(full)
                scale = np.nanstd(np.where(full_valid,full,np.nan),ddof=0) if full_valid.any() else np.nan
                target = np.average(full[full_valid],weights=full_weights[full_valid]) if full_valid.any() else np.nan
                mean = np.average(x[valid],weights=weight[valid]) if valid.any() else np.nan
                qs = np.quantile(x[valid],[0,.25,.5,.75,1],method='linear') if valid.any() else [np.nan]*5
                balance.append(dict(panel_id=panel,policy=policy,feature=feature,
                    feature_type='component_rank' if feature.startswith('rank_') else 'observed_qc',
                    n_valid=int(valid.sum()),n_missing=int((~valid).sum()),missing_reference_mass=float(weight[~valid].sum()),
                    minimum=qs[0],q25=qs[1],median=qs[2],q75=qs[3],maximum=qs[4],
                    weighted_mean=mean,full_collection_target=target,
                    standardization_sd=scale,standardized_difference=(mean-target)/scale if scale>0 else np.nan))
    outputs = dict(panel_rank_summary=pd.DataFrame(ranks),panel_quality_group_counts=pd.DataFrame(counts),
        panel_quality_group_reference_mass=pd.DataFrame(mass),panel_component_balance=pd.DataFrame(balance,columns=[
            'panel_id','policy','feature','feature_type','n_valid','n_missing','missing_reference_mass',
            'minimum','q25','median','q75','maximum','weighted_mean','full_collection_target',
            'standardization_sd','standardized_difference']),
        panel_concentration_summary=pd.DataFrame(concentration),source_family_reference_mass=pd.DataFrame(sources),
        reference_concentration_curves=pd.DataFrame(curves))
    discrepancies = []
    for left,right in itertools.combinations(sorted(table.panel_id.unique()),2):
        a,b = [table.loc[table.panel_id==p] for p in (left,right)]
        grid = np.sort(table.global_rank.unique())
        cdf_a = (a.global_rank.to_numpy()[:,None] <= grid).mean(axis=0)
        cdf_b = (b.global_rank.to_numpy()[:,None] <= grid).mean(axis=0)
        discrepancies.append(dict(panel_a=left,panel_b=right,feature='global_rank_ECDF',
            discrepancy=float(np.abs(cdf_a-cdf_b).max()),definition='Maximum absolute unweighted ECDF difference; descriptive, no p-value.'))
        discrepancies.append(dict(panel_a=left,panel_b=right,feature='global_rank_Wasserstein',
            discrepancy=float(wasserstein_distance(a.global_rank,b.global_rank)),definition='Unweighted first Wasserstein distance, rank units.'))
        for feature in features:
            x,y = [pd.to_numeric(f[feature],errors='coerce') for f in (a,b)]
            scale = pd.to_numeric(table[feature],errors='coerce').std(ddof=0)
            discrepancies.append(dict(panel_a=left,panel_b=right,feature=feature,
                discrepancy=(x.mean()-y.mean())/scale if scale>0 else np.nan,
                definition='Unweighted mean difference / full retained collection observed-value population SD (ddof=0).'))
        for g in range(1,5):
            discrepancies.append(dict(panel_a=left,panel_b=right,feature=f'quality_group_{g}',
                discrepancy=float((a.global_quality_group==g).mean()-(b.global_quality_group==g).mean()),definition='Difference in dataset proportions.'))
    outputs['panel_pair_design_discrepancies'] = pd.DataFrame(discrepancies)
    return outputs


def plot_audit(table, summaries, groups, args, destination):
    destination.mkdir(parents=True, exist_ok=True)
    style = {**LATEX_PAPER_RC, 'font.size':9, 'axes.labelsize':9, 'axes.titlesize':11,
             'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
             'text.usetex':not args.no_tex}
    if args.no_tex:
        style['font.serif'] = ['DejaVu Serif']
    panels = sorted(table.panel_id.unique())
    labels = [f'P{i+1:02}' for i in range(len(panels))]
    def save(fig, name):
        for fmt in ('svg','pdf'):
            fig.savefig(destination/f'{name}.{fmt}',bbox_inches='tight')
        fig.savefig(destination/f'{name}.png',dpi=160,bbox_inches='tight')  # inspection preview only
        plt.close(fig)
    with matplotlib.rc_context(style):
        fig, ax = plt.subplots(figsize=(6.5,3.2),layout='constrained')
        rng = np.random.default_rng(args.jitter_seed)
        points=[]
        for i,panel in enumerate(panels):
            frame = table.loc[table.panel_id==panel].sort_values('dataset_id')
            x=i+rng.uniform(-.17,.17,len(frame))
            points.extend(dict(dataset_id=d,panel_id=panel,plot_x=float(px),global_rank=float(r))
                          for d,px,r in zip(frame.dataset_id,x,frame.global_rank))
            ax.scatter(x,frame.global_rank,s=19,
                color=COLORS[i],alpha=.8,linewidths=.25,edgecolors='white')
            qs = np.quantile(frame.global_rank,[.25,.5,.75],method='linear')
            ax.plot([i,i],[qs[0],qs[2]],color='black',lw=3,solid_capstyle='butt')
            ax.plot([i-.10,i+.10],[qs[1],qs[1]],color='black',lw=2)
        ax.set(xticks=range(4),xticklabels=[f'{l}\n(n={int((table.panel_id==p).sum())})' for l,p in zip(labels,panels)],
            ylabel='Global quality rank (1 = best)',title='A. Global rank distribution')
        ax.set_ylim(args._global_rank_max+2, -1)
        ax.set_yticks(sorted(set([1,*range(20,int(args._global_rank_max),20),args._global_rank_max])))
        ax.grid(axis='y',alpha=.35)
        pd.DataFrame(points).to_csv(destination.parent/'figure_A_point_positions.csv',index=False)
        save(fig,'A_global_rank_distribution')

        fig, ax = plt.subplots(figsize=(7.3,3.2),layout='constrained')
        m = summaries['panel_quality_group_reference_mass']
        for i,panel in enumerate(panels):
            for offset,policy in [(-.19,'equal'),(.19,'ranked')]:
                bottom = 0
                for g in range(1,5):
                    value = m.loc[(m.panel_id==panel)&(m.policy==policy)&(m.group==g),'reference_mass'].item()
                    ax.bar(i+offset,value,bottom=bottom,width=.34,color=COLORS[g-1],edgecolor='white',lw=.3,
                        label=f'G{g}: ranks {groups["groups"][g-1]["observed_min"]:g}--{groups["groups"][g-1]["observed_max"]:g}' if i==0 and policy=='equal' and groups['groups'][g-1]['count'] else None)
                    bottom += value
        ax.set(ylim=(0,1),ylabel='Gamma-reference mass',title='B. Reference mass by global quality group',
               xticks=[i+d for i in range(4) for d in (-.19,.19)],xticklabels=['Equal','Ranked']*4)
        for i,label in enumerate(labels):
            ax.text(i,-.20,label,ha='center',transform=ax.get_xaxis_transform())
        ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),title='G1 = highest quality')
        save(fig,'B_quality_group_reference_mass')

        b = summaries['panel_component_balance']
        features = list(dict.fromkeys(b.feature))
        fig,axes = plt.subplots(1,2,figsize=(7.8,max(3.4,.28*len(features))),layout='constrained',sharey=True)
        maximum = max(1,float(b.standardized_difference.abs().max()))
        cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#d0d0d0')
        for ax,policy in zip(axes,('equal','ranked')):
            if not features:
                ax.text(.5,.5,'No component or raw QC columns available',ha='center',va='center',transform=ax.transAxes,wrap=True)
                ax.set_axis_off()
                continue
            subset = b.loc[b.policy==policy]
            z = subset.pivot(index='feature',columns='panel_id',values='standardized_difference').reindex(index=features,columns=panels)
            missing = subset.pivot(index='feature',columns='panel_id',values='n_missing').reindex(index=features,columns=panels)
            im = ax.pcolormesh(np.arange(5)-.5,np.arange(len(features)+1)-.5,
                np.ma.masked_invalid(z.to_numpy()),cmap=cmap,vmin=-maximum,vmax=maximum,
                shading='flat',rasterized=False)
            ax.set_ylim(len(features)-.5,-.5)
            ax.set_facecolor('#d0d0d0')
            for j in range(len(features)):
                for i in range(4):
                    if missing.iloc[j,i]:
                        ax.text(i,j,f'm={int(missing.iloc[j,i])}',ha='center',va='center',fontsize=8)
            ax.set(xticks=range(4),xticklabels=labels,yticks=range(len(features)),
                yticklabels=[x.replace('_',' ') for x in features],title='Unweighted composition' if policy=='equal' else r'$\pi$-weighted composition')
        if features:
            colorbar=fig.colorbar(im,ax=axes,label='Difference from same-policy full collection / population SD',shrink=.7)
            colorbar.solids.set_rasterized(False)
        fig.suptitle('C. QC composition (m = missing datasets; gray = undefined)')
        save(fig,'C_qc_component_balance')

        fig,axes = plt.subplots(2,2,figsize=(7.4,4.4),layout='constrained',sharey=True)
        c = summaries['reference_concentration_curves']; conc = summaries['panel_concentration_summary']
        for ax,panel,label in zip(axes.flat,panels,labels):
            annotations = []
            for policy,color in [('equal','#777777'),('ranked','#0072B2')]:
                curve = c.loc[(c.panel_id==panel)&(c.policy==policy)]
                ax.plot([0,*curve.position],[0,*curve.cumulative_mass],color=color,label=policy.capitalize(),lw=1.4)
                row = conc.loc[(conc.panel_id==panel)&(conc.policy==policy)].iloc[0]
                annotations.append(f'{policy.capitalize()}: '+r'$N_{\rm ref}$'+f'={row.N_ref:.1f}, max source={row.maximum_source_mass:.1%}')
            ax.set(title=label,xlabel='Datasets, sorted by reference weight',ylabel='Cumulative reference mass',ylim=(0,1.03))
            annotation = '\n'.join(annotations)
            if not args.no_tex:
                annotation = annotation.replace('%',r'\%')
            ax.text(.97,.07,annotation,transform=ax.transAxes,ha='right',va='bottom',fontsize=9)
        axes[0,0].legend(loc='upper left')
        fig.suptitle('D. Reference concentration (design diagnostic)')
        save(fig,'D_reference_concentration')


def provenance_audit(args, panels, metadata, tracked, issues):
    configs, reliability, historical_weights = {}, {}, []
    for panel,names in panels.items():
        base = args.config_root/panel
        candidates = [base/'hydra/.hydra/config.yaml', base/'resolved_config.yaml']
        path = next((p for p in candidates if p.is_file()),None)
        if path:
            tracked[str(path)] = sha256(path)
            configs[panel] = dict(path=str(path),config=yaml.safe_load(path.read_text()))
            cfg=configs[panel]['config']
            selected=canonical_ids(cfg['experiment']['dataset'],metadata['alias_mapping'],f'{panel} configuration')
            gamma=cfg['model']['gamma_centering'];reference=gamma['reference']
            reference_names=reference.get('dataset_names')
            reference_names=selected if reference_names is None else canonical_ids(reference_names,metadata['alias_mapping'],f'{panel} reference')
            if selected!=names or set(reference_names)!=set(names) or gamma['mode']!='fixed_reference':
                issues.append(dict(severity='hard_error',detail=f'{panel}: resolved training configuration does not preserve frozen dataset ordering and the full-panel fixed gamma-reference universe.'))
            if reference['weighting']=='quality_rank' and reference['quality_rank_power']!=1.:
                issues.append(dict(severity='hard_error',detail=f'{panel}: resolved ranked reference power is not the audited p=1 policy.'))
            configs[panel]['reference_definition']=dict(mode=gamma['mode'],dataset_names=reference_names,
                weighting=reference['weighting'],quality_rank_power=reference.get('quality_rank_power'),
                ranking_configuration=cfg['data'].get('dataset_quality_ranking'))
        path = args.reliability_root/panel/'reliability_reference_manifest.json'
        if path.is_file():
            tracked[str(path)] = sha256(path)
            reliability[panel] = dict(path=str(path),manifest=read_json(path))
        path = args.panel_manifest.parent/panel/'run_manifest.json'
        if path.is_file():
            tracked[str(path)] = sha256(path)
            run=read_json(path)
            ref = run.get('fixed_gamma_reference',{})
            if ref and (canonical_ids(run['selected_datasets'],metadata['alias_mapping'],str(path))!=names or
                        set(canonical_ids(ref['dataset_names'],metadata['alias_mapping'],str(path)))!=set(names)):
                issues.append(dict(severity='hard_error',detail=f'{panel}: run manifest reference universe differs from its frozen panel.'))
            policy = {'equal':'equal','quality_rank':'ranked'}.get(ref.get('weighting'))
            if policy:
                historical_weights.extend(dict(policy=policy,panel=panel,dataset=d,pi=v,source=str(path)) for d,v in ref['pi'].items())
    if args.historical_ranked_root:
        root = args.historical_ranked_root
        path = root/'panel_manifest.json'; tracked[str(path)] = sha256(path)
        historical = read_json(path)
        strategy = historical.get('gamma_reference_strategy',{})
        metadata['historical_ranked_strategy'] = strategy
        metadata['historical_recorded_ranking_hash_matches'] = strategy.get('ranking_table_sha256') == metadata['sha256']
        if not metadata['historical_recorded_ranking_hash_matches']:
            issues.append(dict(severity='limitation',detail='Requested ranking differs from historical ranked provenance; it is a new weighting definition, not substituted historical weights.'))
        if historical.get('panels') != panels:
            issues.append(dict(severity='hard_error',detail='Historical ranked panel memberships/order differ from the audited manifest.'))
        elif metadata['historical_recorded_ranking_hash_matches']:
            for panel in panels:
                path = root/panel/'run_manifest.json'; tracked[str(path)] = sha256(path)
                run=read_json(path);ref=run['fixed_gamma_reference']
                if run['selected_datasets']!=panels[panel] or set(ref['dataset_names'])!=set(panels[panel]):
                    issues.append(dict(severity='hard_error',detail=f'{panel}: historical ranked reference universe differs from its frozen panel.'))
                if ref.get('quality_rank_power') != 1.:
                    issues.append(dict(severity='hard_error',detail=f'{panel}: historical reference power is not one.'))
                historical_weights.extend(dict(policy='ranked',panel=panel,dataset=d,pi=v,source=str(path)) for d,v in ref['pi'].items())
    return configs,reliability,pd.DataFrame(historical_weights)


def write_report(out, manifest, summaries):
    esc = html.escape
    rank = manifest.get('ranking',{})
    sections = ['<h1>Four-panel dataset-QC and gamma-reference audit</h1>',
        '<p>CPU-only design diagnostics. No prediction arrays, checkpoint weights, or performance tables were loaded. '
        'Dataset QC is not biological ground truth; pi is the fixed gamma-reference weight; w_dt is the separate local loss reliability weight. '
        'Pi is not multiplied into w_dt or directly into the training loss.</p>',
        '<h2>Inputs and ranking definition</h2><pre>'+esc(json.dumps(rank,indent=2))+'</pre>',
        '<h2>Validation and limitations</h2>'+pd.DataFrame(manifest['issues'],columns=['severity','detail']).to_html(index=False,escape=True),
        '<p>Component ranks are lower-is-better QC summaries. Raw QC values are reported only when present in the inputs. '
        'The original balancing variables were log(1+read density), positive-codon coverage, transcript support and replica PCC. '
        'This is not the same feature set as the composite QC ranking.</p>']
    if summaries:
        ranks = summaries['panel_rank_summary']; counts = summaries['panel_quality_group_counts']; mass = summaries['panel_quality_group_reference_mass']
        concentration = summaries['panel_concentration_summary']; balance = summaries['panel_component_balance']
        q1 = counts.loc[counts.group==1].sort_values('proportion',ascending=False)
        top = q1.iloc[0]; bottom=q1.iloc[-1]
        discrepancy=summaries['panel_pair_design_discrepancies']
        component_differences=discrepancy.loc[discrepancy.feature.str.startswith('rank_')].dropna(subset=['discrepancy'])
        component_answer='No usable component-rank contrast is available; component-level comparability is not verified.'
        if not component_differences.empty:
            contrast=component_differences.loc[component_differences.discrepancy.abs().idxmax()]
            component_answer=(f'The largest pairwise component-rank mean difference is {esc(contrast.feature)} between '
                f'{esc(contrast.panel_a)} and {esc(contrast.panel_b)}: {contrast.discrepancy:+.3f} full-retained population SD. '
                'Component-level differences remain even when composite-rank summaries appear similar.')
        ranked_concentration=concentration.loc[concentration.policy=='ranked']
        largest_source=ranked_concentration.loc[ranked_concentration.maximum_source_mass.idxmax()]
        source_rows=summaries['source_family_reference_mass']
        source_equal=source_rows.loc[(source_rows.policy=='equal')&(source_rows.panel_id==largest_source.panel_id)&
            source_rows.source_family.isin(largest_source.largest_sources.split(';')),'source_mass'].max()
        sections.append('<h2>Answers from the saved design</h2><ul>'+''.join('<li>'+x+'</li>' for x in [
            f'Highest-quality group G1: {esc(top.panel_id)} contains {int(top.n)} datasets ({top.proportion:.1%}); '
            f'{esc(bottom.panel_id)} contains {int(bottom.n)} ({bottom.proportion:.1%}). These are descriptive composition differences, not performance effects.',
            f'Every panel contains all four global quality groups: {bool(counts.n.gt(0).all())}. The exact rank ranges below show which parts of the full range are absent; spanning four groups does not mean spanning every rank.',
            'Ranked reference mass in G1 ranges from '+f'{mass.loc[(mass.policy=="ranked")&(mass.group==1),"reference_mass"].min():.1%} to '
            f'{mass.loc[(mass.policy=="ranked")&(mass.group==1),"reference_mass"].max():.1%} across panels. Equal and ranked masses use the same frozen group boundaries.',
            f'Ranked-reference mean rank ranges from {ranked_concentration.reference_weighted_mean_rank.min():.2f} to '
            f'{ranked_concentration.reference_weighted_mean_rank.max():.2f}, while N_ref ranges from {ranked_concentration.N_ref.min():.2f} to '
            f'{ranked_concentration.N_ref.max():.2f}. Similar concentration is not similar quality composition.',
            component_answer,
            f'The largest ranked source share is {esc(largest_source.largest_sources)} in {esc(largest_source.panel_id)}: '
            f'{largest_source.maximum_source_mass:.1%}, versus {source_equal:.1%} for the same source under equal weighting. '
            f'Across all panels the maximum individual ranked dataset weight is {ranked_concentration.maximum_dataset_weight.max():.1%}. '
            'No threshold is used to declare domination, and no equal-per-source reweighting was applied.',
        ])+'</ul>')
        for title,key in [('Rank distribution','panel_rank_summary'),('Global quality groups (dataset counts)','panel_quality_group_counts'),
                          ('Reference mass by global group','panel_quality_group_reference_mass'),('Dataset/source concentration','panel_concentration_summary')]:
            display=summaries[key].copy()
            if key=='panel_concentration_summary':
                display.loc[display.policy=='equal','largest_datasets']='All panel datasets (tied equal weights)'
            sections.append(f'<h2>{title}</h2>'+display.to_html(index=False,float_format=lambda x:f'{x:.4f}',escape=True))
        sections.append('<p>Figure A: one point per dataset, black span = 25th–75th percentiles, black horizontal line = median; '
            'no truncation of global ranks. Exact jitter coordinates are in figure_A_point_positions.csv. '
            'See <a href="source_family_reference_mass.csv">all source-family shares</a> and '
            '<a href="panel_concentration_summary.csv">unabridged concentration identities</a>.</p>')
        sections.append('<h2>QC-component differences and missingness</h2>'+balance.to_html(index=False,float_format=lambda x:f'{x:.3f}',escape=True))
        sections.append('<p>Standardization denominator: population SD (ddof=0) of observed values over all retained datasets, fixed across panels and policies. '
            'Targets are the full retained collection under uniform dataset weights for equal and under the same frozen q for ranked. '
            'Available-value means are renormalized over observed entries; missing counts and missing reference mass are explicit. '
            'Constant/all-missing features have undefined standardized differences. No nonsignificance-based balance claims or independent-dataset tests are used.</p>')
        for path in sorted((out/'figures').glob('*.svg')):
            sections.append(f'<h2>{esc(path.stem.replace("_"," "))}</h2>'+path.read_text()[path.read_text().index('<svg'):])
    sections += ['<h2>Interpretation and rerun choices</h2><p>N_ref = 1/sum(pi²) describes weight concentration, not independent experiments. '
        'Related conditions belong to source families. Similar N_ref does not imply similar quality. '
        'No default threshold declares balance or domination; optional warnings are explicitly prespecified heuristics.</p>',
        '<p>Existing-panel reruns isolate reference policy within the old composition; they do not repair across-panel quality imbalance. '
        'A QC/rank-balanced repartition is a separate sensitivity design and requires matched fresh equal and ranked models. '
        'Do not select a partition, ranking or seed from model performance. Both preparation modes audit first and never train.</p>',
        '<h2>Historical equal-control reuse</h2><pre>'+esc(json.dumps(manifest.get('historical_control_reuse'),indent=2))+'</pre>',
        '<h2>Provenance and regeneration</h2><pre>'+esc(json.dumps({k:manifest[k] for k in ('command','software','input_sha256','source_sha256')},indent=2))+'</pre>']
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Panel reference quality audit</title><style>body{font:15px/1.5 system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#172536}h2{margin-top:2em}table{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto}th,td{padding:5px 9px;border:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f6f8;padding:16px}svg{width:100%;height:auto}</style><body>'+''.join(sections)+'</body></html>'
    (out/'audit_report.html').write_text(document,encoding='utf-8')


def run_audit(args):
    out = args.output_root
    protected = [args.panel_manifest.parent,args.config_root,args.reliability_root]
    if args.historical_ranked_root:
        protected.append(args.historical_ranked_root)
    if any(out==p or out.is_relative_to(p) for p in protected):
        raise ValueError('Use a separate new --output-root, outside historical input directories.')
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory is not empty: {out}. Choose a new root; existing audit outputs are not overwritten.')
    out.mkdir(parents=True,exist_ok=True)
    tracked,issues,summaries = {},[],{}
    manifest = dict(audit_version=1,created_at_utc=utc_timestamp(),mode=args.mode,cpu_only=True,training_launched=False,
        performance_data_loaded=False,command=shlex.join([sys.executable,*sys.argv]),
        software=software_provenance(),input_sha256=tracked,issues=issues,
        seeds=dict(jitter=args.jitter_seed,partition=args.partition_seed,training=args.training_seed),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'analyses/audit_four_panel_reference_quality.py',Path(__file__),
            ROOT/'Utils/real_panel_convergence.py',ROOT/'Utils/external_transcript_split.py',
            ROOT/'Dataloaders/RiboUnmixMultiDataset/RiboUnmixMultiDatasetDataModule.py']},
        warning_heuristics={'source_mass':args.source_mass_warning,'selection_uses_warning_threshold':False})
    table=panels=split=metadata=panel_manifest=configs=reliability=None
    try:
        table,panels,panel_manifest,split,metadata,components = load_inputs(args,tracked,issues)
        manifest['ranking'] = metadata
        actual = dict(datasets=len(table),source_families=table.source_family.nunique(),
                      panel_sizes=table.groupby('panel_id').size().tolist(),global_rank_max=metadata['global_rank_max'])
        expected = dict(datasets=114,source_families=85,panel_sizes=[29,29,28,28],global_rank_max=115)
        manifest['documented_expectations'] = expected; manifest['observed_design'] = actual
        for key in actual:
            if actual[key] != expected[key]:
                issues.append(dict(severity='discrepancy',detail=f'{key}: documented={expected[key]}, observed={actual[key]}; not repaired.'))
        configs,reliability,historical_weights = provenance_audit(args,panels,metadata,tracked,issues)
        manifest['resolved_config_sources'] = {p:c['path'] for p,c in configs.items()}
        manifest['resolved_reference_definitions'] = {p:c['reference_definition'] for p,c in configs.items()}
        manifest['historical_control_reuse'] = dict(status='unverified',fresh_equal_controls_required=True,
            historical_git=panel_manifest.get('git'),
            reason='No complete historical source snapshot, package environment and cryptographic data identity linked to training were established. Matching configurations or current data hashes alone do not prove historical implementation identity.')
        manifest['local_reliability_provenance'] = {p:{k:r['manifest'].get(k) for k in
            ('reference_split','heldout_rows_used_for_fitting','panel_training_transcript_id_hash')} for p,r in reliability.items()}
        if not any(x['severity']=='hard_error' for x in issues):
            table = add_weights(table)
            table['global_quality_group'],groups = global_quality_groups(table.global_rank)
            write_json(out/'global_quality_group_definitions.json',groups)
            checks=[]
            if not historical_weights.empty:
                for source,frame in historical_weights.groupby('source',sort=True):
                    # Each run manifest supplies exactly its own panel's weights.
                    checks.append(check_saved_weights(frame,table.loc[table.panel_id.isin(frame.panel)],metadata['alias_mapping'],origin=source))
            if args.reference_weights:
                tracked[str(args.reference_weights)] = sha256(args.reference_weights)
                checks.append(check_saved_weights(pd.read_csv(args.reference_weights),table,metadata['alias_mapping'],args.duplicate_weight_policy,str(args.reference_weights)))
            checks = pd.concat(checks,ignore_index=True) if checks else pd.DataFrame(columns=['source','policy','panel_id','dataset_id','saved_pi','reconstructed_pi','difference','status'])
            checks.to_csv(out/'weight_reconstruction_checks.csv',index=False)
            for row in checks.loc[checks.status!='match'].to_dict('records'):
                issues.append(dict(severity='hard_error',detail=f'Saved weight mismatch: {row}'))
            features = [c for c in [*components,*RAW_COMPONENTS.values(),*ORIGINAL_QC] if c in table]
            summaries = summarize(table,features)
            for name,frame in summaries.items():
                frame.to_csv(out/f'{name}.csv',index=False)
            if args.source_mass_warning is not None:
                warning_rows=summaries['panel_concentration_summary'].query('maximum_source_mass > @args.source_mass_warning')
                for row in warning_rows.itertuples():
                    issues.append(dict(severity='heuristic_warning',detail=f'{row.panel_id}/{row.policy}: {row.largest_sources} share {row.maximum_source_mass:.3f} exceeds declared heuristic {args.source_mass_warning}; not a hard design constraint.'))
            args._global_rank_max = metadata['global_rank_max']
            plot_audit(table,summaries,groups,args,out/'figures')
        table.to_csv(out/'dataset_rank_reference_table.csv',index=False)
    except (ValueError,KeyError,FileNotFoundError,AssertionError,RuntimeError) as exc:
        issues.append(dict(severity='hard_error',detail=f'{type(exc).__name__}: {exc}'))
        if table is not None:
            table.to_csv(out/'dataset_rank_reference_table.csv',index=False)
    for path,digest in tracked.items():
        if sha256(path)!=digest:
            issues.append(dict(severity='hard_error',detail=f'Input changed during audit: {path}'))
    manifest['hard_errors'] = [x['detail'] for x in issues if x['severity']=='hard_error']
    manifest['grouping'] = locals().get('groups')
    write_json(out/'audit_manifest.json',manifest)
    write_report(out,manifest,summaries)
    print(f'CPU audit: {out}; hard errors: {len(manifest["hard_errors"])}',flush=True)
    return dict(manifest=manifest,table=table,panels=panels,panel_manifest=panel_manifest,split=split,
                configs=configs,reliability=reliability,summaries=summaries)
