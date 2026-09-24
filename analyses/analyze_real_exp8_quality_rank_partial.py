#!/usr/bin/env python3
"""Strict partial-results analysis for the two real-data Exp8 designs.

The supported designs are (i) cumulative top-quality panels with a
quality-ranked gamma reference and (ii) quality-matched replicate panels with
an equal gamma reference.  Their estimands are deliberately different:
cumulative runs are compared across nested panel sizes, whereas equal-weight
runs use the pre-designated disjoint A--B comparisons at the same panel size.

Only original sequence-only ``best_val_loss`` exports are read. Missing or
invalid exports are audited and omitted, never replaced with checkpoints or a
different panel. The script never trains or modifies a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import itertools
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses.analyze_real_exp8_stability import (
    _compare_profiles,
    _load_profiles,
    _task_directory,
)
from Utils.publication_plot_style import LATEX_PAPER_RC
from Utils.reliability_references import transcript_id_hash
from Utils.tensorboard_scalars import find_event_runs, load_scalars

DEFAULT_RUN = ROOT / 'results/real_exp8_L_stability_quality_rank_10components/cumulative_qrank10components_p1.0_seed42'
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#555555']
RANKED_DESIGN = 'cumulative_top_quality'
EQUAL_DESIGN = 'quality_matched_equal_reference'


def identify_design(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return the analysis contract declared by an Exp8 manifest."""
    if manifest.get('experiment_design') == 'cumulative_top_quality':
        if manifest.get('gamma_reference_weighting', 'quality_rank') != 'quality_rank':
            raise ValueError('Cumulative ranked Exp8 does not declare quality-rank gamma weighting.')
        return dict(
            key=RANKED_DESIGN,
            expected_weighting='quality_rank',
            output_name='analysis_partial_quality_rank',
            short_name='cumulative quality-ranked',
        )
    if (
        manifest.get('experiment_name') == 'real_exp8_L_stability'
        and manifest.get('gamma_centering_mode') == 'fixed_reference'
        and manifest.get('gamma_pi_strategy') == 'uniform within each selected subset'
        and manifest.get('sampling_mode') == 'quality_matched'
    ):
        return dict(
            key=EQUAL_DESIGN,
            expected_weighting='equal',
            output_name='analysis_partial_equal_reference',
            short_name='quality-matched equal-reference',
        )
    raise ValueError(
        'Unsupported Exp8 contract. Expected either cumulative_top_quality or '
        'real_exp8_L_stability with quality-matched subsets and a uniform fixed reference.'
    )


def resolve_task_directory(root: Path, task: Mapping[str, Any]) -> Path:
    """Resolve both legacy explicit paths and canonical equal-design paths."""
    if task.get('directory'):
        return root / str(task['directory'])
    return _task_directory(root, task)


def prepare_tasks(root: Path, manifest: Mapping[str, Any], design: str) -> list[dict[str, Any]]:
    """Validate the planned design and attach task-relative directories/colors."""
    raw_tasks = list(manifest.get('tasks', []))
    if not raw_tasks or len({str(t['run_id']) for t in raw_tasks}) != len(raw_tasks):
        raise ValueError('The experiment contains no tasks or duplicate run IDs.')
    sizes = sorted({int(t['N']) for t in raw_tasks})
    tasks: list[dict[str, Any]] = []
    for raw in raw_tasks:
        task = dict(raw)
        task['N'] = int(task['N'])
        task['training_seed'] = int(task['training_seed'])
        if len(task['datasets']) != task['N'] or len(set(task['datasets'])) != task['N']:
            raise ValueError(f"{task['run_id']} has an invalid planned dataset count.")
        directory = resolve_task_directory(root, task)
        task['directory'] = str(directory.relative_to(root))
        task['color'] = COLORS[sizes.index(task['N']) % len(COLORS)]
        tasks.append(task)

    if design == RANKED_DESIGN:
        for seed in sorted({t['training_seed'] for t in tasks}):
            chain = sorted((t for t in tasks if t['training_seed'] == seed), key=lambda t: t['N'])
            if len({t['N'] for t in chain}) != len(chain):
                raise ValueError(f'Cumulative design has more than one task at an N for seed {seed}.')
            for left, right in zip(chain, chain[1:]):
                if left['N'] >= right['N'] or right['datasets'][:left['N']] != left['datasets']:
                    raise ValueError('Cumulative design is not a strictly nested top-N prefix chain.')
        return tasks

    allowed = {'designated_disjoint_pair', 'large_N_subset', 'full_collection'}
    if {t['kind'] for t in tasks} - allowed:
        raise ValueError('Equal-reference Exp8 contains an unsupported task kind.')
    groups: dict[tuple[int, int, str], dict[str, dict[str, Any]]] = {}
    for task in tasks:
        if task['kind'] != 'designated_disjoint_pair':
            continue
        key = (task['training_seed'], task['N'], str(task['pair_id']))
        side = str(task['side'])
        if side in groups.setdefault(key, {}):
            raise ValueError(f'Duplicate side {side} in designated pair {key}.')
        groups[key][side] = task
    for key, sides in groups.items():
        if set(sides) != {'A', 'B'}:
            raise ValueError(f'Incomplete designated A--B pair {key}.')
        left, right = sides['A'], sides['B']
        if set(left['datasets']) & set(right['datasets']):
            raise ValueError(f'Designated pair {key} overlaps in dataset identity.')
        if set(left.get('source_families', [])) & set(right.get('source_families', [])):
            raise ValueError(f'Designated pair {key} overlaps in source-family identity.')
    return tasks


def read_json(path):
    return json.loads(Path(path).read_text())


def relocated(root, directory, saved):
    """Resolve cluster paths by exact task-relative suffix, never by basename."""
    saved = Path(saved)
    anchor = directory.relative_to(root).parts
    parts = saved.parts
    candidates = []
    if not saved.is_absolute():
        candidates.extend([root / saved, directory / saved])
    else:
        candidates.append(saved)
    for i in range(len(parts)-len(anchor)+1):
        if parts[i:i+len(anchor)] == anchor:
            candidates.append(directory.joinpath(*parts[i+len(anchor):]))
    valid = {p.resolve() for p in candidates if p.is_file() and p.resolve().is_relative_to(directory.resolve())}
    if len(valid) != 1:
        raise ValueError(f'Expected one local task-scoped file for {saved}; found {len(valid)}.')
    return valid.pop()


def load_export(root, task, ids, tolerance, expected_weighting='quality_rank'):
    directory = resolve_task_directory(root, task)
    selected_path = directory / 'selected_checkpoint.json'
    selected = read_json(selected_path) if selected_path.exists() else None
    if selected:
        if selected['checkpoint_variant'] != 'best_val_loss' or selected['run_id'] != task['run_id']:
            raise ValueError('Selected checkpoint variant or run identity mismatch.')
        if selected['test_transcript_id_hash'] != transcript_id_hash(ids):
            raise ValueError('Selected checkpoint held-out hash mismatch.')
        runtime_path = relocated(root, directory, selected['source_runtime_manifest'])
    else:
        paths = list(directory.glob('predictions/**/prediction_checkpoint_manifest.json'))
        if len(paths) != 1:
            raise ValueError(f'Expected one prediction runtime manifest; found {len(paths)}.')
        runtime_path = paths[0]
    runtime = read_json(runtime_path)['best_val_loss']
    if runtime.get('sequence_only_shared_profile_prediction') is not True or runtime.get('split_name') != 'test':
        raise ValueError('Export is not a sequence-only test prediction.')
    if runtime['transcript_id_hash'] != transcript_id_hash(ids) or runtime['transcript_count'] != len(ids):
        raise ValueError('Runtime held-out hash/count mismatch.')
    profile_path = relocated(root, directory, runtime['shared_profile_output_path'])
    if profile_path.parent != runtime_path.parent:
        raise ValueError('Profile and runtime manifest are not co-located.')
    if selected:
        if relocated(root, directory, selected['shared_profile_path']) != profile_path:
            raise ValueError('Selected/runtime profile paths differ.')
        if selected['checkpoint_path'] != runtime['checkpoint_path']:
            raise ValueError('Selected/runtime checkpoint identities differ.')
    identity = pd.read_parquet(profile_path, columns=['run_id', 'N'])
    if set(identity.run_id.astype(str)) != {task['run_id']} or set(identity.N) != {task['N']}:
        raise ValueError('Prediction table has the wrong run ID or N.')
    subset = read_json(directory / 'subset_manifest.json')
    gamma = read_json(profile_path.parent / 'gamma_reference_manifest.json')
    reference = subset['fixed_gamma_reference']
    if subset['datasets'] != task['datasets'] or set(gamma['reference_dataset_names']) != set(task['datasets']):
        raise ValueError('Dataset membership differs between design and runtime.')
    if gamma['centering_mode'] != 'fixed_reference' or gamma['weighting'] != expected_weighting:
        raise ValueError(
            f"Runtime gamma reference is not fixed {expected_weighting}; "
            f"observed mode={gamma.get('centering_mode')}, weighting={gamma.get('weighting')}."
        )
    if reference.get('weighting', expected_weighting) != expected_weighting:
        raise ValueError('Subset and experiment gamma-weighting contracts disagree.')
    expected_power = float(reference.get('quality_rank_power', 0.0))
    if not np.isclose(float(gamma.get('quality_rank_power', 0.0)), expected_power):
        raise ValueError('Gamma rank power mismatch.')
    np.testing.assert_allclose(gamma['reference_pi'],
        [reference['pi'][name] for name in gamma['reference_dataset_names']], rtol=1e-6, atol=1e-8)
    pi = np.asarray(gamma['reference_pi'], dtype=np.float64)
    if len(pi) != task['N'] or np.any(pi <= 0) or not np.isclose(pi.sum(), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError('Gamma reference weights are not a positive, normalized N-vector.')
    if expected_weighting == 'equal':
        np.testing.assert_allclose(pi, np.full(task['N'], 1.0 / task['N']), rtol=1e-7, atol=1e-9)
    before = profile_path.stat()
    profiles, checks = _load_profiles(path=profile_path, expected_ids=ids, mean_one_tolerance=tolerance)
    for row in profiles.values():
        if not np.all(row['mask'][:row['length']]):
            raise ValueError('Missing interior CDS positions; refusing truncated/intersected comparison.')
        if np.any(row['values'][row['mask']] < 0):
            raise ValueError('Shared profile contains negative amplitudes.')
    digest = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    after = profile_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Prediction file changed while reading; rerun after copy/export finishes.')
    checks['run_id'] = task['run_id']
    provenance = dict(run_id=task['run_id'], N=task['N'], profile_path=str(profile_path),
        profile_sha256=digest, runtime_manifest=str(runtime_path), checkpoint_path=runtime['checkpoint_path'],
        selected_checkpoint_present=bool(selected), checkpoint_variant='best_val_loss',
        selected_epoch=selected.get('epoch') if selected else None,
        selected_validation_loss=selected.get('validation_loss') if selected else None,
        gamma_reference_weighting=expected_weighting,
        gamma_quality_rank_power=float(gamma.get('quality_rank_power', 0.0)),
        gamma_effective_reference_count=float(1.0 / np.square(pi).sum()),
        test_transcript_id_hash=transcript_id_hash(ids), transcript_count=len(ids))
    return profiles, checks, provenance


def failure_reason(directory, status):
    path = directory / 'logs/launcher.log'
    if path.exists():
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size-200000))
            tail = handle.read().decode(errors='replace')
        if 'FloatingPointError: Non-finite gradients' in tail:
            return 'Non-finite gradients (recorded launcher error)'
    if status.get('return_code') == -9:
        return 'Process killed by signal 9; cause not established by status alone'
    return 'No verified export; ' + ('unsuccessful saved exit' if status.get('completed') is False else 'not yet exported / status unknown')


def collect(root, manifest, ids, tolerance, expected_weighting='quality_rank'):
    profiles, checks, provenance, availability = {}, [], [], []
    for task in manifest['tasks']:
        directory = resolve_task_directory(root, task)
        try:
            status = read_json(directory / 'training_status.json') if (directory / 'training_status.json').exists() else {}
        except (OSError, ValueError):
            # A status file may be in the middle of being copied/written.
            status = {}
        row = dict(run_id=task['run_id'], N=task['N'], training_seed=task['training_seed'],
            training_completed=status.get('completed'), return_code=status.get('return_code'),
            available=False, export_status='missing', reason=failure_reason(directory, status))
        if (directory / 'selected_checkpoint.json').exists() or list(directory.glob('predictions/**/prediction_checkpoint_manifest.json')):
            try:
                profile, check, source = load_export(
                    root, task, ids, tolerance, expected_weighting=expected_weighting
                )
            except Exception as exc:
                row.update(export_status='invalid', reason=f'{type(exc).__name__}: {exc}')
            else:
                profiles[task['run_id']] = profile
                checks.append(check)
                provenance.append(source)
                row.update(available=True, export_status='verified', reason='Verified best_val_loss sequence-only export')
        availability.append(row)
    return profiles, checks, provenance, pd.DataFrame(availability)


def planned_comparisons(tasks, design=RANKED_DESIGN):
    """Construct comparisons implied by the design, before checking availability."""
    comparisons = []
    for seed in sorted({t['training_seed'] for t in tasks}):
        planned = [t for t in tasks if t['training_seed'] == seed]
        if design == RANKED_DESIGN:
            chain = sorted(planned, key=lambda t: t['N'])
            positions = {t['run_id']: i for i, t in enumerate(chain)}
            for left, right in itertools.combinations(chain, 2):
                comparisons.append(dict(
                    left=left,
                    right=right,
                    comparison_kind='nested_cross_N',
                    pair_id='',
                    adjacent_planned=positions[right['run_id']] - positions[left['run_id']] == 1,
                    to_full=right['kind'] == 'full_collection',
                    primary_comparison=True,
                ))
            continue

        designated: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
        for task in planned:
            if task['kind'] == 'designated_disjoint_pair':
                designated.setdefault((task['N'], str(task['pair_id'])), {})[str(task['side'])] = task
        for (N, pair_id), sides in sorted(designated.items()):
            comparisons.append(dict(
                left=sides['A'], right=sides['B'],
                comparison_kind='designated_disjoint_same_N', pair_id=pair_id,
                adjacent_planned=False, to_full=False, primary_comparison=True,
            ))
        for N in sorted({t['N'] for t in planned if t['kind'] == 'large_N_subset'}):
            same_n = sorted(
                (t for t in planned if t['kind'] == 'large_N_subset' and t['N'] == N),
                key=lambda t: t['run_id'],
            )
            for left, right in itertools.combinations(same_n, 2):
                comparisons.append(dict(
                    left=left, right=right,
                    comparison_kind='overlapping_large_subset_same_N', pair_id='',
                    adjacent_planned=False, to_full=False, primary_comparison=False,
                ))
        full = [t for t in planned if t['kind'] == 'full_collection']
        if len(full) > 1:
            raise ValueError(f'Multiple full-collection tasks for training seed {seed}.')
        if full:
            reference = full[0]
            for task in sorted((t for t in planned if t is not reference), key=lambda t: (t['N'], t['run_id'])):
                comparisons.append(dict(
                    left=task, right=reference,
                    comparison_kind='convergence_to_full', pair_id='',
                    adjacent_planned=False, to_full=True, primary_comparison=False,
                ))
    return comparisons


def compare_available(tasks, profiles, ids, design=RANKED_DESIGN):
    rows = []
    for specification in planned_comparisons(tasks, design=design):
        left, right = specification['left'], specification['right']
        if left['run_id'] not in profiles or right['run_id'] not in profiles:
            continue
        pair = _compare_profiles(
            left=profiles[left['run_id']], right=profiles[right['run_id']], transcript_ids=ids
        )
        comparison_id = f"{left['run_id']}__{right['run_id']}"
        dataset_intersection = len(set(left['datasets']) & set(right['datasets']))
        source_family_intersection = len(
            set(left.get('source_families', [])) & set(right.get('source_families', []))
        )
        for item in pair:
            rows.append(dict(
                training_seed=left['training_seed'], run_a=left['run_id'], run_b=right['run_id'],
                N_a=left['N'], N_b=right['N'], comparison_id=comparison_id,
                comparison_kind=specification['comparison_kind'], pair_id=specification['pair_id'],
                dataset_intersection_count=dataset_intersection,
                source_family_intersection_count=source_family_intersection,
                adjacent_planned=specification['adjacent_planned'], to_full=specification['to_full'],
                primary_comparison=specification['primary_comparison'], **item,
            ))
    return pd.DataFrame(rows)


def summarize(values):
    rows = []
    keys = [
        'training_seed', 'run_a', 'run_b', 'N_a', 'N_b', 'comparison_id',
        'comparison_kind', 'pair_id',
    ]
    for key, group in values.groupby(keys, sort=True, dropna=False):
        v = group.PCC.replace([np.inf,-np.inf], np.nan).dropna()
        rows.append(dict(zip(keys, key),
            n_total=len(group), n_usable=len(v), n_undefined=len(group)-len(v),
            mean_PCC=v.mean(), median_PCC=v.median(), q10_PCC=v.quantile(.1), q90_PCC=v.quantile(.9),
            min_PCC=v.min(), max_PCC=v.max(), median_RMSE=group.RMSE.median(),
            median_Spearman=group.Spearman.median(), adjacent_planned=bool(group.adjacent_planned.iloc[0]),
            to_full=bool(group.to_full.iloc[0]),
            primary_comparison=bool(group.primary_comparison.iloc[0]),
            dataset_intersection_count=int(group.dataset_intersection_count.iloc[0]),
            source_family_intersection_count=int(group.source_family_intersection_count.iloc[0])))
    return pd.DataFrame(rows)


def select_examples(values):
    if 'comparison_id' in values:
        columns = 'comparison_id'
    elif {'run_a', 'run_b'} <= set(values):
        columns = ['run_a', 'run_b']
    else:
        columns = ['N_a', 'N_b']
    pivot = values.pivot(index='transcript_id', columns=columns, values='PCC')
    selection = pd.DataFrame({'usable_pairs':pivot.notna().sum(axis=1), 'm_t':pivot.median(axis=1)})
    selection['eligible'] = selection.usable_pairs == len(pivot.columns)
    eligible = selection.loc[selection.eligible,'m_t']
    if eligible.empty:
        return selection.reset_index(), None
    target = eligible.quantile(.5)
    selection['empirical_p50'] = target
    selection['distance_to_p50'] = (selection.m_t-target).abs()
    chosen = selection.loc[selection.eligible].reset_index().sort_values(
        ['distance_to_p50','transcript_id'], kind='stable').iloc[0].transcript_id
    selection['selected_typical'] = selection.index == chosen
    return selection.reset_index(), chosen


def tex(value):
    return str(value).replace('_', r'\_')


def save(fig, output, stem):
    fig.savefig(output / f'{stem}.pdf')
    fig.savefig(output / f'{stem}.png', dpi=300)
    plt.close(fig)


def representative_profile_tasks(tasks, design):
    """Choose profile overlays from metadata alone, never from outcomes."""
    if design == RANKED_DESIGN:
        return sorted(tasks, key=lambda task: task['N'])
    selected = []
    for N in sorted({task['N'] for task in tasks}):
        candidates = sorted(
            (task for task in tasks if task['N'] == N),
            key=lambda task: (
                task['kind'] != 'designated_disjoint_pair',
                str(task.get('pair_id', '')) != 'pair01',
                str(task.get('side', '')) != 'A',
                str(task.get('subset_id', '')) != 'subset01',
                task['run_id'],
            ),
        )
        selected.append(candidates[0])
    return selected


def main_figure(values, tasks, profiles, chosen, selection, output, width, seed, design):
    primary = values.loc[values.primary_comparison]
    if design == RANKED_DESIGN:
        # Prefer immediate planned additions. Never label a gap as a successive step.
        pairs = list(primary.loc[primary.adjacent_planned, ['N_a','N_b']].drop_duplicates().itertuples(index=False, name=None))
        if not pairs:
            pairs = list(primary[['N_a','N_b']].drop_duplicates().itertuples(index=False,name=None))
        distributions = [
            primary.loc[(primary.N_a == a) & (primary.N_b == b), 'PCC'].dropna().to_numpy()
            for a, b in pairs
        ]
        tick_labels = [rf'${a}\!:\!{b}$' for a, b in pairs]
        x_label = 'Dataset counts compared'
        panel_title = 'A. Incremental agreement'
        comparison_labels = [f'{a}:{b}' for a, b in pairs]
    else:
        sizes = sorted(primary.N_a.unique())
        pairs = [(int(N), int(N)) for N in sizes]
        distributions = [primary.loc[primary.N_a == N, 'PCC'].dropna().to_numpy() for N in sizes]
        tick_labels = [rf'${N}$' for N in sizes]
        x_label = r'Datasets per model, $N$'
        panel_title = 'A. Disjoint-panel agreement'
        comparison_labels = [f'N={N}' for N in sizes]

    display_tasks = representative_profile_tasks(tasks, design)
    extra_legend_height=.28*(int(np.ceil(len(display_tasks)/3))-1)
    height=3.65+extra_legend_height
    fig, axes = plt.subplots(1,2,figsize=(width,height),gridspec_kw={'width_ratios':[.95,1.1]})
    fig.subplots_adjust(left=.10,right=.985,bottom=(1.12+extra_legend_height)/height,
                        top=1-.48/height,wspace=.42)
    ax = axes[0]
    for i,v in enumerate(distributions,1):
        if len(v)>1 and np.ptp(v)>0:
            # Matplotlib evaluates each KDE only between the observed extrema.
            violin=ax.violinplot([v], positions=[i], widths=.78,showextrema=False,points=160)
            violin['bodies'][0].set(facecolor='#82a9bf',edgecolor='#426a82',alpha=.6,linewidth=.6)
        if len(v):
            ax.boxplot([v],positions=[i],widths=.18,showfliers=False,patch_artist=True,
                boxprops={'facecolor':'white','edgecolor':'#263b47','linewidth':.7},
                medianprops={'color':'#15242d','linewidth':1.1},
                whiskerprops={'color':'#263b47','linewidth':.7},capprops={'color':'#263b47','linewidth':.7})
    finite=np.concatenate([v for v in distributions if len(v)]) if any(len(v) for v in distributions) else np.array([-1.,1.])
    ax.set_ylim(np.floor((float(finite.min())-.025)*10)/10, max(1.02,float(finite.max())+.02))
    ax.set_xticks(range(1,len(pairs)+1),tick_labels)
    ax.set(xlabel=x_label,ylabel='Full-CDS PCC',title=panel_title)
    ax.grid(axis='y',alpha=.35)
    ax=axes[1]
    if chosen is not None:
        ymax=0
        for task in display_tasks:
            profile=profiles[task['run_id']][chosen]
            original=profile['values'][profile['mask']]
            ymax=max(ymax,float(original.max()))
            ax.plot(np.arange(1,len(original)+1),original,color=task['color'],lw=.7,label=rf'$N={task["N"]}$')
        mt=selection.set_index('transcript_id').loc[chosen,'m_t']
        ax.set_ylim(0,ymax*1.25)
        ax.text(.02,.97,rf'{tex(chosen)}'+'\n'+rf'$m_t={mt:.3f}$',transform=ax.transAxes,va='top')
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.74,.015),
                   ncol=min(3,len(display_tasks)),columnspacing=.8,handlelength=1.1,handletextpad=.3)
    else:
        ax.text(.5,.5,'No transcript has valid PCC\nfor every available pair',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Codon position',ylabel=r'Shared profile $L_t$ (mean one)',title='B. Typical profile')
    stem=f'partial_stability_seed{seed}'
    save(fig,output,stem)
    return comparison_labels, stem, [task['run_id'] for task in display_tasks]


def diagnostic_figures(values, summary, planned, available, output, width, seed, design):
    if design == EQUAL_DESIGN:
        return equal_design_diagnostic_figures(values, summary, output, width, seed)
    sizes=[t['N'] for t in planned]
    matrix=np.full((len(sizes),len(sizes)),np.nan)
    for row in summary.itertuples():
        i,j=sizes.index(row.N_a),sizes.index(row.N_b)
        matrix[i,j]=matrix[j,i]=row.median_PCC
    legend_rows=int(np.ceil(len(summary)/6))
    height=3.2+.20*legend_rows
    fig,axes=plt.subplots(1,2,figsize=(width,height))
    fig.subplots_adjust(left=.09,right=.99,bottom=(.50+.19*legend_rows)/height,top=.88,wspace=.40)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e7e7e7')
    axes[0].imshow(matrix,vmin=-1,vmax=1,cmap=cmap)
    axes[0].set_anchor('N')
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            v=matrix[i,j]
            axes[0].text(j,i,f'{v:.2f}' if np.isfinite(v) else '--',ha='center',va='center',
                         color='white' if np.isfinite(v) and v<.2 else '#17242a')
    axes[0].set(xticks=range(len(sizes)),xticklabels=sizes,yticks=range(len(sizes)),yticklabels=sizes,
                xlabel='Number of datasets',ylabel='Number of datasets',title='A. Median PCC')
    for i,((a,b),group) in enumerate(values.groupby(['N_a','N_b'])):
        v=np.sort(group.PCC.dropna().to_numpy())
        if len(v):
            axes[1].step(v,np.arange(1,len(v)+1)/len(v),where='post',lw=1.1,
                color=COLORS[i%len(COLORS)],linestyle=['-','--',':'][i//len(COLORS)%3],label=rf'${a}:{b}$')
    axes[1].set(xlabel='Full-CDS PCC',ylabel='Fraction of transcripts',title='B. Full distributions',ylim=(0,1.02))
    axes[1].grid(alpha=.25)
    fig.legend(*axes[1].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.01),
               ncol=6,columnspacing=.8,handlelength=1.2)
    save(fig,output,f'pairwise_distributions_seed{seed}')
    # Shape/amplitude agreement for only truly adjacent planned sizes.
    q=summary.loc[summary.adjacent_planned]
    if not q.empty:
        fig,axes=plt.subplots(1,2,figsize=(width,2.4),layout='constrained')
        x=np.arange(len(q))
        axes[0].errorbar(x,q.median_PCC,yerr=np.vstack([q.median_PCC-q.q10_PCC,q.q90_PCC-q.median_PCC]),
            fmt='o',capsize=3,color='#0072B2',markersize=4)
        axes[1].plot(x,q.median_RMSE,'o-',color='#D55E00',markersize=4)
        for ax in axes:
            ax.set_xticks(x,[rf'${a}:{b}$' for a,b in zip(q.N_a,q.N_b)])
            ax.set_xlabel('Dataset counts compared');ax.grid(axis='y',alpha=.3)
        axes[0].set(ylabel=r'Median PCC (10--90\%)',title='A. Profile shape')
        axes[1].set(ylabel='Median RMSE',title='B. Profile amplitude')
        save(fig,output,f'incremental_metrics_seed{seed}')
    stems = [f'pairwise_distributions_seed{seed}']
    if not q.empty:
        stems.append(f'incremental_metrics_seed{seed}')
    return stems


def equal_design_diagnostic_figures(values, summary, output, width, seed):
    """Plot equal-reference estimands without inventing a nested trajectory."""
    stems = []
    same_n = summary.loc[summary.comparison_kind != 'convergence_to_full'].copy()
    primary_values = values.loc[values.primary_comparison]
    if not same_n.empty:
        fig, axes = plt.subplots(1, 2, figsize=(width, 3.0), layout='constrained')
        for kind, marker, label in [
            ('designated_disjoint_same_N', 'o', 'Designated disjoint pair'),
            ('overlapping_large_subset_same_N', 's', 'Large-subset pair'),
        ]:
            group = same_n.loc[same_n.comparison_kind == kind]
            if not group.empty:
                axes[0].scatter(group.N_a, group.median_PCC, marker=marker, s=30, label=label)
        axes[0].set_xscale('log', base=2)
        ticks = sorted(same_n.N_a.unique())
        axes[0].set_xticks(ticks, [str(int(value)) for value in ticks])
        axes[0].set(xlabel=r'Datasets per model, $N$', ylabel='Median full-CDS PCC',
                    title='A. Run-pair summaries')
        axes[0].grid(axis='y', alpha=.3)
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend()
        for i, (N, group) in enumerate(primary_values.groupby('N_a', sort=True)):
            finite = np.sort(group.PCC.dropna().to_numpy())
            if len(finite):
                axes[1].step(finite, np.arange(1, len(finite) + 1) / len(finite), where='post',
                             color=COLORS[i % len(COLORS)], lw=1.1, label=rf'$N={int(N)}$')
        axes[1].set(xlabel='Full-CDS PCC', ylabel='Fraction of transcripts',
                    title='B. Designated-pair distributions', ylim=(0, 1.02))
        axes[1].grid(alpha=.25)
        if axes[1].get_legend_handles_labels()[0]:
            axes[1].legend(ncol=2)
        save(fig, output, f'pairwise_distributions_seed{seed}')
        stems.append(f'pairwise_distributions_seed{seed}')

    if not primary_values.empty:
        aggregate = []
        for N, group in primary_values.groupby('N_a', sort=True):
            pcc = group.PCC.dropna()
            aggregate.append(dict(
                N=int(N), median_PCC=pcc.median(), q10_PCC=pcc.quantile(.1), q90_PCC=pcc.quantile(.9),
                median_RMSE=group.RMSE.median(), number_of_available_pairs=group.comparison_id.nunique(),
            ))
        aggregate = pd.DataFrame(aggregate)
        aggregate.to_csv(output / f'equal_reference_stability_by_N_seed{seed}.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(width, 2.6), layout='constrained')
        axes[0].errorbar(
            aggregate.N, aggregate.median_PCC,
            yerr=np.vstack([aggregate.median_PCC-aggregate.q10_PCC, aggregate.q90_PCC-aggregate.median_PCC]),
            fmt='o-', capsize=3, color='#0072B2', markersize=4,
        )
        axes[1].plot(aggregate.N, aggregate.median_RMSE, 'o-', color='#D55E00', markersize=4)
        for axis in axes:
            axis.set_xscale('log', base=2)
            axis.set_xticks(aggregate.N, [str(int(value)) for value in aggregate.N])
            axis.set_xlabel(r'Datasets per model, $N$')
            axis.grid(axis='y', alpha=.3)
        axes[0].set(ylabel=r'Median PCC (10--90\%)', title='A. Profile shape')
        axes[1].set(ylabel='Median RMSE', title='B. Profile amplitude')
        save(fig, output, f'stability_metrics_seed{seed}')
        stems.append(f'stability_metrics_seed{seed}')

    to_full = summary.loc[summary.to_full]
    if not to_full.empty:
        fig, axis = plt.subplots(figsize=(width, 3.0), layout='constrained')
        for row in to_full.itertuples():
            color = COLORS[sorted(to_full.N_a.unique()).index(row.N_a) % len(COLORS)]
            axis.scatter(row.N_a, row.median_PCC, color=color)
        grouped = to_full.groupby('N_a', as_index=False).median_PCC.median()
        axis.plot(grouped.N_a, grouped.median_PCC, color='#333333', lw=.9)
        axis.set_xscale('log', base=2)
        axis.set_xticks(grouped.N_a, [str(int(value)) for value in grouped.N_a])
        axis.set(xlabel=r'Datasets in subset, $N$', ylabel=r'Median PCC to full $N=114$ model',
                 title='Convergence to the full collection')
        axis.grid(axis='y', alpha=.3)
        save(fig, output, f'convergence_to_full_seed{seed}')
        stems.append(f'convergence_to_full_seed{seed}')
    return stems


def training_history(root,tasks):
    rows,errors=[],[]
    for task in tasks:
        for directory in find_event_runs(root/task['directory']/'logs'):
            try:
                scalars=load_scalars(directory)
            except Exception as exc:
                errors.append(dict(run_id=task['run_id'],reason=str(exc)));continue
            epochs=pd.DataFrame([(v.step,v.wall_time,v.value) for v in scalars.get('epoch',[])],
                                columns=['step','wall_time','epoch'])
            if not epochs.empty:
                epochs=epochs.sort_values('wall_time').drop_duplicates('step',keep='last').set_index('step')
            for tag in ('val_loss','val_mu_pcc'):
                for event in scalars.get(tag,[]):
                    # Match logged epoch at the exact global step; no interpolated epoch labels.
                    epoch=float(epochs.loc[event.step,'epoch']) if event.step in epochs.index else np.nan
                    rows.append(dict(run_id=task['run_id'],N=task['N'],training_seed=task['training_seed'],
                        tag=tag,step=event.step,epoch=epoch,wall_time=event.wall_time,value=event.value,event_directory=str(directory)))
    frame=pd.DataFrame(rows)
    if not frame.empty:
        frame=frame.sort_values('wall_time').drop_duplicates(['run_id','tag','step'],keep='last')
    return frame,errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--figure-width', type=float, default=7.2, help='Final figure width in inches.')
    parser.add_argument('--mean-one-tolerance', type=float, default=1e-4)
    parser.add_argument('--skip-training-curves', action='store_true')
    args = parser.parse_args(argv)
    if not np.isfinite(args.figure_width) or args.figure_width <= 0:
        raise ValueError('--figure-width must be finite and positive.')
    if not np.isfinite(args.mean_one_tolerance) or args.mean_one_tolerance <= 0:
        raise ValueError('--mean-one-tolerance must be finite and positive.')

    root = args.run_root.expanduser().resolve()
    manifest = read_json(root / 'experiment_manifest.json')
    contract = identify_design(manifest)
    tasks = prepare_tasks(root, manifest, contract['key'])
    normalized_manifest = dict(manifest, tasks=tasks)
    output = (args.output_dir or root / contract['output_name']).expanduser().resolve()
    if output == root or (args.output_dir is None and not output.is_relative_to(root)):
        raise ValueError('Analysis output must be a separate directory.')
    output.mkdir(parents=True, exist_ok=True)

    common = read_json(root / 'common_test_manifest.json')
    ids = list(map(str, common['common_test_ids']))
    expected_hash = transcript_id_hash(ids)
    if not ids or len(set(ids)) != len(ids) or common.get('transcript_id_hash', expected_hash) != expected_hash:
        raise ValueError('Invalid common held-out IDs/hash.')
    profiles, checks, provenance, availability = collect(
        root, normalized_manifest, ids, args.mean_one_tolerance,
        expected_weighting=contract['expected_weighting'],
    )
    availability.to_csv(output / 'run_availability.csv', index=False)
    pd.DataFrame(provenance).to_csv(output / 'prediction_provenance.csv', index=False)
    if checks:
        pd.concat(checks).to_csv(output / 'mean_one_checks.csv', index=False)
    print(availability[['run_id', 'N', 'export_status', 'return_code', 'reason']].to_string(index=False), flush=True)

    comparison_audit = []
    for specification in planned_comparisons(tasks, design=contract['key']):
        left, right = specification['left'], specification['right']
        comparison_audit.append(dict(
            run_a=left['run_id'], run_b=right['run_id'], N_a=left['N'], N_b=right['N'],
            comparison_kind=specification['comparison_kind'], pair_id=specification['pair_id'],
            primary_comparison=specification['primary_comparison'],
            both_exports_available=left['run_id'] in profiles and right['run_id'] in profiles,
            dataset_intersection_count=len(set(left['datasets']) & set(right['datasets'])),
            source_family_intersection_count=len(
                set(left.get('source_families', [])) & set(right.get('source_families', []))
            ),
        ))
    pd.DataFrame(comparison_audit).to_csv(output / 'comparison_plan_and_availability.csv', index=False)

    values = compare_available(tasks, profiles, ids, design=contract['key'])
    summary = summarize(values) if not values.empty else pd.DataFrame()
    if not values.empty:
        values.to_parquet(output / 'pairwise_per_transcript.parquet', index=False)
    summary.to_csv(output / 'pairwise_summary.csv', index=False)
    history, history_errors = (pd.DataFrame(), []) if args.skip_training_curves else training_history(root, tasks)
    if not history.empty:
        history.to_csv(output / 'training_history.csv', index=False)

    sizes = sorted({task['N'] for task in tasks})
    figures, captions = [], []
    with plt.rc_context(LATEX_PAPER_RC):
        for seed in sorted({task['training_seed'] for task in tasks}):
            planned = sorted(
                (task for task in tasks if task['training_seed'] == seed),
                key=lambda task: (task['N'], task['run_id']),
            )
            available = [task for task in planned if task['run_id'] in profiles]
            if values.empty:
                continue
            seed_values = values.loc[values.training_seed == seed]
            seed_summary = summary.loc[summary.training_seed == seed]
            primary = seed_values.loc[seed_values.primary_comparison]
            if primary.empty:
                continue
            selection_values = seed_values if contract['key'] == RANKED_DESIGN else primary
            selection, chosen = select_examples(selection_values)
            selection.to_csv(output / f'example_selection_seed{seed}.csv', index=False)
            labels, stem, displayed_runs = main_figure(
                seed_values, available, profiles, chosen, selection, output,
                args.figure_width, seed, contract['key'],
            )
            figures.append(stem)
            if chosen is not None:
                examples = []
                displayed = {run_id for run_id in displayed_runs}
                for task in available:
                    if task['run_id'] not in displayed:
                        continue
                    profile = profiles[task['run_id']][chosen]
                    data = profile['values'][profile['mask']]
                    examples.extend(
                        dict(transcript_id=chosen, N=task['N'], run_id=task['run_id'],
                             codon_position=i, L_t=float(value))
                        for i, value in enumerate(data, 1)
                    )
                pd.DataFrame(examples).to_csv(output / f'typical_profile_source_seed{seed}.csv', index=False)
            figures.extend(diagnostic_figures(
                seed_values, seed_summary, planned, available, output,
                args.figure_width, seed, contract['key'],
            ))

            count_text = []
            for N in sorted({task['N'] for task in planned}):
                n_planned = sum(task['N'] == N for task in planned)
                n_available = sum(task['N'] == N for task in available)
                count_text.append(f'N={N}: {n_available}/{n_planned}')
            common_text = (
                f'Partial {contract["short_name"]} Exp8 (training seed {seed}); verified runs by panel size: '
                f'{", ".join(count_text)}. All comparisons use the same {len(ids):,} held-out transcripts. '
                f'(A) Full-CDS PCC distributions for {", ".join(labels)}. Violins contain every finite observation; '
                'boxes show the interquartile range and median, with whiskers to observations within 1.5 IQR. '
                f'(B) Transcript {chosen or "unavailable"} is closest to the empirical median of its median PCC '
                'among transcripts with defined PCC for every primary comparison. Original, unsmoothed, mean-one '
                'profiles are shown without offsets or rescaling. '
            )
            if contract['key'] == RANKED_DESIGN:
                interpretation = (
                    'Panels are nested top-N prefixes, so adjacent exported sizes describe incremental agreement. '
                    'Dataset reuse, independently optimized models, and common transcripts make the comparisons dependent. '
                    'The gamma reference changes with N; this is not a ranking-versus-equal-weight ablation. '
                )
                label = f'fig:exp8-ranked-partial-{seed}'
            else:
                interpretation = (
                    'Primary distributions pool the pre-designated source-family-disjoint A--B comparisons at each N. '
                    'Arbitrary cross-N runs are not compared because the equal-weight subsets are not nested. Panel B uses '
                    'one deterministic metadata-selected run per available N (pair01-A preferred), not an outcome-selected run. '
                    'Repeated transcripts and three subset-pair replicates within N are dependent descriptive observations. '
                )
                label = f'fig:exp8-equal-partial-{seed}'
            caption = common_text + interpretation + (
                'PCC measures reproducibility, not biological accuracy, and the largest available subset is never substituted '
                'for an unavailable full-collection model.'
            )
            captions.append(caption)
            (output / f'{stem}_caption.txt').write_text(caption + '\n')
            (output / f'{stem}.tex').write_text(
                '\\begin{figure}[t]\n\\centering\n'
                f'\\includegraphics[width=\\textwidth]{{{stem}.pdf}}\n'
                f'\\caption{{{tex(caption)}}}\n\\label{{{label}}}\n\\end{{figure}}\n'
            )

        # Planned-subset diagnostics are valid even when training is unfinished.
        quality = pd.read_csv(root / 'subset_quality_report.csv')
        reference_rows = []
        for task in tasks:
            reference = read_json(root / task['directory'] / 'subset_manifest.json')['fixed_gamma_reference']
            pi = np.asarray(list(reference['pi'].values()), dtype=np.float64)
            reference_rows.append(dict(
                run_id=task['run_id'], N=task['N'], training_seed=task['training_seed'],
                gamma_weighting=reference.get('weighting', contract['expected_weighting']),
                effective_reference_count=float(1.0 / np.square(pi).sum()),
                available=task['run_id'] in profiles,
            ))
        design_table = pd.DataFrame(reference_rows).merge(
            quality[['run_id', 'quality_mismatch']], on='run_id', validate='one_to_one'
        )
        design_table.to_csv(output / 'design_diagnostics.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(args.figure_width, 3.0), layout='constrained')
        for _, group in design_table.groupby('training_seed'):
            for axis, column in zip(axes, ['quality_mismatch', 'effective_reference_count']):
                for row in group.itertuples():
                    color = COLORS[sizes.index(row.N)]
                    axis.scatter(row.N, getattr(row, column), s=26,
                                 facecolor=color if row.available else 'white',
                                 edgecolor=color, linewidth=.9, zorder=3)
                trend = group.groupby('N', as_index=False)[column].median().sort_values('N')
                axis.plot(trend.N, trend[column], '-', color='#888888', lw=.8)
                axis.set_xscale('log', base=2)
                axis.set_xticks(sizes, [str(value) for value in sizes])
                axis.set_xlabel(r'Number of datasets, $N$')
                axis.grid(alpha=.25)
        axes[0].set(ylabel='QC mismatch to full pool', title='A. Quality composition')
        axes[1].plot(sizes, sizes, ':', color='#777777', lw=.8, label='Uniform reference')
        axes[1].set(ylabel=r'Effective count $1/\sum_d\pi_d^2$', title='B. Reference weights')
        axes[1].legend()
        save(fig, output, 'design_diagnostics')
        figures.append('design_diagnostics')

        if not history.empty:
            for seed in sorted(history.training_seed.unique()):
                fig, axes = plt.subplots(1, 2, figsize=(args.figure_width, 4.35))
                fig.subplots_adjust(left=.11, right=.98, bottom=.38, top=.86, wspace=.38)
                labelled = set()
                for task in (task for task in tasks if task['training_seed'] == seed):
                    for axis, tag in zip(axes, ['val_loss', 'val_mu_pcc']):
                        group = history.loc[
                            (history.run_id == task['run_id']) & (history.tag == tag)
                        ].sort_values('step')
                        if group.empty:
                            continue
                        label = rf'$N={task["N"]}$' if (tag, task['N']) not in labelled else None
                        labelled.add((tag, task['N']))
                        axis.plot(group.step, group.value, color=task['color'], lw=1,
                                  linestyle='-' if task['run_id'] in profiles else '--', label=label)
                        axis.set_xlabel('Optimizer step')
                        axis.grid(alpha=.25)
                axes[0].set(ylabel='Validation NB loss', title='A. Training diagnostic')
                axes[1].set(ylabel=r'Validation $\mu$ PCC', title='B. Observed-profile fit')
                fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
                           bbox_to_anchor=(.5, .015), ncol=4,
                           columnspacing=.9, handlelength=1.5)
                save(fig, output, f'training_diagnostics_seed{seed}')
                figures.append(f'training_diagnostics_seed{seed}')

    command = shlex.join([
        str(Path(sys.executable).absolute()), str(Path(__file__).resolve()),
        '--run-root', str(root), '--output-dir', str(output),
        '--figure-width', str(args.figure_width),
        '--mean-one-tolerance', str(args.mean_one_tolerance),
    ] + (['--skip-training-curves'] if args.skip_training_curves else []))
    regenerate = output / 'regenerate.sh'
    regenerate.write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + command + '\n')
    regenerate.chmod(0o755)
    audit = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), run_root=str(root),
        experiment_design=contract['key'], gamma_reference_weighting=contract['expected_weighting'],
        comparison_estimand=('nested cross-N agreement' if contract['key'] == RANKED_DESIGN
                              else 'same-N designated disjoint-panel agreement'),
        common_test_count=len(ids), common_test_hash=expected_hash,
        verified_run_count=len(profiles), planned_run_count=len(tasks),
        available_primary_run_pair_count=(
            int(summary.loc[summary.primary_comparison, 'comparison_id'].nunique())
            if not summary.empty else 0
        ),
        available_runs=list(profiles), planned_runs=[task['run_id'] for task in tasks],
        figures=figures, latex=True, figure_width_inches=args.figure_width,
        labels_pt=12, titles_pt=12, training_history_errors=history_errors,
        regeneration_command=command,
    )
    (output / 'analysis_manifest.json').write_text(json.dumps(audit, indent=2) + '\n')

    design_explanation = (
        'Nested top-N models are compared across N; only adjacent planned sizes are called incremental.'
        if contract['key'] == RANKED_DESIGN else
        'Primary inference uses only the pre-designated source-family-disjoint A--B pairs at the same N. '
        'Large-N overlapping subsets and convergence to N=114 are secondary and appear only when both required exports exist. '
        'No arbitrary cross-N equal-panel comparison is interpreted as an incremental effect.'
    )
    note = f'''# Partial {contract['short_name']} Exp8 analysis

This is a snapshot of an incomplete experiment. Only verified best-validation-loss
sequence-only profile exports enter agreement analyses. Missing and invalid runs
are listed in `run_availability.csv`; nothing is interpolated or replaced. Exact
task-relative paths are relocated to this local copy. Prediction SHA-256 hashes,
checkpoint identities, gamma-reference contracts, and mean-one checks are saved.
The script never trains, replays a checkpoint, alters an artifact, or renormalizes
a profile.

{design_explanation}

The figures use real LaTeX/Latin Modern typography with all labels at least 12 pt.
Each figure has vector PDF and 300-dpi PNG companions. `regenerate.sh` records the
exact command. Undefined constant-profile PCCs are excluded and counted. Error
bars in the descriptive metric panel are transcript 10th--90th percentiles, not
confidence intervals. Repeated transcripts and related model comparisons are not
independent replicates.

PCC here measures reproducibility of frozen inferred profiles, not recovery of an
unobserved biological ground truth. Equal versus quality-ranked gamma weighting
cannot be inferred by contrasting unmatched subset designs; that requires the
separate matched-weighting comparison. An unavailable N=114 model is never
replaced by the largest completed subset.

## Main figure captions

'''
    (output / 'README.md').write_text(note + '\n\n'.join(captions) + '\n')
    page_title = f"Partial {contract['short_name']} Exp8"
    gallery = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(page_title)}</title><style>
body{{max-width:1000px;margin:32px auto;padding:0 20px;font:16px/1.5 system-ui;color:#203040}}
figure{{margin:32px 0;border-top:1px solid #ddd;padding-top:20px}}img{{max-width:100%;height:auto}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:16px}}a{{color:#17658c}}
table{{border-collapse:collapse;font-size:14px}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}
.scroll{{overflow-x:auto}}</style><h1>{html.escape(page_title)}</h1>
<p>Only verified best-validation-loss exports contribute. Missing runs are not estimated;
PCC measures model agreement, not biological accuracy.</p>
<p><a href="README.md">Methods and interpretation</a> · <a href="pairwise_summary.csv">Pairwise source table</a>
 · <a href="prediction_provenance.csv">Prediction provenance</a></p>'''
    gallery += '<div class="scroll">' + availability[['run_id', 'N', 'export_status', 'reason']].to_html(index=False, border=0) + '</div>'
    for stem in figures:
        gallery += f'<figure><h2>{html.escape(stem.replace("_", " "))}</h2><a href="{stem}.pdf">Vector PDF</a> · '
        gallery += f'<a href="{stem}.png">300-dpi PNG</a><img src="{stem}.png" alt="{html.escape(stem)}" loading="lazy">'
        caption_path = output / f'{stem}_caption.txt'
        if caption_path.exists():
            gallery += '<figcaption>' + html.escape(caption_path.read_text()) + '</figcaption>'
        gallery += '</figure>'
    gallery += '<h2>Regenerate this snapshot</h2><pre>' + html.escape(command) + '</pre></html>'
    (output / 'index.html').write_text(gallery)
    print(f'Wrote {len(figures)} figure sets and source tables to {output}', flush=True)
    if not summary.empty:
        columns = ['comparison_kind', 'N_a', 'N_b', 'n_usable', 'median_PCC', 'median_RMSE']
        print(summary[columns].to_string(index=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
