"""Small report helpers for incomplete, single-seed stability experiments."""
from __future__ import annotations

from datetime import datetime, timezone
import html
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


STATUS_LABELS = ['No local output', 'Logs / checkpoint only', 'Validated prediction', 'Failed / invalid']
STATUS_COLORS = ['#edf0f4', '#f4d38a', '#79baa0', '#df8f89']


def audit_transcript_folds(source_folds, *, require_common_validation=False, require_common_training=False):
    """Check held-out roles and measure how training cohorts differ between fits."""
    folds = {name: {part: set(fold[f'{part}_ids']) for part in ('train', 'validation', 'test')}
             for name, fold in source_folds.items()}
    first = next(iter(folds.values()))
    controls = {f'identical_{part}_ids': all(fold[part] == first[part] for fold in folds.values())
                for part in first}
    counts = []
    for name, fold in folds.items():
        if any(fold[a] & fold[b] for a, b in combinations(first, 2)):
            raise ValueError(f'{name}: training, validation and test transcript roles overlap.')
        counts.append(dict(fold_id=name, **{f'n_{part}': len(ids) for part, ids in fold.items()}))
    if not controls['identical_test_ids']:
        raise ValueError('All compared models must share test transcript IDs.')
    if require_common_validation and not controls['identical_validation_ids']:
        raise ValueError('This design requires identical validation transcript IDs.')
    if require_common_training and not controls['identical_train_ids']:
        raise ValueError('This design requires identical training transcript IDs.')
    pairs = []
    for a, b in combinations(folds, 2):
        fa, fb = folds[a], folds[b]
        pairs.append(dict(fold_a=a, fold_b=b, shared_train=len(fa['train'] & fb['train']),
            shared_validation=len(fa['validation'] & fb['validation']),
            a_train_in_b_validation=len(fa['train'] & fb['validation']),
            a_validation_in_b_train=len(fa['validation'] & fb['train'])))
    controls['within_model_folds_disjoint'] = True
    return (controls, pd.DataFrame(counts), pd.DataFrame(pairs, columns=[
        'fold_a', 'fold_b', 'shared_train', 'shared_validation',
        'a_train_in_b_validation', 'a_validation_in_b_train']))


def snapshot_text(availability, n_test):
    complete = int(availability.status.eq('validated_predictions').sum())
    partial = int(availability.status.isin(['logs_only', 'checkpoint_without_export']).sum())
    absent = int(availability.status.eq('no_local_outputs').sum())
    problems = int(availability.status.isin(['invalid_artifacts', 'recorded_failure']).sum())
    recorded_running = int(availability.recorded_status.eq('running').sum())
    return (f'<p class="note"><b>{complete}/{len(availability)} validated model exports</b>; '
            f'{n_test:,} common test transcripts. {partial} tasks have only logs/checkpoints, '
            f'{absent} have no local output, and {problems} are failed or invalid. '
            f'{recorded_running} tasks are recorded as running in the downloaded snapshot; '
            'this report does not query the live scheduler.</p>')


def plot_availability(availability, group_column, groups, arms, out):
    mapping = {'no_local_outputs': 0, 'logs_only': 1, 'checkpoint_without_export': 1,
               'validated_predictions': 2, 'invalid_artifacts': 3, 'recorded_failure': 3}
    frame = availability.pivot(index=group_column, columns='arm', values='status').reindex(index=groups, columns=arms)
    matrix = np.array([[mapping.get(value, 0) for value in row] for row in frame.to_numpy()])
    fig, ax = plt.subplots(figsize=(10, max(3.5, .5 * len(groups) + 2)))
    ax.imshow(matrix, cmap=ListedColormap(STATUS_COLORS), vmin=-.5, vmax=3.5, aspect='auto')
    for i in range(len(groups)):
        for j in range(len(arms)):
            ax.text(j, i, ['—', 'Partial', 'Ready', 'Check'][matrix[i, j]], ha='center', va='center', fontsize=9)
    ax.set_xticks(range(len(arms)), arms, rotation=20, ha='right')
    ax.set_yticks(range(len(groups)), [str(g) for g in groups])
    ax.set_ylabel('Dataset count' if group_column == 'N' else 'Panel')
    ax.set_title('Available artifacts in this snapshot')
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(STATUS_COLORS, STATUS_LABELS)],
              loc='upper center', bbox_to_anchor=(.5, -.24), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'availability.svg', bbox_inches='tight')
    plt.close(fig)


def comparison_readiness(availability, group_column, pairs, arms):
    ready = set(zip(availability.loc[availability.status.eq('validated_predictions'), group_column],
                    availability.loc[availability.status.eq('validated_predictions'), 'arm']))
    return pd.DataFrame([
        dict(endpoint_a=a, endpoint_b=b, arm=arm, ready=(a, arm) in ready and (b, arm) in ready,
             missing_endpoints=','.join(str(g) for g in (a, b) if (g, arm) not in ready))
        for a, b in pairs for arm in arms])


def effect_sentence(effects, baseline, policy, *, kind=None, unit='comparisons'):
    """Describe signs on explicit matched comparisons, without pooling unequal sets."""
    selected = effects[(effects.baseline == baseline) & (effects.policy == policy)]
    if kind is not None:
        selected = selected[selected.kind == kind]
    pcc = selected[(selected.metric == 'PCC') & selected.mean_improvement.notna()]
    rmse = selected[(selected.metric == 'RMSE') & selected.mean_improvement.notna()]
    label = f'{html.escape(policy)} versus {html.escape(baseline)}'
    if pcc.empty:
        return f'{label}: no matched {unit} are available yet.'
    positive = int((pcc.mean_improvement > 0).sum())
    text = (f'{label}: PCC is higher in {positive}/{len(pcc)} available {unit} '
            f'(ΔPCC {pcc.mean_improvement.min():+.4f} to {pcc.mean_improvement.max():+.4f})')
    if not rmse.empty:
        text += f', while RMSE is lower in {int((rmse.mean_improvement > 0).sum())}/{len(rmse)}'
    return text + '; these are descriptive effects conditional on the fitted single-seed models.'


def profile_variation_text(variance):
    if variance.empty:
        return 'Profile variation cannot yet be assessed because no validated profiles are available.'
    flat = int(variance.near_constant.sum())
    return (f'{flat:,}/{len(variance):,} exported model–transcript profiles have variance ≤ 10⁻¹²; '
            'variance and peak agreement help detect flattening but do not establish biological accuracy.')


def plot_validation_history(logs, availability, group_column, groups, arms, colors, out):
    """Keep unfinished loss curves as progress diagnostics, never as final test results."""
    metadata = availability.set_index('task_id')[group_column]
    frame = logs[logs.tag == 'val_loss'].copy()
    frame['group'] = frame.task_id.map(metadata)
    columns = 2 if len(groups) <= 4 else 3
    rows = int(np.ceil(len(groups) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.1 * columns, 3.1 * rows), squeeze=False)
    for ax, group in zip(axes.flat, groups):
        observed = False
        for arm in arms:
            part = frame[(frame.group == group) & (frame.arm == arm)].dropna(subset=['epoch'])
            part = part.sort_values('wall_time').drop_duplicates('epoch', keep='last').sort_values('epoch')
            if part.empty:
                continue
            observed = True
            ax.plot(part.epoch, part.value, '--' if 'reverse' in arm else '-', color=colors[arm], label=arm)
            chosen = availability[(availability[group_column] == group) & (availability.arm == arm)].selected_epoch
            if len(chosen) and np.isfinite(chosen.iloc[0]):
                point = part[part.epoch == chosen.iloc[0]]
                if not point.empty:
                    ax.scatter(point.epoch.iloc[-1], point.value.iloc[-1], color=colors[arm], marker='o', s=26)
        ax.set(title=f'N={group}' if group_column == 'N' else str(group), xlabel='Epoch', ylabel='Validation objective')
        ax.grid(alpha=.2)
        if observed:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, 'No validation logs downloaded', ha='center', transform=ax.transAxes, fontsize=9)
    for ax in list(axes.flat)[len(groups):]:
        ax.axis('off')
    fig.suptitle('Validation progress; dots mark selected checkpoints when available')
    fig.tight_layout()
    fig.savefig(out / 'validation_history.svg')
    plt.close(fig)


def report_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')
