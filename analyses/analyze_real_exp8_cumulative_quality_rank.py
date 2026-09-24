#!/usr/bin/env python3
"""Analyze cumulative Exp8 predictions without treating nested panels as independent."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from analyses.analyze_real_exp8_stability import _load_profiles, _compare_profiles
from Utils.reliability_references import transcript_id_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--bootstrap-seed', type=int, default=202608,
                        help='Compatibility argument; this descriptive analysis does not bootstrap.')
    parser.add_argument('--mean-one-tolerance', type=float, default=1e-4)
    args = parser.parse_args(argv)
    root = args.run_root.resolve()
    experiment = json.loads((root / 'experiment_manifest.json').read_text())
    if experiment.get('experiment_design') != 'cumulative_top_quality':
        raise ValueError('Expected a cumulative quality-rank experiment manifest.')
    common = json.loads((root / 'common_test_manifest.json').read_text())
    ids = common['common_test_ids']
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Common held-out transcript set is empty or duplicated.')
    test_hash = transcript_id_hash(ids)
    profiles, checks, provenance = {}, [], []
    tasks = experiment['tasks']
    if not tasks or len({t['run_id'] for t in tasks}) != len(tasks):
        raise ValueError('Task list is empty or duplicated.')
    for task in tasks:
        selected = json.loads((root / task['directory'] / 'selected_checkpoint.json').read_text())
        if selected.get('checkpoint_variant') != 'best_val_loss':
            raise ValueError('Analysis requires best_val_loss predictions.')
        if selected.get('test_transcript_id_hash') != test_hash:
            raise ValueError('Selected checkpoint has a different held-out set.')
        source = Path(selected['shared_profile_path'])
        if not source.is_absolute():
            source = root / source
        profiles[task['run_id']], check = _load_profiles(path=source,
            expected_ids=ids, mean_one_tolerance=args.mean_one_tolerance)
        # Sequence-only profiles must cover the entire CDS, with no invalid interior positions.
        for row in profiles[task['run_id']].values():
            if not np.all(row['mask'][:row['length']]):
                raise ValueError('Sequence-only profile has missing positions inside the CDS.')
        check['run_id'] = task['run_id']
        checks.append(check)
        provenance.append(dict(run_id=task['run_id'], profile_path=str(source),
            checkpoint_path=selected['checkpoint_path'], test_transcript_id_hash=test_hash))
    rows = []
    for seed in sorted({t['training_seed'] for t in tasks}):
        chain = sorted([t for t in tasks if t['training_seed'] == seed], key=lambda t: t['N'])
        full = chain[-1]
        if full['kind'] != 'full_collection':
            raise ValueError('Every training seed needs a full-collection reference.')
        if len({t['N'] for t in chain}) != len(chain):
            raise ValueError('Multiple runs at the same N and training seed.')
        for smaller, larger in zip(chain, chain[1:]):
            if not set(smaller['datasets']) < set(larger['datasets']):
                raise ValueError('Dataset panels are not strictly nested.')
        pairs = [('to_full', t, full) for t in chain[:-1]]
        pairs += [('successive', a, b) for a, b in zip(chain, chain[1:])]
        for kind, left, right in pairs:
            for values in _compare_profiles(left=profiles[left['run_id']],
                    right=profiles[right['run_id']], transcript_ids=ids):
                rows.append(dict(comparison=kind, training_seed=seed,
                    run_a=left['run_id'], run_b=right['run_id'], N_a=left['N'], N_b=right['N'],
                    **values))
    if not rows:
        raise ValueError('At least two dataset sizes are needed for convergence analysis.')
    output = (args.output_dir or root / 'analysis').resolve()
    output.mkdir(parents=True, exist_ok=True)
    values = pd.DataFrame(rows)
    values.to_parquet(output / 'cumulative_agreement_per_transcript.parquet', index=False)
    summaries = []
    keys = ['comparison', 'training_seed', 'run_a', 'run_b', 'N_a', 'N_b']
    for key, group in values.groupby(keys, sort=True):
        usable = group.PCC.replace([np.inf, -np.inf], np.nan).dropna()
        summaries.append(dict(zip(keys, key), usable_transcripts=len(usable),
            unusable_transcripts=len(group)-len(usable), mean_PCC=usable.mean(),
            median_PCC=usable.median(), q10_PCC=usable.quantile(.1),
            q90_PCC=usable.quantile(.9), mean_RMSE=group.RMSE.mean()))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'cumulative_agreement_summary.csv', index=False)
    pd.concat(checks, ignore_index=True).to_csv(output / 'L_mean_one_checks.csv', index=False)
    pd.DataFrame(provenance).to_csv(output / 'prediction_provenance.csv', index=False)
    with plt.rc_context({'font.family': 'serif', 'font.size': 9, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), layout='constrained')
        for axis, kind, title in zip(axes, ['to_full', 'successive'],
                ['Agreement with full ranked model', 'Agreement between successive sizes']):
            for seed, group in summary.loc[summary.comparison == kind].groupby('training_seed'):
                group = group.sort_values('N_a')
                axis.plot(group.N_a, group.mean_PCC, 'o-', label=f'Seed {seed}')
            axis.set(xlabel='Number of datasets in smaller panel', ylabel='Mean full-CDS PCC', title=title)
            axis.grid(alpha=.25)
            if len(summary.training_seed.unique()) > 1:
                axis.legend()
        for suffix in ('pdf', 'png'):
            fig.savefig(output / f'cumulative_convergence.{suffix}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    (output / 'README.md').write_text(
        'Cumulative top-quality Exp8 convergence\n\n'
        f'All comparisons use the same {len(ids)} held-out transcripts and original '
        'unsmoothed, mean-one complete-CDS predictions from best_val_loss checkpoints. '
        'PCC measures agreement between inferred profiles, not biological accuracy. '
        'The full ranked model is an empirical reference, not ground truth. '
        'Nested panels, reused models, and shared transcripts make comparisons dependent. '
        'The curve combines adding data with a changing quality composition and gamma reference. '
        'Q10/Q90 in the summary are transcript distribution quantiles, not confidence intervals. '
        'Constant-profile PCCs are undefined and are counted as unusable. '
        'Every N was independently initialized; cumulative refers to dataset membership only.\n')
    print(f'Wrote cumulative analysis: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
