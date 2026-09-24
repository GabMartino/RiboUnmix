#!/usr/bin/env python3
"""Refresh appendix figures from audited scalar data, without loading models.

Re-run the four-panel and cumulative analyzers first to validate current
exports. This renderer keeps the original and directional cohorts distinct.
"""
from pathlib import Path
import os
import sys
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from Utils.publication_plot_style import publication_rc
from analyses.analyze_synthetic_observation_layers import plot_observation_agreement, plot_deterministic_bias
from analyses.analyze_four_panel_quality_score_directionality import DISPLAY_ARMS, ARM_LABELS, ARM_COLORS

OUT = ROOT / 'analyses/artifacts/manuscript_revision_20260921'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'analyses/artifacts/real_data/four_panel_quality_score_directional_balanced_seed42'
    available = pd.read_csv(data / 'availability.csv')
    assert len(available) == 28 and available.status.eq('validated_predictions').all()
    metrics = pd.read_csv(data / 'cross_panel_metrics.csv')
    assignment = pd.read_csv(ROOT / 'results/four_panel_quality_score_directional_balanced_seed42/inputs/panel_assignment.csv')
    assert metrics.groupby(['arm', 'pair']).size().eq(714).all()
    assert metrics.PCC.notna().all() and metrics.arm.nunique() == 7
    assert not assignment.dataset_id.duplicated().any()
    assert assignment.groupby('source_family').panel_id.nunique().eq(1).all()
    rc = publication_rc()
    rc.update({'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 16,
               'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
               'axes.linewidth': 1.4, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
    if rc['text.usetex']:
        rc['text.latex.preamble'] += r'\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}'
    with matplotlib.rc_context(rc):
        fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), layout='constrained')
        axes = axes.ravel()
        panels = sorted(assignment.panel_id.unique())
        axes[0].boxplot([assignment.loc[assignment.panel_id.eq(p),'global_rank'] for p in panels],
                        tick_labels=['P1','P2','P3','P4'], showfliers=False, whis=(0,100))
        axes[0].set_title('A  Panel QC-rank balance', loc='left')
        axes[0].set_ylabel('Global QC rank')
        axes[0].set_xlabel('Source-disjoint panel')
        for i, (axis, arm) in enumerate(zip(axes[1:], DISPLAY_ARMS), 1):
            frame = metrics.loc[metrics.arm.eq(arm)]
            groups = list(frame.groupby('pair', sort=True))
            boxes = axis.boxplot([g.PCC.to_numpy() for _,g in groups],
                tick_labels=[p.replace('panel_0','').replace('__','--') for p,_ in groups],
                showfliers=False, whis=(5,95), patch_artist=True)
            for box in boxes['boxes']:
                box.set_facecolor(ARM_COLORS[arm]); box.set_alpha(.65)
            axis.set_title(chr(65+i)+'  '+ARM_LABELS[arm].replace('-oriented', ''), loc='left')
            axis.set_ylim(-.05,1.01)
            axis.set_xlabel('Panel pair')
            if i in (1,4): axis.set_ylabel(r'Transcript PCC of $L_t$')
        for axis in axes:
            axis.spines[['top','right']].set_visible(False)
        for ext in ('pdf','png'):
            fig.savefig(OUT / f'four_panel_rank_and_cross_panel_pcc.{ext}', dpi=350)
        plt.close(fig)
    metrics.to_csv(OUT / 'four_panel_source.csv', index=False)
    assignment.to_csv(OUT / 'four_panel_assignment_source.csv', index=False)
    config = yaml.safe_load((ROOT / 'analyses/configs/synthetic_observation_layers.yaml').read_text())
    src = ROOT / config['outputs']['results_directory']
    plot_observation_agreement(pd.read_csv(src / 'observation_layer_agreement_summary.csv'),
                               config=config, output_directory=OUT)
    plot_deterministic_bias(pd.read_csv(src / 'deterministic_bias_effects_summary.csv'),
                            config=config, output_directory=OUT)
    from analyses.analyze_synthetic_reference_target_audit import (
        plot_main, plot_reference_weight_sensitivity, plot_distributions,
        plot_reference_scatter, plot_representatives)
    audit = ROOT / 'analyses/artifacts/synthetic/manuscript_revision/reference_target_best_val_loss'
    summary = pd.read_csv(audit / 'aggregate_summary.csv')
    reference = pd.read_csv(audit / 'reference_only_union_cohort_summary.csv')
    plot_main(summary, reference, OUT, 350)
    plot_reference_weight_sensitivity(summary, OUT, 350)
    # Scalar tables only; legacy column names Q are retained for reproducibility.
    metrics = pd.read_parquet(audit / 'per_transcript_metrics.parquet')
    plot_distributions(metrics, OUT, 350)
    plot_reference_scatter(metrics, pd.read_csv(audit / 'reference_error_associations.csv'), OUT, 350)
    plot_representatives(pd.read_csv(audit / 'representative_profile_source.csv'), OUT, 350)

if __name__ == '__main__':
    main()
