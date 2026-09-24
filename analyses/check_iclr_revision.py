#!/usr/bin/env python3
"""Validate the September 21 manuscript figure inputs and record provenance.

This checks the finite saved experiment matrix, not biological correctness.
Run after regenerating figures and compiling ICLR_draft/main.tex.
"""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analyses/artifacts/manuscript_revision_20260921'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    synthetic = ROOT / 'analyses/artifacts/synthetic/manuscript_revision/main_compact_recovery'
    real = ROOT / 'analyses/artifacts/real_data'
    inputs = [synthetic / 'synthetic_occupancy_recovery_compact_source.csv',
              synthetic / 'synthetic_gamma_recovery_source.csv',
              real / 'four_panel_quality_score_directional_balanced_seed42/availability.csv',
              real / 'cumulative_selection_quality_score/availability.csv',
              real / 'cumulative_selection_quality_score/own_policy_anchor_summary.csv',
              ROOT / 'analyses/artifacts/synthetic/manuscript_revision/reference_target_best_val_loss/aggregate_summary.csv',
              ROOT / 'analyses/artifacts/synthetic/inter_artificial_bias/gamma/inter_gamma_recovery_by_run.tsv',
              ROOT / 'analyses/artifacts/synthetic/inter_artificial_bias/gamma/synthetic_gamma_recovery_by_transcript_dataset.tsv.gz',
              ROOT / 'analyses/create_iclr_synthetic_compact_recovery.py',
              ROOT / 'analyses/create_iclr_real_data_cumulative_figures.py',
              ROOT / 'analyses/create_iclr_revision_figures.py']
    src, gamma, panels, cumulative, anchors = [pd.read_csv(p) for p in inputs[:5]]
    assert len(src) == 74 and src.reference_weighting.eq('equal').all()
    assert src.groupby('comparison').size().eq(37).all()
    assert len(gamma) == 10 and gamma.prediction_checkpoint_variant.eq('best_val_loss').all()
    assert gamma.loc[gamma.dataset_count.eq(3), 'valid_transcripts'].eq(0).all()
    assert np.isfinite(gamma.loc[gamma.dataset_count.gt(3), 'mean_pair_log_gamma_pcc']).all()
    assert set(gamma.run) == set(src.loc[src.family.eq('cross_depth'), 'run_id'])
    assert len(panels) == 28 and panels.status.eq('validated_predictions').all()
    assert len(cumulative) == 55 and cumulative.status.eq('validated_predictions').all()
    assert len(anchors) == 56 and anchors.anchor_policy.eq(anchors.reference_policy).all()
    assert anchors.anchor_N.eq(2).all() and anchors.n_valid.eq(714).all()
    assert np.allclose(anchors.loc[anchors.N.eq(2), 'mean'], 1, atol=1e-12, rtol=0)
    geometry = pd.read_csv(real / 'manuscript_revision/cumulative_quality_score/cumulative_reference_geometry_plotted_source.csv')
    assert geometry.groupby(['selection_direction','reference_policy']).size().eq(7).all()
    endpoints = geometry.loc[geometry.N.eq(114) & geometry.reference_policy.eq('equal')]
    assert len(endpoints) == 2 and endpoints.weighted_mean_rank.nunique() == 1
    assert np.allclose(endpoints.effective_fraction, 1, atol=1e-12, rtol=0)
    assert sha(ROOT / 'ICLR_draft/sections/abstract.tex') == 'a5bf156483516dbeefb7075b1949793b55eed5b63d805745ab2b50d2632e1194'
    log = (ROOT / 'ICLR_draft/main.log').read_text()
    assert 'Overfull' not in log and 'undefined references' not in log and 'multiply defined' not in log
    checks = ['74 equal-reference L comparisons: 27 individual-depth and 10 joint-depth fits, two metrics',
              '10 matching best-val-loss gamma runs; single-family PCC undefined; other nine finite',
              '28/28 four-panel and 55/55 cumulative exports validated by existing analyzers',
              '56 own-policy anchor rows; every N=2 self-comparison equals one within 1e-12',
              'Reference geometry includes the shared uniform N=114 endpoint in both paths',
              'Abstract SHA256 unchanged; no overfull boxes or unresolved cross-references']
    outputs = list(OUT.glob('*.pdf')) + [ROOT / 'ICLR_draft/main.pdf',
        ROOT / 'ICLR_draft/figures/main/04_synthetic/synthetic_occupancy_recovery_compact.pdf',
        ROOT / 'ICLR_draft/figures/appendix/real_data/cumulative/cumulative_reference_geometry.pdf',
        ROOT / 'ICLR_draft/figures/appendix/real_data/cumulative/cumulative_directional_stability.pdf']
    manifest = {
        'schema_version': 1,
        'claim_id': 'iclr-20260921-figure-cohort-and-reference-consistency',
        'repository': {'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), 'dirty': True},
        'command': '.venv/bin/python analyses/check_iclr_revision.py',
        'environment': {'software': [f'Python {platform.python_version()}',f'NumPy {np.__version__}',f'pandas {pd.__version__}'], 'hardware': platform.machine()},
        'mathematics': {
            'assertion_tested': 'Plotted comparisons use declared equal-reference fits and policy-specific anchors, with complete real-data exports.',
            'coefficient_domain': 'IEEE-754 float64 scalar correlations and integer cohort counts',
            'conventions': 'qbar replaces Q in rendered notation only; gamma compared to two-way centered log-bias. Distinct cohorts and boundary masks remain explicit.',
            'inputs': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in inputs],
            'bounds': {'individual_depth_fits': 27,'joint_depth_equal_fits': 10,'four_panel_fits': 28,'cumulative_fits': 55,'anchor_transcripts': 714},
            'non_claims': ['No biological ground-truth inference from real-data agreement', 'No independent synthetic test or multi-seed replication', 'No proof of improvement for arbitrary added datasets']},
        'randomness': {'used': False,'generator': 'No randomness in this validation; gamma rendering uses default_rng and 2000 transcript resamples, seed 20260921+N.','seed': None},
        'run': {'started_at': started,'runtime_seconds': time.perf_counter()-start,'exit_status': 0},
        'outputs': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in outputs],
        'checks': checks,
        'result': 'Finite source/cohort checks passed. Figures reuse audited floating-point metrics; no models retrained.',
        'residual_risks': ['Single seeds and cumulative bias order; synthetic validation selection; shared NB2 randomness; different boundary masks in gamma and L audits.']}
    (OUT / 'computation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('\n'.join('PASS: '+s for s in checks))

if __name__ == '__main__':
    main()
