# Article analyses

Start here instead of choosing scripts by their numeric prefix or modification date.
The active entry points below use the seven article result folders. Experiment
folders, saved predictions, checkpoints, manifests and existing figures were left
in place during the 2026-09-11 cleanup.

## Main entry points

For a **design-only audit** of global dataset ranks, gamma-reference mass and
source-family concentration, use `python analyses/audit_four_panel_reference_quality.py`
from the repository root. It defaults to CPU-only audit and never reads model
performance. Run it with `--help` for its two optional preparation modes.

All paths in this table are relative to `results/`.

For the manuscript's **ten-component ranking-effect Figure 2 (C/D)**, use
`create_real_data_ranking_effect_figure.py` and inspect its `--help` output for
the matching and regeneration interface.
It checks true equal/ranked matches before evaluating arrays. The currently
available six-component panel run and cumulative ranked Exp8 run are not valid
counterparts; unavailable effects are reported, not plotted.

| Article result folder | Analysis / figure scripts |
| --- | --- |
| `synthetic_single_dataset_mu_pcc` | `analyze_synthetic_single_dataset_mu_pcc.py` generates its tables from the original single-dataset predictions in `riboai_synthetic_experiments`; `plot_iclr_synthetic_recovery_overview.py` uses the summary. |
| `riboai_synthetic_experiments` | `13_gamma_ablation_recovery.py` → latent-profile recovery; `20_gamma_ablation_gamma_recovery.py` → gamma recovery; `plot_iclr_synthetic_recovery_overview.py` → manuscript overview. |
| `riboai_benchmarking_experiments` | `plot_benchmarking_mu_pcc.py` → held-out count-profile PCC benchmarking. |
| `my_exp8_a100_b32_20260906_114340` | `analyze_real_exp8_stability.py` → full equal-reference Exp8 analysis; `analyze_real_exp8_partial.py` → available-run analysis. |
| `real_exp8_L_stability_quality_rank_10components` | `analyze_real_exp8_quality_rank_partial.py` → cumulative ten-component Exp8, allowing unfinished tasks; `create_exp8_quality_rank_setup_report.py` → saved design/QC explanation. Actual run subfolder: `cumulative_qrank10components_p1.0_seed42`. |
| `my_panels_a100_b32_20260906_114323` | `analyze_real_panel_convergence.py` → four-panel statistics; `create_four_panel_reproducibility_figure.py` → compact manuscript figure. |
| `my_panels_qrank_a100_b32_20260908_103510` | The same two panel scripts, with this explicit `--run-root`. This historical ranked run used **six**, not ten, ranking components. |

Do not relabel a saved experiment using today's launcher defaults. The historical
panel-ranking provenance and membership checks are documented in
[the saved comparison audit](panels_equal_vs_ranked_comparison/EXPERIMENT_AUDIT_20260911.md).
Those equal/ranked runs have the same four dataset panels; membership balancing
used observed QC variables with source families kept together, not the scalar
six-component ranking. The ranking changes the gamma-centering reference weights.

The original equal Exp8 and cumulative ranked Exp8 have different subset designs
as well as different weights: their comparison is **not a weighting-only ablation**.
Real cross-model PCC measures agreement/reproducibility, not biological accuracy.
Pairs sharing transcripts or trained models are dependent.

For a manuscript-style comparison of the available adjacent-size summaries,
run `create_real_data_ranked_cumulative_panel_d.py`. It overlays Figure 1's
uniform representative-subset series with the cumulative ranked series at
2:5, 5:10, 10:20, 20:40, 40:80 and 80:114 datasets, using the common held-out
cohort and joint transcript-bootstrap intervals. Because the uniform subsets
are not nested while the ranked subsets are nested top-quality prefixes, this
is a descriptive design-level comparison—not an isolated weighting effect or
a causal effect of adding datasets.

## Regeneration commands

Run from the repository root with the project's activated Python environment.
Publication plots use the existing LaTeX style; install LaTeX, Latin Modern and
`dvipng` where required. These commands regenerate derived outputs; they were not
run over existing article outputs during cleanup. Use `--output-dir` for a new
version when the selected script supports it.

### Synthetic manuscript overview

```bash
python analyses/synthetic_gamma_ablation_lbio_recovery.py
python analyses/synthetic_gamma_ablation_gamma_recovery.py
python analyses/analyze_synthetic_single_dataset_mu_pcc.py \
  --results-root results/riboai_synthetic_experiments \
  --run-id single_20260830_205110 --checkpoint-variant best_pcc
python analyses/plot_iclr_synthetic_recovery_overview.py
```

The overview reads the recovery and gamma-recovery tables under
`riboai_synthetic_experiments/gamma_ablation_analysis/`, and the single-dataset
table under `synthetic_single_dataset_mu_pcc/single_20260830_205110/best_pcc/`.
Thus scripts `13` and `20` are required despite their old-style names.
Keep `gamma_ablation/common.py` and `gamma_ablation/__init__.py` with them.

The single-dataset `best_pcc` choice is explicit to preserve the existing figure's
provenance. Do not mix checkpoint variants silently. Synthetic count-profile
PCC compares predicted mu with the observed count target, not with latent biology.
The latent/gamma recovery scripts default to excluding five codons from each CDS
end; this differs from the real-panel full-CDS comparison.

### Benchmarking

```bash
python analyses/plot_benchmarking_mu_pcc.py \
  --run-dir results/riboai_benchmarking_experiments/benchmark_20260829_194806 \
  --checkpoint-variant best_pcc
```

An explicit bundle avoids silently selecting a newer run on another machine.

### Equal and cumulative ten-component Exp8

```bash
python analyses/analyze_real_exp8_partial.py \
  --run-root results/my_exp8_a100_b32_20260906_114340
python analyses/analyze_real_exp8_quality_rank_partial.py \
  --run-root results/real_exp8_L_stability_quality_rank_10components/cumulative_qrank10components_p1.0_seed42
python analyses/create_exp8_quality_rank_setup_report.py
python analyses/compare_real_exp8_weighting.py
```

Use `analyze_real_exp8_stability.py --run-root ...` for the full original Exp8
analysis. The ten-component analyzer audits missing/invalid exports instead of
substituting another checkpoint or another N. The setup report reads the saved,
hash-verified frozen ranking and writes under the ranked run's `analysis_setup/`.
The comparison now defaults to the ten-component run and writes to
`archive/exp8_equal_vs_qrank10_comparison`, leaving the older
`archive/exp8_equal_vs_ranked_comparison` untouched.

### Four-panel analyses and comparison

Regenerate both runs' statistics/figures and their comparison with the existing
wrapper (it writes into each run's `analysis/` and the comparison directory):

```bash
PYTHON_BIN=python bash analyses/run_real_panel_weighting_analysis.sh \
  results/my_panels_a100_b32_20260906_114323 \
  results/my_panels_qrank_a100_b32_20260908_103510 \
  results/panels_equal_vs_ranked_comparison
```

For just one run, use `analyze_real_panel_convergence.py --run-root ...`, then
`create_four_panel_reproducibility_figure.py --run-root ...` once all four panels
are complete. The figure script retains the current compact two-axis A/B layout;
"four panel" in its name refers to the four independently trained models. It
saves source and example-selection tables as well as the manuscript exports.
`compare_real_panel_weighting.py` is the standalone comparison entry point.

## Retained supporting analyses

These are not redundant copies of the main figures:

- `14_gamma_ablation_biological_quality.py` and `15_gamma_ablation_report.py`:
  biological/CSS diagnostics and joined synthetic reports. With a single
  configuration, a report is not evidence for an ablation effect.
- `analyze_synthetic_recovery.py` and `analyze_synthetic_gamma_recovery.py`:
  general recovery engines and helpers used by other retained analyses.
- `analyze_synthetic_inter_shared_signal.py` and
  `analyze_synthetic_inter_artificial_bias.py`: shared-signal and artificial-bias
  diagnostics for the synthetic experiment bundles.
- `analyze_synthetic_alpha_causality.py`,
  `analyze_synthetic_gamma_compensation.py`,
  `analyze_synthetic_gamma_failure_modes.py`,
  `plot_synthetic_missed_gamma_examples.py` and
  `compare_synthetic_gamma_results.py`: causal controls, compensation and failure
  analyses. Keep these for supplementary evidence, including negative findings.
- `analyze_real_panel_robustness_streaming.py`: memory-bounded peak, boundary and
  position-mask sensitivity analyses. Prefer this over the archived non-streaming
  position analysis.
- `analyze_real_panel_posthoc_robustness_streaming.py`: separate residual
  diagnostics. It may replay frozen models to obtain profiles and fit post-hoc
  diagnostic regressions; it is not merely plotting saved summary tables.
  Prefer this over the archived non-streaming post-hoc script. Keep these
  diagnostics separate from the main manuscript figure.
- `analyze_real_exp8_cumulative_quality_rank.py`: retained because the cumulative
  launcher and tests still depend on it. It requires completed exports; the
  recommended plotting entry point is `analyze_real_exp8_quality_rank_partial.py`.
- `analyze_synthetic_pi_demo.py`: a recent, separate equal/quality/reversed-ranking
  control under `synthetic_pi_demo_fixed_20260909/`, outside the seven primary
  folders. Retained as optional supplementary evidence, not silently merged
  with the main synthetic experiment.
- `replay_synthetic_best_val_loss_predictions.py`: prediction-recovery utility,
  not an ordinary analysis step. It loads checkpoints and writes predictions;
  do not run it as part of routine figure regeneration or cleanup.

## Cleanup and deployment

The top-level Python inventory decreased from 46 to 29. Seventeen legacy or
superseded scripts, five loose legacy CSVs, five hand-generated schematic SVGs
and one launcher log were moved, not deleted. The complete list, reasons,
SHA-256 hashes and restore command are in
[_archive/2026-09-11_article_cleanup/README.md](_archive/2026-09-11_article_cleanup/README.md)
and its [manifest](_archive/2026-09-11_article_cleanup/manifest.json).
Archived scripts keep their original contents; restore them to their original
paths before using a historical workflow. No saved scientific outputs were
rewritten. Active script names were preserved to avoid breaking imports,
launchers or regeneration commands.

Changed/new active files to deploy for this cleanup:

- `README.md` (repository root).
- `results/README.md` (this index).
- `analyses/analyze_real_exp8_quality_rank_partial.py` (correct ten-component default).
- `analyses/compare_real_exp8_weighting.py` (ten-component input and distinct output default).
- `analyses/create_exp8_quality_rank_setup_report.py` (saved ranking provenance,
  relocated frozen-table path and ten-component default).

The archive and its two metadata files are optional for execution but should be
preserved somewhere for reversibility. No training code, Slurm scripts,
hyperparameters, checkpoint-selection rules or analysis formulas changed.

Verification: all 29 retained script CLIs passed `--help`, active imports have no
archived-module dependencies, and all 28 moved files passed SHA-256 verification.
The setup report was generated in a temporary directory from the actual saved
manifests: 34 original tasks, seven ranked tasks and ten ranking components.
Of 83 focused analysis tests, 82 passed. The remaining, pre-existing
`Tests/test_gamma_ablation_depth_partitioning.py::test_plotters_reject_mixed_depth_input`
calls the nonexistent `plot_recovery` function in unchanged script `13`; this
stale test was not modified as part of file cleanup.
