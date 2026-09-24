# RiboUnmix analyses

This directory is the single home for post-hoc analysis code and generated
analysis products. Training outputs remain under `results/`; source datasets
remain under `Datasets/`; final manuscript exports remain under `figures/`.

## Layout

| Path | Purpose |
|---|---|
| `analyses/*.py` | Reusable analysis, audit, comparison, and plotting entry points. |
| `analyses/configs/` | Analysis-only configurations. These are not training configs. |
| `analyses/launchers/` | CPU/SLURM launchers that execute analyses, not training. |
| `analyses/gamma_ablation/` | Shared implementation used by the synthetic gamma-ablation analyses. |
| `analyses/templates/` | Report templates. |
| `analyses/artifacts/synthetic/` | Derived synthetic-data tables, reports, logs, and working plots. |
| `analyses/artifacts/real_data/` | Derived real-data tables, reports, and working plots. |
| `analyses/artifacts/benchmarking/` | Derived benchmark tables and figures. |
| `figures/` | Stable manuscript-facing PDF/PNG exports; intentionally not moved here. |
| `results/` | Frozen model runs, checkpoints, predictions, and experiment metadata. |

No raw observations, checkpoints, prediction exports, or experiment manifests
were renamed during this reorganization.

## Naming convention

Entry points are named by action and scientific scope:

- `analyze_*`: compute scientific metrics and tables;
- `audit_*`: validate assumptions, cohorts, masks, or numerical identities;
- `compare_*`: matched comparisons between experiment families;
- `plot_*` and `create_*`: render figures or reports from saved compact tables;
- `replay_*` and `reevaluate_*`: frozen-checkpoint evaluation only;
- `report_*` and `build_*`: assemble already-computed outputs.

The former numbered gamma scripts now have descriptive names:

| Current entry point | Scientific role |
|---|---|
| `synthetic_gamma_ablation_lbio_recovery.py` | Shared-profile recovery. |
| `synthetic_gamma_ablation_biological_quality.py` | Motif, ramp, and terminal diagnostics. |
| `synthetic_gamma_ablation_gamma_recovery.py` | Identifiable correction recovery. |
| `synthetic_gamma_ablation_report.py` | Combined report assembly. |

## Artifact index

### Synthetic data

| Artifact directory | Contents | Main entry point(s) |
|---|---|---|
| `artifacts/synthetic/gamma_ablation/` | Shared-profile and gamma recovery, biological diagnostics, ICLR figures. | `synthetic_gamma_ablation_*.py`, `plot_iclr_synthetic_recovery_overview.py` |
| `artifacts/synthetic/read_depth/` | Read-depth audit, alpha/gamma recovery, and reconstruction diagnostics. | `audit_synthetic_read_depth.py`, `plot_synthetic_read_depth_effect.py` |
| `artifacts/synthetic/reference_target/` | Reference-defined target versus occupancy audit. | `analyze_synthetic_reference_target_audit.py` |
| `artifacts/synthetic/mass_factor/` | Aggregate mass-factor audit. | `analyze_synthetic_mass_factor.py` |
| `artifacts/synthetic/individual_dataset/` | Replicate agreement and individual-dataset prediction audit. | `analyze_synthetic_individual_datasets.py` |
| `artifacts/synthetic/input_data/` | Simulator/input-layer audit. | `analyze_synthetic_input_data.py` |
| `artifacts/synthetic/prediction_robustness/` | Single- versus multi-dataset robustness appendix. | `analyze_synthetic_prediction_robustness.py` |
| `artifacts/synthetic/preprocessed_datasets/` | Post-preprocessing transcript counts and reliability weights. | `analyze_synthetic_preprocessed_datasets.py` |
| `artifacts/synthetic/hierarchy/` | K-to-q hierarchy diagnostics and representative examples. | `plot_synthetic_hierarchy.py` |
| `artifacts/synthetic/pi_demo/` | Equal, correct, and reversed reference-weight demonstrations. | `analyze_synthetic_pi_demo.py`, `analyze_synthetic_pi_mu_reconstruction.py` |

### Real data

| Artifact directory | Contents | Main entry point(s) |
|---|---|---|
| `artifacts/real_data/cumulative_stability/` | Dataset-count stability under the original cumulative design. | `analyze_cumulative_reference_weight_stability.py` |
| `artifacts/real_data/four_panel_stability/` | Source-disjoint four-panel reproducibility and ablations. | `analyze_four_panel_stability.py` |
| `artifacts/real_data/cumulative_selection_direction/` | Best-first versus worst-first rank-direction study. | `analyze_cumulative_selection_direction.py` |
| `artifacts/real_data/cumulative_selection_quality_score/` | Direction study using measured quality scores. | `analyze_cumulative_selection_direction_quality_score.py` |
| `artifacts/real_data/four_panel_quality_score_directional_balanced_seed42/` | Matched source-disjoint four-panel analysis of equal, best-oriented, and worst-oriented quality-score references. | `analyze_four_panel_quality_score_directionality.py` |
| `artifacts/real_data/four_quality_strata/` | Four quality-strata comparison. | `analyze_four_quality_strata.py` |
| `artifacts/real_data/hek293_metadata/` | NCBI/GEO metadata used for dataset annotations. | `fetch_hek293_ncbi_metadata.py` |

### Benchmarking

`artifacts/benchmarking/mu_fit/` contains the derived observation-fit tables and
figures for the benchmark campaign. The underlying trained runs remain in
`results/riboai_benchmarking_experiments/`.

## Reproducibility examples

Run entry points from the repository root so relative dataset and result paths
resolve consistently:

```bash
.venv/bin/python analyses/analyze_synthetic_mass_factor.py \
  --config analyses/configs/synthetic_mass_factor.yaml

.venv/bin/python analyses/analyze_synthetic_reference_target_audit.py \
  --config analyses/configs/synthetic_reference_target.yaml

RIBOUNMIX_PLOT_TEX=1 .venv/bin/python analyses/plot_synthetic_read_depth_effect.py
```

See `MIGRATION_MANIFEST.tsv` for the exact old-to-new path mapping. Saved
provenance embedded in existing artifacts can legitimately contain pre-move
paths; those records are historical and were not rewritten.
