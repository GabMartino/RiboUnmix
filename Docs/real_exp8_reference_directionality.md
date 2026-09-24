# Matched cumulative reference-direction experiment

This extends the existing ten-component cumulative experiment, not the
source-disjoint four-panel experiment. It uses **63 fresh models**:
7 prefix sizes × 3 reference policies × 3 training seeds. No old checkpoint is
used for initialization or as a matched control.

## What is held fixed

Source design:
`results/real_exp8_L_stability_quality_rank_10components/cumulative_qrank10components_p1.0_seed42`.

- Prefixes: N = 2, 5, 10, 20, 40, 80, 114, in the existing global-rank order.
- All three arms use the same datasets, their ID ordering, per-N training and
  validation IDs, and numerical training-only reliability references `w_dt`.
  These inputs are copied, not re-estimated.
- The **test list is common: 1,771 transcripts**. Validation is **N-specific**
  (1,285–1,594 transcripts), as in the source design; the legacy
  `common_validation_ids` field only aliases the first subset. No new split
  is drawn. Each model excludes its own validation set and the common test
  set from training.
- Training seeds 42, 43, 44 vary initialization/training randomness, not folds
  or reference fitting. Initial trainable parameter hashes are verified equal
  across the three arms within every N/seed, and rechecked by production code.
- The mass-free, learned-alpha, grouped, transcript-balanced NB2 + consensus
  shape objective, optimizer, schedule and best-val-loss selection are retained.
  Mixed BF16 is used outside the FP32 recurrent kernels, with full BPTT.
  Data-loader workers are zero to avoid copying the in-memory dataset into
  spawned processes. Execution microbatch budgets remain those in the templates.

All tasks use one new production-code snapshot. The old experiment has no
verified matching code/data version, so its old ranked models are not reused.
Current input bytes, code, configurations and relevant package versions are
frozen and checked. Do not copy a workstation-prepared output directory onto
the cluster: let the cluster freeze its own absolute paths and environment.

## Reference interventions

The frozen ranking is `HEK_riboseq_profile_quality_rank_components.tsv`, SHA256
`5811cadf68c56740e83b232b2990299d527630326205db9cba8bc7024d3cf1f8`.
It contains 115 global entries and ten component ranks: periodicity, CDS
enrichment, depth, transcript support, RPF-length center/spread, replicate
agreement, rRNA/tRNA contamination, total mapping, and unique mapping.
Its transcript scope is not verified as training-only.

- Equal: `pi = 1/N`.
- Ranked: global `q = (115-r+1)/115`, then normalize within the selected prefix.
- Reverse: reassign **the same prefix q multiset** in reverse global-rank
  order (dataset ID breaks ties); use production explicit-reference weights.
  This preserves dataset-weight concentration, not source-family shares. The
  authoritative ranking file is never rewritten with artificial ranks.

At N=2, ranked pi is only 0.502183/0.497817. At N=10 its range is
0.095928–0.104072. Equal/ranked/reverse are almost the same reference at small
N by construction. At N=114 ranked effective reference count is 86.47 versus
114 under equal weighting. Do not retune the exponent after examining results
to manufacture a larger policy effect.

## One submission, no separate preparation job

Deploy `run_real_exp8_reference_directionality.py` and
`run_real_exp8_reference_directionality_univie.slurm` into the existing, working
repository on UNIVIE. They reuse the helpers already deployed for the four-panel
directionality experiment. The first array worker freezes configurations under
a lock; subsequent workers reuse them. This does not repeat partition search.

From the cluster repository, start the seed-42 milestone (21 trainings):

```bash
sbatch run_real_exp8_reference_directionality_univie.slurm
```

Then run the prespecified repetitions (42 trainings):

```bash
sbatch --array=21-62%4 run_real_exp8_reference_directionality_univie.slurm
```

Alternatively submit all 63 at once, **instead of** the two submissions above:

```bash
sbatch --array=0-62%4 run_real_exp8_reference_directionality_univie.slurm
```

Concurrency is four, not the number of experiments. Each array element requests
one GPU, 8 CPUs, 128 GB RAM and four days in `p_csunivie_gres`, excluding `dgx1`.
It loads `python/3.11`, then activates
`$HOME/venvs/queueing_riboai_venv` (override `UNIVIE_VENV_PATH` if needed).
Scheduler CUDA visibility is preserved; each model uses logical device zero.
Output is `./results/real_exp8_cumulative_qrank10_directionality`.

Optional CPU-only inspection in the activated project environment:

```bash
python run_real_exp8_reference_directionality.py --dry-run
```

Or inspect through a single allocated array element, without training:

```bash
DRY_RUN=1 sbatch --array=0 run_real_exp8_reference_directionality_univie.slurm
```

The dry run writes the frozen configurations, task matrix, reference weights,
initialization audit and comparison contract; it does not start a trainer.
The Slurm script automatically enables supported full-state resume for earlier
attempts. A completed task is skipped only after revalidating predictions.
To retry selected array indices, keep the output root unchanged, for example:

```bash
sbatch --array=18,19,20%3 run_real_exp8_reference_directionality_univie.slurm
```

Resume uses the production recovery helper's most advanced usable checkpoint
with optimizer/scheduler state and the matching task-contract hash. Corrupt
last-state files may require an earlier full-state checkpoint. Replayed work
and framework RNG/sampler recovery need not be bitwise identical. A best-loss
checkpoint's weights alone are never an exact continuation.

## Analysis, including partial downloads

From the repository in the project environment:

```bash
python analyses/analyze_real_exp8_reference_directionality.py \
  --run-root results/real_exp8_cumulative_qrank10_directionality
```

This is CPU-only analysis of saved arrays/logs: it does not run training or
inference. Repeat the same command after downloading additional results.
Use `--no-tex` on a cluster without a working LaTeX installation; serif and
math fonts still work. `--skip-training-logs` omits the TensorBoard curves.

Outputs are in `analysis_directionality/` under the run root:

- `analysis_report.html`: available comparisons, exact missing models,
  methodological limitations, figures and links to numerical tables.
- `figures/`: native vector PDF/SVG and 600-dpi PNG. Adjacent-size PCC/RMSE,
  matched policy effects when estimable, same-N cross-policy distributions
  and profile variance, validation curves, availability and reference design.
- `adjacent_transcript_metrics.csv`, `adjacent_summary.csv`,
  `adjacent_policy_effects.csv`: original per-transcript values, mean estimates,
  and paired effects with 95% intervals. Descriptive medians are named separately.
- `same_N_policy_*.csv`, `same_N_seed_*.csv`, `profile_diagnostics.csv`:
  reference sensitivity, optimization variability, and amplitude/collapse checks.
- `training_availability.csv`, `training_progress.csv`, `training_scalars.csv`:
  downloaded task status, selected versus best-so-far validation values, and
  numerical sources for learning curves. A missing export does not imply failure.
- `bootstrap_cohorts.csv`, `metric_exclusions.csv`, `split_summary.csv`,
  `verified_input_hashes.csv`, `analysis_manifest.json`, `regenerate.sh`:
  reproducibility, exclusions and input provenance.

Scientific comparisons use only completed, validated `best_val_loss` exports.
Actual training configurations, global ranking and full-prefix gamma weights,
folds, initialization hashes, and fitted reliability references are checked.
Downloaded cluster paths are resolved by an exact experiment-root prefix mapping;
the original files are not rewritten. Sequence-only exports contain dummy
observation-dependent fields: their `mu`/targets are **not** used as real
reconstruction results. Observation-fit context comes from validation logs.

Missing models leave gaps: no bridging 2→10 when N=5 is absent, cross-seed
substitution, or placeholder policy effects. Within each domain/metric the
finite evaluation cohort is shared across all *available* N/policy/seed
comparisons. All entries for a sampled transcript travel together in each of
5,000 bootstrap draws (seed 20260910). Intervals are pointwise and conditional
on the fitted models; each training seed remains a separate estimate. As more
exports arrive this common cohort may change, so its hash and membership are
saved on each invocation. Full-CDS and interior-20 amplitudes are not normalized
again. Undefined correlations and alignment mismatches are explicitly excluded.

Early validation curves are preliminary diagnostics. Comparing best-so-far
losses from unequal training durations is not a final policy contrast, and
validation loss levels at different N use different validation sets/observations.

## What the analysis answers

Primary: adjacent-size shared-profile agreement, **2→5, 5→10, …, 80→114**,
separately by reference policy and training seed, on a common finite test
cohort. This is not convergence to an N=114 reference model. Use paired
transcript-cluster bootstrap intervals, with all policies and adjacent
comparisons carried by the same sampled transcript. Keep per-seed estimates
visible before averaging.

Secondary: same-N cross-policy sensitivity, same-N cross-seed variation,
amplitude/RMSE and observation-fit checks. Reference weights only enter gamma
centering; they are never multiplied into `w_dt` or directly into the loss.

The chain adds progressively poorer-ranked datasets and does not preserve
source-family atomicity. Adjacent models overlap extensively. These are
stability comparisons, not independent source-disjoint replications or
biological-accuracy measurements. Report null or unfavorable effects too.

## Existing four-panel repetitions

The four-panel design already contains 36 configurations. As of the local
2026-09-14 analysis, all 12 seed-42 exports are available, but no seed-43/44
outputs are present. If those tasks are not already pending/running on the
cluster, run the existing worker directly—no preparation rerun is needed:

```bash
sbatch --array=12-35%4 run_rank_balanced_reference_directionality_univie.slurm
```

This is a separate budget of 24 models, not part of the 63 cumulative models.
Keep its original frozen snapshot and environment unchanged. Missing local
downloads alone do not establish that a job failed or was never submitted.
