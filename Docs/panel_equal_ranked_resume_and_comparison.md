# Equal and quality-ranked panels: continuation and matched plots

## Original experiments

| Results directory | Original submission script | Orchestrator | Fixed gamma-reference weights |
|---|---|---|---|
| `my_panels_a100_b32_20260906_114323` | `run_real_independent_panel_convergence.slurm` | `run_real_independent_panel_convergence.py` | Equal within each panel |
| `my_panels_qrank_a100_b32_20260908_103510` | `run_real_independent_panel_convergence_quality_rank.slurm` | `run_real_independent_panel_convergence_quality_rank.py` | Global QC ranks, power 1, normalized within each panel |

Both use `main_ribounmix_multidataset.py` and the shared analysis
`analyses/analyze_real_panel_convergence.py`. The equal run's publication figure
comes from `analyses/create_four_panel_reproducibility_figure.py`.

The ranked design reuses the equal run's dataset/source-family memberships and
transcript splits. The panels contain 29, 29, 28 and 28 datasets. The comparison
checks these identities, the saved training hyperparameters, per-panel local
reliability references, the runtime gamma weights, and the held-out export
identities before computing any statistics.

## What is unfinished (local artifacts inspected 2026-09-10)

The equal run has usable best-validation-loss exports for all four panels.
The ranked run has usable exports for panels 1 and 2. Ranked panels 3 and 4
failed before training with:

```text
ModuleNotFoundError: No module named 'Utils.transcript_batch_metadata'
```

They have no checkpoints. Their first continuation is therefore a **fresh
start of the saved panel task**, not a checkpoint continuation. Their saved
panel memberships, external split and reliability manifests, random seed 42,
batch size 32, BF16 precision, reference chunk 16, and 512-row/256000-token
execution limits are preserved. Future interrupted submissions use the existing
full-state checkpoint selector. Weights-only warm starts are not silently allowed.

## Resume on Leonardo

Synchronize the repository updates to Leonardo, including
`Utils/transcript_batch_metadata.py`, the new ranked resume Python/Slurm files,
and the updated `resume_real_experiment_from_checkpoints.py`. Do not run the
original design-building orchestrator again inside the existing results root.

From the repository with the training virtual environment activated, audit first:

```bash
python resume_real_independent_panel_convergence_quality_rank.py \
  --run-root /leonardo_work/EUHPC_D35_089/my_panels_qrank_a100_b32_20260908_103510 \
  --dry-run
```

Then submit:

```bash
sbatch resume_real_independent_panel_convergence_quality_rank.slurm
```

The default root is the path above. To override it:

```bash
sbatch --export=ALL,RUN_ROOT=/your/existing/ranked/run \
  resume_real_independent_panel_convergence_quality_rank.slurm
```

This uses two GPUs, skips completed panels, and does not modify pi, panel
assignments, or split/reliability manifests. It checks that the frozen ranking
TSV has SHA-256
`07f440ca13c9193f3814d8f529c1e50d8a09ec19fbc1c4a7aa91ecae860be125`.
Every pi vector is independently checked against the global rank formula and
normalization. The old resolved YAML is copied into `resume_inputs/frozen_config.yaml`
with its embedded dataset mapping; today's base YAML defaults cannot silently
replace the saved settings. Paths are relocated and Hydra composition is checked
before launch. Dry-run writes these preparation/audit files but starts no training.

Current source hashes are recorded. Original git commit IDs were not recorded,
so matching saved hyperparameters cannot guarantee identical historical code
or bitwise training trajectories. The scripts do not change the model implementation.

## Run both panel designs from scratch on UNIVIE

The two independent UNIVIE launchers are:

```bash
sbatch run_real_independent_panel_convergence_univie.slurm
sbatch run_real_independent_panel_convergence_quality_rank_univie.slurm
```

Both use `p_csunivie_gres`, load `python/3.11`, activate
`~/venvs/queueing_riboai_venv`, request two GPUs, and write under `./results`.
Each constructs its own deterministic panel assignment and transcript split
from scratch. The quality-ranked run is explicitly recorded as standalone and
does not require an existing equal-weight results directory. Run IDs include
the Slurm job ID, so the two output directories cannot collide.

## Reproduce the same plots for each run, then combine

On the machine with the result exports and LaTeX installed:

```bash
bash analyses/run_real_panel_weighting_analysis.sh
```

The defaults select the two directories above under `results/`. Custom paths:

```bash
PYTHON_BIN=/path/to/venv/bin/python bash analyses/run_real_panel_weighting_analysis.sh \
  /path/to/equal/run /path/to/ranked/run /path/to/comparison/output
```

This runs the **same** `analyze_real_panel_convergence.py` for both experiments,
writing each run's agreement distributions, agreement matrix, representative
profiles, balance plots, and source tables into its `analysis/` directory.
It also regenerates `analysis/publication_figure/four_panel_reproducibility.*`
when that run has all four panels; it does not invent a four-panel figure for
an incomplete run. Existing posthoc regression/peak/boundary analyses are not
refitted or overwritten by this workflow.

To run either original plot workflow separately:

```bash
python analyses/analyze_real_panel_convergence.py \
  --run-root results/my_panels_a100_b32_20260906_114323
python analyses/create_four_panel_reproducibility_figure.py \
  --run-root results/my_panels_a100_b32_20260906_114323

python analyses/analyze_real_panel_convergence.py \
  --run-root results/my_panels_qrank_a100_b32_20260908_103510
# After all ranked panels finish:
python analyses/create_four_panel_reproducibility_figure.py \
  --run-root results/my_panels_qrank_a100_b32_20260908_103510
```

For the combined analysis only:

```bash
python analyses/compare_real_panel_weighting.py
# Final paper check, after all four panels are available in BOTH runs:
python analyses/compare_real_panel_weighting.py --require-all-panels
```

`results/panels_equal_vs_ranked_comparison/` contains PDF/PNG figures, compact
CSV/Parquet source tables, configuration/weight/cohort audits, bootstrap
summaries, a results README, and an exact `reproduce.sh`. The shell workflow
redirects detailed execution logs into the output directories. Typography uses
the same actual LaTeX + Latin Modern configuration as the original publication
figure, with labels at least 12 pt. Install LaTeX, lmodern and dvipng if missing;
there is no automatic fallback to a different font.

## Interpretation and current results

Currently only panels 1 and 2 are matched: **one pair under equal weights versus
the identical pair under ranked weights**, on all 1,593 common test transcripts.
We never compare all six equal-weight pairs with just one ranked pair. The
common test identity hash is
`2ce579fa766c31a3f3e07cc96e124ac2b7ce7732bc2eb9fc9aff451524b40777`.

| Domain | Metric | Equal median | Ranked median | Ranked minus equal, 95% paired bootstrap CI |
|---|---|---:|---:|---|
| Full CDS | PCC | 0.7183 | 0.7313 | +0.0130 [0.0084, 0.0187] |
| Full CDS | Inter-panel RMSE | 0.4374 | 0.4646 | +0.0273 [0.0215, 0.0341] |
| Interior, trim 20 per end | PCC | 0.7026 | 0.7066 | +0.0040 [-0.0004, 0.0098] |
| Interior, trim 20 per end | Inter-panel RMSE | 0.4077 | 0.4334 | +0.0257 [0.0198, 0.0319] |

Thus the currently available ranked comparison has slightly higher full-CDS
shape agreement, but greater absolute profile disagreement. The interior PCC
difference is small and its interval includes zero. This is a mixed, provisional
result, not evidence that ranked weighting uniformly improves recovery.
PCC is insensitive to relative scaling/offset while RMSE is not, so the two
metrics need not select the same policy even for full profiles with mean one.

The bootstrap resamples transcripts, retaining all panel-pair values and both
policies from each sampled transcript together. The statistic is the median
across transcript/pair values; the change is the difference of policy medians.
Default: 5,000 replicates, seed 20260910, percentile 95% intervals. These intervals
do not include uncertainty across new training seeds or new panel assignments.
Constant/nearly constant PCCs are excluded with explicit reason codes on matched
cohorts. No such exclusions occurred in this current comparison.

The same-panel equal-versus-ranked median PCCs are 0.8494 (panel 1) and 0.8560
(panel 2), showing sensitivity to the reference weights. There is no observed
biological ground truth here: neither cross-panel PCC nor inter-panel RMSE is
a direct measure of biological accuracy. No profile normalization, smoothing,
fitting, or changes to model parameters are performed in this comparison.
