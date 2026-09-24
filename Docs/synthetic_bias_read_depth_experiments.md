# Synthetic bias/read-depth experiment plan

For the complete data-generation, preprocessing, split, batching, loss,
shared-signal metric, gamma-recovery metric, and preliminary-results report,
see [Synthetic bias/read-depth recovery](synthetic_data_analysis_report.md).

This plan compares recovery of the shared biological load (`L_bio`) when the
same synthetic kinetics are observed at three read depths:

| depth label | source directory |
| --- | --- |
| `0p25_per_codon` | `Datasets/data/weighted_synthetic/0p25_per_codon` |
| `2_per_codon` | `Datasets/data/weighted_synthetic/2_per_codon` |
| `20_per_codon` | `Datasets/data/weighted_synthetic/20_per_codon` |

The unbiased `artificial_ground_truth.parquet` is not a training condition in
these runs. It remains available to the validation-only synthetic diagnostics.
The bias conditions are added cumulatively in this fixed order:

1. `artificial_bias_3prime_aa`
2. `artificial_bias_3prime_cc`
3. `artificial_bias_3prime_gg`
4. `artificial_bias_3prime_uu`
5. `artificial_bias_5prime_aa`
6. `artificial_bias_5prime_cc`
7. `artificial_bias_5prime_gg`
8. `artificial_bias_5prime_uu`
9. `artificial_bias_gc_fraction_gt_0p7`
10. `artificial_bias_au_fraction_gt_0p7`

## Runs launched by the Slurm array

`run_synthetic_bias_read_depth.slurm` has 47 array tasks:

- 27 within-depth runs: 9 cumulative panels (2 through 10 conditions) at
  each of the three depths;
- 20 inter-depth runs: one three-dataset triplet for each of the 10 bias
  conditions, with the three depths combined once using equal gamma-reference
  weights and once using a quality ranking.

The array ranges are deterministic: tasks `0--8` are the nine panels at
`0p25_per_codon`, `9--17` are the panels at `2_per_codon`, `18--26` are the
panels at `20_per_codon`, and `27--46` are the inter-depth triplets (equal and
quality-ranked tasks alternate for each bias condition).

For an inter-depth triplet the dataset aliases are, for example,
`artificial_bias_3prime_aa_0p25_per_codon`,
`artificial_bias_3prime_aa_2_per_codon`, and
`artificial_bias_3prime_aa_20_per_codon`. The generated ranking is:

| depth | quality rank | gamma reference weight for power 1 |
| --- | ---: | ---: |
| `20_per_codon` | 1 (best) | 1.000 |
| `2_per_codon` | 2 | 0.667 |
| `0p25_per_codon` | 3 (worst) | 0.333 |

The dataloader first converts rank values with
`quality=(max_rank-rank+1)/max_rank`, then the model raises those positive
quality values to `quality_rank_power`. The launcher therefore uses the
ordering `1, 2, 3` exactly as requested, while equal weighting bypasses the
values and assigns one to every reference dataset. Set
`CROSS_DEPTH_RANK_POWER` to another non-negative value to run a softer or
sharper ranking in a separate submission.

## Reproducibility choices

All tasks use the same seed and the same sequence table. The launcher defaults
to the minimum supported `split.validation_weight_bins=2`, which reduces (but
does not eliminate) depth-dependent reliability stratification. The complete
depth-specific master universe makes all nine cumulative panels at one depth
share their validation panel. The current main program has no external split-
manifest override, so validation IDs can still differ between read depths.
Set `VALIDATION_WEIGHT_BINS=10` to reproduce the ordinary, finer reliability-
stratified split.

The Slurm launcher defaults to `data.batch_size=2`, `data.num_workers=0`, and
automatic grouped gradient accumulation targeting 32 unique transcripts per
optimizer update. Its batch size and worker count can be overridden with the
environment variables shown below. The separate local launcher has different
runtime behavior, described in the next section.

Example submissions:

```bash
sbatch run_synthetic_bias_read_depth.slurm
sbatch --array=0-26%3 run_synthetic_bias_read_depth.slurm       # within-depth only
sbatch --array=27-46%3 run_synthetic_bias_read_depth.slurm      # inter-depth only
sbatch --export=ALL,BATCH_SIZE=1,NUM_WORKERS=0 run_synthetic_bias_read_depth.slurm
```

## Normal-server execution

Use `run_synthetic_bias_read_depth_local.sh` on a server without Slurm. It is
standalone and does not call `sbatch`, `srun`, `torchrun`, or the Slurm
launcher. One GPU is requested by default. The current synthetic YAML supplies
`data.batch_size=32`; the local launcher does not define a `BATCH_SIZE`
environment override.

```bash
tmux new -s riboai-synthetic
GPU_INDEX=0 TASK_RANGE=0-46 \
  ./run_synthetic_bias_read_depth_local.sh
# detach with Ctrl-b d; reconnect with: tmux attach -t riboai-synthetic
```

For two GPUs, use their indices as seen by the process:

```bash
GPU_DEVICES=0,1 TASK_RANGE=0-46 \
  ./run_synthetic_bias_read_depth_local.sh
```

The launcher assigns tasks round-robin to two independent GPU queues. Each
queue is sequential, while the queues run concurrently. Every experiment is a
single-GPU process with `trainer.devices=[0]` after `CUDA_VISIBLE_DEVICES`
remapping; this is experiment-level parallelism, not DDP for one experiment.

For a persistent non-interactive launch, put assignments before `nohup` or use
`env`:

```bash
nohup setsid env TASK_RANGE=0-46 GPU_INDEX=0 \
  ./run_synthetic_bias_read_depth_local.sh \
  > results/riboai_synthetic_launcher.log 2>&1 < /dev/null &
```

`setsid` creates a separate session so the launcher does not receive the
terminal's hang-up signal when you log out. `tmux` is an alternative if you
want to reconnect interactively. Useful controls are `TASKS="0 1 27"`,
`TASK_RANGE=27-46`, `RUN_ID=...`, `OUTPUT_ROOT=...`, `GPU_DEVICES=0,1`,
`NUM_WORKERS=0`, `PREDICT_NUM_WORKERS=0`, `VALIDATION_WEIGHT_BINS=2`,
`CROSS_DEPTH_RANK_POWER=1.0`, `DRY_RUN=1`, and `SKIP_GPU_CHECK=1`.
Each task receives a separate log under
`<OUTPUT_ROOT>/riboai_synthetic_experiment_logs/<RUN_ID>/`.

## Ground-truth recovery report

After runs have produced checkpoints (and, where possible, prediction
parquets), build the cross-run latent-recovery report with:

```bash
python analyses/analyze_synthetic_recovery.py
```

The command writes a plot, panel-level table, bias-case table, compressed
per-transcript table, and an interpretation guide under
`analyses/artifacts/synthetic/shared_profile_recovery/`. By default it
only reads immediate run directories beginning `riboai_synthetic_within_` and
records incomplete runs in a skipped-runs table. It selects the same
maximum-`val_mu_pcc` checkpoint used by the prediction path; the latent truth
is never used for checkpoint selection. `L_bio` and the deterministic kinetic
truth are normalized independently to mean one before calculating equal-
transcript PCC, MSE, and MAE summaries and the square root of the mean MSE.
Runs that have only checkpoint-time
TensorBoard scalars are shown as aggregate-only points and are not assigned a
fabricated per-transcript distribution.

The primary recovery quantity is `L_bio` versus latent kinetics. Dataset-level
`mu` is expected to retain artificial bias and is shown only to demonstrate
the intended factorization. For the same reason, the report recomputes target
and `mu` baselines from each run's prediction parquet rather than using an
unmatched observation-depth reference.

To test the other side of the factorization—whether `gamma` recovers each
programmed deterministic bias—run:

```bash
python analyses/analyze_synthetic_gamma_recovery.py
```

This comparison first converts `added_bias` to the physical multiplier
`1 + added_bias`, then applies the model's joint cross-dataset and positional
log-gamma gauge to both truth and prediction. A raw multiplier comparison is
not meaningful because common positional bias belongs to `L_bio` under the
cross-dataset gauge, while dataset-constant multipliers disappear under shape
normalization. The generated `GAMMA_RECOVERY.md` documents the equation and
writes panel-, bias-, and transcript-level results beside the shared-signal
report.

## Artifact naming and interpretation

The main program writes each run below:

```text
<paths.results>/<dataset-signature>/<main-run-tag>/
<paths.checkpoints>/<dataset-signature>/<main-run-tag>/<logger-version>/
```

The main run tag includes the selected dataset names (or a hash for long
lists), all four loss coefficients, the sample reduction, sampler, gamma
centering mode/weighting/power, and sequence-feature routes. It does **not**
include the source read-depth directory. The launcher therefore gives every
array task a distinct `paths.results`, `paths.logs`, and `paths.checkpoints`
root containing the depth/panel/inter-depth/gamma label. This prevents two
depths with the same dataset names from overwriting one another while leaving
the repository's standard internal tags intact.

The loss remains the configured weighted sum of replica NB, raw consensus PCC,
NB-VST consensus PCC, and gamma regularization. Synthetic ground-truth MSE/PCC
are validation diagnostics only; they are not optimization or checkpoint
selection metrics. Checkpoints continue to monitor `val_loss`, with the
additional PCC-best checkpoint produced by the main program.

The launcher creates only task-local symlinks and small encoding/ranking files
for inter-depth aliases under `TMPDIR`; it never copies or modifies parquet
datasets. These temporary aliases are necessary because the dataloader derives
the dataset name from the parquet filename.
