# Parallel resume of the unfinished N20/N40/N80 runs

Run audited locally: `results/my_exp8_a100_b32_20260906_114340`.
These are copied artifacts, not a live query of Leonardo's scheduler.

## Frozen task mapping

| Array index | Task | Latest local full-state checkpoint epoch | Step |
|---|---|---:|---:|
| 0 | N080/subset01 | none: fresh start | — |
| 1 | N080/subset02 | 0 | 413 |
| 2 | N080/subset03 | none: fresh start | — |
| 3 | N040/pair01_A | 5 | 2466 |
| 4 | N040/pair01_B | 37 | 15428 |
| 5 | N040/pair02_A | 12 | 5226 |
| 6 | N040/pair02_B | none: fresh start | — |
| 7 | N040/pair03_A | 21 | 8954 |
| 8 | N040/pair03_B | 27 | 11424 |
| 9 | N020/pair03_A | 11 | 4776 |
| 10 | N020/pair03_B | 1 | 784 |

The other four N20 tasks have readable 1771-row compact test exports and are
excluded. No N40 or N80 task has a compact test export in the audited copy.
All eight listed checkpoints load with optimizer and scheduler state; the
three other tasks have no checkpoint locally. N114 is intentionally excluded
and can use its existing dedicated 256-GiB wrapper.

`resume_real_exp8_N020_N040_N080_array.slurm` embeds this stable mapping. It does
not recompute array indices from a changing list of missing tasks. Each element
rechecks completion and selects the newest usable checkpoint **on the cluster**
when it starts. Later checkpoints need not be copied into this table.

## Resources and behavior

- One Slurm allocation and one training process per model: no DDP.
- `--array=0-10%4`: at most four array jobs running concurrently, subject to
  site/account limits and available resources. Each receives one GPU, eight
  CPUs, 128 GiB host RAM, and up to four days in `boost_qos_lprod`.
- The percent suffix limits array concurrency, not GPU count within a model.
  See [Slurm's array documentation](https://slurm.schedmd.com/job_array.html).
- Preserve Slurm's `CUDA_VISIBLE_DEVICES` verbatim via the shared helper's new
  `--gpus inherit` mode. Do not substitute physical GPU 0; Slurm may assign
  another ordinal or a GPU UUID. The model uses logical device 0 within that
  mask. See [Slurm GPU resource handling](https://slurm.schedmd.com/gres.html).
- Reuse saved dataset subsets, split manifests, reliability references, batch
  settings, losses, and gamma reference weights. Execution limits are unchanged
  unless explicitly overridden as described below.
- Default to zero DataLoader workers to avoid replicating the large in-memory
  dataset. `DATA_NUM_WORKERS` overrides this if explicitly needed.
- Enable the existing FP32 bias-GRU repair by default; Trainer/head precision
  remains as recorded. `BIAS_GRU_PRECISION=inherit` explicitly opts out. Precision
  changes preserve the logical objective, not the original bitwise trajectory.
- Restore full state where available; never silently use a weights-only warm
  start. `ALLOW_WARM_RESUME=1` is required to permit that separate mode.
- Each task writes its own logs and `resume_summary_<array-job>_<index>.json`.
  The new `--summary-path` option avoids concurrent writes to the experiment's
  single root summary. The root summary may therefore be older than these jobs.
- A per-task advisory `flock` rejects duplicate submissions of this wrapper.
  This does not protect against old launchers that do not use the same lock;
  do not launch those for these same run directories concurrently.

## Submit

Synchronize the new Slurm file, updated `resume_real_experiment_from_checkpoints.py`,
and the earlier bias-GRU repair (`DatasetBiasSubmodel.py` and the main training
entrypoint) to Leonardo. Submit from the project directory:

```bash
sbatch --array=0-10%4 \
  --export=ALL,RUN_ROOT=/leonardo_work/EUHPC_D35_089/my_exp8_a100_b32_20260906_114340 \
  resume_real_exp8_N020_N040_N080_array.slurm
```

Use `%2` instead of `%4` for at most two concurrent jobs. To retry only N80,
use `--array=0-2%3`; only N40: `--array=3-8%4`; only the remaining N20 pair:
`--array=9-10%2`. Keep the index-to-task mapping fixed while jobs are queued.

Inspect the mapping without importing Python or changing results:

```bash
bash resume_real_exp8_N020_N040_N080_array.slurm --list-tasks
```

`DRY_RUN=1` performs command/checkpoint preparation without training, but writes
resume audit files for the selected task, following the existing helper's
behavior. `LOAD_PYTHON_MODULE=0` permits such checks outside the cluster module
environment. Do not run a training job on a login node.

Slurm stdout/stderr use `job_<array-job>_<index>_riboai_exp8_remaining.out/.err`.
Per-run `logs/resume_launcher.log` remains the detailed, append-only trainer log.

## Optional higher forward limits (2026-09-10)

The saved A100/b32 runs already use **512 pair rows / 256000 padded codon tokens**
per forward. The wrapper now forwards these optional environment variables to
the existing resume helper. Unset variables retain the original launch settings;
invalid or nonpositive values fail before preparing a task.

| Environment variable | Suggested trial | Saved value |
|---|---:|---:|
| `MAX_PAIR_ROWS_PER_FORWARD` | 1024 | 512 |
| `MAX_PADDED_CODON_TOKENS_PER_FORWARD` | 512000 | 256000 |
| `LOG_EVERY_N_STEPS` | 50 | 25 |

After copying the updated wrapper to Leonardo, submit:

```bash
sbatch --array=0-10%4 \
  --export=ALL,RUN_ROOT=/leonardo_work/EUHPC_D35_089/my_exp8_a100_b32_20260906_114340,MAX_PAIR_ROWS_PER_FORWARD=1024,MAX_PADDED_CODON_TOKENS_PER_FORWARD=512000,LOG_EVERY_N_STEPS=50 \
  resume_real_exp8_N020_N040_N080_array.slurm
```

These are **unbenchmarked trial limits**, not a guaranteed speedup or a guarantee
that every transcript fits GPU memory. Prefer testing the command with
`--array=1` (N80/subset02) first; avoid concurrent submissions for the same task.
Compare epoch duration/optimizer-step throughput, not just execution-batch
iterations per second, because a larger forward changes the number of chunks.
Monitor peak GPU memory over an epoch, including long transcripts and validation.
If memory is exhausted, return to 512/256000 using the same two variables.
Leonardo Booster GPUs have 64 GiB of device memory; the Slurm `--mem=128G` is host
RAM, not additional GPU memory. See [CINECA's Leonardo hardware documentation](https://docs.hpc.cineca.it/hpc/leonardo.html).

Both forward budgets constrain packing; raising only one may leave the other
as the bottleneck. Complete transcript groups remain intact. This does not
increase `data.batch_size=32`, the target transcripts per optimizer update, or
the accumulation ceiling. It leaves reference chunk size 16, uniform pi, loss,
optimizer schedule, and the FP32 bias-GRU repair in place. The repair should not
be disabled for speed. Larger execution chunks can reduce forward/backward
overhead but cost more GPU memory. Numerical rounding and dropout draws can
change with repartitioning: restoring checkpoint state does **not** promise a
bitwise-identical continuation.

Do not select the historical `safe-faster` preset here: its defaults of
256/128000 are smaller than this run's saved limits. No base YAML files or
original launch commands are changed by the optional overrides. Requested
values are recorded in each task's `resume_manifest.json` command and printed
in the Slurm log. Reuse the same environment variables on later submissions to
keep the trial limits; omitting them returns to the original launch settings.

## Validation

Tests cover the exact eleven-run mapping, inherited numeric/UUID GPU masks,
one-task shell dry-run selection, duplicate-task lock rejection, separate
summary files, checkpoint restoration, and preservation of the existing N114
resume behavior. The mapping was compared read-only against the actual run
tree. Bash syntax, Python compilation, and `git diff --check` also pass.
No Slurm jobs were submitted from this workstation.

The higher-limit checks exercise a real shell dry-run against temporary fixtures
with a synthetic full-state checkpoint: both budget overrides and logging
interval reach the trainer command exactly once, checkpoint restoration remains
enabled, logical/reference settings are preserved, and original launch files
remain untouched. No production run artifacts are used for these write tests.
