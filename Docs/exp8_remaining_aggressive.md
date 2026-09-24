# Resume the remaining original Exp8 runs

Audit date: 2026-09-10. Source: the local copy of
`results/my_exp8_a100_b32_20260906_114340`, not live Slurm state.
The complete snapshot is in `exp8_remaining_audit_20260910.csv`.
Of 34 saved task launchers, 24 have readable nonempty common-test profile
Parquet metadata (1,771 rows each), and 10 have no completed prediction artifact.
This checks output availability, not the scientific validity of every prediction.
All available checkpoints for incomplete runs were loaded on CPU: none were
rejected as corrupt, and nine runs have full optimizer and scheduler state.

This directory is **equal-weight Exp8**, not cumulative quality-ranked Exp8.
Do not point the cumulative quality-ranked experiment-preparation launcher at
it: that would change the experiment. The new resume launcher reuses the same
execution-profile utility without rebuilding subsets, splits, or rank weights.

## Frozen task mapping

| Array index | Existing task | Resume epoch (zero-based) | Global step |
|---:|---|---:|---:|
| 0 | N114/full | 25 | 10842 |
| 1 | N080/subset01 | 34 | 14525 |
| 2 | N080/subset02 | 33 | 14042 |
| 3 | N080/subset03 | 29 | 12390 |
| 4 | N040/pair02_A | 18 | 7638 |
| 5 | N040/pair02_B | No checkpoint: fresh start | — |
| 6 | N040/pair03_A | 21 | 8954 |
| 7 | N040/pair03_B | 27 | 11424 |
| 8 | N020/pair03_A | 11 | 4776 |
| 9 | N020/pair03_B | 1 | 784 |

N040/pair01_A and pair01_B now have completed predictions and are excluded.
The array indices do not change as runs complete. On the allocated node the
helper rechecks completion and picks the most advanced readable full-state
checkpoint available there. A run completed since this audit is skipped.
If only a weights-only checkpoint is present, it blocks rather than silently
discarding optimizer state. If no usable checkpoint exists, it starts fresh
and records that fact. Epoch numbers are checkpoint metadata, not an estimate
of epochs remaining or evidence that a job is currently running.

## Submit from the project directory

Use the actual copied run root on the cluster:

```bash
export RUN_ROOT="$PWD/results/my_exp8_a100_b32_20260906_114340"
export EXP8_RUNTIME_PROFILE=aggressive

# Leonardo: nine smaller runs, at most four concurrent tasks.
sbatch --array=1-9%4 resume_real_exp8_remaining_aggressive.slurm
# The full-N114 run retains the earlier launcher's 256G host-memory request.
sbatch --array=0 --mem=256G resume_real_exp8_remaining_aggressive.slurm
```

If the run is still under `/leonardo_work/EUHPC_D35_089/`, set `RUN_ROOT` to that
existing directory instead; no new results root is created.

On UNIVIE use the same exports and these commands instead:

```bash
sbatch --array=1-9%4 resume_real_exp8_remaining_aggressive_univie.slurm
sbatch --array=0 --mem=256G resume_real_exp8_remaining_aggressive_univie.slurm
```

The UNIVIE wrapper uses `p_csunivie_gres`, `module load python/3.11`, and
`~/venvs/queueing_riboai_venv/bin/activate`. Both versions request one GPU per
array task. Separate allocations can share a physical node. Together these
two submissions allow up to five concurrent trainings. The default array is
1–9, so **both commands are needed to include N114**. Host RAM (`--mem`) is not
GPU VRAM; 256G preserves a prior resource choice, not a measured minimum.

Do not overlap these jobs with earlier launchers training the same task
directories. The new launcher shares the old eleven-task array's advisory
`.resume_slurm.lock`; unrelated launchers may not honor that lock.

## What aggressive means on different GPUs

GPU VRAM is queried inside each allocation, not assumed from the partition
name. For N >= 40 the default aggressive settings are:

| Visible GPU total VRAM (GiB) | Pair-row budget | Padded-codon token budget | Reference chunk |
|---|---:|---:|---:|
| <30 | 256 | 128000 | 8 |
| 30 to <60 | 512 | 256000 | 16 |
| 60 to <75 | 1024 | 512000 | 32 |
| >=75 | 2048 | 1024000 | 64 |

N20 retains 512 / 256000 / 16 on GPUs with at least 30 GiB; it uses the smaller
row/token budgets on GPUs below 30 GiB. Reference chunks are always bounded by
N. This wrapper uses zero loader workers for every N to avoid worker copies of
the in-memory datasets, and logs training metrics every 100 steps. Worker
count can be changed with `RUNTIME_NUM_WORKERS` after measuring CPU stalls.

**This does not guarantee compatibility or optimal speed on every node.**
The helper now checks that the GPU natively supports the saved BF16 precision
when selecting an execution profile. Unsupported GPUs fail early, without
silently switching to FP16/FP32 or BF16 emulation. Request suitable GPU types
using the actual GRES/constraint labels configured by the cluster; no hardware
constraint names are guessed here. This check requires the project's pinned
PyTorch version (see `requirements.txt`).

VRAM thresholds are heuristics based on total capacity, not a benchmark or an
automatic OOM-retry system. Long transcripts, other allocations, GPU model,
driver/cuDNN versions, host RAM and CPU stalls can still affect memory and
speed. Whole-transcript groups cannot always fit a nominal microbatch budget;
reference chunking also retains training autograd graphs. Monitor peak VRAM
and time per complete epoch; iterations/second changes meaning when execution
chunks change.

To lower the budget on a resubmission, `EXP8_RUNTIME_PROFILE=auto` uses
1024 / 512000 / 32 instead of the largest aggressive tier. It does not solve
unsupported BF16, host OOM or numerical-gradient failures. Explicit execution
overrides take precedence over either profile, for example:

```bash
EXP8_RUNTIME_PROFILE=auto \
RUNTIME_MAX_PAIR_ROWS=512 \
RUNTIME_MAX_PADDED_TOKENS=256000 \
RUNTIME_REFERENCE_CHUNK=16 \
sbatch --array=0 --mem=256G resume_real_exp8_remaining_aggressive.slurm
```

The saved resolved config is frozen and Hydra-composed before training.
Batch size 32, logical optimizer-batch target, loss, dataset selection, equal
reference weights, and split remain unchanged. The existing bias-GRU float32
repair is retained by default (`BIAS_GRU_PRECISION=float32`) because historical
logs include non-finite gradients. This changes numerical execution; full-state
resume and execution repartitioning do **not** promise bitwise-identical dropout
or optimization trajectories or that a past numerical failure is fixed.

Each task writes `resume_manifest.json` with effective settings and checkpoint
metadata, a frozen snapshot under `resume_inputs/`, and a unique summary in
`logs/resume_summary_<job>_<index>.json`. Original saved launch commands and
resolved configs are not replaced. The original experiment output tree is
reused; temporary training files use `/tmp`, never `/leonardo_scratch`.

## Repeated N040/pair02_A gradient failure (2026-09-10)

The new UNIVIE traceback reports epoch 20, batch 91, lengths 4968--5089 and
non-finite bias-embedding/first-layer GRU gradients. It does not report the
first failing autograd operation or the effective bias-GRU precision. The
local copy of this run is older: its last checkpoint is epoch 18, step 7638,
with finite model and optimizer tensors. It is not an exact failure replay.

Do not disable the local FP32 bias-GRU protection merely because the NB loss
has been rewritten in log space. A long signed recurrent Jacobian product is
not bounded by that rewrite. The trainer can remain `bf16-mixed` with
`BIAS_GRU_PRECISION=float32`: only bias embeddings, the bias GRU and its output
normalization are protected. Other AMP-enabled neural layers retain their
policy. No groups are skipped and no non-finite gradients are zeroed. This is
a targeted mitigation, not a guarantee that the exact cluster failure is fixed.

If the failed job's `resume_manifest.json` says `bias_gru_precision_override`
was `inherit`, resume only array index 4 from the latest usable full-state
checkpoint:

```bash
RUN_ROOT="$PWD/results/my_exp8_a100_b32_20260906_114340" \
BIAS_GRU_PRECISION=float32 EXP8_RUNTIME_PROFILE=aggressive \
sbatch --array=4 resume_real_exp8_remaining_aggressive_univie.slurm
```

If it already says `float32`, the command above does not change precision.
Use `DETECT_ANOMALY=1` on that single task to trace NaNs back to an autograd
operation and its forward stack. Anomaly detection is diagnostic and slower;
it does not fix gradients and may not localize an Inf-only failure. The
existing finite-gradient guard still catches both Inf and NaN. Startup and
failure logs now include the instantiated GRU policy, trainer precision,
formulation version, torch/CUDA/cuDNN versions and GPU name. These fields help
distinguish a stale deployment from an ineffective numerical mitigation.

Local verification for this follow-up: 41 tests passed, one CUDA-only test
skipped. The controlled adversarial GRU test has a finite log-space NB loss
and non-finite BF16 backward, while its FP32 counterpart remains finite.
A small synthetic model also completed two backward/AdamW updates at 5089
codons with FP32 bias recurrence and BF16 heads. Neither is a replay of the
UNIVIE epoch-20 batch, which remains unverified. Shell syntax checks passed.

Sync the shared `resume_real_exp8_remaining_aggressive.slurm` (the UNIVIE
wrapper calls it) and `Models/RiboUnmixLightningModule.py` for these
diagnostics. The numerical model files described in
`Docs/bf16_log_space_numerics.md` must also be present on the cluster.

## Inspection and deployment

```bash
bash resume_real_exp8_remaining_aggressive.slurm --list-tasks
```

CPU-only inspection on a **scratch copy** of a run (dry-run writes resume audit
files but does not train) can simulate 80 GiB as follows:

```bash
LOAD_PYTHON_MODULE=0 VENV_PATH="$PWD/.venv" \
DRY_RUN=1 PROFILE_GPU_MEMORY_GIB=80 SLURM_ARRAY_TASK_ID=0 \
bash resume_real_exp8_remaining_aggressive.slurm
```

Never set `PROFILE_GPU_MEMORY_GIB` for real training; the helper rejects it.
Sync the updated project code to the cluster, not just the Slurm file. In
particular, the launcher needs `resume_real_experiment_from_checkpoints.py`,
`Utils/exp8_runtime_profile.py`, the updated model's checkpoint chunk-override
support, and the existing training dependencies such as
`Utils/transcript_batch_metadata.py`. Historical logs contain an import failure
for the latter, so an incomplete code copy will fail before training regardless
of GPU settings. The resume manifest records configs, not a frozen code version.

All ten actual saved configs also passed CPU-only frozen-config preparation in
temporary copies: nine full-state resumes and one fresh start, each retaining
batch size 32, equal weighting and rank power 0.0. Original result files were
not modified during this check.

Local regression tests cover the fixed task list, shell-to-helper CPU dry runs
at 24/64/80 GiB, saved-config composition, state and design preservation,
per-task locking and native-BF16 preflight. No GPU speed benchmark or live Slurm
submission is performed by these tests.
