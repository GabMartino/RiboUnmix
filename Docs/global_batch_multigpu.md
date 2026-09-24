# One model, multiple GPUs, unchanged global logical batches

`training.execution_microbatching.distributed_mode=global_batch` is opt-in.
The default `single_process` path and existing independent-model array scripts
remain unchanged. This mode accelerates **one model** across GPUs on **one node**;
it is not an array of independently trained models.

## What stays fixed

Keep `data.batch_size=32`, including with two or four GPUs. In this sampler it
is a **per-dataset row quota**, not an exact count of distinct transcripts.
If every transcript occurs in every selected dataset, a full logical batch has
32 transcripts and `32*N` pairs. With incomplete coverage the actual number of
transcripts can differ. Dividing this setting by the GPU count would change the
experiment.

The existing deterministic sampler first constructs the same global logical
batches, with the same transcript membership and ordering, as one process.
Only their execution is partitioned. A transcript and all its observed dataset
pairs stay together. Every process holds the full fixed gamma-reference panel,
with the original equal or quality-rank weights. No reference datasets,
likelihood terms, PCC terms or transcript reliability weights are removed.

The accumulation factor `A` is resolved using the original one-process plan,
independently of GPU count. The legacy `target_scope` flag does not divide the
target again in this mode. Local forward-memory budgets still apply per GPU.
Assignment balances an estimated cost proportional to
`CDS length * (observed pairs + reference datasets)`; it is a heuristic, not a
measured optimal scheduling policy.

## Loss and gradients

For transcript `t`, with observed datasets `D_t`, the existing objective is

\[
\ell_t = \frac{\sum_{d\in D_t} w_{td}\ell_{td}}
                    {\sum_{d\in D_t} w_{td}},\qquad
L_b = \frac1{G_b}\sum_{t\in b}\ell_t.
\]

For a local execution chunk containing `g` of the logical batch's `G_b`
transcripts, backpropagation uses `chunk_loss * g / G_b / A`.
The custom strategy **sums**, rather than averages, accumulated gradients across
processes once at the global optimizer boundary:

\[
g_{\mathrm{update}} = \sum_r g_r
 = \frac1A \sum_{b\in\mathrm{window}} \nabla L_b.
\]

There is **no world-size multiplier** in the local loss. This differs from
ordinary averaged-gradient DDP. One global norm clipping operation follows the
sum, then one AdamW update on each identical model replica. Globally unused
parameters keep `grad=None`, retaining AdamW's usual skip semantics.

Ranks with less work use metadata-only idle slots. They contribute zero local
gradient but join the optimizer update. No real transcript is duplicated to
equalize batch counts. The strategy does not wrap the model in ordinary DDP,
so its per-forward/per-backward collectives cannot conflict with uneven work.
Lightning still handles process launching, precision, optimizer and checkpoint
plumbing through its [strategy extension interface](https://lightning.ai/docs/pytorch/stable/extensions/strategy.html).
Gradient communication uses PyTorch's [distributed collectives](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce).

Validation loss and NB diagnostics are reduced as transcript-weighted sums and
counts; `val_mu_pcc` remains an unweighted mean over transcript-dataset pairs.
The scheduler, early stopping and checkpoint selection receive the same global
values on every rank. Optional per-batch plots and secondary TensorBoard
diagnostics are omitted in this mode; prediction exports remain available.

## Boundaries and limitations

- BF16 mixed and FP32/FP64 are supported; scaled FP16 is not. The existing
  CUDA-AMP FP32 policy for both GRUs remains active; the other AMP-enabled
  operations can still use BF16. No new TBPTT approximation is introduced.
- One node only. Each process holds its own model, optimizer and data state.
  Host RAM therefore needs particular attention for large `N`.
- Use complete epochs and end-of-epoch validation. Do not cut an epoch in the
  middle of a logical batch using a batch limit. Checkpoints must be at optimizer
  boundaries; partially accumulated gradients are not serialized.
- A final incomplete accumulation window keeps the original `1/A` scale.
  In `global_batch` it is applied **before validation**. The legacy
  `single_process` implementation flushes that partial window in
  `on_train_epoch_end`, after validation. Thus, for `A>1` with a remainder,
  historical validation/checkpoint timing is not identical. Global mode is
  consistent between one and multiple GPUs and checkpoints the evaluated state.
- Same logical objective and optimizer windows do **not** mean bitwise-identical
  training. Dropout draws, matrix shapes, summation order and BF16 rounding can
  change. GPU speedup has not been established by the local tests. Compare wall
  time per **global optimizer update** or complete epoch, not execution-chunk
  iterations/second.
- Continue completed-epoch checkpoints when changing GPU count. A full-state
  restoration keeps Adam/scheduler state but is not a claim of an identical
  stochastic trajectory.
- Do not edit the checkout underneath running jobs. The matched-panel launcher
  pins source hashes: an already prepared matched experiment may correctly
  refuse this changed code. Use a separate checkout/new experiment root for
  those source-pinned designs; do not bypass their identity checks.

## Commands

For a fresh training invocation, append these overrides to its existing
dataset/design arguments:

```bash
python main_ribounmix_multidataset.py \
  'trainer.devices=[0,1]' \
  trainer.precision=bf16-mixed \
  trainer.use_distributed_sampler=false \
  data.batch_size=32 \
  training.execution_microbatching.enabled=true \
  training.execution_microbatching.distributed_mode=global_batch
```

The command above does not select an experiment panel; keep that run's existing
dataset list, split manifest, ranking and output arguments. The main entrypoint
selects the custom strategy automatically.

To continue one saved Exp8 task on UNIVIE with two GPUs:

```bash
RUN_ROOT=./results/my_exp8_a100_b32_20260906_114340 \
RUN_ID=real_exp8_N040_pair02_A \
sbatch resume_real_global_batch_univie.slurm
```

This requests one node, one parent Slurm task and two GPUs, excludes `dgx1`,
loads Python 3.11 and the UNIVIE virtual environment, and keeps the saved output
directory/configuration. Lightning launches its own local worker processes.
Do **not** also set `--ntasks=2`. This launcher changes neither the saved batch
size nor the quality ranking and retains the saved TBPTT window unless explicitly
overridden. It skips already-complete tasks under the resume helper's existing
completion checks and acquires the same per-task lock as the resume arrays.

Four GPUs require both the allocation and the worker count to change:

```bash
RUN_ROOT=./results/my_exp8_a100_b32_20260906_114340 \
RUN_ID=real_exp8_N040_pair02_A GLOBAL_BATCH_GPUS=4 \
sbatch --gres=gpu:4 resume_real_global_batch_univie.slurm
```

Use the same `--global-batch-gpus` option from another cluster's environment
inside a single-node multi-GPU allocation:

```bash
python resume_real_experiment_from_checkpoints.py \
  --run-root ./results/my_exp8_a100_b32_20260906_114340 \
  --include-run-ids real_exp8_N040_pair02_A \
  --gpus inherit --global-batch-gpus 2 --use-saved-resolved-config \
  --throughput-profile unchanged --exp8-runtime-profile unchanged \
  --data-num-workers 0
```

Add `--dry-run` to inspect the generated command without training (it still
writes the task's resume-command/audit files). If needed, set explicit per-GPU
`--max-pair-rows-per-forward` and `--max-padded-codon-tokens-per-forward` limits;
the resume helper disallows automatic throughput profiles in this mode so they
cannot silently change the intended execution policy.

## Tests and deployment

```bash
python -m unittest Tests.test_global_batch_distributed
python -m Tests.global_batch_launch_smoke --accelerator cpu --devices 2 \
  --output ./results/global_batch_cpu_smoke
# Inside a two-GPU allocation, before a long production continuation:
python -m Tests.global_batch_launch_smoke --accelerator gpu --devices 2 \
  --precision bf16-mixed --output ./results/global_batch_gpu_smoke
```

The unit tests compare 1/2/3-process gradients, AdamW updates, validation and
scheduler state against an independent monolithic objective, including unequal
group sizes, idle slots, globally unused parameters, partial final accumulation
and completed-epoch checkpoint continuation across different world sizes.
Sampler membership is additionally checked through eight ranks.
The smoke test uses small synthetic arrays with the actual GRUs, fixed-reference
quality centering and loss/prediction code; it is not a large-`N` benchmark.

Local verification on 2026-09-11: 44 focused/legacy checks passed. The real-model
FP32 smoke tests on one versus two CPU processes had identical validation loss
and a maximum absolute final-parameter difference of `1.49e-8`. Training,
validation and prediction also passed on one RTX 3070 Laptop GPU with BF16 mixed
precision and NCCL. The UNIVIE launcher passed a dry-run test with composed
saved Hydra configuration. Only one physical GPU was available: multiple-GPU
CUDA execution and production speedup still require the cluster smoke/benchmark.

Changed/new files to deploy for this implementation:

```text
Dataloaders/RiboUnmixMultiDataset/RiboUnmixMultiDatasetDataModule.py
Dataloaders/RiboUnmixMultiDataset/RiboUnmixMultiDataset.py
Models/RiboUnmixLightningModule.py
Utils/global_batch.py
Utils/global_batch_strategy.py
main_ribounmix_multidataset.py
config/config_ribounmix_multidataset.yaml
resume_real_experiment_from_checkpoints.py
resume_real_global_batch_univie.slurm
Tests/test_global_batch_distributed.py
Tests/global_batch_launch_smoke.py
Docs/global_batch_multigpu.md
```

These are incremental changes on top of the current repository, including the
previous numerical fixes and existing test helpers. They are not a standalone
replacement for missing dependencies from earlier deployments.
