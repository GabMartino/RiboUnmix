# N=114 non-finite backward investigation, 2026-09-08

The underlying failing autograd operation is **not yet identified**. The copied
log records only a parameter-gradient check after norm clipping. This checkout
has no available CUDA device, so the Leonardo failure has not been replayed.
The changes below improve failure localization; they are not evidence that the
training instability is solved.

## Evidence from the actual run

The log in
`results/my_exp8_a100_b32_20260906_114340/N114/full/logs/resume_launcher.log`
contains the reported failure at epoch 1, 2073/4371 execution batches.
Its saved `resolved_config.yaml` specifies:

- `trainer.precision=bf16-mixed`;
- `model.dataset_bias_params.raw_log_gamma_bound=8.0`;
- `model.mass_conservation=false`;
- standard NB, with replica NB weight 1 and both consensus PCC weights 0.5;
- execution gradient norm clipping at 1.0, and transcript-balanced reduction.

The existing stabilized `masked_pcc` implementation already adds epsilon to
each variance. Recommending that same earlier fix or reapplying the gamma bound
does not identify this recurrence. Epoch-aggregated profile/gamma diagnostics
cannot identify a single failing execution chunk.

The copied `last.ckpt` loads successfully, reports epoch 0 and global step 417,
and contains optimizer state. All inspected biological parameters are finite.
It precedes the failure; it cannot establish what the parameters or activations
were at the failing epoch-1 backward pass.

## Confirmed diagnostic defect

Previously, manual optimization called `clip_gradients` before `optimizer.step`.
Lightning invoked `on_before_optimizer_step` from inside that step. Therefore
the non-finite check examined **already clipped** gradients.

Norm clipping uses one norm for all participating parameters. A single NaN in
the dataset branch makes that norm NaN, and multiplying the healthy biological
gradients by the resulting coefficient contaminates them too. A regression
test reproduces this behavior. The reported first ten GRU parameters follow
parameter registration order; they do not establish that the GRU caused the
failure.

A finite displayed loss also does not prove finite backward arithmetic.
`nan_to_num` and masking of an invalid forward result do not repair all of its
upstream derivatives. This general mechanism is documented in
[PyTorch's autograd notes](https://docs.pytorch.org/docs/stable/notes/autograd.html#division-by-zero-in-autograd).
It is a possible mechanism here, not a diagnosis of a particular loss term.

## Changes

- Check gradients after each backward pass when no GradScaler is active (BF16).
  This identifies the first execution chunk that makes accumulated gradients
  invalid, before a later chunk or clipping can obscure it.
- Check again before manual clipping, keeping the existing optimizer-step guard.
- Include epoch, batch index, logical/execution chunk metadata, transcript IDs,
  dataset IDs, length range, and affected-parameter count in the exception.
  These identify the contributing chunk, not necessarily one culpable row.
- Wire `trainer.detect_anomaly` into the real-data Trainer. It previously was
  absent from the explicitly constructed Trainer kwargs.
- Add `--detect-anomaly` to the resume helper and `DETECT_ANOMALY=1` to the
  dedicated N=114 Slurm wrapper. An autograd RuntimeError during manual backward
  also gets the current chunk context as an exception note.

No precision, model, loss, batch, optimizer, or gamma-bound setting is changed.
Gradient checks add a parameter scan/device synchronization per backward pass.
Autograd anomaly tracing is off by default and is slower when enabled.

## Diagnostic continuation on Leonardo

After copying the changed code/config and resume scripts to the cluster:

```bash
sbatch --export=ALL,RUN_ROOT=/path/to/existing/exp8,DETECT_ANOMALY=1 \
  resume_real_exp8_N114_256G.slurm
```

The existing resume helper selects a usable full-state checkpoint, retains the
saved experiment and BF16 precision, and the wrapper keeps `data.num_workers=0`.
On a numerical failure, retain the anomaly warning's forward traceback as well
as the final exception and backward context. Those are needed to distinguish a
loss-operation failure from a recurrent/head backward failure. Anomaly detection
traces NaN-producing backward operations; the finite-gradient guards also catch
infinities.

The next targeted numerical fix should follow that evidence. Changing the
whole Trainer to 32-bit precision, suppressing the exception, or replacing bad
gradients with zeros is not part of this patch.

## Local validation

28 focused checks passed: gradient diagnostics (including an actual Lightning
CPU BF16 training failure with no optimizer update), existing loss contracts,
checkpoint-resume command preparation, unique biological forward equivalence,
and execution microbatch grouping/loss-gradient equivalence. Python compilation,
Slurm shell syntax, and `git diff --check` also passed. These checks do not
reproduce the Leonardo CUDA workload.

## Follow-up: can gamma's exponential overflow?

For this saved run, **finite bounded scores cannot overflow gamma's exp**.
The checkpoint's `model._extra_state` confirms `raw_log_gamma_bound=8.0`,
`fixed_reference` centering, and `geometric_mean_one` positional gauge.
`init_gamma=1.0` in the saved configuration.

Let q be a requested bounded score, c the weighted reference score at a
position, m the requested positional mean, and a the weighted reference
positional mean. The final log-gamma is `q - c - m + a`. Each of these four
quantities lies in [-8, 8], so a conservative final bound is [-32, 32].
The raw bound must not be mistaken for the final centered bound.

| Quantity | Reachable/conservative range |
|---|---|
| Raw gamma | exp(-8) to exp(8), approximately 0.000335 to 2981 |
| Final gamma, both gauges | exp(-32) to exp(32), approximately 1.27e-14 to 7.90e13 |
| BF16 largest finite number | 3.39e38; exp overflows near log-input 88.72 |

Reference centering explicitly accumulates and returns log-gamma in float32
inside the existing BF16 mixed-precision run. Even a BF16 exponential at +/-32
is finite, positive, and has a finite derivative in the local test. No Trainer
precision change was required.

Tests exercise the actual fixed-reference centering method with 114 adversarial
bounded dataset profiles under CPU BF16 autocast, including the separate
training reference evaluations and evaluation-time reference reuse. They
reach absolute log-gamma 31.5954 and gamma up to 5.27e13, with finite backward
gradients through the reference scores.

This rules out ordinary **gamma exponential range overflow** under the checked
bound and finite-input assumptions. It does not repair NaNs already present in
the neural head or exclude an overflow when gamma multiplies a sufficiently
large incoming gradient.

There is a distinct downstream numerical limitation: an adversarial profile
with gamma peak 5.27e13, mean-one biological peak 128.5, and target mean about
996 gives a finite mu peak 6.74e18. The existing raw PCC computation multiplies
its variances and overflows, reporting zero instead of the float64 reference
PCC -0.108042. Its gradients remained finite in this reproduction, so this is
**not a reproduction of the observed non-finite-gradient crash**. It shows
that a safe exponential alone does not bound all loss intermediates. The case
deliberately approaches gamma saturation; the copied epoch-aggregated logs
cannot establish whether the failing batch approached it.

A whole-model logarithmic rewrite is not required by these findings and has
not been made. If the diagnostic rerun implicates PCC or the mean product,
stabilize that specific reduction/product; simply computing gamma in log space
would not by itself repair a PCC variance product after exponentiation.
