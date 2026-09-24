# Bias-GRU TBPTT: 1,024-codon trial

Current CUDA execution uses [FP32 GRU protection](fp32_cuda_gru_policy.md)
for both encoders, including legacy `inherit` configurations and TBPTT.
The original trial observations below are historical. Precision protection
does not itself require TBPTT or change an existing truncation window.

## What changes mathematically

For each GRU layer and direction, the original update is

\[
h_s = F_\theta(x_s, h_{s-1}).
\]

Every K tokens, the next window receives `h.detach()`: the *same numerical
hidden state*, with its history removed from autograd. Within a window,
backpropagation is unchanged. Parameters remain shared across windows.

The forward direction starts at the CDS beginning; the reverse direction
starts at each transcript's **actual CDS end**, not the batch's padded end.
Both complete directional outputs of a layer are concatenated before the
next layer runs. Applying a stacked bidirectional GRU separately to isolated
CDS chunks would not preserve this forward function; this implementation
does not do that.

The forward model is mathematically unchanged. BF16/kernel partitioning can
change roundoff, so bitwise equality is not promised. Only the bias branch
uses TBPTT; the biological encoder is unchanged. The requested-dataset and
fixed-reference-dataset bias evaluations use the same policy.

Preserved: all CDS positions, full-CDS NB/PCC/PCC-VST losses, gamma centering,
dataset weights, full reference-panel derivatives apart from the explicitly
truncated temporal paths, logical batch/optimizer accumulation, model
parameter names/order, and optimizer-state compatibility. There is no hidden
reset, per-window PCC, per-window optimizer step, discarded batch, gradient
sanitization, or automatic precision fallback. Prediction/evaluation uses the
original full GRU, including when computing evaluation-time input gradients.

**This is a biased training gradient, not exact full BPTT.** A K-token window
limits each layer/direction's uninterrupted temporal chain, not every possible
path through the stacked bidirectional network and globally coupled losses.
For a publication comparison, report K and the checkpoint/epoch at which it
was enabled; do not label mixed-policy continuations as an unchanged training
protocol. See [Aicher et al., adaptive TBPTT](https://arxiv.org/abs/1905.07473).

## Why 1,024, and what it cannot guarantee

1,024 is a configurable first trial, not a measured optimum. Compared with
256, it keeps longer temporal gradient paths but permits more Jacobian
amplification. An unstable recurrence can overflow even within 1,024 steps;
truncation cannot guarantee finite gradients for arbitrary GRU parameters.
If that trial fails, compare an explicitly recorded 512-window continuation
from the same finite checkpoint, rather than silently changing the policy.

The existing finite-gradient guard and stable norm clipping remain enabled.
Clipping *after* backward cannot repair an infinity already generated inside
backward. TBPTT addresses the length of the recurrent gradient product instead.
The loss is still evaluated on the full CDS, and all window graphs live until
that loss is backpropagated. Therefore neither activation-memory reduction nor
a speedup is guaranteed. Fused GRU kernels are retained inside each window;
short batches that fit in one window retain the original fused stacked GRU.

## Configuration and checkpoint continuation

The opt-in Hydra settings are:

```text
trainer.precision=bf16-mixed
++model.dataset_bias_params.context_gru_precision=inherit
++model.dataset_bias_params.context_gru_tbptt_window=1024
```

The default window is **0 (full BPTT)** so existing experiments do not silently
change method. Nonzero recurrent dropout is incompatible with the split
execution's forward-equivalence contract; the saved runs use zero. Dropout
in the bias heads is untouched. Whole-GRU failure capture is also incompatible
with nonzero-window initial states and is rejected; use the ordinary numerical
guard/anomaly diagnostics with TBPTT.

The resume helper accepts `--bias-gru-tbptt-window 1024`. Explicit CLI choice
takes precedence, followed by a previous resume's recorded choice, then the
checkpoint hyperparameter. Otherwise the original command/config is retained.
An explicit `0` disables TBPTT. The audit records the window, its source, and
the changed-gradient caveat; new checkpoints record `bias_gru_tbptt_window`.
The startup/failure log reports the actual instantiated encoder's window.
`resume_mode=exact_full_state` describes *state restoration*, not equivalence
of the subsequent full-BPTT and TBPTT optimization trajectories.

The `BIAS_GRU_TBPTT_WINDOW` environment variable is wired into:

- `resume_real_exp8_remaining_aggressive.slurm` and its UNIVIE wrapper;
- `run_real_exp8_L_stability_quality_rank.slurm` and its UNIVIE wrapper.

After syncing the modified source files to UNIVIE, this command explicitly
continues the failed N040/pair02_A task **in its existing run directory**:

Sync the new `Models/utils/gru_tbptt.py` as well as
`Models/RiboUnmixModel/DatasetBiasSubmodel.py`,
`Models/RiboUnmixLightningModule.py`,
`resume_real_experiment_from_checkpoints.py`, and the two shared Slurm
launchers above. Sync `Tests/test_bias_gru_tbptt.py` for the GPU preflight.
The base config documents the default; frozen old configs accept the explicit
`++` override without being rewritten.

```bash
RUN_ROOT="$PWD/results/my_exp8_a100_b32_20260906_114340" \
BIAS_GRU_PRECISION=inherit BIAS_GRU_TBPTT_WINDOW=1024 \
CAPTURE_GRU_FAILURE=0 DETECT_ANOMALY=0 EXP8_RUNTIME_PROFILE=aggressive \
sbatch --array=4 resume_real_exp8_remaining_aggressive_univie.slurm
```

For a separately retained scientific comparison, make an independent copy of
the saved run first and point `RUN_ROOT` to that copy. Do not run two jobs
against the same task directory. The launcher restores the most advanced
usable full-state checkpoint, not necessarily the file named `last.ckpt`.
On Leonardo, use the same command with the non-UNIVIE `.slurm` filename.

For the cumulative quality-ranking launcher, pass the same precision/window
variables to `run_real_exp8_L_stability_quality_rank_univie.slurm`; its array
indices differ: N040=2, N080=1, N114=0. This does not change ranking, split,
reliability references, or the selected cumulative datasets.

## Verification

Reproducible automated checks:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest \
  Tests.test_bias_gru_tbptt \
  Tests.test_bias_gru_precision \
  Tests.test_stable_training_numerics \
  Tests.test_resume_real_experiment_from_checkpoints \
  Tests.test_exp8_remaining_aggressive \
  Tests.test_nonfinite_gradient_diagnostics \
  Tests.test_gru_failure_capture -v
```

Local result on PyTorch 2.10.0: **55 passed, 3 CUDA-only tests skipped**.
The tests cover the original GRU's forward values, independent scalar-cell
TBPTT parameter/input gradients, padding and reverse boundaries, state carry
versus state reset, full-BPTT gradients when K covers the CDS, checkpoint and
Adam-state loading, all-reference-dataset gradient flow, full-CDS combined
loss/optimizer BF16 smoke tests at 5,089 codons, and shell/Hydra resume wiring.

A controlled unstable one-dimensional GRU gives identical finite forward
values and a finite log-space NB loss: full CPU BF16 backward overflows,
while K=1,024 has finite gradients. This demonstrates the mechanism; it is
not the reported production failure.

A read-only check of the downloaded N040/pair02_A `last-v1.ckpt`
(epoch 19, global step 8040) used `ENST00000615648.2`, all 5,089 codons, dataset
ID 6, and the saved panel's dataset-embedding center. Strict bias-submodel
state loading passed. Full versus windowed FP64 GRU outputs differed by at
most **1.1213e-14**. With CPU BF16 autocast, `inherit`, K=1,024 and head dropout
disabled, a diagnostic NB loss (target 2, alpha 1, raw log-gamma as log-mean)
was **2.646733**; all generated parameter gradients were finite, with maximum
absolute value **1.015625**. This diagnostic does not use the actual training
targets, coupled reference loss, dropout state, or failing epoch-20 weights.

No usable local CUDA device was available. On the allocated native-BF16 GPU,
run at least `python -m unittest Tests.test_bias_gru_tbptt -v` before production;
its CUDA test must **pass, not skip**. The helper isolates PyTorch's internal
packed `_VF.gru` operator (the same primitive used by `nn.GRU.forward`); this
integration test is important when the cluster's PyTorch/cuDNN version differs.
Neither the exact H100 failure nor production throughput has been validated
locally. No saved experiment artifact was changed and no job was submitted
during this implementation.
