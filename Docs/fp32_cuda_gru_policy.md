# FP32 recurrence with BF16 mixed-precision heads

## Implemented policy

`cuda-amp-gru-fp32-v1` disables CUDA autocast **inside both GRUs**, using the
original FP32 parameters and FP32 recurrent inputs. The biological head,
dataset-bias heads, and other neural operations retain the surrounding
BF16 mixed-precision policy. Scalar loss/normalization algebra remains as
previously implemented.

Coverage includes training, validation, prediction, packed/unpacked bias
inputs, the complete fixed-reference panel, and every TBPTT window. The
TBPTT helper is protected even if called directly. Old frozen configurations
with a missing precision field or `context_gru_precision=inherit` receive
the same CUDA protection; no launcher override can silently opt back into
the old autocast behavior. Direct true-low-precision model weights are rejected
when protected execution is required: retain FP32 master parameters.

This removes the confirmed, unintended FP16 recurrence. It is **not a promise
that arbitrary learned recurrences, inputs, GPU failures, or memory limits
can never cause an error**. Finite-gradient checks and stable norm clipping
remain active. There is no gradient replacement, skipped update, automatic
learning-rate reduction, or silent fallback after an invalid backward pass.

Preserved: parameter identities/order, model state-dict keys, saved optimizer
mapping, loss coefficients and reductions, dataset weights, split/panel
design, full-CDS supervision, hidden-state propagation, and configured TBPTT
window. Full BPTT is still the default. The new numerical precision changes
roundoff and therefore the optimization trajectory; it does not claim
bitwise continuation of the former FP16-GRU run.

CPU execution and explicit non-AMP FP64 reference calculations are unchanged
unless the existing explicit `float32` bias option is selected.

## Implementation and verification

The shared context is [gru_precision.py](../Models/utils/gru_precision.py).
It wraps the existing recurrence calls; it does not create new recurrent
modules or parameters. The policy is recorded in new Lightning checkpoint
hyperparameters, startup/failure logs, and resume manifests.

CUDA regression checks verify:

- actual FP32 recurrent outputs, not merely FP32 parameter storage;
- BF16 head outputs and restoration of the outer autocast context;
- agreement of full-BPTT forward values and gradients with explicit FP32;
- packed padding/order, direct TBPTT calls, and train/evaluation paths;
- strict checkpoint and Adam-state compatibility;
- the controlled finite-loss overflow case now has finite gradients with
  **no truncation**, followed by a finite clipped Adam update.

The strengthened long-sequence tests run the full model and all three active
losses at 1,807, 3,175, and 5,089 positions, including optimizer updates.
The numerical/CUDA suites passed all 41 tests. The guard, resume, Slurm,
cumulative-ranking and independent-panel suites passed all 40 test executions.

The real-data check used the downloaded N114 epoch-11 checkpoint and all five
reported transcripts, three dropout seeds, all 411 observed pairs across
those transcripts, and the complete 114-dataset reference. All **15 backward
passes were finite**. All 90 traced recurrent calls (15 biological, 75 bias)
were FP32 with CUDA autocast disabled internally.
[Actual results](../results/gru_fp32_fix_N114_epoch11/probes.jsonl) and
[audit](../results/gru_fp32_fix_N114_epoch11/audit.json).

For this 8-GB local GPU only, saved backward activations were stored on CPU
and transferred back without changing dtype. This diagnostic option is
**not** enabled in training launchers. Each test used one complete transcript
group, scaled by 1/33 to match its logical-batch contribution; no dataset pairs
or reference derivatives were dropped. Peak GPU allocation was about 4.88 GiB.
These checks validate the patch at the saved state, not an exact replay of
the later failed cluster update.
The two smaller complete transcript groups were also rechecked without CPU
offload: both passed, with loss differences at most 1.79e-7 and differences
between corresponding parameter-gradient maximum magnitudes at most 1.46e-11.
[No-offload checks](../results/gru_fp32_fix_N114_no_offload/probes.jsonl).
The [before/after controlled probe](../results/gru_fp32_fix_mechanism/probes.jsonl)
retains the failing raw-autocast baseline alongside the passing protected paths.

## Deployment

Copy the updated project code to the cluster before resuming. In particular,
include these runtime files together (do not copy only the Slurm launcher):

```text
Models/utils/gru_precision.py
Models/utils/gru_tbptt.py
Models/RiboUnmixModel/DatasetBiasSubmodel.py
Models/RiboUnmixModel/SharedProfileModel.py
Models/RiboUnmixLightningModule.py
resume_real_experiment_from_checkpoints.py
config/config_ribounmix_multidataset.yaml
run_real_exp8_L_stability_quality_rank.slurm
resume_real_exp8_remaining_aggressive.slurm
```

Keep the other existing dependencies, including `stable_numerics.py` and
`gru_failure_capture.py`. The UNIVIE wrappers use the shared launchers.
Already-running Python processes will not acquire the change; a newly
started worker must print:

```text
gru_compute_policy=cuda-amp-gru-fp32-v1 (both GRUs FP32 under CUDA AMP)
```

`trainer.precision` should remain `bf16-mixed`. There is no need to select
full-model `32-true`, reset the optimizer, or start the experiment from scratch.
The `float32` bias option is explicit in the base config and the quality-rank
launcher default; legacy `inherit` is also safe under CUDA AMP.

FP32 recurrence requires more GPU memory than the former FP16 recurrence.
For the first cluster continuation use the existing `auto` execution profile
instead of the largest aggressive budget. These remain hardware-dependent
starting points, not OOM guarantees; no execution profile changes numerical
precision, dataset membership, or logical batch size.

To resume just cumulative quality-ranked N114 on UNIVIE, from the project
root after copying the code:

```bash
EXP8_RUNTIME_PROFILE=auto BIAS_GRU_PRECISION=float32 \
sbatch --array=0 run_real_exp8_L_stability_quality_rank_univie.slurm
```

This retains the saved TBPTT policy. To make a deliberate full-BPTT continuation
from a previously truncated run, additionally set `BIAS_GRU_TBPTT_WINDOW=0`;
that is a separate training-gradient choice and is not applied automatically.
The quality-rank array mapping is N114=0, N080=1, N040=2.

For equal-weight Exp8 N040/pair02_A, the corresponding command is:

```bash
EXP8_RUNTIME_PROFILE=auto BIAS_GRU_PRECISION=float32 \
sbatch --array=4 resume_real_exp8_remaining_aggressive_univie.slurm
```

Both scripts retain their existing `dgx1` exclusion and results locations.
Do not submit overlapping jobs writing the same task directory. No training
jobs were submitted and no existing checkpoint/results were overwritten by
this implementation.

## Recheck on CUDA

Use the activated cluster environment's `python`, or `.venv/bin/python` locally:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest \
  Tests.test_cuda_gru_precision \
  Tests.test_bias_gru_precision \
  Tests.test_bias_gru_tbptt \
  Tests.test_gru_failure_capture \
  Tests.test_stable_training_numerics -v
```

The CUDA tests must execute, not skip. A standalone before/after mechanism
probe is also available:

```bash
python diagnose_gru_precision_cuda.py --output-dir results/gru_precision_verified
```

Its deliberately unprotected `autocast_bf16` baseline may fail; the
`protected_inherit` and `gru_fp32` cases must show FP32 recurrence and finite
backward. The fixed production paths never use that diagnostic bypass.
