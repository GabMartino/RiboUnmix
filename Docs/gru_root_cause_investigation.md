# Repeated Exp8 GRU failure: evidence and exact-operation capture

Subsequent user-approved experiment: [state-carrying bias-GRU TBPTT](bias_gru_tbptt.md)
is now available as an explicit training-gradient change (default remains off).
Its tests and limitations are separate from the exact-operation capture below.

## What is known, and what is not

The supplied UNIVIE manifest confirms `log-space-nb2-v1`, H100 80 GB,
`context_gru_precision=inherit`, and full-state resume from epoch 18, step
7638. The reported failure is at epoch 20, batch 91, on six transcripts of
4968--5089 codons. Eight bias-embedding/first-layer-GRU parameter tensors have
non-finite gradients. Those are locations of bad *accumulated* gradients, not
proof of the first failing operation.

The local epoch-18 model and optimizer tensors are finite. Its first-layer
candidate recurrent spectral norms are approximately 9.55 (forward) and 8.72
(reverse); the gate recurrent matrices are also unconstrained. These norms
indicate that a contractivity guarantee is absent, **not** that the realized
gated recurrent Jacobian necessarily expands.

Actual-weight/actual-sequence CPU checks used `ENST00000615648.2` (5089 codons)
with dataset IDs 6 and 130, the saved selected-panel embedding center, and the
unchanged bias submodel. A diagnostic mean log-space NB loss (target 2, alpha
1) produced the following results:

| Bias precision policy under CPU BF16 AMP | Dataset ID | Loss | Maximum absolute parameter gradient | All gradients finite |
|---|---:|---:|---:|---|
| inherit | 6 | 3.40922 | 2.9375 | yes |
| inherit | 130 | 4.35070 | 5.625 | yes |
| float32 | 6 | 3.42001 | 2.921875 | yes |
| float32 | 130 | 4.36942 | 5.625 | yes |

These checks use **the older checkpoint, CPU kernels, and a diagnostic loss
with dropout disabled**, not the failed epoch-20 weights, CUDA kernel, loss,
reference evaluation, or dropout realization. They do not reproduce or rule
out the reported production failure. No exact H100 fix is claimed.

## Why changing the loss to logs is insufficient

For a recurrent update `h_t = F(h_(t-1), x_t)`, the state adjoint contains
products of `J_t = dF/dh_(t-1)`. Bounded sigmoid/tanh *outputs* do not bound
these products by one. An ordinary GRU has trainable hidden-to-hidden
matrices inside its gates and candidate state. In a one-dimensional example
with zero input/state, reset/update gates equal to 1/2, and candidate recurrent
weight w, the state derivative is `1/2 + w/4`. For w > 2, repeated products
grow exponentially while all forward states remain zero. A finite log-space
NB loss with a small output derivative does not change that fact.

`Tests/test_bias_gru_precision.py` demonstrates this mechanism with a finite
log-space NB loss: BF16 recurrence has non-finite backward, while the FP32
counterpart remains finite in that controlled case. It is not the production
failure replay. Global norm clipping after backward cannot repair a gradient
that overflowed inside the recurrence.

References: [Pascanu et al., On the difficulty of training RNNs](https://arxiv.org/abs/1211.5063),
[PyTorch GRU equations](https://docs.pytorch.org/docs/2.10/generated/torch.nn.GRU.html).

## New diagnostics: preserve the actual failed GRU invocation

The optional `--capture-gru-failure` resume flag, exposed as
`CAPTURE_GRU_FAILURE=1` in the remaining-Exp8 launcher, installs a hook on the
actual `CudnnRnnBackward` node. The hook checks its incoming adjoints and its
generated gradients **before** they reach parameter accumulators:

- Finite incoming adjoints, invalid generated gradients: invalid values were
  generated inside the fused GRU backward.
- Already-invalid incoming adjoints: investigate the downstream loss/head/
  normalization path first; this is not proof that the GRU originated them.
- Both finite: the hook changes nothing and writes no file.

On failure it saves the packed input, recurrent weights, incoming adjoints,
packing metadata, RNG state and runtime settings under the existing task's
`diagnostics/gru_failures/`. Names are unique and no old captures/checkpoints
are overwritten. The payload can be large (potentially GB for a large
reference-panel invocation). No optimizer step, gradient replacement, group
skipping, or precision fallback occurs. A `.pt` capture is **not** a training
checkpoint. Capture requires a CUDA packed GRU with zero recurrent dropout,
as used by this run. A different backend or nonzero recurrent dropout fails
explicitly instead of claiming exact capture support.

`replay_bias_gru_failure.py` then replays only that GRU operation with its
actual incoming adjoints. It does not invent a loss or reconstruct a training
batch. Original, FP32 and FP64 execution can be compared; a non-cuDNN control
is available. A dtype change also changes the forward numerical trajectory,
so this comparison alone is not a proof of a vendor-kernel bug.

## Cluster command, with unchanged BF16 recurrence for diagnosis

Sync these files and their existing dependencies:

- `Models/utils/gru_failure_capture.py`;
- `Models/RiboUnmixModel/DatasetBiasSubmodel.py`;
- `Models/RiboUnmixLightningModule.py`;
- `resume_real_experiment_from_checkpoints.py`;
- `resume_real_exp8_remaining_aggressive.slurm`;
- `replay_bias_gru_failure.py`;
- `Tests/test_gru_failure_capture.py`.

Inside a native-BF16 CUDA allocation, first run:

```bash
python -m unittest Tests.test_gru_failure_capture -v
```

The CUDA integration case must pass, not skip. Local verification: 33 tests
passed and this one CUDA integration test skipped; no usable local GPU is
available. Do not call that a validated CUDA capture deployment.

From the project on UNIVIE, submit only the failed task:

```bash
RUN_ROOT="$PWD/results/my_exp8_a100_b32_20260906_114340" \
BIAS_GRU_PRECISION=inherit EXP8_RUNTIME_PROFILE=aggressive \
CAPTURE_GRU_FAILURE=1 DETECT_ANOMALY=0 \
sbatch --array=4 resume_real_exp8_remaining_aggressive_univie.slurm
```

This is an **instrumented reproduction**, not a claimed repair. Explicitly
disable anomaly detection here so an earlier NaN trap cannot preempt the
node-level capture. The finite-gradient safeguards remain active.

Inside a subsequent GPU allocation, substitute the exact path printed by the
failure, and use a new output filename:

```bash
python replay_bias_gru_failure.py PATH_TO_CAPTURE.pt \
  --modes original,float32,float64 --device cuda \
  --output results/gru_replay_comparison_01.json
```

Replay does not train or alter saved results. A nonzero exit code is expected
if any requested mode reproduces invalid gradients or encounters an error.
Only load captures from trusted sources (`torch.load` uses pickle).

## Architectural fix versus continuing the original experiment

Eliminating recurrent amplification *by construction* requires a constraint
on the recurrence, not just loss algebra. For example, an input-gated update
`h_t = a(x_t) * h_(t-1) + (1-a(x_t)) * u(x_t)`, with 0 <= a <= 1, has a
diagonal temporal Jacobian whose norm is at most one. It removes the
hidden-to-gate feedback of an ordinary GRU; stacked bidirectional variants
can still use complete-CDS context. This bounds the temporal mechanism, not
every possible floating-point operation in the complete network.

Such an encoder changes the model/hypothesis class. It must be an explicitly
versioned experiment, not an in-place continuation labelled as the original
GRU run. No architecture, loss, gradient semantics, saved result, or original
checkpoint has been changed by this diagnostic work. Approval of that
scientific change is pending.
