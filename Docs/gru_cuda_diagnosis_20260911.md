# CUDA diagnosis: requested BF16 is not the cuDNN GRU's actual precision

This report records the **pre-fix** investigation. The subsequent
[FP32 CUDA-GRU protection](fp32_cuda_gru_policy.md) is now implemented:
`inherit` no longer allows the unsafe CUDA autocast path. Existing artifacts
below retain their original results; new probes record the compute-policy
version. The standalone mechanism probe explicitly bypasses protection only
for its labeled legacy baseline. Production failure capture now observes the
protected recurrence, not the old FP16 path.

## Main finding

On the tested RTX 3070 Laptop (8 GB), PyTorch 2.10.0+cu128 / cuDNN 9.10.2,
`autocast("cuda", dtype=torch.bfloat16)` runs the model's fused cuDNN GRU in
**FP16**. Its actual output and incoming backward adjoints are `torch.float16`.
The protected `context_gru_precision=float32` path returns FP32.

This was measured both in the saved model and in an independent minimal GRU.
It agrees with the matching [PyTorch 2.10 cuDNN autocast implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cudnn/AutocastRNN.cpp#L62-L98),
which explicitly casts weights, inputs, and hidden states to `at::kHalf`.
Thus an outer BF16 setting alone does not give this operation BF16's exponent
range. The largest finite FP16 value is 65,504; BF16's is approximately 3.39e38.

**A numerical hazard is confirmed; the exact cluster failure is not yet
reproduced.** The cluster's actual operation/version must still be measured.
No production model, training configuration, or checkpoint was changed by
this diagnostic work. All probes are forward/backward only, without optimizer
updates or gradient sanitization.

## Tests with downloaded checkpoints and actual inputs

Both inspected checkpoints have finite model tensors and finite saved
optimizer tensors:

- N114 quality-ranked: epoch 11, step 5004, SHA-256
  `0a8aad2728bdc0fc78e99ec288e2baeab01a3f01979474cc1def985811c6f98f`.
- N040 equal-weight pair02_A: epoch 19, step 8040, SHA-256
  `671f5ab37a6ecb990d9ef0df38b5e540e868d23975573c6b228f1a32134b81b1`.

The N114 bias-only scan tested all 114 selected dataset identities for each of
the five reported sequences: 570 combinations, in 40 backward calls. All
were finite. The N040 scan tested the six reported 4,968–5,089-codon sequences
against all 40 identities under both requested-BF16 and protected-FP32
policies: 60 backward calls, all finite. These two scans use a deliberately
simple diagnostic NB loss, **not** the original training objective.

The stronger N114 test uses the actual replica counts, arithmetic consensus,
saved training-only reliability references, checkpoint reference-panel
weights, original sequence features, production collator, complete model,
and production NB + raw-PCC + VST-PCC loss. It retains training dropout.
Each probe contains one complete transcript group and all 114 fixed-reference
datasets. Its loss is multiplied by 1/33, matching a complete group's
contribution to the reported 33-group logical batch (accumulation factor 1).

| Reported transcript | Codons | Observed positive-weight dataset pairs |
|---|---:|---:|
| ENST00000374012.8 | 1013 | 114 |
| ENST00000367618.8 | 1013 | 47 |
| ENST00000380099.4 | 1013 | 26 |
| ENST00000366922.3 | 1013 | 114 |
| ENST00000306858.8 | 1012 | 110 |

Three dropout seeds (42, 43, 44) were tested for each transcript and policy:

| Requested policy | Complete, finite backward passes | Not completed |
|---|---:|---:|
| BF16 autocast / inherited full GRU | 15 | 0 |
| BF16 autocast / TBPTT 1024 | 15 | 0 |
| BF16 autocast / TBPTT 512 | 15 | 0 |
| BF16 autocast / TBPTT 256 | 15 | 0 |
| BF16 autocast / protected FP32 bias GRU | 6 | 9 local CUDA OOMs |

The local OOMs are memory-capacity limits, not the cluster's nonfinite-gradient
error. The output JSON preserves them explicitly; they are not counted as
successful numerical tests. The initial five requested-BF16 production-loss
probes also completed successfully.

Limitations: the epoch-11 checkpoint precedes the failed epoch-12 update by
roughly 281 logical optimizer updates. One-group execution changes GPU kernel
shapes and dropout realizations compared with the five-group cluster chunk.
The hardware/software may differ. Consequently these results do not prove
that the production failure is fixed or that the actual failing state is
numerically stable.

The supplied launch command, copied configuration and checkpoint do not
record an enabled TBPTT window. The default is zero. Even an explicit 1024
window leaves these 1012–1013-codon sequences untruncated.

## Controlled mechanism test, distinct from checkpoint replay

A one-unit bidirectional GRU is constructed with zero inputs/hidden states
and one nonzero recurrent candidate weight of 2.2. Along its forward
direction, the local derivative at zero is

\[
\frac{\partial h_t}{\partial h_{t-1}}=\tfrac12+\tfrac14(2.2)=1.05.
\]

The entire forward output stays exactly zero. A small NB loss on its endpoint
is finite and identical in all cases (0.0041588834). Its incoming adjoint is
approximately -0.001. Backward nevertheless contains a long recurrent
Jacobian product that exceeds FP16's range.

| Policy | Measured recurrent output dtype | Backward |
|---|---|---|
| BF16 autocast, full GRU | FP16 | Nonfinite **inside** fused GRU backward |
| Explicit FP32 GRU, BF16 outer context | FP32 | Finite |
| Explicit native BF16 GRU | BF16 | Finite |
| BF16 autocast, TBPTT 1024 | FP16 | Nonfinite |
| BF16 autocast, TBPTT 512 | FP16 | Nonfinite |
| BF16 autocast, TBPTT 256 | FP16 | Finite |

The failure capture confirms **finite incoming adjoints and nonfinite
gradients generated by the cuDNN operation**. Replaying that same captured
input, weights, and incoming adjoints reproduces the original failure;
FP32 and FP64 replay are finite, with maximum absolute gradient about 2.917e19.
This isolates a backward-range failure without a large loss, an invalid
target, an optimizer update, or exponentiating an excessively large log-mean.

The explicit BF16 capability probe takes the native, non-cuDNN recurrence on
this installation. It is not a drop-in, performance-validated training fix.
It also explicitly rounds this toy model's parameters to BF16; a production
mixed-precision implementation should retain FP32 master parameters.

These controlled results do **not** establish a universal safe truncation
window. They show why log-space losses and a BF16 configuration label do not
by themselves protect the recurrent backward pass. Clipping a parameter
gradient after backward cannot repair a nonfinite value already generated
inside that pass.

## What to change next, and what still needs confirmation

First remove the unintended FP16 recurrence from any proposed BF16 training
policy. The existing, immediate option is to keep `trainer.precision=bf16-mixed`
and explicitly protect the bias GRU with
`++model.dataset_bias_params.context_gru_precision=float32`. This is **not**
full-model FP32 training; other operations retain their mixed-precision policy.
The separate biological GRU is also a cuDNN GRU and must be audited separately
if requiring a policy with no FP16 recurrence anywhere.

A genuine BF16 recurrent implementation is another option, but needs actual
dtype/gradient assertions, retained FP32 optimizer parameters, forward and
gradient comparisons, and throughput measurement. Do not call the current
cuDNN-autocast/TBPTT path genuine BF16 merely because the outer context says so.

To establish the cause of the specific cluster crash, capture the actual
failed invocation using the resume helper's `--capture-gru-failure` flag,
keeping inherited precision and full BPTT (`--bias-gru-tbptt-window 0`) for
that diagnostic replay. Use an allocated GPU, preserve the original execution
budgets, and do not run concurrently against the same training directory.
The capture should identify whether the adjoint arrives nonfinite or becomes
nonfinite in the GRU, and can then be replayed independently in FP32/FP64.
`CAPTURE_GRU_FAILURE` is not currently forwarded by the quality-ranking Slurm
launcher, so setting that environment variable alone is insufficient: pass
the Python helper flag explicitly. No diagnostic cluster job was submitted.

## Artifacts and regeneration

- [Checkpoint/real-sequence bias scan](../diagnose_bias_gru_cuda.py).
- [Actual-data production-loss probe](../diagnose_checkpoint_training_cuda.py).
- [Controlled precision/overflow probe](../diagnose_gru_precision_cuda.py).
- [N114 actual-pair source table](../results/gru_cuda_training_N114_epoch11_bf16_v2/observed_pairs.csv).
- [N114 policy comparison](../results/gru_cuda_training_N114_epoch11_comparison/probes.jsonl).
- [N040 checkpoint probes](../results/gru_cuda_diagnostic_N040_epoch19_comparison/probes.jsonl).
- [Controlled mechanism results](../results/gru_cuda_precision_mechanism/probes.jsonl).
- [Captured-operation replay](../results/gru_cuda_precision_mechanism/replay.json).

Each output directory also contains a runtime/checkpoint audit. The labels
`bf16` in earlier JSON runs name the requested autocast policy; the
`gru_traces[].output.dtype` field records the measured precision. TBPTT splits
bypass the `nn.GRU` module hook, so use the controlled probe to inspect their
pre-normalization output dtype.

From the project root, with CUDA available (choose fresh output names):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python diagnose_gru_precision_cuda.py \
  --output-dir results/gru_cuda_precision_recheck
```

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python diagnose_checkpoint_training_cuda.py \
  --task-dir results/real_exp8_L_stability_quality_rank_10components/cumulative_qrank10components_p1.0_seed42/N114/trainseed42 \
  --transcripts ENST00000374012.8,ENST00000367618.8,ENST00000380099.4,ENST00000366922.3,ENST00000306858.8 \
  --modes bf16,fp32,tbptt1024,tbptt512,tbptt256 --seeds 42,43,44 \
  --output-dir results/gru_cuda_training_recheck
```

For fast reuse without rereading the 114 count files, add
`--prepared-batches results/gru_cuda_training_N114_epoch11_bf16_v2/prepared_batches.pt`.
The cache is accepted only for the same checkpoint hash. On the cluster use
`python` from the activated project environment instead of `.venv/bin/python`.

The following existing regression suites passed on this CUDA-enabled session:
28 tests, no skips, 34.1 seconds. CUDA cases now execute rather than skip.
Passing an existing test named BF16 describes its autocast context, not a
guarantee that every internal operation actually used BF16.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m unittest \
  Tests.test_gru_failure_capture \
  Tests.test_bias_gru_tbptt \
  Tests.test_stable_training_numerics -v
```
