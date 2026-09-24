# Classical training performance review, 2026-09-08

The strongest compute bottleneck candidate in
`main_ribounmix_multidataset.py` is the **dataset-bias BiGRU,
including its additional fixed-reference evaluations and backward passes**.
The strongest avoidable overhead candidate is **repeated host synchronization
inside transcript validation and metric reductions**. Both are active in
ordinary `standard_nb` training.

These conclusions combine source inspection, archived A100 training logs, and
small CPU execution probes. Sandboxed commands initially reported
`torch.cuda.is_available() == False`. A subsequent approved check outside the
sandbox detected the local NVIDIA GeForce RTX 3070 Laptop GPU (8 GB nominal,
7.66 GiB reported by PyTorch), and a CUDA matrix forward/backward smoke test
passed. GPU access was restricted by the execution environment; the hardware
and CUDA installation work. Neither a model CUDA phase-time breakdown nor an
end-to-end GPU speedup has been measured. The original review left training
code and configuration unchanged; the subsequent implementation is noted below.

## Implemented: CPU transcript validation only

At the user's request, only the transcript-validation optimization was applied.
The collator now unconditionally verifies repeated codons and optional biological/
bias sequence features on CPU before canonical packing. Equal codons establish
equal lengths, generated masks, base biological features, and position inputs.
`ValidatedTranscriptMetadata` carries immutable Python tuples for group mapping,
canonical row lookup, and lengths through worker IPC and Lightning transfer.

The model consumes that metadata instead of repeating row-wise `equal/allclose`
checks and canonical `nonzero` searches on the GPU. Requested/reference bias GRUs
receive the validated host lengths, including repeated lengths for reference
chunks, instead of copying reduced GPU masks back to CPU. Direct model calls
and older batches without this metadata retain the existing validation path.
Callers that modify collated rows or inputs must discard their metadata.

No NB/metric optimization, reference-encoder reuse, architecture, parameter,
checkpoint, dropout, loss, gradient protection, or training-setting change is
included. The other opportunities below remain proposals. GPU throughput still
requires measurement on the target machine.

## Scope and configuration

The review started with `current_configured_model.md`,
`grouped_optimizer_batching.md`, the performance section of
`project_reanalysis_2026-09-05.html`, and the recent nonfinite-gradient report.
It followed the main entry point through the datamodule, collator, model,
Lightning `training_step`, loss, diagnostics, and optimizer hooks.

The current YAML takes precedence over older documentation snapshots:

- Biological encoder: 97 input channels, two bidirectional GRU layers,
  hidden size 256 per direction.
- Bias encoder: 68 embedding channels, two bidirectional GRU layers,
  hidden size 256 per direction. Position features enter the head.
- Optional sequence features all use `route: none`.
- GRU dropout is zero; gamma and alpha head dropout is 0.1.
- BF16 mixed precision is already enabled; anomaly detection is off.
- Standard replica NB, raw consensus PCC, and NB-VST consensus PCC are active;
  mean-gradient beta and gamma regularization are zero.
- Biological sequence deduplication, length sorting, pinned memory, persistent
  workers when workers are enabled, and disabled training per-dataset plots/
  metrics already exist.

## What one iteration means

`data.batch_size=32` is a **per-dataset logical quota**, not 32 physical rows.
With two fully paired datasets, this means 32 transcripts and 64 pair rows.
The sampler splits large logical batches into complete-transcript execution
chunks; each chunk appears as one Lightning progress-bar iteration.

A closely matching archived run is
`results/my_exp8_a100_b32_20260906_114340/N114/full/`:

| Recorded quantity | Value |
|---|---:|
| GPU | NVIDIA A100-SXM-64GB |
| Training transcripts | 14,347 |
| Transcript-dataset rows per epoch | 1,262,719 |
| Median logical batch | 34 transcripts, 3,044 pair rows |
| Optimizer updates per epoch | 417 |
| Execution chunks per epoch | 4,371 |
| Epoch 0 training time | 59:45, 1.22 chunks/s |
| Subsequent validation time | Approximately 6:47 |
| Average execution chunk | 3.28 transcripts, 289 pair rows |
| Execution chunks per optimizer update | 10.48 |

Source locations: `logs/resume_launcher.log` lines 673–703, 10217, and 11191.
The latest TensorBoard events confirm the execution transcript mean. The
resume manifest overrides workers from four to zero and uses reference chunk
size 16. Current base YAML reference chunk size is 32. The archived timing is
not proof that this is the user's exact run or that it used every subsequent
diagnostic change in the current checkout.

Compare optimizer updates/s, transcripts/s, and epoch time when changing
execution limits. A larger chunk may lower displayed iterations/s while
finishing the same epoch sooner.

## Main findings and small improvements

### 1. Bias recurrence is duplicated for requested reference rows

`Models/RiboUnmixModel/RiboUnmixModel.py:1245` evaluates the requested
dataset rows. Its fixed-reference method at line 913 additionally evaluates
every reference dataset for each unique transcript. Reference computations
remain connected to autograd.

For an execution chunk with B requested rows, U unique transcripts, and R
reference datasets, recurrent sequence work is:

```
biological GRU: U
bias GRU:       B + U*R
```

For complete groups whose requested and reference datasets coincide, the bias
encoder processes the same sequence/dataset inputs twice. In the N114 archived
epoch, the formula gives 1,262,719 requested plus 1,635,558 reference sequences.
With reference chunk 16, each execution chunk calls the bias GRU nine times:
one requested call plus eight reference calls.

The substantial implementation improvement is to compute each deterministic
**bias encoder output** once for the union of requested/reference pairs, then
gather it into the two head evaluations. Keep the requested and reference
dropout head evaluations independent and preserve reference gradients. This
can remove approximately half the bias recurrent work for complete panels,
or about 44% of bias sequence evaluations for the archived partial-support
N114 workload. Those percentages describe recurrent row work, not measured
wall-time savings. Identical random-number trajectories are not guaranteed.

Simply sharing final gamma scores, calling `eval()` during training, detaching
the reference center, reducing the reference panel, or moving dataset identity
after the GRU changes training semantics. Cross-step caching is invalid because
parameters change.

### 2. Repeated transcript checks force many GPU/CPU round trips

`RiboUnmixModel.py:694` compares every repeated row to its canonical row with
`torch.equal` or `torch.allclose`. The calls at line 805 check codons, masks,
positions, and the 97-channel biological input under the current configuration.
They return Python booleans, forcing host decisions from CUDA results.

The count is **4*(B-U) comparisons per execution chunk**. A complete logical
batch of 32 transcripts over 114 datasets performs 14,464 comparisons across
its chunks. Applying the same formula to the archived epoch counts gives
approximately five million comparisons per epoch. These are especially
wasteful for features copied from the same canonical sequence earlier in the
forward pass.

Validate immutable transcript properties in the CPU dataset/collator and pass
validated canonical indices into the model. For public/manual model inputs,
retain validation or perform a single vectorized check per field. Avoid merely
deleting the input contract.

Related small improvements:

- `RiboUnmixModel.py:1124` rebuilds canonical indices using one GPU `nonzero`
  per transcript; collate already knows these indices.
- `DatasetBiasSubmodel.py:73` recomputes lengths from GPU masks and copies them
  to CPU on every GRU call. Pass the existing CPU lengths and repeat them for
  reference rows. Retain packed GRUs so padding does not alter reverse context.
- `RiboUnmixLightningModule.py:205` checks values/weights and regroups
  rows for every scalar aggregation. Cache group indices and denominators
  once per batch and share immutable metadata validation across reductions.

PyTorch explicitly identifies tensor-to-host operations and CUDA-dependent
Python control flow as synchronization hazards in its
[performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization).
[The nonzero documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nonzero.html)
confirms its CUDA synchronization behavior, and
[packing documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.pack_padded_sequence.html)
requires CPU lengths.

### 3. Standard NB computes quantiles of a tensor of ones

`RiboUnmixLightningModule.py:534` sets all mean-gradient weights to one
when beta is zero, but line 651 still computes three quantiles over their
flattened valid positions. The current operation sorts this constant tensor.
Return five constant-one summaries directly for beta zero.

Lines 596–634 also perform four sequence reductions; in standard NB three
reduce the same `raw_nll`, while the alpha-branch diagnostic is zero. Reuse the
raw result and create the zero per-sample diagnostic directly in a dedicated
standard-NB path. Retain other modes' separate behavior.

A temporary CPU probe used the actual standard-NB configuration and a
1024-by-500 input, representing 512 pairs with two replicas. Replacing only
the quantile operation with known constants preserved **all returned tensors
and both mean/alpha gradients exactly**. Seven post-warmup paired measurements
gave median forward/backward times of 88.1 ms originally and 72.1 ms with the
shortcut. A profiler observed 16.5 ms in quantile, including 14.2 ms in sorting.
This is a CPU NB-only measurement, not a GPU or full-training speedup.

### 4. Diagnostic work runs on every execution chunk

`_compute_loss_and_metrics` at Lightning-module line 1684 computes the active
loss, alternate reduction modes, many gamma summaries, extra correlations,
and other diagnostics on every call. `_log_stage` then updates epoch metrics.
Increasing `trainer.log_every_n_steps` does not skip these computations.

Separate the required objective from optional diagnostics. Compute optional
training diagnostics less frequently and under `no_grad`, retaining the full
validation/checkpoint metrics. Sampled training diagnostics must be labeled as
sampled rather than presented as exact epoch means.

The immediate existing switch
`training.grouped_optimizer_batch.log_batch_structure=false` avoids the extra
grouping, GPU checks, and CPU copies at line 2941 without changing optimization.
It only disables that particular diagnostic block, not all metrics.

Current BF16 training also scans parameter gradients after every backward,
before manual clipping, and before optimizer step (lines 3265–3426). There
has been an actual N114 nonfinite-gradient failure, so preserve protection.
If profiling finds this costly, consolidate redundant scans while retaining
failure detection before clipping/update and the desired chunk localization.
Keep anomaly tracing limited to diagnostic reruns.

### 5. Tune execution sizes using memory and epoch throughput

For an archived run using reference chunk 16, test 32 and then 64 if device
memory permits. With 114 references, requested-plus-reference GRU calls per
chunk become 9, 5, and 3 respectively. This changes call sizes and overhead;
the number of reference sequences remains the same. Current base YAML already
uses 32, and a two-dataset run gets no benefit from increasing it past two.

Larger execution row/token caps can similarly amortize Python, logging, and
launch overhead while keeping `data.batch_size` and the optimizer target fixed.
Reference activations remain live until backward; bigger chunks can increase
peak memory, and changing chunk shapes can change dropout draws. Do not assume
chunk size alone bounds total reference activation memory.

## Input pipeline and secondary ideas

The actual CPU `collate_fn` took roughly 10–28 ms for representative synthetic
chunks with 64–456 rows, 1–32 transcripts, lengths 500–4000, and 6–24 replicas
(one CPU thread, three warmups, twenty measurements). This excludes dataset
access, worker IPC, pinning, and transfers. It suggests ordinary collation
alone is unlikely to explain one second per chunk; it does not exclude loader
wait or RAM pressure in the production job.

Increasing workers blindly is unlikely to be the first win. Inputs are already
in memory and spawned workers can duplicate large dataset state. The dedicated
N114 resume deliberately uses zero workers. Profile loader wait and resident
memory before changing that choice.

PackedSequence transfer could be made explicitly nonblocking; the installed
Lightning helper only adds `non_blocking=True` automatically for Tensor
instances. This is secondary to repeated device-to-host lengths and checks.
Padded replica slots also incur unnecessary likelihood work when replica counts
differ, but computing only valid replicas requires preserving the within-pair
replica mean exactly.

BF16 and biological deduplication are already enabled. Whole-model compilation
is a later experiment because the current forward contains Python grouping,
host checks, and packed RNN handling. `cudnn.benchmark=True` is primarily a
convolution autotuning setting, so it is not the obvious lever for these GRUs.
See the [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner).

An operational issue applies only to the old
`main_ribo_queueing_multi_dataset.slurm`: it requests eight Slurm tasks and two
GPUs, then invokes a two-GPU trainer through `srun`. Verify task/rank/GPU
allocation if using that launcher; its topology can create contention or an
invalid distributed launch. The archived Experiment-8 launchers are separate.

## Recommended order and remaining measurement

1. Make standard-NB constant summaries/reductions cheap; turn off unnecessary
   batch-structure diagnostics.
2. Move immutable per-transcript validation and canonical metadata construction
   off the CUDA hot path; share grouping metadata and CPU lengths.
3. Benchmark larger reference/execution chunks on the target GPU.
4. Reuse deterministic bias encoder outputs while preserving independent head
   dropout and reference gradients.

For an exact bottleneck ranking, collect a short CUDA profiler trace after
warmup on representative short, medium, and long execution chunks. Separate
loader wait, transfer, biological forward, requested bias forward, reference
bias forward, objective, diagnostics/logging, backward, gradient checks, and
optimizer. Inspect recurrent kernels alongside `cudaStreamSynchronize`,
`aten::equal/allclose`, scalar extraction, `nonzero`, and quantile/sort. Report
unprofiled steady-state throughput separately because tracing adds overhead.

The entry point builds explicit Trainer kwargs at line 2508. It does not forward
a `trainer.profiler` YAML field, so merely adding a Hydra profiler override
will not activate profiling; profiler support must be passed to Trainer or
attached through a separate profiling harness.
