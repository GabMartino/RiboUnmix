# Group-aware optimizer batching

This document describes the optimizer-batching lifecycle shared by
`data.train_sampling_strategy=transcript_grouped_pairs` and
`transcript_grouped_multidataset_pairs`. The active latter strategy admits only
groups with positive eligible observations in at least two distinct selected
datasets. The profile/weight assertions and support check run before packing.

## Per-dataset physical quota and optimizer target

`data.batch_size` is the maximum number of transcript-dataset pair rows from
each represented dataset placed in one logical grouped batch. With $$D$$
datasets, the nominal logical pair-row ceiling is

$$
B_{\mathrm{pair,max}}=\texttt{data.batch\_size}\times D.
$$

For transcript (t), the grouped sampler treats

$$
G_t=\{(t,d):d\in\mathcal D_t^+\},\qquad |\mathcal D_t^+|\ge2
$$

as one atomic group. Packing enforces the per-dataset quota without splitting
a group. Therefore one logical batch has three relevant
quantities:

- pair rows: the number of `(transcript, dataset)` observations;
- unique transcripts: the number of biological sequences;
- datasets per transcript: the size of each complete group (G_t).

With two selected datasets and the active multidataset-only sampler, every
eligible group has both rows. Thus the active `data.batch_size=32` produces a
full logical batch with 32 rows from each dataset: 64 pair rows representing 32
unique transcripts. With partial support, no dataset exceeds 32, but atomic
greedy packing may leave some dataset quotas underfilled.

The optimizer target is a different quantity. The ideal pair-row window needed
to match a target of $$T_{\mathrm{target}}$$ fully supported biological
sequences is

$$
B_{\mathrm{pair,ideal}}
=T_{\mathrm{target}}D.
$$

Those rows need not be on the GPU simultaneously. The implementation measures
how many transcripts actually fit under the per-dataset quota and approximates
the target optimizer window with gradient accumulation.

## Non-mutating plan and automatic accumulation

Before Lightning constructs the `Trainer`, the entry point performs this
sequence:

1. construct the datamodule;
2. run the idempotent `setup("fit")` needed to create flat pair metadata;
3. preview the epoch-zero grouped sampler without advancing `_iter_count`;
4. measure actual unique transcripts in each packed physical microbatch;
5. select the configured statistic $$G_{\mathrm{micro}}$$, currently the median;
6. resolve

   $$
   A=
   \operatorname{clamp}\!\left(
   \left\lceil
   \frac{T_{\mathrm{local}}}
   {\max(G_{\mathrm{micro}},1)}
   \right\rceil,
   1,A_{\max}
   \right);
   $$

7. when execution microbatching is enabled, keep Lightning accumulation at one
   and pass (A) to the module's manual logical-batch accumulator; otherwise set
   `trainer.accumulate_grad_batches=A`;
8. construct `Trainer` and start training.

The preview uses the sampler's real selection, balancing, sorting, atomic
packing, `drop_last`, and DDP-compatible sharding logic. It counts real
transcript IDs and does not estimate groups by dividing pair rows by a nominal
dataset count.

Supported statistics are `median`, `mean`, `p25`, and `minimum`. The resolved
plan reports

$$
\widehat T_{\mathrm{update}}=A G_{\mathrm{micro}}
$$

and

$$
\widehat B_{\mathrm{pair,update}}
=A\operatorname{median}(B_{\mathrm{pair,micro}}).
$$

## Per-rank and global targets

With `target_scope: per_rank`, every DDP rank uses the configured target:

$$
T_{\mathrm{local}}=T_{\mathrm{configured}}.
$$

With `target_scope: global`, each rank uses

$$
T_{\mathrm{local}}
=\max\!\left(1,
\left\lceil\frac{T_{\mathrm{configured}}}{W}\right\rceil
\right),
$$

where (W) is the configured DDP world size. All ranks resolve the same
accumulation factor from the same deterministic global sampler preview. The run
manifest records both configured and effective local targets.

## Complete-group safety

An eligible row has a one-dimensional, non-empty, finite, non-negative consensus
profile, positive total reads and coverage, and a finite strictly positive local
weight. The flat table must contain exactly one row for each transcript-dataset pair;
duplicates are rejected. Replicas do not count as supporting datasets. Since one group contributes at most one row to each
dataset quota, any complete group fits whenever `data.batch_size >= 1`.

The same safety policy is used by training, validation, and prediction grouped
samplers. Validation and prediction do not use gradient accumulation.

## Relationship to gamma centering

Gradient accumulation does not combine gamma centers across different physical
microbatches. Gamma centering is evaluated independently inside each complete
same-transcript group during each forward pass. Different transcripts collected
in one optimizer accumulation window do not need to be centered together.

In deterministic `fixed_reference` gamma-centering mode, the model additionally
evaluates its checkpointed reference panel internally. Complete requested groups
are still preserved by the sampler, and their loss observations are never split
silently.

## Loss and optimizer semantics

With `loss.sample_reduction=transcript_balanced`, every complete transcript has
one equal outer vote inside its logical batch; local weights compare only the
datasets for that transcript. Execution chunks are multiplied by their
`chunk_group_count / logical_group_count` fraction, so summing all chunks
reconstructs the original transcript-balanced logical-batch loss exactly. If
the resolved accumulation factor is greater than one, each logical-batch mean
receives weight `1/A`; consequently all transcripts are exactly equally
weighted across the optimizer window only when those logical batches contain
the same number of groups. With the active 32-transcript quota, the usual
resolved factor is one.

No loss reduction, learning rate, or weight decay is rescaled. The configured
AdamW values remain:

- `lr_biological: 5.0e-4`;
- `lr_rest: 1.0e-3`;
- `weight_decay_bio: 1.0e-2`;
- `weight_decay_rest: 1.0e-2`.

The current `ReduceLROnPlateau` scheduler runs once per validation epoch. It is
therefore not stepped per raw microbatch. Lightning's
`trainer.estimated_stepping_batches` is logged after the resolved accumulation
factor has been installed.

## Configuration

```yaml
data:
  # Pair-row quota for each represented dataset on each GPU.
  batch_size: 32
  train_sampling_strategy: transcript_grouped_multidataset_pairs
  minimum_positive_datasets_per_transcript: 2

loss:
  sample_reduction: transcript_balanced

training:
  execution_microbatching:
    enabled: true
    gradient_clip_val: 1.0
    gradient_clip_algorithm: norm
    max_pair_rows_per_forward: 512
    max_padded_codon_tokens_per_forward: 256000
    max_transcript_groups_per_forward: null
  grouped_optimizer_batch:
    enabled: true
    target_unique_transcripts_per_optimizer_step: 32
    auto_accumulate_grad_batches: true
    accumulation_statistic: median
    max_accumulate_grad_batches: 32
    target_scope: per_rank
    log_batch_structure: true
```

An explicitly configured `trainer.accumulate_grad_batches` other than one is a
configuration conflict while auto accumulation is enabled. Set
`auto_accumulate_grad_batches: false` to retain an explicit Trainer value.

The lower accumulation bound is intrinsically one and is no longer exposed as
a redundant configuration field. The `resolved` subtree is runtime output, so
the input YAML no longer contains a `resolved: null` placeholder.

The normal entry point resolves and prints the grouped plan before constructing
the Trainer. For a two-dataset fully paired case, quota 32 gives 32 transcripts
per full logical batch. Target 32 therefore resolves accumulation one. The
execution limits can still divide those 64 pair rows into smaller GPU forwards
without changing the logical objective.

Actual planning inspects flat-pair metadata without materializing training
batches. Replica arrays do not change group construction.

## A100 64-GB execution profile

The real-data experiment launchers use this explicit profile by default:

```bash
data.batch_size=32
training.execution_microbatching.max_pair_rows_per_forward=512
training.execution_microbatching.max_padded_codon_tokens_per_forward=256000
training.grouped_optimizer_batch.target_unique_transcripts_per_optimizer_step=32
training.grouped_optimizer_batch.auto_accumulate_grad_batches=true
training.grouped_optimizer_batch.max_accumulate_grad_batches=32
model.gamma_centering.reference.chunk_size=16
```

The row/token/reference chunk sizes control peak execution memory and call
overhead, not subset membership or the gamma gauge. A single transcript group
remains atomic and may exceed a soft limit. If a large-N run exceeds memory,
lower the two execution-forward limits (for example to 256 and 128000) while
leaving `data.batch_size` and the optimizer target at 32. Execution
microbatching is single-process; run one independent model per GPU.

The same information is saved to `grouped_optimizer_batch_plan.json`, embedded
under `training.grouped_optimizer_batch.resolved` in the resolved Hydra config,
and summarized at runtime under the single `train_batch/*` TensorBoard namespace.
