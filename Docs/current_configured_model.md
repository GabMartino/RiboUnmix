# RiboUnmix: historical configuration snapshot

> This document predates the current runnable defaults. Its dataset selection,
> optional-feature routes, and dataset counts are not a current configuration
> contract. Use `python main_ribounmix_multidataset.py --cfg job --resolve` for
> today's values and a saved run's frozen YAML for past experiments. See the
> [repository alignment audit](repository_alignment.md) and [README](../README.md).

This document records an earlier snapshot of the model, objective, data path,
and training settings. References to “active” settings below describe that
snapshot. The runnable YAML and each saved run's resolved configuration remain
the numerical sources of truth.

The default YAML experiment uses `kutay_2021` and `grimson_2019`. The Slurm
launchers override only the dataset selection and execution resources:

- `run_all_datasets.slurm` trains one joint model on all 30 active datasets;
- `run_individual_datasets.slurm` trains one model for each active dataset;
- the architecture and loss below are otherwise unchanged.

## 1. Scope of the prediction

For dataset $$d$$, transcript $$t$$, and valid codon position $$i$$, let
$$y_{dti}\ge 0$$ be the representative Ribo-seq target. The model predicts the
profile shape conditional on the observed transcript/dataset mean:

$$
S_{dt}
=\frac{1}{T_t}\sum_{i\in\mathcal V_t}y_{dti},
$$

where $$\mathcal V_t$$ contains the $$T_t$$ non-padding codons. In code, the
scale is clamped below by $$10^{-8}$$.

The model therefore does **not** predict total footprint depth. It predicts how
the observed depth is distributed along the coding sequence. Comparisons of
profile PCC are consequently comparisons of shape, not library-size
prediction.

The central decomposition is

$$
\boxed{
\mu_{dti}=S_{dt}\,\widetilde h_{dti}
}
$$

with

$$
h_{dti}
=\gamma_{dti}L^{\mathrm{bio}}_{ti},
\qquad
\widetilde h_{dti}
=\frac{h_{dti}}
{T_t^{-1}\sum_{j\in\mathcal V_t}h_{dtj}}.
$$

Here:

- $$L^{\mathrm{bio}}$$ is the dataset-blind shared biological load;
- $$\gamma$$ is the dataset-specific multiplicative adaptation;
- the second equation is the enabled mass-conservation gauge.

Apart from numerical handling inside logarithms and likelihood evaluation,
mass conservation gives

$$
\operatorname{mean}_{i\in\mathcal V_t}\mu_{dti}=S_{dt},
\qquad
\sum_i\mu_{dti}=\sum_i y_{dti}.
$$

Suppressing zero positions therefore reallocates their mass to other positions,
rather than changing the predicted transcript total.

## 2. Input data and batching

### 2.1 Active data source

The Hydra default selects
`weighted_datasets_paths_replicas`. It contains 30 active Ribo-seq parquet
datasets with per-position biological replicas in `ribo_cds_replicas`.

Dataset IDs come from the global 33-entry dataset encoding. The active set has
sparse global IDs reaching 31, so the dataset embedding table has 33 rows even
though only 30 parquet datasets are active.

The sequence parquet also contains optional per-codon features. Their routing is
controlled by `model.additional_sequence_features`. Every entry has a declared
channel count and one of four routes:

| Route | Effect |
|---|---|
| `none` | The column is not loaded and the architecture is unchanged. |
| `biological` | The channels are appended to the 97 one-hot channels before the biological GRU. |
| `dataset_bias` | The channels are appended to the inputs of the independent dataset-bias GRU. |
| `both` | The same observed channels are supplied independently to both GRUs. |

The available configured columns are `exo` (3), `gmp` (3), `tmp` (3),
`openen` (3), `tAI_profile_codon` (1), and `fra` (3), where parentheses show
the per-codon channel count. For example:

```yaml
model:
  additional_sequence_features:
    openen: {dimension: 3, route: biological, scale: 1.0}
    tAI_profile_codon: {dimension: 1, route: both, scale: 1.0, missing_values: [-1000.0], fill_value: 0.0}
```

Only columns whose route is not `none` are read from disk. Each transcript is
checked to have shape $$[T_t,C]$$ (a scalar profile $$[T_t]$$ is promoted to
$$[T_t,1]$$). Values listed in `missing_values`, together with all non-finite
values, are replaced by `fill_value` before `scale` is applied.

In this parquet, tAI is in approximately $$[0.0995,1]$$ when defined, but every
terminal stop codon is encoded as $$-1000$$ and 17 internal stop positions are
NaN. The active tAI rule maps both cases to zero. The `fra` feature is disabled
because it equals `[1,0,0]` at every position and therefore has zero variance.
The active configuration routes `exo`, `gmp`, `tmp`, `openen`, and cleaned tAI
to both branches. This adds 13 channels to each branch; checkpoints trained with
the old input dimensions are not architecture-compatible with this setting.

Routing a feature to `both` does not connect the two branches. It produces

$$
L^{\mathrm{bio}} = f_{\mathrm{bio}}(x_{\mathrm{onehot}},x_{\mathrm{extra}}),
\qquad
(\gamma,a,\alpha) = f_{\mathrm{bias}}(d,x_{\mathrm{codon}},x_{\mathrm{extra}}),
$$

with separate GRU parameters and no path from $$L^{\mathrm{bio}}$$ into the
dataset-bias branch.

### 2.2 Transcript splits

There are only two roles: training and validation. Validation candidates are
the transcript intersection of the configured 114-dataset master universe.
Ten percent of those common candidates are sampled into one deterministic,
fixed validation panel. Reliability decile and CSS-count class jointly
stratify that sample, so CSS is a composition signal rather than a separate
fold or reserved percentage.

Every master-union transcript not selected for validation is a training ID.
Each experiment dataset contributes the subset of those training IDs that it
actually contains, so training sizes remain dataset-specific. Every selected
dataset contributes the same fixed validation IDs. The fixed panel is used for
validation loss, checkpoint selection, and early stopping.

### 2.3 Sampler

Training uses `transcript_grouped_multidataset_pairs`. All selected-dataset rows
for one transcript are kept in one atomic physical microbatch, and groups with
fewer than two represented datasets are excluded. The split still labels every
non-validation union ID as training; this sampler applies the additional
multi-dataset eligibility condition.

`data.batch_size: 32` is a per-dataset quota in the logical grouped batch. With
two selected datasets, every sampler-eligible group has both rows, so a full
logical batch contains 32 unique transcripts and 64 pair rows. With partially
overlapping larger panels, every represented dataset contributes at most 32
rows, although atomic greedy packing can underfill a quota. Execution
microbatching may materialize that logical batch in several smaller forwards.

Before constructing the Trainer, a non-mutating preview of the real group
packing resolves `accumulate_grad_batches` so that each DDP rank targets
approximately 32 unique transcripts per optimizer update. The active median
previewed transcript count is used and accumulation is capped at 32. Thus the
current fully paired case resolves `ceil(32 / 32) = 1`; the cap of 32 is not
itself the accumulation factor. The active execution-microbatching path is
single-process, so the real-data launchers run one independent model process
per physical GPU rather than DDP.

### 2.4 Representative target and replicas

Each batch carries both:

- an arithmetic replica-consensus target used by the main forward pass, its
  scale gauge, both PCC terms, and diagnostics;
- all valid raw biological replica profiles used by replica-level NB2.

Padding is masked throughout. The biological sequence is provided as a packed
sequence, while the dataset-bias GRU internally packs its codon/dataset input
using the same valid lengths.

## 3. Shared biological branch

The biological branch receives the 97 basic one-hot sequence channels plus 13
enabled optional channels, for 110 features at each codon. It never receives
the dataset ID.

The active architecture is:

```text
110 sequence features (97 one-hot + 13 optional)
  -> 2-layer bidirectional GRU, hidden size 256 per direction
  -> 512-dimensional contextual representation
  -> Linear(512, 512) -> GELU -> Dropout(0)
  -> Linear(512, 1) -> Softplus
```

The unnormalized positive output is

$$
w^{\mathrm{raw}}_{ti}
=\operatorname{softplus}(f_{\mathrm{bio}}(x_t)_i).
$$

It is normalized over valid positions:

$$
L^{\mathrm{bio}}_{ti}
=\frac{w^{\mathrm{raw}}_{ti}}
{T_t^{-1}\sum_{j\in\mathcal V_t}w^{\mathrm{raw}}_{tj}}.
$$

Thus

$$
L^{\mathrm{bio}}_{ti}\ge0,
\qquad
\operatorname{mean}_iL^{\mathrm{bio}}_{ti}=1.
$$

The final biological linear layer is initialized with weight standard deviation
$$10^{-2}$$ and a bias corresponding to an initial factor of 1. The output is
therefore close to, but not exactly, flat at initialization.

Queue-inspired derived quantities are

$$
\rho_{ti}=\frac{L^{\mathrm{bio}}_{ti}}
{1+L^{\mathrm{bio}}_{ti}},
\qquad
\lambda_{ti}=\log(1+L^{\mathrm{bio}}_{ti}),
\qquad
J_t=\frac1{T_t}\sum_i\lambda_{ti}.
$$

In this direct-load configuration, $$J$$ is only a diagnostic. There is no
learned flux head or dataset-specific $$J_d$$.

## 4. Dataset-bias branch

The dataset branch is separate from the biological branch. It receives dataset
identity, codon identity, the 13 enabled optional sequence channels, and
deterministic position features; it does not receive $$L^{\mathrm{bio}}$$.

### 4.1 Embeddings and contextual GRU

The inputs are:

- a 32-dimensional learned dataset embedding;
- a 16-dimensional learned embedding for each of 64 codons;
- 13 cleaned optional sequence channels.

Before lookup, the mean dataset embedding over the global 33-row vocabulary is
subtracted from every row. The selected dataset embedding is then repeated over
the transcript. Dataset, codon, and optional inputs are concatenated into 61
features and passed through:

```text
61 features (32 dataset + 16 codon + 13 optional)
  -> 2-layer bidirectional GRU
  -> hidden size 128 per direction
  -> dropout 0.0 between GRU layers
  -> 256 contextual features per codon
  -> channel-wise LayerNorm
```

This GRU is independent of the biological GRU. Consequently,

```text
L_bio <- GRU_bio(one-hot sequence, optional sequence features)
gamma <- GRU_bias(codon identity, dataset identity, optional sequence features)
```

remain separate attribution paths.

### 4.2 Position features

Three deterministic features are appended:

$$
r_i=\frac{i}{\max(T_t-1,1)},
\qquad
s_i=\exp(-i/100),
\qquad
e_i=\exp(-(T_t-1-i)/100).
$$

They represent relative position, proximity to the start, and proximity to the
stop. The complete head input has

$$
32+256+3=291
$$

features per position.

### 4.3 Shared bias representation and outputs

The 291-dimensional input is processed by

```text
Linear(291, 128) -> GELU -> Dropout(0.1)
Linear(128, 128) -> GELU -> Dropout(0.1)
```

One linear output produces the unconstrained log-gamma residual $$q_{dti}$$.
Its weights and bias are initialized to zero, so gamma starts neutral at one.
A separate MLP using the same assembled contextual input predicts the NB2
log-dispersion.

## 5. Fixed-reference gamma centering

Gamma centering is a cross-dataset identifiability constraint applied directly
to the log-score whose exponential is the final gamma. The biological branch
is not included in the center itself.

The initial gamma value is one, so the uncentered log-amplitude is

$$
\ell^{\mathrm{raw}}_{dti}
=q_{dti}+\log(\gamma_{\mathrm{init}}).
$$

In the current configuration $$\gamma_{\mathrm{init}}=1$$, hence
$$\log(\gamma_{\mathrm{init}})=0$$. The raw log-score is not clamped.

Centering is not estimated from the physical batch. The configured
`fixed_reference` path internally evaluates a checkpointed reference panel for
each requested transcript. The center is therefore independent of requested
partner rows, physical batch size, row order, DDP partitioning, and the grouped
sampler. `batch_grouped` remains an explicit historical alternative, but it is
not the configured mechanism described below.

### 5.1 Dataset weights

The reference panel is assigned dataset-level constants loaded from
`Datasets/data/HEK_riboseq_profile_quality_rank.tsv`. Rank 1 is best. If $$R$$
is the largest rank in the complete table, the conversion is

$$
q_d=\frac{R-r_d+1}{R}.
$$

The configured centering mode is `quality_rank`, so the raw centering weight is

$$
w_d=q_d^p,
$$

with `quality_rank_power` $$p=1$$. The alternative `equal` mode sets
$$w_d=1$$ for every distinct dataset. Setting $$p=0$$ also recovers equal
weighting. These weights affect only gamma centering: they do not reweight the
NB loss, PCC loss, or reported metrics. The transcript-level `sample_weight` is
a separate quantity.

The rank and weight contain no ground-truth value from the current transcript.
Target depth, local target coverage, observed zeros, peak height, and the number
of replicas therefore do not determine the center. Padding, missing transcript
identifiers, and invalid dataset identifiers remain ineligible.

### 5.2 Fixed-reference center

By default the reference panel $$\mathcal R$$ is every dataset selected for the
experiment. `reference.dataset_names` may instead define an explicit subset.
For one canonical copy of transcript $$t$$, the model evaluates the dataset
branch internally for every $$d\in\mathcal R$$ and obtains
$$\ell^{\mathrm{ref}}_{dti}$$. The center is

$$
c_{ti}
=\frac{\sum_{d\in\mathcal R}w_d\ell^{\mathrm{ref}}_{dti}}
{\sum_{d\in\mathcal R}w_d}.
$$

The center is applied when the stored reference panel contains at least
`reference.minimum_datasets` datasets, currently two. The requested centered
log-amplitude is

$$
\ell_{dti}=\ell^{\mathrm{raw}}_{dti}-c_{ti}.
$$

The final correction is $$\gamma=\exp(\ell)$$, so this makes its weighted
geometric mean one wherever centering is applied:

$$
\exp\left(
\frac{\sum_{d\in\mathcal R}w_d\ell^{\mathrm{ref,centered}}_{dti}}
{\sum_{d\in\mathcal R}w_d}
\right)=1.
$$

There is no later gamma clamp, support gate, or other nonlinear transform after
centering apart from the exponential. The zero-mean log constraint therefore
applies directly to the final gamma. A profile shape shared by all datasets
cannot remain in gamma's common multiplicative mode and is encouraged to return
to the dataset-blind biological branch.

In equal mode, if two datasets produce amplitude values 2 and 8 at the same
transcript position, the log center is $$\log 4$$ and the centered amplitudes
become 0.5 and 2. Their geometric mean is one. In quality-ranked mode, the
higher-quality dataset contributes more strongly to the reference center, while
the relative difference between the two raw log-gammas is still retained.

An inference batch containing only one requested dataset is still centered when
the loaded checkpoint contains at least two reference datasets: the other
reference versions are evaluated internally and do not have to be batch rows.
A model genuinely trained with a one-dataset reference panel instead skips the
center because its stored panel is below the configured minimum. The diagnostic
`gamma_all_requested_in_reference` additionally identifies inference datasets
outside the checkpoint's reference panel; the reference center is still
applied, but that case is an out-of-reference extrapolation.

The configured weights contain no target information and receive no gradients. The
log-amplitudes remain differentiable: centering couples requested predictions
to the internally evaluated reference scores for the same transcript.

Finally, the gamma regularizer and centering have different roles. Centering
removes the cross-dataset common mode; the regularizer
$$\operatorname{mean}(\ell^2)$$ shrinks the remaining dataset-specific
amplitude departures toward one. Centering does not regularize the biological
branch.

### 5.3 Effect on training and gamma interpretation

Centering is inside the forward pass and therefore changes the gradients used
to train the gamma branch. In schematic equal-reference weighting with
$$D=|\mathcal R|$$,

$$
\ell_d=q_d-\frac{1}{D}\sum_{j=1}^{D}q_j.
$$

If $$g_d=\partial\mathcal L/\partial\ell_d$$, then

$$
\frac{\partial\mathcal L}{\partial q_d}
=g_d-\frac{1}{D}\sum_jg_j.
$$

The quality-ranked implementation uses the corresponding weighted derivative.
Thus the reference-panel common gradient component is removed. Gamma learns relative
dataset-specific effects, while patterns shared across datasets are pushed
toward the dataset-blind $$L_{\mathrm{bio}}$$ branch. The center is not a
detached diagnostic; only the configured reference weights are non-learned
constants.

The final gamma remains

$$
\gamma_{dti}=\exp(\ell_{dti})>0.
$$

NB likelihood and PCC train the final prediction through gamma. The configured
gamma regularizer is disabled:

$$
\mathcal L_{\gamma}=0\cdot\operatorname{mean}(\ell^2).
$$

Fixed-reference gamma centering is distinct from loss weighting. The explicit
`loss.sample_reduction` selector defaults to `transcript_balanced`: local
reliability weights define a weighted mean across the dataset observations of
one transcript, then transcript means receive equal outer weight.
`dataset_balanced` and `global_weighted` remain available ablations. Gamma
itself uses sequence, position, and dataset ID; the overall scale $$S_{dt}$$
remains target-derived separately from gamma.

## 6. Final gamma and prediction shape

The final multiplicative correction is

$$
\boxed{\gamma_{dti}=\exp(\ell_{dti})>0}.
$$

There is no support gate or additive mean branch. Gamma can approach zero for
large negative log-scores and can form high peaks for large positive scores,
but it cannot be exactly zero. The unnormalized prediction shape is

$$
h_{dti}=\gamma_{dti}L^{\mathrm{bio}}_{ti}.
$$

Mass normalization makes this shape mean one before multiplication by the
target-derived scale $$S_{dt}$$. Valid predicted means are finally constrained
to the likelihood's numerical range.

## 7. Negative-Binomial dispersion

A separate MLP receives the same 291 contextual features:

```text
Linear(291, 128) -> GELU -> Dropout(0.1) -> Linear(128, 1)
```

It produces the historical output named `log_sigma`, which is mathematically
the NB2 log-dispersion $$\log\alpha$$. Its last layer starts at
$$\log\alpha=0.2$$. Dispersion is always learned at every position; the model
does not provide a fixed-dispersion mode.

For the main likelihood,

$$
\log\alpha_{dti}\in[-5,1],
\qquad
\alpha_{dti}=\exp(\log\alpha_{dti}),
$$

so approximately $$0.0067\le\alpha\le2.718$$. The NB2 variance is

$$
\operatorname{Var}(Y_{dti}\mid\mu_{dti},\alpha_{dti})
=\mu_{dti}+\alpha_{dti}\mu_{dti}^2.
$$

## 8. Main Negative-Binomial likelihood

Let $$r=1/\alpha$$. The per-position NLL is

$$
\begin{aligned}
\mathcal L^{\mathrm{NB}}_{dti}
={}&\log\Gamma(r)-\log\Gamma(y_{dti}+r)
+\log\Gamma(y_{dti}+1)\\
&-r\log r-y_{dti}\log\mu_{dti}
+(r+y_{dti})\log(r+\mu_{dti}).
\end{aligned}
$$

The target-only $$\log\Gamma(y+1)$$ normalization is retained, making logged
NLL a proper NB log-density even though that term has no parameter gradient.

The active sequence reduction is length-tempered. For valid length $$T_t$$,

$$
w_T
=\operatorname{clamp}
\left[
\left(\frac{T_t}{1000}\right)^{1-0.75},
0.5,
2.0
\right],
$$

and

$$
\mathcal L^{\mathrm{NB}}_{dt}
=w_T\frac1{T_t}\sum_i\mathcal L^{\mathrm{NB}}_{dti}.
$$

## 9. Hybrid PCC loss

Flat-target sequences with variance below $$10^{-6}$$ are excluded from the
correlation loss. The PCC prediction floor is zero, so no threshold is applied
to $$\mu$$.

The raw component is

$$
\mathcal L^{\mathrm{raw}}_{\mathrm{PCC}}
=1-\operatorname{PCC}(\mu,y).
$$

The NB variance-stabilizing transform is

$$
V_\alpha(x)
=\frac{2}{\sqrt\alpha}
\operatorname{asinh}(\sqrt{\alpha x}),
$$

and the transformed component is

$$
\mathcal L^{\mathrm{VST}}_{\mathrm{PCC}}
=1-\operatorname{PCC}
\left(V_\alpha(\mu),V_\alpha(y)\right).
$$

Alpha is detached in this transform, so the PCC term does not train the
dispersion head. The transform uses the same bounded log-dispersion as the NB2
likelihood, $$\log\alpha\in[-5,1]$$.

The two PCC losses are independently weighted, with no mode or enable selector:

$$
0.5\mathcal L^{\mathrm{cons},\mathrm{raw}}_{\mathrm{PCC}}
+0.5\mathcal L^{\mathrm{cons},\mathrm{VST}}_{\mathrm{PCC}}.
$$

Both are evaluated once per transcript-dataset pair against the arithmetic
replica consensus. A component whose coefficient is zero is skipped. The raw
replicas still supervise NB2 individually.

## 10. Zero handling

There is no support classifier, zero/nonzero BCE, or entmax gate in the active
model. Zeros influence training through the NB likelihood and through the PCC
objective. The optional `pcc_prediction_floor` is currently zero, so it does not
threshold predictions for PCC.

## 11. Regularization

The centered log-amplitude regularizer is

$$
\mathcal R_\gamma
=\frac1{T_t}\sum_i\ell_{dti}^2,
\qquad
w_\gamma=0.
$$

Its configured weight is zero, so the calculation is skipped and it does not
affect optimization.

Reference-anchor and separate zero-calibration objectives are not present in
the final implementation.

## 12. Replica targets

There is no replica-loading switch, replica-objective preset, or PCC mode.
Every sample carries replicas. NB2 is computed per replica and averaged across
valid replicas. Raw PCC and NB-VST PCC are computed once against the arithmetic
replica consensus, so additional replicas improve the consensus estimate but do
not add extra PCC terms.

For each valid replica $$r$$, the code computes its own observed mean

$$
S^{(r)}_{dt}
=\frac1{T_t}\sum_i y^{(r)}_{dti}
$$

and evaluates NB NLL against a prediction with this replica-specific scale.
Replica NB losses are averaged within the pair before batch aggregation.

The representative forward path and every replica NB loss use the same final
mass-normalized shape

$$
\widetilde h
=\operatorname{meanNormalize}(\gamma L^{\mathrm{bio}}).
$$

and each replica prediction is

$$
\mu^{(r)}_{dti}=S^{(r)}_{dt}\widetilde h_{dti}.
$$

The replica-specific scale preserves that replica's observed mean depth while
gamma centering and mass normalization remain identical to the consensus
forward path.

## 13. Complete optimized objective

For a representative sample, the active per-sample objective is

$$
\boxed{
\mathcal L_{dt}
=\mathcal L^{\mathrm{NB}}_{dt,\mathrm{rep}}
+\mathcal L^{\mathrm{PCC}}_{dt,\mathrm{cons}}.
}
$$

There is no loss-warmup subsystem. Parquet transcript weights are applied during
aggregation. The default `transcript_balanced` reducer forms a reliability-
weighted mean across the dataset observations of each transcript and then gives
every transcript one equal outer vote. `dataset_balanced` and `global_weighted`
remain explicit ablation modes. Dataset-quality rank is not part of any loss
reducer.

## 14. Gradient paths

| Term | Biological GRU | Dataset GRU/shared trunk | Gamma head | Dispersion head |
|---|---:|---:|---:|---:|
| Hybrid PCC | yes | yes | yes | no, alpha detached |
| Replica NB | yes | yes | yes | yes |
| Gamma regularizer | no direct path | yes | yes | no |

## 15. Optimizer and training control

The optimizer is AdamW with two parameter groups:

| Parameters | Learning rate | Weight decay |
|---|---:|---:|
| Biological branch | $$5\times10^{-4}$$ | $$10^{-2}$$ |
| Dataset branch, heads, dispersion | $$10^{-3}$$ | $$10^{-2}$$ |

A `ReduceLROnPlateau` scheduler monitors `val_loss` in minimum mode, with factor
0.99, patience 5, and minimum learning rate $$10^{-6}$$. Early stopping also
monitors `val_loss`, with patience 10. A second checkpoint independently keeps
the maximum `val_mu_pcc` model, and prediction prefers that PCC checkpoint.

Other active trainer settings are:

- maximum 200 epochs;
- BF16 mixed precision;
- gradient-norm clipping at 1.0;
- no gradient accumulation;
- log every training step;
- no CAGrad or gradient-conflict subsystem is present in the final model.

The all-dataset Slurm launcher uses two DDP ranks, custom rank-sharded grouped
batch samplers, and synchronized distributed metrics. Each individual model is
single-GPU, so distributed metric synchronization is unnecessary.

## 16. Active diagnostics and outputs

The former full-profile residual accumulator and its CSV/JSON export path have
been removed. They were disabled in normal runs, were not part of the
objective or checkpoint selection, and retained large validation tensors solely
for an optional diagnostic report.

Routine scalar logging covers the optimization losses, PCC, gamma and centering
diagnostics, dispersion, and replica diagnostics. Per-dataset validation
logging includes the configured losses and explicitly unweighted mu-PCC metrics.
All names containing `mu_pcc` are ordinary, unweighted arithmetic means. Once
per validation epoch, the already-computed pair PCC values are averaged equally
over the datasets available for each transcript; TensorBoard logs their compact
histogram and count/mean/std/min/quantile/max summary. No second forward pass is
required, and only one CPU float per validation transcript is retained temporarily.

Validation profile plots include:

- target and predicted mean;
- $$L^{\mathrm{bio}}$$ and $$\rho$$;
- raw and centered gamma diagnostics;
- log-dispersion.

Prediction parquet output includes the valid-length sequences for these same
terms, plus gamma-centering reliability, eligibility, application masks, and
dataset-count diagnostics.

## 17. What is and is not identifiable

Interpretability is encouraged by the following gauges:

- $$L^{\mathrm{bio}}$$ is dataset-blind and mean-one;
- observed scale is isolated in $$S$$;
- same-transcript gamma is geometrically centered across distinct datasets;
- the representative predicted profile conserves target mass.

These constraints do not prove a unique biological/technical decomposition.
Sequence-correlated experimental artifacts can enter the shared branch, and
reproducible dataset-specific biology can enter gamma. Centering is absent for
single-dataset transcripts and positions not shared by at least two datasets.
The learned position-wise dispersion can also explain local
uncertainty without changing the predicted mean.

## 18. Configuration summary

| Component | Active value |
|---|---|
| Biological encoder | 2-layer BiGRU, 256 per direction |
| Dataset encoder | independent 2-layer BiGRU, 128 per direction, dropout 0.0 |
| Dataset vocabulary | 33 global IDs; 30 active datasets |
| Gamma | exponential of the equal-dataset centered, unclamped log-score |
| Mass conservation | enabled |
| Likelihood | position-wise NB2 |
| NB log-alpha range | [-5, 1] |
| NLL reduction | length-tempered, exponent 0.25 around length 1000 |
| Consensus raw PCC | weight 0.5 |
| Consensus NB-VST PCC | weight 0.5 |
| Fixed data loss | replica NB 1 + consensus raw PCC 0.5 + consensus NB-VST PCC 0.5 |
| Gamma regularization | disabled (weight 0) |
| Sample reduction | transcript-balanced by default; dataset-balanced and global-weighted available as ablations |
| Zero-calibration auxiliary | disabled |
| Gradient surgery | not present |

## 19. Implementation map

- `Models/RiboUnmixModel/SharedProfileModel.py`: shared biological load.
- `Models/RiboUnmixModel/DatasetBiasSubmodel.py`: embeddings, position
  features input, dataset-bias GRU, and dispersion input.
- `Models/RiboUnmixModel/submodels/DatasetMultiplicativeAllocationBiasHead.py`:
  dataset-conditioned log-gamma residual.
- `Models/RiboUnmixModel/submodels/DatasetLogSigmaHead.py`: position-wise NB2
  dispersion.
- `Models/RiboUnmixModel/RiboUnmixModel.py`: gamma centering, mass
  conservation, and representative prediction.
- `Models/RiboUnmixLightningModule.py`: NB/PCC/replica objective,
  aggregation, diagnostics, and optimization.
- `Dataloaders/RiboUnmixMultiDataset/`: grouped sampling, replica batches,
  masking, and DDP batch sharding.
- `config/config_ribounmix_multidataset.yaml`: active numerical settings.
