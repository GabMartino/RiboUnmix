# Gamma centering

The implemented mean shape is

$$
\widetilde h_{dti}
=
\frac{L^{bio}_{ti}\gamma_{dti}}
{\operatorname{mean}_{j\in\mathcal V_t}
\left(L^{bio}_{tj}\gamma_{dtj}\right)},
\qquad
\mu_{dti}=S_{dt}\widetilde h_{dti}.
$$

Gamma is strictly positive:

$$
q_{dti}=f_\gamma(x_t,d)_i,
\qquad
\gamma_{dti}=\exp(\log\gamma_{dti})>0.
$$

Centering is a direct forward-value operation. It changes both forward values
and gradients. There is no detached center and no straight-through estimator.

## Fixed-reference mode

`fixed_reference` is recommended for new experiments. A fixed panel
$\mathcal D_{\mathrm{ref}}$ is resolved once from the datasets selected for the
experiment, unless an explicit duplicate-free selected subset is configured.
In this mode the dataset-embedding gauge is also computed only over the
checkpointed selected experiment IDs, so inactive entries in the global
vocabulary cannot influence raw gamma through the embedding mean.

$$
c^{ref}_{ti}
=
\frac{
\sum_{e\in\mathcal D_{\mathrm{ref}}}w_e q_{eti}
}{
\sum_{e\in\mathcal D_{\mathrm{ref}}}w_e
},
\qquad
\log\gamma_{dti}=q_{dti}-c^{ref}_{ti}.
$$

The same panel and formula are used during training, validation, prediction,
grouped inference, and singleton inference. The center does not depend on
requested partners, requested order, requested batch size, or DDP rank.
Reference target profiles are not needed.

The reference panel is evaluated in chunks. The implementation accumulates the
weighted sum without detaching it, so gradients flow through reference raw
scores into the dataset branch.

At valid positions:

$$
\sum_{e\in\mathcal D_{\mathrm{ref}}}\pi_e\log\gamma_{eti}=0,
\qquad
\prod_{e\in\mathcal D_{\mathrm{ref}}}\gamma_{eti}^{\pi_e}=1,
\qquad
\pi_e=\frac{w_e}{\sum_r w_r}.
$$

With fewer than `minimum_datasets`, centering is skipped and
$\log\gamma=q$.

## Batch-grouped mode

`batch_grouped` computes the center from the same-transcript observations in
the current physical batch:

$$
c^{batch}_{ti}
=
\frac{
\sum_{d\in\mathcal D^{batch}_{ti}}w_d q_{dti}
}{
\sum_{d\in\mathcal D^{batch}_{ti}}w_d
}.
$$

Duplicate rows are averaged within dataset before the cross-dataset average,
but only datasets present in the current transcript group participate. Thus a
singleton request is uncentered and partner membership can change the focal
prediction. This mode remains an explicit supported option for benchmarking
and for loading checkpoints trained with batch-local centering. A checkpoint
without fixed-panel provenance cannot be loaded under `fixed_reference`; select
`mode: batch_grouped` explicitly to reproduce its training-time function.

## Weights

`equal` gives one vote per distinct reference dataset. `quality_rank` uses:

$$
Q_d=\frac{R_{\max}-r_d+1}{R_{\max}},
\qquad
w_d=Q_d^p.
$$

Here $r_d$ is the external ordinal quality rank and $p$ is
`quality_rank_power`. Setting $p=0$ is numerically identical to equal
weighting. Replica count and requested-row duplication do not change a fixed
reference dataset's vote.

## Optional dataset-branch features

The current `exo`, `gmp`, `tmp`, `openen`, `tAI_profile_codon`, and `fra`
values are read from the shared sequence parquet by transcript. They are
sequence/transcript invariant in the active dataset implementation and can be
reused safely for every synthetic reference ID. Fixed-reference mode checks
that repeated rows of a transcript have identical codon IDs, masks, position
features, and routed optional features; inconsistent values raise an error.

## Configuration

```yaml
gamma_centering:
  mode: fixed_reference
  reference:
    # null defaults to every experiment dataset.
    dataset_names: null
    weighting: quality_rank
    quality_rank_power: 1.0
    chunk_size: 32
    minimum_datasets: 2
```

This centering is a deterministic gamma gauge. It does not establish that
gamma is purely technical or make the biological/technical decomposition
statistically identifiable by itself.
