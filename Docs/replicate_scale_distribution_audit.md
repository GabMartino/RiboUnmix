# Replicates, observed scales, and the NB distribution

Implementation audit, 10 September 2026. Production code and experiment settings
were not changed. The main finding is a mismatch with the interpretation
"all raw replicas of a dataset/transcript/codon share one mean and variance":
the implementation shares shape and dispersion, but gives each replicate its
own observation-derived mean scale.

For the scientific justification, its assumptions, and alternative models,
see [replicate scaling rationale and alternatives](replicate_scaling_rationale_and_alternatives.md).

## The actual path through the implementation

1. `Datasets/data/build_hek_riboseq_replicas.py:77` excludes `merged_mean`, retains
   original sample profiles, and computes the arithmetic consensus from them.
   `replica_ids` preserve source sample provenance. The weighting script preserves
   the profiles; it adds reliability columns and filters ineligible pairs.
2. `Dataloaders/RiboUnmixMultiDataset/RiboUnmixMultiDatasetDataModule.py:1532`
   validates and stacks raw replicas as [R,T], recomputes their arithmetic mean
   as the representative target, and retains all replicas for NB. The collator
   creates [B,R,T] counts, a [B,R] replica mask, and a [B,T] position mask.
   Padded replicas do not enter the loss. Replica sample IDs/exposure factors
   are not inputs to the model branches or likelihood.
3. `Models/RiboUnmixModel/RiboUnmixModel.py:1305` evaluates the dataset branch
   once per transcript/dataset, using sequence, position and dataset identity.
   There is no replicate-specific gamma or alpha. The biological branch's L is
   dataset-blind and mean-one. Gamma is defined by the configured pi reference.
4. `Models/RiboUnmixModel/RiboUnmixModel.py:1383` computes S from the consensus
   target and returns the representative prediction mu = S * h, where h is the
   final dataset-adapted shape. The current base configuration normalizes h to
   mean one (`mass_conservation: true`); the two saved Exp8 designs do not.
5. **`Models/RiboUnmixLightningModule.py:1579` computes a new S for every raw
   replicate. Line 1599 constructs its own mu. Lines 1608-1613 broadcast the same
   log alpha across all replicas.** Thus the NB loss does not broadcast the
   representative output mu to the raw replicas.
6. The ordinary NB2 loss at line 515 uses these replicate-specific means and
   common alpha. Replicate losses are averaged over valid replicas (line 1655),
   after the per-position sequence reduction. The real configurations then
   apply the transcript-balanced reliability reduction.
7. The raw and NB-VST PCC terms compare the representative mu to the arithmetic
   consensus once per transcript/dataset (`_compute_consensus_loss_terms`). They
   do not compare every replicate independently. Alpha is detached from PCC by
   the current configuration. The alpha head receives NB gradients from all
   replicas, but its input context is detached from the shared dataset encoder
   (`DatasetBiasSubmodel.py:345`). This does not freeze the alpha head itself.

## Mathematical meaning

For dataset d, transcript t, replicate r, and codon i, suppress d,t for clarity:

\[
\bar y_i=\frac1R\sum_r y_{ri},\qquad
S=\frac1T\sum_i\bar y_i,\qquad
S_r=\frac1T\sum_i y_{ri}.
\]

Ignoring numerical floors, S is the average of the replicate scales. Both the
representative and replica paths reuse the same final shape:

\[
h_i=\begin{cases}
L_i\gamma_i,&\text{Exp8, mass conservation disabled},\\
L_i\gamma_i/\operatorname{mean}_j(L_j\gamma_j),&\text{mass conservation enabled}.
\end{cases}
\]

The exported representative mean is mu_i = S h_i. The actual NB evaluations are

\[
\mu_{ri}=S_r h_i,\qquad
\alpha_i=\exp(\operatorname{clip}(\texttt{log\_sigma}_i,-5,1)),\qquad
V_{ri}=\mu_{ri}+\alpha_i\mu_{ri}^2.
\]

The historical name `log_sigma` means log NB2 dispersion, not log standard
deviation. Sharing alpha does not share variance when mu differs. All replicas
use one NB family and common shape/dispersion functions, with different mean
parameters. They are not identically distributed raw counts.

The representative mu equals the average of the replica mu values under these
equations. That equality does **not** make it the mean used for every replica.
With mass conservation disabled, even the predicted mean across codons is
S_r * mean(h), not necessarily S_r. Neither changing mass conservation nor
changing pi removes the replicate-specific scale operation.

## Reproduced evidence

The standalone [diagnostic](replicate_scale_distribution_audit.py) calls the
actual `_compute_replica_loss_terms` and captures its inputs to the production
NB loss. It explicitly configures `standard_nb`, beta=0, and uses synthetic
profiles [10,10,10,10] and [30,30,30,30], h=[1,1,1,1], alpha=0.2.

| Quantity at each codon | Current implementation | One common-mean alternative |
|---|---|---|
| Scales used by NB | 10 and 30 | 20 for both |
| Means | 10 and 30 | 20 and 20 |
| Variances at alpha=0.2 | 30 and 210 | 100 and 100 |

Fitting only alpha to this deliberately amplitude-only example gives about
0.00674 (the configured lower bound) with the current replicate scales, versus
about 0.224 with the common mean. The depth discrepancy is absorbed by S_r in
the current model. These are diagnostic counts, not a real-data estimate of
dispersion bias. Actual alpha gradients were finite and nonzero.

The first transcript in three inspected active weighted files,
`ENST00000000233.10` (181 codons), has these actual replicate means:

| Dataset | Raw replica means (counts/codon) |
|---|---|
| martinez_2019 | 8.9669, 2.6354 |
| sharma_2021 | 10.9503, 10.6464, 11.3039 |
| patel_2020 | 15.7514, 14.0718 |

These sampled profiles contain integer counts. This demonstrates that unequal
S_r occurs in the real inputs; it is not a whole-dataset calibration analysis.

Twenty existing weight/loss/NB-gradient tests passed. The existing replica loss
test uses replicas with identical totals, so it cannot distinguish a common-S
implementation from the current separate-S implementation. The new unequal-total
diagnostic makes that distinction explicit.

Reproduce it with:

```bash
MPLCONFIGDIR=/tmp/replica-audit-mpl .venv/bin/python Docs/replicate_scale_distribution_audit.py
```

## Statistical consequences and the intended model

There is no rule that raw sequencing replicates must have identical means:
different sequencing exposures can change means even when the underlying rate
and dispersion are shared. A standard count-model construction is
mu_ri = s_r q_i with sample-level normalization factors. For example, this
separation of normalization factors and expected expression is explicit in the
[DESeq2 methods paper](https://doi.org/10.1186/s13059-014-0550-8).

The stronger assumption here is that the observed mean of **each individual
transcript in each replicate** is treated as its fixed scale. This can absorb
library depth differences, biological transcript-abundance differences, and
sampling fluctuation in transcript totals together. The NB dispersion describes
residual count variation around these fitted scales; it cannot be interpreted
as a calibrated measure of all raw between-replicate variability. Shared
neural parameters and few replicates also mean alpha is learned across many
transcripts/positions; it is not just the sample variance at each codon.

Using the observations to estimate nuisance parameters is not inherently
invalid. However, this implementation plugs S_r(y_r) into a product of ordinary
NB probabilities and does not model uncertainty in S_r or derive a conditional
distribution given the transcript total. It should be described as an NB-based
profile fitting objective with plug-in scales. It is not automatically an exact
likelihood for a new replicate's complete count profile, nor the exact
conditional likelihood given its total. The additional PCC penalties and
per-transcript/per-replica averages further distinguish the total training
objective from an unweighted joint count likelihood.

An all-zero replica inside a positive-consensus pair is an extreme example:
its own S_r is floored near zero, making a near-zero mean largely automatic.
It therefore provides little evidence for dispersion of transcript totals,
although the replica is present in the loss.

Nor is mu + alpha*mu^2, computed from the representative exported mu, the
variance of the arithmetic consensus. Under a hypothetical conditionally
independent NB model with fixed scales/parameters, that variance would be

\[
\operatorname{Var}(\bar Y_i\mid S_1,\ldots,S_R)
=\frac{1}{R^2}\sum_r\left(\mu_{ri}+\alpha_i\mu_{ri}^2\right).
\]

For identical means it reduces to (mu_i+alpha_i*mu_i^2)/R. Parameter uncertainty,
estimated scales, and dependence between replicas require additional treatment.
The code does not implement this as a calibrated consensus uncertainty model.

If the intended experiment is **replicates as repeated observations of one
underlying distribution at a common exposure**, the NB term should share mu_i
and alpha_i across those replicas. A common S estimated from all replicas would
move toward that interpretation; alpha would then see discrepancies that the
current replicate-specific S absorbs. But imposing the same count mean on raw
replicas with different sequencing depths would confound depth with biological
variation.

For raw libraries, a more explicit formulation is

\[
Y_{dtri}\sim\operatorname{NB2}(s_{dr}\,A_{dt}\,h_{dti},\alpha_{dti}),
\]

where s_dr is a library/replicate exposure estimated across many transcripts,
and A_dt is a transcript/dataset abundance shared across replicas. If h is not
mean-one, A is a multiplicative amplitude, not literally the mean abundance.
A joint amplitude/gamma gauge would need to be specified. Biological variation
in abundance can be represented through dispersion or an explicit replicate
random effect, depending on the desired covariance structure. Neither should
be silently replaced by each replicate's realized transcript mean.

If the objective is strictly the allocation of a known transcript total across
codons, an explicit conditional count-vector model is another route. For example,
Poisson counts conditioned on their total give a multinomial shape likelihood;
an overdispersed alternative can be specified separately. Conditioning this
position-specific-alpha NB2 model does not automatically yield the same
likelihood or dispersion interpretation.

The choice is therefore scientific: the current code is consistent with its
documented depth-adjusted profile objective, but it does not implement the
shared raw-replicate distribution described in the question. A change to S
would change both what alpha learns and the training target for L/gamma, and
would require a separately identified experiment rather than relabeling old
checkpoints.
