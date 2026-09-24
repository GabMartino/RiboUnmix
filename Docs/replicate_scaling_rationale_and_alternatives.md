# Why use replicate-specific scales? Rationale and alternatives

10 September 2026. This note explains a defensible statistical rationale for
the implemented choice; it does not establish that this was the historical
reason for choosing it, or that it outperforms the alternatives. Implementation
details and a numerical reproduction are in the
[replicate distribution audit](replicate_scale_distribution_audit.md).

**The justification is that the intended target is relative ribosome-profile
shape, while overall transcript depth is treated as a nuisance quantity.**
Replicates can share that shape without sharing their expected raw counts.
The consequential modeling choice is to estimate a separate depth from every
replicate's own transcript counts. This removes some variation that a model
with one common mean would instead ask dispersion to explain.

## What the two assumptions mean

Fix a dataset and transcript, and let r index replicates and i the T valid
codons. Write y_ri for observed counts, h_i for the common dataset-adapted
profile derived from L_i gamma_i, and alpha_i for NB2 dispersion.

The **identical-distribution assumption**, at a fixed codon, is

\[
Y_{ri}\overset{\mathrm{iid}}{\sim}\operatorname{NB2}(\mu_i,\alpha_i),
\qquad \operatorname{Var}(Y_{ri})=\mu_i+\alpha_i\mu_i^2.
\]

Here every replicate has the same expected count, and its departures from
that mean contribute to estimating dispersion. Observed replicate totals can
still differ by chance. Different observed totals alone are therefore **not
proof** that this assumption is wrong.

The **implemented assumption** uses shared shape and dispersion with separate
estimated scales:

\[
\widehat S_r=\frac1T\sum_i y_{ri},\qquad
\widehat\mu_{ri}=\widehat S_r h_i,\qquad
\widehat V_{ri}=\widehat\mu_{ri}+\alpha_i\widehat\mu_{ri}^{,2}.
\]

Ordinary NB2 losses are evaluated at these means. The hat emphasizes that the
scale is estimated from the observations being fitted. The actual
[replica loss](../Models/RiboUnmixLightningModule.py) computes `scale_rep`
and `mu_rep`, then broadcasts `log_sigma` across replicas. Replicates share
alpha, not numerical variance. No replicate identifier enters the shape or
dispersion network.

The representative exported mu uses the arithmetic consensus's scale,
S = mean_r(S_r). It is also used for the consensus PCC terms; the NB term
uses the separate replicate means above. Raw replicas remain the NB targets,
and their losses are averaged over valid replicas before the batch reduction.

There is a normalization qualification. The current base configuration sets
h = L gamma / mean(L gamma), so mean(h)=1. The saved Exp8 experiments use
h = L gamma without that final normalization. For Exp8, the predicted mean
across codons is S_r mean(h), and S_r is a multiplicative anchor rather than
an exactly conserved predicted mean. This does not change the fact that NB
receives a separate scale for each replicate.

## Why separate scales can be appropriate for this project

Sequencing counts reflect both a relative pattern and how much material was
observed. Sample depth and composition can change expected counts without a
corresponding change in the pattern of interest. This is why RNA-seq count
models use normalization factors; DESeq2, for example, separates its mean
into a normalization factor and an expression term. These methods support
the general exposure/rate distinction, not the specific use of a separate
observed transcript mean adopted here.
[Love, Huber and Anders, 2014](https://doi.org/10.1186/s13059-014-0550-8);
[Robinson and Oshlack, 2010](https://doi.org/10.1186/gb-2010-11-3-r25).

For a model whose target is within-transcript relative ribosome occupancy,
the corresponding working assumptions are:

- Global amplitude differences between replicate profiles are outside the
  target of inference, whether they arise from sequencing depth or transcript
  abundance. That is a choice of scientific target, not a claim that all
  amplitude differences are technical artifacts.
- After allowing for amplitude, replicates of a dataset share a positional
  mean shape. Replicate-specific positional distortions are left to the
  residual model; a scalar cannot remove them.
- Raw-count uncertainty should still depend on depth. A low-depth profile
  and a high-depth profile need not provide equally precise shape evidence.

Under these assumptions, allowing different scales avoids forcing a common
count mean to compromise between libraries with different expected depth.
It also avoids asking alpha to explain an overall amplitude mismatch that
the analysis deliberately treats as irrelevant to shape.

For illustration, suppose the true exposures are fixed at S_1=10 and S_2=30,
and h_i=1. Their expected counts are 10 and 30. A model constrained to a
common mean of 20 fits neither exposure correctly. In contrast, if both true
scales were 20 and the observed 10/30 discrepancy were sampling fluctuation,
assigning fitted scales 10/30 would absorb variation belonging to the common
distribution. **The rationale depends on which of these data-generating
assumptions is appropriate.**

The distinction can be expressed through the law of total variance. In a
hypothetical model with a random true scale S, fixed h and alpha, and
Y conditional on S following NB2(Sh,alpha), let m=h E[S]. Then

\[
\operatorname{Var}(Y)
=m+\alpha m^2+(1+\alpha)h^2\operatorname{Var}(S).
\]

The final term represents additional variation from changing scale. Ignoring
it can make dispersion absorb exposure heterogeneity. This identity does not
imply that the marginal mixture is itself NB2, or prove that estimating S
from each observed profile is optimal.

Keeping the raw counts in the likelihood is useful. Dividing them by a fixed
known scale does not make their distributions identical:

\[
\mathbb E[Y_{ri}/S_r]=h_i,\qquad
\operatorname{Var}(Y_{ri}/S_r)=h_i/S_r+\alpha_i h_i^2.
\]

The normalized means match, but the sampling variances still depend on depth.
These formulas assume known fixed S_r; they cannot be applied unchanged to
the random ratio Y_ri / S_hat_r. Treating normalized fractional counts as
ordinary identically distributed NB counts would lose this distinction.

## Why the observed transcript mean is a reasonable starting point, and its limits

There is an exact justification in a simpler model. Suppose codons are
conditionally independent Poisson counts with means S_r h_i and h fixed.
Ignoring terms independent of S_r, the log likelihood is

\[
\ell(S_r)=\left(\sum_i y_{ri}\right)\log S_r-S_r\sum_i h_i.
\]

Its maximum-likelihood estimate is

\[
\widehat S_r^{\mathrm{Poisson}}=\frac{\sum_i y_{ri}}{\sum_i h_i}.
\]

For mean-one h, this is exactly the implemented observed transcript mean.
It is therefore a natural estimate of a nuisance amplitude for a normalized
shape, rather than an arbitrary scaling rule. It uses the entire CDS and
does not fit an independent mean at every codon.

**This result does not establish maximum-likelihood estimation under the
actual NB2 model.** For fixed h and alpha, the NB2 scale score is

\[
\frac{\partial\ell}{\partial S_r}
=\sum_i\frac{y_{ri}-S_r h_i}{S_r(1+\alpha_i S_r h_i)}.
\]

In general, setting this score to zero does not yield the simple transcript
mean, even when mean(h)=1. The Poisson argument also requires the denominator
sum(h); it does not justify using mean(y) as the Poisson MLE for the
unnormalized Exp8 shape.

The current approach is best described as **NB-based profile fitting with
plug-in transcript/replicate scales**. Its limitations follow from that choice:

- Scale estimation absorbs biological abundance differences and sampling
  fluctuations together with technical depth. Alpha describes residual
  variation around fitted scales, not all variation between raw replicas.
- Scale uncertainty is ignored. This matters especially for low-count
  transcripts; an all-zero replicate receives a scale near zero automatically.
- Using data to estimate nuisance parameters is legitimate, but plugging
  S_hat(y) into ordinary NB probabilities does not by itself give the exact
  distribution conditional on the observed total, or calibrated predictions
  for the total of a new replicate.
- The scalar adjustment cannot distinguish a depth effect from a biological
  change that alters the entire transcript's amplitude. It also cannot correct
  replicate-specific positional bias.
- Shared shape/alpha, consensus PCC penalties and replica averaging are
  additional assumptions. The complete objective is not an unweighted joint
  likelihood, and alpha is not a direct empirical variance estimator.

For a shape-only objective, discarding abundance variation can be intentional.
For a study of abundance, total occupancy variation or replicate uncertainty,
the same decision would remove part of the signal of interest.

## Alternative: library exposures with a shared transcript amplitude

A more explicit model would separate library exposure from transcript signal:

\[
Y_{dtri}\mid s_{dr},A_{dt},h_{dti},\alpha_{dti}
\sim\operatorname{NB2}\left(s_{dr}A_{dt}h_{dti},\alpha_{dti}\right).
\]

Here s_dr is one exposure factor per replicate/library, estimated across many
transcripts under stated normalization assumptions. A_dt is a shared
transcript/dataset amplitude fitted using all its replicates. The shape and
dispersion are shared as before. Normalize the exposure factors, for example
to geometric mean one within a dataset, to fix their scale relative to A.

At identical exposures this reduces to the same-distribution interpretation.
At different exposures the expected raw counts differ, while the underlying
transcript signal remains shared. Crucially, each observed transcript total
does not determine its own mean independently of the other replicas.

This formulation preserves more information about variation between replicate
transcript totals. It requires reliable sample identities/exposure estimates,
a strategy for fitting and regularizing A_dt, and a specification of inference
for a new transcript or library. Observed test profiles cannot be reused to fit
A without labeling the resulting evaluation as conditional on those observations.
The existing held-out sequence-only L evaluation remains a separate task.

If replicate abundance variation should be modeled explicitly, a possible
extension is

\[
\mu_{dtri}=s_{dr}A_{dt}\exp(u_{dtr})h_{dti},
\qquad u_{dtr}\sim\mathcal N(-\tau_{dt}^2/2,\tau_{dt}^2).
\]

The mean-one random multiplier introduces variation shared across all codons
of a transcript replicate. Conditional NB dispersion then describes residual
count variation. Such an extension needs shrinkage and identifiability checks;
few replicates will not reliably identify unrestricted tau and alpha together.

For a mean-one h, A_dt has the interpretation of mean counts per codon at
reference exposure. For Exp8's unnormalized h it is just a multiplicative
amplitude, and its interaction with the L/gamma gauge must be addressed.
Silently adding shape normalization while changing the scale model would
change two aspects of the experiment at once.

If the goal instead explicitly conditions on each transcript's known total,
a second alternative is a count-vector model. Set p_i=h_i/sum(h) and
K_r=sum_i y_ri, and use

\[
\boldsymbol Y_r\mid K_r\sim\operatorname{Multinomial}(K_r,\boldsymbol p).
\]

This follows exactly by conditioning independent Poisson counts on their sum.
A Dirichlet-multinomial could model additional variation in allocation, but
introduces its own covariance and dispersion assumptions. It is not the
conditional equivalent of the current position-specific-alpha NB2 model.
These models deliberately do not explain variation in K_r.

## How to justify the choice in a methods note

The following wording reflects the implemented objective without claiming
that its statistical superiority has already been demonstrated:

> We model relative ribosome-profile shape while treating transcript-level
> amplitude as a nuisance quantity. Replicates therefore share a positional
> shape and an NB2 dispersion function, while their count means use separate
> scales estimated from their observed mean CDS counts. This allows replicate
> profiles with different depths to inform a common shape without imposing
> identical expected raw counts. Raw replicate counts enter the NB loss
> individually; consensus profiles enter the PCC terms. This choice removes
> overall amplitude differences from the variation attributed to dispersion.
> The resulting dispersion describes residual variation around fitted scales,
> rather than total raw between-replicate variability, and uncertainty in the
> estimated scales is not explicitly modeled.

A convincing empirical justification would compare the current scale rule
with a common-S baseline and the exposure/shared-amplitude alternative, holding
dataset membership, pi, transcript splits, architecture, loss coefficients and
training seeds fixed. Use the same shape normalization in every arm. Simulation
should include both equal true scales and unequal true scales, with known
shape and dispersion; compare shape recovery and dispersion calibration.

On real data, assess shape reproducibility and, separately, predictive fit to
held-out replicates. A method receiving the held-out replicate's own total
must be evaluated as conditional on that information, with comparable
information supplied to its competitors. It cannot be credited with predicting
that total. Include repeated seeds and stratification by count depth and
replicate imbalance. The existing equal-pi versus ranked-pi Exp8 comparison
changes panel design and provides no direct evidence for this scale choice.
