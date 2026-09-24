# From programmed kinetics \(K_t\) to stochastic TASEP occupancies \(q_t^{(r)}\)

## Short answer

The two \(q_t\) replicas are **stochastic realizations generated downstream of
\(K_t\)**, but they are **not direct noisy observations drawn around \(K_t\)**.

More precisely:

- \(K_t\) is a deterministic, mean-one profile of programmed P-site residence
  times.
- Those residence times determine the position-specific elongation rates of a
  stochastic, interacting TASEP simulation.
- Each simulated trajectory produces a finite-time P-site occupancy profile
  \(q_t^{(r)}\).
- Ribosome exclusion, initiation and termination boundaries, finite recording,
  and stochastic event timing can all make \(q_t^{(r)}\) differ from \(K_t\).
- The observed Ribo-seq counts are sampled one stage later, from an NB2 model
  whose mean is proportional to \(q_t^{(r)}\) after applying the chosen
  technical-bias multiplier.

The correct dependency is therefore

\[
K_t
\longrightarrow
\text{TASEP dynamics}
\longrightarrow
q_t^{(r)}
\longrightarrow
\mu_{t,f}^{(r)}
\longrightarrow
Y_{t,f}^{(r)},
\]

not

\[
q_t^{(r)} = K_t + \text{simple measurement noise}.
\]

That distinction determines which target should be used for each synthetic
evaluation.

---

## 1. The three different objects

### 1.1 \(K_t\): deterministic programmed kinetics

For transcript \(t\) and P-site position \(i\), let \(\tau_{t,i}\) denote the
programmed residence time. The saved kinetic target is

\[
K_{t,i}
=
\frac{\tau_{t,i}}
     {\ell_t^{-1}\sum_{j=1}^{\ell_t}\tau_{t,j}},
\]

where \(\ell_t\) is the number of modeled P-site states. Consequently,

\[
\frac{1}{\ell_t}\sum_i K_{t,i}=1.
\]

The corresponding unblocked elongation rate is \(\tau_{t,i}^{-1}\). Once the
transcript sequence and the programmed context rules are fixed, \(K_t\) is
fixed. It contains no Gillespie randomness, finite-trajectory variation,
technical bias, or NB2 count noise.

This is confirmed by the metadata of
[`artificial_ground_truth_kinetics_target_mean_one.parquet`](../Datasets/Synthetic_data/artificial_ground_truth_kinetics_target_mean_one.parquet):

- 19,290 transcript profiles;
- mean-one within each transcript;
- TASEP exclusion not applied;
- Gillespie sampling not applied;
- read sampling not applied;
- terminal boundary excluded.

Thus, \(K_t\) is the appropriate reference for the question:

> Did the model recover the sequence-dependent kinetics programmed into the
> simulator?

It is not the conditional mean of the final count observations.

### 1.2 \(q_t^{(r)}\): stochastic finite-time TASEP occupancy

For trajectory replica \(r\), define

\[
Z_{t,i}^{(r)}(s)
=
\begin{cases}
1, & \text{if a ribosome P-site occupies position } i \text{ at time } s,\\
0, & \text{otherwise.}
\end{cases}
\]

The raw time-averaged occupancy recorded over duration \(T_t\) is

\[
O_{t,i}^{(r)}
=
\frac{1}{T_t}\int_0^{T_t} Z_{t,i}^{(r)}(s)\,\mathrm ds.
\]

The mean-one occupancy profile used for shape comparisons and count generation
is

\[
q_{t,i}^{(r)}
=
\frac{O_{t,i}^{(r)}}
     {\ell_t^{-1}\sum_j O_{t,j}^{(r)}}.
\]

The file
[`artificial_ground_truth_tasep_occupancy_replicates.parquet`](../Datasets/Synthetic_data/artificial_ground_truth_tasep_occupancy_replicates.parquet)
contains **raw \(O_t^{(r)}\)** rather than already normalized \(q_t^{(r)}\).
Its metadata describes the values as raw time-averaged P-site occupancy, with
no bias and no NB2 sampling. The current analysis normalizes each trajectory
within transcript before calling it \(q_t^{(r)}\).

The file contains:

- 19,290 transcripts;
- two rows per transcript, or 38,580 rows in total;
- `replicate_1_mean_psite_occupancy`;
- `replicate_2_mean_psite_occupancy`;
- P-site states in codon-index order;
- no terminal boundary position.

Unlike \(K_t\), each \(q_t^{(r)}\) is random because it is a functional of a
finite stochastic Gillespie trajectory.

### 1.3 \(Y_{t,f}^{(r)}\): biased, sampled counts

For technical-bias condition \(f\), nominal read depth \(C\), and trajectory
replica \(r\), the count mean is

\[
\mu_{t,f,i}^{(r)}
=
C\,q_{t,i}^{(r)}b_f(t,i),
\]

where \(b_f(t,i)\) is the deterministic sequence-dependent observation
multiplier. Counts are then sampled as

\[
Y_{t,f,i}^{(r)}\mid q_t^{(r)},b_f,C
\sim
\operatorname{NB2}\!\left(
\mu_{t,f,i}^{(r)},\alpha_{\mathrm{sim}}=0.1
\right),
\]

with

\[
\operatorname{Var}(Y\mid\mu)=\mu+0.1\mu^2.
\]

Therefore an observed count replica contains at least two stochastic layers:

1. variation of the finite TASEP trajectory, represented by
   \(q_t^{(r)}\); and
2. conditional NB2 count-sampling variation, represented by
   \(Y_{t,f}^{(r)}\mid q_t^{(r)}\).

The bias multiplier is fixed across trajectory replicas and sequencing depths.
The same two TASEP trajectories are reused across the technical-bias
conditions and depths. The resulting datasets are paired synthetic views, not
independent regenerations of the complete biological process.

---

## 2. The precise probabilistic interpretation

Let \(\theta_t\) collect all fixed TASEP parameters for transcript \(t\), for
example:

\[
\theta_t
=
\left(
\{\tau_{t,i}^{-1}\}_i,
\text{initiation rate},
\text{termination rate},
\text{footprint length},
\ell_t,
T_t
\right).
\]

Since \(K_t\) is a normalized transformation of the dwell times, it determines
part of \(\theta_t\), but not every component. A normalized trajectory may be
written abstractly as

\[
q_t^{(r)}
=
g_{T_t}\!\left(X_t^{(r)};\theta_t\right),
\]

where \(X_t^{(r)}\) is one stochastic continuous-time TASEP trajectory and
\(g_{T_t}\) computes its time-averaged, mean-one occupancy.

The intended interpretation of the two replicas is therefore

\[
q_t^{(1)},q_t^{(2)}
\sim
\mathcal Q_t(\,\cdot\mid\theta_t),
\]

with separate trajectory realizations. If separate independent random-number
streams were used, they are conditionally independent given \(\theta_t\).
The exported metadata establish two distinct trajectory outputs, although they
do not themselves record enough RNG information to independently audit the
random-stream independence.

Two qualifications are essential.

### 2.1 Conditioning on \(K_t\) alone is incomplete

Initiation, termination, footprint exclusion, transcript length, and recording
duration also determine the distribution of \(q_t^{(r)}\). It is therefore
more accurate to write

\[
q_t^{(r)}\sim\mathcal Q_t(\cdot\mid\theta_t)
\]

than to write only

\[
q_t^{(r)}\sim\mathcal Q(\cdot\mid K_t).
\]

### 2.2 The distribution is not guaranteed to be centered on \(K_t\)

In general,

\[
\mathbb E\!\left[q_t^{(r)}\mid\theta_t\right]
\neq K_t.
\]

The TASEP is an interacting dynamical system, not an additive noise model.
Furthermore, within-transcript normalization is nonlinear, so even

\[
\mathbb E\!\left[
\frac{O_{t,i}^{(r)}}{\bar O_t^{(r)}}
\right]
\]

need not equal

\[
\frac{\mathbb E[O_{t,i}^{(r)}]}
     {\mathbb E[\bar O_t^{(r)}]}.
\]

Thus, it is not statistically correct to describe the replicas as
\(K_t+\varepsilon_t^{(r)}\) with zero-mean independent measurement error.

---

## 3. When would \(q_t\) equal \(K_t\)?

The equality is obtained in an ideal collision-free steady-state argument.
Suppose the same ribosome flux \(J_t\) passes every position and a ribosome
occupies position \(i\) for mean time \(\tau_{t,i}\). Then

\[
O_{t,i}^{\mathrm{cf}}=J_t\tau_{t,i}.
\]

After mean-one normalization,

\[
q_{t,i}^{\mathrm{cf}}
=
\frac{J_t\tau_{t,i}}
     {\ell_t^{-1}\sum_j J_t\tau_{t,j}}
=
\frac{\tau_{t,i}}
     {\ell_t^{-1}\sum_j\tau_{t,j}}
=K_{t,i}.
\]

The flux cancels. This explains why \(K_t\) and \(q_t^{(r)}\) should be
strongly related.

It does **not** prove equality in the simulated system. The derivation assumes:

- negligible ribosome-ribosome blocking;
- a common stationary flux through the modeled positions;
- no important boundary distortion;
- effectively infinite recording time;
- exact estimation of the stationary occupancy.

The actual synthetic simulation uses extended particles with 10-codon
footprints, stochastic initiation and elongation, exclusion, transcript
boundaries, and finite recording. Those are precisely the mechanisms that can
break the collision-free relation.

---

## 4. Why the two \(q_t\) replicas differ from one another and from \(K_t\)

It is useful to separate two conceptually different sources of deviation.

### 4.1 Systematic dynamical transformation

Define \(q_t^\star\) as the normalized stationary P-site occupancy profile of
the specified TASEP, in the infinite-recording limit. Then

\[
q_t^\star-K_t
\]

captures systematic changes induced by the traffic model. Examples include:

- queues upstream of slow positions;
- occupation time caused by blocking rather than the local intrinsic dwell;
- initiation-limited or termination-affected profiles;
- interactions introduced by the 10-codon footprint;
- density-dependent redistribution of occupancy.

This term would remain even with an infinitely long, perfectly measured TASEP
trajectory whenever the stationary interacting system does not preserve the
collision-free profile.

### 4.2 Finite-trajectory Monte Carlo variation

The second component is

\[
q_t^{(r)}-q_t^\star.
\]

It arises because only a finite trajectory is recorded. Event times,
temporary queues, and the number of visits to each position vary across
replicas. Under standard ergodic conditions this component should diminish as
the recording duration increases.

Together,

\[
q_t^{(r)}-K_t
=
\underbrace{(q_t^\star-K_t)}_{\text{traffic/boundary transformation}}
+
\underbrace{(q_t^{(r)}-q_t^\star)}_{\text{finite-trajectory variation}}.
\]

This equation is the most useful mental model. It also shows why one cannot
infer the traffic-induced component from only one trajectory.

---

## 5. What averaging the two occupancy replicas does

The arithmetic mean

\[
\bar q_t=\frac{q_t^{(1)}+q_t^{(2)}}{2}
\]

reduces trajectory-specific variation if the two runs are conditionally
independent. It does not remove the systematic difference
\(q_t^\star-K_t\).

With only two trajectories, \(\bar q_t\) is still a noisy approximation to
\(q_t^\star\). It should therefore be called a **two-trajectory occupancy
consensus**, not the exact stationary occupancy truth.

There is also an exact finite-sample identity. For

\[
m_t=\frac{q_t^{(1)}+q_t^{(2)}}{2},
\]

the position-averaged squared errors obey

\[
\frac{1}{2}\left[
\operatorname{MSE}(q_t^{(1)},K_t)
+
\operatorname{MSE}(q_t^{(2)},K_t)
\right]
=
\operatorname{MSE}(m_t,K_t)
+
\frac{1}{4}\operatorname{MSE}(q_t^{(1)},q_t^{(2)}).
\]

The first term on the right is the residual of the two-trajectory mean from
\(K_t\). It contains both systematic traffic effects and remaining finite-run
noise. The second is the replica-varying component. This algebra does not
require a probabilistic model, but it does not by itself identify
\(q_t^\star-K_t\).

---

## 6. What the exported data show empirically

A direct audit was performed on all 19,290 transcripts by normalizing each raw
occupancy trajectory to mean one and comparing it with the saved \(K_t\) on
the identical P-site coordinates. No smoothing, truncation, or post-hoc
renormalization beyond the required definition of \(q_t^{(r)}\) was used.

| Comparison | Median PCC | PCC IQR | Mean PCC | Median RMSE |
| --- | ---: | ---: | ---: | ---: |
| \(q_t^{(1)}\) vs. \(K_t\) | 0.9294 | 0.9195--0.9383 | 0.9278 | 0.1966 |
| \(q_t^{(2)}\) vs. \(K_t\) | 0.9294 | 0.9193--0.9383 | 0.9277 | 0.1966 |
| \(\bar q_t\) vs. \(K_t\) | 0.9617 | 0.9569--0.9662 | 0.9607 | 0.1410 |
| \(q_t^{(1)}\) vs. \(q_t^{(2)}\) | 0.8659 | 0.8492--0.8819 | 0.8647 | 0.2760 |

These observations support four conclusions:

1. **The programmed kinetics strongly constrain the occupancy profiles.**
   Both trajectories have median correlation about 0.929 with \(K_t\).
2. **The trajectories are not identical.** Their median mutual correlation is
   about 0.866, showing appreciable finite-trajectory variation.
3. **Averaging cancels a substantial trajectory-specific component.** The
   two-trajectory consensus has median correlation about 0.962 with \(K_t\).
4. **High correlation is not identity.** Traffic, boundaries, finite
   recording, and normalization still leave a measurable discrepancy.

The exact MSE decomposition above gives, after equal weighting of transcripts:

| Component | Mean per-transcript MSE | Fraction of mean single-replica MSE |
| --- | ---: | ---: |
| Residual of \(\bar q_t\) from \(K_t\) | 0.02070 | 51.7% |
| Replica-varying component, \(\tfrac14\operatorname{MSE}(q^{(1)},q^{(2)})\) | 0.01935 | 48.3% |
| Mean single-replica MSE from \(K_t\) | 0.04004 | 100% |

This is a descriptive decomposition of these two trajectories. The 51.7%
residual must not be called purely “systematic traffic error”: with only two
replicas, it still contains finite-trajectory noise that did not cancel.
Likewise, the 48.3% component measures between-trajectory variation, not a
general biological variance parameter.

The model-ready weighted collection retains 19,283 of these transcripts
because seven occupancy IDs are absent from the downstream sequence table.
On that retained set, the previously reported median
\(\operatorname{PCC}(q_t^{(1)},q_t^{(2)})\) is 0.86594, effectively unchanged
from the all-transcript audit.

---

## 7. Which reference should be used for which scientific question?

No single target answers every question.

| Scientific question | Appropriate comparison | Interpretation |
| --- | --- | --- |
| Was a sampled count replica generated consistently with its latent trajectory? | \(Y_{t,f}^{(r)}\) vs. matching \(q_t^{(r)}\), accounting for \(b_f\) when testing the full mean | Observation-generation fidelity |
| Does averaging two count replicas reduce count noise? | \(\bar Y_{t,f}\) vs. \(\bar q_t\) | Benefit of replica consensus relative to the two realized trajectories |
| Did the learned shared profile recover the proximal pre-bias occupancy represented in these simulations? | learned \(L_t\) vs. \(\bar q_t\), with the two-replica limitation stated | Traffic-aware shared-profile recovery |
| Did the learned shared profile recover the programmed sequence-dependent dwell pattern? | learned \(L_t\) vs. \(K_t\) | Programmed-kinetic recovery |
| Did the model recover the injected observation effect? | learned \(\gamma_{t,f}\) vs. programmed \(b_f\), after putting both in the same identifiable gamma gauge | Dataset-specific bias recovery |
| How much finite-trajectory variation exists before count sampling? | \(q_t^{(1)}\) vs. \(q_t^{(2)}\) | TASEP trajectory repeatability |
| How reproducible are final observations? | \(Y_{t,f}^{(1)}\) vs. \(Y_{t,f}^{(2)}\) | Combined trajectory and count repeatability under a fixed bias |

Two especially important cautions follow.

### 7.1 \(q_t^{(1)}\)-to-\(q_t^{(2)}\) agreement is not a strict ceiling

A learned sequence model can average information over many transcripts,
datasets, and replicas. It may denoise finite trajectories and correlate more
strongly with \(K_t\) or \(q_t^\star\) than one trajectory correlates with the
other. The replica PCC therefore quantifies trajectory repeatability; it is not
a universal maximum achievable model PCC.

### 7.2 \(\bar q_t\) is not biological ground truth

It is the average of two simulator trajectories. It is an excellent proximal
reference for the generated count layer, but it is neither experimentally
validated biology nor an exact stationary TASEP solution. Conversely,
\(K_t\) is exact programmed simulator truth for the dwell-time construction,
but it is more distal from the count-generating layer.

The strongest analysis reports both targets and names the estimand explicitly.

---

## 8. Recommended terminology

Use the following phrases consistently:

- **programmed kinetic target** for \(K_t\);
- **finite-trajectory TASEP occupancy replica** for \(q_t^{(r)}\);
- **two-trajectory occupancy consensus** for \(\bar q_t\);
- **stationary TASEP occupancy** for the conceptual \(q_t^\star\);
- **biased NB2 observation** for \(Y_{t,f}^{(r)}\);
- **simulator-defined ground truth** only when the exact simulator quantity is
  specified.

Avoid:

- “\(q_t\) is a noisy measurement of \(K_t\)” without explaining the TASEP
  transformation;
- “\(q_t\) is centered on \(K_t\)” unless this is established under a stated
  regime;
- “\(\bar q_t\) is the exact true occupancy”;
- “replica agreement is model accuracy”;
- “biological ground truth” for any of these simulator-defined quantities.

---

## 9. A notation collision in this project

Some real-data split documentation uses \(q_t\) for an unrelated transcript
reliability statistic,

\[
q_t^{\mathrm{split}}=\operatorname{median}_d w_{dt}.
\]

That quantity has nothing to do with TASEP occupancy. To avoid ambiguity in a
paper or shared code, a safer notation is:

\[
Q_{t,i}^{(r)}
\quad\text{or}\quad
q_{t,i}^{\mathrm{occ},(r)}
\]

for the normalized TASEP occupancy, while retaining
\(q_t^{\mathrm{split}}\) only for the split score. If the existing appendix
keeps lowercase \(q_t^{(r)}\), its meaning should be redefined explicitly in
every self-contained section.

---

## 10. The clean generative statement for the manuscript

A compact but accurate description is:

> The deterministic programmed dwell profile \(K_t\) parameterizes the
> transcript-specific elongation rates of an open extended TASEP. Two finite
> Gillespie trajectories yield replicate-specific normalized P-site occupancy
> profiles \(q_t^{(1)}\) and \(q_t^{(2)}\). These occupancies are stochastic
> descendants of \(K_t\), not direct zero-mean-noise observations of it:
> exclusion, boundary conditions, and finite recording can alter their shape.
> For technical-bias condition \(f\), replicate counts are subsequently drawn
> from an NB2 distribution with mean
> \(Cq_{t,i}^{(r)}b_f(t,i)\). We therefore use matching \(q_t^{(r)}\) to audit
> count generation, their two-trajectory mean \(\bar q_t\) as a proximal
> traffic-aware reference for shared-profile recovery, and \(K_t\) separately
> to evaluate recovery of the programmed kinetic signal.

---

## 11. Bottom line

The answer to “are the \(q_t\) replicas probabilistic samples with respect to
\(K_t\)?” is:

> **Yes, they are stochastic trajectory samples generated from a TASEP whose
> elongation rates are determined by the dwell times underlying \(K_t\). No,
> they are not direct samples from a distribution that is guaranteed to have
> \(K_t\) as its mean.**

The ideal hierarchy to remember is

\[
\boxed{
\text{sequence}
\rightarrow \tau_t
\rightarrow K_t
\rightarrow \text{stochastic TASEP}
\rightarrow q_t^{(r)}
\rightarrow Cq_t^{(r)}b_f
\rightarrow Y_{t,f}^{(r)}
}
\]

and each arrow introduces a different scientific question and, in some cases,
a different source of variation.

## Project sources

- [`Datasets/Synthetic_data/artificial_ground_truth_kinetics_target_mean_one.parquet`](../Datasets/Synthetic_data/artificial_ground_truth_kinetics_target_mean_one.parquet)
- [`Datasets/Synthetic_data/artificial_ground_truth_tasep_occupancy_replicates.parquet`](../Datasets/Synthetic_data/artificial_ground_truth_tasep_occupancy_replicates.parquet)
- [`Datasets/Synthetic_data/artificial_ground_truth_tasep_occupancy_replicates.parquet.summary.json`](../Datasets/Synthetic_data/artificial_ground_truth_tasep_occupancy_replicates.parquet.summary.json)
- [`analyses/analyze_synthetic_tasep_occupancy_agreement.py`](../analyses/analyze_synthetic_tasep_occupancy_agreement.py)
- [`analyses/artifacts/synthetic/individual_dataset/tasep_occupancy_provenance.json`](../analyses/artifacts/synthetic/individual_dataset/tasep_occupancy_provenance.json)
- [`Docs/iclr_synthetic_data_analysis_section.tex`](iclr_synthetic_data_analysis_section.tex)
