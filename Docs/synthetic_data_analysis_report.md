# Synthetic bias/read-depth recovery: design, methods, and preliminary results

Report snapshot: **2026-08-11**. This report describes the files and runtime
artifacts currently present in the repository. The experiment matrix is still
in progress: the current results directory contains the 0.25-read/codon
cumulative panels with 2 through 7 datasets; detailed prediction parquets are
available through the 5-dataset panel.

## Executive summary

The synthetic experiment asks two different questions:

1. Can the model recover the one biological profile shared across biased
   datasets? This is tested by comparing learned `L_bio` with the deterministic
   mean-one kinetic truth.
2. Can the model recover each dataset's deliberately programmed positional
   bias? This is tested by comparing learned `gamma` with the programmed bias
   after putting both in the same identifiable gamma gauge.

The preliminary answer to both questions is positive. At 0.25 reads/codon,
the mean held-out transcript `L_bio` PCC increases from **0.7881** with two
biased datasets to **0.9354** with five datasets that have complete prediction
outputs. The selected-checkpoint scalar reaches **0.9395** with six datasets.
For the five-dataset panel, learned and programmed gauge-fixed log-gamma have a
pooled PCC of **0.9956**, a log-RMSE of **0.0317**, and a calibration slope of
**1.0042**. At positions that were explicitly biased, mean multiplicative
error is **3.72%**, with **99.09%** of those positions within 10%.

These are preliminary results, not a final dataset-count law. The panels are
cumulative, so adding a dataset also adds a particular new bias condition.
Only one seed and one read depth have completed result artifacts so far.

![Shared-signal recovery across the completed panels](../results/riboai_synthetic_experiments/recovery_report_1/synthetic_recovery_overview.png)

![Gamma recovery across the completed panels](../results/riboai_synthetic_experiments/recovery_report_1/synthetic_gamma_recovery_overview.png)

## 1. What the synthetic files represent

The upstream simulation generator is not included in this repository. The
description below is reconstructed from the Parquet metadata, the supplied
data description, and the active preprocessing and analysis code.

### 1.1 Deterministic shared kinetic truth

[`artificial_ground_truth_kinetics_target_mean_one.parquet`](../Datasets/Synthetic_data/artificial_ground_truth_kinetics_target_mean_one.parquet)
contains one floating-point profile for each of **19,290 transcripts**. If
`k_ti` is the programmed P-site residence time at codon position `i`, the
stored profile is

```text
K_ti = k_ti / mean_i(k_ti).
```

Thus every transcript profile has mean one. Parquet metadata confirms that
this target is deterministic and has no TASEP exclusion, no Gillespie
sampling, and no finite-read sampling. It is the reference for learned
`L_bio`, not a count target used by the optimizer.

### 1.2 Artificial sequence-dependent biases

Each `artificial_bias_*.parquet` contains codon-resolution P-site counts in
long format:

| column | meaning |
| --- | --- |
| `sample` | bias name followed by `rep1`, `rep2`, or `mean` |
| `transcript_id` | transcript identifier |
| `rib_profile` | integer count vector over P-site codons |

There are **57,870 rows per file**: 19,290 transcripts times two independent
replicates plus one derived mean profile. Metadata states that the mean is a
stable integerization of the arithmetic mean of the two replicas; it is not a
third negative-binomial draw.

The programmed multiplier `b_dti` depends on the ideal 30-nt RPF sequence at
position `i`. Biased expected counts obey

```text
mu_biased,dti = mu_unbiased,ti * b_dti.
```

The multipliers are not renormalized. Consequently, a bias can change both
the positional shape and the total reads of a transcript. The stored bias
annotations use `added_bias = b - 1`.

| condition | programmed multiplier at selected positions | stored nonzero `added_bias` |
| --- | ---: | ---: |
| 3′ AA | 2.8–3.2 | 1.8–2.2 |
| 3′ CC | 5.0–6.0 | 4.0–5.0 |
| 3′ GG | 4.0–4.4 | 3.0–3.4 |
| 3′ UU | 5.8–6.2 | 4.8–5.2 |
| 5′ AA | 3.0–3.4 | 2.0–2.4 |
| 5′ CC | 5.2–5.6 | 4.2–4.6 |
| 5′ GG | 4.2–4.6 | 3.2–3.6 |
| 5′ UU | 5.6–6.0 | 4.6–5.0 |
| AU fraction > 0.7 | 7.0–7.4 | 6.0–6.4 |
| GC fraction > 0.7 | 5.0–5.4 | 4.0–4.4 |

The files under [`bias_profile`](../Datasets/Synthetic_data/bias_profile/)
are depth-independent annotations and are systematic across replicates and
read depths.

### 1.3 Three sequencing-depth regimes

The same latent kinetics and bias rules are observed at three baseline
sequencing depths:

| directory | baseline expected reads per codon | qualitative regime |
| --- | ---: | --- |
| `0p25_per_codon` | 0.25 | sparse and noisy |
| `2_per_codon` | 2 | intermediate |
| `20_per_codon` | 20 | dense and comparatively precise |

Counts are sampled from an NB2 model with dispersion `alpha = 0.1`:

```text
Var(Y_dti) = mu_dti + 0.1 * mu_dti^2.
```

The metadata records bias seed `20260807`, observation seed `20260808`, and a
common source fingerprint across the three depths. The unbiased
`artificial_ground_truth_psite_counts_*` files provide finite-read observations
without sequence-dependent bias. They are useful observation-level controls;
they are not the deterministic kinetic truth.

### 1.4 Coordinate convention

All raw synthetic profiles use 0-based P-site codon coordinates. The terminal
stop boundary is excluded. The model's sequence table includes that terminal
position, so preprocessing appends one zero to the consensus and both replica
profiles. Recovery analyses remove this padded position before comparison
with kinetic truth or bias annotations.

## 2. From raw synthetic files to model-ready pairs

The active preprocessing entry point is
[`weight_synthetic_riboseq_codon_replicas.py`](../Datasets/data/weight_synthetic_riboseq_codon_replicas.py).
For each depth and condition it performs the following pipeline:

```text
rep1 + rep2 + derived mean
        |
        v
pivot to one transcript-dataset row
        |
        v
intersect with the MANE sequence table
        |
        v
validate codons and profile lengths; append terminal zero
        |
        v
validate counts; physically remove zero-information rows
        |
        v
compute a positive, median-normalized reliability weight
```

The wide output keeps `id`, consensus `ribo`, both raw
`ribo_cds_replicas`, and `replica_ids`. Of 19,290 raw transcript IDs, 19,283
are present in the current sequence table. Seven are therefore removed. The
current audit found no invalid codons and no additional zero-coverage removals
in the 33 processed synthetic datasets. Nevertheless, these checks are active:
profiles must be one-dimensional, non-empty, finite, non-negative, have
positive total reads, and have positive coverage.

### 2.1 Transcript reliability weights

For one retained transcript-dataset pair, let

```text
L_dt = aligned profile length
D_dt = total_reads_dt / L_dt
C_dt = positive_positions_dt / L_dt.
```

The active `snr_depth_coverage` scheme fits the dataset-specific reference

```text
tau_d = median_t(D_dt)
```

and calculates

```text
depth_score_dt = sqrt(D_dt) / (sqrt(D_dt) + sqrt(tau_d))
raw_weight_dt  = 0.70 * depth_score_dt + 0.30 * C_dt
weight_dt      = raw_weight_dt / median_t(raw_weight_dt).
```

The final weight is positive, stored as float32, has dataset median one, and
is not clipped: values above one are valid. It is a local measurement-quality
weight, not a dataset-quality rank.

Representative values for the 3′-AA dataset are:

| depth | rows | median coverage | fitted `tau_d` | final weight range |
| --- | ---: | ---: | ---: | ---: |
| 0.25 | 19,283 | 0.2557 | 0.2813 | 0.6173–1.1722 |
| 2 | 19,283 | 0.8674 | 2.2335 | 0.9171–1.1036 |
| 20 | 19,283 | 0.9976 | 22.2984 | 0.9509–1.0556 |

The current fallback fits `tau_d` and the normalization median on the complete
dataset before splitting. A future strict leakage-free protocol should fit
both statistics on training IDs and freeze them for validation/test. This
does not change the recovery calculations, but it is a methodological caveat
for a final locked evaluation.

## 3. Training and validation split

Within one read depth, `split.master_dataset_universe=all` means all 11
processed datasets—10 biases plus the unbiased finite-read dataset—define the
split. A transcript is eligible for validation only when it has a retained,
positive-weight row in every dataset in that master universe.

For each common transcript, the split-only reliability score is the median of
its 11 local weights. The launcher resolves two reliability quantile bins and
jointly stratifies by the sequence table's CSS-count category. CSS is only a
stratification hint; it receives no separate quota. Sampling is deterministic
with seed 42 and takes 10% from the joint strata.

The current 0.25-depth manifest records:

| quantity | count |
| --- | ---: |
| master union | 19,283 |
| common validation candidates | 19,283 |
| training IDs | 17,355 |
| validation IDs | 1,928 |
| validation IDs in lower reliability half | 964 |
| validation IDs in upper reliability half | 964 |

This answers two competing requirements at once: the same validation
transcripts are available in every dataset used by a panel, but validation is
not restricted to only the highest-weight transcripts. All nine cumulative
panels at a given depth use the same master universe and therefore the same
validation IDs. Validation IDs are not guaranteed to be identical across the
three depths because the reliability ordering can change with read depth.

Every remaining common ID belongs to training; each selected dataset then
contributes whatever retained train pairs it contains. There is no separate
CSS validation partition.

## 4. Experiment matrix and launcher

Experiments are launched through
[`run_synthetic_bias_read_depth_local.sh`](../run_synthetic_bias_read_depth_local.sh).
It defines 47 deterministic tasks.

### 4.1 Within-depth cumulative panels: 27 tasks

At each of the three depths, nine panels use the first `N = 2,...,10` biases
from this fixed order:

1. 3′ AA
2. 3′ CC
3. 3′ GG
4. 3′ UU
5. 5′ AA
6. 5′ CC
7. 5′ GG
8. 5′ UU
9. GC fraction > 0.7
10. AU fraction > 0.7

Task ranges are 0–8 for 0.25 reads/codon, 9–17 for 2 reads/codon, and 18–26
for 20 reads/codon. These panels test whether multiple independent views of
the same kinetics help recover a shared signal. Because the panels are
cumulative, however, `N` and the identity of the newly added condition are
confounded.

### 4.2 Cross-depth panels: 20 tasks

For each bias, the launcher also combines that condition at 0.25, 2, and 20
reads/codon. It runs the triplet twice:

- equal gamma-reference weights, with normalized `pi = (1/3, 1/3, 1/3)`;
- quality-ranked gamma-reference weights. At power one, the shallow,
  intermediate, and deep raw quality values are `(1/3, 2/3, 1)`, giving
  normalized gamma-reference weights `pi = (1/6, 1/3, 1/2)`.

Dataset-quality rank is confined to gamma-reference centering. It is not
multiplied into the training loss.

### 4.3 Reproducibility and GPU scheduling

Every task uses experiment seed 42, fixed-reference gamma centering, and the
transcript-grouped sampler. The local launcher runs each experiment as one
single-GPU process (`trainer.devices=[0]` after `CUDA_VISIBLE_DEVICES`
remapping). If `GPU_DEVICES=0,1` is supplied, tasks are assigned round-robin to
two independent sequential queues that execute concurrently. This is parallel
experiment execution, not DDP training of one experiment.

The current local launcher does not expose a `BATCH_SIZE` environment
override. Its resolved runs inherit `data.batch_size=32` from the synthetic
YAML. Worker counts and validation-bin count are launcher overrides.

## 5. Grouped batching and gradient accumulation

The grouped sampler admits a transcript only when it has positive eligible
observations in at least two distinct selected datasets. All selected rows for
that transcript form one atomic group and are never split by the logical
sampler.

`data.batch_size=32` is a **per-dataset pair quota**, not a total row count.
With `D` complete selected datasets, a full logical batch normally contains
32 transcripts and `32 * D` transcript-dataset rows. For example:

| selected datasets | transcript groups | logical pair rows |
| ---: | ---: | ---: |
| 2 | 32 | 64 |
| 5 | 32 | 160 |
| 10 | 32 | 320 |

Execution microbatching may split a large logical batch into smaller GPU
forwards of at most 128 pair rows and roughly 64,000 padded codon tokens. It
splits only between complete transcript groups and rescales chunk losses so
the logical transcript-balanced objective is preserved.

Automatic accumulation targets 32 unique transcripts per optimizer step. In
the audited N=2 run, a typical logical batch already has 32 transcripts, so
the resolved accumulation factor is one. The plan has 543 logical batches,
with 32 groups in ordinary batches and 11 in the final partial batch. It
admits all 17,355 train transcripts and excludes none for insufficient
support.

## 6. What the model is asked to factorize

For dataset `d`, transcript `t`, and codon `i`, the model constructs

```text
shape_dti = gamma_dti * L_bio_ti
mu_dti    = S_dt * shape_dti / mean_i(shape_dti),
```

where `S_dt` is the observed mean count for that transcript-dataset pair.
Therefore:

- `L_bio_ti` is the shared, mean-one biological shape;
- `gamma_dti` is the positive dataset-specific positional shape correction;
- `S_dt` absorbs the pair's overall count scale, including total-read changes
  caused by the non-renormalized artificial bias;
- `mu_dti` reconstructs the biased observation and is not expected to equal
  the unbiased kinetic truth.

This distinction is crucial. A successful model should produce an `L_bio`
close to the latent kinetics, a `gamma` close to the identifiable part of the
programmed bias, and a `mu` close to each biased dataset.

### 6.1 Why gamma has two gauges

Raw gamma is not uniquely defined. Multiplying a pair's gamma by any constant
does not change `mu` after shape normalization. Also, a positional pattern
shared by all datasets can be assigned either to `L_bio` or to every gamma.

Let `a_dti` be raw log-gamma and `pi_d` the normalized fixed-reference weight.
The model removes both ambiguities jointly:

```text
c_ti    = sum_d pi_d * a_dti
m_dt    = mean_i(a_dti)
a_bar_t = sum_d pi_d * m_dt
g_dti   = a_dti - c_ti - m_dt + a_bar_t
gamma_dti = exp(g_dti).
```

The result satisfies

```text
sum_d pi_d * g_dti = 0        at every position
mean_i(g_dti) = 0             for every transcript-dataset pair.
```

Equivalently, the weighted geometric mean across reference datasets and the
positional geometric mean within each pair are both one. Consequently, gamma
cannot and should not reproduce the raw absolute multiplier `b` directly.
It can recover only the relative, position-dependent bias that remains after
these two identifiable constraints.

## 7. Optimized loss and transcript reliability balancing

For each transcript-dataset pair, the configured optimized quantity is

```text
z_dt = 1.0 * mean_replica NB2_NLL
     + 0.5 * (1 - PCC(mu_dt, consensus_dt))
     + 0.5 * (1 - PCC(VST(mu_dt), VST(consensus_dt)))
     + 0.0 * gamma_regularization.
```

NB2 is evaluated separately against the two raw replicas and averaged inside
the pair. Both PCC terms are calculated once against the arithmetic replica
consensus, so a transcript-dataset pair does not receive two PCC votes merely
because it has two replicas. The NB-VST is the model's dispersion-aware
asinh/square-root transform. Codon-position NB loss uses the configured
length-tempered reduction before pair aggregation.

The selected `transcript_balanced` reducer then computes

```text
L_t = sum_d weight_dt * z_dt / sum_d weight_dt
L_batch = mean_t(L_t).
```

Thus every transcript receives one equal outer vote. Reliability weights only
decide how its available datasets divide that vote. They do not increase a
transcript's total mass, and dataset-quality rank is absent from the loss.

This is exactly the intended use of the preprocessing score: a noisier local
observation counts less than a more reliable observation of the same
transcript, without allowing well-sequenced transcripts or transcripts seen
in more datasets to dominate the outer objective.

Training uses bf16 mixed precision for at most 200 epochs. The scheduler and
early stopping monitor transcript-balanced `val_loss` with patience 20. One
checkpoint is selected by minimum `val_loss`; a second is selected by maximum
unweighted `val_mu_pcc`. Prediction and the current post-hoc recovery reports
use the PCC-best checkpoint. The latent synthetic truth is never used for
checkpoint selection.

## 8. Shared-signal recovery metrics

[`analyze_synthetic_recovery.py`](../analyses/analyze_synthetic_recovery.py)
uses the validation prediction parquet when available. It first verifies that
all dataset rows for one transcript contain the same learned `L_bio` (maximum
observed discrepancy is below `2e-6` in the completed panels). It truncates
the padded terminal position and independently normalizes learned and true
profiles to mean one.

For each validation transcript:

```text
PCC_t  = corr(L_bio_t, K_t)
MSE_t  = mean_i((L_bio_ti - K_ti)^2)
MAE_t  = mean_i(abs(L_bio_ti - K_ti)).
```

Panel PCC and MAE are equal-transcript means. Panel RMSE is
`sqrt(mean_t(MSE_t))`. A 95% PCC interval is obtained from 2,000 transcript
bootstrap resamples. This gives every transcript one vote rather than giving
long transcripts more weight.

The utility also compares normalized observed target and normalized `mu` with
the latent profile for each bias case. Those are factorization diagnostics:
`target` is deliberately biased, while `mu` is supposed to reconstruct that
bias. Neither is the primary shared-kinetics output.

## 9. Gamma recovery metrics

[`analyze_synthetic_gamma_recovery.py`](../analyses/analyze_synthetic_gamma_recovery.py)
loads the `*_mean` annotation from `bias_profile` and reconstructs

```text
a_true,dti = log(1 + added_bias_dti).
```

For every validation transcript it stacks all datasets in the experiment,
uses the reference weights saved in the prediction parquet, and applies the
same joint gauge from Section 6.1 to both the programmed log bias and learned
log-gamma. Learned gamma is first truncated to physical P-site coordinates;
both tensors are then re-gauged on exactly those coordinates.

Let

```text
e_dti = g_learned,dti - g_true,dti.
```

The report calculates:

| metric | definition | ideal value |
| --- | --- | ---: |
| log-gamma PCC | Pearson correlation between learned and true `g` | 1 |
| log-RMSE | `sqrt(mean(e^2))` | 0 |
| log-MAE | `mean(abs(e))` | 0 |
| calibration slope | `sum(g_true*g_learned) / sum(g_true^2)` | 1 |
| multiplicative error | `abs(exp(e) - 1)` | 0 |
| within 1/5/10% | fraction whose multiplicative error is below threshold | 1 |

Metrics are reported in several complementary ways:

- equal mean over transcript-dataset pairs;
- one position-pooled metric across the whole panel;
- per-bias case;
- programmed-site-only metrics using the original `added_bias > 0` mask;
- 95% intervals from 2,000 transcript bootstrap resamples for the equal-pair
  PCC summary.

The analysis additionally checks both gauge residuals and verifies that the
stored `gamma` agrees with `exp(log_gamma)`. Gauge residuals in the current
reports are around `1e-16`, and the largest storage-consistency discrepancy is
below `6e-7`.

## 10. Preliminary shared-signal results

The current result root contains six completed training runs at 0.25
reads/codon. N=2 through N=5 have prediction parquets and full post-hoc
distributions. N=6 and N=7 have selected-checkpoint validation scalars but no
prediction parquet, so their case-level and bootstrap metrics are deliberately
left missing.

| N | evidence | validation transcripts | mean `L_bio` PCC | 95% PCC interval | mean-one RMSE | MAE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | prediction parquet | 1,928 | 0.7881 | 0.7868–0.7893 | 0.3862 | 0.2358 |
| 3 | prediction parquet | 1,928 | 0.8919 | 0.8913–0.8926 | 0.2457 | 0.1708 |
| 4 | prediction parquet | 1,928 | 0.9137 | 0.9132–0.9143 | 0.2125 | 0.1590 |
| 5 | prediction parquet | 1,928 | 0.9354 | 0.9348–0.9359 | 0.1821 | 0.1359 |
| 6 | checkpoint scalar only | 1,928 | 0.9395 | — | 0.1738 | — |
| 7 | checkpoint scalar only | 1,928 | 0.9390 | — | 0.1782 | — |

From N=2 to the last fully detailed panel at N=5, PCC rises by 0.1473 and
RMSE falls by 52.8%. The N=6 scalar is the best current shared-signal result;
the small N=7 decline indicates a plateau rather than a guaranteed monotonic
law.

The N=5 case-level factorization is also informative:

| bias case | biased target vs truth PCC | predicted `mu` vs truth PCC | shared `L_bio` vs truth PCC |
| --- | ---: | ---: | ---: |
| 3′ AA | 0.3290 | 0.7470 | 0.9354 |
| 3′ CC | 0.2949 | 0.4517 | 0.9354 |
| 3′ GG | 0.3114 | 0.5918 | 0.9354 |
| 3′ UU | 0.2943 | 0.4797 | 0.9354 |
| 5′ AA | 0.3371 | 0.6365 | 0.9354 |

The shared profile is much closer to truth than any biased target. `mu` stays
dataset-specific, as it should. This is evidence of genuine factorization
rather than the model simply relabeling one observed profile as biology.

## 11. Preliminary gamma-bias results

Detailed gamma evaluation requires prediction parquets, so it currently covers
N=2 through N=5.

| N | mean pair log-gamma PCC | pooled PCC | pooled log-RMSE | calibration slope | biased-site mean multiplicative error | biased sites within 10% |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.9920 | 0.9954 | 0.0277 | 1.0137 | 2.80% | 99.27% |
| 3 | 0.9913 | 0.9947 | 0.0317 | 1.0016 | 3.57% | 97.98% |
| 4 | 0.9916 | 0.9954 | 0.0320 | 0.9907 | 3.12% | 99.21% |
| 5 | 0.9919 | 0.9956 | 0.0317 | 1.0042 | 3.72% | 99.09% |

The result is stable and very close to the identifiable truth, but it is not
literal floating-point equality. In N=5, 97.87% of all codons are within 5%
multiplicative error. Programmed sites are harder: 75.82% are within 5%, but
99.09% are within 10%. Rare large outliers remain, with maximum absolute
log-error around 1.46, so correlation alone would overstate perfection.

![Individual programmed-bias recovery](../results/riboai_synthetic_experiments/recovery_report_1/synthetic_gamma_recovery_individual_biases.png)

| N=5 bias | pooled log-gamma PCC | pooled log-RMSE | biased-site mean multiplicative error | biased sites within 10% |
| --- | ---: | ---: | ---: | ---: |
| 3′ AA | 0.9956 | 0.0261 | 3.43% | 99.27% |
| 3′ CC | 0.9957 | 0.0384 | 4.22% | 99.24% |
| 3′ GG | 0.9945 | 0.0322 | 3.05% | 98.79% |
| 3′ UU | 0.9962 | 0.0327 | 3.81% | 99.27% |
| 5′ AA | 0.9962 | 0.0278 | 3.80% | 98.87% |

The correct conclusion is therefore: **gamma nearly exactly recovers the
gauge-identifiable individual bias shapes**. It does not recover—and is not
mathematically able to recover—the raw dataset-constant multiplier component.

## 12. What the remaining experiments will test

The outstanding matrix is scientifically useful because it separates several
effects that the current preliminary panel cannot:

1. **Read depth:** compare the same cumulative panel at 0.25, 2, and 20
   reads/codon. Shared and gamma recovery should generally improve as sampling
   noise decreases, but NB2 overdispersion prevents unlimited gains.
2. **Dataset count:** extend each depth to N=10. A plateau is expected once the
   shared component is already well identified.
3. **Same bias across depths:** the 20 cross-depth tasks test whether combining
   shallow, medium, and deep observations of one bias can identify shared
   biology.
4. **Gamma reference weighting:** equal versus quality-ranked cross-depth runs
   test whether making the deep observation the strongest reference improves
   the decomposition. This changes only gamma centering, not loss weights.

For a stronger causal claim about dataset count, future runs should also vary
the cumulative order and random seed. Otherwise a gain after adding a dataset
cannot be separated from the particular bias feature that was added.

## 13. Reproduction commands

Regenerate all processed synthetic datasets and weights:

```bash
.venv/bin/python Datasets/data/weight_synthetic_riboseq_codon_replicas.py \
  --depths 0p25_per_codon 2_per_codon 20_per_codon \
  --overwrite
```

Preview the complete local task matrix without training:

```bash
DRY_RUN=1 SKIP_GPU_CHECK=1 TASK_RANGE=0-46 \
  ./run_synthetic_bias_read_depth_local.sh
```

Run the complete matrix using two GPUs as independent experiment workers:

```bash
GPU_DEVICES=0,1 TASK_RANGE=0-46 \
  ./run_synthetic_bias_read_depth_local.sh
```

Run selected tasks, for example the first three 0.25-depth panels and the
first cross-depth pair:

```bash
GPU_DEVICES=0,1 TASKS="0 1 2 27 28" \
  ./run_synthetic_bias_read_depth_local.sh
```

Refresh both reports as new prediction artifacts arrive:

```bash
.venv/bin/python analyses/analyze_synthetic_recovery.py

.venv/bin/python analyses/analyze_synthetic_gamma_recovery.py
```

## 14. Interpretation limits and audit notes

- Current detailed evidence covers only one depth, one seed, and cumulative
  panels through N=5.
- N=6 and N=7 lack prediction parquets; only checkpoint-time aggregate
  synthetic scalars are used for those points. Gamma is not inferred from
  those scalars.
- The 0.25-depth resolved configs retain a legacy `synthetic_ground_truth`
  `observed_path` pointing to the 20-depth unbiased file. The recovery utility
  intentionally ignores those legacy observed-reference scalars and recomputes
  target and `mu` comparisons from each run's own prediction parquet.
- Predictions and reports use the maximum-`val_mu_pcc` checkpoint. The
  likelihood-best checkpoint remains available and may answer a different
  calibration question.
- Weight-reference statistics are currently fitted before the train/validation
  split. A final locked evaluation should fit and freeze them on training IDs.
- The local launcher's multi-GPU mode is independent task parallelism, not
  DDP. DDP-specific optimizer-window interpretations do not apply to these
  preliminary local runs.
- Raw gamma magnitude is not scientifically interpretable under the model's
  normalization. Always compare gauge-fixed log profiles.

## 15. Artifact index

- Experiment launcher:
  [`run_synthetic_bias_read_depth_local.sh`](../run_synthetic_bias_read_depth_local.sh)
- Synthetic configuration:
  [`config_ribounmix_synthetic.yaml`](../config/config_ribounmix_synthetic.yaml)
- Preprocessing:
  [`weight_synthetic_riboseq_codon_replicas.py`](../Datasets/data/weight_synthetic_riboseq_codon_replicas.py)
- Shared-signal analysis:
  [`analyze_synthetic_recovery.py`](../analyses/analyze_synthetic_recovery.py)
- Gamma-bias analysis:
  [`analyze_synthetic_gamma_recovery.py`](../analyses/analyze_synthetic_gamma_recovery.py)
- Machine-readable panel results:
  [`synthetic_recovery_by_panel.tsv`](../results/riboai_synthetic_experiments/recovery_report_1/synthetic_recovery_by_panel.tsv)
  and
  [`synthetic_gamma_recovery_by_panel.tsv`](../results/riboai_synthetic_experiments/recovery_report_1/synthetic_gamma_recovery_by_panel.tsv)
- Generated concise interpretations:
  [`README.md`](../results/riboai_synthetic_experiments/recovery_report_1/README.md)
  and
  [`GAMMA_RECOVERY.md`](../results/riboai_synthetic_experiments/recovery_report_1/GAMMA_RECOVERY.md)
