# Why Gamma Centering Could Help — and Why It Might Not

> **Historical note:** this reasoning document discusses earlier experimental
> reliability and reference modes that have now been removed. The active model
> uses checkpointed fixed-reference centering with configurable equal or
> quality-rank dataset weights. See
> [`gamma_centering_explained.md`](gamma_centering_explained.md) and
> [`current_configured_model.md`](current_configured_model.md) for the active
> implementation.

This is the *reasoning* companion to [`gamma_centering.md`](gamma_centering.md),
which documents the mechanism. Here the question is narrower and more practical:
**what problem is centering actually solving, what should improve if it works,
and what would explain it doing nothing?**

---

## TL;DR

Gamma centering exists to fix an **identifiability** problem, not a capacity
problem. The mean model has a mathematical degree of freedom that lets the shared
biology and the per-dataset correction trade the *same* signal back and forth
without changing the prediction. Centering removes one specific piece of that
freedom, forcing anything the datasets agree on to be explained by the shared
biological branch instead of leaking into the dataset-specific gamma.

- **It helps if** your disentanglement is failing because biological shape is
  hiding in gamma (the "flat-`L_bio`" failure).
- **It does nothing if** the leakage is going through a *different* channel (the
  additive background, the per-position dataset branch), if the datasets rarely co-observe
  the same transcript in a batch, or if the real limit is model/noise capacity
  rather than identifiability.

---

## Where gamma sits

The mean is

```text
mu[d, t, i] = S[d, t] * gate[d, t, i] *
              ( amplitude[d, t, i] * L_bio[t, i] + additive_bias[d, t, i] )
```

- `L_bio[t, i]` — the **shared** biological load. Comes *only* from the sequence
  branch (no dataset id, no dataset embedding). Mean-normalized to 1 over valid
  positions per transcript. This is the thing we actually care about recovering.
- `gamma = amplitude * gate` — a nonnegative, entmax-sparse **per-dataset,
  per-position** correction. Only the centered log-amplitude is pulled toward
  zero by `gamma_reg_weight`; a separately supervised support logit controls
  exact-zero support.
- `additive_bias[d, t, i]` — a **per-dataset, per-position** additive background,
  softplus, pulled toward 0 by an L1 penalty (`additive_bias_l1_weight: 1e-1`).
- `S[d, t]` — an oracle per-transcript scale (mean of the target), not learned.

The whole point of the architecture is that `L_bio` is *dataset-blind*, so it is
forced to carry only what is common across datasets, and gamma / additive carry
the dataset-specific observation bias. Centering is what makes that separation
actually hold.

---

## Why it *could* help: the gauge freedom

Look at just the multiplicative part, `gamma * L_bio`. Pick any positive field
`c[t, i]` and apply, for **every** dataset `d`:

```text
L_bio[t, i]      ->  c[t, i] * L_bio[t, i]
gamma[d, t, i]   ->  gamma[d, t, i] / c[t, i]
```

The product `gamma * L_bio` is unchanged, so `mu` is unchanged, so the loss is
unchanged. The model cannot tell these two configurations apart from the data.
That is a **gauge freedom**: the shape that all datasets *share* in gamma is
mathematically interchangeable with the shape in `L_bio`.

Concretely, this is the **flat-`L_bio` failure mode**: the sequence branch
collapses `L_bio` toward flat/uninformative, and every dataset's gamma quietly
grows the *same* peaks and valleys to reconstruct the profile. Predictions look
fine; the biological signal you wanted has migrated into a per-dataset head that
is supposed to hold only bias. (See the `disentanglement-mechanism-and-leakage`
project note — this is the central modeling risk.)

**What centering pins down.** For each transcript-position observed by ≥2
datasets, centering subtracts the reliability-weighted mean of `log_gamma` across
those datasets. With `strength=1.0` this constrains the cross-dataset **geometric
mean of gamma to 1** at that position. In gauge terms it fixes `c` so that gamma
has no common mode: *any shape shared across datasets can no longer be represented
in gamma and must be explained by `L_bio`.* That closes the leakage channel and
lets the sequence branch reclaim the shared signal.

---

## What should improve if it's working

If the flat-`L_bio` leak was really happening and centering closes it, you'd
expect to see, roughly in this order:

1. **Biology shape sharpens.** `L_bio` (equivalently `rho`, `lambda_bio`) stops
   being flat — `L_bio_max`, `lambda_bio_max`, and the `support_pcc` / biology-vs-
   target correlation that is logged (but not in the loss) go up.
2. **Gamma shrinks toward its job.** `gamma_raw` distributions tighten around 1
   and become more clearly *dataset-idiosyncratic* rather than tracking the
   profile. The `gamma_reg` term stops fighting the fit.
3. **`mu_val_pcc` holds or improves** while the decomposition gets cleaner —
   because the same prediction is now reached with the *correct* factorization,
   which generalizes better across transcripts than a gamma that memorized shape.
4. **Cross-dataset gradient conflict on the shared branch drops.** If both
   datasets are pushing `L_bio` toward the same (now-shared) signal instead of
   each carving its own shape via gamma, the per-dataset gradients on the
   biological params align better (watch `grad_conflict/full/biological/*`).

The honest success signal is #1 + #2 *together*: shape moves into `L_bio` and out
of gamma at roughly constant `mu` quality. If only `mu` moves, that's not
disentanglement, that's just refitting.

---

## Why it might NOT help

Five concrete reasons, most-likely first for this project.

### 1. It only closes *one* leakage channel; the additive branch is still open

Centering constrains the **multiplicative** common mode only. The additive term
`additive_bias[d, t, i]` is a full per-position, per-dataset head (fed by the
dataset-bias BiGRU)
and is **not** identified by centering — the mechanism doc says this explicitly.
If shared biological shape can be reconstructed as a common *additive* background
across datasets, it will simply leak there instead, and `L_bio` stays flat. The
only thing standing in the additive channel's way is the L1 penalty
(`additive_bias_l1_weight: 1e-1`) and its softplus floor — a much weaker
constraint than a hard gauge fix. **If centering "does nothing," suspect the
additive head first.** Test: tighten `additive_bias_l1_weight`, or restrict
`additive_bias_input_mode`, and see whether centering suddenly bites.

### 2. It only fires where two datasets co-observe the same transcript

Centering is `mode: batch_grouped` and needs ≥ `min_distinct_datasets` (=2)
distinct datasets, each clearing `min_reliability` (0.05) and a joint
`min_total_reliability` (0.5), *for the same transcript-position, in the same
batch*. If the batch doesn't group the transcript across datasets, or a transcript
lives in only one dataset, centering returns the raw gamma and marks
`applied=False` there. With only two datasets (kutay + grimson) and partial
transcript overlap, the fraction of positions that actually get centered can be
small — and centering a small fraction changes little. **Check
`gamma_cross_dataset_center_applied` / `gamma_num_distinct_datasets`:** if the
applied fraction is low, the sampler (`transcript_grouped_pairs`) and the
reliability thresholds, not the idea, are the bottleneck.

### 3. The shared-capacity ceiling is a different problem

Centering improves *identifiability*, not *capacity*. If the two datasets' true
biological signals are weakly correlated, a single shared `L_bio` cannot fit both
well no matter how cleanly gamma is centered — you are asking one shape to be two
shapes. This is the same wall CAGrad hits: it can *allocate* the shared branch
between datasets but cannot raise the ceiling (see the `cagrad-conflict` and
`kutay-sako-shared-biology-capacity` notes; re-check the current pair with
`analyse_cross_dataset_correlation.py`). Symptom: centering makes the
decomposition cleaner and gradient conflict lower, but `mu_val_pcc` on the harder
dataset still plateaus below its single-dataset value. That's not centering
failing — it's centering revealing that the shared-biology assumption is the
limit.

### 4. Gamma may already be near 1, so there's nothing to center

`gamma_reg_weight` pulls every centered gamma score toward 0 individually.
Centering pulls only the cross-dataset *common mode* toward 0. If the per-position
reg is strong relative to the fit pressure, gamma barely moves off 1 in the first
place and centering has almost nothing to subtract. The two are complementary but
compete for the same slack: strong `gamma_reg` + centering can be redundant, and
the interesting regime is *loose* per-position reg (so gamma is free to deviate
where a dataset genuinely differs) with centering doing the gauge fixing. If you
enable centering and see no change, try lowering `gamma_reg_weight` so gamma has
room to express dataset-specific deviations that centering can then discipline.

### 5. The real bottleneck is elsewhere

Centering can't help with problems it doesn't touch:
- **Oracle scale `S`.** Because `S` is `mean(target)`, `mu_val_pcc` is a
  shape-only, scale-invariant metric; several failure modes (dead queue, flat
  hazard) don't show up there at all. See the `oracle-scale-and-dead-J` note.
- **Data alignment.** The averaged-`ribo` +1-codon shift bug and 5'/stop-ramp
  underperformance are data/target issues; centering the bias head won't move
  them. (You're now training on `ribo_cds_replicas`, which sidesteps the shift —
  good, but that's orthogonal to gamma.)
- **Noise ceiling.** Replica reproducibility bounds how well *any* model can do;
  a dataset that is reproducible but not sequence-predictable won't improve just
  because gamma is disentangled (`noise-ceiling-vs-model-ceiling`).

---

## How to tell which case you're in

The diagnostics are already logged. Read them in this order:

| Question | Look at | If bad → |
| --- | --- | --- |
| Is centering even firing? | `gamma_cross_dataset_center_applied` (fraction), `gamma_num_distinct_datasets` | Reason **2** — sampler/coverage/thresholds |
| Did shape move into biology? | `L_bio_max`, `lambda_bio_max`, logged `support_pcc` | If not, reason **1** (additive leak) or **4** (gamma pinned) |
| Did gamma become dataset-specific? | `gamma_raw` spread, `gamma_center` magnitude | If gamma still tracks the profile → reason **1** |
| Is the mean fit stuck anyway? | `val_pcc_value` / `mu_val_pcc` per dataset | If clean decomposition but plateaued → reason **3** (capacity) |
| Is conflict gone but ceiling stays? | `grad_conflict/full/biological/*` | Confirms reason **3** |

The clean win looks like: **applied-fraction high → `L_bio` sharpens → gamma
tightens to ~1 common mode → `mu_val_pcc` steady-or-up → conflict down.** Any
other pattern points at one of the five reasons above.

---

## Historical knobs

The options below belonged to the removed batch-reliability implementation;
they are not accepted by the current configuration. The active selector is
`model.gamma_centering.mode`, and the fixed-reference equation and settings are
documented in `Docs/model_mathematics.html` and `Docs/gamma_centering.md`.

The former levers were:

- `strength` (1.0 = hard geometric-mean-to-1; lower = softer nudge).
- `min_distinct_datasets`, `min_reliability`, `min_total_reliability` — lower to
  increase coverage (reason 2), raise to trust only well-supported positions.
- `reliability_mode` (`combined` = depth × local-coverage × replicate-agreement,
  geometric mean) and its `*_kappa` / `tau` scales — how much to down-weight
  shallow or noisy positions when defining the center.
- Pair with `gamma_reg_weight` (reason 4) and `additive_bias_l1_weight`
  (reason 1) — centering is only as effective as the additive channel is closed
  and as gamma is allowed to move.
