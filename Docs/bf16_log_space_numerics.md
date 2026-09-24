# Model/loss numerical audit and log-space rewrite

Version: `log-space-nb2-v1` (2026-09-10).

**Recovery guidance update:** the later N040/pair02_A report still has
non-finite GRU gradients at lengths up to 5089. Do not treat the loss rewrite
as evidence that the FP32 bias-GRU protection can safely be removed. For
affected runs, retain `trainer.precision=bf16-mixed` and explicitly select
`BIAS_GRU_PRECISION=float32` while diagnosing the recurrent backward path.
If that setting was already active, collect an anomaly trace rather than
claiming the same precision override is a new fix. See the N040 section in
`Docs/exp8_remaining_aggressive.md` for the exact single-task command.

## Outcome and limits

The positive-profile algebra and NB2 likelihood now have a log-space path.
The neural networks use the trainer's autocast policy unless the explicit
`context_gru_precision=float32` bias-branch protection is enabled. The
quality-ranked launcher currently defaults to `inherit`, so the protection
must be requested explicitly for affected quality-ranked runs. Scalar normalization, probability and
correlation calculations use FP32 under AMP. **This is BF16 mixed precision,
not a full-FP32 model.** Embeddings/master parameters and sensitive scalar
operations do not have to be BF16 for matrix/recurrent operations to use AMP.

The supplied failure traces locate non-finite accumulated gradients in the
dataset-bias GRU and its embeddings. They do not locate the first non-finite
autograd operation. The saved N80 manifest did not request the FP32 GRU option.
No causal proof ties the cluster failures to any one formula below.

No log rewrite guarantees finite results for every finite input, weight,
sequence length or training trajectory. A GRU has signed hidden states and a
product of recurrent Jacobians; replacing positive outputs by their logs does
not bound that product. This machine has no usable CUDA device. The changes
are numerically tested, but **resolution of the exact UNIVIE failure remains
unverified**. Do not interpret a skipped CUDA test as a passing CUDA test.

## Preserved statistical model

For a transcript t of valid length T, dataset d, replica r, and codon i:

\[
 S_{dtr}=\max\!\left(\epsilon,\frac{1}{T}\sum_i y_{dtri}\right),
 \qquad
 \ell_{ti}=\log\operatorname{softplus}(h_{ti})
 -\left(\operatorname{LSE}_j\log\operatorname{softplus}(h_{tj})-\log T\right).
\]

Then \(L_{ti}=\exp(\ell_{ti})\) is mean-one. The existing biological
GRU, head parameters, dropout and checkpoint tensor names are unchanged.
The implementation evaluates log-softplus directly in its negative tail.
It does not form an underflowed Softplus and then take its logarithm.

Let a be the existing bounded raw dataset log-score and pi the fixed-panel
weights (equal or quality-ranked, unchanged). With both existing gauge
constraints enabled:

\[
 g_{dti}=a_{dti}-\sum_k\pi_k a_{kti}
 -\left(\frac{1}{T}\sum_j a_{dtj}
        -\sum_k\pi_k\frac{1}{T}\sum_j a_{ktj}\right).
\]

The final bias is \(\gamma_{dti}=\exp(g_{dti})\). Neither reference membership,
rank weights, rank power, dropout/reference evaluation nor the centering
gradients have been changed. For the saved score bound 8, the two gauges can
produce |g| up to 32, **not just 8 or 16**. This is below BF16/FP32's exponential
overflow range; the corresponding variance products are not necessarily safe.

For the saved mass-free experiment:

\[
 q_{dti}=g_{dti}+\ell_{ti},\qquad
 \eta_{dtri}=\log S_{dtr}+q_{dti}.
\]

In mass-conserving mode only, subtract
\(\operatorname{LSE}_i(q_{dti})-\log T\) from q. The mean is
\(\mu_{dtri}=\exp(\eta_{dtri})\). Replica NB now consumes eta directly from
`extras.log_normalized_shape`, rather than multiplying S, gamma and L and
then taking a log. Linear-scale outputs remain for raw PCC and predictions;
an unrepresentable observation mean raises instead of being replaced by the
largest float. This is a deliberate limit of retaining a linear-output API.

The old denominators `max(mean(softplus), eps)` and, for mass conservation,
`max(mean(shape), eps)` broke the intended mean-one normalization when all
factors were extremely small. Log normalization removes these artificial
floors. This is equivalent in the ordinary regime and restores the intended
model in the underflow regime; it is not bitwise-equivalent to the old fallback.
The existing target-scale and NB mean floors remain.

## NB2 likelihood

With the same configured clipped log-dispersion A, define
\(k=\exp(-A)\), \(z=\max(\eta,\log\epsilon)+A\). The implemented NLL is:

\[
 \mathcal{L}_{NB}=
 \log\Gamma(k)+\log\Gamma(y+1)-\log\Gamma(y+k)
 +k\operatorname{softplus}(z)+y\operatorname{softplus}(-z).
\]

This retains the target-only gamma normalization and the continuous extension
for non-integer replica-averaged targets. It avoids `exp(eta)`, `alpha*mu`,
and cancellation between `(k+y)*log(k+mu)` and the separate log-mean terms.
Above the unchanged mean floor:

\[
 \frac{\partial\mathcal{L}_{NB}}{\partial\eta}
 =k\sigma(z)-y\sigma(-z)=\frac{\mu-y}{1+\alpha\mu},
 \qquad -y\leq\partial_\eta\mathcal{L}_{NB}\leq k.
\]

Large-target gamma differences use a Stirling difference expressed with
`log1p((k-1)/(y+1))`, including corrections through the inverse seventh power,
for y >= 10000. At this threshold and the experiment's bounded dispersion the
truncation error is below FP64 precision; FP32 rounding still applies. This
avoids losing the entire gamma ratio by subtracting two large `lgamma` values.
Moderate targets use the direct `lgamma` difference. Tests compare values and
both mean/dispersion gradients with an independent FP64 formula.

The existing standard, mean-gradient-reweighted, decoupled and fixed-alpha
modes remain. Detached reweighting uses log-mean directly. Its uncapped
exponential weight can itself be mathematically too large in other experiment
modes; this still raises. The reported Exp8 mode has beta=0 and does not use it.
Invalid predicted means, scores, dispersion and likelihood values are no
longer repaired with `nan_to_num` on the NB training path. Padding/missing
target masks are applied before nonlinear operations; their semantics remain.

## PCC, variance stabilization, reductions and optimization

The optimized PCC still uses
\(\mathrm{cov}(x,y)/\sqrt{(v_x+\epsilon)(v_y+\epsilon)}\).
Its implementation centers after subtracting a reference value, rescales the
variance calculation, obtains standard deviations in log space, and averages
the standardized products. It never squares raw amplitudes or multiplies
variances. The epsilon regularization and target-variance eligibility remain.
The reporting-only PCC retains its separate historical denominator convention
and is detached. Unrepresentable FP32 target variances are reported in FP64;
this is a detached per-profile diagnostic, not FP64 model training.

The NB variance-stabilizing transform remains
\(2/\sqrt{\alpha+\epsilon}\,\operatorname{asinh}\sqrt{\alpha x+\epsilon}\).
For \(u=\log\alpha+\log(x+\epsilon/\alpha)\), use
\(\operatorname{logaddexp}(u/2,\operatorname{softplus}(u)/2)\) for its asinh
factor. Thus alpha*x need not be representable, and x=0 has a finite derivative.
The existing detached-alpha routing remains unchanged.

Means divide by the count before summing. Global L2 clipping in the manual
BF16/FP32 execution path scales by the largest gradient magnitude before
reducing squares. It reproduces the usual clipping coefficient while avoiding
an infinite norm from finite gradients. The existing NaN/Inf gradient guard
still runs **before** clipping; no batches or transcript groups are skipped.
FP16-with-GradScaler retains Lightning's existing clipping path.

The objective coefficients, replica averaging, reliability weighting,
transcript-balanced reduction, logical batch size, optimizer groups,
learning rates, complete-transcript grouping and alpha-head detachment remain
unchanged. AdamW and neural kernels retain their normal finite-precision
limits. Logs cannot make signed recurrence, optimizer state or arbitrary
diagnostic exponentials overflow-proof; the exact saved Exp8 bounds and
finite-gradient checks remain important.

## Verification and checkpoint compatibility

Local result: 117 passing unittest cases and one skipped CUDA-only case in
the 118-case regression run, plus five passing execution-microbatch function
tests (122 passed in total). Shell syntax and tracked diff-whitespace checks
also passed. No cluster training job was submitted.

The numerical suite covers FP64 forward/gradient equivalence and gradcheck,
log-means up to 1000, targets up to 1e20, PCC amplitudes up to 1e30 including
flat profiles, VST inputs up to 1e38, underflowing biological factors, masks,
finite-gradient clipping at 1e30, and CPU BF16-autocast backward/AdamW updates
on a small synthetic model at length 3175. CPU AMP is not a CUDA kernel replay.

The locally copied N80 and N114 `last.ckpt` files (both epoch 0, global step
417) contain finite model-state tensors. Both loaded into the updated model
with `strict=True`, with every state key matching. These are pre-failure
checkpoints, not the exact later failing weights/batch/dropout realization.
No original results, predictions, saved configs or checkpoints were modified.

`numerical_formulation_version=log-space-nb2-v1` is recorded in new resume
manifests and Lightning checkpoint hyperparameters. Resume optimizer state
remains available, but changed numerical arithmetic changes training
trajectories. For a strict numerical-method comparison, use separate runs;
do not relabel previously completed results as generated with this rewrite.

## Run and validate on UNIVIE

Sync the changed project files, especially:

- `Models/utils/stable_numerics.py` (new, required);
- `Models/RiboUnmixModel/SharedProfileModel.py`;
- `Models/RiboUnmixModel/RiboUnmixModel.py`;
- `Models/RiboUnmixModel/DatasetBiasSubmodel.py`;
- `Models/RiboUnmixLightningModule.py`;
- `resume_real_experiment_from_checkpoints.py`;
- `run_real_exp8_L_stability_quality_rank.slurm`;
- `Tests/test_stable_training_numerics.py` and its existing test dependencies.

Inside a **one-GPU allocation** with the project virtual environment active:

```bash
python -m unittest Tests.test_stable_training_numerics
```

The CUDA long-sequence test must run rather than skip. It tests a small
synthetic model at lengths 1807, 3175 and 5089 under both bias-GRU policies, not the full failed
production batch. After that check, resume only N114/N80 using the unchanged
existing output-root/run-ID settings:

```bash
sbatch --array=0-1%2 \
  --export=ALL,EXP8_RUNTIME_PROFILE=auto,BIAS_GRU_PRECISION=float32 \
  run_real_exp8_L_stability_quality_rank_univie.slurm
```

This explicitly retains the FP32 bias-GRU protection. The saved trainer
precision remains `bf16-mixed`. Prefer the auto runtime budget while validating
the numerical change; there is no evidence yet of the optimal throughput.
If a failure recurs, `DETECT_ANOMALY=1` is now forwarded by this launcher to
the resume helper. It is slow and intended to identify the first autograd
operation, not as a speed setting. Keep the resulting traceback and effective
resume manifest. A recurrence inside the GRU would require investigating the
recurrent dynamics/kernel, not claiming that more logarithms can solve it.

Background: [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
explains why intermediate reductions can overflow when the final result fits;
[PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html) documents mixed
operation dtypes. The implementation above is checked against this project's
installed code and tests, not assumed from the newest online release.
