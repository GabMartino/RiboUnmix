# Synthetic read depth, cumulative panels, and π: audit

Date: 2026-09-08. This audit analyzes existing results and runs deterministic CPU diagnostics; it does not change training or launch experiments.

The requested report is actually under `analyses/artifacts/synthetic/inter_shared_signal`, without the `Utils/` prefix. The latter contains an empty recovery-report directory in this checkout.

## What the completed runs establish

The [original report](../../analyses/artifacts/synthetic/inter_shared_signal/INTER_SHARED_SIGNAL_RECOVERY.md) contains 18 available prediction runs and eight comparable equal/quality-rank pairs out of ten configured panels. These use seed 42, best-validation-loss checkpoints, fixed-reference centering, and geometric-mean-one gamma. Available runs use the same 1,920 validation transcripts, hash `b5e7b4a1ec52`. PCC is averaged across transcripts on the CDS interior, excluding five codons at each end; RMSE is the square root of mean transcript MSE after whole-profile mean-one normalization.

| Datasets | Bias families | Equal π PCC | Ranked π PCC | Equal π RMSE | Ranked π RMSE |
|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 0.7424 | 0.7424 | 0.4693 | 0.4689 |
| 6 | 2 | 0.7909 | 0.7863 | 0.4123 | 0.4046 |
| 9 | 3 | 0.8839 | 0.8846 | 0.2693 | 0.2677 |
| 12 | 4 | 0.9188 | 0.9174 | 0.2182 | 0.2198 |
| 15 | 5 | 0.9309 | 0.9314 | 0.2083 | 0.2030 |
| 24 | 8 | 0.9585 | 0.9598 | 0.1537 | 0.1519 |
| 27 | 9 | 0.9624 | 0.9601 | 0.1494 | 0.1561 |
| 30 | 10 | 0.9643 | 0.9655 | 0.1442 | 0.1437 |

The mean paired PCC change is −0.000576, with mixed signs. This is descriptive, not a significance test: panel sizes are nested, not independent replicates. N=18 lacks equal-policy predictions and N=21 lacks ranked-policy predictions. Cumulative recovery improves substantially; the ranking comparison has no consistent benefit. Adding datasets also adds bias families, so these runs do not isolate dataset count from bias diversity.

## Why the current design cancels the systematic effect of π

The raw Parquet metadata confirms that each programmed bias is systematic across samples and depths. For a family j, its physical multiplier b_j(i) is identical at 0.25, 2 and 20 reads/codon. The original launcher adds complete triplets and assigns depth ranks 3, 2, 1. With power one, their weights are proportional to 1, 2, 3.

For J complete families, the reference log-bias center is

    cπ(i) = Σ_j Σ_depth w_depth log b_j(i) / (J Σ_depth w_depth)
          = (1/J) Σ_j log b_j(i).

It is identical under equal and depth-ranked weights. Any depth-only power retains this cancellation, including concentrating on the highest depth. Depth-dependent estimation errors need not cancel, so learned predictions need not match exactly.

The implementation agrees with this interpretation: [RiboUnmixModel.py](../../Models/RiboUnmixModel/RiboUnmixModel.py) constructs reference weights near line 262 and subtracts their weighted log-score center near line 984. The [Lightning sample reduction](../../Models/RiboUnmixLightningModule.py) explicitly excludes dataset-quality ranks from its API near line 170. π changes the factorization constraint; it is not a direct multiplier on each dataset's NB/PCC loss. Transcript reliability weights are a separate mechanism.

Even with perfect knowledge of biased profile shapes, the compatible shared shape is

    Lπ(i) ∝ K(i) exp(cπ(i)), normalized to mean one.

Subtracting the positional mean of each log bias changes only the normalization constant here. Thus a common nonconstant bias remains in Lπ. With one family at three depths, all three observations share the same bias: more reads cannot distinguish it from biology under this reference constraint. With additional families, averaging their different spatial patterns can reduce the distortion. This is a shape-factorization diagnostic, not an assertion of the exact finite-data training optimum or a universal performance ceiling; the mass-free count objective, model capacity and regularization can introduce other compromises.

## Independent checks using the actual synthetic files

[oracle_check.py](oracle_check.py) reads the first 128 kinetic-truth transcripts in file order and their programmed bias annotations. IDs are saved in [oracle_transcript_ids.tsv](oracle_transcript_ids.tsv). This is a deterministic illustrative subset, not the training report's validation set. It applies whole-profile normalization before five-codon trimming, matching the report's metric convention.

For every original complete-triplet panel, equal and ranked oracle profiles match to at most **2.67e−15**. Their oracle PCC rises from 0.7555 at one family to 0.9687 at ten families. The qualitative similarity to the trained curve supports the shared-bias explanation; differing transcript sets prevent a direct residual-error estimate.

Read depth does change observation precision. A separate check averages the two actual `3prime_aa` replicate count profiles, divides by their known physical bias, normalizes, and compares with K:

| Reads/codon | Corrected observation PCC | Corrected observation RMSE |
|---:|---:|---:|
| 0.25 | 0.3317 | 1.4184 |
| 2 | 0.6635 | 0.5687 |
| 20 | 0.8345 | 0.3301 |

These are observation-level controls, not trained-model scores. Higher depth helps, but it cannot remove the shared systematic bias by itself. NB2 metadata specifies variance μ + 0.1 μ², so relative variance is 1/μ + 0.1: even very high depth has a dispersion floor. A sequence model pooling information across many transcripts can also denoise low-depth observations, reducing the difference between trained recovery curves.

## Would changing cumulation solve it?

The repository already contains [run_synthetic_bias_read_depth_cumulative_orders_local.sh](../../run_synthetic_bias_read_depth_cumulative_orders_local.sh). Its Latin orders assign different depths to different bias families, rotating assignments three ways. Early panels have incomplete families, so π changes each family's total contribution. All orders reunite the same 30 datasets at the final panel, where cancellation returns. No corresponding `orderlatin` result directories were found in this checkout.

However, breaking cancellation does not guarantee better K recovery. For the first three families, the noiseless oracle diagnostic gives:

| Assignment | Equal π PCC | Ranked π PCC |
|---|---:|---:|
| latin0 | 0.8940 | 0.8651 |
| latin1 | 0.8940 | 0.8151 |
| latin2 | 0.8940 | 0.9109 |

Higher depth estimates its assigned bias more accurately, but the bias need not be weaker or more compatible with K. Unequal weighting can amplify an unfavorable pattern. These numbers diagnose the proposed design before training; they are not predictions of exact trained performance.

## Proposed experiments and what each would test

1. **Isolate read depth.** Keep bias identities, reference weights, transcript split and training budget fixed; vary depth alone. Use independent observation seeds and multiple model seeds. Include an unbiased or known-bias-corrected observation control and compare learned L with both K and the compatible Lπ. This separates finite-count estimation error from systematic shared-bias distortion. Evaluate identical transcript IDs across depths; a common RNG seed alone does not guarantee identical stratified splits.
2. **Test sensitivity to π with the existing files.** Use the Latin rotations, paired equal/ranked/reversed rankings, and complete-triplet controls. Preselect a small set of panel sizes, such as 3, 9 and 30, rather than launching all 60 runs immediately. Report every rotation, including cases where weighting worsens K recovery. The current launcher supplies equal/ranked only; reversed ranking would require a deliberate extension.
3. **Test when quality ranking should help.** Build a controlled factorial scenario with depth and systematic-bias severity varied separately. Include a positive-control condition where high-depth datasets also have weaker distortion, a condition where severity is independent of depth, and a reversed alignment. The aligned condition tests the explicit assumption that depth is a useful proxy for biological fidelity; it cannot establish that depth alone causes the gain. Keep transcript content, bias family identities and dataset counts matched when swapping assignments.

For a general statistical-precision claim about π alone, use independent zero-mean dataset-specific log-bias perturbations with variance tied to the quality mechanism, enough dataset draws, and multiple seeds. Avoid the exact same family×depth rectangle. Depth-derived π is justified as a reference-quality proxy only to the extent that quality predicts reference distortion or estimation uncertainty; it is not universally an inverse-variance optimal weight. Assess estimation against the compatible Lπ as well as raw K so a change of reference convention is not mistaken for improved learning.

Checkpoint selection should remain based on observed validation data, not K. Fix a common split, report equal-transcript interior PCC and RMSE, and distinguish uncertainty across training/data seeds from uncertainty across validation transcripts. Do not select a favorable cumulative order after inspecting K.

## Reproduction and artifacts

From repository root:

```bash
python Docs/synthetic_depth_pi_audit_2026-09-08/oracle_check.py
```

- [Observed paired metrics](observed_paired_results.tsv), extracted from the existing report.
- [All deterministic gauge diagnostics](oracle_gauge_results.tsv).
- [Known-bias-corrected depth control](oracle_bias_corrected_depth_results.tsv).

No production model, launcher, dataset or existing result artifact was modified.
