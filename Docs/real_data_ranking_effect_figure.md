# Figure 2: ten-component reference-weight effects (C/D)

Run from the repository root, with the project Python environment activated:

```bash
python analyses/create_real_data_ranking_effect_figure.py
```

The script inventories real run manifests before opening predictions. It never
trains, changes reference buffers, chooses a result by performance, or compares
the representative subset experiment with a cumulative top-quality chain.

## Current local availability

The CPU audit finds **no compatible ranked matches**:

| Panel | Available uniform run | Required ranked counterparts |
| --- | --- | --- |
| C | `results/my_panels_a100_b32_20260906_114323` | `panel_01`–`panel_04`, with the identical 29/29/28/28 datasets, using the frozen ten-component reference |
| D | `results/my_exp8_a100_b32_20260906_114340` | At each N=2,5,10,20,40: `pair01_A`, `pair01_B`, `pair02_A`, `pair02_B`, `pair03_A`, `pair03_B` |

These are **4 + 30 missing ranked matches**, not an authorization to train.
`missing_runs.csv` lists every exact collection, ordered membership, seed,
split hashes and reliability-reference hash. The existing four-panel ranked
run `my_panels_qrank_a100_b32_20260908_103510` used **six components**. The
completed ten-component Exp8 run uses **cumulative top-quality subsets**.
Neither supplies the requested contrast, even though checkpoints exist.

Historical equal-run code/environment provenance is also unverified. If it
cannot be recovered, fresh matched equal controls would be needed; merely
downloading ranked outputs does not prove implementation matching.

The script returns **exit status 2** for these scientific blockers. It writes
the report, exact missing-run table, required reference-weight specification,
LaTeX caption and header-only unavailable metric tables. It does **not** create
an article PDF/PNG or fill in effects/intervals. Design weights are explicitly
labelled as requirements, not as evidence of ranked training.

## Supplying compatible results

When matching runs are available, specify their experiment roots explicitly:

```bash
python analyses/create_real_data_ranking_effect_figure.py \
  --panel-equal-root /shared/matched_panels/equal \
  --panel-ranked-root /shared/matched_panels/qrank10components_p1 \
  --stability-equal-root /shared/matched_representative_subsets/equal \
  --stability-ranked-root /shared/matched_representative_subsets/qrank10components_p1 \
  --ranking-table Datasets/data/HEK_riboseq_profile_quality_rank_components.tsv \
  --reference-figure figures/real_data_equal.pdf \
  --output-dir figures
```

Those `/shared/...` paths are illustrative inputs, not existing runs.

The workflow accepts the existing production panel/Exp8 manifest layouts, raw
`L_bio` parquet exports and `prediction_checkpoint_manifest.json`. It checks
the selected `best_val_loss` checkpoint, its reference state, the runtime
reference manifest, full-panel weights, resolved configurations, ordered split
lists and numerical training-only `w_dt` references. Raw repeated dataset
rows must have identical shared outputs. Full-CDS lengths are checked against
the frozen sequence artifact; mismatches are excluded, not truncated.

Training-linked source/environment evidence follows the repository's
`run_real_panels_qrank10_matched.py` contract: `matched_experiment.json`,
`execution_identity`, and hashes of immutable launch/split/reference inputs.
Do not fabricate such a manifest retrospectively from current code. If a
different genuine provenance format exists, extend the reader for that actual
format; there is no `--ignore-matching` switch. Data/sequence content hashes
computed at analysis time are distinguished from historical file signatures.

No suitable inference-only task was found locally. Inference from a six-component
or cumulative checkpoint cannot repair the missing experiment. If valid matched
checkpoints exist elsewhere without exports, use their frozen production
prediction configuration before invoking this script; do not replace training
reference weights or use a different checkpoint-selection criterion.

## Descriptive adjacent-size panel D

The completed ten-component cumulative experiment and Figure 1's verified
uniform source records support a descriptive overlay of adjacent-size agreement
at 2:5, 5:10, 10:20, 20:40, 40:80 and 80:114 datasets. Generate it with:

```bash
python analyses/create_real_data_ranked_cumulative_panel_d.py
```

This writes `figures/real_data_ranked_cumulative_panel_d.{pdf,png}`, a LaTeX
caption, combined per-transcript records, joint bootstrap summaries, a
descriptive-difference table, prediction provenance, and every active ranked
q/pi value. Ranked PCCs are recomputed from saved arrays; the uniform records
are loaded from `figures/real_data_equal_source` and checked against its saved
pair and transition summaries. The uniform curve averages 36, 36, 36, 36, 18
and 3 representative cross-size model pairs; the ranked curve has one nested
top-quality model pair per transition. Their difference is saved for
transparency but deliberately not plotted as a reference-weight effect because
membership, overlap and subset construction differ. Neither curve is a causal
estimate of adding datasets; all models were fitted independently.

## Fixed ranking and statistics

The manuscript ranking checksum is:

```
5811cadf68c56740e83b232b2990299d527630326205db9cba8bc7024d3cf1f8
```

It has 115 rows, global maximum rank **R=115**, and these ten component ranks:
CDS enrichment, depth, periodicity, replicate agreement, RPF-length center,
RPF-length spread, r/tRNA contamination, STAR total mapping, STAR unique mapping,
and transcript support. Rank 1 is best. The production parser supplies
`q=(R-r+1)/R`; ranked pi is q normalized within the selected collection.
The complete ranking is never filtered before computing R. Its historical
transcript scope is not verified; train-only `w_dt` does not imply train-only QC.

- C: `median(PCC_ranked) - median(PCC_equal)` per panel pair, on matched finite
  transcripts. This is **not** `median(PCC_ranked - PCC_equal)`.
- D: per policy, mean transcript PCC for each designated pair, then mean over
  the three pairs. Take ranked minus equal. Use the same finite transcript
  cohort across both policies, all N and all pairs.
- Both: 5,000 paired transcript-cluster resamples, `default_rng(20260910)`;
  linear 2.5/97.5 percentile endpoints. Each sampled transcript carries both
  policies and every comparison. Bootstrap index-stream hashes are recorded.

Intervals are conditional on these fitted models and selected collections,
not independent experiment replications. Positive differences mean increased
agreement, not biological accuracy. Constant, nonfinite, incomplete or
misaligned profiles remain excluded with reasons, never zero-imputed.

## Outputs and rendering

When every required match passes, outputs are:

- `figures/real_data_ranking_effect.pdf`: native vector C/D figure.
- `figures/real_data_ranking_effect.png`: 600-dpi preview.
- `figures/real_data_ranking_effect.tex`: caption and figure inclusion.
- `figures/real_data_ranking_effect_source/`: paired records, C/D summary
  tables, three pair-specific D effects per N, exclusions, matching checks,
  ranking/weight provenance, Figure 1 cohort differences and regeneration command.

Summary statistics are recomputed from round-tripped paired CSVs. Plot artists
are checked against saved summary tables, including all CI endpoints and
negative effects. Categorical C rows are not connected; D alone has a line.

The reference PDF page size is used when available, with the typography of
`create_real_data_equal_figure.py`. Pair order is P1–P2, P1–P3, P1–P4, P2–P3,
P2–P4, P3–P4. Figure 1 currently includes auxiliary N=80/114 diagnostics;
Figure 2 explicitly omits them, retaining log ticks 2,5,10,20,40.
Use `--no-tex` only for an explicitly recorded built-in-font rendering, not
as a claim of identical LaTeX glyphs.

## Verification

```bash
python -m py_compile analyses/create_real_data_ranking_effect_figure.py \
  Tests/test_real_data_ranking_effect.py
python analyses/create_real_data_ranking_effect_figure.py --help
python -m unittest Tests.test_real_data_ranking_effect \
  Tests.test_compare_real_panel_weighting -v
```

Tests cover the complete-table rank conversion, missing/duplicate ranks,
incompatible experiments, source atomicity, configuration/provenance failures,
the two estimands, shared bootstrap draws, the D common cohort, alignment,
invalid PCC handling, vector plot/table equality, and preservation of inputs
when results are unavailable. Test-only rendering fixtures are not article
results and are never written under the manuscript figure name.

Executed locally on 2026-09-12: syntax checks, `--help`, the CPU audit (expected
exit 2 for missing matches), and **21 passing tests** including the existing
matched-panel comparison tests. A real selected-checkpoint/raw-array check
passed for equal `panel_01` (1,593 profiles, checkpoint epoch 44) and equal
`N002/pair01_A` (1,771 profiles, epoch 7), with no invalid profiles in either
checked export. The production checkpoint stores **raw** q (ones for equal);
normalization into pi occurs in centering, not in that saved buffer.

A clearly watermarked test-only LaTeX rendering was visually inspected at the
reference page size (523.2 x 199.2 points). Its PDF contains vector artists and
embedded Latin Modern fonts, with no raster images; its preview was exported at
600 dpi. No article Figure 2 PDF/PNG was generated because no ranked comparison
passed the availability gate. No GPU inference or training was run.
