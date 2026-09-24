# Four-panel global-QC and gamma-reference audit

`audit_four_panel_reference_quality.py` defaults to a CPU-only audit. It disables
CUDA visibility before importing the production ranking loader. It never loads
prediction arrays, checkpoints, validation-performance logs, or test-performance
tables, and never launches training in any mode.

## Inputs and interpretation

The default panel manifest is the historical equal design under
`results/my_panels_a100_b32_20260906_114323/`. Its dataset order, retained universe
and saved source-family identifiers are authoritative. The common split, observed
QC table, panel configurations (runtime Hydra configuration preferred) and
train-only reliability references are discovered at explicit, documented paths
below that root; they can all be overridden with CLI arguments. There is no
fuzzy author/year matching or automatic ranking-table substitution.

The default ranking is **the six-component**
`Datasets/data/HEK_riboseq_profile_quality_rank.tsv`. It contains periodicity,
CDS enrichment, depth, detected-transcript support, RPF-length center and
RPF-length spread ranks, plus their raw QC summaries. To audit a ten-component
definition, explicitly pass its path with `--ranking-table`; it is a different
input definition, not a replacement for historical six-component weights.

The production `load_dataset_quality_ranking` computes `q=(R-r+1)/R`, with R
equal to the maximum rank in the **complete input table**. Panel weights are
`pi_equal=1/N` and `pi_ranked=q/sum_panel(q)`, with power fixed to one. Global
ranks are never recalculated after retention, per panel, or for plotting.
Pi is used only in gamma centering, not as a multiplier on local loss weights
`w_dt`. QC is not biological truth.

Global quality groups use the linear empirical 25th/50th/75th rank quantiles
over all retained datasets. A value exactly on a boundary goes to the lower
numbered group; equal ranks stay together, so groups need not have equal sizes.
The same boundaries apply to every panel and both policies. All plot source
tables are saved, including deterministic scatter jitter coordinates.

Component mean differences use the full retained collection's observed-value
population SD (`ddof=0`) as a fixed denominator. Unweighted targets use uniform
dataset weights; ranked targets use the same frozen q mapping. Available-value
means renormalize over observed entries, while missing counts and missing
reference mass remain explicit. No raw QC is reconstructed from ranks.
ECDF distances and Wasserstein distances are descriptive, without p-values.

N_ref is `1/sum(pi²)`, not a number of independent experiments. Source masses
sum dataset weights inside the frozen source families; no equal-per-source
policy is introduced. There is no automatic “balanced” threshold. An optional
`--source-mass-warning` is a documented descriptive heuristic only and is never
used to select a partition.

Historical global-ranking input data, transcript scope and freeze chronology
are reported as **not verified** unless supported by available provenance.
A matching historical ranking hash establishes content identity, not that its
transcript-derived QC was train-only. Train-only `w_dt` provenance is reported
separately. A supplied `--ranking-provenance` JSON is labeled as documentary
claims, not an independent verification.

## CPU audit command

From the repository root in the project environment:

```bash
python analyses/audit_four_panel_reference_quality.py \
  --mode audit \
  --panel-manifest results/my_panels_a100_b32_20260906_114323/panel_manifest.json \
  --ranking-table Datasets/data/HEK_riboseq_profile_quality_rank.tsv \
  --historical-ranked-root results/my_panels_qrank_a100_b32_20260908_103510 \
  --reference-weights results/panels_equal_vs_ranked_comparison/reference_weights.csv \
  --output-root results/four_panel_reference_quality_audit
```

Only `reference_weights.csv` is read from the comparison directory; no model
metrics are read. An explicit saved weight table must cover the retained
universe for each policy it contains. Duplicate policy/panel/dataset rows fail
by default; `--duplicate-weight-policy collapse-identical` permits exact duplicate
values only and records how many rows were duplicated. Conflicting duplicates,
missing ranks, ambiguous canonical IDs, incorrect membership and split source
families block ranked preparation. Diagnostics are written to the audit report
and manifest; no missing ranks are imputed.

Use a **new** output root for each invocation; the tool refuses nonempty output
directories and paths inside historical input roots. It hashes inputs before
and after the audit. LaTeX/Latin Modern is the default figure style; `--no-tex`
uses Matplotlib serif fonts without changing the numbers. SVG/PDF exports are
native vector, including heatmap cells. PNGs are inspection previews only.

## Exactly two optional preparation designs

Both modes run the audit first. Neither trains, even without `--dry-run`.
Preparation uses the production model entrypoint, external split loader,
ranking conversion, source-group assignment utilities, reliability fitter and
matched configuration/analysis contract. It introduces no distributed trainer.

### 1. Existing partition, ranked-reference retraining

```bash
python analyses/audit_four_panel_reference_quality.py \
  --mode prepare-existing \
  --panel-manifest results/my_panels_a100_b32_20260906_114323/panel_manifest.json \
  --ranking-table Datasets/data/HEK_riboseq_profile_quality_rank.tsv \
  --output-root results/reference_quality_prepare_existing_v2 \
  --gpus inherit --dry-run
```

Preserves dataset order and membership, source identities, training/validation/
test lists, saved numerical `w_dt` reference values, architecture, objective,
optimizer, scheduler and saved training seed. Data paths are relocated by exact
repository suffix or explicit overrides, not by basename. Current data are
hashed. The global ranking is copied in full before commands are written.
An explicit alias map is recorded; if necessary, a canonical-ID copy of the
complete ranking is frozen alongside its original bytes. Explicit nonstandard
ID/rank column names are normalized to `dataset`/`quality_rank` in this copy
for the existing matched analysis; the transformation is recorded and rank
values and the complete global universe are unchanged. Dataset byte sizes are
compared with historical QC provenance where available; a known mismatch blocks
preparation. Size agreement alone does not establish historical byte identity.

The historical source/environment/data identity is not fully established in
the available artifacts. Therefore strict reuse of historical equal controls
is **unverified**, and this implementation conservatively prepares four fresh
equal controls as well as four fresh ranked models: **eight trainings**.
It never changes pi in a trained checkpoint or resumes an old ranked model.

This changes reference policy within the original panels. It does **not**
repair between-panel QC imbalance. A different `--training-seed` is rejected
for this preservation mode rather than silently changing the experiment.

### 2. New global-rank-balanced partition

```bash
python analyses/audit_four_panel_reference_quality.py \
  --mode prepare-rank-balanced \
  --panel-manifest results/my_panels_a100_b32_20260906_114323/panel_manifest.json \
  --ranking-table Datasets/data/HEK_riboseq_profile_quality_rank.tsv \
  --partition-seed 42 --training-seed 42 \
  --output-root results/reference_quality_prepare_rank_balanced_v2 \
  --gpus inherit --dry-run
```

The search configuration is written **before candidate evaluation**. Defaults:
four starts, 2,000 seeded source-swap proposals per start. The existing partition
is an explicit baseline; other starts reuse the existing source-atomic greedy
assignment. Only equal-size intact-family swaps are used in refinement, retaining
exact capacities and every dataset. This is a limited search, not a global
optimum or an infeasibility proof.

The score is a weighted sum of mean squared panel deviations in these blocks:

| Block | Weight | Full-collection target |
| --- | ---: | --- |
| Original observed QC variables | 1 | Uniform dataset weights |
| Global standardized rank and ECDF at seven frozen eighth-quantile cutoffs | 1 | Uniform dataset weights |
| Four fixed quality-group proportions | 1 | Uniform dataset weights |
| Available component ranks | 1 | Uniform dataset weights |
| Ranked reference mass in the four quality groups | 1 | Frozen q weights |
| Ranked-reference component means | 0.5 | Frozen q weights |

Numeric feature scales are the full retained observed-value population SD;
proportions/ECDFs remain on [0,1]. Missingness indicators are included. No ranks
are imputed. A wholly unobserved panel feature receives a fixed squared penalty
of one, stated in the configuration. Constant/unavailable features do not supply
spurious balance evidence. Supply `--design-config` with a JSON containing
`restarts`, `proposal_budget_per_restart`, and/or `block_weights` to prespecify
alternatives; do not tune these after examining model performance.

The best candidate is chosen by QC objective only, with lexicographic
dataset-sorted assignment tie breaking. Proposed dataset ranks, masses,
component summaries, concentration and figures are saved separately. Residual
imbalance and failure to improve are reported explicitly.

Validation and test IDs are frozen. The production support reader recomputes
eligibility from the selected data. An infeasible common validation cohort
stops preparation with `heldout_support_report.csv`; no split is redrawn.
Sequence-only test prediction retains every frozen test ID even when measured
support falls below the validation threshold. Missing sequence artifacts block
preparation. Training lists are recomputed with all old validation/test IDs
excluded. `w_dt` is fitted once on each new panel's training IDs and reused
unchanged by both policies. This design prepares **eight fresh trainings**.

## Prepared artifacts and authorization boundary

Each preparation writes a distinct child root with `rerun_plan.json`,
`resolved_configs/`, `partition_and_split_manifests/`, `comparison_contract.json`,
`launch_commands.sh`, a copied code snapshot and ranking, a package environment
record, and cryptographic data/config/code hashes in `frozen_execution.json`.
Prepare on the machine where the commands will eventually run: absolute data
and environment paths are frozen, not assumed portable after preparation.

All paired models start fresh and select `best_val_loss`. Saved production
architecture, NB2/consensus shape losses, alpha treatment, grouped sampling and
transcript-balanced reduction are retained. Sequence-only shared-profile test
export is explicitly enabled for both arms to satisfy the frozen-test contract;
any difference from a historical export flag is listed with all other historical
configuration differences in `comparison_contract.json`. Numerical code-version
differences from historical training remain unverified, not silently equated.

The command file accepts exactly one task index (0–7), one production process
on one visible GPU. `--gpus inherit` preserves scheduler-provided visibility;
explicit comma-separated tokens assign future commands round-robin. It refuses
execution unless `RIBOUNMIX_TRAINING_AUTHORIZED=1` is supplied **after subsequent
explicit training authorization**. Before execution it verifies the frozen code,
data, configuration and environment. No GPU job is submitted by this tool.

The existing matched analysis command is saved, not run. It compares six
cross-panel pairs per policy and same-panel equal/ranked profiles using PCC,
Spearman and RMSE, full CDS and 20-codon-per-end interiors, no interior
renormalization, matched finite positions and paired transcript-cluster
bootstrap. The six pairs are not six independent training replications.
The existing matched-analysis utility now includes Spearman in its per-transcript
tables and paired bootstrap summaries (average ranks for tied positions, using
the same near-constant-profile exclusion as PCC). The design audit never runs
the performance-analysis stage.

## Verification and deployment

```bash
python analyses/audit_four_panel_reference_quality.py --help
python -m unittest Tests.test_panel_reference_quality_audit Tests.test_real_panel_convergence
```

Changed/new source files to deploy:

- `audit_four_panel_reference_quality.py`
- `Utils/panel_reference_audit.py`
- `Utils/panel_reference_preparation.py`
- `Utils/real_panel_convergence.py` (optional additional QC objective; existing default behavior unchanged)
- `analyses/compare_real_panel_weighting.py` (adds the requested Spearman metric to the existing matched analysis)
- `Tests/test_panel_reference_quality_audit.py`
- `Tests/test_compare_real_panel_weighting.py`
- `Docs/four_panel_reference_quality_audit.md`
- `results/README.md`

No historical result folders or old training configurations are modified.
