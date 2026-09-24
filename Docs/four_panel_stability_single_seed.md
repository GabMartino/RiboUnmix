# Four-panel reference stability — simple single-seed setup

The [HTML methodology report](stability_experiments_methodology.html) explains
both current stability experiments, the five ranking policies, the gamma=1
ablation, exact saved folds, and the analysis, with an interactive view of
panel membership and reference weights.

## Add the gamma = 1 experiment

With the original full-model jobs already submitted, run only the **four additional
shared-only baselines**, all seed 42:

```bash
sbatch --array=20-23%2 run_four_panel_stability_univie.slurm
```

The first worker appends the baselines to the existing setup. It copies each
panel's saved equal-arm configuration, changes `model.mean_correction` to `unity`,
and assigns separate output paths. The original 20 task indices/configurations,
transcript splits and training-only reliability references are reused. No old
checkpoint initializes a baseline: gamma is one throughout fresh training and prediction.

For the minimal comparison from scratch (four full equal models + four baselines):

```bash
sbatch --array=0,5,10,15,20-23%2 run_four_panel_stability_univie.slurm
```

The primary contrast is **full equal versus shared_only**; ranked-versus-shared-only
also changes reference weighting. The report gives paired PCC, RMSE and peak-overlap
effects for all six panel pairs, and normalized profile variance. Positive effect
values favor the full model. Shared-only prediction exports must contain exact unit gamma.

**Interpretation:** learned dispersion is retained, but its detached context encoder
receives no training gradient in the shared-only arm. This is the whole-system
ablation discussed in the experiment proposal, not the optional confirmation with
the same fixed dispersion in both arms. Alpha's output head still learns.

Observation-fit tables use validation metrics logged at the selected checkpoint;
they are selection-dependent diagnostics. Sequence-only test exports have dummy
observation targets, so they cannot measure independent test-count reconstruction.
Peak sets contain positions strictly above the profile's 90th percentile; ties
at the threshold are excluded and empty/flat sets have undefined overlap.

## Complete experiment

From the project directory on UNIVIE:

```bash
sbatch run_four_panel_stability_univie.slurm
```

The first worker builds the shared inputs directly from current data; the other
workers reuse them. No prior results directory, preparation job or audit pipeline
is required. Defaults: one GPU per task, two concurrent tasks, seed 42.
Prepare on the cluster because generated configurations contain local paths.
Resubmission skips validated completed tasks and resumes matching full-state checkpoints.

## Experiment

The default preserves the **original fixed four-panel membership**, read from
`config/experiment_designs/panels_equal_seed42_20260906_114323.json`.
The 114 datasets appear once across panels of **29, 29, 28, 28** datasets;
related datasets from a source stay in the same panel. This is distinct from the
later rank-balanced repartition; no partition search is performed here.

Each panel gets five full models: **equal, ranked_p1, reverse_p1, ranked_p3,
reverse_p3**, plus **shared_only**, all seed 42: **24 training runs**. Original
indices 0–4 correspond to panel 01, 5–9 to panel 02, 10–14 to panel 03 and 15–19
to panel 04. Indices **20, 21, 22, 23** are the shared-only baselines for panels
01, 02, 03, 04 respectively.
The primary p=1 policies use q=(R−rank+1)/R with the full ranking universe R=115;
p=3 is the same stronger-reference sensitivity check as the cumulative experiment.
Reverse assigns the same weights in reverse rank order within each panel.

Validation and test transcripts are common across panels, requiring at least two
usable datasets in every panel. Splits are rebuilt deterministically from current
data; historical transcript hashes are not prerequisites. Each panel's reliability
reference is fitted on its own training transcripts only and shared by all policies.
Architecture, initialization seed, dataset order and training settings are matched
within each panel. π controls gamma centering; w_dt separately weights observation losses.

## Analysis

After training:

```bash
python analyses/analyze_four_panel_stability.py
```

Open the [four-panel results report](../analyses/artifacts/real_data/four_panel_stability/analysis_report.html).
The report shows panel rank composition, weight concentration, and full-CDS
mean-one L_bio agreement for all six panel pairs. It reports paired PCC/RMSE
improvements of ranking over equal and reverse, using matched finite transcript
cohorts. Missing models leave gaps; variance diagnostics expose near-flat profiles.

This tests reproducibility across source-disjoint dataset collections. It does
not test progressive degradation from adding datasets. Greater agreement is not
biological accuracy. One seed cannot estimate optimization variability; the six
panel pairs share models and are not six independent repetitions.

Optional CPU-only inspection:

```bash
python run_four_panel_stability.py --prepare-only
python run_four_panel_stability.py --task-index 23 --dry-run
```

An explicitly selected alternative fixed assignment can be supplied with `--panels`
(or `PANEL_ASSIGNMENT` for Slurm), using CSV columns `dataset_name`, `panel`,
`source_identifier`. It must cover the same configured dataset universe exactly,
with four balanced, source-disjoint panels. Use a new output directory for another
design (`--output-root`, or `EXPERIMENT_ROOT` for Slurm); pass that directory to
analysis with `--experiment-root`.
