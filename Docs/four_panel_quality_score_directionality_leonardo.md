# Four-panel directional quality-score experiment on Leonardo

## Recommended experiment

Run the **balanced source-disjoint four-panel design**:

```bash
cd /path/to/RiboUnmix
export VENV_PATH="$HOME/venvs/queuing_ribo_venv"
bash submit_four_panel_quality_score_directionality_leonardo.sh balanced
```

This submits one preparation job followed by a dependent 28-task GPU array.
There is no array throttle in the script; scheduler and allocation limits decide
how many tasks run concurrently.

The submission wrapper validates `numpy`, `pandas`, `PyYAML`, `pyarrow`,
`torch`, and `lightning` before creating either Slurm job, then exports the same
absolute `VENV_PATH` to preparation and training. This avoids accidentally
selecting a checkout-local `.venv` that exists but lacks the training stack.

The default output is:

```text
/leonardo_work/euhpc_d35_089/$USER/riboai_runs/
four_panel_quality_score_directional_balanced_seed42
```

Set `FOUR_PANEL_SCORE_ROOT` before submission to override it.

## Scientific design

The 114 datasets retain the existing four fixed panels of 29, 29, 28 and 28
datasets. Source families remain confined to one panel and the QC-rank
composition is approximately balanced. Dataset membership therefore remains
constant while the gamma-reference prior changes.

Each panel is trained from scratch with seed 42 under seven policies:

1. `equal`
2. `best_first_score_p1`
3. `worst_first_score_p1`
4. `best_first_score_p3`
5. `worst_first_score_p3`
6. `best_first_score_p5`
7. `worst_first_score_p5`

Let (s_d) be `quality_rank_score`, where lower values indicate better measured
quality. The normalized gamma-reference weights are

\[
\pi_d^{\mathrm{best}} \propto s_d^{-p}, \qquad
\pi_d^{\mathrm{worst}} \propto s_d^{p},
\qquad p\in\{1,3,5\}.
\]

The global minimum and maximum factors used by the implementation only rescale
all raw weights and cancel after normalization.

Use **p=1 as the primary score intervention**. Treat p=3 and p=5 as increasing
concentration stress tests. A preflight calculation on the frozen balanced
panels gives best-oriented `N_eff/N` ranges of 0.822–0.930 for p=1,
0.208–0.478 for p=3, and 0.089–0.239 for p=5. In panel 3, best-oriented p=5
has only about 2.5 effective datasets out of 28 and its largest reference mass
is approximately 0.58. That arm is useful for demonstrating the mechanism but
is too concentrated to present as the default biological model.

“Worst-first” is retained in task names for consistency with the cumulative
campaign. Here it means **worst-oriented reference weighting inside a fixed
panel**. It does not change panel membership or train first on worse datasets.

## Transcript and reliability controls

The preparation step constructs one complete-case transcript cohort across all
114 datasets and partitions it once. Training, validation and test IDs are
identical for every panel and policy. This is stricter than the original
`four_panel_stability_seed42` experiment, whose training eligibility was
panel-specific.

Each dataset's observation reliability reference is fitted once using the
common training IDs. Panel manifests are then exact subsets of that complete
reference. Thus neither transcript membership nor reliability fitting changes
with panel or reference policy.

The complete-case cohort favors transcripts observed broadly across datasets;
that restriction must be reported when interpreting generalization.

## Intended analysis

The primary endpoint is transcript-level PCC of **L_t** across the six panel
pairs under the same reference policy.

- Equal weighting measures baseline cross-panel reproducibility.
- Best-oriented versus equal tests whether a quality-directed gamma reference
  improves reproducibility.
- Worst-oriented versus equal is the directional control.
- Best-oriented versus worst-oriented tests direction, but only after examining
  `N_eff/N` because the same power need not produce the same concentration.
- Same-panel PCC between each score-weighted fit and equal quantifies how much
  the reference changes the selected decomposition.

High cross-panel PCC is evidence of reproducibility, not biological accuracy.
If both best- and worst-oriented references improve agreement similarly, the
result supports a concentration effect rather than the QC direction. One seed
does not estimate optimization variability.

The aggregate quality score is a sum of component ranks. Raising score ratios
to a power treats those ordinal rank sums as if their ratios were cardinal.
This follows the cumulative experiment exactly, but it remains a modeling
assumption. A later concentration-matched analysis should calibrate a separate
temperature per panel and orientation to common target values of `N_eff/N`.

## Optional quality-strata sensitivity

The same runner can instead use the four contiguous quality strata:

```bash
bash submit_four_panel_quality_score_directionality_leonardo.sh quality_strata
```

This creates a separate output directory. It is secondary because panel
membership then changes strongly with quality and related source families may
be split across strata. It mixes quality composition with reference orientation,
whereas the balanced design isolates the reference intervention more cleanly.

## Inspect task mapping

```bash
bash run_four_panel_quality_score_directionality_leonardo.slurm --list-tasks
PANEL_DESIGN=quality_strata \
  bash run_four_panel_quality_score_directionality_leonardo.slurm --list-tasks
```

## Runtime memory controls

Execution microbatch limits can be changed without regenerating the scientific
task plan:

```bash
export RUNTIME_MAX_PAIR_ROWS=762
export RUNTIME_MAX_PADDED_TOKENS=512000
bash submit_four_panel_quality_score_directionality_leonardo.sh balanced
```

Use values supported by the assigned GPU memory. These settings partition the
same logical batch across forward calls; they do not change panel membership,
loss weights or optimizer batch construction.

## Resubmission and recovery

Submitting the same design and output root again is safe. Completed tasks are
validated and skipped. Interrupted tasks resume when a compatible full-state
checkpoint exists. To require a checkpoint instead of allowing a fresh start:

```bash
export REQUIRE_RESUME_CHECKPOINT=1
bash submit_four_panel_quality_score_directionality_leonardo.sh balanced
```

If an older preparation job failed at `import numpy`, cancel its dependent GPU
array if it is still listed, then resubmit with the populated environment:

```bash
squeue -u "$USER" --name=four_score_dir -o "%.18i %.10T %R"
# scancel <old-dependent-array-job-id>

export VENV_PATH="$HOME/venvs/queuing_ribo_venv"
"$VENV_PATH/bin/python" -c \
  'import numpy, pandas, yaml, pyarrow, torch, lightning; print("environment OK")'
bash submit_four_panel_quality_score_directionality_leonardo.sh balanced
```

The failed import occurred before experiment preparation, so the same output
root can be reused; no partial fit was produced by that preparation job.

The preparation step writes `design_report.html`, `task_matrix.csv`,
`reference_weights.csv`, and `reference_concentration.csv` into the experiment
root. Inspect these before interpreting model results.

After copying partial or completed outputs to the analysis machine, generate the
matched PCC report with:

```bash
python analyses/analyze_four_panel_quality_score_directionality.py \
  --experiment-root /path/to/four_panel_quality_score_directional_balanced_seed42
```

The analyzer leaves missing policies and panel pairs as explicit gaps. It
reports cross-panel PCC, same-panel score-versus-equal sensitivity, direct
best-versus-worst orientation agreement, and paired score-minus-equal effects.
