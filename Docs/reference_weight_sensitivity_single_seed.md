# Cumulative quality-reference stability — one seed

The [HTML methodology report](stability_experiments_methodology.html) explains
the current cumulative and four-panel designs, all five ranking policies,
gamma centering, transcript batches, and the paired evaluation, with an
interactive view of the saved dataset weights.

## Run on UNIVIE

From the project directory:

```bash
sbatch run_cumulative_reference_weight_stability_univie.slurm
```

This submits **35 single-GPU tasks**, with two running concurrently. The first
worker builds the shared setup; the others wait for it and reuse it.
`bash submit_cumulative_reference_weight_stability_univie.sh` is an equivalent shortcut.

The setup reads the current dataset YAML, quality-ranking TSV, and configured
parquet/encoding files. **No previous experiment directory or manifest is needed.**
`run_cumulative_stability.py` creates its own inputs and configurations under
`results/cumulative_stability_seed42`, then calls the existing production trainer.
Prepare on the training machine because configurations contain local absolute paths;
do not copy a locally generated setup to the cluster. Keep inputs and training code
unchanged during the experiment. Resubmitting resumes matching full-state checkpoints
and skips completed, validated tasks.

After training, generate the HTML report:

```bash
python analyses/analyze_cumulative_reference_weight_stability.py
```

Open the [cumulative results report](../analyses/artifacts/real_data/cumulative_stability/analysis_report.html).
The linked `design_report.html` shows weight concentration and reference quality.
Analysis can also run before completion: unavailable models remain explicit gaps.

Optional preparation/inspection without GPU training:

```bash
python run_cumulative_stability.py --prepare-only
python run_cumulative_stability.py --task-index 34 --dry-run
```

## Exact experiment

Take the best **2, 5, 10, 20, 40, 80, 114** configured datasets by global QC rank.
Train five fresh models per size, all with **seed 42**: `equal`, `ranked_p1`,
`reverse_p1`, `ranked_p3`, `reverse_p3`. Array indices 0–4 are N=2; 30–34 are N=114.
For global maximum rank R, q=(R−rank+1)/R. Apply q or q³, optionally reverse the
weight assignment within the selected prefix, then normalize. The current ranking
has R=115; 114 datasets are configured (rank 110 is absent).

Within N, every policy uses the same dataset order, seed, transcript split and
training-only reliability references. Across N, test transcripts stay fixed;
training/validation IDs and reliability fits can vary with the available datasets.
π controls **gamma centering**, while w_dt separately weights observation losses.

## What the results mean

Compare each transcript's L_bio at adjacent sizes under each policy. Positive
**PCC_ranked − PCC_equal** and **RMSE_equal − RMSE_ranked** support stabilization.
Ranked versus reverse holds dataset-weight concentration fixed and tests direction.
p=1 is the original mapping; p=3 is a prespecified stronger-reference sensitivity check.

Equal-weight agreement need not decrease monotonically: lower-ranked data can
still help estimation, and nested collections overlap. Stability is not biological
accuracy. The report also measures drift from N=2 and profile variance, using
matched finite transcript cohorts. One seed gives evidence conditional on that
training seed, without estimating variation across training seeds.
