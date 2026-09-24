# Cumulative quality-ranked Experiment 8

This is a separate experiment. The original `run_real_exp8_L_stability.py/.slurm`
and equal-weight results are unchanged.

For a visual explanation comparing the previous design with the verified new
design, open [the offline HTML setup report](exp8_cumulative_quality_rank_setup.html).
It includes the actual membership and reference weights at each N, plus the old
and new quality-mismatch diagnostics. Regenerate it with
`.venv/bin/python analyses/create_exp8_quality_rank_setup_report.py`.

## Design

- Default dataset sizes: **2, 5, 10, 20, 40, 80, 114**.
- Sort active datasets by ascending frozen `quality_rank`, breaking ties by
  dataset name. Each panel is the exact top-N prefix of that order.
- Fit one model per N per training seed, independently initialized from scratch.
  Cumulative refers to membership, not checkpoint warm starts. Default seed: 42.
- Use fixed-reference gamma centering with `weighting=quality_rank` and
  `quality_rank_power=1.0`. For rank r and maximum rank R in the entire frozen
  ranking table, q = (R-r+1)/R and pi_d = q_d^p / sum(q_j^p) over selected datasets.
  The current table has R=115 and covers all 114 active datasets. Do not re-rank
  within a prefix or after removing inactive datasets.
- Preserve the original Exp8 production training settings: learned alpha,
  no mass conservation, standard NB loss (beta=0), grouped batching, gamma bound,
  execution microbatching, and minimum two observed datasets for training rows.
- Use a common held-out sequence-only transcript test set, task-specific
  training/validation eligibility, train-only reliability-reference fitting,
  and `best_val_loss` predictions. Dataset/transcript reliability weights w_dt
  are separate from the gamma-reference weights pi_d.
- Exact top-N selection can split source/publication families. Family atomicity
  is explicitly **not** enforced. Dataset and source overlap reports are saved.

These are nested, dependent convergence comparisons, not independent disjoint
panel replication. Adding datasets changes both their quality composition and
the normalized gamma reference. Comparison against the full model measures
agreement, not biological accuracy, and that model is not ground truth. This
experiment changes both subset selection and gamma weighting relative to the
original Exp8; it does not isolate a weighting-only effect.

## Which ranking file?

The current launcher default is
`Datasets/data/HEK_riboseq_profile_quality_rank_components.tsv`. It contains a
ten-component ranking: periodicity, CDS enrichment, depth, transcript support,
RPF length center/spread, STAR unique/total mapping,
r/tRNA contamination, and replicate agreement. The earlier
`HEK_riboseq_profile_quality_rank.tsv` uses six components. Both tables have 115
entries and cover all active datasets; only five dataset ranks are identical between them.
More components alone do not establish that a ranking is better. Choose the
ranking before examining downstream agreement, and use a separate run ID for a
sensitivity experiment. These are frozen dataset-level QC rankings, not rankings
re-estimated on each training fold.

## Commands

From the repository root, inspect actual membership and gamma-reference weights
without reading profile parquets or launching training:

```bash
.venv/bin/python run_real_exp8_L_stability_quality_rank.py \
  --design-only --run-id inspect_cumulative_rank
```

Validate weighted inputs, construct held-out splits, fit train-only reliability
references, and save all seven training commands without launching them:

```bash
.venv/bin/python run_real_exp8_L_stability_quality_rank.py \
  --dry-run --run-id cumulative_rank_v1
```

`--design-only` and `--dry-run` use separate `_design_only` and `_dry_run` output
directories. The TSV copied into each directory is the immutable ranking source
for its commands. Copy the source repository/data to the cluster and regenerate
there: generated configurations contain absolute local paths.

Submit on the same cluster as the original Exp8. This creates seven independent
Slurm array jobs, one per N, each requesting one GPU:

```bash
sbatch run_real_exp8_L_stability_quality_rank.slurm
```

The array mapping is ordered `114, 80, 40, 20, 10, 5, 2` so the longest jobs
are eligible to start first. The scheduler may place independent array jobs on
the same physical host; requesting node exclusivity would usually reduce queue
priority and is therefore intentionally not enabled. A preparation lock creates
one common frozen design and held-out split before any training starts. The last
successful task automatically runs the combined analysis.

Local training equivalent:

```bash
.venv/bin/python run_real_exp8_L_stability_quality_rank.py \
  --run-id cumulative_qrank10components_p1.0_seed42 --gpus 0,1
```

To reuse an earlier experiment's exact held-out IDs, add
`--experiment1-manifest /path/to/previous/run/common_test_manifest.json` locally,
or export `EXPERIMENT1_MANIFEST` when submitting Slurm. The new ranked full model
must still be trained; an old equal-weight full checkpoint is not interchangeable.

The Slurm defaults create the clearly separated run directory
`real_exp8_L_stability_quality_rank_10components/cumulative_qrank10components_p1.0_seed42`.
The launcher validates that the table contains exactly ten populated component
rank columns before constructing any task. To run the older six-component
ranking explicitly, override both the file and the expected count and use a
separate output root/run ID.

```bash
QUALITY_RANKING_TABLE="$PWD/Datasets/data/HEK_riboseq_profile_quality_rank.tsv" \
EXPECTED_RANK_COMPONENTS=6 \
OUTPUT_ROOT="$WORK/riboai_runs/real_exp8_L_stability_quality_rank_6components" \
RUN_ID=cumulative_qrank6components_p1.0_seed42 \
sbatch --export=ALL run_real_exp8_L_stability_quality_rank.slurm
```

`QUALITY_RANK_POWER` controls p in Slurm (`--quality-rank-power` locally). This
ranked launcher requires p > 0. To run multiple initializations locally, use
`--training-seeds 42,43,44` and a distinct run ID (21 models at default sizes).

Resume with the same run ID, output root, scientific arguments, inputs, ranking,
and held-out-set argument:

```bash
sbatch --export=ALL,RESUME=1 run_real_exp8_L_stability_quality_rank.slurm
```

Resume skips completed checkpoint/prediction pairs after hash checks. Incomplete
runs restart from scratch; this launcher does not resume optimizer state. Do not
run two submissions against the same output directory concurrently. It refuses
changed scientific/input hashes and changed frozen rankings.
On resume, it reuses the saved quality-diagnostic table only after validating
its input-file and sequence signatures, avoiding repeated replica-QC computation.

## Saved outputs and analysis

The run root contains `cumulative_dataset_order.csv`, `overlap_report.csv`, a
frozen ranking TSV, experiment/split/common-test manifests, and a quality report.
Each `Nxxx/trainseed42` directory contains a subset manifest with exact ranks,
pi values, reference effective dataset count, reliability manifest, resolved
configuration, launch command, logs, checkpoints, and predictions.

After successful training, the launcher automatically calls:

```bash
.venv/bin/python analyses/analyze_real_exp8_cumulative_quality_rank.py \
  --run-root /path/to/completed/cumulative_rank_v1
```

The analysis reads original mean-one full-CDS prediction arrays, checks shared
transcript IDs and masks, and compares each smaller model to the full model and
to its successor. It exports per-transcript PCC/Spearman/RMSE parquet tables,
summary CSVs, provenance, mean-one checks, and PDF/300-dpi PNG convergence curves.
Constant-profile PCC is undefined and counted as unusable. Transcript quantiles
are descriptive, not confidence intervals; no independence is assumed.

### Analysis before all runs have finished

Use the separate partial-results script; it does not require the full N=114
model and never substitutes a smaller model for it:

```bash
.venv/bin/python analyses/analyze_real_exp8_quality_rank_partial.py \
  --run-root results/real_exp8_L_stability_quality_rank/cumulative_qrankp1.0_seed42
```

It writes `analysis_partial_quality_rank/index.html`, five types of figures
(main stability/typical profile, pairwise heatmap/ECDF, incremental PCC/RMSE,
design diagnostics, and training histories), vector PDFs, 300-dpi PNGs, source
tables, a selection audit, captions and a LaTeX figure snippet. The default is
5.5-inch figure width, with real LaTeX/Latin Modern, 9-point labels and 12-point
titles; `--figure-width` changes the physical width.

Only sequence-only `best_val_loss` exports passing run, transcript, mask,
mean-one and gamma-reference checks enter agreement statistics. Copied cluster
paths are resolved by exact task-relative suffix, not by ambiguous filenames.
Missing/invalid exports are audited. Failed runs may contribute logged training
curves but not invented shared-profile predictions. Re-run the same command
after copying new completed exports; the selected typical transcript can change
when more model pairs become available. The gallery/analysis manifest identifies
the figures produced by the current invocation.

Verification:

```bash
.venv/bin/python -m unittest Tests.test_real_exp8_cumulative_quality_rank -v
bash -n run_real_exp8_L_stability_quality_rank.slurm
```

Tests exercise frozen global-rank weighting, exact nesting and ties, invalid
designs, held-out exclusion, train-only reliability fitting, dry-run command
generation, resume drift refusal, and analysis using explicitly synthetic arrays.

## UNIVIE submission

Submit the UNIVIE-specific wrapper with:

```bash
sbatch run_real_exp8_L_stability_quality_rank_univie.slurm
```

It uses partition `p_csunivie_gres`, loads `python/3.11`, activates
`~/venvs/queueing_riboai_venv`, and invokes the shared seven-element array
launcher through `srun`. Its output root is local to the repository:
`./results/real_exp8_L_stability_quality_rank_10components`; caches are kept in
`./results/.scratch` rather than a Leonardo path.

### UNIVIE throughput settings

The UNIVIE array defaults to `EXP8_RUNTIME_PROFILE=auto`. Each array element
detects its GPU inside the allocated `srun` step and selects an execution profile
after loading the common saved design. These are unbenchmarked starting points:
GPU memory alone cannot establish the fastest settings or guarantee freedom
from OOM for every transcript length and support pattern.

| Condition | Pair rows per forward | Padded codons per forward | Reference chunk |
|---|---:|---:|---:|
| GPU <30 GiB | 256 | 128000 | min(N, 8) |
| N<40 or GPU 30–<60 GiB | 512 | 256000 | min(N, 16) |
| N>=40 and GPU >=60 GiB | 1024 | 512000 | 32 |
| Explicit aggressive mode, N>=40 and GPU >=75 GiB | 2048 | 1024000 | min(N, 64) |

Both profiles log every 100 execution chunks. They use zero data workers for
N>=40 to avoid copies of the large in-memory dataset under spawn, and two for
smaller panels. Zero workers can be slower if collation is the bottleneck;
measure before choosing more workers on a 128-GB host.

The original logical batch size of 32 **per dataset**, optimizer target, loss,
ranking, and all reference datasets remain in use. At N=114 the logical upper
bound is 3648 rows, not 32. The reference branch also scales with N: chunks of
16/32/64 require 8/4/2 passes respectively. Reference autograd activations are
still retained, so a chunk size is not a strict GPU memory cap. Repartitioning
can change dropout draws and floating-point results, not the intended objective.

Submit normally for the automatic profile:

```bash
sbatch run_real_exp8_L_stability_quality_rank_univie.slurm
```

For an N114-only continuation with the aggressive candidate (index 0):

```bash
sbatch --array=0 --export=ALL,EXP8_RUNTIME_PROFILE=aggressive \
  run_real_exp8_L_stability_quality_rank_univie.slurm
```

Finish or cancel the existing task before submitting another against its output
directory. The launcher resumes usable full-state checkpoints and skips completed
tasks. No optimizer state is discarded to change these execution settings.

To set explicit runtime values, which take precedence over the profile:

```bash
sbatch --array=0 \
  --export=ALL,RUNTIME_MAX_PAIR_ROWS=1024,RUNTIME_MAX_PADDED_TOKENS=512000,RUNTIME_REFERENCE_CHUNK=32,RUNTIME_NUM_WORKERS=1 \
  run_real_exp8_L_stability_quality_rank_univie.slurm
```

Use the `RUNTIME_*` variables for existing experiments. The older preparation
variables change the saved design and may fail its hash checks. Settings are
printed and saved in each task's `resume_manifest.json` and `resume_command.sh`.
An explicit reference-chunk override now survives checkpoint extra-state loading.
`EXP8_RUNTIME_PROFILE=unchanged` restores the saved command's execution settings.

Compare **seconds per full epoch or optimizer updates per second** at the same
N, GPU, and data, after warm-up. Progress-bar iterations count execution chunks;
a larger chunk may reduce iterations/second while still shortening the epoch.
Compare train and validation time separately and inspect GPU utilization/memory
and host RAM before increasing limits further. This change was verified with
CPU tests; no UNIVIE throughput benchmark was available locally.
