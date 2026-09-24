# Ten-component rank-balanced reference-direction experiment

This experiment uses one frozen source-atomic four-panel partition and changes
only the fixed-reference weights used to center `gamma`. It does not use model
performance to construct the panels.

## Why this is separate from `reproducibility_reference_campaign_main_v2`

The existing `main_v2/partition_preparation` loaded
`HEK_riboseq_profile_quality_rank.tsv` (six components; SHA-256
`07f440ca13c9193f3814d8f529c1e50d8a09ec19fbc1c4a7aa91ecae860be125`).
It must not be relabeled as the ten-component experiment.

The new design loads
`HEK_riboseq_profile_quality_rank_components.tsv` (ten components; SHA-256
`5811cadf68c56740e83b232b2990299d527630326205db9cba8bc7024d3cf1f8`).
The complete table has 115 datasets and `R=115`; 114 datasets are retained in
the experiment. Rank 1 is best.

## Frozen partition

The support-constrained QC-only search retains capacities 29, 29, 28 and 28
and 85 intact source families. It selects search seed 44 from the prespecified
four bounded restarts. The combined balance objective changes from 0.057065
for the original panels to 0.020146 for the proposed panels (64.7% lower).

This is not perfect balance. In particular, the original observed-QC block
changes from 0.001108 to 0.002782 while the ten-component and ranked-reference
blocks improve. All 1,593 frozen validation transcripts satisfy the minimum
support rule. Eight panel/test-transcript records do not meet measured support,
but the frozen sequence-only test IDs are retained and these rows do not block
training.

## Three matched arms

For every panel and training seed:

1. `equal`: `pi_d = 1/N_panel`.
2. `ranked`: `q_d=(R-r_d+1)/R`, then `pi_d=q_d/sum_panel(q)`.
3. `reverse`: assign the exact ranked-arm `q` multiset to dataset identities in
   reverse global-rank order, then normalize.

The reverse arm is preferable to a new formula such as `r/R`: it has exactly
the same maximum dataset weight and effective reference count as the ranked
arm. Thus ranked-versus-reverse tests direction of assignment without changing
weight concentration. Equal-versus-ranked changes both concentration and QC
assignment and should be interpreted accordingly.

Dataset-level reversal does not generally preserve the aggregate mass carried
by a multi-dataset source family. `source_family_reference_mass.csv` records
this residual design difference and must accompany the ranked-versus-reverse
analysis; the control isolates dataset-weight concentration, not source-level
concentration.

The panel membership, dataset order, transcript split, training-only `w_dt`
reference, architecture, objective, optimizer, scheduler, training seed and
initial trainable parameters are matched across arms. Only `pi_d` changes.
All arms use best-validation-loss checkpoint selection.

Three seeds (42, 43 and 44) give 36 fresh trainings. A seed-42 resource
milestone contains 12 trainings, but it cannot by itself distinguish a
reference-policy effect from optimization variability.

## UNIVIE commands

The recommended single command submits preparation and places the seed-42
training array behind an `afterok` dependency, so no GPU worker can start
before its immutable task configurations exist:

```bash
bash submit_rank_balanced_reference_directionality_univie.sh seed42
```

Use `all` instead of `seed42` for all 36 tasks, or `repeats` for indices
12--35 after completing the seed-42 milestone.  `MAX_CONCURRENT=4` is the
default.

The equivalent manual preparation command is:

```bash
sbatch prepare_rank_balanced_reference_directionality_univie.slurm
```

The preparation no longer requires
`results/my_panels_a100_b32_20260906_114323` to have been copied to the
cluster.  When `PANEL_MANIFEST` is not supplied, it recreates the original
equal-panel *design metadata* from the configured weighted datasets and then
requires exact agreement with the repository's portable historical identity:

```text
config/experiment_designs/panels_equal_seed42_20260906_114323.json
```

The verification covers panel membership plus the ordered common validation,
test and per-panel training transcript-ID hashes.  A mismatch stops
preparation.  No historical predictions or checkpoints are read, and no model
is trained by this job.  The reconstructed metadata and rank-balanced audit
are written separately under:

```text
results/four_panel_rank_balanced_qrank10_directionality_seed42/source_design/
results/four_panel_rank_balanced_qrank10_directionality_seed42/partition_audit_qrank10_v2/
```

The current default audit directory is `partition_audit_qrank10_v2`. An
interrupted preparation there is resumed in place; another versioned directory
or a new partition search is not needed.

### Recovering an interrupted final audit

`rerun_plan.json` is created **before** preparation finishes. Its existence is
not readiness. The preparation job now checks the saved state and completes
the final matched audit and input freezing before the directionality task
builder runs. If already complete, it verifies and reuses that preparation.

Deploy these together (including the updated matched-analysis module):

```text
Utils/panel_reference_preparation.py
audit_four_panel_reference_quality.py
analyses/compare_real_panel_weighting.py
prepare_rank_balanced_reference_directionality.py
prepare_rank_balanced_reference_directionality_univie.slurm
submit_rank_balanced_reference_directionality_univie.sh
```

Then resubmit using the normal `seed42` command above. No manual JSON status
edits or deletion of the design directory are required. In particular,
`blocked_before_training` is not changed to ready without rerunning the
matched audit and checking the saved artifacts. The historical two-argument
and hardcoded-ranking versions of `analyses/compare_real_panel_weighting.py`
must not be used with the ten-component preparation.

The recovery step does **not** rerun the assignment search, change panel
membership, redraw transcript splits, refit local reliability weights, or
change training configurations. `preparation_recovery.json` records the
previous failure and the finalizer/matched-audit hashes. For older interrupted
preparations without a `preparation_state.json`, saved dataset hashes and
ranking/split identities are checked, but sequence and snapshot/config hashes
are first recorded at recovery; earlier cryptographic verification of those
files is not claimed. Missing scientific artifacts or changed saved hashes
still stop recovery.

The finalization-only CPU command, after environment activation, is:

```bash
python analyses/audit_four_panel_reference_quality.py \
  --resume-preparation results/four_panel_rank_balanced_qrank10_directionality_seed42/partition_audit_qrank10_v2/global_rank_balanced_partition_v2 \
  --gpus inherit
```

A failed preparation makes its old `afterok` training dependency unsatisfiable.
Cancel that pending array before resubmitting; the submitter creates a new
dependency on the new preparation job. New submissions also use
`--kill-on-invalid-dep=yes` so an impossible dependency is cancelled rather
than left pending indefinitely ([Slurm sbatch reference](https://slurm.schedmd.com/sbatch.html#OPT_kill-on-invalid-dep)).

An existing complete historical design can still be used explicitly:

```bash
PANEL_MANIFEST=/absolute/path/to/panel_manifest.json \
  sbatch prepare_rank_balanced_reference_directionality_univie.slurm
```

Inspect the generated task mapping after preparation:

```bash
python prepare_rank_balanced_reference_directionality.py \
  --output-root results/four_panel_rank_balanced_qrank10_directionality_seed42/equal_ranked_reverse_qrank10 \
  --training-seeds 42,43,44 --list-tasks
```

Run the complete 36-model experiment, with at most four simultaneous jobs:

```bash
sbatch run_rank_balanced_reference_directionality_univie.slurm
```

Run only the 12 seed-42 tasks first:

```bash
sbatch --array=0-11%4 run_rank_balanced_reference_directionality_univie.slurm
```

Then run the two prespecified seed repetitions:

```bash
sbatch --array=12-35%4 run_rank_balanced_reference_directionality_univie.slurm
```

The Slurm worker preserves scheduler-provided `CUDA_VISIBLE_DEVICES`; every
array element starts one production model on logical device 0. Preparation is
CPU-only and never launches training.
