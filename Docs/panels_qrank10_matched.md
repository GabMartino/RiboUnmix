# Fresh matched four-panel experiment: equal versus ten-component ranking

## Scientific choice

Keep the historical four panel memberships and transcript splits fixed. This tests changing the gamma-reference weights, not changing the dataset composition. Both arms are newly trained with the same current numerical policy; no historical checkpoint is used. A portable reference extracted from the historical equal-weight run supplies design metadata only; the old results directory does not have to exist on the cluster.

The two arms are:

- `equal`: four independently initialized models with uniform reference weights.
- `qrank10components_p1`: four independently initialized models using the ten-component `HEK_riboseq_profile_quality_rank_components.tsv`, power 1, global ranks normalized within each fixed panel.

These are eight training runs, not eight dataset panels. Each arm has the same four panels (29/29/28/28 datasets), source-family grouping, seed 42 and train/validation/test identities. The ranked arm's configuration is checked against the freshly prepared equal arm; both arms' assignments and transcript splits are checked against the historical equal design. A mismatch fails before any training starts.

Do not repartition to make this primary comparison look better. A new partition balanced explicitly on the ten-component score is a separate design-sensitivity experiment and needs its own equal and ranked controls (another eight models). Comparing old equal panels against newly repartitioned ranked panels would conflate weighting with composition. That separate experiment is not implemented or launched here.

## Numerical policy and provenance

Both arms use BF16 mixed precision, FP32 CUDA GRUs, full BPTT (`context_gru_tbptt_window=0`), logical batch size 32, fixed execution limits of 512 pair rows / 256,000 padded codon tokens and reference chunks of 16. Data workers are zero to avoid dataset-copy memory costs. These conservative execution settings are identical across arms; they are not claimed to maximize throughput on every GPU. Native BF16 hardware is required; `dgx1` is excluded.

Preparation saves a copy of the historical design reference, ranking and base/dataset configurations. It records the ten component names, ranking hash, source-code hashes, Python/library/CUDA/cuDNN versions and count/sequence file size/mtime signatures. Resubmission reuses these files and refuses changed training source/environment or input signatures. File signatures are not cryptographic checksums of the full count arrays. GPUs may differ across jobs, so this is not a bitwise reproducibility guarantee. Do not edit training code in the shared checkout while jobs are queued or running; use a dedicated synchronized checkout/environment for the experiment.

### Portable historical reference

`config/experiment_designs/panels_equal_seed42_20260906_114323.json` is approximately 16 KB. It contains the actual 114 dataset-to-panel/source-family assignments, ordered transcript-list SHA-256 fingerprints and counts (1,593 validation and 1,593 test transcripts, plus each panel's training cohort), and hashes of the two original source metadata files. No predictions or checkpoints are included.

The design builders still compute eligibility, panels, splits and train-only reliability references from the configured data. Both newly prepared arms must match the bundled assignments and transcript fingerprints before training. The ordered-list encoding preserves order and duplicates, matching the previous direct list comparisons; file locations and timestamps are not transcript identity. This is a replacement for access to the old metadata, not permission to silently rebuild a different design. The reference is frozen into `frozen_reference_design.json` and checked on subsequent submissions.

For another explicitly chosen historical reference, supply `--reference-design-root` / `REFERENCE_DESIGN_ROOT` with a directory containing `panel_assignment.csv` and `common_split_manifest.json`, or `--reference-design-manifest` / `REFERENCE_DESIGN_MANIFEST` with an equivalent portable JSON. An explicitly supplied missing reference fails rather than silently falling back. Leave both overrides unset to use the shipped reference.

## Cluster submission

For the missing-historical-folder fix, deploy these changed/new files (relative to the repository root):

- `run_real_panels_qrank10_matched.py`
- `run_real_panels_qrank10_matched_univie.slurm`
- `run_real_panels_qrank10_matched.slurm`
- `config/experiment_designs/panels_equal_seed42_20260906_114323.json` (required)
- `Tests/test_panels_qrank10_matched.py` (verification)
- `Docs/panels_qrank10_matched.md` (this document)

Keep the already deployed current numerical-fix code, both independent-panel builders, the checkpoint-resume helper and the component validator in `run_real_exp8_L_stability_quality_rank.py`. Ensure the ten-component TSV and auditable weighted replica-aware parquet inputs are synchronized. Do not copy the historical training results just to satisfy this launcher; the small reference JSON above is sufficient.

UNIVIE:

```bash
unset REFERENCE_DESIGN_ROOT REFERENCE_DESIGN_MANIFEST
sbatch run_real_panels_qrank10_matched_univie.slurm
```

Leonardo (choose this instead, not in addition against the same output):

```bash
unset REFERENCE_DESIGN_ROOT REFERENCE_DESIGN_MANIFEST
sbatch run_real_panels_qrank10_matched.slurm
```

UNIVIE loads `python/3.11` and `~/venvs/queueing_riboai_venv`; Leonardo defaults to the repository `.venv` and accepts `VENV_PATH`. Both scripts store results under `./results`, never `/leonardo_scratch`.

| Array indices | Arm | Panels |
|---|---|---|
| 0–3 | Equal | 1–4 |
| 4–7 | Ten-component ranked | 1–4 |

Each element requests one GPU on one node. Nodes may be shared with other jobs; physical-node exclusivity is not requested. Default concurrency is four (`0-7%4`). All tasks wait for serialized preparation to finish before training. Outputs:

```text
results/panels_matched_equal_vs_qrank10_seed42_v1/
  matched_experiment.json
  frozen_quality_rank_10components.tsv
  frozen_config.yaml
  frozen_dataset_config.yaml
  frozen_reference_design.json
  equal/panel_01 ... panel_04
  qrank10components_p1/panel_01 ... panel_04
```

Re-submit the same command to continue incomplete runs from their own checkpoints. Completed tasks are checked by the existing resume helper. Use `--array=4-7%4` to select only ranked tasks; a fresh equal control is still needed for the primary comparison. For a genuinely new repeat, set `EXPERIMENT_ROOT` to a new directory. A partially failed preparation is not overwritten automatically; inspect `equal_prepare.log` / `qrank10components_p1_prepare.log`, correct the cause and choose a new output root.

The reported missing-reference exception occurs before the experiment output directory is populated, so that failure alone permits using the same output path after this fix. If a `matched_experiment.json` already exists from an older launcher, its code/environment pin still applies: do not bypass it to mix implementations. Keep that prepared experiment on its pinned code, or use a new output root for the updated launcher.

Optional CPU preparation before submitting (on a suitable CPU node, not an overloaded login node):

```bash
python run_real_panels_qrank10_matched.py --prepare-only
```

Optional selected command audit, without training:

```bash
python run_real_panels_qrank10_matched.py --task-index 4 --dry-run
```

The latter still prepares a missing experiment and may need CUDA for runtime precision checks.

## Analysis after all eight runs finish

```bash
ROOT_RUN=results/panels_matched_equal_vs_qrank10_seed42_v1
python analyses/compare_real_panel_weighting.py \
  --equal-root "$ROOT_RUN/equal" \
  --ranked-root "$ROOT_RUN/qrank10components_p1" \
  --ranking-table "$ROOT_RUN/frozen_quality_rank_10components.tsv" \
  --output-dir "$ROOT_RUN/comparison" \
  --require-all-panels
```

The comparison now resolves the ranking by its saved hash, rather than assuming the six-component TSV. It still checks identical datasets, transcript cohorts, reliability references and reference weights. Report PCC and RMSE together and inspect all six pairs. Paired transcript-bootstrap intervals condition on these fitted models; one seed does not establish robustness over retraining or new panel partitions. Neither agreement metric is biological ground-truth accuracy.

## Verification

The automated integration test runs both real design builders on eight small replica-aware datasets with the historical directory absent, using a portable reference. It checks the matched design/weighting audit and composes all eight saved training commands with Hydra. It exercises ten-component validation, exact assignment/ordered-ID rejection, one-time preparation, frozen-reference/code drift rejection and task mapping. This is not an eight-model cluster training or throughput benchmark.

```bash
python -m unittest Tests.test_panels_qrank10_matched
```
