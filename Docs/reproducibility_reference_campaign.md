# Reproducibility reference campaign

## Status

CPU preparation and the one-process-per-GPU training runner are operational.
No GPU training has been launched. Export/analysis integration beyond native L
validation remains pending; those stages refuse rather than fabricate results.

Main: 48 fresh trainings (12 equal, 12 ranked, 12 gamma=1, 12 shuffled at seed42).
Extended: 96; only main was prepared on actual artifacts. Seeds are 42,43,44.
The six-component HEK_riboseq_profile_quality_rank.tsv is used, with global R=115.
Its historical transcript scope and freeze chronology remain not verified.

The v1 blocked candidate is preserved. The authorized v2 search starts from the
feasible original panels and rejects source-family swaps breaking validation
support before scoring. The original objective and four × 2,000-proposal budget
are unchanged. No model performance is used.

The v2 candidate retains 114 datasets, 85 families, capacities 29,29,28,28 and all
held-out IDs. All 1,593 validation transcripts pass support in every panel.
The total QC objective improves from 0.06580797 to 0.01707783, but the original
observed-QC block worsens from 0.00110824 to 0.00419162. Approval must consider
that tradeoff, not just the total. All 1,593 test IDs remain in sequence-only
export regardless of measured support.

48 production-size CPU initialization/short-gradient probes were executed.
Initial parameters match across arms within panel/seed. Gamma=1 leaves the
detached dataset-context encoder without likelihood gradients; the shared
encoder and alpha head retain gradients. This is not identical learned
dispersion context across full and ablated models.

The runner verifies frozen code/data/configuration/environment, initial
parameters, approved hashes and task budget. Resume requires matching task
identity and full optimizer/scheduler state. Best-val-loss native L is validated
against frozen IDs/lengths, positivity, finiteness, normalization and duplicates.
CPU/mocked execution tests are not GPU tests or an overflow-free guarantee.

## UNIVIE preparation

Deploy the files below and prior campaign dependencies. Prepare ON THE CLUSTER:
the local Python 3.12 frozen environment is not portable to cluster Python 3.11.

```bash
mkdir -p results
CAMPAIGN_STAGE=prepare \
CAMPAIGN_ROOT="$PWD/results/reproducibility_reference_campaign_main_v2" \
sbatch prepare_reproducibility_reference_campaign_univie.slurm
```

Use a new root if that directory already contains another frozen experiment.
RESUME_PREPARATION=1 reuses verified preparation, not migrated inputs.
NO_TEX=1 changes rendering only. Wrappers use p_csunivie_gres, exclude dgx1,
activate ~/venvs/queueing_riboai_venv and keep outputs in ./results.
A report job alone does not prepare training.

Review campaign_report.html, balance_objective_before_after.csv,
sequence_split_audit.json and candidate_partition_manifest.json.
Read the plan hash from campaign_manifest.json only after review.

## Training after approval

```bash
export CAMPAIGN_ROOT="$PWD/results/reproducibility_reference_campaign_main_v2"
export APPROVED_PARTITION_MANIFEST="$CAMPAIGN_ROOT/candidate_partition_manifest.json"
export APPROVED_PLAN_HASH='<reviewed cluster-generated plan hash>'
export AUTHORIZE_TRAINING=1
export MAX_NEW_TRAININGS=8
bash submit_reproducibility_reference_campaign_univie.sh
```

CPU preflight runs before sbatch. Array indices 0–7 execute the eight B
equal/ranked seed-42 models, at most four concurrently, one GPU per model.
After resource/debugging checks, without performance-based selection:

```bash
START_TASK_INDEX=8 MAX_NEW_TRAININGS=48 \
bash submit_reproducibility_reference_campaign_univie.sh
```

The cap admits a prefix of the matrix, not that many GPUs. MAX_CONCURRENT changes
concurrency only. For retries set RESUME_CAMPAIGN=1 and the intended array range.
Scheduler CUDA visibility is preserved. Precision/batch/objective are unchanged.
This document does not itself approve a candidate or authorize training.

## Tests and pending work

```bash
python -m unittest Tests.test_campaign_training Tests.test_reference_campaign_cli \
 Tests.test_reference_campaign_univie_slurm Tests.test_reference_campaign \
 Tests.test_panel_reference_quality_audit Tests.test_real_panel_convergence \
 Tests.test_compare_real_panel_weighting Tests.test_gamma_centering
```

72 CPU tests passed: support constraints, deterministic assignment, task caps,
resume identity, paired configs, global ranks, permutations, numerical primitives
and mocked launchers. Sequence duplicates across folds are reported without
altering splits; gene overlap is unverified without a supplied gene mapping.

Pending: user approval, GPU training, full streamed factor exports,
common-training predictions, integrated multiseed/residual/peak/boundary analyses,
and performance figures. Report separates availability from analyses.
Historical models are not reused as matched controls.

## Files to deploy

- Utils/campaign_training.py (new)
- Utils/reference_campaign.py
- Utils/panel_reference_preparation.py
- Utils/real_panel_convergence.py
- audit_four_panel_reference_quality.py
- main_ribounmix_multidataset.py
- run_reproducibility_reference_campaign.py
- prepare_reproducibility_reference_campaign_univie.slurm
- run_reproducibility_reference_campaign_univie.slurm
- submit_reproducibility_reference_campaign_univie.sh
- Tests/test_campaign_training.py (new)
- Tests/test_reference_campaign_univie_slurm.py
- Docs/reproducibility_reference_campaign.md

Previously implemented explicit-reference and gamma=1 support in
Models/RiboUnmixModel/RiboUnmixModel.py must also be deployed.
