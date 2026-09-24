# UNIVIE mapped campaign submission

These wrappers replicate the Leonardo execution layer, not the experiment design.
The earlier run/submit *_univie files remain unchanged because existing prepared
plans hash them. Use the new *_mapped_univie files for immutable task mapping,
non-contiguous resume, scheduler reconciliation and dependent CPU analysis.

## Infrastructure defaults

- Partition: p_csunivie_gres; excluded node: dgx1.
- No Leonardo account or QoS is supplied. Explicit --account/--qos overrides
  remain available if your allocation needs them.
- One GPU, one task, eight CPUs, 128G, four-day limit per training worker,
  matching the existing UNIVIE campaign training wrapper.
- Analysis: no GPU, eight CPUs, 128G, one-day limit on p_csunivie_gres,
  matching the existing UNIVIE CPU preparation wrapper. Site acceptance of
  that CPU-only request has not been tested here; --analysis-* overrides remain.
- Use UNIVIE_VENV_PATH when set, otherwise the active VIRTUAL_ENV. With neither
  set, choose the sole runnable environment among ~/venvs/queueing_riboai_venv,
  ~/venvs/queuing_ribo_venv and the project .venv. The repository setup scripts
  create the second spelling; earlier UNIVIE launchers assumed the first.
  If several exist, specify the one used for campaign preparation explicitly.
  The order follows run_real_independent_panel_convergence_quality_rank_univie.slurm:
  load a requested Python module, source the selected bin/activate, then execute
  Python. The --version check happens after activation so activation-provided
  runtime settings are available. Discovery probes candidates in subshells and
  activates the selected environment in the launcher itself. Python modules are optional:
  your cluster reports that `python/3.11` is unavailable, so the mapped scripts
  do not load it by default. Override UNIVIE_VENV_PATH explicitly if needed.
  Only set LOAD_PYTHON_MODULE=1 and PYTHON_MODULE=<verified module name> when
  your specific venv requires it.
- Preserve scheduler CUDA visibility; production trainer uses logical device 0.
- Keep all scientific output paths in the already prepared campaign. New logs
  and mappings go under CAMPAIGN_ROOT/slurm_submissions. No Leonardo scratch paths.

## Commands from the deployed repository

### Preparation resume reports only `'ranking'`

An audit manifest is also saved when input loading fails. In that case it may
contain `hard_errors` but no `ranking` section. Older campaign code indexed the
missing section on resume, hiding the original failure. The current code checks
the saved failure first, reports its details and artifact path, and does not
repeat the partition search or modify the audit. Unexpected missing-key errors
now retain a traceback in the batch stderr log.

Inspect the original failure without Python, modules, or another cluster job:

```bash
jq '{hard_errors, issues, ranking, input_sha256}' \
  "$CAMPAIGN_ROOT/partition_preparation/audit_manifest.json"
```

`RESUME_PREPARATION=1` is for successful saved CPU preparation; it does not repair
a failed audit. Resolve its reported input/dependency issue first. Preserve the
failed report. Do not delete approved artifacts, fabricate a ranking section,
or bypass input hashes. A genuinely interrupted successful preparation is a
different case and must retain any already selected partition.

### When the submitting node cannot run the venv

`bin/python -> python3` is normally the first link in a chain whose second link
points to the Python used to create the venv. A broken chain cannot be fixed by
activation. The previous module-not-found message and this broken link suggest
different software availability on the submitting and compute nodes; that cause
is not verified until the same environment is checked on a compute node.

The working independent-panel script loads python/3.11 inside its batch job.
Use this CPU batch entrypoint to run the existing campaign preflight there:

```bash
mkdir -p results
LOAD_PYTHON_MODULE=1 PYTHON_MODULE=python/3.11 \
sbatch submit_reproducibility_reference_campaign_from_compute_univie.slurm \
  --campaign-root "$CAMPAIGN_ROOT" \
  --max-new-trainings 8 --with-analysis --dry-run
```

This submits ONE CPU preflight job. It does not submit any model or analysis
jobs in --dry-run mode. Read `results/job_<jobid>_riboai_campaign_submit.out`
and `.err`. The existing APPROVED_PLAN_HASH and APPROVED_PARTITION_MANIFEST
values must still be exported as described below. No prepared design is rebuilt.

After preflight succeeds, submit the same controller with --submit:

```bash
LOAD_PYTHON_MODULE=1 PYTHON_MODULE=python/3.11 \
sbatch submit_reproducibility_reference_campaign_from_compute_univie.slurm \
  --campaign-root "$CAMPAIGN_ROOT" \
  --max-new-trainings 8 --max-concurrent 4 --with-analysis --submit
```

The outer returned job ID identifies the CPU controller; the training-array and
analysis job IDs are printed in its stdout and saved by the mapped submitter.
Submission from a compute node must be permitted by the site; this has not been
tested locally. The requested CPU resources follow the existing preparation
script, and the controller requests no GPU. It inherits the verified module
selection into its worker submissions.

If the interpreter is also missing on the compute node, inspect the reported
python3 target and pyvenv.cfg home/executable. Restore access to that interpreter
or resolve the environment deployment before running the frozen campaign.
Repointing the symlink to another Python is not an equivalent environment fix.

### Direct submission when the venv is available

For an environment created by slurm_python_venv_setup.slurm or
slurm_python_install_new_packages.slurm, the documented path is:

```bash
export UNIVIE_VENV_PATH="$HOME/venvs/queuing_ribo_venv"
source "$UNIVIE_VENV_PATH/bin/activate"
"$UNIVIE_VENV_PATH/bin/python" --version
```

This selects an existing environment; it does not create or reinstall one.
The campaign's frozen environment checks still apply. If the launcher reports
a broken Python symlink, inspect bin/python and pyvenv.cfg in that environment;
the base interpreter may require the same module/mount used during preparation.

Use the existing approved cluster-resident plan. Do not copy a locally frozen
workstation plan and rewrite paths or hashes to make it pass.

```bash
export CAMPAIGN_ROOT="$PWD/results/reproducibility_reference_campaign_main_v2"
export APPROVED_PLAN_HASH='<approved cluster-generated plan hash>'
export APPROVED_PARTITION_MANIFEST="$CAMPAIGN_ROOT/candidate_partition_manifest.json"

bash submit_reproducibility_reference_campaign_mapped_univie.sh \
  --campaign-root "$CAMPAIGN_ROOT" --max-new-trainings 8 --dry-run

bash submit_reproducibility_reference_campaign_mapped_univie.sh \
  --campaign-root "$CAMPAIGN_ROOT" --max-new-trainings 8 \
  --max-concurrent 4 --with-analysis --submit
```

The first command validates without submitting. The second prints and executes
the exact resolved sbatch commands: an eight-element one-GPU array followed by
CPU analysis with afterok dependencies. --submit propagates explicit training
authorization; the existing approved plan/partition hashes are still required.
Do not submit the batch files directly without their immutable mapping arguments.

An isolated infrastructure/GPU smoke, with no shortened scientific training:

```bash
bash submit_reproducibility_reference_campaign_mapped_univie.sh \
  --campaign-root "$CAMPAIGN_ROOT" --max-new-trainings 1 \
  --task-ids B_equal_seed42_panel01 --smoke --submit
```

A non-contiguous retry keeps the exact scientific task IDs/configurations:

```bash
bash submit_reproducibility_reference_campaign_mapped_univie.sh \
  --campaign-root "$CAMPAIGN_ROOT" --max-new-trainings 2 \
  --task-ids B_equal_seed42_panel04,B_ranked_seed42_panel02 --resume --submit
```

Validated completed tasks are skipped; active scheduler tasks are not duplicated.
Interrupted training requires compatible last-state optimizer/scheduler state;
proven post-training export failures use prediction-only retry. No weights-only
fallback or claim of bitwise mid-epoch recovery is introduced.

Analysis is the same partial integration as on Leonardo: complete four-panel
equal/ranked groups within each seed, plus completeness accounting. Full factor,
cross-seed, shuffled, gamma=1 and robustness analyses are not silently claimed
complete. Eight runs are not the complete main campaign.

Job IDs and monitoring commands are printed on successful submission:

```bash
squeue -j <returned-job-ids>
sacct -j <returned-job-ids> --format=JobID,State,ExitCode,Elapsed,AllocTRES
```

## Deployment and validation

New: submit_reproducibility_reference_campaign_mapped_univie.sh,
run_reproducibility_reference_campaign_mapped_univie.slurm,
analyses/launchers/analyze_reproducibility_reference_campaign_mapped_univie.slurm, and this guide.
Updated: campaign_slurm.py and Tests/test_campaign_slurm.py.
Also deploy Utils/univie_campaign_environment.sh, sourced by all three new wrappers.
The CPU submission entrypoint is
submit_reproducibility_reference_campaign_from_compute_univie.slurm.

The shared adapter selects site-specific batch files/defaults; the production
single-task executor, frozen plan and old UNIVIE wrappers are unchanged.
Do not overwrite campaign_slurm.py while existing mapped jobs are pending/running:
submission mappings hash their infrastructure and correctly refuse changes.

Local syntax/help, mocked submission/dependency tests and offline dry-run are
checked. No UNIVIE allocation, GPU smoke or live scheduler validation is claimed.
