# Benchmark loss ablation

This experiment tests the contribution of the two consensus-correlation terms
in the four single-dataset organism benchmarks.  It uses four matched arms:

| arm | replica NB2 | raw-consensus PCC | NB-VST-consensus PCC |
|---|---:|---:|---:|
| `full` | 1.0 | 0.5 | 0.5 |
| `nb_only` | 1.0 | 0.0 | 0.0 |
| `nb_raw_pcc` | 1.0 | 0.5 | 0.0 |
| `nb_vst_pcc` | 1.0 | 0.0 | 0.5 |

The gamma regularizer remains `1e-4` in every arm. Thus `nb_only` means the
configured replica-NB term is the only **data-fit** term; removing the regularizer
would be a second intervention. Every arm also retains
`experiment_mode=mean_gradient_reweighted_nb` and
`nb_mean_gradient_beta=0.5`: this experiment removes the PCC terms and does not
ablate the NB mean-gradient reweighting. Every organism has one observed profile per transcript, represented
as one replica, so the arithmetic consensus equals that profile. This experiment
does not test multi-replicate aggregation.

The full arm is rerun under the same current code as the ablations. The older
`benchmark_20260829_194806` outputs are not used as matched controls.

## Inspect the frozen task mapping

```bash
.venv/bin/python run_benchmark_loss_ablation.py \
  --training-seeds 42 --list-tasks
```

Indices `0-15` are one training seed: four loss arms by four datasets. The split
seed is fixed at 42. If training seeds `42,43,44` are requested, indices `0-47`
are used while the organism-specific train/validation/test IDs remain fixed.

## UNIVIE

One-seed screening campaign (16 trainings):

```bash
TRAINING_SEEDS=42 \
sbatch --array=0-15%4 run_benchmark_loss_ablation_univie.slurm
```

Three-seed manuscript campaign (48 trainings):

```bash
TRAINING_SEEDS=42,43,44 \
sbatch --array=0-47%4 run_benchmark_loss_ablation_univie.slurm
```

Each array element receives one GPU and launches exactly one model. Resubmitting
the same array skips tasks whose predictions have passed completion checks.
Interrupted tasks start a fresh attempt because the benchmark currently writes
weights-only checkpoints; they are not described as exact optimizer-state
resumes.

## Leonardo

The launcher selects the project `.venv` when present and otherwise uses
`$HOME/venvs/queuing_ribo_venv`. Override this explicitly when the populated
production environment is elsewhere:

```bash
export VENV_PATH="$HOME/venvs/queuing_ribo_venv"
```

Before the first submission, verify the exact interpreter and Hydra import:

```bash
module load python/3.11
"$VENV_PATH/bin/python" -c \
  'import hydra,lightning,torch; print(hydra.__version__, lightning.__version__, torch.__version__)'
```

If that command fails, populate the environment once in a CPU setup job:

```bash
PROJECT_DIR="$PWD" VENV_PATH="$VENV_PATH" \
  sbatch slurm_python_install_new_packages.slurm
```

The setup job targets Leonardo's Booster partition but deliberately does not
hardcode a project account. Obtain the currently active association with
`sacctmgr` and pass it to `sbatch`; do not copy an expired project account from
an old launcher. The setup requests one GPU only because the available project
accounts are not associated with the default `lrd_all_serial` partition; it
does not perform GPU computation.

```bash
sacctmgr -nP show assoc where user="$USER" \
  format=Cluster,Account,Partition,QOS

sbatch --account="$LEONARDO_ACCOUNT" \
  --export=ALL,PROJECT_DIR="$PWD",VENV_PATH="$VENV_PATH" \
  slurm_python_install_new_packages.slurm
```

After that setup job completes successfully, submit the one-seed campaign:

```bash
VENV_PATH="$VENV_PATH" TRAINING_SEEDS=42 \
  sbatch --account="$LEONARDO_ACCOUNT" --array=0-15%4 \
  run_benchmark_loss_ablation.slurm
```

The training launcher never installs packages inside a GPU job. It validates
the environment before creating an experiment attempt, and both the array
adapter and production trainer use the same absolute `VENV_PATH/bin/python`.

## Analysis

For one seed:

```bash
RIBOUNMIX_PLOT_TEX=0 .venv/bin/python \
  analyses/analyze_benchmark_loss_ablation.py \
  --experiment-root results/riboai_benchmarking_experiments/loss_ablation_v1 \
  --training-seeds 42 --require-complete
```

For three seeds:

```bash
RIBOUNMIX_PLOT_TEX=0 .venv/bin/python \
  analyses/analyze_benchmark_loss_ablation.py \
  --experiment-root results/riboai_benchmarking_experiments/loss_ablation_v1 \
  --training-seeds 42,43,44 --require-complete
```

The primary analysis uses `best_nb_nll`, because raw validation NB2 NLL has the
same definition in every arm. `best_val_loss` is objective-specific, and
`best_pcc` directly selects for one of the outcomes; both remain available as
declared sensitivity analyses through `--checkpoint-variant`.

## Direct two-GPU server

`run_benchmark_loss_ablation_2gpu.sh` runs two independent queues: one model on
each GPU, with tasks sequential within a GPU. It does not use DDP and does not
change the logical batch size or scientific configuration.

```bash
nohup env \
  VENV_PATH="$PWD/.venv" \
  GPU_DEVICES=0,1 \
  TRAINING_SEEDS=42 \
  OUTPUT_ROOT="$PWD/results/riboai_benchmarking_experiments/loss_ablation_v1" \
  bash run_benchmark_loss_ablation_2gpu.sh \
  > benchmark_loss_ablation_2gpu.nohup.log 2>&1 &

echo $! > benchmark_loss_ablation_2gpu.pid
```

Resubmitting the same command is safe: tasks with validated checkpoint exports
are skipped by the existing task runner. Per-task logs and the frozen task table
are written under `OUTPUT_ROOT/direct_launcher_logs/RUN_TAG/`. The launcher
stops both queues after the first task failure and copies the last 100 log lines
to the master log. Set `CONTINUE_ON_FAILURE=1` only when failures are known to
be task-specific rather than a shared configuration problem.

Outputs include exact per-transcript metrics, matched configuration checks,
paired effect tables, exclusions, run provenance, a caption, and PDF/SVG/600-dpi
PNG figures. The generated `benchmark_loss_ablation_report.html` includes an
explicit run-availability matrix and remains labeled partial until the requested
matrix is complete. Bootstrap intervals resample matched transcripts and are
conditional on the fitted models and selected benchmark datasets.
