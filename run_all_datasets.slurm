#!/bin/bash
#SBATCH --job-name=riboai_all_datasets
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d35_089
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
# 2 x 2 x 2 x 2 ablation grid. The %4 throttle limits concurrent tasks.
#SBATCH --array=0-15%4
#SBATCH --output=job_%A_%a_riboai_all_datasets.out
#SBATCH --error=job_%A_%a_riboai_all_datasets.err

set -euo pipefail

module load python/3.11
source ../venvs/queuing_ribo_venv/bin/activate

# ---------------------------------------------------------------------------
# Ablation grid: one Slurm array task per configuration.
#
# Dimensions:
#   1. CAGrad: off/on
#   2. Dataset-balanced loss: off/on
#   3. Replica objective: consensus only / consensus + replica auxiliary loss
#   4. PCC loss: raw PCC / raw PCC + NB variance-adjusted PCC
#
# The Python entry point adds these choices to the run tag, so logs, results and
# checkpoints are separated under the common output root.
# ---------------------------------------------------------------------------
CAGRAD_OPTIONS=(false true)
DATASET_BALANCED_OPTIONS=(false true)
REPLICA_OBJECTIVE_OPTIONS=(consensus consensus_plus_replica)
PCC_MODE_OPTIONS=(raw hybrid_raw_nb_vst)
PCC_LABELS=(raw raw_plus_varadj)
PCC_RAW_WEIGHTS=(1.0 0.2)
PCC_NB_VST_WEIGHTS=(0.0 0.8)

TOTAL_ABLATIONS=$(( \
    ${#CAGRAD_OPTIONS[@]} \
    * ${#DATASET_BALANCED_OPTIONS[@]} \
    * ${#REPLICA_OBJECTIVE_OPTIONS[@]} \
    * ${#PCC_MODE_OPTIONS[@]} \
))

ABLATION_ID="${SLURM_ARRAY_TASK_ID:-${ABLATION_ID:-0}}"
if ! [[ "${ABLATION_ID}" =~ ^[0-9]+$ ]]; then
    echo "ABLATION_ID must be an integer, got: ${ABLATION_ID}" >&2
    exit 2
fi
if (( ABLATION_ID < 0 || ABLATION_ID >= TOTAL_ABLATIONS )); then
    echo "ABLATION_ID ${ABLATION_ID} outside valid range 0-$((TOTAL_ABLATIONS - 1))." >&2
    exit 2
fi

idx="${ABLATION_ID}"
pcc_idx=$((idx % ${#PCC_MODE_OPTIONS[@]}))
idx=$((idx / ${#PCC_MODE_OPTIONS[@]}))

replica_idx=$((idx % ${#REPLICA_OBJECTIVE_OPTIONS[@]}))
idx=$((idx / ${#REPLICA_OBJECTIVE_OPTIONS[@]}))

dataset_balanced_idx=$((idx % ${#DATASET_BALANCED_OPTIONS[@]}))
idx=$((idx / ${#DATASET_BALANCED_OPTIONS[@]}))

cagrad_idx=$((idx % ${#CAGRAD_OPTIONS[@]}))

CAGRAD_ENABLED="${CAGRAD_OPTIONS[${cagrad_idx}]}"
DATASET_BALANCED_LOSS="${DATASET_BALANCED_OPTIONS[${dataset_balanced_idx}]}"
REPLICA_OBJECTIVE="${REPLICA_OBJECTIVE_OPTIONS[${replica_idx}]}"
PCC_MODE="${PCC_MODE_OPTIONS[${pcc_idx}]}"
PCC_LABEL="${PCC_LABELS[${pcc_idx}]}"
PCC_RAW_WEIGHT="${PCC_RAW_WEIGHTS[${pcc_idx}]}"
PCC_NB_VST_WEIGHT="${PCC_NB_VST_WEIGHTS[${pcc_idx}]}"

# ---------------------------------------------------------------------------
# Storage: NEVER write to $HOME (small byte + inode quota -> Errno 122).
# Persistent outputs you want to keep  -> $WORK  (project area, not purged).
# Transient/regenerable files (tmp, caches, hydra run dir) -> $SCRATCH (purged).
# ---------------------------------------------------------------------------
SCRATCH_DIR="${SCRATCH:-${CINECA_SCRATCH:-/leonardo_scratch/large/userexternal/${USER}}}"
WORK_DIR="${WORK:-/leonardo_work/${SLURM_JOB_ACCOUNT:-euhpc_d35_089}/${USER}}"

RUN_GROUP="riboai_all_datasets_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
RUN_TAG="${RUN_GROUP}_abl${ABLATION_ID}"
OUT_DIR="${WORK_DIR}/riboai_runs/${RUN_GROUP}"
CKPT_DIR="${OUT_DIR}/checkpoints"
LOG_DIR="${OUT_DIR}/logs"
RESULTS_DIR="${OUT_DIR}/results"

# Caches, scratch tmp and Hydra's per-run output dir -> scratch (off $HOME).
export TMPDIR="${SCRATCH_DIR}/tmp/${RUN_TAG}"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export MPLCONFIGDIR="${SCRATCH_DIR}/.cache/matplotlib"
export TORCH_HOME="${SCRATCH_DIR}/.cache/torch"
export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/.cache/triton"
HYDRA_RUN_DIR="${SCRATCH_DIR}/hydra/${RUN_TAG}"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${RESULTS_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}" \
         "${TORCH_HOME}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${HYDRA_RUN_DIR}"

# Remove the scratch tmp dir on exit so it does not linger between jobs.
trap 'rm -rf "${TMPDIR}"' EXIT

# ---------------------------------------------------------------------------
# Threads: 2 tasks x 8 cpus = 16 cores. Keep BLAS/OpenMP
# from oversubscribing on top of the DataLoader workers.
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

# Silence the benign NCCL heartbeat/TCPStore "Broken pipe" warning at teardown.
export TORCH_NCCL_ENABLE_MONITORING=0

echo "Outputs  -> ${OUT_DIR}"
echo "Scratch  -> ${SCRATCH_DIR} (tmp/caches/hydra)"
echo "Ablation -> ${ABLATION_ID}/$((TOTAL_ABLATIONS - 1))"
echo "             cagrad.enabled=${CAGRAD_ENABLED}"
echo "             loss.dataset_balanced_loss=${DATASET_BALANCED_LOSS}"
echo "             loss.replica_objective=${REPLICA_OBJECTIVE}"
echo "             pcc=${PCC_LABEL} (${PCC_MODE})"
echo "             pcc_raw_weight=${PCC_RAW_WEIGHT}"
echo "             pcc_nb_vst_weight=${PCC_NB_VST_WEIGHT}"

srun python main_ribo_queueing_modeling_multi_dataset.py \
    experiment.dataset=all \
    split.master_dataset_universe=all \
    cagrad.enabled="${CAGRAD_ENABLED}" \
    loss.dataset_balanced_loss="${DATASET_BALANCED_LOSS}" \
    loss.replica_objective="${REPLICA_OBJECTIVE}" \
    loss.pcc_loss_enabled=true \
    loss.pcc_loss_mode="${PCC_MODE}" \
    loss.pcc_raw_component_weight="${PCC_RAW_WEIGHT}" \
    loss.pcc_nb_vst_component_weight="${PCC_NB_VST_WEIGHT}" \
    'trainer.devices=[0,1]' \
    data.batch_size=128 \
    data.num_workers=8 \
    paths.checkpoints="${CKPT_DIR}" \
    paths.logs="${LOG_DIR}" \
    paths.results="${RESULTS_DIR}" \
    hydra.run.dir="${HYDRA_RUN_DIR}"
