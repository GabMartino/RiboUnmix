#!/bin/bash
#SBATCH --job-name=riboai_all_final
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d35_089
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=job_%j_riboai_all_final.out
#SBATCH --error=job_%j_riboai_all_final.err

set -euo pipefail

module load python/3.11

# Submit from the repository root, or provide PROJECT_DIR/VENV_PATH explicitly:
#   sbatch --export=ALL,PROJECT_DIR=/path/to/QueueingRiboAI run_all_datasets.slurm
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/../venvs/queuing_ribo_venv}"

if [[ ! -f "${PROJECT_DIR}/main_ribo_queueing_modeling_multi_dataset.py" ]]; then
    echo "PROJECT_DIR is not the QueueingRiboAI repository: ${PROJECT_DIR}" >&2
    exit 2
fi
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "Python virtual environment not found: ${VENV_PATH}" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
source "${VENV_PATH}/bin/activate"

# Model and loss settings are inherited unchanged from
# config/config_riboai_queuing_multidataset.yaml. These environment variables
# control only execution resources and reproducibility.
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Persistent outputs go to $WORK; transient caches and Hydra files go to
# $SCRATCH. Avoid writing large files to the small home quota.
SCRATCH_DIR="${SCRATCH:-${CINECA_SCRATCH:-/leonardo_scratch/large/userexternal/${USER}}}"
WORK_DIR="${WORK:-/leonardo_work/${SLURM_JOB_ACCOUNT:-euhpc_d35_089}/${USER}}"

RUN_TAG="riboai_all_final_seed${SEED}_${SLURM_JOB_ID:-local}"
OUT_DIR="${WORK_DIR}/riboai_runs/${RUN_TAG}"
CKPT_DIR="${OUT_DIR}/checkpoints"
LOG_DIR="${OUT_DIR}/logs"
RESULTS_DIR="${OUT_DIR}/results"

export TMPDIR="${SCRATCH_DIR}/tmp/${RUN_TAG}"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export MPLCONFIGDIR="${SCRATCH_DIR}/.cache/matplotlib"
export TORCH_HOME="${SCRATCH_DIR}/.cache/torch"
export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/.cache/triton"
HYDRA_RUN_DIR="${SCRATCH_DIR}/hydra/${RUN_TAG}"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${RESULTS_DIR}" "${TMPDIR}" \
    "${MPLCONFIGDIR}" "${TORCH_HOME}" "${HF_HOME}" \
    "${TRITON_CACHE_DIR}" "${HYDRA_RUN_DIR}"

trap 'rm -rf "${TMPDIR}"' EXIT

# Lightning detects the two externally launched Slurm tasks as its two DDP
# ranks. The entry point uses SLURM_PROCID to keep rank-zero writes and logger
# versions coordinated.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ENABLE_MONITORING=0

echo "Starting all-dataset final-model run"
echo "Project  -> ${PROJECT_DIR}"
echo "Outputs  -> ${OUT_DIR}"
echo "Scratch  -> ${SCRATCH_DIR}"
echo "Seed     -> ${SEED}"
echo "Batch    -> ${BATCH_SIZE} per DDP rank"
echo "Workers  -> ${NUM_WORKERS} per DDP rank"
echo "GPUs     -> ${CUDA_VISIBLE_DEVICES:-managed by Slurm}"

srun --ntasks=2 python main_ribo_queueing_modeling_multi_dataset.py \
    experiment.dataset=all \
    split.master_dataset_universe=all \
    experiment.seed="${SEED}" \
    experiment.from_checkpoint=false \
    experiment.train=true \
    experiment.predict=true \
    'trainer.devices=[0,1]' \
    trainer.use_distributed_sampler=false \
    trainer.sync_dist_logs=true \
    data.batch_size="${BATCH_SIZE}" \
    data.num_workers="${NUM_WORKERS}" \
    paths.checkpoints="${CKPT_DIR}" \
    paths.logs="${LOG_DIR}" \
    paths.results="${RESULTS_DIR}" \
    hydra.run.dir="${HYDRA_RUN_DIR}"
