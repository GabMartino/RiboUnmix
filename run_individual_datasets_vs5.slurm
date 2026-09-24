#!/bin/bash
#SBATCH --exclude=dgx1
#SBATCH --job-name=riboai_individual_final
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=job_%j_riboai_individual_final.out
#SBATCH --error=job_%j_riboai_individual_final.err

set -euo pipefail

module load python/3.11

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/../venvs/queuing_ribo_venv}"

if [[ ! -f "${PROJECT_DIR}/main_ribounmix_multidataset.py" ]]; then
    echo "PROJECT_DIR is not the QueueingRiboAI repository: ${PROJECT_DIR}" >&2
    exit 2
fi
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "Python virtual environment not found: ${VENV_PATH}" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
source "${VENV_PATH}/bin/activate"

# All individual models use the same final model/loss configuration as the
# all-dataset run. Only experiment.dataset and the execution device differ.
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Optional-sequence-feature ablation. Supported presets:
#   baseline, biological, dataset_bias, both,
#   exo_bio, gmp_bio, tmp_bio, openen_bio, tai_bio, core_bio
FEATURE_PRESET="${FEATURE_PRESET:-both}"

FEATURE_KEYS=(exo gmp tmp openen tAI_profile_codon fra)
declare -A FEATURE_ROUTES
for feature in "${FEATURE_KEYS[@]}"; do
    FEATURE_ROUTES["${feature}"]="none"
done

case "${FEATURE_PRESET}" in
    baseline)
        ;;
    biological|dataset_bias|both)
        for feature in exo gmp tmp openen tAI_profile_codon; do
            FEATURE_ROUTES["${feature}"]="${FEATURE_PRESET}"
        done
        ;;
    exo_bio) FEATURE_ROUTES[exo]="biological" ;;
    gmp_bio) FEATURE_ROUTES[gmp]="biological" ;;
    tmp_bio) FEATURE_ROUTES[tmp]="biological" ;;
    openen_bio) FEATURE_ROUTES[openen]="biological" ;;
    tai_bio) FEATURE_ROUTES[tAI_profile_codon]="biological" ;;
    core_bio)
        FEATURE_ROUTES[openen]="biological"
        FEATURE_ROUTES[tAI_profile_codon]="biological"
        ;;
    *)
        echo "Unknown FEATURE_PRESET=${FEATURE_PRESET}" >&2
        exit 2
        ;;
esac

FEATURE_OVERRIDES=()
for feature in "${FEATURE_KEYS[@]}"; do
    FEATURE_OVERRIDES+=(
        "model.additional_sequence_features.${feature}.route=${FEATURE_ROUTES[${feature}]}"
    )
done

SCRATCH_DIR="${SCRATCH:-${CINECA_SCRATCH:-/leonardo_scratch/large/userexternal/${USER}}}"
WORK_DIR="${WORK:-/leonardo_work/${SLURM_JOB_ACCOUNT:-euhpc_d35_089}/${USER}}"

RUN_TAG="riboai_individual_${FEATURE_PRESET}_seed${SEED}_${SLURM_JOB_ID:-local}"
OUT_DIR="${WORK_DIR}/riboai_runs/${RUN_TAG}"
CKPT_DIR="${OUT_DIR}/checkpoints"
TB_LOG_DIR="${OUT_DIR}/logs"
RESULTS_DIR="${OUT_DIR}/results"
LOG_DIR="${OUT_DIR}/slurm_logs"

# Use the same active 30-dataset universe as the joint run so single-dataset
# models are evaluated on comparable transcript folds.
SPLIT_MASTER_DATASET_UNIVERSE="${SPLIT_MASTER_DATASET_UNIVERSE:-all}"

export TMPDIR="${SCRATCH_DIR}/tmp/${RUN_TAG}"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export MPLCONFIGDIR="${SCRATCH_DIR}/.cache/matplotlib"
export TORCH_HOME="${SCRATCH_DIR}/.cache/torch"
export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/.cache/triton"
export HYDRA_FULL_ERROR=1
HYDRA_RUN_DIR="${SCRATCH_DIR}/hydra/${RUN_TAG}"

mkdir -p "${CKPT_DIR}" "${TB_LOG_DIR}" "${RESULTS_DIR}" "${LOG_DIR}" \
    "${TMPDIR}" "${MPLCONFIGDIR}" "${TORCH_HOME}" "${HF_HOME}" \
    "${TRITON_CACHE_DIR}" "${HYDRA_RUN_DIR}"

trap 'rm -rf "${TMPDIR}"' EXIT

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "Starting individual-dataset final-model runs"
echo "Project        -> ${PROJECT_DIR}"
echo "Outputs        -> ${OUT_DIR}"
echo "Scratch        -> ${SCRATCH_DIR}"
echo "Seed           -> ${SEED}"
echo "Batch/workers  -> ${BATCH_SIZE}/${NUM_WORKERS} per GPU"
echo "Split universe -> ${SPLIT_MASTER_DATASET_UNIVERSE}"
echo "Feature preset -> ${FEATURE_PRESET}"
echo "Allocated GPUs -> ${CUDA_VISIBLE_DEVICES:-not exposed; using 0,1,2,3}"
nvidia-smi || true

# Four independent queues run concurrently. Each queue processes its assigned
# datasets sequentially on one allocated GPU.
DATASETS_GPU0=(
    "cui_2024"
    "ichihara_2021"
    "iwasaki_2020"
    "kulakovskiy_2024"
    "riepe_2018"
    "song_2019"
    "wang_2016"
    "weber_2022"
)

DATASETS_GPU1=(
    "gillen_2019"
    "iwasaki_2014"
    "iwasaki_2021"
    "kutay_2021"
    "sako_2020"
    "volegova_2018"
    "wangHEK_2022"
    "weber_2024"
)

DATASETS_GPU2=(
    "green_2020"
    "iwasaki_2018"
    "kito_2023"
    "martinez_2019"
    "sauer_2019"
    "wakigawa_2025"
    "weber_2020"
)

DATASETS_GPU3=(
    "grimson_2019"
    "iwasaki_2019"
    "kulakovskiy_2019"
    "sidrauski_2015"
    "wan_2019"
    "wu_2019"
    "wu_2020"
)

# Preserve the GPU identifiers assigned by Slurm. They may be numeric physical
# IDs or UUIDs; each child remaps its one visible device to local CUDA index 0.
IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#ALLOCATED_GPUS[@]} < 4 )); then
    echo "Expected four allocated GPUs, got: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 2
fi

run_queue() {
    local queue_id="$1"
    local visible_gpu="$2"
    shift 2
    local datasets=("$@")

    if (( ${#datasets[@]} == 0 )); then
        echo "[queue ${queue_id}] No datasets assigned."
        return 0
    fi

    echo "[queue ${queue_id}] GPU=${visible_gpu}; datasets=${datasets[*]}"

    for dataset_name in "${datasets[@]}"; do
        local log_file="${LOG_DIR}/run_log_queue${queue_id}_${dataset_name}.txt"

        echo "[queue ${queue_id}] Starting ${dataset_name}; log=${log_file}"
        CUDA_VISIBLE_DEVICES="${visible_gpu}" \
            python main_ribounmix_multidataset.py \
                experiment.dataset="['${dataset_name}']" \
                split.master_dataset_universe="${SPLIT_MASTER_DATASET_UNIVERSE}" \
                experiment.seed="${SEED}" \
                experiment.from_checkpoint=false \
                experiment.train=true \
                experiment.predict=true \
                'trainer.devices=[0]' \
                trainer.use_distributed_sampler=false \
                trainer.sync_dist_logs=false \
                data.batch_size="${BATCH_SIZE}" \
                data.num_workers="${NUM_WORKERS}" \
                "${FEATURE_OVERRIDES[@]}" \
                paths.checkpoints="${CKPT_DIR}" \
                paths.logs="${TB_LOG_DIR}" \
                paths.results="${RESULTS_DIR}" \
                hydra.run.dir="${HYDRA_RUN_DIR}/${dataset_name}" \
                > "${log_file}" 2>&1

        echo "[queue ${queue_id}] Finished ${dataset_name}"
    done
}

pids=()

terminate_queues() {
    echo "Caught interruption; terminating running dataset queues..."
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap terminate_queues INT TERM

run_queue 0 "${ALLOCATED_GPUS[0]}" "${DATASETS_GPU0[@]}" &
pids+=("$!")
run_queue 1 "${ALLOCATED_GPUS[1]}" "${DATASETS_GPU1[@]}" &
pids+=("$!")
run_queue 2 "${ALLOCATED_GPUS[2]}" "${DATASETS_GPU2[@]}" &
pids+=("$!")
run_queue 3 "${ALLOCATED_GPUS[3]}" "${DATASETS_GPU3[@]}" &
pids+=("$!")

echo "Started queue PIDs: ${pids[*]}"

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        echo "Queue process ${pid} failed."
        status=1
    fi
done

if (( status != 0 )); then
    echo "At least one GPU queue failed. Check ${LOG_DIR}." >&2
    exit "${status}"
fi

echo "All individual-dataset runs finished successfully."
