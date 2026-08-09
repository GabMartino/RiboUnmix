#!/bin/bash
#SBATCH --job-name=riboai_qrank_ablation
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d35_089
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
# With quality-rank weighting this is 7 dataset-count conditions x 4 powers =
# 28 jobs. With equal weighting the power is irrelevant and the grid becomes
# 7 jobs (one per dataset count). The allocated GPUs/CPUs/memory remain
# exclusive to each job, but the unused part of a node may host another job.
# For the currently incomplete baseline conditions only, submit with:
#   sbatch --array=20-27%4 \
#     --export=ALL,FEATURE_PRESET=baseline,NUM_WORKERS=0,PREDICT_NUM_WORKERS=0 \
#     run_all_datasets.slurm
#SBATCH --array=0-27%4
#SBATCH --output=job_%A_%a_riboai_qrank_ablation.out
#SBATCH --error=job_%A_%a_riboai_qrank_ablation.err

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

# ---------------------------------------------------------------------------
# Ablation grid
# ---------------------------------------------------------------------------
# The default nested rank-stratified subsets start with the best and worst
# active datasets, then repeatedly add the midpoint of the largest remaining
# rank gap. This spans the quality range at every N, keeps smaller subsets
# inside larger ones, and makes quality-rank-power effects measurable.
# Set DATASET_SUBSET_STRATEGY=top_quality to instead use the best N datasets.
# `all` resolves dynamically to every active dataset in the dataset YAML.
DATASET_COUNTS=(2 5 10 20 40 80 all)
DATASET_SUBSET_STRATEGY="${DATASET_SUBSET_STRATEGY:-rank_stratified}"

# `quality_rank` uses Q_d^p; `equal` gives every active dataset weight 1.
# Equal weighting has no meaningful power ablation, so it uses p=0 only.
GAMMA_CENTERING_WEIGHTING="${GAMMA_CENTERING_WEIGHTING:-quality_rank}"
case "${GAMMA_CENTERING_WEIGHTING}" in
    quality_rank)
        QUALITY_RANK_POWERS=(0.0 0.5 1.0 2.0)
        ;;
    equal)
        QUALITY_RANK_POWERS=(0.0)
        ;;
    *)
        echo "GAMMA_CENTERING_WEIGHTING must be quality_rank or equal, got '${GAMMA_CENTERING_WEIGHTING}'." >&2
        exit 2
        ;;
esac

# p=0 is exactly equal-dataset centering because Q_d**0 = 1. Values below one
# soften the ranking; values above one sharpen it.

# Array mapping for quality_rank (power changes fastest):
#   0-3: N=2, 4-7: N=5, 8-11: N=10, 12-15: N=20,
#   16-19: N=40, 20-23: N=80, 24-27: N=all.
# For equal weighting, the single power p=0 gives tasks 0:N=2, 1:N=5,
# 2:N=10, 3:N=20, 4:N=40, 5:N=80, 6:N=all.

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_POWERS="${#QUALITY_RANK_POWERS[@]}"
NUM_COUNTS="${#DATASET_COUNTS[@]}"
NUM_EXPERIMENTS=$((NUM_POWERS * NUM_COUNTS))
if (( ARRAY_TASK_ID < 0 || ARRAY_TASK_ID >= NUM_EXPERIMENTS )); then
    echo "SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID} is outside 0-$((NUM_EXPERIMENTS - 1))." >&2
    exit 2
fi

COUNT_INDEX=$((ARRAY_TASK_ID / NUM_POWERS))
POWER_INDEX=$((ARRAY_TASK_ID % NUM_POWERS))
DATASET_COUNT_REQUEST="${DATASET_COUNTS[COUNT_INDEX]}"
QUALITY_RANK_POWER="${QUALITY_RANK_POWERS[POWER_INDEX]}"

# Model and loss settings not explicitly overridden below are inherited from
# config/config_riboai_queuing_multidataset.yaml.
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TRAIN_SAMPLING_STRATEGY="${TRAIN_SAMPLING_STRATEGY:-transcript_grouped_multidataset_pairs}"
# One persistent worker per DDP rank retains background loading while limiting
# the in-memory dataset copies made by CUDA-safe `spawn`. Prediction defaults to
# the main DDP process because it runs once and does not benefit from persistent
# workers enough to justify another dataset copy.
NUM_WORKERS="${NUM_WORKERS:-1}"
PREDICT_NUM_WORKERS="${PREDICT_NUM_WORKERS:-0}"
DATA_WORKER_CONTEXT="${DATA_WORKER_CONTEXT:-spawn}"
# Optional-sequence-feature ablation. Submit, for example:
#   sbatch --export=ALL,FEATURE_PRESET=baseline run_all_datasets.slurm
# Supported: baseline, biological, dataset_bias, both, exo_bio, gmp_bio,
# tmp_bio, openen_bio, tai_bio, core_bio.
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

# These historical transcript_grouped_pairs conditions were verified in the
# repository's results/ directory: each
# has a non-empty last.ckpt and readable main-validation and CSS prediction
# Parquets. Keep the full array declaration generic for other feature presets,
# but only skip them when TRAIN_SAMPLING_STRATEGY=transcript_grouped_pairs.
# The new grouped-multidataset optimizer-batching runs are distinct.
# Set SKIP_COMPLETED=0 explicitly to rerun one of these conditions.
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
if [[ "${SKIP_COMPLETED}" != "0" && "${SKIP_COMPLETED}" != "1" ]]; then
    echo "SKIP_COMPLETED must be 0 or 1, got '${SKIP_COMPLETED}'." >&2
    exit 2
fi

declare -A COMPLETED_CONDITIONS=(
    ["rank_stratified|2|0.0|baseline|42"]="riboai_rank_stratified_n2_qrp0p0_baseline_seed42_49833044_0"
    ["rank_stratified|2|0.5|baseline|42"]="riboai_rank_stratified_n2_qrp0p5_baseline_seed42_49859890_1"
    ["rank_stratified|2|1.0|baseline|42"]="riboai_rank_stratified_n2_qrp1p0_baseline_seed42_49859890_2"
    ["rank_stratified|2|2.0|baseline|42"]="riboai_rank_stratified_n2_qrp2p0_baseline_seed42_49859890_3"
    ["rank_stratified|5|0.0|baseline|42"]="riboai_rank_stratified_n5_qrp0p0_baseline_seed42_49833044_4"
    ["rank_stratified|5|0.5|baseline|42"]="riboai_rank_stratified_n5_qrp0p5_baseline_seed42_49833044_5"
    ["rank_stratified|5|1.0|baseline|42"]="riboai_rank_stratified_n5_qrp1p0_baseline_seed42_49833044_6"
    ["rank_stratified|5|2.0|baseline|42"]="riboai_rank_stratified_n5_qrp2p0_baseline_seed42_49833044_7"
    ["rank_stratified|10|0.5|baseline|42"]="riboai_rank_stratified_n10_qrp0p5_baseline_seed42_49833044_9"
    ["rank_stratified|10|0.0|baseline|42"]="riboai_rank_stratified_n10_qrp0p0_baseline_seed42_49909320_8"
    ["rank_stratified|10|1.0|baseline|42"]="riboai_rank_stratified_n10_qrp1p0_baseline_seed42_49909320_10"
    ["rank_stratified|10|2.0|baseline|42"]="riboai_rank_stratified_n10_qrp2p0_baseline_seed42_49833044_11"
    ["rank_stratified|20|0.0|baseline|42"]="riboai_rank_stratified_n20_qrp0p0_baseline_seed42_49833044_12"
    ["rank_stratified|20|0.5|baseline|42"]="riboai_rank_stratified_n20_qrp0p5_baseline_seed42_49833044_13"
    ["rank_stratified|20|1.0|baseline|42"]="riboai_rank_stratified_n20_qrp1p0_baseline_seed42_49909320_14"
    ["rank_stratified|20|2.0|baseline|42"]="riboai_rank_stratified_n20_qrp2p0_baseline_seed42_49909320_15"
    ["rank_stratified|40|0.0|baseline|42"]="riboai_rank_stratified_n40_qrp0p0_baseline_seed42_49833044_16"
    ["rank_stratified|40|0.5|baseline|42"]="riboai_rank_stratified_n40_qrp0p5_baseline_seed42_49909320_17"
    ["rank_stratified|40|1.0|baseline|42"]="riboai_rank_stratified_n40_qrp1p0_baseline_seed42_49833044_18"
    ["rank_stratified|40|2.0|baseline|42"]="riboai_rank_stratified_n40_qrp2p0_baseline_seed42_49833044_19"
)

CURRENT_CONDITION="${DATASET_SUBSET_STRATEGY}|${DATASET_COUNT_REQUEST}|${QUALITY_RANK_POWER}|${FEATURE_PRESET}|${SEED}"
COMPLETED_RUN="${COMPLETED_CONDITIONS[${CURRENT_CONDITION}]-}"
# The completion table describes quality-rank runs only. Equal weighting is a
# distinct experiment even when it uses the same p=0 dataset subset.
if [[ "${SKIP_COMPLETED}" == "1" && "${TRAIN_SAMPLING_STRATEGY}" == "transcript_grouped_pairs" && "${GAMMA_CENTERING_WEIGHTING}" == "quality_rank" && -n "${COMPLETED_RUN}" ]]; then
    echo "Skipping completed condition: ${CURRENT_CONDITION}"
    echo "Verified result: results/${COMPLETED_RUN}"
    exit 0
fi

FEATURE_OVERRIDES=()
for feature in "${FEATURE_KEYS[@]}"; do
    FEATURE_OVERRIDES+=(
        "model.additional_sequence_features.${feature}.route=${FEATURE_ROUTES[${feature}]}"
    )
done

# Resolve the requested nested subset from the active 114-dataset YAML and the
# global quality table. A regular Python file avoids Bash creating a temporary
# here-document on the node-local filesystem. The first output line is N; the
# second is a Hydra list.
SELECTION_OUTPUT="$(python Utils/select_dataset_ablation_subset.py \
    "${DATASET_COUNT_REQUEST}" "${DATASET_SUBSET_STRATEGY}")"
SELECTED_DATASET_COUNT="${SELECTION_OUTPUT%%$'\n'*}"
DATASET_OVERRIDE="${SELECTION_OUTPUT#*$'\n'}"

if [[ -z "${DATASET_OVERRIDE}" || "${DATASET_OVERRIDE}" == "${SELECTION_OUTPUT}" ]]; then
    echo "Failed to construct the dataset subset." >&2
    exit 2
fi

# Persistent outputs go to $WORK; transient caches and Hydra files go to
# $SCRATCH. Avoid writing large files to the small home quota.
SCRATCH_DIR="${SCRATCH:-${CINECA_SCRATCH:-/leonardo_scratch/large/userexternal/${USER}}}"
WORK_DIR="${WORK:-/leonardo_work/${SLURM_JOB_ACCOUNT:-euhpc_d35_089}/${USER}}"

POWER_TAG="${QUALITY_RANK_POWER//./p}"
WEIGHTING_TAG="${GAMMA_CENTERING_WEIGHTING}"
if [[ "${GAMMA_CENTERING_WEIGHTING}" == "quality_rank" ]]; then
    WEIGHTING_TAG="qrank"
else
    WEIGHTING_TAG="equal"
fi
RUN_TAG="riboai_${DATASET_SUBSET_STRATEGY}_n${SELECTED_DATASET_COUNT}_gw${WEIGHTING_TAG}_qrp${POWER_TAG}_${FEATURE_PRESET}_seed${SEED}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${ARRAY_TASK_ID}"
OUT_DIR="${WORK_DIR}/riboai_runs/${RUN_TAG}"
CKPT_DIR="${OUT_DIR}/checkpoints"
LOG_DIR="${OUT_DIR}/logs"
RESULTS_DIR="${OUT_DIR}/results"

# Python multiprocessing creates an AF_UNIX socket below TMPDIR. Linux limits
# that socket path to about 108 bytes, so keep this path deliberately short.
export TMPDIR="/tmp/riboai_a_${SLURM_JOB_ID:-$$}"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export TORCH_HOME="${SCRATCH_DIR}/.cache/torch"
export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
# These caches may be written during a run, so keep them private to the array
# element instead of allowing concurrent jobs to update the same files.
export MPLCONFIGDIR="${TMPDIR}/matplotlib"
export TRITON_CACHE_DIR="${TMPDIR}/triton"
export HYDRA_FULL_ERROR=1
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
# Avoid initializing the CUDA runtime merely to answer an availability check;
# this is safer if workers are explicitly re-enabled. If either DDP rank fails,
# ask NCCL and Slurm to tear down its peer instead of waiting for a long timeout.
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-300}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "Starting dataset-count x quality-rank-power ablation"
echo "Array    -> task ${ARRAY_TASK_ID}/$((NUM_EXPERIMENTS - 1))"
echo "Project  -> ${PROJECT_DIR}"
echo "Outputs  -> ${OUT_DIR}"
echo "Scratch  -> ${SCRATCH_DIR}"
echo "Seed     -> ${SEED}"
echo "Datasets -> ${SELECTED_DATASET_COUNT} (${DATASET_SUBSET_STRATEGY})"
echo "Rank p   -> ${QUALITY_RANK_POWER}"
echo "Gamma wt -> ${GAMMA_CENTERING_WEIGHTING}"
echo "Batch    -> ${BATCH_SIZE} pair rows per dataset per DDP-rank microbatch"
echo "Pair cap -> $((BATCH_SIZE * SELECTED_DATASET_COUNT)) nominal rows per rank"
echo "Sampling -> ${TRAIN_SAMPLING_STRATEGY}"
echo "Workers  -> ${NUM_WORKERS} per DDP rank"
echo "Pred work-> ${PREDICT_NUM_WORKERS} per DDP rank"
echo "Features -> ${FEATURE_PRESET}"
echo "GPUs     -> ${CUDA_VISIBLE_DEVICES:-managed by Slurm}"
echo "Node     -> ${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-unknown}}"
echo "Worker mp-> ${DATA_WORKER_CONTEXT}"

# Fail before training if the allocated node cannot initialize the GPU driver.
# Skip this only for the local DRY_RUN validation path.
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader
fi

COMMAND=(
    srun --ntasks=2 --kill-on-bad-exit=1 --wait=60 \
    python main_ribo_queueing_modeling_multi_dataset.py
    "experiment.dataset=${DATASET_OVERRIDE}"
    split.master_dataset_universe=all
    "model.gamma_centering.reference.quality_rank_power=${QUALITY_RANK_POWER}"
    "model.gamma_centering.reference.weighting=${GAMMA_CENTERING_WEIGHTING}"
    model.gamma_centering.mode=fixed_reference
    "experiment.seed=${SEED}"
    experiment.from_checkpoint=false
    experiment.train=true
    experiment.predict=true
    'trainer.devices=[0,1]'
    trainer.use_distributed_sampler=false
    trainer.sync_dist_logs=true
    "data.batch_size=${BATCH_SIZE}"
    "data.train_sampling_strategy=${TRAIN_SAMPLING_STRATEGY}"
    training.grouped_optimizer_batch.enabled=true
    training.grouped_optimizer_batch.auto_accumulate_grad_batches=true
    training.grouped_optimizer_batch.target_unique_transcripts_per_optimizer_step=32
    "data.num_workers=${NUM_WORKERS}"
    "data.predict_num_workers=${PREDICT_NUM_WORKERS}"
    "data.multiprocessing_context=${DATA_WORKER_CONTEXT}"
    "${FEATURE_OVERRIDES[@]}"
    "paths.checkpoints=${CKPT_DIR}"
    "paths.logs=${LOG_DIR}"
    "paths.results=${RESULTS_DIR}"
    "hydra.run.dir=${HYDRA_RUN_DIR}"
)

# Useful for checking every array condition without launching training:
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash run_all_datasets.slurm
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'Command  -> '
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

"${COMMAND[@]}"
