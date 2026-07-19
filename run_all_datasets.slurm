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
# 7 dataset-count conditions x 4 quality-rank powers = 28 jobs. At most four
# run concurrently; change only the %4 throttle if the allocation permits it.
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

# p=0 is exactly equal-dataset centering because Q_d**0 = 1. Values below one
# soften the ranking; values above one sharpen it.
QUALITY_RANK_POWERS=(0.0 0.5 1.0 2.0)

# Array mapping (power changes fastest):
#   0-3: N=2, 4-7: N=5, 8-11: N=10, 12-15: N=20,
#   16-19: N=40, 20-23: N=80, 24-27: N=all.

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
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
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

FEATURE_OVERRIDES=()
for feature in "${FEATURE_KEYS[@]}"; do
    FEATURE_OVERRIDES+=(
        "model.additional_sequence_features.${feature}.route=${FEATURE_ROUTES[${feature}]}"
    )
done

# Resolve the requested nested subset from the active 114-dataset YAML and the
# global quality table. The first output line is N; the second is a Hydra list.
SELECTION_OUTPUT="$(python - "${DATASET_COUNT_REQUEST}" "${DATASET_SUBSET_STRATEGY}" <<'PY'
import csv
import sys
from pathlib import Path

import yaml

requested = sys.argv[1]
strategy = sys.argv[2].lower()
manifest_path = Path("config/dataset_config/weighted_hek_riboseq_codon_replicas.yaml")
ranking_path = Path("Datasets/data/HEK_riboseq_profile_quality_rank.tsv")

manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
active = list(manifest["dataset_path"])
with ranking_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

rank_by_dataset = {str(row["dataset"]): float(row["quality_rank"]) for row in rows}
missing = sorted(set(active) - set(rank_by_dataset))
if missing:
    raise SystemExit(f"Active datasets missing from quality ranking: {missing}")

ranked_active = sorted(active, key=lambda name: (rank_by_dataset[name], name))
n = len(ranked_active) if requested.lower() == "all" else int(requested)
if n < 2 or n > len(ranked_active):
    raise SystemExit(f"Dataset count must be in [2, {len(ranked_active)}], got {n}")

if strategy == "top_quality":
    ordered = ranked_active
elif strategy == "rank_stratified":
    # Nested one-dimensional farthest-point ordering over the global ranks.
    # Prefix 2 contains both extremes; every later point splits the largest
    # uncovered rank interval. Ties are resolved toward the better rank.
    selected_indices = [0, len(ranked_active) - 1]
    remaining = set(range(1, len(ranked_active) - 1))
    while remaining:
        next_index = max(
            remaining,
            key=lambda index: (
                min(abs(index - chosen) for chosen in selected_indices),
                -index,
            ),
        )
        selected_indices.append(next_index)
        remaining.remove(next_index)
    ordered = [ranked_active[index] for index in selected_indices]
else:
    raise SystemExit(
        "DATASET_SUBSET_STRATEGY must be rank_stratified or top_quality, "
        f"got {strategy!r}"
    )

selected = ordered[:n]
print(n)
print("[" + ",".join(selected) + "]")
PY
)"
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
RUN_TAG="riboai_${DATASET_SUBSET_STRATEGY}_n${SELECTED_DATASET_COUNT}_qrp${POWER_TAG}_${FEATURE_PRESET}_seed${SEED}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${ARRAY_TASK_ID}"
OUT_DIR="${WORK_DIR}/riboai_runs/${RUN_TAG}"
CKPT_DIR="${OUT_DIR}/checkpoints"
LOG_DIR="${OUT_DIR}/logs"
RESULTS_DIR="${OUT_DIR}/results"

# Python multiprocessing creates an AF_UNIX socket below TMPDIR. Linux limits
# that socket path to about 108 bytes, so keep this path deliberately short.
export TMPDIR="/tmp/riboai_a_${SLURM_JOB_ID:-$$}"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export MPLCONFIGDIR="${SCRATCH_DIR}/.cache/matplotlib"
export TORCH_HOME="${SCRATCH_DIR}/.cache/torch"
export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/.cache/triton"
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
export TORCH_NCCL_ENABLE_MONITORING=0

echo "Starting dataset-count x quality-rank-power ablation"
echo "Array    -> task ${ARRAY_TASK_ID}/$((NUM_EXPERIMENTS - 1))"
echo "Project  -> ${PROJECT_DIR}"
echo "Outputs  -> ${OUT_DIR}"
echo "Scratch  -> ${SCRATCH_DIR}"
echo "Seed     -> ${SEED}"
echo "Datasets -> ${SELECTED_DATASET_COUNT} (${DATASET_SUBSET_STRATEGY})"
echo "Rank p   -> ${QUALITY_RANK_POWER}"
echo "Batch    -> ${BATCH_SIZE} per DDP rank"
echo "Workers  -> ${NUM_WORKERS} per DDP rank"
echo "Features -> ${FEATURE_PRESET}"
echo "GPUs     -> ${CUDA_VISIBLE_DEVICES:-managed by Slurm}"

COMMAND=(
    srun --ntasks=2 python main_ribo_queueing_modeling_multi_dataset.py
    "experiment.dataset=${DATASET_OVERRIDE}"
    "experiment.dataset_subset_strategy=${DATASET_SUBSET_STRATEGY}"
    split.master_dataset_universe=all
    "model.gamma_centering.quality_rank_power=${QUALITY_RANK_POWER}"
    model.gamma_centering.weighting=quality_rank
    model.gamma_centering.enabled=true
    model.gamma_centering.scope=batch_grouped
    "experiment.seed=${SEED}"
    experiment.from_checkpoint=false
    experiment.train=true
    experiment.predict=true
    'trainer.devices=[0,1]'
    trainer.use_distributed_sampler=false
    trainer.sync_dist_logs=true
    "data.batch_size=${BATCH_SIZE}"
    "data.num_workers=${NUM_WORKERS}"
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
