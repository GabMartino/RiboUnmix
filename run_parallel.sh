#!/bin/bash

set -uo pipefail

# Split 1: 16 datasets assigned to GPU 0
DATASETS_GPU0=(
    "cui_2024" "gillen_2019" "green_2020" "grimson_2019" "ichihara_2021"
    "iwasaki_2014" "iwasaki_2018" "iwasaki_2019" "iwasaki_2020"
    "iwasaki_2021" "kito_2023" "kulakovskiy_2019" "kulakovskiy_2024" "kutay_2021"
    "martinez_2019" "riepe_2018"
)

# Split 2: 14 datasets assigned to GPU 1
DATASETS_GPU1=(
    "sako_2020" "sauer_2019" "sidrauski_2015" "song_2019"
    "volegova_2018" "wakigawa_2025" "wan_2019" "wang_2016" "wangHEK_2022"
    "weber_2020" "weber_2022" "weber_2024" "wu_2019" "wu_2020"
)

# Function to process a queue of datasets on a specific GPU
run_queue() {
    local gpu_id=$1
    shift
    local datasets=("$@")

    # This is the critical step. It restricts PyTorch Lightning to ONLY see this specific GPU.
    export CUDA_VISIBLE_DEVICES=$gpu_id

    for DS in "${datasets[@]}"; do
        echo "[Queue GPU $gpu_id] Starting $DS..."

        # We explicitly enforce trainer.devices=[0] here because to this isolated process,
        # its assigned GPU is always index 0.
        # We pipe the output to a unique log file so the outputs don't interleave in the terminal.
        python main_ribo_queueing_modeling_multi_dataset.py \
            cagrad.enabled=false \
            experiment.dataset="['$DS']" \
            split.master_dataset_universe="['$DS']" \
            'trainer.devices=[0]' \
            > "run_log_GPU${gpu_id}_${DS}.txt" 2>&1

        if [ $? -ne 0 ]; then
            echo "[Queue GPU $gpu_id] ERROR: Failed on $DS. Halting this queue."
            exit 1
        fi

        echo "[Queue GPU $gpu_id] Finished $DS successfully."
    done

    echo "[Queue GPU $gpu_id] All assigned datasets completed!"
}

echo "Launching parallel training pipelines..."

# Launch GPU 0 queue in the background (using &)
run_queue 0 "${DATASETS_GPU0[@]}" &
PID0=$!

# Launch GPU 1 queue in the background (using &)
run_queue 1 "${DATASETS_GPU1[@]}" &
PID1=$!

echo "Pipelines launched in the background."
echo "- GPU 0 Process ID: $PID0"
echo "- GPU 1 Process ID: $PID1"

# The 'wait' commands keep the master script alive until both background queues finish
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1

if [ "$STATUS" -ne 0 ]; then
    echo "Master script failed: at least one GPU queue stopped with an error."
    exit "$STATUS"
fi

echo "Master script complete: All parallel processing finished."
