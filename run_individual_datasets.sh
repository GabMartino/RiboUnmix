#!/bin/bash

set -e

# Split 1: 17 datasets assigned to GPU 0
DATASETS_GPU0=(
    #"cui_2024" "gillen_2019" "green_2020" "grimson_2019" "ichihara_2021"
    #"ingolia_2014"
    #"iwasaki_2014" "iwasaki_2018" "iwasaki_2019" "iwasaki_2020"
    #"iwasaki_2021" "kito_2023" "kulakovskiy_2019" "kulakovskiy_2024" "kutay_2021"
    #"martinez_2019" "riepe_2018"
)

# Split 2: 16 datasets assigned to GPU 1
DATASETS_GPU1=(
   # "sako_2020" "sauer_2019" "sidrauski_2015" "song_2019" #"tsvetanova_2025"
    "volegova_2018" "wakigawa_2025" "wan_2019" "wang_2016" "wangHEK_2022"
    "weber_2020" "weber_2022" "weber_2024" "wu_2019" "wu_2020" #"eichhorn_2014"
)

run_queue() {
    local gpu_id=$1
    shift
    local datasets=("$@")

    export CUDA_VISIBLE_DEVICES=$gpu_id

    for DS in "${datasets[@]}"; do
        echo "[GPU $gpu_id] Starting $DS..."

        LOG_FILE="run_log_GPU${gpu_id}_${DS}.txt"

        python main_ribo_queueing_modeling_multi_dataset.py \
            experiment.dataset="['$DS']" \
            split.master_dataset_universe="['$DS']" \
            trainer.devices=[0] \
            > "$LOG_FILE" 2>&1

        if [ $? -ne 0 ]; then
            echo "[GPU $gpu_id] ERROR: Failed on $DS. Check log: $LOG_FILE"
            exit 1
        fi

        echo "[GPU $gpu_id] Finished $DS."
    done

    echo "[GPU $gpu_id] All assigned datasets completed."
}

echo "Launching parallel training pipelines..."

run_queue 0 "${DATASETS_GPU0[@]}" &
PID0=$!

run_queue 1 "${DATASETS_GPU1[@]}" &
PID1=$!

echo "GPU 0 PID: $PID0  |  GPU 1 PID: $PID1"

wait $PID0
wait $PID1

echo "All individual dataset runs finished."
