#!/bin/bash
#SBATCH --exclude=dgx1
#SBATCH --job-name=riboai_2gpu_smoke
#SBATCH --partition=p_datamining
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=job_%j_riboai_test.out
#SBATCH --error=job_%j_riboai_test.err

set -euo pipefail

module load python/3.11
source ~/venvs/queueing_riboai_venv/bin/activate

srun python main_ribounmix_multidataset.py \
    trainer.devices=[0,1] \
    training.execution_microbatching.enabled=false \
    data.batch_size=128 \
    data.num_workers=8 \
    experiment.dataset="['kutay_2021', 'grimson_2019', 'sako_2020']" \
    split.master_dataset_universe="['kutay_2021', 'grimson_2019', 'sako_2020']"
