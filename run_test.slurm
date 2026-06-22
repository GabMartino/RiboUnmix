#!/bin/bash
#SBATCH --job-name=riboai_test
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d35_089
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=job_%j_riboai_test.out
#SBATCH --error=job_%j_riboai_test.err

set -euo pipefail

module load python/3.11
source ../venvs/queuing_ribo_venv/bin/activate

srun python main_ribo_queueing_modeling_multi_dataset.py \
    trainer.devices=[0,1,2,3] \
    data.batch_size=8 \
    data.num_workers=8 \
    experiment.dataset="['kutay_2021', 'sako_2020']" \
    split.master_dataset_universe="['kutay_2021', 'sako_2020']" \
