#!/bin/bash
#SBATCH --job-name=slurmtest
#SBATCH --mem=200GB
#SBATCH --output=/local/scratch/yxu81/PhysicialFM/output_logs/rr_interval/finetune/moment_map.log
#SBATCH --gres=gpu:1

cd /local/scratch/yxu81/PhysicialFM/
source venv/bin/activate
python finetune/moment.py \
        --task_name regression \
        --label rr_interval \
        --use_lora False \
        --downsample_size 5000 \
        --config_path finetune/moment_config.yaml \
