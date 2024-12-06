#!/bin/bash
#SBATCH --job-name=slurmtest
#SBATCH --mem=200GB
#SBATCH --output=./output_logs/rr_interval/0shot/moment_map.log
#SBATCH --gres=gpu:1

cd /local/scratch/yxu81/PhysicialFM/
source venv/bin/activate
python zero_shot/moment.py \
        --task_name regression \
        --label rr_interval \
        --use_lora False \
        --downsample_size 5000 \
        --config_path zero_shot/moment_config2.yaml \
