#!/bin/bash
#SBATCH --job-name=slurmtest
#SBATCH --mem=200GB
#SBATCH --output=./output_logs/age/64shot/moment.log
#SBATCH --gres=gpu:1

cd /local/scratch/yxu81/PhysicialFM/
source venv/bin/activate
python moment_trainer.py \
        --task_name regression \
        --label age \
        --use_lora False \
        --downsample_size 5000 \
        --config_path ./moment_config2.yaml
