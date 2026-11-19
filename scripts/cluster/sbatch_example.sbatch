#!/bin/bash
#SBATCH --job-name=smolvla-catch
#SBATCH --gres=gpu:1            # STRICT: exactly one GPU (V100). Do not raise this number.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
# Adapt partition / account lines to your cluster (e.g. --partition=..., a V100 constraint such as --constraint=v100).
#SBATCH --output=slurm-%j.out
cd "$SLURM_SUBMIT_DIR"
# Example: the big training run. Replace with record_shard.sh / eval_catch.sh as needed.
bash scripts/cluster/train_catch.sh checkpoints/4_catch_pilot_003000/pretrained_model data/so101_catch_v2 outputs/catch_big 12000 4 4000
