#!/bin/bash
# GPU job (~10 min): de-risk the night. Record 3 no-marker/HSV episodes on the V100 (timing), merge, train 20 steps.
#SBATCH --job-name=g-pipetest
#SBATCH --partition=gpu
#SBATCH --exclude=wr14,wr20,wr21,wr22,wr23,wr24,wr25
#SBATCH --gres=gpu:1            # STRICT: exactly one GPU (V100). Do not raise this number.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
cd "$SLURM_SUBMIT_DIR"   # submit from the repo root
bash scripts/cluster/record_shard.sh pt_0 990000 3 1 hsv 2>&1 | grep --line-buffered -vE "EGLError|glCheckError|^Traceback|^  File|Exception ignored"
source scripts/cluster/env.sh
python scripts/aggregate_shards.py --out data/pipetest data/catch_pt_0 || { echo PIPETEST_MERGE_FAILED; exit 1; }
rm -rf outputs/pipetest
bash scripts/cluster/train_catch.sh checkpoints/2_stage2_blocking_015000/pretrained_model data/pipetest outputs/pipetest 20 4 20 \
  && ls outputs/pipetest/checkpoints/*/pretrained_model && echo PIPETEST_OK || echo PIPETEST_TRAIN_FAILED
