#!/bin/bash
# GPU job: the main model 5_catch_big_012000 in the same real-world setting (NO_MARKER=1, --tracker hsv), all cells,
# so the new no-marker models compare on identical settings/seeds.
#SBATCH --job-name=g-eval-base-nm
#SBATCH --partition=gpu
#SBATCH --exclude=wr14,wr20,wr21,wr22,wr23,wr24,wr25
#SBATCH --gres=gpu:1            # STRICT: exactly one GPU (V100). Do not raise this number.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -x
cd "$SLURM_SUBMIT_DIR"   # submit from the repo root
CK=checkpoints/5_catch_big_012000/pretrained_model
NO_MARKER=1 bash scripts/cluster/eval_catch.sh $CK outputs/catch_big/eval_hsv_no_marker_cluster --splits hard,indist --n 24 \
     --modes oracle,tabpfn,none --variants smoother --realism 0,1 --tracker hsv --gifs 1
echo BASE_EVAL_DONE
