#!/bin/bash
# GPU job: train SmolVLA on the no-marker, HSV-fed dataset (handout 4.1), then evaluate in the real-world setting
# (NO_MARKER=1, --tracker hsv). Usage:  sbatch scripts/cluster/gpu_train_eval_nomarker.sbatch INIT_CKPT_DIR RUN_NAME [DATA]
#   INIT_CKPT_DIR: e.g. checkpoints/2_stage2_blocking_015000   RUN_NAME: e.g. catch_nomarker_s2
#SBATCH --job-name=g-train-nm
#SBATCH --partition=gpu
#SBATCH --exclude=wr14,wr20,wr21,wr22,wr23,wr24,wr25
#SBATCH --gres=gpu:1            # STRICT: exactly one GPU (V100). Do not raise this number.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
set -x
cd "$SLURM_SUBMIT_DIR"   # submit from the repo root
INIT=$1; NAME=$2; DATA=${3:-data/so101_catch_nm}
[ -f "$DATA/sidecar.json" ] || { echo "dataset $DATA missing (recording/merge failed?)"; exit 1; }
[ -d "$INIT/pretrained_model" ] || { echo "init checkpoint $INIT/pretrained_model missing"; exit 1; }
OUT=outputs/$NAME
if [ ! -f "$OUT/checkpoints/012000/pretrained_model/model.safetensors" ]; then
  rm -rf "$OUT"          # lerobot-train refuses an existing output dir without --resume
  bash scripts/cluster/train_catch.sh "$INIT/pretrained_model" "$DATA" "$OUT" 12000 4 4000 || exit 1
fi
CK=$OUT/checkpoints/012000/pretrained_model
# Real-world inputs: no blue dot, forecaster fed only by HSV detections. Same seeds / n as the main table.
NO_MARKER=1 bash scripts/cluster/eval_catch.sh $CK $OUT/eval_hsv_no_marker --splits hard,indist --n 24 \
     --modes oracle,tabpfn,none --variants smoother --realism 0,1 --tracker hsv --gifs 1
echo TRAIN_EVAL_DONE $NAME
