#!/bin/bash
# GPU job 1: smoke tests (handout section 3), then the inference-only smoothness sweep (handout 4.2) on the main model.
#SBATCH --job-name=g1-smoke-sweep
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
source scripts/cluster/env.sh || exit 1
nvidia-smi
CK=checkpoints/5_catch_big_012000/pretrained_model
# --- checkpoints from the shared Drive folder (main model + the two training inits), skipped if already present
# (the Drive upload may still be running: wait until every file of both pretrained_model folders is listed)
REQ=""
for c in 5_catch_big_012000 2_stage2_blocking_015000; do
  for f in config.json model.safetensors train_config.json policy_preprocessor.json policy_postprocessor.json \
           policy_preprocessor_step_5_normalizer_processor.safetensors policy_postprocessor_step_0_unnormalizer_processor.safetensors \
           tokenizer/tokenizer.json tokenizer/tokenizer_config.json tokenizer/chat_template.jinja; do
    REQ="$REQ,checkpoints/$c/pretrained_model/$f"; done; done
python scripts/cluster/pull_drive.py --only checkpoints --wait-for "${REQ#,}" --wait-hours 8 \
  --prefix checkpoints/5_catch_big_012000/pretrained_model,checkpoints/2_stage2_blocking_015000/pretrained_model || exit 1
ls -la $CK checkpoints/2_stage2_blocking_015000/pretrained_model || exit 1
# --- smoke tests
python challenge/sim/hsv_tracker.py
python challenge/sim/test_catch.py --help
bash scripts/cluster/eval_catch.sh $CK outputs/smoke --splits indist --n 4 --modes oracle --variants smoother --realism 0 --gifs 0
python - <<'PY' || { echo "SMOKE FAILED: held < 2/4, stopping before the sweep"; exit 1; }
import json, sys
r = json.load(open("outputs/smoke/closed_loop.json"))["results"]
h = [v["held"] for k, v in r.items()]
print("SMOKE held", h)
sys.exit(0 if h and h[0] >= 2 else 1)
PY
# --- 4.2 cheap smoothness tests (inference only): oracle targets, clean, both splits, n=24, same seeds as the main table.
# smoother = p5 e0.05 a0.7 (reference). eN = chunk-blending decay (smaller = more blending), pN = re-query period
# (fewer executed steps per chunk), aN = EMA.
VARIANTS=smoother,p5e0.02a0.7,p5e0.01a0.7,p3e0.05a0.7,p2e0.05a0.7,p5e0.05a0.8,p5e0.05a0.9,p3e0.02a0.8
bash scripts/cluster/eval_catch.sh $CK outputs/catch_big/smooth_sweep --splits indist,hard --n 24 --modes oracle \
     --variants $VARIANTS --realism 0 --gifs 1
echo G1_DONE
