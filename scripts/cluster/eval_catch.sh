#!/bin/bash
# Closed-loop evaluation (one GPU).  Extra arguments are passed to eval_catch.py.
#   bash scripts/cluster/eval_catch.sh CKPT_PRETRAINED_DIR OUT_DIR [more eval_catch.py args]
# Main table:  ... --splits hard,indist --n 24 --modes oracle,tabpfn,none --variants smooth,smoother,safe --realism 0,1 --safety
# Real-world inputs: NO_MARKER=1 and  --tracker hsv
set -e
source "$(dirname "$0")/env.sh"
export HF_HUB_OFFLINE=0
CK=$1; OUT=$2; shift 2
python challenge/sim/eval_catch.py --checkpoints "run=$CK" --out "$OUT" "$@"
