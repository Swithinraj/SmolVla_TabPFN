#!/bin/bash
# Record one shard of randomized catch demonstrations (CPU heavy; one GPU for TabPFN and rendering).
#   bash scripts/cluster/record_shard.sh SHARD_NAME SEED0 EPISODES [NO_MARKER=0|1]
set -e
source "$(dirname "$0")/env.sh"
export HF_HUB_OFFLINE=0                      # TabPFN weights may need the hub the first time
[ "${4:-0}" = "1" ] && export NO_MARKER=1
python challenge/sim/record_catch_dataset_v2.py --episodes "$3" --seed0 "$2" --root "data/catch_$1" \
  --repo-id "local/catch_$1" --overwrite
