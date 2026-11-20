#!/bin/bash
# Record one shard of randomized catch demonstrations.
#   bash scripts/cluster/record_shard.sh SHARD_NAME SEED0 EPISODES [NO_MARKER=0|1] [TRACKER=truth|hsv]
# Writes data/catch_<SHARD_NAME>. On a GPU node it sources env.sh (one V100); with CPU_ONLY=1 it sources env_cpu.sh
# (TabPFN + rendering on the CPU, for hpc nodes).
set -e
if [ "${CPU_ONLY:-0}" = "1" ]; then source "$(dirname "$0")/env_cpu.sh"; else source "$(dirname "$0")/env.sh"; fi
export HF_HUB_OFFLINE=0                      # TabPFN weights may need the hub the first time
[ "${4:-0}" = "1" ] && export NO_MARKER=1
python challenge/sim/record_catch_dataset_v2.py --episodes "$3" --seed0 "$2" --root "data/catch_$1" \
  --repo-id "local/catch_$1" --tracker "${5:-truth}" --delay "${DELAY:-measured}" --overwrite
