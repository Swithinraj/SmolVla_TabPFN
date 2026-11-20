#!/bin/bash
# Source this from EVERY job script:  source scripts/cluster/env.sh
# STRICT RULE: exactly ONE GPU (one V100) per job. Never request or use more.
export ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
source "$ROOT/scripts/cluster/disk_guard.sh" || exit 1   # disk budget (default 300 GB), see disk_guard.sh
export CUDA_VISIBLE_DEVICES=0                 # even on a multi-GPU node, only device 0 is visible
export HF_HOME="$ROOT/.cache/huggingface"
export HF_LEROBOT_HOME="$ROOT/.cache/lerobot"
export TABPFN_MODEL_CACHE_DIR="$ROOT/.cache/tabpfn"
export TABPFN_NO_BROWSER=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"          # headless GPU rendering; use MUJOCO_GL=osmesa if EGL is unavailable (much slower)
export TOKENIZERS_PARALLELISM=false
# the venv was built from the cluster module python3/3.13.2; its libpython3.13.so must be on the library path
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load python3/3.13.2
source "$ROOT/.venv/bin/activate"
python - <<'PY'
import sys, torch
n = torch.cuda.device_count()
if n != 1:
    sys.exit(f"STRICT RULE VIOLATED: {n} GPUs visible, exactly 1 (one V100) is allowed")
name = torch.cuda.get_device_name(0)
print("GPU:", name, "| arch list:", torch.cuda.get_arch_list())
if "V100" not in name:
    print("WARNING: expected a V100, got", name)
PY
[ $? -eq 0 ] || exit 1
