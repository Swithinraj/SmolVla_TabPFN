#!/bin/bash
# CPU-only variant of env.sh for recording on a CPU (hpc) node: no GPU is requested or used.
# TabPFN runs on the CPU (forecast.py picks cpu when no GPU is visible); MuJoCo renders with EGL/mesa (software)
# or OSMesa. Do NOT use this for training or policy evaluation (those need the one V100 and env.sh).
export ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
source "$ROOT/scripts/cluster/disk_guard.sh" || exit 1   # disk budget (default 300 GB), see disk_guard.sh
export CUDA_VISIBLE_DEVICES=""
export HF_HOME="$ROOT/.cache/huggingface"
export HF_LEROBOT_HOME="$ROOT/.cache/lerobot"
export TABPFN_MODEL_CACHE_DIR="$ROOT/.cache/tabpfn"
export TABPFN_NO_BROWSER=1
export TABPFN_DEVICE=cpu
export TABPFN_ALLOW_CPU_LARGE_DATASET=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-2}"
# the venv was built from the cluster module python3/3.13.2; its libpython3.13.so must be on the library path
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load python3/3.13.2
source "$ROOT/.venv/bin/activate"
