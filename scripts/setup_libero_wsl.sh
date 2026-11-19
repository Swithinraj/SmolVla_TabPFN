#!/usr/bin/env bash
set -euo pipefail

workspace=/mnt/d/smolVLA_Enhance
venv="$workspace/.venv-libero-wsl"
export PIP_CACHE_DIR="$workspace/.cache/pip-wsl"
export XDG_CACHE_HOME="$workspace/.cache/xdg-wsl"
export HF_HOME="$workspace/.cache/huggingface-wsl"
export TMPDIR="$workspace/.tmp"

mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TMPDIR"
free_bytes=$(df -B1 --output=avail /mnt/d | tail -n 1 | tr -d ' ')
if (( free_bytes < 105000000000 )); then
    echo "Need at least 105 GB free on D: before installing LIBERO" >&2
    exit 1
fi

if [[ ! -x "$venv/bin/python" ]]; then
    python3 -m venv "$venv"
fi
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install 'lerobot[libero,remote]==0.7.0'
"$venv/bin/python" -c 'import lerobot, mujoco, libero; print("LeRobot", lerobot.__version__, "MuJoCo", mujoco.__version__)'
df -h /mnt/d
