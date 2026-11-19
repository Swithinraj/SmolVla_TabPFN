#!/usr/bin/env bash
set -euo pipefail

workspace=/mnt/d/smolVLA_Enhance
export HF_HOME="$workspace/.cache/huggingface-wsl"
export XDG_CACHE_HOME="$workspace/.cache/xdg-wsl"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$workspace/.cache/libero"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$LIBERO_CONFIG_PATH"

exec "$workspace/.venv-libero-wsl/bin/lerobot-env-server" \
    --config_path="$workspace/scripts/libero_task7_server.yaml"
