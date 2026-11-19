#!/usr/bin/env bash
set -euo pipefail

workspace=/mnt/d/smolVLA_Enhance
export LIBERO_CONFIG_PATH="$workspace/.cache/libero"
export HF_HOME="$workspace/.cache/huggingface-wsl"
export XDG_CACHE_HOME="$workspace/.cache/xdg-wsl"
assets="$LIBERO_CONFIG_PATH/assets"
package_assets="$workspace/.venv-libero-wsl/lib/python3.12/site-packages/libero/libero/assets"

free_bytes=$(df -B1 --output=avail /mnt/d | tail -n 1 | tr -d ' ')
if (( free_bytes < 105000000000 )); then
    echo "Need at least 105 GB free on D: before downloading assets" >&2
    exit 1
fi

"$workspace/.venv-libero-wsl/bin/python" -c \
    'from libero.libero.utils.download_utils import download_assets_from_huggingface; download_assets_from_huggingface("/mnt/d/smolVLA_Enhance/.cache/libero/assets")'
if [[ ! -e "$package_assets" ]]; then
    ln -s "$assets" "$package_assets"
fi
df -h /mnt/d
