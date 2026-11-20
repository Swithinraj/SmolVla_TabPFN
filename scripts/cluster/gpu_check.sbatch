#!/bin/bash
#SBATCH --job-name=g-check
#SBATCH --partition=gpu
#SBATCH --exclude=wr14,wr20,wr21,wr22,wr23,wr24,wr25
#SBATCH --gres=gpu:1            # STRICT: exactly one GPU (V100)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
cd "$SLURM_SUBMIT_DIR"   # submit from the repo root
source scripts/cluster/env.sh || exit 1
python -c "import torch; x=torch.randn(512,512,device='cuda'); print('matmul ok', (x@x).sum().item(), torch.cuda.get_device_capability())"
python challenge/sim/hsv_tracker.py 2>&1 | grep -v EGLError
python challenge/sim/test_catch.py --help | head -5
