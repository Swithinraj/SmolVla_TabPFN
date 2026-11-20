#!/bin/bash
# One-time environment setup on a compute node (compute nodes had internet on the tested cluster).
# Submit from the repo root:  mkdir -p logs && sbatch scripts/cluster/setup_env.sbatch
# Recipe that worked on the cluster (2026-11-19): Python 3.13.2 module, torch 2.9.1 from the cu126 index (has sm_70 for the
# V100), lerobot 0.7.0 with the training extra, mujoco pinned to 3.3.7 (pip otherwise pulls 4.x), only the headless opencv.
#SBATCH --job-name=setup-env
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -ex
cd "$SLURM_SUBMIT_DIR"   # submit from the repo root
module load python3/3.13.2
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
PINS="torch==2.9.1 numpy==2.2.6 mujoco==3.3.7 transformers==5.22.2 opencv-python-headless==4.13.0.92"
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install "lerobot[smolvla,training]==0.7.0" tabpfn==10.2.0 matplotlib av gdown $PINS \
    --extra-index-url https://download.pytorch.org/whl/cu126
pip uninstall -y opencv-python || true                     # lerobot pulls the GUI build too; both provide cv2
pip install --force-reinstall --no-deps opencv-python-headless==4.13.0.92
python -c "import torch, mujoco, cv2, numpy; print('TORCH', torch.__version__, torch.version.cuda, 'mujoco', mujoco.__version__, 'cv2', cv2.__version__, 'numpy', numpy.__version__)"
# the simulator meshes/textures are Git LFS pointers in a plain clone: fetch the real files (sha256-checked)
python scripts/cluster/fix_lfs_assets.py
# SmolVLA base + SmolVLM2 backbone + TabPFN v2 weights into .cache/ (training runs offline)
python scripts/cluster/prefetch_models.py
pip freeze > logs/pip_freeze.txt
echo SETUP_DONE
