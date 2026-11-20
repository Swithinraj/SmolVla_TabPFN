#!/bin/bash
# CPU job (one hpc node, no GPU): mirror the whole Drive folder into drive_pull/ and arrange it as in handout section 2.
# Then sanity-check every arranged LeRobot dataset (episode count in meta vs sidecar) and every checkpoint folder.
#SBATCH --job-name=h-pull-drive
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
cd "$SLURM_SUBMIT_DIR"   # submit from the repo root
source scripts/cluster/env_cpu.sh
python scripts/cluster/pull_drive.py --list | tail -1
python scripts/cluster/pull_drive.py || echo "PULL HAD FAILURES (re-submit to resume)"
python - <<'PY'
import json
from pathlib import Path
for d in sorted(Path("data").glob("*")):
    info = d / "meta" / "info.json"
    if not info.exists():
        continue
    n = json.loads(info.read_text()).get("total_episodes")
    side = d / "sidecar.json"
    ns = len(json.loads(side.read_text())) if side.exists() else None
    print(f"dataset {d}: meta episodes {n}, sidecar {ns}", "OK" if ns in (None, n) else "MISMATCH")
for c in sorted(Path("checkpoints").glob("*/pretrained_model")):
    need = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]
    miss = [f for f in need if not (c / f).exists()]
    print(f"checkpoint {c.parent.name}:", "OK" if not miss else f"MISSING {miss}")
PY
du -sh drive_pull checkpoints data outputs/drive_evaluation 2>/dev/null
echo PULL_ALL_DONE
