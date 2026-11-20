#!/bin/bash
# GPU job (one V100, one process at a time): record the no-marker, HSV-fed catch dataset (handout 4.1) shard by shard,
# then merge into data/so101_catch_nm. CPU-only recording was measured at ~1000 s/episode (TabPFN on CPU), so the
# recorder runs on the GPU. Finished shards are skipped, so a re-submit resumes.
#   sbatch scripts/cluster/gpu_record_nomarker.sbatch [NSHARDS] [EPISODES_PER_SHARD] [PREFIX] [OUT]
#SBATCH --job-name=g-record-nm
#SBATCH --partition=gpu
#SBATCH --exclude=wr14,wr20,wr21,wr22,wr23,wr24,wr25
#SBATCH --gres=gpu:1            # STRICT: exactly one GPU (V100). Do not raise this number.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
cd "$SLURM_SUBMIT_DIR"   # submit from the repo root
NS=${1:-8}; EPS=${2:-50}; PREFIX=${3:-nm}; OUT=${4:-data/so101_catch_nm}
for k in $(seq 0 $((NS - 1))); do
  name=${PREFIX}_$k
  if [ -f data/catch_$name/sidecar.json ]; then echo "shard $name already done"; continue; fi
  echo "$(date +%H:%M) shard $name seeds from $((900000 + 10000 * k))"
  bash scripts/cluster/record_shard.sh $name $((900000 + 10000 * k)) $EPS 1 hsv 2>&1 | grep --line-buffered -vE "EGLError|glCheckError|^Traceback|^  File|Exception ignored" || echo "WARNING: shard $name failed"
done
source scripts/cluster/env.sh
python scripts/aggregate_shards.py --out "$OUT" $(for k in $(seq 0 $((NS - 1))); do echo data/catch_${PREFIX}_$k; done) || exit 1
echo RECORD_DONE
