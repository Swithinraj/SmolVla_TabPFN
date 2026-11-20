#!/bin/bash
# The unattended chain that produced the 2026-11-19 cluster results (run from the repo root).
# STRICT: one V100 job at a time -- every GPU job waits for the previous one (afterany), so only one GPU job ever runs.
#   GT  pipeline test: record 3 episodes, merge, train 20 steps (~2 min)
#   GR  record the no-marker, HSV-fed dataset on the GPU: 8 shards x 50 -> data/so101_catch_nm (~1 h)
#       (CPU-only recording was measured at ~1000 s/episode because TabPFN is slow on CPU)
#   G2  train 12k steps from 2_stage2_blocking_015000, then evaluate NO_MARKER=1 --tracker hsv (~2.2 h)
#   G3  the same from 5_catch_big_012000 (~2.2 h)
#   G1  pull checkpoints if missing, smoke tests, inference-only smoothness sweep on 5_catch_big (~1.1 h)
#   G4  5_catch_big in the same NO_MARKER + HSV setting as G2/G3 (~0.8 h)
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs
GT=$(sbatch --parsable scripts/cluster/gpu_pipeline_test.sbatch)
GR=$(sbatch --parsable --dependency=afterany:$GT scripts/cluster/gpu_record_nomarker.sbatch 8 50 nm data/so101_catch_nm)
G2=$(sbatch --parsable --dependency=afterany:$GR scripts/cluster/gpu_train_eval_nomarker.sbatch checkpoints/2_stage2_blocking_015000 catch_nm_from_s2)
G3=$(sbatch --parsable --dependency=afterany:$G2 scripts/cluster/gpu_train_eval_nomarker.sbatch checkpoints/5_catch_big_012000 catch_nm_from_big)
G1=$(sbatch --parsable --dependency=afterany:$G3 scripts/cluster/gpu1_smoke_and_smooth_sweep.sbatch)
G4=$(sbatch --parsable --dependency=afterany:$G1 scripts/cluster/gpu_eval_baseline_nomarker.sbatch)
echo "$(date) GT=$GT GR=$GR G2=$G2 G3=$G3 G1=$G1 G4=$G4" | tee -a logs/night_jobs.txt
