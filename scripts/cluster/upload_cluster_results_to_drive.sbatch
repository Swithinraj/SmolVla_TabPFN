#!/bin/bash
# Upload the 2026-11-19 cluster results to the shared Drive folder, same layout and rclone convention as
# scripts/upload_to_drive.sh (remote "gdrive:", DRIVE_FOLDER_ID = the id you use with that script).
#   DRIVE_FOLDER_ID=<id> sbatch --export=ALL scripts/cluster/upload_cluster_results_to_drive.sbatch
#   ONLY_CHECKPOINTS=1 ...   upload only checkpoints 6 and 7 (skip the new dataset and the evaluation outputs)
#   RCLONE=/path/to/rclone   if rclone is not on PATH
# CPU job, no GPU. Refuses to upload unless the existing layout (checkpoints/5_catch_big_012000) is found at the target.
#SBATCH --job-name=h-upload-drive
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
cd "$SLURM_SUBMIT_DIR"   # submit from the repo root
RCLONE="${RCLONE:-rclone}"
F="${DRIVE_FOLDER_ID:?set DRIVE_FOLDER_ID to the target Drive folder id (as for scripts/upload_to_drive.sh)}"
R="gdrive:SmolVLA_TabPFN_SO101"
RC="$RCLONE copy --drive-root-folder-id $F --transfers 3 --drive-chunk-size 128M --stats 60s --stats-one-line -v"
"$RCLONE" lsf --drive-root-folder-id "$F" "$R/checkpoints" | grep -qx "5_catch_big_012000/" \
  || { echo "ABORT: $R/checkpoints/5_catch_big_012000 not found with this DRIVE_FOLDER_ID (wrong id or remote?)"; exit 1; }
set -ex
$RC outputs/catch_nm_from_s2/checkpoints/012000 "$R/checkpoints/6_catch_nomarker_from_stage2_012000"
$RC outputs/catch_nm_from_big/checkpoints/012000 "$R/checkpoints/7_catch_nomarker_from_catch_big_012000"
if [ "${ONLY_CHECKPOINTS:-0}" != "1" ]; then
  $RC data/so101_catch_nm "$R/datasets/catch_nomarker_hsv_merged"
  $RC outputs/catch_nm_from_s2/eval_hsv_no_marker "$R/evaluation/catch_nomarker/eval_from_stage2"
  $RC outputs/catch_nm_from_big/eval_hsv_no_marker "$R/evaluation/catch_nomarker/eval_from_catch_big"
  $RC outputs/catch_big/eval_hsv_no_marker_cluster "$R/evaluation/catch_nomarker/eval_old_model_cluster"
  $RC outputs/catch_big/smooth_sweep "$R/evaluation/catch_big/smooth_sweep"
fi
$RC docs/DRIVE_README.md "$R"
"$RCLONE" lsf --drive-root-folder-id "$F" "$R/checkpoints"
echo UPLOAD_DONE
