#!/bin/bash
# Upload the 2026-11-19 cluster results to the shared Drive folder in the same layout as scripts/upload_to_drive.sh.
# Needs an rclone remote named "gdrive:" (rclone config). Settings come from the environment or, if unset, from the
# git-ignored scripts/cluster/local.env:  DRIVE_FOLDER_ID (the shared folder's id), RCLONE (path to rclone, default: on PATH).
#   sbatch scripts/cluster/upload_cluster_results_to_drive.sbatch              (from the repo root)
#   ONLY_CHECKPOINTS=1 sbatch --export=ALL ...   upload only checkpoints 6 and 7 (skip the new dataset and evaluation outputs)
# CPU job, no GPU. The target base is the one that already holds checkpoints/5_catch_big_012000: either the folder itself
# (DRIVE_FOLDER_ID = SmolVLA_TabPFN_SO101) or SmolVLA_TabPFN_SO101/ below it (DRIVE_FOLDER_ID = its parent). Otherwise it aborts.
#SBATCH --job-name=h-upload-drive
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
cd "$SLURM_SUBMIT_DIR"   # submit from the repo root
_le=scripts/cluster/local.env
_get() { [ -f "$_le" ] && sed -n "s/^$1=//p" "$_le" | tail -1 | tr -d '"'; }
RCLONE="${RCLONE:-$(_get RCLONE)}"; RCLONE="${RCLONE:-rclone}"
F="${DRIVE_FOLDER_ID:-$(_get DRIVE_FOLDER_ID)}"
[ -n "$F" ] || { echo "ABORT: set DRIVE_FOLDER_ID (environment or $_le)"; exit 1; }
"$RCLONE" listremotes | grep -qx "gdrive:" || { echo "ABORT: no rclone remote 'gdrive:' (run: $RCLONE config)"; exit 1; }
R=""          # path prefix, used as "${R}checkpoints/..."
for base in "gdrive:" "gdrive:SmolVLA_TabPFN_SO101/"; do
  if "$RCLONE" lsf --drive-root-folder-id "$F" "${base}checkpoints" 2>/dev/null | grep -qx "5_catch_big_012000/"; then
    R="$base"; break
  fi
done
[ -n "$R" ] || { echo "ABORT: checkpoints/5_catch_big_012000 not found under this DRIVE_FOLDER_ID (wrong id or remote?)"; exit 1; }
echo "uploading to base '$R' (root folder id $F)"
RC="$RCLONE copy --drive-root-folder-id $F --transfers 3 --drive-chunk-size 128M --stats 60s --stats-one-line -v"
set -ex
$RC outputs/catch_nm_from_s2/checkpoints/012000 "${R}checkpoints/6_catch_nomarker_from_stage2_012000"
$RC outputs/catch_nm_from_big/checkpoints/012000 "${R}checkpoints/7_catch_nomarker_from_catch_big_012000"
if [ "${ONLY_CHECKPOINTS:-0}" != "1" ]; then
  $RC data/so101_catch_nm "${R}datasets/catch_nomarker_hsv_merged"
  $RC outputs/catch_nm_from_s2/eval_hsv_no_marker "${R}evaluation/catch_nomarker/eval_from_stage2"
  $RC outputs/catch_nm_from_big/eval_hsv_no_marker "${R}evaluation/catch_nomarker/eval_from_catch_big"
  $RC outputs/catch_big/eval_hsv_no_marker_cluster "${R}evaluation/catch_nomarker/eval_old_model_cluster"
  $RC outputs/catch_big/smooth_sweep "${R}evaluation/catch_big/smooth_sweep"
fi
$RC docs/DRIVE_README.md "$R"
"$RCLONE" lsf --drive-root-folder-id "$F" "${R}checkpoints"
echo UPLOAD_DONE
