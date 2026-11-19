#!/bin/bash
# Upload checkpoints, datasets and evaluation outputs to the shared Drive folder with rclone (remote "gdrive:").
cd /d/smolVLA_Enhance
F=18BWYkCl592-N_dSM5L2XeRnGBhfThkyl
R="gdrive:SmolVLA_TabPFN_SO101"
RC="rclone copy --drive-root-folder-id $F --transfers 3 --drive-chunk-size 128M --stats 60s --stats-one-line -v"
set -x
$RC /tmp/DRIVE_README.md "$R"
$RC --exclude "training_state/**" outputs/so101_pickplace_20261117_191510/checkpoints/020000 "$R/checkpoints/1_stage1_real_pickplace_020000"
$RC --exclude "training_state/**" outputs/so101_ball_20261118_011951/checkpoints/015000 "$R/checkpoints/2_stage2_blocking_015000"
$RC --exclude "training_state/**" outputs/so101_ball_stage3/checkpoints/010000 "$R/checkpoints/3_stage3_blocking_noisy_010000"
$RC --exclude "training_state/**" outputs/catch_pilot_s2/checkpoints/003000 "$R/checkpoints/4_catch_pilot_003000"
$RC outputs/catch_big/checkpoints/012000 "$R/checkpoints/5_catch_big_012000"
$RC data/so101_catch_v2 "$R/datasets/catch_v2_merged"
$RC data/catch_v2_A "$R/datasets/catch_v2_shards/A"
$RC data/catch_v2_B "$R/datasets/catch_v2_shards/B"
$RC data/so101_ball_train "$R/datasets/blocking_stage2_train"
$RC data/so101_ball_fc_train "$R/datasets/blocking_stage3_forecast_train"
$RC data/so101_ball_heldout "$R/datasets/blocking_heldout"
$RC outputs/catch_big/eval "$R/evaluation/catch_big/eval_main"
$RC outputs/catch_big/eval_hsv "$R/evaluation/catch_big/eval_hsv_no_marker"
$RC outputs/catch_big/eval_nomarker "$R/evaluation/catch_big/eval_no_marker_true_track"
$RC outputs/catch_big/forecast_check.txt "$R/evaluation/catch_big"
$RC outputs/catch_pilot_s2/eval "$R/evaluation/catch_pilot"
$RC outputs/so101_ball_stage3/closed_loop_compare "$R/evaluation/blocking/stage2_vs_stage3"
$RC outputs/so101_ball_20261118_011951/closed_loop_hard "$R/evaluation/blocking/stage2_hard"
echo "=== upload done"
