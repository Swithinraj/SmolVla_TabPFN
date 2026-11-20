# SmolVLA + TabPFN on the SO-101: checkpoints, datasets, evaluation outputs

Code, docs and graphs: https://github.com/Swithinraj/SmolVla_TabPFN. Everything here is simulation-only except stage 1,
which starts from real SO-101 pick-and-place data. Nothing has been run on a real arm.

## Layout

```
checkpoints/
  1_stage1_real_pickplace_020000/      SmolVLA fine-tuned on real SO-101 pick-and-place (lerobot/svla_so101_pickplace), 20,000 steps
  2_stage2_blocking_015000/            + sim blocking demonstrations with exact targets, 15,000 steps (init for stage 3 and for the catch pilot)
  3_stage3_blocking_noisy_010000/      + blocking with noisy-forecast targets: NEGATIVE result, the policy ignores the target
  4_catch_pilot_003000/                catch and hold, 3,000 steps from stage 2
  5_catch_big_012000/                  catch and hold, 12,000 steps from the pilot  <-- the main model (trained WITH the marker)
  6_catch_nomarker_from_stage2_012000/ catch and hold, NO marker, HSV-fed forecasts, 12,000 steps from checkpoint 2 (cluster, 2026-11-19)
  7_catch_nomarker_from_catch_big_012000/  the same data, 12,000 steps from checkpoint 5 (cluster, 2026-11-19)
      pretrained_model/                  load this with lerobot (SmolVLAPolicy.from_pretrained); includes normalizer statistics
      training_state/                    optimizer / scheduler state, only needed to resume training
datasets/                              LeRobot v3 format (parquet + AV1 video); open with LeRobotDataset(root=...)
  catch_v2_merged/                     400 catch demonstrations, 37,636 frames (what checkpoints 4 and 5 were trained on)
  catch_v2_shards/A, B                 the two recording shards the merge was made from (200 episodes each) + sidecar.json
  catch_nomarker_hsv_merged/           400 catch demonstrations WITHOUT the marker, forecaster fed only by HSV detections,
                                       36,984 frames (what checkpoints 6 and 7 were trained on); sidecar.json per episode
  blocking_stage2_train/               blocking demos with exact targets
  blocking_stage3_forecast_train/      blocking demos with noisy forecast targets
  blocking_heldout/                    held-out blocking launches
evaluation/                            evaluator outputs (JSON per condition, rollout GIFs)
  catch_big/                           main evaluation (24 launches per cell), marker-off and HSV-only runs, forecast accuracy check,
                                       smooth_sweep/ (inference-only smoothing sweep, cluster)
  catch_nomarker/                      checkpoints 6, 7 and 5 with no marker + HSV-fed TabPFN (cluster, n = 24 per cell)
  catch_pilot/
  blocking/
```

## Provenance and how to use

- Every checkpoint is SmolVLA (`lerobot/smolvla_base` architecture), trained with only the action expert unfrozen, lr 1e-4.
- Chain: smolvla_base -> 1 -> 2 -> 3, 2 -> 4 -> 5, 2 -> 6 and 5 -> 7 (each stage starts from the previous checkpoint). The catch runs start
  from checkpoint 2, so keep it to reproduce them (see docs/REPRODUCE.md in the GitHub repo).
- State size: checkpoint 1 uses the 6 joints; checkpoint 2 uses 8 entries (6 joints + 2 target entries); checkpoints 3, 4 and 5
  use 12 (6 joints + 6 forecast values: crossing y, time to cross, sigma, three path points); so do 6 and 7. Each checkpoint stores its own
  normalization statistics in the `policy_preprocessor` / `policy_postprocessor` files. Do not reuse statistics from another
  robot or dataset.
- The blocking sets and catch_v2 contain the rendered blue goal marker in the images; catch_nomarker_hsv_merged does not
  (recorded with NO_MARKER=1 and --tracker hsv). Results of checkpoints 6 and 7: docs/RESULTS.md section 4.
- Results summary: docs/RESULTS.md in the repository.
