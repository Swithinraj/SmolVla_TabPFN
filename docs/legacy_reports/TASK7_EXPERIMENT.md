# Single-task SmolVLA training: two-object basket placement

Task: **put both the cream cheese box and the butter in the basket** (`task_index=7`). It has 49 demonstrations, the largest count among the LIBERO tasks that explicitly require two object placements. The two tasks with 50 demonstrations are single-action tasks.

## Split

- Training: 47 complete episodes, 12,178 frames.
- Testing: episodes **53** and **347**, 524 frames combined. These were the most visually different pair of initial views among the task's 49 episodes when comparing both cameras at 64 × 64 resolution. Pixel difference was 10.337 versus 7.727 for the median pair. Both starting views were inspected.
- The exact episode IDs are in [`data/libero/task7_split.json`](../data/libero/task7_split.json). Neither test episode is passed to the training command.

## Training

- Starting weights: exact [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) weights, downloaded to `models/smolvla_base/`.
- Shared architecture and preprocessing choices: map the two camera views consistently to LeRobot's `camera1`/`camera2` keys; resize with padding to 512 x 512; freeze the vision-language backbone and train the action expert using 50-action chunks. The camera key names are a LeRobot compatibility requirement, not a paper training prescription.
- The [SmolVLA paper](https://arxiv.org/pdf/2506.01844) does **not** specify a stochastic augmentation recipe. For this run, LeRobot's optional training-time image transforms apply either brightness or contrast jitter in the mild range 0.9–1.1 to training images. There are no geometric shifts, and evaluation uses unaugmented frames. See [LeRobot image transforms](https://huggingface.co/docs/lerobot/lerobot-dataset-v3#image-transforms).
- Batch 1, learning rate 1e-5 with 100-step warmup and cosine decay to 1e-6, 24,356 steps (about two passes over the 12,178 training frames). Save after one pass and at the end. No Hub upload.
- A two-step smoke run verified data loading, augmentation, feature dimensions, and GPU execution. Its model weights are not used for the baseline or main run.
- Main training launched 2026-11-16 at 17:47 CET with PID 29924. Logs: `.tmp/task7_base_aug_20261116_1748.stdout.log` and `.tmp/task7_base_aug_20261116_1748.stderr.log`.

The exact base weights are available in `outputs/task7_base_baseline/pretrained_model/` with the task-compatible input/output metadata and normalization produced by the smoke-run setup. The training process saves to `outputs/task7_base_aug_20261116_1748/`.

## Evaluation

A background watcher waits for training to finish, then compares the exact base weights with the 12,178-step and 24,356-step checkpoints on 16 fixed, unaugmented frames from each test episode. It writes `evaluation.log`, `evaluation_32_012178.json`, `evaluation_32_024356.json`, and `evaluation_status.txt` under the run directory. The evaluation script reports flow-matching loss and first-action MAE. These are offline imitation metrics, **not** LIBERO rollout success. Native LIBERO simulation requires Linux.

The dataset's supplied global normalization statistics include the test episodes. Thus the episode actions/images are held out from this local training run, but the split is not a strict unseen-data benchmark. The older `smolvla_libero` checkpoint was trained on the full LIBERO dataset and is not used as the starting model or baseline here.

## Completed results

Training completed all 24,356 steps on 2026-11-16 at 20:09 CET, taking about 2 hours 23 minutes. The automatic evaluation stopped because PowerShell treated a harmless Windows stderr warning as a terminating error. The same fixed-frame evaluation was then run directly and both reports were saved.

| Checkpoint | Held-out flow-matching loss | First-action MAE |
| --- | ---: | ---: |
| Exact base weights | 2.1061 | 0.3572 |
| 12,178 steps (about one pass) | 0.5606 | 0.1948 |
| 24,356 steps (about two passes) | **0.5431** | **0.1755** |

The final checkpoint reduced sampled held-out loss by 74.2% and first-action MAE by 50.9% relative to base. Both test episodes individually improved on both metrics. The final checkpoint is the best of the three on this offline check. The reports are [`evaluation_32_012178.json`](task7_base_aug_20261116_1748/evaluation_32_012178.json) and [`evaluation_32_024356.json`](task7_base_aug_20261116_1748/evaluation_32_024356.json). These numbers use 16 fixed frames per held-out episode, not all 524 frames; closed-loop results are reported below.

## Viewer inference export

The final checkpoint was run independently on every recorded observation in the two held-out episodes, with the policy reset before each one-step prediction. Per-frame 7D actions, fixed-seed flow-matching loss, and first-action MAE are saved in [`episode-53.json`](../viewer/predictions/episode-53.json) and [`episode-347.json`](../viewer/predictions/episode-347.json). The viewer at <http://127.0.0.1:8765/> displays these against the synchronized demonstration video and ground-truth action trace.

| Test episode | Frames | Full-episode mean loss | Full-episode first-action MAE |
| --- | ---: | ---: | ---: |
| 53 | 242 | 0.4867 | 0.1763 |
| 347 | 282 | 0.6420 | 0.1992 |

These full-episode numbers use a different fixed random seed sequence than the 32-frame checkpoint comparison, so compare models using the 32-frame reports rather than mixing those loss values. The frame-aligned videos and loss plots remain offline demonstration comparisons.

## Closed-loop LIBERO rollout

On 2026-11-17, the final 24,356-step checkpoint was evaluated against the real LIBERO MuJoCo environment. The simulator ran in WSL Ubuntu; the policy ran on the Windows CUDA GPU through LeRobot's remote environment transport. The dataset calls this task `task_index=7`, while the `libero_10` simulator suite calls the same language task `task_id=1`. The task description was checked before evaluation.

| Policy execution | Initial-state seed | Episode limit | Simulator result |
| --- | ---: | ---: | --- |
| Checkpoint default, 50 actions per chunk | 1000 | 520 steps at 20 Hz | **Failure (0/1)** |
| Replan every 10 actions | 1000 | 520 steps at 20 Hz | **Failure (0/1)** |
| Replan every action, as in the paper's simulation setup | 1000 | 520 steps at 20 Hz | **Failure (0/1)** |
| Full-LIBERO-trained `smolvla_libero` control, replan every action | 1000 | 520-step limit at 20 Hz | **Success (1/1), at step 232** |

The simulator returned zero reward and no success for all three single-task runs. These are one trial per setting, not reliable success-rate estimates. The recorded [50-action rollout video](../viewer/rollouts/task7_model_rollout.mp4) is displayed in the viewer with a failure label. The [single-task one-action video](../viewer/rollouts/task7_replan1.mp4) and [successful full-LIBERO control video](../viewer/rollouts/libero_full_control_replan1.mp4) use the same simulator seed and action interval. Metrics are saved for [50 actions](task7_libero_rollout/eval_info.json), [10 actions](task7_libero_rollout_replan10/eval_info.json), [one action](task7_libero_rollout_replan1/eval_info.json), and the [control](libero_control_seed1000_action1/eval_info.json). The offline loss curves do not measure closed-loop rollout loss, and the simulator's seed-1000 initial state is independent of held-out episodes 53 and 347.

LeRobot 0.7.0 passes remote `render_fps` as a float to PyAV's MP4 writer, which rejects it. The local [`eval_with_video_fix.py`](../scripts/eval_with_video_fix.py) wrapper converts only the FPS value to a rational number for video encoding; it does not alter policy actions or simulator stepping. Each failed rollout video contains all 520 simulator frames at 20 fps; the successful control video ends after 232 frames. The assets and Linux environment are under this workspace; D: had about 220 GB free after installation.

## Paper alignment and failure audit

The [SmolVLA paper](https://arxiv.org/pdf/2506.01844) reports LIBERO success using multi-task training over all 40 tasks and 1,693 demonstrations, 100,000 fine-tuning steps with batch size 64, and a new action prediction after every simulator step. The [LeRobot LIBERO guide](https://huggingface.co/docs/lerobot/libero) gives the same 100,000-step, batch-64 training example and recommends 10 evaluation episodes per task. This local run used 47 demonstrations from one long-horizon task, 24,356 steps with batch size 1 (about two passes over its 12,178 frames), and one evaluation seed. It is a single-task experiment, not a reproduction of the paper's LIBERO benchmark.

The local run did match several model choices: 512-pixel padded inputs, a 50-action prediction chunk, flow matching with 10 inference integration steps, and a frozen VLM with trainable action expert. It added mild brightness/contrast jitter; the paper does not prescribe that stochastic augmentation. It started from `smolvla_base`, while the paper's LIBERO simulation comparison describes a VLM-initialized model without robotics pretraining. The initial 50-action rollout did not match the paper's one-action simulation execution; the later one-action rollout still failed.

The one-step offline mean action MAE over the 524 held-out frames is 0.1886, but it mixes action components with very different scales. Translation MAE is 0.2608 per component versus 0.3151 mean absolute ground-truth translation, and gripper MAE is 0.4153. Those errors can compound during closed-loop control; the low flow-matching loss is not a success measure. The checkpoint's normalizer uses the dataset-wide statistics (`action.count=273465`) rather than statistics from the 12,178 selected training frames. The successful full-LIBERO control uses the same dataset-wide statistics, so this is not evidence of a broken simulator interface or the sole cause of failure. The control's success on the same seed strongly supports that the camera mapping, relative action mode, and simulator can work, although one trial cannot establish either model's overall success rate.
