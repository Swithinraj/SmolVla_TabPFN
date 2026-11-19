# SO-101 red-ball interception: TabPFN + SmolVLA

This repository is now aimed at the Prior Labs spatial-awareness challenge. The selected first task is an **SO-101 robot blocking a rolling red ball in simulation**. TabPFN will predict the ball's crossing point from time-stamped coordinates; SmolVLA will move the SO-101 toward that predicted intercept. The implementation sequence, evaluation baselines, and hardware limits are in [challenge/PLAN.md](../challenge/PLAN.md). The simulator and policy for this new task have not been built yet.

The earlier LIBERO work is preserved below as reference. Its dataset and checkpoints control a Franka Panda and are not SO-101 training data.

## Previous LIBERO project

This project contains a local LeRobot/SmolVLA environment, the complete LIBERO v3 dataset, a LIBERO-fine-tuned checkpoint, and a browser viewer for recorded simulation episodes. All project downloads and caches are under this directory.

## What is ready

- Python 3.13 virtual environment: `.venv/`
- LeRobot 0.7.0 with training and SmolVLA dependencies
- CUDA PyTorch 2.9.1 (`cu128`), tested on an RTX 3050 Laptop GPU with 4 GB VRAM
- Full LIBERO dataset: 1,693 episodes, 273,465 frames, 40 tasks, both camera streams, states and actions
- Local `lerobot/smolvla_libero` checkpoint
- WSL Ubuntu LIBERO/MuJoCo simulator with local assets and a remote link to the Windows CUDA policy
- Browser viewer with H.264 preview copies, frame stepping, speed control, action traces and robot states; first 8 episodes prepared by default

The original AV1 MP4 files under `data/libero/` are retained for LeRobot. The H.264 files under `viewer/media/` are browser previews.

## View the episodes

Double-click `start_viewer.cmd`, or run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start_viewer.ps1
```

Open <http://127.0.0.1:8765>. The default page shows the trained model's offline predictions alongside ground-truth actions and per-frame loss for test episodes 53 and 347. A separate **Model-controlled simulation** panel contains the real closed-loop rollout video and its success result. Select either test episode, then use the action-dimension selector, timeline, or left/right arrow keys to compare recorded frames. Press Space to play/pause the demonstration; use the rollout video's own controls for the independent simulator run. The viewer runs on localhost and does not upload data. The original demonstration-only page remains at <http://127.0.0.1:8765/index.html>.

The complete dataset is available to LeRobot. Browser previews are generated for the first 8 episodes plus the two test episodes by default because transcoding every video takes time and disk space. To prepare more, run `.\.venv\Scripts\python.exe scripts\prepare_viewer.py --limit 40 --episode-ids 53,347` and refresh the page, or launch `start_viewer.cmd -Limit 40` when the server is stopped.

## Run one model prediction

```powershell
.\.venv\Scripts\python.exe scripts\smoke_inference.py
```

This loads one downloaded observation and prints the demonstration action and the model's predicted action. It was tested on CUDA. The first run may need the SmolVLM tokenizer/backbone from Hugging Face; the files are now cached under `.cache/huggingface/`.

## Fine-tune on the local dataset

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\train_subset.ps1 -Steps 1000 -BatchSize 1
```

The script reads the episode list from `data/libero/subset.json` (now all 1,693 episodes), uses the local dataset/checkpoint, renames the cameras to match this checkpoint, disables Hub uploads, and writes a timestamped run in `outputs/`. Set `-Device cpu` for CPU training. A one-step CUDA run with batch size 1 completed successfully and saved a checkpoint. Use `-EpisodeIds '1,2,4'` to restrict a run to selected episodes.

The script defaults to a conservative learning rate of `0.000001`. At batch size 1, one pass over the full 273,465-frame dataset takes about 273,465 steps; the command above is a short trial. A 2,700-step, three-epoch experiment on three selected episodes and its held-out checks are documented in [outputs/TUNING_RESULTS.md](legacy_reports/TUNING_RESULTS.md). The original checkpoint remains the recommended model from these measurements.

## Current single-task experiment

The two-object basket task has 49 demonstrations. Episodes 53 and 347 were reserved for offline testing; the other 47 were used for training. A 24,356-step run from `smolvla_base` completed with mild LeRobot brightness/contrast augmentation. On 32 fixed held-out frames, the final checkpoint improved flow-matching loss from 2.1061 to 0.5431 and first-action MAE from 0.3572 to 0.1755 versus the base weights. See [outputs/TASK7_EXPERIMENT.md](legacy_reports/TASK7_EXPERIMENT.md) for the split, training settings, evaluation limits, and artifacts. To launch another run with those settings, use `powershell -NoProfile -ExecutionPolicy Bypass -File .\train_task7.ps1` with a new output directory.

For a frame-by-frame offline inference view of the final checkpoint, the project exported one-step predictions for **all 524 frames** of test episodes 53 and 347. Each saved prediction is made from the corresponding recorded observation; it does not drive the simulated robot. The viewer overlays predicted and demonstration actions, shows per-frame flow-matching loss and action MAE, and reports episode averages. To re-export predictions, run `.\.venv\Scripts\python.exe scripts\export_task7_inference.py`.

## Dataset download

The full `lerobot/libero` dataset is downloaded under `data/libero/`: 451 required Parquet/MP4 files, about 1.94 GB. The downloader checks that D: stays above 100 GB free. To verify or resume the download, run:

```powershell
python scripts\download_libero.py --episodes 1693
```

The viewer preparation step makes browser previews and posters for the number of episodes you select with `--limit`.

## Run a LIBERO simulator rollout

The Linux-only LIBERO environment runs in WSL Ubuntu. Its Python environment is `.venv-libero-wsl/`, and the simulator assets are in `.cache/libero/assets/`. The Windows policy connects to the simulator at `tcp/127.0.0.1:7448`. The dataset's task index 7 is simulator task ID 1 in `libero_10`.

In one terminal, start the simulator:

```powershell
wsl.exe -d Ubuntu -- bash /mnt/d/smolVLA_Enhance/scripts/run_libero_task7_server.sh
```

In another terminal, run one 520-step episode and publish its video to the viewer:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_task7_rollout.ps1 -ActionSteps 50
.\.venv\Scripts\python.exe scripts\publish_task7_rollout.py
```

The final single-task checkpoint failed the task in the recorded seed-1000 episode: **0/1 success at 520 steps** with 50, 10, and 1 action per replanning cycle. The already full-LIBERO-trained `smolvla_libero` checkpoint succeeded on the same seed with one-action replanning in 232 steps. The [single-task one-action video](viewer/rollouts/task7_replan1.mp4) and [successful control video](viewer/rollouts/libero_full_control_replan1.mp4) are saved for comparison. See [outputs/TASK7_EXPERIMENT.md](legacy_reports/TASK7_EXPERIMENT.md) for the paper alignment audit and precise evaluation limits. The viewer's labeled model video is the original 50-action rollout; its loss plots still use recorded demonstrations.

## Sources

- [LeRobot installation](https://huggingface.co/docs/lerobot/installation)
- [LeRobot LIBERO dataset and training](https://huggingface.co/docs/lerobot/libero)
- [LIBERO dataset](https://huggingface.co/datasets/lerobot/libero)
- [SmolVLA LIBERO checkpoint](https://huggingface.co/lerobot/smolvla_libero)
