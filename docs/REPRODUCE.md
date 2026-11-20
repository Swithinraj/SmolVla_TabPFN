# Reproducing the results

Tested on Windows 11, PowerShell, Python 3.13, an RTX 3050 laptop GPU with **4 GB** VRAM, 16 CPU cores and about 16 GB RAM.
GPU memory is tight: SmolVLA training (batch 4) fits; the evaluator needs the policy and TabPFN (about 1 GB) at once, so run
only one GPU job at a time. The PowerShell scripts contain absolute paths `d:\smolVLA_Enhance`; either clone to that path or edit the
`Set-Location` / `$env:` lines at the top of each script.

## 0. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\pip install "lerobot[smolvla]==0.7.0" tabpfn mujoco opencv-python matplotlib av
# torch 2.9.1+cu128 was installed with the CUDA 12.8 wheel index; the exact install commands were not saved, so pin the versions below
```

Versions used: lerobot 0.7.0, torch 2.9.1+cu128, mujoco 3.3.7, tabpfn 10.2.0, transformers 5.22.2, numpy 2.2.6.
Required downloads (one time, needs the Hugging Face hub; set `HF_HUB_OFFLINE=0`): `lerobot/smolvla_base`, the SmolVLM2-500M
backbone it loads, and the TabPFN v2 regressor weights (a Hugging Face login may be requested; the scripts set
`TABPFN_NO_BROWSER=1`, put the weights in `.cache/tabpfn` via `TABPFN_MODEL_CACHE_DIR`). The training data scripts
set `HF_LEROBOT_HOME=.cache/lerobot`.

## 1. Stages and where each checkpoint comes from

| stage | what | script | output |
|---|---|---|---|
| 1 | SmolVLA on real SO-101 pick-and-place (`lerobot/svla_so101_pickplace`), `train_so101.ps1` | `train_so101.ps1` | `outputs/so101_pickplace_*` |
| 2 | blocking demonstrations in sim, 15,000 steps | `challenge/sim/record_dataset.py`, `train_ball.ps1` | `outputs/so101_ball_<timestamp>` |
| 3 | blocking with noisy-forecast targets (negative result) | `scripts/stage3_pipeline.ps1` | `outputs/so101_ball_stage3` |
| catch pilot | catch demos, 3,000 steps from stage 2 | `scripts/record_v2_shard.ps1`, `scripts/catch_v2_pipeline.ps1` | `outputs/catch_pilot_s2` |
| catch big | 12,000 steps from the pilot | `scripts/catch_big_pipeline.ps1` | `outputs/catch_big` |

Stage 2 and 3 are the earlier blocking task and are only needed to reproduce Section 1 of the results. **To reproduce the
catch results you need the stage-2 checkpoint (`outputs/so101_ball_*/checkpoints/015000`) as the initial weights**; the
pipelines look for the oldest `outputs/so101_ball_2*` run. To skip it, pass `-Init lerobot/smolvla_base` to `train_ball.ps1` (the
result of doing so was not tested).

## 2. Record the catch dataset (randomized, sim-to-real)

Two shards of 200 episodes each, run in parallel (a few hours on the tested machine; the exact time was not recorded; 25% of episodes clean, the rest
randomized; only episodes where the expert holds the ball are kept):

```powershell
powershell -File scripts\record_v2_shard.ps1 -Shard A -Seed0 1000000 -Episodes 200
powershell -File scripts\record_v2_shard.ps1 -Shard B -Seed0 2000000 -Episodes 200
.\.venv\Scripts\python.exe scripts\aggregate_v2.py      # -> data/so101_catch_v2 (400 episodes, 37,636 frames)
```

These are the seeds used (shard A used seeds 900000–900257, B 1000000–1000259 after skipping launches the expert could not catch). Targets inside the data are a mix of exact (30%), TabPFN (50%) and
constant-velocity (20%) forecasts; the forecaster delay is clamped to 0.15–0.30 s.

## 3. Train

```powershell
# pilot (about 35 min)
powershell -File train_ball.ps1 -Init <stage2>\checkpoints\015000\pretrained_model -Dataset data\so101_catch_v2 `
   -Steps 3000 -BatchSize 4 -SaveFreq 3000 -OutputDir outputs\catch_pilot_s2
# big run (about 2.3 h): from the pilot checkpoint
powershell -File train_ball.ps1 -Init outputs\catch_pilot_s2\checkpoints\003000\pretrained_model -Dataset data\so101_catch_v2 `
   -Steps 12000 -BatchSize 4 -SaveFreq 4000 -OutputDir outputs\catch_big
```

`train_ball.ps1` sets learning rate 1e-4 with a 200-step warm-up and cosine decay, trains only the action expert (vision
encoder and VLM frozen), brightness/contrast jitter 0.8–1.2, and the camera rename top→camera1, front→camera2.
Normalization statistics are recomputed from the new dataset (never reuse statistics from another robot).

## 4. Evaluate

```powershell
$env:HF_HUB_OFFLINE = '0'
# main table: 24 launches per cell, clean and randomized, every variant, with the safety filter
.\.venv\Scripts\python.exe challenge\sim\eval_catch.py --checkpoints "big=outputs\catch_big\checkpoints\012000\pretrained_model" `
   --splits hard,indist --n 24 --modes oracle,tabpfn,none --variants smooth,smoother,safe --realism 0,1 --safety `
   --gifs 2 --out outputs\catch_big\eval
# forecast accuracy by lead time (about 8 min)
.\.venv\Scripts\python.exe challenge\sim\forecast_check.py
# marker hidden / HSV-only ball tracking (real-world inputs)
$env:NO_MARKER = '1'
.\.venv\Scripts\python.exe challenge\sim\eval_catch.py --checkpoints "big=...\012000\pretrained_model" --splits hard,indist `
   --n 24 --modes tabpfn --variants smoother --realism 0,1 --tracker hsv --out outputs\catch_big\eval_hsv
# spatial brain (zones, gate, few-shot adaptation)
.\.venv\Scripts\python.exe challenge\sim\spatial_brain.py
# HSV detector accuracy on its own
.\.venv\Scripts\python.exe challenge\sim\hsv_tracker.py
# graphs and results/summary.json
.\.venv\Scripts\python.exe scripts\make_figures.py
```

A cell takes about 10–15 minutes (24 launches with policy inference every 5 steps); the full main table takes about 7 hours.
`--references` adds the do-nothing and scripted-expert baselines. `eval_catch.py` selects launches by seed, so the same seeds
reproduce the same launches; the randomized condition uses seed `777 + launch seed`.

## 5. Sim-to-real layer

`challenge/sim/sim2real.py`, per episode: camera position ±2 cm and orientation ±2.5°, field of view ±4°, light and headlight
intensity ±40%, table, floor, wall and **arm** colours (the real arm is not turquoise), ball tint (stays red-dominant), image
noise and blur, actuator gain ±20%, joint damping and friction, joint-state bias (1°) and noise, control latency of 0–3 steps,
a servo speed limit of 7–12°/step, and tracker noise, dropouts and one-frame lag. Strength 0 is the clean simulator.

## 6. Safety

`challenge/sim/safety.py` and [DEPLOYMENT_SAFETY.md](../DEPLOYMENT_SAFETY.md). Read the latter before any real-arm use:
nothing here has been validated on hardware.

## 7. On a Slurm cluster (one V100)

Tested on 2026-11-19/20 ([EXPERIMENT_LOG](EXPERIMENT_LOG.md#phase-e-on-the-cluster-one-v100-2026-11-1920)). Every job uses
exactly one GPU; `scripts/cluster/env.sh` aborts if PyTorch sees more than one. Partition names and the node exclusion list in
the `.sbatch` files are those of the tested cluster; adapt them. Submit everything from the repo root.

```bash
mkdir -p logs && sbatch scripts/cluster/setup_env.sbatch        # venv, pinned versions, LFS meshes, model weights
sbatch scripts/cluster/cpu_pull_drive_all.sbatch                # checkpoints/datasets from the shared Drive folder
sbatch scripts/cluster/gpu_check.sbatch                         # one V100, sm_70, EGL rendering, HSV detector
bash scripts/cluster/submit_night.sh                            # the 2026-11-19 chain: record (no marker, HSV) -> train x2 -> eval -> sweep
```

`scripts/cluster/disk_guard.sh` keeps the workspace under a disk budget (jobs refuse to start at 270 GB and are cancelled at
290 GB); set the measured folder in the git-ignored `scripts/cluster/local.env` (`DISK_WS=...`).

