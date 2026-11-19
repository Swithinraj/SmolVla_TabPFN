# Continue on the cluster: hand-over guide

Repository: https://github.com/Swithinraj/SmolVla_TabPFN  
Checkpoints, datasets and evaluation outputs: the shared Google Drive folder `SmolVLA_TabPFN_SO101` (link: `<DRIVE_FOLDER_LINK>`,
add it here; it is deliberately not written into this public file). Its own README lists the layout.

## STRICT RULE: one GPU, one V100

Every job uses **exactly one GPU, a V100**. No exceptions, no multi-GPU training, no data parallel, no "two jobs on two GPUs".

- Slurm: `#SBATCH --gres=gpu:1` and nothing larger. Do not raise it.
- Every script sources [scripts/cluster/env.sh](../scripts/cluster/env.sh), which sets `CUDA_VISIBLE_DEVICES=0` and **aborts if
  PyTorch sees anything other than exactly one GPU** (it warns if the GPU is not a V100). Keep using it; do not bypass it.
- Do not launch `accelerate launch --num_processes>1` or `torchrun` with more than one process.
- Run one GPU job at a time per allocation; TabPFN (about 1 GB) and the policy share that one GPU.
- On a V100 stay in fp32 (what we trained in). V100 has no native bf16; do not enable bf16 or autocast without testing.

## Where the project stands (2026-11-19)

Simulation only; nothing was run on a real arm. An SO-101 arm in MuJoCo catches and holds a rolling ball. SmolVLA turns images,
joint state and a TabPFN forecast of the ball's arrival into joint commands. Main model: checkpoint
`5_catch_big_012000` (Drive) = 12,000 steps on 400 randomized demonstrations. Results (24 launches per cell, "smoother"
controller; see [RESULTS.md](RESULTS.md)):

| target given to the policy | clean in-dist | clean hard | randomized in-dist | randomized hard |
|---|---|---|---|---|
| exact forecast | 19/24 | 19/24 | 18/24 | 14/24 |
| TabPFN forecast | 16/24 | 9/24 | 8/24 | 5/24 |
| none | 10/24 | 6/24 | 11/24 | 1/24 |

Three problems to fix, in order of importance:
1. **The policy depends on a blue marker** that the simulator draws at the predicted crossing. A real camera cannot show that.
   With the marker hidden and TabPFN targets, hard launches drop from 9/24 to 1/24.
2. **Jerk is 9k–16k deg/s³** against the 4000 score threshold (the scripted expert is about 1.6k); catches score about 0.3 of a point.
3. **TabPFN targets catch much less than exact targets on hard launches** (9 vs 19 of 24); likely the unpredictable mid-roll nudge
   plus the policy not trusting a noisy forecast.

## 1. Set up the cluster environment

The repository's `.ps1` scripts are Windows-only and hard-code `d:\smolVLA_Enhance`. Use `scripts/cluster/*.sh` instead
(written from the PowerShell scripts, **not yet run on a cluster**; expect small fixes).

```bash
git clone https://github.com/Swithinraj/SmolVla_TabPFN.git && cd SmolVla_TabPFN
python3.13 -m venv .venv && source .venv/bin/activate          # Python 3.13 was used; 3.11+ should work
pip install "lerobot[smolvla]==0.7.0" tabpfn mujoco opencv-python matplotlib av
python -c "import torch;print(torch.__version__, torch.cuda.get_arch_list())"
```

- **V100 is compute capability 7.0 (`sm_70`).** Check that `sm_70` is in the printed arch list. We used torch 2.9.1+cu128 on an
  RTX 3050; if the wheel you get lacks `sm_70`, install a torch 2.9.x wheel built for CUDA 12.6 or older
  (`pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126`), which still ships Volta kernels. Not verified here.
- Versions that produced the results: lerobot 0.7.0, torch 2.9.1, mujoco 3.3.7, tabpfn 10.2.0, transformers 5.22.2, numpy 2.2.6.
- **Headless rendering:** `MUJOCO_GL=egl` (default in env.sh). If EGL fails on the node, use `MUJOCO_GL=osmesa` (CPU, far slower).
- **Internet on compute nodes** may be blocked. On a login node, once: download `lerobot/smolvla_base`, the SmolVLM2-500M backbone it
  loads, and the TabPFN v2 regressor weights into `.cache/` (`HF_HOME`, `TABPFN_MODEL_CACHE_DIR` as set in env.sh; run
  `python -c "from tabpfn import TabPFNRegressor; ..."` or one short evaluation to trigger the download). The training script
  runs offline after that; recording and evaluation set `HF_HUB_OFFLINE=0` for TabPFN, so either allow the hub or pre-download.
- RAM: recording and evaluation want 16 GB or more; training 32 GB is comfortable.

## 2. Get the checkpoints and datasets

```bash
rclone copy gdrive:SmolVLA_TabPFN_SO101 ./drive_pull --drive-root-folder-id <FOLDER_ID> --progress   # or download in the browser / gdown
mkdir -p checkpoints data outputs/catch_big
mv drive_pull/checkpoints/* checkpoints/
mv drive_pull/datasets/catch_v2_merged data/so101_catch_v2
mv drive_pull/datasets/catch_v2_shards/A data/catch_v2_A ; mv drive_pull/datasets/catch_v2_shards/B data/catch_v2_B
mv drive_pull/datasets/blocking_stage2_train data/so101_ball_train
mv drive_pull/datasets/blocking_stage3_forecast_train data/so101_ball_fc_train
mv drive_pull/datasets/blocking_heldout data/so101_ball_heldout
```

The checkpoint folders contain a `pretrained_model/` directory (that is what `--policy.path` and `--checkpoints` take).

## 3. Smoke tests before any long job (about 30 minutes in total)

```bash
source scripts/cluster/env.sh                     # must print one GPU
python challenge/sim/hsv_tracker.py                 # detector accuracy: expect ~236/240 clean, 0.05 cm median error
python challenge/sim/test_catch.py --help           # simulator imports and renders headless
bash scripts/cluster/eval_catch.sh checkpoints/5_catch_big_012000/pretrained_model outputs/smoke --splits indist --n 4 \
     --modes oracle --variants smoother --realism 0 --gifs 0
```

Pass criterion: the smoke evaluation prints `held x/4` with `x >= 2` (oracle, clean, in-distribution; the reference is 19/24). If it prints 0,
fix the environment (rendering, versions, camera-name mapping) before doing anything else.

## 4. What to do next, in order

Each item has its success test; report with the same evaluation settings (same seeds, n = 24, then larger) so numbers compare.

### 4.1 Remove the marker for real (highest priority)
- Re-record the dataset without the blue dot: `NO_MARKER=1` is already supported by the recorder and the evaluator
  (`bash scripts/cluster/record_shard.sh A 900000 200 1`, shard B with another seed range such as 1000000; merge with
  `python scripts/aggregate_v2.py`; check that it reads the shard names you used).
- **Code still to write:** the recorder feeds the TabPFN forecaster the exact ball track plus Gaussian noise; the deployment
  setting is HSV detections. Make `record_catch_dataset_v2.py` render the top camera each step and use `hsv_tracker.HSVTracker`
  (as `eval_catch.py --tracker hsv` already does) so training and deployment inputs match.
- Train from `checkpoints/2_stage2_blocking_015000` or from the big checkpoint (both are plausible; try the cheaper one first):
  `bash scripts/cluster/train_catch.sh <init>/pretrained_model data/<new_dataset> outputs/catch_nomarker 12000 4 4000`.
- Evaluate with `NO_MARKER=1 ... --tracker hsv`. **Success:** TabPFN targets on the hard split at or above 9/24 without the marker.

### 4.2 Smoothness
- Cheap tests first (inference only): more chunk blending, executing fewer steps of each chunk, a larger EMA (the 0.7 setting is
  already the best on points).
- Then a training change: fine-tune on the expert's own smooth min-jerk trajectories resampled to the chunk length, or add a
  jerk penalty. **Success:** median jerk under 6000 with the held rate unchanged; points per launch above 0.6 with exact targets.

### 4.3 Make the policy use noisy forecasts
- Noise augmentation of the forecast inputs inside the data, randomized forecast delay (0.15–0.30 s) and sigma-aware training.
- **Success:** hard-launch TabPFN rate closer to exact targets (9 → 14+ of 24).

### 4.4 The gate, done properly
- Use TabPFN sigma to decide *when to commit the close and lift*, not to hide the forecast (hiding it gave 2/24). Test with the
  scripted expert first, then the policy; add a scripted fallback when confidence is low.

### 4.5 Evidence quality
- n ≥ 60 launches per cell and 2–3 training seeds. At n = 24 one standard error is about ±10 points.
- A run where the nudge time is revealed to the forecaster, to confirm it explains the exact/TabPFN gap.

### 4.6 The queued TabPFN spatial-brain task
LIBERO backend numbers (check the checkpoint repo ids exist: `BurgQ/smolvla-libero-bs32-100k`, `crislmfroes/smolvla_libero`;
never reuse LIBERO normalization for the SO-101), the multi-zone SO-101 demo (4–6 zones, 10 demos per zone, canonical prompts),
live few-shot adaptation of the ball forecaster, and a deterministic fallback executor. Simulated zone prediction, the confidence
gate and few-shot adaptation are done ([RESULTS.md §5](RESULTS.md#5-tabpfn-spatial-brain-zone-prediction-uncertainty-gate-few-shot-adaptation)).

### 4.7 Before any real arm
Everything in [DEPLOYMENT_SAFETY.md](../DEPLOYMENT_SAFETY.md). Nothing here was validated on hardware.

## 5. Running jobs on the cluster

Template: [scripts/cluster/sbatch_example.sbatch](../scripts/cluster/sbatch_example.sbatch) (`--gres=gpu:1`).

| job | script | one-GPU resource use (our machine, a 4 GB RTX 3050) |
|---|---|---|
| train, batch 4, 12,000 steps | `train_catch.sh` | 2.3 h at 1.4 steps/s; a V100 should be faster (not measured) |
| record 200 episodes | `record_shard.sh` | hours, CPU-bound (rendering + TabPFN); give it many CPU cores |
| evaluate, 24 launches per cell | `eval_catch.sh` | 10–15 min per cell; the full main table about 7 h |

Do not change the batch size or learning rate casually: all results used batch 4, lr 1e-4, 200 warm-up steps, cosine decay to
lr/40. A larger batch on the V100's 16/32 GB may be tried as a separate, labelled experiment.

## 6. Rules and pitfalls
- **One V100, always** (see the top).
- Keep evaluation seeds and the launch filter as they are (launches must be catchable by the scripted expert); change n, not the seeds.
- Never train or evaluate with another robot's normalization statistics; each checkpoint carries its own.
- State what is simulated. No real-hardware claims.
- After new results: run `python scripts/make_figures.py`, update [RESULTS.md](RESULTS.md), [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) and
  [TODO.md](TODO.md), and push. Put new checkpoints and datasets on the Drive folder in the same layout.
- Known weak spots in the code: the PowerShell scripts' absolute paths; `scripts/aggregate_v2.py` expects the shard folders
  `data/catch_v2_A` and `data/catch_v2_B` and was only tested on Windows; `make_figures.py` reads the forecast-check numbers from the
  printed log (`results/catch_big/forecast_check.txt`), not from JSON.
