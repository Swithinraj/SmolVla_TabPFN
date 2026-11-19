# SmolVLA local LIBERO tuning results

Run date: 2026-11-16. Hardware: RTX 3050 Laptop GPU (4 GB VRAM). LeRobot 0.7.0, CUDA PyTorch 2.9.1.

## Experiment

Starting model: `models/smolvla_libero` (`lerobot/smolvla_libero`). The initial fine-tuning runs used LIBERO episodes **1, 2, 4** (907 frames, two task types), 100 steps, batch size 1, action expert training, and no Hub upload. Episodes **3 and 5** were excluded from these local training runs. At batch size 1, 100 steps covered only about **0.11 epoch**.

For offline evaluation, the same fixed frames from episodes 3 and 5 were passed through the original and tuned policies. `policy loss` is LeRobot's SmolVLA forward loss. `first-action MAE` is the mean absolute difference between the model's first predicted 7D action and the demonstration action. Lower is better.

| Model | Learning rate | Held-out frames | Policy loss | First-action MAE | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Original LIBERO checkpoint | — | 32 | 0.06337 | **0.01742** | Recommended |
| Conservative tuning | 0.000001 | 32 | **0.06336** | 0.01765 | Effectively unchanged |
| Original LIBERO checkpoint | — | 8 | **0.03094** | **0.02074** | Reference for aggressive run |
| Aggressive tuning | 0.0001 | 8 | 0.34117 | 0.07584 | Worse |

The conservative 32-frame loss changed by less than 0.1%; action MAE rose by 1.3%. This sample does not show a useful improvement. The high-rate run clearly degraded performance. Continue using the original checkpoint for now. The training script defaults to the conservative learning rate for any future local experiments.

The original checkpoint was already trained on LIBERO, including this dataset. These frames are held out only from **our local tuning runs**. This is an offline imitation check, not a fresh generalization test or simulator success rate. Live LIBERO simulation remains unavailable on this Windows setup.

## Longer three-epoch run

To test whether the 100-step result was premature, the same three training episodes were used for **2,700 steps** at learning rate `0.000001` (about **2.98 epochs**). Checkpoints at 900, 1,800, and 2,700 steps were each evaluated on the same 32 fixed frames from episodes 3 and 5.

| Checkpoint | Approx. epochs | Held-out policy loss | Held-out first-action MAE |
| --- | ---: | ---: | ---: |
| Original LIBERO model | 0 | **0.06337** | **0.01742** |
| 900 steps | 0.99 | 0.06361 | 0.01752 |
| 1,800 steps | 1.99 | 0.06357 | 0.01761 |
| 2,700 steps | 2.98 | 0.06348 | 0.01765 |

![Training and held-out metric curves](tuning_20261116_162927/convergence.png)

Training loss fluctuated without a clear downward trend. Held-out loss stayed nearly flat, while first-action error rose slightly at each checkpoint. More epochs on these three episodes did not improve the model on this offline check. Continue using the original checkpoint.

The **complete 1,693-episode, 273,465-frame dataset** was downloaded after the initial experiments and is available locally. This longer run still used only the same three episodes so its checkpoints remain comparable. A full-dataset epoch at batch size 1 would take about 273,465 steps. The starting checkpoint was already trained on LIBERO, so this check cannot establish fresh generalization or simulator success.

## Artifacts

- Conservative run: `tuning_20261116_161415/checkpoints/000100/pretrained_model/`
- 32-frame report: `tuning_20261116_161415/evaluation_32.json`
- Conservative training log: `tuning_20261116_161415/training.log`
- Aggressive run: `tuning_20261116_161124/checkpoints/000100/pretrained_model/`
- 8-frame report: `tuning_20261116_161124/evaluation.json`
- Longer run training log: `tuning_20261116_162927/training.log`
- Longer run checkpoints: `tuning_20261116_162927/checkpoints/000900/`, `001800/`, `002700/`
- Longer run reports: `tuning_20261116_162927/evaluation_32_000900.json`, `evaluation_32_001800.json`, `evaluation_32_002700.json`
- Loss and action error chart: `tuning_20261116_162927/convergence.png`

To repeat the 32-frame evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tuning.py --tuned outputs\tuning_20261116_161415\checkpoints\000100\pretrained_model --samples-per-episode 16
```
