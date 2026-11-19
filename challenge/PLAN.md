# SO-101 red-ball interception: TabPFN + SmolVLA

Status: design selected; simulation first. No SO-101 interception simulator, dataset, or trained policy exists yet. The previous LIBERO data and checkpoints are for a Franka Panda and cannot be used as SO-101 demonstrations.

## Goal

An SO-101 arm blocks a **rolling red ball** on a tabletop before it crosses a fixed line. Use a small paddle attached to or held by the gripper in simulation. A rolling ball gives a reachable, measurable first task; flying-ball catching would require depth sensing, faster actuation, and a separate feasibility study.

## Why both models are here

1. An overhead camera observes the ball and robot. Calibration maps image pixels to table coordinates (x, y); the ball center is at a known height z above the tabletop. Tracking supplies time-stamped positions and estimated velocity. Raw pixels go to SmolVLA, while the measured coordinates go to TabPFN.
2. TabPFN predicts the ball's future crossing position and time, with quantiles for uncertainty. It consumes a table of recent and previous launch trajectories. This is the challenge's central spatial-prediction component.
3. A goal marker is rendered at the predicted intercept point in the camera image. SmolVLA sees the marked image, the SO-101 joint state, and an instruction such as "Block the red ball at the marked point." It predicts SO-101 joint actions. The same marker generation must be present in demonstration recording and deployment so the policy learns to use it.
4. A reachability and timing gate sends only feasible targets. If the ball arrives before the arm can reach or the prediction is too uncertain, the arm stays at a safe ready pose. A low-level controller executes joint targets and enforces joint limits. The ball prediction must measurably change the commanded blocking position.

## Simulation and data

- Build a local MuJoCo scene with **only an SO-101 robot**, based on the official [SO-101 tabletop environment](https://huggingface.co/lerobot/so101-tabletop-mujoco). Replace the cubes with a red ball, launch lane, blocking line, and gripper paddle. Run the simulator in WSL if its dependencies require Linux; keep Windows for training and the browser viewer.
- Randomize ball launch point and speed, rolling friction, camera placement, and lighting. Start with slow, straight rolls; add bounces and friction changes only after the basic task works.
- Record a LeRobot dataset of successful SO-101 blocking demonstrations, with images, joint states, actions, task text, timestamps, ball coordinates, and the rendered goal marker. Generate a held-out set of launch seeds and friction settings before training.
- Use `lerobot/smolvla_base` for the **SO-101 dataset**. The current `smolvla_libero` checkpoint controls a Franka Panda with a different action space and is unsuitable as the interception policy.
- Train TabPFN regressors on causal trajectory rows: current/past ball x, y, velocity, elapsed time, and prediction horizon as features; future x/y or crossing x/time as targets. A row may only use observations available at its timestamp. Quantiles provide an uncertainty estimate, but coverage must be checked on held-out launches.

## Evaluation that proves the idea

Report at least 30 held-out launches per condition, including new friction or launch speeds. Compare:

| Condition | Ball forecast | SO-101 control |
| --- | --- | --- |
| SmolVLA alone | None | SmolVLA |
| Physics baseline + SmolVLA | Constant-velocity / fitted deceleration | SmolVLA with marker |
| TabPFN + SmolVLA | TabPFN crossing prediction | SmolVLA with marker |
| Oracle ceiling | Simulator's true crossing point | SmolVLA with marker |

Measure interception success, crossing-point error, arrival-time error, quantile coverage, arm travel time, and end-to-end decision latency. The TabPFN condition needs to improve success or robustness under changed dynamics to support the challenge claim. Quantile intervals are predictions, not guaranteed confidence intervals.

## Build order

1. SO-101 MuJoCo red-ball environment and a browser viewer showing camera video, ball track, predicted crossing point, arm target, and result.
2. Tracker/calibration and a simple physics forecast. Verify reachable launches and control timing before involving either model.
3. TabPFN forecast benchmark and latency measurement. Prefer the official local `TabPFNRegressor` API; pin the model version and respect the weight license.
4. LeRobot SO-101 demonstrations with goal marker; fine-tune SmolVLA. Evaluate with an oracle marker first, then TabPFN's marker.
5. Ablations and domain-shift runs with held-out seeds. Only then consider physical SO-101 deployment.

## Practical limits

- The current Windows RTX 3050 has 4 GB VRAM. Running TabPFN and SmolVLA simultaneously on it may not fit or meet the ball's deadline; benchmark separately, use CPU/offload or a second GPU for TabPFN if needed, and keep the ball slow enough for a measured control loop.
- SO-101 joint-position commands and camera calibration must match the simulated robot and the eventual real arm. Establish limits and an emergency stop before any physical trial.
- TabPFN's newer local weights have a non-commercial license. The [Prior Labs repository](https://github.com/PriorLabs/TabPFN) distinguishes code and weight licenses; model access may require accepting terms. The regressor supports [quantile predictions](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/regressor.py).

Sources: [SmolVLA usage and SO-101 deployment](https://huggingface.co/docs/lerobot/main/smolvla), [LeRobot simulated robot interface](https://huggingface.co/docs/lerobot/simulation), [official SO-101 tabletop environment](https://huggingface.co/lerobot/so101-tabletop-mujoco), [Prior Labs TabPFN](https://github.com/PriorLabs/TabPFN).
