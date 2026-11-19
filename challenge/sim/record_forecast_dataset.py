"""Record SO-101 blocking demos where the policy's target inputs are a realistic FORECAST (v2), not the oracle.

Policy inputs (state dims 7-12 + blue marker at the forecast crossing point), exactly as in deployment:
  [y_cross, time_to_cross (countdown), sigma_y (calibrated 10-90% band), y(+0.3 s), y(+0.6 s), y(+0.9 s)]
  from a noisy 4 mm tracker, a forecast every 3 steps, available only after the compute delay,
  absent (zeros, marker hidden) until 6 track frames exist. Forecaster: TabPFN (70%) or constant velocity (30%).
Labels (actions): scripted expert planned from the TRUE crossing point (privileged teacher).
Launch mix: half easy, half irregular (hidden slope + rough patch + 50% random mid-roll nudge).
Usage: python record_forecast_dataset.py --episodes 400 --root data\\so101_ball_fc_train
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
from ball_env import JOINTS, BallBlockEnv  # noqa: E402
from expert import X_INT, Expert  # noqa: E402
from forecast import ForecastPipeline, TabPFNForecasterV2, make_rows_v2  # noqa: E402
from launches import easy_launch, irregular_launch, mixed_irregular  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

TASK = "Block the red ball at the marked point."
NAMES = [f"{j}.pos" for j in JOINTS]
EXTRA = ["target_y", "time_to_cross", "sigma_y", "y_plus_0.3s", "y_plus_0.6s", "y_plus_0.9s"]
TAIL_STEPS = 15
NOISE = 0.004


def build_tabpfn(env, expert):
    rng0 = np.random.default_rng(1)
    X, T, P = make_rows_v2(expert.ghost, env, lambda e, s: mixed_irregular(e, s), 1500, 0, 4, rng0, noise=NOISE)
    tp = TabPFNForecasterV2().fit(X, T, P)
    probe = X[0]
    tp.predict(probe)
    t1 = time.time()
    for _ in range(5):
        tp.predict(probe)
    return tp, (time.time() - t1) / 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed0", type=int, default=800_000)
    ap.add_argument("--tabpfn-frac", type=float, default=0.7)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if root.exists():
        if not a.overwrite:
            sys.exit(f"{root} exists; pass --overwrite")
        shutil.rmtree(root)
    env = BallBlockEnv(cameras=("top", "front"), height=a.height, width=a.width, episode_seconds=4.0)
    expert = Expert(env)
    tp, delay = build_tabpfn(env, expert)
    print(f"tabpfn ready: forecast latency {1000 * delay:.0f} ms", flush=True)

    img = {"dtype": "video", "shape": (a.height, a.width, 3), "names": ["height", "width", "channels"]}
    spec = {
        "observation.images.top": img, "observation.images.front": img,
        "observation.state": {"dtype": "float32", "shape": (6 + len(EXTRA),), "names": NAMES + EXTRA},
        "action": {"dtype": "float32", "shape": (6,), "names": NAMES},
    }
    ds = LeRobotDataset.create(repo_id="local/so101_ball_fc_train", fps=env.fps, features=spec, root=root,
                               robot_type="so101_follower", use_videos=True)
    sidecar, kept, tried = [], 0, 0
    while kept < a.episodes:
        seed = a.seed0 + tried
        tried += 1
        rng = np.random.default_rng(30_000 + seed)
        hard = seed % 2 == 0
        (irregular_launch if hard else easy_launch)(env, seed, joint_noise=2.0)
        c = expert.plan(start_delay=float(rng.uniform(0.0, 0.3)), min_move=float(rng.uniform(0.3, 0.6)))
        if c is None or not expert.ok:
            continue
        use_tabpfn = rng.random() < a.tabpfn_frac
        pipe = ForecastPipeline("tabpfn" if use_tabpfn else "cv", tp=tp, delay=delay if use_tabpfn else 0.0)
        tt, xt, yt, _ = expert.ghost.track()                       # arm-independent ball track
        xo, yo = xt + rng.normal(0, NOISE, len(xt)), yt + rng.normal(0, NOISE, len(yt))
        frames, tail, done, errs = [], 0, False, []
        while tail < TAIL_STEPS:
            i, t = env.t, env.t / env.fps
            if i < len(tt):
                pipe.update(i, t, xo[i], yo[i])
            tv, my = pipe.vector(t)
            env.set_marker(*((X_INT, my) if my is not None else (0.8, 0.0)))
            if my is not None:
                errs.append(abs(my - c[0]))
            obs = env.observe()
            action = expert.act(t).astype(np.float32)
            frames.append(dict(top=obs["top"], front=obs["front"],
                               state=np.concatenate([obs["state"], tv]).astype(np.float32), action=action))
            _, done, info = env.step(action, render=False)
            if done:
                tail += 1
        if info["crossed"]:
            continue  # expert failed; do not keep failures
        for f in frames:
            ds.add_frame({"observation.images.top": f["top"], "observation.images.front": f["front"],
                          "observation.state": f["state"], "action": f["action"], "task": TASK})
        ds.save_episode()
        sidecar.append(dict(episode=kept, seed=seed, irregular=hard, forecaster="tabpfn" if use_tabpfn else "cv",
                            launch=env.launch, crossing_y=c[0], crossing_t=c[1],
                            mean_target_y_err_m=float(np.mean(errs)) if errs else None))
        kept += 1
        if kept % 25 == 0:
            print(f"{kept}/{a.episodes} episodes ({tried} launches tried)", flush=True)
    ds.finalize()
    (root / "sidecar.json").write_text(json.dumps(sidecar))
    print(f"done: {kept} episodes, {ds.meta.total_frames} frames -> {root}")


if __name__ == "__main__":
    main()
