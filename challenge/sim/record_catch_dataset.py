"""Record SO-101 catch-and-hold demos (scripted expert, privileged teacher) with forecast-conditioned inputs.

Policy inputs (state dims 7-12 + blue marker at the target crossing point):
  [y_cross, time_to_cross (countdown), sigma_y, y(+0.3 s), y(+0.6 s), y(+0.9 s)]
  from one of: oracle (exact, 30%), TabPFN (50%, noisy 4 mm tracker, forecast every 3 steps, compute delay),
  constant-velocity (20%). Mixing exact and noisy targets teaches the policy to FOLLOW the target first and to
  tolerate forecast error second (stage 3, trained on noisy targets only, learned to ignore the target).
Labels (actions): scripted catch expert planned from the TRUE crossing. Only episodes that end holding the ball are kept.
Launch mix: half easy, half irregular (hidden slope + rough patch + 50% random mid-roll nudge).
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
from ball_env import JOINTS  # noqa: E402
from catch_env import CatchEnv  # noqa: E402
from catch_expert import CatchExpert  # noqa: E402
from expert import X_INT  # noqa: E402
from forecast import PATH_TAUS, ForecastPipeline, TabPFNForecasterV2, make_rows_v2  # noqa: E402
from launches import easy_launch, irregular_launch, mixed_irregular  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

TASK = "Catch the red ball at the marked point and hold it."
NAMES = [f"{j}.pos" for j in JOINTS]
EXTRA = ["target_y", "time_to_cross", "sigma_y", "y_plus_0.3s", "y_plus_0.6s", "y_plus_0.9s"]
NOISE = 0.004
HOLD_AFTER_LIFT = 0.7
KIND_P = (("oracle", 0.3), ("tabpfn", 0.5), ("cv", 0.2))


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


def oracle_vector(t, tr, ys, y_c, t_c):
    path = [np.interp(min(t + tau, t_c), tr, ys) for tau in PATH_TAUS]
    y = float(np.clip(y_c, -0.22, 0.22))
    return np.array([y, max(t_c - t, 0.0), 0.0, *np.clip(path, -0.22, 0.22)], np.float32), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if root.exists():
        if not a.overwrite:
            sys.exit(f"{root} exists; pass --overwrite")
        shutil.rmtree(root)
    env = CatchEnv(cameras=("top", "front"), height=a.height, width=a.width, episode_seconds=7.0)
    expert = CatchExpert(env)
    tp, delay = build_tabpfn(env, expert)
    print(f"tabpfn ready: forecast latency {1000 * delay:.0f} ms", flush=True)

    img = {"dtype": "video", "shape": (a.height, a.width, 3), "names": ["height", "width", "channels"]}
    spec = {
        "observation.images.top": img, "observation.images.front": img,
        "observation.state": {"dtype": "float32", "shape": (6 + len(EXTRA),), "names": NAMES + EXTRA},
        "action": {"dtype": "float32", "shape": (6,), "names": NAMES},
    }
    ds = LeRobotDataset.create(repo_id="local/so101_catch_train", fps=env.fps, features=spec, root=root,
                               robot_type="so101_follower", use_videos=True)
    kinds, probs = zip(*KIND_P)
    sidecar, kept, tried = [], 0, 0
    while kept < a.episodes:
        seed = a.seed0 + tried
        tried += 1
        rng = np.random.default_rng(30_000 + seed)
        hard = seed % 2 == 0
        (irregular_launch if hard else easy_launch)(env, seed, joint_noise=2.0)
        c = expert.plan(x_c=X_INT, start_delay=float(rng.uniform(0.0, 0.25)), min_move=0.5)
        if c is None or not expert.ok:
            continue
        kind = str(rng.choice(kinds, p=probs))
        pipe = ForecastPipeline(kind, tp=tp, delay=delay) if kind in ("tabpfn", "cv") else None
        tt, xt, yt, cc = expert.ghost.track()                      # arm-independent ball track (kick-aware)
        if cc is None:
            continue
        tr, ys = np.append(tt, cc[1]), np.append(yt, cc[0])
        xo, yo = xt + rng.normal(0, NOISE, len(xt)), yt + rng.normal(0, NOISE, len(yt))
        t_end = expert.t_lift + expert.lift_dur + HOLD_AFTER_LIFT
        frames, errs = [], []
        info = {}
        while env.t / env.fps < t_end and env.t < env.max_steps:
            i, t = env.t, env.t / env.fps
            if pipe is not None:
                if i < len(tt):
                    pipe.update(i, t, xo[i], yo[i])
                tv, my = pipe.vector(t)
            else:
                tv, my = oracle_vector(t, tr, ys, cc[0], cc[1])
            env.set_marker(*((X_INT, my) if my is not None else (0.8, 0.0)))
            if my is not None:
                errs.append(abs(my - cc[0]))
            obs = env.observe()
            action = expert.act(t).astype(np.float32)
            frames.append(dict(top=obs["top"], front=obs["front"],
                               state=np.concatenate([obs["state"], tv]).astype(np.float32), action=action))
            _, done, info = env.step(action, render=False)
            if done:
                break
        if not info.get("held", False):
            continue  # expert failed to hold; do not keep failures
        for f in frames:
            ds.add_frame({"observation.images.top": f["top"], "observation.images.front": f["front"],
                          "observation.state": f["state"], "action": f["action"], "task": TASK})
        ds.save_episode()
        sidecar.append(dict(episode=kept, seed=seed, irregular=hard, kind=kind, launch=env.launch,
                            crossing_y=cc[0], crossing_t=cc[1], frames=len(frames),
                            mean_target_y_err_m=float(np.mean(errs)) if errs else None))
        kept += 1
        if kept % 10 == 0:
            print(f"{kept}/{a.episodes} episodes ({tried} launches tried, {ds.meta.total_frames} frames)", flush=True)
    ds.finalize()
    (root / "sidecar.json").write_text(json.dumps(sidecar))
    print(f"done: {kept} episodes, {ds.meta.total_frames} frames -> {root}")


if __name__ == "__main__":
    main()
