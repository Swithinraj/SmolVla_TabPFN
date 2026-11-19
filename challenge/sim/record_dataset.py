"""Record scripted SO-101 ball-blocking demonstrations as a LeRobot v3 dataset.

Cameras: top (sees ball, arm and the blue goal marker at the predicted crossing point) and front.
Ball ground-truth tracks and launch parameters go to a sidecar (not LeRobot features) for TabPFN training.
Usage: python record_dataset.py --episodes 300 --root <dir> [--split train|heldout]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ball_env import HOME, JOINTS, BallBlockEnv  # noqa: E402
from expert import Expert  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

TASK = "Block the red ball at the marked point."
NAMES = [f"{j}.pos" for j in JOINTS]
TAIL_STEPS = 15
# Friction ranges: train covers rolling friction up to 1e-4; "heldout" is a shifted regime.
SPLITS = {"train": dict(rf=(0.0, 1.0e-4), speed=(0.10, 0.22)),
          "heldout": dict(rf=(1.2e-4, 2.0e-4), speed=(0.18, 0.26))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train", choices=list(SPLITS))
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if root.exists():
        if not a.overwrite:
            sys.exit(f"{root} exists; pass --overwrite")
        shutil.rmtree(root)
    cfg = SPLITS[a.split]
    env = BallBlockEnv(cameras=("top", "front"), height=a.height, width=a.width, episode_seconds=4.0)
    expert = Expert(env)
    img = {"dtype": "video", "shape": (a.height, a.width, 3), "names": ["height", "width", "channels"]}
    features = {
        "observation.images.top": img,
        "observation.images.front": img,
        # 6 joints + numeric forecast: crossing y at the paddle plane (m) and time until the ball gets there (s).
        "observation.state": {"dtype": "float32", "shape": (8,), "names": NAMES + ["target_y", "time_to_cross"]},
        "action": {"dtype": "float32", "shape": (6,), "names": NAMES},
    }
    ds = LeRobotDataset.create(repo_id=f"local/so101_ball_{a.split}", fps=env.fps, features=features,
                               root=root, robot_type="so101_follower", use_videos=True)
    sidecar, kept, tried = [], 0, 0
    while kept < a.episodes:
        seed = a.seed0 + tried
        tried += 1
        rng = np.random.default_rng(10_000 + seed)
        obs = env.reset(seed=seed, speed=rng.uniform(*cfg["speed"]), rolling_friction=rng.uniform(*cfg["rf"]))
        c = expert.plan(start_delay=float(rng.uniform(0.0, 0.3)), min_move=float(rng.uniform(0.3, 0.6)))
        if c is None or not expert.ok:
            continue
        env.set_marker(expert.ik.paddle(np.deg2rad(expert.q_goal[:5]))[0][0], c[0])
        obs = env.observe()
        track, frames, done, tail = [], [], False, 0
        while tail < TAIL_STEPS:
            action = expert.act(env.t / env.fps).astype(np.float32)
            target = np.array([c[0], max(c[1] - env.t / env.fps, 0.0)], dtype=np.float32)
            frames.append(dict(top=obs["top"], front=obs["front"],
                               state=np.concatenate([obs["state"], target]).astype(np.float32), action=action))
            track.append([env.t / env.fps, *env.ball_pos()[:2]])
            obs, done, info = env.step(action)
            if done:
                tail += 1
        if info["crossed"]:
            continue  # expert failed; do not keep failures
        for f in frames:
            ds.add_frame({"observation.images.top": f["top"], "observation.images.front": f["front"],
                          "observation.state": f["state"], "action": f["action"], "task": TASK})
        ds.save_episode()
        sidecar.append(dict(episode=kept, seed=seed, launch=env.launch, crossing_y=c[0], crossing_t=c[1],
                            track=np.round(track, 5).tolist()))
        kept += 1
        if kept % 25 == 0:
            print(f"{kept}/{a.episodes} episodes ({tried} launches tried)", flush=True)
    ds.finalize()
    (root / "ball_sidecar.json").write_text(json.dumps(sidecar))
    print(f"done: {kept} episodes, {ds.meta.total_frames} frames -> {root}")


if __name__ == "__main__":
    main()
