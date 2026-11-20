"""v2 (sim-to-real): same as record_catch_dataset.py plus per-episode randomization (sim2real.Realism): camera/light/colour
randomization, arm gain/friction, state bias+noise, 0-3 step control latency (labels lead the latency), tracker dropouts.

Record SO-101 catch-and-hold demos (scripted expert, privileged teacher) with forecast-conditioned inputs.

Policy inputs (state dims 7-12 + blue marker at the target crossing point):
  [y_cross, time_to_cross (countdown), sigma_y, y(+0.3 s), y(+0.6 s), y(+0.9 s)]
  from one of: oracle (exact, 30%), TabPFN (50%, noisy 4 mm tracker, forecast every 3 steps, compute delay),
  constant-velocity (20%). Mixing exact and noisy targets teaches the policy to FOLLOW the target first and to
  tolerate forecast error second (stage 3, trained on noisy targets only, learned to ignore the target).
Labels (actions): scripted catch expert planned from the TRUE crossing. Only episodes that end holding the ball are kept.
Launch mix: half easy, half irregular (hidden slope + rough patch + 50% random mid-roll nudge).
--tracker hsv: the forecaster (tabpfn / cv episodes) is fed ONLY by HSV detections on the rendered (and, with realism,
randomized) top camera image, exactly as in eval_catch.py --tracker hsv, so training inputs match deployment.
Combine with NO_MARKER=1 to record the real-world setting (no blue dot).
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
from sim2real import Realism  # noqa: E402
from launches import easy_launch, irregular_launch, mixed_irregular  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

TASK = "Catch the red ball at the marked point and hold it."
NAMES = [f"{j}.pos" for j in JOINTS]
EXTRA = ["target_y", "time_to_cross", "sigma_y", "y_plus_0.3s", "y_plus_0.6s", "y_plus_0.9s"]
NOISE = 0.004
HOLD_AFTER_LIFT = 0.7
KIND_P = (("oracle", 0.3), ("tabpfn", 0.5), ("cv", 0.2))


def tabpfn_context(env, expert):
    """The forecaster's 1500-launch context rows. Deterministic (fixed seeds), so they are cached in .cache/ and shared
    by all recording shards. Rendering is switched off while building them: env.reset() would otherwise render both
    cameras for every launch (wasted, and very slow with software rendering on CPU nodes); physics and RNG are unchanged."""
    cache = Path(__file__).resolve().parents[2] / ".cache" / f"tabpfn_ctx_v2_1500_0_4_{NOISE}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["T"], z["P"]
    rng0 = np.random.default_rng(1)
    rend, env.renderer = env.renderer, None
    try:
        X, T, P = make_rows_v2(expert.ghost, env, lambda e, s: mixed_irregular(e, s), 1500, 0, 4, rng0, noise=NOISE)
    finally:
        env.renderer = rend
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.npz")
    np.savez(tmp, X=X, T=T, P=P)
    os.replace(tmp, cache)
    return X, T, P


def build_tabpfn(env, expert):
    t0 = time.time()
    X, T, P = tabpfn_context(env, expert)
    print(f"tabpfn context: {len(X)} rows in {time.time() - t0:.0f} s", flush=True)
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
    ap.add_argument("--realism", type=float, default=1.0)
    ap.add_argument("--clean-frac", type=float, default=0.25, help="fraction of episodes recorded with randomization off")
    ap.add_argument("--repo-id", default="local/so101_catch_train")
    ap.add_argument("--tracker", default="truth", choices=["truth", "hsv"],
                    help="hsv: forecaster sees only HSV ball detections on the top camera (deployment setting)")
    ap.add_argument("--delay", default="measured", choices=["measured", "random"],
                    help="forecast compute delay: measured TabPFN latency clipped to 0.15-0.30 s (original), or drawn "
                         "per episode from U(0.15, 0.30) s (independent of the recording hardware; CPU nodes measure >0.3 s)")
    a = ap.parse_args()
    root = Path(a.root)
    if root.exists():
        if not a.overwrite:
            sys.exit(f"{root} exists; pass --overwrite")
        shutil.rmtree(root)
    env = CatchEnv(cameras=("top", "front"), height=a.height, width=a.width, episode_seconds=7.0)
    expert = CatchExpert(env)
    hsv = None
    if a.tracker == "hsv":
        from hsv_tracker import HSVTracker
        hsv = HSVTracker(env)
        print("HSV tracker calibrated with", hsv.n_calib, "points", flush=True)
    tp, delay = build_tabpfn(env, expert)
    delay = float(np.clip(delay, 0.15, 0.30))      # deployment-like compute delay, independent of recording load
    print(f"tabpfn ready: forecast delay {1000 * delay:.0f} ms", flush=True)
    rl = Realism(env, a.realism)

    img = {"dtype": "video", "shape": (a.height, a.width, 3), "names": ["height", "width", "channels"]}
    spec = {
        "observation.images.top": img, "observation.images.front": img,
        "observation.state": {"dtype": "float32", "shape": (6 + len(EXTRA),), "names": NAMES + EXTRA},
        "action": {"dtype": "float32", "shape": (6,), "names": NAMES},
    }
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=env.fps, features=spec, root=root,
                               robot_type="so101_follower", use_videos=True)
    kinds, probs = zip(*KIND_P)
    sidecar, kept, tried = [], 0, 0
    t_start = time.time()
    while kept < a.episodes:
        seed = a.seed0 + tried
        tried += 1
        rng = np.random.default_rng(30_000 + seed)
        rl.strength = 0.0 if rng.random() < a.clean_frac else a.realism
        rl.start_episode(np.random.default_rng(50_000 + seed))
        hard = seed % 2 == 0
        (irregular_launch if hard else easy_launch)(env, seed, joint_noise=2.0)
        c = expert.plan(x_c=X_INT, start_delay=float(rng.uniform(0.0, 0.25)), min_move=0.5)
        if c is None or not expert.ok:
            continue
        kind = str(rng.choice(kinds, p=probs))
        ep_delay = float(rng.uniform(0.15, 0.30)) if a.delay == "random" else delay
        pipe = ForecastPipeline(kind, tp=tp, delay=ep_delay) if kind in ("tabpfn", "cv") else None
        tt, xt, yt, cc = expert.ghost.track()                      # arm-independent ball track (kick-aware)
        if cc is None:
            continue
        tr, ys = np.append(tt, cc[1]), np.append(yt, cc[0])
        d = rl.p["delay"]
        t_end = expert.t_lift + expert.lift_dur + HOLD_AFTER_LIFT
        frames, errs = [], []
        info = {}
        while env.t / env.fps < t_end and env.t < env.max_steps:
            i, t = env.t, env.t / env.fps
            if pipe is not None and hsv is not None:     # deployment setting: forecaster fed ONLY by HSV detections
                img = rl.obs(dict(state=env.joint_state(), top=hsv._render_top()))["top"]
                pt = hsv(img)
                if pt is not None:
                    pipe.update(i, t, pt[0], pt[1])
                tv, my = pipe.vector(t)
            elif pipe is not None:
                if i < len(tt):
                    rd = rl.track(xt[i], yt[i])
                    if rd is not None:
                        pipe.update(i, t, rd[0], rd[1])
                tv, my = pipe.vector(t)
            else:
                tv, my = oracle_vector(t, tr, ys, cc[0], cc[1])
            env.set_marker(*((X_INT, my) if (my is not None and not os.environ.get("NO_MARKER")) else (0.8, 0.0)))   # NO_MARKER=1: record without the blue dot
            if my is not None:
                errs.append(abs(my - cc[0]))
            obs = rl.obs(env.observe())
            action = expert.act(t + d / env.fps).astype(np.float32)   # privileged label leads the control latency
            frames.append(dict(top=obs["top"], front=obs["front"],
                               state=np.concatenate([obs["state"], tv]).astype(np.float32), action=action))
            _, done, info = env.step(rl.act(action, env.joint_state()), render=False)
            if done:
                break
        if not info.get("held", False):
            continue  # expert failed to hold; do not keep failures
        for f in frames:
            ds.add_frame({"observation.images.top": f["top"], "observation.images.front": f["front"],
                          "observation.state": f["state"], "action": f["action"], "task": TASK})
        ds.save_episode()
        sidecar.append(dict(episode=kept, seed=seed, irregular=hard, kind=kind, delay_steps=d, realism=rl.strength, launch=env.launch,
                            crossing_y=cc[0], crossing_t=cc[1], frames=len(frames), tracker=a.tracker, forecast_delay_s=ep_delay,
                            no_marker=bool(os.environ.get("NO_MARKER")),
                            mean_target_y_err_m=float(np.mean(errs)) if errs else None))
        kept += 1
        print(f"{kept}/{a.episodes} episodes ({tried} launches tried, {ds.meta.total_frames} frames, "
              f"{time.time() - t_start:.0f} s)", flush=True)
    ds.finalize()
    (root / "sidecar.json").write_text(json.dumps(sidecar))
    print(f"done: {kept} episodes, {ds.meta.total_frames} frames -> {root}")


if __name__ == "__main__":
    main()
