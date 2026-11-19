"""Closed-loop evaluation of SmolVLA checkpoints in the SO-101 ball-blocking simulator (v2).

Target conditions (what the policy is told about the ball's future):
  tabpfn   TabPFN forecast from a noisy 4 mm tracker, forecast every 3 steps, compute delay applied
           (crossing y, time-to-cross, confidence, ball y at +0.3/+0.6/+0.9 s; marker at the crossing point)
  cv/quad  constant-velocity / quadratic-fit physics forecasts through the same pipeline (no delay)
  oracle   exact simulator values (an upper bound, not available in deployment)
  wrong    oracle shifted by +/-WRONG_SHIFT m in y (checks that the policy follows the target)
  none     no target: extras zero, marker hidden
Launch splits: indist, heldout (friction/speed shift), hard (= irregular: hidden slope + rough patch + 50% nudges).
Policies trained with 8 state dims (stage 2) receive only the first two extras; 12-dim policies receive all six.
Also reports non-policy references: noop (arm stays home) and the scripted expert with the oracle.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402

from ball_env import HOME, BallBlockEnv  # noqa: E402
from expert import X_INT, Expert  # noqa: E402
from forecast import PATH_TAUS, STATE_EXTRA, ForecastPipeline  # noqa: E402
from launches import easy_launch, irregular_launch  # noqa: E402
from smoothness import jerk_rms, step_rms  # noqa: E402

TASK = "Block the red ball at the marked point."
WRONG_SHIFT = 0.10
TRACK_NOISE = 0.004
OFFSCREEN = (0.8, 0.0)
SEEDS = {"indist": 500_000, "heldout": 600_000, "hard": 700_000}
SPLITS = {"indist": dict(rf=(0.0, 1.0e-4), speed=(0.10, 0.22)),
          "heldout": dict(rf=(1.2e-4, 2.0e-4), speed=(0.18, 0.26))}
FORECAST_KINDS = ("tabpfn", "cv", "quad")


def sample_launch(env, split, seed):
    if split == "hard":
        irregular_launch(env, seed, joint_noise=2.0)
    elif split == "indist":
        easy_launch(env, seed, joint_noise=2.0)
    else:  # heldout: shifted friction and speed
        rng = np.random.default_rng(10_000 + seed)
        env.reset(seed=seed, speed=rng.uniform(*SPLITS[split]["speed"]),
                  rolling_friction=rng.uniform(*SPLITS[split]["rf"]))


def make_launches(split: str, n: int, env: BallBlockEnv, expert: Expert):
    """Fresh seeds; keep launches the scripted expert can solve. Stored so every run replays the same launch."""
    out, seed = [], SEEDS[split]
    while len(out) < n:
        sample_launch(env, split, seed)
        c = expert.plan()
        if c is not None and expert.ok:
            L = env.launch
            out.append(dict(seed=seed, start=L["start"], speed=L["speed"], angle=L["angle"], rf=L["rolling_friction"],
                            hard=L["hard"], slope=L["slope"], patch_rf=L["patch_rf"], kick=L["kick"]))
        seed += 1
    return out


def reset_launch(env: BallBlockEnv, l: dict):
    return env.reset(seed=l["seed"], start=l["start"], speed=l["speed"], angle=l["angle"], rolling_friction=l["rf"],
                     hard=l.get("hard", False), slope=l.get("slope", 0.0), patch_rf=l.get("patch_rf", 1.5e-4),
                     kick=l.get("kick"))


class Target:
    """Per-episode source of the policy's target extras [6] and the marker position."""

    def __init__(self, mode: str, env: BallBlockEnv, expert: Expert, sign: float, pipes: dict, seed: int):
        self.mode, self.env, self.sign = mode, env, sign
        self.pipe = pipes.get(mode)
        if self.pipe is not None:
            self.pipe.reset()
        self.rng = np.random.default_rng(seed)
        if mode in ("oracle", "wrong"):
            t, x, y, c = expert.ghost.track()
            self.tr, self.ys = np.append(t, c[1]), np.append(y, c[0])
            self.y_cross, self.t_cross = c

    def observe(self):
        if self.pipe is not None:
            p = self.env.ball_pos()
            self.pipe.update(self.env.t, self.env.t / self.env.fps, p[0] + self.rng.normal(0, TRACK_NOISE),
                             p[1] + self.rng.normal(0, TRACK_NOISE))

    def vector(self):
        t = self.env.t / self.env.fps
        if self.mode == "none":
            return np.zeros(STATE_EXTRA, np.float32), None
        if self.pipe is not None:
            return self.pipe.vector(t)
        shift = self.sign * WRONG_SHIFT if self.mode == "wrong" else 0.0
        path = [np.interp(min(t + tau, self.t_cross), self.tr, self.ys) for tau in PATH_TAUS]
        y = float(np.clip(self.y_cross + shift, -0.22, 0.22))
        v = np.array([y, max(self.t_cross - t, 0.0), 0.0, *np.clip(np.array(path) + shift, -0.22, 0.22)], np.float32)
        return v, y


def run_episode(env, launch, mode, controller, expert, sign, pipes, record_frames=False):
    reset_launch(env, launch)
    target = Target(mode, env, expert, sign, pipes, launch["seed"]) if mode else None
    if controller == "expert":
        expert.plan()
    if hasattr(controller, "reset"):
        controller.reset()
    frames, done = [], False
    v0 = launch["speed"]
    min_speed, max_vx = v0, -1.0
    acts = []
    while not done:
        if target is not None:
            target.observe()
            extras, my = target.vector()
        else:
            extras, my = np.zeros(STATE_EXTRA, np.float32), None
        env.set_marker(*((X_INT, my) if my is not None else OFFSCREEN))
        obs = env.observe()
        if controller == "noop":
            action = np.array(HOME, dtype=np.float32)
        elif controller == "expert":
            action = expert.act(env.t / env.fps).astype(np.float32)
        else:
            state = np.concatenate([obs["state"], extras[:controller.n_extra]]).astype(np.float32)
            action = controller.act(obs, state)
        acts.append(np.asarray(action, dtype=np.float64))
        if record_frames:
            frames.append(obs["top"])
        _, done, info = env.step(action, render=False)
        v = env.ball_vel()
        min_speed = min(min_speed, float(np.linalg.norm(v[:2])))
        max_vx = max(max_vx, float(v[0]))
    touched = (min_speed < 0.5 * v0) or max_vx > 0.01
    return dict(blocked=bool(info["blocked"]), crossed=bool(info["crossed"]), touched=bool(touched),
                steps=env.t, final_ball=[float(info["ball"][0]), float(info["ball"][1])],
                jerk=jerk_rms(acts), speed_rms=step_rms(acts)), frames


class PolicyController:
    def __init__(self, ckpt: Path, n_action_steps: int):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        self.torch = torch
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.n_action_steps = n_action_steps
        self.n_extra = int(cfg.input_features["observation.state"].shape[0]) - 6
        self.policy = SmolVLAPolicy.from_pretrained(ckpt, config=cfg, local_files_only=True).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=ckpt, preprocessor_overrides={"device_processor": {"device": cfg.device}})
        self.times = []

    def reset(self):
        self.policy.reset()

    def act(self, obs, state):
        torch = self.torch
        t0 = time.time()
        img = lambda a: torch.from_numpy(a).permute(2, 0, 1).float().div(255.0)  # noqa: E731
        batch = {"observation.images.top": img(obs["top"]), "observation.images.front": img(obs["front"]),
                 "observation.state": torch.from_numpy(state), "task": TASK}
        batch = self.pre(batch)
        with torch.inference_mode():
            action = self.post(self.policy.select_action(batch))
        self.times.append(time.time() - t0)
        return action.detach().cpu().numpy().reshape(-1)[:6].astype(np.float32)


def summarize(rs):
    n = len(rs)
    return dict(n=n, blocked=sum(r["blocked"] for r in rs), block_rate=sum(r["blocked"] for r in rs) / n,
                touched=sum(r["touched"] for r in rs), crossed=sum(r["crossed"] for r in rs))


def save_gif(frames, path):
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames[::2]]
    if ims:
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=66, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[], help="name=path/to/pretrained_model")
    ap.add_argument("--n", type=int, default=30, help="episodes per split")
    ap.add_argument("--n-action-steps", type=int, default=5)
    ap.add_argument("--modes", default="oracle,tabpfn,cv,none")
    ap.add_argument("--splits", default="hard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gifs", type=int, default=2, help="GIFs saved per (checkpoint, mode, split)")
    ap.add_argument("--references", action="store_true", help="also run noop and scripted-expert references")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "gifs").mkdir(parents=True, exist_ok=True)
    env = BallBlockEnv(cameras=("top", "front"), height=240, width=320, episode_seconds=4.0)
    expert = Expert(env)
    splits = a.splits.split(",")
    launches = {s: make_launches(s, a.n, env, expert) for s in splits}
    report = {"n_per_split": a.n, "n_action_steps": a.n_action_steps, "wrong_shift_m": WRONG_SHIFT,
              "tracking_noise_m": TRACK_NOISE, "results": {}}
    modes = a.modes.split(",")
    pipes = {}
    if any(m in FORECAST_KINDS for m in modes):
        from record_forecast_dataset import build_tabpfn
        tp, lat = build_tabpfn(env, expert)
        report["tabpfn_latency_ms"] = 1000 * lat
        print(f"tabpfn ready: forecast latency {1000 * lat:.0f} ms", flush=True)
        pipes = {"tabpfn": ForecastPipeline("tabpfn", tp=tp, delay=lat), "cv": ForecastPipeline("cv"),
                 "quad": ForecastPipeline("quad")}

    def save():
        (out / "closed_loop.json").write_text(json.dumps(report, indent=2))

    if a.references:
        for split, ls in launches.items():
            for name, ctrl, mode in (("noop", "noop", None), ("expert_oracle", "expert", "oracle")):
                rs = [run_episode(env, l, mode, ctrl, expert, 1.0, pipes)[0] for l in ls]
                report["results"][f"{name}|{split}"] = summarize(rs)
                print(name, split, report["results"][f"{name}|{split}"], flush=True)
        save()

    for spec in a.checkpoints:
        name, path = spec.split("=", 1)
        ctrl = PolicyController(Path(path), a.n_action_steps)
        for split, ls in launches.items():
            for mode in modes:
                rs = []
                for i, l in enumerate(ls):
                    sign = 1.0 if (l["seed"] % 2 == 0) else -1.0
                    r, fr = run_episode(env, l, mode, ctrl, expert, sign, pipes, record_frames=i < a.gifs)
                    r["seed"] = l["seed"]
                    r["kick"] = l.get("kick") is not None
                    rs.append(r)
                    if fr:
                        save_gif(fr, out / "gifs" / f"{name}_{split}_{mode}_{i}.gif")
                s = summarize(rs)
                s["episodes"] = rs
                s["policy_latency_ms"] = 1000 * float(np.mean(ctrl.times)) if ctrl.times else None
                kk = [r for r in rs if r["kick"]]
                s["blocked_kicked"] = f"{sum(r['blocked'] for r in kk)}/{len(kk)}"
                report["results"][f"{name}|{split}|{mode}"] = s
                print(f"{name} {split} {mode}: blocked {s['blocked']}/{s['n']} (nudged {s['blocked_kicked']}) "
                      f"touched {s['touched']}", flush=True)
                save()
        del ctrl
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    save()
    env.close()


if __name__ == "__main__":
    main()
