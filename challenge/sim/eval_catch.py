"""Closed-loop evaluation of SmolVLA checkpoints on the catch-and-hold task.

Success ("held"): at the end of the episode the ball is lifted (centre z > 0.035 m) and within 9 cm of the gripper.
Smoothness: jerk (RMS 3rd difference of the 5 arm-joint commands, deg/s^3) relative to the scripted expert's;
  smooth_score = min(1, expert_ref / jerk)  (1.0 = at least as smooth as the expert)
  points       = held * smooth_score        (a clean catch counts fully; a jittery catch counts less)
Target conditions as in eval_closed_loop (oracle / tabpfn / cv / wrong / none). Launches are fresh seeds the scripted
expert can solve; the share it cannot solve is reported. Controller variants: raw (re-query every 5 steps) vs smooth
(fixed noise + chunk ensembling + EMA 0.5; smoother = EMA 0.7; measured 4x lower jerk on stage 2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402

import smoothness  # noqa: E402
from ball_env import HOME  # noqa: E402
from catch_env import CatchEnv  # noqa: E402
from catch_expert import CatchExpert  # noqa: E402
from eval_closed_loop import (FORECAST_KINDS, OFFSCREEN, SEEDS, Target, reset_launch,  # noqa: E402
                              sample_launch)
from expert import X_INT  # noqa: E402
from forecast import STATE_EXTRA, ForecastPipeline  # noqa: E402
from safety import SafetyFilter  # noqa: E402
from sim2real import Realism  # noqa: E402

TASK = "Catch the red ball at the marked point and hold it."
HOLD_AFTER_LIFT = 0.7
MARGIN = 0.5
SMOOTH = dict(period=5, fixed_noise=True, ens_m=0.05, ema=0.5)
SMOOTHER = dict(period=5, fixed_noise=True, ens_m=0.05, ema=0.7)
SAFE = dict(period=5, fixed_noise=True, ens_m=0.05, ema=0.5, max_step=10.0)   # + per-step joint limit (deployment)
RAW = dict(period=5)


def parse_variant(v: str) -> dict:
    """Named variant (raw/smooth/smoother/safe) or a parameter string for smoothing sweeps:
    p<period>[e<ens_m>][a<ema>][w<ens_window>][m<max_step>], e.g. p3e0.02a0.8 (fixed noise is always on for these)."""
    import re
    named = {"raw": RAW, "smooth": SMOOTH, "smoother": SMOOTHER, "safe": SAFE}
    if v in named:
        return named[v]
    m = re.fullmatch(r"p(\d+)(?:e([\d.]+))?(?:a([\d.]+))?(?:w(\d+))?(?:m([\d.]+))?", v)
    if m is None:
        raise ValueError(f"unknown variant {v!r}")
    kw = dict(period=int(m[1]), fixed_noise=True, ens_m=float(m[2] or 0.0), ema=float(m[3] or 0.0))
    if m[4]:
        kw["ens_window"] = int(m[4])
    if m[5]:
        kw["max_step"] = float(m[5])
    return kw


class GatedPipe:
    """Uncertainty / time gate: the policy only sees the TabPFN target once it is trustworthy (time-to-cross < gate_t
    and/or sigma_y < gate_s); before that it gets the 'no target' input (zeros, no marker)."""

    def __init__(self, pipe, gate_t=None, gate_s=None):
        self.pipe, self.gate_t, self.gate_s = pipe, gate_t, gate_s

    def reset(self):
        self.pipe.reset()

    def update(self, *a):
        self.pipe.update(*a)

    def vector(self, t):
        v, y = self.pipe.vector(t)
        if y is None or (self.gate_t is not None and v[1] > self.gate_t) or (self.gate_s is not None and v[2] > self.gate_s):
            return np.zeros(STATE_EXTRA, np.float32), None
        return v, y


HSV = None   # set in main() when --tracker hsv


def plan_expert(expert, seed_rng_start=0.1):
    return expert.plan(x_c=X_INT, start_delay=seed_rng_start, min_move=0.5)


def make_launches(split, n, env, expert):
    out, seed, tried = [], SEEDS[split], 0
    while len(out) < n:
        sample_launch(env, split, seed)
        seed += 1
        tried += 1
        c = plan_expert(expert)
        if c is None or not expert.ok:
            continue
        horizon = expert.t_lift + expert.lift_dur + HOLD_AFTER_LIFT + MARGIN
        env.max_steps = int(horizon * env.fps)
        done = False
        while not done:
            _, done, info = env.step(expert.act(env.t / env.fps), render=False)
        if not info["held"]:
            continue
        L = env.launch
        out.append(dict(seed=seed - 1, start=L["start"], speed=L["speed"], angle=L["angle"], rf=L["rolling_friction"],
                        hard=L["hard"], slope=L["slope"], patch_rf=L["patch_rf"], kick=L["kick"], horizon=horizon))
    return out, n / tried


def run_episode(env, launch, mode, controller, expert, sign, pipes, record_frames=False, rl=None, sf=None):
    if rl is not None:
        rl.start_episode(np.random.default_rng(777 + launch["seed"]))
    else:
        rl0 = None
    reset_launch(env, launch)
    env.max_steps = int(launch["horizon"] * env.fps)
    target = Target(mode, env, expert, sign, pipes, launch["seed"]) if mode else None
    if controller == "expert":
        plan_expert(expert)
    if hasattr(controller, "reset"):
        controller.reset()
    if sf is not None:
        sf.reset(env.joint_state())
    frames, acts, done = [], [], False
    v0 = launch["speed"]
    min_speed, max_vx, ever_held = v0, -1.0, False
    while not done:
        if target is not None:
            if HSV is not None and target.pipe is not None:      # real-world setting: forecaster fed ONLY by HSV detections
                img = HSV._render_top()
                if rl is not None:
                    img = rl.obs(dict(state=env.joint_state(), top=img))["top"]
                pt = HSV(img)
                if pt is not None:
                    target.pipe.update(env.t, env.t / env.fps, pt[0], pt[1])
            elif rl is None:
                target.observe()
            elif target.pipe is not None:
                p = env.ball_pos()
                rd = rl.track(p[0], p[1])
                if rd is not None:
                    target.pipe.update(env.t, env.t / env.fps, rd[0], rd[1])
            extras, my = target.vector()
        else:
            extras, my = np.zeros(STATE_EXTRA, np.float32), None
        env.set_marker(*((X_INT, my) if (my is not None and not os.environ.get("NO_MARKER")) else OFFSCREEN))   # NO_MARKER=1: hide the blue dot (real-world setting)
        obs = env.observe()
        if rl is not None:
            obs = rl.obs(obs)
        if controller == "noop":
            action = np.array(HOME, dtype=np.float32)
        elif controller == "expert":
            lead = 0.0 if rl is None else rl.p["delay"] / env.fps
            action = expert.act(env.t / env.fps + lead).astype(np.float32)
        else:
            state = np.concatenate([obs["state"], extras[:controller.n_extra]]).astype(np.float32)
            action = controller.act(obs, state)
        if sf is not None:
            action = sf(action, env.joint_state())
        acts.append(np.asarray(action, dtype=np.float64))
        if record_frames:
            frames.append(np.concatenate([obs["top"], obs["front"]], axis=1))
        _, done, info = env.step(action if rl is None else rl.act(action, env.joint_state()), render=False)
        v = env.ball_vel()
        min_speed = min(min_speed, float(np.linalg.norm(v[:2])))
        max_vx = max(max_vx, float(v[0]))
        ever_held |= info["held"]
    jerk = smoothness.jerk_rms(acts)
    held = bool(info["held"])
    return dict(held=held, ever_held=bool(ever_held), touched=bool((min_speed < 0.5 * v0) or max_vx > 0.01),
                jerk=jerk, speed_rms=smoothness.step_rms(acts), smooth=smoothness.smooth_score(jerk),
                points=float(held) * smoothness.smooth_score(jerk), steps=env.t,
                safety=(dict(sf.stats) if sf is not None else None),
                final_ball=[float(x) for x in info["ball"]]), frames


def summarize(rs):
    n = len(rs)
    return dict(n=n, held=sum(r["held"] for r in rs), ever_held=sum(r["ever_held"] for r in rs),
                touched=sum(r["touched"] for r in rs), jerk_median=float(np.median([r["jerk"] for r in rs])),
                smooth_mean=float(np.mean([r["smooth"] for r in rs])), points=float(np.sum([r["points"] for r in rs])))


def save_gif(frames, path):
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames[::2]]
    if ims:
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=66, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[], help="name=path/to/pretrained_model")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--modes", default="oracle,tabpfn,cv,none")
    ap.add_argument("--splits", default="hard,indist")
    ap.add_argument("--variants", default="raw,smooth,smoother")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gifs", type=int, default=2)
    ap.add_argument("--references", action="store_true")
    ap.add_argument("--tracker", default="truth", choices=["truth", "hsv"], help="hsv: TabPFN sees only HSV ball detections")
    ap.add_argument("--safety", action="store_true", help="run every command through safety.SafetyFilter")
    ap.add_argument("--realism", default="0", help="comma list of Realism strengths, e.g. 0,1 (0 = clean sim)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "gifs").mkdir(parents=True, exist_ok=True)
    env = CatchEnv(cameras=("top", "front"), height=240, width=320)
    expert = CatchExpert(env)
    if a.tracker == "hsv":
        global HSV
        from hsv_tracker import HSVTracker
        HSV = HSVTracker(env)
        print("HSV tracker calibrated with", HSV.n_calib, "points", flush=True)
    launches, solvable = {}, {}
    for s in a.splits.split(","):
        launches[s], solvable[s] = make_launches(s, a.n, env, expert)
    report = {"n_per_split": a.n, "expert_solvable_fraction": solvable, "smooth_ref_jerk": smoothness.EXPERT_JERK_REF,
              "results": {}}
    modes = a.modes.split(",")
    pipes = {}
    if any(m.split("_")[0] in FORECAST_KINDS for m in modes):
        from record_catch_dataset import build_tabpfn
        tp, lat = build_tabpfn(env, expert)
        report["tabpfn_latency_ms"] = 1000 * lat
        print(f"tabpfn ready: forecast latency {1000 * lat:.0f} ms", flush=True)
        lat = float(np.clip(lat, 0.15, 0.30))   # deployment-like compute delay
        pipes = {"tabpfn": ForecastPipeline("tabpfn", tp=tp, delay=lat), "cv": ForecastPipeline("cv")}
        for m in modes:                      # gated modes: tabpfn_t0.6 (time gate), tabpfn_s0.02 (sigma gate)
            if m.startswith("tabpfn_"):
                kind, val = m[7], float(m[8:])
                pipes[m] = GatedPipe(ForecastPipeline("tabpfn", tp=tp, delay=lat), gate_t=val if kind == "t" else None,
                                     gate_s=val if kind == "s" else None)

    sf = SafetyFilter(env=env) if a.safety else None
    conds = {("clean" if float(x) == 0 else f"real{x}"): (None if float(x) == 0 else Realism(env, float(x)))
             for x in a.realism.split(",")}

    def save():
        (out / "closed_loop.json").write_text(json.dumps(report, indent=2))

    if a.references:
        for cond, rl in conds.items():
            for split, ls in launches.items():
                for name, ctrl, mode in (("noop", "noop", None), ("expert_oracle", "expert", "oracle")):
                    rs = [run_episode(env, l, mode, ctrl, expert, 1.0, pipes, rl=rl, sf=sf)[0] for l in ls]
                    report["results"][f"{name}|{cond}|{split}"] = summarize(rs)
                    print(name, cond, split, report["results"][f"{name}|{cond}|{split}"], flush=True)
        save()

    from smooth_controller import SmoothPolicyController
    for spec in a.checkpoints:
        name, path = spec.split("=", 1)
        ctrl = SmoothPolicyController(Path(path), **RAW)
        ctrl.task = TASK
        for variant in a.variants.split(","):
            kw = parse_variant(variant)
            ctrl.period, ctrl.fixed_noise, ctrl.ens_m, ctrl.ema = kw["period"], kw.get("fixed_noise", False), kw.get("ens_m", 0.0), kw.get("ema", 0.0)
            ctrl.max_step = kw.get("max_step")
            ctrl.ens_window = kw.get("ens_window", 10)
            for cond, rl in conds.items():
              for split, ls in launches.items():
                for mode in modes:
                    rs = []
                    for i, l in enumerate(ls):
                        sign = 1.0 if (l["seed"] % 2 == 0) else -1.0
                        r, fr = run_episode(env, l, mode, ctrl, expert, sign, pipes, record_frames=i < a.gifs, rl=rl, sf=sf)
                        r["seed"], r["kick"] = l["seed"], l.get("kick") is not None
                        rs.append(r)
                        if fr:
                            save_gif(fr, out / "gifs" / f"{name}_{variant}_{cond}_{split}_{mode}_{i}.gif")
                    s = summarize(rs)
                    s["episodes"] = rs
                    kk = [r for r in rs if r["kick"]]
                    s["held_kicked"] = f"{sum(r['held'] for r in kk)}/{len(kk)}"
                    report["results"][f"{name}|{variant}|{cond}|{split}|{mode}"] = s
                    print(f"{name} {variant} {cond} {split} {mode}: held {s['held']}/{s['n']} (nudged {s['held_kicked']}) "
                          f"touched {s['touched']} jerk {s['jerk_median']:.0f} points {s['points']:.1f}", flush=True)
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
