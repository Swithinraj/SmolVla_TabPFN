"""Benchmark: constant-velocity vs quadratic fit vs TabPFN crossing forecasts on held-out hard launches."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ball_env import BallBlockEnv  # noqa: E402
from expert import Expert  # noqa: E402
from forecast import (FEATURES, MIN_PTS, OBS_NOISE, TabPFNForecaster, cv_forecast, features,  # noqa: E402
                      make_rows, quad_forecast)

LEADS = (1.0, 0.7, 0.5, 0.3)  # seconds before the ball reaches the paddle plane
TOL = 0.05                     # |y error| under 5 cm ~ the paddle can still block


def hard_launch(env: BallBlockEnv, seed: int):
    rng = np.random.default_rng(20_000 + seed)
    env.reset(seed=seed, start=(rng.uniform(0.38, 0.46), rng.uniform(-0.13, 0.13)), speed=rng.uniform(0.16, 0.28),
              angle=rng.uniform(-0.35, 0.35), rolling_friction=rng.uniform(0, 1e-4), hard=True,
              slope=rng.uniform(-0.3, 0.3), joint_noise=0)


def mixed_launch(env: BallBlockEnv, seed: int):
    """Even seeds: hard dynamics; odd seeds: the easy regime the policy was first trained on."""
    if seed % 2 == 0:
        return hard_launch(env, seed)
    rng = np.random.default_rng(10_000 + seed)
    env.reset(seed=seed, speed=rng.uniform(0.10, 0.22), rolling_friction=rng.uniform(0, 1e-4), joint_noise=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-launches", type=int, default=1500)
    ap.add_argument("--rows-per-launch", type=int, default=4)
    ap.add_argument("--test-launches", type=int, default=300)
    ap.add_argument("--noise", type=float, default=OBS_NOISE)
    ap.add_argument("--max-context", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = BallBlockEnv(render=False)
    ex = Expert(env)
    ghost = ex.ghost
    rng = np.random.default_rng(1)
    t0 = time.time()
    Xtr, Ttr, _ = make_rows(ghost, env, lambda e, s: hard_launch(e, s), a.train_launches, 0, a.rows_per_launch, rng,
                            noise=a.noise)
    print(f"train rows {len(Xtr)} from {a.train_launches} launches ({time.time() - t0:.0f}s)", flush=True)

    # held-out evaluation set: fixed lead times, seeds disjoint from training
    rng_t = np.random.default_rng(2)
    cases = []  # (hist features, truth, lead)
    seed, kept = 1_000_000, 0
    while kept < a.test_launches:
        hard_launch(env, seed)
        seed += 1
        t, x, y, c = ghost.track()
        if c is None:
            continue
        xo, yo = x + rng_t.normal(0, a.noise, len(x)), y + rng_t.normal(0, a.noise, len(y))
        row = []
        for lead in LEADS:
            k = int(np.argmin(np.abs((c[1] - t) - lead)))
            if k + 1 < MIN_PTS or abs((c[1] - t[k]) - lead) > 0.05:
                row.append(None)
                continue
            row.append((t[:k + 1], xo[:k + 1], yo[:k + 1], c[0], c[1] - t[k]))
        cases.append(row)
        kept += 1

    tp = TabPFNForecaster(max_context=a.max_context).fit(Xtr, Ttr)
    res = {}
    for li, lead in enumerate(LEADS):
        sel = [r[li] for r in cases if r[li] is not None]
        Xq = np.array([features(h[0], h[1], h[2]) for h in sel])
        truth_y = np.array([h[3] for h in sel])
        truth_t = np.array([h[4] for h in sel])
        out = {}
        for name, fn in (("cv", cv_forecast), ("quad", quad_forecast)):
            p = np.array([fn(h[0], h[1], h[2]) for h in sel])
            out[name] = dict(y_pred=p[:, 0], t_pred=p[:, 1])
        py, pt, (lo, hi) = tp.predict(Xq)
        out["tabpfn"] = dict(y_pred=py, t_pred=pt, lo=lo, hi=hi)
        res[str(lead)] = {"n": len(sel)}
        for name, o in out.items():
            ey, et = np.abs(o["y_pred"] - truth_y), np.abs(o["t_pred"] - truth_t)
            r = dict(y_mae_cm=100 * ey.mean(), y_p90_cm=100 * np.percentile(ey, 90), within_5cm=float((ey < TOL).mean()),
                     t_mae_ms=1000 * et.mean(), t_p90_ms=1000 * np.percentile(et, 90))
            if name == "tabpfn":
                r["interval_10_90_coverage"] = float(((truth_y >= o["lo"]) & (truth_y <= o["hi"])).mean())
                r["interval_mean_width_cm"] = float(100 * (o["hi"] - o["lo"]).mean())
            res[str(lead)][name] = r
        print(f"lead {lead}s (n={len(sel)})", flush=True)
        for name in ("cv", "quad", "tabpfn"):
            print("  %-7s" % name, {k: round(v, 3) for k, v in res[str(lead)][name].items()}, flush=True)

    # latency: single-row predict (what a control loop would do)
    xs = Xq[:1]
    tp.predict(xs)
    t1 = time.time()
    for _ in range(5):
        tp.predict(xs)
    res["latency_ms_single_row_y_and_t_and_quantiles"] = 1000 * (time.time() - t1) / 5
    t1 = time.time()
    for _ in range(5):
        tp.ry.predict(xs)
        tp.rt.predict(xs)
    res["latency_ms_single_row_y_and_t"] = 1000 * (time.time() - t1) / 5
    res["context_rows"] = len(Xtr) if not a.max_context else min(a.max_context, len(Xtr))
    res["obs_noise_m"] = a.noise
    print("latency (ms):", {k: round(v, 1) for k, v in res.items() if k.startswith("latency")}, "context", res["context_rows"])
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
