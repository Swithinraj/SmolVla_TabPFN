"""v2 benchmark: raw TabPFN vs TabPFN + correction filter vs constant velocity, on irregular launches (with nudges).
Also checks the nudge is identical in the real simulator and the oracle ghost, path accuracy and band coverage."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ball_env import HOME, BallBlockEnv  # noqa: E402
from expert import Expert  # noqa: E402
from forecast import (F, MIN_PTS, PATH_TAUS, CrossingFilter, TabPFNForecasterV2, cv_forecast, features,  # noqa: E402
                      make_rows_v2)
from launches import irregular_launch, mixed_irregular  # noqa: E402

LEADS = (1.0, 0.7, 0.5, 0.3)
NOISE = 0.004


def check_ghost(env, ex, n=20):
    """Ghost crossing must match what the real simulator does when the arm stays at home."""
    errs = []
    for s in range(1000, 1000 + n):
        irregular_launch(env, s)
        if env.kick is None:
            continue
        c = ex.ghost.crossing(0.22)
        if c is None:
            continue
        xs = None
        while True:
            _, done, info = env.step(HOME, render=False)
            if env.ball_pos()[0] <= 0.22 or done:
                break
        if env.ball_pos()[0] <= 0.22:
            errs.append((abs(env.ball_pos()[1] - c[0]), abs(env.t / env.fps - c[1])))
    e = np.array(errs)
    print(f"ghost vs real sim (arm at home), {len(e)} kicked launches: y err mean {1000*e[:,0].mean():.1f} mm, "
          f"max {1000*e[:,0].max():.1f} mm; t err max {1000*e[:,1].max():.0f} ms", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-launches", type=int, default=1500)
    ap.add_argument("--test-launches", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = BallBlockEnv(render=False, episode_seconds=6.0)
    ex = Expert(env)
    check_ghost(env, ex)
    rng = np.random.default_rng(1)
    X, T, P = make_rows_v2(ex.ghost, env, lambda e, s: mixed_irregular(e, s), a.train_launches, 0, 4, rng, noise=NOISE)
    tp = TabPFNForecasterV2().fit(X, T, P)
    print(f"fit on {len(X)} rows", flush=True)

    rng_t = np.random.default_rng(2)
    FILTERS = {"f012": 0.012, "f03": 0.03, "f06": 0.06, "f10": 0.10}   # process-noise std per update (m)
    rec = {k: {lead: [] for lead in LEADS} for k in ("raw", "cv", *FILTERS)}
    path_err, cover, widths, lat = [[] for _ in PATH_TAUS], [], [], []
    kick_flags, seed, kept = [], 1_000_000, 0
    while kept < a.test_launches:
        irregular_launch(env, seed)
        seed += 1
        t, x, y, c = ex.ghost.track()
        if c is None:
            continue
        xo, yo = x + rng_t.normal(0, NOISE, len(x)), y + rng_t.normal(0, NOISE, len(y))
        ts, ys = np.append(t, c[1]), np.append(y, c[0])
        filts = {n: CrossingFilter(qy=q ** 2) for n, q in FILTERS.items()}
        last = {}
        for k in range(len(t)):
            if k + 1 >= MIN_PTS and k % 3 == 0:
                fx = features(t[:k + 1], xo[:k + 1], yo[:k + 1])
                t0 = time.time()
                f = tp.predict(fx)
                lat.append(time.time() - t0)
                ycv, tcv = cv_forecast(t[:k + 1], xo[:k + 1], yo[:k + 1])
                last = dict(raw=(f["y_cross"], t[k] + f["t_rem"]), cv=(ycv, t[k] + tcv))
                for n, fl in filts.items():
                    yf, sf, tf = fl.update(float(np.clip(f["y_cross"], -.22, .22)), f["sigma_y"], t[k] + f["t_rem"])
                    last[n] = (yf, tf)
                rem = c[1] - t[k]
                for lead in LEADS:
                    if abs(rem - lead) < 0.05:
                        for name, (yy, tt) in last.items():
                            rec[name][lead].append((abs(yy - c[0]), abs(tt - c[1]), env.kick is not None))
                for j, tau in enumerate(PATH_TAUS):
                    if rem > tau + 0.05:
                        path_err[j].append(abs(f["path"][j] - np.interp(t[k] + tau, ts, ys)))
                if rem > 0.2:
                    cover.append(abs(f["y_cross"] - c[0]) <= 1.2816 * f["sigma_y"])
                    widths.append(f["sigma_y"])
        kick_flags.append(env.kick is not None)
        kept += 1
    res = {"kicked_fraction": float(np.mean(kick_flags)), "latency_ms_per_forecast": 1000 * float(np.mean(lat))}
    for lead in LEADS:
        res[str(lead)] = {}
        for name in ("cv", "raw", *FILTERS):
            arr = np.array(rec[name][lead])
            if not len(arr):
                continue
            ey, et = arr[:, 0], arr[:, 1]
            res[str(lead)][name] = dict(n=len(arr), y_mae_cm=100 * ey.mean(), within_5cm=float((ey < 0.05).mean()),
                                        t_mae_ms=1000 * et.mean(),
                                        y_mae_cm_kicked=100 * ey[arr[:, 2] == 1].mean() if (arr[:, 2] == 1).any() else None)
        print(f"lead {lead}s:", {n: {k: (round(v, 2) if v is not None else None) for k, v in d.items()}
                                 for n, d in res[str(lead)].items()}, flush=True)
    res["path_mae_cm"] = {str(tau): 100 * float(np.mean(e)) for tau, e in zip(PATH_TAUS, path_err)}
    res["sigma_band_coverage_80"] = float(np.mean(cover))
    res["mean_sigma_cm"] = 100 * float(np.mean(widths))
    print("path MAE (cm):", res["path_mae_cm"], f"| band coverage(80pct) {res['sigma_band_coverage_80']:.2f}",
          f"| mean sigma {res['mean_sigma_cm']:.1f} cm", f"| latency {res['latency_ms_per_forecast']:.0f} ms")
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
