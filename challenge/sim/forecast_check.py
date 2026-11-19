"""Is TabPFN forecasting the crossing correctly? Error of (y_cross, t_cross) vs the ghost truth, by lead time, vs cv baseline."""
import json, os, sys
from pathlib import Path
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from catch_env import CatchEnv
from catch_expert import CatchExpert
from eval_catch import make_launches
from eval_closed_loop import reset_launch
from expert import X_INT
from forecast import ForecastPipeline
from record_catch_dataset import build_tabpfn
from sim2real import Realism
from ball_env import HOME

env = CatchEnv(cameras=("top", "front"), height=120, width=160)
expert = CatchExpert(env)
tp, lat = build_tabpfn(env, expert)
LEADS = (1.0, 0.7, 0.5, 0.35, 0.2)
out = {}
for cond, rl in (("clean", None), ("real1", Realism(env, 1.0))):
    for split in ("indist", "hard"):
        ls, _ = make_launches(split, 24, env, expert)
        err = {k: {L: [] for L in LEADS} for k in ("tabpfn", "cv")}
        terr = {k: {L: [] for L in LEADS} for k in ("tabpfn", "cv")}
        for l in ls:
            if rl is not None:
                rl.start_episode(np.random.default_rng(777 + l["seed"]))
            reset_launch(env, l)
            yc, tc = expert.ghost.crossing(X_INT)
            pipes = {k: ForecastPipeline(k, tp=tp if k == "tabpfn" else None, delay=0.0) for k in ("tabpfn", "cv")}
            rng = np.random.default_rng(l["seed"])
            done = False
            while not done and env.t / env.fps < tc + 0.05:
                t = env.t / env.fps
                p = env.ball_pos()
                rd = (p[0] + rng.normal(0, .004), p[1] + rng.normal(0, .004)) if rl is None else rl.track(p[0], p[1])
                if rd is not None:
                    for k, pp in pipes.items():
                        pp.update(env.t, t, rd[0], rd[1])
                for L in LEADS:
                    if abs((tc - t) - L) < 0.5 / env.fps:
                        for k, pp in pipes.items():
                            if pp.current is not None:
                                err[k][L].append(abs(pp.current["y"] - yc))
                                terr[k][L].append(abs(pp.current["t_abs"] - tc))
                _, done, _ = env.step(np.array(HOME, dtype=np.float32), render=False)
        res = {}
        for k in err:
            res[k] = {str(L): dict(n=len(err[k][L]), y_cm_med=100 * float(np.median(err[k][L])) if err[k][L] else None,
                                   y_cm_p90=100 * float(np.percentile(err[k][L], 90)) if err[k][L] else None,
                                   t_ms_med=1000 * float(np.median(terr[k][L])) if terr[k][L] else None) for L in LEADS}
        out[f"{cond}|{split}"] = res
        print(cond, split, flush=True)
        for k in res:
            print("  ", k, {L: (round(v["y_cm_med"], 1), round(v["y_cm_p90"], 1), round(v["t_ms_med"])) for L, v in res[k].items() if v["n"]}, flush=True)
Path(__file__).resolve().parents[2] / "outputs/catch_big/forecast_check.json".write_text(json.dumps(out, indent=2))
