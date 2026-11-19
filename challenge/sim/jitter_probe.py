"""Compare controller smoothing variants on one checkpoint: jerk of the executed commands vs task success."""
import sys, json, time; sys.path.insert(0, 'challenge/sim')
import numpy as np
from pathlib import Path
from ball_env import BallBlockEnv
from expert import Expert
from eval_closed_loop import make_launches, run_episode
from forecast import ForecastPipeline
from smooth_controller import SmoothPolicyController

ckpt = Path(sys.argv[1]); n = int(sys.argv[2]); out = sys.argv[3]
env = BallBlockEnv(cameras=("top", "front"), height=240, width=320, episode_seconds=4.0)
expert = Expert(env)
launches = make_launches("indist", n, env, expert)
VARIANTS = {
    "base (period 5)":               dict(period=5),
    "fixed noise":                   dict(period=5, fixed_noise=True),
    "ensemble m=0.05":               dict(period=5, ens_m=0.05),
    "fixed noise + ensemble":        dict(period=5, fixed_noise=True, ens_m=0.05),
    "fixed noise + ens + ema 0.5":   dict(period=5, fixed_noise=True, ens_m=0.05, ema=0.5),
    "fixed noise + ens + ema 0.7":   dict(period=5, fixed_noise=True, ens_m=0.05, ema=0.7),
}
res = {}
ctrl = None
for name, kw in VARIANTS.items():
    if ctrl is None: ctrl = SmoothPolicyController(ckpt, **kw)
    ctrl.period, ctrl.fixed_noise, ctrl.ens_m, ctrl.ema = kw.get("period", 5), kw.get("fixed_noise", False), kw.get("ens_m", 0.0), kw.get("ema", 0.0)
    rs = [run_episode(env, l, "oracle", ctrl, expert, 1.0, {})[0] for l in launches]
    res[name] = dict(blocked=sum(r["blocked"] for r in rs), n=len(rs), jerk_median=float(np.median([r["jerk"] for r in rs])),
                     speed_rms=float(np.median([r["speed_rms"] for r in rs])))
    print(name, res[name], flush=True)
Path(out).write_text(json.dumps(res, indent=2))
