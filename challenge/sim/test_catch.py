import sys, argparse; sys.path.insert(0, 'challenge/sim')
import numpy as np
from catch_env import CatchEnv
from catch_expert import CatchExpert
from launches import easy_launch, irregular_launch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20); ap.add_argument("--hard", action="store_true")
ap.add_argument("--close-lead", type=float, default=0.25); ap.add_argument("--z-off", type=float, default=0.0)
ap.add_argument("--arrive-lead", type=float, default=0.35); ap.add_argument("--x-c", type=float, default=None)
a = ap.parse_args()
env = CatchEnv(render=False)
ex = CatchExpert(env, close_lead=a.close_lead, z_off=a.z_off, arrive_lead=a.arrive_lead, min_time=0.0)
ok = held = planned = 0
rows = []
for s in range(a.n):
    (irregular_launch if a.hard else easy_launch)(env, 5000 + s, joint_noise=2.0)
    c = ex.plan(x_c=a.x_c)
    if c is None or not ex.ok:
        rows.append((s, "noplan")); continue
    planned += 1
    done = False; hz = []
    while not done:
        _, done, info = env.step(ex.act(env.t / env.fps), render=False)
        hz.append(info["held"])
    held += info["held"]
    p = info["ball"]
    rows.append((s, "HELD" if info["held"] else f"miss ball=({p[0]:.2f},{p[1]:.2f},{p[2]:.3f}) speed={env.launch['speed']:.2f} y={c[0]:+.2f}"))
for r in rows: print(r)
print(f"planned {planned}/{a.n}  held {held}/{planned}")
