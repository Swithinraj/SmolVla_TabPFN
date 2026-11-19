import sys, itertools, time; sys.path.insert(0, 'challenge/sim')
import numpy as np
from catch_env import CatchEnv
from catch_expert import CatchExpert
from launches import irregular_launch
env = CatchEnv(render=False)
t0 = time.time()
for kl, kd in itertools.product((-0.06, -0.03, 0.0, 0.03, 0.06), (0.0, 0.03)):
    ex = CatchExpert(env, k_lat=kl, k_dep=kd)
    held = n = 0
    for s in range(120):
        irregular_launch(env, 6000 + s, joint_noise=2.0)
        if ex.plan() is None: continue
        n += 1; done = False
        while not done:
            _, done, info = env.step(ex.act(env.t / env.fps), render=False)
        held += info["held"]
    print(f"k_lat={kl:+.2f} k_dep={kd:.2f}: held {held}/{n}  ({time.time()-t0:.0f}s)", flush=True)
