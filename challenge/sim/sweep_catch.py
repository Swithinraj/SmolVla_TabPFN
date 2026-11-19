import sys, itertools, time; sys.path.insert(0, 'challenge/sim')
import numpy as np
from catch_env import CatchEnv
from catch_expert import CatchExpert
from launches import easy_launch
env = CatchEnv(render=False)
res = []
t0 = time.time()
for dx, dz, zo, cl in itertools.product((-0.03, -0.05, -0.065), (0.03, 0.045, 0.06), (0.0, 0.012), (0.06, 0.14)):
    ex = CatchExpert(env, close_lead=cl, z_off=zo, c_loc=(dx, 0.0, dz))
    held = n = 0
    for s in range(10):
        easy_launch(env, 5000 + s, joint_noise=2.0)
        if ex.plan() is None or not ex.ok: continue
        n += 1; done = False
        while not done:
            _, done, info = env.step(ex.act(env.t / env.fps), render=False)
        held += info["held"]
    res.append((held, n, dx, dz, zo, cl))
    print(f"dx={dx} dz={dz} z_off={zo} lead={cl}: held {held}/{n}  ({time.time()-t0:.0f}s)", flush=True)
res.sort(reverse=True)
print("BEST", res[:5])
