import sys, itertools, time; sys.path.insert(0, 'challenge/sim')
import numpy as np
from catch_env import CatchEnv
from catch_expert import CatchExpert
from launches import irregular_launch
from smoothness import jerk_rms
env = CatchEnv(render=False)
t0 = time.time()
for kl, ld, ml in [(-0.06,0.5,0.5),(-0.09,0.5,0.5),(-0.12,0.5,0.5),(-0.09,0.9,0.5),(-0.09,0.9,0.8)]:
    ex = CatchExpert(env, k_lat=kl, lift_dur=ld)
    held = n = 0; J=[]
    for s in range(120):
        irregular_launch(env, 6000 + s, joint_noise=2.0)
        if ex.plan(min_move=ml) is None: continue
        n += 1; done = False; acts=[]
        while not done:
            a=ex.act(env.t / env.fps); acts.append(a)
            _, done, info = env.step(a, render=False)
        held += info["held"]; J.append(jerk_rms(acts))
    print(f"k_lat={kl:+.2f} lift_dur={ld} min_move={ml}: held {held}/{n} jerk median {np.median(J):.0f} ({time.time()-t0:.0f}s)", flush=True)
