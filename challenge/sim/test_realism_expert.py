import sys; sys.path.insert(0,'challenge/sim')
import numpy as np
from catch_env import CatchEnv
from catch_expert import CatchExpert
from launches import easy_launch, irregular_launch
from sim2real import Realism
env = CatchEnv(render=False); ex = CatchExpert(env); rl = Realism(env, 1.0)
for comp in (False, True):
    for name, fn in (("easy", easy_launch), ("hard", irregular_launch)):
        held = n = 0; byd = {0:[0,0],1:[0,0],2:[0,0],3:[0,0]}
        for s in range(120):
            rl.start_episode(np.random.default_rng(1000+s))
            fn(env, 90000+s, joint_noise=2.0)
            if ex.plan(x_c=0.24, start_delay=0.1) is None: continue
            d = rl.p["delay"]; n += 1; done = False
            while not done:
                t = env.t / env.fps + (d / env.fps if comp else 0.0)
                a = rl.act(ex.act(t), env.joint_state())
                _, done, info = env.step(a, render=False)
            held += info["held"]; byd[d][0] += info["held"]; byd[d][1] += 1
        print(f"lookahead-compensation={comp} {name}: held {held}/{n}  by delay {{{', '.join(f'{k}:{v[0]}/{v[1]}' for k,v in byd.items())}}}", flush=True)
