import sys; sys.path.insert(0,'challenge/sim')
import numpy as np
import catch_ik, catch_expert
from catch_env import CatchEnv
from launches import easy_launch, irregular_launch
env = CatchEnv(render=False)
for margin in (0, 6, 10, 14):
    for opn in (100, 75, 60):
        catch_ik.MARGIN_DEG = margin; catch_expert.OPEN = float(opn)
        ex = catch_expert.CatchExpert(env)
        out=[]; lim=[]
        for name, fn in (("easy", easy_launch), ("hard", irregular_launch)):
            held=n=0
            for s in range(80):
                fn(env, 90000+s, joint_noise=2.0)
                if ex.plan(x_c=0.24, start_delay=0.1) is None: continue
                n+=1; done=False
                while not done: _,done,info=env.step(ex.act(env.t/env.fps),render=False)
                held+=info["held"]; lim.append(np.rad2deg(ex.q_catch[3]) if False else ex.q_catch[3])
            out.append(f"{name} {held}/80 (planned {n})")
        print(f"margin {margin:2d} deg, open {opn:3d}: "+" | ".join(out)+f" | min wrist_flex {min(lim):.0f}", flush=True)
