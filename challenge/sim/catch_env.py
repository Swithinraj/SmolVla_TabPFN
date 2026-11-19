"""Catch task: the gripper must grab the rolling ball and hold it (lifted) until the end of the episode."""
from __future__ import annotations

import mujoco
import numpy as np

from ball_env import BALL_R, BallBlockEnv

HOLD_Z = 0.035          # ball centre must be at least this high (it rests at 0.02) ...
HOLD_DIST = 0.09        # ... and within this distance of the gripper frame, at the end of the episode


class CatchEnv(BallBlockEnv):
    def __init__(self, *a, episode_seconds: float = 4.5, paddle: bool = False, **k):
        super().__init__(*a, episode_seconds=episode_seconds, paddle=paddle, **k)

    def held(self) -> bool:
        p = self.ball_pos()
        g = self.data.site_xpos[self.grip_site]
        return bool(p[2] > HOLD_Z and np.linalg.norm(p - g) < HOLD_DIST)

    def step(self, action, render: bool = True):
        d = self.data
        d.ctrl[self.act] = self.to_ctrl(action)
        if self.kick is not None and self.t == int(self.kick[0]):
            d.qvel[self.ball_vadr] += self.kick[1]
            d.qvel[self.ball_vadr + 1] += self.kick[2]
        mujoco.mj_step(self.model, d, nstep=self.substeps)
        self.t += 1
        p = self.ball_pos()
        done = self.t >= self.max_steps or p[2] < -0.02
        info = dict(ball=p, held=self.held(), ball_speed=float(np.linalg.norm(self.ball_vel()[:2])))
        return (self.observe() if render else None), done, info
