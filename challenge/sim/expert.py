"""Scripted SO-101 blocking expert: oracle/forecast crossing point -> IK -> smooth joint trajectory."""
from __future__ import annotations

import copy

import mujoco
import numpy as np
from scipy.optimize import least_squares

from ball_env import BALL_R, HOME, JOINTS, LINE_X, BallBlockEnv

X_INT = 0.24          # forecast / catch plane (world x) where the arm meets the ball (blocking line: LINE_X=0.15)
PADDLE_Z = 0.03       # paddle centre height so the plate spans the ball


class Ghost:
    """Ball-only physics copy (arm geoms disabled) used for the oracle crossing and for tests."""

    def __init__(self, env: BallBlockEnv):
        self.env = env
        self.model = copy.deepcopy(env.model)
        keep = {env.ball_geom, env.table_geom, env.patch_geom, *env.rail_geoms}
        for g in range(self.model.ngeom):
            if g not in keep:
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0
        self.data = mujoco.MjData(self.model)

    def crossing(self, x_plane: float, horizon: float = 6.0):
        """Exact (y, t) where the ball centre reaches x_plane, or None if it never does."""
        e = self.env
        m, d = self.model, self.data
        m.geom_friction[:] = e.model.geom_friction
        for g in e.rail_geoms:
            m.geom_contype[g] = e.model.geom_contype[g]
            m.geom_conaffinity[g] = e.model.geom_conaffinity[g]
        d.xfrc_applied[:] = e.data.xfrc_applied
        d.qpos[:] = e.data.qpos
        d.qvel[:] = e.data.qvel
        mujoco.mj_forward(m, d)
        t = 0.0
        a = e.ball_qadr
        kick, kicked = e.kick, False
        prev = d.qpos[a:a + 3].copy()
        while t < horizon:
            if kick is not None and not kicked and t + 1e-9 >= kick[0] / e.fps:
                d.qvel[e.ball_vadr] += kick[1]
                d.qvel[e.ball_vadr + 1] += kick[2]
                kicked = True
            mujoco.mj_step(m, d)
            t += m.opt.timestep
            cur = d.qpos[a:a + 3]
            if cur[0] <= x_plane:
                f = (prev[0] - x_plane) / max(prev[0] - cur[0], 1e-9)
                return float(prev[1] + f * (cur[1] - prev[1])), t
            prev = cur.copy()
        return None


    def track(self, fps: int = 30, until_x: float = X_INT, horizon: float = 5.0):
        """Exact ball trajectory sampled at `fps`: arrays t, x, y until the ball centre reaches until_x.
        Returns (t, x, y, crossing) where crossing=(y, t) at until_x, or None if it never gets there."""
        e = self.env
        m, d = self.model, self.data
        m.geom_friction[:] = e.model.geom_friction
        for g in e.rail_geoms:
            m.geom_contype[g] = e.model.geom_contype[g]
            m.geom_conaffinity[g] = e.model.geom_conaffinity[g]
        d.xfrc_applied[:] = e.data.xfrc_applied
        d.qpos[:] = e.data.qpos
        d.qvel[:] = e.data.qvel
        mujoco.mj_forward(m, d)
        a = e.ball_qadr
        sub = max(1, round(1 / (fps * m.opt.timestep)))
        ts, xs, ys = [0.0], [d.qpos[a]], [d.qpos[a + 1]]
        for k in range(int(horizon * fps)):
            prev = np.array([d.qpos[a], d.qpos[a + 1]])
            if e.kick is not None and k == int(e.kick[0]):
                d.qvel[e.ball_vadr] += e.kick[1]
                d.qvel[e.ball_vadr + 1] += e.kick[2]
            mujoco.mj_step(m, d, nstep=sub)
            cur = np.array([d.qpos[a], d.qpos[a + 1]])
            t = (k + 1) / fps
            if cur[0] <= until_x:
                f = (prev[0] - until_x) / max(prev[0] - cur[0], 1e-9)
                return np.array(ts), np.array(xs), np.array(ys), (float(prev[1] + f * (cur[1] - prev[1])), t - (1 - f) / fps)
            ts.append(t)
            xs.append(cur[0])
            ys.append(cur[1])
        return np.array(ts), np.array(xs), np.array(ys), None


class IK:
    def __init__(self, env: BallBlockEnv):
        self.env = env
        self.m = copy.deepcopy(env.model)
        self.d = mujoco.MjData(self.m)
        self.pid = self.m.geom("paddle").id
        self.qa = env.qadr
        jid = [self.m.joint(j).id for j in JOINTS[:5]]
        self.lo, self.hi = self.m.jnt_range[jid].T
        self.q_home = np.deg2rad(np.array(HOME[:5]))

    def paddle(self, q):
        self.d.qpos[self.qa[:5]] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.geom_xpos[self.pid].copy(), self.d.geom_xmat[self.pid].reshape(3, 3).copy()

    def solve(self, x, y, z=PADDLE_Z, q0=None):
        q0 = self.q_home if q0 is None else q0
        target = np.array([x, y, z])

        def res(q):
            p, R = self.paddle(q)
            return np.concatenate([(p - target) * 20, R[:, 0] - [1, 0, 0], 0.02 * (q - self.q_home)])

        best = None
        for s in range(3):
            start = q0 if s == 0 else np.clip(q0 + np.random.default_rng(s).uniform(-0.5, 0.5, 5), self.lo, self.hi)
            r = least_squares(res, start, bounds=(self.lo, self.hi))
            p, _ = self.paddle(r.x)
            err = float(np.linalg.norm(p - target))
            if best is None or err < best[1]:
                best = (r.x, err)
            if err < 1e-3:
                break
        return np.rad2deg(best[0]), best[1]


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class Expert:
    """Plans once at the start of an episode from a crossing prediction (oracle or forecast)."""

    def __init__(self, env: BallBlockEnv):
        self.env = env
        self.ik = IK(env)
        self.ghost = Ghost(env)

    def plan(self, crossing=None, x_int: float = X_INT, start_delay: float = 0.1, min_move: float = 0.35):
        """crossing: (y, t) at x_int; default oracle. Returns None if the ball will not reach the plane."""
        c = self.ghost.crossing(x_int) if crossing is None else crossing
        if c is None:
            self.q_goal, self.ok = np.array(HOME, dtype=float), False
            self.t0, self.dur, self.c = 0.0, 1.0, None
            return None
        y, t_arr = c
        q_goal, err = self.ik.solve(x_int, float(np.clip(y, -0.22, 0.22)))
        q_goal = np.append(q_goal, HOME[5])
        self.q_goal, self.ok, self.c = q_goal, err < 5e-3, c
        self.q_start = self.env.joint_state().astype(float)
        # arrive ~0.15 s before the ball reaches the plane (ball surface arrives earlier than centre)
        lead = (BALL_R + 0.004) / max(self.env.launch["speed"], 0.05)
        self.dur = max(min_move, t_arr - lead - start_delay - 0.15)
        self.t0 = start_delay
        return c

    def act(self, t: float) -> np.ndarray:
        s = min_jerk((t - self.t0) / self.dur)
        return self.q_start + s * (self.q_goal - self.q_start)
