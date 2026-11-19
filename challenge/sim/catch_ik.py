"""IK for the catch pose: gripper mouth faces the oncoming ball, jaws open sideways (horizontal)."""
from __future__ import annotations

import copy

import mujoco
import numpy as np
from scipy.optimize import least_squares

from ball_env import HOME, JOINTS, BallBlockEnv

MARGIN_DEG = 10.0
C_LOC = np.array([-0.05, 0.0, 0.035])   # where the ball centre should sit in the site frame (mid-mouth, depth)


class CatchIK:
    def __init__(self, env: BallBlockEnv):
        self.env = env
        self.m = copy.deepcopy(env.model)
        self.d = mujoco.MjData(self.m)
        self.site = self.m.site("gripperframe").id
        self.qa = env.qadr
        jid = [self.m.joint(j).id for j in JOINTS[:5]]
        self.lo, self.hi = self.m.jnt_range[jid].T
        self.q_home = np.deg2rad(np.array(HOME[:5]))
        self.lo = self.lo.copy(); self.hi = self.hi.copy()
        margin = np.deg2rad(MARGIN_DEG)                     # stay away from the hardware joint limits (calibration slack)
        self.lo[:4] += margin; self.hi[:4] -= margin
        self.lo[4], self.hi[4] = np.deg2rad(-115), np.deg2rad(-65)   # camera on top, jaws open sideways

    def frame(self, q):
        self.d.qpos[self.qa[:5]] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, x, y, z=0.02, c_loc=C_LOC, q0=None, pitch=0.7, w_ori=1.0):
        """Place the ball-centre point (site frame c_loc) at world (x, y, z). Mouth pitch is `pitch` rad nose-down,
        jaws open horizontally; yaw follows the radial direction (set by the shoulder pan)."""
        q0 = self.q_home if q0 is None else q0
        q0 = np.clip(q0, self.lo, self.hi)
        target = np.array([x, y, z])

        def res(q):
            p, R = self.frame(q)
            return np.concatenate([(p + R @ c_loc - target) * 20,
                                   [w_ori * (R[2, 0] + np.sin(pitch)), w_ori * R[2, 2]],
                                   0.01 * (q - self.q_home)])

        best = None
        for s in range(5):
            start = q0 if s == 0 else np.clip(q0 + np.random.default_rng(s).uniform(-0.6, 0.6, 5), self.lo, self.hi)
            start[4] = np.clip(start[4], self.lo[4], self.hi[4])
            r = least_squares(res, start, bounds=(self.lo, self.hi))
            p, R = self.frame(r.x)
            perr = float(np.linalg.norm(p + R @ c_loc - target))
            oerr = float(abs(R[2, 0] + np.sin(pitch)) + abs(R[2, 2]))
            if best is None or perr + 0.02 * oerr < best[2]:
                best = (r.x, perr, perr + 0.02 * oerr, oerr)
            if perr < 1e-3 and oerr < 0.03:
                break
        return np.rad2deg(best[0]), best[1], best[3]

    def mouth_heading(self, q):
        _, R = self.frame(np.deg2rad(q) if np.max(np.abs(q)) > 3.2 else q)
        return float(np.arctan2(R[1, 0], R[0, 0]))


def table_contacts(ik: "CatchIK", q, min_pen: float = 0.0) -> list:
    """Names of arm geoms touching the table at joint config q (rad), ball excluded. min_pen: only count penetrations
    deeper than this (m); 0 counts any contact."""
    m, d = ik.m, ik.d
    d.qpos[ik.qa[:5]] = q
    mujoco.mj_forward(m, d)
    tab = m.geom("table").id
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        if tab in (c.geom1, c.geom2):
            o = c.geom2 if c.geom1 == tab else c.geom1
            if o != m.geom("ball_geom").id and c.dist < -min_pen:
                out.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o) or f"geom{o}")
    return out
