"""Scripted catch expert: pre-position the open gripper at the ball's crossing point, close on arrival, lift, hold."""
from __future__ import annotations

import numpy as np

from ball_env import BALL_R, HOME
from catch_ik import C_LOC, CatchIK, table_contacts
from expert import Ghost, min_jerk

XS = (0.24,)   # default catch plane = the forecast plane; one fixed plane keeps the demos unambiguous
PITCH = np.deg2rad(12)
OPEN, CLOSED = 100.0, 0.0   # OPEN is overridden by the sweep below if a smaller opening still catches
LIFT_Z = 0.075


class CatchExpert:
    def __init__(self, env, close_lead: float = 0.25, close_dur: float = 0.18, lift_delay: float = 0.20,
                 lift_dur: float = 0.9, arrive_lead: float = 0.35, z_off: float = 0.0, c_loc=None, min_time: float = 0.0, k_lat: float = -0.06, k_dep: float = 0.0):
        self.env, self.ik, self.ghost = env, CatchIK(env), Ghost(env)
        self.close_lead, self.close_dur, self.lift_delay, self.lift_dur = close_lead, close_dur, lift_delay, lift_dur
        self.arrive_lead, self.z_off, self.min_time = arrive_lead, z_off, min_time
        self.k_lat, self.k_dep = k_lat, k_dep
        self.c_loc = C_LOC if c_loc is None else np.asarray(c_loc)

    def _candidate(self, x_c):
        c = self.ghost.crossing(x_c)
        c2 = self.ghost.crossing(x_c + 0.03)
        if c is None or c2 is None:
            return None
        y, t_c = c
        y = float(np.clip(y, -0.2, 0.2))
        ball_hd = np.arctan2(c2[0] - c[0], 0.03)                         # direction from catch point toward the ball
        c_loc = self.c_loc
        for _ in range(2):                                                 # 2nd pass: correct for the heading mismatch
            q, perr, _ = self.ik.solve(x_c, y, z=BALL_R + self.z_off, pitch=PITCH, c_loc=c_loc)
            hd = self.ik.mouth_heading(q)
            delta = (hd - ball_hd + np.pi) % (2 * np.pi) - np.pi
            c_loc = self.c_loc + np.array([self.k_dep * abs(np.sin(delta)), 0.0, self.k_lat * np.sin(delta)])
        feasible = perr < 5e-3 and not table_contacts(self.ik, np.deg2rad(q))
        mismatch = abs(delta)
        return dict(x_c=x_c, c=c, q=q, feasible=feasible, mismatch=mismatch, perr=perr, c_loc=c_loc)

    def plan(self, crossing=None, x_c=None, start_delay: float = 0.0, min_move: float = 0.5, xs=XS):
        """Pick the reachable catch point along the ball path whose mouth heading best matches the ball direction."""
        self.ok, self.c = False, None
        cands = [self._candidate(x) for x in ([x_c] if x_c is not None else xs)]
        cands = [k for k in cands if k is not None]
        good = [k for k in cands if k["feasible"] and abs(np.rad2deg(self.ik.mouth_heading(k["q"]))) < 70
                and k["c"][1] >= self.min_time]
        if not good:
            return None
        k = min(good, key=lambda k: k["mismatch"])
        self.x_c, self.c, y, t_c = k["x_c"], k["c"], k["c"][0], k["c"][1]
        y = float(np.clip(y, -0.2, 0.2))
        qa, _, _ = self.ik.solve(self.x_c, y, z=LIFT_Z, pitch=PITCH, q0=np.deg2rad(k["q"]), c_loc=k["c_loc"])
        self.q_catch, self.q_lift = np.append(k["q"], CLOSED), np.append(qa, CLOSED)
        self.ok, self.mismatch = True, k["mismatch"]
        self.q_start = self.env.joint_state().astype(float)
        self.t0, self.t_c = start_delay, t_c
        self.dur = max(min_move, t_c - self.arrive_lead - start_delay)
        self.t_close = t_c - self.close_lead
        self.t_lift = t_c + self.lift_delay
        return self.c

    def act(self, t: float) -> np.ndarray:
        s = min_jerk((t - self.t0) / self.dur)
        q = self.q_start[:5] + s * (self.q_catch[:5] - self.q_start[:5])
        # gripper: opens during the approach, closes at t_close
        g_open = self.q_start[5] + min_jerk((t - self.t0) / (0.5 * self.dur)) * (OPEN - self.q_start[5])
        sc = min_jerk((t - self.t_close) / self.close_dur)
        g = g_open + sc * (CLOSED - g_open)
        # lift after the catch
        sl = min_jerk((t - self.t_lift) / self.lift_dur)
        q = q + sl * (self.q_lift[:5] - self.q_catch[:5])
        return np.append(q, g)
