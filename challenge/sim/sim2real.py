"""Sim-to-real randomization for the SO-101 catch task.

Per episode, `Realism.start_episode(rng)` samples one perturbation set; call it BEFORE env.reset().
  visual     camera pose and FOV jitter, light intensity/position, table / floor / wall / ARM colour (the real arm is not
             teal), ball tint (stays red-dominant), image noise and slight blur
  dynamics   arm position-actuator gain (kp) and damping, joint friction (the ball physics is NOT touched: the oracle
             forecast must stay exact)
  sensing    joint-state bias (calibration offset) and noise, tracker noise / dropouts / one-frame lag
  timing     control latency of 0-3 steps (0-100 ms) and a servo speed limit of 7-12 deg/step (210-360 deg/s)
`strength` scales every range (0 = clean sim). Held-out evaluation uses the same ranges with fresh seeds.
"""
from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

ARM_COLOR = (0.30, 0.62, 0.68)


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _small_rot(rng, max_deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-9
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg))
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis, ang)
    return q


def _pick_color(rng, lo, hi, avoid_red=True, avoid_blue=False):
    """Random RGB that is not red-dominant (the ball is red) and optionally not blue-dominant (the marker is blue)."""
    for _ in range(50):
        c = rng.uniform(lo, hi, 3)
        if avoid_red and c[0] > max(c[1], c[2]) + 0.15:
            continue
        if avoid_blue and c[2] > max(c[0], c[1]) + 0.2:
            continue
        return c
    return np.array([0.6, 0.6, 0.6])


class Realism:
    def __init__(self, env, strength: float = 1.0):
        self.env, self.strength = env, strength
        m = env.model
        self.m = m
        self.cams = [m.camera(n).id for n in ("front", "top")]
        self.nom = dict(
            cam_pos=m.cam_pos.copy(), cam_quat=m.cam_quat.copy(), cam_fovy=m.cam_fovy.copy(),
            light_diffuse=m.light_diffuse.copy(), light_pos=m.light_pos.copy(),
            head_diffuse=np.array(m.vis.headlight.diffuse), head_ambient=np.array(m.vis.headlight.ambient),
            mat_rgba=m.mat_rgba.copy(), gain=m.actuator_gainprm.copy(), bias=m.actuator_biasprm.copy(),
            damping=m.dof_damping.copy(), floss=m.dof_frictionloss.copy())
        name = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i)  # noqa: E731
        self.arm_mats = [i for i in range(m.nmat) if np.allclose(m.mat_rgba[i, :3], ARM_COLOR, atol=1e-3)]
        self.mat = {n: m.material(n).id for n in ("table", "ball", "floor", "wall", "table_leg")}
        self.arm_dofs = np.array([m.joint(j).dofadr[0] for j in
                                  ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")])
        self.p = {}
        self.delay_q = deque()
        self.prev_track = None

    # ------------------------------------------------------------------ sampling
    def start_episode(self, rng: np.random.Generator):
        s, m, n = self.strength, self.m, self.nom
        u = lambda lo, hi: rng.uniform(lo, hi) * 1.0  # noqa: E731
        # --- visual
        for ci in self.cams:
            m.cam_pos[ci] = n["cam_pos"][ci] + s * rng.uniform(-0.02, 0.02, 3)
            m.cam_quat[ci] = _quat_mul(n["cam_quat"][ci], _small_rot(rng, 2.5 * s))
            m.cam_fovy[ci] = n["cam_fovy"][ci] + s * u(-4, 4)
        k = 1 + s * u(-0.4, 0.4)
        m.light_diffuse[:] = n["light_diffuse"] * k
        m.light_pos[:] = n["light_pos"] + s * rng.uniform(-0.15, 0.15, n["light_pos"].shape)
        m.vis.headlight.diffuse[:] = n["head_diffuse"] * (1 + s * u(-0.4, 0.4))
        m.vis.headlight.ambient[:] = n["head_ambient"] * (1 + s * u(-0.4, 0.4))
        m.mat_rgba[:] = n["mat_rgba"]
        base = _pick_color(rng, 0.35, 0.95, avoid_red=True)
        m.mat_rgba[self.mat["table"], :3] = (1 - s) * n["mat_rgba"][self.mat["table"], :3] + s * base
        m.mat_rgba[self.mat["floor"], :3] = np.clip(n["mat_rgba"][self.mat["floor"], :3] * (1 + s * u(-0.4, 0.4)), 0, 1)
        m.mat_rgba[self.mat["wall"], :3] = np.clip(n["mat_rgba"][self.mat["wall"], :3] * (1 + s * u(-0.3, 0.1)), 0, 1)
        arm = (1 - s) * np.array(ARM_COLOR) + s * _pick_color(rng, 0.05, 1.0, avoid_red=True, avoid_blue=True)
        for i in self.arm_mats:
            m.mat_rgba[i, :3] = arm
        ball = n["mat_rgba"][self.mat["ball"], :3] + s * np.array([u(-0.1, 0.05), u(0.0, 0.30), u(0.0, 0.15)])
        m.mat_rgba[self.mat["ball"], :3] = np.clip(ball, 0, 1)
        # --- dynamics (arm only)
        kp = 1 + s * u(-0.2, 0.2)
        m.actuator_gainprm[:] = n["gain"]
        m.actuator_biasprm[:] = n["bias"]
        m.actuator_gainprm[:, 0] *= kp
        m.actuator_biasprm[:, 1] *= kp
        m.dof_damping[:] = n["damping"]
        m.dof_frictionloss[:] = n["floss"]
        m.dof_damping[self.arm_dofs] *= 1 + s * u(-0.3, 0.6)
        m.dof_frictionloss[self.arm_dofs] *= 1 + s * u(-0.3, 1.0)
        # --- sensing / timing (per-episode draws, used by obs()/act()/track())
        self.p = dict(
            state_bias=np.append(rng.normal(0, 1.0 * s, 5), rng.normal(0, 2.0 * s)),
            state_noise=0.2 * s, img_noise=s * u(0.0, 4.0), blur=bool(s > 0 and rng.random() < 0.3),
            delay=int(round(s * rng.integers(0, 4))), track_noise=0.004 + s * u(0.0, 0.004),
            drop=s * u(0.0, 0.10), slew=(1e9 if s == 0 else float(rng.uniform(7.0, 12.0))),
            rng=np.random.default_rng(rng.integers(1 << 31)))
        self.delay_q = deque()
        self.prev_track = None
        self.last_exec = None
        return self.p

    # ------------------------------------------------------------------ runtime hooks
    def obs(self, obs: dict) -> dict:
        p, out = self.p, dict(obs)
        r = p["rng"]
        out["state"] = (obs["state"] + p["state_bias"] + r.normal(0, p["state_noise"], 6)).astype(np.float32)
        for cam in ("top", "front"):
            if cam in obs:
                im = obs[cam].astype(np.float32)
                if p["img_noise"] > 0:
                    im = im + r.normal(0, p["img_noise"], im.shape)
                if p["blur"]:
                    im = 0.5 * im + 0.25 * (np.roll(im, 1, 0) + np.roll(im, 1, 1))
                out[cam] = np.clip(im, 0, 255).astype(np.uint8)
        return out

    def act(self, action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Control latency: the command applied now is the one issued `delay` steps ago."""
        d = self.p["delay"]
        self.delay_q.append(np.asarray(action, dtype=np.float32))
        while len(self.delay_q) <= d:
            self.delay_q.appendleft(np.asarray(current_state, dtype=np.float32))
        cmd = self.delay_q.popleft().astype(np.float64)
        # servo speed limit (Feetech STS3215 are slower than an ideal sim actuator): arm joints move <= slew deg/step
        if self.last_exec is None:
            self.last_exec = np.asarray(current_state, dtype=np.float64)
        out = cmd.copy()
        out[:5] = self.last_exec[:5] + np.clip(cmd[:5] - self.last_exec[:5], -self.p["slew"], self.p["slew"])
        self.last_exec = out
        return out.astype(np.float32)

    def track(self, x: float, y: float):
        """Tracker reading: noise, occasional dropout (None), one-frame lag."""
        p, r = self.p, self.p["rng"]
        reading = (x + r.normal(0, p["track_noise"]), y + r.normal(0, p["track_noise"]))
        out = self.prev_track if self.prev_track is not None else reading
        self.prev_track = reading
        return None if r.random() < p["drop"] else out

    def restore(self):
        """Back to the nominal (clean) model."""
        m, n = self.m, self.nom
        m.cam_pos[:], m.cam_quat[:], m.cam_fovy[:] = n["cam_pos"], n["cam_quat"], n["cam_fovy"]
        m.light_diffuse[:], m.light_pos[:] = n["light_diffuse"], n["light_pos"]
        m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = n["head_diffuse"], n["head_ambient"]
        m.mat_rgba[:] = n["mat_rgba"]
        m.actuator_gainprm[:], m.actuator_biasprm[:] = n["gain"], n["bias"]
        m.dof_damping[:], m.dof_frictionloss[:] = n["damping"], n["floss"]
