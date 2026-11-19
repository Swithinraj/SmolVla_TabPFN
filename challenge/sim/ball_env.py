"""SO-101 red-ball interception environment (MuJoCo).

Reuses the official lerobot/so101-tabletop-mujoco arm model. Joint state/action units match LeRobot's
so101_follower with degrees: degrees for arm joints, 0-100 for the gripper; absolute position targets.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).absolute().parent
ARM_ROOT = HERE / "so101_tabletop"
SCENE = HERE / "so101_ball.xml"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
HOME = (0.0, -60.0, 60.0, 35.0, 0.0, 30.0)
LINE_X = 0.15
BALL_R = 0.02
ARM_COLOR = (0.30, 0.62, 0.68)
# Paddle: a flat plate on the gripper body, in the gripper frame (tip points along -z, see gripperframe).
PADDLE_POS = (0.012, 0.0, -0.105)
PADDLE_HALF = (0.004, 0.035, 0.03)


def build_model(paddle: bool = True) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENE))
    arm = mujoco.MjSpec.from_file(str(ARM_ROOT / "robotstudio_so101" / "so101.xml"))
    for m in arm.materials:
        if not m.name.startswith("sts3215"):
            m.rgba = [*ARM_COLOR, 1.0]
    if ET.parse(SCENE).getroot().find("option") is None:
        for name in dir(arm.option):
            if not name.startswith("_"):
                setattr(spec.option, name, getattr(arm.option, name))
    spec.attach(arm, site=spec.site("so101_mount"), prefix="")
    if paddle:
        body = spec.body("gripper")
        body.add_geom(
            name="paddle", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(PADDLE_HALF), pos=list(PADDLE_POS),
            material="paddle", friction=[1.0, 0.005, 0.0001], condim=3, mass=0.005,
        )
    return spec.compile()


class BallBlockEnv:
    """Joint-position control; ball rolls toward the arm and must be stopped before LINE_X."""

    def __init__(self, fps: int = 30, height: int = 480, width: int = 640, cameras=("top", "wrist_cam"),
                 episode_seconds: float = 4.0, paddle: bool = True, render: bool = True):
        self.model = build_model(paddle)
        self.fps = fps
        self.substeps = max(1, round(1 / (fps * self.model.opt.timestep)))
        self.model.opt.timestep = 1 / (fps * self.substeps)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width) if render else None
        self.cameras = tuple(cameras)
        self.max_steps = int(episode_seconds * fps)
        m = self.model
        self.qadr = np.array([m.joint(j).qposadr[0] for j in JOINTS])
        self.act = np.array([m.actuator(j).id for j in JOINTS])
        self.lo, self.hi = m.actuator_ctrlrange[self.act].T.copy()
        self.ball_qadr = m.joint("ball_free").qposadr[0]
        self.ball_vadr = m.joint("ball_free").dofadr[0]
        self.ball_geom = m.geom("ball_geom").id
        self.table_geom = m.geom("table").id
        self.ball_body = m.body("ball").id
        self.patch_geom = m.geom("patch").id
        self.rail_geoms = [m.geom("rail_l").id, m.geom("rail_r").id]
        for g in [self.patch_geom, *self.rail_geoms]:  # hidden dynamics are not rendered
            m.geom_rgba[g, 3] = 0.0
        self.marker_mocap = m.body("goal_marker").mocapid[0]
        self.grip_site = m.site("gripperframe").id
        self.t = 0
        self.kick = None
        self.launch = {}

    # --- unit conversion -------------------------------------------------
    def to_ctrl(self, v: np.ndarray) -> np.ndarray:
        c = np.deg2rad(np.asarray(v, dtype=np.float64))
        c[5] = self.lo[5] + v[5] / 100 * (self.hi[5] - self.lo[5])
        return np.clip(c, self.lo, self.hi)

    def joint_state(self) -> np.ndarray:
        q = self.data.qpos[self.qadr]
        s = np.rad2deg(q)
        s[5] = (q[5] - self.lo[5]) / (self.hi[5] - self.lo[5]) * 100
        return s.astype(np.float32)

    def ball_pos(self) -> np.ndarray:
        return self.data.qpos[self.ball_qadr:self.ball_qadr + 3].copy()

    def ball_vel(self) -> np.ndarray:
        return self.data.qvel[self.ball_vadr:self.ball_vadr + 3].copy()

    # --- episode ---------------------------------------------------------
    def reset(self, seed: int | None = None, start=None, speed: float | None = None, angle: float | None = None,
              rolling_friction: float | None = None, sliding_friction: float | None = None,
              joints=HOME, joint_noise: float = 2.0, hard: bool = False, slope: float = 0.0,
              patch_rf: float = 1.5e-4, kick=None):
        """Launch the ball from `start`=(x,y) with `speed` m/s toward -x rotated by `angle` radians.

        hard=True activates the hidden dynamics: a rough patch at x in [0.27, 0.35] (rolling friction
        `patch_rf`) and a lateral slope acceleration `slope` (m/s^2, +y). Neither is visible in the images.
        kick=(step, dvx, dvy): at control step `step` the ball's velocity jumps by (dvx, dvy) m/s, an
        unpredictable mid-roll nudge (hard mode only)."""
        rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        j = np.asarray(joints, dtype=np.float64) + rng.uniform(-joint_noise, joint_noise, 6)
        d.qpos[self.qadr] = self.to_ctrl(j)
        d.ctrl[self.act] = d.qpos[self.qadr]
        if start is None:
            start = (rng.uniform(0.38, 0.46), rng.uniform(-0.12, 0.12))
        if speed is None:
            speed = rng.uniform(0.10, 0.22)
        if angle is None:
            angle = rng.uniform(-0.25, 0.25)
        if rolling_friction is None:
            rolling_friction = 5e-5
        if sliding_friction is None:
            sliding_friction = 1.0
        m.geom_friction[self.ball_geom] = [sliding_friction, 0.005, rolling_friction]
        m.geom_friction[self.table_geom] = [sliding_friction, 0.005, rolling_friction]
        m.geom_friction[self.patch_geom] = [sliding_friction, 0.005, patch_rf if hard else rolling_friction]
        for g in self.rail_geoms:
            m.geom_contype[g] = m.geom_conaffinity[g] = 0  # rails only stick the ball (no useful bounce); disabled
        d.xfrc_applied[self.ball_body, 1] = m.body_mass[self.ball_body] * slope if hard else 0.0
        vx, vy = -speed * np.cos(angle), speed * np.sin(angle)
        a = self.ball_qadr
        d.qpos[a:a + 3] = [start[0], start[1], BALL_R]
        d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        v = self.ball_vadr
        d.qvel[v:v + 3] = [vx, vy, 0.0]
        d.qvel[v + 3:v + 6] = [-vy / BALL_R, vx / BALL_R, 0.0]  # rolling without slipping
        d.mocap_pos[self.marker_mocap] = [LINE_X, 0.0, 0.001]
        mujoco.mj_forward(m, d)
        self.t = 0
        self.crossed = False
        self.launch = dict(start=tuple(map(float, start)), speed=float(speed), angle=float(angle),
                           rolling_friction=float(rolling_friction), sliding_friction=float(sliding_friction),
                           hard=bool(hard), slope=float(slope), patch_rf=float(patch_rf),
                           kick=list(map(float, kick)) if (hard and kick is not None) else None)
        self.kick = tuple(kick) if (hard and kick is not None) else None
        return self.observe()

    def set_marker(self, x: float, y: float):
        self.data.mocap_pos[self.marker_mocap] = [x, y, 0.001]

    def observe(self) -> dict:
        obs = {"state": self.joint_state(), "t": self.t / self.fps}
        if self.renderer is not None:
            for cam in self.cameras:
                self.renderer.update_scene(self.data, camera=cam)
                obs[cam] = self.renderer.render().copy()
        return obs

    def step(self, action, render: bool = True):
        d = self.data
        d.ctrl[self.act] = self.to_ctrl(action)
        if self.kick is not None and self.t == int(self.kick[0]):
            d.qvel[self.ball_vadr] += self.kick[1]
            d.qvel[self.ball_vadr + 1] += self.kick[2]
        mujoco.mj_step(self.model, d, nstep=self.substeps)
        self.t += 1
        p = self.ball_pos()
        if p[0] < LINE_X - BALL_R:  # ball fully past the line
            self.crossed = True
        speed = float(np.linalg.norm(self.ball_vel()[:2]))
        stopped = speed < 0.005 and self.t > 5
        off_table = p[2] < -0.02
        done = self.crossed or stopped or off_table or self.t >= self.max_steps
        info = dict(ball=p, ball_speed=speed, crossed=self.crossed,
                    blocked=bool(done and not self.crossed and not off_table))
        return (self.observe() if render else None), done, info

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
