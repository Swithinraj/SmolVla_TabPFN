"""HSV red-ball tracker on the TOP camera image only -> table (x, y). This is the sole input of the TabPFN forecaster
in the real-world setting (no ground-truth ball position, no marker).
  detect:  RGB -> HSV, red = hue in [0,10] U [170,180], S >= S_MIN, V >= V_MIN; morphological open; largest blob; the
           blob must be roughly round and of plausible area. Centroid (u, v) in pixels.
  map:     homography pixel -> table plane (ball-centre height), fitted once from a grid of known ball positions
           (the real-world equivalent: place the ball at 4+ marked table points and click / detect).
Returns None when the ball is not found (occluded by the arm, glare, out of view)."""
from __future__ import annotations

import cv2
import numpy as np
import mujoco

from ball_env import BALL_R

S_MIN, V_MIN = 110, 50
AREA_MIN, AREA_MAX_FRAC = 6, 0.02


def red_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = (((hsv[..., 0] <= 10) | (hsv[..., 0] >= 170)) & (hsv[..., 1] >= S_MIN) & (hsv[..., 2] >= V_MIN)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def detect(rgb: np.ndarray):
    m = red_mask(rgb)
    n, lab, stats, cen = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    a = stats[k, cv2.CC_STAT_AREA]
    w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
    if a < AREA_MIN or a > AREA_MAX_FRAC * m.size or max(w, h) > 2.2 * min(w, h) + 2:
        return None
    return float(cen[k][0]), float(cen[k][1])


class HSVTracker:
    def __init__(self, env):
        self.env, self.H = env, None
        self.calibrate()

    def _render_top(self):
        e = self.env
        e.renderer.update_scene(e.data, camera="top")
        return e.renderer.render().copy()

    def calibrate(self):
        e = self.env
        e.reset(seed=0)
        xs, ys, pts_px, pts_w = np.linspace(0.12, 0.48, 6), np.linspace(-0.22, 0.22, 6), [], []
        a = e.ball_qadr
        for x in xs:
            for y in ys:
                e.data.qpos[a:a + 3] = [x, y, BALL_R]
                e.data.qvel[:] = 0
                mujoco.mj_forward(e.model, e.data)
                p = detect(self._render_top())
                if p is not None:
                    pts_px.append(p); pts_w.append((x, y))
        self.H, _ = cv2.findHomography(np.array(pts_px, np.float32), np.array(pts_w, np.float32), cv2.RANSAC, 0.01)
        self.n_calib = len(pts_px)

    def world(self, uv):
        p = self.H @ np.array([uv[0], uv[1], 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def __call__(self, rgb: np.ndarray):
        uv = detect(rgb)
        return None if uv is None else self.world(uv)


if __name__ == "__main__":      # accuracy test: clean and randomized sim, ball swept over the lane
    import sys
    from catch_env import CatchEnv
    from sim2real import Realism
    env = CatchEnv(cameras=("top",), height=240, width=320)
    tr = HSVTracker(env)
    print("calibration points:", tr.n_calib, flush=True)
    rl = Realism(env, 1.0)
    rng = np.random.default_rng(3)
    for name, strength in (("clean", 0.0), ("realistic", 1.0)):
        rl.strength = strength
        err, miss, n = [], 0, 0
        for ep in range(30):
            rl.start_episode(rng)
            env.reset(seed=ep)
            for _ in range(8):
                x, y = rng.uniform(0.14, 0.46), rng.uniform(-0.2, 0.2)
                a = env.ball_qadr
                env.data.qpos[a:a + 3] = [x, y, BALL_R]; mujoco.mj_forward(env.model, env.data)
                img = rl.obs(dict(state=env.joint_state(), top=tr._render_top()))["top"]
                p = tr(img); n += 1
                if p is None:
                    miss += 1
                else:
                    err.append(np.hypot(p[0] - x, p[1] - y))
        err = np.array(err)
        print(f"{name}: detected {n - miss}/{n}; error median {100 * np.median(err):.2f} cm, p90 {100 * np.percentile(err, 90):.2f} cm, max {100 * err.max():.1f} cm", flush=True)
        rl.restore()
