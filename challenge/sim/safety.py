"""Safety filter between the policy and the arm (use in sim AND on the real SO-101).

Limits are the hardware joint ranges (degrees mode, LeRobot so101_follower) minus a margin, so calibration error cannot push a
joint into its hard stop. Order of checks per command:
  1. non-finite command  -> hold the last safe command
  2. joint range clamp (+ margin)           gripper clamped to [GRIP_MIN, GRIP_MAX]
  3. per-step speed limit (deg/step)        the real servos are slower than an ideal sim actuator anyway
  4. table check (optional, needs the model): a pose whose arm geometry touches the table is rejected -> hold last command
  5. stall guard: if the measured joint lags the command by > STALL_DEG for STALL_STEPS steps, freeze at the measured pose
Everything the filter changed is counted in `stats` so a deployment can report how often it intervened.
"""
from __future__ import annotations

import numpy as np

# hardware ranges (deg) of the SO-101 joints as in the MuJoCo model / LeRobot degrees mode
RANGE = np.array([[-110.0, 110.0], [-100.0, 100.0], [-96.8, 96.8], [-95.0, 95.0], [-157.0, 163.0]])
MARGIN = 6.0            # deg kept clear of each hard stop
MAX_STEP = 12.0         # deg per control step (30 fps) = 360 deg/s, above the 7-12 deg/step servo limit used in the sim
GRIP_MIN, GRIP_MAX = 0.0, 100.0
GRIP_STEP = 35.0
STALL_DEG, STALL_STEPS = 35.0, 15
TABLE_PEN = 0.020       # m: reject poses that push the arm >20 mm into the table (skimming contact is normal; 12 mm over-triggered)


class SafetyFilter:
    def __init__(self, margin: float = MARGIN, max_step: float = MAX_STEP, grip_step: float = GRIP_STEP, env=None):
        self.ik = None
        if env is not None:                      # optional table-collision check with the MuJoCo kinematic model
            from catch_ik import CatchIK
            self.ik = CatchIK(env)
        self.lo = RANGE[:, 0] + margin
        self.hi = RANGE[:, 1] - margin
        self.max_step, self.grip_step = max_step, grip_step
        self.reset()

    def reset(self, state: np.ndarray | None = None):
        self.last = None if state is None else np.asarray(state, dtype=np.float64).copy()
        self.stall = 0
        self.stats = dict(nonfinite=0, range=0, speed=0, stall=0, table=0, steps=0)

    def __call__(self, action, state):
        s = np.asarray(state, dtype=np.float64)
        a = np.asarray(action, dtype=np.float64).copy()
        if self.last is None:
            self.last = s.copy()
        self.stats["steps"] += 1
        if not np.all(np.isfinite(a)):
            self.stats["nonfinite"] += 1
            return self.last.astype(np.float32)
        before = a.copy()
        a[:5] = np.clip(a[:5], self.lo, self.hi)
        a[5] = np.clip(a[5], GRIP_MIN, GRIP_MAX)
        self.stats["range"] += int(np.any(a != before))
        before = a.copy()
        a[:5] = self.last[:5] + np.clip(a[:5] - self.last[:5], -self.max_step, self.max_step)
        a[5] = self.last[5] + np.clip(a[5] - self.last[5], -self.grip_step, self.grip_step)
        self.stats["speed"] += int(np.any(a != before))
        if self.ik is not None:
            from catch_ik import table_contacts
            if table_contacts(self.ik, np.deg2rad(a[:5]), min_pen=TABLE_PEN):   # pose would drive the arm into the table
                self.stats["table"] += 1
                a[:5] = self.last[:5]
        lag = np.abs(s[:5] - a[:5]).max()
        self.stall = self.stall + 1 if lag > STALL_DEG else 0
        if self.stall >= STALL_STEPS:        # the arm is not following (collision, torque limit, lost power): stop pushing
            self.stats["stall"] += 1
            a[:5] = s[:5]
        self.last = a
        return a.astype(np.float32)
