"""Smoothness of an executed command sequence. Jitter shows up as high jerk (3rd derivative) of the joint targets.

jerk_rms: RMS of the 3rd finite difference of the 5 arm joints (deg/s^3). The gripper is excluded (it moves in
deliberate steps). smooth_score compares to a reference (the scripted expert): 1.0 = as smooth as the expert."""
from __future__ import annotations

import numpy as np

# Median arm jerk of the scripted catch expert (measured: 3.6k easy, 4.3k hard launches).
EXPERT_JERK_REF = 4000.0


def jerk_rms(actions, fps: int = 30) -> float:
    a = np.asarray(actions, dtype=np.float64)[:, :5]
    if len(a) < 5:
        return 0.0
    j = np.diff(a, n=3, axis=0) * fps ** 3
    return float(np.sqrt(np.mean(j ** 2)))


def step_rms(actions, fps: int = 30) -> float:
    """RMS joint speed (deg/s), context for the jerk number."""
    a = np.asarray(actions, dtype=np.float64)[:, :5]
    return float(np.sqrt(np.mean((np.diff(a, axis=0) * fps) ** 2))) if len(a) > 1 else 0.0


def smooth_score(jerk: float, ref: float | None = None) -> float:
    ref = EXPERT_JERK_REF if ref is None else ref
    return float(min(1.0, ref / max(jerk, 1e-9)))
