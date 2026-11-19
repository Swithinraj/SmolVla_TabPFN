"""Launch generators for the irregular-motion regime (hidden slope + rough patch + random mid-roll nudge)."""
from __future__ import annotations

import numpy as np

from ball_env import BallBlockEnv

KICK_PROB = 0.5


def irregular_launch(env: BallBlockEnv, seed: int, joint_noise: float = 0.0):
    """Hard dynamics as in forecast_benchmark.hard_launch, plus (50% of launches) an unpredictable nudge."""
    rng = np.random.default_rng(20_000 + seed)
    start = (rng.uniform(0.38, 0.46), rng.uniform(-0.13, 0.13))
    speed, angle = rng.uniform(0.16, 0.28), rng.uniform(-0.35, 0.35)
    rf, slope = rng.uniform(0, 1e-4), rng.uniform(-0.3, 0.3)
    kr = np.random.default_rng(40_000 + seed)
    kick = None
    if kr.random() < KICK_PROB:
        kick = (int(kr.integers(10, 26)), kr.uniform(-0.03, 0.03), kr.uniform(-0.06, 0.06))  # step 0.33-0.83 s
    env.reset(seed=seed, start=start, speed=speed, angle=angle, rolling_friction=rf, hard=True, slope=slope,
              kick=kick, joint_noise=joint_noise)


def easy_launch(env: BallBlockEnv, seed: int, joint_noise: float = 0.0):
    rng = np.random.default_rng(10_000 + seed)
    env.reset(seed=seed, speed=rng.uniform(0.10, 0.22), rolling_friction=rng.uniform(0, 1e-4),
              joint_noise=joint_noise)


def mixed_irregular(env: BallBlockEnv, seed: int, joint_noise: float = 0.0):
    """Even seeds: irregular (hard + nudges); odd seeds: easy."""
    (irregular_launch if seed % 2 == 0 else easy_launch)(env, seed, joint_noise)
