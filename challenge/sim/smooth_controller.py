"""SmolVLA controller with action smoothing.

Jitter in the closed loop comes from (1) a fresh flow-matching noise sample at every re-query, so consecutive action
chunks disagree, (2) hard switches between chunks every `period` steps, and (3) per-step noise in the arm targets.
Each fix is a switch so its effect can be measured with smoothness.jerk_rms:
  fixed_noise  one noise tensor per episode (the chunk then depends on the observation only)
  ens_m        temporal ensembling: blend every overlapping chunk, weight exp(-ens_m * age); >0 favours older plans
  ema          first-order low-pass on the 5 arm joints (the gripper is left fast)
  period       re-query interval in control steps (chunks are 50 steps long)
"""
from __future__ import annotations

import time

import numpy as np

from eval_closed_loop import PolicyController


class SmoothPolicyController(PolicyController):
    def __init__(self, ckpt, period: int = 5, fixed_noise: bool = False, ens_m: float = 0.0, ema: float = 0.0,
                 ens_window: int = 10, seed: int = 0, max_step: float | None = None):
        super().__init__(ckpt, n_action_steps=period)
        self.max_step = max_step
        self.period, self.fixed_noise, self.ens_m, self.ema = period, fixed_noise, ens_m, ema
        self.ens_window = ens_window
        self.seed = seed
        self.reset()

    def reset(self):
        self.policy.reset()
        self.i = 0
        self.chunks = []          # (start_step, (T, 6) array)
        self.prev = None
        self.noise = None
        if self.fixed_noise:
            g = self.torch.Generator(device="cpu").manual_seed(self.seed)
            cfg = self.policy.config
            self.noise = self.torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=g).to(cfg.device)

    def _query(self, obs, state):
        torch = self.torch
        img = lambda a: torch.from_numpy(a).permute(2, 0, 1).float().div(255.0)  # noqa: E731
        batch = {"observation.images.top": img(obs["top"]), "observation.images.front": img(obs["front"]),
                 "observation.state": torch.from_numpy(state), "task": self.task}
        batch = self.pre(batch)
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch, noise=self.noise)
            chunk = self.post(chunk)
        return chunk.detach().cpu().numpy().reshape(-1, 6).astype(np.float64)

    task = "Block the red ball at the marked point."

    def act(self, obs, state):
        t0 = time.time()
        if self.i % self.period == 0:
            self.chunks.append((self.i, self._query(obs, state)))
            self.chunks = [c for c in self.chunks if self.i - c[0] < len(c[1])][-self.ens_window:]
        self.times.append(time.time() - t0 if self.i % self.period == 0 else 0.0)
        live = [(s, c) for s, c in self.chunks if 0 <= self.i - s < len(c)]
        if self.ens_m == 0.0 or len(live) == 1:
            s, c = live[-1]
            a = c[self.i - s]
        else:
            ages = np.array([self.i - s for s, _ in live], dtype=np.float64)
            w = np.exp(-self.ens_m * ages)
            a = (w[:, None] * np.stack([c[self.i - s] for s, c in live])).sum(0) / w.sum()
        if self.ema > 0.0 and self.prev is not None:
            a = a.copy()
            a[:5] = self.ema * self.prev[:5] + (1 - self.ema) * a[:5]
        if self.max_step is not None and self.prev is not None:   # max relative target: limit per-step arm motion
            a = a.copy()
            a[:5] = self.prev[:5] + np.clip(a[:5] - self.prev[:5], -self.max_step, self.max_step)
        self.prev = a
        self.i += 1
        return a.astype(np.float32)
