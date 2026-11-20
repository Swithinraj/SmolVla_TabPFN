"""Ball crossing forecasters: constant velocity, constant acceleration (quadratic fit) and TabPFN.

A forecaster sees only a *noisy, time-stamped* track of the ball (t, x, y) up to now and predicts where
(y) and when (time remaining) the ball centre reaches the paddle plane x = X_INT.
Rows only use observations available at their timestamp (causal).
"""
from __future__ import annotations

import numpy as np

from expert import X_INT

OBS_NOISE = 0.004     # std of the tracker's position error per axis (m)
MIN_PTS = 6           # frames needed before any forecast
VEL_WIN, ACC_WIN = 6, 10
FEATURES = ["x", "y", "vx", "vy", "ax", "ay", "elapsed", "x0", "y0", "y_cv", "t_cv"]


def _fit(t, v, deg):
    return np.polyfit(t - t[-1], v, deg)  # coefficients about the latest timestamp


def cv_forecast(t, x, y):
    """Constant-velocity extrapolation from a linear fit of the last VEL_WIN points. -> (y_cross, t_remaining)"""
    n = min(VEL_WIN, len(t))
    px, py = _fit(t[-n:], x[-n:], 1), _fit(t[-n:], y[-n:], 1)
    vx, vy, x_now, y_now = px[0], py[0], px[1], py[1]
    vx = min(vx, -0.03)  # moving toward the arm; avoid division blow-ups when the track is noisy
    tr = max((x_now - X_INT) / -vx, 0.0)
    return y_now + vy * tr, tr


def quad_forecast(t, x, y):
    """Constant-acceleration extrapolation (quadratic LS fit of the last ACC_WIN points)."""
    n = min(ACC_WIN, len(t))
    if n < 5:
        return cv_forecast(t, x, y)
    tt = t[-n:] - t[-1]
    cx, cy = np.polyfit(tt, x[-n:], 2), np.polyfit(tt, y[-n:], 2)
    # earliest tau >= 0 where x(tau) = X_INT: solve a tau^2 + b tau + (c - X_INT) = 0
    roots = np.roots([cx[0], cx[1], cx[2] - X_INT])
    real = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real >= 0]
    if not real:
        return cv_forecast(t, x, y)
    tau = min(real)
    if tau > 3.0:
        return cv_forecast(t, x, y)
    return float(np.polyval(cy, tau)), float(tau)


def features(t, x, y):
    """Feature row from the observed history (arrays up to and including the current frame)."""
    n_v, n_a = min(VEL_WIN, len(t)), min(ACC_WIN, len(t))
    px, py = _fit(t[-n_v:], x[-n_v:], 1), _fit(t[-n_v:], y[-n_v:], 1)
    qx, qy = _fit(t[-n_a:], x[-n_a:], 2), _fit(t[-n_a:], y[-n_a:], 2)
    y_cv, t_cv = cv_forecast(t, x, y)
    return np.array([px[1], py[1], px[0], py[0], 2 * qx[0], 2 * qy[0], t[-1] - t[0], x[0], y[0], y_cv, t_cv],
                    dtype=np.float64)


def make_rows(ghost, env, launch_fn, n_launch: int, seed0: int, rows_per_launch: int, rng: np.random.Generator,
              noise: float = OBS_NOISE, min_remaining: float = 0.15, fps: int = 30):
    """Simulate n launches ball-only; return (X, targets[y_cross, t_remaining], meta) rows from random frames."""
    X, T, meta = [], [], []
    seed = seed0
    kept = 0
    while kept < n_launch:
        launch_fn(env, seed)
        seed += 1
        t, x, y, c = ghost.track(fps=fps)
        if c is None or len(t) < MIN_PTS + 3:
            continue
        xo = x + rng.normal(0, noise, len(x))
        yo = y + rng.normal(0, noise, len(y))
        t_cross, y_cross = c[1], c[0]
        valid = [k for k in range(MIN_PTS - 1, len(t)) if t_cross - t[k] >= min_remaining]
        if not valid:
            continue
        for k in rng.choice(valid, size=min(rows_per_launch, len(valid)), replace=False):
            X.append(features(t[:k + 1], xo[:k + 1], yo[:k + 1]))
            T.append([y_cross, t_cross - t[k]])
            meta.append(dict(seed=seed - 1, k=int(k), remaining=float(t_cross - t[k])))
        kept += 1
    return np.array(X), np.array(T), meta


# ----------------------------------------------------------------------------------------------------------
# v2: path prediction + confidence + online correction
# ----------------------------------------------------------------------------------------------------------
PATH_TAUS = (0.3, 0.6, 0.9)   # seconds ahead at which the ball's y is also predicted
TAU_CROSS = 3.0               # "at or beyond the crossing" -> crossing y
F = {n: i for i, n in enumerate(FEATURES)}
STATE_EXTRA = 6               # [y_cross, time_to_cross, sigma_y, y(+0.3), y(+0.6), y(+0.9)]
SIGMA_SCALE = 1.5             # calibration of TabPFN's 10-90% band (see forecast_benchmark_v2)


def make_rows_v2(ghost, env, launch_fn, n_launch: int, seed0: int, rows_per_launch: int, rng: np.random.Generator,
                 noise: float = OBS_NOISE, min_remaining: float = 0.15, fps: int = 30):
    """Like make_rows, plus the true ball y at each PATH_TAUS ahead (capped at the crossing)."""
    X, T, P = [], [], []
    seed, kept = seed0, 0
    while kept < n_launch:
        launch_fn(env, seed)
        seed += 1
        t, x, y, c = ghost.track(fps=fps)
        if c is None or len(t) < MIN_PTS + 3:
            continue
        xo, yo = x + rng.normal(0, noise, len(x)), y + rng.normal(0, noise, len(y))
        y_cross, t_cross = c
        valid = [k for k in range(MIN_PTS - 1, len(t)) if t_cross - t[k] >= min_remaining]
        if not valid:
            continue
        ts, ys = np.append(t, t_cross), np.append(y, y_cross)
        for k in rng.choice(valid, size=min(rows_per_launch, len(valid)), replace=False):
            X.append(features(t[:k + 1], xo[:k + 1], yo[:k + 1]))
            T.append([y_cross, t_cross - t[k]])
            P.append([np.interp(min(t[k] + tau, t_cross), ts, ys) for tau in PATH_TAUS])
        kept += 1
    return np.array(X), np.array(T), np.array(P)


def _default_device() -> str:
    """TABPFN_DEVICE overrides; otherwise cuda when a GPU is visible, else cpu (CPU-only recording nodes)."""
    import os
    d = os.environ.get("TABPFN_DEVICE")
    if d:
        return d
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class TabPFNForecasterV2:
    """TabPFN forecaster that predicts the ball's path (y at +0.3/+0.6/+0.9 s), the crossing y with a 10-90%
    band, and the crossing time. The y-regressor takes the horizon tau as an extra feature so all four y outputs
    come from one batched forward pass; the time regressor is a second pass."""

    def __init__(self, device: str | None = None, n_estimators: int = 4, ctx_y: int = 1500, ctx_t: int = 1500):
        import os
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
        device = device or _default_device()
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        mk = lambda: TabPFNRegressor.create_default_for_version(  # noqa: E731
            ModelVersion.V2, device=device, n_estimators=n_estimators, fit_mode="fit_with_cache")
        self.ry, self.rt = mk(), mk()
        self.ctx_y, self.ctx_t = ctx_y, ctx_t
        self.taus = list(PATH_TAUS) + [TAU_CROSS]

    def fit(self, X, T, P, seed: int = 0):
        rng = np.random.default_rng(seed)
        rows, res = [], []
        for j, tau in enumerate(self.taus):
            y_cv = X[:, F["y"]] + X[:, F["vy"]] * np.minimum(tau, X[:, F["t_cv"]])
            truth = T[:, 0] if tau == TAU_CROSS else P[:, j]
            rows.append(np.column_stack([X, np.full(len(X), tau)]))
            res.append(truth - y_cv)
        A, r = np.vstack(rows), np.concatenate(res)
        if len(A) > self.ctx_y:
            idx = rng.choice(len(A), self.ctx_y, replace=False)
            A, r = A[idx], r[idx]
        self.ry.fit(A, r)
        it = rng.choice(len(X), min(self.ctx_t, len(X)), replace=False)
        self.rt.fit(X[it], (T[:, 1] - X[:, F["t_cv"]])[it])
        return self

    def predict(self, x):
        """x: one feature row. -> dict(y_cross, sigma_y, path[3], t_rem)"""
        q = np.column_stack([np.tile(x, (len(self.taus), 1)), self.taus])
        out = self.ry.predict(q, output_type="full")
        qs = np.array(out["quantiles"])                       # (n_quantiles, 4) default 0.1 ... 0.9
        y_cv = x[F["y"]] + x[F["vy"]] * np.minimum(self.taus, x[F["t_cv"]])
        y_pred = y_cv + np.asarray(out["mean"])
        # 10-90% band -> sigma of a normal, x1.5: measured 80% coverage of the raw band was only 61%
        sigma = SIGMA_SCALE * float((qs[-1, -1] - qs[0, -1]) / 2.563)
        t_rem = max(float(x[F["t_cv"]] + self.rt.predict(x[None])[0]), 0.0)
        return dict(y_cross=float(y_pred[-1]), sigma_y=max(sigma, 0.0), path=y_pred[:-1].astype(float), t_rem=t_rem)


class CrossingFilter:
    """Scalar Kalman filters on the stream of forecasts: each new forecast corrects the running estimate,
    weighted by its confidence; process noise lets it follow a sudden change (e.g. a nudge)."""

    def __init__(self, qy: float = 0.012 ** 2, qt: float = 0.02 ** 2, rt: float = 0.06 ** 2, sy_floor: float = 0.005):
        self.qy, self.qt, self.rt, self.floor = qy, qt, rt, sy_floor
        self.y = None

    def update(self, y: float, sy: float, t_abs: float):
        ry = max(sy, self.floor) ** 2
        if self.y is None:
            self.y, self.py, self.t, self.pt = y, ry, t_abs, self.rt
        else:
            self.py += self.qy
            k = self.py / (self.py + ry)
            self.y += k * (y - self.y)
            self.py *= 1 - k
            self.pt += self.qt
            kt = self.pt / (self.pt + self.rt)
            self.t += kt * (t_abs - self.t)
            self.pt *= 1 - kt
        return self.y, float(np.sqrt(self.py)), self.t


class ForecastPipeline:
    """Tracker readings -> (TabPFN | cv | quad) forecast every PERIOD steps -> compute delay -> optional filter.
    Used identically when recording demonstrations and when evaluating, so training matches deployment."""

    def __init__(self, kind: str, tp: TabPFNForecasterV2 | None = None, delay: float = 0.0, period: int = 3,
                 use_filter: bool = False):  # filtering successive forecasts hurt in the benchmark -> off
        self.kind, self.tp, self.delay, self.period, self.use_filter = kind, tp, delay, period, use_filter
        self.reset()

    def reset(self):
        self.hist, self.pending, self.current = [], [], None
        self.filt = CrossingFilter()

    def update(self, step: int, t: float, xo: float, yo: float):
        self.hist.append((t, xo, yo))
        if len(self.hist) >= MIN_PTS and step % self.period == 0:
            h = np.array(self.hist)
            tt, xx, yy = h[:, 0], h[:, 1], h[:, 2]
            if self.kind == "tabpfn":
                f = self.tp.predict(features(tt, xx, yy))
                y, tr, sig, path = f["y_cross"], f["t_rem"], f["sigma_y"], f["path"]
                y = float(np.clip(y, -0.22, 0.22))
                if self.use_filter:
                    y, sig, t_abs = self.filt.update(y, sig, t + tr)
                    y = float(np.clip(y, -0.22, 0.22))
                else:
                    t_abs = t + tr
            else:
                y, tr = (cv_forecast if self.kind == "cv" else quad_forecast)(tt, xx, yy)
                y, t_abs, sig = float(np.clip(y, -0.22, 0.22)), t + tr, 0.05
                fx = features(tt, xx, yy)
                path = np.array([fx[F["y"]] + fx[F["vy"]] * min(tau, tr) for tau in PATH_TAUS])
            self.pending.append((t + self.delay, dict(y=y, t_abs=float(t_abs), sigma=float(sig), path=np.asarray(path))))
        while self.pending and self.pending[0][0] <= t:
            self.current = self.pending.pop(0)[1]

    def vector(self, t: float):
        """-> (state extras [6], marker y or None)"""
        c = self.current
        if c is None:
            return np.zeros(STATE_EXTRA, np.float32), None
        v = np.array([c["y"], max(c["t_abs"] - t, 0.0), c["sigma"], *c["path"]], dtype=np.float32)
        return v, c["y"]


class TabPFNForecaster:
    """Two TabPFN regressors (y and time) trained on residuals against the constant-velocity forecast."""

    def __init__(self, device: str | None = None, n_estimators: int = 4, max_context: int | None = None):
        import os
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
        device = device or _default_device()
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        self.device, self.max_context = device, max_context
        mk = lambda: TabPFNRegressor.create_default_for_version(  # noqa: E731
            ModelVersion.V2, device=device, n_estimators=n_estimators, fit_mode="fit_with_cache")
        self.ry, self.rt = mk(), mk()

    def fit(self, X, T, seed: int = 0):
        if self.max_context and len(X) > self.max_context:
            idx = np.random.default_rng(seed).choice(len(X), self.max_context, replace=False)
            X, T = X[idx], T[idx]
        y_res = T[:, 0] - X[:, FEATURES.index("y_cv")]
        t_res = T[:, 1] - X[:, FEATURES.index("t_cv")]
        self.ry.fit(X, y_res)
        self.rt.fit(X, t_res)
        return self

    def predict(self, X, quantiles=(0.1, 0.9)):
        """-> y_cross, t_remaining, (y_lo, y_hi)"""
        X = np.atleast_2d(X)
        y_cv, t_cv = X[:, FEATURES.index("y_cv")], X[:, FEATURES.index("t_cv")]
        ry = self.ry.predict(X)
        rt = self.rt.predict(X)
        q = np.array(self.ry.predict(X, output_type="quantiles", quantiles=list(quantiles)))  # (nq, n)
        return y_cv + ry, np.maximum(t_cv + rt, 0.0), (y_cv + q[0], y_cv + q[-1])
