"""TabPFN 'spatial brain': few-shot zone prediction + uncertainty gate + few-shot adaptation (sim, no policy needed).
Task: the ball rolls toward the arm; the brain predicts which of 4 zones (y-bands at the catch plane) it will arrive in and
fires a canonical SmolVLA prompt ("catch the ball in zone k") only when confident. A hidden rule flip (zones reversed)
tests few-shot adaptation. Rows = the tracker features at one frame (+ session id)."""
import json, os, sys
from pathlib import Path
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from catch_env import CatchEnv
from catch_expert import CatchExpert
from expert import X_INT
from forecast import MIN_PTS, features
from launches import mixed_irregular
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

EDGES = np.array([-0.10, 0.0, 0.10])      # 4 zones across the lane
NZ = 4
PROMPTS = [f"Catch the red ball in zone {k + 1} and hold it." for k in range(NZ)]
zone = lambda y: int(np.searchsorted(EDGES, y))  # noqa: E731

env = CatchEnv(cameras=("top",), height=60, width=80)
ex = CatchExpert(env)
rng = np.random.default_rng(0)


def sequences(n, seed0):
    out, seed = [], seed0
    while len(out) < n:
        mixed_irregular(env, seed); seed += 1
        t, x, y, c = ex.ghost.track(fps=30)
        if c is None or len(t) < MIN_PTS + 3:
            continue
        xo, yo = x + rng.normal(0, .004, len(x)), y + rng.normal(0, .004, len(y))
        rows = []
        for k in range(MIN_PTS - 1, len(t), 3):
            if c[1] - t[k] < 0.1:
                break
            rows.append((c[1] - t[k], features(t[:k + 1], xo[:k + 1], yo[:k + 1])))
        if rows:
            out.append(dict(z=zone(c[0]), y=c[0], rows=rows))
    return out


def clf(): return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4)


def flat(seqs, per, flip=False, session=0):
    X, Y = [], []
    for s in seqs:
        idx = rng.choice(len(s["rows"]), size=min(per, len(s["rows"])), replace=False)
        for i in idx:
            X.append(np.append(s["rows"][i][1], session)); Y.append(NZ - 1 - s["z"] if flip else s["z"])
    return np.array(X), np.array(Y)


train, test = sequences(500, 0), sequences(120, 50_000)
print("sequences ready", len(train), len(test), flush=True)
res = {}
# ---- 1) accuracy + calibration along the approach, and the uncertainty gate
X, Y = flat(train, 4)
m = clf().fit(X, Y)
P, Z, TR = [], [], []
for s in test:
    pr = m.predict_proba(np.array([np.append(r[1], 0) for r in s["rows"]]))
    P.append(pr); Z.append(s["z"]); TR.append([r[0] for r in s["rows"]])
by_lead = {}
for L in (1.2, 0.9, 0.6, 0.4, 0.25):
    acc, conf = [], []
    for pr, z, tr in zip(P, Z, TR):
        k = int(np.argmin(np.abs(np.array(tr) - L)))
        if abs(tr[k] - L) < 0.1:
            acc.append(int(pr[k].argmax() == z)); conf.append(pr[k].max())
    by_lead[L] = dict(n=len(acc), acc=float(np.mean(acc)), mean_conf=float(np.mean(conf)))
res["acc_by_lead"] = by_lead
print("accuracy / mean confidence by time-to-arrival:", by_lead, flush=True)
gate = {}
for name, thr in (("fire_at_first_row", 0.0), ("conf>=0.6", 0.6), ("conf>=0.8", 0.8), ("conf>=0.9", 0.9)):
    fired = ok = 0; lead = []
    for pr, z, tr in zip(P, Z, TR):
        k = next((i for i in range(len(pr)) if pr[i].max() >= thr), None)
        if k is None:
            continue
        fired += 1; ok += int(pr[k].argmax() == z); lead.append(tr[k])
    gate[name] = dict(fired=f"{fired}/{len(P)}", correct_of_fired=ok / max(fired, 1), wasted_chunks=fired - ok,
                      mean_lead_s=float(np.mean(lead)) if lead else None)
res["gate"] = gate
print("gate:", json.dumps(gate, indent=1), flush=True)
# ---- 2) few-shot adaptation: zones reversed (hidden rule flip), with / without old context
newseq = sequences(80, 90_000)
Xt, Yt = [], []
for s in test:
    k = int(np.argmin(np.abs(np.array([r[0] for r in s["rows"]]) - 0.6)))
    Xt.append(np.append(s["rows"][k][1], 1)); Yt.append(NZ - 1 - s["z"])
Xt, Yt = np.array(Xt), np.array(Yt)
Xo, Yo = flat(train, 2)
adapt = {}
for n in (0, 3, 5, 10, 20):
    a_old, a_new = [], []
    for rep in range(5 if n else 1):
        sub = [newseq[i] for i in rng.choice(len(newseq), n, replace=False)] if n else []
        if n:
            Xn, Yn = flat(sub, 1, flip=True, session=1)
            a_old.append((clf().fit(np.vstack([Xo, Xn]), np.concatenate([Yo, Yn])).predict(Xt) == Yt).mean())
            a_new.append((clf().fit(Xn, Yn).predict(Xt) == Yt).mean())
        else:
            a_old.append((clf().fit(Xo, Yo).predict(Xt) == Yt).mean()); a_new.append(float("nan"))
    adapt[n] = dict(old_plus_new=float(np.mean(a_old)), old_plus_new_sd=float(np.std(a_old)), new_only=float(np.nanmean(a_new)))
    print("adapt n_new =", n, adapt[n], flush=True)
res["adapt"] = adapt
Path(__file__).resolve().parents[2].joinpath("outputs/spatial_brain.json").write_text(json.dumps(res, indent=2))
print("done")
