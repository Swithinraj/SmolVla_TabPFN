"""Build docs/figures/*.png and results/summary.json from the evaluation JSONs (run from the repo root).
Colours: Okabe-Ito (colour-blind safe). Every bar carries its held/n count."""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(".")
OUT = R / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)
C = {"oracle": "#009E73", "tabpfn": "#0072B2", "cv": "#E69F00", "none": "#999999", "expert": "#CC79A7"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def load(p):
    return json.load(open(R / p)) if (R / p).exists() else None


def bars(ax, groups, series, get, title, ylabel, colors):
    w = 0.8 / len(series)
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        for j, g in enumerate(groups):
            v = get(g, s)
            if v is None:
                continue
            xx = x[j] + i * w - 0.4 + w / 2
            ax.bar(xx, v[0] / v[1] * 100, w * 0.92, color=colors[s], label=s if j == 0 else None)
            ax.text(xx, v[0] / v[1] * 100 + 1.5, f"{v[0]}/{v[1]}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 112)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9)
    ax.legend(frameon=False, ncol=len(series), fontsize=8, loc="upper right")


summary = {}

# 1) blocking stages (earlier task): which target the policy trusted
cmp_ = load("outputs/so101_ball_stage3/closed_loop_compare/closed_loop.json")
if cmp_:
    r = cmp_["results"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, split in zip(axs, ("indist", "hard")):
        def get(g, s, split=split):
            v = r[f"{'stage2' if g.startswith('stage 2') else 'stage3'}|{split}|{s}"]
            return v["blocked"], v["n"]
        bars(ax, ["stage 2 (exact targets)", "stage 3 (noisy targets)"], ["oracle", "tabpfn", "cv", "none"], get,
             f"Blocking task, {split} launches", "blocked (%)", C)
    fig.suptitle("Blocking (superseded task): stage 3 trained on noisy forecasts and learned to ignore the target",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "blocking_stage2_vs_stage3.png", bbox_inches="tight")
    plt.close(fig)

# 2) catch: big run, held by target mode
big = load("outputs/catch_big/eval/closed_loop.json")
pil = load("outputs/catch_pilot_s2/eval/closed_loop.json")
if big:
    r = big["results"]
    summary["catch_big"] = {k: dict(held=v["held"], n=v["n"], jerk=v["jerk_median"], points=v["points"])
                            for k, v in r.items() if k.startswith("big|")}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, cond in zip(axs, ("clean", "real1")):
        def get(g, s, cond=cond):
            v = r[f"big|smoother|{cond}|{g}|{s}"]
            return v["held"], v["n"]
        bars(ax, ["indist", "hard"], ["oracle", "tabpfn", "none"], get,
             f"Catch & hold, {'clean sim' if cond == 'clean' else 'randomized sim-to-real'} ('smoother' controller)",
             "held at end (%)", C)
    fig.tight_layout()
    fig.savefig(OUT / "catch_held_by_target.png", bbox_inches="tight")
    plt.close(fig)

    # 3) smoothness: jerk and points by controller variant (clean, in-distribution, oracle)
    rows = []
    for name, d, key in (("pilot raw", pil, "pilot|raw|clean|indist|oracle"),
                         ("pilot smooth", pil, "pilot|smooth|clean|indist|oracle"),
                         ("big smooth", big, "big|smooth|clean|indist|oracle"),
                         ("big smoother", big, "big|smoother|clean|indist|oracle"),
                         ("big safe", big, "big|safe|clean|indist|oracle")):
        if d and key in d["results"]:
            v = d["results"][key]
            rows.append((name, v["jerk_median"], v["points"] / v["n"]))
    ref = (pil or big)["results"].get("expert_oracle|clean|indist")
    if ref:
        rows.append(("scripted expert", ref["jerk_median"], ref["points"] / ref["n"]))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    names = [x[0] for x in rows]
    axs[0].barh(names, [x[1] for x in rows], color="#0072B2")
    axs[0].set_xscale("log")
    axs[0].axvline(4000, color="#D55E00", ls="--")
    axs[0].set_xlabel("median jerk (deg/s^3, log); dashed = 4000 score threshold")
    axs[0].set_title("Jerk of the arm commands (lower = smoother)", loc="left", fontsize=9)
    axs[0].invert_yaxis()
    axs[1].barh(names, [x[2] for x in rows], color="#009E73")
    axs[1].set_xlabel("points per launch = held x min(1, 4000/jerk)")
    axs[1].set_title("Score per launch (clean, in-distribution, oracle targets)", loc="left", fontsize=9)
    axs[1].invert_yaxis()
    axs[1].set_yticklabels([])
    fig.tight_layout()
    fig.savefig(OUT / "catch_smoothness.png", bbox_inches="tight")
    plt.close(fig)

# 4) forecast error vs lead time (parsed from the printed log)
txt = Path("outputs/catch_big/forecast_check.txt")
if txt.exists():
    cur, data = None, {}
    for line in txt.read_text().splitlines():
        m = re.match(r"^(clean|real1) (indist|hard)$", line.strip())
        if m:
            cur = f"{m.group(1)} {m.group(2)}"
            continue
        m = re.match(r"^\s+(tabpfn|cv) (\{.*\})$", line)
        if m and cur:
            data.setdefault(cur, {})[m.group(1)] = eval(m.group(2))
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, (k, v) in zip(axs, data.items()):
        for s, col in (("tabpfn", C["tabpfn"]), ("cv", C["cv"])):
            leads = sorted(v[s], key=float, reverse=True)
            ax.plot([float(l) for l in leads], [v[s][l][0] for l in leads], "o-", color=col, label=f"{s} median")
            ax.plot([float(l) for l in leads], [v[s][l][1] for l in leads], "o--", color=col, alpha=0.5, label=f"{s} p90")
        ax.set_title(k, loc="left")
        ax.set_xlabel("time before arrival (s)")
        ax.invert_xaxis()
    axs[0].set_ylabel("crossing-position error (cm)")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "forecast_error_vs_lead.png", bbox_inches="tight")
    plt.close(fig)

# 5) spatial brain
sb = load("outputs/spatial_brain.json")
if sb:
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    L = sorted(sb["acc_by_lead"], key=float, reverse=True)
    axs[0].plot([float(l) for l in L], [100 * sb["acc_by_lead"][l]["acc"] for l in L], "o-", color=C["tabpfn"], label="accuracy")
    axs[0].plot([float(l) for l in L], [100 * sb["acc_by_lead"][l]["mean_conf"] for l in L], "o--", color=C["none"],
                label="mean confidence")
    axs[0].invert_xaxis()
    axs[0].set_ylim(50, 100)
    axs[0].set_xlabel("time before arrival (s)")
    axs[0].set_ylabel("%")
    axs[0].legend(frameon=False)
    axs[0].set_title("4-zone prediction vs lead time", loc="left", fontsize=9)
    g = sb["gate"]
    names = list(g)
    axs[1].bar(range(len(names)), [g[n]["wasted_chunks"] for n in names], color="#D55E00")
    for i, n in enumerate(names):
        axs[1].text(i, g[n]["wasted_chunks"] + 0.7, f"fired {g[n]['fired']}", ha="center", fontsize=7)
    axs[1].set_xticks(range(len(names)))
    axs[1].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    axs[1].set_ylabel("wrong-zone prompts fired")
    axs[1].set_title("Uncertainty gate (120 test launches)", loc="left", fontsize=9)
    ad = sb["adapt"]
    ns = sorted(ad, key=int)
    axs[2].plot([int(n) for n in ns], [100 * ad[n]["old_plus_new"] for n in ns], "o-", color=C["tabpfn"], label="old + new rows")
    axs[2].plot([int(n) for n in ns[1:]], [100 * ad[n]["new_only"] for n in ns[1:]], "o--", color=C["cv"], label="new rows only")
    axs[2].axhline(25, color=C["none"], ls=":")
    axs[2].text(12, 27, "chance", fontsize=8, color=C["none"])
    axs[2].set_xlabel("examples after the rule flip")
    axs[2].set_ylabel("zone accuracy (%)")
    axs[2].legend(frameon=False)
    axs[2].set_title("Few-shot adaptation (zones reversed)", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "spatial_brain.png", bbox_inches="tight")
    plt.close(fig)

# 6) marker / tracker ablation (catch, clean + randomized, 'smoother', TabPFN targets)
abl = []


def grab(path, label):
    d = load(path)
    if not d:
        return
    for split in ("indist", "hard"):
        for cond in ("clean", "real1"):
            k = f"big|smoother|{cond}|{split}|tabpfn"
            if k in d["results"]:
                abl.append((label, f"{split}/{cond}", d["results"][k]["held"], d["results"][k]["n"]))


grab("outputs/catch_big/eval/closed_loop.json", "marker + true track")
grab("outputs/catch_big/eval_nomarker/closed_loop.json", "no marker, true track")
grab("outputs/catch_big/eval_hsv/closed_loop.json", "no marker, HSV track")
if abl:
    labels = ["marker + true track", "no marker, true track", "no marker, HSV track"]
    groups = sorted({a[1] for a in abl})
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.26
    x = np.arange(len(groups))
    cols = ["#0072B2", "#E69F00", "#009E73"]
    for i, lab in enumerate(labels):
        for j, g in enumerate(groups):
            v = [a for a in abl if a[0] == lab and a[1] == g]
            if v:
                ax.bar(x[j] + (i - 1) * w, 100 * v[0][2] / v[0][3], w * 0.92, color=cols[i], label=lab if j == 0 else None)
                ax.text(x[j] + (i - 1) * w, 100 * v[0][2] / v[0][3] + 1.5, f"{v[0][2]}/{v[0][3]}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("held (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Real-world inputs: no marker, forecaster fed only by HSV detections (TabPFN targets)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "marker_hsv_ablation.png", bbox_inches="tight")
    plt.close(fig)
    summary["marker_hsv_ablation"] = [dict(setting=a[0], split_cond=a[1], held=a[2], n=a[3]) for a in abl]

json.dump(summary, open(R / "results/summary.json", "w"), indent=1)
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
