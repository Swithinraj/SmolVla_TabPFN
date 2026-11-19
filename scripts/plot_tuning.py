"""Plot training loss and held-out metrics for the three-epoch run."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "tuning_20261116_162927"
log = (RUN / "training.log").read_text(encoding="utf-8", errors="replace")
points = [(float(epoch), float(loss)) for epoch, loss in re.findall(r"epch:([0-9.]+) loss:([0-9.]+)", log)]
reports = [json.loads((RUN / f"evaluation_32_{step:06d}.json").read_text()) for step in (900, 1800, 2700)]
epochs = [0, 1, 2, 3]
base_mae = reports[0]["baseline"]["mean_action_mae"]
mae = [base_mae] + [report["tuned"]["mean_action_mae"] for report in reports]
base_loss = reports[0]["baseline"]["mean_loss"]
held_loss = [base_loss] + [report["tuned"]["mean_loss"] for report in reports]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
axes[0].plot([p[0] for p in points], [p[1] for p in points], color="#3b82f6", marker="o", markersize=3)
axes[0].set(title="Training loss (logged every 100 steps)", xlabel="Approximate epochs", ylabel="SmolVLA loss")
axes[0].grid(alpha=0.25)
axes[1].plot(epochs, held_loss, color="#9b5de5", marker="o")
axes[1].axhline(base_loss, color="#888", linestyle="--", linewidth=1, label="Original checkpoint")
axes[1].set(title="Held-out policy loss (32 frames)", xlabel="Local fine-tuning epochs", ylabel="SmolVLA loss", xticks=epochs)
axes[1].set_ylim(bottom=0)
axes[1].grid(alpha=0.25)
axes[2].plot(epochs, mae, color="#ef6c52", marker="o")
axes[2].axhline(base_mae, color="#888", linestyle="--", linewidth=1, label="Original checkpoint")
axes[2].set(title="Held-out first-action error (32 frames)", xlabel="Local fine-tuning epochs", ylabel="Mean absolute error", xticks=epochs)
axes[2].set_ylim(bottom=0)
axes[2].grid(alpha=0.25)
fig.suptitle("SmolVLA local tuning on three LIBERO episodes", fontsize=14, fontweight="bold")
path = RUN / "convergence.png"
fig.savefig(path, dpi=170)
print(path)
