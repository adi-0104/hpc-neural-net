"""
Milestone 3 plots:
  Figure 1 — 2x2 individual validation-loss curves (one per implementation)
  Figure 2 — Overlay of all 4 curves on one axis for direct comparison

Usage:
  python make_plots.py
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")   # non-interactive — write files without opening a window
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# data sources 
BASE = os.path.dirname(__file__)

SOURCES = [
    ("CPU Native (TILE=32, OMP=32)",  os.path.join(BASE, "cpu", "cpu_native.csv"), "#2196F3"),
    ("CPU BLAS  (OpenBLAS, OMP=32)",  os.path.join(BASE, "cpu", "cpu_blas.csv"),   "#4CAF50"),
    ("GPU Native",                    os.path.join(BASE, "gpu", "gpu_native.csv"), "#FF9800"),
    ("GPU cuBLAS",                    os.path.join(BASE, "gpu", "gpu_cublas.csv"), "#F44336"),
]

# helpers 
def parse_csv(path):
    epochs, losses = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["val_loss"]))
    return epochs, losses


def style_ax(ax, title, color):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Validation Loss", fontsize=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # shade the final val-loss value
    ax.axhline(y=ax.lines[0].get_ydata()[-1], color=color,
               linestyle=":", linewidth=0.8, alpha=0.6)


# Figure 1: 2x2 individual 
fig1, axes = plt.subplots(2, 2, figsize=(11, 7.5))
axes = axes.flatten()

for ax, (label, path, color) in zip(axes, SOURCES):
    epochs, losses = parse_csv(path)
    ax.plot(epochs, losses, color=color, linewidth=1.8, label=label)
    final = losses[-1]
    ax.annotate(f"{final:.4f}", xy=(epochs[-1], final),
                xytext=(-4, 6), textcoords="offset points",
                fontsize=7.5, color=color, ha="right")
    style_ax(ax, label, color)

fig1.suptitle(
    "Validation Loss per Epoch — MNIST  (α=0.1 · batch=500 · 50 epochs · 50K train / 10K val)",
    fontsize=11, fontweight="bold", y=1.01
)
plt.tight_layout()
fig1.savefig(os.path.join(BASE, "loss_curves_individual.png"), dpi=150, bbox_inches="tight")
print("Saved  loss_curves_individual.png")


# Figure 2: overlay comparison
fig2, ax2 = plt.subplots(figsize=(8, 5))

for label, path, color in SOURCES:
    epochs, losses = parse_csv(path)
    ax2.plot(epochs, losses, color=color, linewidth=1.8, label=f"{label}  (final {losses[-1]:.4f})")

ax2.set_xlabel("Epoch", fontsize=10)
ax2.set_ylabel("Validation Loss", fontsize=10)
ax2.set_title(
    "All Implementations — Validation Loss Overlay\n"
    "α=0.1 · batch=500 · 50 epochs · 50K train / 10K val",
    fontsize=11, fontweight="bold"
)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax2.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
fig2.savefig(os.path.join(BASE, "loss_curves_overlay.png"), dpi=150, bbox_inches="tight")
print("Saved  loss_curves_overlay.png")

print("Done.")
