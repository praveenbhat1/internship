"""
Render the 5 validation metrics as a clean slide (metrics_summary.png).
Reads features/metrics.json -> update those numbers with your training [done] line, then re-run.
Usage: python metrics_card.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = json.load(open("features/metrics.json"))
cards = [
    ("mA", M["mA"], "mean Accuracy\n(balanced, main PAR metric)", "#2e8b57"),
    ("Accuracy", M["Accuracy"], "instance accuracy\n(attributes correct per person)", "#5b9bd5"),
    ("Precision", M["Precision"], "of predicted attributes,\nhow many are right", "#8e44ad"),
    ("Recall", M["Recall"], "of true attributes,\nhow many found", "#e67e22"),
    ("F1", M["F1"], "balance of\nprecision & recall", "#c0392b"),
]

fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
for ax, (name, val, desc, col) in zip(axes, cards):
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8, color=col, alpha=0.12,
                               transform=ax.transAxes, zorder=0))
    ax.text(0.5, 0.72, f"{val:.1f}%", ha="center", va="center", fontsize=30,
            fontweight="bold", color=col, transform=ax.transAxes)
    ax.text(0.5, 0.42, name, ha="center", va="center", fontsize=15,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.22, desc, ha="center", va="center", fontsize=9,
            color="#444", transform=ax.transAxes)

fig.suptitle(f"Validation metrics  —  {M['model']}  (on {M['n_test']:,} held-out PA-100K test images)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("metrics_summary.png", dpi=160, bbox_inches="tight")
print("saved metrics_summary.png")
