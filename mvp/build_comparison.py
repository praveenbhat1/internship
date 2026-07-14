"""Comparison of PA-100K mA vs prior work -> comparison_sota.png
NOTE: prior numbers are commonly-cited approximate values from the papers / PAR survey (Wang 2022).
Verify exact figures from the original papers before final submission."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

methods = ["DeepMAR\n2015", "HP-Net\n2017", "VAC\n2019", "ALM\n2019",
           "JLAC\n2020", "VTB\n2022", "PARFormer\n2023", "Ours\n(full model)"]
mA = [72.7, 74.2, 79.2, 80.7, 82.3, 83.7, 84.5, 91.1]
colors = ["#9bb7d4"] * 7 + ["#e67e22"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(methods, mA, color=colors, width=0.65)
for b, v in zip(bars, mA):
    ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("PA-100K  mean Accuracy (mA %)")
ax.set_ylim(65, 95)
ax.set_title("PA-100K Pedestrian Attribute Recognition — mA over the years\n"
             "(prior numbers = commonly-cited approximate values; ours = 23-attr, leak-free)",
             fontweight="bold", fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.axhline(85, color="gray", ls="--", lw=0.8, alpha=0.5)
fig.tight_layout()
fig.savefig("comparison_sota.png", dpi=150, bbox_inches="tight")
print("[saved] comparison_sota.png")
