"""Orientation Head + OCFR routing flowchart -> ocfr_flowchart.png (paper-ready)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(8.5, 11))
ax.set_xlim(0, 10); ax.set_ylim(0, 13.5); ax.axis("off")

B, P, O, G, GR = "#d6e4f7", "#f3d9f3", "#ffe6c7", "#d5f0e0", "#eeeeee"


def box(x, y, w, h, text, fc, fs=10.5, bold=True):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.08",
                                lw=1.5, edgecolor="#333", facecolor=fc))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2, txt="", lbl_dx=0.25):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, lw=1.4, color="#333"))
    if txt:
        ax.text((x1+x2)/2 + lbl_dx, (y1+y2)/2, txt, fontsize=8.5, style="italic", color="#555")


# two inputs
box(2.4, 12.7, 3.2, 0.85, "Pooled Visual\nFeature  $p$", B, 10)
box(7.3, 12.7, 3.6, 0.85, "Attribute-aware Features\n$F_{attr}$  (from CMAA)", G, 9.5)

# orientation head path (left)
arrow(2.4, 12.25, 2.4, 11.55)
box(2.4, 11.1, 3.4, 0.85, "Orientation Head\n(Linear)  $o = W_o p + b_o$", P, 9)
arrow(2.4, 10.65, 2.4, 9.95)
box(2.4, 9.5, 3.0, 0.8, "Softmax\n$p_v = \\mathrm{softmax}(o)$", O, 9)
arrow(2.4, 9.1, 2.4, 8.45)

# three viewpoint branches
for cx, lab in [(1.0, "Front"), (2.4, "Side"), (3.8, "Back")]:
    box(cx, 8.0, 1.15, 0.65, lab, GR, 9.5)
    ax.add_patch(FancyArrowPatch((2.4, 8.45), (cx, 8.35), arrowstyle="-|>", mutation_scale=10, lw=1.1, color="#666"))
    ax.add_patch(FancyArrowPatch((cx, 7.65), (2.4, 7.15), arrowstyle="-|>", mutation_scale=10, lw=1.1, color="#666"))

box(2.4, 6.7, 3.6, 0.85, "FiLM Generator\n$[\\gamma,\\beta]=W_f p_v + b_f$", B, 9)

# routing (merge)
arrow(2.4, 6.25, 3.6, 5.35, "$\\gamma,\\beta$", -0.5)
arrow(7.3, 12.25, 6.2, 5.35, "$F_{attr}$", 0.35)
box(5.0, 4.9, 6.2, 1.0, "Feature Routing (FiLM Modulation)\n$F' = F_{attr}\\odot(1+\\gamma)+\\beta$", O, 10)
arrow(5.0, 4.4, 5.0, 3.65)
box(5.0, 3.2, 4.6, 0.85, "Orientation-aware\nFeatures  $F'$", G, 10.5)
arrow(5.0, 2.77, 5.0, 2.15)
box(5.0, 1.7, 4.2, 0.8, "to Hybrid DACG", GR, 9.5, bold=False)

ax.set_title("Orientation Head + OCFR (Orientation-Conditioned Feature Routing)",
             fontsize=12.5, fontweight="bold", pad=12)
fig.savefig("ocfr_flowchart.png", dpi=170, bbox_inches="tight")
print("[saved] ocfr_flowchart.png")
