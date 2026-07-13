"""Generate a paper-ready architecture diagram -> architecture.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(9, 12.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis("off")


def box(x, y, w, h, text, fc, fs=11, bold=True):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.08",
                                linewidth=1.5, edgecolor="#333", facecolor=fc))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.4, color="#333"))


G, B, O, P, R, Y = "#d5f0e0", "#d6e4f7", "#ffe6c7", "#f3d9f3", "#f8d0d0", "#fff6c8"
cx = 5
box(cx, 12.3, 3.4, 0.8, "Input: Pedestrian Image", G)
arrow(cx, 11.9, cx, 11.5)
box(cx, 11.1, 4.4, 0.8, "Preprocessing\n(square-pad · resize · normalize)", O, 10)
arrow(cx, 10.7, 3.2, 10.3); arrow(cx, 10.7, 6.8, 10.3)

box(3.2, 9.9, 3.4, 0.9, "SigLIP-2 Image Encoder\n+ LoRA (frozen backbone)", B, 9.5)
box(6.8, 9.9, 3.4, 0.9, "SigLIP-2 Text Encoder\n(23 attribute prompts)", G, 9.5)
arrow(3.2, 9.45, 3.2, 9.05); arrow(6.8, 9.45, 6.8, 9.05)
box(3.2, 8.6, 3.2, 0.8, "Visual Features (1024-d)\n+ patch tokens", B, 9.5)
box(6.8, 8.6, 3.2, 0.8, "23 Attribute Embeddings", G, 9.5)
arrow(3.2, 8.2, 4.4, 7.75); arrow(6.8, 8.2, 5.6, 7.75)

box(cx, 7.3, 5.4, 0.9, "CMAA — Cross-Modal Attribute Attention\n(each attribute attends its region)", O, 10)
arrow(cx, 6.85, cx, 6.5)
box(cx, 6.05, 4.6, 0.8, "Orientation Head (Front / Side / Back)", P, 10)
arrow(cx, 5.65, cx, 5.3)
box(cx, 4.85, 5.2, 0.9, "OCFR — Orientation-Conditioned\nFeature Routing (FiLM)", B, 10)
arrow(cx, 4.4, cx, 4.05)
box(cx, 3.6, 5.2, 0.9, "DACG — Dynamic Attribute Correlation Graph\n(static prior + dynamic + graph conv)", P, 9.5)
arrow(cx, 3.15, cx, 2.8)
box(cx, 2.35, 4.4, 0.8, "Linear Classifier + Sigmoid", R, 10)
arrow(cx, 1.95, cx, 1.6)
box(cx, 1.15, 4.2, 0.8, "23 Predicted Attributes", G, 11)

# loss side-note
ax.text(0.15, 2.35, "Loss:\nweighted BCE\n+ orientation CE\n+ CCLoss\n(consistency)",
        fontsize=8.5, va="center", ha="left", style="italic", color="#555")

ax.set_title("Multimodal PAR — Architecture", fontsize=15, fontweight="bold", pad=14)
fig.savefig("architecture.png", dpi=160, bbox_inches="tight")
print("[saved] architecture.png")
