"""Render the presentation to slides.pdf (16:9) using matplotlib — no external tools."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

W, H = 13.33, 7.5
ACCENT = "#2e3b4e"

SLIDES = [
    {"type": "title", "title": "Multimodal Pedestrian\nAttribute Recognition",
     "subtitle": "with Correlation-Aware Learning",
     "lines": ["Interpretable, viewpoint-aware PAR on SigLIP-2 + LoRA",
               "with a zero-shot cross-dataset generalization study", "", "Internship Project"]},
    {"type": "bullets", "title": "The Problem",
     "bullets": ["Predict a person's attributes (clothing, accessories, viewpoint) from one cropped image",
                 "Multi-label — each attribute gets its own yes/no (sigmoid)",
                 "Used in surveillance, retail analytics, person retrieval",
                 "Describes people by attributes without storing identity",
                 "Hard: occlusion, viewpoint, low-res crops, class imbalance"]},
    {"type": "image", "title": "Our Architecture", "image": "mvp/architecture.png",
     "caption": "Frozen SigLIP-2 + LoRA  ->  CMAA  ->  OCFR  ->  DACG  ->  classifier", "iw": 0.42},
    {"type": "bullets", "title": "The Four Modules",
     "bullets": ["CMAA — each attribute attends to its own image region (interpretable heatmaps)",
                 "OCFR — predicts viewpoint (Front/Side/Back) and reweights features",
                 "DACG — 23x23 attribute-correlation graph; predictions reinforce each other",
                 "CCLoss — enforces logical consistency (one viewpoint, one sleeve length)",
                 "LoRA fine-tuning (~1% of params) is the main accuracy driver"]},
    {"type": "bullets", "title": "Datasets & Honest Evaluation",
     "bullets": ["PA-100K — 100,000 images (train + 10k test)",
                 "PETA — 19,000 images, used ZERO-SHOT for cross-dataset validation",
                 "Age removed (near chance); 23-attribute final model",
                 "Leak-free protocol: thresholds + best epoch chosen on VALIDATION,",
                 "     reported once on TEST  (we found & fixed a data-leakage bug: val 91.9 vs test 91.1)"]},
    {"type": "image", "title": "Results — Accuracy Progression", "image": "mvp/accuracy_progression.png",
     "caption": "Zero-shot 69.5  ->  trained head 85.5  ->  LoRA + modules ~91", "iw": 0.6},
    {"type": "image", "title": "Ablation — Each Module's Contribution", "image": "mvp/ablation_chart.png",
     "caption": "mA flat (backbone dominates); Accuracy & F1 peak at +CMAA+OCFR", "iw": 0.85},
    {"type": "image", "title": "Final Model — Validation Metrics", "image": "mvp/metrics_summary.png",
     "caption": "91.12 mA on 10,000 held-out test images (leak-free)", "iw": 0.92},
    {"type": "image", "title": "Interpretability — What Each Stage Does", "image": "mvp/stage_test_crop.png",
     "caption": "CMAA heatmaps show WHERE the model looks per attribute -> trustworthy", "iw": 0.8},
    {"type": "image", "title": "Cross-Dataset Generalization (PETA)", "image": "mvp/peta_comparison.png",
     "caption": "Same model, no retraining -> 77.8 mA on PETA (14,437 imgs). Gap expected; well above chance.", "iw": 0.5},
    {"type": "image", "title": "Cross-Dataset — Real Examples", "image": "mvp/peta_examples.png",
     "caption": "Model prediction vs PETA ground truth on unseen images (79-100% per image)", "iw": 0.34},
    {"type": "bullets", "title": "Novelty & Responsible Design",
     "bullets": ["1. Orientation-aware routing (OCFR)",
                 "2. Attribute-correlation graph (DACG)",
                 "3. Consistency loss (CCLoss)",
                 "4. Cross-dataset generalization study  (strongest novelty)",
                 "5. Gender abstention — reports gender only when confident + face visible"]},
    {"type": "bullets", "title": "Honest Limitations",
     "bullets": ["~91% mA — not perfect; errors on fine textures (UpperPlaid) + small accessories (Glasses)",
                 "Gender is appearance-based & unreliable -> abstention policy",
                 "Modules add interpretability + small F1 more than raw mA",
                 "Cross-dataset gap on attributes with different definitions (Trousers)",
                 "Reported transparently — rigor is a strength"]},
    {"type": "bullets", "title": "Conclusion",
     "bullets": ["91.12 mA in-domain (SOTA-level, leak-free)",
                 "77.81 mA zero-shot on PETA — proves generalization",
                 "Interpretable, viewpoint-aware, responsible design",
                 "Full ablation + honest evaluation",
                 "The strength is the complete rigorous package, not one number",
                 "Repo: github.com/praveenbhat1/internship"]},
    {"type": "title", "title": "Thank You", "subtitle": "",
     "lines": ["Live demo:  python3 mvp/demo_full.py", "", "Questions?"]},
]


def render(s, pdf):
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if s["type"] == "title":
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=ACCENT))
        ax.text(0.5, 0.66, s["title"], color="white", fontsize=34, fontweight="bold", ha="center", va="center")
        if s.get("subtitle"):
            ax.text(0.5, 0.5, s["subtitle"], color="#cfe3ff", fontsize=20, ha="center", va="center")
        for i, ln in enumerate(s.get("lines", [])):
            ax.text(0.5, 0.36 - i * 0.06, ln, color="#e8eef5", fontsize=13, ha="center", va="center")
    else:
        ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, color=ACCENT))
        ax.text(0.05, 0.94, s["title"], color="white", fontsize=22, fontweight="bold", va="center")
        if s["type"] == "bullets":
            for i, b in enumerate(s["bullets"]):
                ax.text(0.07, 0.78 - i * 0.11, "•  " + b, fontsize=15, va="top", wrap=True)
        elif s["type"] == "image":
            try:
                img = mpimg.imread(s["image"]); ih, iw = img.shape[0], img.shape[1]
                w = s.get("iw", 0.6); h = w * (ih / iw) * (W / H)
                h = min(h, 0.66); w = h * (iw / ih) * (H / W)
                iax = fig.add_axes([0.5 - w/2, 0.14, w, h]); iax.imshow(img); iax.axis("off")
            except Exception as e:
                ax.text(0.5, 0.5, f"[image: {s['image']}]\n{e}", ha="center")
            ax.text(0.5, 0.07, s.get("caption", ""), fontsize=13, ha="center", style="italic", color="#444")
    pdf.savefig(fig); plt.close(fig)


with PdfPages("slides.pdf") as pdf:
    for s in SLIDES:
        render(s, pdf)
print(f"[saved] slides.pdf ({len(SLIDES)} slides)")
