"""Make one combined 'Visual Features' slide figure for the mentor presentation."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageOps

IMG = "data/PA-100K/data/release_data/release_data"
feats = np.load("features/test_feats.npy").astype(np.float32)
labels = np.load("features/test_labels.npy").astype(int)
names = np.load("features/test_names.npy")
attrs = json.load(open("features/attributes.json"))
gi = attrs.index("Female") if "Female" in attrs else 0

idx = 1
img = Image.open(f"{IMG}/{names[idx]}").convert("RGB")
s = max(img.size)
disp = ImageOps.pad(img, (s, s), color=(0, 0, 0)).resize((224, 224))
vec = feats[idx]

# PCA (numpy SVD) on a subset
sub = np.random.RandomState(0).permutation(len(feats))[:2000]
X = feats[sub] - feats[sub].mean(0)
_, _, Vt = np.linalg.svd(X, full_matrices=False)
XY = X @ Vt[:2].T
c = labels[sub, gi]

fig = plt.figure(figsize=(12, 7))
fig.suptitle("Visual Feature Extraction with Frozen SigLIP-2", fontsize=16, fontweight="bold")
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.6], height_ratios=[1, 1],
              hspace=0.35, wspace=0.25)

# (1) preprocessed image
axA = fig.add_subplot(gs[0, 0])
axA.imshow(disp); axA.axis("off")
axA.set_title("1. Preprocessed image (224x224)", fontsize=11)

# (2) feature vector
axB = fig.add_subplot(gs[0, 1])
axB.bar(range(len(vec)), vec, width=1.0, color="#1a73e8")
axB.set_title("2.  -> frozen SigLIP-2 ->  768-number feature vector", fontsize=11)
axB.set_xlabel("feature dimension (0..767)")

# (3) PCA — meaningful
axC = fig.add_subplot(gs[1, 0])
for v, col, lab in [(1, "crimson", "Female"), (0, "steelblue", "Not female")]:
    axC.scatter(XY[c == v, 0], XY[c == v, 1], s=6, alpha=0.5, c=col, label=lab)
axC.legend(fontsize=8); axC.set_xlabel("PC 1"); axC.set_ylabel("PC 2")
axC.set_title("3. Meaningful: features separate by gender (PCA)", fontsize=11)

# (4) stats — useful
axD = fig.add_subplot(gs[1, 1]); axD.axis("off")
txt = ("4.  Features are useful\n\n"
       "- 100,000 images  ->  768 numbers each\n"
       "  (train 80k / val 10k / test 10k)\n\n"
       "- Backbone FROZEN; features cached once\n\n"
       "- Zero-shot baseline ........  69.5 mA\n"
       "- Trained linear head .......  85.5 mA\n\n"
       "=> the features alone enable\n"
       "   near state-of-the-art accuracy")
axD.text(0.0, 0.98, txt, va="top", ha="left", fontsize=11.5, family="monospace",
         bbox=dict(boxstyle="round,pad=0.6", fc="#f6f9ff", ec="#1a73e8", lw=1.5))

plt.savefig("features_overview.png", dpi=150, bbox_inches="tight")
print("saved features_overview.png")
