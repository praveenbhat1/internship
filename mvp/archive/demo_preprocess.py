"""Show preprocessing (before -> after) for a few sample images, as one grid. No model needed."""
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

ANN = "data/PA-100K/annotation/annotation.mat"
IMG = "data/PA-100K/data/release_data/release_data"


def unwrap(v):
    while isinstance(v, np.ndarray):
        if v.size == 0:
            return ""
        v = v.flat[0]
    return v


names = [str(unwrap(x)) for x in sio.loadmat(ANN)["test_images_name"]]
idxs = [0, 1, 50, 200]                      # 4 sample images

fig, ax = plt.subplots(len(idxs), 2, figsize=(6, 3 * len(idxs)))
for r, i in enumerate(idxs):
    orig = Image.open(f"{IMG}/{names[i]}").convert("RGB")
    s = max(orig.size)
    after = ImageOps.pad(orig, (s, s), color=(0, 0, 0)).resize((224, 224))   # pad -> resize
    ax[r, 0].imshow(orig); ax[r, 0].set_title(f"BEFORE  {orig.size}", fontsize=9); ax[r, 0].axis("off")
    ax[r, 1].imshow(after); ax[r, 1].set_title("AFTER  224x224", fontsize=9); ax[r, 1].axis("off")

plt.tight_layout()
plt.savefig("demo_preprocess_grid.png", dpi=130)
print("saved demo_preprocess_grid.png for:", [names[i] for i in idxs])
