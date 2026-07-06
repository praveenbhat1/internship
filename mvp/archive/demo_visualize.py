"""Demo: show preprocessing (before/after) and the extracted visual features for one image."""
import numpy as np
import scipy.io as sio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

ANN = "data/PA-100K/annotation/annotation.mat"
IMG_DIR = "data/PA-100K/data/release_data/release_data"
MODEL_ID = "google/siglip2-base-patch16-224"


def unwrap(v):
    while isinstance(v, np.ndarray):
        if v.size == 0:
            return ""
        v = v.flat[0]
    return v


# ---- load annotation, pick a sample ----
m = sio.loadmat(ANN)
attrs = [str(unwrap(a)) for a in m["attributes"]]
names = [str(unwrap(x)) for x in m["test_images_name"]]
labels = np.asarray(m["test_label"])
print("[attributes from .mat]:", attrs)

idx = 1
fname = names[idx]
gt = [attrs[i] for i in range(len(attrs)) if labels[idx, i] == 1]
print(f"\nsample image: {fname}")
print("ground-truth attributes:", gt)

# ---- STEP 1: preprocessing (before / after) ----
orig = Image.open(f"{IMG_DIR}/{fname}").convert("RGB")
s = max(orig.size)
padded = ImageOps.pad(orig, (s, s), color=(0, 0, 0))
after = padded.resize((224, 224))
print(f"\n[preprocess] original {orig.size} -> square-pad -> resize -> {after.size}")

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(orig); ax[0].set_title(f"BEFORE  {orig.size}"); ax[0].axis("off")
ax[1].imshow(after); ax[1].set_title("AFTER  224x224 (pad+resize+normalize)"); ax[1].axis("off")
plt.tight_layout(); plt.savefig("demo_before_after.png", dpi=130); plt.close()

# ---- STEP 2: visual feature extraction ----
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n[features] device = {device}")
proc = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
px = proc(images=padded, return_tensors="pt")["pixel_values"].to(device)
with torch.no_grad():
    out = model.get_image_features(pixel_values=px)
    pooled = out.pooler_output[0]        # (768,)  the pooled visual feature
    patches = out.last_hidden_state[0]   # (196,768) per-region features

vec = pooled.float().cpu().numpy()
print(f"\npooled visual feature shape : {tuple(pooled.shape)}   <- this is 'the visual features'")
print(f"patch-token feature shape   : {tuple(patches.shape)}   <- per-region features (for CMAA later)")
print("first 8 feature numbers     :", [round(float(x), 3) for x in vec[:8]])

plt.figure(figsize=(9, 2.6))
plt.bar(range(len(vec)), vec, width=1.0)
plt.title(f"The extracted visual feature vector ({len(vec)} numbers) for this image")
plt.xlabel("feature dimension (0..767)"); plt.ylabel("value")
plt.tight_layout(); plt.savefig("demo_features.png", dpi=130); plt.close()

print("\nsaved: demo_before_after.png , demo_features.png")
