"""
Full per-stage extraction figure for the presentation (works WITHOUT the trained checkpoint).
Uses the frozen SigLIP-2 backbone (cached) + real PA-100K label correlations to show, for one image:
  Stage 1  SigLIP  -> visual feature vector + image<->attribute-text match
  Stage 2  CMAA    -> attribute attention heatmaps (where each attribute looks)
  Stage 3  OCFR    -> viewpoint (Front/Side/Back) via image<->viewpoint-text match
  Stage 4  DACG    -> attribute correlation matrix (from PA-100K labels)
Saves: stage_<imagename>.png
Usage: python stage_extraction.py [image.jpg]
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip2-base-patch16-224"
IMG = sys.argv[1] if len(sys.argv) > 1 else "test_crop.jpg"
NAMES = ['Female', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Front', 'Side', 'Back', 'Hat',
         'Glasses', 'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 'ShortSleeve',
         'LongSleeve', 'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice', 'LowerStripe',
         'LowerPattern', 'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots']
HEAT = ["Hat", "Backpack", "ShortSleeve", "Trousers", "Glasses"]


def square_pad(img):
    s = max(img.size)
    return ImageOps.pad(img.convert("RGB"), (s, s), color=(0, 0, 0))


print(f"[load] {MODEL_ID} (cached) | image {IMG} ...", flush=True)
proc = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).eval()

pil = Image.open(IMG)
img224 = np.array(square_pad(pil).resize((224, 224)))
px = proc(images=square_pad(pil), return_tensors="pt")["pixel_values"]
attr_txt = proc(text=[f"a photo of a person, {a}" for a in NAMES],
                padding="max_length", max_length=64, return_tensors="pt")
view_txt = proc(text=["a photo of a person seen from the front",
                      "a photo of a person seen from the side",
                      "a photo of a person seen from the back"],
                padding="max_length", max_length=64, return_tensors="pt")
with torch.no_grad():
    vout = model.vision_model(pixel_values=px)
    patches, pooled = vout.last_hidden_state[0], vout.pooler_output[0]
    T = getattr(model.get_text_features(**attr_txt), "pooler_output", None)
    T = (T if T is not None else model.get_text_features(**attr_txt)).float()
    V = getattr(model.get_text_features(**view_txt), "pooler_output", None)
    V = (V if V is not None else model.get_text_features(**view_txt)).float()

pn = torch.nn.functional.normalize(patches, dim=-1)
tn = torch.nn.functional.normalize(T, dim=-1)
vn = torch.nn.functional.normalize(V, dim=-1)
pooln = torch.nn.functional.normalize(pooled.unsqueeze(0), dim=-1)
img_attr = (pooln @ tn.t())[0].numpy()
patch_attr = (pn @ tn.t()).numpy()
patch_attr = patch_attr - patch_attr.mean(axis=1, keepdims=True)     # attribute-distinctive
vp = (pooln @ vn.t())[0].numpy()
vp = np.exp(vp / 0.02); vp = vp / vp.sum()                           # viewpoint softmax


def heat(vec):
    g = int(round(len(vec) ** 0.5))
    hm = vec.reshape(g, g)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    hm = np.array(Image.fromarray((hm * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)) / 255.0
    return ((0.5 * (img224 / 255.0) + 0.5 * cm.jet(hm)[..., :3]) * 255).astype("uint8")


fig = plt.figure(figsize=(15, 11))
gs = gridspec.GridSpec(3, 5, height_ratios=[1.1, 1, 1.1], hspace=0.4, wspace=0.35)

# Row 0 -- Stage 1
ax = fig.add_subplot(gs[0, 0]); ax.imshow(img224); ax.axis("off")
ax.set_title("Input image", fontsize=12, fontweight="bold")
ax = fig.add_subplot(gs[0, 1:3])
ax.bar(range(len(pooled)), pooled.numpy(), width=1.0, color="#1a73e8")
ax.set_title("Stage 1 - SigLIP visual feature\n(image -> %d numbers)" % len(pooled), fontsize=11)
ax.set_xlabel("feature dimension")
ax = fig.add_subplot(gs[0, 3:5])
top = np.argsort(-img_attr)[:8]
ax.barh([NAMES[j] for j in top][::-1], img_attr[top][::-1], color="#8e44ad")
ax.set_title("Stage 1 - SigLIP image <-> attribute-text match", fontsize=11)

# Row 1 -- Stage 2 CMAA
for c, a in enumerate(HEAT):
    ax = fig.add_subplot(gs[1, c])
    ax.imshow(heat(patch_attr[:, NAMES.index(a)])); ax.axis("off")
    ax.set_title(f"Stage 2 - CMAA\n'{a}'", fontsize=11)

# Row 2 -- Stage 3 OCFR + Stage 4 DACG
ax = fig.add_subplot(gs[2, 0:2])
vn_lbl = ["Front", "Side", "Back"]
ax.bar(vn_lbl, vp * 100, color=["#2e8b57", "#d4a017", "#c0392b"])
ax.set_ylabel("%"); ax.set_ylim(0, 100)
ax.set_title("Stage 3 - OCFR: predicted viewpoint\n(image <-> front/side/back text)", fontsize=11)

ax = fig.add_subplot(gs[2, 2:5])
try:
    lab = np.load("features/train_labels.npy")
    C = np.corrcoef(lab.T)
    im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(NAMES))); ax.set_xticklabels(NAMES, rotation=90, fontsize=5)
    ax.set_yticks(range(len(NAMES))); ax.set_yticklabels(NAMES, fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Stage 4 - DACG: attribute correlations (from PA-100K labels)", fontsize=11)
except Exception as e:
    ax.text(0.5, 0.5, f"(labels not found)\n{e}", ha="center"); ax.axis("off")

fig.suptitle("How the model extracts & reasons, stage by stage  "
             "(SigLIP -> CMAA -> OCFR -> DACG)", fontsize=14, fontweight="bold", y=0.995)
out = f"stage_{os.path.splitext(os.path.basename(IMG))[0]}.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"[saved] {out}")
