"""
PROOF SHEET: run N images (>=5) through the trained FULL model and save ONE figure that
proves it works + is explainable. For each image, one row shows:
    [ input | CMAA 'Hat' | CMAA 'Backpack' | CMAA 'Trousers' | predicted attributes ]
The heatmaps show WHERE each attribute looks (CMAA), the text shows the final prediction.

Needs the FULL model `par_full.pt` (+ features/attributes.json) in mvp/ or mvp/features/.
Usage:  python proof_sheet.py img1.jpg img2.jpg img3.jpg img4.jpg img5.jpg
        python proof_sheet.py            # uses any 5 images it finds in ./proof_images/
"""
import json, os, sys, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

from train_par_full import FullPAR, square_pad

MODEL_ID = "google/siglip2-large-patch16-256"
CKPT = "par_full.pt" if os.path.exists("par_full.pt") else "features/par_full.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # CPU on Mac (MPS unstable with SigLIP-2)
NAMES = json.load(open("features/attributes.json")) if os.path.exists("features/attributes.json") \
    else json.load(open("attributes.json"))
SHOW_ATTRS = [a for a in ["Hat", "Backpack", "Trousers"] if a in NAMES]  # 3 heatmap columns

# --- collect images ---
imgs = sys.argv[1:] or sorted(glob.glob("proof_images/*"))[:5]
if len(imgs) < 1:
    raise SystemExit("Give image paths, or put images in ./proof_images/. Need >=5 for a good proof.")

print(f"[load] {MODEL_ID} + {CKPT} on {DEVICE} ({len(NAMES)} attributes) ...", flush=True)
proc = AutoProcessor.from_pretrained(MODEL_ID)
full = AutoModel.from_pretrained(MODEL_ID)
dim = full.vision_model.config.hidden_size
tin = proc(text=[f"a photo of a person, {a}" for a in NAMES],
           padding="max_length", max_length=64, return_tensors="pt")
with torch.no_grad():
    T = full.get_text_features(**tin); T = getattr(T, "pooler_output", T).float()
vision = get_peft_model(full.vision_model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
model = FullPAR(vision, dim, T.shape[1], T, nattr=len(NAMES)).to(DEVICE).eval()
model.load_state_dict(torch.load(CKPT, map_location=DEVICE), strict=False)  # small ckpt: backbone from HF
print("[ready]", flush=True)


def overlay(img224, attn_vec):
    g = int(round(len(attn_vec) ** 0.5))
    hm = attn_vec.reshape(g, g)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    hm = np.array(Image.fromarray((hm * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)) / 255.0
    return ((0.55 * (img224 / 255.0) + 0.45 * cm.jet(hm)[..., :3]) * 255).astype("uint8")


ncol = 2 + len(SHOW_ATTRS)                       # input + heatmaps + text
fig, axes = plt.subplots(len(imgs), ncol, figsize=(3.0 * ncol, 3.2 * len(imgs)))
if len(imgs) == 1:
    axes = axes[None, :]

for r, path in enumerate(imgs):
    pil = Image.open(path)
    img224 = np.array(square_pad(pil).resize((224, 224)))
    px = proc(images=square_pad(pil), return_tensors="pt")["pixel_values"].to(DEVICE)
    with torch.no_grad():
        logits, o_logits, cmaa_attn, _, _ = model.forward_explain(px)
    probs = torch.sigmoid(logits)[0].float().cpu().numpy()
    attn = cmaa_attn[0].float().cpu().numpy()

    axes[r, 0].imshow(img224); axes[r, 0].set_title("input" if r == 0 else "", fontsize=11)
    for c, a in enumerate(SHOW_ATTRS, start=1):
        axes[r, c].imshow(overlay(img224, attn[NAMES.index(a)]))
        axes[r, c].set_title(f"CMAA: {a}" if r == 0 else "", fontsize=11)
    for c in range(1 + len(SHOW_ATTRS)):
        axes[r, c].axis("off")

    # predictions column (top-8 by probability)
    order = np.argsort(-probs)
    detected = [f"{NAMES[j]}  {probs[j]*100:.0f}%" for j in order if probs[j] >= 0.5][:8]
    vn = ["Front", "Side", "Back"]
    view = vn[int(torch.softmax(o_logits, -1)[0].argmax())]
    txt = f"viewpoint: {view}\n\n" + "\n".join(detected) if detected else f"viewpoint: {view}\n(none >50%)"
    axes[r, -1].axis("off")
    axes[r, -1].text(0.0, 0.5, txt, fontsize=10, va="center", family="monospace")
    if r == 0:
        axes[r, -1].set_title("predicted", fontsize=11)

fig.suptitle("Proof — trained model on real images: CMAA localizes each attribute, then predicts",
             fontsize=13, y=0.995)
fig.tight_layout()
fig.savefig("proof_sheet.png", dpi=140, bbox_inches="tight")
print("[saved] proof_sheet.png")
