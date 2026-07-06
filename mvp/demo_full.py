"""
Step-by-step explainable demo: upload one image -> see what EACH STAGE does.
  Step 1  Plain SigLIP-2  -> the visual feature it extracts (+ text encoder)
  Step 2  CMAA            -> 5 attention heatmaps (where it looks per attribute)
  Step 3  OCFR            -> predicted viewpoint (Front/Side/Back) + routing
  Step 4  DACG            -> 26x26 attribute-correlation heatmap
  Step 5  Prediction      -> the 26 YES/NO

Needs the FULL model `par_full.pt` (from the full-config ablation run).
Run:  python demo_full.py     -> prints a public share link.
"""
import json, os
import numpy as np
import torch
import gradio as gr
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
NAMES = json.load(open("features/attributes.json")) if os.path.exists("features/attributes.json") else \
    ['Female', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Front', 'Side', 'Back', 'Hat', 'Glasses',
     'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 'ShortSleeve', 'LongSleeve',
     'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice', 'LowerStripe', 'LowerPattern',
     'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots']
HEATMAP_ATTRS = ["Hat", "Backpack", "ShortSleeve", "Trousers", "Glasses"]  # no gender/age
AGE = [NAMES.index(a) for a in ["AgeOver60", "Age18-60", "AgeLess18"] if a in NAMES]
VIEW = [NAMES.index(a) for a in ["Front", "Side", "Back"] if a in NAMES]

print(f"[load] {MODEL_ID} + {CKPT} on {DEVICE} ...", flush=True)
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

try:
    _M = json.load(open("features/metrics.json"))
    METRICS_MD = (f"### ✅ Model validated on {_M['n_test']:,} held-out test images\n"
                  f"**mA {_M['mA']:.1f}%**  ·  Accuracy {_M['Accuracy']:.1f}%  ·  "
                  f"Precision {_M['Precision']:.1f}%  ·  Recall {_M['Recall']:.1f}%  ·  F1 {_M['F1']:.1f}%")
except Exception:
    METRICS_MD = ""


def overlay(img224, attn_vec):
    g = int(round(len(attn_vec) ** 0.5))
    hm = attn_vec.reshape(g, g)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    hm = np.array(Image.fromarray((hm * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)) / 255.0
    heat = cm.jet(hm)[..., :3]
    return ((0.55 * (img224 / 255.0) + 0.45 * heat) * 255).astype("uint8")


@torch.no_grad()
def run(image):
    if image is None:
        return None, [], "Upload an image.", None, "—"
    img224 = np.array(square_pad(image).resize((224, 224)))
    px = proc(images=square_pad(image), return_tensors="pt")["pixel_values"].to(DEVICE)
    logits, o_logits, cmaa_attn, dacg_A, pooled = model.forward_explain(px)
    probs = torch.sigmoid(logits)[0].float().cpu().numpy()

    # STEP 1 — plain SigLIP-2: (a) the visual feature vector, (b) plain image<->attribute match
    vec = pooled[0].float().cpu().numpy()
    pv = torch.nn.functional.normalize(pooled.float(), dim=-1)      # image feature
    tv = torch.nn.functional.normalize(model.text_emb.float(), dim=-1)  # 26 attribute texts
    sim = (pv @ tv.t())[0].cpu().numpy()                            # plain SigLIP affinity
    top = np.argsort(-sim)[:10]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.8))
    a1.bar(range(len(vec)), vec, width=1.0, color="#1a73e8")
    a1.set_title(f"Step 1a - SigLIP-2 VISUAL FEATURE: image -> {len(vec)} numbers")
    a1.set_xlabel("feature dimension")
    a2.barh([NAMES[j] for j in top][::-1], sim[top][::-1], color="#8e44ad")
    a2.set_title("Step 1b - PLAIN SigLIP image<->attribute-text match (before any module)")
    fig.tight_layout(); fig.savefig("_feat.png", dpi=110); plt.close(fig)

    # STEP 2 — CMAA heatmaps
    gallery = []
    if cmaa_attn is not None:
        attn = cmaa_attn[0].float().cpu().numpy()
        for a in HEATMAP_ATTRS:
            gallery.append((overlay(img224, attn[NAMES.index(a)]), f"'{a}'"))

    # STEP 3 — OCFR / orientation
    o = torch.softmax(o_logits, -1)[0].cpu().numpy()
    vn = ["Front", "Side", "Back"]
    orient = (f"### Step 3 — OCFR (viewpoint)\n**{vn[int(o.argmax())]}** "
              f"(Front {o[0]*100:.0f}% / Side {o[1]*100:.0f}% / Back {o[2]*100:.0f}%)\n\n"
              f"*OCFR reweights the features for this viewpoint.*")

    # STEP 4 — DACG correlations
    dacg_img = None
    if dacg_A is not None:
        A = dacg_A[0].float().cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(A, cmap="viridis")
        ax.set_title("Step 4 — DACG: attribute correlations (26x26)")
        ax.set_xticks(range(len(NAMES))); ax.set_xticklabels(NAMES, rotation=90, fontsize=5)
        ax.set_yticks(range(len(NAMES))); ax.set_yticklabels(NAMES, fontsize=5)
        fig.colorbar(im); fig.tight_layout()
        fig.savefig("_dacg.png", dpi=110); plt.close(fig)
        dacg_img = "_dacg.png"

    # STEP 5 — final prediction
    pred = probs >= 0.5
    for grp in (AGE, VIEW):
        if grp:
            w = grp[int(np.argmax(probs[grp]))]
            for j in grp:
                pred[j] = (j == w)
    detected = [NAMES[j] for j in range(len(NAMES)) if pred[j]]
    final = "### Step 5 — Final prediction (26 attributes)\n**Detected:** " + \
            (", ".join(detected) if detected else "(none)")
    return "_feat.png", gallery, orient, dacg_img, final


with gr.Blocks(title="Explainable PAR — step by step") as demo:
    gr.Markdown("# Pedestrian Attribute Recognition — what each step does\n"
                "Upload a cropped photo of **one person** and see each stage of the model.")
    if METRICS_MD:
        gr.Markdown(METRICS_MD)
    with gr.Row():
        inp = gr.Image(type="pil", label="Input image")
        btn = gr.Button("Analyze", variant="primary")
    gr.Markdown("## Step 1 — Plain SigLIP-2: feature extraction (image + text encoders)")
    feat_out = gr.Image(label="Extracted visual feature vector")
    gr.Markdown("## Step 2 — CMAA: where the model looks for each attribute")
    gallery = gr.Gallery(label="CMAA attention heatmaps", columns=5, height=240)
    with gr.Row():
        orient_md = gr.Markdown()      # Step 3
        final_md = gr.Markdown()       # Step 5
    gr.Markdown("## Step 4 — DACG: attribute correlations for this image")
    dacg_out = gr.Image(label="DACG 26x26 correlation matrix")
    btn.click(run, inp, [feat_out, gallery, orient_md, dacg_out, final_md])
    inp.upload(run, inp, [feat_out, gallery, orient_md, dacg_out, final_md])

if __name__ == "__main__":
    demo.launch(share=True)
