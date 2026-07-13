"""
Step-by-step explainable demo: upload one image -> see what EACH STAGE does.
  Step 1  Plain SigLIP-2  -> the visual feature it extracts (+ text encoder)
  Step 2  CMAA            -> attention heatmap for EVERY attribute (where it looks)
  Step 3  OCFR            -> predicted viewpoint (Front/Side/Back) + routing
  Step 4  DACG            -> attribute-correlation heatmap
  Step 5  Prediction      -> all attributes YES/NO with confidence

Needs the trained `par_full.pt` + `features/attributes.json` (+ optional `features/thresholds.json`).
Run:  python demo_full.py            (local, http://127.0.0.1:7860)
      python demo_full.py --share    (public temporary link for a remote demo)
"""
import argparse, json, os
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
ATTR_FILE = "features/attributes.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # CPU on Mac (MPS unstable with SigLIP-2)

if not os.path.exists(ATTR_FILE):
    raise SystemExit(f"[!] {ATTR_FILE} not found. Download it together with par_full.pt from Kaggle.")
NAMES = json.load(open(ATTR_FILE))                        # the exact attribute set the model was trained on
N = len(NAMES)

# per-attribute thresholds (calibrated during training); fall back to 0.5 if not present
THRESH = None
if os.path.exists("features/thresholds.json"):
    _t = json.load(open("features/thresholds.json"))
    THRESH = np.array([_t.get(a, 0.5) for a in NAMES], dtype=np.float32)

# mutually-exclusive groups (only those that exist in this attribute set)
def _idx(names):
    return [NAMES.index(a) for a in names if a in NAMES]
AGE    = _idx(["AgeOver60", "Age18-60", "AgeLess18"])
VIEW   = _idx(["Front", "Side", "Back"])
SLEEVE = _idx(["ShortSleeve", "LongSleeve"])
LOWER  = _idx(["Trousers", "Shorts", "Skirt&Dress"])   # lower-body garment: at most one

print(f"[load] {MODEL_ID} + {CKPT} on {DEVICE} | {N} attributes ...", flush=True)
proc = AutoProcessor.from_pretrained(MODEL_ID)
full = AutoModel.from_pretrained(MODEL_ID)
dim = full.vision_model.config.hidden_size
tin = proc(text=[f"a photo of a person, {a}" for a in NAMES],
           padding="max_length", max_length=64, return_tensors="pt")
with torch.no_grad():
    T = full.get_text_features(**tin); T = getattr(T, "pooler_output", T).float()
vision = get_peft_model(full.vision_model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
model = FullPAR(vision, dim, T.shape[1], T, nattr=N).to(DEVICE).eval()
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
    """Resize the per-patch attention to the image size and alpha-blend it on the body."""
    g = int(round(len(attn_vec) ** 0.5))
    hm = attn_vec.reshape(g, g)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    hm = np.array(Image.fromarray((hm * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)) / 255.0
    heat = cm.jet(hm)[..., :3]
    return ((0.5 * (img224 / 255.0) + 0.5 * heat) * 255).astype("uint8")


def apply_exclusive(pred, probs, grp, exactly_one):
    """Fix mutually-exclusive groups. exactly_one=True -> keep the top one (age/viewpoint).
    exactly_one=False -> 'at most one': if >1 fire, keep only the strongest (sleeves)."""
    if not grp:
        return
    firing = [j for j in grp if pred[j]]
    if exactly_one or len(firing) > 1:
        w = grp[int(np.argmax(probs[grp]))]
        for j in grp:
            pred[j] = (j == w)


@torch.no_grad()
def run(image):
    if image is None:
        return None, [], "Upload an image.", None, "—"
    img224 = np.array(square_pad(image).resize((224, 224)))
    px = proc(images=square_pad(image), return_tensors="pt")["pixel_values"].to(DEVICE)
    logits, o_logits, cmaa_attn, dacg_A, pooled = model.forward_explain(px)
    probs = torch.sigmoid(logits)[0].float().cpu().numpy()

    # STEP 1 — plain SigLIP-2: (a) visual feature vector, (b) plain image<->attribute match
    vec = pooled[0].float().cpu().numpy()
    pv = torch.nn.functional.normalize(pooled.float(), dim=-1)
    tv = torch.nn.functional.normalize(model.text_emb.float(), dim=-1)
    sim = (pv @ tv.t())[0].cpu().numpy()
    top = np.argsort(-sim)[:10]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.8))
    a1.bar(range(len(vec)), vec, width=1.0, color="#1a73e8")
    a1.set_title(f"Step 1a - SigLIP-2 VISUAL FEATURE: image -> {len(vec)} numbers")
    a1.set_xlabel("feature dimension")
    a2.barh([NAMES[j] for j in top][::-1], sim[top][::-1], color="#8e44ad")
    a2.set_title("Step 1b - PLAIN SigLIP image<->attribute-text match (before any module)")
    fig.tight_layout(); fig.savefig("_feat.png", dpi=110); plt.close(fig)

    # STEP 5 (compute first so heatmap labels can show YES/NO) — thresholds + mutual exclusion
    thr = THRESH if THRESH is not None else 0.5
    pred = probs >= thr
    apply_exclusive(pred, probs, AGE, True)
    apply_exclusive(pred, probs, VIEW, True)
    apply_exclusive(pred, probs, SLEEVE, False)
    apply_exclusive(pred, probs, LOWER, False)

    # STEP 2 — CMAA heatmap for EVERY attribute (attribute-distinctive: subtract cross-attribute mean)
    gallery = []
    if cmaa_attn is not None:
        attn = cmaa_attn[0].float().cpu().numpy()              # (N, P)
        attn = attn - attn.mean(axis=0, keepdims=True)         # where each attribute looks *relatively*
        order = np.argsort(-probs)                             # detected attributes first
        for j in order:
            tag = "YES" if pred[j] else "no"
            gallery.append((overlay(img224, attn[j]), f"{NAMES[j]} — {probs[j]*100:.0f}% [{tag}]"))

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
        ax.set_title(f"Step 4 — DACG: attribute correlations ({N}x{N})")
        ax.set_xticks(range(N)); ax.set_xticklabels(NAMES, rotation=90, fontsize=5)
        ax.set_yticks(range(N)); ax.set_yticklabels(NAMES, fontsize=5)
        fig.colorbar(im); fig.tight_layout()
        fig.savefig("_dacg.png", dpi=110); plt.close(fig)
        dacg_img = "_dacg.png"

    # STEP 5 — final prediction text (with confidence)
    det = [f"**{NAMES[j]}** ({probs[j]*100:.0f}%)" for j in np.argsort(-probs) if pred[j]]
    final = (f"### Step 5 — Final prediction ({N} attributes)\n**Detected:** " +
             (", ".join(det) if det else "(none above threshold)"))
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
    gr.Markdown("## Step 2 — CMAA: where the model looks for **each** attribute (detected ones first)")
    gallery = gr.Gallery(label="CMAA attention heatmaps (all attributes)", columns=5, height=520)
    with gr.Row():
        orient_md = gr.Markdown()      # Step 3
        final_md = gr.Markdown()       # Step 5
    gr.Markdown(f"## Step 4 — DACG: attribute correlations for this image")
    dacg_out = gr.Image(label=f"DACG {N}x{N} correlation matrix")
    btn.click(run, inp, [feat_out, gallery, orient_md, dacg_out, final_md])
    inp.upload(run, inp, [feat_out, gallery, orient_md, dacg_out, final_md])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="open a public temporary link (default: local only)")
    args = ap.parse_args()
    demo.launch(share=args.share)
