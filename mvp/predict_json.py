"""
Headless version of demo_full.py — runs the REAL trained model on sample images
and dumps predictions.json (+ per-image feature / DACG / heatmap PNGs) for the website.

No Gradio. Reuses the exact model, thresholds, mutual-exclusion and gender-abstention
logic from demo_full.py so the website shows genuine model output.

Run:  python predict_json.py --out ../website/public/assets
"""
import argparse, json, os
# Load models from the local HF cache — no internet needed (override with HF_HUB_OFFLINE=0)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
ATTR_FILE = "features/attributes.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENDER_CONF = 0.85

NAMES = json.load(open(ATTR_FILE))
N = len(NAMES)

THRESH = None
if os.path.exists("features/thresholds.json"):
    _t = json.load(open("features/thresholds.json"))
    THRESH = np.maximum(np.array([_t.get(a, 0.5) for a in NAMES], dtype=np.float32), 0.5)


def _idx(names):
    return [NAMES.index(a) for a in names if a in NAMES]


AGE = _idx(["AgeOver60", "Age18-60", "AgeLess18"])
VIEW = _idx(["Front", "Side", "Back"])
SLEEVE = _idx(["ShortSleeve", "LongSleeve"])
LOWER = _idx(["Trousers", "Shorts", "Skirt&Dress"])
FEMALE = NAMES.index("Female") if "Female" in NAMES else -1
BACK = NAMES.index("Back") if "Back" in NAMES else -1


def apply_exclusive(pred, probs, grp, exactly_one):
    if not grp:
        return
    firing = [j for j in grp if pred[j]]
    if exactly_one or len(firing) > 1:
        w = grp[int(np.argmax(probs[grp]))]
        for j in grp:
            pred[j] = (j == w)


def overlay(img224, attn_vec):
    g = int(round(len(attn_vec) ** 0.5))
    hm = attn_vec.reshape(g, g)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    hm = np.array(Image.fromarray((hm * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)) / 255.0
    heat = cm.jet(hm)[..., :3]
    return ((0.5 * (img224 / 255.0) + 0.5 * heat) * 255).astype("uint8")


print(f"[load] {MODEL_ID} + {CKPT} on {DEVICE} | {N} attributes ...", flush=True)
proc = AutoProcessor.from_pretrained(MODEL_ID)
full = AutoModel.from_pretrained(MODEL_ID)
dim = full.vision_model.config.hidden_size
tin = proc(text=[f"a photo of a person, {a}" for a in NAMES],
           padding="max_length", max_length=64, return_tensors="pt")
with torch.no_grad():
    T = full.get_text_features(**tin)
    T = getattr(T, "pooler_output", T).float()
vision = get_peft_model(full.vision_model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
model = FullPAR(vision, dim, T.shape[1], T, nattr=N).to(DEVICE).eval()
model.load_state_dict(torch.load(CKPT, map_location=DEVICE), strict=False)
print("[ready]", flush=True)


@torch.no_grad()
def analyze(path, stem, out_dir, save_stage_imgs):
    image = Image.open(path).convert("RGB")
    img224 = np.array(square_pad(image).resize((224, 224)))
    px = proc(images=square_pad(image), return_tensors="pt")["pixel_values"].to(DEVICE)
    logits, o_logits, cmaa_attn, dacg_A, pooled = model.forward_explain(px)
    probs = torch.sigmoid(logits)[0].float().cpu().numpy()

    o = torch.softmax(o_logits, -1)[0].cpu().numpy()
    ov = int(np.argmax(o))  # 0=Front, 1=Side, 2=Back
    viewpoint = {"Front": round(float(o[0]) * 100), "Side": round(float(o[1]) * 100),
                 "Back": round(float(o[2]) * 100)}

    thr = THRESH if THRESH is not None else 0.5
    pred = probs >= thr
    apply_exclusive(pred, probs, AGE, True)
    if VIEW:  # view attribute matches the authoritative OCFR head
        win = VIEW[ov]
        for j in VIEW:
            pred[j] = (j == win)
    apply_exclusive(pred, probs, SLEEVE, False)
    apply_exclusive(pred, probs, LOWER, False)

    # gender abstention keyed off the OCFR viewpoint
    gender = {"label": None, "conf": None, "reported": False, "why": ""}
    if FEMALE >= 0:
        is_back = ov == 2  # back view -> face not visible -> abstain
        pf = float(probs[FEMALE])
        if is_back or (0.15 < pf < GENDER_CONF):
            pred[FEMALE] = False
            gender.update(reported=False,
                          why="back view (face not visible)" if is_back else f"only {round(pf*100)}% confident")
        elif pf >= GENDER_CONF:
            gender.update(label="Female", conf=round(pf * 100), reported=True)
        else:
            gender.update(label="Male", conf=round((1 - pf) * 100), reported=True)

    order = np.argsort(-probs)
    attrs = [{"name": NAMES[j], "prob": round(float(probs[j]) * 100), "pred": bool(pred[j])}
             for j in order]

    # optional: save the real Step-1 feature plot and Step-4 DACG matrix for THIS image
    if save_stage_imgs:
        vec = pooled[0].float().cpu().numpy()
        pv = torch.nn.functional.normalize(pooled.float(), dim=-1)
        tv = torch.nn.functional.normalize(model.text_emb.float(), dim=-1)
        sim = (pv @ tv.t())[0].cpu().numpy()
        topk = np.argsort(-sim)[:10]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.0))
        a1.bar(range(len(vec)), vec, width=1.0, color="#7c5cff")
        a1.set_title(f"SigLIP-2 visual feature: image -> {len(vec)} numbers")
        a2.barh([NAMES[j] for j in topk][::-1], sim[topk][::-1], color="#20e3b2")
        a2.set_title("Plain SigLIP image<->attribute match (before modules)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "_feat.png"), dpi=110); plt.close(fig)

        if dacg_A is not None:
            A = dacg_A[0].float().cpu().numpy()
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(A, cmap="viridis")
            ax.set_title(f"DACG attribute correlations ({N}x{N})")
            ax.set_xticks(range(N)); ax.set_xticklabels(NAMES, rotation=90, fontsize=5)
            ax.set_yticks(range(N)); ax.set_yticklabels(NAMES, fontsize=5)
            fig.colorbar(im); fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "_dacg.png"), dpi=110); plt.close(fig)

        # a single combined CMAA heatmap sheet for the top detected attributes
        if cmaa_attn is not None:
            attn = cmaa_attn[0].float().cpu().numpy()
            attn = attn - attn.mean(axis=0, keepdims=True)
            show = [j for j in order if pred[j]][:6] or list(order[:6])
            cols = len(show)
            fig, axes = plt.subplots(1, cols, figsize=(2.1 * cols, 2.4))
            if cols == 1:
                axes = [axes]
            for ax, j in zip(axes, show):
                ax.imshow(overlay(img224, attn[j]))
                ax.set_title(f"{NAMES[j]}\n{round(probs[j]*100)}%", fontsize=8)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "_cmaa.png"), dpi=120,
                        bbox_inches="tight"); plt.close(fig)

    return {"image": f"{stem}.jpg", "viewpoint": viewpoint, "gender": gender, "attrs": attrs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../website/public/assets")
    ap.add_argument("--images", nargs="+",
                    default=["test_images/090001.jpg", "test_images/094166.jpg",
                             "test_images/095832.jpg"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = []
    for i, p in enumerate(args.images):
        stem = os.path.splitext(os.path.basename(p))[0]
        print(f"[run] {p}", flush=True)
        # save stage PNGs only for the first (primary) image to keep it light
        results.append(analyze(p, stem, args.out, save_stage_imgs=(i == 0)))

    payload = {"model": "Full 23-attr (SigLIP2 + OCFR + CMAA + DACG + CCLoss)",
               "metrics": json.load(open("features/metrics.json")),
               "results": results}
    with open(os.path.join(args.out, "predictions.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] wrote {os.path.join(args.out, 'predictions.json')}", flush=True)


if __name__ == "__main__":
    main()
