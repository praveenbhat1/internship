"""
FastAPI backend that serves the REAL trained PAR model to the website.

Loads the model once, then answers POST /predict with:
  - viewpoint (OCFR softmax), gender decision (with abstention), all 23 attributes
  - base64 PNGs for the SigLIP feature plot, CMAA heatmap sheet, and DACG matrix
so the website can reproduce demo_full.py's output for any uploaded image.

Run:  cd mvp && python serve.py         (http://127.0.0.1:8000)
Docs: http://127.0.0.1:8000/docs
"""
import base64, io, json, os
# Use the local HF cache when the backbone is already downloaded (no internet needed).
# On a fresh machine (not cached) it stays online so the model can download the first time.
_hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
if os.path.isdir(os.path.join(_hf_home, "hub", "models--google--siglip2-large-patch16-256")):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
try:  # optional: lets iPhone HEIC/HEIF uploads work
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
try:
    METRICS = json.load(open("features/metrics.json"))
except Exception:
    METRICS = {}


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


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="#111208")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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

app = FastAPI(title="PAR model API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# olive/dark styling for the generated matplotlib figures
OLIVE = "#a8b14b"
plt.rcParams.update({
    "text.color": "#e9e9de", "axes.labelcolor": "#c9caba",
    "xtick.color": "#c9caba", "ytick.color": "#c9caba",
    "axes.edgecolor": "#3a3d28", "axes.facecolor": "#111208",
    "figure.facecolor": "#111208",
})


@app.get("/health")
def health():
    return {"status": "ok", "attributes": N, "metrics": METRICS}


@torch.no_grad()
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Couldn't read this image. Please upload a JPG or PNG."},
        )
    img224 = np.array(square_pad(image).resize((224, 224)))
    px = proc(images=square_pad(image), return_tensors="pt")["pixel_values"].to(DEVICE)
    logits, o_logits, cmaa_attn, dacg_A, pooled = model.forward_explain(px)
    probs = torch.sigmoid(logits)[0].float().cpu().numpy()

    # OCFR viewpoint is the authoritative orientation head — use it as the single
    # source of truth for the displayed viewpoint, the detected view attribute, and gender.
    o = torch.softmax(o_logits, -1)[0].cpu().numpy()
    ov = int(np.argmax(o))  # 0=Front, 1=Side, 2=Back
    viewpoint = {"Front": round(float(o[0]) * 100), "Side": round(float(o[1]) * 100),
                 "Back": round(float(o[2]) * 100)}

    thr = THRESH if THRESH is not None else 0.5
    pred = probs >= thr
    apply_exclusive(pred, probs, AGE, True)
    if VIEW:  # force the view attribute to match OCFR so nothing contradicts the viewpoint
        win = VIEW[ov]
        for j in VIEW:
            pred[j] = (j == win)
    apply_exclusive(pred, probs, SLEEVE, False)
    apply_exclusive(pred, probs, LOWER, False)

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

    # Step 1 — SigLIP feature + plain image<->attribute match
    vec = pooled[0].float().cpu().numpy()
    pv = torch.nn.functional.normalize(pooled.float(), dim=-1)
    tv = torch.nn.functional.normalize(model.text_emb.float(), dim=-1)
    sim = (pv @ tv.t())[0].cpu().numpy()
    topk = np.argsort(-sim)[:10]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4.6))
    a1.bar(range(len(vec)), vec, width=1.0, color=OLIVE)
    a1.set_title(f"SigLIP-2 visual feature ({len(vec)} dims)")
    a2.barh([NAMES[j] for j in topk][::-1], sim[topk][::-1], color="#c8d17a")
    a2.set_title("Plain image <-> attribute match")
    fig.tight_layout()
    feat_png = fig_to_b64(fig)

    # Step 2 — CMAA heatmaps for top attributes
    cmaa_png = None
    if cmaa_attn is not None:
        attn = cmaa_attn[0].float().cpu().numpy()
        attn = attn - attn.mean(axis=0, keepdims=True)
        show = [j for j in order if pred[j]][:6] or list(order[:6])
        cols = len(show)
        fig, axes = plt.subplots(1, cols, figsize=(2.0 * cols, 2.3))
        if cols == 1:
            axes = [axes]
        for ax, j in zip(axes, show):
            ax.imshow(overlay(img224, attn[j]))
            ax.set_title(f"{NAMES[j]}\n{round(probs[j]*100)}%", fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        cmaa_png = fig_to_b64(fig)

    # Step 4 — DACG correlation matrix
    dacg_png = None
    if dacg_A is not None:
        A = dacg_A[0].float().cpu().numpy()
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        im = ax.imshow(A, cmap="cividis")
        ax.set_title(f"DACG correlations ({N}x{N})")
        ax.set_xticks(range(N)); ax.set_xticklabels(NAMES, rotation=90, fontsize=5)
        ax.set_yticks(range(N)); ax.set_yticklabels(NAMES, fontsize=5)
        fig.colorbar(im)
        fig.tight_layout()
        dacg_png = fig_to_b64(fig)

    return {"viewpoint": viewpoint, "gender": gender, "attrs": attrs,
            "images": {"feature": feat_png, "cmaa": cmaa_png, "dacg": dacg_png}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
