"""
Web demo — TRAINED PAR model (frozen SigLIP-2 + trained linear head).

Upload a cropped image of one person -> YES / NO for all 26 attributes.
Uses the trained classifier (features/baseline_linear.pt) with per-attribute thresholds
calibrated on the validation set (so it doesn't predict everything as NO like raw zero-shot).

Run:
    python app.py        # then open http://127.0.0.1:7860
"""
import json
import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip2-base-patch16-224"
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

NAMES = json.load(open("features/attributes.json"))          # 26 names, label order
AGE = [NAMES.index(a) for a in ["AgeOver60", "Age18-60", "AgeLess18"] if a in NAMES]
VIEW = [NAMES.index(a) for a in ["Front", "Side", "Back"] if a in NAMES]

print(f"[load] {MODEL_ID} on {DEVICE} ...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
smodel = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
head = nn.Linear(768, 26).to(DEVICE)
head.load_state_dict(torch.load("features/baseline_linear.pt", map_location=DEVICE))
head.eval()


@torch.no_grad()
def _calibrate():
    """Per-attribute threshold that maximizes balanced accuracy on the validation set."""
    Xva = torch.tensor(np.load("features/val_feats.npy").astype(np.float32)).to(DEVICE)
    Yva = np.load("features/val_labels.npy").astype(int)
    p = torch.sigmoid(head(Xva)).cpu().numpy()
    thr = np.full(26, 0.5)
    for j in range(26):
        s, y = p[:, j], Yva[:, j]
        best_t, best = 0.5, -1.0
        for t in np.quantile(s, np.linspace(0.05, 0.95, 19)):
            pr = s >= t
            tp = (pr & (y == 1)).sum(); fn = ((~pr) & (y == 1)).sum()
            tn = ((~pr) & (y == 0)).sum(); fp = (pr & (y == 0)).sum()
            ba = 0.5 * (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9))
            if ba > best:
                best, best_t = ba, t
        thr[j] = best_t
    return thr


THR = _calibrate()
print("[ready]", flush=True)


def square_pad(img):
    s = max(img.size)
    return ImageOps.pad(img.convert("RGB"), (s, s), color=(0, 0, 0))


@torch.no_grad()
def run(image, sensitivity, groups):
    if image is None:
        return [], "Upload an image first."
    px = processor(images=square_pad(image), return_tensors="pt")["pixel_values"].to(DEVICE)
    feat = smodel.get_image_features(pixel_values=px).pooler_output          # (1, 768)
    probs = torch.sigmoid(head(feat))[0].cpu().numpy()                       # (26,)
    pred = probs >= (THR - sensitivity)                                      # higher sens -> more YES
    if groups:
        for grp in (AGE, VIEW):
            if grp:
                win = grp[int(np.argmax(probs[grp]))]
                for j in grp:
                    pred[j] = (j == win)
    rows = [[NAMES[j], "YES" if pred[j] else "NO"] for j in range(26)]
    pos = [NAMES[j] for j in range(26) if pred[j]]
    summary = "### Detected attributes (YES)\n" + (
        ", ".join(f"**{p}**" for p in pos) if pos else "_(none)_")
    return rows, summary


with gr.Blocks(title="PAR — trained model") as demo:
    gr.Markdown("# Pedestrian Attribute Recognition (trained model)\n"
                "Frozen SigLIP-2 + trained classifier. Upload a cropped photo of **one person**.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Pedestrian image")
            sens = gr.Slider(-0.2, 0.2, value=0.0, step=0.02,
                             label="Sensitivity (higher = more YES)")
            grp = gr.Checkbox(value=True, label="Single age + single viewpoint")
            btn = gr.Button("Predict attributes", variant="primary")
        with gr.Column():
            out_md = gr.Markdown()
            out_table = gr.Dataframe(headers=["Attribute", "Answer"],
                                     label="All 26 attributes (YES / NO)",
                                     wrap=True, row_count=26)
    btn.click(run, [inp, sens, grp], [out_table, out_md])
    inp.upload(run, [inp, sens, grp], [out_table, out_md])

if __name__ == "__main__":
    demo.launch()
