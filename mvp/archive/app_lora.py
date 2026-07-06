"""
Live demo using the TRAINED LoRA model (LoRA large SigLIP-2 + Orientation Head + OCFR, ~91 mA).

Loads features/lora_ocfr.pt (download it from Kaggle -> /kaggle/working/lora_ocfr.pt first).
Upload a person image -> YES/NO for all 26 attributes, from the ~91% model.

Run:  python app_lora.py    # first run downloads siglip2-large (~3.5 GB), then loads your weights
"""
import json
import os
import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

from train_par_kaggle import OCFRModel, square_pad   # reuse the trained model architecture

MODEL_ID = "google/siglip2-large-patch16-256"
CKPT = "features/lora_ocfr.pt"
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# attribute names in the training (label) order
DEFAULT_NAMES = ['Female', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Front', 'Side', 'Back',
                 'Hat', 'Glasses', 'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront',
                 'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo', 'UpperPlaid',
                 'UpperSplice', 'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers',
                 'Shorts', 'Skirt&Dress', 'boots']
NAMES = json.load(open("features/attributes.json")) if os.path.exists("features/attributes.json") else DEFAULT_NAMES
AGE = [NAMES.index(a) for a in ["AgeOver60", "Age18-60", "AgeLess18"] if a in NAMES]
VIEW = [NAMES.index(a) for a in ["Front", "Side", "Back"] if a in NAMES]

if not os.path.exists(CKPT):
    raise SystemExit(
        f"\n[!] Model file '{CKPT}' not found.\n"
        f"    Download lora_ocfr.pt from Kaggle (Output -> /kaggle/working/lora_ocfr.pt)\n"
        f"    into the features/ folder, then re-run this.\n"
        f"    Meanwhile you can run the lighter base demo:  python3 app.py\n")

print(f"[load] {MODEL_ID} on {DEVICE} (first run downloads ~3.5 GB) ...", flush=True)
proc = AutoProcessor.from_pretrained(MODEL_ID)
vision = AutoModel.from_pretrained(MODEL_ID).vision_model
dim = vision.config.hidden_size
vision = get_peft_model(vision, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
model = OCFRModel(vision, dim).to(DEVICE).eval()

ckpt = torch.load(CKPT, map_location=DEVICE)
model.vision.load_state_dict(ckpt["lora"])
model.orient_head.load_state_dict(ckpt["orient"])
model.film.load_state_dict(ckpt["film"])
model.head.load_state_dict(ckpt["head"])
print("[ready] trained LoRA model loaded", flush=True)


@torch.no_grad()
def run(image, sensitivity, groups):
    if image is None:
        return [], "Upload an image first."
    px = proc(images=square_pad(image), return_tensors="pt")["pixel_values"].to(DEVICE)
    logits, _ = model(px)
    probs = torch.sigmoid(logits)[0].float().cpu().numpy()
    pred = probs >= (0.5 - sensitivity)
    if groups:
        for grp in (AGE, VIEW):
            if grp:
                win = grp[int(np.argmax(probs[grp]))]
                for j in grp:
                    pred[j] = (j == win)
    rows = [[NAMES[j], "YES" if pred[j] else "NO"] for j in range(len(NAMES))]
    pos = [NAMES[j] for j in range(len(NAMES)) if pred[j]]
    summary = "### Detected attributes (YES)\n" + (
        ", ".join(f"**{p}**" for p in pos) if pos else "_(none)_")
    return rows, summary


with gr.Blocks(title="PAR — trained LoRA model (~91%)") as demo:
    gr.Markdown("# Pedestrian Attribute Recognition — trained model (~91% mA)\n"
                "LoRA-adapted SigLIP-2 + Orientation Head + OCFR. Upload a cropped photo of **one person**.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Pedestrian image")
            sens = gr.Slider(-0.3, 0.3, value=0.0, step=0.02, label="Sensitivity (higher = more YES)")
            grp = gr.Checkbox(value=True, label="Single age + single viewpoint")
            btn = gr.Button("Predict attributes", variant="primary")
        with gr.Column():
            out_md = gr.Markdown()
            out_table = gr.Dataframe(headers=["Attribute", "Answer"],
                                     label="All 26 attributes (YES / NO)", wrap=True, row_count=26)
    btn.click(run, [inp, sens, grp], [out_table, out_md])
    inp.upload(run, [inp, sens, grp], [out_table, out_md])

if __name__ == "__main__":
    demo.launch()
