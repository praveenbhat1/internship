"""
Cross-dataset (ZERO-SHOT) validation: PA-100K-trained model -> PETA.
No retraining. We run our 22-attr model on PETA images and score ONLY the attributes that
have a confident correspondence in PETA (name + meaning + polarity match) — this is the
standard, honest way to do cross-domain PAR.

Pipeline: load par_full.pt -> run each PETA image -> map our attributes to PETA columns ->
compute mA / Accuracy / Precision / Recall / F1 on the matched attributes.

Run (on Kaggle where PETA lives, or locally if you have PETA):
  python peta_eval.py --peta_img_dir /path/to/peta/images \
                      --peta_labels  /path/to/peta_labels.csv \
                      --ckpt features/par_full.pt

peta_labels.csv format: one row per image, a column with the image filename + one 0/1 column
per PETA attribute (named as in ATTRIBUTE_MAP below). Adjust the map / column names to match
whatever PETA dataset you use, then re-run.
"""
import argparse, json, os, glob
import numpy as np
import torch
import pandas as pd
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

from train_par_full import FullPAR, square_pad

MODEL_ID = "google/siglip2-large-patch16-256"

# --- Attribute correspondence: OUR PA-100K attribute -> PETA attribute column name ---
# Only include CONFIDENT matches (same meaning + polarity). Comment out anything uncertain.
# PETA column names vary by dataset upload — edit the right-hand side to match YOUR csv headers.
ATTRIBUTE_MAP = {
    "Hat":         "accessoryHat",
    "Glasses":     "accessorySunglasses",     # approx (glasses vs sunglasses)
    "Backpack":    "carryingBackpack",
    "ShoulderBag": "carryingMessengerBag",     # approx
    "ShortSleeve": "upperBodyShortSleeve",
    "LongSleeve":  "upperBodyLongSleeve",       # if present in your csv
    "UpperLogo":   "upperBodyLogo",
    "UpperPlaid":  "upperBodyPlaid",
    "Trousers":    "lowerBodyTrousers",
    "Shorts":      "lowerBodyShorts",
    "Skirt&Dress": "lowerBodyShortSkirt",       # approx
    # Front/Side/Back, HandBag, HoldObjectsInFront, UpperStride/Splice, LowerStripe/Pattern, boots
    # have no clean PETA-35 correspondence -> intentionally excluded.
}


def par_metrics_subset(pred, gt):
    """pred, gt: (N, K) binary arrays over the MATCHED attributes only."""
    e = 1e-12
    tp = ((pred == 1) & (gt == 1)).sum(0); tn = ((pred == 0) & (gt == 0)).sum(0)
    fp = ((pred == 1) & (gt == 0)).sum(0); fn = ((pred == 0) & (gt == 1)).sum(0)
    per_attr_mA = 0.5 * (tp / (tp + fn + e) + tn / (tn + fp + e))
    mA = per_attr_mA.mean()
    inter = ((pred == 1) & (gt == 1)).sum(1); pc = (pred == 1).sum(1); gc = (gt == 1).sum(1)
    acc = (inter / (pc + gc - inter + e)).mean()
    prec = (inter / (pc + e)).mean(); rec = (inter / (gc + e)).mean()
    f1 = 2 * prec * rec / (prec + rec + e)
    return mA, acc, prec, rec, f1, per_attr_mA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peta_img_dir", required=True, help="folder with PETA images")
    ap.add_argument("--peta_labels", required=True, help="CSV: filename col + one 0/1 col per PETA attr")
    ap.add_argument("--ckpt", default="features/par_full.pt")
    ap.add_argument("--attrs", default="features/attributes.json")
    ap.add_argument("--img_col", default="", help="filename column in the csv (auto-detect if empty)")
    ap.add_argument("--thr", type=float, default=0.5, help="decision threshold")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="limit #images for a quick test")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    NAMES = json.load(open(args.attrs))
    print(f"[device] {device} | our model: {len(NAMES)} attributes")

    # --- load labels + resolve which attributes we can actually score ---
    df = pd.read_csv(args.peta_labels)
    img_col = args.img_col or next((c for c in df.columns if df[c].astype(str).str.contains(r"\.(jpg|png|jpeg|bmp)", case=False, regex=True).any()), df.columns[0])
    print(f"[labels] {len(df)} rows | filename column: '{img_col}'")

    matched = [(our, peta) for our, peta in ATTRIBUTE_MAP.items()
               if our in NAMES and peta in df.columns]
    skipped = [our for our, peta in ATTRIBUTE_MAP.items() if peta not in df.columns]
    if not matched:
        raise SystemExit("[!] No attributes matched. Edit ATTRIBUTE_MAP to your PETA csv column names "
                         f"(csv has columns: {list(df.columns)})")
    print(f"[aligned] scoring {len(matched)} attributes: {[m[0] for m in matched]}")
    if skipped:
        print(f"[skipped] not found in csv (adjust names if needed): {skipped}")
    our_idx = [NAMES.index(our) for our, _ in matched]
    peta_cols = [peta for _, peta in matched]

    # --- load model (same as the demo; small checkpoint + strict=False) ---
    print("[model] loading SigLIP-2 + par_full.pt ...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    full = AutoModel.from_pretrained(MODEL_ID)
    dim = full.vision_model.config.hidden_size
    tin = proc(text=[f"a photo of a person, {a}" for a in NAMES], padding="max_length", max_length=64, return_tensors="pt")
    with torch.no_grad():
        T = full.get_text_features(**tin); T = getattr(T, "pooler_output", T).float()
    vision = get_peft_model(full.vision_model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    model = FullPAR(vision, dim, T.shape[1], T, nattr=len(NAMES)).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    print("[model] ready")

    # --- run inference over PETA ---
    rows = df.to_dict("records")
    if args.limit:
        rows = rows[:args.limit]
    preds, gts, done, missing = [], [], 0, 0
    batch_imgs, batch_gt = [], []

    def flush(batch_imgs, batch_gt):
        if not batch_imgs:
            return
        px = proc(images=batch_imgs, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            logits, _ = model(px)
        p = torch.sigmoid(logits)[:, our_idx].float().cpu().numpy()
        preds.append((p >= args.thr).astype(int))
        gts.append(np.array(batch_gt))

    for r in rows:
        fn = str(r[img_col])
        path = os.path.join(args.peta_img_dir, fn)
        if not os.path.exists(path):
            hit = glob.glob(os.path.join(args.peta_img_dir, "**", fn), recursive=True)
            if not hit:
                missing += 1; continue
            path = hit[0]
        try:
            batch_imgs.append(square_pad(Image.open(path)))
            batch_gt.append([int(r[c]) for c in peta_cols])
        except Exception:
            missing += 1; continue
        if len(batch_imgs) == args.batch:
            flush(batch_imgs, batch_gt); done += len(batch_imgs); batch_imgs, batch_gt = [], []
            print(f"  processed {done} images", flush=True)
    flush(batch_imgs, batch_gt); done += len(batch_imgs)

    if not preds:
        raise SystemExit(f"[!] No images processed (missing={missing}). Check --peta_img_dir.")
    P = np.concatenate(preds); G = np.concatenate(gts)
    mA, acc, prec, rec, f1, per_attr = par_metrics_subset(P, G)

    print("\n" + "=" * 60)
    print(f"ZERO-SHOT PA-100K -> PETA  ({done} images, {missing} missing, {len(matched)} attributes)")
    print("=" * 60)
    print(f"  mA {mA*100:.2f} | Accuracy {acc*100:.2f} | Precision {prec*100:.2f} | "
          f"Recall {rec*100:.2f} | F1 {f1*100:.2f}")
    print("\n  per-attribute mA:")
    for (our, _), m in sorted(zip(matched, per_attr), key=lambda x: -x[1]):
        print(f"    {our:15s} {m*100:.1f}")

    out = {"n_images": int(done), "n_attrs": len(matched),
           "attrs": [m[0] for m in matched],
           "mA": round(float(mA*100), 2), "Accuracy": round(float(acc*100), 2),
           "Precision": round(float(prec*100), 2), "Recall": round(float(rec*100), 2),
           "F1": round(float(f1*100), 2),
           "per_attr_mA": {m[0]: round(float(v*100), 1) for m, v in zip(matched, per_attr)}}
    json.dump(out, open("peta_results.json", "w"), indent=2)
    print("\n[saved] peta_results.json")


if __name__ == "__main__":
    main()
