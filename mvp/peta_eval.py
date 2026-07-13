"""
Cross-dataset (ZERO-SHOT) validation: PA-100K-trained model -> PETA. No retraining.
Parses PETA's per-subdataset Label.txt (each line: "<id> <attr> <attr> ...") and scores only
the attributes with a confident PA-100K<->PETA correspondence.

Run on Kaggle (PETA + par_full.pt there):
  !python peta_eval.py \
      --peta_root "/kaggle/input/datasets/ayushk1409/peta-dataset/PETA dataset" \
      --ckpt /kaggle/working/par_full.pt --attrs /kaggle/working/attributes.json \
      --thresholds /kaggle/working/thresholds.json
"""
import argparse, json, os, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

from train_par_full import FullPAR, square_pad

MODEL_ID = "google/siglip2-large-patch16-256"

# our PA-100K attribute -> list of PETA attribute names (GT positive if ANY are present)
ATTRIBUTE_MAP = {
    "Female":      ["personalFemale"],
    "Hat":         ["accessoryHat"],
    "Glasses":     ["accessorySunglasses"],
    "Backpack":    ["carryingBackpack"],
    "ShoulderBag": ["carryingMessengerBag"],
    "ShortSleeve": ["upperBodyShortSleeve"],
    "LongSleeve":  ["upperBodyLongSleeve"],
    "UpperLogo":   ["upperBodyLogo"],
    "UpperPlaid":  ["upperBodyPlaid"],
    "UpperStride": ["upperBodyThinStripes", "upperBodyThickStripes"],
    "Trousers":    ["lowerBodyTrousers"],
    "Shorts":      ["lowerBodyShorts"],
    "Skirt&Dress": ["lowerBodyShortSkirt", "lowerBodyLongSkirt"],
    "boots":       ["footwearBoots"],
}
IMG_EXT = ("*.png", "*.bmp", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.BMP")


def par_metrics(pred, gt):
    e = 1e-12
    tp = ((pred == 1) & (gt == 1)).sum(0); tn = ((pred == 0) & (gt == 0)).sum(0)
    fp = ((pred == 1) & (gt == 0)).sum(0); fn = ((pred == 0) & (gt == 1)).sum(0)
    per = 0.5 * (tp / (tp + fn + e) + tn / (tn + fp + e))
    inter = ((pred == 1) & (gt == 1)).sum(1); pc = (pred == 1).sum(1); gc = (gt == 1).sum(1)
    acc = (inter / (pc + gc - inter + e)).mean()
    prec = (inter / (pc + e)).mean(); rec = (inter / (gc + e)).mean()
    return per.mean(), acc, prec, rec, 2 * prec * rec / (prec + rec + e), per


def load_peta(root):
    """Return [(image_path, set_of_positive_peta_attrs)] across all sub-datasets."""
    samples = []
    for lab in sorted(glob.glob(os.path.join(root, "*", "archive", "Label.txt"))):
        subroot = os.path.dirname(os.path.dirname(lab))          # .../3DPeS
        lm = {}
        for line in open(lab).read().strip().split("\n"):
            t = line.split()
            if len(t) < 2:
                continue
            s = set(t[1:]); lm[t[0]] = s
            if t[0].isdigit():
                lm[str(int(t[0]))] = s                           # int-normalized (drop leading zeros)
        imgs = []
        for ext in IMG_EXT:
            imgs += glob.glob(os.path.join(subroot, "**", ext), recursive=True)
        n0 = len(samples)
        for p in imgs:
            stem = os.path.splitext(os.path.basename(p))[0]
            for cand in (stem, stem.split("_")[0], stem.split("_")[0].lstrip("0") or "0"):
                if cand in lm:
                    samples.append((p, lm[cand])); break
        print(f"  {os.path.basename(subroot):14s} {len(imgs):5d} imgs -> {len(samples)-n0:5d} matched labels")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peta_root", required=True)
    ap.add_argument("--ckpt", default="features/par_full.pt")
    ap.add_argument("--attrs", default="features/attributes.json")
    ap.add_argument("--thresholds", default="features/thresholds.json")
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (horizontal-flip averaging)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NAMES = json.load(open(args.attrs))
    matched = [(our, peta) for our, peta in ATTRIBUTE_MAP.items() if our in NAMES]
    our_idx = [NAMES.index(our) for our, _ in matched]
    print(f"[align] scoring {len(matched)} shared attributes: {[m[0] for m in matched]}")

    thr = np.full(len(matched), 0.5, dtype=np.float32)
    if os.path.exists(args.thresholds):
        _t = json.load(open(args.thresholds))
        thr = np.array([_t.get(our, 0.5) for our, _ in matched], dtype=np.float32)

    print("[peta] scanning sub-datasets ...")
    samples = load_peta(args.peta_root)
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        raise SystemExit("[!] No PETA images matched labels — check --peta_root path.")
    print(f"[peta] total {len(samples)} labeled images")

    print("[model] loading SigLIP-2 + checkpoint ...", flush=True)
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
    print("[model] ready\n")

    MN = [m[0] for m in matched]
    preds, gts, done, examples = [], [], 0, []
    for i in range(0, len(samples), args.batch):
        chunk = samples[i:i + args.batch]
        pil, gt_rows = [], []
        for path, attrset in chunk:
            try:
                pil.append(square_pad(Image.open(path)))
            except Exception:
                continue
            gt_rows.append([1 if any(pa in attrset for pa in peta) else 0 for _, peta in matched])
        if not pil:
            continue
        px = proc(images=pil, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            logits, _ = model(px)
            if args.tta:                                     # average with horizontal flip (no labels used)
                logits = (logits + model(torch.flip(px, dims=[3]))[0]) / 2
        p = torch.sigmoid(logits)[:, our_idx].float().cpu().numpy()
        pr = (p >= thr).astype(int); gr = np.array(gt_rows)
        preds.append(pr); gts.append(gr); done += len(pil)
        for k in range(len(pil)):                            # keep a few examples for the figure
            if len(examples) < 5:
                examples.append((pil[k], pr[k], gr[k]))
        if i % (args.batch * 20) == 0:
            print(f"  processed {done}/{len(samples)}", flush=True)

    # --- example figure: model prediction vs PETA ground truth ---
    if examples:
        fig, axes = plt.subplots(len(examples), 2, figsize=(9, 2.7 * len(examples)),
                                 gridspec_kw={"width_ratios": [1, 2.2]})
        if len(examples) == 1:
            axes = axes[None, :]
        for r, (pil, pr, gr) in enumerate(examples):
            axes[r, 0].imshow(pil.resize((150, 300))); axes[r, 0].axis("off")
            axes[r, 1].axis("off")
            rows = []
            for j, name in enumerate(MN):
                if pr[j] or gr[j]:
                    mark = "OK " if pr[j] == gr[j] else "X  "
                    rows.append(f"{mark}{name:12s} model={'Yes' if pr[j] else 'No':3s}  PETA={'Yes' if gr[j] else 'No'}")
            acc = (pr == gr).mean() * 100
            axes[r, 1].text(0, 0.5, f"shared-attribute accuracy: {acc:.0f}%\n\n" + "\n".join(rows),
                            fontsize=8.5, va="center", family="monospace")
        fig.suptitle("Zero-shot on PETA (unseen dataset): model prediction vs PETA ground truth",
                     fontweight="bold", y=1.0)
        fig.tight_layout()
        fig.savefig("peta_examples.png", dpi=140, bbox_inches="tight")
        print("[saved] peta_examples.png")

    P, G = np.concatenate(preds), np.concatenate(gts)
    mA, acc, prec, rec, f1, per = par_metrics(P, G)
    print("\n" + "=" * 62)
    print(f"ZERO-SHOT  PA-100K -> PETA   ({done} images, {len(matched)} shared attributes)")
    print("=" * 62)
    print(f"  mA {mA*100:.2f} | Accuracy {acc*100:.2f} | Precision {prec*100:.2f} | "
          f"Recall {rec*100:.2f} | F1 {f1*100:.2f}")
    print("\n  per-attribute mA:")
    for (our, _), m in sorted(zip(matched, per), key=lambda x: -x[1]):
        print(f"    {our:14s} {m*100:.1f}")
    out = {"n_images": int(done), "n_attrs": len(matched), "attrs": [m[0] for m in matched],
           "mA": round(float(mA*100), 2), "Accuracy": round(float(acc*100), 2),
           "Precision": round(float(prec*100), 2), "Recall": round(float(rec*100), 2), "F1": round(float(f1*100), 2),
           "per_attr_mA": {m[0]: round(float(v*100), 1) for m, v in zip(matched, per)}}
    json.dump(out, open("peta_results.json", "w"), indent=2)
    print("\n[saved] peta_results.json")


if __name__ == "__main__":
    main()
