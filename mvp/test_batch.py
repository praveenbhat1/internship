"""
Batch accuracy test: run N images (>=10) through the trained model, show predictions +
confidence, and — for PA-100K test images we have labels for — compute per-image and
aggregate accuracy (mA / Accuracy / Precision / Recall / F1) on that batch.

Needs: features/par_full.pt + attributes.json (+ thresholds.json), and
       features/test_names.npy + test_labels.npy (ground truth, already in the repo).
Images: put PA-100K TEST images in a folder (their filenames match test_names.npy).

Usage:  python test_batch.py --dir test_images
"""
import argparse, json, os, glob
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

from train_par_full import FullPAR, square_pad

MODEL_ID = "google/siglip2-large-patch16-256"
# the 26 original PA-100K attribute names, in test_labels.npy column order (for ground-truth alignment)
ORIG26 = ['Female', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Front', 'Side', 'Back', 'Hat', 'Glasses',
          'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 'ShortSleeve', 'LongSleeve',
          'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice', 'LowerStripe', 'LowerPattern',
          'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots']


def par_metrics(pred, gt):
    e = 1e-12
    tp = ((pred == 1) & (gt == 1)).sum(0); tn = ((pred == 0) & (gt == 0)).sum(0)
    fp = ((pred == 1) & (gt == 0)).sum(0); fn = ((pred == 0) & (gt == 1)).sum(0)
    mA = (0.5 * (tp / (tp + fn + e) + tn / (tn + fp + e))).mean()
    inter = ((pred == 1) & (gt == 1)).sum(1); pc = (pred == 1).sum(1); gc = (gt == 1).sum(1)
    acc = (inter / (pc + gc - inter + e)).mean()
    prec = (inter / (pc + e)).mean(); rec = (inter / (gc + e)).mean()
    return mA, acc, prec, rec, 2 * prec * rec / (prec + rec + e)


def apply_exclusive(pred, probs, grp, exactly_one):
    if not grp:
        return
    firing = [j for j in grp if pred[j]]
    if exactly_one or len(firing) > 1:
        w = grp[int(np.argmax(probs[grp]))]
        for j in grp:
            pred[j] = (j == w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="test_images", help="folder of person images")
    ap.add_argument("--ckpt", default="features/par_full.pt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NAMES = json.load(open("features/attributes.json")); N = len(NAMES)
    THRESH = None
    if os.path.exists("features/thresholds.json"):
        _t = json.load(open("features/thresholds.json"))
        THRESH = np.array([_t.get(a, 0.5) for a in NAMES], dtype=np.float32)
        THRESH = np.maximum(THRESH, 0.5)                     # floor: fewer false positives in display

    def _idx(names): return [NAMES.index(a) for a in names if a in NAMES]
    AGE, VIEW, SLEEVE = _idx(["AgeOver60", "Age18-60", "AgeLess18"]), _idx(["Front", "Side", "Back"]), _idx(["ShortSleeve", "LongSleeve"])
    LOWER = _idx(["Trousers", "Shorts", "Skirt&Dress"])

    # ground truth (optional)
    gt_map = {}
    if os.path.exists("features/test_names.npy") and os.path.exists("features/test_labels.npy"):
        tn = np.load("features/test_names.npy", allow_pickle=True)
        tl = np.load("features/test_labels.npy").astype(int)
        colmap = [ORIG26.index(a) for a in NAMES]                # model attr -> 26-col index
        for name, row in zip(tn, tl):
            gt_map[os.path.basename(str(name))] = row[colmap]    # aligned (N,) ground truth
        print(f"[gt] loaded ground truth for {len(gt_map)} test images")

    print(f"[load] {MODEL_ID} + {args.ckpt} on {device} | {N} attributes ...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    full = AutoModel.from_pretrained(MODEL_ID)
    dim = full.vision_model.config.hidden_size
    tin = proc(text=[f"a photo of a person, {a}" for a in NAMES], padding="max_length", max_length=64, return_tensors="pt")
    with torch.no_grad():
        T = full.get_text_features(**tin); T = getattr(T, "pooler_output", T).float()
    vision = get_peft_model(full.vision_model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    model = FullPAR(vision, dim, T.shape[1], T, nattr=N).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    print("[ready]\n")

    imgs = sorted(glob.glob(os.path.join(args.dir, "*")))
    if len(imgs) < 1:
        raise SystemExit(f"[!] No images in {args.dir}/. Put PA-100K test images there (>=10).")

    all_pred, all_gt, per_img_acc = [], [], []
    for path in imgs:
        fn = os.path.basename(path)
        with torch.no_grad():
            px = proc(images=square_pad(Image.open(path)), return_tensors="pt")["pixel_values"].to(device)
            logits, _ = model(px)
            probs = torch.sigmoid(logits)[0].float().cpu().numpy()
        thr = THRESH if THRESH is not None else 0.5
        pred = (probs >= thr)
        apply_exclusive(pred, probs, AGE, True)
        apply_exclusive(pred, probs, VIEW, True)
        apply_exclusive(pred, probs, SLEEVE, False)
        apply_exclusive(pred, probs, LOWER, False)
        det = [f"{NAMES[j]}({probs[j]*100:.0f}%)" for j in np.argsort(-probs) if pred[j]]
        line = f"{fn:28s} -> " + ", ".join(det[:8])
        if fn in gt_map:
            gt = gt_map[fn].astype(int)
            acc = float((pred.astype(int) == gt).mean())
            per_img_acc.append(acc)
            all_pred.append(pred.astype(int)); all_gt.append(gt)
            miss = [NAMES[j] for j in range(N) if pred[j] != gt[j]]
            line += f"   | attr-acc {acc*100:.0f}%" + (f"  (wrong: {', '.join(miss)})" if miss else "  ✓ all correct")
        print(line)

    if all_pred:
        P, G = np.array(all_pred), np.array(all_gt)
        mA, acc, prec, rec, f1 = par_metrics(P, G)
        print("\n" + "=" * 60)
        print(f"BATCH ACCURACY on {len(all_pred)} labeled images:")
        print(f"  per-attribute correct: {np.mean(per_img_acc)*100:.1f}%")
        print(f"  mA {mA*100:.1f} | Accuracy {acc*100:.1f} | Precision {prec*100:.1f} | "
              f"Recall {rec*100:.1f} | F1 {f1*100:.1f}")
        print("=" * 60)
        print("(small-batch numbers are noisy; the 10k-test metrics in metrics.json are the real validation)")
    else:
        print("\n[note] no ground truth matched these filenames — showing predictions only.")


if __name__ == "__main__":
    main()
