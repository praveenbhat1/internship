"""
Zero-shot baseline evaluation + feature visualization for the SigLIP-2 MVP on PA-100K.

Uses the CACHED features from extract_features.py (fast — no image re-encoding):
  - computes zero-shot attribute probabilities via SigLIP-2 image-text similarity,
  - reports the 5 PAR metrics (mA + instance Accuracy/Precision/Recall/F1) — your BASELINE,
  - saves a PCA scatter of features colored by gender (features/feature_pca.png) — proof the
    features are meaningful/separable.

Run (after extract_features.py has produced features/):
  python evaluate_zeroshot.py --split test
  python evaluate_zeroshot.py --split test --groups        # enforce single age + viewpoint
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoProcessor

from attributes import ATTRIBUTES, NAMES, PROMPTS, AGE_GROUP, VIEW_GROUP

MODEL_ID = "google/siglip2-base-patch16-224"
clean = lambda s: "".join(c for c in str(s).lower() if c.isalnum())


def align_prompts(mat_attrs):
    """Match each .mat attribute (label column) to its prompt, so prompts line up with labels."""
    by_clean = {clean(n): p for n, p in ATTRIBUTES}
    prompts = []
    for i, a in enumerate(mat_attrs):
        if clean(a) in by_clean:
            prompts.append(by_clean[clean(a)])
        else:
            prompts.append(PROMPTS[i] if i < len(PROMPTS) else f"a photo of a person, {a}")
            print(f"[warn] no prompt match for .mat attr '{a}' -> using fallback")
    return prompts


@torch.no_grad()
def get_text_embeds(prompts, model, processor, device):
    inp = processor(text=prompts, padding="max_length", max_length=64,
                    return_tensors="pt").to(device)
    t = model.get_text_features(**inp)
    t = getattr(t, "pooler_output", t)                                 # handle output object
    return F.normalize(t, dim=-1)                                      # (A, D)


def calibrate_thresholds(scores, labels):
    """Per-attribute threshold that maximizes balanced accuracy (zero-shot upper bound)."""
    A = scores.shape[1]
    pred = np.zeros_like(scores, dtype=np.int64)
    thr = np.zeros(A)
    for j in range(A):
        s, y = scores[:, j], labels[:, j]
        cands = np.quantile(s, np.linspace(0.02, 0.98, 97))
        best_t, best_ba = 0.5, -1.0
        for t in cands:
            p = s >= t
            tp = (p & (y == 1)).sum(); fn = ((~p) & (y == 1)).sum()
            tn = ((~p) & (y == 0)).sum(); fp = (p & (y == 0)).sum()
            ba = 0.5 * (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9))
            if ba > best_ba:
                best_ba, best_t = ba, t
        thr[j], pred[:, j] = best_t, (s >= best_t)
    return pred, thr


def par_metrics(pred, gt):
    """pred, gt: (N, A) in {0,1}. Returns mA, per-attr mA, instance acc/prec/rec/f1."""
    eps = 1e-12
    tp = ((pred == 1) & (gt == 1)).sum(0)
    tn = ((pred == 0) & (gt == 0)).sum(0)
    fp = ((pred == 1) & (gt == 0)).sum(0)
    fn = ((pred == 0) & (gt == 1)).sum(0)
    mA_per = 0.5 * (tp / (tp + fn + eps) + tn / (tn + fp + eps))
    inter = ((pred == 1) & (gt == 1)).sum(1)
    p_cnt = (pred == 1).sum(1)
    g_cnt = (gt == 1).sum(1)
    acc = (inter / (p_cnt + g_cnt - inter + eps)).mean()
    prec = (inter / (p_cnt + eps)).mean()
    rec = (inter / (g_cnt + eps)).mean()
    f1 = 2 * prec * rec / (prec + rec + eps)
    return mA_per.mean(), mA_per, acc, prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--fixed", action="store_true",
                    help="use fixed --threshold instead of per-attribute calibration")
    ap.add_argument("--groups", action="store_true")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    fp = lambda name: os.path.join(args.features, name)
    if not os.path.exists(fp(f"{args.split}_feats.npy")):
        raise SystemExit(f"Run extract_features.py first — {fp(args.split + '_feats.npy')} not found")

    feats = np.load(fp(f"{args.split}_feats.npy")).astype(np.float32)
    labels = np.load(fp(f"{args.split}_labels.npy")).astype(np.int64)
    mat_attrs = json.load(open(fp("attributes.json")))
    print(f"[data] {feats.shape[0]} images, {labels.shape[1]} attributes")

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[load] {args.model} on {device}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    # zero-shot probabilities from cached image features + text prompts
    txt = get_text_embeds(align_prompts(mat_attrs), model, processor, device)   # (A, D)
    img = F.normalize(torch.tensor(feats, device=device), dim=-1)               # (N, D)
    with torch.no_grad():
        logits = model.logit_scale.exp() * img @ txt.t() + model.logit_bias
        probs = torch.sigmoid(logits).cpu().numpy()

    # mutually-exclusive groups: keep only the top-1 within age and within viewpoint
    name_idx = {clean(a): i for i, a in enumerate(mat_attrs)}
    eff = probs.copy()
    if args.groups:
        for grp in (AGE_GROUP, VIEW_GROUP):
            idx = [name_idx[clean(g)] for g in grp if clean(g) in name_idx]
            if len(idx) > 1:
                winner = np.array(idx)[probs[:, idx].argmax(1)]
                for j in idx:
                    eff[winner != j, j] = 0.0

    if args.fixed:
        pred = (eff >= args.threshold).astype(np.int64)
        mode = f"fixed threshold {args.threshold}"
    else:
        pred, _ = calibrate_thresholds(eff, labels)
        mode = "per-attribute calibrated thresholds"
    mA, mA_per, acc, prec, rec, f1 = par_metrics(pred, labels)

    print(f"\n===== ZERO-SHOT BASELINE (SigLIP-2, no training; {mode}) =====")
    print(f"  mA        : {mA * 100:.2f}")
    print(f"  Accuracy  : {acc * 100:.2f}")
    print(f"  Precision : {prec * 100:.2f}")
    print(f"  Recall    : {rec * 100:.2f}")
    print(f"  F1        : {f1 * 100:.2f}")
    print("\n  per-attribute mA (worst first):")
    for a, m in sorted(zip(mat_attrs, mA_per), key=lambda x: x[1]):
        print(f"    {a:<22} {m * 100:5.1f}")

    # feature visualization: PCA-2D colored by gender (numpy SVD, no sklearn needed)
    gi = name_idx.get("female")
    if gi is not None:
        sub = np.random.RandomState(0).permutation(len(feats))[:2000]
        X = feats[sub] - feats[sub].mean(0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        XY = X @ Vt[:2].T
        c = labels[sub, gi]
        plt.figure(figsize=(6, 5))
        for v, col, lab in [(1, "crimson", "Female"), (0, "steelblue", "Not female")]:
            plt.scatter(XY[c == v, 0], XY[c == v, 1], s=6, alpha=0.5, c=col, label=lab)
        plt.legend(); plt.title("SigLIP-2 features (PCA) colored by gender")
        plt.tight_layout(); plt.savefig(fp("feature_pca.png"), dpi=150)
        print(f"\n[saved] {fp('feature_pca.png')}")


if __name__ == "__main__":
    main()
