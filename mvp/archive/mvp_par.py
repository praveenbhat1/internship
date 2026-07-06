"""
MVP: Zero-shot Pedestrian Attribute Recognition with SigLIP-2 only.

Idea: SigLIP-2 aligns images and text in one space. We ask it 26 yes/no questions
(one natural-language prompt per PA-100K attribute) and read off how strongly the
image matches each prompt. NO TRAINING -- this is a pure zero-shot baseline that
also serves as the starting point the full model (CMAA/OCFR/DACG/CCLoss) builds on.

Usage:
    python mvp_par.py --image path/to/person.jpg
    python mvp_par.py --image person.jpg --groups          # enforce age/view single-choice
    python mvp_par.py --folder path/to/crops/ --threshold 0.5
    python mvp_par.py --image person.jpg --model google/siglip2-base-patch16-224

Output: a table of attribute -> probability -> YES/NO, and a JSON file next to it.
"""
import argparse, json, os, sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

from attributes import NAMES, PROMPTS, AGE_GROUP, VIEW_GROUP


def square_pad(img: Image.Image) -> Image.Image:
    """Pad a tall pedestrian crop to a square so the 224x224 resize doesn't squash it."""
    s = max(img.size)
    return ImageOps.pad(img.convert("RGB"), (s, s), color=(0, 0, 0))


def load_model(model_id, device):
    print(f"[load] {model_id} (first run downloads weights) ...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return processor, model


@torch.no_grad()
def predict(image_path, processor, model, device):
    """Return a dict attribute_name -> probability in [0,1]."""
    img = square_pad(Image.open(image_path))
    inputs = processor(
        text=PROMPTS, images=img,
        padding="max_length", max_length=64, return_tensors="pt",
    ).to(device)
    out = model(**inputs)
    # SigLIP is trained with a sigmoid objective -> sigmoid(logit) ~ P(prompt matches image)
    probs = torch.sigmoid(out.logits_per_image)[0]  # (26,)
    return {name: float(p) for name, p in zip(NAMES, probs)}


def apply_group_constraints(probs):
    """Within each mutually-exclusive group keep only the top-1 as positive."""
    result = dict(probs)
    for group in (AGE_GROUP, VIEW_GROUP):
        winner = max(group, key=lambda n: probs[n])
        for n in group:
            result[n] = probs[n] if n == winner else min(probs[n], 0.0)
    return result


def render(probs, threshold, groups):
    scored = apply_group_constraints(probs) if groups else probs
    rows = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    print("\n  attribute             prob    pred")
    print("  " + "-" * 38)
    positives = []
    for name, p in rows:
        eff = scored[name]
        yes = (eff >= threshold) and (not groups or eff > 0.0)
        if yes:
            positives.append(name)
        print(f"  {name:<20} {p:6.3f}   {'YES' if yes else 'NO'}")
    print("\n  PREDICTED ATTRIBUTES:", ", ".join(positives) if positives else "(none)")
    return positives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="path to a single pedestrian crop")
    ap.add_argument("--folder", help="path to a folder of crops")
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--groups", action="store_true",
                    help="enforce single-choice for age and viewpoint groups")
    args = ap.parse_args()

    if not args.image and not args.folder:
        ap.error("provide --image or --folder")

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}")
    processor, model = load_model(args.model, device)

    paths = []
    if args.image:
        paths.append(args.image)
    if args.folder:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths += [str(p) for p in sorted(Path(args.folder).iterdir())
                  if p.suffix.lower() in exts]

    all_results = {}
    for path in paths:
        print(f"\n==== {path} ====")
        probs = predict(path, processor, model, device)
        positives = render(probs, args.threshold, args.groups)
        all_results[path] = {"probs": probs, "predicted": positives}

    out_path = Path(args.folder or os.path.dirname(args.image) or ".") / "mvp_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
