"""
Extract & cache frozen SigLIP-2 features for PA-100K.

Reads PA-100K's annotation.mat (HydraPlus-Net format) + the images, runs each image
through a FROZEN SigLIP-2 image encoder, and caches the features + labels to disk so
downstream training never has to touch the heavy backbone again. This is the
"visual features extracted" deliverable.

Outputs (in --out, default ./features/):
  {split}_feats.npy     (N, 768)     pooled image features            [fp16]
  {split}_patches.npy   (N,196,768)  patch tokens  (only with --patches; large)
  {split}_labels.npy    (N, 26)      binary attribute labels
  {split}_names.npy     (N,)         image filenames
  attributes.json                    the 26 attribute names from the .mat (for alignment)

Run (after PA-100K is in data/PA100K/):
  python extract_features.py --data_dir data/PA100K
  python extract_features.py --data_dir data/PA100K --splits test       # quick test-only run
  python extract_features.py --data_dir data/PA100K --patches           # also cache patch tokens (for CMAA later)

IMPORTANT: check the printed "[attributes from .mat]" order matches the order in
attributes.py. If it differs, reorder attributes.py so prompts line up with the labels.
"""
import argparse
import json
import os

import numpy as np
import torch
import scipy.io as sio
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip2-base-patch16-224"


def square_pad(img):
    """Pad a tall crop to square so the 224x224 resize doesn't distort the body."""
    s = max(img.size)
    return ImageOps.pad(img.convert("RGB"), (s, s), color=(0, 0, 0))


def _unwrap(v):
    """PA-100K .mat stores strings as nested arrays; peel down to the scalar."""
    while isinstance(v, np.ndarray):
        if v.size == 0:
            return ""
        v = v.flat[0]
    return v


def load_annotation(mat_path):
    """Return {split: (names, labels(N,26))} and the 26 attribute names."""
    m = sio.loadmat(mat_path)
    names = lambda k: [str(_unwrap(x)) for x in m[k]]
    labels = lambda k: np.asarray(m[k]).astype(np.float32)
    data = {
        "train": (names("train_images_name"), labels("train_label")),
        "val":   (names("val_images_name"),   labels("val_label")),
        "test":  (names("test_images_name"),  labels("test_label")),
    }
    attrs = [str(_unwrap(a)) for a in m["attributes"]]
    return data, attrs


@torch.no_grad()
def extract_split(names, labels, img_dir, processor, model, device, out, split,
                  batch=32, want_patches=False, pool=0, fp16=False):
    import numpy.lib.format as npf
    feats = []
    n = len(names)
    pmm = None
    if want_patches:
        ntok = pool * pool if pool else 196
        pmm = npf.open_memmap(os.path.join(out, f"{split}_patches.npy"),
                              mode="w+", dtype=np.float16, shape=(n, ntok, 768))
    for i in range(0, n, batch):
        chunk = names[i:i + batch]
        imgs = [square_pad(Image.open(os.path.join(img_dir, nm))) for nm in chunk]
        px = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        if fp16:
            px = px.half()
        vis = model.get_image_features(pixel_values=px)
        pooled = getattr(vis, "pooler_output", vis)                    # (B, 768)
        feats.append(pooled.float().cpu().numpy())
        if want_patches:
            pt = vis.last_hidden_state.float()                         # (b, 196, 768)
            if pool:
                b, g = pt.shape[0], int(round(pt.shape[1] ** 0.5))     # g = 14
                pt = pt.reshape(b, g, g, -1).permute(0, 3, 1, 2)       # (b,768,14,14)
                pt = torch.nn.functional.adaptive_avg_pool2d(pt, (pool, pool))
                pt = pt.permute(0, 2, 3, 1).reshape(b, pool * pool, -1)  # (b,pool^2,768)
            pmm[i:i + len(chunk)] = pt.cpu().numpy().astype(np.float16)
        if (i // batch) % 20 == 0:
            print(f"  [{split}] {min(i + batch, n)}/{n}", flush=True)

    feats = np.concatenate(feats).astype(np.float16)
    np.save(os.path.join(out, f"{split}_feats.npy"), feats)
    np.save(os.path.join(out, f"{split}_labels.npy"), labels)
    np.save(os.path.join(out, f"{split}_names.npy"), np.array(names))
    print(f"  [{split}] saved feats {feats.shape}, labels {labels.shape}")
    if pmm is not None:
        pmm.flush()
        print(f"  [{split}] saved patches ({n}, {pmm.shape[1]}, 768)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/PA100K")
    ap.add_argument("--ann", default=None, help="annotation.mat (default: data_dir/annotation.mat)")
    ap.add_argument("--img_dir", default=None, help="image folder (default: data_dir/release_data)")
    ap.add_argument("--out", default="features")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patches", action="store_true", help="also cache patch tokens")
    ap.add_argument("--pool", type=int, default=0,
                    help="pool patch grid to NxN tokens (e.g. 7 -> 49 tokens; 0 = keep 196)")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    ann = args.ann or os.path.join(args.data_dir, "annotation.mat")
    img_dir = args.img_dir or os.path.join(args.data_dir, "release_data")
    # PA-100K is sometimes nested one level deeper (release_data/release_data/*.jpg)
    if os.path.isdir(os.path.join(img_dir, "release_data")):
        img_dir = os.path.join(img_dir, "release_data")

    assert os.path.exists(ann), f"annotation not found: {ann}"
    assert os.path.isdir(img_dir), f"image dir not found: {img_dir}"
    os.makedirs(args.out, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    fp16 = device == "cuda"                     # fp16 only helps on CUDA; CPU/MPS stay fp32
    print(f"[device] {device}  (fp16={fp16})")
    print(f"[images] {img_dir}")
    print(f"[load] {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    if fp16:
        model = model.half()

    data, attrs = load_annotation(ann)
    json.dump(attrs, open(os.path.join(args.out, "attributes.json"), "w"), indent=2)
    print(f"[attributes from .mat] {attrs}")
    print("  --> verify this order matches attributes.py before training!")

    for split in args.splits:
        names, labels = data[split]
        print(f"\n=== {split}: {len(names)} images ===")
        extract_split(names, labels, img_dir, processor, model, device, args.out,
                      split, batch=args.batch, want_patches=args.patches,
                      pool=args.pool, fp16=fp16)

    print("\n[done] features cached in", args.out)


if __name__ == "__main__":
    main()
