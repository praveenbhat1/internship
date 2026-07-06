"""
LoRA fine-tune a LARGE SigLIP-2 for Pedestrian Attribute Recognition.

Unlike the frozen baseline, this ADAPTS the backbone via LoRA (small trainable adapters
inside the vision transformer) + a classifier head. Only the LoRA adapters + head train;
the base weights stay fixed, so it fits a free Colab T4.

Self-contained (metrics inline) so only this file + the dataset are needed.

Run (on Colab T4, after data is unzipped):
  python train_lora.py --ann ANN --img_dir IMG --epochs 8 --batch 32
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import scipy.io as sio
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

MODEL_ID = "google/siglip2-large-patch16-256"       # the LARGE variant (1024-dim)


# ---------- data ----------
def _unwrap(v):
    while isinstance(v, np.ndarray):
        if v.size == 0:
            return ""
        v = v.flat[0]
    return v


def load_ann(mat_path):
    m = sio.loadmat(mat_path)
    names = lambda k: [str(_unwrap(x)) for x in m[k]]
    labels = lambda k: np.asarray(m[k]).astype(np.float32)
    data = {"train": (names("train_images_name"), labels("train_label")),
            "val":   (names("val_images_name"),   labels("val_label")),
            "test":  (names("test_images_name"),  labels("test_label"))}
    return data, [str(_unwrap(a)) for a in m["attributes"]]


def square_pad(img):
    s = max(img.size)
    return ImageOps.pad(img.convert("RGB"), (s, s), color=(0, 0, 0))


class PARData(Dataset):
    def __init__(self, names, labels, img_dir, processor):
        self.names, self.labels, self.img_dir, self.proc = names, labels, img_dir, processor

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = square_pad(Image.open(f"{self.img_dir}/{self.names[i]}"))
        px = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        return px, torch.tensor(self.labels[i])


# ---------- model ----------
class LoRAPAR(nn.Module):
    def __init__(self, vision, dim, nattr=26):
        super().__init__()
        self.vision = vision
        self.head = nn.Linear(dim, nattr)

    def forward(self, px):
        feat = self.vision(pixel_values=px).pooler_output       # (B, dim)
        return self.head(feat)


# ---------- metrics ----------
def par_metrics(pred, gt):
    eps = 1e-12
    tp = ((pred == 1) & (gt == 1)).sum(0); tn = ((pred == 0) & (gt == 0)).sum(0)
    fp = ((pred == 1) & (gt == 0)).sum(0); fn = ((pred == 0) & (gt == 1)).sum(0)
    mA = (0.5 * (tp / (tp + fn + eps) + tn / (tn + fp + eps))).mean()
    inter = ((pred == 1) & (gt == 1)).sum(1); pc = (pred == 1).sum(1); gc = (gt == 1).sum(1)
    acc = (inter / (pc + gc - inter + eps)).mean()
    prec = (inter / (pc + eps)).mean(); rec = (inter / (gc + eps)).mean()
    return mA, acc, prec, rec, 2 * prec * rec / (prec + rec + eps)


def calibrate(scores, labels):
    pred = np.zeros_like(scores, dtype=np.int64)
    for j in range(scores.shape[1]):
        s, y = scores[:, j], labels[:, j]
        best_t, best = 0.5, -1.0
        for t in np.quantile(s, np.linspace(0.05, 0.95, 19)):
            p = s >= t
            tp = (p & (y == 1)).sum(); fn = ((~p) & (y == 1)).sum()
            tn = ((~p) & (y == 0)).sum(); fp = (p & (y == 0)).sum()
            ba = 0.5 * (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9))
            if ba > best:
                best, best_t = ba, t
        pred[:, j] = s >= best_t
    return pred


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, Y = [], []
    for px, y in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(px.to(device))
        P.append(torch.sigmoid(logits).float().cpu().numpy()); Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
    return par_metrics(calibrate(P, Y), Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="features")
    ap.add_argument("--limit_train", type=int, default=0, help="cap train size (0 = all)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device} | model {args.model}")
    proc = AutoProcessor.from_pretrained(args.model)
    full = AutoModel.from_pretrained(args.model)
    vision = full.vision_model
    dim = vision.config.hidden_size
    print(f"[backbone] vision hidden dim = {dim}")

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
    vision = get_peft_model(vision, lora)
    vision.print_trainable_parameters()
    model = LoRAPAR(vision, dim).to(device)

    data, attrs = load_ann(args.ann)
    json.dump(attrs, open(f"{args.out}/attributes.json", "w"), indent=2)

    def make(split, shuffle, limit=0):
        names, labels = data[split]
        if limit and len(names) > limit:
            idx = np.random.RandomState(0).permutation(len(names))[:limit]
            names = [names[i] for i in idx]; labels = labels[idx]
        ds = PARData(names, labels, args.img_dir, proc)
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=2), labels

    tr_loader, tr_labels = make("train", True, args.limit_train)
    te_loader, _ = make("test", False)

    pos = tr_labels.sum(0)
    pos_weight = torch.tensor((len(tr_labels) - pos) / (pos + 1e-6)).clamp(max=20).to(device)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        for b, (px, y) in enumerate(tr_loader):
            px, y = px.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                loss = lossfn(model(px), y)
            loss.backward(); opt.step()
            if b % 100 == 0:
                print(f"  epoch {epoch+1} batch {b}/{len(tr_loader)} loss {loss.item():.3f}", flush=True)
        mA, acc, prec, rec, f1 = evaluate(model, te_loader, device)
        print(f"=== epoch {epoch+1}: test mA {mA*100:.2f} | F1 {f1*100:.2f}", flush=True)

    torch.save({"lora": vision.state_dict(), "head": model.head.state_dict()},
               f"{args.out}/lora_par.pt")
    print(f"\n[done] saved -> {args.out}/lora_par.pt")
    print(f"  final: mA {mA*100:.2f}  Acc {acc*100:.2f}  Prec {prec*100:.2f}  Rec {rec*100:.2f}  F1 {f1*100:.2f}")
    print("  baselines: zero-shot 69.47 | frozen linear 85.49")


if __name__ == "__main__":
    main()
