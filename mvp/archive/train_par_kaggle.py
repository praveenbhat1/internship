"""
LoRA large SigLIP-2 + Orientation Head + OCFR + attribute classifier.
CSV version for the Kaggle 'yuulind/pa-100k' dataset (train/val/test.csv + data/ images).

Run:
  python train_par_kaggle.py --csv_dir CSV --img_dir IMG --epochs 6 --batch 32
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

MODEL_ID = "google/siglip2-large-patch16-256"


def load_csv(csv_dir):
    data, attrs = {}, None
    for s in ["train", "val", "test"]:
        df = pd.read_csv(f"{csv_dir}/{s}.csv")
        if attrs is None:
            attrs = [c for c in df.columns if c != "Image"]
        data[s] = (df["Image"].tolist(), df[attrs].values.astype(np.float32))
    return data, attrs


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


class OCFRModel(nn.Module):
    def __init__(self, vision, dim, nattr=26):
        super().__init__()
        self.vision = vision
        self.orient_head = nn.Linear(dim, 3)
        self.film = nn.Linear(3, dim * 2)
        self.head = nn.Linear(dim, nattr)

    def forward(self, px):
        feat = self.vision(pixel_values=px).pooler_output
        o_logits = self.orient_head(feat)
        gamma, beta = self.film(torch.softmax(o_logits, -1)).chunk(2, -1)
        return self.head(feat * (1 + gamma) + beta), o_logits


def par_metrics(pred, gt):
    e = 1e-12
    tp = ((pred == 1) & (gt == 1)).sum(0); tn = ((pred == 0) & (gt == 0)).sum(0)
    fp = ((pred == 1) & (gt == 0)).sum(0); fn = ((pred == 0) & (gt == 1)).sum(0)
    mA = (0.5 * (tp / (tp + fn + e) + tn / (tn + fp + e))).mean()
    inter = ((pred == 1) & (gt == 1)).sum(1); pc = (pred == 1).sum(1); gc = (gt == 1).sum(1)
    acc = (inter / (pc + gc - inter + e)).mean()
    prec = (inter / (pc + e)).mean(); rec = (inter / (gc + e)).mean()
    return mA, acc, prec, rec, 2 * prec * rec / (prec + rec + e)


def calibrate(scores, labels):
    pred = np.zeros_like(scores, dtype=np.int64)
    for j in range(scores.shape[1]):
        s, y = scores[:, j], labels[:, j]
        bt, bb = 0.5, -1.0
        for t in np.quantile(s, np.linspace(0.05, 0.95, 19)):
            p = s >= t
            tp = (p & (y == 1)).sum(); fn = ((~p) & (y == 1)).sum()
            tn = ((~p) & (y == 0)).sum(); fp = (p & (y == 0)).sum()
            ba = 0.5 * (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9))
            if ba > bb:
                bb, bt = ba, t
        pred[:, j] = s >= bt
    return pred


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, Y = [], []
    for px, y in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits, _ = model(px.to(device))
        P.append(torch.sigmoid(logits).float().cpu().numpy()); Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
    return par_metrics(calibrate(P, Y), Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--orient_w", type=float, default=0.5)
    ap.add_argument("--out", default=".")
    ap.add_argument("--limit_train", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device} | model {args.model}")
    proc = AutoProcessor.from_pretrained(args.model)
    vision = AutoModel.from_pretrained(args.model).vision_model
    dim = vision.config.hidden_size
    print(f"[backbone] hidden dim = {dim}")

    vision = get_peft_model(vision, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    vision.print_trainable_parameters()
    model = OCFRModel(vision, dim).to(device)

    data, attrs = load_csv(args.csv_dir)
    json.dump(attrs, open(f"{args.out}/attributes.json", "w"), indent=2)
    orient_idx = [attrs.index(a) for a in ["Front", "Side", "Back"]]
    print(f"[orientation] Front/Side/Back at {orient_idx}")

    def make(split, shuffle, limit=0):
        names, labels = data[split]
        if limit and len(names) > limit:
            idx = np.random.RandomState(0).permutation(len(names))[:limit]
            names = [names[i] for i in idx]; labels = labels[idx]
        return DataLoader(PARData(names, labels, args.img_dir, proc),
                          batch_size=args.batch, shuffle=shuffle, num_workers=2), labels

    tr_loader, tr_labels = make("train", True, args.limit_train)
    te_loader, _ = make("test", False)

    pos = tr_labels.sum(0)
    pw = torch.tensor((len(tr_labels) - pos) / (pos + 1e-6)).clamp(max=20).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))    # fp16 loss scaling (stability)

    best_mA, best_metrics = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        for b, (px, y) in enumerate(tr_loader):
            px, y = px.to(device), y.to(device)
            otgt = y[:, orient_idx].argmax(1)
            opt.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                logits, o_logits = model(px)
                loss = bce(logits, y) + args.orient_w * ce(o_logits, otgt)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # prevent divergence
            scaler.step(opt); scaler.update()
            if b % 100 == 0:
                print(f"  epoch {epoch+1} batch {b}/{len(tr_loader)} loss {loss.item():.3f}", flush=True)
        mA, acc, prec, rec, f1 = evaluate(model, te_loader, device)
        print(f"=== epoch {epoch+1}: test mA {mA*100:.2f} | F1 {f1*100:.2f}", flush=True)
        if mA > best_mA:                                               # keep only the BEST epoch
            best_mA, best_metrics = mA, (mA, acc, prec, rec, f1)
            torch.save({"lora": model.vision.state_dict(), "orient": model.orient_head.state_dict(),
                        "film": model.film.state_dict(), "head": model.head.state_dict()},
                       f"{args.out}/lora_ocfr.pt")
            print(f"    (saved best so far: mA {mA*100:.2f})", flush=True)

    mA, acc, prec, rec, f1 = best_metrics
    print(f"\n[done] BEST mA {mA*100:.2f}  Acc {acc*100:.2f}  Prec {prec*100:.2f}  Rec {rec*100:.2f}  F1 {f1*100:.2f}")
    print("  baselines: zero-shot 69.47 | frozen linear 85.49")


if __name__ == "__main__":
    main()
