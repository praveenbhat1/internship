"""
Day 4b - SPATIAL CMAA (the real version).

26 attribute queries (initialized from SigLIP-2 text embeddings) cross-attend over the
49 spatial patch tokens of each image (a 7x7 grid), so each attribute can focus on its
own body region. This is the localization mechanism the pooled CMAA lacked.

Needs patch features:
  python extract_features.py --ann ANN --img_dir IMG --splits train val test \
      --out features --patches --pool 7
Then:
  python cmaa_spatial.py
"""
import json
import numpy as np
import torch
import torch.nn as nn

from evaluate_zeroshot import (par_metrics, calibrate_thresholds,
                               align_prompts, get_text_embeds)
from transformers import AutoModel, AutoProcessor

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_ID = "google/siglip2-base-patch16-224"


def load_patches(split, subsample=None):
    X = np.load(f"features/{split}_patches.npy", mmap_mode="r")        # (N, 49, 768) on disk
    Y = np.load(f"features/{split}_labels.npy").astype(np.float32)
    if subsample and len(Y) > subsample:                              # RAM-friendly subset
        idx = np.sort(np.random.RandomState(0).permutation(len(Y))[:subsample])
        X = np.asarray(X[idx]); Y = Y[idx]
    else:
        X = np.asarray(X)                                            # materialize into RAM
    return X, Y


def batch(Xmm, idx):
    return torch.tensor(np.asarray(Xmm[idx]).astype(np.float32))


class SpatialCMAA(nn.Module):
    def __init__(self, text_init, dim=768, d=256, heads=4):
        super().__init__()
        self.q_proj = nn.Linear(dim, d)
        self.kv_proj = nn.Linear(dim, d)
        self.queries = nn.Parameter(text_init.clone())   # (26, 768) text-initialized
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.cls = nn.Linear(d, 1)

    def forward(self, patches):                          # (B, 49, 768)
        B = patches.shape[0]
        kv = self.kv_proj(patches)                       # (B, 49, d)
        q = self.q_proj(self.queries).unsqueeze(0).expand(B, -1, -1)   # (B, 26, d)
        a, _ = self.attn(q, kv, kv)                      # each attribute attends 49 regions
        a = self.norm(a + q)
        return self.cls(a).squeeze(-1)                   # (B, 26)


@torch.no_grad()
def evaluate(model, Xmm, Y):
    model.eval()
    probs = []
    for i in range(0, len(Y), 4096):
        xb = batch(Xmm, np.arange(i, min(i + 4096, len(Y)))).to(DEVICE)
        probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    pred, _ = calibrate_thresholds(np.concatenate(probs), Y.astype(int))
    return par_metrics(pred, Y.astype(int))


def main():
    print(f"[device] {DEVICE}")
    Xtr, Ytr = load_patches("train", subsample=40000)   # 40k in RAM (fast, memory-safe)
    Xva, Yva = load_patches("val")
    Xte, Yte = load_patches("test")
    print(f"[patches, in RAM] train {Xtr.shape} | val {Xva.shape} | test {Xte.shape}")

    mat_attrs = json.load(open("features/attributes.json"))
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    smodel = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    text_init = get_text_embeds(align_prompts(mat_attrs), smodel, proc, DEVICE).detach().cpu()

    pos = Ytr.sum(0)
    pos_weight = torch.tensor((len(Ytr) - pos) / (pos + 1e-6)).clamp(max=20).to(DEVICE)
    model = SpatialCMAA(text_init).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n, best_mA, best_state = len(Ytr), -1.0, None
    for epoch in range(50):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, 2048):
            idx = np.sort(perm[i:i + 2048])              # sorted for fast memmap reads
            xb = batch(Xtr, idx).to(DEVICE)
            yb = torch.tensor(Ytr[idx]).to(DEVICE)
            opt.zero_grad()
            lossfn(model(xb), yb).backward()
            opt.step()
        if (epoch + 1) % 5 == 0:
            mA = float(evaluate(model, Xva, Yva)[0])
            print(f"  epoch {epoch+1:2d} | val mA {mA*100:.2f}")
            if mA > best_mA:
                best_mA = mA
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    mA, _, acc, prec, rec, f1 = evaluate(model, Xte, Yte)
    torch.save(model.state_dict(), "features/cmaa_spatial.pt")

    print("\n===== SPATIAL CMAA (cross-attention over 49 patch tokens) =====")
    print(f"  mA        : {mA*100:.2f}")
    print(f"  Accuracy  : {acc*100:.2f}")
    print(f"  Precision : {prec*100:.2f}")
    print(f"  Recall    : {rec*100:.2f}")
    print(f"  F1        : {f1*100:.2f}")
    print(f"\n  baselines:  linear head 85.49  |  pooled CMAA 84.78")
    print(f"  spatial CMAA vs linear head: {mA*100 - 85.49:+.2f} mA")


if __name__ == "__main__":
    main()
