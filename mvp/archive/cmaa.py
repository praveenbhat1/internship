"""
Day 4 - CMAA (Cross-Modal Attribute Attention).

26 attribute queries -- initialized from SigLIP-2's text embeddings of the attribute
prompts -- attend (cross-attention) to the image, producing one attribute-aware feature
per attribute, then classify. This injects the *meaning* of each attribute (from the
frozen text encoder) into the prediction, instead of a plain linear head.

Note: this is the POOLED variant (image is expanded into a few context tokens), because
the full spatial version needs the 196 patch tokens (~24 GB to cache). The mechanism is
the same; the spatial version is the natural upgrade when run on Colab.

Run:
  python cmaa.py
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


def load(split):
    return (torch.tensor(np.load(f"features/{split}_feats.npy").astype(np.float32)),
            torch.tensor(np.load(f"features/{split}_labels.npy").astype(np.float32)))


class CMAA(nn.Module):
    def __init__(self, text_init, dim=768, d=256, n_ctx=8, heads=4):
        super().__init__()
        self.n_ctx, self.d = n_ctx, d
        self.to_ctx = nn.Linear(dim, n_ctx * d)          # image -> context tokens
        self.q_proj = nn.Linear(dim, d)
        self.queries = nn.Parameter(text_init.clone())   # (26, 768) text-initialized
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.cls = nn.Linear(d, 1)

    def forward(self, x):                                # x: (B, 768)
        B = x.shape[0]
        ctx = self.to_ctx(x).view(B, self.n_ctx, self.d)
        q = self.q_proj(self.queries).unsqueeze(0).expand(B, -1, -1)   # (B, 26, d)
        a, _ = self.attn(q, ctx, ctx)                    # each attribute attends image
        a = self.norm(a + q)
        return self.cls(a).squeeze(-1)                   # (B, 26)


def evaluate(model, X, Y):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X.to(DEVICE))).cpu().numpy()
    yint = Y.numpy().astype(int)
    pred, _ = calibrate_thresholds(probs, yint)
    return par_metrics(pred, yint)


def main():
    print(f"[device] {DEVICE}")
    Xtr, Ytr = load("train"); Xva, Yva = load("val"); Xte, Yte = load("test")
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)

    # text embeddings (in label order) -> initialize the 26 attribute queries
    mat_attrs = json.load(open("features/attributes.json"))
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    smodel = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    text_init = get_text_embeds(align_prompts(mat_attrs), smodel, proc, DEVICE).detach().cpu()
    print(f"[CMAA] text-initialized queries: {tuple(text_init.shape)}")

    pos = Ytr.sum(0); pos_weight = ((Ytr.shape[0] - pos) / (pos + 1e-6)).clamp(max=20).to(DEVICE)
    model = CMAA(text_init).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_mA, best_state, n = -1.0, None, Xtr.shape[0]
    for epoch in range(40):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, 2048):
            idx = perm[i:i + 2048]
            opt.zero_grad()
            lossfn(model(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        if (epoch + 1) % 5 == 0:
            mA = float(evaluate(model, Xva, Yva)[0])
            print(f"  epoch {epoch+1:2d} | val mA {mA*100:.2f}")
            if mA > best_mA:
                best_mA = mA
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    mA, _, acc, prec, rec, f1 = evaluate(model, Xte, Yte)
    torch.save(model.state_dict(), "features/cmaa.pt")

    print("\n===== CMAA (frozen SigLIP-2 + cross-modal attribute attention) =====")
    print(f"  mA        : {mA*100:.2f}")
    print(f"  Accuracy  : {acc*100:.2f}")
    print(f"  Precision : {prec*100:.2f}")
    print(f"  Recall    : {rec*100:.2f}")
    print(f"  F1        : {f1*100:.2f}")
    print(f"\n  baselines:  zero-shot 69.47  |  linear head 85.49")
    print(f"  CMAA vs linear head: {mA*100 - 85.49:+.2f} mA")


if __name__ == "__main__":
    main()
