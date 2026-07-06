"""
Day 3 - Trained baseline: a linear classifier on frozen SigLIP-2 features.

Trains a single Linear(768 -> 26) head on the cached train features with weighted BCE
(inverse-frequency, to handle class imbalance), selects the best epoch on val, and
evaluates on test with the SAME protocol as the zero-shot baseline (per-attribute
calibrated thresholds) so the comparison is fair.

Run:
  python train_baseline.py
"""
import numpy as np
import torch
import torch.nn as nn

from evaluate_zeroshot import par_metrics, calibrate_thresholds

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


def load(split):
    X = np.load(f"features/{split}_feats.npy").astype(np.float32)
    Y = np.load(f"features/{split}_labels.npy").astype(np.float32)
    return torch.tensor(X), torch.tensor(Y)


def evaluate(model, X, Y):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X.to(DEVICE))).cpu().numpy()
    yint = Y.numpy().astype(int)
    pred, _ = calibrate_thresholds(probs, yint)
    return par_metrics(pred, yint)


def main():
    print(f"[device] {DEVICE}")
    Xtr, Ytr = load("train")
    Xva, Yva = load("val")
    Xte, Yte = load("test")
    print(f"[data] train {tuple(Xtr.shape)} | val {tuple(Xva.shape)} | test {tuple(Xte.shape)}")

    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)

    # inverse-frequency positive weights for imbalance
    pos = Ytr.sum(0)
    neg = Ytr.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(max=20.0).to(DEVICE)

    model = nn.Linear(768, 26).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_mA, best_state = -1.0, None
    n = Xtr.shape[0]
    for epoch in range(40):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, 2048):
            idx = perm[i:i + 2048]
            opt.zero_grad()
            loss = lossfn(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
        if (epoch + 1) % 5 == 0:
            mA = float(evaluate(model, Xva, Yva)[0])
            print(f"  epoch {epoch+1:2d} | val mA {mA*100:.2f}")
            if mA > best_mA:
                best_mA = mA
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    mA, _, acc, prec, rec, f1 = evaluate(model, Xte, Yte)
    torch.save(model.state_dict(), "features/baseline_linear.pt")

    print("\n===== TRAINED BASELINE (frozen SigLIP-2 + linear head + weighted BCE) =====")
    print(f"  mA        : {mA*100:.2f}")
    print(f"  Accuracy  : {acc*100:.2f}")
    print(f"  Precision : {prec*100:.2f}")
    print(f"  Recall    : {rec*100:.2f}")
    print(f"  F1        : {f1*100:.2f}")
    print("\n  (zero-shot baseline was mA 69.47)  ->  improvement: "
          f"+{mA*100 - 69.47:.2f} mA")
    print("  saved model -> features/baseline_linear.pt")


if __name__ == "__main__":
    main()
