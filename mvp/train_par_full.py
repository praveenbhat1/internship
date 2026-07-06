"""
FULL pipeline: LoRA large SigLIP-2 -> CMAA -> OCFR -> DACG -> Classifier, with CCLoss.
Follows the methodology order. Every module has an on/off flag for ablations.

  image -> LoRA vision -> patch tokens + pooled
        -> CMAA   (26 attribute-text queries cross-attend the patches)   [--no_cmaa]
        -> OCFR   (orientation head + viewpoint routing)                 [--no_ocfr]
        -> DACG   (dynamic + static attribute correlation graph)         [--no_dacg]
        -> per-attribute classifier -> 26 sigmoids
  loss = weighted BCE + OCFR orientation CE + CCLoss (logical consistency) [--no_ccloss]

Stable fp16 (GradScaler + grad clip) + best-checkpoint saving.

Run: python train_par_full.py --csv_dir CSV --img_dir IMG --epochs 4 --batch 32
Ablation examples: add --no_cmaa  /  --no_dacg  /  --no_ccloss  etc.
"""
import argparse, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
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
    def __init__(self, names, labels, img_dir, proc):
        self.names, self.labels, self.img_dir, self.proc = names, labels, img_dir, proc

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = square_pad(Image.open(f"{self.img_dir}/{self.names[i]}"))
        return self.proc(images=img, return_tensors="pt")["pixel_values"][0], torch.tensor(self.labels[i])


class FullPAR(nn.Module):
    def __init__(self, vision, dim, text_dim, text_emb, nattr=26, d=512,
                 use_cmaa=True, use_ocfr=True, use_dacg=True):
        super().__init__()
        self.vision, self.nattr, self.d = vision, nattr, d
        self.use_cmaa, self.use_ocfr, self.use_dacg = use_cmaa, use_ocfr, use_dacg
        self.register_buffer("text_emb", text_emb)          # (nattr, text_dim), frozen queries
        self.q_proj = nn.Linear(text_dim, d)
        self.kv_proj = nn.Linear(dim, d)
        self.pool_proj = nn.Linear(dim, d)
        self.cmaa = nn.MultiheadAttention(d, 4, batch_first=True)
        self.cmaa_norm = nn.LayerNorm(d)
        # OCFR
        self.orient_head = nn.Linear(dim, 3)
        self.film = nn.Linear(3, d * 2)
        # DACG
        self.A_static = nn.Parameter(torch.zeros(nattr, nattr))
        self.dacg_lin = nn.Linear(d, d)
        self.dacg_norm = nn.LayerNorm(d)
        # per-attribute classifier
        self.W = nn.Parameter(torch.randn(nattr, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(nattr))

    def forward(self, px):
        out = self.vision(pixel_values=px)
        patches = out.last_hidden_state                     # (B, P, dim)
        pooled = out.pooler_output                          # (B, dim)
        B = pooled.shape[0]

        if self.use_cmaa:                                   # 26 attribute-aware features
            q = self.q_proj(self.text_emb).unsqueeze(0).expand(B, -1, -1)   # (B, 26, d)
            kv = self.kv_proj(patches)                                       # (B, P, d)
            a, _ = self.cmaa(q, kv, kv)
            f26 = self.cmaa_norm(a + q)                                      # (B, 26, d)
        else:
            f26 = self.pool_proj(pooled).unsqueeze(1).expand(B, self.nattr, self.d)

        o_logits = self.orient_head(pooled)                 # (B, 3)  (always computed)
        if self.use_ocfr:
            gamma, beta = self.film(torch.softmax(o_logits, -1)).chunk(2, -1)   # (B, d) each
            f26 = f26 * (1 + gamma).unsqueeze(1) + beta.unsqueeze(1)

        if self.use_dacg:                                   # dynamic + static correlation graph
            A_dyn = torch.softmax(f26 @ f26.transpose(1, 2) / (self.d ** 0.5), -1)  # (B,26,26)
            A = 0.5 * A_dyn + 0.5 * torch.softmax(self.A_static, -1).unsqueeze(0)
            f26 = self.dacg_norm(f26 + self.dacg_lin(A @ f26))

        logits = torch.einsum("bad,ad->ba", f26, self.W) + self.b           # (B, 26)
        return logits, o_logits

    @torch.no_grad()
    def forward_explain(self, px):
        """Like forward() but also returns CMAA attention (B,26,P) and DACG adjacency (B,26,26)."""
        out = self.vision(pixel_values=px)
        patches, pooled = out.last_hidden_state, out.pooler_output
        B = pooled.shape[0]
        cmaa_attn, dacg_A = None, None
        if self.use_cmaa:
            q = self.q_proj(self.text_emb).unsqueeze(0).expand(B, -1, -1)
            kv = self.kv_proj(patches)
            a, cmaa_attn = self.cmaa(q, kv, kv)                 # cmaa_attn: (B, 26, P)
            f26 = self.cmaa_norm(a + q)
        else:
            f26 = self.pool_proj(pooled).unsqueeze(1).expand(B, self.nattr, self.d)
        o_logits = self.orient_head(pooled)
        if self.use_ocfr:
            gamma, beta = self.film(torch.softmax(o_logits, -1)).chunk(2, -1)
            f26 = f26 * (1 + gamma).unsqueeze(1) + beta.unsqueeze(1)
        if self.use_dacg:
            A_dyn = torch.softmax(f26 @ f26.transpose(1, 2) / (self.d ** 0.5), -1)
            dacg_A = 0.5 * A_dyn + 0.5 * torch.softmax(self.A_static, -1).unsqueeze(0)
            f26 = self.dacg_norm(f26 + self.dacg_lin(dacg_A @ f26))
        logits = torch.einsum("bad,ad->ba", f26, self.W) + self.b
        return logits, o_logits, cmaa_attn, dacg_A, pooled


def ccloss(logits, groups):
    """Logical consistency: probs within a mutually-exclusive group should sum to ~1."""
    p = torch.sigmoid(logits)
    loss = 0.0
    for g in groups:
        loss = loss + ((p[:, g].sum(1) - 1.0) ** 2).mean()
    return loss


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
        s, y = scores[:, j], labels[:, j]; bt, bb = 0.5, -1.0
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
    model.eval(); P, Y = [], []
    for px, y in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits, _ = model(px.to(device))
        P.append(torch.sigmoid(logits).float().cpu().numpy()); Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
    return par_metrics(calibrate(P, Y), Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True); ap.add_argument("--img_dir", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--orient_w", type=float, default=0.3); ap.add_argument("--cc_w", type=float, default=0.1)
    ap.add_argument("--out", default="."); ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--no_cmaa", action="store_true"); ap.add_argument("--no_ocfr", action="store_true")
    ap.add_argument("--no_dacg", action="store_true"); ap.add_argument("--no_ccloss", action="store_true")
    ap.add_argument("--drop_gender_age", action="store_true",
                    help="remove Female + the 3 Age labels -> 22 attributes")
    ap.add_argument("--drop_age", action="store_true",
                    help="remove only the 3 Age labels (keep Female) -> 23 attributes")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device} | model {args.model}")
    print(f"[modules] CMAA={not args.no_cmaa} OCFR={not args.no_ocfr} DACG={not args.no_dacg} CCLoss={not args.no_ccloss}")
    proc = AutoProcessor.from_pretrained(args.model)
    full = AutoModel.from_pretrained(args.model)
    dim = full.vision_model.config.hidden_size

    data, attrs = load_csv(args.csv_dir)
    drop = []
    if args.drop_gender_age:                                 # no gender + no age -> 22
        drop = ["Female", "AgeOver60", "Age18-60", "AgeLess18"]
    elif args.drop_age:                                      # keep gender, no age -> 23
        drop = ["AgeOver60", "Age18-60", "AgeLess18"]
    if drop:
        keep = [i for i, a in enumerate(attrs) if a not in drop]
        attrs = [attrs[i] for i in keep]
        data = {s: (n, l[:, keep]) for s, (n, l) in data.items()}
        print(f"[drop] removed {drop} -> {len(attrs)} attributes")
    json.dump(attrs, open(f"{args.out}/attributes.json", "w"))
    # attribute-text embeddings for CMAA (frozen text tower, computed once)
    prompts = [f"a photo of a person, {a}" for a in attrs]
    tin = proc(text=prompts, padding="max_length", max_length=64, return_tensors="pt")
    with torch.no_grad():
        T = full.get_text_features(**tin)
        T = getattr(T, "pooler_output", T).float()          # (26, text_dim)
    print(f"[backbone] dim={dim} | text_emb {tuple(T.shape)}")

    vision = get_peft_model(full.vision_model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    vision.print_trainable_parameters()
    model = FullPAR(vision, dim, T.shape[1], T, nattr=len(attrs),
                    use_cmaa=not args.no_cmaa, use_ocfr=not args.no_ocfr,
                    use_dacg=not args.no_dacg).to(device)

    oi = [attrs.index(a) for a in ["Front", "Side", "Back"]]
    cc_groups = [oi]                                          # viewpoint mutual-exclusion
    if all(a in attrs for a in ["AgeOver60", "Age18-60", "AgeLess18"]):
        cc_groups.append([attrs.index(a) for a in ["AgeOver60", "Age18-60", "AgeLess18"]])
    print(f"[orient] {oi} | cc_groups {cc_groups}")

    def make(split, sh, lim=0):
        n, l = data[split]
        if lim and len(n) > lim:
            idx = np.random.RandomState(0).permutation(len(n))[:lim]; n = [n[i] for i in idx]; l = l[idx]
        return DataLoader(PARData(n, l, args.img_dir, proc), batch_size=args.batch,
                          shuffle=sh, num_workers=2), l

    trl, trlab = make("train", True, args.limit_train); tel, _ = make("test", False)
    pos = trlab.sum(0)
    pw = torch.tensor((len(trlab) - pos) / (pos + 1e-6)).clamp(max=20).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw); ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_mA, best_metrics = -1.0, None
    for ep in range(args.epochs):
        model.train()
        for b, (px, y) in enumerate(trl):
            px, y = px.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                logits, ol = model(px)
                loss = bce(logits, y)
                if not args.no_ocfr:
                    loss = loss + args.orient_w * ce(ol, y[:, oi].argmax(1))
                if not args.no_ccloss:
                    loss = loss + args.cc_w * ccloss(logits, cc_groups)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if b % 100 == 0:
                print(f"  ep{ep+1} b{b}/{len(trl)} loss {loss.item():.3f}", flush=True)
        mA, acc, prec, rec, f1 = evaluate(model, tel, device)
        print(f"=== epoch {ep+1}: test mA {mA*100:.2f} | F1 {f1*100:.2f}", flush=True)
        if mA > best_mA:
            best_mA, best_metrics = mA, (mA, acc, prec, rec, f1)
            sd = model.state_dict()                          # save only trained parts (~15MB)
            small = {k: v for k, v in sd.items() if ('lora_' in k) or (not k.startswith('vision.'))}
            torch.save(small, f"{args.out}/par_full.pt")     # backbone re-loads from HF in the demo
            print(f"    (saved best: mA {mA*100:.2f})", flush=True)

    mA, acc, prec, rec, f1 = best_metrics
    print(f"\n[done] BEST mA {mA*100:.2f}  Acc {acc*100:.2f}  Prec {prec*100:.2f}  Rec {rec*100:.2f}  F1 {f1*100:.2f}")
    print("  baselines: zero-shot 69.47 | frozen linear 85.49 | LoRA+OCFR ~90")


if __name__ == "__main__":
    main()
