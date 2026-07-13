"""
Build the cross-dataset presentation figures from peta_results.json + features/metrics.json:
  peta_comparison.png  - in-domain (PA-100K) vs cross-domain (PETA) mA / Accuracy / F1
  peta_perattr.png     - per-attribute mA on PETA (which attributes transfer well)
Run AFTER downloading peta_results.json from Kaggle:  python peta_report.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("peta_results.json"))
M = json.load(open("features/metrics.json"))

# --- 1. in-domain vs cross-domain ---
labels = ["mA", "Accuracy", "F1"]
indom = [M["mA"], M["Accuracy"], M["F1"]]
cross = [R["mA"], R["Accuracy"], R["F1"]]
x = range(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(7, 4.5))
b1 = ax.bar([i - w/2 for i in x], indom, w, label="In-domain (PA-100K, 10k test)", color="#2e8b57")
b2 = ax.bar([i + w/2 for i in x], cross, w, label=f"Cross-domain (PETA, {R['n_images']} imgs)", color="#e67e22")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8, f"{b.get_height():.1f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylabel("%"); ax.set_ylim(0, 100)
ax.set_title("Generalization: same model, no retraining\nin-domain (PA-100K) vs cross-domain (PETA)", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("peta_comparison.png", dpi=150, bbox_inches="tight")
print("[saved] peta_comparison.png")

# --- 2. per-attribute mA on PETA ---
pa = R["per_attr_mA"]
items = sorted(pa.items(), key=lambda kv: kv[1])
names = [k for k, _ in items]; vals = [v for _, v in items]
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#c0392b" if v < 60 else "#e67e22" if v < 75 else "#2e8b57" for v in vals]
ax.barh(names, vals, color=colors)
for i, v in enumerate(vals):
    ax.text(v + 0.5, i, f"{v:.0f}", va="center", fontsize=9)
ax.set_xlabel("mA (%) on PETA"); ax.set_xlim(0, 100)
ax.set_title(f"Cross-dataset per-attribute accuracy on PETA ({R['n_attrs']} shared attributes)\n"
             "green = transfers well, red = domain gap", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig("peta_perattr.png", dpi=150, bbox_inches="tight")
print("[saved] peta_perattr.png")
print(f"\nCross-domain summary: mA {R['mA']} | Acc {R['Accuracy']} | F1 {R['F1']} on {R['n_images']} PETA images")
