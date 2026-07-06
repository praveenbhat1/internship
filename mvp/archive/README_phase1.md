# Phase 1 — Preprocessing, Visual Feature Extraction & Zero-Shot Baseline

Everything for the first phase of the project, in run order. Phase 1 delivers:
**preprocessing → cached SigLIP-2 features → a zero-shot baseline score → a feature plot.**

---

## 0. Setup (once)

```bash
cd /Users/praveenbhat/mvp
pip install -r requirements.txt        # torch, transformers, scipy, pillow, sentencepiece, protobuf
pip install gradio                     # only needed for the web UI (app.py)
```
The SigLIP-2 model downloads once (~1.5 GB) and is cached in `~/.cache/huggingface` — it does
**not** re-download on later runs.

---

## 1. Get the dataset

Download **PA-100K** (search "PA-100K dataset download" — on Kaggle / the HydraPlus-Net repo).
Place it exactly like this:

```
mvp/data/PA100K/
├── release_data/        # the 100,000 .jpg images
└── annotation.mat       # 26 labels + train/val/test split
```

---

## 2. Extract & cache features  →  `extract_features.py`

```bash
# quick check on the test split first (fast), then the full dataset
python3 extract_features.py --data_dir data/PA100K --splits test
python3 extract_features.py --data_dir data/PA100K

# optional: also cache patch tokens for CMAA later (~24 GB, only when you start Day 4)
python3 extract_features.py --data_dir data/PA100K --patches
```

Produces in `features/`: `{split}_feats.npy` (N×768), `{split}_labels.npy` (N×26),
`{split}_names.npy`, and `attributes.json`.

> ⚠️ **Check the printed `[attributes from .mat]` order matches `attributes.py`.** They must
> line up so each prompt scores the right label column. If they differ, reorder `attributes.py`.
> (`evaluate_zeroshot.py` also auto-aligns prompts by name as a safety net.)

---

## 3. Baseline score + feature plot  →  `evaluate_zeroshot.py`

```bash
python3 evaluate_zeroshot.py --split test --groups
```

Prints your **baseline** numbers (the bottom row of the whole project):
- `mA` + instance `Accuracy / Precision / Recall / F1`
- per-attribute mA (worst first) — shows where SigLIP is weak (age, fine clothing) vs strong
  (gender, bags). This is the **motivation** for adding the 4 modules.

Saves `features/feature_pca.png` — a PCA scatter of features colored by gender (proof the
features are meaningful).

---

## 4. Try single images (YES/NO)  →  MVP

```bash
# web interface: upload an image, see YES/NO for all 26 attributes
python3 app.py            # then open http://127.0.0.1:7860   (Ctrl+C to stop)

# command line on one image
python3 mvp_par.py --image test_crop.jpg --groups
```

---

## Phase-1 deliverables checklist
- [ ] Preprocessing working (square-pad → 224 → normalize) — built into all scripts
- [ ] `features/*_feats.npy` — cached visual features (shapes printed)
- [ ] Baseline `mA` + 4 instance metrics (from `evaluate_zeroshot.py`)
- [ ] `features/feature_pca.png` — feature visualization
- [ ] A few YES/NO predictions from the MVP / UI

## Files
| File | Purpose |
|---|---|
| `attributes.py` | the 26 attributes + their text prompts + exclusive groups |
| `mvp_par.py` | command-line attribute prediction (YES/NO) |
| `app.py` | web UI: upload image → YES/NO table |
| `extract_features.py` | cache frozen SigLIP-2 features for PA-100K |
| `evaluate_zeroshot.py` | baseline mA/F1 + PCA feature plot |

## Common issues
- **`annotation not found`** → check the path; pass `--ann path/to/annotation.mat` explicitly.
- **`image dir not found`** → pass `--img_dir path/to/images`. The script auto-handles a nested
  `release_data/release_data/` folder.
- **Model re-downloading** → it shouldn't after the first complete download; if interrupted,
  run `hf download google/siglip2-base-patch16-224` once to finish it.
- **Slow on CPU** → fine for `--splits test`; for the full train split a GPU is much faster.

---

**Next after Phase 1:** send the baseline `mA` number — then we start Day 4 (the CMAA module).
