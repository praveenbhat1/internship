# PAR MVP — Pedestrian Attribute Recognition

Multimodal PAR: **SigLIP-2 + LoRA → CMAA → OCFR → DACG → classifier**, trained on PA-100K.
Best model ~91% mA. This folder holds the working pipeline, demos, and presentation figures.

## 🎬 Demos (run these for the mentor)
| File | What it does | Run |
|---|---|---|
| `demo_full.py` | Live step-by-step demo — upload an image, see every stage (SigLIP → CMAA → OCFR → DACG → prediction) | `python3 demo_full.py` |
| `proof_sheet.py` | 5 images → predictions + confidence % + CMAA heatmaps → `proof_sheet.png` | `python3 proof_sheet.py` |
| `stage_extraction.py` | Per-stage figure (works WITHOUT the trained checkpoint) | `python3 stage_extraction.py img.jpg` |
| `app.py` | Base 85.5 demo (YES/NO, runs locally — no checkpoint download needed) | `python3 app.py` |

> `demo_full.py` / `proof_sheet.py` need `features/par_full.pt` (the 15 MB trained checkpoint — download from Kaggle).

## 🏋️ Training (run on Kaggle GPU, not the Mac)
| File | Model |
|---|---|
| `train_par_full.py` | Full pipeline (CMAA/OCFR/DACG/CCLoss, ablation flags, `--drop_gender_age` / `--drop_age`). Saves a ~15 MB checkpoint. |
| `extract_features.py` | Cache frozen SigLIP features (regenerates the deleted `.npy` caches if needed) |
| `train_baseline.py` | Trained linear head on frozen features (85.5 baseline) |
| `evaluate_zeroshot.py` | Zero-shot baseline (69.5) |

## 📊 Presentation figures
| File | Shows |
|---|---|
| `stage_test_crop.png`, `stage_quadrant.png` | Full per-stage pipeline on real images |
| `ablation_chart.png` | Each module's contribution (mA / Accuracy / F1 bars) |
| `ablation_table.png` | 5-metric ablation table (mA/Acc/Prec/Rec/F1 per module) |
| `accuracy_progression.png` | 69.5 → 85.5 → 91.6 story |
| `features_overview.png`, `training_curve.png` | Feature PCA + baseline training curve |

## 📁 features/
- `attributes.json` — attribute name list (overwrite with the 22/23-attr version from the matching model)
- `par_full.pt` — **download from Kaggle** (the trained model; loaded with `strict=False`, backbone re-fetched from HF)
- `baseline_linear.pt` — the 85.5 baseline head
- `*_labels.npy`, `*_names.npy` — PA-100K labels/filenames (used for DACG correlation + eval)
- `cmaa.pt`, `cmaa_spatial.pt`, `feature_pca.png` — earlier CMAA experiment records

## 🗄️ archive/
Superseded scripts, old notebooks, and old figures — kept for reference, not needed for the final work.

## Docs (in `../` internship/)
`methodology.md` (module-by-module "Say:" lines), `explain.md` (step-by-step), `results_slide.md` (tables + script).
