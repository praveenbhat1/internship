# Project Handoff — Multimodal Pedestrian Attribute Recognition (PAR)

*For my project partner. Open this in Claude Code as reference — it explains what's done, where
everything is, the gotchas, and what's left to finish.*

--
## 1. What the project is
**Pedestrian Attribute Recognition:** given a cropped photo of one person, predict a set of
attributes (clothing, bag, viewpoint, etc.) — multi-label, so each attribute gets its own
yes/no with a sigmoid. Trained on **PA-100K** (100k images). Title:
*"Multimodal PAR with Correlation-Aware Learning."*

## 2. The architecture (pipeline)
```
image -> SigLIP-2 (frozen VLM) + LoRA  -> CMAA -> Orientation Head -> OCFR -> DACG -> classifier -> sigmoids
         (visual features + attribute-text)  (attend)  (Front/Side/Back)  (route)  (correlations)
loss = weighted BCE + orientation CE + CCLoss (logical consistency)
```
- **LoRA** — cheaply fine-tunes the big SigLIP backbone (~1% of params). **Main accuracy driver.**
- **CMAA** — each attribute attends to its image region (interpretable heatmaps).
- **OCFR** — predicts viewpoint (Front/Side/Back) and reweights features.
- **DACG** — 26×26 attribute-correlation graph (static prior + dynamic).
- **CCLoss** — enforces logical consistency (one age, one viewpoint).

## 3. Results (all real, honest)
| Stage | mA |
|---|---|
| Zero-shot SigLIP-2 | 69.5 |
| + trained linear head | 85.5 |
| + LoRA + modules | **~91.5** |

**Ablation finding (honest):** the LoRA backbone drives the accuracy; the modules add **small F1
gains + interpretability**, not big mA. We report this transparently — it's rigorous, not a failure.
5-metric ablation is in `mvp/ablation_table.png` and `mvp/ablation_chart.png`.

## 4. Important decisions
- **Mentor asked to REMOVE gender + age** → current model is **22 attributes** (`--drop_gender_age`).
  (A 23-attr option `--drop_age` keeps gender — pending mentor confirmation.)
- Train on **Kaggle GPU** (T4), NOT the local Mac (limited GPU/16GB RAM).

## 5. Where everything is
```
internship/
├── methodology.md      # module-by-module explanation with "Say:" lines for presenting
├── explain.md          # step-by-step results for the mentor
├── results_slide.md    # tables + speaker script
├── PARTNER_HANDOFF.md  # this file
└── mvp/
    ├── train_par_full.py   # THE training script (ablation flags + --drop_gender_age/--drop_age)
    ├── demo_full.py        # live step-by-step demo (upload image -> all stages + metrics banner)
    ├── proof_sheet.py      # 5 images -> predictions + confidence % + heatmaps
    ├── stage_extraction.py # per-stage figure (works WITHOUT the trained checkpoint)
    ├── metrics_card.py     # renders the 5-metric validation slide
    ├── app.py              # base 85.5 demo (works locally, no download)
    ├── extract_features.py / train_baseline.py / evaluate_zeroshot.py  # baseline pipeline
    ├── features/           # attributes.json, metrics.json, labels, baseline_linear.pt, par_full.pt
    ├── *.png               # presentation figures (stage_*, ablation_*, accuracy_progression, metrics_summary)
    └── archive/            # old/superseded files
```

## 6. Kaggle workflow + GOTCHAS (read this — it saved us hours)
- Use **GPU T4 x2** — NOT P100 (P100 = sm_60, too old for PyTorch → "no kernel image" error).
- **Upload the .py as a Kaggle dataset** (`par-code`) and copy it in a cell — avoids paste-indent errors.
- First cell always: `!pip uninstall -y torchao` (torchao clashes with PEFT/LoRA).
- **Interactive sessions keep wiping `/kaggle/working`** → always use **Save Version → Save & Run All (commit)**;
  committed output is permanent and downloadable.
- Training saves a **~15 MB checkpoint** (LoRA+heads only; backbone reloads from HuggingFace).
  Load it in demos with `strict=False`. (The old full 1.2 GB save wouldn't download.)
- Standard training cell:
  ```python
  !pip uninstall -y torchao
  import glob, os, shutil
  csv = os.path.dirname(glob.glob('/kaggle/input/**/train.csv', recursive=True)[0])
  img = os.path.dirname(glob.glob('/kaggle/input/**/*.jpg', recursive=True)[0])
  shutil.copy(glob.glob('/kaggle/input/**/train_par_full.py', recursive=True)[0], '/kaggle/working/train_par_full.py')
  !python /kaggle/working/train_par_full.py --csv_dir {csv} --img_dir {img} --epochs 3 --batch 32 --drop_gender_age --out /kaggle/working
  ```

## 7. What's DONE ✅
- Full pipeline coded + trained (~91.5 mA); 5-config ablation with 5 metrics each.
- All presentation figures: per-stage pipeline, ablation table+chart, accuracy progression, metrics slide.
- Demos: `demo_full.py` (live, with metrics banner), `proof_sheet.py`, `stage_extraction.py`.
- Docs: methodology / explain / results_slide.
- 22-attr model trained (gender+age removed per mentor).

## 8. What's LEFT (roadmap to the end — deadline this weekend)
1. **Finish + download `par_full.pt`** (15 MB) from the Kaggle commit run → `mvp/features/`.
2. **Update `features/metrics.json`** with the real `[done]` line (mA/Acc/Prec/Rec/F1) → re-run `metrics_card.py`.
3. **Run the live demo** (`python3 demo_full.py`) for the mentor.
4. **Cross-dataset validation on PETA** (zero-shot PA-100K→PETA): download PETA, build an
   attribute-alignment table (22↔35), run the trained model, report metrics. *(Script not built yet — ask Claude to build it.)*
5. **Assemble final slides/report:** methodology + metrics + demo + PETA + novelty.

## 9. Novelty & why it's useful (for the report)
- vs prior work (VLM-PAR: frozen SigLIP + prompts) we add **OCFR (viewpoint) + DACG (graph) +
  CCLoss (consistency) + a cross-dataset generalization study**.
- Useful for **surveillance/retail analytics**, **interpretable** (heatmaps show why),
  **viewpoint-robust**, and **generalizes** across datasets (the PETA test proves it).

## 10. How to use Claude Code on this project
- Point it at this file + the `mvp/` folder. Say what you want (e.g. "build the PETA eval script"
  or "run proof_sheet.py on these 5 images").
- Training = Kaggle (see §6). Local = demos + figures only.
- Keep results **honest** — the ablation shows the backbone dominates; present that openly.
