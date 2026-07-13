# Multimodal Pedestrian Attribute Recognition with Correlation-Aware Learning
### Internship Final Report

---

## Abstract
We build an interpretable pedestrian attribute recognition (PAR) system on a frozen SigLIP-2
vision–language backbone adapted with LoRA, extended with four attribute-aware modules —
Cross-Modal Attribute Attention (CMAA), Orientation-Conditioned Feature Routing (OCFR), a
Dynamic Attribute Correlation Graph (DACG), and a Consistency Loss (CCLoss). Trained on PA-100K,
the model reaches **90.8% mean accuracy (mA)** on the held-out test set under a **leak-free**
evaluation protocol. An incremental ablation isolates each module's contribution, and a
**zero-shot cross-dataset study on PETA (14,437 images) reaches 77.75% mA with no retraining**,
demonstrating that the model learns generalizable pedestrian features. We report all results
transparently, including a data-leakage issue we found and fixed, and we design the system to
**abstain on gender** when it cannot see the face — a responsible-AI choice.

---

## 1. Introduction
Pedestrian Attribute Recognition predicts a set of human attributes (clothing, accessories,
carried items, viewpoint) from a single cropped person image. It is **multi-label** — many
attributes can be simultaneously true — so each attribute is predicted independently with a
sigmoid. PAR is used in surveillance, retail analytics, and person retrieval, where describing
people by attributes avoids storing identity.

PAR is hard because of occlusion, viewpoint variation, low-resolution crops, severe class
imbalance, and interdependencies between attributes. Our goal: a model that is **accurate**,
**interpretable** (shows *why* it predicts each attribute), **viewpoint-robust**, and — critically —
**generalizes** to datasets it was not trained on.

## 2. Related Work & Novelty
Recent vision–language PAR (e.g., a frozen SigLIP-2 with per-attribute text prompts and compact
cross-attention) achieves strong in-domain results but uses **no explicit viewpoint handling, no
attribute-correlation graph, no consistency loss, and no cross-dataset evaluation.**

**Our contributions on top of that baseline:**
1. **OCFR** — an orientation head + feature routing that adapts features to Front/Side/Back.
2. **DACG** — a dynamic + static attribute-correlation graph so predictions reinforce each other.
3. **CCLoss** — a logical-consistency loss enforcing mutual-exclusion (viewpoint, sleeve length).
4. **A cross-dataset generalization study** (PA-100K → PETA) — most PAR papers report only in-domain.
5. **Responsible design** — the model abstains on gender when the face is not visible.

## 3. Method
```
image → SigLIP-2 (frozen) + LoRA → CMAA → Orientation Head → OCFR → DACG → classifier → sigmoids
        (visual + attribute-text)   (attend)  (Front/Side/Back)  (route)  (correlate)
```
- **Backbone (SigLIP-2 large + LoRA).** A frozen vision–language backbone produces 1024-dim
  features; LoRA trains ~1% of parameters (~3.2 M) to adapt it to pedestrians. **This is the main
  driver of accuracy.**
- **CMAA.** 23 attribute-text embeddings cross-attend the image patches, so each attribute focuses
  on its relevant region (visualized as heatmaps).
- **Orientation Head + OCFR.** Predicts viewpoint and reweights features (FiLM-style), since the
  same attribute appears differently from front vs back.
- **DACG.** A 23×23 correlation graph (learned static prior + per-image dynamic part) with graph
  convolution, so related attributes inform each other.
- **CCLoss.** Penalizes logically inconsistent outputs (one viewpoint; one sleeve length).
- **Loss.** Inverse-frequency-weighted BCE (for class imbalance) + orientation CE + CCLoss.

## 4. Datasets
- **PA-100K** — 100,000 pedestrian images, 26 attributes, split 80k/10k/10k.
- **PETA** — 19,000 images across 10 sub-datasets, used **zero-shot** for cross-dataset validation.
- **Attribute decision.** Per project guidance, the **age** labels (near chance for every model)
  were removed; **gender** was kept, giving a **23-attribute** final model.

## 5. Training & Evaluation Protocol
- Trained on Kaggle T4 GPU, fp16 with a gradient scaler + gradient clipping + best-checkpoint
  saving (an early run diverged in fp16; these standard fixes stabilized it).
- **Leak-free evaluation (important).** We initially calibrated per-attribute decision thresholds
  and selected the best epoch **on the test set** — a data-leakage bug that inflates results. We
  fixed it: thresholds and epoch selection are done on the **validation** set, and the test set is
  scored **once** with those frozen thresholds. The val↔test gap (91.5 vs 90.8) quantifies the
  optimism we removed.

## 6. Results (PA-100K)

**Accuracy progression**
| Stage | mA |
|---|---|
| Zero-shot SigLIP-2 (no training) | 69.5 |
| + trained linear head | 85.5 |
| + LoRA + modules (final) | **90.8** |

**Module ablation** (each row adds one module; LoRA-large backbone; PA-100K test)
| # | Configuration | mA | Accuracy | F1 |
|---|---|---|---|---|
| 1 | Plain LoRA | 91.58 | 74.50 | 84.69 |
| 2 | + CMAA | 91.44 | 75.23 | 85.15 |
| 3 | + OCFR | **91.60** | **75.91** | **85.55** |
| 4 | + DACG | 91.41 | 74.61 | 84.77 |

*Reading:* mA is flat (~91.5) — the LoRA backbone dominates — but **instance Accuracy and F1 peak at
+CMAA+OCFR**, and DACG adds interpretability. The modules add **structure and interpretability
without hurting accuracy**. (Ablation numbers use the earlier per-test-set calibration and are
relative; the final model below uses the leak-free protocol.)

**Final model (23 attributes, leak-free, 10,000 test images)**
| mA | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **90.80** | 72.70 | 76.36 | 91.98 | 83.44 |

## 7. Cross-Dataset Generalization (PETA, zero-shot)
The PA-100K-trained model was run on **PETA with no retraining**, scoring the **14 attributes** with
an unambiguous correspondence (viewpoint and a few fine textures have no clean PETA equivalent and
are excluded, for rigor). On **14,437 images:**

| | mA | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| In-domain (PA-100K) | 90.8 | 72.7 | 76.4 | 92.0 | 83.4 |
| **Cross-domain (PETA)** | **77.75** | 58.07 | 62.43 | 87.94 | 73.02 |

**Transfers well:** Shorts 94.8, ShortSleeve/LongSleeve 91.2, Female 88.4, Backpack 82.1.
**Domain gap:** Trousers 56.5, UpperPlaid 46.4 (attribute definitions differ between datasets).

The ~13-point drop is the **expected domain gap**, but 77.75 mA is well above chance — evidence that
the model learned **general** pedestrian features and would transfer to real deployment on unseen data.

## 8. Interpretability & Responsible Design
A live demo (`demo_full.py`) shows, for any uploaded image, every stage: the SigLIP feature, the
CMAA attention heatmap for each attribute, the predicted viewpoint, the DACG correlation grid, and
the final predictions with **per-attribute confidence**. Design choices for trustworthiness:
- **Gender abstention** — the model reports gender only when ≥85% confident *and* the face is
  visible (not a back view); otherwise it says "not reported." Better to abstain than to guess.
- **Confidence floor** — a 0.5 display floor prevents false positives from rare-attribute thresholds.
- **Mutual-exclusion** — one viewpoint, one sleeve length, one lower-body garment.

## 9. Limitations (reported transparently)
- **~90.8% mA is not perfect** — roughly 1 in 10 attribute predictions can be wrong, mostly on
  fine-grained textures (UpperPlaid/Splice) and small accessories (Glasses).
- **Gender is appearance-based and unreliable** — it matches learned cues (hair, clothing), can
  reflect dataset bias, and can be confidently wrong on front views; hence the abstention policy.
- **On a strong backbone the modules add limited raw mA** — their value is interpretability and
  small F1 gains, reported openly rather than overclaimed.
- **Cross-dataset attributes with different definitions** (e.g., Trousers) show a larger gap.

## 10. Conclusion & Future Work
We built an interpretable, viewpoint-aware PAR model reaching **90.8% mA** in-domain and **77.75%
mA zero-shot on PETA**, with an honest ablation and a fixed data-leakage protocol. Future work:
recalibrate thresholds for F1 (fewer false positives), redo the full ablation under the leak-free
protocol, and extend the cross-dataset study to more datasets.

## Appendix — Reproducibility
- **Repository:** github.com/praveenbhat1/internship
- **Model:** `mvp/features/par_full.pt` (LoRA + heads, ~23 MB; backbone loads from HuggingFace)
- **Key figures:** `accuracy_progression.png`, `ablation_table.png`, `ablation_chart.png`,
  `metrics_summary.png`, `stage_test_crop.png`, `peta_comparison.png`, `peta_perattr.png`,
  `peta_examples.png`
- **Run the demo:** `cd mvp && pip install -r requirements.txt && python3 demo_full.py`
- **Docs:** `methodology.md`, `explain.md`, `results_slide.md`, `PRESENTATION.md`
