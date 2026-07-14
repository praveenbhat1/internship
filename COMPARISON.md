# Comparison with Prior Work (PA-100K)

## Published PA-100K results vs ours
| Method | Year | Backbone | mA (%) |
|---|---|---|---|
| DeepMAR | 2015 | CaffeNet | 72.7 |
| HP-Net (HydraPlus) | 2017 | Inception | 74.2 |
| VAC | 2019 | ResNet-50 | 79.2 |
| ALM | 2019 | BN-Inception | 80.7 |
| JLAC | 2020 | ResNet-50 | 82.3 |
| VTB | 2022 | ViT-B | 83.7 |
| PARFormer | 2023 | Transformer | ~84.5 |
| **Ours (full model)** | 2026 | **SigLIP-2 + LoRA + CMAA + OCFR + DACG + CCLoss** | **~91** |

> ⚠️ Prior numbers are **commonly-cited approximate values** from the papers / the PAR survey
> (Wang et al., 2022). **Verify exact figures from the original papers** before submitting.
> Figure: `mvp/comparison_sota.png`.

## How much did we improve?
- **vs classic baseline (DeepMAR 72.7):** +18 mA
- **vs recent transformer SOTA (VTB 83.7):** +7 mA
- **Within our own project:** zero-shot 69.5 → linear 85.5 → full model 91.1 (**+22 mA**)

## ⚠️ Honest caveats (STATE THESE — do not overclaim)
Our ~91 is **not a strict apples-to-apples** with the published numbers, for three reasons:
1. **Attribute set:** ours is **23 attributes** (age removed). Prior work uses the standard **26**.
   Age is near-chance and drags mA down, so removing it *raises* our average — we can't claim to
   "beat" 26-attr methods directly.
2. **Backbone:** we use **SigLIP-2**, a much stronger, more recent vision-language backbone than the
   CNNs/ViTs in older work. Much of the gain comes from the backbone, which we state openly.
3. **Protocol:** ours is **leak-free** (val-calibrated). Some prior numbers may use different
   thresholding/eval protocols.

## What we CAN honestly claim
- Our model is **at the level of / competitive with recent state-of-the-art** PA-100K methods.
- The **modern vision-language backbone (SigLIP-2 + LoRA)** is the main driver of the gain.
- Our distinctive contributions are **interpretability, viewpoint-aware routing, the attribute graph,
  responsible gender abstention, and — most importantly — the cross-dataset generalization study**,
  which most prior PA-100K papers do NOT report.

## 🎤 How to say it to your mentor
> *"Compared to prior PA-100K work — DeepMAR 72.7, ALM 80.7, VTB 83.7 — our model reaches around 91,
> which is at the level of recent state-of-the-art. But I want to be transparent: this isn't a strict
> head-to-head — we use 23 attributes instead of 26 (age removed), a stronger SigLIP-2 backbone, and a
> leak-free protocol. So the fairer statement is that our model is competitive with recent SOTA, the
> modern backbone drives most of the accuracy, and our real contribution is the interpretability and the
> cross-dataset generalization study that prior papers don't do."*
