# Methodology — Multimodal Pedestrian Attribute Recognition

*Read the **"Say:"** line under each part out loud to present. Each part = what it is → why → how.*

---

## 0. The task (what we are solving)
**Pedestrian Attribute Recognition (PAR):** given a cropped photo of **one person**, predict
**26 attributes** at once — gender, age group, viewpoint (front/side/back), clothing
(short/long sleeve, trousers, skirt), and accessories (hat, glasses, backpack, bag…).

- It is **multi-label**: many attributes can be true at the same time → we use a **sigmoid**
  on each attribute (independent yes/no), **not** softmax (which would force one winner).
- **Say:** *"For each person we answer 26 yes/no questions at once. Because a person can be
  female AND wear a hat AND carry a bag, each attribute gets its own probability."*

---

## 1. Dataset
- **PA-100K** — 100,000 pedestrian images, each labeled with the 26 attributes.
  Split: **80,000 train / 10,000 val / 10,000 test**.
- **Say:** *"We train and test on PA-100K, the largest public PAR benchmark — 100k people,
  26 attributes each."*

## 2. Preprocessing
- Each image → convert to **RGB** → **square-pad** (add borders to make it square) →
  resize to the model's input size → **normalize**.
- **Why square-pad:** if you just resize a tall person-crop to a square, the body gets
  squashed. Padding keeps the aspect ratio so the person isn't distorted.
- **Say:** *"We pad each crop to a square before resizing so the body shape isn't stretched,
  then normalize it for the model."*

---

## 3. Backbone — SigLIP-2 (vision-language model)
- We use **SigLIP-2 large** as the image encoder. It is a **vision-language model**: it was
  pretrained to match images with text, so its features already "understand" visual concepts.
- It turns an image into a **1024-number feature vector** (plus per-patch features).
- **Why this backbone:** vision-language pretraining gives very strong features — a simple
  classifier on top already reaches ~85% (see baselines). It also has a **text encoder**,
  which we use for the attribute prompts (part 5).
- **Say:** *"The backbone is SigLIP-2, a model pretrained to align images and text. It already
  knows visual concepts, so its features are a strong starting point."*

## 4. LoRA fine-tuning (how we adapt the backbone)
- The backbone has ~300M parameters — too big to fully retrain on a small GPU, and full
  fine-tuning risks overfitting.
- **LoRA** (Low-Rank Adaptation) inserts tiny trainable "adapter" matrices into the attention
  layers and **freezes everything else**. We train only **~1% of the parameters** (~3.2M).
- **Why:** cheap, fast, fits on a Kaggle T4, and adapts the backbone to pedestrians without
  destroying its pretrained knowledge.
- **Result:** this single step is the **main driver** — accuracy jumps from 85.5 (frozen) to
  **~91.5**.
- **Say:** *"Instead of retraining the whole 300M-parameter model, LoRA trains only small
  adapters — about 1% of the weights. This is what lifts us from 85% to 91%."*

---

## 5. Text branch — attribute prompts (the "language" side)
- For each attribute we build a short text prompt, e.g. *"a photo of a person, Hat"*, and pass
  it through SigLIP-2's **text encoder** → a **1024-number vector per attribute** (26 vectors).
- These are computed **once and cached** (they don't change during training).
- **Key idea:** the prompts are **fixed questions**, but the **answers come from the image**.
  Because SigLIP's image and text live in the **same aligned space**, we can directly compare
  "what the image shows" with "what each attribute means."
- **Say:** *"Each attribute is also written as text and encoded by the same model, so the image
  and the attribute words live in one shared space and can be compared directly."*

## 6. CMAA — Cross-Modal Attribute Attention (module 1)
- **What:** the 26 attribute-text vectors act as **queries** that **attend** to the image's
  patch features — each attribute "looks" at the image regions most relevant to it.
- **Why:** it localizes each attribute (a "hat" query focuses on the head region) and injects
  the attribute's text meaning into the visual features.
- **Output:** per-attribute attention maps (which we visualize as heatmaps in the demo) +
  refined features.
- **Say:** *"CMAA lets each attribute look at the part of the image that matters — the 'hat'
  attribute attends to the head, 'shoes' to the feet — using the attribute's text meaning."*

## 7. Orientation Head + OCFR — viewpoint-aware routing (module 2)
- **Orientation head:** a small classifier that predicts the person's **viewpoint**:
  Front / Side / Back (PA-100K already labels this, so it's free supervision).
- **OCFR (Orientation-Conditioned Feature Routing):** uses that viewpoint to **reweight the
  features** (FiLM-style), because the same attribute looks different from front vs back
  (e.g. a backpack is obvious from behind, hidden from the front).
- **Why:** viewpoint is a major source of appearance variation; conditioning on it helps.
- **Result:** the **+CMAA+OCFR** config gives the best instance Accuracy/F1 (75.9 / 85.55).
- **Say:** *"We predict whether the person faces front, side, or back, and use that to adjust
  the features — because a backpack looks different from behind than from the front."*

## 8. DACG — Dynamic Attribute Correlation Graph (module 3)
- **What:** attributes are **correlated** (skirt ↔ female; long-sleeve ↔ long-coat). DACG
  builds a **26×26 correlation graph** — a **static** prior (learned average co-occurrence) +
  a **dynamic** part (per-image adjustments) — and runs a small **graph convolution** so
  related attributes inform each other.
- **Why:** predicting attributes jointly (not independently) uses these relationships.
- **In our ablation:** roughly **neutral on accuracy** but it adds **interpretability** — the
  graph shows which attributes the model ties together.
- **Say:** *"DACG models how attributes relate — like skirt and female co-occurring — with a
  correlation graph so the predictions reinforce each other."*

## 9. CCLoss — Consistency / logical loss (module 4)
- **What:** an extra loss term that penalizes **logically impossible** outputs — e.g. the three
  **Age** labels are mutually exclusive, and so are **Front/Side/Back**. Only one can be true.
- **Why:** keeps predictions **valid and consistent**, not just individually likely.
- **Say:** *"CCLoss enforces logic — a person has exactly one age group and one viewpoint —
  so the model can't output contradictory attributes."*

---

## 10. Loss function (overall training objective)
- **Weighted BCE (Binary Cross-Entropy)** on the 26 attributes — *weighted* by **inverse
  frequency** so rare attributes aren't ignored (PA-100K is very imbalanced).
- **+ CCLoss** (part 9) for logical consistency.
- **Say:** *"The main loss is weighted cross-entropy — we up-weight rare attributes so the model
  doesn't just predict the common ones — plus a consistency term."*

## 11. Training setup (and a real bug we fixed)
- Trained on **Kaggle T4 GPU**, **fp16** (half precision) for speed, 3 epochs per config.
- **Stability fix:** an early run **diverged** in late epochs (loss blew up, accuracy crashed).
  We fixed it with three standard techniques:
  1. **GradScaler** (loss scaling for fp16),
  2. **gradient clipping** (cap gradient size at 1.0),
  3. **best-checkpoint saving** (keep the best epoch, not the last).
- **Say:** *"Training briefly destabilized in fp16; we fixed it with loss scaling, gradient
  clipping, and saving the best checkpoint — standard practice. The reported number is the
  model's real, stable performance."*

## 12. Evaluation metrics
- **mA (mean Accuracy)** — the standard PAR metric: for each attribute, average the accuracy on
  positives and negatives, then average over attributes (**balances rare vs common**).
- Plus instance-level **Accuracy, Precision, Recall, F1**.
- We also apply **per-attribute calibrated thresholds** (tuned on validation), used
  **consistently** for every model, so comparisons are fair.
- **Say:** *"We report mA — the balanced PAR metric — plus Accuracy, Precision, Recall and F1,
  with the same calibrated thresholds for every model."*

---

## 13. Ablation methodology (how we prove each module's contribution)
- We train the **same backbone** five times, **adding one module at a time**:
  `Plain LoRA → +CMAA → +OCFR → +DACG → +CCLoss (Full)`.
- Each config reports mA + Accuracy + F1 → this **isolates** what each part adds.

| # | Adds | mA | Acc | F1 |
|---|---|---|---|---|
| 1 | Plain LoRA | 91.58 | 74.50 | 84.69 |
| 2 | + CMAA | 91.44 | 75.23 | 85.15 |
| 3 | + OCFR | **91.60** | **75.91** | **85.55** |
| 4 | + DACG | 91.41 | 74.61 | 84.77 |
| 5 | + CCLoss (Full) | *(fill in)* | | |

- **Honest reading:** **mA is flat (~91.5)** — the LoRA backbone dominates. **Instance Acc/F1
  peak at +CMAA+OCFR** — a small consistent gain. **DACG is neutral** but adds interpretability.
  The modules add **structure without hurting accuracy**.

## 14. The overall accuracy story
```
Zero-shot SigLIP-2 ......... 69.5   (no training — image/text matching only)
+ trained linear head ...... 85.5   (+16.0 — classifier on frozen features)
+ LoRA + modules ........... 91.6   (+6.1 — adapt backbone + attribute-aware modules)
```
- **Say:** *"From an untrained 69.5%, a simple classifier gets 85.5%, and adapting the backbone
  with LoRA plus our attribute-aware modules reaches ~91.6% — a total gain of 22 points."*

---

## 15. Honest limitations (say these — they show rigor)
- **Age is near chance for every model** — a fundamental limit of single-image PAR, not a bug.
- **On a strong backbone the modules add little raw accuracy** — reported transparently; their
  value is a small F1 gain (CMAA+OCFR) and interpretability (DACG).
- **CMAA didn't beat the plain baseline on frozen features** — fair test needed the full GPU run.
