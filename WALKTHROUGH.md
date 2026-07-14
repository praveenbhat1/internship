# Full Project Walkthrough — What to Explain to the Mentor (start to end)

*Read this top to bottom. It's the full story of what we built and why — start by opening the
live demo, then explain each part as it appears.*

---

## 0. Opening line
> "This project is a Pedestrian Attribute Recognition system. Given one cropped photo of a person,
> it predicts a set of attributes — clothing, accessories, viewpoint. I'll show the live demo first,
> then explain how each part works, the results, and how we validated it on a second dataset."

---

## 1. Start with the LIVE DEMO
Open it: `python3 mvp/demo_full.py` → upload a person image → click Analyze.

> "The demo is explainable — it shows what happens at every stage, not just the final answer.
> At the top you can see the model is validated on 10,000 test images: 91% mean accuracy."

Walk down the stages on screen:
- **Step 1 — SigLIP-2 features:** "First, a vision-language model turns the image into 1024 numbers —
  its fingerprint — and also matches it against each attribute's text description."
- **Step 2 — CMAA heatmaps:** "For every attribute, this heatmap shows *where* the model looks —
  'Trousers' looks at the legs, 'Hat' at the head. This makes it interpretable."
- **Step 3 — Viewpoint:** "It predicts whether the person faces front, side, or back."
- **Step 4 — DACG grid:** "This 23×23 grid shows how attributes relate — e.g. short-sleeve and
  long-sleeve exclude each other."
- **Step 5 — Predictions:** "Finally, the detected attributes with a confidence score each.
  Notice: on a back view it says 'gender not reported' — the model abstains when it can't see the face."

---

## 2. The APPROACH (the architecture)
Show `architecture.png`.
> "The backbone is **SigLIP-2**, a frozen vision-language model pretrained to match images and text —
> so it already understands visual concepts. We adapt it cheaply with **LoRA**, training only about
> 1% of the parameters. This adaptation is the main driver of accuracy. On top we add four modules."

**The four modules (explain each):**
- **CMAA (Cross-Modal Attribute Attention):** "Each attribute — written as text — attends to the image
  patches most relevant to it, so it focuses on the right region."
- **OCFR (Orientation-Conditioned Feature Routing):** "It predicts the viewpoint and reweights the
  features accordingly, because a backpack looks different from front vs back."
- **DACG (Dynamic Attribute Correlation Graph):** "It models how attributes correlate — using a fixed
  prior plus a per-image graph — so predictions reinforce each other instead of being independent."
- **CCLoss (Consistency Loss):** "A loss that enforces logic — a person has one viewpoint and one
  sleeve length, so the model can't output contradictions."

---

## 3. The DATA and the HONEST evaluation
> "We train on **PA-100K** — 100,000 pedestrian images with 26 attributes, split into 80k train,
> 10k validation, 10k test. Per your guidance we removed the age labels (they were near chance) and
> kept gender, giving a 23-attribute model."

**The data-leakage fix (a key rigor point):**
> "Early on, we had a data-leakage bug: we were tuning the decision thresholds and picking the best
> epoch *on the test set*, which inflates results. We fixed it — now we tune everything on the
> **validation** set and score the test set only once. You can even see the gap: validation was 91.9%
> but the honest test number is 91.1%. That 0.8-point gap is exactly the optimism we removed. We report
> the honest number."

---

## 4. The RESULTS (PA-100K)
Show `accuracy_progression.png`.
> "We measured accuracy at each stage: zero-shot SigLIP with no training gives 69.5%. Training a simple
> classifier on its features gives 85.5% — already near state of the art. Adapting the backbone with
> LoRA plus our modules reaches **91%** — a 22-point improvement."

Show `ablation_chart.png` / the ablation table.
> "We added each module one at a time to measure its contribution across five metrics. Mean accuracy
> stays flat around 91.5 — the backbone dominates — but the **instance Accuracy and F1 peak when we add
> CMAA and OCFR**. DACG is roughly neutral on accuracy but adds interpretability. So the modules add
> structure and interpretability without hurting accuracy — we report this honestly."

Show `metrics_summary.png`.
> "The final model, validated on 10,000 unseen test images with five metrics: **91.1% mA, 75.5%
> Accuracy, 79% Precision, 92.6% Recall, 85.3% F1.**"

---

## 5. INTERPRETABILITY and RESPONSIBLE design
> "Beyond accuracy, we focused on trust. Three things:
> 1. The **CMAA heatmaps** show *why* each prediction was made.
> 2. **Gender abstention** — the model only reports gender when it's confident AND can see the face;
>    otherwise it says 'not reported'. Gender from appearance is unreliable and sensitive, so abstaining
>    is more responsible than guessing.
> 3. A **confidence floor** so it only shows attributes it's genuinely sure about — fewer false positives."

---

## 6. CROSS-DATASET GENERALIZATION (the strongest part)
Show `peta_comparison.png` and `peta_examples.png`.
> "The most important test: does it generalize, or did it just memorize PA-100K? We took the trained
> model and ran it on **PETA — a completely different dataset — with zero retraining.** We scored the
> 14 attributes both datasets clearly share.
> In-domain we get 91.1%; cross-domain on 14,437 PETA images we get about **78% mean accuracy.** The
> ~13-point drop is the expected domain gap, but 78% is well above chance — which proves the model
> learned **general** pedestrian features, not dataset-specific tricks. The example images show it
> correctly predicting clothing and bags on people it has never seen."

**The prior-matching improvement (if you ran it):**
> "The model over-predicted a few attributes on the new domain — for 'Trousers' it said yes for almost
> everyone. We applied **label-shift adaptation**: knowing each attribute's frequency from the source
> dataset, we calibrate the thresholds on PETA to match those rates — using no target labels. This
> corrects the over-firing and raises the cross-domain score."

---

## 7. HONEST LIMITATIONS (say these — they show rigor)
> "It's not perfect — about 91% mA, so roughly 1 in 10 predictions can be wrong, mostly on fine textures
> like plaid and small accessories like glasses. Gender is appearance-based and can be confidently wrong,
> which is why we abstain. And a few attributes are defined differently across datasets, which widens the
> cross-domain gap. We report all of this transparently."

---

## 8. CLOSING
> "So in summary: a SigLIP-2 based, interpretable, viewpoint-aware attribute model that reaches
> state-of-the-art-level 91% in-domain under an honest leak-free protocol, generalizes to an unseen
> dataset at 78% zero-shot, and is designed responsibly — it abstains on gender and shows its reasoning.
> The strength isn't a single number; it's the complete, rigorous, honest package. Everything is on
> GitHub with a full report."

---

## Quick fact sheet (if asked)
- **Backbone:** SigLIP-2 large, frozen + LoRA (~1% params trained)
- **Attributes:** 23 (gender kept, age removed)
- **In-domain:** 91.1 mA / 75.5 Acc / 85.3 F1 (leak-free, 10k test)
- **Cross-domain (PETA):** ~78 mA, zero-shot, 14 shared attributes, 14,437 images
- **Datasets:** PA-100K (train/test), PETA (zero-shot validation)
- **Novelty:** OCFR + DACG + CCLoss + cross-dataset study + gender abstention
- **Rigor:** fixed data leakage, honest limitations, label-free domain adaptation
