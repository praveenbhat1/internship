# References & Paper-Writing Guide

## How to write your paper (structure + which papers to model it on)
Use a standard vision-paper structure. Two good **templates to model your methodology + diagram on**:
- **VTB** (Cheng et al., 2022) — a clean visual–textual PAR baseline; good template for a
  method figure + ablation table.
- **Rethinking of PAR** (Jia et al., 2020) — the standard strong-baseline paper; good template
  for datasets, metrics (mA/Acc/Prec/Rec/F1), and honest evaluation protocol.
- **PAR Survey** (Wang et al., 2022) — cite for the problem definition, datasets, and metrics.

**Your paper sections:** Abstract → Introduction → Related Work → Method (with the architecture
figure) → Datasets → Experiments (in-domain results + ablation) → Cross-Dataset Study (PETA) →
Limitations → Conclusion. (`FINAL_REPORT.md` already follows this — reuse it as your draft.)
Architecture figure: `mvp/architecture.png` (from `build_architecture.py`).

## Key references to cite
**Backbone & adaptation**
1. Zhai, Mustafa, Kolesnikov, Beyer. *Sigmoid Loss for Language Image Pre-Training (SigLIP).* ICCV 2023.
2. Tschannen et al. *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic
   Understanding, Localization, and Dense Features.* 2025. (your backbone)
3. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
4. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. (your fine-tuning)

**Module inspirations**
5. Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018. (basis for OCFR)
6. Chen et al. *Multi-Label Image Recognition with Graph Convolutional Networks (ML-GCN).* CVPR 2019.
   (basis for DACG attribute-correlation graph)

**Pedestrian Attribute Recognition**
7. Wang, Zhang, Gao, Wang, Shen, Tan. *Pedestrian Attribute Recognition: A Survey.* Pattern Recognition, 2022.
8. Li, Chen, Huang. *Multi-attribute Learning for Pedestrian Attribute Recognition in Surveillance
   Scenarios (DeepMAR).* ACPR 2015.
9. Cheng et al. *A Simple Visual-Textual Baseline for Pedestrian Attribute Recognition (VTB).*
   IEEE TCSVT, 2022.
10. Jia, Huang, et al. *Rethinking of Pedestrian Attribute Recognition: Realistic Datasets with
    Efficient Method.* arXiv 2020.

**Datasets**
11. Liu et al. *HydraPlus-Net: Attentive Deep Features for Pedestrian Analysis.* ICCV 2017. (introduces **PA-100K**)
12. Deng et al. *Pedestrian Attribute Recognition At Far Distance.* ACM Multimedia 2014. (introduces **PETA**)

> Verify exact authors/pages on Google Scholar before final submission — the titles/venues above
> are correct; format them to your required citation style (IEEE/APA).
