# jefworks-structure_segmentation
# Deep Literature Review: Segmentation of Renal Structures


A curated deep literature review and resource hub for the semantic segmentation of kidney structures (glomeruli, tubules, vessels), tracking the evolution from heuristic methods to gene-informed Foundation Models.

## Overview

Segmentation of kidney pathology is a "Holy Grail" task in computational pathology. This repository synthesizes the transition of State-of-the-Art (SOTA) methods through four distinct eras:

1.  **The U-Net Era** (CNN baselines)
2.  **The Competition Era** (Data engineering & HuBMAP)
3.  **The Transformer Era** (Attention mechanisms & Global Context)
4.  **The Multimodal Frontier** (Integrating Gene Expression/Spatial Transcriptomics)

---

## The Biological Challenge

To understand the algorithms, one must understand the morphology:

* **The Target:** Glomeruli (tufts of capillaries encases in Bowman's capsule) and Tubules.
* **The "Borderless" Problem:** While healthy glomeruli have distinct boundaries, **sclerotic (scarred) glomeruli** collapse and lose the Bowman's capsule, making them nearly indistinguishable from interstitial fibrosis.
* **Stain Variance:** Models trained on PAS often fail on H&E, Silver, or Trichrome due to domain shift.
* **Scale:** Glomeruli are large (requires 5x-10x context), while peritubular capillaries are tiny (requires 40x).

---

## Phase I: The U-Net Baseline (2015-2019)

The gold standard for biomedical image segmentation. Despite newer architectures, a well-tuned U-Net remains the strongest baseline.

| Architecture | Key Innovation | Limitation |
| :--- | :--- | :--- |
| **U-Net** | Encoder-Decoder with Skip Connections. | Struggles with large scale variance; local focus only. |
| **Res-U-Net** | Residual blocks (ResNet) to prevent vanishing gradients. | Computationally heavier. |
| **U-Net++** | Nested, dense skip connections. | High memory usage; marginal gains on simple structures. |

> **Key Insight:** Skip connections are non-negotiable for preserving the fine details of the Bowman's space.

---

## Phase II: Engineering & Competitions (HuBMAP)

Lessons learned from the **HuBMAP "Hacking the Kidney"** Kaggle competitions. The winning solutions proved that *data engineering > architecture*.

* **Stain Normalization:** Techniques like *Macenko* or *Reinhard* normalization are mandatory to force all slides into a standard color space.
* **Tiling Strategy:** **Overlap-tile strategy** (predicting center crops and discarding edges) is required to remove stitching artifacts.
* **Pseudo-Labeling:** "Noisy Student" approaches (train on labeled $\rightarrow$ predict on unlabeled $\rightarrow$ retrain on both) yielded the highest leaderboard jumps.

---

## Phase III: Vision Transformers (2021-Present)

Moving beyond CNNs to capture **global context** (e.g., distinguishing a glomerulus from a similar-looking vascular structure based on surrounding tissue location).

* **Swin-U-Net:** Hierarchical Vision Transformer using shifted windows. Excellent for *sclerotic* glomeruli where local boundaries are missing but context implies a scar.
* **TransU-Net:** Hybrid CNN-Transformer. Uses CNN for feature extraction and Transformers for encoding global context.
* **HookNet:** Multi-resolution approach (two branches: context vs. target) that mimics a pathologist zooming in and out.

---

## Phase IV: Foundation Models (2023+)

Adapting massive pre-trained models to nephrology via **Parameter-Efficient Fine-Tuning (PEFT)**.

* **SAM (Segment Anything Model):** Out-of-the-box performance is weak on histology.
* **MedSAM / SAM-Adapter:** Freezes the massive SAM weights and trains a lightweight "adapter" layer. Achieves SOTA with few-shot learning (e.g., 10 examples).
* **SegGPT:** Treats segmentation as an in-context visual prompting task ("Here is one marked glomerulus; find the rest").

---

## Gene-Informed Segmentation (Multimodal)

The cutting edge: integrating **Spatial Transcriptomics (ST)** to define boundaries invisible to the human eye (e.g., Proximal S1 vs. S2 tubules).

### 1. Gene-to-Segmentation (Ground Truth Generation)
* **Proseg / Baysor:** Uses RNA transcript density to probabilistically define cell borders.
    * *Use Case:* Segmenting immune infiltration in renal tumors where T-cells blend into cancer cells.
* **GIST (Graph-Integration):** Fuses visual features with gene expression graphs to identify "functional domains" (e.g., hypoxic regions).

### 2. Image-to-Gene (Virtual Staining)
* **ST-Net / HE2RNA:** Predicts gene expression heatmaps (e.g., *PODXL* or *NPHS1*) directly from H&E images.
    * *Use Case:* Implicitly learns perfect segmentation of podocytes without human annotation by trying to predict the molecular signal.

---

## Datasets

1.  **[HuBMAP (Human BioMolecular Atlas Program)](https://hubmapconsortium.org/)**: The premier dataset. Contains paired PAS/H&E and Spatial Transcriptomics (Visium/CODEX).
2.  **[PANDA (Prostate cANcer graDe Assessment)](https://www.kaggle.com/c/prostate-cancer-grade-assessment)**: Useful for studying tiling/normalization pipelines standard in kidney work.
3.  **[Kidney Pathology Challenge (KPC)](https://www.google.com/search?q=kidney+pathology+segmentation+dataset)**: Various smaller datasets for glomerulus detection.