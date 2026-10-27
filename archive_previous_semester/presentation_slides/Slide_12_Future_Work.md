# Slide 12: Future Work — Problem Statement

## Unsolved Issues in the Current Work

1. **IDH and 1p/19q use pseudo-labels, not real molecular data**
   - Our pseudo-labels approximate clinical distributions but don't capture true per-patient molecular status
   - Real molecular labels require biopsy-confirmed data from clinical records

2. **2D slice processing instead of 3D volumetric processing**
   - We process individual MRI slices, losing inter-slice spatial context
   - Tumor location, 3D shape, and volumetric features are invisible to our model
   - The paper's 3D ResNet-10 captures these features

3. **No biological interpretation (Phase 3 of the paper)**
   - We cannot perform multi-omics validation (GSEA, immune profiling, methylation analysis)
   - This requires matched gene expression, proteomics, and epigenomics data (available in TCGA)

4. **Single-phase training instead of two-phase**
   - The paper trains classification heads first, then freezes them for survival branch training
   - Our simultaneous training may not fully optimize the survival branch

## Limitations Identified During Implementation

1. **GPU memory constraint (4 GB)** prevented 3D volumetric processing — the core methodology of the paper
2. **BraTS dataset lacks molecular markers** — forced use of pseudo-labels for 2 out of 3 classification tasks
3. **No patient-level aggregation** — paper aggregates predictions across 3D volume; we evaluate per-slice
4. **Limited test diversity** — single dataset source vs paper's 5 institution cross-validation
5. **No C-index computation** — paper's primary survival metric not implemented
6. **1p/19q AUC is inflated** — pseudo-labels correlated with grade make this metric unreliable

## Proposed Problem Statement for Future Enhancement

**"End-to-end 3D multi-task deep learning for glioma molecular profiling with real molecular labels and biological interpretability"**

### Specific objectives:
1. Acquire TCGA-LGG + TCGA-GBM datasets with real IDH and 1p/19q molecular labels from clinical records
2. Implement full 3D processing using a GPU with 12+ GB VRAM (e.g., RTX 3080, A100, or cloud GPU)
3. Implement the paper's two-phase training strategy (classification first, then survival)
4. Add C-index and time-dependent AUC (1-year, 3-year, 5-year) for survival evaluation
5. Perform multi-omics analysis on TCGA data to validate biological interpretability of DPS

## Possible Extensions and Research Directions

### Short-term (within semester):
1. **Download raw NIfTI 3D volumes** from TCIA (TCGA-LGG/GBM) and implement true 3D processing on cloud GPU (Google Colab Pro, Kaggle GPU)
2. **Cross-reference BraTS patients with TCGA clinical data** using the name_mapping.csv TCGA IDs to get real IDH/1p/19q labels for overlapping patients
3. **Implement patient-level prediction aggregation** — average slice predictions per patient for more stable results
4. **Add attention mechanism** to highlight which image regions drive each prediction (Grad-CAM visualization)

### Medium-term (next semester):
5. **Vision Transformer (ViT) backbone** — replace ResNet with medical image transformers (e.g., Swin-UNETR) for potentially better feature extraction
6. **Multi-center validation** — test on UCSF-PDGM and EGD datasets to assess cross-institution generalization
7. **Uncertainty quantification** — add Monte Carlo dropout or ensemble methods to provide confidence intervals for predictions

### Long-term (research extension):
8. **Multi-omics integration** — combine MRI predictions with genomics, transcriptomics, and proteomics for comprehensive biological profiling
9. **Treatment response prediction** — extend DPS to predict response to specific therapies (temozolomide, radiation, immunotherapy)
10. **Federated learning** — train across multiple hospitals without sharing patient data, addressing privacy concerns
11. **Clinical deployment** — develop a web-based tool where clinicians can upload MRI and receive molecular predictions with interpretability reports
