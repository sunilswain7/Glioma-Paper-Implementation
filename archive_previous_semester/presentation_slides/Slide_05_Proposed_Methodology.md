# Slide 5: Proposed Methodology in the Paper

## Overall Architecture / Framework

The paper proposes a **three-phase framework:**

### Phase 1: Model Development
```
4-channel MRI Input (T1, T1ce, T2, FLAIR)
         |
    [mmFormer Tumor Segmentation]
         |
    [3D ResNet-10 Shared Backbone]
         |
    512-dimensional shared features
         |
    ┌────┼────┬────┐
    |    |    |    |
  IDH  1p/19q Grade Survival
  head  head  head  branch
  (512→2) (512→2) (512→2) (512→64→16)
                          |
                    concat with task scores
                          |
                    Deep Prognosis Score (DPS)
```

### Phase 2: Model Assessment
- ROC curves, confusion matrices for classification tasks
- Kaplan-Meier survival curves for DPS stratification
- Calibration curves for probability reliability
- Validated on 3 external cohorts (TCGA, UCSF, EGD)

### Phase 3: Biological Interpretation
- Radio-multiomics analysis on TCGA data
- Gene Set Enrichment Analysis (GSEA)
- Immune infiltration profiling (ESTIMATE, ssGSEA)
- DNA methylation and proteomics analysis

## Main Algorithms and Techniques

1. **3D ResNet-10:** Residual neural network with skip connections for feature extraction from 3D MRI volumes. Uses 4 residual blocks with batch normalization and ReLU activation

2. **Multi-task Learning:** Shared backbone extracts common features; task-specific fully connected heads produce independent predictions. Total loss = L_IDH + L_1p19q + L_grade + L_survival

3. **Gradient Detachment (1p/19q):** The 1p/19q classification head receives gradient-detached features — its loss doesn't update the shared backbone. This prevents conflicting gradients since 1p/19q is biologically linked to IDH

4. **Cox Partial Likelihood Loss:** For survival prediction, the model uses the Cox proportional hazards framework. It learns to rank patients by risk without assuming a specific survival distribution

5. **Deep Prognosis Score (DPS):** A novel composite score combining:
   - 16 deep prognosis features (from 512→64→16 reduction)
   - Softmax scores from all 3 classification tasks (6 values)
   - Final mapping: 22 → 32 → 1 scalar risk score

## Workflow of the Proposed System
```
Step 1: Acquire multi-modal MRI (T1, T1ce, T2, FLAIR)
Step 2: Automated tumor segmentation via mmFormer
Step 3: Normalize images (z-score) and resize to 88x112x88
Step 4: Phase 1 training — train classification heads (IDH, 1p/19q, Grade)
Step 5: Phase 2 training — freeze classification, train survival branch
Step 6: Compute DPS for each patient
Step 7: Stratify patients into low-DPS and high-DPS groups
Step 8: Validate across external cohorts
Step 9: Multi-omics analysis for biological interpretation
```
