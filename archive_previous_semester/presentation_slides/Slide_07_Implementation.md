# Slide 7: Implementation Details

## Modules Developed

### Module 1: Data Pipeline
- **Metadata loading:** Reads name_mapping.csv (grade), survival_info.csv (OS data), meta_data.csv (slice info)
- **Pseudo-label generation:** Creates IDH and 1p/19q labels using WHO 2021 epidemiological correlations
  - LGG: 80% IDH-mutant, HGG: 10% IDH-mutant
  - IDH-mutant LGG: 50% codeleted, IDH-wildtype: 0% codeleted
- **Slice selection:** Selects top 8 tumor-rich slices per patient based on tumor pixel count
- **Patient-level split:** 80/20 train/test ensuring no patient appears in both sets

### Module 2: Dataset Class (BraTSMultiTaskDataset)
- Reads HDF5 files containing 4-channel MRI images (240x240x4)
- Applies z-score normalization per channel (matching paper's Equation 1)
- Resizes to 128x128 for GPU memory efficiency
- Data augmentation: random horizontal and vertical flips
- Returns: image tensor, IDH label, 1p/19q label, grade label, OS time, event indicator

### Module 3: Multi-task Model (MultiTaskResNet)
- **Shared backbone:** ResNet18 pretrained on ImageNet, first conv adapted from 3-channel RGB to 4-channel MRI
- **IDH head:** Linear(512, 2) — binary classification
- **1p/19q head:** Linear(512, 2) with gradient detachment — `feat.detach()`
- **Grade head:** Linear(512, 2) — binary classification
- **Survival branch:** 512→64→16 (prognosis features), concatenated with 6 task scores, →32→1 (DPS)

### Module 4: Loss Functions
- CrossEntropyLoss for IDH and Grade
- Weighted CrossEntropyLoss [1.0, 4.0] for 1p/19q (class imbalance handling)
- NegativeLogLikelihood (Cox partial likelihood) for survival

### Module 5: Training Engine
- Adam optimizer (lr=0.001, weight_decay=1e-4)
- Polynomial LR decay: lr * (1 - epoch/max_epoch)^0.9
- Mixed precision training (torch.cuda.amp)
- Best model saved based on mean validation AUC (min epoch 5)

### Module 6: Evaluation and Visualization
- 7 visualization functions generating 10 publication-quality figures
- Optimal threshold selection using Youden's J statistic
- Classification reports with AUC, accuracy, precision, recall, F1

## System Design / Block Diagram

```
┌─────────────────────────────────────────────────────┐
│                    INPUT PIPELINE                     │
│  BraTS H5 Files → 4-ch MRI (T1,T1ce,T2,FLAIR)      │
│  → Z-score Normalize → Resize 128x128 → Augment     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              SHARED BACKBONE (ResNet18)               │
│  Conv1(4→64) → BN → ReLU → MaxPool                  │
│  → Layer1 → Layer2 → Layer3 → Layer4                 │
│  → AdaptiveAvgPool → 512-dim feature vector          │
└───┬──────────┬──────────┬──────────┬────────────────┘
    │          │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼─────────────────┐
│ IDH   │ │1p/19q │ │ Grade │ │ Survival Branch      │
│ Head  │ │ Head  │ │ Head  │ │ 512→64→16 (feat)     │
│512→2  │ │512→2  │ │512→2  │ │ concat(16 + 6 scores)│
│       │ │(detach)│ │       │ │ 22→32→1 = DPS        │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───────────────────┘
    │         │         │         │
┌───▼─────────▼─────────▼─────────▼───────────────────┐
│                  LOSS COMPUTATION                     │
│  L_total = CE(IDH) + wCE(1p19q) + CE(Grade) + Cox   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               OUTPUT / EVALUATION                     │
│  ROC Curves │ Confusion Matrices │ Calibration       │
│  Kaplan-Meier │ DPS Distribution │ Performance Chart │
└─────────────────────────────────────────────────────┘
```

## Key Implementation Steps
1. Parse metadata CSVs to build patient-level label mappings
2. Generate pseudo-labels for IDH and 1p/19q based on grade correlations
3. Select top 8 tumor-rich slices per patient from 57,195 available slices
4. Split 369 patients into 295 training + 74 testing (patient-level, no leakage)
5. Initialize ResNet18 backbone with ImageNet weights adapted for 4-channel input
6. Train for 35 epochs with multi-task loss, polynomial LR decay, mixed precision
7. Save best model based on mean validation AUC (epoch 7 selected)
8. Generate 10 paper-style evaluation figures and classification reports
