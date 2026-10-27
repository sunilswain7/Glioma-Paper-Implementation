# Slide 8: Experimental Setup and Evaluation Metrics

## Hardware / Software Environment

| Component | Paper | Our Setup |
|---|---|---|
| **GPU** | NVIDIA RTX 3080 Ti (12 GB VRAM) | NVIDIA RTX 3050 Laptop (4 GB VRAM) |
| **VRAM Usage** | ~10 GB (3D volumes, batch=32) | ~3 GB (2D slices, batch=16, mixed precision) |
| **Framework** | PyTorch 0.4.0, Python 3.6 | PyTorch 2.x, Python 3.13 |
| **OS** | Not specified | Linux (Fedora 42) |
| **Training time** | Not reported | 4.3 minutes (35 epochs) |
| **Mixed precision** | No | Yes (FP16 via torch.cuda.amp) |

## Testing Procedure

### Data Split:
- 369 patients split at **patient level** (critical: no patient in both train and test)
- 295 patients (80%) for training → 2,360 slices
- 74 patients (20%) for external validation → 592 slices
- 8 most tumor-rich slices selected per patient

### Training Configuration:
| Parameter | Value |
|---|---|
| Epochs | 35 |
| Batch size | 16 |
| Learning rate | 0.001 (polynomial decay, exponent 0.9) |
| Optimizer | Adam (weight_decay = 1e-4) |
| Best model | Selected at epoch 7 (highest mean validation AUC) |
| Seed | 42 (fixed for reproducibility) |

### Validation:
- Model evaluated on held-out test set (74 patients) after each epoch
- Best model checkpoint saved based on highest mean AUC across 3 classification tasks
- Final evaluation generates all metrics and figures from the best model

## Performance Metrics Used

### Classification Metrics:
| Metric | What It Measures | Formula |
|---|---|---|
| **AUC** | Overall discrimination ability (threshold-independent) | Area under ROC curve |
| **Accuracy** | Fraction of correct predictions | (TP+TN) / (TP+TN+FP+FN) |
| **Precision** | Of predicted positives, how many are correct | TP / (TP+FP) |
| **Recall (Sensitivity)** | Of actual positives, how many are detected | TP / (TP+FN) |
| **F1-Score** | Balance between precision and recall | 2*(P*R)/(P+R) |
| **Optimal Threshold** | Best classification cutoff (Youden's J) | argmax(TPR - FPR) |

### Survival Metrics:
| Metric | What It Measures |
|---|---|
| **Kaplan-Meier Curve** | Survival probability over time for each DPS group |
| **Log-rank Test (p-value)** | Statistical significance of survival difference between groups |

### Visualization Outputs:
| Figure | Purpose | Paper Equivalent |
|---|---|---|
| ROC Curves | Discrimination ability per task | Paper Fig. 3a-c |
| Confusion Matrices | Classification accuracy breakdown | Paper Fig. 3d-f |
| Calibration Curves | Probability reliability assessment | Paper Fig. 4b |
| Kaplan-Meier Curves | Survival stratification by DPS | Paper Fig. 4a |
| DPS Distribution | Score distribution by molecular status | Paper Supplementary |
| Training Curves | Loss and AUC convergence | Not in paper |
| Performance Comparison | Our results vs paper bar chart | Not in paper |
