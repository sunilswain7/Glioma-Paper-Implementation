# Slide 6: Tools, Technologies, and Dataset Used

## Software and Libraries

### Paper's Tools:
| Tool | Purpose |
|---|---|
| PyTorch 0.4.0 | Deep learning framework |
| Python 3.6 | Programming language |
| mmFormer | 3D brain tumor segmentation (Transformer-based) |
| R 4.2.3 | Multi-omics statistical analysis |
| reportROC (R) | ROC curve analysis |
| rms (R) | Survival prediction evaluation |
| clusterProfiler (R) | Gene set enrichment analysis (GSEA) |
| GSVA (R) | Gene set variation analysis for immune profiling |
| ESTIMATE (R) | Immune and stromal score computation |
| DESeq2 (R) | Differential gene expression analysis |
| OncoPredict (R) | Drug sensitivity prediction |

### Our Implementation Tools:
| Tool | Purpose |
|---|---|
| PyTorch 2.x | Deep learning framework |
| Python 3.13 | Programming language |
| torchvision | Pretrained ResNet18 model |
| h5py | Reading HDF5 data files |
| scikit-learn | ROC, confusion matrix, calibration, metrics |
| lifelines | Kaplan-Meier survival analysis, log-rank test |
| matplotlib + seaborn | Visualization (plots, heatmaps) |
| pandas + numpy | Data processing |
| torch.cuda.amp | Mixed precision training (memory optimization) |

## Programming Language / Framework
- **Primary:** Python with PyTorch deep learning framework
- **GPU acceleration:** CUDA with mixed precision (FP16) for memory efficiency
- **Statistical analysis:** scikit-learn for classification metrics, lifelines for survival analysis

## Dataset Description

### Paper's Dataset:
| Cohort | Source | Patients | Purpose |
|---|---|---|---|
| TC (Training) | FAHZU + HNPH (private hospitals) | 1,352 | Model training |
| IVC (Internal Validation) | FAHZU + HNPH | 579 | Internal validation |
| EVC1 | TCGA (public) | 213 | External validation |
| EVC2 | UCSF (public) | 410 | External validation |
| EVC3 | EGD (public) | 222 | External validation |
| **Total** | **5 institutions** | **2,776** | |

### Our Dataset (BraTS 2020):
| Attribute | Details |
|---|---|
| **Source** | BraTS 2020 Challenge (Brain Tumor Segmentation) |
| **Patients** | 369 (293 HGG + 76 LGG) |
| **Format** | Pre-processed HDF5 slices (from 3D NIfTI volumes) |
| **MRI Modalities** | T1, T1ce, T2, FLAIR — 4 channels (same as paper) |
| **Image size** | 240 x 240 per slice, resized to 128 x 128 |
| **Total slices** | 57,195 (24,422 with tumor content) |
| **Selected slices** | 2,952 (top 8 tumor-rich slices per patient) |
| **Grade labels** | REAL (HGG/LGG from expert annotation) |
| **Survival data** | REAL (236 patients with OS days, age, resection status) |
| **IDH labels** | Pseudo (grade-correlated, matching WHO 2021 epidemiology) |
| **1p/19q labels** | Pseudo (IDH+grade-correlated, matching literature rates) |
| **Download size** | ~8 GB |
