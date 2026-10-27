# Slide 9: Results and Analysis

## Experimental Results

### AUC Performance Summary:

| Task | Our Training | Our Ext. Validation | Paper EVC1 |
|---|---|---|---|
| **IDH Mutation** | 0.898 | 0.758 | 0.895 |
| **1p/19q Codeletion** | 0.931 | 0.955 | 0.806 |
| **WHO Grade** | 0.986 | 0.928 | 0.850 |

### Detailed Metrics (External Validation, Optimal Thresholds):

| Metric | IDH Mutation | 1p/19q Codeletion | WHO Grade |
|---|---|---|---|
| **AUC** | 0.758 | 0.955 | 0.928 |
| **Accuracy** | 81.8% | 84.5% | 83.6% |
| **Precision** | 57.4% | 40.4% | 97.8% |
| **Recall** | 60.9% | 92.2% | 82.3% |
| **F1-Score** | 59.1% | 56.2% | 89.4% |
| **Threshold** | 0.453 | 0.226 | 0.906 |

### Training Details:
- Best model selected at **epoch 7** out of 35
- Training loss decreased from 3.49 to 0.73 (steady convergence)
- Total training time: **4.3 minutes** on RTX 3050 (4 GB VRAM)

## Graphs / Tables / Comparisons

### Figures generated (in results_brats/ folder):
1. **training_curves.png** — Loss convergence and per-task AUC over 35 epochs
2. **roc_curves_ext._validation.png** — ROC curves showing discrimination for each task
3. **confusion_matrices_ext._validation.png** — Classification accuracy breakdown at optimal thresholds
4. **calibration_curves_ext._validation.png** — Predicted vs observed probability agreement
5. **kaplan_meier_ext._validation.png** — Survival curves: low-DPS vs high-DPS groups
6. **dps_distribution_ext._validation.png** — DPS score distributions by molecular status
7. **performance_summary.png** — Bar chart: Our Train AUC vs Our Test AUC vs Paper EVC1

## Interpretation of Results

### WHO Grade (AUC = 0.928) — Strongest Task
- **Exceeds paper by +7.8 percentage points** (0.928 vs 0.850)
- Uses REAL clinical labels from BraTS expert annotation
- Grade has the strongest visual signal in MRI — necrosis, ring enhancement, mass effect, edema are all visible in individual 2D slices
- High precision (97.8%) ensures very few false positives
- Our binary classification (HGG vs LGG) is simpler than the paper's 3-class (Grade II vs III vs IV), contributing to higher AUC

### 1p/19q Codeletion (AUC = 0.955) — Highest AUC but Cautionary
- High AUC is partly because pseudo-labels are derived from grade
- The model effectively uses its strong grade prediction to infer codeletion status
- With real independent molecular labels, this number would likely be lower
- Demonstrates that the multi-task architecture captures inter-task correlations

### IDH Mutation (AUC = 0.758) — Most Honest Comparison
- Lower than paper (0.758 vs 0.895) due to pseudo-labels and 2D processing
- IDH mutation has subtle MRI correlates (tumor location, border characteristics) that benefit from 3D context
- Still significantly above random chance (0.5), confirming meaningful feature learning

### Survival Prediction (DPS)
- Kaplan-Meier curves show statistically significant separation between low-DPS and high-DPS groups
- Uses REAL overall survival data from 236 BraTS patients
- DPS successfully integrates predictions from all three classification tasks
