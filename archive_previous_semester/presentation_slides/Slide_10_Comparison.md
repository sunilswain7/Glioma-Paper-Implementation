# Slide 10: Comparison with Existing Methods

## Accuracy / Performance Comparison

### AUC Comparison Table:

| Task | Our Result | Paper TC | Paper IVC | Paper EVC1 | Paper EVC2 | Paper EVC3 |
|---|---|---|---|---|---|---|
| **IDH** | 0.758 | 0.907 | 0.872 | 0.895 | 0.903 | 0.892 |
| **1p/19q** | 0.955* | 0.874 | 0.845 | 0.806 | 0.894 | 0.710 |
| **Grade** | **0.928** | 0.863 | 0.842 | 0.850 | 0.875 | 0.879 |

*1p/19q AUC inflated due to pseudo-label correlation with grade

### Paper's Multi-task vs Single-task Comparison:
The paper showed that multi-task learning outperforms single-task models:

| Task | Multi-task (Paper) | Single-task (Paper) | Improvement |
|---|---|---|---|
| 1p/19q | 0.806 | 0.705 | +10.1% |
| Survival C-index | 0.723 (TC) | 0.705 (TC) | +1.8% |

This validates the multi-task approach — shared features improve minority-class prediction (1p/19q).

## Advantages of the Proposed Work (Paper)

1. **First unified model** predicting all 3 molecular markers + survival simultaneously
2. **Gradient detachment** for 1p/19q prevents conflicting optimization signals
3. **DPS** provides a clinically actionable single prognostic score
4. **Validated across 5 institutions** (cross-center generalization)
5. **Biological interpretation** through multi-omics analysis (GSEA, immune, methylation)

## Advantages of Our Implementation

1. **Real grade labels** — AUC 0.928 exceeds paper's 0.850 for WHO grade
2. **4-channel MRI input** — matches paper's exact modality configuration
3. **Computationally efficient** — trains in 4.3 minutes on consumer GPU (vs paper's 12GB GPU)
4. **Mixed precision training** — enables operation on 4GB VRAM without performance loss
5. **ImageNet pretrained backbone** — provides strong initial features for medical imaging

## Key Differences Affecting Results

| Factor | Paper (Advantage) | Ours (Limitation) | Impact |
|---|---|---|---|
| Labels | Real molecular markers from biopsy | Pseudo-labels for IDH, 1p/19q | IDH AUC -13.7% |
| Dimensions | 3D volumetric processing | 2D slice processing | Loses spatial context |
| Dataset size | 2,776 patients, multi-institution | 369 patients, single source | Less diversity |
| Grade classes | 3-class (II, III, IV) | 2-class (HGG, LGG) | Our task is easier |
| Initialization | Kaiming random init | ImageNet pretrained | Our advantage |

## Honest Assessment

- **Grade result is genuinely strong** but benefits from binary classification and pretrained weights
- **1p/19q result is artificially high** due to pseudo-label correlation with grade
- **IDH result is the most fair comparison** and shows the architecture works but needs real labels and 3D context for full performance
- **Survival stratification works** with real OS data, validating the DPS concept
