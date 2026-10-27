# Slide 4: Literature Review / Existing Methods

## Previous Approaches

| Study | Method | Task | Limitation |
|---|---|---|---|
| Zhang et al. (2023) [Ref 26] | Multi-level feature exploration + fusion network | IDH prediction only | Single-task, no survival prediction |
| Lee et al. (2023) [Ref 27] | 3D CNN-based survival prediction | Survival from whole-brain MRI | No molecular subtyping |
| Yan et al. (2021) [Ref 28] | Quantitative MRI-based radiomics | Molecular subtypes + survival | Sequential design, not end-to-end |
| van der Voort et al. (2023) [Ref 33] | Combined subtyping, grading, segmentation | Grade + segmentation | No prognosis prediction |
| Decuyper et al. (2021) [Ref 18] | MRI pipeline for segmentation + prediction | IDH, 1p/19q, grade | No survival, single institution |
| Choi et al. (2021) [Ref 19] | Hybrid approach for IDH | IDH prediction via radiomics + DL | Single marker only |

## Limitations of Existing Systems
1. **Single-task approaches:** Most models predict ONE marker (IDH OR grade OR survival) — they don't capture relationships between tasks
2. **Sequential pipelines:** Some studies predict molecular markers first, then use them for survival prediction in a separate model — not end-to-end
3. **No biological interpretation:** Deep learning models are treated as black boxes without linking predictions to known molecular pathways
4. **Limited validation:** Most studies validate on only 1-2 datasets from the same institution
5. **No unified prognostic score:** Previous work doesn't combine molecular predictions into a single, clinically actionable prognosis indicator

## Research Gap Identified
- No existing work simultaneously predicts ALL THREE molecular markers (IDH, 1p/19q, grade) AND overall survival in a single end-to-end multi-task model
- No previous study provides biological interpretation of deep learning prognosis predictions through comprehensive multi-omics analysis
- The concept of a "Deep Prognosis Score" that integrates molecular predictions with deep features for survival stratification is novel
