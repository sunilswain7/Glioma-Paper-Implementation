# Slide 3: Research Problem Definition

## Problem Addressed in the Paper
- How to **simultaneously predict** IDH mutation, 1p/19q codeletion, WHO grade, and overall survival from MRI scans using a single unified deep learning model
- How to provide **biological interpretability** for the model's predictions by linking deep features to known molecular pathways

## Existing Challenges
1. **Invasive diagnosis:** Current molecular profiling requires tissue from biopsy/surgery — risky, expensive, and subject to sampling errors
2. **Single-task limitation:** Most existing AI models predict only one biomarker, ignoring inter-task biological relationships (e.g., 1p/19q codeletion occurs almost exclusively with IDH mutation)
3. **Lack of interpretability:** Deep learning models in medical imaging are often "black boxes" — clinicians cannot understand WHY the model makes its predictions
4. **Multi-center generalization:** Models trained on one hospital's data often fail on data from other institutions due to differences in MRI scanners and protocols
5. **Prognosis integration:** Previous models separate molecular subtyping from survival prediction, missing the opportunity to create a unified prognostic score

## Objectives of the Proposed Work
1. Develop an end-to-end **multi-task deep learning (MDL)** pipeline that concurrently predicts IDH mutation, 1p/19q codeletion, WHO grade, and overall survival
2. Generate a **Deep Prognosis Score (DPS)** that integrates molecular predictions into a single prognostic indicator
3. Validate the model across **multiple international cohorts** (5 institutions, 2,776 patients)
4. Provide **biological interpretation** of the DPS through radio-multiomics analysis (gene expression, methylation, proteomics, immune profiling)
