# Slide 2: Introduction and Background

## Domain Overview
- **Glioma** is a primary brain tumor originating from glial cells and is the third leading cause of cancer-related deaths worldwide
- The WHO 2021 Classification of CNS Tumors incorporates both histopathological and molecular markers for diagnosis:
  - **IDH mutation status** — Isocitrate Dehydrogenase gene mutation
  - **1p/19q codeletion status** — Co-deletion of chromosomal arms 1p and 19q
  - **Histological grade** — Grade II (low), III, IV (high/GBM)
- These molecular biomarkers are critical for treatment planning and prognosis estimation

## Importance of the Problem Area
- Current methods to determine molecular markers require **invasive biopsy or surgery**
- Biopsy is prone to **sampling errors** (tumor heterogeneity) and carries surgical risk
- MRI is routinely acquired for all brain tumor patients — it is **non-invasive, global, and repeatable**
- If molecular markers can be predicted from MRI alone, it enables:
  - Pre-operative treatment planning without waiting for pathology
  - Non-invasive monitoring during treatment
  - Personalized clinical decision-making

## Motivation Behind the Research
- Existing deep learning models for glioma are mostly **single-task** (predict one marker at a time)
- Single-task models ignore the biological interrelationships between molecular markers (e.g., IDH and 1p/19q are biologically linked)
- **Multi-task learning (MTL)** can simultaneously predict multiple related outcomes, offering:
  - Shared feature learning across tasks
  - Better generalization with limited data
  - Reduced computational cost (one model instead of four)
- Additionally, most existing models are **black boxes** — this paper adds biological interpretability through multi-omics analysis
