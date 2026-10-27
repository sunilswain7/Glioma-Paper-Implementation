# Corrections to `design_solution_preview.pdf`

Your friend drafted the template from a **vague idea of the project**, not from the
actual code. As a result several sections describe an architecture that does **not
match** `glioma_seg/cst/` (the real implementation). This file lists every
mismatch, with the correct text to fill in instead — all verified against the
source (`config.py`, `dataset.py`, `model.py`, `losses.py`, `metrics.py`,
`train.py`).

Severity legend: 🔴 **wrong — contradicts the code** · 🟡 **needs aligning** ·
🟢 **already correct, keep it**.

---

## The single biggest issue

The PDF invents a **5-stage pipeline** with two modules that do not exist:

> Lightweight Transformer Encoder → Cross-Slice Transformer → **Adaptive Modality
> Fusion** → Segmentation Decoder, with an **adaptive gating mechanism**.

The **actual** system is a **3-stage** pipeline:

> **Shared 2-D U-Net (convolutional) Encoder** → **Cross-Slice Transformer at the
> bottleneck** → **2-D U-Net Decoder (centre-slice skips)**.

There is **no transformer encoder**, **no Adaptive Modality Fusion module**, and
**no learned gating network**. The four MRI modalities are simply the 4 input
channels of the first convolution. Fusing neighbour slices is done by *standard
multi-head self-attention across the slice axis* — the attention weights are the
only "gating" that exists; there is no separate MLP gate and no residual add.

Almost every 🔴 below traces back to this one over-invented design.

---

## Section 3 — System Architecture (page 4)

🔴 **"Lightweight Transformer Encoder … patch embedding followed by stacked
self-attention blocks."**
→ The encoder is a **convolutional U-Net encoder**: four double-conv blocks
(Conv3×3–BatchNorm–ReLU ×2) with 2×2 max-pooling, widths **48 → 96 → 192 → 384**.
Weights are **shared across all K slices**. There is no patch embedding and no
self-attention in the encoder.

🔴 **"Adaptive Modality Fusion module, which learns attention-based weights across
the four MRI modalities."**
→ **Delete this module — it does not exist.** T1/T1ce/T2/FLAIR enter as the 4
channels of the first Conv2d; they are never separately re-weighted.

🔴 **"adaptive gating mechanism learning how much each neighboring slice should
contribute."**
→ There is **no explicit gate**. The Cross-Slice Transformer is a 2-layer,
4-head `nn.TransformerEncoder` applied to a length-K token sequence (one token per
slice) at each spatial position, plus a **learnable slice-position embedding**;
the centre token is read out as the fused feature. The self-attention weights
themselves are the only "contribution weighting".

🟢 **Correct — keep:** five consecutive axial slices as input; the cross-slice
module is the core novelty; it addresses the slice-independence limitation; output
regions = necrotic/non-enhancing core, peritumoral edema, enhancing tumor.

**Suggested replacement paragraph:**
> The proposed system is a three-stage 2.5-D pipeline. A window of five consecutive
> axial slices (each with four MRI modalities, T1/T1ce/T2/FLAIR) is fed to a
> **shared 2-D U-Net convolutional encoder** that embeds every slice independently
> into multi-scale features and a 384-channel bottleneck. The K bottleneck feature
> maps are passed to the **Cross-Slice Transformer**, the core novelty: at each
> spatial position it forms a length-K token sequence, adds a learnable
> slice-position embedding, and applies multi-head self-attention **across the slice
> axis** so that the centre-slice representation is enriched with neighbouring-slice
> context. The fused centre-slice bottleneck, together with the **centre slice's
> skip connections**, is decoded by a **2-D U-Net decoder** (transposed
> convolutions) into a per-pixel prediction for the three tumour sub-regions.
> Setting the window size to one collapses the model to a plain 2-D U-Net, giving a
> clean single-flag ablation of the cross-slice contribution.

---

## Section 4 — Module Description (page 5)

🔴 **Data Loader row:** *"Reads raw BraTS2020 NIfTI volumes … applies z-score
normalization … remaps BraTS label 4 to a dense class index."*
→ Wrong on every point. The pipeline reads the **pre-built per-slice HDF5 store**
(`volume_{V}_slice_{S}.h5`), where **images are already z-score normalised**
(240×240×4) and **masks are already one-hot 3-channel** `[NCR/NET, ED, ET]`
(240×240×3). There is **no NIfTI reading** and **no label-4 remapping** — that step
was already done when the HDF5 build was created.
Correct wording: *"Reads the pre-processed BraTS 2020 per-slice HDF5 store
(image 240×240×4 already z-scored, mask 240×240×3 already one-hot [NCR/NET, ED,
ET]); indexes files into volumes; splits at patient/volume level; assembles a
5-slice window centred on each foreground (tumour-bearing) slice with edge-clamping;
applies label-safe random flips shared across the whole stack."*

🔴 **"Lightweight Transformer Encoder … convolutional patch embedding + self-
attention blocks."** → Replace with the **convolutional U-Net encoder** description
(see Section 3 fix). Input: `(B·K, 4, 240, 240)`. Output: bottleneck
`(B·K, 384, 30, 30)` + three skip tensors.

🔴 **Cross-Slice Transformer row — "adaptive gating mechanism … adapted from
Abdikenov et al. (2026)."** → Keep the citation, but correct the mechanism: *"At
each spatial position, forms a length-K token sequence (one token per slice, dim
384), adds a learnable slice-position embedding, and applies a 2-layer / 4-head
Transformer encoder that self-attends across the slice axis; the centre token is
read out as the fused feature."* No MLP gate, no residual.

🔴 **Adaptive Modality Fusion row — delete the whole row.** The module does not
exist.

🟡 **Segmentation Decoder row.** Correct but under-specified. It is a **U-Net
decoder** that upsamples with transposed convolutions **and concatenates the centre
slice's skip connections** at each scale (384→192→96→48), then a 1×1 conv to the
**3 channels**. Note "3", not 4 — see Loss/Output fix below.

🔴 **Loss row — "weighted cross-entropy."** → It is **binary cross-entropy with
logits (BCE)**, not (class-)weighted cross-entropy, blended 50/50 with soft Dice
and applied per channel after a **sigmoid** (channels are non-exclusive). No class
weights are used; Dice handles the imbalance.

---

## Section 5 — System Workflow (page 6)

🟢 **Mostly correct** — patient-level split (no leakage), 5-slice window (2 before /
target / 2 after) with edge-clamping, Dice reported over WT/TC/ET. Keep these.

🔴 **"the decoder output is passed through an argmax operation to produce the final
tumor segmentation mask."** → Wrong. The 3 output channels are **independent
(multi-label)**; inference applies **`sigmoid` then a 0.5 threshold per channel**,
and the three nested regions (WT/TC/ET) are derived from those binary channels.
There is **no softmax and no argmax**.

🔴 **"remaps labels"** in preprocessing → remove (data is already one-hot in HDF5).

🟡 Remove the mention of **Adaptive Modality Fusion** in the workflow narrative.

---

## Section 6 — Domain-Specific Design (page 8)

🟡 **ML Model Architecture row** lists *Lightweight Transformer Encoder → Cross-Slice
Transformer → Adaptive Modality Fusion → Convolutional Segmentation Decoder.*
→ Correct to: *Shared 2-D U-Net convolutional encoder → Cross-Slice Transformer
(self-attention across a 5-slice window at the bottleneck) → 2-D U-Net decoder.*

🟡 **Dataset Description** — say it is the **pre-processed per-slice HDF5 build of
BraTS 2020 (369 volumes)**, not raw multi-institutional NIfTI, to match the actual
pipeline.

🟢 Database/UML/API/Security/Deployment "Not applicable" rows are fine.

---

## Section 7 — Algorithms / Methodology (pages 9–10)

🔴 **The prose** repeats the invented gate/residual/modality-fusion story — rewrite
to match Section 3's replacement paragraph.

🔴 **Algorithm 1 "Cross-Slice Adaptive Fusion"** is entirely invented (MLP gate,
`gate_weights`, weighted sum, residual). **Replace with the real algorithm:**

```
Algorithm 1: Cross-Slice Fusion (as implemented)
Input : bottleneck features of K slices, feats ∈ R^(B×K×C×h×w)   (C=384, h=w=30)
Output: fused centre-slice features, F ∈ R^(B×C×h×w)

1. Reshape to token sequences: X ← reshape(feats) ∈ R^((B·h·w) × K × C)
     # at every spatial position, a sequence of K slice-tokens
2. X ← X + slice_position_embedding          # learnable, broadcast over positions
3. X ← TransformerEncoder(X)                 # 2 layers, 4 heads, dim_ff=768,
                                             #   self-attention ACROSS the K slices
4. reshape X → R^(B×h×w×K×C);  centre ← X[:, :, :, K//2, :]   # read out centre token
5. F ← permute centre → R^(B×C×h×w)
6. return F                                  # (no MLP gate, no residual add)
```

🟢 **Loss formula** `L = (1−λ)·L_dice + λ·L_bce`, λ = 0.5 is numerically correct —
but label the cross-entropy term **BCE** (`binary_cross_entropy_with_logits`), not
CE. The soft-Dice uses smooth = 1.0.

🟢 **Dice formula** and the WT/TC/ET region definitions are correct — keep.

🟡 Add the real class setup: the head outputs **3 sigmoid channels** [NCR/NET, ED,
ET]; WT/TC/ET are nested regions **derived** from them (WT = union of all three,
TC = NCR/NET ∪ ET, ET = ET).

---

## Section 8 — Technology Stack (page 10)

🔴 **Libraries row** lists *einops, nibabel, SciPy, scikit-image, Albumentations,
TensorBoard, PyYAML* — **none of these are used.** The actual dependencies are:

| Category | Correct entry |
|----------|---------------|
| Language | **Python 3.13** (verified 3.13.13) |
| Framework | **PyTorch 2.7** (`+cu126` build) |
| Core libraries | **h5py 3.16** (HDF5 I/O), **NumPy 2.2** |
| NN modules | `torch.nn` Conv2d / BatchNorm2d / ConvTranspose2d / **TransformerEncoder** |
| Mixed precision | `torch.cuda.amp` (autocast + GradScaler) |
| Data pipeline | `torch.utils.data` Dataset/DataLoader |
| Std-lib only | `argparse`, `csv`, `json`, `os`, `re`, `glob`, `dataclasses` |
| Logging | plain **JSON** history (`history.json`) — **not** TensorBoard |
| Augmentation | manual `torch.flip` — **not** Albumentations |
| Data format | **HDF5** — **not** NIfTI, so **no nibabel** |
| Version control | Git |

🔴 **Cloud / Deployment "Google Colab / Kaggle (GPU training)."** → Training runs
**locally on the laptop's own GPU** (RTX 3050, 4 GB). Colab/Kaggle are not used and
are not part of the design.

🟡 Python "3.10+" and PyTorch "2.1.0+" are acceptable as *minimums*, but the real
versions are **3.13 / 2.7**; state the actual ones for accuracy.

🟡 OS "Ubuntu 20.04+ / Windows 10+" → actual dev OS is **Linux (Fedora 42)**.

---

## Section 9 — Hardware & Software Requirements (page 11)

🔴 **GPU: "NVIDIA GPU with 6 GB+ VRAM minimum (GTX 1660); 12 GB+ recommended (RTX
3060/3080)."** → This **contradicts the entire project rationale.** The real target
is an **RTX 3050 Laptop GPU with 4 GB VRAM**, and the whole 2.5-D design (windowed
cross-slice attention instead of 3-D, mixed precision, `base_ch=48`, `batch_size=4`)
exists **because** only 4 GB is available. Correct it to:
*"NVIDIA GPU with ≥ 4 GB VRAM (developed on an RTX 3050 Laptop, 4 GB). The model
trains in this budget via mixed precision at batch size 4."*

🔴 **Other Dependencies: "CUDA Toolkit 11.8+."** → Actual is **CUDA 12.6** (driver
580.x). PyTorch is the `cu126` build.

🟡 **CPU** "quad-core min / 6+ recommended" — fine; actual dev CPU is an **Intel
i5-12500H (12-core)**.

🟡 **RAM** "16 GB min / 32 GB recommended" — fine; actual machine has **16 GB**.

🟡 **Libraries row** — same fix as Section 8 (drop einops/nibabel/Albumentations/
TensorBoard/etc.; use torch + h5py + numpy).

🟡 **OS** — actual is Fedora 42 Linux; the "Ubuntu/Windows/macOS" range is fine to
keep as a portability claim, but note that only Linux has been used.

---

## Section 10 — Design Considerations (page 12)

🟢 Scalability, Reliability (patient-level split, checkpointing), Maintainability
(modular code), Privacy, Ethics — all consistent with the code. **Keep as-is.**

🟡 **Scalability** says "≈370 patient volumes" — exact figure is **369**.

🟡 **Performance** mentions "windowed cross-slice attention rather than full 3-D
attention keeps cost manageable on a single GPU" — correct in spirit; you can
strengthen it by naming the **4 GB** constraint that motivated it.

---

## Section 11 — Expected Outcome (page 13)

🟡 Mentions ablating **"the Cross-Slice Transformer and Adaptive Modality Fusion
modules individually."** → Drop Adaptive Modality Fusion (it doesn't exist). The
real, clean ablation is the **single flag `n_slices`**: `baseline_k1` (K=1, plain
2-D U-Net, **4.33 M params**) vs `xslice_k5` (K=5, cross-slice, **6.70 M params**),
same code/data/seed — the mean-Dice gap isolates the cross-slice contribution.
🟢 Everything else (deliverables, hardest-region = ET, Phase-II milestone) is fine.

---

## Sections that are already correct — no change

- 🟢 **Title page / team / guide / SDG mapping (Sec 1)** — good.
- 🟢 **Project timeline (Sec 2)** — consistent.
- 🟢 **References (Sec 12)** — the base paper (Abdikenov et al., *IEEE Access* 2026,
  ref 5) is cited correctly; ref 2 (Song et al., *npj Precision Oncology*) is the
  earlier-thread base paper and is fine to keep as related work.

---

## One-line summary for your friend

Replace the **transformer encoder**, the **Adaptive Modality Fusion module**, and
the **adaptive-gating pseudocode** with the real design (**conv U-Net encoder →
self-attention across a 5-slice window at the bottleneck → U-Net decoder**), fix the
**data description** (pre-processed HDF5, already z-scored / one-hot — no NIfTI, no
label remap), the **output** (3 sigmoid channels + 0.5 threshold, not 4-class
argmax), the **loss** (Dice + **BCE**), the **libraries** (torch/h5py/numpy only),
and the **hardware** (**RTX 3050, 4 GB, local** — not 6–12 GB cloud). Everything
else in his template stands.
