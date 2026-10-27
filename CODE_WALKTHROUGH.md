# Code Walkthrough — Cross-Slice Transformer (viva prep)

Everything in `glioma_seg/`. 6 code files in `cst/` + `sweep.py`. Total ~690 lines.
Every claim below has a **`file:line`** so you can point to it live.

---

## 0. The 30-second pitch (say this first)

> "It segments brain tumours from MRI. A normal 2-D U-Net looks at one slice at a
> time and ignores that the tumour is a 3-D object. My contribution is a
> **Cross-Slice Transformer** that takes 5 neighbouring slices and uses attention
> **across the slice axis** to fuse their context into the centre slice, so the
> segmentation stays consistent in 3-D. The clever part: setting the slice count to
> 1 turns the whole thing back into a plain U-Net — so I get a clean ablation from
> the *same code, one flag*."

The "one flag" is `n_slices` in `config.py:23`. K=1 → baseline, K=5 → my model.

---

## 1. File map — "where is everything?"

| File | Lines | What lives here |
|------|-------|-----------------|
| `cst/config.py`   | 50  | **All hyperparameters** — one dataclass |
| `cst/dataset.py`  | 144 | **Data loading** — HDF5 → 5-slice stacks, the train/val split |
| `cst/model.py`    | 123 | **The network** — Encoder, Cross-Slice Transformer, Decoder |
| `cst/losses.py`   | 23  | **Loss function** — Dice + BCE |
| `cst/metrics.py`  | 36  | **Evaluation** — WT/TC/ET region Dice |
| `cst/train.py`    | 185 | **Training loop** — optimiser, epochs, checkpointing |
| `sweep.py`        | 127 | **Experiments** — the design-justification runs |

Mnemonic to recite: **config → dataset → model → losses → metrics → train.**
Data flows left to right; that's also the import order in `train.py:14-21`.

---

## 2. `config.py` — the control panel

**Where:** `Config` dataclass at `config.py:10`.

Key things she may point at:
- `n_slices = 5` (**`config.py:23`**) — THE experiment knob. Must be odd (centre + neighbours).
- `base_ch = 48` (**`config.py:29`**) — U-Net width; chosen so params ≈ 6.7M ≈ the paper's 7.97M and fits 4 GB VRAM.
- `batch_size = 4` (**`config.py:35`**) — sized for the 4 GB laptop GPU.
- `lr = 2e-4`, `epochs`, `weight_decay = 1e-5`, `val_frac = 0.2`, `seed = 42` (**`config.py:34-42`**).
- `half` property (**`config.py:49`**) — returns `n_slices // 2`, the neighbour radius (2 for K=5).

**Why a dataclass?** "So every experiment differs by one field, nothing is
hard-coded in the model — that's what makes the ablation fair."

---

## 3. `dataset.py` — getting data in

The data on disk = one HDF5 file **per 2-D slice**: `volume_{V}_slice_{S}.h5`,
each with `image (240,240,4)` (4 MRI modalities, already z-scored) and
`mask (240,240,3)` (one-hot [NCR/NET, ED, ET]). Header note at `dataset.py:1-10`.

Four pieces, in order:

1. **`build_foreground_set()`** (**`dataset.py:29`**) — reads the metadata CSV of
   per-slice tumour pixel counts, returns the set of `(volume, slice)` that contain
   tumour. Purpose: skip empty slices *instantly* instead of opening 57k files.

2. **`build_volume_index()`** (**`dataset.py:46`**) — scans filenames with a regex
   (`dataset.py:20`) and groups slices by volume → `{volume: [slice ids]}`.

3. **`split_volumes()`** (**`dataset.py:60`**) — **the most important 10 lines for a
   viva.** It shuffles and splits by **VOLUME, not slice** (`dataset.py:66-68`).
   → **Why it matters:** adjacent slices of one patient are nearly identical; if some
   went to train and some to val, the model would "cheat" and val scores would be
   fake. Splitting whole patients prevents this **data leakage**. Expect her to ask
   this — it's the #1 methodology question.

4. **`BraTSCrossSlice(Dataset)`** (**`dataset.py:71`**) — the PyTorch dataset:
   - `_collect()` (**`dataset.py:88`**) — builds the list of `(volume, centre_slice)`
     training targets, keeping only foreground slices.
   - `_load_slice()` (**`dataset.py:103`**) — loads one slice, **clamps to volume
     edges** (`dataset.py:105`) so a stack near the top/bottom of the brain repeats
     the edge slice instead of crashing.
   - `__getitem__()` (**`dataset.py:114`**) — **the core**: for offsets −2…+2 around
     the centre it stacks 5 slices → `x (5,4,240,240)`, and takes the **centre
     slice's** mask as the label `y (3,240,240)` (`dataset.py:118-128`).
   - `_augment()` (**`dataset.py:138`**) — random H/V flips applied **identically to
     all 5 slices and the mask** (`dataset.py:140-143`), so geometry stays aligned.

---

## 4. `model.py` — the network (the heart; she WILL dig here)

Header diagram at `model.py:1-17`. Four classes:

### `conv_block()` — `model.py:22`
The U-Net building block: Conv3×3 → BatchNorm → ReLU, twice. Nothing fancy.

### `Encoder` — `model.py:31`
Standard U-Net encoder. 4 stages, widths **48→96→192→384**, max-pool between
(`model.py:35-39`). `forward` (`model.py:41`) returns the **bottleneck** `b` and the
three **skip connections** `(s1,s2,s3)`. Runs on **every slice with shared weights**.
- Input: `(B·K, 4, 240, 240)` · Output: bottleneck `(B·K, 384, 30, 30)` + skips.
- Say: "Weight-shared, so encoder cost doesn't grow with the number of slices."

### `CrossSliceTransformer` — `model.py:49` ⭐ THE NOVELTY
This is the module to know cold. `forward` at **`model.py:63`**:
```
feats: (B, K, C, h, w)          # K=5 slices, C=384, h=w=30
→ permute+reshape → (B*h*w, K, C)   # line 66: at EACH spatial position, a sequence of K tokens
→ + slice_pos embedding             # line 67: learnable position per slice
→ TransformerEncoder                # line 68: self-attention ACROSS the K slices
→ reshape, take centre token        # lines 69-70: K//2 = the centre slice
→ (B, C, h, w)                      # line 71: fused centre features
```
- Built at `model.py:56-60`: `nn.TransformerEncoderLayer` → `nn.TransformerEncoder`,
  **2 layers, 4 heads** (from config), feed-forward `dim*2`.
- **The one-sentence answer** if she asks "what's novel?":
  > "Standard transformers attend across *space* within one image. Mine attends
  > across the *slice axis* — at every pixel location it looks at that same location
  > in the 5 neighbouring slices and fuses them. That's the adaptive slice fusion."
- `slice_pos` (`model.py:61`) = learnable embedding so the model knows *which* slice
  is which (centre vs neighbour).

### `Decoder` — `model.py:74`
U-Net decoder: 3 transpose-conv upsampling stages, each concatenating the matching
**centre-slice** skip and refining (`model.py:87-89`), then a 1×1 conv **head** to 3
channels (`model.py:83`, `model.py:90`). Output: logits `(B, 3, 240, 240)`.

### `CrossSliceUNet` — `model.py:93` (ties it together)
`forward` at **`model.py:106`** — trace this end-to-end if asked:
1. `(B,K,4,H,W)` → reshape to `(B·K,4,H,W)` so encoder sees each slice (`model.py:109`).
2. Encode all slices (`model.py:110`).
3. **If K>1** (`model.py:113`): reshape to `(B,K,...)`, run the Cross-Slice
   Transformer (`model.py:114-115`), and **keep only the centre slice's skips**
   (`model.py:117-120`).
4. **If K==1** (`model.py:121-122`): skip the transformer entirely → plain U-Net.
   **← this `if` is the whole ablation.** `use_xslice` is set at `model.py:99`.
5. Decode → logits (`model.py:123`).

---

## 5. `losses.py` — `DiceBCELoss` at `losses.py:7`

`forward` (`losses.py:13`): `0.5·(1 − softDice) + 0.5·BCE`, per channel, after a
**sigmoid** (`losses.py:14`).
- **Why sigmoid, not softmax?** The 3 channels are **not mutually exclusive** (a
  voxel can be in multiple regions), so each channel is an independent yes/no →
  sigmoid + BCE, not softmax.
- **Why Dice AND BCE?** Dice handles the huge class imbalance (tumour is a tiny part
  of the image); BCE gives stable per-pixel gradients. My sweep proved the combo
  (0.716) beats Dice-only (0.706) and BCE-only (0.690).

---

## 6. `metrics.py` — the BraTS score

- `to_regions()` (**`metrics.py:14`**) — converts the 3 raw channels into the 3
  **nested clinical regions** (`metrics.py:16-18`):
  - **WT** (Whole Tumour) = NCR/NET ∪ ED ∪ ET
  - **TC** (Tumour Core) = NCR/NET ∪ ET
  - **ET** (Enhancing Tumour) = ET
- `dice_score()` (**`metrics.py:22`**) — `2|A∩B|/(|A|+|B|)` per region + their mean.
- **Why these 3 regions?** "It's the BraTS challenge standard — everyone reports
  WT/TC/ET so my numbers are comparable to published work."

---

## 7. `train.py` — the training loop

- `set_seed()` (**`train.py:23`**) — reproducibility (numpy + torch).
- `evaluate()` (**`train.py:28`**) — `@torch.no_grad()`, runs val set, applies
  `sigmoid > 0.5` (**`train.py:35`**) to get binary predictions, returns WT/TC/ET Dice.
- `main()` (**`train.py:43`**):
  - CLI flags incl. my sweep knobs `--optimizer/--scheduler/--lr/--bce_weight`
    (**`train.py:53-62`**).
  - Build index → split → datasets → loaders (**`train.py:73-91`**).
  - Build model, print param count (**`train.py:93-96`**).
  - **Optimizer** — swappable AdamW/Adam/SGD/RMSprop (**`train.py:98-111`**).
  - **Scheduler** — cosine/step/none (**`train.py:113-120`**).
  - **Mixed precision**: `GradScaler` + `autocast` (**`train.py:102, 112`** region) —
    this is what fits training into 4 GB.
  - Epoch loop (**`train.py:106+`**): forward → loss → backward → step; then
    `evaluate`; **save `best.pth` only when mean Dice improves** (**`train.py:130-135`**);
    write `history.json` every epoch; write `summary.json` at the end.

---

## 8. `sweep.py` — how I justified every choice

`sweep.py:1` header explains it. The `MATRIX` (**`sweep.py:41`**) lists 11 runs, each
varying **one factor** (K, optimizer, scheduler, loss, lr) off the chosen config
(`CHOSEN`, **`sweep.py:29`**), fixed 16 volumes / 10 epochs / seed 42. `collect()`
(**`sweep.py:66`**) aggregates every `summary.json` into `runs/sweep_results.csv` +
a printed table. **This file is your evidence** that the design choices are
data-driven, not arbitrary.

---

## 9. Questions she's likely to ask — with your answers

**Q: Where's the novelty / what's new?**
→ `model.py:49` `CrossSliceTransformer`, specifically the `forward` at `model.py:63`:
attention across the slice axis. Everything else is standard U-Net.

**Q: How do you prevent data leakage?**
→ `dataset.py:60` `split_volumes` — split by whole patient/volume, never by slice.

**Q: Why does K=1 give a baseline?**
→ `model.py:113` the `if self.use_xslice`. K=1 skips the transformer → plain 2-D
U-Net. Same code, one flag = a clean ablation.

**Q: Why sigmoid + Dice+BCE, not softmax + cross-entropy?**
→ `losses.py:14` — regions overlap (non-exclusive), so per-channel sigmoid; Dice for
imbalance + BCE for stable gradients. Sweep proved the combo wins.

**Q: How do you handle the 4 GB GPU limit?**
→ 2.5-D not 3-D; mixed precision (`train.py` GradScaler/autocast); `base_ch=48`,
`batch_size=4` in `config.py`.

**Q: What are WT/TC/ET?**
→ `metrics.py:16-18` — the 3 nested BraTS regions; standard for comparability.

**Q: How is the tumour stack built?**
→ `dataset.py:114` `__getitem__` — offsets −2…+2 around the centre, edge-clamped
(`dataset.py:105`), label is the centre slice's mask.

**Q: How do you pick the final model?**
→ `train.py:130-135` — checkpoint the epoch with best **mean** validation Dice.

**Q: How many parameters?**
→ K=1 baseline = **4.33M**, K=5 proposed = **6.70M**. The +2.37M is the transformer.

**Q: What's your base paper?**
→ Abdikenov et al., *IEEE Access* 2026 — Cross-Slice Transformer with Adaptive Slice
Fusion, originally for CT lymph nodes. I adapted the cross-slice attention to
multi-modal brain MRI.

---

## 10. Two honest things to be ready for (don't get caught out)

1. **"Is this exactly the paper's method?"** — No, and say so: "I adapted the core
   idea (cross-slice attention) to a U-Net backbone for 4-modality MRI. I did not
   reimplement their statistical descriptor extraction — mine uses learned attention
   with a slice-position embedding." Honesty > overclaiming.

2. **Results maturity** — the full baseline-vs-proposed numbers are from 30-epoch
   runs on the whole dataset (finishing now). If asked for final Dice, say "baseline
   is 0.795 mean; the proposed model's full run is completing — I'll report the gap
   from the saved checkpoints." Don't quote a number you haven't verified.
