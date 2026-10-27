# Context-Aware Cross-Slice Transformer — Glioma Segmentation (BraTS 2020)

Final-year project. Segments glioma sub-regions from multi-modal brain MRI, with
a **cross-slice transformer** that fuses context from adjacent slices so the 3D
tumour boundary stays consistent (the novelty, adapted from the 2026 Cross-Slice
Transformer / Adaptive Slice Fusion paper, originally for CT lymph nodes).

## Data
Full BraTS 2020 training set (369 volumes), HDF5 per-slice format already on disk:
`Brats datset/BraTS2020_training_data/content/data/volume_{V}_slice_{S}.h5`
- `image` (240,240,4) = T1, T1ce, T2, FLAIR (z-score normalised)
- `mask`  (240,240,3) = one-hot [NCR/NET, ED, ET]

Evaluation regions (BraTS standard): WT = all three, TC = NCR/NET+ET, ET = ET.

## Architecture (`cst/model.py`)
`stack of K adjacent slices (B,K,4,H,W)`
-> shared 2D U-Net **encoder** (per slice)
-> **Cross-Slice Transformer** at the bottleneck (self-attention across the K
   slices at each spatial position -> fuses neighbour context into centre slice)
-> 2D U-Net **decoder** (centre-slice skips) -> mask (B,3,H,W).

Set `n_slices=1` and it collapses to a plain 2D U-Net = the baseline. **Same code,
one flag** -> a clean ablation that isolates the cross-slice contribution.

## Run
```bash
cd glioma_seg

# Proposed model (cross-slice, K=5)
python -m cst.train --exp xslice_k5  --n_slices 5

# Baseline for the ablation (no cross-slice, K=1)
python -m cst.train --exp baseline_k1 --n_slices 1

# Fast sanity check (few volumes, 1 epoch)
python -m cst.train --exp smoke --n_slices 5 --epochs 1 --limit_vols 6
```
Outputs (`runs/<exp>/`): `best.pth`, `history.json`.
Compare `xslice_k5` vs `baseline_k1` mean Dice — that gap is your result.

## Files
- `config.py`  — all hyperparameters (edit here)
- `dataset.py` — H5 cross-slice loader, volume-level split, fast foreground CSV
- `model.py`   — encoder / CrossSliceTransformer / decoder
- `losses.py`  — Dice + BCE
- `metrics.py` — WT/TC/ET Dice
- `train.py`   — training + validation loop

## Tips (RTX 3050, 4 GB)
- Uses mixed precision; ~0.5 GB at batch 4. You can raise `batch_size` to 8–12.
- If OOM: lower `batch_size`, or `base_ch` 32 -> 24.
