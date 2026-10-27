"""BraTS 2020 cross-slice dataset.

The Kaggle HDF5 build stores one file per 2D slice:
    volume_{V}_slice_{S}.h5  with  image (240,240,4)  and  mask (240,240,3)
Images are already z-score normalised; masks are one-hot [NCR/NET, ED, ET].

For a cross-slice model we don't want a single slice -- we want a *stack* of
n_slices adjacent slices from the SAME volume, centred on the target slice.
The model predicts the mask of the centre slice using the neighbours as context.
"""
import os
import re
import csv
import glob
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

_FNAME = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")

# Optional metadata CSV with per-slice tumour pixel counts. If present we use it
# to find foreground slices instantly instead of opening every .h5 file.
_META_CSV = (
    "/home/reaper/PaperCancer/Brats datset/BraTS20 Training Metadata.csv"
)


def build_foreground_set(meta_csv=_META_CSV):
    """Return set of (vol, slice) that contain tumour, or None if no CSV."""
    if not os.path.exists(meta_csv):
        return None
    fg = set()
    with open(meta_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = _FNAME.search(row["slice_path"])
            if not m:
                continue
            counts = [row.get(f"label{i}_pxl_cnt", "0") for i in range(3)]
            if any(int(float(c or 0)) > 0 for c in counts):
                fg.add((int(m.group(1)), int(m.group(2))))
    return fg


def build_volume_index(data_dir):
    """Return {volume_id: sorted list of slice indices present on disk}."""
    index = {}
    for path in glob.glob(os.path.join(data_dir, "*.h5")):
        m = _FNAME.search(os.path.basename(path))
        if not m:
            continue
        vol, sl = int(m.group(1)), int(m.group(2))
        index.setdefault(vol, []).append(sl)
    for vol in index:
        index[vol].sort()
    return index


def split_volumes(index, val_frac, seed):
    """Split by VOLUME (never leak slices of one patient across the split)."""
    vols = sorted(index.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(vols)
    n_val = int(round(len(vols) * val_frac))
    val_vols = set(vols[:n_val])
    train_vols = set(vols[n_val:])
    return train_vols, val_vols


class BraTSCrossSlice(Dataset):
    def __init__(self, cfg, volumes, index, train=True, foreground=None):
        self.cfg = cfg
        self.index = index
        self.train = train
        self.half = cfg.half
        # foreground: optional set of (vol,slice) with tumour (from CSV)
        self.foreground = foreground
        self.samples = self._collect(volumes)

    def _slice_path(self, vol, sl):
        return os.path.join(self.cfg.data_dir, f"volume_{vol}_slice_{sl}.h5")

    def _has_tumour(self, vol, sl):
        with h5py.File(self._slice_path(vol, sl), "r") as f:
            return f["mask"][()].any()

    def _collect(self, volumes):
        """List of (vol, center_slice) targets. Optionally foreground-only."""
        samples = []
        for vol in sorted(volumes):
            slices = self.index[vol]
            for sl in slices:
                if self.cfg.foreground_only:
                    if self.foreground is not None:
                        if (vol, sl) not in self.foreground:
                            continue
                    elif not self._has_tumour(vol, sl):  # slow fallback
                        continue
                samples.append((vol, sl))
        return samples

    def _load_slice(self, vol, sl, slices):
        """Load one slice, clamping to volume bounds at the edges."""
        sl = min(max(sl, slices[0]), slices[-1])
        with h5py.File(self._slice_path(vol, sl), "r") as f:
            img = f["image"][()].astype(np.float32)   # (240,240,4)
            msk = f["mask"][()].astype(np.float32)     # (240,240,3)
        return img, msk

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        vol, center = self.samples[i]
        slices = self.index[vol]

        stack = []
        center_mask = None
        for off in range(-self.half, self.half + 1):
            img, msk = self._load_slice(vol, center + off, slices)
            stack.append(img)
            if off == 0:
                center_mask = msk

        # (n_slices, 240,240,4) -> (n_slices, 4, 240,240)
        x = np.stack(stack, axis=0).transpose(0, 3, 1, 2)
        y = center_mask.transpose(2, 0, 1)              # (3,240,240)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        if self.train:
            x, y = self._augment(x, y)
        return x, y

    @staticmethod
    def _augment(x, y):
        # cheap, label-safe flips (same geometry for every slice in the stack)
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1]); y = torch.flip(y, dims=[-1])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-2]); y = torch.flip(y, dims=[-2])
        return x, y
