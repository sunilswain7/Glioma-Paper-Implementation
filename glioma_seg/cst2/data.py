"""v2 data: patient-level k-fold split, memory-mapped per-patient arrays, 2.5D stacks.

Label map stored on disk: 0=bg 1=NCR/NET 2=ED 3=ET.
Training targets are the BraTS evaluation REGIONS directly (v1 used raw classes):
    WT = {1,2,3}   TC = {1,3}   ET = {3}
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def to_regions_np(seg):
    """uint8 label map (...,H,W) -> float32 regions (...,3,H,W) [WT,TC,ET]."""
    wt = seg > 0
    tc = (seg == 1) | (seg == 3)
    et = seg == 3
    return np.stack([wt, tc, et], axis=-3).astype(np.float32)


def list_patients(data_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        return [c["pid"] for c in json.load(f)["cases"]]


def make_split(pids, fold, n_folds=5, val_frac=0.1, seed=42, limit_train=None):
    """Patient-level split. Test = fold `fold` of `n_folds`; val = val_frac of the rest.

    limit_train (data-size ablation) subsamples the TRAIN patients only, so the
    val/test patients are identical across all data sizes.
    """
    rng = np.random.RandomState(seed)
    order = list(pids)
    rng.shuffle(order)
    folds = np.array_split(np.array(order), n_folds)
    test = list(folds[fold])
    rest = [p for p in order if p not in set(test)]
    n_val = int(round(len(rest) * val_frac))
    val, train = rest[:n_val], rest[n_val:]
    if limit_train:
        train = list(np.random.RandomState(seed + 1000).permutation(train)[:limit_train])
    return sorted(train), sorted(val), sorted(test)


class Volumes:
    """Lazily memory-maps each patient's arrays (shared by train/eval code)."""
    def __init__(self, data_dir, pids, in_ram=False):
        self.data_dir, self.pids = data_dir, list(pids)
        self.img, self.seg = {}, {}
        for p in self.pids:
            mode = None if in_ram else "r"
            self.img[p] = np.load(os.path.join(data_dir, f"{p}_img.npy"), mmap_mode=mode)
            self.seg[p] = np.load(os.path.join(data_dir, f"{p}_seg.npy"), mmap_mode=mode)


def stack_indices(center, n_slices, depth):
    """Indices of the K slices around `center`, clamped to the volume edges."""
    h = n_slices // 2
    return [min(max(center + o, 0), depth - 1) for o in range(-h, h + 1)]


class SliceStacks(Dataset):
    """Item = (x (K,4,H,W) float16->float32, y (3,H,W) regions of the centre slice)."""
    def __init__(self, vols, n_slices):
        self.v, self.K = vols, n_slices
        self.items, self.is_fg = [], []
        for p in vols.pids:
            fg = (np.asarray(vols.seg[p]) > 0).any((1, 2))
            for z in range(len(fg)):
                self.items.append((p, z))
                self.is_fg.append(bool(fg[z]))
        self.is_fg = np.array(self.is_fg)

    def sample_weights(self, fg_weight):
        """Foreground slices drawn fg_weight x as often as background (paper: 5x)."""
        return np.where(self.is_fg, float(fg_weight), 1.0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, z = self.items[i]
        img = self.v.img[p]
        idx = stack_indices(z, self.K, img.shape[0])
        x = np.asarray(img[idx], dtype=np.float32)              # (K,4,H,W)
        y = to_regions_np(np.asarray(self.v.seg[p][z]))         # (3,H,W)
        return torch.from_numpy(x), torch.from_numpy(y)
