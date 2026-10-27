"""BraTS-standard volumetric evaluation (what v1 was missing).

Per patient, over the WHOLE 3D volume:
  * Dice for WT / TC / ET (GT empty & pred empty -> 1, one empty -> 0), then
    averaged over patients (each patient weighted equally) -- BraTS / base-paper rule.
  * HD95 (mm; BraTS voxels are 1 mm isotropic). Empty-vs-nonempty -> 373.13 (BraTS).
  * fp_slices: slices with predicted tumour but no GT tumour (3D consistency signal).
ET post-processing: if a predicted ET region is smaller than `et_min` voxels it is
removed (standard BraTS trick for patients without enhancing tumour); et_min is
tuned on VALIDATION patients only, then applied unchanged to TEST.

CLI:  python -m cst2.evaluate --run_dir RUNS/xslice_k5_f0 --data_dir DATA [--split test]
"""
import os
import json
import argparse
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion

from .data import Volumes, stack_indices, to_regions_np

REGIONS = ("WT", "TC", "ET")
HD_MAX = 373.13


@torch.no_grad()
def predict_volume(model, img, n_slices, device, bs=32):
    """img (S,4,H,W) array -> probabilities (S,3,H,W) float32 numpy."""
    model.eval()
    S = img.shape[0]
    vol = torch.from_numpy(np.asarray(img, dtype=np.float32))
    out = []
    for st in range(0, S, bs):
        idx = [stack_indices(z, n_slices, S) for z in range(st, min(st + bs, S))]
        x = vol[torch.tensor(idx)].to(device, non_blocking=True)   # (b,K,4,H,W)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        out.append(torch.sigmoid(logits.float()).cpu())
    probs = torch.cat(out).numpy()
    brain = (np.asarray(img) != 0).any(1)                  # no tumour outside the brain
    return probs * brain[:, None]


def dice3d(p, t):
    ps, ts = p.sum(), t.sum()
    if ts == 0:
        return 1.0 if ps == 0 else 0.0
    return float(2 * (p & t).sum() / (ps + ts))


def hd95(p, t):
    if not p.any() and not t.any():
        return 0.0
    if not p.any() or not t.any():
        return HD_MAX
    sp, st = p ^ binary_erosion(p), t ^ binary_erosion(t)
    d = np.concatenate([distance_transform_edt(~st)[sp], distance_transform_edt(~sp)[st]])
    return float(np.percentile(d, 95))


def binarise(probs, et_min=0):
    b = probs > 0.5                                         # (S,3,H,W) [WT,TC,ET]
    if et_min and b[:, 2].sum() < et_min:
        b[:, 2] = False
    return b


def case_metrics(pred, gt, with_hd=True):
    """pred, gt: bool (S,3,H,W)."""
    r = {}
    for i, k in enumerate(REGIONS):
        r[k] = dice3d(pred[:, i], gt[:, i])
        if with_hd:
            r["HD95_" + k] = hd95(pred[:, i], gt[:, i])
    gt_any = gt[:, 0].any((1, 2))
    r["fp_slices"] = int((pred[:, 0].any((1, 2)) & ~gt_any).sum())
    r["gt_et_empty"] = bool(not gt[:, 2].any())
    return r


def summarise(rows):
    keys = [k for k in rows[0] if k not in ("pid", "gt_et_empty")]
    s = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    s["mean_dice"] = float(np.mean([s[k] for k in REGIONS]))
    s["n"] = len(rows)
    return s


def evaluate_split(model, vols, n_slices, device, et_min=0, with_hd=True, keep_probs=False):
    rows, cache = [], {}
    for p in vols.pids:
        probs = predict_volume(model, vols.img[p], n_slices, device)
        gt = to_regions_np(np.asarray(vols.seg[p])) > 0.5
        if keep_probs:
            cache[p] = (probs, gt)
        rows.append({"pid": p, **case_metrics(binarise(probs, et_min), gt, with_hd)})
    return rows, cache


def tune_et_min(cache, grid=(0, 50, 100, 200, 300, 500, 750, 1000)):
    """Pick the ET size threshold that maximises mean ET Dice on validation."""
    best = (0, -1.0)
    for t in grid:
        d = np.mean([dice3d(binarise(pr, t)[:, 2], gt[:, 2]) for pr, gt in cache.values()])
        if d > best[1]:
            best = (t, float(d))
    return best[0]


def main():
    from .model import CrossSliceUNet
    from .data import list_patients, make_split
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()
    ck = torch.load(os.path.join(args.run_dir, "best.pth"), map_location="cpu",
                    weights_only=False)
    a = ck["args"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossSliceUNet(a["n_slices"], a["base_ch"], layers=a.get("xslice_layers", 2)).to(device)
    model.load_state_dict(ck["ema"] if ck.get("ema") else ck["model"])
    tr, va, te = make_split(list_patients(args.data_dir), a["fold"], a["n_folds"],
                            seed=a["split_seed"], limit_train=a["limit_train"])
    vols = Volumes(args.data_dir, te if args.split == "test" else va)
    rows, _ = evaluate_split(model, vols, a["n_slices"], device, ck.get("et_min", 0))
    print(json.dumps(summarise(rows), indent=1))


if __name__ == "__main__":
    main()
