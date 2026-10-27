"""Convert BraTS2020 into compact per-patient arrays for v2 training.

Source (preferred): the ORIGINAL NIfTI release as the Kaggle zip
    awsaf49/brats20-dataset-training-validation  (read straight from the zip,
    no 42 GB unzip). Only the 369 *training* cases are used (validation has no masks).
Fallback: the per-slice HDF5 build we used on the laptop (v1).

Per patient we write two .npy files (memory-mappable):
    <pid>_img.npy  float16 (S, 4, 192, 192)   modalities [FLAIR, T1, T1ce, T2]
    <pid>_seg.npy  uint8   (S, 192, 192)      labels 0=bg 1=NCR/NET 2=ED 3=ET (BraTS 4)

What changes vs v1 (and why):
  * intensity: z-score PER PATIENT, PER MODALITY, over BRAIN voxels only, background=0.
    (v1 data was z-scored per slice incl. background -> neighbouring slices had
    different scales, which hurts cross-slice fusion.)
  * crop: 192x192 window centred on the brain bounding box, empty z-slices dropped
    (~2x less compute; tumour voxels lost are counted and must be 0).

Usage:
  python -m cst2.preprocess --zip ~/Project1/data/brats20-dataset-training-validation.zip \
         --out ~/Project1/data/brats_npy --workers 8
  python -m cst2.preprocess --h5_dir ".../content/data" --out ... --limit 8   # fallback
"""
import os
import re
import io
import gzip
import json
import glob
import zipfile
import argparse
from multiprocessing import Pool

import numpy as np

CROP = 192
MODS = ("flair", "t1", "t1ce", "t2")
_CASE = re.compile(r"BraTS20_Training_(\d{3})")


# ---------------------------------------------------------------- shared steps
def normalise_and_crop(img, seg):
    """img (S,4,H,W) float32 raw, seg (S,H,W) uint8 {0,1,2,3} -> cropped arrays + info."""
    brain = (img != 0).any(1)                                  # (S,H,W)
    zs = np.where(brain.any((1, 2)))[0]
    img, seg, brain = img[zs[0]:zs[-1] + 1], seg[zs[0]:zs[-1] + 1], brain[zs[0]:zs[-1] + 1]

    out = np.zeros_like(img)
    for c in range(img.shape[1]):
        v = img[:, c][brain]
        out[:, c][brain] = (v - v.mean()) / (v.std() + 1e-8)

    ys, xs = np.where(brain.any(0))
    H, W = brain.shape[1:]
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    y0 = int(np.clip(cy - CROP // 2, 0, H - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, W - CROP))
    lost = int((seg > 0).sum() - (seg[:, y0:y0 + CROP, x0:x0 + CROP] > 0).sum())
    lost_brain = int(brain.sum() - brain[:, y0:y0 + CROP, x0:x0 + CROP].sum())
    out = out[:, :, y0:y0 + CROP, x0:x0 + CROP].astype(np.float16)
    seg = seg[:, y0:y0 + CROP, x0:x0 + CROP].astype(np.uint8)
    info = {"slices": int(out.shape[0]), "z0": int(zs[0]), "y0": y0, "x0": x0,
            "tumour_voxels": int((seg > 0).sum()), "tumour_lost": lost,
            "brain_lost": lost_brain, "et_voxels": int((seg == 3).sum())}
    return out, seg, info


def save(out_dir, pid, img, seg, info):
    np.save(os.path.join(out_dir, f"{pid}_img.npy"), img)
    np.save(os.path.join(out_dir, f"{pid}_seg.npy"), seg)
    info["pid"] = pid
    return info


# ---------------------------------------------------------------- NIfTI (zip)
def _load_nii(zf, name):
    import nibabel as nib
    raw = zf.read(name)
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
    return np.asarray(nib.Nifti1Image.from_bytes(raw).dataobj, dtype=np.float32)


def index_zip(zip_path):
    """{case_id: {mod: member, 'seg': member}} for training cases only."""
    cases = {}
    with zipfile.ZipFile(zip_path) as zf:
        for n in zf.namelist():
            if "Training" not in n or not re.search(r"\.nii(\.gz)?$", n):
                continue
            m = _CASE.search(os.path.dirname(n))
            if not m:
                continue
            base = os.path.basename(n).lower()
            d = cases.setdefault(m.group(1), {})
            for mod in MODS:
                if re.search(rf"_{mod}\.nii(\.gz)?$", base):
                    d[mod] = n
                    break
            else:
                if "seg" in base:          # case 355's seg is misnamed '..._Segm.nii'
                    d["seg"] = n
    return {k: v for k, v in sorted(cases.items()) if all(x in v for x in MODS + ("seg",))}


def _do_zip_case(args):
    zip_path, out_dir, cid, members = args
    with zipfile.ZipFile(zip_path) as zf:
        vols = [_load_nii(zf, members[m]) for m in MODS]       # each (240,240,155)
        seg = _load_nii(zf, members["seg"])
    img = np.stack(vols, 0).transpose(3, 0, 1, 2)              # (155,4,240,240)
    seg = np.rint(seg).astype(np.uint8).transpose(2, 0, 1)     # (155,240,240)
    seg[seg == 4] = 3
    img, seg, info = normalise_and_crop(img, seg)
    return save(out_dir, f"BraTS20_{cid}", img, seg, info)


# ---------------------------------------------------------------- HDF5 fallback
def _do_h5_case(args):
    import h5py
    h5_dir, out_dir, vol = args
    files = sorted(glob.glob(os.path.join(h5_dir, f"volume_{vol}_slice_*.h5")),
                   key=lambda p: int(re.search(r"slice_(\d+)", p).group(1)))
    imgs, segs = [], []
    for p in files:
        with h5py.File(p, "r") as f:
            x = f["image"][()].astype(np.float32)               # (240,240,4) per-slice z-scored
            m = f["mask"][()]                                   # (240,240,3) [NCR,ED,ET]
        bg = (x == x.reshape(-1, 4).min(0)).all(-1)             # background = slice minimum
        x[bg] = 0
        imgs.append(x.transpose(2, 0, 1))
        s = np.zeros(m.shape[:2], np.uint8)
        s[m[..., 1] > 0] = 2; s[m[..., 0] > 0] = 1; s[m[..., 2] > 0] = 3
        segs.append(s)
    img, seg = np.stack(imgs), np.stack(segs)
    # keep v1 per-slice values (cannot be undone); only background zeroed + crop
    brain = (img != 0).any(1)
    zs = np.where(brain.any((1, 2)))[0]
    img, seg, brain = img[zs[0]:zs[-1] + 1], seg[zs[0]:zs[-1] + 1], brain[zs[0]:zs[-1] + 1]
    ys, xs = np.where(brain.any(0))
    y0 = int(np.clip((ys.min() + ys.max()) // 2 - CROP // 2, 0, 240 - CROP))
    x0 = int(np.clip((xs.min() + xs.max()) // 2 - CROP // 2, 0, 240 - CROP))
    lost = int((seg > 0).sum() - (seg[:, y0:y0 + CROP, x0:x0 + CROP] > 0).sum())
    img = img[:, :, y0:y0 + CROP, x0:x0 + CROP].astype(np.float16)
    seg = seg[:, y0:y0 + CROP, x0:x0 + CROP]
    info = {"slices": int(img.shape[0]), "tumour_voxels": int((seg > 0).sum()),
            "tumour_lost": lost, "et_voxels": int((seg == 3).sum())}
    return save(out_dir, f"BraTS20_{vol:03d}", img, seg, info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    ap.add_argument("--h5_dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only first N cases (testing)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.zip:
        cases = index_zip(args.zip)
        print(f"found {len(cases)} complete training cases in zip", flush=True)
        jobs = [(args.zip, args.out, cid, m) for cid, m in cases.items()]
        fn = _do_zip_case
    else:
        vols = sorted({int(re.search(r"volume_(\d+)_", p).group(1))
                       for p in glob.glob(os.path.join(args.h5_dir, "*.h5"))})
        jobs = [(args.h5_dir, args.out, v) for v in vols]
        fn = _do_h5_case
    if args.limit:
        jobs = jobs[:args.limit]

    infos = []
    with Pool(args.workers) as pool:
        for i, info in enumerate(pool.imap_unordered(fn, jobs), 1):
            infos.append(info)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done", flush=True)
    infos.sort(key=lambda d: d["pid"])
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"source": "nifti_zip" if args.zip else "h5_v1", "crop": CROP,
                   "cases": infos}, f, indent=1)
    lost = sum(d["tumour_lost"] for d in infos)
    print(f"DONE {len(infos)} cases | tumour voxels lost by crop: {lost} "
          f"| cases without ET: {sum(d['et_voxels'] == 0 for d in infos)}")


if __name__ == "__main__":
    main()
