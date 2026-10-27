"""BraTS evaluation metrics.

The model outputs the 3 raw class channels [NCR/NET, ED, ET]. BraTS is scored on
three *nested regions* derived from them:
    WT (Whole Tumour)   = NCR/NET | ED | ET   (all three)
    TC (Tumour Core)    = NCR/NET | ET
    ET (Enhancing)      = ET
"""
import torch

CH_NCR, CH_ED, CH_ET = 0, 1, 2


def to_regions(binary):
    """binary: (B,3,H,W) {0,1} -> dict of region masks (B,H,W)."""
    ncr, ed, et = binary[:, CH_NCR], binary[:, CH_ED], binary[:, CH_ET]
    wt = ((ncr + ed + et) > 0).float()
    tc = ((ncr + et) > 0).float()
    return {"WT": wt, "TC": tc, "ET": et}


def dice_score(pred_bin, target_bin, eps=1e-6):
    """Per-region Dice, averaged over the batch. Returns {WT,TC,ET,mean}."""
    preds = to_regions(pred_bin)
    tgts = to_regions(target_bin)
    out = {}
    for r in ("WT", "TC", "ET"):
        p, t = preds[r], tgts[r]
        dims = (1, 2)
        inter = (p * t).sum(dims)
        denom = p.sum(dims) + t.sum(dims)
        # if both empty -> perfect (Dice 1); handled by eps convention below
        d = (2 * inter + eps) / (denom + eps)
        out[r] = d.mean().item()
    out["mean"] = (out["WT"] + out["TC"] + out["ET"]) / 3
    return out
