"""Soft Dice + BCE over the 3 region channels [WT, TC, ET] (weight 0.5 chosen by the v1 sweep)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0, bce_weight=0.5):
        super().__init__()
        self.smooth, self.bce_weight = smooth, bce_weight

    def forward(self, logits, target):
        logits = logits.float()
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return (1 - self.bce_weight) * (1 - dice.mean()) + self.bce_weight * bce
