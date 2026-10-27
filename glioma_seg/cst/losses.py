"""Segmentation loss: soft Dice + BCE, averaged over the 3 raw class channels."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0, bce_weight=0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        # soft Dice per channel
        dims = (0, 2, 3)
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1 - dice.mean()

        bce = F.binary_cross_entropy_with_logits(logits, target)
        return (1 - self.bce_weight) * dice_loss + self.bce_weight * bce
