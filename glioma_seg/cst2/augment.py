"""Batched GPU augmentation. The SAME geometric transform is applied to all K
slices of a stack (and its mask), so the cross-slice alignment is preserved."""
import math
import torch
import torch.nn.functional as F


def gpu_augment(x, y, p_flip=0.5, rot_deg=15.0, scale=(0.85, 1.15),
                p_affine=0.5, p_int=0.5, noise_std=0.05):
    """x (B,K,C,H,W), y (B,3,H,W) on GPU -> augmented copies."""
    B, K, C, H, W = x.shape
    dev = x.device
    x = x.reshape(B, K * C, H, W)

    # flips (left-right and up-down), per sample
    for dim in (-1, -2):
        m = torch.rand(B, device=dev) < p_flip
        if m.any():
            x[m] = x[m].flip(dim)
            y[m] = y[m].flip(dim)

    # random rotation + isotropic scale, per sample
    m = torch.rand(B, device=dev) < p_affine
    if m.any():
        n = int(m.sum())
        ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(rot_deg)
        s = torch.empty(n, device=dev).uniform_(*scale)
        cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
        theta = torch.zeros(n, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = cos, -sin, sin, cos
        grid = F.affine_grid(theta, (n, 1, H, W), align_corners=False)
        x[m] = F.grid_sample(x[m], grid, mode="bilinear", padding_mode="zeros",
                             align_corners=False)
        y[m] = (F.grid_sample(y[m], grid, mode="nearest", padding_mode="zeros",
                              align_corners=False) > 0.5).float()

    # per-modality intensity scale/shift + gaussian noise (brain voxels only)
    m = torch.rand(B, device=dev) < p_int
    if m.any():
        xs = x.reshape(B, K, C, H, W)
        brain = (xs != 0).any(2, keepdim=True)
        a = torch.empty(B, 1, C, 1, 1, device=dev).uniform_(0.9, 1.1)
        b = torch.empty(B, 1, C, 1, 1, device=dev).uniform_(-0.1, 0.1)
        noise = torch.randn_like(xs) * noise_std
        aug = (xs * a + b + noise) * brain
        mm = m.view(B, 1, 1, 1, 1)
        x = torch.where(mm, aug, xs).reshape(B, K * C, H, W)
    return x.reshape(B, K, C, H, W), y
