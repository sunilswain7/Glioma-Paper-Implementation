"""v2 Cross-Slice U-Net. Same architecture as v1 (cst/model.py) with one fix:

  v1 added ONE shared vector `slice_pos` to every slice token, so the transformer
  could not tell slices apart (above vs below vs centre). v2 learns a separate
  position embedding per slice offset: slice_pos has shape (1, K, C).

K=1 -> no transformer -> plain 2D U-Net baseline (same code, one flag).
"""
import torch
import torch.nn as nn


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, cin, base):
        super().__init__()
        self.e1 = conv_block(cin, base)
        self.e2 = conv_block(base, base * 2)
        self.e3 = conv_block(base * 2, base * 4)
        self.e4 = conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b = self.e4(self.pool(s3))                      # (H/8, W/8)
        return b, (s1, s2, s3)


class CrossSliceTransformer(nn.Module):
    """Self-attention across the K slices at every bottleneck position.
    (B, K, C, h, w) -> fused centre-slice features (B, C, h, w)."""
    def __init__(self, dim, heads, layers, n_slices):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            batch_first=True, dropout=0.1,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=layers,
                                        enable_nested_tensor=False)
        self.slice_pos = nn.Parameter(torch.randn(1, n_slices, dim) * 0.02)

    def forward(self, feats):
        B, K, C, h, w = feats.shape
        x = feats.permute(0, 3, 4, 1, 2).reshape(B * h * w, K, C)
        x = self.tf(x + self.slice_pos)
        center = x[:, K // 2].reshape(B, h, w, C)
        return center.permute(0, 3, 1, 2).contiguous()


class Decoder(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = conv_block(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, b, skips):
        s1, s2, s3 = skips
        x = self.d3(torch.cat([self.up3(b), s3], 1))
        x = self.d2(torch.cat([self.up2(x), s2], 1))
        x = self.d1(torch.cat([self.up1(x), s1], 1))
        return self.head(x)


class CrossSliceUNet(nn.Module):
    def __init__(self, n_slices=5, base=48, in_ch=4, num_classes=3, heads=4, layers=2):
        super().__init__()
        self.K = n_slices
        self.encoder = Encoder(in_ch, base)
        self.xslice = (CrossSliceTransformer(base * 8, heads, layers, n_slices)
                       if n_slices > 1 else None)
        self.decoder = Decoder(base, num_classes)

    def forward(self, x):                               # x: (B, K, 4, H, W)
        B, K, C, H, W = x.shape
        b, skips = self.encoder(x.reshape(B * K, C, H, W))
        if self.xslice is not None:
            b = self.xslice(b.reshape(B, K, *b.shape[1:]))
            skips = tuple(s.reshape(B, K, *s.shape[1:])[:, K // 2] for s in skips)
        return self.decoder(b, skips)                   # logits (B, 3, H, W)
