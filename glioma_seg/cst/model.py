"""Context-Aware Cross-Slice Transformer for glioma segmentation.

Design (fits a 4 GB GPU, matches the proposal slides):

  input: stack of K adjacent slices, each 4-channel  ->  (B, K, 4, H, W)

  1. A SHARED 2D U-Net encoder runs on every slice independently
     -> per-slice bottleneck features + skip connections for the centre slice.
  2. CROSS-SLICE TRANSFORMER at the bottleneck: for each spatial position we
     have K tokens (one per slice); self-attention across the slice axis fuses
     neighbour context into the centre slice. This is the novel module.
  3. A 2D U-Net decoder reconstructs the CENTRE slice mask from the fused
     bottleneck + the centre slice's skip connections.

Ablation: set K=1 and the cross-slice attention collapses to identity, giving a
plain 2D U-Net baseline -- same code, one flag.
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
    """Standard U-Net encoder; returns bottleneck + skips."""
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
        b = self.e4(self.pool(s3))          # bottleneck (H/8, W/8)
        return b, (s1, s2, s3)


class CrossSliceTransformer(nn.Module):
    """Self-attention across the SLICE axis at each spatial location.

    features: (B, K, C, h, w) -> fused centre features (B, C, h, w)
    """
    def __init__(self, dim, heads, layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            batch_first=True, dropout=0.1,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=layers)
        self.slice_pos = nn.Parameter(torch.zeros(1, 1, dim))  # broadcast pos

    def forward(self, feats):
        B, K, C, h, w = feats.shape
        # tokens: for every (batch, spatial position) -> sequence of K slices
        x = feats.permute(0, 3, 4, 1, 2).reshape(B * h * w, K, C)
        x = x + self.slice_pos
        x = self.tf(x)                                  # attend across slices
        x = x.reshape(B, h, w, K, C)
        center = x[:, :, :, K // 2, :]                  # take centre slice
        return center.permute(0, 3, 1, 2).contiguous()  # (B, C, h, w)


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
        return self.head(x)                             # logits (B, C, H, W)


class CrossSliceUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        base = cfg.base_ch
        self.encoder = Encoder(cfg.in_modalities, base)
        self.use_xslice = cfg.n_slices > 1
        if self.use_xslice:
            self.xslice = CrossSliceTransformer(
                base * 8, cfg.xslice_heads, cfg.xslice_layers
            )
        self.decoder = Decoder(base, cfg.num_classes)

    def forward(self, x):
        # x: (B, K, 4, H, W)
        B, K, C, H, W = x.shape
        flat = x.reshape(B * K, C, H, W)
        b, skips = self.encoder(flat)                   # b: (B*K, 8base, h, w)
        _, Cf, h, w = b.shape

        if self.use_xslice:
            b = b.reshape(B, K, Cf, h, w)
            b = self.xslice(b)                          # (B, 8base, h, w)
            # keep only the CENTRE slice's skip connections for decoding
            center = K // 2
            skips = tuple(
                s.reshape(B, K, *s.shape[1:])[:, center] for s in skips
            )
        else:
            b = b                                       # K==1, already centre
        return self.decoder(b, skips)                   # (B, num_classes, H, W)
