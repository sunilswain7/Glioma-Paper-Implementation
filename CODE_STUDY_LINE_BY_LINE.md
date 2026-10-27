# Line-by-Line Code Study — Cross-Slice Transformer (Viva Prep)

Read this top to bottom. Every meaningful line of every file is explained in plain
language. Files are in the order data flows through them:
**config → dataset → model → losses → metrics → train → sweep.**

Tensor-shape shorthand used throughout:
`B` = batch size, `K` = number of slices (5 for the model, 1 for baseline),
`C` = channels, `H,W` = 240×240 image, `h,w` = 30×30 bottleneck.

---

# 1. `cst/config.py` — the control panel

```python
from dataclasses import dataclass, field
```
Imports the `dataclass` decorator — a Python shortcut that turns a class into a plain
"struct" of named fields with default values, no boilerplate `__init__` needed.

```python
@dataclass
class Config:
```
`Config` holds **every** tunable number in the project in one place. Nothing is
hard-coded inside the model — the model reads from this object. That's what makes the
baseline-vs-proposed comparison fair: only the fields here change.

**Data fields:**
```python
    data_dir: str = "/home/reaper/PaperCancer/Brats datset/BraTS2020_training_data/content/data"
```
Absolute path to the folder of per-slice HDF5 files.
```python
    img_size: int = 240
```
Each MRI slice is 240×240 pixels.
```python
    in_modalities: int = 4          # T1, T1ce, T2, FLAIR
```
4 input channels — the four MRI sequences stacked together per slice.
```python
    num_classes: int = 3            # NCR/NET, ED, ET (raw BraTS channels)
```
3 output channels = the three raw tumour tissue classes.

**The experiment knob:**
```python
    n_slices: int = 5               # must be odd; center slice is the target
```
**The single most important line.** 5 = the proposed cross-slice model; 1 = plain
2-D U-Net baseline. Odd so there's a clear centre slice with equal neighbours either
side.
```python
    foreground_only: bool = True
```
Only train on slices that actually contain tumour (skip empty brain slices) — faster
and standard for 2-D BraTS.

**Model fields:**
```python
    base_ch: int = 48               # U-Net width -> ~6.7M params, fits 4GB
```
The width of the first U-Net layer; all other layers scale from it (48→96→192→384).
Chosen so the model is ~6.7M params (near the paper's 7.97M) and fits 4 GB of VRAM.
```python
    xslice_heads: int = 4           # attention heads in the cross-slice module
    xslice_layers: int = 2          # transformer encoder layers
```
The transformer's size: 4 attention heads, 2 stacked layers. Kept small on purpose
(lightweight, low memory).

**Training fields:**
```python
    batch_size: int = 4             # 2.5 GB VRAM. Do NOT raise on a 4 GB card.
    epochs: int = 80
    lr: float = 2e-4
    weight_decay: float = 1e-5
    val_frac: float = 0.2           # fraction of *volumes* held out
    num_workers: int = 4
    amp: bool = True                # mixed precision -> saves VRAM
    seed: int = 42
```
- `batch_size=4`: sized for the 4 GB GPU.
- `lr=2e-4`: the learning rate the sweep proved optimal.
- `val_frac=0.2`: 20% of **volumes** (patients) held out for validation.
- `amp=True`: mixed precision (float16 where safe) — halves memory use.
- `seed=42`: fixes randomness so runs are reproducible.

```python
    out_dir: str = ".../glioma_seg/runs"
    exp_name: str = "xslice_k5"
```
Where results are saved, and the default experiment name.

```python
    @property
    def half(self) -> int:
        return self.n_slices // 2
```
A helper that returns the "neighbour radius". For `n_slices=5`, `half=2` → we take 2
slices on each side of the centre. `@property` means you call it as `cfg.half`, not
`cfg.half()`.

---

# 2. `cst/dataset.py` — turning HDF5 files into training samples

**Top-level setup:**
```python
_FNAME = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")
```
A regular expression that pulls the **volume number** and **slice number** out of a
filename like `volume_23_slice_100.h5`. The two `(\d+)` groups capture those numbers.
```python
_META_CSV = ".../BraTS20 Training Metadata.csv"
```
Path to a CSV listing, per slice, how many tumour pixels it has. Used to find
tumour-containing slices instantly.

### `build_foreground_set()` — line ~29
```python
def build_foreground_set(meta_csv=_META_CSV):
    if not os.path.exists(meta_csv):
        return None
```
If the CSV isn't there, return `None` (the dataset then falls back to opening files).
```python
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
```
Reads every CSV row, extracts `(volume, slice)`, checks the three tumour pixel-count
columns, and if **any** is > 0 the slice has tumour → add it to the set `fg`.
Returns a `set` of `(volume, slice)` pairs — fast O(1) membership test later.

### `build_volume_index()` — line ~46
```python
def build_volume_index(data_dir):
    index = {}
    for path in glob.glob(os.path.join(data_dir, "*.h5")):
        m = _FNAME.search(os.path.basename(path))
        if not m: continue
        vol, sl = int(m.group(1)), int(m.group(2))
        index.setdefault(vol, []).append(sl)
    for vol in index:
        index[vol].sort()
    return index
```
Scans all `.h5` files and builds a dictionary `{volume: [sorted slice numbers]}`.
`setdefault(vol, [])` creates an empty list the first time a volume is seen. The final
loop sorts each volume's slices so neighbours are truly adjacent.

### `split_volumes()` — line ~60 ⭐ (the anti-leakage split)
```python
def split_volumes(index, val_frac, seed):
    vols = sorted(index.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(vols)
    n_val = int(round(len(vols) * val_frac))
    val_vols = set(vols[:n_val])
    train_vols = set(vols[n_val:])
    return train_vols, val_vols
```
- Takes the list of **volumes** (patients), shuffles them with a **fixed seed** (so
  the split is identical every run).
- Puts the first 20% into validation, the rest into training.
- **KEY POINT for the viva:** the split is by whole **volume/patient**, never by
  slice. Adjacent slices of one patient look almost identical; if some went to train
  and some to val, the model would effectively see the answer → **data leakage** and
  fake-high scores. Splitting whole patients prevents that.

### `class BraTSCrossSlice(Dataset)` — line ~71
A PyTorch `Dataset`: it must implement `__len__` (how many samples) and `__getitem__`
(return sample #i). PyTorch's `DataLoader` calls these to build batches.

```python
    def __init__(self, cfg, volumes, index, train=True, foreground=None):
        self.cfg = cfg
        self.index = index
        self.train = train
        self.half = cfg.half
        self.foreground = foreground
        self.samples = self._collect(volumes)
```
Stores config, the volume→slices index, whether this is the training set (controls
augmentation), the neighbour radius, the foreground set, and builds the list of
targets via `_collect`.

```python
    def _slice_path(self, vol, sl):
        return os.path.join(self.cfg.data_dir, f"volume_{vol}_slice_{sl}.h5")
```
Builds the file path for one slice.

```python
    def _has_tumour(self, vol, sl):
        with h5py.File(self._slice_path(vol, sl), "r") as f:
            return f["mask"][()].any()
```
Slow fallback (used only if there's no CSV): opens the file and checks if the mask has
any non-zero pixel.

```python
    def _collect(self, volumes):
        samples = []
        for vol in sorted(volumes):
            slices = self.index[vol]
            for sl in slices:
                if self.cfg.foreground_only:
                    if self.foreground is not None:
                        if (vol, sl) not in self.foreground:
                            continue
                    elif not self._has_tumour(vol, sl):
                        continue
                samples.append((vol, sl))
        return samples
```
Builds the list of `(volume, centre_slice)` **targets**. If `foreground_only`, it
keeps a slice only if it's in the CSV foreground set (fast path) or, without a CSV,
if opening it shows tumour (slow path). Each kept item is one training example.

```python
    def _load_slice(self, vol, sl, slices):
        sl = min(max(sl, slices[0]), slices[-1])
        with h5py.File(self._slice_path(vol, sl), "r") as f:
            img = f["image"][()].astype(np.float32)   # (240,240,4)
            msk = f["mask"][()].astype(np.float32)     # (240,240,3)
        return img, msk
```
Loads one slice's image + mask. The `min(max(...))` **clamps** the slice number to the
volume's valid range — so when we ask for a neighbour beyond the first/last slice, it
just repeats the edge slice instead of crashing. (**Edge handling** — a likely
question.)

```python
    def __len__(self):
        return len(self.samples)
```
Number of training examples = number of foreground centre-slices.

```python
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
```
**The core of the dataset.** For example #i, it loops over offsets `-2,-1,0,+1,+2`
(because `half=2`), loads each neighbour slice, and collects the 5 images into
`stack`. When `off==0` (the centre), it also grabs that slice's **mask** as the label
— because the model predicts the **centre** slice using the neighbours only as
context.

```python
        x = np.stack(stack, axis=0).transpose(0, 3, 1, 2)   # (5,4,240,240)
        y = center_mask.transpose(2, 0, 1)                  # (3,240,240)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
```
Stacks the 5 slices into one array shaped `(5, 4, 240, 240)` — the `transpose` moves
the channel axis to PyTorch's expected position (channels-first). The mask becomes
`(3, 240, 240)`. Both are converted from NumPy to PyTorch tensors.

```python
        if self.train:
            x, y = self._augment(x, y)
        return x, y
```
On the training set only, apply data augmentation, then return the `(input, label)`
pair.

```python
    @staticmethod
    def _augment(x, y):
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1]); y = torch.flip(y, dims=[-1])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-2]); y = torch.flip(y, dims=[-2])
        return x, y
```
Cheap, label-safe augmentation: 50% chance of a horizontal flip, 50% of a vertical
flip. **Crucially the same flip is applied to all 5 slices AND the mask** (`dims=[-1]`
is width, `[-2]` is height, which are shared axes) — so the geometry stays perfectly
aligned. `@staticmethod` = it doesn't need `self`.

---

# 3. `cst/model.py` — the neural network (know this file best)

### `conv_block()` — line ~22
```python
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )
```
The standard U-Net building block: two rounds of **Conv 3×3 → BatchNorm → ReLU**.
- `padding=1` keeps the image size unchanged.
- `bias=False` because BatchNorm makes the bias redundant.
- `BatchNorm` stabilises/speeds training; `ReLU` is the non-linearity.
- `inplace=True` saves a little memory.

### `class Encoder` — line ~31
```python
    def __init__(self, cin, base):
        self.e1 = conv_block(cin, base)          # 4  -> 48
        self.e2 = conv_block(base, base * 2)     # 48 -> 96
        self.e3 = conv_block(base * 2, base * 4) # 96 -> 192
        self.e4 = conv_block(base * 4, base * 8) # 192-> 384  (bottleneck)
        self.pool = nn.MaxPool2d(2)
```
Four conv blocks that progressively **double the channels** (48→96→192→384) and one
max-pool that **halves the spatial size** each stage. This is the classic U-Net
"contracting path" — it trades resolution for richer features.

```python
    def forward(self, x):
        s1 = self.e1(x)                  # 48ch  @240x240
        s2 = self.e2(self.pool(s1))      # 96ch  @120x120
        s3 = self.e3(self.pool(s2))      # 192ch @60x60
        b  = self.e4(self.pool(s3))      # 384ch @30x30  (bottleneck)
        return b, (s1, s2, s3)
```
Runs the input through the four stages. Returns the deepest features `b` (the
**bottleneck**) plus the three intermediate outputs `s1,s2,s3` — the **skip
connections** that the decoder will reuse to recover fine detail.

### `class CrossSliceTransformer` — line ~49 ⭐⭐ THE NOVELTY
```python
    def __init__(self, dim, heads, layers):
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            batch_first=True, dropout=0.1,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=layers)
        self.slice_pos = nn.Parameter(torch.zeros(1, 1, dim))
```
- Builds a standard Transformer encoder: `dim=384` feature size, `heads=4`,
  feed-forward size `dim*2=768`, 2 layers, 10% dropout.
- `batch_first=True` means tensors are shaped `(batch, sequence, features)`.
- `slice_pos` is a **learnable positional embedding** — a small trainable vector added
  to every token so the transformer can tell slice positions apart. `nn.Parameter`
  means it's learned during training.

```python
    def forward(self, feats):
        B, K, C, h, w = feats.shape
        x = feats.permute(0, 3, 4, 1, 2).reshape(B * h * w, K, C)
```
**This is the heart of the whole project.** Input `feats` = bottleneck features of all
K slices, shape `(B, K, 384, 30, 30)`.
- `permute` reorders to `(B, h, w, K, C)`, then `reshape` to `(B·h·w, K, C)`.
- Meaning: for **each spatial position** (each of the 30×30 locations, across the
  batch) we now have a **sequence of K tokens** — one token per slice, each a
  384-dim feature vector.

```python
        x = x + self.slice_pos
        x = self.tf(x)                       # attention ACROSS the K slices
```
Add the learnable slice-position embedding, then run the transformer. The
self-attention here operates **along the K (slice) axis** — so at every pixel
location, the centre slice can look at the same location in its neighbours and pull in
their context. **This "attention across slices, not across space" is what's novel.**

```python
        x = x.reshape(B, h, w, K, C)
        center = x[:, :, :, K // 2, :]                  # take centre slice
        return center.permute(0, 3, 1, 2).contiguous()  # (B, C, h, w)
```
Reshape back to a grid, then **keep only the centre slice's token** (`K//2` = index 2
for K=5) — because we only segment the centre slice. `permute` puts channels back
first → fused bottleneck `(B, 384, 30, 30)`. `.contiguous()` just tidies memory layout.

### `class Decoder` — line ~74
```python
    def __init__(self, base, num_classes):
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = conv_block(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)
```
The U-Net "expanding path". Each `ConvTranspose2d` **upsamples** (doubles spatial size,
halves channels). Each `conv_block` refines after merging a skip. The final `head` is
a 1×1 conv that maps 48 channels → the 3 output classes.

```python
    def forward(self, b, skips):
        s1, s2, s3 = skips
        x = self.d3(torch.cat([self.up3(b), s3], 1))
        x = self.d2(torch.cat([self.up2(x), s2], 1))
        x = self.d1(torch.cat([self.up1(x), s1], 1))
        return self.head(x)                             # logits (B, C, H, W)
```
Upsample the bottleneck, **concatenate the matching-resolution skip** (`torch.cat(...,
1)` joins along channels), refine, repeat three times back to full 240×240, then the
1×1 head produces raw scores (**logits**) for the 3 classes. Skips restore the fine
spatial detail that pooling threw away.

### `class CrossSliceUNet` — line ~93 (assembles everything)
```python
    def __init__(self, cfg):
        base = cfg.base_ch
        self.encoder = Encoder(cfg.in_modalities, base)
        self.use_xslice = cfg.n_slices > 1
        if self.use_xslice:
            self.xslice = CrossSliceTransformer(base * 8, cfg.xslice_heads, cfg.xslice_layers)
        self.decoder = Decoder(base, cfg.num_classes)
```
Builds the three parts. **`use_xslice = cfg.n_slices > 1`** is the ablation switch: the
transformer is only created when K>1. For K=1 (baseline) it's never built — the model
literally becomes a plain U-Net.

```python
    def forward(self, x):
        B, K, C, H, W = x.shape
        flat = x.reshape(B * K, C, H, W)
        b, skips = self.encoder(flat)                   # b: (B*K, 384, h, w)
        _, Cf, h, w = b.shape
```
Input `(B, K, 4, 240, 240)`. Flatten the slice axis into the batch so the **shared**
encoder processes every slice independently in one pass. Get bottleneck + skips.

```python
        if self.use_xslice:
            b = b.reshape(B, K, Cf, h, w)
            b = self.xslice(b)                          # fuse -> (B, 384, h, w)
            center = K // 2
            skips = tuple(
                s.reshape(B, K, *s.shape[1:])[:, center] for s in skips
            )
        else:
            b = b                                       # K==1, already centre
        return self.decoder(b, skips)                   # (B, num_classes, H, W)
```
- **Proposed path (K>1):** reshape bottleneck back to per-slice, run the Cross-Slice
  Transformer to fuse neighbours into the centre, and **keep only the centre slice's
  skip connections** for decoding (since we output the centre slice).
- **Baseline path (K==1):** do nothing extra — the single slice *is* the centre.
- Either way, decode → final logits `(B, 3, 240, 240)`.

**This `if/else` is the entire ablation.** Same class, same weights structure, one flag.

---

# 4. `cst/losses.py` — the training objective

```python
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0, bce_weight=0.5):
        self.smooth = smooth
        self.bce_weight = bce_weight
```
A combined loss. `bce_weight=0.5` blends Dice and BCE equally. `smooth=1.0` prevents
divide-by-zero in the Dice formula.

```python
    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
```
Convert raw scores to probabilities with **sigmoid** (not softmax) — because the 3
classes are **independent** (a voxel can belong to more than one region).

```python
        dims = (0, 2, 3)
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1 - dice.mean()
```
Computes **soft Dice** per class channel: `2·intersection / (prediction + target)`.
`dims=(0,2,3)` sums over batch, height, width — leaving one Dice value per class.
Dice is high when overlap is high, so the **loss** is `1 − Dice`.

```python
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return (1 - self.bce_weight) * dice_loss + self.bce_weight * bce
```
Adds **binary cross-entropy** (per-pixel classification loss), computed directly from
logits (numerically stable). Final loss = weighted sum of the two.
- **Why both?** Dice handles the severe class imbalance (tumour is a tiny fraction of
  the image); BCE gives smooth, stable per-pixel gradients. The sweep confirmed the
  combination beats either alone.

---

# 5. `cst/metrics.py` — how performance is scored

```python
CH_NCR, CH_ED, CH_ET = 0, 1, 2
```
Names for the 3 raw output channels: NCR/NET (necrotic core), ED (edema), ET
(enhancing tumour).

```python
def to_regions(binary):
    ncr, ed, et = binary[:, CH_NCR], binary[:, CH_ED], binary[:, CH_ET]
    wt = ((ncr + ed + et) > 0).float()
    tc = ((ncr + et) > 0).float()
    return {"WT": wt, "TC": tc, "ET": et}
```
Converts the 3 raw channels into the **3 nested BraTS evaluation regions**:
- **WT (Whole Tumour)** = everything (NCR ∪ ED ∪ ET)
- **TC (Tumour Core)** = NCR ∪ ET (the solid core, without edema)
- **ET (Enhancing Tumour)** = ET only
These are the official challenge regions — using them makes your scores comparable to
published papers.

```python
def dice_score(pred_bin, target_bin, eps=1e-6):
    preds = to_regions(pred_bin)
    tgts = to_regions(target_bin)
    out = {}
    for r in ("WT", "TC", "ET"):
        p, t = preds[r], tgts[r]
        dims = (1, 2)
        inter = (p * t).sum(dims)
        denom = p.sum(dims) + t.sum(dims)
        d = (2 * inter + eps) / (denom + eps)
        out[r] = d.mean().item()
    out["mean"] = (out["WT"] + out["TC"] + out["ET"]) / 3
    return out
```
Computes the **Dice coefficient** (overlap: `2|A∩B|/(|A|+|B|)`) for each region,
averages over the batch, and returns WT, TC, ET plus their **mean** (the single number
we track for "best model"). `eps` avoids division by zero on empty masks.

---

# 6. `cst/train.py` — the training loop

**Imports (lines 7–20):** standard libs + the project's own modules.

```python
def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
```
Fixes all random number generators for reproducibility.

### `evaluate()` — line 27
```python
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    agg = {"WT": 0, "TC": 0, "ET": 0, "mean": 0}
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).float()
        d = dice_score(pred, y)
        for k in agg: agg[k] += d[k]
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}
```
- `@torch.no_grad()` + `model.eval()`: turn off gradient tracking and dropout/BN
  updates — this is evaluation, not training.
- For each validation batch: run the model, convert logits to a **binary** mask via
  `sigmoid > 0.5`, score it with `dice_score`, and accumulate.
- Returns the **average** WT/TC/ET/mean Dice over the validation set.

### `main()` — line 43

**Argument parsing (lines 44–59):** defines command-line flags. The basic ones
(`--exp`, `--n_slices`, `--epochs`, `--batch_size`, `--limit_vols`) and the
**sweep knobs** I added (`--optimizer`, `--scheduler`, `--lr`, `--bce_weight`) so each
design factor can be changed from the command line without editing code.

**Config overrides (lines 61–66):** start from `Config()` defaults, then override any
field the user passed on the command line.

```python
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
```
Seed everything, pick GPU if available, and make the output folder for this run.

**Data setup (lines 73–91):**
```python
    index = build_volume_index(cfg.data_dir)
    train_vols, val_vols = split_volumes(index, cfg.val_frac, cfg.seed)
    if args.limit_vols:
        train_vols = set(list(train_vols)[:args.limit_vols])
        val_vols = set(list(val_vols)[:max(2, args.limit_vols // 4)])
```
Index the files, split by volume, and — for quick tests — optionally keep only the
first N volumes (`--limit_vols`, used by the sweep).
```python
    fg = build_foreground_set()
    train_ds = BraTSCrossSlice(cfg, train_vols, index, train=True, foreground=fg)
    val_ds   = BraTSCrossSlice(cfg, val_vols,   index, train=False, foreground=fg)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True)
```
Build the foreground set, the two datasets, and their `DataLoader`s (which batch and
parallel-load with 4 worker processes; `shuffle=True` only for training; `drop_last`
avoids a ragged final batch; `pin_memory` speeds CPU→GPU transfer).

```python
    model = CrossSliceUNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
```
Build the model, move it to GPU, and count its parameters (in millions).

**Optimizer (lines 98–110):** an `if/elif` chain that builds AdamW / Adam / SGD /
RMSprop depending on `--optimizer`. Default AdamW. (SGD adds momentum 0.9.) This
swappability is what let the sweep compare optimizers fairly.

**Scheduler (lines 112–119):** builds a cosine / step / constant learning-rate
schedule depending on `--scheduler`. Cosine slowly anneals the LR to zero over
training; "none" keeps it constant.

```python
    loss_fn = DiceBCELoss(bce_weight=args.bce_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
```
Create the loss (with the chosen Dice/BCE weighting) and the mixed-precision gradient
scaler (needed so tiny float16 gradients don't underflow to zero).

**The epoch loop (lines 126–160):**
```python
    best = -1.0
    history = []
    for ep in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for i, (x, y) in enumerate(train_ld):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
```
For each epoch, for each batch: move data to GPU, **zero old gradients**, run a
**mixed-precision forward pass** (`autocast`), compute the loss, **backpropagate**
(`scaler.scale(...).backward()`), and **update the weights** (`scaler.step`,
`scaler.update`). `running` accumulates the loss for reporting.

```python
        sched.step()
        val = evaluate(model, val_ld, device)
        rec = {"epoch": ep, "train_loss": running / len(train_ld), **val}
        history.append(rec)
```
After each epoch: step the LR schedule, evaluate on validation, and record the epoch's
loss + Dice scores.

```python
        if val["mean"] > best:
            best = val["mean"]
            torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                        "val": val, "epoch": ep}, os.path.join(run_dir, "best.pth"))
```
**Checkpointing:** save the model to `best.pth` **only when** the mean validation Dice
improves. So even if the model gets worse later, you keep the best version.

```python
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
```
Write the full per-epoch history to disk every epoch (so nothing is lost if it stops).

**Summary (lines 162–179):** after training, find the best epoch, and write a compact
`summary.json` with the settings + best WT/TC/ET/mean — this is the file `sweep.py`
reads to build the comparison table.

```python
if __name__ == "__main__":
    main()
```
Standard Python entry point — runs `main()` when the file is executed.

---

# 7. `sweep.py` — the design-justification experiments (know the idea, not every line)

- `CHOSEN` (line ~29): the chosen configuration (K5 / AdamW / cosine / Dice+BCE /
  lr 2e-4).
- `MATRIX` (line ~41): a list of 11 runs, each varying **one** factor off `CHOSEN`
  (K=1/3, adam/sgd, none/step, dice-only/bce-only, lr 1e-3/1e-4), all on a fixed
  16-volume / 10-epoch budget with the same seed.
- `run_one()`: launches one training run via `subprocess` with the right flags.
- `collect()` (line ~66): reads every run's `summary.json` and writes a combined
  `runs/sweep_results.csv` + prints a comparison table.
- **The purpose:** produce evidence that each design choice is data-driven, not
  arbitrary — exactly what justifies the choices in the report.

---

# 8. One-paragraph summary to say out loud

> "Data comes in as per-slice HDF5 files. `dataset.py` groups them by patient, splits
> by patient to avoid leakage, and for each target slice builds a stack of 5 adjacent
> slices with the centre slice's mask as the label. `model.py` runs a shared U-Net
> encoder on every slice, then my Cross-Slice Transformer applies self-attention
> across the slice axis at each pixel location to fuse the neighbours into the centre,
> and a U-Net decoder produces the segmentation. Setting `n_slices=1` skips the
> transformer, giving a plain-U-Net baseline from the same code. `losses.py` trains it
> with Dice+BCE, `metrics.py` scores it on the WT/TC/ET regions, and `train.py` runs
> the loop and saves the best checkpoint. `sweep.py` is how I justified every design
> choice with controlled experiments."
