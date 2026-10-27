"""Central configuration for the Context-Aware Cross-Slice Transformer project.

Everything tunable lives here so experiments (baseline vs cross-slice) differ by
one flag only. This is what makes the ablation study clean.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- data ----
    data_dir: str = (
        "/home/reaper/PaperCancer/Brats datset/"
        "BraTS2020_training_data/content/data"
    )
    img_size: int = 240
    in_modalities: int = 4          # T1, T1ce, T2, FLAIR
    num_classes: int = 3            # NCR/NET, ED, ET (raw BraTS channels)

    # ---- the cross-slice knob (THE experiment) ----
    #   n_slices = 1  -> baseline 2D U-Net (no cross-slice context)
    #   n_slices = 5  -> Context-Aware Cross-Slice Transformer
    n_slices: int = 5               # must be odd; center slice is the target

    # keep only slices whose CENTER contains tumour (fast, standard for 2D BraTS)
    foreground_only: bool = True

    # ---- model ----
    base_ch: int = 48               # U-Net width. 48 -> ~6.7M params (~paper's
                                    # 7.97M), 2.5 GB VRAM at batch 4 -> safe on 4 GB.
    xslice_heads: int = 4           # attention heads in the cross-slice module
    xslice_layers: int = 2          # transformer encoder layers

    # ---- training ----
    batch_size: int = 4             # 2.5 GB VRAM. Do NOT raise on a 4 GB card.
    epochs: int = 80                # let it run long; best checkpoint is kept.
    lr: float = 2e-4
    weight_decay: float = 1e-5
    val_frac: float = 0.2           # fraction of *volumes* held out
    num_workers: int = 4
    amp: bool = True                # mixed precision -> saves VRAM
    seed: int = 42

    # ---- io ----
    out_dir: str = "/home/reaper/PaperCancer/glioma_seg/runs"
    exp_name: str = "xslice_k5"

    @property
    def half(self) -> int:
        return self.n_slices // 2
