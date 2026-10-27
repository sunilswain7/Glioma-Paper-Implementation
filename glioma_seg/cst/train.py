"""Training loop for the Context-Aware Cross-Slice Transformer.

Usage:
    python -m cst.train --exp xslice_k5 --n_slices 5   # proposed model
    python -m cst.train --exp baseline_k1 --n_slices 1 # baseline (ablation)
"""
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import (
    BraTSCrossSlice, build_volume_index, split_volumes, build_foreground_set,
)
from .model import CrossSliceUNet
from .losses import DiceBCELoss
from .metrics import dice_score


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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
        for k in agg:
            agg[k] += d[k]
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--n_slices", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--limit_vols", type=int, default=None,
                    help="use only N volumes (fast smoke test)")
    # ---- design-justification sweep knobs (one flag per factor) ----
    ap.add_argument("--optimizer", default="adamw",
                    choices=["adamw", "adam", "sgd", "rmsprop"])
    ap.add_argument("--scheduler", default="cosine",
                    choices=["cosine", "step", "none"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--bce_weight", type=float, default=0.5,
                    help="1.0=pure BCE, 0.0=pure Dice, 0.5=Dice+BCE (default)")
    args = ap.parse_args()

    cfg = Config()
    if args.exp: cfg.exp_name = args.exp
    if args.n_slices is not None: cfg.n_slices = args.n_slices
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.lr is not None: cfg.lr = args.lr

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    print("Indexing volumes...")
    index = build_volume_index(cfg.data_dir)
    train_vols, val_vols = split_volumes(index, cfg.val_frac, cfg.seed)
    if args.limit_vols:
        train_vols = set(list(train_vols)[:args.limit_vols])
        val_vols = set(list(val_vols)[:max(2, args.limit_vols // 4)])
    print(f"  volumes: {len(train_vols)} train / {len(val_vols)} val")

    fg = build_foreground_set()
    print(f"  foreground: {'CSV' if fg is not None else 'scan (slow)'}")
    train_ds = BraTSCrossSlice(cfg, train_vols, index, train=True, foreground=fg)
    val_ds = BraTSCrossSlice(cfg, val_vols, index, train=False, foreground=fg)
    print(f"  slices : {len(train_ds)} train / {len(val_ds)} val")

    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True,
                          drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    model = CrossSliceUNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  model  : {n_params:.2f}M params | cross-slice="
          f"{model.use_xslice} (K={cfg.n_slices})")

    # ---- optimizer (swappable for the design-justification sweep) ----
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    elif args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
    elif args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9,
                              weight_decay=cfg.weight_decay)
    else:  # rmsprop
        opt = torch.optim.RMSprop(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    # ---- lr schedule ----
    if args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    elif args.scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    else:  # none -> constant lr
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0)

    loss_fn = DiceBCELoss(bce_weight=args.bce_weight)
    print(f"  train  : opt={args.optimizer} sched={args.scheduler} "
          f"lr={cfg.lr:g} bce_weight={args.bce_weight}")
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

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
            if i % 50 == 0:
                print(f"  ep{ep} it{i}/{len(train_ld)} loss={loss.item():.4f}")
        sched.step()

        val = evaluate(model, val_ld, device)
        rec = {"epoch": ep, "train_loss": running / len(train_ld), **val}
        history.append(rec)
        print(f"[ep {ep}] loss={rec['train_loss']:.4f} | "
              f"Dice WT={val['WT']:.3f} TC={val['TC']:.3f} "
              f"ET={val['ET']:.3f} mean={val['mean']:.3f}")

        if val["mean"] > best:
            best = val["mean"]
            torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                        "val": val, "epoch": ep},
                       os.path.join(run_dir, "best.pth"))
            print(f"  saved best (mean Dice {best:.3f})")

        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    best_rec = max(history, key=lambda r: r["mean"])
    summary = {
        "exp": cfg.exp_name,
        "n_slices": cfg.n_slices,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "lr": cfg.lr,
        "bce_weight": args.bce_weight,
        "epochs": cfg.epochs,
        "params_M": round(n_params, 2),
        "best_epoch": best_rec["epoch"],
        "WT": round(best_rec["WT"], 4),
        "TC": round(best_rec["TC"], 4),
        "ET": round(best_rec["ET"], 4),
        "mean": round(best_rec["mean"], 4),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Best mean Dice = {best:.3f}. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
