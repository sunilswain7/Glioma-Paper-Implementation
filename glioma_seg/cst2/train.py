"""v2 training: fixed protocol, HPC-ready (Astra A100, torch 2.5.1).

  python -m cst2.train --data_dir DATA --out_dir RUNS --exp xslice_k5_f0 --n_slices 5 --fold 0

Changes vs v1 (cst/train.py):
  * patient-level train / val / TEST split (k-fold); test is touched only once, at the end
  * trains on ALL slices, tumour slices sampled 5x more often (v1: tumour slices only)
  * region targets WT/TC/ET; GPU augmentation (flip, rotate/scale, intensity, noise)
  * AdamW + linear warmup + cosine, grad clipping, EMA weights, bf16 autocast
  * model selection on VOLUMETRIC 3D Dice of validation patients (EMA model)
  * automatic resume from last.pth + --time_limit_h (safe under scheduler walltime)
  * final test evaluation: 3D Dice, HD95, false-positive slices, ET post-processing
"""
import os
import copy
import json
import math
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import Volumes, SliceStacks, list_patients, make_split
from .model import CrossSliceUNet
from .losses import DiceBCELoss
from .augment import gpu_augment
from .evaluate import evaluate_split, summarise, tune_et_min


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n_slices", type=int, default=5)
    ap.add_argument("--base_ch", type=int, default=48)
    ap.add_argument("--xslice_layers", type=int, default=2)
    ap.add_argument("--params_json", default=None,
                    help="JSON of tuned hyper-parameters (from cst2.tune) overriding defaults")
    # split
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--limit_train", type=int, default=None,
                    help="data-size ablation: use only N training patients")
    ap.add_argument("--seed", type=int, default=0, help="training seed (init/sampling)")
    # optimisation
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--iters_per_epoch", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--bce_weight", type=float, default=0.5)
    ap.add_argument("--fg_weight", type=float, default=5.0)
    ap.add_argument("--no_aug", action="store_true")
    # validation / runtime
    ap.add_argument("--val_every", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10, help="in validation checks")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--in_ram", action="store_true", help="load train volumes into RAM")
    ap.add_argument("--time_limit_h", type=float, default=None)
    ap.add_argument("--deadline_h", type=float, default=None,
                    help="stop training after this many hours and still do the final test evaluation")
    ap.add_argument("--tune", action="store_true", help=argparse.SUPPRESS)  # set by cst2.tune
    args = ap.parse_args(argv)
    if args.params_json:
        with open(args.params_json) as f:
            tuned = json.load(f)["params"]
        explicit = {a.split("=")[0].lstrip("-") for a in (argv or __import__("sys").argv[1:])
                    if a.startswith("--")}
        for k, v in tuned.items():          # command-line flags (e.g. --n_slices 1) win
            if k not in explicit:
                setattr(args, k, v)
    return args


@torch.no_grad()
def ema_update(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for e, b in zip(ema.buffers(), model.buffers()):
        e.copy_(b)


def log(run_dir, msg):
    print(msg, flush=True)
    with open(os.path.join(run_dir, "log.txt"), "a") as f:
        f.write(msg + "\n")


def main(argv=None, trial=None):
    """trial: optuna trial when called from cst2.tune (reports val Dice, may prune)."""
    args = get_args(argv)
    t_start = time.time()
    run_dir = os.path.join(args.out_dir, args.exp)
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "summary.json")):
        print(f"{args.exp}: already finished, skipping"); return
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va, te = make_split(list_patients(args.data_dir), args.fold, args.n_folds,
                            seed=args.split_seed, limit_train=args.limit_train)
    with open(os.path.join(run_dir, "split.json"), "w") as f:
        json.dump({"train": tr, "val": va, "test": te}, f, indent=1)
    train_ds = SliceStacks(Volumes(args.data_dir, tr, in_ram=args.in_ram), args.n_slices)
    val_vols = Volumes(args.data_dir, va)
    n_samples = args.iters_per_epoch * args.batch_size
    sampler = WeightedRandomSampler(train_ds.sample_weights(args.fg_weight),
                                    num_samples=n_samples, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)

    model = CrossSliceUNet(args.n_slices, args.base_ch, layers=args.xslice_layers).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total, warm = args.epochs * args.iters_per_epoch, args.warmup_epochs * args.iters_per_epoch
    min_ratio = 1e-6 / args.lr
    lr_at = lambda it: (it + 1) / warm if it < warm else \
        min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, total - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    loss_fn = DiceBCELoss(bce_weight=args.bce_weight)

    start_ep, best, bad, history = 1, -1.0, 0, []
    last_path = os.path.join(run_dir, "last.pth")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, best, bad, history = ck["epoch"] + 1, ck["best"], ck["bad"], ck["history"]
        log(run_dir, f"resumed from epoch {ck['epoch']} (best {best:.4f})")
    log(run_dir, f"{args.exp}: K={args.n_slices} params={n_params:.2f}M | patients "
                 f"train {len(tr)} val {len(va)} test {len(te)} | slices {len(train_ds)} "
                 f"(fg {int(train_ds.is_fg.sum())}) | device {device}")

    stop_reason = "epochs"
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            if not args.no_aug:
                x, y = gpu_augment(x, y)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); sched.step()
            ema_update(ema, model, args.ema)
            running += loss.item()
        rec = {"epoch": ep, "train_loss": running / args.iters_per_epoch,
               "lr": opt.param_groups[0]["lr"], "epoch_s": round(time.time() - t0, 1),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if device == "cuda" else 0}

        if ep % args.val_every == 0 or ep == args.epochs:
            rows, _ = evaluate_split(ema, val_vols, args.n_slices, device, with_hd=False)
            s = summarise(rows)
            rec.update({f"val_{k}": round(s[k], 4) for k in ("WT", "TC", "ET", "mean_dice", "fp_slices")})
            if trial is not None:
                import optuna
                trial.report(s["mean_dice"], ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if s["mean_dice"] > best:
                best, bad = s["mean_dice"], 0
                if not args.tune:
                    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                                "args": vars(args), "epoch": ep, "val": s},
                               os.path.join(run_dir, "best.pth"))
            else:
                bad += 1
        history.append(rec)
        log(run_dir, json.dumps(rec))
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
        if not args.tune:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                        "best": best, "bad": bad, "history": history}, last_path)
        if bad >= args.patience:
            stop_reason = "early_stop"; break
        if args.deadline_h and (time.time() - t_start) / 3600 > args.deadline_h:
            stop_reason = "deadline"; break
        if args.time_limit_h and (time.time() - t_start) / 3600 > args.time_limit_h:
            log(run_dir, "time limit reached -> resubmit the same job to resume"); return

    if args.tune:
        return best

    # ---------------- final: tune ET threshold on VAL, evaluate TEST once ----------------
    ck = torch.load(os.path.join(run_dir, "best.pth"), map_location=device, weights_only=False)
    ema.load_state_dict(ck["ema"])
    _, cache = evaluate_split(ema, val_vols, args.n_slices, device, with_hd=False, keep_probs=True)
    et_min = tune_et_min(cache); del cache
    ck["et_min"] = et_min
    torch.save(ck, os.path.join(run_dir, "best.pth"))
    test_vols = Volumes(args.data_dir, te)
    rows_raw, _ = evaluate_split(ema, test_vols, args.n_slices, device, et_min=0)
    rows, _ = evaluate_split(ema, test_vols, args.n_slices, device, et_min=et_min)
    with open(os.path.join(run_dir, "test_per_patient.json"), "w") as f:
        json.dump({"et_min": et_min, "postproc": rows, "raw": rows_raw}, f, indent=1)
    summary = {"exp": args.exp, "n_slices": args.n_slices, "fold": args.fold, "seed": args.seed,
               "limit_train": args.limit_train, "n_train": len(tr), "params_M": round(n_params, 2),
               "best_epoch": ck["epoch"], "stop": stop_reason, "et_min": et_min,
               "val_mean_dice": round(best, 4),
               "hparams": {k: getattr(args, k) for k in ("lr", "weight_decay", "bce_weight",
                           "fg_weight", "base_ch", "xslice_layers", "batch_size", "epochs")},
               "test": {k: round(v, 4) for k, v in summarise(rows).items()},
               "test_no_postproc": {k: round(v, 4) for k, v in summarise(rows_raw).items()},
               "hours": round((time.time() - t_start) / 3600, 2)}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(run_dir, "TEST " + json.dumps(summary["test"]))


if __name__ == "__main__":
    main()
