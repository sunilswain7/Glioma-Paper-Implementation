"""Design-justification sweep.

Runs a controlled comparative study: a fixed volume subset and epoch budget,
seed held constant, with ONE design factor varied per run. The point is not
absolute SOTA Dice but a fair *relative ranking* that justifies each design
choice (cross-slice K, optimizer, LR schedule, loss, learning rate).

Each run writes runs/<exp>/summary.json. After all runs finish, this script
aggregates them into runs/sweep_results.csv and prints a comparison table.

Usage:
    python sweep.py            # run the full matrix, then aggregate
    python sweep.py --collect  # only aggregate existing summaries
"""
import os
import csv
import json
import argparse
import subprocess

LIMIT_VOLS = 16
EPOCHS = 10

# chosen config = K5 / adamw / cosine / bce_weight 0.5 / lr 2e-4
CHOSEN = dict(n_slices=5, optimizer="adamw", scheduler="cosine",
              bce_weight=0.5, lr=2e-4)


def cfg(**over):
    c = dict(CHOSEN); c.update(over); return c


# (factor, exp_name, overrides)  -- the chosen config is run once as the anchor
MATRIX = [
    ("anchor",    "chosen_k5",     cfg()),
    # cross-slice contribution (the novelty)
    ("xslice",    "abl_k1",        cfg(n_slices=1)),
    ("xslice",    "abl_k3",        cfg(n_slices=3)),
    # optimizer
    ("optimizer", "opt_adam",      cfg(optimizer="adam")),
    ("optimizer", "opt_sgd",       cfg(optimizer="sgd")),
    # lr schedule
    ("scheduler", "sched_none",    cfg(scheduler="none")),
    ("scheduler", "sched_step",    cfg(scheduler="step")),
    # loss
    ("loss",      "loss_diceonly", cfg(bce_weight=0.0)),
    ("loss",      "loss_bceonly",  cfg(bce_weight=1.0)),
    # learning rate
    ("lr",        "lr_1e3",        cfg(lr=1e-3)),
    ("lr",        "lr_1e4",        cfg(lr=1e-4)),
]

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


def run_one(exp, c):
    cmd = [
        "python", "-m", "cst.train",
        "--exp", exp,
        "--limit_vols", str(LIMIT_VOLS),
        "--epochs", str(EPOCHS),
        "--n_slices", str(c["n_slices"]),
        "--optimizer", c["optimizer"],
        "--scheduler", c["scheduler"],
        "--bce_weight", str(c["bce_weight"]),
        "--lr", str(c["lr"]),
    ]
    print(f"\n=== {exp} :: {c} ===", flush=True)
    subprocess.run(cmd, cwd=os.path.dirname(__file__), check=True)


def collect():
    rows = []
    for factor, exp, _ in MATRIX:
        path = os.path.join(RUNS_DIR, exp, "summary.json")
        if not os.path.exists(path):
            print(f"  (missing) {exp}")
            continue
        with open(path) as f:
            s = json.load(f)
        s["factor"] = factor
        rows.append(s)

    if not rows:
        print("No summaries found yet.")
        return

    cols = ["factor", "exp", "n_slices", "optimizer", "scheduler", "lr",
            "bce_weight", "params_M", "best_epoch", "WT", "TC", "ET", "mean"]
    out_csv = os.path.join(RUNS_DIR, "sweep_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    # pretty table
    print("\n" + "=" * 92)
    print(f"{'factor':<10}{'exp':<15}{'K':>2} {'opt':<8}{'sched':<7}"
          f"{'lr':>7} {'bce':>4} {'par':>5} {'WT':>6}{'TC':>6}{'ET':>6}{'mean':>7}")
    print("-" * 92)
    for r in sorted(rows, key=lambda x: (x["factor"], -x["mean"])):
        print(f"{r['factor']:<10}{r['exp']:<15}{r['n_slices']:>2} "
              f"{r['optimizer']:<8}{r['scheduler']:<7}{r['lr']:>7g} "
              f"{r['bce_weight']:>4} {r['params_M']:>5} "
              f"{r['WT']:>6.3f}{r['TC']:>6.3f}{r['ET']:>6.3f}{r['mean']:>7.3f}")
    print("=" * 92)
    print(f"\nSaved: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true",
                    help="only aggregate existing summaries")
    args = ap.parse_args()

    if not args.collect:
        for _, exp, c in MATRIX:
            try:
                run_one(exp, c)
            except subprocess.CalledProcessError as e:
                print(f"  !! {exp} failed: {e}")
    collect()


if __name__ == "__main__":
    main()
