"""Aggregate every runs/*/summary.json into runs/results.csv and print a table,
plus paired K5-vs-K1 (and K3-vs-K1) comparisons over the same test patients."""
import os
import sys
import csv
import json
import glob
import numpy as np


def main(out):
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "*", "summary.json"))):
        s = json.load(open(p))
        t = s["test"]
        rows.append({"exp": s["exp"], "K": s["n_slices"], "fold": s["fold"],
                     "n_train": s["n_train"], "params_M": s["params_M"],
                     "best_ep": s["best_epoch"], "et_min": s["et_min"],
                     "WT": t["WT"], "TC": t["TC"], "ET": t["ET"], "mean": t["mean_dice"],
                     "HD95_WT": t["HD95_WT"], "HD95_TC": t["HD95_TC"], "HD95_ET": t["HD95_ET"],
                     "fp_slices": t["fp_slices"], "hours": s["hours"]})
    if not rows:
        print("no finished runs yet"); return
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'exp':<18}{'K':>2}{'ntr':>5}{'WT':>7}{'TC':>7}{'ET':>7}{'mean':>7}"
          f"{'HD_WT':>7}{'HD_TC':>7}{'HD_ET':>7}{'FPsl':>6}{'h':>5}")
    for r in rows:
        print(f"{r['exp']:<18}{r['K']:>2}{r['n_train']:>5}{r['WT']:>7.3f}{r['TC']:>7.3f}"
              f"{r['ET']:>7.3f}{r['mean']:>7.3f}{r['HD95_WT']:>7.1f}{r['HD95_TC']:>7.1f}"
              f"{r['HD95_ET']:>7.1f}{r['fp_slices']:>6.1f}{r['hours']:>5.1f}")

    # paired per-patient comparisons: same prefix/fold, different K
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return
    per = {}
    for p in glob.glob(os.path.join(out, "*", "test_per_patient.json")):
        exp = os.path.basename(os.path.dirname(p))
        per[exp] = {r["pid"]: r for r in json.load(open(p))["postproc"]}
    print("\npaired test-patient comparison vs K=1 (pooled over folds)")
    for prefix in sorted({e.split("_k")[0] for e in per}):
        for kb in (3, 5):
            a_all, b_all = {}, {}
            for e in per:
                if e.startswith(prefix + "_k1_"):
                    eb = e.replace("_k1_", f"_k{kb}_")
                    if eb in per:
                        for pid in per[e]:
                            a_all[(e, pid)], b_all[(e, pid)] = per[e][pid], per[eb][pid]
            if not a_all:
                continue
            keys = list(a_all)
            line = f"  {prefix} K{kb}-K1 (n={len(keys)}):"
            for m in ("WT", "TC", "ET", "HD95_WT", "HD95_TC", "HD95_ET", "fp_slices"):
                d = np.array([b_all[k][m] - a_all[k][m] for k in keys])
                pv = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                line += f" {m} {d.mean():+.3f} (p={pv:.3f})"
            print(line)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Project1/runs"))
