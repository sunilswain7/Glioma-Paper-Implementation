"""Build report figures + statistics from the Astra results bundle.

    python3 report/make_figures.py <extracted results dir>
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

R = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
C1, C5 = "#8a8f98", "#1f5fa8"          # baseline grey, proposed blue
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
MET = ["WT", "TC", "ET", "HD95_WT", "HD95_TC", "HD95_ET", "fp_slices"]


def load(exp):
    p = os.path.join(R, exp)
    return (json.load(open(f"{p}/summary.json")), json.load(open(f"{p}/history.json")),
            {r["pid"]: r for r in json.load(open(f"{p}/test_per_patient.json"))["postproc"]})


folds = range(5)
S = {k: [load(f"safe_k{k}_f{f}") for f in folds] for k in (1, 5)}
stats = {}

# ---- per-fold table + fold-level summary
pf = {k: {m: [s[0]["test"][m] for s in S[k]] for m in MET + ["mean_dice"]} for k in (1, 5)}
stats["per_fold"] = pf
stats["fold_mean_sd"] = {k: {m: (float(np.mean(v)), float(np.std(v, ddof=1))) for m, v in pf[k].items()}
                         for k in (1, 5)}

# ---- pooled per-patient paired tests (369 patients)
pairs = []
for f in folds:
    a, b = S[1][f][2], S[5][f][2]
    pairs += [(a[p], b[p]) for p in a]
stats["n_patients"] = len(pairs)
pooled = {}
for m in MET + ["mean"]:
    if m == "mean":
        x = np.array([np.mean([a[r] for r in ("WT", "TC", "ET")]) for a, _ in pairs])
        y = np.array([np.mean([b[r] for r in ("WT", "TC", "ET")]) for _, b in pairs])
    else:
        x = np.array([a[m] for a, _ in pairs]); y = np.array([b[m] for _, b in pairs])
    d = y - x
    pooled[m] = {"k1_mean": float(x.mean()), "k5_mean": float(y.mean()), "k1_median": float(np.median(x)),
                 "k5_median": float(np.median(y)), "diff_mean": float(d.mean()), "diff_median": float(np.median(d)),
                 "k5_better_pct": float(np.mean(d > 0) * 100 if not m.startswith(("HD", "fp")) else np.mean(d < 0) * 100),
                 "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
stats["pooled"] = pooled

# ---- overnight (confounded) experiment, folds with both runs
lo = [f for f in folds if os.path.exists(f"{R}/long_kx_f{f}/summary.json")]
stats["long_folds"] = lo
stats["long"] = {f: {k: json.load(open(f"{R}/long_{k}_f{f}/summary.json")) for k in ("k1", "kx")} for f in lo}
stats["optuna"] = json.load(open(f"{R}/optuna_best.json"))
stats["optuna_trials"] = json.load(open(f"{R}/optuna_trials.json"))
json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1, default=float)

# ---- Fig: per-fold mean Dice
x = np.arange(5); w = 0.38
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.bar(x - w / 2, pf[1]["mean_dice"], w, color=C1, label="Baseline 2D U-Net (K=1)")
ax.bar(x + w / 2, pf[5]["mean_dice"], w, color=C5, label="Cross-slice transformer (K=5)")
for i in x:
    ax.text(i - w / 2, pf[1]["mean_dice"][i] + .002, f"{pf[1]['mean_dice'][i]:.3f}", ha="center", fontsize=7)
    ax.text(i + w / 2, pf[5]["mean_dice"][i] + .002, f"{pf[5]['mean_dice'][i]:.3f}", ha="center", fontsize=7)
ax.set_xticks(x, [f"Fold {i}" for i in x]); ax.set_ylim(0.78, 0.89); ax.set_ylabel("Mean 3D Dice (WT, TC, ET)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.savefig(f"{OUT}/fold_mean_dice.png"); plt.close(fig)

# ---- Fig: HD95 per region (fold means)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, m in zip(axs, ("HD95_WT", "HD95_TC", "HD95_ET")):
    ax.bar([0, 1], [np.mean(pf[1][m]), np.mean(pf[5][m])], yerr=[np.std(pf[1][m], ddof=1), np.std(pf[5][m], ddof=1)],
           color=[C1, C5], capsize=4)
    ax.set_xticks([0, 1], ["K=1", "K=5"]); ax.set_title(m.replace("HD95_", "HD95 "), fontsize=10)
axs[0].set_ylabel("mm (lower is better)")
fig.tight_layout(); fig.savefig(f"{OUT}/hd95.png"); plt.close(fig)

# ---- Fig: per-patient paired differences (boxplots)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
dd = [[b[r] - a[r] for a, b in pairs] for r in ("WT", "TC", "ET")]
axs[0].boxplot(dd, showfliers=False); axs[0].axhline(0, color="k", lw=.8, ls="--")
axs[0].set_xticks([1, 2, 3], ["WT", "TC", "ET"]); axs[0].set_ylabel("Dice(K=5) − Dice(K=1)")
axs[0].set_title("Per-patient Dice difference", fontsize=10)
hh = [[b[r] - a[r] for a, b in pairs] for r in ("HD95_WT", "HD95_TC", "HD95_ET")]
axs[1].boxplot(hh, showfliers=False); axs[1].axhline(0, color="k", lw=.8, ls="--")
axs[1].set_xticks([1, 2, 3], ["WT", "TC", "ET"]); axs[1].set_ylabel("HD95(K=5) − HD95(K=1) [mm]")
axs[1].set_title("Per-patient HD95 difference", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/paired_diff.png"); plt.close(fig)

# ---- Fig: training curves fold 0
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
for k, c in ((1, C1), (5, C5)):
    h = S[k][0][1]
    axs[0].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], color=c, label=f"K={k}")
    v = [r for r in h if "val_mean_dice" in r]
    axs[1].plot([r["epoch"] for r in v], [r["val_mean_dice"] for r in v], "o-", ms=3, color=c, label=f"K={k}")
axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Training loss"); axs[0].legend(frameon=False)
axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Validation mean 3D Dice"); axs[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/training_curves.png"); plt.close(fig)

# ---- Fig: Optuna trials + importance
tr = [t for t in stats["optuna_trials"] if t["state"] == "COMPLETE"]
imp = stats["optuna"]["importance"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
for t in stats["optuna_trials"]:
    if t["state"] in ("COMPLETE",):
        axs[0].scatter(t["number"], t["value"], color={3: "#d98c1f", 5: C5, 7: "#3a9a5b"}[t["n_slices"]], s=30)
for kk, cc in ((3, "#d98c1f"), (5, C5), (7, "#3a9a5b")):
    axs[0].scatter([], [], color=cc, label=f"K={kk}")
axs[0].set_xlabel("Trial number"); axs[0].set_ylabel("Validation 3D Dice (short trial)"); axs[0].legend(frameon=False, fontsize=8)
names = list(imp)[::-1]
axs[1].barh(names, [imp[n] for n in names], color=C5); axs[1].set_xlabel("Hyper-parameter importance (fANOVA)")
fig.tight_layout(); fig.savefig(f"{OUT}/optuna.png"); plt.close(fig)

print(json.dumps({"pooled": pooled, "fold_mean_sd": stats["fold_mean_sd"], "n": len(pairs)}, indent=1, default=float))
