"""Experiment plans. One plan = one Astra GPU job (1 A100); its runs execute
`concurrency` at a time on that GPU. Edit here, `git pull` on Astra, re-run
make_jobs.py, submit job_<plan>.py. Training runs resume from last.pth and skip
themselves once summary.json exists; the Optuna study resumes from its journal.

Run entry = (name, flags) for cst2.train, or (name, flags, "cst2.tune").
{RUNS} in flags is replaced by the project's runs directory.

Budgets are sized for the single lab day (28 Oct): every run stops at its
--deadline_h and still evaluates on the test patients.
"""
COMMON = "--epochs 30 --iters_per_epoch 400 --batch_size 32 --deadline_h 1.5"
FINAL = "--epochs 30 --iters_per_epoch 400 --batch_size 32 --deadline_h 1.25"
BEST = "--params_json {RUNS}/optuna_best.json"


def safe(fold, deadline_h=3.0):
    flags = f"--fold {fold} --epochs 30 --iters_per_epoch 400 --batch_size 32 --deadline_h {deadline_h}"
    return [(f"safe_k5_f{fold}", f"--n_slices 5 {flags}"),
            (f"safe_k1_f{fold}", f"--n_slices 1 {flags}")]


def final(fold):
    """Proposed model (Optuna's best settings, incl. its K) vs 2D baseline K=1 with
    the SAME settings -- the only difference is the cross-slice context."""
    return [(f"final_kx_f{fold}", f"{BEST} --fold {fold} {FINAL}"),
            (f"final_k1_f{fold}", f"{BEST} --n_slices 1 --fold {fold} {FINAL}")]


LONG = "--epochs 150 --iters_per_epoch 400 --batch_size 32 --val_every 4 --patience 10"


def long(fold, deadline_h=15):
    flags = f"{LONG} --deadline_h {deadline_h}"
    return [(f"long_kx_f{fold}", f"{BEST} --fold {fold} {flags}"),
            (f"long_k1_f{fold}", f"{BEST} --n_slices 1 --fold {fold} {flags}")]


PLANS = {
    # ~10 min: measures speed + GPU memory, checks the whole pipeline incl. test eval
    "smoke": dict(concurrency=2, runs=[
        ("smoke_k1", "--n_slices 1 --epochs 2 --iters_per_epoch 100 --val_every 1"),
        ("smoke_k5", "--n_slices 5 --epochs 2 --iters_per_epoch 100 --val_every 1"),
    ]),
    # Optuna search: 3 workers share one study on one GPU (stops after timeout_h)
    "tune": dict(concurrency=2, runs=[
        (f"tune_w{w}", f"--worker {w} --timeout_h 1.0 --epochs 8 --iters 200", "cst2.tune")
        for w in range(2)
    ]),
    # safety net with the v1-sweep defaults (K=5 vs K=1), in case tuning disappoints
    "safety": dict(concurrency=2, runs=[
        ("safe_k5_f0", f"--n_slices 5 --fold 0 {COMMON}"),
        ("safe_k1_f0", f"--n_slices 1 --fold 0 {COMMON}"),
    ]),
    # same protocol as "safety" (fold 0) on folds 1-4 -> clean 5-fold CV with ONE fixed
    # protocol (v1-sweep settings, 30 epochs, cosine fully annealed). collect.py pools all
    # safe_* runs. Packed for 3 GPUs under 40 GB each (K=5 peaks ~16 GB, K=1 ~5 GB).
    "safety_A": dict(concurrency=3, runs=[safe(1)[0], safe(1)[1], safe(4)[1]]),
    "safety_B": dict(concurrency=3, runs=[safe(2)[0], safe(2)[1], safe(3)[1]]),
    "safety_C": dict(concurrency=2, runs=[safe(3)[0], safe(4)[0]]),
    # final comparison with tuned settings: one GPU job per fold
    "final_f0": dict(concurrency=2, runs=final(0)),
    "final_f1": dict(concurrency=2, runs=final(1)),
    "final_f2": dict(concurrency=2, runs=final(2)),
    # OVERNIGHT, max 3 GPU jobs allowed: each job = 2 folds x (tuned K, K=1) = 4 runs on
    # one A100 (~33 GB). 150 epochs, early stop after 10 checks (40 epochs) w/o improvement,
    # hard stop + test evaluation after 15 h.
    "night_A": dict(concurrency=4, runs=long(0) + long(1)),
    "night_B": dict(concurrency=4, runs=long(2) + long(3)),
    # third GPU: fold 4 -> full 5-fold CV finishes overnight
    "night_C": dict(concurrency=2, runs=long(4)),
    # optional: data-size ablation on fold 0 (same val/test patients for every size)
    "datasize": dict(concurrency=4, runs=[
        (f"ds{n}_k{k}_f0", f"{BEST} --n_slices {k} --fold 0 --limit_train {n} {COMMON}"
         if k == 1 else f"{BEST} --fold 0 --limit_train {n} {COMMON}")
        for n in (16, 64) for k in (1, "x")
    ]),
}
