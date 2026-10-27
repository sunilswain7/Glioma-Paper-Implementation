"""Experiment plans. One plan = one Astra GPU job (1 A100); its runs execute
`concurrency` at a time on that GPU. Edit here, `git pull` on Astra, re-run
make_jobs.py, submit job_<plan>.py. Training runs resume from last.pth and skip
themselves once summary.json exists; the Optuna study resumes from its journal.

Run entry = (name, flags) for cst2.train, or (name, flags, "cst2.tune").
{RUNS} in flags is replaced by the project's runs directory.
"""
COMMON = "--epochs 60 --iters_per_epoch 500 --batch_size 32"
BEST = "--params_json {RUNS}/optuna_best.json"


def final(fold):
    """Proposed model (Optuna's best settings, incl. its K) vs 2D baseline K=1 with
    the SAME settings -- the only difference is the cross-slice context."""
    return [(f"final_kx_f{fold}", f"{BEST} --fold {fold} {COMMON}"),
            (f"final_k1_f{fold}", f"{BEST} --n_slices 1 --fold {fold} {COMMON}")]


PLANS = {
    # ~10 min: measures speed + GPU memory, checks the whole pipeline incl. test eval
    "smoke": dict(concurrency=2, runs=[
        ("smoke_k1", "--n_slices 1 --epochs 2 --iters_per_epoch 100 --val_every 1"),
        ("smoke_k5", "--n_slices 5 --epochs 2 --iters_per_epoch 100 --val_every 1"),
    ]),
    # Optuna search: 3 workers share one study on one GPU (stops after timeout_h)
    "tune": dict(concurrency=3, runs=[
        (f"tune_w{w}", f"--worker {w} --timeout_h 3.0", "cst2.tune") for w in range(3)
    ]),
    # safety net with the v1-sweep defaults (K=5 vs K=1), in case tuning disappoints
    "safety": dict(concurrency=2, runs=[
        ("safe_k5_f0", f"--n_slices 5 --fold 0 {COMMON}"),
        ("safe_k1_f0", f"--n_slices 1 --fold 0 {COMMON}"),
    ]),
    # final comparison with tuned settings: one plan per pair of folds (2 GPUs -> 4 folds)
    "final_f01": dict(concurrency=4, runs=final(0) + final(1)),
    "final_f23": dict(concurrency=4, runs=final(2) + final(3)),
    "final_f4": dict(concurrency=2, runs=final(4)),
    # optional: data-size ablation on fold 0 (same val/test patients for every size)
    "datasize": dict(concurrency=4, runs=[
        (f"ds{n}_k{k}_f0", f"{BEST} --n_slices {k} --fold 0 --limit_train {n} {COMMON}"
         if k == 1 else f"{BEST} --fold 0 --limit_train {n} {COMMON}")
        for n in (16, 64) for k in (1, "x")
    ]),
}
