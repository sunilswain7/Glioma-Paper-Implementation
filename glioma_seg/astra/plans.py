"""Experiment plans. One plan = one Astra GPU job (1 A100), runs executed `concurrency`
at a time on that GPU. Edit here, `git pull` on Astra, resubmit the same wrapper.
Every run resumes automatically from last.pth and skips itself once summary.json exists.
"""
COMMON = "--epochs 60 --iters_per_epoch 500 --batch_size 32"


def runs(prefix, ks=(1, 3, 5), folds=(0,), extra=""):
    return [(f"{prefix}_k{k}_f{f}", f"--n_slices {k} --fold {f} {COMMON} {extra}")
            for f in folds for k in ks]


PLANS = {
    # ~10 min: measures speed + GPU memory, checks the whole pipeline incl. test eval
    "smoke": dict(concurrency=2, runs=[
        ("smoke_k1", "--n_slices 1 --epochs 2 --iters_per_epoch 100 --batch_size 32 --val_every 1"),
        ("smoke_k5", "--n_slices 5 --epochs 2 --iters_per_epoch 100 --batch_size 32 --val_every 1"),
    ]),
    # headline: K=1 vs K=3 vs K=5, one plan per fold (submit as separate GPU jobs)
    "main_f0": dict(concurrency=3, runs=runs("main", folds=(0,))),
    "main_f1": dict(concurrency=3, runs=runs("main", folds=(1,))),
    "main_f2": dict(concurrency=3, runs=runs("main", folds=(2,))),
    "main_f3": dict(concurrency=3, runs=runs("main", folds=(3,))),
    "main_f4": dict(concurrency=3, runs=runs("main", folds=(4,))),
    # data-size ablation on fold 0 (same val/test patients for every size)
    "datasize": dict(concurrency=3, runs=[
        (f"ds{n}_k{k}_f0", f"--n_slices {k} --fold 0 --limit_train {n} {COMMON}")
        for n in (16, 64, 150) for k in (1, 5)
    ]),
}
