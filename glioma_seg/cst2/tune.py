"""Automatic hyper-parameter search with Optuna (replaces manual trial-and-error).

Searches the settings we previously picked by hand -- learning rate, weight decay,
Dice/BCE balance, tumour-slice oversampling, cross-slice depth K, transformer
layers and network width -- with a TPE sampler and median pruning (hopeless trials
are stopped early). Objective = best VALIDATION 3D mean Dice of fold 0; the test
patients are never used here.

Several workers can run at once (same GPU or several jobs); they share one
journal file, so the study simply continues when a job is resubmitted.

  python -m cst2.tune --data_dir DATA --out_dir RUNS --worker 0 --timeout_h 3
  python -m cst2.tune --out_dir RUNS --report          # writes RUNS/optuna_best.json
"""
import os
import json
import argparse
import optuna
import torch

from . import train

STUDY = "cst_glioma_v2"


def storage(out_dir):
    path = os.path.join(out_dir, "optuna_journal.log")
    try:
        from optuna.storages.journal import JournalFileBackend
        backend = JournalFileBackend(path)
    except ImportError:                                   # optuna < 4
        backend = optuna.storages.JournalFileStorage(path)
    return optuna.storages.JournalStorage(backend)


def search_space(trial):
    return {
        "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "bce_weight": trial.suggest_float("bce_weight", 0.2, 0.8),
        "fg_weight": trial.suggest_float("fg_weight", 1.0, 10.0),
        "n_slices": trial.suggest_categorical("n_slices", [3, 5, 7]),
        "xslice_layers": trial.suggest_int("xslice_layers", 1, 3),
        "base_ch": trial.suggest_categorical("base_ch", [32, 48]),
    }


def make_objective(a):
    def objective(trial):
        p = search_space(trial)
        argv = ["--data_dir", a.data_dir, "--out_dir", os.path.join(a.out_dir, "optuna_trials"),
                "--exp", f"trial_{trial.number:03d}", "--fold", "0", "--tune",
                "--epochs", str(a.epochs), "--iters_per_epoch", str(a.iters),
                "--batch_size", str(a.batch_size), "--warmup_epochs", "1",
                "--val_every", "2", "--patience", "100", "--workers", str(a.workers)]
        for k, v in p.items():
            argv += [f"--{k}", str(v)]
        try:
            return train.main(argv, trial=trial)
        except torch.cuda.OutOfMemoryError:
            # config too big for the GPU share -> skip it, keep the process healthy
            torch.cuda.empty_cache()
            raise optuna.TrialPruned("CUDA OOM")
    return objective


def report(out_dir):
    st = optuna.load_study(study_name=STUDY, storage=storage(out_dir))
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in st.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"trials: {len(done)} complete, {len(pruned)} pruned, {len(st.trials)} total")
    for t in sorted(done, key=lambda t: -t.value)[:10]:
        print(f"  #{t.number:3d} val Dice {t.value:.4f}  {t.params}")
    best = st.best_trial
    try:
        imp = {k: round(v, 3) for k, v in optuna.importance.get_param_importances(st).items()}
    except Exception:
        imp = {}
    out = {"trial": best.number, "val_dice": best.value, "params": best.params,
           "importance": imp, "n_complete": len(done), "n_pruned": len(pruned)}
    with open(os.path.join(out_dir, "optuna_best.json"), "w") as f:
        json.dump(out, f, indent=1)
    rows = [{"number": t.number, "state": t.state.name, "value": t.value, **t.params}
            for t in st.trials]
    with open(os.path.join(out_dir, "optuna_trials.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("best:", json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--timeout_h", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.report:
        report(a.out_dir); return
    study = optuna.create_study(
        study_name=STUDY, storage=storage(a.out_dir), direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=a.worker, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4))
    if a.worker == 0 and not study.trials:
        # start from the configuration validated by the v1 design sweep
        study.enqueue_trial({"lr": 2e-4, "weight_decay": 1e-5, "bce_weight": 0.5, "fg_weight": 5.0,
                             "n_slices": 5, "xslice_layers": 2, "base_ch": 48})
    study.optimize(make_objective(a), timeout=a.timeout_h * 3600, gc_after_trial=True,
                   catch=(RuntimeError,))          # e.g. CUDA OOM on a huge config -> next trial
    report(a.out_dir)


if __name__ == "__main__":
    main()
