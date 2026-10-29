#!/bin/bash
# Pack everything worth keeping into two files in ~/Project1/ for download (Files -> right-click -> Download):
#   results_<date>.tar.gz      small: tables, per-patient metrics, histories, logs, Optuna files, split lists
#   checkpoints_<date>.tar     best.pth of every run (model weights, ~30 MB each)
set -e
R=$HOME/Project1/runs
D=$(date +%m%d_%H%M)
~/Project1/env/bin/python $HOME/Project1/code/glioma_seg/astra/collect.py $R > $R/results_table.txt || true
cd $R
tar -czf $HOME/Project1/results_$D.tar.gz results.csv results_table.txt optuna_best.json optuna_trials.json \
    optuna_journal.log $(ls -d */ | grep -v optuna_trials | sed 's|/$||' | while read d; do
      for f in summary.json test_per_patient.json history.json log.txt split.json stdout.log; do
        [ -f "$d/$f" ] && echo "$d/$f"; done; done) 2>/dev/null
tar -cf $HOME/Project1/checkpoints_$D.tar $(ls */best.pth 2>/dev/null | grep -v optuna_trials)
ls -lh $HOME/Project1/results_$D.tar.gz $HOME/Project1/checkpoints_$D.tar
