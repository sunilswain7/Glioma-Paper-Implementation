#!/bin/bash
# Downloads the original BraTS2020 NIfTI zip (~4 GB) into ~/Project1/data. Do NOT unzip it.
# Needs ~/.kaggle/kaggle.json (or kaggle.json uploaded to ~/Project1). Runs in background; log: ~/Project1/data/download.log
P=$HOME/Project1
if [ ! -f ~/.kaggle/kaggle.json ]; then
  [ -f $P/kaggle.json ] || { echo "Upload kaggle.json to $P first"; exit 1; }
  mkdir -p ~/.kaggle && mv $P/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
fi
source $P/env/bin/activate
cd $P/data
nohup kaggle datasets download -d awsaf49/brats20-dataset-training-validation -p $P/data > download.log 2>&1 &
echo "Download started in background. Check progress with:  tail -c 300 $P/data/download.log ; ls -lh $P/data"
