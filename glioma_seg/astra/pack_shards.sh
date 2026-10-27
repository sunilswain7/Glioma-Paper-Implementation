#!/bin/bash
# LAPTOP side: build v2 data from the local HDF5 copy and pack it into <5 GB tar shards
# for browser upload to Astra (Files -> /home/<you>/Project1/data/).
#   bash astra/pack_shards.sh /run/media/<user>/<HDD>
set -e
DST=${1:?usage: pack_shards.sh <output dir on external drive>}
H5="/home/reaper/PaperCancer/Brats datset/BraTS2020_training_data/content/data"
cd "$(dirname "$0")/.."
python3 -m cst2.preprocess --h5_dir "$H5" --out "$DST/brats_npy" --workers 8
cd "$DST"
ls brats_npy | grep _img.npy | sed 's/_img.npy//' | sort > pids.txt
split -n l/4 -d pids.txt part_
for f in part_0*; do
  n=${f#part_0}
  sed 's/$/_img.npy/' $f > list; sed 's/$/_seg.npy/' $f >> list; echo meta.json >> list
  tar -cf brats_npy_part$n.tar -C "$DST" $(sed 's|^|brats_npy/|' list)
done
rm -f list part_0* pids.txt
ls -lh "$DST"/*.tar
