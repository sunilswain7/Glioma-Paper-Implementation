#!/bin/bash
# One-time Astra setup. From the Jupyter terminal:
#   bash ~/Project1/code/glioma_seg/astra/setup_astra.sh
set -e
P=$HOME/Project1
mkdir -p $P/data $P/runs $P/run_on_astra

echo "===== SYSTEM ====="
python3 --version; nproc; free -g; df -h $HOME
echo "===== INTERNET ====="
curl -s -o /dev/null -w "pypi   %{http_code}\n" https://pypi.org || echo "pypi   NO"
curl -s -o /dev/null -w "kaggle %{http_code}\n" https://www.kaggle.com || echo "kaggle NO"

echo "===== VENV + PACKAGES ====="
cd $P
[ -d env ] || python3 -m venv env
source env/bin/activate
python -m pip install -q --upgrade pip setuptools wheel
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -q -r $P/code/glioma_seg/requirements_astra.txt
python -c "import torch,nibabel,scipy,h5py; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "===== GPU TEST WRAPPER ====="
cat > $P/run_on_astra/run_gputest.py <<PYEOF
#!/usr/bin/env python3
import subprocess
print("Start running on Astra", flush=True)
subprocess.run(["$P/env/bin/python", "$P/code/glioma_seg/astra/gpu_test.py"], check=True)
print("Done running on Astra", flush=True)
PYEOF
echo "Wrote $P/run_on_astra/run_gputest.py"
echo "SETUP OK"
