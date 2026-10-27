"""Write one scheduler wrapper per plan into ~/Project1/run_on_astra/ (absolute paths,
as the Astra guide requires). Run after every `git pull`:
    ~/Project1/env/bin/python ~/Project1/code/glioma_seg/astra/make_jobs.py
Then submit job_<plan>.py via Jobs -> Shell Script (gpuq, 1 GPU).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plans import PLANS  # noqa: E402

P = os.path.realpath(os.path.expanduser("~/Project1"))
PY = f"{P}/env/bin/python"
CODE = f"{P}/code/glioma_seg"
JOBS = f"{P}/run_on_astra"

TEMPLATE = '''#!/usr/bin/env python3
import subprocess
print("Start running on Astra: {name}", flush=True)
subprocess.run({cmd!r}, cwd={cwd!r}, check=True)
print("Done running on Astra: {name}", flush=True)
'''


def write(name, cmd, cwd=CODE):
    path = os.path.join(JOBS, f"job_{name}.py")
    with open(path, "w") as f:
        f.write(TEMPLATE.format(name=name, cmd=cmd, cwd=cwd))
    print("wrote", path)


os.makedirs(JOBS, exist_ok=True)
write("preprocess", [PY, "-m", "cst2.preprocess",
                     "--zip", f"{P}/data/brats20-dataset-training-validation.zip",
                     "--out", f"{P}/data/brats_npy", "--workers", "16"])
for plan in PLANS:
    write(plan, [PY, f"{CODE}/astra/run_many.py", plan, "--project", P])
