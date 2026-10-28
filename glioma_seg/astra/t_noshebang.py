import os, subprocess
print("Start running on Astra (no-shebang test)", flush=True)
os.system("hostname; nvidia-smi; which python3")
subprocess.run([os.path.expanduser("~/Project1/env/bin/python"),
                os.path.expanduser("~/Project1/code/glioma_seg/astra/gpu_test.py")])
print("Done running on Astra", flush=True)
