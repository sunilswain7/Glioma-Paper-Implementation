#!/usr/bin/env python3
"""Astra setup WITHOUT a terminal: submit this file as a Shell Script job (cpuq, 16 CPUs, 32 GB).

Upload it to /home/<you>/ (and kaggle.json to /home/<you>/Project1/ or /home/<you>/),
then Jobs -> Shell Script -> drag bootstrap.py into Job Script -> Submit.

Every step is skipped if already done, so re-submitting = "git pull + continue".
Output: the job's Diagnostics/.o file, and ~/Project1/bootstrap.log
"""
import os
import glob
import shutil
import subprocess
import sys

HOME = os.path.expanduser("~")
P = os.path.join(HOME, "Project1")
CODE = os.path.join(P, "code")
REPO = "https://github.com/sunilswain7/Glioma-Paper-Implementation.git"
ZIP = os.path.join(P, "data", "brats20-dataset-training-validation.zip")
NPY = os.path.join(P, "data", "brats_npy")
LOG = open(os.path.join(HOME, "Project1_bootstrap.log"), "a") if not os.path.isdir(P) \
    else open(os.path.join(P, "bootstrap.log"), "a")


def say(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n"); LOG.flush()


def sh(cmd, check=True):
    say("$ " + cmd)
    r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)
    for line in r.stdout.splitlines()[-40:]:
        say("  " + line)
    if check and r.returncode != 0:
        say("!! FAILED (exit %d): %s" % (r.returncode, cmd))
        sys.exit(1)
    return r.returncode


say("===== 0. system / internet =====")
sh("hostname; nproc; free -g | head -2; df -h %s | tail -1; which python3; python3 --version" % HOME, check=False)
sh("curl -s -m 15 -o /dev/null -w 'github %{http_code}\\n' https://github.com || echo 'github NO INTERNET'", check=False)
sh("curl -s -m 15 -o /dev/null -w 'pypi   %{http_code}\\n' https://pypi.org || echo 'pypi NO INTERNET'", check=False)
sh("curl -s -m 15 -o /dev/null -w 'kaggle %{http_code}\\n' https://www.kaggle.com || echo 'kaggle NO INTERNET'", check=False)

say("===== 1. code =====")
os.makedirs(os.path.join(P, "data"), exist_ok=True)
if os.path.isdir(os.path.join(CODE, ".git")):
    sh("cd %s && git pull" % CODE)
else:
    sh("git clone %s %s" % (REPO, CODE))

say("===== 2. python environment =====")
sh("bash %s/glioma_seg/astra/setup_astra.sh" % CODE)
PY = os.path.join(P, "env", "bin", "python")

say("===== 3. dataset download =====")
for cand in (os.path.join(P, "kaggle.json"), os.path.join(HOME, "kaggle.json")):
    if os.path.exists(cand) and not os.path.exists(os.path.join(HOME, ".kaggle", "kaggle.json")):
        os.makedirs(os.path.join(HOME, ".kaggle"), exist_ok=True)
        shutil.copy(cand, os.path.join(HOME, ".kaggle", "kaggle.json"))
        os.chmod(os.path.join(HOME, ".kaggle", "kaggle.json"), 0o600)
shards = sorted(glob.glob(os.path.join(P, "data", "brats_npy_part*.tar")))
if shards and not os.path.exists(os.path.join(NPY, "meta.json")):
    say("found %d uploaded data shards -> extracting" % len(shards))
    for t in shards:
        sh("tar -xf %s -C %s/data" % (t, P))
    sh("ls %s | wc -l; du -sh %s" % (NPY, NPY), check=False)
if os.path.exists(os.path.join(NPY, "meta.json")):
    say("preprocessed data already present -> skip download")
elif os.path.exists(ZIP) and os.path.getsize(ZIP) > 3e9:
    say("zip already present (%.1f GB)" % (os.path.getsize(ZIP) / 1e9))
elif not os.path.exists(os.path.join(HOME, ".kaggle", "kaggle.json")):
    say("!! kaggle.json not found: upload it to %s and resubmit" % P); sys.exit(1)
else:
    sh("%s/env/bin/kaggle datasets download -d awsaf49/brats20-dataset-training-validation -p %s/data"
       % (P, P))

say("===== 4. preprocessing =====")
if os.path.exists(os.path.join(NPY, "meta.json")):
    say("already done")
else:
    zips = [ZIP] if os.path.exists(ZIP) else glob.glob(os.path.join(P, "data", "*.zip"))
    if not zips:
        say("!! no zip found in %s/data" % P); sys.exit(1)
    ncpu = os.environ.get("NCPUS", str(os.cpu_count()))
    sh("cd %s/glioma_seg && %s -m cst2.preprocess --zip %s --out %s --workers %s"
       % (CODE, PY, zips[0], NPY, ncpu))
    sh("du -sh %s" % NPY, check=False)

say("===== 5. job wrappers =====")
sh("%s %s/glioma_seg/astra/make_jobs.py" % (PY, CODE))
say("BOOTSTRAP OK -> next: submit run_on_astra/job_smoke.py (gpuq, 1 GPU)")
