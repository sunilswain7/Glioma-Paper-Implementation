"""Run one plan from plans.py on the GPU this job was given.

    python run_many.py PLAN --project ~/Project1
Each run's console output -> <project>/runs/<exp>/stdout.log. At the end the
results table is rebuilt (collect.py).
"""
import os
import sys
import time
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plans import PLANS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--project", default=os.path.expanduser("~/Project1"))
    ap.add_argument("--time_limit_h", type=float, default=None,
                    help="stop runs cleanly after this many hours (resubmit to resume)")
    args = ap.parse_args()
    plan = PLANS[args.plan]
    P = args.project
    py = os.path.join(P, "env", "bin", "python")
    code = os.path.join(P, "code", "glioma_seg")
    data, out = os.path.join(P, "data", "brats_npy"), os.path.join(P, "runs")
    ncpu = int(os.environ.get("NCPUS", len(os.sched_getaffinity(0))))
    conc = plan["concurrency"]
    workers = max(2, ncpu // conc - 1)
    print(f"plan {args.plan}: {len(plan['runs'])} runs, {conc} at a time, "
          f"{ncpu} CPUs -> {workers} loader workers each", flush=True)
    os.system("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        # both GPUs of a node are visible to every job: use the least-used one
        out_q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                                "--format=csv,noheader,nounits"],
                               stdout=subprocess.PIPE, universal_newlines=True).stdout
        free = min((int(m), i.strip()) for i, m in
                   (l.split(",") for l in out_q.strip().splitlines()))[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = free
    print("using GPU", os.environ["CUDA_VISIBLE_DEVICES"], flush=True)

    pending, active = list(plan["runs"]), []
    while pending or active:
        while pending and len(active) < conc:
            exp, flags, *mod = pending.pop(0)
            mod = mod[0] if mod else "cst2.train"
            os.makedirs(os.path.join(out, exp), exist_ok=True)
            cmd = [py, "-m", mod, "--data_dir", data, "--out_dir", out,
                   "--workers", str(workers)] + flags.replace("{RUNS}", out).split()
            if mod == "cst2.train":
                cmd += ["--exp", exp]
            if args.time_limit_h and mod == "cst2.train":
                cmd += ["--time_limit_h", str(args.time_limit_h)]
            logf = open(os.path.join(out, exp, "stdout.log"), "a")
            active.append((exp, subprocess.Popen(cmd, cwd=code, stdout=logf,
                                                 stderr=subprocess.STDOUT), logf, time.time()))
            print(f"[{time.strftime('%H:%M')}] started {exp}", flush=True)
        time.sleep(20)
        for item in list(active):
            exp, proc, logf, t0 = item
            if proc.poll() is not None:
                logf.close(); active.remove(item)
                print(f"[{time.strftime('%H:%M')}] finished {exp} (exit {proc.returncode}, "
                      f"{(time.time() - t0) / 60:.0f} min)", flush=True)
    if any(len(r) > 2 for r in plan["runs"]):
        subprocess.run([py, "-m", "cst2.tune", "--out_dir", out, "--report"], cwd=code)
    subprocess.run([py, os.path.join(code, "astra", "collect.py"), out])


if __name__ == "__main__":
    main()
