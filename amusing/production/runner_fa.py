#!/usr/bin/env python3
"""runner_fa.py (2026-10-21: copy of mac_runner_free.py with a line-buffered runner.log)
mac_runner.py BUDGET [NPROC] : advance queued nested-sampling jobs (jobs/*.json, in order of their 'priority') for at most
BUDGET seconds of wall time with NPROC concurrent processes (BLAS threads 1 each).  Each process is started under
`timeout -s KILL` so it ends before the bridge call does; dynesty checkpoints every 15 s, so a killed job resumes later
from its last checkpoint (verified bit-identical to an uninterrupted run).  Completed jobs (runs/<name>.npz) are skipped.
Writes a per-call log line (PID, job, start, end, exit) to runs/runner.log."""
import sys, os, json, time, glob, subprocess
B = float(sys.argv[1]); NP = int(sys.argv[2]) if len(sys.argv) > 2 else 4; JD = sys.argv[3] if len(sys.argv) > 3 else "jobs"; RD = sys.argv[4]; t0 = time.time()
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
jobs = []
for f in glob.glob(f"{JD}/*.json"):
    j = json.load(open(f)); j["file"] = f
    if not os.path.exists(j["out"] + ".npz"): jobs.append(j)
jobs.sort(key=lambda j: (0 if os.path.exists(j["out"] + ".ckpt") else 1, j.get("priority", 9), j["file"]))
run = []; log = open(f"{RD}/runner.log", "a", buffering=1)
while jobs or run:
    for p, j, ts in list(run):
        if p.poll() is not None:
            run.remove((p, j, ts)); log.write(f"{time.strftime('%H:%M:%S')} end {j['file']} pid {p.pid} rc {p.returncode} {time.time()-ts:.0f}s\n")
    left = B - (time.time()-t0)
    while jobs and len(run) < NP and left > 30:
        j = jobs.pop(0); name = os.path.basename(j["file"])[:-5]
        p = subprocess.Popen(["timeout", "-s", "KILL", str(int(left)), "python3", "mac_fit.py", j["file"]], env=env,
                             stdout=open(f"{RD}/{name}.log", "a"), stderr=subprocess.STDOUT)
        run.append((p, j, time.time())); log.write(f"{time.strftime('%H:%M:%S')} start {j['file']} pid {p.pid}\n")
    if left <= 30 and not run: break
    if left <= 0: break
    time.sleep(1)
log.flush()
done = len(glob.glob(f"{RD}/*.npz")); pend = sum(1 for f in glob.glob(f"{JD}/*.json") if not os.path.exists(json.load(open(f))["out"] + ".npz"))
print(f"elapsed {time.time()-t0:.0f}s; completed runs {done}; pending {pend}")
