#!/usr/bin/env python3
"""Accelerator check on the actual posteriors: for each finished run, re-evaluate ln L at posterior draws with the
exact-alpha table evaluator (no (ell, eps) grid) and compare with the stored (fast-evaluator) ln L."""
import sys, json, glob, os
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]/"ktzobs"))
import si_fit
out = {}
for jf in sys.argv[2:]:
    job = json.load(open(jf)); f = job["out"]+".npz"
    if not os.path.exists(f): continue
    d = np.load(f); w = np.exp(d["logwt"]-d["logz"][-1]); w /= w.sum()
    idx = np.random.default_rng(3).choice(len(w), 60, p=w)
    ex = si_fit.LogL(job["cfg"], job["tables"], job["model"], "manuscript", None)
    ll = np.array([ex(d["samples"][i]) for i in idx]); dl = ll-d["logl"][idx]
    out[os.path.basename(job["out"])] = dict(max_abs_dlnL=float(np.max(np.abs(dl))), mean_dlnL=float(np.mean(dl)))
    print(os.path.basename(job["out"]), f"max|dlnL| {np.max(np.abs(dl)):.2e} mean {np.mean(dl):+.2e}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
