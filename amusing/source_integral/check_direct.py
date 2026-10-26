#!/usr/bin/env python3
"""Compare si_direct.py results (exact ages/slope/ell_src, no tables) with the table evaluators (exact-alpha and fast)."""
import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from si_tables import Config
from si_model import Tables
cfg = Config(json.load(open(sys.argv[1]))); Tx = Tables(sys.argv[2]); Tf = Tables(sys.argv[2], fast=sys.argv[3])
fsig = cfg.diag["noise_signal_fraction"]; bias = cfg.diag["noise_additive_bias"]
C = np.load(Path(cfg.cfg["dir"])/f"{cfg.galaxy}_fit_covariance.npz")["covariance"]; Li = np.linalg.inv(np.linalg.cholesky(C))
for f in sys.argv[4:]:
    d = json.load(open(f)); p = d["params"]
    rx = Tx.terms(p["ell"], p["eps"], p["slope"], p["ell_src"], p["eta"], cfg.sb2)
    rf = Tf.terms(p["ell"], p["eps"], p["slope"], p["ell_src"], p["eta"], cfg.sb2)
    print(f"== {Path(f).name}: {p}")
    for k in ("p0", "p1", "mean", "cnp"):
        D = np.array(d[k]); sc = np.abs(D).max(axis=1, keepdims=True)
        ex = (np.abs(rx[k][0]-D)/sc).max(); fa = (np.abs(rf[k][0]-D)/sc).max()
        print(f"  {k}: table(exact alpha) max rel {ex:.2e}   fast {fa:.2e}")
    for amp in ((1.0, 1.0), (0.3, 3.0)):
        beta, bA = amp
        def pred(r):
            num = r["p0"][0][0]+p["eta"]*r["p1"][0][0]+beta*r["mean"][0][0]+bA*r["cnp"][0][0]
            den = r["p0"][0][1]+p["eta"]*r["p1"][0][1]+beta*r["mean"][0][1]+bA*r["cnp"][0][1]
            return fsig*num/den+bias
        Dd = {k: np.array(d[k])[None] for k in ("p0", "p1", "mean", "cnp")}
        xd, xx, xf = pred(Dd), pred(rx), pred(rf)
        obs = cfg.diag["xi_centered"]; chi = lambda v: float(np.sum(((v-obs) @ Li.T)**2))
        print(f"  beta {beta} betaA {bA}: max|dxi| exact {np.nanmax(np.abs(xx-xd)):.2e} fast {np.nanmax(np.abs(xf-xd)):.2e};"
              f" dchi2 exact {chi(xx)-chi(xd):+.2e} fast {chi(xf)-chi(xd):+.2e} (chi2 {chi(xd):.1f})")
