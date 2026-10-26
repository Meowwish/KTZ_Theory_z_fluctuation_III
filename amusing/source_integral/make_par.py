#!/usr/bin/env python3
"""make_par.py POST_NPZ MODEL M1 OUT.json [label] : posterior median (per sampled coordinate, equal-weight resampled
posterior of the dominant piece) -> physical parameter JSON for si_generate / si_regsig.
Coordinates: ln ell, logit eps, ln beta, ln eta, slope, ln betaA, ln ell_src, q (A_dip = q)."""
import sys, json, numpy as np
from scipy.special import expit
z = np.load(sys.argv[1], allow_pickle=True); mdl = sys.argv[2]; m1 = float(sys.argv[3]); out = sys.argv[4]
X = z[f"{mdl}_samples"]; x = np.median(X, 0)
par = dict(ell=float(np.exp(x[0])), eps=float(expit(x[1])), beta=float(np.exp(x[2])), eta=float(np.exp(x[3])), slope=float(x[4]),
           betaA=float(np.exp(x[5])), ell_src=float(np.exp(x[6])), A_dip=float(x[7]) if X.shape[1] > 7 else 0.0, m1=m1, amp="auto",
           source=(sys.argv[5] if len(sys.argv) > 5 else "") + f" {mdl} coordinate-wise posterior median of {sys.argv[1].split('/')[-1]}")
json.dump(par, open(out, "w"), indent=1); print(json.dumps(par))
