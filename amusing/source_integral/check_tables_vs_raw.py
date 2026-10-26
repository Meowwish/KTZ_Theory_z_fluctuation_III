#!/usr/bin/env python3
"""Compressed (Tucker) evaluator vs uncompressed tables at random prior points; and per-call timing."""
import sys, json, glob, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from si_tables import Config
from si_model import Tables
from si_eval import V_weights, bary
cfg = Config(json.load(open(sys.argv[1]))); T = Tables(sys.argv[2]); cnp_dir = Path(sys.argv[3]); sb2 = cfg.sb2
nV, ns, nb = len(cfg.V_nodes), len(cfg.s_nodes), T.nb
X = T.cnp.reshape(ns, len(T.L_nodes), -1, 2, nb)
mean = T.mean.reshape(ns, 2*nV, 2*nV, 2, nb)
rng = np.random.default_rng(7); n = 200
mu, sd = cfg.morph["event_rate_log_slope_kpc_inv"], cfg.morph["event_rate_log_slope_error_kpc_inv"]
ell = np.exp(rng.uniform(np.log(0.08), np.log(3.2), n)); eps = rng.uniform(0.015, 0.9, n)
slope = np.clip(rng.normal(mu, 3*sd, n), T.s_lo, T.s_hi); Ls = np.exp(rng.uniform(np.log(T.L_lo), np.log(T.L_hi), n)); eta = rng.uniform(0, 1, n)
t0 = time.time(); r = T.terms(ell, eps, slope, Ls, eta, sb2); t1 = time.time()
for i in range(20): T.terms(ell[i:i+1], eps[i:i+1], slope[i:i+1], Ls[i:i+1], eta[i:i+1], sb2)
t2 = time.time()
print(f"batch {n}: {1e3*(t1-t0)/n:.2f} ms/sample; single calls {1e3*(t2-t1)/20:.2f} ms/call")
alpha, f = V_weights(T.lnV, ell, eps, sb2); ws = bary(T.s_nodes, slope); wl = bary(np.log(T.L_nodes), np.log(Ls))
iu = np.triu_indices(nV); pw = np.where(iu[0] == iu[1], 1.0, 2.0)
cnp_raw = np.einsum("ni,nl,ilpab,np->nab", ws, wl, X, alpha[:, iu[0]]*alpha[:, iu[1]]*pw[None], optimize=True)/T.counts
c = np.concatenate([alpha, eta[:, None]*alpha], axis=1)
mean_raw = np.einsum("ni,iabxy,na,nb->nxy", ws, mean, c, c, optimize=True)/T.counts
for name, a, b in (("cnp", r["cnp"], cnp_raw), ("mean", r["mean"], mean_raw)):
    rel = np.abs(a-b)/np.maximum(np.abs(b).max(axis=2, keepdims=True), 1e-300)
    ratio_a = a[:, 0]/a[:, 1]; ratio_b = b[:, 0]/b[:, 1]
    print(f"{name}: max rel (per-sample/num-den scale) {rel.max():.2e}; max |delta ratio| {np.nanmax(np.abs(ratio_a-ratio_b)):.2e}")
