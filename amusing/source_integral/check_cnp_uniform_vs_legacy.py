#!/usr/bin/env python3
"""Uniform-source limit of the C_NP tables (slope 0): the legacy production evaluator (compiled unified tables) is exact
for a uniform rate, so the new polar-harmonic C_NP must reproduce it on the full actual mask; also prints the same-event
check through the assembled Tables evaluator.  usage: CONFIG TABLES.npz LEGACY_COMPILED.npz"""
import sys, json
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0]/"scripts"))
from si_tables import Config
from si_model import Tables
import ktz_unified_response as kur
cfg = Config(json.load(open(sys.argv[1]))); T = Tables(sys.argv[2]); comp = kur.load_compiled(sys.argv[3])
rng = np.random.default_rng(5); worst = {"p0": 0, "cnp": 0}
pts = [(0.08, 0.015, 0.12), (0.12, 0.13, 1.0), (0.13, 0.18, 1.07), (0.5, 0.5, 0.3), (1.2, 0.02, 2.0), (3.2, 0.9, 2.5), (3.2, 0.015, 0.12)]
pts += [(float(np.exp(rng.uniform(np.log(.08), np.log(3.2)))), float(rng.uniform(.015, .9)), float(np.exp(rng.uniform(np.log(T.L_lo), np.log(T.L_hi))))) for _ in range(13)]
for ell, eps, Ls in pts:
    r = T.terms(np.array([ell]), np.array([eps]), np.array([0.0]), np.array([Ls]), np.array([0.0]), cfg.sb2, need=("p", "cnp"))
    lo = np.sqrt(ell**2+cfg.sb2); le = np.sqrt((ell*eps)**2+cfg.sb2)/lo
    oldp = comp.pinterp(np.array([[np.log(lo), np.log(le), 0.0]]))[0, 0]
    oldn = comp.ninterp(np.array([[np.log(lo), np.log(le), 0.0, np.log(Ls)]]))[0]
    sel = T.counts > 0; out = []
    for key, new, old in (("p0", r["p0"][0], oldp), ("cnp", r["cnp"][0], oldn)):
        rel = np.max(np.abs(new[:, sel]-old[:, sel]), axis=1)/np.max(np.abs(old[:, sel]), axis=1)
        worst[key] = max(worst[key], rel.max()); out.append(f"{key} num {rel[0]:.1e} den {rel[1]:.1e}")
    dxi = np.nanmax(np.abs(r["cnp"][0][0]/r["cnp"][0][1]-oldn[0]/oldn[1]))
    print(f"ell {ell:.3f} eps {eps:.3f} L {Ls:.3f}: " + "; ".join(out) + f"; max|d(num/den)| cnp {dxi:.1e}")
print("worst", worst)
