#!/usr/bin/env python3
"""End-to-end regression in the uniform-source limit (slope 0, eta 0): for a spatially uniform event rate the
legacy midpoint kernel is exact, so the new same-event tables must reproduce the legacy production evaluator
(compiled unified tables) p0 numerator/denominator on the full actual mask.  Tests the S-map algebra, the
annular projection, the bins, the source quadrature, the ln V Chebyshev interpolation and the age quadrature."""
import sys, json, glob
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0]/"scripts"))
from si_tables import Config
from si_eval import V_weights, bary
import ktz_unified_response as kur

cfg = Config(json.load(open(sys.argv[1]))); comp = kur.load_compiled(sys.argv[2])
nV = len(cfg.V_nodes)
SE = np.stack([np.load(cfg.out/f"se_V{j:03d}.npz")["se"] for j in range(nV)], axis=2)   # (ns, 2 pieces, nV, 2, nb)
from si_geometry import Geometry
geom = cfg.geometry(); cnt = geom.counts
rng = np.random.default_rng(3); worst = 0
pts = [(0.08, 0.015), (0.08, 0.9), (0.12, 0.05), (0.35, 0.1), (0.5, 0.5), (1.0, 0.02), (3.2, 0.015), (3.2, 0.9)] + [(float(np.exp(rng.uniform(np.log(.08), np.log(3.2)))), float(rng.uniform(.015, .9))) for _ in range(12)]
ws = bary(cfg.s_nodes, 0.0)[0]
for ell, eps in pts:
    a, f = V_weights(np.log(cfg.V_nodes), ell, eps, cfg.sb2)
    new = f[0]*np.einsum("j,i,ijab->ab", a[0], ws, SE[:, 0])/cnt[None]
    lo = np.sqrt(ell**2+cfg.sb2); le = np.sqrt((ell*eps)**2+cfg.sb2)/lo
    old = comp.pinterp(np.array([[np.log(lo), np.log(le), 0.0]]))[0, 0]           # (2, nb)
    sel = cnt > 0
    rel = np.max(np.abs(new[:, sel]-old[:, sel]), axis=1)/np.max(np.abs(old[:, sel]), axis=1)
    worst = max(worst, rel.max())
    print(f"ell {ell:.3f} eps {eps:.3f}: max rel diff num {rel[0]:.2e} den {rel[1]:.2e}")
print("worst", worst)
