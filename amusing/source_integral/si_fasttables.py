#!/usr/bin/env python3
"""Fast evaluator tables (2026-10-26): contract the age weights alpha(ell, eps) into the C_NP and mean tables on a
Chebyshev grid in (ln ell_*, eps^2), so that one likelihood call reads only a local window of nodes.

usage: si_fasttables.py CONFIG.json TABLES.npz OUT.npz [--n-ell 24 --n-eps2 16]
Y_cnp[s, L, i, k, 2nb]   = sum_{j,j'} alpha_j alpha_j' X[s, L, jj', 2nb]                   at (ell_i, eps_k)
Y_mean[s, i, k, pp', 2nb] = sum_{j,j'} alpha_j alpha_j' M[s, (p,j), (p',j'), 2nb]         pp' in (00, 01+10, 11)
p0/p1 keep the exact per-sample alpha path (cheap).  Accuracy against the exact-alpha path: check in si_model
(Tables.validate_fast)."""
import sys, json, os, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from si_tables import Config, cl_nodes, atomic_savez
from si_eval import V_weights

cfg = Config(json.load(open(sys.argv[1]))); z = dict(np.load(sys.argv[2])); out = sys.argv[3]
ne = int(sys.argv[sys.argv.index("--n-ell")+1]) if "--n-ell" in sys.argv else 24
nk = int(sys.argv[sys.argv.index("--n-eps2")+1]) if "--n-eps2" in sys.argv else 16
t0 = time.time()
# coordinates: x1 = ln ell, x2 = position of ln V_lo = ln(ell^2 eps^2 + sb2) between its bounds at eps = 0.015, 0.9
lnl = cl_nodes(ne, np.log(0.08)-1e-6, np.log(3.2)+1e-6); e2 = cl_nodes(nk, -1e-6, 1+1e-6)
L, K = np.meshgrid(lnl, e2, indexing="ij"); ell_n = np.exp(L.ravel())
lo, hi = np.log(ell_n**2*0.015**2+cfg.sb2), np.log(ell_n**2*0.9**2+cfg.sb2)
eps_n = np.sqrt(np.clip((np.exp(lo+K.ravel()*(hi-lo))-cfg.sb2)/ell_n**2, 1e-12, None))
alpha, f = V_weights(z["lnV_nodes"], ell_n, eps_n, cfg.sb2)          # (ne*nk, nV)
nV = alpha.shape[1]; iu = np.triu_indices(nV); pw = np.where(iu[0] == iu[1], 1.0, 2.0)
aa = alpha[:, iu[0]]*alpha[:, iu[1]]*pw[None]                                                 # (nek, npair)
X = z["cnp"]; ns, nL = X.shape[:2]
Yc = np.einsum("slpx,np->slnx", X, aa, optimize=True).reshape(ns, nL, ne, nk, -1)
M = z["mean"].reshape(ns, 2, nV, 2, nV, -1)
Ym = np.stack([np.einsum("sjkx,nj,nk->snx", M[:, 0, :, 0], alpha, alpha, optimize=True),
               np.einsum("sjkx,nj,nk->snx", M[:, 0, :, 1]+M[:, 1, :, 0].transpose(0, 2, 1, 3), alpha, alpha, optimize=True),
               np.einsum("sjkx,nj,nk->snx", M[:, 1, :, 1], alpha, alpha, optimize=True)], axis=2).reshape(ns, ne, nk, 3, -1)
atomic_savez(out, lnell_nodes=lnl, eps2_nodes=e2, x2_coordinate=np.asarray("lnVlo_fraction"), sb2=cfg.sb2, Y_cnp=Yc.astype(np.float32), Y_mean=Ym.astype(np.float32),
             source=os.path.basename(sys.argv[2]), meta=json.dumps(dict(n_ell=ne, n_eps2=nk, created=time.strftime("%Y-%m-%dT%H:%M:%S"))))
print(f"wrote {out}: Y_cnp {Yc.shape}, Y_mean {Ym.shape}, {time.time()-t0:.0f}s")
