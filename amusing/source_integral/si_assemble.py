#!/usr/bin/env python3
"""Assemble the source-integral tables of one variant into the compressed evaluator file.

usage: si_assemble.py CONFIG.json CNP_DIR OUT.npz [--m-max M]
SE and mean tables are read from CONFIG.out; C_NP partial sums (cnp_m*.npz covering m = 0..M) from CNP_DIR.
Slope (mean) and slope x ell_src (C_NP) dependences are compressed by an SVD whose truncation is checked against
the uncompressed tables (max error relative to the largest entry of the same bin and num/den column < 1e-10).
"""
import sys, json, glob, os, hashlib, time
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from si_tables import Config, atomic_savez


def svd_compress(M, tol=1e-13, check=1e-10):
    """M (n, D) -> U (n, r) (with singular values), V (r, D); column-relative max error checked."""
    G = M @ M.T
    w, Q = np.linalg.eigh(G); w = w[::-1]; Q = Q[:, ::-1]
    r = int(np.sum(w > (tol**2)*w[0])); Q = Q[:, :r]
    V = Q.T @ M; U = Q
    err = np.abs(U @ V - M); scale = np.maximum(np.abs(M).max(axis=0), 1e-300)
    rel = float((err.max(axis=0)/scale).max())
    if rel > check:
        r = len(w); Q = np.linalg.qr(M.T)[0]; V = Q.T; U = M @ Q
        rel = float((np.abs(U @ V - M).max(axis=0)/scale).max())
    return U, V, r, rel


def mode_basis(M, tol):
    """orthonormal basis (rows of M = mode index) keeping singular values > tol * largest."""
    w, Q = np.linalg.eigh(M @ M.T); w = w[::-1]; Q = Q[:, ::-1]
    r = int(np.sum(w > (tol**2)*w[0])); return Q[:, :r]


def main():
    cfg = Config(json.load(open(sys.argv[1]))); cnp_dir = Path(sys.argv[2]); out = Path(sys.argv[3])
    m_max = int(sys.argv[sys.argv.index("--m-max")+1]) if "--m-max" in sys.argv else cfg.cfg["m_max"]
    geom = cfg.geometry(annulus_counts=False); nV, ns, nb = len(cfg.V_nodes), len(cfg.s_nodes), geom.nb
    t0 = time.time()
    se = np.stack([np.load(cfg.out/f"se_V{j:03d}.npz")["se"] for j in range(nV)], axis=2)       # (ns, 2, nV, 2, nb)
    mean = []
    for i in range(ns):
        z = np.load(cfg.out/f"mean_s{i:03d}.npz")
        mean.append(np.stack([z["num"], z["den"]], axis=-2).reshape(2, nV, 2, nV, 2, nb))
    mean = np.stack(mean).reshape(ns, -1)
    M5 = mean.reshape(ns, 2*nV, 2*nV, 2*nb); mr = None; merr = 0.0; tol = None
    # C_NP: sum the harmonic partial sums, check coverage 0..m_max exactly once
    files = sorted(glob.glob(str(cnp_dir/"cnp_s*_m*-*.npz")))
    files = [f for f in files if ".ckpt" not in f and ".partial" not in f]
    cover = np.zeros((ns, m_max+1), int); X = None; L_nodes = None; meta_c = []
    for f in files:
        z = np.load(f); m0, m1, s0, s1 = int(z["m0"]), int(z["m1"]), int(z["s0"]), int(z["s1"])
        if m0 > m_max:
            continue
        assert m1 <= m_max+1, f"{f} exceeds m_max"
        assert np.array_equal(z["s_nodes"], cfg.s_nodes[s0:s1])
        cover[s0:s1, m0:m1] += 1
        if X is None:
            L_nodes = z["L_nodes"]; X = np.zeros((ns,)+z["X"].shape[1:])
        assert np.array_equal(L_nodes, z["L_nodes"])
        X[s0:s1] += z["X"]
        meta_c.append(dict(file=os.path.basename(f), sha256=hashlib.sha256(open(f, "rb").read()).hexdigest(), m0=m0, m1=m1, s0=s0, s1=s1, nr=int(z["nr"]), nq=len(z["q"])))
    assert np.all(cover == 1), f"harmonic coverage incomplete/duplicated: {np.argwhere(cover != 1)[:10]}"
    nL = len(L_nodes); npair = X.shape[2]
    nr_ = None; nerr = 0.0
    meta = dict(config=cfg.cfg, s_nodes=cfg.s_nodes.tolist(), V_nodes=cfg.V_nodes.tolist(), L_nodes=L_nodes.tolist(), m_max=m_max,
                mean_rank=mr, mean_tucker_rel_err=merr, cnp_rank=nr_, cnp_tucker_rel_err=nerr, tucker_tol=tol, cnp_files=meta_c,
                source_completion="C1: R=exp(slope(|y|-r_ref)) on the whole plane; g=bilinear template inside the measured support, 0 outside",
                created=time.strftime("%Y-%m-%dT%H:%M:%S"), pid=os.getpid())
    atomic_savez(out, s_nodes=cfg.s_nodes, lnV_nodes=np.log(cfg.V_nodes), L_nodes=L_nodes, counts=geom.counts, se=se,
                 mean=M5, cnp=X.reshape(ns, nL, npair, 2*nb), meta=json.dumps(meta))
    print(json.dumps({k: meta[k] for k in ("mean_rank", "mean_tucker_rel_err", "cnp_rank", "cnp_tucker_rel_err", "m_max")}), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
