#!/usr/bin/env python3
"""C_NP source-integral table for one variant, accumulated harmonic by harmonic (2026-10-26).

usage: si_cnp_build.py CONFIG.json m0 m1      -> cnp_m{m0}-{m1}.npz (partial sum over m in [m0, m1))
Table X[s_i, L_n, (j<=k), 2 (num, den), b] = sum_m c_m 2 pi sum_q Ehat_{L_n}(q) q w_q a_j(.,q)^T Hhat_{b,m} a_k(.,q)
(c_0 = 1, c_m = 2), a_j = a^{V_j}_m for slope node s_i; see si_cnp_polar.py for the derivation.
Normalisation: C_NP/(betaA A_ref) = sum_{j,k} alpha_j alpha_k X_jk / ordered counts, alpha the age weights on the
V nodes (si_eval.V_weights).  Raw (un-normalised) pair sums are stored.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.special import ive, jv
sys.path.insert(0, str(Path(__file__).resolve().parent))
from si_tables import Config, cl_nodes, atomic_savez
from si_cnp_polar import PolarWindow


def q_grid(qmax, period, pts_per_period=6, order=12):
    npan = max(4, int(np.ceil(qmax/period*pts_per_period/order)))
    x, w = np.polynomial.legendre.leggauss(order); e = np.linspace(0, qmax, npan+1)
    q = (0.5*(e[1:]-e[:-1])[:, None]*(x[None]+1)+e[:-1, None]).ravel()
    qw = (0.5*(e[1:]-e[:-1])[:, None]*w[None]).ravel()
    return q, qw


class CNPSetup:
    def __init__(self, cfg, geom):
        c = cfg.cfg
        self.cfg, self.geom = cfg, geom
        self.L_lo, self.L_hi = c["L_lo"], c["L_hi"]
        self.L_nodes = np.exp(cl_nodes(c["n_L"], np.log(self.L_lo)-1e-6, np.log(self.L_hi)+1e-6))
        self.pw = PolarWindow(geom, c["r_step"], order=c.get("r_order", 4))
        rmax = self.pw.nodes.max()
        Vmin = cfg.V_nodes.min()
        self.qmax = np.sqrt(c.get("q_tail", 50.0)/(0.5*self.L_lo**2+Vmin))
        self.q, self.qw = q_grid(self.qmax, 2*np.pi/(2*rmax), c.get("q_pts_per_period", 6))
        self.Ew = np.stack([2*np.pi*L*L*np.exp(-0.5*L*L*self.q**2)*self.q*self.qw for L in self.L_nodes])   # (nL, nq)
        self.smax = max(abs(cfg.s_lo), abs(cfg.s_hi))
        self.rho_div = c.get("rho_div", 4.0)

    def kernels(self, m, V):
        """(K*rho*w (nr, nrho), rho, J (nrho, nq_cut), n_q_cut) for one V and harmonic m."""
        r = self.pw.nodes; sV = np.sqrt(V)
        qcut = min(self.qmax, np.sqrt(60.0/V))
        step = min(sV/self.rho_div, np.pi/(5*qcut))
        lo = max(0.0, r.min()-9*sV); hi = r.max()+9*sV+self.smax*V
        rho = np.arange(lo, hi+step, step)
        w = np.full(len(rho), step); w[0] *= 0.5 if lo == 0 else 1.0
        K = np.exp(-(r[:, None]-rho[None])**2/(2*V))*ive(m, r[:, None]*rho[None]/V)/(2*np.pi*V)
        nqc = int(np.sum(self.q <= qcut))
        J = jv(m, self.q[None, :nqc]*rho[:, None])
        return K*(rho*w)[None], rho, J, nqc

    def a_slope(self, kern, s):
        """a^V_m(r_i, q) for every V node at one slope: (nV, nr, nq), zero beyond each V's cut-off."""
        out = np.zeros((len(kern), self.pw.nr, len(self.q)))
        for j, (KW, rho, J, nqc) in enumerate(kern):
            out[j, :, :nqc] = (KW*np.exp(s*(rho-self.cfg.r_ref))[None]) @ J
        return out


def build(cfg_path, m0, m1, budget=None, s0=None, s1=None, stop_at=None):
    cfg = Config(json.load(open(cfg_path))); geom = cfg.geometry(annulus_counts=False)
    s0 = 0 if s0 is None else s0; s1 = len(cfg.s_nodes) if s1 is None else s1
    tag = f"cnp_s{s0:02d}-{s1:02d}_m{m0:03d}-{m1:03d}"          # checkpoint identity (original lane)
    m_end = m1 if stop_at is None else stop_at                  # a lane may be truncated; output named by its true range
    out = cfg.out/f"cnp_s{s0:02d}-{s1:02d}_m{m0:03d}-{m_end:03d}.npz"
    if out.exists():
        print("exists", out); return
    S = CNPSetup(cfg, geom)
    ns, nL, nV, nb = s1-s0, len(S.L_nodes), len(cfg.V_nodes), geom.nb
    iu = np.triu_indices(nV)
    X = np.zeros((ns, nL, len(iu[0]), 2, nb))
    ckd = Path(cfg.cfg.get("ckpt_dir", str(cfg.out))); ckd.mkdir(parents=True, exist_ok=True)
    ck = ckd/f"{cfg.out.name}_{tag}.ckpt.npz"; m_start = m0
    if ck.exists():
        zc = np.load(ck); X = zc["X"]; m_start = int(zc["m_done"])+1
    tstart = time.time(); per_m = None
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')} pid {os.getpid()}] {s}", flush=True)
    log(f"nr {S.pw.nr} nq {len(S.q)} qmax {S.qmax:.1f} L {S.L_nodes.round(3).tolist()} m {m0}-{m1}")
    t0 = time.time()
    for m in range(m_start, m_end):
        tm = time.time()
        Hh, Dh = S.pw.harmonics(m)                                                   # (nb, nr, nr)
        kern = [S.kernels(m, V) for V in cfg.V_nodes]
        nqv = [k[3] for k in kern]
        nqv = np.minimum.accumulate(np.asarray(nqv))                                 # non-increasing with V
        cm = (1.0 if m == 0 else 2.0)*2*np.pi
        nr, nq = S.pw.nr, len(S.q)
        Mall = np.concatenate([Hh, Dh]).reshape(2*nb*nr, nr)                         # (2 nb r, t)
        # q segments with a fixed active set of V nodes (a^V = 0 for q beyond its cut-off)
        segs = []
        for J in range(nV, 0, -1):
            qa = int(nqv[J]) if J < nV else 0
            qb = int(nqv[J-1])
            for q0 in range(qa, qb, 24):
                segs.append((q0, min(qb, q0+24), J))
        for i in range(ns):
            Ai = S.a_slope(kern, cfg.s_nodes[s0+i])
            XF = np.zeros((nL, nV, 2*nb, nV))
            for qa, qb, J in segs:
                As = Ai[:J, :, qa:qb]                                              # (J, r, nqs)
                nqs = qb-qa
                P = (Mall @ As.transpose(1, 2, 0).reshape(nr, nqs*J)).reshape(2*nb, nr, nqs, J)
                P = np.ascontiguousarray(P.transpose(2, 1, 0, 3)).reshape(nqs, nr, 2*nb*J)
                G = np.matmul(np.ascontiguousarray(As.transpose(2, 0, 1)), P).reshape(nqs, J, 2*nb, J)
                XF[:, :J, :, :J] += np.tensordot(S.Ew[:, qa:qb], G, axes=(1, 0))
            XF = XF.reshape(nL, nV, 2, nb, nV)
            X[i] += cm*XF[:, iu[0], :, :, iu[1]].transpose(1, 0, 2, 3)               # (nL, npair, 2, nb)
        per_m = time.time()-tm if per_m is None else max(per_m, time.time()-tm)
        log(f"m={m} done ({time.time()-t0:.0f}s, {time.time()-tm:.0f}s for this m)")
        if m < m_end-1 and ((budget is not None and time.time()-tstart+per_m+8 > budget) or (m-m0) % 10 == 9):
            atomic_savez(ck, X=X, m_done=m)
            if budget is not None and time.time()-tstart+per_m+8 > budget:
                log(f"budget reached after m={m}; checkpoint written"); return
    assert m_start <= m_end, "checkpoint is beyond the truncated lane end"
    atomic_savez(out, X=X, m0=m0, m1=m_end, s0=s0, s1=s1, s_nodes=cfg.s_nodes[s0:s1], L_nodes=S.L_nodes, q=S.q, nr=S.pw.nr, r_nodes=S.pw.nodes, pid=os.getpid(), seconds=time.time()-t0)
    log(f"wrote {out}")


if __name__ == "__main__":
    bud = float(sys.argv[sys.argv.index("--budget")+1]) if "--budget" in sys.argv else None
    if bud is not None and "SLICE_LEFT" in os.environ:
        bud = min(bud, float(os.environ["SLICE_LEFT"])-22)
    ss = [int(v) for v in sys.argv[sys.argv.index("--s")+1].split(":")] if "--s" in sys.argv else (None, None)
    stop = int(sys.argv[sys.argv.index("--stop-at")+1]) if "--stop-at" in sys.argv else None
    build(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), bud, *ss, stop_at=stop)
