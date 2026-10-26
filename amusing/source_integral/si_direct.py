#!/usr/bin/env python3
"""Direct (no table interpolation) source-integral responses at one parameter point, for accelerator validation.

usage: si_direct.py CONFIG.json OUT.json ell eps slope ell_src eta [--n-age 48] [--mmax M] [--part se|cnp --m0 A --m1 B]
--part splits the work into resumable units (same-event+mean, or C_NP harmonics m0..m1-1); si_direct_combine.py sums the
parts into the same output as a single run.
Same-event: sensitivity-map contractions at the exact age-node variances V_s and the exact slope (si_same_event).
Mean: transported fields sum_s w_s G_{V_s} * R(1+eta g) at the exact ages, projected, exact FFT pair sums.
C_NP: polar-harmonic form with the exact age mixture and exact ell_src and slope (si_cnp_polar algebra).
All quantities are per unit amplitude and per ordered pair (numerator, denominator), as Tables.terms.
"""
import sys, json, time, copy
import numpy as np
from pathlib import Path
from scipy.special import ive, jv
sys.path.insert(0, str(Path(__file__).resolve().parent))
import si_tables as st
from si_same_event import SameEventMaps, midpoint_maps
from si_eval import age_nodes
from si_cnp_build import q_grid


def main():
    a = sys.argv; cfgd = json.load(open(a[1])); out = a[2]
    ell, eps, slope, Ls, eta = map(float, a[3:8])
    n_age = int(a[a.index("--n-age")+1]) if "--n-age" in a else 48
    part = a[a.index("--part")+1] if "--part" in a else "all"
    cfg = st.Config(cfgd); geom = cfg.geometry(); t0 = time.time()
    u, w = age_nodes(eps, n_age); u, w = u[0], w[0]; Vs = ell**2*u+cfg.sb2; f4 = 2*np.pi*ell**2*(1-eps**2)
    cfg.s_nodes = np.array([slope])
    se = np.zeros((2, 2, geom.nb)); mean_f = np.zeros((2, geom.H, geom.W))
    a0 = int(a[a.index("--a0")+1]) if "--a0" in a else 0; a1 = int(a[a.index("--a1")+1]) if "--a1" in a else n_age
    for V, wt in zip(Vs[a0:a1], w[a0:a1]) if part in ("all", "se") else ():
        Y = midpoint_maps(geom, [V])[:, 0]
        fine, coarse = cfg.grids(geom, V)
        for k, (grid, gp) in enumerate(((coarse, False), (fine, True))):
            Hs = st.rates(cfg, grid)
            if gp:
                Hs = Hs*cfg.g_on(grid)[None]
            se[k] += f4*wt*SameEventMaps(geom, grid, V, Y, Hs).contract()[0]
            mean_f[k] += wt*st.smooth_to_cells(geom, grid, V, Hs)[0]
    t1 = time.time()
    m = geom.project(mean_f[0]+eta*mean_f[1]); mn, md = geom.pair_sums(m)
    # C_NP (exact age mixture; per-age adaptive radial source grids as in the table builder)
    from si_cnp_build import CNPSetup
    cfg.V_nodes = Vs; cfgd2 = dict(cfgd); cfgd2.update(L_lo=Ls, L_hi=Ls, n_L=2); cfg.cfg = cfgd2
    S = CNPSetup(cfg, geom)
    Eh = 2*np.pi*Ls**2*np.exp(-0.5*Ls**2*S.q**2)*S.q*S.qw
    mmax = int(a[a.index("--mmax")+1]) if "--mmax" in a else cfgd["m_max"]
    m0 = int(a[a.index("--m0")+1]) if "--m0" in a else 0; m1 = int(a[a.index("--m1")+1]) if "--m1" in a else mmax+1
    cn = np.zeros((2, geom.nb))
    for mm in range(m0, min(m1, mmax+1)) if part in ("all", "cnp") else ():
        kern = [S.kernels(mm, V) for V in Vs]
        amix = np.einsum("s,srq->rq", w, S.a_slope(kern, slope))
        Mm = 2*np.pi*(amix*Eh[None]) @ amix.T
        Hh, Dh = S.pw.harmonics(mm)
        c = 1.0 if mm == 0 else 2.0
        cn[0] += c*np.einsum("ij,bij->b", Mm, Hh); cn[1] += c*np.einsum("ij,bij->b", Mm, Dh)
    pw = S.pw; q = S.q; rho = np.zeros(0)
    cnt = geom.counts
    res = dict(params=dict(ell=ell, eps=eps, slope=slope, ell_src=Ls, eta=eta), n_age=n_age, mmax=mmax, part=part, m0=m0, m1=m1,
               a0=a0, a1=a1, raw=dict(se=se.tolist(), mean_field=mean_f.tolist() if part == "se" else None, mean_num=np.asarray(mn).tolist(), mean_den=np.asarray(md).tolist(), cnp=cn.tolist(), counts=cnt.tolist()),
               p0=(se[0]/cnt).tolist(), p1=(se[1]/cnt).tolist(), mean=(np.array([mn, md])/cnt).tolist(), cnp=(cn/cnt).tolist(),
               seconds_se_mean=t1-t0, seconds_total=time.time()-t0, rho_points=len(rho), nq=len(q), nr=pw.nr)
    tmp = out+f".{__import__('os').getpid()}.partial"; json.dump(res, open(tmp, "w")); __import__('os').replace(tmp, out)
    print(f"wrote {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
