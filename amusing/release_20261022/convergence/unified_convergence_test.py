"""Convergence and regression test of the unified stochastic-response evaluation (2026-10-16).

Reference: independent brute-force finite-window pair sums (direct_pairs.direct_terms) with every kernel evaluated at the
exact integer-lag pair separations and the C_NP age integral to 64x64 Gauss-Legendre (no kernel-radius basis, no slope
or (ell, eps, ell_src) interpolation), computed for the points of unified_points.py (stored posterior samples of every
published-prior piece, inter-node slopes, domain-edge stress points).
Compared: the production evaluation (65 Chebyshev-Lobatto slope nodes, cubic-Hermite kernel basis, exact per-sample
kernels, C_NP 24 nodes/piece), nested node subsets 33/17/9, C_NP 8/12/16 nodes/piece, and the legacy interpolated model.
Per component (C_2p,0, C_2p,N, C_NP; C_M from the exact mean-field library): max relative error of the projected
numerator in fitted bins with |exact| >= 1e-3 max|exact|; max absolute error in the remaining (near-zero) bins in units
of max|exact|; max relative error of the denominator; and max |Delta xi_total| of the full model statistic of the
sample's own model (masked components, m=1 nuisance included).
Exit 1 if the production setting violates: every component relative error < 5e-3 and max |Delta xi| < 1e-3.
usage: unified_convergence_test.py CASE [CASE ...]
"""
import sys, glob, json, time, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); from _setup import setup
sys.path.insert(0, str(HERE.parent/"scripts")); import ktz_unified_response as kur
CASES = {"S": ("NGC2466", "scal_mockcov_fid0_v2_pixbeam", 1, 0.45, 4.0), "D16": ("NGC2466", "d16_mockcov_fid0_v2_pixbeam", 9, 0.45, 4.0),
         "N1084D16": ("NGC1084", "d16_mockcov_v2_pixbeam", -87, 0.12, 2.5), "N1084t05": ("NGC1084", "d16_mockcov_t05_v2_pixbeam", -87, 0.12, 2.5)}
TOL_REL, TOL_XI = 5e-3, 1e-3
COMP = {"C2p0": (0, "p0"), "C2pN": (1, "pN"), "CM": (2, "m"), "CNP": (3, "np")}
MODEL_COMPONENTS = {"L0": [0], "L1": [0, 1, 2], "L1n": [0, 3], "L2": [0, 1, 2, 3]}
report = {}; fail = []
for case in sys.argv[1:]:
    t0 = time.time(); G_, V, M1, RMIN, SRC = CASES[case]
    G = setup(G_, V + "_unif65", M1, RMIN, SRC); m = G["model"]; keep = ~G["excl"]
    P = dict(np.load(HERE/f"unified_points_{case}.npz")); X = P["X"].copy(); lab = [str(l) for l in P["label"]]; n = len(X)
    # component-level coordinates: unit amplitude where the sample's model lacks the component (as in the direct run)
    Xc = X.copy(); Xc[:, 2] = np.log(P["beta"]); Xc[:, 3] = np.log(P["eta"]); Xc[:, 5] = np.log(P["betaA"])
    D = {}
    for i in range(n):
        f = HERE/"direct_mac"/"direct"/f"{case}_{i:03d}.npz"; z = np.load(f); D[i] = {k: z[k] for k in ("p0", "pN", "np", "m")}
    cont = sorted(glob.glob(f"/home/claude/cmfix2/directc/direct/{case}_*.npz"))   # container duplicates: cross-machine check
    xm = max([float(np.nanmax(np.abs(np.load(f)[k] - D[int(np.load(f)["index"])][k]) / np.nanmax(np.abs(D[int(np.load(f)["index"])][k]))))
              for f in cont for k in ("p0", "pN", "np", "m")], default=None)
    same = m.sub["same"]; legacy = dict(same); legacy["pinterp"], legacy["ninterp"] = same["pinterp_legacy"], same["ninterp_legacy"]
    Lib = dict(np.load(G["out"]/f"{G['g']}_stochastic_response_H65.npz"))
    pg, ng = same["pinterp_legacy"], same["ninterp_legacy"]; sb = m.beam_fwhm_kpc/2.354820045
    def evaluator(**kw):
        U = kur.UnifiedStochasticResponse(Lib, pg.grid[0], pg.grid[1], ng.grid[3], pg.grid[2], sb, **kw)
        t = dict(same); t["pinterp"], t["ninterp"] = U.pinterp, U.ninterp; return t
    setups = {"production_65_cnp24": same, "nodes33": evaluator(node_stride=2), "nodes17": evaluator(node_stride=4), "nodes9": evaluator(node_stride=8),
              "cnp16": evaluator(cnp_n=16), "cnp12": evaluator(cnp_n=12), "cnp8": evaluator(cnp_n=8), "legacy": legacy}
    sbf, bias = m.data["noise_signal_fraction"], m.data["noise_additive_bias"]
    def xi(t, comps):
        tt = t.copy(); mask = np.ones(5, bool); mask[comps + [4]] = False; tt[mask] = 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            return sbf*tt[:, 0].sum(0)/tt[:, 1].sum(0)+bias
    res = {}
    for name, interp in setups.items():
        T = m.raw_terms(Xc, interp)                                  # (n, 5, 2, nb)
        Tq = m.raw_terms(X, interp)[:, 4]                            # m=1 nuisance of the actual sample
        per = {c: dict(rel=0.0, near_zero_abs=0.0, den_rel=0.0) for c in COMP}; dxi = {"posterior": 0.0, "internode": 0.0, "edge": 0.0}
        worst = {}
        for i in range(n):
            ex = np.zeros_like(T[i])
            for c, (k, key) in COMP.items():
                e = np.asarray(D[i][key]); ex[k] = e; a = T[i, k]
                ne, na_ = e[0][keep], a[0][keep]; mx = np.max(np.abs(ne)); big = np.abs(ne) >= 1e-3*mx
                r = float(np.max(np.abs(na_[big]-ne[big])/np.abs(ne[big]))); z = float(np.max(np.abs(na_[~big]-ne[~big]))/mx) if (~big).any() else 0.0
                dr = float(np.max(np.abs(a[1][keep]-e[1][keep])/np.abs(e[1][keep])))
                if r > per[c]["rel"]: worst[c] = lab[i]
                per[c] = dict(rel=max(per[c]["rel"], r), near_zero_abs=max(per[c]["near_zero_abs"], z), den_rel=max(per[c]["den_rel"], dr))
            L = lab[i].split(":")[1] if lab[i].startswith("posterior") else "L2"
            ex[4] = Tq[i]; ta = T[i].copy(); ta[4] = Tq[i]
            # actual amplitudes of the sample's own model (inactive components are masked)
            d = float(np.max(np.abs(xi(ta, MODEL_COMPONENTS[L])-xi(ex, MODEL_COMPONENTS[L]))[keep]))
            k = lab[i].split(":")[0]; dxi[k] = max(dxi[k], d)
        res[name] = dict(components=per, max_abs_dxi=dxi, worst_point=worst)
        print(case, name, {c: f"{v['rel']:.1e}/{v['near_zero_abs']:.1e}/{v['den_rel']:.1e}" for c, v in per.items()}, {k: f"{v:.1e}" for k, v in dxi.items()}, flush=True)
    prod = res["production_65_cnp24"]
    ok = all(v["rel"] < TOL_REL for v in prod["components"].values()) and max(prod["max_abs_dxi"].values()) < TOL_XI
    if not ok: fail.append(case)
    report[case] = dict(variant=V, n_points=n, point_kinds={k: sum(l.startswith(k) for l in lab) for k in ("posterior", "internode", "edge")},
                        results=res, production_pass=ok, cross_machine_direct_max_rel=xm, seconds=time.time()-t0)
    print(case, "PASS" if ok else "FAIL", f"(cross-machine direct-pair agreement {xm})", flush=True)
(HERE/f"unified_convergence_report_{'_'.join(sys.argv[1:])}.json").write_text(json.dumps(report, indent=1))
sys.exit(1 if fail else 0)
