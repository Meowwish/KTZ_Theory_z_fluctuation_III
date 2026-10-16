"""Regression test (2026-10-16): the unified evaluator keeps the legacy prediction domain (effective prior support).
For every published prior piece of each case, at 4000 random coordinates spanning well beyond the grids, the NaN pattern
of every term (same-pointing responses) must be identical between the legacy grid interpolators and the unified
evaluator, the cross-pointing zero responses must be bit-identical, and for every model (L0..L2 with the m=1 nuisance)
the set of unit-cube points with finite log-likelihood must be identical. Exit 1 on any mismatch.
usage: unified_domain_test.py CASE"""
import sys, json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); from _setup import setup
CASES = {"S": ("NGC2466", "scal_mockcov_fid0_v2_pixbeam", 1, 0.45, 4.0, [(0.08, 0.22), (0.22, 0.35), (0.35, 0.5), (0.5, 3.2)]),
         "D16": ("NGC2466", "d16_mockcov_fid0_v2_pixbeam", 9, 0.45, 4.0, [(0.08, 0.22), (0.22, 0.35), (0.35, 0.5), (0.5, 3.2)]),
         "N1084D16": ("NGC1084", "d16_mockcov_v2_pixbeam", -87, 0.12, 2.5, [(0.08, 0.5), (0.5, 3.2)]),
         "N1084t05": ("NGC1084", "d16_mockcov_t05_v2_pixbeam", -87, 0.12, 2.5, [(0.08, 0.5), (0.5, 3.2)]),
         "N1084S": ("NGC1084", "scal_mockcov_v2_pixbeam", -71, 0.12, 2.5, [(0.08, 0.5), (0.5, 3.2)])}
case = sys.argv[1]; G_, V, M1, RMIN, SRC, pieces = CASES[case]; rep = {}; bad = False
for lo, hi in pieces:
    G = setup(G_, V + "_unif65", M1, RMIN, SRC, lo, hi); m = G["model"]
    from scripts.ktz_exact_window_evidence import NestedModel
    rng = np.random.default_rng(17); n = 4000
    X = np.column_stack([rng.uniform(np.log(0.03), np.log(5.0), n), rng.uniform(-7, 5, n), rng.uniform(np.log(1e-3), np.log(1e3), n),
                         rng.uniform(np.log(1e-3), np.log(3), n), rng.uniform(-1.0, 1.0, n), rng.uniform(np.log(1e-5), np.log(1e5), n),
                         rng.uniform(np.log(0.002), np.log(8.0), n), rng.uniform(0, 3, n)])
    # boundary-focused points: observed (ell, eps), slope and ell_src placed within +-1e-9 (relative) of every grid edge
    pg = m.sub["same"]["pinterp_legacy"]; ng = m.sub["same"]["ninterp_legacy"]; sb = m.beam_fwhm_kpc/2.354820045
    B = []
    for k in range(3000):
        x = X[rng.integers(n)].copy(); which = rng.integers(4); off = rng.choice([-1e-9, -1e-12, 0.0, 1e-12, 1e-9])
        lo_obs = rng.uniform(pg.grid[0][0], pg.grid[0][-1]); eo_obs = rng.uniform(pg.grid[1][0], pg.grid[1][-1])
        if which == 0: lo_obs = rng.choice([pg.grid[0][0], pg.grid[0][-1]])*(1+off)
        if which == 1: eo_obs = rng.choice([pg.grid[1][0], pg.grid[1][-1]])*(1+off)
        ell2 = np.exp(2*lo_obs)-sb**2; a2 = np.exp(2*(lo_obs+eo_obs))-sb**2
        if ell2 <= 0 or a2 <= 0: continue
        x[0] = 0.5*np.log(ell2); e = np.sqrt(a2/ell2)
        if not (0 < e < 1): continue
        x[1] = np.log(e/(1-e))
        if which == 2: x[4] = rng.choice([pg.grid[2][0], pg.grid[2][-1]])*(1+off)
        if which == 3: x[6] = rng.choice([ng.grid[3][0], ng.grid[3][-1]])*(1+off)
        B.append(x)
    X = np.vstack([X, np.array(B)])
    same, cross = m.sub["same"], m.sub["cross"]
    leg_s = dict(same); leg_s["pinterp"], leg_s["ninterp"] = same["pinterp_legacy"], same["ninterp_legacy"]
    leg_c = dict(cross); leg_c["pinterp"], leg_c["ninterp"] = cross["pinterp_legacy"], cross["ninterp_legacy"]
    tu, tl = m.raw_terms(X, same), m.raw_terms(X, leg_s)
    nan_same = bool(np.array_equal(np.isnan(tu), np.isnan(tl)))
    cu, cl = m.raw_terms(X, cross), m.raw_terms(X, leg_c)
    cross_bitwise = bool(np.array_equal(cu, cl, equal_nan=True))
    ll = {}
    for L in ("L0", "L1", "L1n", "L2"):
        nm = NestedModel(m, L + "q"); U = rng.uniform(size=(1500, len(nm.active)))
        fu = np.isfinite([nm.loglike(u) for u in U])
        m.sub["same"], m.sub["cross"] = leg_s, leg_c; fl = np.isfinite([nm.loglike(u) for u in U]); m.sub["same"], m.sub["cross"] = same, cross
        ll[L] = dict(identical_support=bool(np.array_equal(fu, fl)), n_finite=int(fu.sum()))
    ok = nan_same and cross_bitwise and all(v["identical_support"] for v in ll.values())
    bad |= not ok
    rep[f"{lo:g}-{hi:g}"] = dict(nan_pattern_identical=nan_same, n_outside=int(np.isnan(tl).any(axis=(1, 2, 3)).sum()), cross_bitwise_identical=cross_bitwise, loglike_support=ll, ok=ok)
    print(case, lo, hi, rep[f"{lo:g}-{hi:g}"], flush=True)
(HERE/f"unified_domain_report_{case}.json").write_text(json.dumps(rep, indent=1))
sys.exit(1 if bad else 0)
