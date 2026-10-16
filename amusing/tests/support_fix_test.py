"""Regression test of the intrinsic ell_* prior-support fix (2026-10-16).  Exit 1 on any failure.

For every published prior piece of the case (log-uniform in ell_* within the piece):
 (1) the log-likelihood is finite for every model (except where the model's own validity rule applies: a non-positive
     projected endpoint variance, possible because the cell covariance uses the pair-midpoint rate; counted and reported) (L0, L1, L1n, L2, all with the m=1 nuisance) at every point allowed by the
     prior -- the piece edges, the former gap positions (where the legacy grid box made beam-convolved ell_obs fall outside
     the tabulated range), the high-eps corner and random points; points outside the prior are -inf, as before;
 (2) the mapping ell_obs = sqrt(ell_*^2 + sigma_eff^2), eps_obs ell_obs = sqrt((eps ell_*)^2 + sigma_eff^2) is exact;
 (3) at every point that was finite under the legacy support, the predictions and log-likelihoods of the legacy-support and
     intrinsic-support evaluators agree to numerical precision (|dxi| < 1e-9, |dlnL| < 1e-5; the two use 1-D kernel tables
     over different ranges): the fix only removes the artificial -inf region;
 (4) adjacent pieces join without an artificial gap: at the shared edge the two piece models give identical predictions,
     and predictions are continuous across the edge (relative change < 1e-6 for a 1e-7 relative step in ell_*);
 (5) each term enters once: beta_src -> 0 removes exactly C_M, beta_src A_src -> 0 removes exactly C_NP, and the
     prediction equals s_b * sum_c num_c / sum_c den_c + n_b recomputed from the separately returned terms (measurement
     noise only through s_b and n_b; the likelihood covariance is fixed and contains no model-dependent noise term);
 (6) the split-evidence combination uses the log-uniform prior masses of the pieces (weights ln(hi/lo)/ln(ell_max/ell_min),
     contiguous pieces covering the full interval), so a constant likelihood gives the same evidence for any split.
usage: support_fix_test.py CASE  (S, D16, N1084D16, N1084t05, N1084S)
"""
import sys, json, numpy as np
from pathlib import Path
from scipy.special import logit, expit
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); from _setup import setup
CASES = {"S": ("NGC2466", "scal_mockcov_fid0_v2_pixbeam", 1, 0.45, 4.0, [(0.08, 0.22), (0.22, 0.35), (0.35, 0.5), (0.5, 3.2)]),
         "D16": ("NGC2466", "d16_mockcov_fid0_v2_pixbeam", 9, 0.45, 4.0, [(0.08, 0.22), (0.22, 0.35), (0.35, 0.5), (0.5, 3.2)]),
         "N1084D16": ("NGC1084", "d16_mockcov_v2_pixbeam", -87, 0.12, 2.5, [(0.08, 0.5), (0.5, 3.2)]),
         "N1084t05": ("NGC1084", "d16_mockcov_t05_v2_pixbeam", -87, 0.12, 2.5, [(0.08, 0.5), (0.5, 3.2)]),
         "N1084S": ("NGC1084", "scal_mockcov_v2_pixbeam", -71, 0.12, 2.5, [(0.08, 0.5), (0.5, 3.2)])}
case = sys.argv[1]; G_, V, M1, RMIN, SRC, pieces = CASES[case]
import os; EPS = os.environ.get("EPS_PRIOR", "legacy")
fail = []; rep = {}
def check(name, ok, detail=None):
    rep[name] = dict(ok=bool(ok), detail=detail)
    if not ok: fail.append(name)
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)
models = {}
for lo, hi in pieces:
    G = setup(G_, V + "_unif65", M1, RMIN, SRC, lo, hi); m = G["model"]; models[(lo, hi)] = m
    import gc
    import ktz_unified_response as kur
    from scripts.ktz_exact_window_evidence import NestedModel
    same = m.sub["same"]; cross = m.sub["cross"]
    # legacy-support and intrinsic-support evaluators on the same library
    Lib = dict(np.load(G["out"]/f"{G['g']}_stochastic_response_H65.npz")); pg, ng = same["pinterp_legacy"], same["ninterp_legacy"]
    sb = m.beam_fwhm_kpc/2.354820045
    U_leg = kur.UnifiedStochasticResponse(Lib, pg.grid[0], pg.grid[1], ng.grid[3], pg.grid[2], sb, support="legacy")
    U_int = kur.UnifiedStochasticResponse(Lib, pg.grid[0], pg.grid[1], ng.grid[3], pg.grid[2], sb, support="intrinsic")
    del Lib
    for k in ("_L", "_lib", "_st", "resp", "zero", "UNI"): G.pop(k, None)
    gc.collect()
    def use(U):
        same["pinterp"], same["ninterp"] = U.pinterp, U.ninterp; cross["pinterp"], cross["ninterp"] = U.zero_p, U.zero_n
    key = f"{lo:g}-{hi:g}"; rng = np.random.default_rng(23)
    # (2) mapping
    X = np.zeros((200, 8)); X[:, 0] = rng.uniform(np.log(lo), np.log(hi), 200); X[:, 1] = rng.uniform(logit(0.015), logit(0.9), 200)
    co = m.kernel_coordinates(X); ell, eps = np.exp(X[:, 0]), expit(X[:, 1])
    lo_ = np.sqrt(ell**2+sb**2); ao_ = np.sqrt((eps*ell)**2+sb**2)
    check(f"{key} mapping ell_obs, eps_obs", np.allclose(np.exp(co[:, 0]), lo_, rtol=1e-13) and np.allclose(np.exp(co[:, 0]+co[:, 1]), ao_, rtol=1e-13))
    for L in ("L0", "L1", "L1n", "L2"):
        nm = NestedModel(m, L+"q", eps_prior=EPS); j0 = nm.active.index(0); j1 = nm.active.index(1)
        # points: random prior draws, ell_* at piece edges / former gaps / eps corner
        U_ = rng.uniform(size=(1200, len(nm.active)))
        sp = np.r_[lo, hi, lo*(1+1e-9), hi*(1-1e-9), np.exp(np.linspace(np.log(lo), np.log(hi), 40))]
        if G_ == "NGC2466": sp = np.r_[sp, [x for x in (0.105, 0.15, 0.2, 0.292, 0.3, 0.34, 0.461, 0.48, 0.499) if lo <= x <= hi]]
        else: sp = np.r_[sp, [x for x in (0.497, 0.499, 0.5) if lo <= x <= hi]]
        E = rng.uniform(size=(len(sp)*6, len(nm.active))); E[:, j0] = np.repeat(np.log(sp/lo)/np.log(hi/lo), 6)
        E[::3, j1] = 1.0 - 1e-9                                             # eps at its upper prior bound (high-eps corner)
        Uall = np.vstack([U_, E])
        Xf = nm.fill(Uall); inprior = nm.bounds_ok(Xf)
        use(U_int); ll_i = np.array([nm.loglike(u) for u in Uall])
        use(U_leg); ll_l = np.array([nm.loglike(u) for u in Uall])
        # (1) finite everywhere inside the prior, -inf outside
        # the only admissible -inf inside the prior is the model's own validity rule (ExactWindowKTZ._ratio): a non-positive
        # projected endpoint variance in some bin (possible because the cell covariance uses the pair-midpoint rate, which is
        # not positive definite for steep rate profiles); every other in-prior point must be finite
        use(U_int); tt = m.raw_terms(Xf, same)
        tq = tt.copy(); tq[:, nm.mask] = 0.0; den_bad = (tq[:, :, 1].sum(1) <= 0).any(axis=1)
        nonfin = inprior & ~np.isfinite(ll_i)
        check(f"{key} {L} finite on full intrinsic support", np.all(den_bad[nonfin]) and np.all(~np.isfinite(ll_i[~inprior])),
              dict(n_in_prior=int(inprior.sum()), n_finite_intrinsic=int(np.isfinite(ll_i).sum()), n_finite_legacy=int(np.isfinite(ll_l).sum()),
                   n_in_prior_nonpositive_projected_variance=int((inprior & den_bad).sum()), n_in_prior_nonfinite_other=int((nonfin & ~den_bad).sum())))
        # (3) identical where the legacy support was finite
        f = np.isfinite(ll_l)
        use(U_int); pi_, _ = nm.predict_joint(Xf[f]); use(U_leg); pl_, _ = nm.predict_joint(Xf[f])
        dll = float(np.max(np.abs(ll_i[f]-ll_l[f]))) if f.any() else 0.0; dxi = float(np.max(np.abs(pi_-pl_))) if f.any() else 0.0
        # the two supports use 1-D kernel tables over different ln L / ln D ranges, so agreement is to table precision (~1e-13)
        check(f"{key} {L} legacy-finite points unchanged", dll < 1e-5 and dxi < 1e-9, dict(n=int(f.sum()), max_abs_dlnL=dll, max_abs_dxi=dxi))
    use(U_int)
    # (5) single entry of C_M, C_NP and noise
    x = nm.fill(rng.uniform(size=(1, len(NestedModel(m, "L2q", eps_prior=EPS).active))))
    nm2 = NestedModel(m, "L2q", eps_prior=EPS); x = nm2.fill(rng.uniform(size=(50, len(nm2.active)))); x = x[nm2.bounds_ok(x)][:5]
    t = m.raw_terms(x, same); sbf, nbias = m.data["noise_signal_fraction"], m.data["noise_additive_bias"]
    pred, _ = nm2.predict_joint(x)
    manual = sbf*t[:, :, 0].sum(1)/t[:, :, 1].sum(1)+nbias
    check(f"{key} prediction = s_b*sum(num)/sum(den)+n_b (noise once)", np.allclose(pred, manual, rtol=1e-12, atol=1e-14))
    xb = x.copy(); xb[:, 2] = -np.inf; tb = m.raw_terms(xb, same)          # beta_src -> 0
    tm = t.copy(); tm[:, 2] = 0.0
    check(f"{key} beta_src->0 removes exactly C_M", np.array_equal(tb, tm) and np.all(tb[:, [0, 1, 3, 4]] == t[:, [0, 1, 3, 4]]))
    xa = x.copy(); xa[:, 5] = -np.inf; ta = m.raw_terms(xa, same)          # beta_src A_src -> 0
    check(f"{key} beta_src A_src->0 removes exactly C_NP", np.all(ta[:, 3] == 0) and np.all(ta[:, [0, 1, 2, 4]] == t[:, [0, 1, 2, 4]]))
    check(f"{key} likelihood covariance fixed (no model-dependent noise)", np.array_equal(nm2.m.precision_root_joint, m.precision_root_joint))
    # cross term identically zero (single pointing): no double counting of the same-pointing responses
    check(f"{key} cross-pointing terms zero", np.all(m.raw_terms(x, cross)[:, :4] == 0))
# (4) junctions
for (a, b), (c, d) in zip(pieces[:-1], pieces[1:]):
    ma, mb = models[(a, b)], models[(c, d)]; assert b == c
    for L in ("L0", "L2"):
        na, nb_ = NestedModel(ma, L+"q", eps_prior=EPS), NestedModel(mb, L+"q", eps_prior=EPS); rng = np.random.default_rng(5)
        X = na.fill(rng.uniform(size=(400, len(na.active)))); X = X[na.bounds_ok(X)]; X[:, 0] = np.log(b)
        pa0, _ = na.predict_joint(X); X = X[np.isfinite(pa0).all(axis=1)][:20]     # exclude the model-invalid region (see (1))
        pa, _ = na.predict_joint(X); pb, _ = nb_.predict_joint(X)
        Xm, Xp = X.copy(), X.copy(); Xm[:, 0] += np.log(1-1e-7); Xp[:, 0] += np.log(1+1e-7)
        pm_, _ = na.predict_joint(Xm); pp_, _ = nb_.predict_joint(Xp)
        cont = np.max(np.abs(pp_-pm_))/np.max(np.abs(pa))
        check(f"junction {b:g} {L}: identical predictions from both pieces, continuous", np.all(np.isfinite(pa)) and np.array_equal(pa, pb) and cont < 1e-6, dict(rel_jump=float(cont)))
# (6) evidence weights of combine_split_evidence.py
src = (HERE.parent/"scripts"/"combine_split_evidence.py").read_text()
edges = [p[0] for p in pieces]+[pieces[-1][1]]
w = [np.log(hi/lo)/np.log(pieces[-1][1]/pieces[0][0]) for lo, hi in pieces]
ok = "L = np.log(pieces[-1][1]/pieces[0][0]); f = [np.log(hi/lo)/L for lo, hi in pieces]" in src and "w = np.log(f)+lz; z = np.logaddexp.reduce(w)" in src
contig = all(pieces[i][1] == pieces[i+1][0] for i in range(len(pieces)-1))
lz = np.full(len(pieces), -3.7); comb = np.logaddexp.reduce(np.log(w)+lz)
check("split-evidence combination uses log-uniform prior masses", ok and contig and abs(sum(w)-1) < 1e-14 and abs(comb-(-3.7)) < 1e-12, dict(weights=[float(x) for x in w]))
for sub in ([p for p in pieces if p[1] <= 0.5],):
    if len(sub) > 1:
        ws = [np.log(hi/lo)/np.log(sub[-1][1]/sub[0][0]) for lo, hi in sub]
        check("bounded-fit weights (pieces to 0.5 kpc)", abs(sum(ws)-1) < 1e-14, dict(weights=[float(x) for x in ws]))
(HERE/f"support_fix_report_{case}_{EPS}.json").write_text(json.dumps(rep, indent=1))
print(case, "ALL PASS" if not fail else f"{len(fail)} FAIL")
sys.exit(1 if fail else 0)
