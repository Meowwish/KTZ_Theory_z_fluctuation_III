"""Regression test of the manuscript epsilon prior (2026-10-16).  Exit 1 on any failure.
p(eps) = [ln((1+eps_max)/(1+eps_min))]^-1 (1+eps)^-1 on [0.015, 0.9]; inverse CDF eps = (1+eps_min)((1+eps_max)/(1+eps_min))^u - 1.
Checks: CDF / inverse-CDF round trip; the density integrates to 1; quantiles of the prior transform equal the analytic
quantiles; the edges map exactly and show no pile-up (edge-bin counts consistent with the density); the log-likelihood
contains no epsilon weight (equals -chi^2/2 recomputed independently, and is independent of eps at fixed predictions);
every model draws eps through the transform once (the legacy logit-uniform draw and 2/(1+eps) factor are absent).
usage: eps_prior_test.py CASE  (uses the same model construction as support_fix_test.py)"""
import sys, json, numpy as np
from pathlib import Path
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import chisquare
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); from _setup import setup
sys.path.insert(0, str(HERE.parents[1]/"ktzobs"))
from scripts.ktz_exact_window_evidence import NestedModel, eps_from_u, u_from_eps, EPS_MIN, EPS_MAX
fail = []; rep = {}
def check(name, ok, detail=None):
    rep[name] = dict(ok=bool(ok), detail=detail); print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)
    if not ok: fail.append(name)
C = np.log((1+EPS_MAX)/(1+EPS_MIN)); pdf = lambda e: 1/((1+e)*C)
u = np.linspace(0, 1, 100001)
check("inverse CDF / CDF round trip", np.max(np.abs(u_from_eps(eps_from_u(u))-u)) < 1e-14, float(np.max(np.abs(u_from_eps(eps_from_u(u))-u))))
I, err = quad(pdf, EPS_MIN, EPS_MAX, epsabs=1e-14, epsrel=1e-14); check("density integrates to 1", abs(I-1) < 1e-12, float(I))
check("edges map exactly", abs(eps_from_u(0.0)-EPS_MIN) < 1e-15 and abs(eps_from_u(1.0)-EPS_MAX) < 1e-15, [float(eps_from_u(0.0)), float(eps_from_u(1.0))])
case = sys.argv[1] if len(sys.argv) > 1 else "S"
CASES = {"S": ("NGC2466", "scal_mockcov_fid0_v2_pixbeam", 1, 0.45, 4.0, (0.22, 0.35)), "N1084D16": ("NGC1084", "d16_mockcov_v2_pixbeam", -87, 0.12, 2.5, (0.08, 0.5))}
G_, V, M1, RMIN, SRC, (lo, hi) = CASES[case]
m = setup(G_, V + "_unif65i", M1, RMIN, SRC, lo, hi, extra=["--support", "intrinsic"])["model"]
rng = np.random.default_rng(31)
for L in ("L0", "L1", "L1n", "L2"):
    nm = NestedModel(m, L+"q", eps_prior="manuscript"); j1 = nm.active.index(1)
    U = rng.uniform(size=(200000, len(nm.active))); eps = expit(nm.fill(U)[:, 1])
    qs = np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])
    emp = np.quantile(eps, qs); ana = eps_from_u(qs)
    check(f"{L} prior-transform quantiles = analytic", np.max(np.abs(emp-ana)) < 3e-3, dict(max_abs=float(np.max(np.abs(emp-ana)))))
    edges = np.linspace(EPS_MIN, EPS_MAX, 41); h, _ = np.histogram(eps, edges); expct = len(eps)*np.diff(u_from_eps(edges))
    chi = chisquare(h, expct); check(f"{L} no edge pile-up / histogram = density", chi.pvalue > 1e-3 and abs(h[0]/expct[0]-1) < 0.05 and abs(h[-1]/expct[-1]-1) < 0.05,
                                     dict(p=float(chi.pvalue), first_bin_ratio=float(h[0]/expct[0]), last_bin_ratio=float(h[-1]/expct[-1])))
    # likelihood carries no epsilon weight
    Uu = rng.uniform(size=(300, len(nm.active))); X = nm.fill(Uu); ok = nm.bounds_ok(X)
    ll = np.array([nm.loglike(x) for x in Uu])
    pred, _ = nm.predict_joint(X[ok]); chi2 = np.sum(((pred-m.y) @ m.precision_root_joint.T)**2, axis=1); ll_ref = -0.5*chi2
    f = np.isfinite(ll[ok])
    check(f"{L} loglike = -chi2/2 (no eps weight)", np.allclose(ll[ok][f], ll_ref[f], rtol=0, atol=1e-10), dict(n=int(f.sum())))
    leg = NestedModel(m, L+"q", eps_prior="legacy"); Xl = leg.fill(Uu)
    check(f"{L} manuscript draw differs from legacy logit draw (legacy path not used)", not np.allclose(expit(Xl[:, 1]), expit(X[:, 1])))
(HERE/f"eps_prior_report_{case}.json").write_text(json.dumps(rep, indent=1))
print(case, "ALL PASS" if not fail else f"{len(fail)} FAIL"); sys.exit(1 if fail else 0)
