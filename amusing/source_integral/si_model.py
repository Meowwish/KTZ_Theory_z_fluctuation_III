"""Production likelihood with the source-position-dependent (legal, PSD) KTZ covariance integrals (2026-10-26).

Replaces ONLY the predicted stochastic/mean responses of the legacy single-pointing model (mac_model.build):
data vector, covariance (incl. the sub-beam exclusion), bins, mask, priors (split ell_* piece, manuscript eps
prior transform once, flat eta, Gaussian slope, ell_src bounds), the m=1 dipole nuisance and the noise layer
(noise_signal_fraction x ratio + noise_additive_bias, applied once) are reproduced verbatim from mac_model.py.

Components per bin b (raw ordered-pair sums / ordered-pair counts, numerator and endpoint denominator):
  0  C_2p,0  : 4 pi d sum_s w_s int R(y) S_{b,V_s}(y) dy                      (same event, rate R)
  1  C_2p,N  : eta * 4 pi d sum_s w_s int R(y) g(y) S_{b,V_s}(y) dy          (same event, modulation g)
  2  C_M     : beta * pairsum[(P T (R + eta R g))^2]                         (transported mean, one outer product)
  3  C_NP    : betaA * sum_{s,s'} w_s w_s' [G_Vs D_R E D_R G_Vs'] projected  (distinct events, Gaussian excess)
  4  dipole  : q^2 * dipole response (unchanged)
Source completion C1 (user decision): R continued over the whole plane, g = 0 outside the measured template.
"""
from __future__ import annotations
import json, sys, os, glob
from pathlib import Path
import numpy as np
from scipy.special import expit
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]/"ktzobs")); sys.path.insert(0, str(HERE.parents[0]/"scripts"))
from scripts import ktz_exact_window as kew
from si_eval import V_weights, bary, age_nodes

SLOPE_BOUND, TARGET_FWHM_KPC = 0.9, 0.12
ELL0, EPS_GRID, SRC0 = np.geomspace(0.08, 3.2, 11), np.geomspace(0.015, 0.9, 10), np.geomspace(0.005, 2.5, 17)


def local_lagrange(nodes, x, p):
    """indices (n, p) of the p nodes nearest to x (contiguous window) and Lagrange weights (n, p)."""
    nodes = np.asarray(nodes); x = np.atleast_1d(x); n = len(nodes)
    i0 = np.clip(np.searchsorted(nodes, x)-p//2, 0, n-p)
    idx = i0[:, None]+np.arange(p)[None]
    xn = nodes[idx]; w = np.ones((len(x), p))
    for a in range(p):
        for b in range(p):
            if a != b:
                w[:, a] *= (x-xn[:, b])/(xn[:, a]-xn[:, b])
    return i0, w


class Tables:
    """Source-integral tables of one variant (raw pair sums; divided by ordered-pair counts at evaluation).
    Slope and ell_src: local p-point Lagrange interpolation on the Chebyshev-Lobatto nodes (window of the nearest
    nodes); ln V: global barycentric interpolation through the age weights.  Accuracy against the global
    barycentric interpolant is checked in check_tables_vs_raw.py."""
    def __init__(self, path, p_s=12, p_L=12, fast=None, p_e=8):
        z = np.load(path)
        self.fast = None
        if fast is not None:
            zf = np.load(fast)
            self.fast = dict(lnl=zf["lnell_nodes"], e2=zf["eps2_nodes"], Yc=zf["Y_cnp"], Ym=zf["Y_mean"], sb2=float(zf["sb2"]))
            self.p_e = p_e
        self.s_nodes, self.lnV, self.L_nodes = z["s_nodes"], z["lnV_nodes"], z["L_nodes"]
        self.lnL = np.log(self.L_nodes)
        self.counts = z["counts"]; self.nb = len(self.counts)
        self.se = np.ascontiguousarray(z["se"])           # (ns, 2 piece, nV, 2, nb)
        self._path, self._mean, self._cnp = path, None, None   # mean/cnp (ns, 2nV, 2nV, 2nb)/(ns, nL, npair, 2nb): loaded
        if fast is None:                                         # on first use (not needed on the fast path; memory)
            self._mean, self._cnp = np.ascontiguousarray(z["mean"]), np.ascontiguousarray(z["cnp"])
        self.nV = self.se.shape[2]
        self.iu = np.triu_indices(self.nV)
        self.pair_w = np.where(self.iu[0] == self.iu[1], 1.0, 2.0)
        self.s_lo, self.s_hi = float(self.s_nodes.min()), float(self.s_nodes.max())
        self.L_lo, self.L_hi = float(self.L_nodes.min()), float(self.L_nodes.max())
        self.p_s, self.p_L = min(p_s, len(self.s_nodes)), min(p_L, len(self.L_nodes))
        self.meta = json.loads(str(z["meta"]))

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.ascontiguousarray(np.load(self._path)["mean"])
        return self._mean

    @property
    def cnp(self):
        if self._cnp is None:
            self._cnp = np.ascontiguousarray(np.load(self._path)["cnp"])
        return self._cnp

    def terms(self, ell, eps, slope, Ls, eta, sb2, n_age=48, need=("p", "mean", "cnp")):
        """per-sample normalised responses (n, 2 [num, den], nb): p0, p1 (per unit eta), mean (per unit beta, with this
        eta) and C_NP (per unit betaA).  NaN outside the tabulated slope (all terms) / ell_src (C_NP) range."""
        ell, eps, slope, Ls, eta = (np.atleast_1d(np.asarray(a, float)) for a in (ell, eps, slope, Ls, eta))
        n = len(ell); nV, nb = self.nV, self.nb
        alpha, f = V_weights(self.lnV, ell, eps, sb2, n_age)
        i0, ws = local_lagrange(self.s_nodes, np.clip(slope, self.s_lo, self.s_hi), self.p_s)
        out = {}
        if "p" in need:
            out["p0"] = np.empty((n, 2*nb)); out["p1"] = np.empty((n, 2*nb))
        if "mean" in need:
            out["mean"] = np.empty((n, 2*nb))
        if "cnp" in need:
            out["cnp"] = np.empty((n, 2*nb))
            l0, wl = local_lagrange(self.lnL, np.log(np.clip(Ls, self.L_lo, self.L_hi)), self.p_L)
            aa = alpha[:, self.iu[0]]*alpha[:, self.iu[1]]*self.pair_w[None]
        fast = self.fast is not None and not getattr(self, "force_exact", False)
        if fast:
            F = self.fast
            e0, we = local_lagrange(F["lnl"], np.log(ell), self.p_e)
            lo, hi = np.log(ell**2*0.015**2+F["sb2"]), np.log(ell**2*0.9**2+F["sb2"])
            k0, wk = local_lagrange(F["e2"], (np.log(ell**2*eps**2+F["sb2"])-lo)/(hi-lo), self.p_e)
        for k in range(n):
            sl = slice(i0[k], i0[k]+self.p_s)
            if fast and ("mean" in need or "cnp" in need):
                se_, le_ = slice(e0[k], e0[k]+self.p_e), slice(k0[k], k0[k]+self.p_e)
                if "mean" in need:
                    Ym = F["Ym"][sl, se_, le_].astype(float)                                   # (p_s, p_e, p_e, 3, 2nb)
                    ym = np.tensordot(ws[k], np.tensordot(we[k], np.tensordot(wk[k], Ym, axes=(0, 2)), axes=(0, 1)), axes=(0, 0))
                    out["mean"][k] = ym[0]+eta[k]*ym[1]+eta[k]**2*ym[2]
                if "cnp" in need:
                    Yc = F["Yc"][sl, l0[k]:l0[k]+self.p_L, se_, le_].astype(float)             # (p_s, p_L, p_e, p_e, 2nb)
                    out["cnp"][k] = np.tensordot(ws[k], np.tensordot(wl[k], np.tensordot(we[k], np.tensordot(wk[k], Yc, axes=(0, 3)), axes=(0, 2)), axes=(0, 1)), axes=(0, 0))
            if "p" in need:
                se = self.se[sl].reshape(self.p_s, 2, nV, 2*nb)
                p = f[k]*np.tensordot(alpha[k], np.tensordot(ws[k], se, axes=(0, 0)), axes=(0, 1))   # (2 pieces, 2nb)
                out["p0"][k], out["p1"][k] = p[0], p[1]
            if "mean" in need and not fast:
                c = np.concatenate([alpha[k], eta[k]*alpha[k]])
                Mk = np.tensordot(ws[k], self.mean[sl], axes=(0, 0))                          # (2nV, 2nV, 2nb)
                out["mean"][k] = np.tensordot(c, np.tensordot(c, Mk, axes=(0, 0)), axes=(0, 0))
            if "cnp" in need and not fast:
                slab = self.cnp[sl, l0[k]:l0[k]+self.p_L]                                      # (p_s, p_L, npair, 2nb)
                v = np.tensordot(slab, aa[k], axes=(2, 0))                                     # (p_s, p_L, 2nb)
                out["cnp"][k] = np.tensordot(ws[k], np.tensordot(wl[k], v, axes=(0, 1)), axes=(0, 0))
        bad_s = (slope < self.s_lo-1e-12) | (slope > self.s_hi+1e-12)
        bad_L = (Ls < self.L_lo-1e-12) | (Ls > self.L_hi+1e-12)
        for key in out:
            out[key] = out[key].reshape(n, 2, nb)/self.counts
            out[key][bad_s] = np.nan
        if "cnp" in out:
            out["cnp"][bad_L] = np.nan
        return out


def build(cfg, tables_path, log=print, fast_path=None):
    """Same model as production mac_model.build (lean), with the new responses installed."""
    out = Path(cfg["dir"]); g = cfg["g"]
    pm = dict(np.load(out/f"{g}_pixel_map.npz")); morph = json.loads((out/f"{g}_morphology.json").read_text())
    diag = np.genfromtxt(out/f"{g}_diagnostics.csv", delimiter=",", names=True); fit_cov = np.load(out/f"{g}_fit_covariance.npz")["covariance"].copy()
    excl = (diag["r_kpc"] < cfg["rmin_fit"]) | (diag["r_kpc"] > cfg.get("rmax_fit", 1e9))
    if excl.any():
        fit_cov[excl, :] = 0.0; fit_cov[:, excl] = 0.0; fit_cov[np.ix_(excl, excl)] = np.diag(np.full(int(excl.sum()), 1e4))
    nb = len(diag["r_kpc"])
    T = Tables(tables_path, fast=fast_path); assert T.nb == nb
    split = dict(D=np.zeros(10), cbin=np.zeros(nb, int), pairs_same=np.ones(nb), pairs_cross=np.ones(nb), coarse_edges=np.arange(0, 3.01, 0.3))
    lo, hi, smax = cfg["ell_min"], cfg["ell_max"], cfg["src_max"]
    ELL = ELL0 if (hi == 3.2 and lo == 0.08) else np.geomspace(lo, hi, max(7, int(round(11*np.log(hi/lo)/np.log(40)))))
    SRC = SRC0 if smax == 2.5 else np.geomspace(0.005, smax, max(17, int(round(17*np.log(smax/0.005)/np.log(500)))))
    grids = dict(ell_grid=ELL, epsilon_grid=EPS_GRID, source_length_grid=SRC)
    X_, Y_ = pm["x_grid_kpc"], pm["y_grid_kpc"]; valid_ = pm["valid_mask"]; R_ = np.hypot(X_, Y_); PHI_ = np.arctan2(Y_, X_)
    f1 = kew.detrend(np.where(valid_, np.cos(PHI_ - np.deg2rad(cfg["m1"])), 0.0), valid_, R_, width=(0.4 if float(pm["pixel_size_kpc"]) >= 0.1 else 0.2))
    dr = diag["r_kpc"][1]-diag["r_kpc"][0]; edges = np.concatenate(([diag["r_kpc"][0]-0.5*dr], diag["r_kpc"]+0.5*dr))
    ps1 = kew.PairStatistic(valid_, pixel=float(pm["pixel_size_kpc"]), edges=edges); n1, e1, c1 = ps1.sums(f1, valid_)
    assert np.array_equal(np.rint(c1), T.counts), "table window/bins differ from the likelihood window"
    kr = np.array([0.0, 0.01]); slopes = np.array([-SLOPE_BOUND, 0.0, SLOPE_BOUND])
    resp = dict(kernel_radius=kr, rate_log_slope_grid=slopes, response=np.zeros((3, 3, 2, nb, 2)), mean_response=np.zeros((3, 3, 2, nb)),
                ordered_pair_counts=c1.astype(float),
                delta_response=np.asarray([np.divide(n1, c1, out=np.zeros(nb), where=c1 > 0), np.divide(e1, c1, out=np.zeros(nb), where=c1 > 0)]))
    zero = {k: (np.zeros_like(v) if k in ("response", "mean_response", "delta_response", "ordered_pair_counts") else v) for k, v in resp.items()}
    kew.ExactWindowKTZ._kernels = lambda self, kr_: (np.zeros((len(self.ell_grid), len(self.eps_grid), len(kr_))),
                                                     np.zeros((len(self.ell_grid), len(self.eps_grid), len(self.src_grid), len(kr_))))
    model = kew.SplitKTZ(morph, diag, fit_cov, resp, resp, zero, split, fit_cov, grids, use_D=False, sigma_q_max=3.0, src_length_bounds=(0.12, None),
                         eta_prior="flat", rate_log_slope_bound=SLOPE_BOUND, beam_fwhm_kpc=morph.get("effective_fwhm_kpc", TARGET_FWHM_KPC))
    sb2 = (model.beam_fwhm_kpc/2.354820045)**2
    assert T.L_lo <= model.src_lo+1e-12 and T.L_hi >= model.src_hi-1e-12, (T.L_lo, T.L_hi, model.src_lo, model.src_hi)
    model.si_tables, model.si_sb2 = T, sb2
    delta = resp["delta_response"]

    def raw_terms(x, interp=None):
        x = np.atleast_2d(x); n = len(x)
        if interp is not None and interp is model.sub["cross"]:
            return np.zeros((n, 5, 2, nb))
        ell, eps = np.exp(x[:, 0]), expit(x[:, 1])
        beta, eta, betaA = np.exp(x[:, 2]), np.exp(x[:, 3]), np.exp(x[:, 5])
        need = ["p"] + (["mean"] if "mean" not in model.si_skip else []) + (["cnp"] if "cnp" not in model.si_skip else [])
        r = T.terms(ell, eps, x[:, 4], np.exp(x[:, 6]), eta, sb2, need=need)
        z = np.zeros((n, 2, nb))
        mean = beta[:, None, None]*r["mean"] if "mean" in r else z
        cnp = betaA[:, None, None]*r["cnp"] if "cnp" in r else z
        q = x[:, 7]**2 if x.shape[1] > 7 else np.zeros(n)
        pointing = q[:, None, None]*delta[None]
        return np.stack((r["p0"], eta[:, None, None]*r["p1"], mean, cnp, pointing), axis=1)

    model.raw_terms = raw_terms
    model.si_skip = set()      # set to {"mean"} for L1n, {"cnp"} for L1, {"mean", "cnp"} for L0 (components masked anyway)
    model.si_meta = dict(tables=str(tables_path), table_meta=T.meta, sb2=sb2)
    log(f"source-integral model: {nb} bins, slope table [{T.s_lo:.3f},{T.s_hi:.3f}], ell_src table [{T.L_lo},{T.L_hi}]")
    return model
