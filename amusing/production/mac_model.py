"""Lean reconstruction of the paper_single_pointing.py likelihood (2026-10-16) for remote workers: identical data, mask,
bins, covariance (incl. the sub-beam exclusion), priors (split ell_* piece, eps/src/eta/slope priors), m=1 nuisance and
exact C_M library as paper_single_pointing.py --cm-nodes 65 --unified-response, with the stochastic terms from a compiled
unified evaluator.  The legacy (ell, eps, ell_src) kernel grids are not computed (they are not used: the same- and
cross-pointing interpolators are replaced); their axes are identical.  Verified against the full model in the container.
config keys: dir, g, m1, rmin_fit, rmax_fit, src_max, ell_min, ell_max, compiled"""
import json, sys, os
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO/"ktzobs"))
sys.path.insert(0, str(REPO/"amusing/scripts"))
from scripts import ktz_exact_window as kew
import ktz_unified_response as kur
SLOPE_BOUND, TARGET_FWHM_KPC = 0.9, 0.12
ELL0, EPS_GRID, SRC0 = np.geomspace(0.08, 3.2, 11), np.geomspace(0.015, 0.9, 10), np.geomspace(0.005, 2.5, 17)


def _fast_kernels(self, kr):   # shapes only: these grids are replaced by the compiled evaluator
    return np.zeros((len(self.ell_grid), len(self.eps_grid), len(kr))), np.zeros((len(self.ell_grid), len(self.eps_grid), len(self.src_grid), len(kr)))


def build(cfg, lean=True, log=print):
    if lean: kew.ExactWindowKTZ._kernels = _fast_kernels
    out = Path(cfg["dir"]); g = cfg["g"]
    pm = dict(np.load(out/f"{g}_pixel_map.npz")); morph = json.loads((out/f"{g}_morphology.json").read_text())
    diag = np.genfromtxt(out/f"{g}_diagnostics.csv", delimiter=",", names=True); fit_cov = np.load(out/f"{g}_fit_covariance.npz")["covariance"].copy()
    excl = (diag["r_kpc"] < cfg["rmin_fit"]) | (diag["r_kpc"] > cfg.get("rmax_fit", 1e9))
    if excl.any():
        fit_cov[excl, :] = 0.0; fit_cov[:, excl] = 0.0; fit_cov[np.ix_(excl, excl)] = np.diag(np.full(int(excl.sum()), 1e4))
    resp = dict(np.load(out/f"{g}_window_response_all.npz"))
    nb = len(diag["r_kpc"])
    split = dict(D=np.zeros(10), cbin=np.zeros(nb, int), pairs_same=np.ones(nb), pairs_cross=np.ones(nb), coarse_edges=np.arange(0, 3.01, 0.3))
    lo, hi, smax = cfg["ell_min"], cfg["ell_max"], cfg["src_max"]
    ELL = ELL0 if (hi == 3.2 and lo == 0.08) else np.geomspace(lo, hi, max(7, int(round(11*np.log(hi/lo)/np.log(40)))))
    SRC = SRC0 if smax == 2.5 else np.geomspace(0.005, smax, max(17, int(round(17*np.log(smax/0.005)/np.log(500)))))
    grids = dict(ell_grid=ELL, epsilon_grid=EPS_GRID, source_length_grid=SRC)
    X_, Y_ = pm["x_grid_kpc"], pm["y_grid_kpc"]; valid_ = pm["valid_mask"]; R_ = np.hypot(X_, Y_); PHI_ = np.arctan2(Y_, X_)
    f1 = kew.detrend(np.where(valid_, np.cos(PHI_ - np.deg2rad(cfg["m1"])), 0.0), valid_, R_, width=(0.4 if float(pm["pixel_size_kpc"]) >= 0.1 else 0.2))
    dr = diag["r_kpc"][1]-diag["r_kpc"][0]; edges = np.concatenate(([diag["r_kpc"][0]-0.5*dr], diag["r_kpc"]+0.5*dr))
    ps1 = kew.PairStatistic(valid_, pixel=float(pm["pixel_size_kpc"]), edges=edges); n1, e1, c1 = ps1.sums(f1, valid_)
    resp["delta_response"] = np.asarray([np.divide(n1, c1, out=np.zeros(nb), where=c1 > 0), np.divide(e1, c1, out=np.zeros(nb), where=c1 > 0)])
    aw = 0.4 if float(pm["pixel_size_kpc"]) >= 0.1 else 0.2
    chk = kew.exact_mean_field_response(pm, morph["r_ref_kpc"], resp["rate_log_slope_grid"], edges, aw)
    err = np.nanmax(np.abs(chk-np.asarray(resp["mean_response"])))/np.nanmax(np.abs(resp["mean_response"])); assert err < 1e-10, err
    resp.update(kew.mean_field_response_library(pm, morph["r_ref_kpc"], edges, aw, SLOPE_BOUND, 65))
    zero = {k: (np.zeros_like(v) if k in ("response", "mean_response", "mean_response_cheb", "delta_response", "ordered_pair_counts") else v) for k, v in resp.items()}
    model = kew.SplitKTZ(morph, diag, fit_cov, resp, resp, zero, split, fit_cov, grids, use_D=False, sigma_q_max=3.0, src_length_bounds=(0.12, None),
                         eta_prior="flat", rate_log_slope_bound=SLOPE_BOUND, beam_fwhm_kpc=morph.get("effective_fwhm_kpc", TARGET_FWHM_KPC))
    if not cfg.get("compiled"):
        return model
    U = kur.load_compiled(cfg["compiled"])
    assert abs(U.sigma_beam - model.beam_fwhm_kpc/2.354820045) < 1e-15 and U.support == "intrinsic"
    # axes agree to the last bits (exp/log of different numpy builds may differ by 1 ulp)
    assert np.allclose(U.ax["src"], model.interp_all["ninterp"].grid[3], rtol=0, atol=1e-12) and np.allclose(U.ax["slope"], model.interp_all["pinterp"].grid[2], rtol=0, atol=1e-12)
    kur.install_evaluator(model, U, log=log)
    return model
