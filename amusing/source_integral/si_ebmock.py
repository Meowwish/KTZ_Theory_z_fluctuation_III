#!/usr/bin/env python3
"""EB covariance (empirical sensitivity test) rebuilt with a new L2 median curve, exactly following the production
recipe (amusing_ktz_stage.py --cov mock --mock-curve [--mock-stochastic] --mock-noise none, then assemble_eb_cov.py):
  Sigma_field = s_b s_b' Cov[300 stationary proxy fields with the curve as input correlation, scaled to the observed
                signal variance (x stochastic share for --stochastic), detrended, pair statistic];
  Sigma = (1-s) (Sigma_field + Sigma_MC) + s diag(.) + floor^2 I, with Sigma_MC (measurement) and the jackknife block
  taken unchanged from the archived EB product, shrinkage/floor from its morphology.
The stage drew the fields after its line-flux MC from rng(1084); the MC draws are not reproducible here (no cubes), so the
fields use rng(SEED) and the regression check (archived curve -> archived Sigma_field) holds to Monte Carlo precision.
usage: si_ebmock.py NULL_INPUT_DIR ARCHIVED_EB_DIR CURVE.npz OUT.npz [--stochastic] [--n 300] [--seed 1084]"""
import sys, json
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1]/"ktzobs"))
from scripts import ktz_exact_window as kew
from scripts.stage2_injection_recovery import gaussian_field
a = sys.argv; nd, ebd, cv, out = Path(a[1]), Path(a[2]), a[3], a[4]
stoch = "--stochastic" in a; n = int(a[a.index("--n")+1]) if "--n" in a else 300; seed = int(a[a.index("--seed")+1]) if "--seed" in a else 1084
g = [p for p in nd.glob("*_pixel_map.npz")][0].name.split("_pixel_map")[0]
pm = dict(np.load(nd/f"{g}_pixel_map.npz")); morph = json.loads((ebd/f"{g}_morphology.json").read_text())
diag = np.genfromtxt(nd/f"{g}_diagnostics.csv", delimiter=",", names=True); O = np.load(ebd/f"{g}_fit_covariance.npz")
valid = pm["valid_mask"].astype(bool); X, Y = pm["x_grid_kpc"], pm["y_grid_kpc"]; R = np.hypot(X, Y); pix = float(pm["pixel_size_kpc"])
width = 0.4 if pix >= 0.1 else 0.2
resid = np.asarray(pm["residual_dex"], float); z_err = np.asarray(pm["metallicity_error_dex"], float)
r = diag["r_kpc"]; dr = r[1]-r[0]; edges = np.concatenate(([r[0]-0.5*dr], r+0.5*dr)); ps = kew.PairStatistic(valid, pixel=pix, edges=edges)
C_ = np.load(cv); share = 1.0
if stoch:
    cm = C_["comp_median"]; st_ = cm[0]+cm[1]+cm[3]; tot = cm[:4].sum(0); share = float(np.clip(st_[0]/tot[0], 0.05, 1.0))
    r_c, xi_c = np.concatenate(([0.0], C_["r"])), np.concatenate(([1.0], st_/st_[0]))
else:
    r_c, xi_c = np.concatenate(([0.0], C_["r"])), np.concatenate(([1.0], C_["ktz_median"]))
corr = lambda rr: np.where(rr <= r_c[-1], np.interp(rr, r_c, xi_c), xi_c[-1]*np.exp(-(rr-r_c[-1])/1.0))
sig = np.sqrt(max(resid[valid].var()-np.nanmean(z_err[valid]**2), 1e-6))*np.sqrt(share)
rng = np.random.default_rng(seed); curves = []
for k in range(n):
    f = gaussian_field(rng, 256, pix, corr)[:X.shape[0], :X.shape[1]]*sig
    z_ = kew.detrend(np.where(valid, f, 0.0), valid, R, width=width); n_, e_, _ = ps.sums(z_, valid)
    curves.append(np.divide(n_, e_, out=np.full(len(r), np.nan), where=e_ > 0))
sf = diag["noise_signal_fraction"]
mock = np.outer(sf, sf)*np.cov(np.nan_to_num(np.array(curves)), rowvar=False, ddof=1)
s, fl = float(morph["covariance_shrinkage"]), float(morph["systematic_floor"])
raw = mock+O["measurement"]; cov = (1-s)*raw+s*np.diag(np.diag(raw))+fl**2*np.eye(len(raw))
tmp = out+".partial.npz"; np.savez_compressed(tmp, covariance=cov, raw_covariance=raw, jackknife=O["jackknife"], measurement=O["measurement"], mock=mock,
                                              curve=cv, stochastic=stoch, share=share, signal_rms=sig, n=n, seed=seed)
import os; os.replace(tmp, out)
dd = np.sqrt(np.diag(O["mock"])); rel = np.diag(mock)/np.diag(O["mock"])
print(f"{out}: share {share:.3f} signal rms {sig:.4f}; diag(mock)/diag(archived mock): median {np.median(rel):.3f} range [{rel.min():.3f},{rel.max():.3f}]; "
      f"max|cov-archived|/sqrt(diag diag) {np.max(np.abs(cov-O['covariance'])/np.sqrt(np.outer(np.diag(O['covariance']), np.diag(O['covariance'])))):.3f}")
