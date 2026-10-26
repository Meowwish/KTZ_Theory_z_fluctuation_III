#!/usr/bin/env python3
"""Template-filter comparison (A): the C_M term with the H-alpha template transported by the source-integral kernels
(sum_s w_s G_{V_s} * R(1+eta g), beam once - the production term) versus the former UNFILTERED approximation
(P R, P R g on the cells: exact_mean_field_response, the pre-repair C_M), at posterior samples of a fit.
usage: si_tmplcmp.py JOB.json POST.npz MODEL OUT.json [--n 400]
All other components, the noise layer and Sigma_data are identical; reports per sample the change of the predicted
curve in units of sqrt(diag Sigma_data) and Delta chi^2 = chi2(unfiltered) - chi2(filtered), plus the C_M share."""
import sys, json
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]/"ktzobs"))
import si_model, si_tables as st
from scripts import ktz_exact_window as kew
a = sys.argv; job = json.load(open(a[1])); post = np.load(a[2], allow_pickle=True); name = a[3]; out = a[4]
n = int(a[a.index("--n")+1]) if "--n" in a else 400
cfg = job["cfg"]; model = si_model.build(cfg, job["tables"], log=lambda m: None, fast_path=job.get("fast")); model.si_skip = {"L0": {"mean", "cnp"}, "L1": {"cnp"}, "L1n": {"mean"}, "L2": set()}[name]   # as si_fit.SKIP
X = post[f"{name}_samples"]; rng = np.random.default_rng(3); X = X[rng.choice(len(X), size=min(n, len(X)), replace=False)]
d = Path(cfg["dir"]); pm = dict(np.load(d/f"{cfg['g']}_pixel_map.npz")); diag = np.genfromtxt(d/f"{cfg['g']}_diagnostics.csv", delimiter=",", names=True)
r = diag["r_kpc"]; dr = r[1]-r[0]; edges = np.concatenate(([r[0]-0.5*dr], r+0.5*dr))
geom_aw = 0.4 if float(pm["pixel_size_kpc"]) >= 0.1 else 0.2
r_ref = float(json.load(open(d/f"{cfg['g']}_morphology.json"))["r_ref_kpc"])   # = Config.r_ref (si_tables reads the same key)
slopes = X[:, 4]
old = kew.exact_mean_field_response(pm, r_ref, slopes, edges, geom_aw)          # (n, 3, 2, nb), per ordered pair
t_new = model.raw_terms(X); beta, eta = np.exp(X[:, 2]), np.exp(X[:, 3])
t_old = t_new.copy(); t_old[:, 2] = beta[:, None, None]*(old[:, 0]+eta[:, None, None]*old[:, 1]+eta[:, None, None]**2*old[:, 2])
bias = model.data["noise_additive_bias"]
p_new = model._ratio(t_new).sum(1)+bias; p_old = model._ratio(t_old).sum(1)+bias
y = model.y; L = model.precision_root_joint; keep = np.diag(model.covariance_joint) < 1e3; sd = np.sqrt(np.diag(model.covariance_joint))
c_new = np.sum(((p_new-y) @ L.T)**2, 1); c_old = np.sum(((p_old-y) @ L.T)**2, 1)
dz = (p_old-p_new)[:, keep]/sd[keep]
cm_new = model._ratio(t_new)[:, 2][:, keep]; cm_old = model._ratio(t_old)[:, 2][:, keep]
res = dict(job=a[1], post=a[2], model=name, n=len(X), n_finite=int(np.sum(np.isfinite(c_new-c_old))), r_ref=r_ref,
           dchi2_old_minus_new=dict(median=float(np.nanmedian(c_old-c_new)), p16=float(np.nanpercentile(c_old-c_new, 16)), p84=float(np.nanpercentile(c_old-c_new, 84))),
           chi2_new_median=float(np.nanmedian(c_new)), chi2_old_median=float(np.nanmedian(c_old)),
           max_abs_dxi_over_sigma=dict(median=float(np.nanmedian(np.nanmax(np.abs(dz), 1))), p84=float(np.nanpercentile(np.nanmax(np.abs(dz), 1), 84))),
           CM_zero_lag_like_first_bin=dict(new=float(np.nanmedian(cm_new[:, 0])), old=float(np.nanmedian(cm_old[:, 0]))),
           CM_rms_ratio_old_over_new=float(np.nanmedian(np.sqrt(np.mean(cm_old**2, 1)/np.mean(cm_new**2, 1)))),
           r_kpc=r[keep].tolist(), median_CM_new=np.nanmedian(cm_new, 0).tolist(), median_CM_old=np.nanmedian(cm_old, 0).tolist(),
           median_dxi_over_sigma=np.nanmedian(dz, 0).tolist())
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("model", "n", "n_finite", "dchi2_old_minus_new", "chi2_new_median", "chi2_old_median", "max_abs_dxi_over_sigma", "CM_zero_lag_like_first_bin", "CM_rms_ratio_old_over_new")}))
