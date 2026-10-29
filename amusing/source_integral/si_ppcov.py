#!/usr/bin/env python3
"""si_ppcov.py NULL_INPUT_DIR POINTS.json GEN_DIR OUT_INPUT_DIR [--shrink 0.35 --floor 0.01]
Legal-field posterior-predictive SENSITIVITY covariance (review round 3, 2026-10-26).  It does not replace the null
(primary) covariance or the rebuilt EB covariance; it is a separately labelled third covariance.

Input: si_gen_joint.py draws at the parameter points of POINTS.json (complete L2 posterior over all ell_* intervals,
stratified, weights w_k summing to 1).  Statistic per draw: xi = N/D of the complete model (fixed mean m, legal e, random
dipole c ~ N(0, q^2), one noise map).  With mu_k, C_k the mean and covariance of xi at point k:
    C_cond   = sum_k w_k C_k                         conditional sampling covariance, averaged over the posterior
    C_betw   = sum_k w_k (mu_k - mu)(mu_k - mu)^T    spread of the conditional means = parameter uncertainty of the
                                                       expected statistic (the model prediction itself varies with theta
                                                       in the likelihood, so this block is NOT added to the likelihood
                                                       covariance; it is reported for comparison)
    C_pooled = C_cond + C_betw                       pooled mixture (posterior-predictive) covariance
The sensitivity covariance used in the fits is reg(C_cond) with the documented production regularisation
reg(C) = (1-s) C + s diag(C) + floor^2 I (s = 0.35, floor = 0.01).  Measurement noise is inside the draws (once), so no
Sigma_MC block is added.  Blocks reported (all conditional, weighted):  noise = C_cond[full] - C_cond[no noise];
cross = C_cond[full] - C_cond[x0], x0 = (N_nomean + N_m)/(D_nomean + D_m) (mean-fluctuation cross terms incl. dipole).
Checks: PSD (eigenvalues), MC convergence (split halves of the draws, bootstrap over points), amplitude MC errors."""
import sys, json, os, glob, hashlib, time
import numpy as np
from pathlib import Path
a = sys.argv; nd, ptsf, gd, od = Path(a[1]), a[2], Path(a[3]), Path(a[4])
s_ = float(a[a.index("--shrink")+1]) if "--shrink" in a else 0.35; fl = float(a[a.index("--floor")+1]) if "--floor" in a else 0.01
P = json.load(open(ptsf)); pts = P["points"]; n = len(pts)
files = [gd/f"point_{i:03d}.npz" for i in range(n)]; missing = [str(f) for f in files if not f.exists()]
assert not missing, f"missing {len(missing)} point files"
g = [p for p in nd.glob("*_pixel_map.npz")][0].name.split("_pixel_map")[0]
w = np.array([p["w"] for p in pts]); w = w/w.sum()
reg = lambda C: (1-s_)*C+s_*np.diag(np.diag(C))+fl**2*np.eye(len(C))
def stats(Z, key, sl=slice(None)):
    if key == "x0":
        N = Z["nomean"][sl, 0]+Z["mean_only"][0]; D = Z["nomean"][sl, 1]+Z["mean_only"][1]
    else:
        N, D = Z[key][sl, 0], Z[key][sl, 1]
    x = N/D; return x.mean(0), np.cov(x.T, ddof=1), N.mean(0)/D.mean(0)
Zs = [dict(np.load(f)) for f in files]; K = Zs[0]["full"].shape[0]
def weighted(key, sl=slice(None), wt=w, idx=None):
    idx = range(n) if idx is None else idx
    mus, Cs = [], []
    for i in idx:
        m_, C_, _ = stats(Zs[i], key, sl); mus.append(m_); Cs.append(C_)
    mus = np.array(mus); ww = wt[list(idx)]; ww = ww/ww.sum()
    Cc = np.einsum("k,kij->ij", ww, np.array(Cs)); mu = ww @ mus; dm = mus-mu
    return Cc, np.einsum("k,ki,kj->ij", ww, dm, dm), mu, mus
Cc, Cb, mu, mus = weighted("full"); Cc_nn, _, _, _ = weighted("full_nn"); Cc_x0, _, _, _ = weighted("x0")
Cp = Cc+Cb; R = reg(Cc); Rp = reg(Cp)
# MC convergence: split halves of the draws; bootstrap over parameter points
h1, _, _, _ = weighted("full", slice(0, K//2)); h2, _, _, _ = weighted("full", slice(K//2, K))
rng = np.random.default_rng(5); bd = []
for _ in range(200):
    idx = rng.choice(n, n, replace=True); C_, _, _, _ = weighted("full", idx=list(idx)); bd.append(np.diag(C_))
bd = np.array(bd)
diag_ = np.genfromtxt(nd/f"{g}_diagnostics.csv", delimiter=",", names=True); r = diag_["r_kpc"]
null = np.load(nd/f"{g}_fit_covariance.npz")["covariance"]
eig = lambda C: (float(np.linalg.eigvalsh(C).min()), float(np.linalg.eigvalsh(C).max()))
amps = np.array([float(Z["amp"]) for Z in Zs]); amp_se = np.array([float(Z["amp_rel_se"]) for Z in Zs])
summ = dict(n_points=n, draws_per_point=K, n_draws=n*K, weights_by_piece={str(pc): float(w[[i for i, p in enumerate(pts) if p["piece"] == pc]].sum()) for pc in map(list, {tuple(p["piece"]) for p in pts})},
            excluded_piece_mass=P["excluded_mass"], shrink=s_, floor=fl,
            psd=dict(cond_raw=eig(Cc), cond_reg=eig(R), pooled_raw=eig(Cp), between=eig(Cb)),
            diag_ratio_median=dict(cond_over_null=float(np.median(np.diag(Cc)/np.diag(null))), pooled_over_cond=float(np.median(np.diag(Cp)/np.diag(Cc))),
                                   noise_block_share=float(np.median(np.diag(Cc-Cc_nn)/np.diag(Cc))), cross_block_share=float(np.median(np.diag(Cc-Cc_x0)/np.diag(Cc)))),
            mc=dict(split_half_diag_rel_diff_median=float(np.median(np.abs(np.diag(h1)-np.diag(h2))/np.diag(Cc))),
                    split_half_diag_rel_diff_max=float(np.max(np.abs(np.diag(h1)-np.diag(h2))/np.diag(Cc))),
                    bootstrap_points_diag_rel_sd_median=float(np.median(bd.std(0)/np.diag(Cc))),
                    expected_rel_sd_of_diag_for_N_draws=float(np.sqrt(2/(n*(K-1))))),
            amplitude=dict(median=float(np.median(amps)), range=[float(amps.min()), float(amps.max())], mc_rel_se_median=float(np.median(amp_se)),
                           mc_rel_se_max=float(amp_se.max()), implied_cov_scale_rel_se_median=float(2*np.median(amp_se))),
            per_r_sd=dict(r=[float(x) for x in r], cond=np.sqrt(np.diag(Cc)).tolist(), pooled=np.sqrt(np.diag(Cp)).tolist(), null=np.sqrt(np.diag(null)).tolist()))
od.mkdir(parents=True, exist_ok=True)
for f in ("diagnostics.csv", "morphology.json", "pixel_map.npz"):
    dst = od/f"{g}_{f}"
    if not dst.exists():
        import shutil; shutil.copy2(nd/f"{g}_{f}", dst)
cov = od/f"{g}_fit_covariance.npz"; assert not cov.exists(), "refusing to overwrite"
tmp = str(cov)+f".{os.getpid()}.partial.npz"
np.savez_compressed(tmp, covariance=R, raw_covariance=Cc, conditional=Cc, between=Cb, pooled=Cp, pooled_regularised=Rp,
                    noise_block=Cc-Cc_nn, cross_block=Cc-Cc_x0, cond_no_noise=Cc_nn, cond_crossfree=Cc_x0, split_half_a=h1, split_half_b=h2,
                    mu_pooled=mu, mu_points=mus, weights=w)
os.replace(tmp, cov)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
prov = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), label="legal-field posterior-predictive SENSITIVITY covariance (not the primary null, not the EB)",
            null_input=str(nd), points=ptsf, points_sha256=sha(ptsf), gen_dir=str(gd), gen_files={f.name: sha(f) for f in files},
            code=dict(si_ppcov=sha(__file__)), covariance_used_in_fits="reg(conditional)", summary=summ,
            sha256={f: sha(od/f"{g}_{f}") for f in ("diagnostics.csv", "morphology.json", "pixel_map.npz", "fit_covariance.npz")})
json.dump(prov, open(od/"PROVENANCE.json", "w"), indent=1)
print(json.dumps({k: summ[k] for k in ("n_points", "n_draws", "psd", "diag_ratio_median", "mc", "amplitude")}, indent=1))
