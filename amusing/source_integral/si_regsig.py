#!/usr/bin/env python3
"""H-alpha template-regression significance against LEGAL stochastic null fields (2026-10-26).

usage: si_regsig.py CONFIG.json PAR.json OUT.json [--n 2000] [--seed 11] [--post POST.npz MODEL K]
With --post the null is posterior-predictive: K equal-weight posterior samples of MODEL (dominant piece) are drawn and
n/K fields generated at each (amplitude renormalised per sample); PAR then only supplies m1.
Replaces the null of amusing/scripts/mean_field_significance.py (stationary Gaussian fields with the correlation of
the old midpoint-rate L2 median curve) by realisations of the stochastic part of the new source-integral L2 fit:
    z_null = amp * (e_P + e_NP)       (same-event Gram draw with h = R(1 + eta g), distinct-event draw; no mean field,
                                       no dipole - cos/sin phi are regression controls),
amp fixed so that amp * (m + e) has the observed signal variance (signal_amplitude, as for the B check).
Regression, detrending, support, m=1 controls and the line-flux-noise slope variance are identical to the production
script (--m1-control path); the noise variance is added once, in quadrature.
"""
import sys, json, time
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]/"ktzobs"))
import si_generate as sg
from scripts import ktz_exact_window as kew


def main():
    a = sys.argv; cfgd = json.load(open(a[1])); par = json.load(open(a[2])); out = a[3]
    n = int(a[a.index("--n")+1]) if "--n" in a else 2000; seed = int(a[a.index("--seed")+1]) if "--seed" in a else 11
    post = None
    if "--post" in a:
        i = a.index("--post"); post = (a[i+1], a[i+2], int(a[i+3]))
    rng = np.random.default_rng(seed)
    if post is None:
        pars = [par]
    else:
        from scipy.special import expit
        z = np.load(post[0], allow_pickle=True); X = z[f"{post[1]}_samples"]; X = X[rng.choice(len(X), size=post[2], replace=False)]
        pars = [dict(par, ell=float(np.exp(x[0])), eps=float(expit(x[1])), beta=float(np.exp(x[2])), eta=float(np.exp(x[3])), slope=float(x[4]),
                     betaA=float(np.exp(x[5])), ell_src=float(np.exp(x[6])), A_dip=float(x[7])) for x in X]
    G = sg.Generator(cfgd, pars[0], rng=rng); pm = G.cfg.pm; t0 = time.time()
    amp, sig2 = sg.signal_amplitude(G)
    valid = pm["valid_mask"].astype(bool); X, Y = pm["x_grid_kpc"], pm["y_grid_kpc"]; R = np.hypot(X, Y); PHI = np.arctan2(Y, X)
    mod = np.asarray(pm["arm_source_modulation"], float); sup = pm["template_support"].astype(bool) & valid
    width = G.geom.aw; resid = np.asarray(pm["residual_dex"], float)

    def slope(z, m):
        gg = mod[m]-mod[m].mean(); A = np.c_[gg, np.cos(PHI[m]), np.sin(PHI[m])]
        return float(np.linalg.lstsq(A, z[m], rcond=None)[0][0])
    b_obs = slope(resid, sup)
    nb = np.array([slope(np.nan_to_num(np.asarray(nm, float)), sup & np.isfinite(nm)) for nm in pm["noise_realisations"]])
    var_noise = float(nb.var())
    bs = np.empty(n); bs_P = np.empty(n); per = int(np.ceil(n/len(pars))); amps = []
    for k in range(n):
        if k % per == 0 and k > 0 and post is not None:
            G = sg.Generator(cfgd, pars[k//per], rng=rng); amp, _ = sg.signal_amplitude(G)
        if k % per == 0:
            amps.append(amp)
        e, enp = G.draw_stochastic()
        bs[k] = slope(kew.detrend(np.where(valid, amp*(e+enp), 0.0), valid, R, width=width), sup)
        bs_P[k] = slope(kew.detrend(np.where(valid, amp*e, 0.0), valid, R, width=width), sup)
        if (k+1) % 200 == 0:
            print(f"{k+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    sd = float(np.sqrt(bs.var()+var_noise))
    res = dict(par=par, amp=amp, signal_var=sig2, slope_obs=b_obs, noise_slope_std=float(np.sqrt(var_noise)), n_noise=len(nb),
               null_slope_mean=float(bs.mean()), null_slope_std_field=float(bs.std()), slope_std=sd,
               significance=float((b_obs-bs.mean())/sd), field_only_P_slope_std=float(bs_P.std()),
               p_two_sided_empirical_field_only=float(np.mean(np.abs(bs-bs.mean()) >= abs(b_obs-bs.mean()))),
               n_mocks=n, seed=seed, posterior_predictive=None if post is None else dict(npz=post[0], model=post[1], K=post[2], amps=amps), n_age=len(G.V), seconds=time.time()-t0, m1_control=True,
               null="legal source-integral stochastic field amp*(e_P+e_NP) of the L2 parameter point")
    json.dump(res, open(out, "w"), indent=1)
    print(f"observed {b_obs:+.4f}; null {bs.mean():+.4f} +- {sd:.4f} (field {bs.std():.4f}, noise {np.sqrt(var_noise):.4f}) -> {res['significance']:.2f} sigma")


if __name__ == "__main__":
    main()
