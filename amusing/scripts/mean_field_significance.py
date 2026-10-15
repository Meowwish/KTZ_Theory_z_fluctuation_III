"""Direct mean-field test: regression slope of the detrended abundance residual on the (annulus-normalised) H-alpha template,
compared with its null distribution from stochastic mock fields (no mean field) on the same footprint.
    python scripts/mean_field_significance.py --variant outputs/NGC2466/ktz/d16_mockcov_fid [--n 300]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ktzobs")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts import ktz_exact_window as kew
from scripts.stage2_injection_recovery import gaussian_field
from scipy.special import exp1
ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); ap.add_argument("--n", type=int, default=2000); ap.add_argument("--curve", help="curves npz providing the stochastic correlation (default: the variant's own)"); ap.add_argument("--m1-control", action="store_true", help="control for an m=1 (cos phi, sin phi) pattern in the regression"); ap.add_argument("--out-suffix", default=""); a = ap.parse_args()
d = Path(a.variant); g = [p for p in d.glob("*_pixel_map.npz")][0].name.split("_pixel_map")[0]
pm = dict(np.load(d / f"{g}_pixel_map.npz")); morph = json.loads((d / f"{g}_morphology.json").read_text()); C = np.load(a.curve) if a.curve else np.load(d / f"{g}_curves.npz")
valid = pm["valid_mask"]; resid = pm["residual_dex"].astype(float); mod = pm["arm_source_modulation"].astype(float); sup = pm["template_support"] & valid
X, Y = pm["x_grid_kpc"], pm["y_grid_kpc"]; R = np.hypot(X, Y); pix = float(pm["pixel_size_kpc"]); z_err = np.nan_to_num(pm["metallicity_error_dex"].astype(float))
width = float(morph["annulus_kpc"]) if "annulus_kpc" in morph else (0.2 if pix < 0.1 else 0.4)
PHI = np.arctan2(Y, X)
def slope(z, m):
    zz = z[m]; gg = mod[m]; gg = gg - gg.mean()
    if a.m1_control:   # partial regression: template slope controlling for an m=1 pattern (cos phi, sin phi)
        A = np.c_[gg, np.cos(PHI[m]), np.sin(PHI[m])]; coef = np.linalg.lstsq(A, zz, rcond=None)[0]; return float(coef[0]), float(np.corrcoef(zz, gg)[0, 1])
    return float(np.sum(zz*gg)/np.sum(gg**2)), float(np.corrcoef(zz, gg)[0, 1])
b_obs, r_obs = slope(resid, sup)
rng = np.random.default_rng(7); N = 256
beam2 = (morph["effective_fwhm_kpc"]/2.354820045)**2   # full beam variance (corrected 2026-10-10)
def kt18_corr(ell, eps):
    ell_o = np.sqrt(ell**2+beam2); eps_o = np.sqrt((ell*eps)**2+beam2)/ell_o
    def f(rr):
        r2 = np.maximum(rr, 1e-9)**2
        return np.where(rr > 0, 0.5*(exp1(r2/(4*ell_o**2))-exp1(r2/(4*(eps_o*ell_o)**2))), -np.log(eps_o))/(-np.log(eps_o))
    return f
cm = C["comp_median"]; st_ = cm[0]+cm[1]+cm[3]; tot = cm[:4].sum(0); share = float(np.clip(st_[0]/tot[0], 0.05, 1)); r_c = np.concatenate(([0.0], C["r"])); xi_c = np.concatenate(([1.0], st_/st_[0]))
stoch = lambda rr: np.where(rr <= r_c[-1], np.interp(rr, r_c, xi_c), xi_c[-1]*np.exp(-(rr-r_c[-1])))
sig = np.sqrt(max(resid[valid].var()-np.mean(z_err[valid]**2), 1e-6))
out = dict(galaxy=g, calibration=morph["calibration"], slope_obs=b_obs, corr_obs=r_obs, template_rms=float(mod[sup].std()), resid_rms=float(resid[valid].std()), mean_field_rms_dex=float(abs(b_obs)*mod[sup].std()))
# measurement-noise contribution to the slope: from the line-flux Monte Carlo noise realisations propagated through the pipeline
# (beam-correlated, selection-aware), added in quadrature to the signal-field null; falls back to white noise if they are absent
if "noise_realisations" in pm and len(pm["noise_realisations"]) > 0:
    nb = np.array([slope(np.nan_to_num(nm.astype(float)), sup & np.isfinite(nm)) [0] for nm in pm["noise_realisations"]]); var_noise = float(nb.var()); noise_kind = f"line-flux MC ({len(nb)} realisations)"
else:
    nb = np.array([slope(kew.detrend(rng.normal(size=X.shape)*z_err*valid, valid, R, width=width), sup)[0] for _ in range(100)]); var_noise = float(nb.var()); noise_kind = "white (fallback)"
out["noise_slope_std"] = float(np.sqrt(var_noise)); out["noise_kind"] = noise_kind
for name, corr, amp in (("kt18_mock", kt18_corr(0.35, 0.1), sig), ("stochastic_L2_mock", stoch, sig*np.sqrt(share))):
    bs = []
    for k in range(a.n):
        f = gaussian_field(rng, N, pix, corr)[:X.shape[0], :X.shape[1]]*amp
        z_ = kew.detrend(np.where(valid, f, 0.0), valid, R, width=width); bs.append(slope(z_, sup)[0])
    bs = np.array(bs); std_tot = float(np.sqrt(bs.var() + var_noise))
    # the null field is Gaussian and the slope is linear in it, so the null slope is Gaussian: the z-score is exact given its variance;
    # the empirical p-value from a.n realisations is reported alongside (resolution 1/a.n)
    out[name] = dict(slope_mean=float(bs.mean()), slope_std_field=float(bs.std()), slope_std=std_tot, significance=float((b_obs-bs.mean())/std_tot),
                     p_two_sided_empirical_field_only=float(np.mean(np.abs(bs-bs.mean()) >= abs(b_obs-bs.mean()))), n_mocks=a.n)
    print(f"[{g} {morph['calibration']}] {name}: null slope {bs.mean():+.4f} +- {std_tot:.4f} (field {bs.std():.4f}, noise {np.sqrt(var_noise):.4f}, {noise_kind}); observed {b_obs:+.4f} -> {out[name]['significance']:.1f} sigma")
print(f"[{g} {morph['calibration']}] observed slope {b_obs:+.4f} dex per unit contrast, r = {r_obs:+.3f}; mean-field rms {out['mean_field_rms_dex']:.4f} dex of residual rms {out['resid_rms']:.4f} ({(out['mean_field_rms_dex']/out['resid_rms'])**2*100:.1f} % of variance)")
out["m1_control"] = bool(a.m1_control); json.dump(out, (d / f"{g}_mean_field_significance{a.out_suffix}.json").open("w"), indent=1)
