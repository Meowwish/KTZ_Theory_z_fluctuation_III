"""Injection-recovery test of the stage-2 joint fit on the real NGC 628 footprint (appendix robustness check).

    python -m scripts.stage2_injection_recovery [--steps 30000] [--only NAME ...]

Each realisation is a simulated abundance-residual map on the observed valid-pixel mask:
  * a stationary Gaussian random field with the KT18 connected correlation at the stage-2 posterior median
    (ell*, epsilon), beam-convolved as in the model;
  * optionally a wide Gaussian-correlated component (length L, variance share f_NP) mimicking a source-pair excess;
  * optionally a deterministic mean field P[rate(R)(1 + eta g)] built from the stage-1 H-alpha template
    (variance share f_M) at the posterior-median eta and h_R^-1;
  * pixel noise from the released error map;
  * one constant offset per pointing drawn from N(0, sigma_off) (sigma_off = 0 for the null realisations).
The realisation then goes through the identical stage-2 pipeline (annular detrending, same/cross pair
statistic, 48-block jackknife, joint covariance recipe, exact window responses, joint fit with the
rank-K pointing term) and the recovered (ell*, eta, sigma_q, pointing variance fraction) are compared
with the truth.  Writes outputs/implications/NGC628_exact_window_joint/robustness/ngc628_injection_recovery.json.
"""
from __future__ import annotations
import json, sys, time
import numpy as np
from scipy.special import exp1
from scripts import ktz_exact_window as kew
from scripts.stage2_robustness import load_products, make_model, run_fit, replace_field, ensure_positive_definite

GALAXY = "NGC628"
SIGMA_OFF_DEX = 0.0106
# name: (seed, sigma_off, wide component (L_kpc, share) or None, mean field share or None)
REALISATIONS = {
    "kt18_off_1": (1, SIGMA_OFF_DEX, None, None), "kt18_off_2": (2, SIGMA_OFF_DEX, None, None), "kt18_off_3": (3, SIGMA_OFF_DEX, None, None),
    "kt18_zero_4": (4, 0.0, None, None), "kt18_zero_5": (5, 0.0, None, None),
    "np_off_6": (6, SIGMA_OFF_DEX, (0.6, 0.15), None), "np_off_7": (7, SIGMA_OFF_DEX, (0.6, 0.15), None), "np_zero_8": (8, 0.0, (0.6, 0.15), None),
    "mf_off_11": (11, SIGMA_OFF_DEX, None, 0.23), "mf_off_12": (12, SIGMA_OFF_DEX, None, 0.23), "mf_zero_13": (13, 0.0, None, 0.23),
    "mfnp_off_14": (14, SIGMA_OFF_DEX, (0.6, 0.10), 0.23),
    # offsets at the amplitude fitted to NGC 628 (sigma_q ~ 0.36)
    "kt18_big_21": (21, 0.016, None, None), "kt18_big_22": (22, 0.016, None, None), "mf_big_23": (23, 0.016, None, 0.23), "mf_big_24": (24, 0.016, None, 0.23),
}


def gaussian_field(rng, N, pix, corr):
    """Stationary Gaussian random field with the given correlation function corr(r) via circulant embedding."""
    fy, fx = np.meshgrid(np.fft.fftfreq(N) * N * pix, np.fft.fftfreq(N) * N * pix, indexing="ij"); rr = np.hypot(fx, fy)
    Pk = np.clip(np.real(np.fft.fft2(corr(rr))), 0, None)
    white = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    field = np.real(np.fft.ifft2(np.sqrt(Pk) * np.fft.fft2(white))) / np.sqrt(2)
    return field / field.std()


def simulate(P, rng, sigma_off, wide, mf_share, truth):
    pm = P["pixel_map"]; valid = pm["valid_mask"].astype(bool); h, w = valid.shape; pix = float(pm["pixel_size_kpc"])
    pointing = P["labels"]["pointing"]; N = 512
    beam2 = (kew.BEAM_FWHM_KPC / 2.354820045)**2   # full beam variance (corrected 2026-10-10)
    ell_o = np.sqrt(truth["ell_star"]**2 + beam2); eps_o = np.sqrt((truth["ell_star"] * truth["epsilon"])**2 + beam2) / ell_o
    def kt18(rr):
        r2 = np.maximum(rr, 1e-9)**2
        return np.where(rr > 0, 0.5 * (exp1(r2 / (4 * ell_o**2)) - exp1(r2 / (4 * (eps_o * ell_o)**2))), -np.log(eps_o)) / (-np.log(eps_o))
    field = gaussian_field(rng, N, pix, kt18)
    if wide is not None:
        L, share = wide; g = gaussian_field(rng, N, pix, lambda rr: np.exp(-rr**2 / (2 * L**2)))
        field = np.sqrt(1 - share) * field + np.sqrt(share) * g
    field = field[:h, :w]
    if mf_share is not None:
        R = np.hypot(pm["x_grid_kpc"], pm["y_grid_kpc"])
        fm = np.where(valid, np.exp(truth["rate_log_slope"] * (R - P["morphology"]["r_ref_kpc"])) * (1 + truth["eta"] * pm["arm_source_modulation"]), 0.0)
        fm = kew.detrend(fm, valid, R); fm /= fm[valid].std()
        field = np.sqrt(1 - mf_share) * field + np.sqrt(mf_share) * fm
    err = np.nan_to_num(pm["metallicity_error_dex"], nan=0.0)
    sig_signal = np.sqrt(max(pm["residual_dex"][valid].var() - (err[valid]**2).mean(), 1e-6))
    sim = sig_signal * field + rng.normal(size=(h, w)) * err
    offsets = rng.normal(0, sigma_off, int(pointing.max()) + 1) if sigma_off > 0 else np.zeros(int(pointing.max()) + 1)
    sim = sim + np.where(pointing >= 0, offsets[np.clip(pointing, 0, None)], 0.0)
    return sim, offsets, float(sig_signal)


def main(n_steps=30000, only=None):
    P = load_products(GALAXY); g = P["prefix"]; out_dir = P["stage2"] / "robustness"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{g}_injection_recovery.json"; log_path = out_dir / f"{g}_injection_recovery.log"
    def log(msg):
        print(f"[{GALAXY}] {msg}", flush=True); log_path.open("a").write(msg + "\n")
    pA = P["summary"]["A"]["params"]
    truth = dict(ell_star=pA["ell_star"][1], epsilon=pA["epsilon"][1], eta=pA["eta"][1], rate_log_slope=-pA["inv_h_R"][1])
    results = json.loads(out_path.read_text())["realisations"] if out_path.exists() else {}
    nb = len(P["split"]["xi_all"])
    for name, (seed, sigma_off, wide, mf_share) in REALISATIONS.items():
        if only and name not in only: continue
        if name in results: log(f"{name}: cached"); continue
        rng = np.random.default_rng(seed); t0 = time.time()
        sim, offsets, sig_signal = simulate(P, rng, sigma_off, wide, mf_share, truth)
        split = kew.split_statistics(replace_field(P["pixel_map"], sim), P["labels"]["pointing"], log=lambda m: None)
        C, cross_scale = ensure_positive_definite(kew.joint_covariance(P["fit_covariance"], split["cov_jk"], nb), nb)
        if cross_scale < 1: log(f"{name}: cross block scaled by {cross_scale} for positive definiteness")
        diagnostics = P["diagnostics"].copy(); diagnostics["xi_centered"] = split["xi_all"]
        truth_var_frac = sigma_off**2 / (sigma_off**2 + sig_signal**2)
        total_rms = float(np.sqrt(P["pixel_map"]["residual_dex"][P["pixel_map"]["valid_mask"].astype(bool)].var() + sigma_off**2))
        truth_sigma_q = sigma_off / total_rms
        log(f"{name}: sigma_off {sigma_off:.4f} dex (pointing variance fraction {truth_var_frac:.3f}), signal rms {sig_signal:.4f}; statistics in {time.time() - t0:.0f} s; "
            f"D/err {np.round(split['D'] / split['D_err'], 1)}")
        model = make_model(P, split=split, covariance=C, diagnostics=diagnostics)
        summary, _, _ = run_fit(model, P["chain_A"], n_steps, seed=100 + seed, tag=name, log=log)
        results[name] = dict(seed=seed, sigma_off_dex=sigma_off, wide_component=wide, mean_field_share=mf_share, truth=truth, truth_pointing_variance_fraction=truth_var_frac, truth_sigma_q=truth_sigma_q,
                             signal_rms_dex=sig_signal, offsets_dex=offsets.tolist(), params=summary["params"], pointing_variance_fraction=summary["pointing_variance_fraction"],
                             chi2_joint_median=summary["chi2_joint_median"], rhat_max=summary["rhat_max"], D=split["D"].tolist(), D_err=split["D_err"].tolist())
        json.dump(dict(galaxy=GALAXY, n_steps=n_steps, truth=truth, realisations=results), out_path.open("w"), indent=1)
    log("all realisations done")


if __name__ == "__main__":
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 30000
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
    main(steps, only)
