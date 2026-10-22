"""Stage 1 + 2 for a single-pointing MUSE field, following the PHANGS conventions, then the exact-window KTZ fit.

    python scripts/amusing_ktz_stage.py --name NGC1084 --distance 19.7 --ba 0.918 --r25-arcmin 1.6 \
        [--pa 209] [--calib scal|d16] [--rmin 0.3 --rmax 3.9] [--steps 50000] [--evidence]

Stage 1 (map):  PSF-match the line maps to 0.12 kpc FWHM, resample onto a 0.06-kpc face-on grid (kinematic PA, b/a with
q0 = 0.13), Li21 selection, PG16 S calibration (or D16), 0.2-kpc annular detrending; exhaustive pair statistic in 0.06-kpc
bins to 5 kpc; 48-block jackknife + 200 line-flux Monte-Carlo covariance with shrinkage 0.35 and floor 0.01; noise
transfer (signal fraction, additive bias) from 50 correlated-noise realisations; 1-kpc H-alpha source template on the
same grid (annulus-normalised modulation, positivity limit); H-alpha rate slope and SFR for the reference event rate.
Stage 2 (fit):  exact window projection P C P^T of every KTZ component on the actual mask (one pointing, so no
zero-point term), 7-coordinate KTZ chain (flat eta prior, ell_src >= 0.12 kpc) and a KT18 reference chain (stretch move,
64 walkers), optional nested-model evidence L0/L1/L1n/L2 with dynesty.
Writes outputs/<name>/ktz/<calib>/ : <g>_pixel_map.npz, <g>_diagnostics.csv, <g>_fit_covariance.npz, <g>_morphology.json,
<g>_window_response_all.npz, <g>_chain_{KTZ,KT18}.npz, <g>_summary.json, <g>_curves.npz, <g>_fit.png, <g>_posterior.png,
<g>_evidence.json.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.stats import linregress
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for cand in (ROOT.parent / "ktzobs", ROOT.parent / "KTZ_PHANGS", ROOT.parent / "KTZ_theory_observations_2026-09-19"):
    if (cand / "scripts" / "ktz_exact_window.py").exists():
        sys.path.insert(0, str(cand)); break
from scripts import ktz_exact_window as kew
from scripts.phangs_source_inputs import sample_line
sys.path.insert(0, str(HERE))
from amusing_firstlook import LINES, WAVE, K, scal, d16, Q0

TARGET_FWHM_KPC = 0.12; PIXEL_KPC = 0.06; ANNULUS_KPC = 0.20; LINE_SN_MIN = 2.0; EW_MIN = 6.0; COVERAGE_MIN = 0.8
ERROR_INFLATION = 1.12; N_FLUX_MC = 200; N_NOISE = 50; MIN_PIX_ANNULUS = 20; TEMPLATE_FWHM_KPC = 1.0
EDGES = kew.FINE_EDGES; NB = len(EDGES) - 1
SLOPE_BOUND = 0.9   # |rate_log_slope| bound and response grid half-width [kpc^-1]; PHANGS used 0.35
ELL_GRID = np.geomspace(0.08, 3.2, 11); EPS_GRID = np.geomspace(0.015, 0.9, 10); SRC_GRID = np.geomspace(0.005, 2.5, 17)


def kinematic_pa(vgas, dx, dy, incl):
    vok = np.isfinite(vgas) & (np.hypot(dx, dy) > 0.3); best = (None, -np.inf)
    for pa_deg in np.arange(0, 180, 1.0):
        p = np.deg2rad(pa_deg); xm = dx*np.sin(p)+dy*np.cos(p); ym = (-dx*np.cos(p)+dy*np.sin(p))/np.cos(incl)
        A = np.c_[np.ones(vok.sum()), (xm/np.hypot(xm, ym))[vok]]; coef = np.linalg.lstsq(A, vgas[vok], rcond=None)[0]
        score = -np.sum((vgas[vok]-A@coef)**2)
        if score > best[1]: best = (pa_deg if coef[1] > 0 else (pa_deg+180) % 360, score)
    return best[0]


def halpha_cell_flux_factor(wcs, kpc_per_arcsec, cell_kpc, incl):
    """Flux-area factor for summing resampled line maps (2026-10-21 fix).
    The MAPS line fluxes are integrated fluxes per native spaxel (BUNIT 1e-20 erg/s/cm2, FLUXUNIT 'BUNIT*A').
    sample_line() smooths with a normalised kernel and interpolates at the face-on cell centres, so a cell value is the
    per-native-spaxel flux at that position (a surface brightness in units of one native spaxel).  The integrated flux of
    a cell is that value times the sky area of the cell divided by the sky area of a native spaxel.  For the thin-disc
    deprojection of build_maps() the sky area of a face-on cell of side cell_kpc is cell_kpc^2 cos(i) (Jacobian of the
    face-on -> sky mapping); the native spaxel area is taken from the WCS (proj_plane_pixel_area, exact for these
    rectilinear TAN grids).  Returns (factor, native spaxel area [kpc^2], cell sky area [kpc^2])."""
    from astropy.wcs.utils import proj_plane_pixel_area
    spaxel_kpc2 = float(proj_plane_pixel_area(wcs))*3600.0**2*kpc_per_arcsec**2
    cell_sky_kpc2 = float(cell_kpc)**2*float(np.cos(incl))
    return cell_sky_kpc2/spaxel_kpc2, spaxel_kpc2, cell_sky_kpc2


def halpha_sfr(cell_flux, valid, distance_mpc, factor):
    """SFR [Msun/yr] = 5.5e-42 L(Halpha) from the valid-cell fluxes (units 1e-20 erg/s/cm2 per native spaxel) times the
    flux-area factor; factor = 1 reproduces the pre-2026-10-21 (unscaled) value."""
    dist_cm = distance_mpc*1e6*3.085677581491367e18
    lum = float(np.sum(np.asarray(cell_flux)[valid].astype(np.float64))*float(factor)*1e-20*4*np.pi*dist_cm**2)
    return 5.5e-42*lum


def build_maps(a, out):
    g = a.name.lower(); t0 = time.time()
    mp = ROOT / "outputs" / a.name / f"{a.name}_MAPS_muse.fits"
    if not mp.exists(): mp = mp.with_suffix(".fits.gz")
    hd = fits.open(mp); h = hd["HA6562_FLUX"].header; wcs = WCS(h).celestial
    kpc_per_arcsec = a.distance*1e3*np.pi/180/3600; native_kpc = 0.2*kpc_per_arcsec
    psf_arcsec = h.get("HIERARCH ESO OCS SGS FWHM MED", h.get("ESO OCS SGS FWHM MED", 0.9)); psf_kpc = psf_arcsec*kpc_per_arcsec
    ra0, dec0 = (a.ra if a.ra is not None else h["RA"]), (a.dec if a.dec is not None else h["DEC"]); incl = np.arccos(np.sqrt(max((a.ba**2-Q0**2)/(1-Q0**2), 1e-4)))
    ny, nx = hd["HA6562_FLUX"].data.shape; yy, xx = np.mgrid[:ny, :nx]; ra, dec = wcs.pixel_to_world_values(xx, yy)
    dx = (ra-ra0)*np.cos(np.deg2rad(dec0))*3600*kpc_per_arcsec; dy = (dec-dec0)*3600*kpc_per_arcsec
    pa_kin = kinematic_pa(np.asarray(hd["V_GAS"].data, float), dx, dy, incl); pa = np.deg2rad(a.pa if a.pa is not None else pa_kin)
    # face-on grid
    half = int(np.ceil(a.rmax/PIXEL_KPC)); axis = np.arange(-half, half+1)*PIXEL_KPC; Y, X = np.meshgrid(axis, axis, indexing="ij"); R = np.hypot(X, Y)
    dxs = X*np.sin(pa) - Y*np.cos(incl)*np.cos(pa); dys = X*np.cos(pa) + Y*np.cos(incl)*np.sin(pa)
    ras = ra0 + dxs/(3600*kpc_per_arcsec)/np.cos(np.deg2rad(dec0)); decs = dec0 + dys/(3600*kpc_per_arcsec)
    px, py = wcs.world_to_pixel_values(ras, decs); coords = np.vstack([py.ravel(), px.ravel()])
    effective_fwhm = max(TARGET_FWHM_KPC, psf_kpc)   # PSF-match to the target, or keep the native PSF when it is already coarser
    sigma_pix = np.sqrt(max(TARGET_FWHM_KPC**2 - psf_kpc**2, 1e-6))/2.354820045/native_kpc
    flux, err, cov = {}, {}, {}
    native_flux = {n: np.asarray(hd[e+"_FLUX"].data, np.float32) for n, e in LINES.items()}
    native_err = {n: np.asarray(hd[e+"_FLUX_ERR"].data, np.float32) for n, e in LINES.items()}
    for n in LINES:
        f, e, c = sample_line(native_flux[n], native_err[n], coords, sigma_pix, X.shape); flux[n], err[n], cov[n] = f, ERROR_INFLATION*e, c
    cont, cont_err, cont_cov = sample_line(np.asarray(hd["HA6562_CONT"].data, np.float32), np.asarray(hd["HA6562_CONT_ERR"].data, np.float32), coords, sigma_pix, X.shape)
    geometric = (R >= a.rmin) & (R <= a.rmax) & (cont_cov >= COVERAGE_MIN)
    for n in LINES: geometric &= (cov[n] >= COVERAGE_MIN) & np.isfinite(flux[n]) & np.isfinite(err[n])
    calib = scal if a.calib == "scal" else (lambda f: (d16(f), np.zeros_like(f["ha"]), f))
    def select_and_metallicity(fl):
        with np.errstate(divide="ignore", invalid="ignore"):
            ew = fl["ha"]/cont; z, ebv, corr = calib(fl)
            bx = np.log10(corr["nii"]/corr["ha"]); by = np.log10(corr["oiii"]/corr["hb"])
            bpt = np.isfinite(bx) & np.isfinite(by) & (bx < 0.4) & (by < 0.61/(bx-0.47)+1.19)
            sn = {n: fl[n]/err[n] for n in LINES}
        need = LINES if a.calib == "scal" else ("ha", "nii", "sii6716", "sii6730")
        sel = geometric & bpt & (ew >= EW_MIN) & np.isfinite(z)
        for n in need: sel &= sn[n] >= LINE_SN_MIN
        return sel, z, corr
    def robust(sel, z, nsig=5.0):
        """Drop cells whose detrended residual exceeds nsig x MAD-sigma (calibration-branch failures), then re-detrend."""
        r0 = kew.detrend(np.nan_to_num(z), sel, R, width=ANNULUS_KPC); v = r0[sel]; mad = 1.4826*np.median(np.abs(v-np.median(v)))
        keep = sel & (np.abs(r0) <= nsig*mad); return keep, kew.detrend(np.nan_to_num(z), keep, R, width=ANNULUS_KPC)
    valid, zmap, corr = select_and_metallicity(flux); n_before = int(valid.sum())
    valid, resid = robust(valid, zmap); n_outliers = n_before - int(valid.sum())
    ps = kew.PairStatistic(valid, pixel=PIXEL_KPC, edges=EDGES)
    num, end, cnt = ps.sums(resid, valid); xi_obs = np.divide(num, end, out=np.full(NB, np.nan), where=end > 0)
    print(f"[{a.name}] stage 1: PSF {psf_arcsec:.2f}\" = {psf_kpc*1e3:.0f} pc -> {TARGET_FWHM_KPC*1e3:.0f} pc; PA_kin {pa_kin:.0f}, i {np.rad2deg(incl):.0f}; {valid.sum()} cells; rms {resid[valid].std():.4f} dex ({time.time()-t0:.0f} s)", flush=True)
    # jackknife (4 x 12)
    rr = R[valid]; redges = np.sqrt(np.linspace(rr.min()**2, rr.max()**2+1e-9, 5)); PHI = np.mod(np.arctan2(Y, X), 2*np.pi)
    block = 12*np.clip(np.digitize(R, redges)-1, 0, 3) + np.floor(PHI/(2*np.pi)*12).astype(int); blocks = np.unique(block[valid]); jk = []
    for L in blocks:
        m = valid & (block != L); z_ = kew.detrend(np.nan_to_num(zmap), m, R, width=ANNULUS_KPC); n_, e_, _ = ps.sums(z_, m); jk.append(np.divide(n_, e_, out=np.full(NB, np.nan), where=e_ > 0))
    jk = np.array(jk); fac = (len(blocks)-1)/len(blocks); dv = np.nan_to_num(jk-np.nanmean(jk, 0)); cov_jk = fac*dv.T@dv
    # line-flux Monte Carlo: perturb the native line maps, pass them through the same smoothing/resampling, selection and
    # calibration (dynamic selection); gives the measurement covariance, the per-cell metallicity error and the noise transfer
    rng = np.random.default_rng(1084); mc = []; zdraws = []; noise_num = []; noise_end = []; noise_maps = []
    obs_end = np.divide(end, cnt, out=np.zeros(NB), where=cnt > 0)
    for k in range(N_FLUX_MC):
        fl = {}
        for n in LINES:
            pert = native_flux[n] + rng.normal(size=native_flux[n].shape).astype(np.float32)*np.nan_to_num(native_err[n])
            fl[n], _, _ = sample_line(pert, native_err[n], coords, sigma_pix, X.shape)
        s_, z_, _ = select_and_metallicity(fl); s_, r_ = robust(s_, z_); zdraws.append(np.where(s_, z_, np.nan))
        n_, e_, _ = ps.sums(r_, s_); mc.append(np.divide(n_, e_, out=np.full(NB, np.nan), where=e_ > 0))
        if k < N_NOISE:   # correlated metallicity-noise realisation on the observed cells
            common = valid & s_; noise = np.where(common, z_ - zmap, 0.0); noise = kew.detrend(noise, common, R, width=ANNULUS_KPC); noise_maps.append(np.where(common, noise, np.nan).astype(np.float32))
            n2, e2, c2 = ps.sums(noise, common); noise_num.append(np.divide(n2, c2, out=np.zeros(NB), where=c2 > 0)); noise_end.append(np.divide(e2, c2, out=np.zeros(NB), where=c2 > 0))
        if (k+1) % 50 == 0: print(f"[{a.name}]   MC draw {k+1}/{N_FLUX_MC} ({time.time()-t0:.0f} s)", flush=True)
    mc = np.array(mc); cov_mc = np.cov(np.nan_to_num(mc), rowvar=False, ddof=1); z_err = np.nanstd(np.array(zdraws), 0)
    noise_num = np.mean(noise_num, 0); noise_end = np.mean(noise_end, 0)
    signal_fraction = np.clip(np.divide(np.clip(obs_end-noise_end, 0, None), obs_end, out=np.ones(NB), where=obs_end > 0), 0.05, 1.0)
    noise_bias = np.divide(noise_num, obs_end, out=np.zeros(NB), where=obs_end > 0)
    # covariance: jackknife (default) or mock-based (stochastic KT18 field + correlated noise on the same footprint)
    cov_mock = None
    if a.cov == "mock":
        from scipy.special import exp1
        from scripts.stage2_injection_recovery import gaussian_field
        beam2 = (effective_fwhm/2.354820045)**2; ell_o = np.sqrt(a.mock_ell**2 + beam2); eps_o = np.sqrt((a.mock_ell*a.mock_eps)**2 + beam2)/ell_o
        def kt18(rr):
            r2 = np.maximum(rr, 1e-9)**2
            return np.where(rr > 0, 0.5*(exp1(r2/(4*ell_o**2)) - exp1(r2/(4*(eps_o*ell_o)**2))), -np.log(eps_o))/(-np.log(eps_o))
        if a.mock_curve:   # posterior-predictive mock: correlation taken from a previous complete-model median curve
            C_ = np.load(a.mock_curve); stoch_share = 1.0
            if a.mock_stochastic:   # stochastic terms only (C_2p,0 + C_2p,N + C_NP): the deterministic mean field is not sample variance
                cm = C_["comp_median"]; st_ = cm[0]+cm[1]+cm[3]; tot = cm[:4].sum(0); stoch_share = float(np.clip(st_[0]/tot[0], 0.05, 1.0))
                r_c, xi_c = np.concatenate(([0.0], C_["r"])), np.concatenate(([1.0], st_/st_[0]))
            else:
                r_c, xi_c = np.concatenate(([0.0], C_["r"])), np.concatenate(([1.0], C_["ktz_median"]))
            def kt18(rr):
                return np.where(rr <= r_c[-1], np.interp(rr, r_c, xi_c), xi_c[-1]*np.exp(-(rr-r_c[-1])/1.0))
            print(f"[{a.name}] mock correlation taken from {a.mock_curve} ({'stochastic terms only, variance share %.2f' % stoch_share if a.mock_stochastic else 'complete curve'}; extrapolated beyond {r_c[-1]:.2f} kpc)", flush=True)
        sig_signal = np.sqrt(max(resid[valid].var() - np.nanmean(z_err[valid]**2), 1e-6))*np.sqrt(stoch_share if a.mock_curve else 1.0); N = 256; curves = []; nums = []; ends = []
        # Sigma_field: sample covariance of the SIGNAL field alone.  Measurement noise is NOT added here (corrected 2026-10-12):
        # it enters only through Sigma_MC, which propagates the line-flux errors (including their beam correlation, the selection and the
        # signal-noise cross terms) through the identical pipeline.  --mock-noise white restores the former (double-counted) behaviour.
        for k in range(a.n_mock):
            f = gaussian_field(rng, N, PIXEL_KPC, kt18)[:X.shape[0], :X.shape[1]]*sig_signal
            if a.mock_noise == "white": f = f + rng.normal(size=X.shape)*np.nan_to_num(z_err)
            z_ = kew.detrend(np.where(valid, f, 0.0), valid, R, width=ANNULUS_KPC); n_, e_, _ = ps.sums(z_, valid); curves.append(np.divide(n_, e_, out=np.full(NB, np.nan), where=e_ > 0)); nums.append(n_); ends.append(e_)
        cov_mock = np.cov(np.nan_to_num(np.array(curves)), rowvar=False, ddof=1)
        if a.mock_noise == "none":
            # the data statistic is normalised by the total endpoint variance (signal + noise), so the signal-field fluctuations of the
            # data statistic are those of the noise-free mock statistic scaled by the signal fraction of each bin: Sigma_field = s_i s_j Cov_mock
            cov_mock = np.outer(signal_fraction, signal_fraction)*cov_mock
        # estimator check: mean of the per-realisation ratio vs ratio of the mean numerator and denominator (the model uses the latter)
        ratio_bias = np.nanmean(np.array(curves), 0) - np.divide(np.mean(nums, 0), np.mean(ends, 0), out=np.zeros(NB), where=np.mean(ends, 0) > 0)
        print(f"[{a.name}] estimator ratio bias (mean of ratios minus ratio of means) over {a.n_mock} mocks: max |bias| = {np.nanmax(np.abs(ratio_bias)):.4f}", flush=True)
        i1, i2, i3 = [int(np.argmin(np.abs(ps.r-q))) for q in (1.0, 2.0, 3.3)]
        print(f"[{a.name}] mock covariance from {a.n_mock} KT18 realisations (ell* {a.mock_ell}, eps {a.mock_eps}, signal rms {sig_signal:.4f}); diag err at 1/2/3.3 kpc: "
              f"{np.sqrt(cov_mock[i1,i1]):.3f}/{np.sqrt(cov_mock[i2,i2]):.3f}/{np.sqrt(cov_mock[i3,i3]):.3f} (jackknife {np.sqrt(cov_jk[i1,i1]):.3f}/{np.sqrt(cov_jk[i2,i2]):.3f}/{np.sqrt(cov_jk[i3,i3]):.3f})", flush=True)
        raw = cov_mock + cov_mc
    else:
        raw = cov_jk + cov_mc
    fit_cov = (1-a.shrink)*raw + a.shrink*np.diag(np.diag(raw)) + a.floor**2*np.eye(NB)
    # H-alpha source template (1 kpc FWHM, annulus-normalised) and rate profile
    tracer = np.where(valid, corr["ha"], 0.0); known = geometric; sig_t = a.template_fwhm/2.354820045/PIXEL_KPC
    wgt = gaussian_filter(known.astype(float), sig_t); inten = np.divide(gaussian_filter(tracer, sig_t), wgt, out=np.full(X.shape, np.nan), where=wgt > 0.05)
    modulation = np.zeros(X.shape); lab = np.floor(R/ANNULUS_KPC).astype(int); support = np.zeros(X.shape, bool)
    for L in np.unique(lab[np.isfinite(inten)]):
        m = (lab == L) & np.isfinite(inten) & (R >= a.rmin) & (R <= a.rmax)
        if m.sum() >= MIN_PIX_ANNULUS and np.nanmean(inten[m]) > 0: modulation[m] = inten[m]/np.nanmean(inten[m]) - 1; support[m] = True
    mod_min = float(modulation[support].min()); eta_max = float(1/abs(mod_min)) if mod_min < 0 else 10.0
    contrast = float(np.sqrt(np.mean(modulation[support]**2)))
    w = corr["ha"][valid]; r_ref = float(np.exp(np.sum(w*np.log(R[valid]))/np.sum(w)))
    redges2 = np.arange(a.rmin, a.rmax+1e-9, ANNULUS_KPC); hist, _ = np.histogram(R[valid], bins=redges2, weights=corr["ha"][valid]); areas = np.pi*(redges2[1:]**2-redges2[:-1]**2)
    sd = hist/areas; pos = sd > 0; fit = linregress(0.5*(redges2[:-1]+redges2[1:])[pos], np.log(sd[pos])); slope, slope_err = float(fit.slope), float(fit.stderr)
    area_factor, spaxel_kpc2, cell_sky_kpc2 = halpha_cell_flux_factor(wcs, kpc_per_arcsec, PIXEL_KPC, incl)
    sfr = halpha_sfr(corr["ha"], valid, a.distance, area_factor); sfr_unscaled = halpha_sfr(corr["ha"], valid, a.distance, 1.0)
    morph = dict(galaxy=a.name, distance_mpc=a.distance, psf_arcsec=float(psf_arcsec), psf_kpc=float(psf_kpc), target_fwhm_kpc=TARGET_FWHM_KPC, effective_fwhm_kpc=float(effective_fwhm), pixel_kpc=PIXEL_KPC, pa_kinematic_deg=float(pa_kin),
                 pa_used_deg=float(np.rad2deg(pa)), inclination_deg=float(np.rad2deg(incl)), ba=a.ba, calibration=a.calib, n_cells=int(valid.sum()), n_outliers_removed=n_outliers, outlier_cut='5 x MAD sigma of the detrended residual', residual_rms_dex=float(resid[valid].std()),
                 analysis_rmin_kpc=a.rmin, analysis_rmax_kpc=a.rmax, r_ref_kpc=r_ref, mask_contrast_amplitude=contrast, mask_contrast_amplitude_error=0.0, maximum_positive_eta=eta_max,
                 event_rate_log_slope_kpc_inv=slope, inv_h_R_halpha_kpc_inv=-slope, event_rate_log_slope_error_kpc_inv=slope_err, halpha_sfr_msun_per_yr=sfr, halpha_flux_area_factor=area_factor, native_spaxel_area_kpc2=spaxel_kpc2, cell_sky_area_kpc2=cell_sky_kpc2, halpha_sfr_unscaled_pre20261021_msun_per_yr=sfr_unscaled, template_fwhm_kpc=a.template_fwhm,
                 n_jackknife_blocks=int(len(blocks)), n_flux_mc=N_FLUX_MC, covariance_kind=a.cov, mock_settings=(dict(ell=a.mock_ell, eps=a.mock_eps, n=a.n_mock) if a.cov == "mock" else None), covariance_shrinkage=a.shrink, systematic_floor=a.floor, mock_noise=a.mock_noise, ratio_bias_max=(float(np.nanmax(np.abs(ratio_bias))) if a.cov == 'mock' else None), selection_bias_term="not included (single realisation)")
    np.savez_compressed(out / f"{g}_pixel_map.npz", residual_dex=resid.astype(np.float32), valid_mask=valid, x_grid_kpc=X.astype(np.float32), y_grid_kpc=Y.astype(np.float32), metallicity_error_dex=np.where(valid, z_err, np.nan).astype(np.float32),
                        metallicity_12_log_oh=np.where(valid, zmap, np.nan).astype(np.float32), arm_source_modulation=modulation.astype(np.float32), template_support=support, halpha_flux=np.where(valid, corr["ha"], np.nan).astype(np.float32), pixel_size_kpc=PIXEL_KPC, target_fwhm_kpc=TARGET_FWHM_KPC, noise_realisations=np.array(noise_maps))
    diag = np.rec.fromarrays([ps.r, cnt, xi_obs, np.sqrt(np.diag(fit_cov)), np.sqrt(np.diag(cov_jk)), np.sqrt(np.diag(cov_mc)), signal_fraction, noise_bias],
                             names="r_kpc,pair_count,xi_centered,total_error,spatial_48_jackknife_error,line_flux_mc_error,noise_signal_fraction,noise_additive_bias")
    with (out / f"{g}_diagnostics.csv").open("w") as f:
        f.write(",".join(diag.dtype.names)+"\n"); np.savetxt(f, np.c_[[diag[n] for n in diag.dtype.names]].T, delimiter=",")
    np.savez_compressed(out / f"{g}_fit_covariance.npz", covariance=fit_cov, raw_covariance=raw, jackknife=cov_jk, measurement=cov_mc, **({"mock": cov_mock} if cov_mock is not None else {}))
    json.dump(morph, (out / f"{g}_morphology.json").open("w"), indent=1)
    print(f"[{a.name}] stage 1 done ({time.time()-t0:.0f} s): r_ref {r_ref:.2f} kpc, slope {slope:+.3f}+-{slope_err:.3f}, eta_max {eta_max:.2f}, template rms {contrast:.2f}, SFR {sfr:.2f} Msun/yr; "
          f"xi 1-2 kpc {np.nanmean(xi_obs[(ps.r >= 1) & (ps.r <= 2)]):+.3f}, 3-4.5 kpc {np.nanmean(xi_obs[(ps.r >= 3) & (ps.r <= 4.5)]):+.3f}", flush=True)
    return morph


def run_fit(a, out):
    g = a.name.lower(); t0 = time.time()
    pm = dict(np.load(out / f"{g}_pixel_map.npz")); morph = json.loads((out / f"{g}_morphology.json").read_text())
    diag = np.genfromtxt(out / f"{g}_diagnostics.csv", delimiter=",", names=True); fit_cov = np.load(out / f"{g}_fit_covariance.npz")["covariance"]
    rp = out / f"{g}_window_response_all.npz"
    if rp.exists():
        resp = dict(np.load(rp)); print(f"[{a.name}] loaded window response", flush=True)
    else:
        labels = np.zeros(pm["valid_mask"].shape, int)
        resp = kew.build_window_response(pm, labels, morph["r_ref_kpc"], selection="all", rate_log_slope_grid=np.linspace(-SLOPE_BOUND, SLOPE_BOUND, 7), edges=EDGES, annulus_width=ANNULUS_KPC, log=lambda m: print(f"[{a.name}] {m}", flush=True))
        np.savez_compressed(rp, **resp); print(f"[{a.name}] window response built in {time.time()-t0:.0f} s", flush=True)
    nuisance_max = 0.0
    if a.m1_nuisance is not None:   # deterministic unit-amplitude m=1 field cos(phi - phi0) through the same projector; amplitude A1 >= 0 fitted like sigma_q
        X, Y = pm["x_grid_kpc"], pm["y_grid_kpc"]; valid = pm["valid_mask"]; R = np.hypot(X, Y); PHI = np.arctan2(Y, X)
        f1 = kew.detrend(np.where(valid, np.cos(PHI - np.deg2rad(a.m1_nuisance)), 0.0), valid, R, width=ANNULUS_KPC)
        ps1 = kew.PairStatistic(valid, pixel=PIXEL_KPC, edges=EDGES); n1, e1, c1 = ps1.sums(f1, valid)
        resp = dict(resp); resp["delta_response"] = np.asarray([np.divide(n1, c1, out=np.zeros(NB), where=c1 > 0), np.divide(e1, c1, out=np.zeros(NB), where=c1 > 0)]); nuisance_max = 3.0
        print(f"[{a.name}] m=1 nuisance template at phi0 = {a.m1_nuisance} deg: projected shape at 1/2/3.3 kpc = {resp['delta_response'][0][16]/resp['delta_response'][1][16]:+.3f}/{resp['delta_response'][0][32]/resp['delta_response'][1][32]:+.3f}/{resp['delta_response'][0][54]/resp['delta_response'][1][54]:+.3f}", flush=True)
    zero = {k: (np.zeros_like(v) if k in ("response", "mean_response", "delta_response", "ordered_pair_counts") else v) for k, v in resp.items()}
    split = dict(D=np.zeros(10), cbin=np.zeros(len(diag["r_kpc"]), int), pairs_same=np.ones(len(diag["r_kpc"])), pairs_cross=np.ones(len(diag["r_kpc"])), coarse_edges=np.arange(0, 3.01, 0.3))
    grids = dict(ell_grid=ELL_GRID, epsilon_grid=EPS_GRID, source_length_grid=SRC_GRID)
    model = kew.SplitKTZ(morph, diag, fit_cov, resp, resp, zero, split, fit_cov, grids, use_D=False, sigma_q_max=nuisance_max, src_length_bounds=(0.12, None), eta_prior="flat", rate_log_slope_bound=SLOPE_BOUND, beam_fwhm_kpc=morph.get("effective_fwhm_kpc", TARGET_FWHM_KPC))
    print(f"[{a.name}] model ready ({time.time()-t0:.0f} s)", flush=True)
    rng = np.random.default_rng(1084)
    def start_from_prior(n, kt18=False):
        cands = []
        while sum(len(c) for c in cands) < 4*n:
            x = np.c_[rng.uniform(np.log(0.1), np.log(0.6), 4000), rng.uniform(-3, 0, 4000), rng.uniform(-2, 3, 4000), np.log(rng.uniform(0.05, min(model.eta_positive_limit, 2.0), 4000)),
                      rng.normal(model.slope_mu, min(model.slope_sd, 0.1), 4000), np.zeros(4000), np.log(rng.uniform(0.15, 1.5, 4000)), (rng.uniform(0.05, 1.5, 4000) if nuisance_max > 0 else np.zeros(4000))]
            x[:, 5] = x[:, 2] + np.log(rng.uniform(0.02, 50, 4000))   # ln(beta A_src) with A_src inside its prior
            if kt18: x = x[:, :2]
            x = x[np.isfinite(model.log_probability(x, kt18))]; cands.append(x)
        c = np.vstack(cands); return c[rng.choice(len(c), n, replace=False)]
    chains = {}
    for tag, kt18, nw in (("KTZ", False, 64), ("KT18", True, 32)):
        path = out / f"{g}_chain_{tag}.npz"
        if path.exists() and not a.refit:
            chains[tag] = dict(np.load(path)); print(f"[{tag}] cached chain", flush=True); continue
        start = start_from_prior(nw, kt18); t1 = time.time()
        res = kew.run_ensemble(model, start, a.steps if not kt18 else max(a.steps//5, 5000), kt18=kt18, seed=int(rng.integers(2**31)), thin=5, burn=a.steps//4 if not kt18 else max(a.steps//20, 1000),
                               progress=lambda k: print(f"[{tag}] step {k:,} ({time.time()-t1:.0f} s)", flush=True) if k % 10000 == 0 else None)
        chains[tag] = dict(samples=res["samples"], sample_lp=res["sample_lp"], rhat=res["rhat"], acceptance=res["acceptance"]); np.savez_compressed(path, **chains[tag])
        print(f"[{tag}] done {time.time()-t1:.0f} s; R-hat max {np.nanmax(res['rhat']):.3f}; acceptance {res['acceptance'].mean():.2f}", flush=True)
    # summaries
    s = chains["KTZ"]["samples"]; summary, curves = kew.summarise_joint(model, s)
    summary["rhat_max"] = float(np.nanmax(chains["KTZ"]["rhat"])); summary["chi2_dof"] = f"{len(diag['r_kpc'])} data, 7 parameters"
    k = chains["KT18"]["samples"]; d_ = k[rng.choice(len(k), min(2000, len(k)), replace=False)]; kt = model.predict_kt18(d_); kt_med = np.nanmedian(kt, 0)
    der = np.asarray(model.derived(np.c_[k[:, :2], np.zeros((len(k), 5))])).T
    summary_kt18 = dict(ell_star=np.percentile(der[:, 0], [16, 50, 84]).tolist(), epsilon=np.percentile(der[:, 1], [16, 50, 84]).tolist(), chi2_median_curve=model.chi2(kt_med), rhat_max=float(np.nanmax(chains["KT18"]["rhat"])))
    r = diag["r_kpc"]; obs = diag["xi_centered"]; err = np.sqrt(np.diag(fit_cov))
    tl = (r >= 3) & (r <= 4.5); sh = (r >= 1) & (r <= 2)
    summary["tail_obs"] = float(np.nanmean(obs[tl])); summary["tail_ktz"] = float(np.nanmean(curves["xi_median"][tl])); summary["tail_kt18"] = float(np.nanmean(kt_med[tl]))
    summary["shoulder_obs"] = float(np.nanmean(obs[sh])); summary["shoulder_ktz"] = float(np.nanmean(curves["xi_median"][sh])); summary["shoulder_kt18"] = float(np.nanmean(kt_med[sh]))
    summary["delta_chi2_kt18_minus_ktz"] = summary_kt18["chi2_median_curve"] - summary["chi2_all_median_curve"]
    if nuisance_max > 0: summary["m1_amplitude"] = summary["params"]["sigma_q"]; summary["m1_variance_fraction"] = summary["pointing_variance_fraction"]; summary["m1_phase_deg"] = a.m1_nuisance
    json.dump(dict(KTZ=summary, KT18=summary_kt18, settings=dict(steps=a.steps, eta_prior="flat", ell_src_lower=0.12, calibration=a.calib, rmin=a.rmin, rmax=a.rmax)), (out / f"{g}_summary.json").open("w"), indent=1)
    np.savez_compressed(out / f"{g}_curves.npz", r=r, obs=obs, err=err, ktz_median=curves["xi_median"], ktz_p16=curves["xi_p16"], ktz_p84=curves["xi_p84"], comp_median=curves["comp_median"], kt18_median=kt_med,
                        kt18_p16=np.nanpercentile(kt, 16, 0), kt18_p84=np.nanpercentile(kt, 84, 0))
    p = summary["params"]
    print(f"[{a.name}] KTZ: ell* {p['ell_star'][1]:.3f} [{p['ell_star'][0]:.3f},{p['ell_star'][2]:.3f}], eps {p['epsilon'][1]:.3f}, beta {p['beta_src'][1]:.2f}, eta {p['eta'][1]:.2f} [{p['eta'][0]:.2f},{p['eta'][2]:.2f}], "
          f"h_R^-1 {p['inv_h_R'][1]:+.3f}, ell_src {p['ell_src'][1]:.2f}; chi2 {summary['chi2_all_median_curve']:.1f}; tail obs/ktz/kt18 {summary['tail_obs']:+.3f}/{summary['tail_ktz']:+.3f}/{summary['tail_kt18']:+.3f}", flush=True)
    print(f"[{a.name}] KT18: ell* {summary_kt18['ell_star'][1]:.3f} [{summary_kt18['ell_star'][0]:.3f},{summary_kt18['ell_star'][2]:.3f}], chi2 {summary_kt18['chi2_median_curve']:.1f}; delta chi2 = {summary['delta_chi2_kt18_minus_ktz']:.1f}", flush=True)
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ax = axes[0]; ax.errorbar(r, obs, err, fmt="o", ms=2.5, color="k", lw=0.7, label="observed")
    ax.fill_between(r, np.nanpercentile(kt, 16, 0), np.nanpercentile(kt, 84, 0), color="#1f77b4", alpha=0.2); ax.plot(r, kt_med, color="#1f77b4", lw=1.3, label=f"KT18 ($\\chi^2$ = {summary_kt18['chi2_median_curve']:.1f})")
    ax.fill_between(r, curves["xi_p16"], curves["xi_p84"], color="#7b3294", alpha=0.25); ax.plot(r, curves["xi_median"], color="#7b3294", lw=1.6, label=f"complete KTZ ($\\chi^2$ = {summary['chi2_all_median_curve']:.1f})")
    ax.axhline(0, color="0.5", lw=0.6); ax.set(xlabel="r [kpc]", ylabel=r"$\xi_{3,\rm pp}(r)$", title=f"{a.name}: exact-window fits ({a.calib})", ylim=(-0.15, 0.5)); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]; cm = curves["comp_median"]; names = [r"$C_{2p,0}$ (KT18 transport)", r"$C_{2p,N}$", r"$C_{\rm M}$ (mean field)", r"$C_{\rm NP}$"]; cols = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    for i in range(4): ax.plot(r, cm[i], color=cols[i], lw=1.3, label=names[i])
    ax.plot(r, curves["xi_median"], color="#7b3294", lw=1.6, label="total"); ax.errorbar(r, obs, err, fmt="o", ms=2, color="k", lw=0.5)
    ax.axhline(0, color="0.5", lw=0.6); ax.set(xlabel="r [kpc]", ylabel="contribution", title="posterior-median components", ylim=(-0.15, 0.5)); ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / f"{g}_fit.png", dpi=120)
    # posterior figure
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True); derv = np.asarray(model.derived(s[:, :7])).T
    for ax, (vals, lab) in zip(axes.flat, [(derv[:, 0], r"$\ell_\ast$ [kpc]"), (derv[:, 1], r"$\epsilon$"), (np.log10(derv[:, 2]), r"$\log\beta_{\rm src}$"), (derv[:, 3], r"$\eta$"), (-s[:, 4], r"$h_R^{-1}$ [kpc$^{-1}$]"),
                                           (np.log10(derv[:, 7]), r"$\log{\cal A}_{\rm src}$"), (derv[:, 8], r"$\ell_{\rm src}$ [kpc]"), (derv[:, 5], r"$t_\ast$ [Gyr]")]):
        ax.hist(vals, 50, color="#7b3294", alpha=0.7); ax.set_xlabel(lab); ax.set_yticks([])
    fig.suptitle(f"{a.name}: KTZ posterior ({a.calib})"); fig.savefig(out / f"{g}_posterior.png", dpi=110)
    if a.evidence:
        from scripts.ktz_exact_window_evidence import evidence_table
        tab = evidence_table(model, models=(("L0", "L0q", "L1", "L1q", "L1n", "L1nq", "L2", "L2q") if nuisance_max > 0 else ("L0", "L1", "L1n", "L2")), nlive=400, log=lambda m: print(f"[{a.name}] {m}", flush=True))
        json.dump(tab, (out / f"{g}_evidence.json").open("w"), indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--distance", type=float, required=True); ap.add_argument("--ba", type=float, required=True)
    ap.add_argument("--pa", type=float); ap.add_argument("--calib", default="scal", choices=("scal", "d16")); ap.add_argument("--rmin", type=float, default=0.3); ap.add_argument("--rmax", type=float, default=3.9)
    ap.add_argument("--r25-arcmin", type=float); ap.add_argument("--steps", type=int, default=50000); ap.add_argument("--evidence", action="store_true"); ap.add_argument("--refit", action="store_true")
    ap.add_argument("--cov", default="jackknife", choices=("jackknife", "mock")); ap.add_argument("--mock-ell", type=float, default=0.35); ap.add_argument("--mock-eps", type=float, default=0.1); ap.add_argument("--n-mock", type=int, default=300); ap.add_argument("--rsep-max", type=float, default=5.0, help="largest separation bin edge [kpc] (with --bin)"); ap.add_argument("--mock-stochastic", action="store_true", help="with --mock-curve: use only the stochastic components as the mock correlation"); ap.add_argument("--mock-curve", help="npz with r/ktz_median: use this curve as the mock correlation")
    ap.add_argument("--m1-nuisance", type=float, help="phase [deg, face-on] of a unit m=1 nuisance template; its amplitude is fitted"); ap.add_argument("--tag", default=""); ap.add_argument("--template-fwhm", type=float, default=TEMPLATE_FWHM_KPC, help="H-alpha source-template smoothing FWHM [kpc]")
    ap.add_argument("--maps-only", action="store_true"); ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--mock-noise", default="none", choices=("none", "white"), help="add white per-cell noise to the field mocks (former behaviour; double-counts Sigma_MC)"); ap.add_argument("--shrink", type=float, default=0.35, help="shrinkage of the raw covariance towards its diagonal"); ap.add_argument("--floor", type=float, default=0.01, help="additive floor on the per-bin standard deviation")
    ap.add_argument("--ell-max", type=float, default=3.2, help="upper bound of the ell_star grid/prior [kpc]"); ap.add_argument("--src-max", type=float, default=2.5, help="upper bound of the ell_src grid/prior [kpc]"); ap.add_argument("--ra", type=float, help="galaxy centre RA [deg] (default: cube header)"); ap.add_argument("--dec", type=float); ap.add_argument("--pixel", type=float, help="face-on cell size [kpc] (default 0.06)"); ap.add_argument("--annulus", type=float, help="detrending annulus width [kpc] (default 0.2)"); ap.add_argument("--bin", type=float, help="separation bin width [kpc] (default 0.06; bins run to 5 kpc)")
    a = ap.parse_args()
    global PIXEL_KPC, ANNULUS_KPC, EDGES, NB, ELL_GRID, SRC_GRID
    if a.ell_max != 3.2: ELL_GRID = np.geomspace(0.08, a.ell_max, max(7, int(round(11*np.log(a.ell_max/0.08)/np.log(40)))))
    if a.src_max != 2.5: SRC_GRID = np.geomspace(0.005, a.src_max, max(17, int(round(17*np.log(a.src_max/0.005)/np.log(500)))))
    if a.pixel: PIXEL_KPC = a.pixel
    if a.annulus: ANNULUS_KPC = a.annulus; kew.ANNULUS_WIDTH = a.annulus
    if a.bin: EDGES = np.arange(a.bin/2, a.rsep_max+a.bin/2+1e-9, a.bin); NB = len(EDGES)-1
    out = ROOT / "outputs" / a.name / "ktz" / ((a.calib if a.cov == "jackknife" else f"{a.calib}_mockcov") + a.tag); out.mkdir(parents=True, exist_ok=True)
    if not a.fit_only: build_maps(a, out)
    if not a.maps_only: run_fit(a, out)


if __name__ == "__main__":
    main()
