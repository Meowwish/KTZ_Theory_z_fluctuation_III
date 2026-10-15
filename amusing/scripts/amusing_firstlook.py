"""First look at a single-pointing MUSE galaxy for KTZ: abundance residual map and pair statistic.

    python scripts/amusing_firstlook.py --name NGC4030 --maps outputs/NGC4030/NGC4030_MAPS_muse.fits --distance 20.5 --pa 125 --ba 0.838 \
        [--ra 180.0985 --dec -1.1000] [--li22-lcorr 0.118] [--psf-arcsec 0.76]

Reproduces the PHANGS stage-1 conventions on a single MUSE field:
  * O'Donnell (1994) extinction from the Balmer decrement (E(B-V) clipped to [0, 2]);
  * Pilyugin & Grebel (2016) S calibration (same function as the PHANGS notebooks) and the H-beta-free Dopita et al.
    (2016) N2S2-Halpha calibration;
  * Li et al. (2021) spaxel selection: line S/N >= 2 on Ha, Hb, [N II], [S II]x2, [O III]; Kewley et al. (2001)
    star-forming side of the BPT; EW(Ha) >= 6 A; galactocentric radius inside the field;
  * deprojection with the catalogue PA and b/a (q0 = 0.13 as in Li21/Li22) onto a face-on grid at the native spaxel
    scale, annular detrending in 0.1-kpc rings, exhaustive FFT pair statistic (ktz_exact_window.PairStatistic) in
    0.06-kpc bins to r_max = 0.7 x median radius (Li22 edge rule) and a white-noise "blue-noise" band for the PSF.
Writes outputs/<name>/<name>_firstlook.{npz,json,png}.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, map_coordinates
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ktzobs"))   # cloud layout: KTZ_PHANGS snapshot next to this workspace
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "KTZ_PHANGS"))
from scripts import ktz_exact_window as kew

LINES = {"hb": "HB4861", "oiii": "OIII5006", "ha": "HA6562", "nii": "NII6583", "sii6716": "SII6716", "sii6730": "SII6730"}
WAVE = {"hb": 4861.35, "oiii": 5006.84, "ha": 6562.79, "nii": 6583.45, "sii6716": 6716.44, "sii6730": 6730.82}
Q0 = 0.13


def odonnell_k(w, rv=3.1):
    x = 1e4/w; y = x-1.82
    a = 1+0.104*y-0.609*y**2+0.701*y**3+1.137*y**4-1.718*y**5-0.827*y**6+1.647*y**7-0.505*y**8
    b = 1.952*y+2.908*y**2-3.989*y**3-7.985*y**4+11.102*y**5+5.491*y**6-10.805*y**7+3.347*y**8
    return rv*a+b


K = {n: odonnell_k(w) for n, w in WAVE.items()}


def scal(flux):
    with np.errstate(divide="ignore", invalid="ignore"):
        ebv = np.clip(2.5/(K["hb"]-K["ha"])*np.log10((flux["ha"]/flux["hb"])/2.86), 0, 2)
        c = {n: f*10**(0.4*K[n]*ebv) for n, f in flux.items()}
        r3 = 1.35*c["oiii"]/c["hb"]; n2 = 1.34*c["nii"]/c["hb"]; s2 = (c["sii6716"]+c["sii6730"])/c["hb"]
        lr = np.log10(r3/s2); ln = np.log10(n2); ls = np.log10(s2)
        upper = ln >= -0.6
        z = np.where(upper, 8.424+0.030*lr+0.751*ln+(-0.349+0.182*lr+0.508*ln)*ls, 8.072+0.789*lr+0.726*ln+(1.069-0.170*lr+0.022*ln)*ls)
        z[~((r3 > 0) & (n2 > 0) & (s2 > 0))] = np.nan
    return z, ebv, c


def d16(flux):
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log10(flux["nii"]/(flux["sii6716"]+flux["sii6730"])) + 0.264*np.log10(flux["nii"]/flux["ha"])
        z = 8.77 + y + 0.45*(y+0.3)**5
        return np.where((z >= 7.63) & (z <= 9.23), z, np.nan)   # D16 validity range used by Li22


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--maps", required=True); ap.add_argument("--distance", type=float, required=True)
    ap.add_argument("--pa", type=float, required=True); ap.add_argument("--ba", type=float, required=True)
    ap.add_argument("--ra", type=float); ap.add_argument("--dec", type=float); ap.add_argument("--li22-lcorr", type=float)
    ap.add_argument("--psf-arcsec", type=float); ap.add_argument("--sn-min", type=float, default=2.0); ap.add_argument("--ew-min", type=float, default=6.0)
    ap.add_argument("--annulus-kpc", type=float, default=0.10); ap.add_argument("--pa-from-catalog", action="store_true"); ap.add_argument("--template-fwhm-kpc", type=float, default=0.5); ap.add_argument("--error-inflation", type=float, default=1.12)
    a = ap.parse_args(); out = Path(a.maps).parent; name = a.name
    hd = fits.open(a.maps); h = hd["HA6562_FLUX"].header; wcs = WCS(h).celestial
    flux = {n: np.asarray(hd[e+"_FLUX"].data, float) for n, e in LINES.items()}
    err = {n: a.error_inflation*np.asarray(hd[e+"_FLUX_ERR"].data, float) for n, e in LINES.items()}
    ew = np.asarray(hd["HA6562_EW"].data, float)
    ra0 = a.ra if a.ra is not None else h.get("RA"); dec0 = a.dec if a.dec is not None else h.get("DEC")
    psf = a.psf_arcsec if a.psf_arcsec is not None else h.get("HIERARCH ESO OCS SGS FWHM MED", h.get("ESO OCS SGS FWHM MED", 0.9))
    # geometry: deproject sky offsets to face-on kpc
    ny, nx = flux["ha"].shape; yy, xx = np.mgrid[:ny, :nx]
    ra, dec = wcs.pixel_to_world_values(xx, yy)
    kpc_per_arcsec = a.distance*1e3*np.pi/180/3600
    dx = (ra-ra0)*np.cos(np.deg2rad(dec0))*3600*kpc_per_arcsec; dy = (dec-dec0)*3600*kpc_per_arcsec   # sky offsets [kpc], east positive
    incl = np.arccos(np.sqrt(max((a.ba**2-Q0**2)/(1-Q0**2), 1e-4)))
    # kinematic position angle from the H-alpha velocity field (PA of the receding major axis, N through E);
    # the catalogue PA is only used if --pa-from-catalog is given
    vgas = np.asarray(hd["V_GAS"].data, float); vok = np.isfinite(vgas) & (np.hypot(dx, dy) > 0.3)
    best = (None, -np.inf)
    for pa_deg in np.arange(0, 180, 1.0):
        p_ = np.deg2rad(pa_deg); xm_ = dx*np.sin(p_)+dy*np.cos(p_); ym_ = (-dx*np.cos(p_)+dy*np.sin(p_))/np.cos(incl)
        cosphi = xm_/np.hypot(xm_, ym_)
        A = np.c_[np.ones(vok.sum()), cosphi[vok]]; coef, res, *_ = np.linalg.lstsq(A, vgas[vok], rcond=None)
        score = -np.sum((vgas[vok]-A@coef)**2)
        if score > best[1]: best = (pa_deg if coef[1] > 0 else (pa_deg+180) % 360, score, coef)
    pa_kin = best[0]; print(f"[{name}] kinematic PA = {pa_kin:.0f} deg (catalogue value {a.pa}); vsys fit {best[2][0]:.1f} km/s, projected rotation amplitude {abs(best[2][1]):.1f} km/s")
    pa = np.deg2rad(a.pa if a.pa_from_catalog else pa_kin)
    # rotate so that the major axis is along x_face: PA measured from north through east
    xm = dx*np.sin(pa)+dy*np.cos(pa); ym = (-dx*np.cos(pa)+dy*np.sin(pa))/np.cos(incl)
    R = np.hypot(xm, ym)
    # selection
    with np.errstate(divide="ignore", invalid="ignore"):
        sn = {n: flux[n]/err[n] for n in LINES}
        z_s, ebv, corr = scal(flux); z_d = d16(flux)
        bx = np.log10(corr["nii"]/corr["ha"]); by = np.log10(corr["oiii"]/corr["hb"])
        bpt = np.isfinite(bx) & np.isfinite(by) & (bx < 0.4) & (by < 0.61/(bx-0.47)+1.19)
    good = np.all([np.isfinite(flux[n]) & (sn[n] >= a.sn_min) for n in LINES], axis=0) & bpt & (ew >= a.ew_min) & np.isfinite(z_s)
    good_d = np.all([np.isfinite(flux[n]) & (sn[n] >= a.sn_min) for n in ("ha", "nii", "sii6716", "sii6730")], axis=0) & bpt & (ew >= a.ew_min) & np.isfinite(z_d)
    # face-on grid at the native spaxel scale (0.2"); resample residual-free quantities by nearest neighbour
    pix = 0.2*kpc_per_arcsec; half = int(np.ceil(np.nanmax(R[good])/pix))+2
    axis = np.arange(-half, half+1)*pix; Yf, Xf = np.meshgrid(axis, axis, indexing="ij")
    # inverse transform face-on -> sky pixel
    dxs = Xf*np.sin(pa) - Yf*np.cos(incl)*np.cos(pa); dys = Xf*np.cos(pa) + Yf*np.cos(incl)*np.sin(pa)
    ras = ra0 + dxs/(3600*kpc_per_arcsec)/np.cos(np.deg2rad(dec0)); decs = dec0 + dys/(3600*kpc_per_arcsec)
    px, py = wcs.world_to_pixel_values(ras, decs); coords = np.vstack([py.ravel(), px.ravel()])
    def samp(arr, order=0, cval=np.nan):
        return map_coordinates(np.asarray(arr, float), coords, order=order, mode="constant", cval=cval).reshape(Xf.shape)
    results = {}
    for tag, zmap, gmask in (("scal", z_s, good), ("d16", z_d, good_d)):
        zf = samp(np.where(gmask, zmap, np.nan)); valid = np.isfinite(zf)
        Rf = np.hypot(Xf, Yf)
        # radial profile (annular means) and residual
        resid = kew.detrend(np.nan_to_num(zf), valid, Rf, width=a.annulus_kpc)
        ps = kew.PairStatistic(valid, pixel=pix, edges=np.arange(0.03, 5.031, 0.06))
        num, end, cnt = ps.sums(resid, valid)
        with np.errstate(invalid="ignore", divide="ignore"):
            xi = num/end
        rmax = 0.7*np.median(Rf[valid])
        # delete-one block jackknife (4 equal-area radial x 8 azimuthal blocks), re-detrending each time
        rr = Rf[valid]; redges = np.sqrt(np.linspace(rr.min()**2, rr.max()**2+1e-9, 5)); PHIf = np.mod(np.arctan2(Yf, Xf), 2*np.pi)
        block = 8*np.clip(np.digitize(Rf, redges)-1, 0, 3) + np.floor(PHIf/(2*np.pi)*8).astype(int); blocks = np.unique(block[valid]); vecs = []
        for L in blocks:
            m = valid & (block != L); z_ = kew.detrend(np.nan_to_num(zf), m, Rf, width=a.annulus_kpc); n_, e_, _ = ps.sums(z_, m)
            with np.errstate(invalid="ignore", divide="ignore"):
                vecs.append(n_/e_)
        vecs = np.array(vecs); fac = (len(blocks)-1)/len(blocks); dv = np.nan_to_num(vecs-np.nanmean(vecs, 0)); cov_jk = fac*dv.T@dv; xi_err = np.sqrt(np.diag(cov_jk))
        windows = {}
        for lo_, hi_ in ((1.0, 2.0), (2.0, 3.0), (3.0, 4.5)):
            tw = (ps.r >= lo_) & (ps.r <= hi_) & np.isfinite(xi) & (cnt > 0)
            windows[f"{lo_:.0f}-{hi_:.1f}"] = (float(np.nanmean(xi[tw])), float(np.sqrt(np.mean(cov_jk[np.ix_(tw, tw)])))) if tw.sum() > 2 else (np.nan, np.nan)
        tail_mean, tail_err = windows["1-2.0"]
        # blue-noise band: unit white noise convolved with the PSF, same mask, same detrending, 30 realisations
        rng = np.random.default_rng(1); sig_pix = psf*kpc_per_arcsec/2.354820045/pix; bn = []
        for _ in range(30):
            w = gaussian_filter(rng.normal(size=Xf.shape), sig_pix); w = kew.detrend(w, valid, Rf, width=a.annulus_kpc)
            n_, e_, _ = ps.sums(w, valid); bn.append(n_/e_)
        bn = np.array(bn)
        results[tag] = dict(xi=xi, xi_err=xi_err, cov_jk=cov_jk, tail_mean=tail_mean, tail_err=tail_err, windows=windows, r=ps.r, cnt=cnt, rmax=rmax, blue_p16=np.nanpercentile(bn, 16, 0), blue_p84=np.nanpercentile(bn, 84, 0), blue_med=np.nanmedian(bn, 0),
                            resid=resid, valid=valid, zf=zf, n_valid=int(valid.sum()), rms=float(np.std(resid[valid])), median_R=float(np.median(Rf[valid])))
        print(f"[{name}] {tag}: {valid.sum()} face-on cells ({pix*1e3:.0f} pc), residual rms {np.std(resid[valid]):.4f} dex, r_max {rmax:.2f} kpc; xi(0.06..0.3) = {np.round(xi[:5], 3)}; "
              "windows: " + "; ".join(f"{k} kpc {v[0]:+.3f}+-{v[1]:.3f}" for k, v in windows.items()))
    # ---- arm-template diagnostic: annulus-normalised, smoothed H-alpha modulation g = I/<I>_annulus - 1 ----
    Rf = np.hypot(Xf, Yf); ha_f = samp(np.where(np.isfinite(flux["ha"]) & (sn["ha"] >= 3), flux["ha"], np.nan), order=1)
    known = np.isfinite(ha_f); sig_t = a.template_fwhm_kpc/2.354820045/pix
    smooth_w = gaussian_filter(known.astype(float), sig_t); smooth_i = gaussian_filter(np.where(known, ha_f, 0.0), sig_t)
    intensity = np.where(smooth_w > 0.3, smooth_i/np.where(smooth_w > 0, smooth_w, 1), np.nan)
    g = np.full(Xf.shape, np.nan); lab = np.floor(Rf/a.annulus_kpc).astype(int)
    for L in np.unique(lab[np.isfinite(intensity)]):
        m = (lab == L) & np.isfinite(intensity)
        if m.sum() >= 20 and np.nanmean(intensity[m]) > 0: g[m] = intensity[m]/np.nanmean(intensity[m]) - 1
    arm = {}
    for tag, d in results.items():
        v = d["valid"] & np.isfinite(g); res = d["resid"]
        # binned mean residual versus template contrast
        edges = np.percentile(g[v], np.linspace(0, 100, 9)); cb = np.clip(np.digitize(g[v], edges)-1, 0, 7)
        gm = np.array([np.mean(g[v][cb == k]) for k in range(8)]); rm = np.array([np.mean(res[v][cb == k]) for k in range(8)])
        re_ = np.array([np.std(res[v][cb == k])/np.sqrt(np.sum(cb == k)) for k in range(8)])
        slope = np.polyfit(g[v], res[v], 1)[0]; corr = np.corrcoef(g[v], res[v])[0, 1]
        # exactly projected mean-field shape: pair product of the detrended template on the same mask
        f_t = kew.detrend(np.where(v, g, 0.0), v, Rf, width=a.annulus_kpc); ps = kew.PairStatistic(v, pixel=pix, edges=np.arange(0.03, 5.031, 0.06))
        n_, e_, c_ = ps.sums(f_t, v); cm_shape = n_/e_
        arm[tag] = dict(g_bins=gm, resid_bins=rm, resid_bin_err=re_, slope=float(slope), corr=float(corr), cm_shape=cm_shape, arm_rms=float(np.nanstd(g[v])))
        print(f"[{name}] {tag}: residual vs arm template: slope {slope:+.4f} dex per unit contrast, r = {corr:+.3f}; template rms contrast {np.nanstd(g[v]):.2f}")
    np.savez_compressed(out / f"{name}_firstlook.npz", X=Xf, Y=Yf, pixel_kpc=pix, arm_template=g, **{f"{t}_{k}": v for t, d in results.items() for k, v in d.items() if isinstance(v, np.ndarray)},
                        **{f"arm_{t}_{k}": v for t, d in arm.items() for k, v in d.items() if isinstance(v, np.ndarray)})
    summary = dict(name=name, distance_mpc=a.distance, pa_catalog=a.pa, pa_kinematic=float(pa_kin), pa_used=float(np.rad2deg(pa)), ba=a.ba, inclination_deg=float(np.rad2deg(incl)), psf_arcsec=psf, psf_pc=psf*kpc_per_arcsec*1e3, pixel_pc=pix*1e3,
                   n_spaxels_selected_scal=int(good.sum()), n_spaxels_selected_d16=int(good_d.sum()),
                   **{f"{t}_{k}": d[k] for t, d in results.items() for k in ("n_valid", "rms", "rmax", "median_R", "tail_mean", "tail_err", "windows")}, li22_lcorr_kpc=a.li22_lcorr, template_fwhm_kpc=a.template_fwhm_kpc, **{f'arm_{t}_{k}': d[k] for t, d in arm.items() for k in ('slope', 'corr', 'arm_rms')},
                   ebv_median=float(np.nanmedian(ebv[good])), metallicity_median_scal=float(np.nanmedian(z_s[good])), metallicity_median_d16=float(np.nanmedian(z_d[good_d])))
    json.dump(summary, (out / f"{name}_firstlook.json").open("w"), indent=1); print(json.dumps(summary, indent=1))
    # figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 13.5), constrained_layout=True)
    ext = [axis[0], axis[-1], axis[0], axis[-1]]
    im = axes[0, 0].imshow(np.log10(np.clip(samp(flux["ha"], order=1), 1, None)), origin="lower", extent=ext, cmap="gray_r"); axes[0, 0].set(title=r"log H$\alpha$ (face-on)", xlabel="x [kpc]", ylabel="y [kpc]")
    for j, (tag, lab) in enumerate((("scal", "S calibration"), ("d16", "D16 (H$\\beta$-free)"))):
        d = results[tag]
        im = axes[0, j+1].imshow(np.where(d["valid"], d["resid"], np.nan), origin="lower", extent=ext, cmap="RdBu_r", vmin=-0.12, vmax=0.12)
        axes[0, j+1].set(title=f"{lab}: residual, {d['n_valid']} cells, rms {d['rms']:.3f} dex", xlabel="x [kpc]"); fig.colorbar(im, ax=axes[0, j+1], label="dex")
        ax = axes[1, j+1]; r = d["r"]; m = r <= d["rmax"]
        ax.fill_between(r, d["blue_p16"], d["blue_p84"], color="0.75", label="PSF blue-noise band")
        ok = np.isfinite(d["xi"]) & (d["cnt"] > 0)
        ax.errorbar(r[ok], d["xi"][ok], d["xi_err"][ok], fmt="o", ms=2.5, color="k", lw=0.8, label=r"$\xi(r)$ with 32-block jackknife errors")
        ax.axvline(d["rmax"], color="0.4", lw=0.8, ls="-.", label=f"0.7 x median radius = {d['rmax']:.1f} kpc")
        ax.text(0.98, 0.95, f"1-2 kpc mean {d['tail_mean']:+.3f} ± {d['tail_err']:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.axhline(0, color="0.5", lw=0.6); ax.set(xlabel="r [kpc]", ylabel=r"$\xi(r)$", title=f"{lab}: pair statistic", ylim=(-0.15, 0.7), xlim=(0, 5))
        if a.li22_lcorr: ax.axvline(a.li22_lcorr, color="#d7301f", ls="--", lw=1, label=f"Li22 $l_{{\\rm corr}}$ = {a.li22_lcorr} kpc")
        ax.axvline(psf*kpc_per_arcsec, color="#1f77b4", ls=":", lw=1, label=f"PSF FWHM {psf*kpc_per_arcsec*1e3:.0f} pc"); ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]; ax.imshow(np.where(results["scal"]["valid"], results["scal"]["zf"], np.nan), origin="lower", extent=ext, cmap="viridis", vmin=8.3, vmax=8.75); ax.set(title="12+log(O/H), S calibration", xlabel="x [kpc]", ylabel="y [kpc]")
    ax = axes[2, 0]; im = ax.imshow(g, origin="lower", extent=ext, cmap="PuOr_r", vmin=-1, vmax=1); ax.set(title=f"H$\\alpha$ arm template ({a.template_fwhm_kpc:.1f} kpc FWHM), annulus-normalised", xlabel="x [kpc]", ylabel="y [kpc]"); fig.colorbar(im, ax=ax, label="$g$")
    for j, (tag, lab) in enumerate((("scal", "S calibration"), ("d16", "D16"))):
        d = results[tag]; A = arm[tag]; ax = axes[2, j+1]
        ax.errorbar(A["g_bins"], A["resid_bins"], A["resid_bin_err"], fmt="o-", color="k", label=f"binned residual; slope {A['slope']:+.3f} dex, r = {A['corr']:+.2f}")
        ax.axhline(0, color="0.5", lw=0.6); ax.set(xlabel="template contrast $g$", ylabel="mean abundance residual [dex]", title=f"{lab}: residual versus arm template"); ax.legend(frameon=False, fontsize=8)
        ins = ax.inset_axes([0.55, 0.55, 0.43, 0.4]); r = d["r"]; m = r <= d["rmax"]; sh = (r >= 0.3) & (r <= 1.0)
        cm = A["cm_shape"]; scale = np.nanmean(d["xi"][sh])/np.nanmean(cm[sh]) if np.nanmean(cm[sh]) > 0 else np.nan
        ins.plot(r[m], d["xi"][m], "k.", ms=3); ins.plot(r[m], (cm*scale)[m], color="#ff7f0e", lw=1.2); ins.axhline(0, color="0.5", lw=0.5)
        ins.set(title="$C_{\\rm M}$ shape (scaled) vs $\\xi$", xlabel="r [kpc]", ylim=(-0.05, 0.3)); ins.tick_params(labelsize=7); ins.title.set_size(8)
    fig.suptitle(f"{name}: D = {a.distance} Mpc, PSF {psf:.2f}\" = {psf*kpc_per_arcsec*1e3:.0f} pc, i = {np.rad2deg(incl):.0f}°, PA = {np.rad2deg(pa):.0f}° (kin)")
    fig.savefig(out / f"{name}_firstlook.png", dpi=110)


if __name__ == "__main__":
    main()
