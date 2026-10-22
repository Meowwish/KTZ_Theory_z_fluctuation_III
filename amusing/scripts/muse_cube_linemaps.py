"""Emission-line maps from an ESO Phase-3 MUSE cube (DATA/STAT), written in the PHANGS-DAP MAPS layout.

    python scripts/muse_cube_linemaps.py --cube data/eso/ADP.xxx.fits --name NGC4030 --vsys 1465 --out outputs/NGC4030

Method (per spaxel, fully vectorised):
  1. H-alpha + [N II] window: a grid in (velocity, velocity dispersion) is searched; at each node the line amplitudes
     (H-alpha, [N II]6583 with 6548 tied to 1/2.94) and a linear local continuum are solved by weighted linear least
     squares with the STAT variances; the chi^2 minimum is refined by parabolic interpolation in velocity.
  2. With (v, sigma) fixed, every other line group ([S II] doublet, H-beta, [O III]5007 with 4959 tied to 1/2.98,
     [O I]6300) is solved by weighted linear least squares with its own local linear continuum.  The line width
     combines the fitted dispersion with the MUSE line-spread function (Bacon et al. 2017 polynomial).
  3. Fluxes = amplitude * sigma_lambda * sqrt(2 pi); errors from the least-squares covariance (STAT variances).
     H-alpha and H-beta are corrected for underlying stellar absorption with a fixed equivalent width
     (default 2 A x local continuum), which is recorded in the header as a systematic.
  4. EW(H-alpha) = corrected flux / local continuum.  A pseudo r-band image (median of line-free 6000-6800 A)
     with DATA/STAT/DQ extensions is written for the stage-1 EW/continuum code path.
Output: <out>/<name>_MAPS_muse.fits with extensions <LINE>_FLUX, <LINE>_FLUX_ERR for HA6562, HB4861, NII6583, NII6548,
SII6716, SII6730, OIII5006, OIII4958, OI6300; HA6562_EW, HA6562_EW_ERR, HA6562_CONT, V_GAS, SIGMA_GAS, CHI2_HA, NPIX_HA;
<out>/<name>_rband_muse.fits; QA figures <out>/<name>_linemaps_qa.png.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

C_KMS = 299792.458
REST = dict(HB4861=4861.333, OIII4958=4958.911, OIII5006=5006.843, OI6300=6300.304, NII6548=6548.050, HA6562=6562.819, NII6583=6583.460,
            SII6716=6716.440, SII6730=6730.815)
# fitting groups: lines with free amplitude; tied lines (name, reference, ratio)
GROUPS = {
    "HA": dict(free=["HA6562", "NII6583"], tied=[("NII6548", "NII6583", 1/2.94)], window=(6520, 6610), cont=[(6480, 6520), (6610, 6650)]),
    "SII": dict(free=["SII6716", "SII6730"], tied=[], window=(6690, 6760), cont=[(6650, 6690), (6760, 6800)]),
    "HB": dict(free=["HB4861"], tied=[], window=(4840, 4885), cont=[(4800, 4840), (4885, 4920)]),
    "OIII": dict(free=["OIII5006"], tied=[("OIII4958", "OIII5006", 1/2.98)], window=(4940, 5030), cont=[(4900, 4940), (5030, 5070)]),
    "OI": dict(free=["OI6300"], tied=[], window=(6280, 6320), cont=[(6240, 6280), (6320, 6360)]),
}
BALMER_ABSORPTION_EW = 2.0   # Angstrom, applied to H-alpha and H-beta (systematic; recorded in header)


def lsf_fwhm(lam):
    """MUSE line-spread-function FWHM [A] (Bacon et al. 2017, UDF)."""
    return 5.866e-8*lam**2 - 9.187e-4*lam + 6.040


def read_window(hdul, lam, lo, hi):
    sel = np.where((lam >= lo) & (lam <= hi))[0]
    sl = slice(sel[0], sel[-1]+1)
    data = np.asarray(hdul["DATA"].data[sl], np.float32); var = np.asarray(hdul["STAT"].data[sl], np.float32)
    return lam[sl], data, var


def solve_group(lam, data, var, v, sig_v, free, tied, rest_window, cont_windows, amp_only=False):
    """Weighted linear least squares for [amplitudes..., c0, c1] at fixed (v, sigma_v) per spaxel.

    data, var: (nl, ny, nx); v, sig_v: (ny, nx).  Returns amplitudes (nfree, ny, nx), their errors, continuum at the
    group centre, continuum error, chi2 and the number of valid pixels."""
    nl, ny, nx = data.shape
    z = 1 + v/C_KMS
    lam_obs = {n: REST[n]*z for n in free + [t[0] for t in tied]}           # (ny, nx)
    lam_c = 0.5*(rest_window[0]+rest_window[1])*z
    good = np.isfinite(data) & np.isfinite(var) & (var > 0)
    w = np.where(good, 1/np.where(var > 0, var, 1), 0).astype(np.float64)
    y = np.where(good, data, 0).astype(np.float64)
    # basis functions: one Gaussian per free line (tied lines added to their reference), continuum 1 and (lam-lam_c)
    basis = []
    for n in free:
        sl = REST[n]*z*np.sqrt((sig_v/C_KMS)**2 + (lsf_fwhm(REST[n]*z)/2.354820045/(REST[n]*z))**2)   # sigma_lambda (ny,nx)
        g = np.exp(-0.5*((lam[:, None, None]-lam_obs[n][None])/sl[None])**2)
        for tname, ref, ratio in tied:
            if ref == n:
                slt = REST[tname]*z*np.sqrt((sig_v/C_KMS)**2 + (lsf_fwhm(REST[tname]*z)/2.354820045/(REST[tname]*z))**2)
                g = g + ratio*np.exp(-0.5*((lam[:, None, None]-lam_obs[tname][None])/slt[None])**2)
        basis.append(g)
    basis.append(np.ones_like(data, np.float64)); basis.append(np.broadcast_to((lam[:, None, None]-lam_c[None]), data.shape).astype(np.float64))
    B = np.stack(basis, axis=0)                                            # (nb, nl, ny, nx)
    nb = B.shape[0]
    # normal equations per spaxel
    A = np.einsum("ilyx,jlyx,lyx->yxij", B, B, w, optimize=True)         # (ny, nx, nb, nb)
    b = np.einsum("ilyx,lyx->yxi", B, w*y, optimize=True)
    npix = good.sum(0)
    ok = npix >= nb + 3
    A[~ok] = np.eye(nb)
    try:
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(A)
    p = np.einsum("yxij,yxj->yxi", cov, b)
    model = np.einsum("yxi,ilyx->lyx", p, B)
    chi2 = np.sum(w*(y-model)**2, axis=0)
    amps = np.moveaxis(p[..., :len(free)], -1, 0); amp_err = np.moveaxis(np.sqrt(np.maximum(np.einsum("yxii->yxi", cov)[..., :len(free)], 0)), -1, 0)
    cont = p[..., len(free)]; cont_err = np.sqrt(np.maximum(cov[..., len(free), len(free)], 0))
    for arr in (amps, amp_err):
        arr[:, ~ok] = np.nan
    cont[~ok] = np.nan; cont_err[~ok] = np.nan; chi2[~ok] = np.nan
    return amps, amp_err, cont, cont_err, chi2, npix


def sigma_lambda(name, v, sig_v):
    lo = REST[name]*(1+v/C_KMS)
    return lo*np.sqrt((sig_v/C_KMS)**2 + (lsf_fwhm(lo)/2.354820045/lo)**2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True); ap.add_argument("--name", required=True); ap.add_argument("--vsys", type=float, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--vrange", type=float, default=450.0); ap.add_argument("--vstep", type=float, default=20.0)
    ap.add_argument("--sigmas", type=str, default="12,25,40,60,90,140,220"); ap.add_argument("--balmer-abs-ew", type=float, default=BALMER_ABSORPTION_EW)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    hdul = fits.open(args.cube, memmap=True); hd = hdul["DATA"].header; h0 = hdul[0].header
    nl = hd["NAXIS3"]; lam = hd["CRVAL3"] + hd["CD3_3"]*(np.arange(nl)+1-hd["CRPIX3"])
    zs = 1 + args.vsys/C_KMS
    print(f"[{args.name}] cube {hd['NAXIS1']}x{hd['NAXIS2']}x{nl}, {lam[0]:.1f}-{lam[-1]:.1f} A, vsys {args.vsys} km/s", flush=True)

    # ---- 1. velocity / dispersion grid on the H-alpha group ----
    G = GROUPS["HA"]; lo = (G["cont"][0][0])*zs - 5; hi = (G["cont"][-1][1])*zs + 5
    lam_w, data_w, var_w = read_window(hdul, lam, lo, hi)
    ny, nx = data_w.shape[1:]
    vgrid = args.vsys + np.arange(-args.vrange, args.vrange+1e-6, args.vstep); sgrid = np.array([float(s) for s in args.sigmas.split(",")])
    best_chi2 = np.full((ny, nx), np.inf); best_v = np.full((ny, nx), np.nan); best_s = np.full((ny, nx), np.nan)
    chi2_cube = np.full((len(sgrid), len(vgrid), ny, nx), np.nan, np.float32)
    for i, s in enumerate(sgrid):
        for j, v in enumerate(vgrid):
            V = np.full((ny, nx), v); S = np.full((ny, nx), s)
            _, _, _, _, chi2, npix = solve_group(lam_w, data_w, var_w, V, S, G["free"], G["tied"], G["window"], G["cont"])
            chi2_cube[i, j] = chi2
        print(f"  sigma {s:5.0f} km/s scanned ({time.time()-t0:.0f} s)", flush=True)
    flat = np.nan_to_num(chi2_cube, nan=np.inf).reshape(len(sgrid)*len(vgrid), ny, nx)
    k = np.argmin(flat, axis=0); i_s, j_v = np.unravel_index(k, (len(sgrid), len(vgrid)))
    best_s = sgrid[i_s].astype(float); best_v = vgrid[j_v].astype(float)
    # parabolic refinement in v at the best sigma
    c0 = np.take_along_axis(chi2_cube, i_s[None, None], 0)[0]            # (nv, ny, nx)
    jm = np.clip(j_v, 1, len(vgrid)-2)
    ym, y0, yp = (np.take_along_axis(c0, (jm+d)[None], 0)[0] for d in (-1, 0, 1))
    den = ym - 2*y0 + yp
    shift = np.where(den > 0, 0.5*(ym-yp)/np.where(den > 0, den, 1), 0.0)
    best_v = vgrid[jm] + np.clip(shift, -1, 1)*args.vstep
    # final H-alpha group solution at refined (v, sigma)
    amps, aerr, cont_ha, cont_ha_err, chi2_ha, npix_ha = solve_group(lam_w, data_w, var_w, best_v, best_s, G["free"], G["tied"], G["window"], G["cont"])
    valid = np.isfinite(amps[0]) & (aerr[0] > 0) & (amps[0]/aerr[0] > 3)
    best_v[~valid] = np.nan; best_s[~valid] = np.nan
    # fill velocity/sigma of weak spaxels with a smoothed field so that the other groups can still be solved
    from scipy.ndimage import gaussian_filter
    def smooth_fill(f, default):
        m = np.isfinite(f); num = gaussian_filter(np.where(m, f, 0.0), 3); den = gaussian_filter(m.astype(float), 3)
        filled = np.where(den > 0.05, num/np.where(den > 0, den, 1), default)
        return np.where(m, f, filled)
    v_use = smooth_fill(best_v, args.vsys); s_use = smooth_fill(best_s, 60.0)
    print(f"  velocity field: {valid.sum()} spaxels with S/N(Ha)>3; median v {np.nanmedian(best_v):.1f}, median sigma {np.nanmedian(best_s):.1f} km/s ({time.time()-t0:.0f} s)", flush=True)

    # ---- 2. all groups at fixed kinematics ----
    maps = {}; hdr_notes = {}
    for gname, G in GROUPS.items():
        lo = (G["cont"][0][0])*zs - 5; hi = (G["cont"][-1][1])*zs + 5
        lam_w, data_w, var_w = read_window(hdul, lam, lo, hi)
        amps, aerr, cont, cont_err, chi2, npix = solve_group(lam_w, data_w, var_w, v_use, s_use, G["free"], G["tied"], G["window"], G["cont"])
        for n, a, e in zip(G["free"], amps, aerr):
            sl = sigma_lambda(n, v_use, s_use); f = a*sl*np.sqrt(2*np.pi); fe = e*sl*np.sqrt(2*np.pi)
            maps[n+"_FLUX"] = f.astype(np.float32); maps[n+"_FLUX_ERR"] = fe.astype(np.float32)
            maps[n+"_CONT"] = cont.astype(np.float32); maps[n+"_CONT_ERR"] = cont_err.astype(np.float32)
        for tname, ref, ratio in G["tied"]:
            maps[tname+"_FLUX"] = (ratio*maps[ref+"_FLUX"]).astype(np.float32); maps[tname+"_FLUX_ERR"] = (ratio*maps[ref+"_FLUX_ERR"]).astype(np.float32)
        maps["CHI2_"+gname] = chi2.astype(np.float32); maps["NPIX_"+gname] = npix.astype(np.int16)
        print(f"  group {gname}: done ({time.time()-t0:.0f} s)", flush=True)
    # Balmer absorption correction (fixed EW x local continuum); keep the uncorrected values too
    for n in ("HA6562", "HB4861"):
        maps[n+"_FLUX_NOABS"] = maps[n+"_FLUX"].copy()
        corr = args.balmer_abs_ew*np.nan_to_num(np.clip(maps[n+"_CONT"], 0, None))
        maps[n+"_FLUX"] = (maps[n+"_FLUX"] + corr).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ew = maps["HA6562_FLUX"]/maps["HA6562_CONT"]
        ew_err = np.sqrt((maps["HA6562_FLUX_ERR"]/maps["HA6562_CONT"])**2 + (maps["HA6562_FLUX"]*maps["HA6562_CONT_ERR"]/maps["HA6562_CONT"]**2)**2)
    maps["HA6562_EW"] = ew.astype(np.float32); maps["HA6562_EW_ERR"] = ew_err.astype(np.float32)
    maps["V_GAS"] = best_v.astype(np.float32); maps["SIGMA_GAS"] = best_s.astype(np.float32)

    # ---- 3. pseudo r-band continuum image ----
    rsel = (lam >= 6000*zs) & (lam <= 6800*zs)
    for n in ("OI6300", "NII6548", "HA6562", "NII6583", "SII6716", "SII6730"):
        rsel &= np.abs(lam - REST[n]*zs) > 12
    idx = np.where(rsel)[0]; sl = slice(idx[0], idx[-1]+1); keep = rsel[sl]
    dat = np.asarray(hdul["DATA"].data[sl], np.float32)[keep]; var = np.asarray(hdul["STAT"].data[sl], np.float32)[keep]
    with np.errstate(invalid="ignore"):
        rimg = np.nanmedian(dat, axis=0); rvar = np.nanmedian(var, axis=0)*np.pi/2/np.sum(np.isfinite(dat), axis=0).clip(1)
    rdq = (~np.isfinite(rimg) | ~np.isfinite(rvar) | (rvar <= 0)).astype(np.int16)

    # ---- 4. write ----
    wcs2 = WCS(hd).celestial; hdr2 = wcs2.to_header()
    for k in ("OBJECT", "RA", "DEC", "EXPTIME", "TEXPTIME", "NCOMBINE", "PROG_ID", "DATE-OBS", "MJD-OBS", "SPEC_RES", "SKY_RES", "SKY_RERR", "ABMAGLIM"):
        if k in h0: hdr2[k] = h0[k]
    for k in ("ESO OCS SGS FWHM MED", "ESO TEL IA FWHMLINOBS", "ESO TEL AMBI FWHM START", "ESO TEL AMBI FWHM END"):
        if k in h0: hdr2["HIERARCH " + k] = h0[k]
    prim = fits.PrimaryHDU(header=fits.Header([("ORIGIN", "muse_cube_linemaps.py"), ("CUBE", Path(args.cube).name), ("GALAXY", args.name), ("VSYS", args.vsys, "km/s, grid centre"),
                                               ("BALMABS", args.balmer_abs_ew, "A, stellar absorption EW added to Ha, Hb"), ("LSF", "Bacon+2017 polynomial"), ("FLUXUNIT", "1e-20 erg/s/cm2 (BUNIT*A)")]))
    hdus = [prim]
    for k, arr in maps.items():
        h = hdr2.copy(); h["EXTNAME"] = k; h["BUNIT"] = "1e-20 erg/s/cm2" if k.endswith("FLUX") or k.endswith("FLUX_ERR") or k.endswith("NOABS") else ("Angstrom" if "EW" in k else ("km/s" if k in ("V_GAS", "SIGMA_GAS") else ""))
        hdus.append(fits.ImageHDU(arr, header=h, name=k))
    fits.HDUList(hdus).writeto(out / f"{args.name}_MAPS_muse.fits", overwrite=True)
    rh = hdr2.copy(); rh["BUNIT"] = "1e-20 erg/s/cm2/A"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(rimg.astype(np.float32), header=rh, name="DATA"), fits.ImageHDU(rvar.astype(np.float32), header=rh, name="STAT"),
                  fits.ImageHDU(rdq, header=rh, name="DQ")]).writeto(out / f"{args.name}_rband_muse.fits", overwrite=True)
    summary = dict(name=args.name, cube=Path(args.cube).name, vsys_grid=args.vsys, v_median=float(np.nanmedian(best_v)), sigma_median=float(np.nanmedian(best_s)),
                   n_spaxels=int(ny*nx), n_ha_sn3=int(valid.sum()), n_all_lines_sn3=int(np.sum(np.all([maps[n+"_FLUX"]/maps[n+"_FLUX_ERR"] > 3 for n in ("HA6562", "HB4861", "NII6583", "SII6716", "SII6730", "OIII5006")], axis=0))),
                   psf_sgs_fwhm_arcsec=h0.get("ESO OCS SGS FWHM MED"), psf_tel_fwhm_arcsec=h0.get("ESO TEL IA FWHMLINOBS"), sky_res_arcsec=h0.get("SKY_RES"),
                   balmer_absorption_ew=args.balmer_abs_ew, time_s=time.time()-t0)
    json.dump(summary, (out / f"{args.name}_linemaps_summary.json").open("w"), indent=1); print(json.dumps(summary, indent=1))

    # ---- 5. QA figure ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), constrained_layout=True)
    def show(ax, a, title, log=False, **kw):
        im = ax.imshow(np.log10(np.clip(a, 1e-3, None)) if log else a, origin="lower", cmap=kw.pop("cmap", "viridis"), **kw); ax.set_title(title); fig.colorbar(im, ax=ax, shrink=0.8)
    show(axes[0, 0], maps["HA6562_FLUX"], r"log H$\alpha$ flux", log=True, vmin=1, vmax=4.5)
    show(axes[0, 1], maps["HB4861_FLUX"], r"log H$\beta$ flux", log=True, vmin=0.5, vmax=4)
    show(axes[0, 2], maps["SII6716_FLUX"]/maps["SII6716_FLUX_ERR"], "S/N [S II]6716", vmin=0, vmax=20)
    show(axes[0, 3], maps["HA6562_EW"], r"EW(H$\alpha$) [A]", vmin=0, vmax=60)
    show(axes[1, 0], best_v - args.vsys, r"$v - v_{\rm sys}$ [km/s]", cmap="RdBu_r", vmin=-250, vmax=250)
    show(axes[1, 1], best_s, r"$\sigma_v$ [km/s]", vmin=30, vmax=150)
    with np.errstate(invalid="ignore", divide="ignore"):
        show(axes[1, 2], np.log10(maps["NII6583_FLUX"]/maps["HA6562_FLUX"]), r"log [N II]/H$\alpha$", cmap="magma", vmin=-1, vmax=0)
        show(axes[1, 3], maps["HA6562_FLUX"]/maps["HB4861_FLUX"], r"H$\alpha$/H$\beta$", cmap="cividis", vmin=2.5, vmax=6)
    fig.suptitle(f"{args.name}: MUSE line maps ({Path(args.cube).name})"); fig.savefig(out / f"{args.name}_linemaps_qa.png", dpi=110)
    print(f"[{args.name}] done in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
