"""Unit/area conservation test of the 2026-10-21 flux-area factor on synthetic per-spaxel line maps.
A TAN WCS with 0.2" spaxels carries (i) a uniform and (ii) a smooth exponential-disc per-spaxel flux.  The face-on
cell grid, sky mapping, PSF-matching smoothing and interpolation replicate amusing_ktz_stage.build_maps() (sample_line).
Compared: sum over cells x factor  vs  sum of native spaxels whose deprojected position falls in those cells,
for all cells and for interior cells only (cells whose 8 neighbours are also in the footprint), which separates the
finite-resolution/boundary difference from the missing area factor.  usage: python test_flux_area_synthetic.py"""
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(W, "..", "ktzobs")); sys.path.insert(0, os.path.join(W, "scripts"))
from astropy.wcs import WCS
from scripts.phangs_source_inputs import sample_line
import amusing_ktz_stage as st
out = {}
for case in [dict(name="NGC1084-like", D=19.7, ba=0.918, pa=209.0, pix=0.06, rmin=0.3, rmax=3.9, psf=0.715),
             dict(name="NGC2466-like", D=73.8, ba=0.95, pa=212.0, pix=0.12, rmin=1.0, rmax=10.0, psf=1.252)]:
    kpa = case["D"]*1e3*np.pi/180/3600; native = 0.2*kpa
    n = int(np.ceil(2*1.15*case["rmax"]/native)) | 1
    w = WCS(naxis=2); w.wcs.ctype = ["RA---TAN", "DEC--TAN"]; w.wcs.crval = [40.0, -10.0]; w.wcs.crpix = [n/2+0.5, n/2+0.5]
    w.wcs.cdelt = [-0.2/3600, 0.2/3600]
    yy, xx = np.mgrid[:n, :n]; ra, dec = w.pixel_to_world_values(xx, yy); ra0, dec0 = 40.0, -10.0
    incl = np.arccos(np.sqrt(max((case["ba"]**2-st.Q0**2)/(1-st.Q0**2), 1e-4))); pa = np.deg2rad(case["pa"])
    sx = (ra-ra0)*np.cos(np.deg2rad(dec0))*3600*kpa; sy = (dec-dec0)*3600*kpa
    Xn = sx*np.sin(pa)+sy*np.cos(pa); Yn = (-sx*np.cos(pa)+sy*np.sin(pa))/np.cos(incl); Rn = np.hypot(Xn, Yn)
    fields = {"uniform": np.full((n, n), 100.0, np.float32), "exp_disc": (1000.0*np.exp(-Rn/2.0)).astype(np.float32)}
    err = np.ones((n, n), np.float32)
    half = int(np.ceil(case["rmax"]/case["pix"])); axis = np.arange(-half, half+1)*case["pix"]; Y, X = np.meshgrid(axis, axis, indexing="ij"); R = np.hypot(X, Y)
    dxs = X*np.sin(pa) - Y*np.cos(incl)*np.cos(pa); dys = X*np.cos(pa) + Y*np.cos(incl)*np.sin(pa)
    ras = ra0 + dxs/(3600*kpa)/np.cos(np.deg2rad(dec0)); decs = dec0 + dys/(3600*kpa)
    px, py = w.world_to_pixel_values(ras, decs); coords = np.vstack([py.ravel(), px.ravel()])
    sig = np.sqrt(max(0.12**2-(case["psf"]*kpa)**2, 1e-6))/2.354820045/native
    valid = (R >= case["rmin"]) & (R <= case["rmax"])
    inner = valid.copy()
    for dy_ in (-1, 0, 1):
        for dx_ in (-1, 0, 1): inner &= np.roll(np.roll(valid, dy_, 0), dx_, 1)
    factor, spx, csky = st.halpha_cell_flux_factor(w, kpa, case["pix"], incl)
    ix = np.rint((Xn-X[0, 0])/case["pix"]).astype(int); iy = np.rint((Yn-Y[0, 0])/case["pix"]).astype(int)
    ok = (ix >= 0) & (ix < X.shape[1]) & (iy >= 0) & (iy < X.shape[0])
    res = dict(factor=factor, native_spaxel_kpc2=spx, cell_sky_kpc2=csky, analytic=case["pix"]**2*np.cos(incl)/native**2)
    for fname, f in fields.items():
        fc, _, _ = sample_line(f, err, coords, sig, X.shape)
        for lab, m in (("all_valid", valid), ("interior", inner)):
            sel = np.zeros((n, n), bool); sel[ok] = m[iy[ok], ix[ok]]
            A = float(np.sum(fc[m])); B = float(np.sum(f[sel]))
            res[f"{fname}_{lab}"] = dict(cells_sum_unscaled=A, cells_sum_times_factor=A*factor, native_sum_same_footprint=B,
                                         ratio_native_over_unscaled=B/A, ratio_native_over_corrected=B/(A*factor), n_native=int(sel.sum()), n_cells=int(m.sum()))
    out[case["name"]] = res
    print(case["name"], f"factor {factor:.4f} (analytic {res['analytic']:.4f})",
          {k: round(v["ratio_native_over_corrected"], 5) for k, v in res.items() if isinstance(v, dict)})
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json")+1], "w"), indent=1)
for name, res in out.items():
    assert abs(res["factor"]/res["analytic"]-1) < 1e-6
    for k, v in res.items():
        if isinstance(v, dict) and k.startswith("uniform"):
            assert abs(v["ratio_native_over_corrected"]-1) < (2e-3 if "interior" in k else 1e-2), (name, k, v)
print("PASS")
