"""Write the flux-area-corrected morphology for an existing stage-1 output directory without re-running the stage
(2026-10-21).  Only the H-alpha SFR (which sets lambda0 / gamma_area) is changed; it is recomputed with exactly the
fixed stage functions (halpha_cell_flux_factor, halpha_sfr) from the stored valid-cell H-alpha (pixel map 'halpha_flux',
= the stage's corr['ha'] on the valid cells) and the WCS of the MAPS the stage read.  The unscaled value is recomputed
first and must reproduce the archived halpha_sfr_msun_per_yr (regression check).  Every other key is copied unchanged.
usage: apply_sfr_area_fix.py SRC_DIR GALAXY MAPS OUT_MORPHOLOGY_JSON"""
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(W, "..", "ktzobs")); sys.path.insert(0, HERE)
from astropy.io import fits; from astropy.wcs import WCS
import amusing_ktz_stage as st
src, gal, maps, dst = sys.argv[1:5]; g = gal.lower()
m = json.load(open(os.path.join(src, f"{g}_morphology.json"))); pm = np.load(os.path.join(src, f"{g}_pixel_map.npz"))
v = pm["valid_mask"]; hcell = pm["halpha_flux"]
wcs = WCS(fits.open(maps)["HA6562_FLUX"].header).celestial
kpa = m["distance_mpc"]*1e3*np.pi/180/3600
factor, spx, csky = st.halpha_cell_flux_factor(wcs, kpa, float(m["pixel_kpc"]), np.deg2rad(m["inclination_deg"]))
old = st.halpha_sfr(hcell, v, m["distance_mpc"], 1.0)
assert abs(old/m["halpha_sfr_msun_per_yr"]-1) < 1e-6, (old, m["halpha_sfr_msun_per_yr"])
new = dict(m); new.update(halpha_sfr_msun_per_yr=st.halpha_sfr(hcell, v, m["distance_mpc"], factor), halpha_flux_area_factor=factor,
                          native_spaxel_area_kpc2=spx, cell_sky_area_kpc2=csky, halpha_sfr_unscaled_pre20261021_msun_per_yr=m["halpha_sfr_msun_per_yr"],
                          flux_area_fix="2026-10-21: SFR from valid-cell H-alpha x (cell sky area / native spaxel area); apply_sfr_area_fix.py")
tmp = dst+".partial"; json.dump(new, open(tmp, "w"), indent=1); os.replace(tmp, dst)
print(f"{src}: SFR {m['halpha_sfr_msun_per_yr']:.6f} -> {new['halpha_sfr_msun_per_yr']:.6f} Msun/yr (factor {factor:.6f}; unscaled reproduced to {abs(old/m['halpha_sfr_msun_per_yr']-1):.1e})")
