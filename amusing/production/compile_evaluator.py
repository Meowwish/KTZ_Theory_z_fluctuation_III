"""compile_evaluator.py DIR G SRC_MAX OUT : intrinsic-support unified evaluator tables for one variant (all prior pieces)."""
import sys, json, hashlib, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import mac_model, ktz_unified_response as kur
d, g, smax, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
m = mac_model.build(dict(dir=d, g=g, m1=0.0, rmin_fit=0.0, src_max=smax, ell_min=0.08, ell_max=3.2), lean=True)
pg, ng = m.interp_all["pinterp"], m.interp_all["ninterp"]; lib = f"{d}/{g}_stochastic_response_H65.npz"
U = kur.UnifiedStochasticResponse(lib, pg.grid[0], pg.grid[1], ng.grid[3], pg.grid[2], m.beam_fwhm_kpc/2.354820045, support="intrinsic", ell_prior_max=3.2)
kur.save_compiled(U, out, provenance=dict(library=lib, library_md5=hashlib.md5(open(lib, "rb").read()).hexdigest(), beam_fwhm_kpc=m.beam_fwhm_kpc, src_max=smax))
print(out, U.rank, U.check, U.lnL, U.lnD)
