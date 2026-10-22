"""assemble_eb_cov.py NEW_STAGE_DIR ARCHIVED_EB_DIR GALAXY OUT_NPZ : EB covariance after the flux-area fix.
Sigma_field (the 'mock' block: stationary proxy fields with the new L2 curve as input correlation, x s_b s_b') is taken
from the re-run stage; Sigma_MC ('measurement', line-flux Monte Carlo; independent of the curve) and the jackknife
diagnostic are taken from the archived EB product, so that a re-run with the archived curve reproduces the archived
covariance exactly (regression: assemble with the *_regress stage output).  Shrinkage and floor as in the stage
(morphology covariance_shrinkage / systematic_floor):  Sigma = (1-s) raw + s diag(raw) + floor^2 I, raw = mock + measurement."""
import sys, json, numpy as np
new, old, g, out = sys.argv[1], sys.argv[2], sys.argv[3].lower(), sys.argv[4]
N = np.load(f"{new}/{g}_fit_covariance.npz"); O = np.load(f"{old}/{g}_fit_covariance.npz"); m = json.load(open(f"{old}/{g}_morphology.json"))
s, fl = float(m["covariance_shrinkage"]), float(m["systematic_floor"])
raw = N["mock"] + O["measurement"]; cov = (1-s)*raw + s*np.diag(np.diag(raw)) + fl**2*np.eye(len(raw))
chk_old = (1-s)*(O["mock"]+O["measurement"]) + s*np.diag(np.diag(O["mock"]+O["measurement"])) + fl**2*np.eye(len(raw))
assert np.allclose(chk_old, O["covariance"], rtol=0, atol=1e-15) and np.allclose(O["raw_covariance"], O["mock"]+O["measurement"], rtol=0, atol=1e-15)
np.savez_compressed(out, covariance=cov, raw_covariance=raw, jackknife=O["jackknife"], measurement=O["measurement"], mock=N["mock"])
print(out, "max|cov-archived|", float(np.max(np.abs(cov-O["covariance"]))), "rel", float(np.max(np.abs(cov-O["covariance"])/np.sqrt(np.outer(np.diag(O['covariance']),np.diag(O['covariance']))))))
