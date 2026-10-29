#!/usr/bin/env python3
"""si_post_subset.py (2026-10-26, source-integral likelihood; otherwise verbatim from production post_lean.py)
post_lean.py GALAXY VARIANT TAG M1 RMIN [RMAX] -- RUNFILE_L0 RUNFILE_L1 RUNFILE_L1n RUNFILE_L2 -- JOBJSON_L0 ... JOBJSON_L2
(2026-10-21 flux-area fix)  Post-processes stored Mac nested-sampling results (mac_fit.py npz) into the production summary
files <g>_paper_nested_<TAG>.{json,npz} under outputs/<GALAXY>/ktz/<VARIANT>/, with the summary block of
paper_single_pointing.py (posterior resampling 4000 draws with default_rng(seed 2), derived parameters, chi2 = -2 max lnL
for the manuscript eps prior, prediction percentiles, MAP prediction, median normalised components, shoulder/tail) copied
verbatim, evaluated with the lean remote model (mac_model.build with the compiled unified evaluator; documented equal to
the full model to 2e-12 in lnL).  The H65 stochastic libraries needed by paper_single_pointing.py --unified-response are
not archived; this script is validated by reproducing the archived production summaries of unchanged runs (see
validate_post_lean.py).  Each model's stored lnL is re-evaluated at 40 samples (must agree to 1e-4)."""
import sys, os, json, time, numpy as np
from pathlib import Path
from scipy.special import expit
HERE = Path(__file__).resolve().parent; MF = Path(os.environ.get("MACFIT", str(HERE)))
sys.path.insert(0, str(MF)); sys.path.insert(0, str(HERE.parents[1]/"ktzobs"))
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]/"ktzobs"))
import si_model
from scripts.ktz_exact_window_evidence import NestedModel
args = sys.argv[1:]; i1 = args.index("--"); head = args[:i1]; rest = args[i1+1:]; i2 = rest.index("--")
runs, jobs = rest[:i2], rest[i2+1:]
G, V, TAG, M1, RMIN = head[:5]; out = Path(os.environ["OUTROOT"])/G/"ktz"/V; out.mkdir(parents=True, exist_ok=True); g = G.lower()
NAMES = ["ell_star", "epsilon", "beta_src", "eta", "kappa", "t_star", "sigma", "A_src", "ell_src"]
res_all = {}; arrays = {}; cache = {}
MODELS = os.environ.get("SI_MODELS", "L0,L2").split(",")   # round 3: subset (e.g. L0,L2) for the sensitivity fits; default unchanged
for name, rf, jf in zip(MODELS, runs, jobs):
    d = dict(np.load(rf)); job = json.load(open(jf)); cfg = job["cfg"]; assert job["model"] == name+"q" and str(d["eps_prior"]) == "manuscript"
    key = json.dumps(cfg, sort_keys=True)
    key = json.dumps([cfg, job["tables"], job.get("fast")], sort_keys=True)
    if key not in cache: cache[key] = si_model.build(cfg, job["tables"], log=lambda m: None, fast_path=job.get("fast"))
    model = cache[key]; model.si_skip = set()
    diag = np.genfromtxt(Path(cfg["dir"])/f"{g}_diagnostics.csv", delimiter=",", names=True); r = diag["r_kpc"]; obs = diag["xi_centered"]
    err = np.sqrt(np.diag(np.load(Path(cfg["dir"])/f"{g}_fit_covariance.npz")["covariance"]))
    excl = (diag["r_kpc"] < float(cfg["rmin_fit"])) | (diag["r_kpc"] > float(cfg.get("rmax_fit", 1e9))); nb = len(r)
    nm = NestedModel(model, name+"q", eps_prior="manuscript"); nd = len(nm.active); t0 = time.time()
    assert d["samples"].shape[1] == nd
    idx = np.linspace(0, len(d["logl"])-1, 40).astype(int); ll = np.array([nm.loglike(d["samples"][i]) for i in idx]); st_ = d["logl"][idx]
    both = (~np.isfinite(ll) | (ll <= -1e299)) & (st_ <= -1e299); dev = float(np.max(np.abs(np.where(both, 0.0, ll-st_)))); assert dev < 1e-4, dev
    class _R: pass
    res = _R(); res.samples, res.logwt, res.logz, res.logzerr, res.logl = d["samples"], d["logwt"], d["logz"], d["logzerr"], d["logl"]
    # ---- verbatim from paper_single_pointing.py (2026-10-16 audit copy), lines 91-107 ----
    w = np.exp(res.logwt - res.logz[-1]); w /= w.sum(); rng = np.random.default_rng(2)
    X = nm.fill(res.samples[rng.choice(len(w), 4000, p=w)]); der = np.asarray(model.derived(X[:, :7])).T
    xmap = nm.fill(res.samples[np.argmax(res.logl)][None])[0]; chi2 = float(-2*res.logl.max())
    pred, _ = nm.predict_joint(X)
    terms = model.raw_terms(X); terms[:, nm.mask] = 0.0; comps = model._ratio(terms)
    pmap, _ = nm.predict_joint(xmap[None])
    summ = dict(logZ=float(res.logz[-1]), logZ_err=float(res.logzerr[-1]), n_params=nd, chi2_min=chi2, n_data=int(nb-excl.sum()), n_excluded=int(excl.sum()), m1_phase=float(M1), m1_amplitude=np.percentile(X[:, 7], [16, 50, 84]).tolist(), time_s=float(d["elapsed"]),
                chi2_median_curve=float(model.chi2(np.median(pred, 0)) if hasattr(model, "chi2") else np.nan),
                params={k: np.percentile(der[:, i], [16, 50, 84]).tolist() for i, k in enumerate(NAMES)}, inv_h_R=np.percentile(-X[:, 4], [16, 50, 84]).tolist(),
                map_params={k: float(v) for k, v in zip(NAMES, np.asarray(model.derived(xmap[None, :7])).ravel())}, map_inv_h_R=float(-xmap[4]),
                shoulder=float(np.median(pred[:, (r >= .6) & (r <= 2.5)].mean(1))), tail=float(np.median(pred[:, (r >= 3) & (r <= 4.5)].mean(1))),
                P_ell_src_gt_0p6=float(np.mean(der[:, 8] > 0.6)) if 6 in nm.active else None, P_ell_src_gt_ell_star=float(np.mean(der[:, 8] > der[:, 0])) if 6 in nm.active else None)
    arrays[f"{name}_pred_p16"], arrays[f"{name}_pred_p50"], arrays[f"{name}_pred_p84"] = np.percentile(pred, [16, 50, 84], 0)
    arrays[f"{name}_pred_map"] = pmap[0]; arrays[f"{name}_comp_p50"] = np.median(comps, 0); arrays[f"{name}_samples"] = X; arrays[f"{name}_derived"] = der; arrays[f"{name}_logl"] = np.asarray([0.0])
    # ---- end verbatim ----
    summ.update(eps_prior="manuscript", support="intrinsic", worker="si_fit.py (source-integral likelihood); si_post_subset.py", tables=job["tables"], fast_tables=job.get("fast"), source_completion="C1", ncall=int(np.sum(d["ncall"])), niter=int(d["niter"]),
                max_abs_dlnL_recheck_40pts=dev, run_file=os.path.basename(rf), run_sha256=__import__("hashlib").sha256(open(rf, "rb").read()).hexdigest(), post_time_s=time.time()-t0)
    res_all[name] = summ; print(f"{TAG} {name}: logZ {summ['logZ']:.3f} chi2 {chi2:.2f} ell* {np.round(summ['params']['ell_star'],3)} eta {np.round(summ['params']['eta'],3)} dev {dev:.1e}", flush=True)
npz = out/f"{g}_paper_nested_{TAG}.npz"; js = out/f"{g}_paper_nested_{TAG}.json"
tmp = str(npz)+".partial.npz"; np.savez_compressed(tmp, r=r, obs=obs, err=err, **arrays); os.replace(tmp, npz)
tmp = str(js)+".partial"; json.dump(res_all, open(tmp, "w"), indent=1); os.replace(tmp, js); print("wrote", js)
