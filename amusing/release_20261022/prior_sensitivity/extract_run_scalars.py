"""extract_run_scalars.py OUT_PREFIX JOBGLOB [JOBGLOB ...]
Scalar summary of every nested-sampling run named by the job JSONs (no sample arrays).  For each job: the run file is
job['out']+'.npz' (or job['reused_run'] for reused L0/L1n post-processing jobs); fields: run file, its SHA256, final
logz / logzerr, max(logl), niter, total ncall, eps_prior, seed, nlive, model, transport bounds (ell_min, ell_max), cuts,
the SFR of the morphology actually used (cfg.dir), and SHA256 of the input fit_covariance.npz and morphology.json.
Writes OUT_PREFIX.csv and OUT_PREFIX.json."""
import sys, os, json, glob, csv, hashlib
import numpy as np
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
out, globs = sys.argv[1], sys.argv[2:]; rows = []; covsha = {}
for jf in sorted(f for g in globs for f in glob.glob(g)):
    j = json.load(open(jf)); cfg = j["cfg"]
    run = j.get("reused_run") or (j["out"] + ".npz")
    d = np.load(run, allow_pickle=False); g = cfg["g"]
    cov = os.path.realpath(os.path.join(cfg["dir"], f"{g}_fit_covariance.npz")); mor = os.path.join(cfg["dir"], f"{g}_morphology.json")
    if cov not in covsha: covsha[cov] = sha(cov)
    rows.append(dict(job=os.path.basename(jf), run_file=os.path.basename(run), run_sha256=sha(run), reused=bool(j.get("reused_run")),
                     sens_variant=j.get("sens_variant", "baseline"), model=j["model"], galaxy=g, data_dir=os.path.basename(cfg["dir"]),
                     ell_min=cfg["ell_min"], ell_max=cfg["ell_max"], m1=cfg["m1"], rmin_fit=cfg["rmin_fit"], src_max=cfg["src_max"],
                     nlive=j["nlive"], seed=j["seed"], eps_prior=str(d["eps_prior"]), job_eps_prior=j.get("eps_prior"),
                     cuts=json.dumps(j.get("cuts")), sfr_msun_per_yr=json.load(open(mor))["halpha_sfr_msun_per_yr"],
                     logz=float(d["logz"][-1]), logzerr=float(d["logzerr"][-1]), max_logl=float(d["logl"].max()), chi2_min=float(-2*d["logl"].max()),
                     niter=int(d["niter"]), ncall_total=int(np.sum(d["ncall"])), n_samples=int(len(d["logl"])),
                     fit_covariance_file=cov, fit_covariance_sha256=covsha[cov], morphology_sha256=sha(mor),
                     dynesty=str(d["dynesty"]), numpy=str(d["numpy"])))
with open(out + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(rows, open(out + ".json", "w"), indent=1); print(len(rows), "runs ->", out)
