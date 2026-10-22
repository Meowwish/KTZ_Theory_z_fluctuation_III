#!/usr/bin/env python3
"""mac_fit.py JOB.json : one nested-sampling run (one model of one prior piece) with dynesty checkpointing, so it can be
advanced in slices of a bounded wall time and resumed after the process is killed.  Settings identical to
paper_single_pointing.py: NestedSampler(nm.loglike, identity, nd, nlive, rstate=default_rng(seed), bound="multi",
sample="rwalk").run_nested(dlogz=0.05).  On completion writes <out>.npz (samples, logwt, logz, logzerr, logl, ncall, niter)
atomically; the container post-processes it with the unmodified paper_single_pointing.py summary code."""
import sys, os, json, time, pickle
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mac_model
from scripts.ktz_exact_window_evidence import NestedModel
import dynesty

_CACHE = {}
def identity(u): return u

class LogL:
    """Picklable log-likelihood: pickles only its configuration; the model is rebuilt (once per process) on unpickling."""
    def __init__(self, cfg, name, eps_prior="legacy"):
        self.cfg, self.name, self.eps_prior = cfg, name, eps_prior; key = json.dumps(cfg, sort_keys=True)
        if key not in _CACHE: _CACHE[key] = mac_model.build(cfg, log=lambda m: None)
        self.nm = NestedModel(_CACHE[key], name, eps_prior=eps_prior)
    def __call__(self, u): return self.nm.loglike(u)
    def __reduce__(self): return (LogL, (self.cfg, self.name, self.eps_prior))

if __name__ == "__main__":
    job = json.load(open(sys.argv[1])); budget = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
    out = job["out"]; ck = out + ".ckpt"; t0 = time.time()
    if os.path.exists(out + ".npz"): print("done already"); sys.exit(0)
    ll = LogL(job["cfg"], job["model"], job.get("eps_prior", "legacy")); nd = len(ll.nm.active)
    elapsed0 = 0.0
    if os.path.exists(ck):
        sampler = dynesty.NestedSampler.restore(ck); resume = True
        elapsed0 = json.load(open(out + ".time.json"))["elapsed"] if os.path.exists(out + ".time.json") else 0.0
    else:
        sampler = dynesty.NestedSampler(ll, identity, nd, nlive=job["nlive"], rstate=np.random.default_rng(job["seed"]), bound="multi", sample="rwalk"); resume = False
    print(f"setup {time.time()-t0:.1f}s resume={resume}", flush=True)
    # record wall time used so far (excluding setup) before each checkpoint interval: approximate, for the summary only
    t1 = time.time()
    try:
        sampler.run_nested(dlogz=0.05, print_progress=False, checkpoint_file=ck, checkpoint_every=job.get("checkpoint_every", 20), resume=resume)
    finally:
        tmp = out + ".time.json.partial"; json.dump(dict(elapsed=elapsed0 + time.time()-t1), open(tmp, "w")); os.replace(tmp, out + ".time.json")
    res = sampler.results
    d = dict(samples=res.samples, logwt=res.logwt, logz=res.logz, logzerr=res.logzerr, logl=res.logl, ncall=np.asarray(res.ncall), niter=np.asarray(res.niter),
             elapsed=np.asarray(elapsed0 + time.time()-t1), eps_prior=np.asarray(job.get("eps_prior", "legacy")), dynesty=np.asarray(dynesty.__version__), numpy=np.asarray(np.__version__))
    np.savez(out + ".partial.npz", **d); os.replace(out + ".partial.npz", out + ".npz"); print(f"finished logZ {res.logz[-1]:.3f} in {time.time()-t0:.0f}s", flush=True)
