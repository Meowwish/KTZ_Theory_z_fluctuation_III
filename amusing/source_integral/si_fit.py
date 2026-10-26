#!/usr/bin/env python3
"""si_fit.py JOB.json : one nested-sampling run (one model of one prior piece) with the source-integral likelihood
(si_model.build), dynesty checkpointing and resume; settings identical to the production mac_fit.py:
NestedSampler(loglike, identity, nd, nlive, rstate=default_rng(seed), bound="multi", sample="rwalk").run_nested(dlogz=0.05).
On completion writes <out>.npz atomically (samples, logwt, logz, logzerr, logl, ncall, niter, elapsed, versions)."""
import sys, os, json, time
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]/"ktzobs"))
import si_model
from scripts.ktz_exact_window_evidence import NestedModel
import dynesty

_CACHE = {}
SKIP = {"L0": {"mean", "cnp"}, "L1": {"cnp"}, "L1n": {"mean"}, "L2": set()}


def identity(u): return u


class LogL:
    def __init__(self, cfg, tables, name, eps_prior="manuscript", fast=None):
        self.cfg, self.tables, self.name, self.eps_prior, self.fast = cfg, tables, name, eps_prior, fast
        key = json.dumps([cfg, tables, fast], sort_keys=True)
        if key not in _CACHE:
            _CACHE[key] = si_model.build(cfg, tables, log=lambda m: None, fast_path=self.fast)
        m = _CACHE[key]; m.si_skip = SKIP[name.rstrip("q")]
        self.nm = NestedModel(m, name, eps_prior=eps_prior)
    def __call__(self, u): return self.nm.loglike(u)
    def __reduce__(self): return (LogL, (self.cfg, self.tables, self.name, self.eps_prior, self.fast))


if __name__ == "__main__":
    job = json.load(open(sys.argv[1])); out = job["out"]; ck = out+".ckpt"; t0 = time.time()
    if os.path.exists(out+".npz"):
        print("done already"); sys.exit(0)
    import fcntl   # one process per job (lock released automatically when the process ends or is killed)
    lockdir = os.environ.get("SI_LOCKDIR", os.path.dirname(out)); os.makedirs(lockdir, exist_ok=True)
    lockf = open(os.path.join(lockdir, os.path.basename(out)+".lock"), "a")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("job locked by another process", flush=True); sys.exit(0)
    ll = LogL(job["cfg"], job["tables"], job["model"], job.get("eps_prior", "manuscript"), job.get("fast")); nd = len(ll.nm.active)
    elapsed0 = json.load(open(out+".time.json"))["elapsed"] if os.path.exists(out+".time.json") else 0.0
    if os.path.exists(ck):
        sampler = dynesty.NestedSampler.restore(ck); resume = True
    else:
        sampler = dynesty.NestedSampler(ll, identity, nd, nlive=job["nlive"], rstate=np.random.default_rng(job["seed"]), bound="multi", sample="rwalk"); resume = False
    print(f"setup {time.time()-t0:.1f}s resume={resume} pid {os.getpid()}", flush=True)
    t1 = time.time()
    try:
        sampler.run_nested(dlogz=0.05, print_progress=False, checkpoint_file=ck, checkpoint_every=job.get("checkpoint_every", 15), resume=resume)
    finally:
        tmp = out+f".time.json.{os.getpid()}.partial"; json.dump(dict(elapsed=elapsed0+time.time()-t1), open(tmp, "w")); os.replace(tmp, out+".time.json")
    res = sampler.results
    d = dict(samples=res.samples, logwt=res.logwt, logz=res.logz, logzerr=res.logzerr, logl=res.logl, ncall=np.asarray(res.ncall), niter=np.asarray(res.niter),
             elapsed=np.asarray(elapsed0+time.time()-t1), eps_prior=np.asarray(job.get("eps_prior", "manuscript")), dynesty=np.asarray(dynesty.__version__),
             numpy=np.asarray(np.__version__), tables=np.asarray(job["tables"]), fast=np.asarray(str(job.get("fast"))), job=np.asarray(json.dumps(job)))
    tmp = out+f".{os.getpid()}.partial.npz"; np.savez(tmp, **d); os.replace(tmp, out+".npz")
    print(f"finished logZ {res.logz[-1]:.3f} in {time.time()-t0:.0f}s", flush=True)
