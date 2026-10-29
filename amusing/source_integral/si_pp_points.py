#!/usr/bin/env python3
"""si_pp_points.py OUTDIR VARIANT GALAXY TAGPREFIX M1 N OUT.json MODEL [--seed S] [--min-weight 0.005]
Copy of si_pp_points.py (10-26, sha 81d24549...) generalised to the posterior of MODEL (L0 or L2): interval masses w_p are the
conditioned evidence weights of MODEL (evidence_<VARIANT>.json), equal-weight posterior draws <MODEL>_samples (si_post, 4000 per
interval) are sampled uniformly, stratified n_p = max(2, round(N w_p)) for w_p >= min-weight, weight w_p/n_p renormalised over
the included intervals (excluded mass recorded).  Components absent from MODEL are set to zero in the generator parameters:
L0 has no deterministic mean (beta = 0; the stored L0 sample column ln beta = 0 is an unused filler), no template modulation
(eta = 0) and no distinct-event excess (betaA = 0).  ell, eps, slope and A_dip are taken from the sample exactly as in
si_pp_points.py (A_dip = column 7 = A_dip/A_ref^{1/2}; random dipole coefficient c ~ N(0, A_dip^2) in si_gen_joint)."""
import sys, json, numpy as np
from scipy.special import expit
a = sys.argv; d, v, g, tp, m1, N, out, model = a[1], a[2], a[3], a[4], float(a[5]), int(a[6]), a[7], a[8]
seed = int(a[a.index("--seed")+1]) if "--seed" in a else 31; wmin = float(a[a.index("--min-weight")+1]) if "--min-weight" in a else 0.005
ev = json.load(open(f"{d}/evidence_{v}.json"))[v]; pieces = json.load(open(f"{d}/case_{v}.json"))[v]["pieces"]; w = np.asarray(ev[model]["weights"])
rng = np.random.default_rng(seed); pts = []; inc = w >= wmin; excl_mass = float(w[~inc].sum())
for p, ((lo, hi), wp) in enumerate(zip(pieces, w)):
    if wp < wmin: continue
    n_p = max(2, int(round(N*wp))); X = np.load(f"{d}/{g}_paper_nested_{tp}_ell{lo:g}-{hi:g}.npz", allow_pickle=True)[f"{model}_samples"]
    for j in rng.choice(len(X), n_p, replace=False):
        x = X[j]
        q = dict(ell=float(np.exp(x[0])), eps=float(expit(x[1])), beta=float(np.exp(x[2])), eta=float(np.exp(x[3])), slope=float(x[4]),
                 betaA=float(np.exp(x[5])), ell_src=float(np.exp(x[6])), A_dip=float(x[7]), m1=m1, amp="auto",
                 piece=[lo, hi], piece_weight=float(wp), w=float(wp/n_p/w[inc].sum()), idx=int(j), model=model)
        if model == "L0":
            assert q["eta"] == 0.0 and q["betaA"] == 0.0, (q["eta"], q["betaA"])
            q["beta"] = 0.0
        pts.append(q)
json.dump(dict(variant=v, model=model, pieces=pieces, piece_weights=w.tolist(), excluded_mass=excl_mass, seed=seed, n=len(pts), points=pts), open(out, "w"), indent=1)
print(v, model, "points", len(pts), "per piece", [sum(1 for q in pts if q["piece"] == list(pc)) for pc in pieces], "excluded mass", excl_mass)
