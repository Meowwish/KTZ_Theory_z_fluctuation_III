#!/usr/bin/env python3
"""make_jobs.py ROOT_INPUTS ROOT_TABLES ROOT_RUNS OUT_DIR VARIANT... : job JSONs for si_fit.py replicating the production
fits (settings from TITANIC/fluxarea_fix_20261021_074830/macfit/jobs_fa, jobs_post, jobs_eb_fa: nlive 400, seed 2,
eps_prior manuscript, dipole phase m1, r_min/r_max fit cuts, ell_src upper bound, split intrinsic ell_* pieces)."""
import sys, json, os
INP, TAB, RUNS, OUT = sys.argv[1:5]
V = {  # name: (input dir, galaxy, m1, rmin_fit, rmax_fit, src_max, pieces, tag, table set)
    "S":          ("S_pixbeam_fa", "ngc2466", 1.0, 0.45, None, 4.0, [(0.08, .22), (.22, .35), (.35, .5), (.5, 3.2)], "v3", "N2466S"),
    "S_r5":       ("S_pixbeam_fa", "ngc2466", 1.0, 0.45, 5.0, 4.0, [(0.08, .22), (.22, .35), (.35, .5), (.5, 3.2)], "v3r5", "N2466S"),
    "D16":        ("D16_pixbeam_fa", "ngc2466", 9.0, 0.45, None, 4.0, [(0.08, .22), (.22, .35), (.35, .5), (.5, 3.2)], "v3", "N2466D16"),
    "N1084D16":   ("N1084D16_pixbeam_fa", "ngc1084", -87.0, 0.12, None, 2.5, [(0.08, .5), (.5, 3.2)], "v3", "N1084D16"),
    "N1084t05":   ("N1084t05_pixbeam_fa", "ngc1084", -87.0, 0.12, None, 2.5, [(0.08, .5), (.5, 3.2)], "v3", "N1084t05"),
    "N1084S":     ("N1084S_pixbeam_fa", "ngc1084", -71.0, 0.12, None, 2.5, [(0.08, .5), (.5, 3.2)], "v3", "N1084S"),
    "S_EB":       ("S_ppe_fa", "ngc2466", 1.0, 0.45, None, 4.0, [(0.08, .22), (.22, .35), (.35, .5), (.5, 3.2)], "v3", "N2466S"),
    "S_r5_EB":    ("S_ppe_fa", "ngc2466", 1.0, 0.45, 5.0, 4.0, [(0.08, .22), (.22, .35), (.35, .5), (.5, 3.2)], "v3r5", "N2466S"),
    "D16_EB":     ("D16_ppe_fa", "ngc2466", 9.0, 0.45, None, 4.0, [(0.08, .22), (.22, .35), (.35, .5), (.5, 3.2)], "v3", "N2466D16"),
    "N1084D16_EB": ("N1084D16_ppe_fa", "ngc1084", -87.0, 0.12, None, 2.5, [(0.08, .5), (.5, 3.2)], "v3", "N1084D16"),
    # EB covariances rebuilt (si_ebmock.py, 2026-10-26) from the new source-integral L2 median curves:
    "S_EB2":      ("S_ppe_si1026", "ngc2466", 1.0, 0.45, None, 4.0, [(0.08, .22), (.22, .35), (.35, .5), (.5, 3.2)], "v3", "N2466S"),
    "S_r5_EB2":   ("S_ppe_si1026", "ngc2466", 1.0, 0.45, 5.0, 4.0, [(0.08, .22), (.22, .35), (.35, .5), (.5, 3.2)], "v3r5", "N2466S"),
    "D16_EB2":    ("D16_ppe_si1026", "ngc2466", 9.0, 0.45, None, 4.0, [(0.08, .22), (.22, .35), (.35, .5), (.5, 3.2)], "v3", "N2466D16"),
    "N1084D16_EB2": ("N1084D16_ppe_si1026", "ngc1084", -87.0, 0.12, None, 2.5, [(0.08, .5), (.5, 3.2)], "v3", "N1084D16"),
}
os.makedirs(OUT, exist_ok=True); n = 0
for name in sys.argv[5:]:
    d, g, m1, rmin, rmax, smax, pieces, tag, tab = V[name]
    for lo, hi in pieces:
        for model in ("L0q", "L1q", "L1nq", "L2q"):
            cfg = dict(dir=f"{INP}/{d}", g=g, m1=m1, rmin_fit=rmin, src_max=smax, ell_min=lo, ell_max=hi)
            if rmax is not None: cfg["rmax_fit"] = rmax
            fast = f"{TAB}/{tab}_fast_{d}.npz"
            job = dict(out=f"{RUNS}/{name}__{tag}_ell{lo:g}-{hi:g}__{model}", cfg=cfg, tables=f"{TAB}/{tab}_tables.npz", fast=fast, model=model,
                       nlive=400, seed=2, checkpoint_every=15, eps_prior="manuscript", variant=name, tag=f"{tag}_ell{lo:g}-{hi:g}",
                       priority={"L0q": 0, "L1q": 1, "L1nq": 1, "L2q": 2}[model])
            json.dump(job, open(f"{OUT}/{name}__{tag}_ell{lo:g}-{hi:g}__{model}.json", "w"), indent=1); n += 1
print(n, "jobs")
