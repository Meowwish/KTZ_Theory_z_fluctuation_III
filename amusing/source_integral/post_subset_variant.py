#!/usr/bin/env python3
"""post_subset_variant.py JOBDIR RUNDIR OUTROOT VARIANT_NAME OUT_VARIANT_DIRNAME : post-process every piece of one variant with
si_post.py (production summary code) and condition the split-piece evidences with the production condition_all.py.
Writes OUTROOT/<GALAXY>/ktz/<OUT_VARIANT_DIRNAME>/<g>_paper_nested_<tag>.{json,npz} and evidence JSON."""
import sys, os, json, glob, subprocess
J, RD, OUTROOT, VN, OVD = sys.argv[1:6]
HERE = os.path.dirname(os.path.abspath(__file__))
MQ = [m+"q" for m in os.environ.get("SI_MODELS", "L0,L2").split(",")]   # round 3: model subset; default unchanged
jobs = [json.load(open(f)) for f in sorted(glob.glob(f"{J}/{VN}__*.json")) if json.load(open(f))["model"] in MQ]
tags = sorted({j["tag"] for j in jobs}, key=lambda t: float(t.split("ell")[1].split("-")[0]))
gal = {"ngc1084": "NGC1084", "ngc2466": "NGC2466"}[jobs[0]["cfg"]["g"]]
pieces = []
for tag in tags:
    jj = {j["model"]: j for j in jobs if j["tag"] == tag}
    runs = [jj[m]["out"]+".npz" for m in MQ]
    if not all(os.path.exists(r) for r in runs):
        print("missing runs for", tag); sys.exit(1)
    jf = [f"{J}/{VN}__{tag}__{m}.json" for m in MQ]
    c = jj["L0q"]["cfg"]; head = [gal, OVD, tag, str(c["m1"]), str(c["rmin_fit"])] + ([str(c["rmax_fit"])] if "rmax_fit" in c else [])
    env = dict(os.environ, OUTROOT=OUTROOT)
    lo, hi = tag.split("ell")[1].split("-")
    done_npz = f"{OUTROOT}/{gal}/ktz/{OVD}/{jobs[0]['cfg']['g']}_paper_nested_{tag}.npz"
    if os.path.exists(done_npz) and os.path.exists(done_npz[:-4]+".json") and not os.environ.get("FORCE"):
        print("piece done:", done_npz); pieces.append([float(lo), float(hi)]); continue   # resumable (outputs are atomic)
    r = subprocess.run([sys.executable, f"{HERE}/si_post_subset.py", *head, "--", *runs, "--", *jf], env=env, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:]); assert r.returncode == 0
    lo, hi = tag.split("ell")[1].split("-"); pieces.append([float(lo), float(hi)])
case = {VN: dict(galaxy=gal, variant=OVD, dir=f"{OUTROOT}/{gal}/ktz/{OVD}", tagprefix=tags[0].split("_ell")[0], pieces=pieces,
                 morphology=f"{jobs[0]['cfg']['dir']}/{jobs[0]['cfg']['g']}_morphology.json")}
cf = f"{OUTROOT}/{gal}/ktz/{OVD}/case_{VN}.json"; json.dump(case, open(cf, "w"), indent=1)
env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..", "..", "ktzobs"))
r = subprocess.run([sys.executable, f"{HERE}/si_condition_subset.py", cf, f"{OUTROOT}/{gal}/ktz/{OVD}/evidence_{VN}.json"], capture_output=True, text=True, env=env)
print(r.stdout, r.stderr[-1500:]); assert r.returncode == 0
