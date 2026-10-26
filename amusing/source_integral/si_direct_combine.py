#!/usr/bin/env python3
"""si_direct_combine.py CONFIG.json OUT.json PART.json... : sum si_direct --part outputs ('se' parts covering the age
nodes exactly once, C_NP parts covering m = 0..mmax exactly once) into the single-run output format; the transported
mean field is summed over the age parts before the projection and pair sums (quadratic)."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import si_tables as st
cfgd = json.load(open(sys.argv[1])); out, parts = sys.argv[2], [json.load(open(f)) for f in sys.argv[3:]]
sep = [p for p in parts if p["part"] == "se"]; cp = [p for p in parts if p["part"] == "cnp"]
ac = np.zeros(sep[0]["n_age"], int)
for p in sep:
    assert p["params"] == sep[0]["params"]; ac[p["a0"]:p["a1"]] += 1
assert np.all(ac == 1), ac
geom = st.Config(cfgd).geometry(); eta = sep[0]["params"]["eta"]
mf = sum(np.asarray(p["raw"]["mean_field"]) for p in sep); mn, md = geom.pair_sums(geom.project(mf[0]+eta*mf[1]))
se = [dict(sep[0], raw=dict(sep[0]["raw"], se=sum(np.asarray(p["raw"]["se"]) for p in sep).tolist(), mean_num=np.asarray(mn).tolist(), mean_den=np.asarray(md).tolist()))]
mmax = se[0]["mmax"]; cover = np.zeros(mmax+1, int)
for p in cp:
    assert p["params"] == se[0]["params"] and p["n_age"] == se[0]["n_age"]; cover[p["m0"]:min(p["m1"], mmax+1)] += 1
assert np.all(cover == 1), cover
r = se[0]["raw"]; cnt = np.asarray(r["counts"], float); sev = np.asarray(r["se"]); cn = sum(np.asarray(p["raw"]["cnp"]) for p in cp)
res = dict(params=se[0]["params"], n_age=se[0]["n_age"], mmax=mmax, parts=sys.argv[3:], p0=(sev[0]/cnt).tolist(), p1=(sev[1]/cnt).tolist(),
           mean=(np.array([r["mean_num"], r["mean_den"]])/cnt).tolist(), cnp=(cn/cnt).tolist(),
           seconds_total=sum(p["seconds_total"] for p in parts))
json.dump(res, open(out, "w")); print("wrote", out)
