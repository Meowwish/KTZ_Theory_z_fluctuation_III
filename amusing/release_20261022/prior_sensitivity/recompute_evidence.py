"""recompute_evidence.py : independent recomputation of the conditioned evidences from the scalar tables only.
Per model: ln Z_box = logsumexp_i[ln f_i + logz_i], f_i = ln(hi_i/lo_i)/ln(3.2/0.08); ln Z = ln Z_box - ln F_M,
F_M = physical_prior_cuts.retained_prior_mass(inputs_from_metadata(morphology of the run), model, C=cuts of the run).
Variant L1/L2 come from variants_*.csv; L0/L1n of every variant are the baseline runs (they do not depend on SFR or cuts).
Prints dlnZ = lnZ(M) - lnZ(L0) per map, variant and model, the change relative to baseline, and writes recomputed.json."""
import csv, json, sys, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "code"))
from physical_prior import inputs_from_metadata
import physical_prior_cuts as pc
def load(f): return list(csv.DictReader(open(os.path.join(H, f))))
MAPS = {"null": {"S": "S_pixbeam", "D16": "D16_pixbeam", "N1084": "N1084D16_pixbeam"}, "eb": {"S": "S_ppe", "D16": "D16_ppe", "N1084": "N1084D16_ppe"}}
DEF = (0.05, 1.5, 0.05, 3.0); out = {}
for cov in ("null", "eb"):
    base, var = load(f"baseline_{cov}.csv"), load(f"variants_{cov}.csv")
    for mp, key in MAPS[cov].items():
        def runs(rows, model, dpre):
            return [r for r in rows if r["model"] == model + "q" and r["data_dir"].startswith(dpre)]
        for v in ("baseline", "sfrx0.5", "sfrx2", "cutswide"):
            res = {}
            for L in ("L0", "L1", "L1n", "L2"):
                if v == "baseline" or L in ("L0", "L1n"):
                    rr = [r for r in runs(base, L, key + "_fa") if r["data_dir"] == key + "_fa"]
                else:
                    rr = [r for r in runs(var, L, key) if r["sens_variant"] == v]
                rr.sort(key=lambda r: float(r["ell_min"])); lo, hi = float(rr[0]["ell_min"]), float(rr[-1]["ell_max"])
                assert (lo, hi) == (0.08, 3.2) and all(abs(float(a["ell_max"]) - float(b["ell_min"])) < 1e-12 for a, b in zip(rr[:-1], rr[1:]))
                f = np.array([np.log(float(r["ell_max"]) / float(r["ell_min"])) for r in rr]) / np.log(hi / lo)
                box = float(np.logaddexp.reduce(np.log(f) + np.array([float(r["logz"]) for r in rr])))
                cuts = json.loads(rr[0]["cuts"]); C = tuple(cuts) if cuts else DEF
                morph = json.load(open(os.path.join(H, "morphology", f"{rr[0]['data_dir']}__{rr[0]['galaxy']}_morphology.json")))
                F = pc.retained_prior_mass(inputs_from_metadata(morph, (lo, hi)), L, C=C)
                res[L] = dict(lnZ=box - np.log(F), lnZ_box=box, F=F, chi2_min=min(float(r["chi2_min"]) for r in rr), n_pieces=len(rr))
            for L in res: res[L]["dlnZ"] = res[L]["lnZ"] - res["L0"]["lnZ"]
            out[f"{cov}|{mp}|{v}"] = res
summary = {}
for k, r in out.items():
    cov, mp, v = k.split("|"); b = out[f"{cov}|{mp}|baseline"]
    summary[k] = {L: dict(dlnZ=round(r[L]["dlnZ"], 4), change_vs_baseline=round(r[L]["dlnZ"] - b[L]["dlnZ"], 4)) for L in ("L1", "L1n", "L2")}
json.dump(dict(recomputed=out, summary=summary), open(os.path.join(H, "recomputed.json"), "w"), indent=1)
for k, s in summary.items():
    print(f"{k:22s}", "  ".join(f"{L}: {s[L]['dlnZ']:+7.3f} ({s[L]['change_vs_baseline']:+6.3f})" for L in ("L1", "L1n", "L2")))
