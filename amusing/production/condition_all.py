"""condition_all.py CASES_JSON OUT_JSON : physical-prior conditioning (conditional_physical_v1) of split-piece box evidences
from the post-processed summaries: ln Z_box = logsumexp_i[ln f_i + ln Z_i], f_i = ln(hi/lo)/ln(range); ln Z = ln Z_box - ln F_M,
F_M = retained_prior_mass(inputs_from_metadata(morphology, range), model) with the morphology of the variant (new gamma)."""
import json, sys, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "ktzobs"))
from scripts.physical_prior import inputs_from_metadata, retained_prior_mass
cases = json.load(open(sys.argv[1])); out = {}
for name, c in cases.items():
    g = c["galaxy"].lower(); d = c.get("dir") or os.path.join(H, "outputs", c["galaxy"], "ktz", c["variant"]); pieces = c["pieces"]
    morph = json.load(open(c["morphology"])); lo, hi = pieces[0][0], pieces[-1][1]; inp = inputs_from_metadata(morph, (lo, hi))
    J = [json.load(open(os.path.join(d, f"{g}_paper_nested_{c['tagprefix']}_ell{a:g}-{b:g}.json"))) for a, b in pieces]
    f = np.array([np.log(b/a) for a, b in pieces])/np.log(hi/lo); res = {}
    for L in ("L0", "L1", "L1n", "L2"):
        lz = np.array([j[L]["logZ"] for j in J]); le = np.array([j[L]["logZ_err"] for j in J]); w = np.log(f)+lz; box = float(np.logaddexp.reduce(w)); pw = np.exp(w-box)
        F = retained_prior_mass(inp, L)
        res[L] = dict(logZ_box=box, F=F, logZ=box-np.log(F), logzerr=float(np.sqrt(np.sum((pw*le)**2))), weights=pw.tolist(), logZ_pieces=lz.tolist(),
                      chi2_min=float(min(j[L]["chi2_min"] for j in J)), chi2_pieces=[j[L]["chi2_min"] for j in J], best_piece=int(np.argmin([j[L]["chi2_min"] for j in J])))
    for L in res: res[L]["dlnZ"] = res[L]["logZ"]-res["L0"]["logZ"]
    res["_meta"] = dict(pieces=pieces, gamma_area=inp["gamma_area"], halpha_sfr=morph["halpha_sfr_msun_per_yr"], morphology=c["morphology"], variant=c["variant"])
    out[name] = res; print(name, {L: (round(res[L]["dlnZ"], 3), round(res[L]["chi2_min"], 2), round(res[L]["F"], 7)) for L in ("L0", "L1", "L1n", "L2")}, "w_L2", np.round(res["L2"]["weights"], 3))
json.dump(out, open(sys.argv[2], "w"), indent=1)
