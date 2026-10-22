"""delta_summary.py : from recomputed.json, (a) max |dlnZ(variant) - dlnZ(baseline)| per covariance and model,
(b) the span max-min over all four settings (baseline + 3 variants), (c) pairwise differences that decide orderings."""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); s = json.load(open(os.path.join(H, "recomputed.json")))["summary"]
V = ("sfrx0.5", "sfrx2", "cutswide"); out = {}
for cov in ("null", "eb"):
    for L in ("L1", "L2", "L1n"):
        ch = [(abs(s[f"{cov}|{m}|{v}"][L]["change_vs_baseline"]), m, v) for m in ("S", "D16", "N1084") for v in V]
        sp = [(max(s[f"{cov}|{m}|{v}"][L]["dlnZ"] for v in ("baseline",)+V)-min(s[f"{cov}|{m}|{v}"][L]["dlnZ"] for v in ("baseline",)+V), m) for m in ("S", "D16", "N1084")]
        out[f"{cov}|{L}"] = dict(max_abs_change_vs_baseline=max(ch), max_span_all_settings=max(sp))
    for m in ("S", "D16", "N1084"):
        out[f"{cov}|{m}|pairs"] = {v: {p: round(s[f"{cov}|{m}|{v}"][p.split('-')[0]]["dlnZ"]-s[f"{cov}|{m}|{v}"][p.split('-')[1]]["dlnZ"], 3) for p in ("L2-L1", "L2-L1n", "L1-L1n")} for v in ("baseline",)+V}
json.dump(out, open(os.path.join(H, "delta_summary.json"), "w"), indent=1)
for k, v in out.items(): print(k, v)
