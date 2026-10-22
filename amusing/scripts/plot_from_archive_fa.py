"""plot_from_archive.py -- pure-plotting reproduction of Figs 7/8 (ngc2466/ngc1084_single_pointing).

Runs the *load* and *figure* sections of scripts/paper_single_pointing.py verbatim (the source text between
"A = dict(np.load(npz))" and "# ---------------- LaTeX table rows") on archived final-fit products.
No model, response library, likelihood, sampler or fit is constructed or run.

Equivalent to the original generator call
  paper_single_pointing.py --name NAME --variant V --plot-only --eps-prior manuscript --evidence-prior box \
      --tag L2TAG --l0-tag L0TAG --rmin-fit RMIN --figname STEM ...
whose figure section only reads: the pixel map, morphology, diagnostics (r, xi_centered), the diagonal of the
fit covariance (plotting errors), the nested-fit npz/json of the L2-dominant interval, and the L0 arrays/summary of
L0's own dominant interval.

usage: python plot_from_archive.py --script scripts/paper_single_pointing.py --name NGC2466 --outdir figs [--dump X.json]
"""
import argparse, json, hashlib, types
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = {  # final intrinsic-support + manuscript-epsilon-prior products (variant suffix _unif65e)
    "NGC2466": dict(variant="scal_mockcov_fid0_v2_pixbeam_unif65e", tag="_v3_ell0.08-0.22", l0_tag="_v3_ell0.5-3.2", rmin=0.45, stem="ngc2466_single_pointing"),
    "NGC1084": dict(variant="d16_mockcov_v2_pixbeam_unif65e", tag="_v3_ell0.08-0.5", l0_tag="_v3_ell0.5-3.2", rmin=0.12, stem="ngc1084_single_pointing"),
}
HERE = Path(__file__).resolve().parent

def h(x):
    a = np.ascontiguousarray(np.asarray(x, dtype=float)); return hashlib.sha256(a.tobytes()).hexdigest()[:16]

ap = argparse.ArgumentParser(); ap.add_argument("--script", required=True); ap.add_argument("--name", required=True, choices=sorted(CASES))
ap.add_argument("--outdir", required=True); ap.add_argument("--inputs", default=str(HERE.parent / "release_20261022/figure_inputs")); ap.add_argument("--dump"); ap.add_argument("--pyver-note", default="")
cl = ap.parse_args(); c = CASES[cl.name]; g = cl.name.lower(); out = Path(cl.inputs) / cl.name
src = Path(cl.script).read_text(); i0 = src.index("A = dict(np.load(npz))"); i1 = src.index("# ---------------- LaTeX table rows")
outdir = Path(cl.outdir); outdir.mkdir(parents=True, exist_ok=True)
# --- the same input reads as paper_single_pointing.py (lines 27-28, 81) ---
pm = dict(np.load(out / f"{g}_pixel_map.npz")); morph = json.loads((out / f"{g}_morphology.json").read_text())
diag = np.genfromtxt(out / f"{g}_diagnostics.csv", delimiter=",", names=True)
r = diag["r_kpc"]; obs = diag["xi_centered"]; err = np.sqrt(np.diag(np.load(out / f"{g}_fit_covariance.npz")["covariance"]))
excl = (diag["r_kpc"] < c["rmin"]) | (diag["r_kpc"] > 1e9)
a = types.SimpleNamespace(name=cl.name, variant=c["variant"], l0_tag=c["l0_tag"], figname=c["stem"])
npz = out / f"{g}_paper_nested{c['tag']}.npz"; js = out / f"{g}_paper_nested{c['tag']}.json"
ns = dict(np=np, json=json, plt=plt, Path=Path, ROOT=outdir.parent if outdir.name == "figs" else None, out=out, g=g, a=a,
          npz=npz, js=js, pm=pm, morph=morph, r=r, obs=obs, err=err, excl=excl)
code = src[i0:i1]
# The shared source figure block now implements the accepted visible-point legend
# rule directly. No run-time rewriting of the scientific or plotting code is needed.
if ns["ROOT"] is None:  # save into --outdir instead of ROOT/figs
    code = code.replace('fig.savefig(ROOT / "figs" / f"{stem}.{p}", dpi=150)', 'fig.savefig(OUTDIR / f"{stem}.{p}", dpi=150)'); ns["OUTDIR"] = outdir
exec(compile(code, cl.script + " [load+figure sections]", "exec"), ns)
fig = ns["fig"]
if cl.dump:  # fingerprint of every plotted array (lines, bands, error bars, image, contours) and every text string
    rec = dict(arrays={}, texts={})
    for k, ax in enumerate(fig.axes):
        for j, ln in enumerate(ax.get_lines()): rec["arrays"][f"ax{k}_line{j}"] = h(ln.get_xydata())
        for j, col in enumerate(ax.collections):
            paths = col.get_paths(); rec["arrays"][f"ax{k}_coll{j}_{type(col).__name__}"] = h(np.concatenate([p.vertices for p in paths]) if paths else [])
            if hasattr(col, "get_segments"):
                try: rec["arrays"][f"ax{k}_coll{j}_segments"] = h(np.concatenate(col.get_segments()) if col.get_segments() else [])
                except Exception: pass
        for j, im in enumerate(ax.get_images()): rec["arrays"][f"ax{k}_image{j}"] = h(np.ma.filled(im.get_array(), np.nan))
        leg = ax.get_legend()
        rec["texts"][f"ax{k}"] = dict(title=ax.get_title(), xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(), legend=[t.get_text() for t in leg.get_texts()] if leg else [],
                                      xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()))
    rec["inputs_used"] = {"L0_pred_p16": h(ns["A"]["L0_pred_p16"]), "L0_pred_p50": h(ns["A"]["L0_pred_p50"]), "L0_pred_p84": h(ns["A"]["L0_pred_p84"]),
                          "L2_pred_p16": h(ns["A"]["L2_pred_p16"]), "L2_pred_p50": h(ns["A"]["L2_pred_p50"]), "L2_pred_p84": h(ns["A"]["L2_pred_p84"]),
                          "L1_pred_p50": h(ns["A"]["L1_pred_p50"]), "L1n_pred_p50": h(ns["A"]["L1n_pred_p50"]), "L2_comp_p50": h(ns["A"]["L2_comp_p50"]),
                          "r": h(r), "obs": h(obs), "err": h(err), "residual_dex": h(pm["residual_dex"]), "arm_source_modulation": h(pm["arm_source_modulation"]),
                          "L0_ell_star": ns["S"]["L0"]["params"]["ell_star"], "L2_ell_star": ns["S"]["L2"]["params"]["ell_star"]}
    Path(cl.dump).write_text(json.dumps(rec, indent=1))
print("wrote", [str(p) for p in sorted(outdir.glob(c["stem"] + ".*"))])
