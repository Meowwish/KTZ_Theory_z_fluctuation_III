#!/usr/bin/env python3
"""si_plot_fig78.py -- Figs 7/8 (ngc2466/ngc1084_single_pointing) from the source-integral posteriors (2026-10-26, review fix).

Differs from amusing/scripts/plot_from_archive_fa.py (which exec'd the figure section of paper_single_pointing.py) in ONE
respect, requested in review: every model's representative curve is taken from that model's OWN highest-posterior-mass
transport interval (piece weights of the conditioned evidence file), not from the L2 interval.  Each nested-model curve is a
single conditional median prediction; its legend gives the interval and the conditional median ell_*.  The L0 and L2 bands
(16-84 per cent conditional intervals) and the L2 component panel are kept exactly as before; no band or posterior curve
bundle is added.  All other drawing code (layout, colours, limits, map panel, contours, error bars) is copied verbatim from
paper_single_pointing.py's figure section.

usage: si_plot_fig78.py --name NGC2466|NGC1084 --inputs DIR --evidence EVIDENCE.json --variant KEY --outdir DIR [--dump X.json]
DIR/<NAME>/ must hold the pixel map, morphology, diagnostics, fit covariance and the <g>_paper_nested_<tag>_ell<lo>-<hi>
npz/json of every interval used.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = {"NGC2466": dict(tag="v3", rmin=0.45, stem="ngc2466_single_pointing"),
         "NGC1084": dict(tag="v3", rmin=0.12, stem="ngc1084_single_pointing")}


def h(x):
    a = np.ascontiguousarray(np.asarray(x, dtype=float)); return hashlib.sha256(a.tobytes()).hexdigest()[:16]


ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True, choices=sorted(CASES)); ap.add_argument("--inputs", required=True)
ap.add_argument("--evidence", required=True); ap.add_argument("--variant", required=True); ap.add_argument("--outdir", required=True); ap.add_argument("--dump")
cl = ap.parse_args(); c = CASES[cl.name]; g = cl.name.lower(); out = Path(cl.inputs)/cl.name; outdir = Path(cl.outdir); outdir.mkdir(parents=True, exist_ok=True)
pm = dict(np.load(out/f"{g}_pixel_map.npz")); morph = json.loads((out/f"{g}_morphology.json").read_text())
diag = np.genfromtxt(out/f"{g}_diagnostics.csv", delimiter=",", names=True)
r = diag["r_kpc"]; obs = diag["xi_centered"]; err = np.sqrt(np.diag(np.load(out/f"{g}_fit_covariance.npz")["covariance"]))
excl = (diag["r_kpc"] < c["rmin"]) | (diag["r_kpc"] > 1e9)
ev = json.load(open(cl.evidence))[cl.variant]; pieces = json.load(open(Path(cl.evidence).with_name(f"case_{cl.variant}.json")))[cl.variant]["pieces"]
dom, A, S = {}, {}, {}
for m in ("L0", "L1", "L1n", "L2"):
    i = int(np.argmax(ev[m]["weights"])); lo, hi = pieces[i]; dom[m] = (lo, hi, float(ev[m]["weights"][i]))
    stem = f"{g}_paper_nested_{c['tag']}_ell{lo:g}-{hi:g}"
    z = dict(np.load(out/f"{stem}.npz", allow_pickle=True)); j = json.loads((out/f"{stem}.json").read_text())
    for k in z:
        if k.startswith(m+"_"): A[k] = z[k]
    S[m] = j[m]
ell = {m: S[m]["params"]["ell_star"][1] for m in S}
lab_iv = lambda m: f"[{dom[m][0]:g},{dom[m][1]:g}]"   # dominant interval of ell_* [kpc]; ell_* = conditional median
# ---------------- figure (verbatim from paper_single_pointing.py except the four curve labels/sources) ----------------
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.2, "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42})
C_KT18, C_KTZ, C_NP, C_M, C_2P = "#1f77b4", "#7b3294", "#9467bd", "#e6550d", "#1f77b4"
PLOT_YLIM = (-0.17, 0.45)
excluded_visible = bool(np.any(excl & (obs >= PLOT_YLIM[0]) & (obs <= PLOT_YLIM[1])))
fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.3), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.0, 1.25, 1.25]))
ax = axes[0]; X, Y, valid = pm["x_grid_kpc"], pm["y_grid_kpc"], pm["valid_mask"]; res_map = np.where(valid, pm["residual_dex"], np.nan)
dx = float(pm["pixel_size_kpc"]); ext = [X.min()-dx/2, X.max()+dx/2, Y.min()-dx/2, Y.max()+dx/2]
im = ax.imshow(res_map, origin="lower", extent=ext, cmap="RdBu_r", vmin=-0.15, vmax=0.15, interpolation="nearest")
mod = np.where(pm["template_support"], pm["arm_source_modulation"], np.nan)
ax.contour(X, Y, np.nan_to_num(mod, nan=-9), levels=[0.3, 0.8] if morph["analysis_rmax_kpc"] < 6 else [0.5, 1.5], colors="k", linewidths=[0.6, 0.9], linestyles=["-", "-"])
for rr in (morph["analysis_rmin_kpc"], morph["analysis_rmax_kpc"]): ax.add_patch(plt.Circle((0, 0), rr, fill=False, color="0.3", lw=0.7, ls="--"))
ax.set(xlabel="$x$ [kpc]", ylabel="$y$ [kpc]", aspect="equal", xlim=(-1.08*morph["analysis_rmax_kpc"], 1.08*morph["analysis_rmax_kpc"]), ylim=(-1.08*morph["analysis_rmax_kpc"], 1.08*morph["analysis_rmax_kpc"]), title=f"{cl.name}: abundance residual ({'D16' if morph['calibration']=='d16' else 'S'})")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(r"$\delta\,[12+\log({\rm O/H})]$ [dex]")
ax = axes[1]
ax.errorbar(r[~excl], obs[~excl], err[~excl], fmt="o", ms=2.6, color="k", lw=0.7, label="observed", zorder=5); ax.errorbar(r[excl], obs[excl], err[excl], fmt="o", ms=2.6, mfc="w", color="k", lw=0.7, label="excluded (sub-beam)" if excluded_visible else None, zorder=5)
ax.fill_between(r, A["L0_pred_p16"], A["L0_pred_p84"], color=C_KT18, alpha=0.2, lw=0)
ax.plot(r, A["L0_pred_p50"], color=C_KT18, lw=1.5, label=f"L0 {lab_iv('L0')}: $\\ell_\\ast={ell['L0']:.2f}$ kpc")
ax.fill_between(r, A["L2_pred_p16"], A["L2_pred_p84"], color=C_KTZ, alpha=0.25, lw=0)
ax.plot(r, A["L2_pred_p50"], color=C_KTZ, lw=1.8, label=f"L2 {lab_iv('L2')}: $\\ell_\\ast={ell['L2']:.2f}$ kpc")
ax.plot(r, A["L1n_pred_p50"], color=C_NP, lw=1.2, ls="--", label=f"L1n {lab_iv('L1n')}: $\\ell_\\ast={ell['L1n']:.2f}$ kpc")
ax.plot(r, A["L1_pred_p50"], color=C_M, lw=1.2, ls=":", label=f"L1 {lab_iv('L1')}: $\\ell_\\ast={ell['L1']:.2f}$ kpc")
ax.axhline(0, color="0.5", lw=0.6); ax.set(xlabel="$r$ [kpc]", ylabel=r"$\xi_{3,\rm pp}(r)$", xlim=(0, r.max()+0.1), ylim=PLOT_YLIM, title="finite-window statistic, nested models")
ax.legend(frameon=False, loc="upper right", fontsize=7.2)
ax = axes[2]; cm = A["L2_comp_p50"]
for i, (lab, col, ls) in enumerate([(r"$C_{{\rm 2p},0}$ (same event, unmodulated)", C_2P, "-"), (r"$C_{{\rm 2p},N}$ (same event, H$\alpha$-modulated)", "#2ca02c", "-"), (r"$C_{\rm M}$ (unremoved mean field)", C_M, "-"), (r"$C_{\rm NP}$ (isotropic distinct-event term)", C_NP, "-")]):
    ax.plot(r, cm[i], color=col, lw=1.4, ls=ls, label=lab)
ax.plot(r, A["L2_pred_p50"], color=C_KTZ, lw=1.8, label="L2 total"); ax.errorbar(r[~excl], obs[~excl], err[~excl], fmt="o", ms=2, color="k", lw=0.5, alpha=0.6); ax.errorbar(r[excl], obs[excl], err[excl], fmt="o", ms=2, mfc="w", color="k", lw=0.5, alpha=0.6)
ax.axhline(0, color="0.5", lw=0.6); ax.set(xlabel="$r$ [kpc]", ylabel="contribution to $\\xi_{3,\\rm pp}$", xlim=(0, r.max()+0.1), ylim=PLOT_YLIM, title="complete KTZ (L2): components")
ax.legend(frameon=False, loc="upper right", fontsize=7.2)
for p in ("pdf", "png"): fig.savefig(outdir/f"{c['stem']}.{p}", dpi=150)
if cl.dump:
    rec = dict(dominant_intervals={m: dict(interval=[dom[m][0], dom[m][1]], posterior_weight=dom[m][2], ell_star_conditional=S[m]["params"]["ell_star"]) for m in dom},
               arrays={}, texts={}, inputs_used={})
    for k, a_ in enumerate(fig.axes):
        for j, ln in enumerate(a_.get_lines()): rec["arrays"][f"ax{k}_line{j}"] = h(ln.get_xydata())
        for j, col in enumerate(a_.collections):
            paths = col.get_paths(); rec["arrays"][f"ax{k}_coll{j}_{type(col).__name__}"] = h(np.concatenate([p.vertices for p in paths]) if paths else [])
        for j, im_ in enumerate(a_.get_images()): rec["arrays"][f"ax{k}_image{j}"] = h(np.ma.filled(im_.get_array(), np.nan))
        leg = a_.get_legend()
        rec["texts"][f"ax{k}"] = dict(title=a_.get_title(), legend=[t.get_text() for t in leg.get_texts()] if leg else [], xlim=list(a_.get_xlim()), ylim=list(a_.get_ylim()))
    for k in ("L0_pred_p16", "L0_pred_p50", "L0_pred_p84", "L2_pred_p16", "L2_pred_p50", "L2_pred_p84", "L1_pred_p50", "L1n_pred_p50", "L2_comp_p50"):
        rec["inputs_used"][k] = h(A[k])
    rec["inputs_used"].update(r=h(r), obs=h(obs), err=h(err), residual_dex=h(pm["residual_dex"]), arm_source_modulation=h(pm["arm_source_modulation"]))
    Path(cl.dump).write_text(json.dumps(rec, indent=1))
print("wrote", [str(p) for p in sorted(outdir.glob(c["stem"]+".*"))], {m: (dom[m], round(ell[m], 3)) for m in dom})
