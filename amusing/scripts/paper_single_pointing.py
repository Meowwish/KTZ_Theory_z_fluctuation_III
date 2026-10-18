"""Paper-III single-pointing demonstration (NGC 1084, D16, mock covariance): nested-sampling posteriors of the
L0 (KT18) / L1 / L1n / L2 hierarchy with full parameter summaries and posterior-predictive curves, plus the
journal figure figs/ngc1084_single_pointing.pdf and the LaTeX table rows.

    python scripts/paper_single_pointing.py [--name NGC1084 --variant d16_mockcov --nlive 400]
Writes outputs/<name>/ktz/<variant>/<g>_paper_nested.npz / .json and figs/ngc1084_single_pointing.{pdf,png}.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.special import expit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[0] / "ktzobs")); sys.path.insert(0, str(ROOT / "scripts"))
from scripts import ktz_exact_window as kew
from scripts.ktz_exact_window_evidence import NestedModel
from scripts.physical_prior import inputs_from_model, retained_prior_mass, condition_summary
import amusing_ktz_stage as st

ap = argparse.ArgumentParser(); ap.add_argument("--name", default="NGC1084"); ap.add_argument("--variant", default="d16_mockcov"); ap.add_argument("--nlive", type=int, default=400); ap.add_argument("--seed", type=int, default=2)
ap.add_argument("--evidence-prior", choices=("physical", "box"), help="default: physical for --eps-prior manuscript, box for legacy; physical conditions the joint base prior on all physical cuts, after sampling")
ap.add_argument("--models", default="L0,L1,L1n,L2"); ap.add_argument("--ell-max", type=float, default=3.2); ap.add_argument("--ell-min", type=float, default=0.08, help="lower bound of the ell_star grid/prior [kpc] (split-prior runs)"); ap.add_argument("--tag", default="", help="suffix for the output files (split-prior runs)"); ap.add_argument("--l0-tag", help="plot-only: take the L0 curves and summary from the run with this tag (its dominant prior piece)"); ap.add_argument("--m1", type=float, help="phase [deg] of an m=1 nuisance pattern cos(phi-phi0); its amplitude is marginalised in every model (models become L0q..L2q)"); ap.add_argument("--rmax-fit", type=float, default=1e9, help="exclude separation bins with r > rmax_fit from the likelihood"); ap.add_argument("--rmin-fit", type=float, default=0.0, help="exclude separation bins with r < rmin_fit from the likelihood (variance set to 1e4, no covariance)"); ap.add_argument("--src-max", type=float, default=2.5); ap.add_argument("--plot-only", action="store_true"); ap.add_argument("--cm-nodes", type=int, default=0, help="if > 0: evaluate the projected mean-field (C_M) responses from an exact Chebyshev-Lobatto library with this many nodes in the rate log-slope instead of the 7-node linear interpolation (added 2026-10-15)"); ap.add_argument("--figname", help="output figure stem (default <g>_single_pointing_<variant>)"); ap.add_argument("--eps-prior", default="legacy", choices=("legacy", "manuscript"), help="'manuscript': normalised p(eps) ~ 1/(1+eps) through the prior transform, no epsilon weight in the likelihood (2026-10-16 fix)"); ap.add_argument("--support", default="legacy", choices=("legacy", "intrinsic"), help="with --unified-response: prediction support; 'intrinsic' evaluates every point of the intrinsic ell_*/eps prior (fix of the legacy grid truncation, 2026-10-16)"); ap.add_argument("--unified-response", action="store_true", help="evaluate C_2p,0, C_2p,N and C_NP from the 65-node Chebyshev-Lobatto, cubic-Hermite stochastic response library <g>_stochastic_response_H65.npz with per-sample kernels (no slope or (ell, eps, ell_src) grid interpolation; added 2026-10-16)")
a = ap.parse_args(); g = a.name.lower(); out = ROOT / "outputs" / a.name / "ktz" / a.variant
pm = dict(np.load(out / f"{g}_pixel_map.npz")); morph = json.loads((out / f"{g}_morphology.json").read_text())
diag = np.genfromtxt(out / f"{g}_diagnostics.csv", delimiter=",", names=True); fit_cov = np.load(out / f"{g}_fit_covariance.npz")["covariance"].copy()
excl = (diag["r_kpc"] < a.rmin_fit) | (diag["r_kpc"] > a.rmax_fit)
if excl.any():   # sub-beam bins removed from the likelihood: huge variance, no covariance -> zero weight, model-independent constant in ln Z
    fit_cov[excl, :] = 0.0; fit_cov[:, excl] = 0.0; fit_cov[np.ix_(excl, excl)] = np.diag(np.full(int(excl.sum()), 1e4)); print(f"excluding {int(excl.sum())} bins with r < {a.rmin_fit} kpc from the fit")
resp = dict(np.load(out / f"{g}_window_response_all.npz"))
zero = {k: (np.zeros_like(v) if k in ("response", "mean_response", "delta_response", "ordered_pair_counts") else v) for k, v in resp.items()}
nb = len(diag["r_kpc"])
split = dict(D=np.zeros(10), cbin=np.zeros(nb, int), pairs_same=np.ones(nb), pairs_cross=np.ones(nb), coarse_edges=np.arange(0, 3.01, 0.3))
ELL = st.ELL_GRID if (a.ell_max == 3.2 and a.ell_min == 0.08) else np.geomspace(a.ell_min, a.ell_max, max(7, int(round(11*np.log(a.ell_max/a.ell_min)/np.log(40)))))
SRC = st.SRC_GRID if a.src_max == 2.5 else np.geomspace(0.005, a.src_max, max(17, int(round(17*np.log(a.src_max/0.005)/np.log(500)))))
grids = dict(ell_grid=ELL, epsilon_grid=st.EPS_GRID, source_length_grid=SRC)
nuisance_max = 0.0
if a.m1 is not None:   # deterministic unit-amplitude m=1 field cos(phi - phi0) through the same projector (as in amusing_ktz_stage.py)
    X_, Y_ = pm["x_grid_kpc"], pm["y_grid_kpc"]; valid_ = pm["valid_mask"]; R_ = np.hypot(X_, Y_); PHI_ = np.arctan2(Y_, X_)
    f1 = kew.detrend(np.where(valid_, np.cos(PHI_ - np.deg2rad(a.m1)), 0.0), valid_, R_, width=(0.4 if float(pm["pixel_size_kpc"]) >= 0.1 else 0.2))
    ps1 = kew.PairStatistic(valid_, pixel=float(pm["pixel_size_kpc"]), edges=np.concatenate(([diag["r_kpc"][0]-0.5*(diag["r_kpc"][1]-diag["r_kpc"][0])], diag["r_kpc"]+0.5*(diag["r_kpc"][1]-diag["r_kpc"][0])))); n1, e1, c1 = ps1.sums(f1, valid_)
    resp = dict(resp); resp["delta_response"] = np.asarray([np.divide(n1, c1, out=np.zeros(nb), where=c1 > 0), np.divide(e1, c1, out=np.zeros(nb), where=c1 > 0)]); nuisance_max = 3.0
    zero = {k: (np.zeros_like(v) if k in ("response", "mean_response", "delta_response", "ordered_pair_counts") else v) for k, v in resp.items()}
if a.cm_nodes > 0:   # exact C_M responses: Chebyshev-Lobatto library, verified first against the stored response nodes
    _dr = diag["r_kpc"][1]-diag["r_kpc"][0]; _edges = np.concatenate(([diag["r_kpc"][0]-0.5*_dr], diag["r_kpc"]+0.5*_dr))
    _aw = 0.4 if float(pm["pixel_size_kpc"]) >= 0.1 else 0.2
    _chk = kew.exact_mean_field_response(pm, morph["r_ref_kpc"], resp["rate_log_slope_grid"], _edges, _aw)
    _ref = np.asarray(resp["mean_response"]); _err = np.nanmax(np.abs(_chk-_ref))/np.nanmax(np.abs(_ref))
    assert _err < 1e-10, f"exact mean-field response does not reproduce the stored response nodes (max rel {_err:.2e})"
    lib = kew.mean_field_response_library(pm, morph["r_ref_kpc"], _edges, _aw, st.SLOPE_BOUND, a.cm_nodes)
    np.savez_compressed(out / f"{g}_mean_response_cheb{a.cm_nodes}.npz", **lib, check_vs_stored_nodes_max_rel=_err)
    resp = dict(resp); resp.update(lib)
    zero = {k: (np.zeros_like(v) if k in ("response", "mean_response", "mean_response_cheb", "delta_response", "ordered_pair_counts") else v) for k, v in resp.items()}
    print(f"C_M responses: exact Chebyshev-Lobatto library, {a.cm_nodes} nodes on |slope| <= {st.SLOPE_BOUND}; stored-node check max rel {_err:.1e}", flush=True)
model = kew.SplitKTZ(morph, diag, fit_cov, resp, resp, zero, split, fit_cov, grids, use_D=False, sigma_q_max=nuisance_max, src_length_bounds=(0.12, None), eta_prior="flat",
                     rate_log_slope_bound=st.SLOPE_BOUND, beam_fwhm_kpc=morph.get("effective_fwhm_kpc", st.TARGET_FWHM_KPC))
if a.unified_response:   # unified converged stochastic responses (2026-10-16); library provenance and consistency checked first
    import hashlib, ktz_unified_response as kur
    from scipy.special import exp1 as _E1
    _L = dict(np.load(out / f"{g}_stochastic_response_H65.npz"))
    _dr = diag["r_kpc"][1]-diag["r_kpc"][0]; _edges = np.concatenate(([diag["r_kpc"][0]-0.5*_dr], diag["r_kpc"]+0.5*_dr))
    assert str(_L["pixel_map_md5"]) == hashlib.md5(open(out / f"{g}_pixel_map.npz", "rb").read()).hexdigest(), "library built from a different pixel map"
    assert np.allclose(_L["edges"], _edges, rtol=0, atol=1e-12) and float(_L["annulus_width"]) == (0.4 if float(pm["pixel_size_kpc"]) >= 0.1 else 0.2)
    assert np.array_equal(_L["ordered_pair_counts"], resp["ordered_pair_counts"]) and str(_L["basis"]) == "hermite"
    # shared slope nodes (-0.9, 0, +0.9): library and stored 7-node response must give the same contraction with smooth
    # test kernels up to the legacy hat-basis discretisation (stored response: linear basis on 10/60 pc knots)
    _kr, _krs = _L["kernel_radius"], resp["kernel_radius"]; _nk = len(_kr); _maxe = 0.0
    for _ell, _eps in ((1.0, 0.5), (2.0, 0.3)):
        _kh = np.r_[-np.log(_eps), 0.5*(_E1(_kr[1:]**2/(4*_ell**2))-_E1(_kr[1:]**2/(4*(_eps*_ell)**2))), 0.0, -(np.exp(-_kr[1:]**2/(4*_ell**2))-np.exp(-_kr[1:]**2/(4*(_eps*_ell)**2)))/_kr[1:]]
        _ks = np.r_[-np.log(_eps), 0.5*(_E1(_krs[1:]**2/(4*_ell**2))-_E1(_krs[1:]**2/(4*(_eps*_ell)**2)))]
        for _s_i, _s in enumerate(resp["rate_log_slope_grid"]):
            _j = int(np.argmin(np.abs(_L["rate_log_slope_grid"]-_s)))
            if abs(_L["rate_log_slope_grid"][_j]-_s) < 1e-12:
                _a = np.nan_to_num(_L["response"][_j, :2]) @ _kh; _b = np.nan_to_num(resp["response"][_s_i, :2]) @ _ks
                _maxe = max(_maxe, float(np.max(np.abs(_a-_b))/np.max(np.abs(_b))))
    assert _maxe < 2e-3, f"library inconsistent with the stored response at the shared nodes ({_maxe:.1e})"
    print(f"stochastic library: shared-node consistency with the stored response {_maxe:.1e} (legacy hat-basis discretisation)", flush=True)
    UNI = kur.install(model, _L, log=print, support=a.support)
r = diag["r_kpc"]; obs = diag["xi_centered"]; err = np.sqrt(np.diag(np.load(out / f"{g}_fit_covariance.npz")["covariance"]))   # plotting: unmasked errors
NAMES = ["ell_star", "epsilon", "beta_src", "eta", "kappa", "t_star", "sigma", "A_src", "ell_src"]
sfx = a.tag
physical_evidence = (a.evidence_prior or ("physical" if a.eps_prior == "manuscript" else "box")) == "physical"
if physical_evidence and a.eps_prior != "manuscript":
    raise ValueError("Conditional physical evidence requires the manuscript epsilon prior")
if physical_evidence:
    sfx += "_physical"
    if a.l0_tag:
        raise ValueError("Do not compare a dominant-piece L0 evidence with a global physical prior; use the combined split summary")
npz = out / f"{g}_paper_nested{sfx}.npz"; js = out / f"{g}_paper_nested{sfx}.json"

if not a.plot_only:
    import dynesty
    res_all = {}; arrays = {}
    for name in a.models.split(","):
        nm = NestedModel(model, name + ("q" if a.m1 is not None else ""), eps_prior=a.eps_prior); nd = len(nm.active); t0 = time.time()
        sampler = dynesty.NestedSampler(nm.loglike, lambda u: u, nd, nlive=a.nlive, rstate=np.random.default_rng(a.seed), bound="multi", sample="rwalk")
        sampler.run_nested(dlogz=0.05, print_progress=False); res = sampler.results
        w = np.exp(res.logwt - res.logz[-1]); w /= w.sum(); rng = np.random.default_rng(a.seed)
        X = nm.fill(res.samples[rng.choice(len(w), 4000, p=w)]); der = np.asarray(model.derived(X[:, :7])).T
        xmap = nm.fill(res.samples[np.argmax(res.logl)][None])[0]; chi2 = float(-2*(res.logl.max() - np.log(2) + np.log1p(expit(xmap[1])))) if a.eps_prior == "legacy" else float(-2*res.logl.max())
        pred, _ = nm.predict_joint(X)
        terms = model.raw_terms(X); terms[:, nm.mask] = 0.0; comps = model._ratio(terms)  # (n, 5, nb) normalised components
        pmap, _ = nm.predict_joint(xmap[None])
        summ = dict(logZ=float(res.logz[-1]), logZ_err=float(res.logzerr[-1]), n_params=nd, chi2_min=chi2, n_data=int(nb-excl.sum()), n_excluded=int(excl.sum()), m1_phase=a.m1, m1_amplitude=(np.percentile(X[:, 7], [16, 50, 84]).tolist() if a.m1 is not None else None), time_s=time.time()-t0,
                    chi2_median_curve=float(model.chi2(np.median(pred, 0)) if hasattr(model, "chi2") else np.nan),
                    params={k: np.percentile(der[:, i], [16, 50, 84]).tolist() for i, k in enumerate(NAMES)}, inv_h_R=np.percentile(-X[:, 4], [16, 50, 84]).tolist(),
                    map_params={k: float(v) for k, v in zip(NAMES, np.asarray(model.derived(xmap[None, :7])).ravel())}, map_inv_h_R=float(-xmap[4]),
                    shoulder=float(np.median(pred[:, (r >= .6) & (r <= 2.5)].mean(1))), tail=float(np.median(pred[:, (r >= 3) & (r <= 4.5)].mean(1))),
                    P_ell_src_gt_0p6=float(np.mean(der[:, 8] > 0.6)) if 6 in nm.active else None, P_ell_src_gt_ell_star=float(np.mean(der[:, 8] > der[:, 0])) if 6 in nm.active else None)
        if physical_evidence:
            mass = retained_prior_mass(inputs_from_model(model), name)
            summ = condition_summary(summ, mass)
        res_all[name] = summ
        arrays[f"{name}_pred_p16"], arrays[f"{name}_pred_p50"], arrays[f"{name}_pred_p84"] = np.percentile(pred, [16, 50, 84], 0)
        arrays[f"{name}_pred_map"] = pmap[0]; arrays[f"{name}_comp_p50"] = np.median(comps, 0); arrays[f"{name}_samples"] = X; arrays[f"{name}_derived"] = der; arrays[f"{name}_logl"] = np.asarray([0.0])
        print(f"{name}: logZ {summ['logZ']:.2f} +- {summ['logZ_err']:.2f}, chi2_min {chi2:.2f}, chi2(median curve) {summ['chi2_median_curve']:.2f}, ell* {summ['params']['ell_star']}, "
              f"ell_src {summ['params']['ell_src']}, eta {summ['params']['eta']}, A_src {summ['params']['A_src']}  [{summ['time_s']:.0f} s]", flush=True)
    np.savez_compressed(npz, r=r, obs=obs, err=err, **arrays); json.dump(res_all, js.open("w"), indent=1)

A = dict(np.load(npz)); S = json.loads(js.read_text())
if a.l0_tag:   # L0 from its own dominant prior piece
    A0 = dict(np.load(out / f"{g}_paper_nested{a.l0_tag}.npz")); S0 = json.loads((out / f"{g}_paper_nested{a.l0_tag}.json").read_text())
    for k in list(A0): 
        if k.startswith("L0_"): A[k] = A0[k]
    S["L0"] = S0["L0"]
# ---------------- figure ----------------
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.2, "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42})
C_KT18, C_KTZ, C_NP, C_M, C_2P = "#1f77b4", "#7b3294", "#9467bd", "#e6550d", "#1f77b4"
fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.3), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.0, 1.25, 1.25]))
ax = axes[0]; X, Y, valid = pm["x_grid_kpc"], pm["y_grid_kpc"], pm["valid_mask"]; res_map = np.where(valid, pm["residual_dex"], np.nan)
dx = float(pm["pixel_size_kpc"]); ext = [X.min()-dx/2, X.max()+dx/2, Y.min()-dx/2, Y.max()+dx/2]
im = ax.imshow(res_map, origin="lower", extent=ext, cmap="RdBu_r", vmin=-0.15, vmax=0.15, interpolation="nearest")
mod = np.where(pm["template_support"], pm["arm_source_modulation"], np.nan)
ax.contour(X, Y, np.nan_to_num(mod, nan=-9), levels=[0.3, 0.8] if morph["analysis_rmax_kpc"] < 6 else [0.5, 1.5], colors="k", linewidths=[0.6, 0.9], linestyles=["-", "-"])
for rr in (morph["analysis_rmin_kpc"], morph["analysis_rmax_kpc"]): ax.add_patch(plt.Circle((0, 0), rr, fill=False, color="0.3", lw=0.7, ls="--"))
ax.set(xlabel="$x$ [kpc]", ylabel="$y$ [kpc]", aspect="equal", xlim=(-1.08*morph["analysis_rmax_kpc"], 1.08*morph["analysis_rmax_kpc"]), ylim=(-1.08*morph["analysis_rmax_kpc"], 1.08*morph["analysis_rmax_kpc"]), title=f"{a.name}: abundance residual ({'D16' if morph['calibration']=='d16' else 'S'})")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(r"$\delta\,[12+\log({\rm O/H})]$ [dex]")
ax = axes[1]
ax.errorbar(r[~excl], obs[~excl], err[~excl], fmt="o", ms=2.6, color="k", lw=0.7, label="observed", zorder=5); ax.errorbar(r[excl], obs[excl], err[excl], fmt="o", ms=2.6, mfc="w", color="k", lw=0.7, label="excluded (sub-beam)" if excl.any() else None, zorder=5)
ax.fill_between(r, A["L0_pred_p16"], A["L0_pred_p84"], color=C_KT18, alpha=0.2, lw=0)
ax.plot(r, A["L0_pred_p50"], color=C_KT18, lw=1.5, label=f"L0: projected $C_{{{{\\rm 2p}},0}}$, $\\ell_\\ast={S['L0']['params']['ell_star'][1]:.2f}$ kpc")
ax.fill_between(r, A["L2_pred_p16"], A["L2_pred_p84"], color=C_KTZ, alpha=0.25, lw=0)
ax.plot(r, A["L2_pred_p50"], color=C_KTZ, lw=1.8, label=f"L2: complete, $\\ell_\\ast={S['L2']['params']['ell_star'][1]:.2f}$ kpc")
ax.plot(r, A["L1n_pred_p50"], color=C_NP, lw=1.2, ls="--", label="L1n: $C_{{\\rm 2p},0}+C_{\\rm NP}$")
ax.plot(r, A["L1_pred_p50"], color=C_M, lw=1.2, ls=":", label="L1: $C_{{\\rm 2p},0}+C_{{\\rm 2p},N}+C_{\\rm M}$")
ax.axhline(0, color="0.5", lw=0.6); ax.set(xlabel="$r$ [kpc]", ylabel=r"$\xi_{3,\rm pp}(r)$", xlim=(0, r.max()+0.1), ylim=(-0.17, 0.45), title="finite-window statistic, nested models")
ax.legend(frameon=False, loc="upper right", fontsize=6.8)
ax = axes[2]; cm = A["L2_comp_p50"]
for i, (lab, col, ls) in enumerate([(r"$C_{{\rm 2p},0}$ (same event, uniform)", C_2P, "-"), (r"$C_{{\rm 2p},N}$ (same event, arm-modulated)", "#2ca02c", "-"), (r"$C_{\rm M}$ (unremoved mean field)", C_M, "-"), (r"$C_{\rm NP}$ (isotropic distinct-event term)", C_NP, "-")]):
    ax.plot(r, cm[i], color=col, lw=1.4, ls=ls, label=lab)
ax.plot(r, A["L2_pred_p50"], color=C_KTZ, lw=1.8, label="sum (L2)"); ax.errorbar(r[~excl], obs[~excl], err[~excl], fmt="o", ms=2, color="k", lw=0.5, alpha=0.6); ax.errorbar(r[excl], obs[excl], err[excl], fmt="o", ms=2, mfc="w", color="k", lw=0.5, alpha=0.6)
ax.axhline(0, color="0.5", lw=0.6); ax.set(xlabel="$r$ [kpc]", ylabel="contribution to $\\xi_{3,\\rm pp}$", xlim=(0, r.max()+0.1), ylim=(-0.17, 0.45), title="complete KTZ (L2): components")
ax.legend(frameon=False, loc="upper right", fontsize=6.8)
stem = a.figname or f"{g}_single_pointing_{a.variant}"
for p in ("pdf", "png"): fig.savefig(ROOT / "figs" / f"{stem}.{p}", dpi=150)
# ---------------- LaTeX table rows ----------------
def fmt(p, nd=2): return f"${p[1]:.{nd}f}^{{+{p[2]-p[1]:.{nd}f}}}_{{-{p[1]-p[0]:.{nd}f}}}$"
lines = []
for name, label in (("L0", r"L0: $C_{{\rm 2p},0}$ (\citetalias{KT18})"), ("L1", r"L1: $+C_{{\rm 2p},N}+C_{\rm M}$"), ("L1n", r"L1n: $+C_{\rm NP}$"), ("L2", r"L2: complete")):
    s = S[name]; p = s["params"]
    lines.append(f"{label} & {s['n_params']} & {fmt(p['ell_star'])} & {fmt(p['eta']) if name in ('L1', 'L2') else '--'} & {fmt(p['ell_src']) if name in ('L1n', 'L2') else '--'} & {s['chi2_min']:.1f} & "
                 f"${s['logZ']-S['L0']['logZ']:+.1f}$ \\\\")
(out / f"{g}_paper_table_rows{sfx}.tex").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("params", "map_params")} for k, v in S.items()}, indent=1))
print(json.dumps({k: v["params"] for k, v in S.items()}, indent=1))
