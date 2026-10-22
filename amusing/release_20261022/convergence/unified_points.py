"""Stage 1 of the unified-response convergence test (2026-10-16): choose test points for one case and write the
parameters needed by the independent brute-force pair calculation (direct_pairs.py).
Points: stored posterior samples (L1, L1n, L2 of every published-prior piece of the 2026-10-15 corrected runs),
inter-node stress points (slopes at midpoints of adjacent 65-node and 33-node Chebyshev-Lobatto nodes and of the
legacy 7-node grid), and boundary stress points (slope at and next to +-0.9, ell_obs/eps_obs at the domain edges,
ell_src at its prior bounds, eta at its positivity bound).
usage: unified_points.py CASE   (CASE in S, D16, N1084D16, N1084t05)"""
import sys, glob, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _setup import setup
CASES = {"S": ("NGC2466", "scal_mockcov_fid0_v2_pixbeam", 1, 0.45, 4.0), "D16": ("NGC2466", "d16_mockcov_fid0_v2_pixbeam", 9, 0.45, 4.0),
         "N1084D16": ("NGC1084", "d16_mockcov_v2_pixbeam", -87, 0.12, 2.5), "N1084t05": ("NGC1084", "d16_mockcov_t05_v2_pixbeam", -87, 0.12, 2.5)}
case = sys.argv[1]; G_, V, M1, RMIN, SRC = CASES[case]
G = setup(G_, V + "_unif65", M1, RMIN, SRC); m = G["model"]; out = G["out"]; g = G["g"]
cm = Path(str(out).replace("_unif65", "_cm65")); rng = np.random.default_rng(2026)
pts, lab = [], []
for f in sorted(glob.glob(str(cm/f"{g}_paper_nested_v3_ell*.npz"))):
    if "_L1L2" in f: continue
    z = np.load(f); piece = f.split("_ell")[-1][:-4]
    for L in ("L1", "L1n", "L2"):
        X = z[f"{L}_samples"]
        for i in rng.choice(len(X), 3, replace=False):
            pts.append(X[i]); lab.append(f"posterior:{L}:{piece}")
base = np.load(sorted(glob.glob(str(cm/f"{g}_paper_nested_v3_ell*.npz")))[0])["L2_samples"]
med = np.median(base, axis=0)
cl65 = -0.9*np.cos(np.pi*np.arange(65)/64); cl33 = cl65[::2]; leg = np.linspace(-0.9, 0.9, 7)
mids = np.r_[0.5*(cl65[1:]+cl65[:-1])[rng.choice(64, 6, replace=False)], 0.5*(cl33[1:]+cl33[:-1])[rng.choice(32, 3, replace=False)], 0.5*(leg[1:]+leg[:-1])[[0, 2, 5]]]
for s in mids:
    x = base[rng.integers(len(base))].copy(); x[4] = s; pts.append(x); lab.append("internode")
sig_b = m.beam_fwhm_kpc/2.354820045
ell_ax, eps_ax = m.interp_all["pinterp_legacy"].grid[0], m.interp_all["pinterp_legacy"].grid[1]
from scipy.special import logit
def from_obs(lo, eo):   # internal (ln ell*, logit eps) giving observed (ell_obs, eps_obs)
    ell = np.sqrt(np.exp(2*lo)-sig_b**2); a = np.sqrt((np.exp(lo+eo))**2-sig_b**2); return np.log(ell), logit(a/ell)
edge = []
for s in (-0.9, 0.9, -0.8999, 0.8999): edge.append(("slope_edge", dict(s=s)))
edge += [("ell_obs_max", dict(lo=ell_ax[-1]-1e-9)), ("ell_obs_min", dict(lo=max(ell_ax[0], np.log(sig_b)+0.05)+1e-9)),
         ("eps_obs_max", dict(eo=eps_ax[-1]-1e-9)), ("src_min", dict(src=np.log(0.12))), ("src_max", dict(src=np.log(SRC))), ("eta_max", dict(eta=np.log(m.eta_positive_limit*0.999)))]
for name, d in edge:
    x = med.copy(); co = m.kernel_coordinates(x[None])[0]
    if "s" in d: x[4] = d["s"]
    if "lo" in d or "eo" in d:
        lo = d.get("lo", co[0]); eo = d.get("eo", co[1])
        if "eo" in d: lo = min(co[0], np.log(1.5))
        try:
            l_, e_ = from_obs(lo, eo)
        except Exception: continue
        if not (np.isfinite(l_) and np.isfinite(e_)): continue
        x[0], x[1] = l_, e_
    if "src" in d: x[6] = d["src"]
    if "eta" in d: x[3] = d["eta"]
    pts.append(x); lab.append("edge:"+name)
X = np.array(pts); co = m.kernel_coordinates(X)
# unit amplitudes for components absent from a sample's model (component-level comparison); actual amplitudes kept otherwise
amp = lambda v: np.where(np.isfinite(v), np.exp(v), 1.0)
P = dict(X=X, label=np.array(lab), ell_obs=np.exp(co[:, 0]), eps_obs=np.exp(co[:, 1]), slope=X[:, 4], eta=amp(X[:, 3]), beta=amp(X[:, 2]),
         betaA=amp(X[:, 5]), ell_src=np.exp(X[:, 6]), r_ref=G["morph"]["r_ref_kpc"],
         edges=np.concatenate(([G["diag"]["r_kpc"][0]-0.5*(G["diag"]["r_kpc"][1]-G["diag"]["r_kpc"][0])], G["diag"]["r_kpc"]+0.5*(G["diag"]["r_kpc"][1]-G["diag"]["r_kpc"][0]))),
         annulus_width=0.4 if float(G["pm"]["pixel_size_kpc"]) >= 0.1 else 0.2, case=case)
inside = np.isfinite(m.raw_terms(X)).all(axis=(1, 2, 3)); P["inside_domain"] = inside
np.savez(Path(__file__).resolve().parent/f"unified_points_{case}.npz", **P)
print(case, len(X), "points;", dict(zip(*np.unique([l.split(":")[0] for l in lab], return_counts=True))), "inside domain", int(inside.sum()))
