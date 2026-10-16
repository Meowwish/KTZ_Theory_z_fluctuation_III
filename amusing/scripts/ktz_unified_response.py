"""Unified converged evaluation of the projected stochastic KTZ responses (2026-10-16).

Replaces, for C_2p,0, C_2p,N and C_NP, the two interpolation layers of ExactWindowKTZ._interpolators
  (i)  linear interpolation over 7 rate-log-slope nodes, and
  (ii) multilinear interpolation of kernel contractions on (ln ell_obs, ln eps_obs[, ln ell_src]) grids,
with an evaluation that is converged in every discretisation:

* slope: barycentric polynomial interpolation through a 65-node Chebyshev-Lobatto library of exact window responses
  (response_library.py; identical algebra to build_window_response at every node);
* kernel radius: the responses are tabulated on a cubic-Hermite basis in the pair separation (values and derivatives
  at 10/20 pc knots), so a kernel enters through K(r_k) and K'(r_k) (error < 1e-5 against the brute-force pair sums);
* kernels: evaluated for each sample's own (ell_obs, eps_obs, ell_src), without any (ell, eps, ell_src) grid.
  The KT18 kernel is the difference of a one-parameter family, K = G_ell - G_{eps ell}, G_L(r) = E1(r^2/4L^2)/2
  (G_L(0) = ln L), and the C_NP kernel is a positive mixture of Gaussians exp(-r^2/2D) (cnp_fast.cnp_nodes, age
  integral converged to ~1e-11).  Contractions of the library with G_L and with the Gaussians are therefore smooth
  1-D functions of ln L and ln D; they are tabulated on Chebyshev-Lobatto nodes in ln L / ln D (n chosen so the
  table error is < 1e-9 of the largest response, checked at install against direct contractions at random points),
  and compressed over the slope index by an SVD (truncation 1e-13).
The prediction domain of the legacy model is kept exactly: the legacy grid interpolators return NaN (likelihood
-inf) whenever ln ell_obs, ln eps_obs, the slope or ln ell_src fall outside their grids; the new callables return
NaN under the identical comparisons, so the effective prior support is unchanged.
C_M (exact 65-node mean-field library of 2026-10-15), the m=1 nuisance and the cross-pointing (zero) responses are
untouched.
"""
import json
import numpy as np
from scipy.special import exp1
from cnp_fast import cnp_nodes


def cl_nodes(n, lo, hi):
    return 0.5*(lo+hi) - 0.5*(hi-lo)*np.cos(np.pi*np.arange(n)/(n-1))


def barycentric_weights(nodes, s):
    """Interpolation weights (len(s), n) for Chebyshev-Lobatto nodes (second-kind barycentric formula)."""
    n = len(nodes); w = (-1.0)**np.arange(n); w[0] *= 0.5; w[-1] *= 0.5
    s = np.atleast_1d(np.asarray(s, float)); diff = s[:, None]-nodes[None, :]
    hit = diff == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        k = w[None, :]/diff; out = k/k.sum(axis=1, keepdims=True)
    for i in np.nonzero(hit.any(axis=1))[0]:
        out[i] = 0.0; out[i, np.nonzero(hit[i])[0][0]] = 1.0
    return out


def kt18_family(kr, L):
    """[G_L(kr), G_L'(kr)] for each L: shape (len(L), 2 nk); G_L(0) := ln L, G_L'(0) = 0."""
    L = np.atleast_1d(L); nk = len(kr); out = np.zeros((len(L), 2*nk))
    out[:, 0] = np.log(L)
    z = kr[None, 1:]**2/(4*L[:, None]**2)
    out[:, 1:nk] = 0.5*exp1(z); out[:, nk+1:] = -np.exp(-z)/kr[None, 1:]
    return out


def gauss_family(kr, D):
    """[exp(-r^2/2D), d/dr] for each D: shape (len(D), 2 nk)."""
    D = np.atleast_1d(D); e = np.exp(-kr[None, :]**2/(2*D[:, None]))
    return np.concatenate([e, -(kr[None, :]/D[:, None])*e], axis=1)


class UnifiedStochasticResponse:
    def __init__(self, library, ell_axis, eps_axis, src_axis, slope_axis, sigma_beam, n_L=96, n_D=160, svd_tol=1e-13,
                 table_tol=1e-9, cnp_n=24, node_stride=1, support="legacy", ell_prior_max=3.2, log=print):
        L = dict(np.load(library)) if isinstance(library, str) else dict(library)
        assert str(L.get("basis", "hat")) == "hermite", "unified evaluator expects a cubic-Hermite library"
        R = np.nan_to_num(np.asarray(L["response"], float))[::node_stride]          # (ns, 3, 2, nb, 2nk)
        self.nodes = np.asarray(L["rate_log_slope_grid"], float)[::node_stride]; self.kr = kr = np.asarray(L["kernel_radius"], float)
        self.ns, _, _, self.nb, _ = R.shape; self.cnp_n = cnp_n
        self.ax = dict(ell=np.asarray(ell_axis), eps=np.asarray(eps_axis), src=np.asarray(src_axis), slope=np.asarray(slope_axis))
        # 1-D table ranges covering the whole prediction domain.
        # support="legacy": prediction domain = the legacy (ell_obs, eps_obs, slope, ell_src) grid box (reproduces the published runs,
        #   including their unintended truncation of the intrinsic ell_* / eps prior);
        # support="intrinsic" (2026-10-16 fix): every point allowed by the intrinsic prior is evaluated; tables cover
        #   ell_obs = sqrt(ell_*^2 + sigma_eff^2) for ell_* <= ell_prior_max and all eps in (0, 1].
        assert support in ("legacy", "intrinsic"); self.support = support
        sb2 = sigma_beam**2; self.sigma_beam = sigma_beam
        lmax = self.ax["ell"][-1] if support == "legacy" else 0.5*np.log(ell_prior_max**2 + sb2)
        self.lnL = (np.log(np.sqrt(sb2))-1e-6, lmax+1e-6)
        Dmin = np.exp(2*self.ax["src"][0]) + 2*sb2; Dmax = np.exp(2*self.ax["src"][-1]) + 2*np.exp(2*lmax)
        self.lnD = (np.log(Dmin)-1e-6, np.log(Dmax)+1e-6)
        self.xL, self.xD = cl_nodes(n_L, *self.lnL), cl_nodes(n_D, *self.lnD)
        VL = kt18_family(kr, np.exp(self.xL)); VD = gauss_family(kr, np.exp(self.xD))
        self.tab, self.check = {}, {}
        rng = np.random.default_rng(5)
        for c, V, x, rng_x, fam in ((0, VL, self.xL, self.lnL, kt18_family), (1, VL, self.xL, self.lnL, kt18_family),
                                    (2, VD, self.xD, self.lnD, gauss_family)):
            T = (R[:, c].reshape(self.ns*2*self.nb, -1) @ V.T).reshape(self.ns, 2*self.nb, len(x))
            T = np.ascontiguousarray(T.transpose(0, 2, 1)).reshape(self.ns, len(x)*2*self.nb)                # (ns, nx*2nb)
            U, S, Vt = np.linalg.svd(T, full_matrices=False); r = int(np.sum(S > svd_tol*S[0]))
            self.tab[c] = ((U[:, :r]*S[:r]), np.ascontiguousarray(Vt[:r].reshape(r, len(x), 2*self.nb).transpose(1, 0, 2)).reshape(len(x), r*2*self.nb))
            # table check: random slopes / scale parameters against direct contraction
            xs = rng.uniform(*rng_x, 24); ss = self.nodes[rng.integers(self.ns, size=24)]   # at slope nodes: tests the 1-D table only
            direct = (R[:, c].reshape(self.ns*2*self.nb, -1) @ fam(kr, np.exp(xs)).T).reshape(self.ns, 2, self.nb, len(xs)).transpose(0, 3, 1, 2)
            ws = barycentric_weights(self.nodes, ss); dref = np.einsum("ks,skan->kan", ws, direct)
            approx = self._table(c, ws, barycentric_weights(x, xs))
            scale = np.max(np.abs(dref), axis=(0, 2), keepdims=True)
            self.check[c] = float(np.max(np.abs(approx-dref)/np.where(scale > 0, scale, 1)))
            assert self.check[c] < table_tol, f"table for component {c} not converged ({self.check[c]:.1e} >= {table_tol:.0e})"
        self.rank = {c: self.tab[c][0].shape[1] for c in self.tab}

    def _table(self, c, ws, ux):
        """ws (n, ns) slope weights, ux (n, nx) scale-parameter weights -> (n, 2, nb)."""
        A, Bx = self.tab[c]
        a = ws @ A                                                  # (n, r)
        bx = (ux @ Bx).reshape(len(ws), a.shape[1], 2*self.nb)      # (n, r, 2nb)
        return np.einsum("nr,nrp->np", a, bx).reshape(len(ws), 2, self.nb)

    def _outside(self, c, with_src):
        if self.support == "intrinsic":   # only the slope prior bound and the (prior-covering) table range
            bad = ((np.abs(c[:, 2]) > self.ax["slope"][-1]) | ~np.isfinite(c[:, :3]).all(axis=1) | (c[:, 0] > self.lnL[1])
                   | (c[:, 0]+c[:, 1] < self.lnL[0]))
            if with_src:
                bad |= ~np.isfinite(c[:, 3]) | (c[:, 3] < self.ax["src"][0]) | (c[:, 3] > self.ax["src"][-1])
            return bad
        bad = ((c[:, 0] < self.ax["ell"][0]) | (c[:, 0] > self.ax["ell"][-1]) | (c[:, 1] < self.ax["eps"][0]) | (c[:, 1] > self.ax["eps"][-1])
               | (c[:, 2] < self.ax["slope"][0]) | (c[:, 2] > self.ax["slope"][-1]) | ~np.isfinite(c[:, :3]).all(axis=1))
        if with_src:
            bad |= (c[:, 3] < self.ax["src"][0]) | (c[:, 3] > self.ax["src"][-1]) | ~np.isfinite(c[:, 3])
        return bad

    def zero_p(self, coords):
        c = np.atleast_2d(coords); out = np.zeros((len(c), 2, 2, self.nb)); out[self._outside(c, False)] = np.nan; return out

    def zero_n(self, coords):
        c = np.atleast_2d(coords); out = np.zeros((len(c), 2, self.nb)); out[self._outside(c, True)] = np.nan; return out

    def pinterp(self, coords):
        c = np.atleast_2d(coords); out = np.full((len(c), 2, 2, self.nb), np.nan)
        ok = ~self._outside(c, False)
        if ok.any():
            lnl = c[ok, 0]; lna = c[ok, 0]+c[ok, 1]                 # ln ell_obs, ln(eps_obs ell_obs)
            assert lna.min() >= self.lnL[0] and lnl.max() <= self.lnL[1]
            ux = barycentric_weights(self.xL, lnl)-barycentric_weights(self.xL, lna)
            ws = barycentric_weights(self.nodes, c[ok, 2])
            out[ok, 0] = self._table(0, ws, ux); out[ok, 1] = self._table(1, ws, ux)
        return out

    def ninterp(self, coords):
        c = np.atleast_2d(coords); out = np.full((len(c), 2, self.nb), np.nan)
        ok = ~self._outside(c, True)
        if ok.any():
            idx = np.nonzero(ok)[0]; ux = np.zeros((len(idx), len(self.xD)))
            for j, i in enumerate(idx):
                D, W = cnp_nodes(np.exp(c[i, 0]), np.exp(c[i, 1]), np.exp(c[i, 3]), self.cnp_n)
                lnD = np.log(D); assert lnD.min() >= self.lnD[0] and lnD.max() <= self.lnD[1]
                ux[j] = W @ barycentric_weights(self.xD, lnD)
            out[ok] = self._table(2, barycentric_weights(self.nodes, c[ok, 2]), ux)
        return out


def install(model, library, log=print, support="legacy", **kw):
    """Swap the stochastic interpolators of a (Split)ExactWindowKTZ model for the unified converged evaluator.
    The grid axes (and therefore the NaN prediction domain) are taken from the legacy interpolators themselves."""
    pg, ng = model.interp_all["pinterp"], model.interp_all["ninterp"]
    ell_ax, eps_ax, slope_ax = pg.grid[0], pg.grid[1], pg.grid[2]
    assert np.array_equal(ng.grid[0], ell_ax) and np.array_equal(ng.grid[1], eps_ax) and np.array_equal(ng.grid[2], slope_ax)
    sigma_b = model.beam_fwhm_kpc/2.354820045
    U = UnifiedStochasticResponse(library, ell_ax, eps_ax, ng.grid[3], slope_ax, sigma_b, log=log, support=support,
                                  ell_prior_max=max(3.2, float(model.ell_grid[-1])), **kw)
    for t in [model.interp_all] + ([model.sub["same"]] if hasattr(model, "sub") else []):
        t["pinterp_legacy"], t["ninterp_legacy"] = t["pinterp"], t["ninterp"]
        t["pinterp"], t["ninterp"] = U.pinterp, U.ninterp
    if hasattr(model, "sub") and not np.any(np.asarray(model.sub["cross"]["pinterp"].values)) and not np.any(np.asarray(model.sub["cross"]["ninterp"].values)):
        # single-pointing runs: the cross-pointing responses are identically zero; replace the grid look-up of zeros by
        # zeros with the same NaN domain (bit-identical output, verified in the regression test)
        t = model.sub["cross"]; t["pinterp_legacy"], t["ninterp_legacy"] = t["pinterp"], t["ninterp"]
        t["pinterp"], t["ninterp"] = U.zero_p, U.zero_n
    log(f"unified stochastic responses: {U.ns} Chebyshev-Lobatto slope nodes, cubic-Hermite kernel-radius basis (nk={len(U.kr)}), "
        f"exact per-sample kernels via 1-D tables (n_L={len(U.xL)}, n_D={len(U.xD)}, max table error {max(U.check.values()):.1e}, "
        f"slope SVD ranks {U.rank}), C_NP age integral n={U.cnp_n}/piece; prediction support: {support}"
        + (" (intrinsic ell_*/eps prior, no grid truncation)" if support == "intrinsic" else " (legacy grid box)"))
    return U


# ----------------------------------------------------------------------------------
# compiled evaluator (2026-10-16): the converged 1-D tables alone, so a remote worker can evaluate the model without the
# (0.3-0.6 GB) response library; identical arithmetic to the in-memory evaluator.
# ----------------------------------------------------------------------------------
def save_compiled(U, path, provenance=None):
    d = dict(nodes=U.nodes, xL=U.xL, xD=U.xD, lnL=np.asarray(U.lnL), lnD=np.asarray(U.lnD), nb=np.asarray(U.nb), ns=np.asarray(U.ns),
             cnp_n=np.asarray(U.cnp_n), support=np.asarray(U.support), sigma_beam=np.asarray(U.sigma_beam),
             ax_ell=U.ax["ell"], ax_eps=U.ax["eps"], ax_src=U.ax["src"], ax_slope=U.ax["slope"],
             check=np.asarray([U.check[c] for c in range(3)]), provenance=np.asarray(json.dumps(provenance or {})))
    for c in range(3):
        d[f"A{c}"], d[f"B{c}"] = U.tab[c]
    tmp = str(path) + ".partial.npz"; np.savez(tmp, **d); import os; os.replace(tmp, path)


def load_compiled(path):
    z = np.load(path); U = UnifiedStochasticResponse.__new__(UnifiedStochasticResponse)
    U.nodes, U.xL, U.xD = z["nodes"], z["xL"], z["xD"]; U.lnL, U.lnD = tuple(z["lnL"]), tuple(z["lnD"])
    U.nb, U.ns, U.cnp_n, U.support, U.sigma_beam = int(z["nb"]), int(z["ns"]), int(z["cnp_n"]), str(z["support"]), float(z["sigma_beam"])
    U.ax = dict(ell=z["ax_ell"], eps=z["ax_eps"], src=z["ax_src"], slope=z["ax_slope"])
    U.tab = {c: (z[f"A{c}"], z[f"B{c}"]) for c in range(3)}; U.check = {c: float(z["check"][c]) for c in range(3)}
    U.rank = {c: U.tab[c][0].shape[1] for c in range(3)}; U.kr = np.zeros(0)
    return U


def install_evaluator(model, U, log=print):
    """Install an already built (or loaded compiled) evaluator; same replacement rules as install()."""
    for t in [model.interp_all] + ([model.sub["same"]] if hasattr(model, "sub") else []):
        t["pinterp_legacy"], t["ninterp_legacy"] = t["pinterp"], t["ninterp"]
        t["pinterp"], t["ninterp"] = U.pinterp, U.ninterp
    if hasattr(model, "sub"):
        assert not np.any(np.asarray(model.sub["cross"]["pinterp"].values)) and not np.any(np.asarray(model.sub["cross"]["ninterp"].values)), \
            "cross-pointing responses are not zero: compiled evaluator supports single-pointing runs only"
        t = model.sub["cross"]; t["pinterp_legacy"], t["ninterp_legacy"] = t["pinterp"], t["ninterp"]
        t["pinterp"], t["ninterp"] = U.zero_p, U.zero_n
    log(f"compiled unified evaluator installed: support {U.support}, slope ranks {U.rank}, table check {max(U.check.values()):.1e}")
    return U
