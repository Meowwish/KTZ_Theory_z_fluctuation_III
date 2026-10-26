#!/usr/bin/env python3
"""Legal generative realisations of the KTZ field on the actual mask (2026-10-26), for the targeted error-propagation
checks (B) and for the EB posterior-predictive covariance.

z(x) = m(x) + e_P(x) + e_NP(x) [+ q f_dip(x)] [+ n(x)]
  m    : deterministic transported mean sqrt(beta) sum_s w_s G_{V_s} * R(1 + eta g)            (fixed, not randomised)
  e_P  : same-event field, Cov = 4 pi d sum_s w_s int R(1+eta g) G_s G_s^T dy  (Gram draw: independent white noise per age
         node on a uniform source grid, weight sqrt(h dy)); PSD by construction
  e_NP : distinct-event field, Cov = betaA T D_R E D_R T^T; draw T D_R (sqrt(2 pi L^2) G_{L^2/2} * white)
  f_dip: the production m=1 dipole template.  In the likelihood the dipole is a ZERO-MEAN random mode whose coefficient has
         rms q (it adds q^2 * pair-sums of f_dip, no mean and no cross term with m).  Corrected 2026-10-26 after review: the
         'emdn' scenario now draws the coefficient c ~ N(0, q^2) per realisation.  B-check files gen_B_*.npz written before
         this fix used a FIXED coefficient +q (an extra deterministic pattern with spurious cross terms with m); their 'emdn'
         arrays are retained for provenance but are not valid for a full-model (KTZ + dipole) calibration.  The e, em, en,
         emn scenarios (KTZ core, no dipole) are unaffected.
  n    : one of the line-flux Monte Carlo metallicity-noise realisations stored with the pixel map (measurement layer, once)
The statistic is computed exactly as for the data: annular projection on the actual mask, ordered-pair numerator and
endpoint denominator per separation bin, ratio per realisation.
Age integral: n_age-point quadrature (default 16; converged to <3e-6 in the binned expectations, see logs).
Source quadrature: uniform grid, spacing min(pixel/2, 0.5 sqrt(V/2)), bilinear template (as the tables' coarse rule);
the expectation of the generated statistics is checked against the exact tables in --check mode.
"""
from __future__ import annotations
import sys, json, time, os
import numpy as np
from pathlib import Path
from scipy.signal import fftconvolve
sys.path.insert(0, str(Path(__file__).resolve().parent))
import si_tables as st
from si_same_event import SourceGrid, g1
from si_eval import age_nodes


class Generator:
    def __init__(self, cfgd, par, n_age=16, rng=None):
        self.cfg = st.Config(cfgd); self.geom = self.cfg.geometry(annulus_counts=False); self.par = par
        ell, eps = par["ell"], par["eps"]
        u, w = age_nodes(eps, n_age); self.w = w[0]; self.V = ell**2*u[0]+self.cfg.sb2; self.c = 2*np.pi*ell**2*(1-eps**2)*self.w
        self.rng = rng or np.random.default_rng(0)
        g = self.geom; self.pix = self.cfg.pixel
        mod = np.asarray(self.cfg.pm["arm_source_modulation"], float)
        self.grids = []
        for V in self.V:
            sp = np.sqrt(V/2); step = min(self.pix/2, 0.5*sp)
            ext = np.sqrt(40*V)+0.9*V+0.5
            grid = SourceGrid(g.X1[0]-ext, g.X1[-1]+ext, g.Y1[0]-ext, g.Y1[-1]+ext, step)
            R = np.exp(par["slope"]*(grid.radius()-self.cfg.r_ref)); gg = self.cfg.g_on(grid)
            h = R*(1+par["eta"]*gg)
            assert h.min() >= 0, "1 + eta g must be non-negative"
            Gy = g1(g.Y1[:, None]-grid.ys[None, :], V); Gx = g1(g.X1[:, None]-grid.xs[None, :], V)
            self.grids.append((grid, np.sqrt(h)*step, Gy, Gx, R))
        # deterministic transported mean at the cells
        m = np.zeros((g.H, g.W))
        for (grid, sh, Gy, Gx, R), wt in zip(self.grids, self.w):
            hh = (sh/grid.step)**2
            m += wt*(Gy @ (hh*grid.step**2) @ Gx.T)
        self.mean = np.sqrt(par["beta"])*m
        # C_NP source grid
        L = par["ell_src"]; self.L = L
        Vmin = self.V.min(); stepn = min(self.pix/2, 0.45*np.sqrt(0.5*L*L+Vmin))
        ext = np.sqrt(40*self.V.max())+0.9*self.V.max()+4*L+0.5
        self.ngrid = SourceGrid(g.X1[0]-ext, g.X1[-1]+ext, g.Y1[0]-ext, g.Y1[-1]+ext, stepn)
        self.nR = np.exp(par["slope"]*(self.ngrid.radius()-self.cfg.r_ref))
        kx = np.arange(-int(np.ceil(5*L/np.sqrt(2)/stepn)), int(np.ceil(5*L/np.sqrt(2)/stepn))+1)*stepn
        k1 = g1(kx, L*L/2)                                          # 1-D Gaussian G_{L^2/2}
        self.nk = np.sqrt(2*np.pi*L*L)*np.outer(k1, k1)*stepn       # zeta(y) = sqrt(2 pi L^2) sum_z dz G(y-z) xi_z, dz = step (white noise of unit variance per node)
        self.nG = [(g1(g.Y1[:, None]-self.ngrid.ys[None, :], V)*stepn, g1(g.X1[:, None]-self.ngrid.xs[None, :], V)*stepn) for V in self.V]
        # dipole and noise layer
        X, Y = np.meshgrid(g.X1, g.Y1); PHI = np.arctan2(Y, X)
        import importlib; sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"ktzobs"))
        from scripts import ktz_exact_window as kew
        self.fdip = kew.detrend(np.where(g.valid, np.cos(PHI-np.deg2rad(par.get("m1", 0.0))), 0.0), g.valid, np.hypot(X, Y), width=g.aw)
        nr = self.cfg.pm.get("noise_realisations")
        self.noise = None if nr is None or len(nr) == 0 else np.nan_to_num(np.asarray(nr, float))

    def draw_stochastic(self):
        g = self.geom; e = np.zeros((g.H, g.W))
        for (grid, sh, Gy, Gx, R), cs in zip(self.grids, self.c):
            xi = self.rng.standard_normal(grid.shape)
            e += np.sqrt(cs)*(Gy @ (sh*xi) @ Gx.T)
        enp = np.zeros((g.H, g.W))
        if self.par.get("betaA", 0) > 0:
            zeta = fftconvolve(self.rng.standard_normal(self.ngrid.shape), self.nk, mode="same")
            src = self.nR*zeta
            for (Gy, Gx), wt in zip(self.nG, self.w):
                enp += wt*(Gy @ src @ Gx.T)
            enp *= np.sqrt(self.par["betaA"])
        return e, enp

    def statistic(self, z):
        pz = self.geom.project(z); n, d = self.geom.pair_sums(pz)
        return n, d


def signal_amplitude(G, npilot=20):
    """amp such that the mean projected cell variance of amp*(m + e) equals the observed signal variance
    var(residual) - mean(z_err^2) on the valid cells (the same normalisation as the production mock covariance)."""
    pm = G.cfg.pm; v = G.geom.valid
    sig2 = max(float(np.var(np.asarray(pm["residual_dex"], float)[v])-np.nanmean(np.asarray(pm["metallicity_error_dex"], float)[v]**2)), 1e-8)
    acc = []
    for _ in range(npilot):
        e, enp = G.draw_stochastic(); acc.append(np.mean(G.geom.project(e+enp+G.mean)[v]**2))
    return float(np.sqrt(sig2/np.mean(acc))), sig2


def main():
    a = sys.argv; cfgd = json.load(open(a[1])); par = json.load(open(a[2])); out = a[3]; nd = int(a[4]); seed = int(a[5]) if len(a) > 5 else 1
    G = Generator(cfgd, par, rng=np.random.default_rng(seed)); g = G.geom; t0 = time.time()
    if par.get("amp") == "auto":
        par["amp"], par["signal_var"] = signal_amplitude(G); print(f"signal amplitude {par['amp']:.4g} (signal variance {par['signal_var']:.3g})", flush=True)
    q = par.get("A_dip", 0.0); amp = par.get("amp", 1.0)       # amp: physical signal amplitude (signal field scaled to the observed signal rms)
    rec = {k: [] for k in ("e", "em", "en", "emn", "emdn")}
    for i in range(nd):
        e, enp = G.draw_stochastic(); s = amp*(e+enp)
        n = G.noise[G.rng.integers(len(G.noise))] if G.noise is not None else 0.0
        M = amp*G.mean
        for key, z in (("e", s), ("em", s+M), ("en", s+n), ("emn", s+M+n), ("emdn", s+M+amp*q*G.rng.standard_normal()*G.fdip+n)):
            rec[key].append(np.array(G.statistic(z)))
        if (i+1) % 25 == 0:
            print(f"{i+1}/{nd} draws ({time.time()-t0:.0f}s)", flush=True)
    mn, md = G.statistic(amp*G.mean)
    np.savez(out, **{k: np.array(v) for k, v in rec.items()}, mean_only=np.array([mn, md]), counts=g.counts, par=json.dumps(par), seed=seed,
             n_age=len(G.V), seconds=time.time()-t0)
    print(f"wrote {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
