"""Exact distinct-event (C_NP) source integral on the actual projected window, polar-harmonic form (2026-10-26).

C_NP(x, x') / (betaA A_ref) = sum_{s,s'} w_s w_s' int int G_{V_s}(x-y) R(y) E(y-y') R(y') G_{V_s'}(x'-y') dy dy'
with R(y) = exp(slope (|y| - r_ref)) the radial event rate (default zeroth order in the arm modulation; source
completion C1: continued over the whole plane) and E(Delta) = exp(-|Delta|^2/(2 L^2)) (L = ell_src).
It is the Gram form A E A^T with E of non-negative spectrum 2 pi L^2 exp(-L^2 q^2/2): PSD for every
slope, L, age weights and quadrature (positive q weights).

Because R is radial and every kernel isotropic, C_NP is invariant under a common rotation:
    C(x, x') = sum_m exp(i m (theta - theta')) M_m(r, r'),
    M_m(r, r') = 2 pi int_0^inf q dq Ehat(q) a^V_m(r, q) a^{V'}_m(r', q),
    a^V_m(r, q) = int_0^inf K^V_m(r, rho) R(rho) J_m(q rho) rho drho,
    K^V_m(r, rho) = exp(-(r - rho)^2/(2V)) ive(m, r rho/V) / (2 pi V)       [angular harmonic of G_V]
(Jacobi-Anger expansion of the Gaussian and of the plane waves of E).  The window enters only through
    Hhat_{b,m}(i, j) = Re sum_{(k,l) in b} conj(F_{m,i}(k)) F_{m,j}(l),   F_{m,i} = P[ exp(i m theta) psi_i(r) ]
(and the endpoint analogue), psi_i the radial interpolation basis at the actual cell radii.
Discretisations, each converged explicitly: radial interpolation of M_m at the cell radii (local cubic
Lagrange), harmonic truncation |m| <= m_max, source radial quadrature, q quadrature (Gauss-Legendre panels).
"""
from __future__ import annotations
import numpy as np
from scipy.special import ive, jv
from scipy.signal import fftconvolve


def lagrange_p(r, nodes, p=4):
    """(n_cells, p) indices and weights of local p-point Lagrange interpolation on uniform nodes."""
    h = nodes[1]-nodes[0]; t = (r-nodes[0])/h
    i0 = np.clip(np.floor(t).astype(int)-(p//2-1), 0, len(nodes)-p)
    idx = i0[:, None]+np.arange(p)[None]
    u = t-i0
    L = np.empty((len(r), p))
    for a in range(p):
        num = np.ones(len(r)); den = 1.0
        for b in range(p):
            if b != a:
                num *= (u-b); den *= (a-b)
        L[:, a] = num/den
    return idx, L


class PolarWindow:
    """Window harmonics Hhat_{b,m}, Dhat_{b,m} (real parts) for one mask; computed m by m."""
    def __init__(self, geom, r_step, pad=3, order=4):
        self.g = geom
        yy, xx = np.meshgrid(geom.Y1, geom.X1, indexing="ij")
        self.R = np.hypot(xx, yy); self.TH = np.arctan2(yy, xx)
        rv = self.R[geom.valid]
        lo, hi = rv.min()-pad*r_step, rv.max()+pad*r_step
        n = int(np.ceil((hi-lo)/r_step))+1
        self.nodes = lo+r_step*np.arange(n); self.nr = n
        idx, L = lagrange_p(rv, self.nodes, order)
        # psi fields: psi_i on the lattice (zero off valid cells)
        self.psi = np.zeros((n,)+geom.valid.shape)
        vy, vx = np.nonzero(geom.valid)
        for a in range(order):
            np.add.at(self.psi, (idx[:, a], vy, vx), L[:, a])

    def harmonics(self, m):
        g = self.g; H, W = g.H, g.W
        ph = np.exp(1j*m*self.TH)
        F = g.project(self.psi*ph[None])                                       # (nr, H, W) complex
        FF = np.fft.fft2(F, s=(2*H-1, 2*W-1))
        nb, nr = g.nb, self.nr
        Hh = np.zeros((nb, nr, nr)); Dh = np.zeros((nb, nr, nr))
        sel = g.lagbin >= 0; lb = g.lagbin[sel]
        # lag index: corr[lag] = sum_k conj(F_i(k)) F_j(k+lag); with FFT, ifft(conj(FF_i)*FF_j)[t] at t = lag mod size
        Ly, Lx = g.ly[sel] % (2*H-1), g.lx[sel] % (2*W-1)
        active = [i for i in range(nr) if np.any(F[i] != 0)]
        for i in active:
            for j0 in range(i, nr, 12):                                          # chunked (memory)
                prod = np.conj(FF[i])[None]*FF[j0:j0+12]
                corr = np.fft.ifft2(prod)[:, Ly, Lx].real                        # (<=12, nsel)
                for jj in range(corr.shape[0]):
                    v = np.bincount(lb, weights=corr[jj], minlength=nb)
                    Hh[:, i, j0+jj] = v; Hh[:, j0+jj, i] = v
        Fv = F[:, g.valid]                                                       # (nr, ncells)
        ev = g.e[:, g.valid]                                                     # (nb, ncells)
        for b in range(nb):
            Dh[b] = np.real((Fv*ev[b][None]) @ np.conj(Fv).T)
        return Hh, Dh


class RadialKernels:
    """a^V_m(r_i, q) for V nodes, one slope (rate exponent) and one harmonic m."""
    def __init__(self, r_nodes, rho, rho_w, q, q_w):
        self.r, self.rho, self.rho_w, self.q, self.q_w = r_nodes, rho, rho_w, q, q_w

    def a(self, V, m, rate):
        K = np.exp(-(self.r[:, None]-self.rho[None])**2/(2*V))*ive(m, self.r[:, None]*self.rho[None]/V)/(2*np.pi*V)
        J = jv(m, self.q[None]*self.rho[:, None])                                # (nrho, nq)
        return (K*(rate*self.rho_w)[None]) @ J                                   # (nr, nq)
