"""Same-event source integral on the actual projected finite window (2026-10-26).

For one age (per-axis response variance V = intrinsic v(a) + beam variance, beam entering once):

    C_V(x, x') = int h(y) G_V(x - y) G_V(x' - y) d^2y ,

a Gram integral, PSD for h >= 0.  For every separation bin b the projected pair numerator and
endpoint denominator are LINEAR functionals of the source rate h:

    num_b = int h(y) S^num_{b,V}(y) dy,   S^num_{b,V}(y) = sum_{(k,l) in b} (P phi_y)_k (P phi_y)_l
    den_b = int h(y) S^den_{b,V}(y) dy,   S^den_{b,V}(y) = sum_k e_b(k) (P phi_y)_k^2,   phi_y = G_V(. - y)

The sensitivity maps S are evaluated exactly (no kernel-radius grid, no midpoint rule):
    sum_{(k,l) in b} G_V(x_k-y)G_V(x_l-y) = G_{V/2} * Y_{b,V},  Y_{b,V}(mid) = sum_{(k,l) in b} G_{2V}(d_kl) delta(mid_kl)
(Gaussian product identity, exact), the annulus-mean terms are products of Gaussian-smoothed
cell fields, and G_V(x-y)^2 = G_{V/2}(x-y)/(4 pi V).  All Gaussian smoothings are separable
dense matrix products evaluated at the exact stored cell coordinates.  The only discretisation
is the source quadrature on a uniform grid (trapezoid), which is converged explicitly.
"""
from __future__ import annotations
import numpy as np


def g1(t, V):
    return np.exp(-0.5*t*t/V)/np.sqrt(2*np.pi*V)


class SourceGrid:
    """tensor source grid: nodes ys, xs with quadrature weights wy, wx (uniform trapezoid by default)."""
    def __init__(self, x0, x1, y0, y1, step, xs=None, wx=None, ys=None, wy=None):
        if xs is None:
            nx = int(np.ceil((x1-x0)/step))+1; ny = int(np.ceil((y1-y0)/step))+1
            self.xs = x0+step*np.arange(nx); self.ys = y0+step*np.arange(ny)
            self.wx = np.full(nx, step); self.wy = np.full(ny, step)
        else:
            self.xs, self.wx, self.ys, self.wy = map(np.asarray, (xs, wx, ys, wy))
        self.step = step
        self.shape = (len(self.ys), len(self.xs))

    @classmethod
    def gauss_cells(cls, edges_x, edges_y, p):
        """composite p-point Gauss-Legendre on the cells between consecutive edges (exact for polynomials of
        degree 2p-1 within each cell; used with edges at the template pixel centres where the bilinear g has kinks)."""
        t, w = np.polynomial.legendre.leggauss(p)
        def nodes(e):
            a, b = e[:-1, None], e[1:, None]
            return (0.5*(b-a)*(t[None]+1)+a).ravel(), (0.5*(b-a)*w[None]).ravel()
        xs, wx = nodes(np.asarray(edges_x)); ys, wy = nodes(np.asarray(edges_y))
        return cls(None, None, None, None, float(np.min(np.diff(edges_x)))/p, xs, wx, ys, wy)

    def radius(self):
        return np.hypot(self.xs[None, :], self.ys[:, None])

    def weights(self):
        return self.wy[:, None]*self.wx[None, :]


def half_lattice(v):
    out = np.empty(2*len(v)-1); out[0::2] = v; out[1::2] = 0.5*(v[:-1]+v[1:]); return out


def midpoint_maps(geom, V_list, bins=None):
    """Y[b, j] on the half lattice: sum over ordered pairs in bin b of G_{2V_j}(d) at the pair midpoint."""
    H, W = geom.H, geom.W; m = geom.m
    bins = range(geom.nb) if bins is None else bins
    bins = list(bins); bidx = {b: i for i, b in enumerate(bins)}
    V = np.asarray(V_list, float)
    Y = np.zeros((len(bins), len(V), 2*H-1, 2*W-1))
    sel = np.nonzero(np.isin(geom.lagbin, bins))
    for iy, ix in zip(*sel):
        dy, dx = int(geom.ly[iy, ix]), int(geom.lx[iy, ix]); b = int(geom.lagbin[iy, ix])
        d = geom.lag[iy, ix]
        i0, i1 = max(0, -dy), min(H, H-dy); j0, j1 = max(0, -dx), min(W, W-dx)
        P = m[i0:i1, j0:j1]*m[i0+dy:i1+dy, j0+dx:j1+dx]
        if not P.any():
            continue
        w = np.exp(-d*d/(4*V))/(4*np.pi*V)                      # G_{2V}(d)
        Y[bidx[b], :, 2*i0+dy:2*(i1-1)+dy+1:2, 2*j0+dx:2*(j1-1)+dx+1:2] += w[:, None, None]*P[None]
    return Y


class SameEventMaps:
    """Contractions int h S^num_b, int h S^den_b for source rates H (n_h, ny, nx) on a source grid, one variance V.
    The sensitivity maps are built bin by bin and contracted immediately (memory ~ (n_h + 2 n_annuli) grid)."""
    def __init__(self, geom, grid, V, Y_V, H):
        self.g, self.grid, self.V = geom, grid, V
        Gy = g1(grid.ys[:, None]-geom.Y1[None, :], V); Gx = g1(grid.xs[:, None]-geom.X1[None, :], V)
        Gy2 = g1(grid.ys[:, None]-geom.Y1[None, :], V/2); Gx2 = g1(grid.xs[:, None]-geom.X1[None, :], V/2)
        Yh, Xh = half_lattice(geom.Y1), half_lattice(geom.X1)
        Gyh = g1(grid.ys[:, None]-Yh[None, :], V/2); Gxh = g1(grid.xs[:, None]-Xh[None, :], V/2)
        sm = lambda F, A, B: np.einsum("ij,...jk,lk->...il", A, F, B, optimize=True)
        Hw = H*grid.weights()[None]                                       # (n_h, ny, nx)
        U = sm(geom.u, Gy, Gx)                                            # (na, ny, nx)
        Ua = U/geom.n_a[:, None, None]
        nb = geom.nb
        res = np.zeros((len(H), 2, nb))
        for b in range(nb):
            num = sm(Y_V[b], Gyh, Gxh)
            num -= 2*np.einsum("ayx,ayx->yx", Ua, sm(geom.c[b], Gy, Gx))
            coef = geom.Nab[b]/np.outer(geom.n_a, geom.n_a)
            num += np.einsum("ayx,ayx->yx", U, np.einsum("ac,cyx->ayx", coef, U))
            den = sm(geom.e[b], Gy2, Gx2)/(4*np.pi*V)
            den -= 2*np.einsum("ayx,ayx->yx", Ua, sm(geom.e[b][None]*geom.u, Gy, Gx))
            den += np.einsum("a,ayx->yx", geom.Eab[b]/geom.n_a**2, U*U)
            res[:, 0, b] = np.einsum("hyx,yx->h", Hw, num)
            res[:, 1, b] = np.einsum("hyx,yx->h", Hw, den)
        self.res = res

    def contract(self, H=None):
        return self.res
