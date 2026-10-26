"""Per-sample evaluation of the source-integral tables (2026-10-26).

Age integral: uniform physical age a in [0,1]; per-axis intrinsic variance ell^2 u, u = eps^2+(1-eps^2) a;
Gauss-Legendre in ln u (positive weights, sum 1), as source_integral_reference.age_quadrature.
Response variance V = ell^2 u + sigma_eff^2 (beam once).  Tables are tabulated on Chebyshev-Lobatto nodes in
ln V and in the rate slope; interpolation is barycentric (second kind).
"""
from __future__ import annotations
import numpy as np

_GL = {}


def age_nodes(eps, n=48):
    """u (n,), w (n,) for each eps (vectorised over eps: returns (len(eps), n))."""
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    x, w = _GL[n]
    eps = np.atleast_1d(np.asarray(eps, float)); lo = 2*np.log(eps)[:, None]
    u = np.exp(0.5*(-lo*x[None]+lo))
    wt = -0.5*lo*w[None]*u/(1-eps[:, None]**2)
    return u, wt


def bary(nodes, s):
    n = len(nodes); w = (-1.0)**np.arange(n); w[0] *= 0.5; w[-1] *= 0.5
    s = np.atleast_1d(np.asarray(s, float)); diff = s[..., None]-nodes
    hit = diff == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        k = w/diff; out = k/k.sum(axis=-1, keepdims=True)
    if hit.any():
        idx = np.nonzero(hit.any(axis=-1))
        out[idx] = 0.0
        out[hit] = 1.0
    return out


def V_weights(lnV_nodes, ell, eps, sb2, n_age=48):
    """alpha (n_samples, nV) = sum_s w_s L_j(ln V_s), and the factor 4 pi d = 2 pi ell^2 (1-eps^2)."""
    ell = np.atleast_1d(ell); eps = np.atleast_1d(eps)
    u, w = age_nodes(eps, n_age)
    lnV = np.log(ell[:, None]**2*u+sb2)
    B = bary(lnV_nodes, lnV)                                    # (n, nage, nV)
    return np.einsum("sa,saj->sj", w, B), 2*np.pi*ell**2*(1-eps**2)
