"""C_NP Gaussian-source kernel, same integral as phangs_template_ktz.gaussian_source_cnp_kernel:
K(r) = E_{a,b~U(0,1)} [ L^2/D exp(-r^2/(2D)) ],  D = L^2 + ell^2 (w_a + w_b),  w = eps^2 + (1-eps^2) a.
s = w_a + w_b has a triangular density on [2 eps^2, 2] (mode 1+eps^2), so the double age integral is a 1-D integral;
with u = ln D on each linear piece the integrand (L^2/ell^2) p(s(u)) exp(-r^2/(2 e^u)) is smooth and n-point
Gauss-Legendre per piece converges exponentially (verified against the brute-force double quadrature)."""
import numpy as np


_GL = {}


def cnp_nodes(ell, eps, L, n=24):
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    x, w = _GL[n]; d = 1-eps**2; e2 = eps**2
    Dn, Wn = [], []
    for s0, s1, dens in ((2*e2, 1+e2, lambda s: (s-2*e2)/d**2), (1+e2, 2.0, lambda s: (2-s)/d**2)):
        u0, u1 = np.log(L**2+ell**2*s0), np.log(L**2+ell**2*s1)
        u = 0.5*(u1-u0)*x+0.5*(u1+u0); D = np.exp(u); s = (D-L**2)/ell**2
        Dn.append(D); Wn.append(0.5*(u1-u0)*w*dens(s)*L**2/ell**2)    # ds = D du/ell^2 ; integrand L^2/D -> L^2/ell^2
    return np.concatenate(Dn), np.concatenate(Wn)


def cnp_kernel_fast(r, ell, eps, L, n=24):
    D, W = cnp_nodes(ell, eps, L, n)
    r = np.asarray(r, float)
    return np.exp(-r.ravel()[:, None]**2/(2*D[None, :])) @ W
