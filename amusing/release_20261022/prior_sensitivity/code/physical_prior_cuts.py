"""physical_prior_cuts.py (2026-10-21 prior sensitivity): physical_prior.retained_prior_mass with the kappa/t_* cut
values as a parameter C=(kmin,kmax,tmin,tmax); the default reproduces physical_prior.py exactly (checked)."""
import numpy as np
from scipy.integrate import quad
from scipy.stats import truncnorm

def _ell_breaks(lo, hi, C=(.05, 1.5, .05, 3.)):
    kmin, kmax, tmin, tmax = C
    # Split at the published computational boundaries and at every change in
    # the sigma/epsilon/extent integration bounds (extent = kappa*t_star).
    points = [.22, .35, .5, 1, .25/.9]
    for extent in (kmin*tmin, tmin*kmax, kmin*tmax, kmax*tmax):
        points += [np.sqrt(2*extent+s*s) for s in (.015, .25)]
        points += [np.sqrt(2*extent/(1-e*e)) for e in (.015, .9)]
    return sorted(set([lo, hi]+[p for p in points if lo < p < hi]))


def retained_prior_mass(inputs, name, order=64, C=(.05, 1.5, .05, 3.)):
    kmin, kmax, tmin, tmax = C
    """Integrate the hard-cut mass; beta and beta*A are integrated analytically.

    Gauss-Legendre rules integrate log(ell), epsilon and the truncated-Gaussian
    slope CDF. Epsilon intervals are split at the kappa/t constraint switches.
    L0's sigma-cut probability uses independent adaptive one-dimensional quad.
    """
    name = name.removesuffix("q")
    if name not in ("L0", "L1", "L1n", "L2"):
        raise ValueError(name)
    lo, hi = inputs["ell_bounds"]
    if not 0 < lo < hi:
        raise ValueError("Invalid intrinsic ell bounds")
    log_range, eps_log_range = np.log(hi/lo), np.log(1.9/1.015)
    breaks = _ell_breaks(lo, hi, C)
    def eps_mass(ell):
        a, b = max(.015, .015/ell), min(.9, .25/ell)
        return max(0., np.log((1+b)/(1+a))/eps_log_range)
    f0 = sum(quad(lambda l: eps_mass(np.exp(l)), np.log(a), np.log(b), epsabs=1e-12)[0]
             for a, b in zip(breaks[:-1], breaks[1:]))/log_range
    if name in ("L0", "L1n"):
        return float(f0*(5/13 if name == "L1n" else 1))
    x, w = np.polynomial.legendre.leggauss(order)
    u = (x+1)/2
    mu, sd, bound = (inputs[k] for k in ("slope_mu", "slope_sd", "slope_bound"))
    slope = truncnorm.ppf(u, (-bound-mu)/sd, (bound-mu)/sd, loc=mu, scale=sd)
    rel = np.exp(slope[:, None]*(np.asarray(inputs["area_radius"])-inputs["r_ref"])) @ np.asarray(inputs["area_weight"])
    log_c = np.log(4*np.pi*inputs["gamma_area"]*1e3*inputs["mass_factor"]/rel)
    total = 0.
    for a, b in zip(breaks[:-1], breaks[1:]):
        log_a, log_b = np.log(a), np.log(b)
        for ell, wl in zip(np.exp(log_a+(log_b-log_a)*u), w*(log_b-log_a)/2):
            emin = max(.015, .015/ell, np.sqrt(max(0., 1-2*kmax*tmax/ell**2)))
            emax = min(.9, .25/ell, np.sqrt(max(0., 1-2*kmin*tmin/ell**2)))
            if emin >= emax:
                continue
            edges = sorted(set([emin, emax]+[p for q in (tmin*kmax, kmin*tmax)
                      if 1-2*q/ell**2 > 0 for p in [np.sqrt(1-2*q/ell**2)] if emin < p < emax]))
            for e0, e1 in zip(edges[:-1], edges[1:]):
                eps = e0+(e1-e0)*u
                extent = .5*ell**2*(1-eps*eps)
                lower = log_c[None, :]+np.log(np.maximum(tmin*extent, extent*extent/kmax))[:, None]
                upper = log_c[None, :]+np.log(np.minimum(tmax*extent, extent*extent/kmin))[:, None]
                fraction = np.maximum(0., np.minimum(np.log(1e4), upper)-np.maximum(np.log(1e-4), lower))/np.log(1e8)
                total += wl*np.sum(w*(e1-e0)/2/(eps_log_range*(1+eps))*(fraction @ (w/2)))
    return float(total/log_range*(5/13 if name == "L2" else 1))


