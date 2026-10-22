"""Normalisation of the manuscript priors conditional on physical cuts.

The likelihood and the independent coordinate-box transform stay unchanged.
For a completed box-prior run, Z_physical = Z_box / retained_mass. For split
runs, combine box evidences with box masses first, then use the GLOBAL mass.
Equivalently combine conditional piece evidences with weights f_i F_i / F.
No posterior reweighting, curve regeneration, or sampler rerun is involved.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.integrate import quad
from scipy.stats import truncnorm

CONVENTION = "conditional_physical_v1"


def inputs_from_model(model, ell_bounds=None):
    if model.eta_prior_kind != "flat":
        raise ValueError("This normalisation is for the manuscript flat-eta prior")
    if not np.allclose([model.eps_grid[0], model.eps_grid[-1]], [.015, .9], rtol=0, atol=1e-12):
        raise ValueError("This normalisation is for the manuscript epsilon bounds")
    return dict(ell_bounds=list(ell_bounds or (model.ell_grid[0], model.ell_grid[-1])),
                slope_mu=model.slope_mu, slope_sd=model.slope_sd, slope_bound=model.slope_bound,
                area_radius=np.asarray(model.area_radius).tolist(),
                area_weight=np.asarray(model.area_weight).tolist(),
                r_ref=model.metadata["r_ref_kpc"], gamma_area=model.gamma_area,
                mass_factor=model.mass_factor)


def inputs_from_metadata(metadata, ell_bounds):
    """Reconstruct the same scalar inputs without loading large response arrays."""
    rmin, rmax = metadata["analysis_rmin_kpc"], metadata["analysis_rmax_kpc"]
    node, weight = np.polynomial.legendre.leggauss(32)
    norm = 1/(1/100-1/1e6)
    mean = norm*np.log(1e4)
    return dict(ell_bounds=list(ell_bounds),
                slope_mu=metadata["event_rate_log_slope_kpc_inv"],
                slope_sd=metadata["event_rate_log_slope_error_kpc_inv"], slope_bound=.9,
                area_radius=(.5*((rmax-rmin)*node+rmax+rmin)).tolist(),
                area_weight=(weight*.5*((rmax-rmin)*node+rmax+rmin)/(rmax+rmin)).tolist(),
                r_ref=metadata["r_ref_kpc"],
                gamma_area=metadata["halpha_sfr_msun_per_yr"]/(np.pi*(rmax*rmax-rmin*rmin)*mean)*1e6,
                mass_factor=mean*mean/(norm*(1e6-100)))


def _ell_breaks(lo, hi):
    # Split at the published computational boundaries and at every change in
    # the sigma/epsilon/extent integration bounds (extent = kappa*t_star).
    points = [.22, .35, .5, 1, .25/.9]
    for extent in (.05*.05, .05*1.5, .05*3, 1.5*3):
        points += [np.sqrt(2*extent+s*s) for s in (.015, .25)]
        points += [np.sqrt(2*extent/(1-e*e)) for e in (.015, .9)]
    return sorted(set([lo, hi]+[p for p in points if lo < p < hi]))


def retained_prior_mass(inputs, name, order=64):
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
    breaks = _ell_breaks(lo, hi)
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
            emin = max(.015, .015/ell, np.sqrt(max(0., 1-9/ell**2)))
            emax = min(.9, .25/ell, np.sqrt(max(0., 1-.005/ell**2)))
            if emin >= emax:
                continue
            edges = sorted(set([emin, emax]+[p for q in (.075, .15)
                      if 1-2*q/ell**2 > 0 for p in [np.sqrt(1-2*q/ell**2)] if emin < p < emax]))
            for e0, e1 in zip(edges[:-1], edges[1:]):
                eps = e0+(e1-e0)*u
                extent = .5*ell**2*(1-eps*eps)
                lower = log_c[None, :]+np.log(np.maximum(.05*extent, extent*extent/1.5))[:, None]
                upper = log_c[None, :]+np.log(np.minimum(3*extent, extent*extent/.05))[:, None]
                fraction = np.maximum(0., np.minimum(np.log(1e4), upper)-np.maximum(np.log(1e-4), lower))/np.log(1e8)
                total += wl*np.sum(w*(e1-e0)/2/(eps_log_range*(1+eps))*(fraction @ (w/2)))
    return float(total/log_range*(5/13 if name == "L2" else 1))


def condition_summary(summary, mass):
    """Idempotent conversion; preserve the original logZ and every posterior."""
    result = copy.deepcopy(summary)
    if not 0 < mass <= 1:
        raise ValueError("Retained prior mass must lie in (0, 1]")
    if result.get("evidence_prior") not in (None, "coordinate_box", CONVENTION):
        raise ValueError("Unknown evidence convention")
    box = result.get("logZ_box", result["logZ"])
    result.update(logZ_box=box, logZ=float(box-np.log(mass)),
                  retained_prior_mass=float(mass), evidence_prior=CONVENTION)
    return result


def condition_split_summary(summary, inputs, order=64):
    """Condition a combined split result without changing its posterior weights."""
    result = copy.deepcopy(summary)
    pieces = result["_pieces"]
    lo, hi = inputs["ell_bounds"]
    if not np.allclose([pieces[0][0], pieces[-1][1]], [lo, hi], rtol=0, atol=1e-12):
        raise ValueError("Global bounds and computational pieces disagree")
    if any(abs(a[1]-b[0]) > 1e-12 for a, b in zip(pieces[:-1], pieces[1:])):
        raise ValueError("Computational pieces must partition the global prior")
    f = np.array([np.log(b/a)/np.log(hi/lo) for a, b in pieces])
    for name, old in summary.items():
        if name.startswith("_"):
            continue
        masses = []
        for a, b in pieces:
            piece_inputs = dict(inputs, ell_bounds=[a, b])
            masses.append(retained_prior_mass(piece_inputs, name, order))
        masses = np.asarray(masses)
        # This partition sum is the normalisation of precisely the same joint
        # base prior as used by the split sampler, not a new per-piece prior.
        mass = float(f @ masses)
        new = condition_summary(old, mass)
        raw = np.asarray(old.get("logZ_box_pieces", old["logZ_pieces"]))
        weights = f*masses/mass
        conditional = raw-np.log(masses)
        joined = float(np.logaddexp.reduce(np.log(weights)+conditional))
        if abs(joined-new["logZ"]) > 1e-9:
            raise ValueError("Stored split evidence does not match its pieces")
        posterior_weights = np.exp(np.log(weights)+conditional-joined)
        if not np.allclose(posterior_weights, old["weights"], rtol=0, atol=1e-9):
            raise ValueError("Prior conditioning unexpectedly changed posterior piece weights")
        new.update(logZ_box_pieces=raw.tolist(), logZ_pieces=conditional.tolist(),
                   prior_masses_box=f.tolist(), retained_prior_mass_pieces=masses.tolist(),
                   prior_masses_physical=weights.tolist())
        result[name] = new
    return result
