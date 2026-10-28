# %% Block 1: parameters
import os, sys, json, time
from pathlib import Path
import numpy as np
from numpy.random import default_rng
from scipy.special import exp1
from scipy.integrate import quad, simpson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(os.environ.get('KTZ_VALIDATION_ROOT', Path.cwd()))
FIG_DIR = ROOT / 'figs'; FIG_DIR.mkdir(exist_ok=True)
OUT_DIR = ROOT / 'outputs'; OUT_DIR.mkdir(exist_ok=True)
METRICS = {}
T_START = time.time()

def figure_path(stem):
    return FIG_DIR / f'validation_{stem}.png'

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # 1/2.3548

# Disc geometry and transport (scalar experiment)
R_in, R_out = 3.0, 15.0
kappa = 0.3579          # kpc^2 Gyr^-1 (KT25 neutral ISM)
sigma_scalar = 0.06     # kpc, injection dispersion
t_star = 0.50           # Gyr

# Spiral: logarithmic, m arms, pitch angle i; h(Theta) = sum_n g_n cos(n Theta + alpha_n), a_n = eta g_n
m_arms = 3
pitch_angle = 20.0
pitch_rad = np.deg2rad(pitch_angle)
Theta0 = 0.0
eta = 0.40
harmonic_n = np.array([1, 2, 3], dtype=int)
harmonic_g = np.array([1.00, 0.40, 0.20])
harmonic_alpha = np.zeros(3)

def h_func(Theta):
    Theta = np.asarray(Theta, dtype=float)
    out = np.zeros_like(Theta)
    for n, g, a in zip(harmonic_n, harmonic_g, harmonic_alpha):
        out = out + g * np.cos(n * Theta + a)
    return out

Theta_check = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
assert 1.0 + eta * np.min(h_func(Theta_check)) > 0

lambda0_0 = 5.0e2       # kpc^-2 Gyr^-1
h_R = 4.8
Y_val = 1.0
validation_rmax = 5.0
sector_phase_halfwidth = np.deg2rad(30.0)
sector_validation_rmax = 1.5
sector_analysis_R_in = R_in + 1.5
sector_analysis_R_out = R_out - 1.5

# Residual source clustering (Thomas process): a fraction f_cl of events are children of parent
# complexes that follow lambda(x); each complex has a Poisson number of children (mean N_ch)
# displaced by isotropic Gaussian offsets of one-dimensional dispersion sigma_cl.
cluster_fraction = 0.50
cluster_radius = 0.35
mean_children_per_cluster = 20.0
cluster_theory_nodes = 24

# Grid and Gaussian observation operator
dx = 0.20
fwhm_beam = 2.0 * dx                         # 0.40 kpc
sigma_beam = fwhm_beam * FWHM_TO_SIGMA       # 0.1699 kpc
grid_radius = R_out + 1.0
QUICK = os.getenv('KTZ_QUICK', '0') == '1'
n_realizations = 2 if QUICK else 10
rng = default_rng(42)

s_eff_scalar = np.sqrt(sigma_scalar**2 + sigma_beam**2)
METRICS['scalar_setup'] = dict(
    dx_kpc=dx, fwhm_beam_kpc=fwhm_beam, sigma_beam_kpc=float(sigma_beam),
    s_eff_kpc=float(s_eff_scalar), q=float(dx / s_eff_scalar),
    ell_star_kpc=float(np.sqrt(sigma_scalar**2 + 2 * kappa * t_star)),
    cluster_fraction=cluster_fraction, sigma_cl_kpc=cluster_radius, N_ch=mean_children_per_cluster,
    ell_src_kpc=float(np.sqrt(2) * cluster_radius), n_realizations=n_realizations)
print(METRICS['scalar_setup'])

# %% Block 2: spiral geometry
def spiral_phase(varpi, phi):
    Phi = (m_arms / np.tan(pitch_rad)) * np.log(varpi / R_in)
    return Phi - m_arms * phi + Theta0

def lambda_rate(varpi, phi):
    return lambda0_0 * np.exp(-varpi / h_R) * (1.0 + eta * h_func(spiral_phase(varpi, phi)))

def k_perp(varpi):
    dPhi = (m_arms / np.tan(pitch_rad)) / varpi
    return np.sqrt(dPhi**2 + (m_arms / varpi)**2)


grid_x=np.arange(-grid_radius, grid_radius + dx/2, dx); X,Y=np.meshgrid(grid_x,grid_x); R_grid=np.hypot(X,Y)
# %% Block 4: deterministic axisymmetric mean and residual
disk_mask = (R_grid >= R_in) & (R_grid <= R_out)
window_margin = np.sqrt(2.0) * dx
analysis_R_in = R_in + window_margin
analysis_R_out = R_out - window_margin
Z_bg = np.where(disk_mask, Y_val * lambda0_0 * t_star * np.exp(-R_grid / h_R), np.nan)

radii_test = np.round(np.linspace(analysis_R_in + 0.5, analysis_R_out - 0.5, 6), 2)
midpoint_annulus_halfwidth = 0.5 * dx
max_pair_separation = R_out
r_bins = np.linspace(0.1, max_pair_separation, 30)
r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
sector_last_bin = np.flatnonzero(r_centers <= sector_validation_rmax)[-1]
# %% Block 6: KTZ prediction with the same Gaussian observation operator
from functools import lru_cache
# u-coordinate in physical units: p = 2 B_s = sigma^2 + 2 kappa s; the beam adds sigma_b^2 to every kernel
p0_val = sigma_scalar**2 + sigma_beam**2
p_ast_val = sigma_scalar**2 + 2 * kappa * t_star + sigma_beam**2
component_order = ['CP0', 'CPN', 'CM', 'CNP']
component_colors = {'CP0': '#7fc97f', 'CPN': '#ffff99', 'CM': '#fdc086', 'CNP': '#beaed4'}
component_labels = {'CP0': r'$C_{\rm U}$', 'CPN': r'$C_{\rm T}$', 'CM': r'$C_{\rm M}$', 'CNP': r'$C_{\rm D}$'}

@lru_cache(maxsize=None)
def I0(r):
    if r == 0:
        return 0.5 * np.log(p_ast_val / p0_val)
    return 0.5 * (exp1(r**2 / (4 * p_ast_val)) - exp1(r**2 / (4 * p0_val)))

@lru_cache(maxsize=None)
def InP(r, Qn):
    f = lambda p: 0.5 / p * np.exp(-r**2 / (4 * p) - Qn**2 * p / 4)
    return quad(f, p0_val, p_ast_val, limit=100)[0]

@lru_cache(maxsize=None)
def Tn(Qn):
    """int_0^t* exp[-(B_s + sigma_b^2/2) Q^2] ds (beam-attenuated mean-field transfer times t*)."""
    if Qn == 0:
        return t_star
    return np.exp(-0.5 * (sigma_scalar**2 + sigma_beam**2) * Qn**2) * (1 - np.exp(-kappa * t_star * Qn**2)) / (kappa * Qn**2)

# C_NP for the Gaussian pair excess E_src = A_src exp(-Delta^2 / 2 ell_src^2), ell_src = sqrt(2) sigma_cl.
_qx, _qw = np.polynomial.legendre.leggauss(cluster_theory_nodes)
_t_nodes = 0.5 * t_star * (_qx + 1.0); _t_w = 0.5 * t_star * _qw
_p_nodes = sigma_scalar**2 + 2 * kappa * _t_nodes + sigma_beam**2
_l2 = 2.0 * cluster_radius**2

@lru_cache(maxsize=None)
def I_cluster_np(r):
    den = _l2 + _p_nodes[:, None] + _p_nodes[None, :]
    return float(np.sum(_t_w[:, None] * _t_w[None, :] * (_l2 / den) * np.exp(-r**2 / (2.0 * den))))

def CNP_prefactor(varpi_m):
    """C_NP / A_2p for the Thomas component: lambda_0^2 A_src ... with Sigma_src = lambda_0 t*."""
    lam_local = lambda0_0 * np.exp(-varpi_m / h_R)
    sigma_src = lam_local * t_star
    A_src = cluster_fraction * mean_children_per_cluster / (4 * np.pi * sigma_src * cluster_radius**2)
    # C_NP = Y^2 lambda_0^2 A_src * int int (l^2/den) e^{-r^2/2den}; divide by A_2p = Y^2 lambda_0 /(4 pi kappa)
    return 4 * np.pi * kappa * lam_local * A_src

def A_2p_factor(varpi_m):
    return Y_val**2 * lambda0_0 * np.exp(-varpi_m / h_R) / (4 * np.pi * kappa)

def source_harmonic_damping(Qn):
    return (1.0 - cluster_fraction) + cluster_fraction * np.exp(-0.5 * (Qn * cluster_radius)**2)

def local_cov_component_parts(varpi_m, phi_m, r, psi):
    Theta_m = spiral_phase(varpi_m, phi_m)
    Qb = k_perp(varpi_m)
    dPhi = (m_arms / np.tan(pitch_rad)) / varpi_m
    gx = dPhi * np.cos(phi_m) + (m_arms / varpi_m) * np.sin(phi_m)
    gy = dPhi * np.sin(phi_m) - (m_arms / varpi_m) * np.cos(phi_m)
    dot = r * (gx * np.cos(psi) + gy * np.sin(psi))
    Tm, Tp = Theta_m - 0.5 * dot, Theta_m + 0.5 * dot
    lam_local = lambda0_0 * np.exp(-varpi_m / h_R)
    R_val = 4 * np.pi * kappa * lam_local * eta**2
    cpn = cvm = cvp = mm = mp = 0.0
    for n, g, a in zip(harmonic_n, harmonic_g, harmonic_alpha):
        Qn = n * Qb; d = source_harmonic_damping(Qn)
        S_m, S_mi, S_pl = g * d * np.cos(n * Theta_m + a), g * d * np.cos(n * Tm + a), g * d * np.cos(n * Tp + a)
        cpn += InP(r, Qn) * S_m; cvm += InP(0, Qn) * S_mi; cvp += InP(0, Qn) * S_pl
        t = Tn(Qn); mm += t * S_mi; mp += t * S_pl
    cnp = CNP_prefactor(varpi_m)
    nums = {'CP0': I0(r), 'CPN': eta * cpn, 'CM': R_val * mm * mp, 'CNP': cnp * I_cluster_np(r)}
    vm_ = {'CP0': I0(0), 'CPN': eta * cvm, 'CM': R_val * mm**2, 'CNP': cnp * I_cluster_np(0.0)}
    vp_ = {'CP0': I0(0), 'CPN': eta * cvp, 'CM': R_val * mp**2, 'CNP': cnp * I_cluster_np(0.0)}
    return nums, vm_, vp_

def endpoint_window_ok(varpi_m, phi_m, r, psi):
    mx, my = varpi_m * np.cos(phi_m), varpi_m * np.sin(phi_m)
    hx, hy = 0.5 * r * np.cos(psi), 0.5 * r * np.sin(psi)
    rm, rp = np.hypot(mx - hx, my - hy), np.hypot(mx + hx, my + hy)
    return (analysis_R_in <= rm <= analysis_R_out) and (analysis_R_in <= rp <= analysis_R_out)

def average_parts(parts, weights):
    """Pair-product average, normalised after averaging; returns components and the no-C_NP total."""
    if not parts:
        return {k: np.nan for k in component_order + ['noCNP']}
    w = np.asarray(weights, float)
    num = {k: np.sum(w * np.array([p[0][k] for p in parts])) / w.sum() for k in component_order}
    vm = {k: np.sum(w * np.array([p[1][k] for p in parts])) / w.sum() for k in component_order}
    vp = {k: np.sum(w * np.array([p[2][k] for p in parts])) / w.sum() for k in component_order}
    den = np.sqrt(sum(vm.values()) * sum(vp.values()))
    out = {k: num[k] / den for k in component_order}
    keep = [k for k in component_order if k != 'CNP']
    out['noCNP'] = sum(num[k] for k in keep) / np.sqrt(sum(vm[k] for k in keep) * sum(vp[k] for k in keep))
    return out

_gx, _gw = np.polynomial.legendre.leggauss(5)
def bin_nodes(lo, hi):
    r = 0.5 * (hi - lo) * _gx + 0.5 * (hi + lo)
    return r, 0.5 * (hi - lo) * _gw * r

def xi2_theory(varpi_m, lo, hi, Nphi=24 if QUICK else 40, Npsi=8 if QUICK else 40):
    phis = np.linspace(0, 2 * np.pi, Nphi, endpoint=False); psis = np.linspace(0, 2 * np.pi, Npsi, endpoint=False)
    parts, wts = [], []
    rs, rw = bin_nodes(lo, hi) if hi > lo else (np.array([lo]), np.array([1.0]))
    for r, w in zip(rs, rw):
        for ph in phis:
            for ps in psis:
                if endpoint_window_ok(varpi_m, ph, r, ps):
                    parts.append(local_cov_component_parts(varpi_m, ph, r, ps)); wts.append(w)
    return average_parts(parts, wts)

def xi3_theory(lo, hi, phase_limited=False, Nvarpi=6 if QUICK else 30, Nphi=24 if QUICK else 60, Npsi=6 if QUICK else 20):
    rin = sector_analysis_R_in if phase_limited else analysis_R_in
    rout = sector_analysis_R_out if phase_limited else analysis_R_out
    e2 = np.linspace(rin**2, rout**2, Nvarpi + 1); vs = np.sqrt(0.5 * (e2[:-1] + e2[1:]))
    phis = np.linspace(0, 2 * np.pi, Nphi, endpoint=False); psis = np.linspace(0, 2 * np.pi, Npsi, endpoint=False)
    rs, rw = bin_nodes(lo, hi) if hi > lo else (np.array([lo]), np.array([1.0]))
    parts, wts = [], []
    for r, w in zip(rs, rw):
        for v in vs:
            A = A_2p_factor(v)
            for ph in phis:
                if phase_limited and abs(np.angle(np.exp(1j * spiral_phase(v, ph)))) >= sector_phase_halfwidth:
                    continue
                for ps in psis:
                    if endpoint_window_ok(v, ph, r, ps):
                        parts.append(local_cov_component_parts(v, ph, r, ps)); wts.append(w * A)
    return average_parts(parts, wts)

def to_arrays(lst):
    return {k: np.array([c[k] for c in lst]) for k in component_order + ['noCNP']}


_orig = local_cov_component_parts
def local_cov_modulated(varpi_m, phi_m, r, psi):
    nums, vm_, vp_ = _orig(varpi_m, phi_m, r, psi)
    th = spiral_phase(varpi_m, phi_m); Qb = k_perp(varpi_m)
    f = 1.0 + eta*sum(g*np.exp(-0.25*(n*Qb*cluster_radius)**2)*np.cos(n*th+a) for n,g,a in zip(harmonic_n,harmonic_g,harmonic_alpha))
    nums['CNP'] *= f; vm_['CNP'] *= f; vp_['CNP'] *= f
    return nums, vm_, vp_
smask = r_centers <= sector_validation_rmax
local_cov_component_parts = local_cov_modulated
th = [xi3_theory(lo, hi, True) for lo, hi in zip(r_bins[:-1][smask], r_bins[1:][smask])]
tot = np.array([sum(c[k] for k in component_order) for c in th])
d = np.load(ROOT/'outputs'/'scalar_simulation_statistics.npz'); sim = np.nanmean(d['sector'],0)[smask]
print('sector theory with midpoint-modulated C_NP:', tot, 'sim', sim, 'resid', sim-tot, 'rmse', np.sqrt(np.mean((sim-tot)**2)))
