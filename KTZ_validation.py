# %% [markdown]
# # Unified clustered-source validation of the KTZ covariance (Paper III, Section 3)
#
# One Monte Carlo experiment tests the complete observable covariance
# $C_{\rm obs}=C_{{\rm 2p},0}+C_{{\rm 2p},N}+C_{\rm M}+C_{\rm NP}$.
# Half of the enrichment events are drawn in Thomas-process complexes whose parents follow the
# spiral-modulated intensity $\lambda(\boldsymbol x)$, so the distinct-event term $C_{\rm NP}$ is present
# in addition to the same-event and mean-field terms.
#
# Observation operator (identical in simulation and theory): the metal field is convolved with a
# Gaussian beam of ${\rm FWHM}_b=2\Delta x$ and sampled at fixed grid centres. Because every event
# kernel is Gaussian, the beam simply adds $\sigma_b^2$ to the per-axis variance of every kernel,
# $s^2\to s^2+\sigma_b^2$, in the simulated fields and in every analytic term.
#
# Residual maps subtract the known axisymmetric ensemble mean $\bar\mu(\varpi)$ rather than an annular
# mean estimated from each map. The scalar estimators enumerate every unordered pixel pair in the
# analysis window; analytic predictions are averaged over the identical separation bins with the
# planar $r\,{\rm d}r$ pair measure.
#
# Multi-element same-event term: the KTZ prediction uses the direct two-dimensional Gaussian drift
# average, <exp[-v^2 dt^2/(4S)]> = [1 + sigma_v^2 dt^2/(2S)]^{-1}, integrated over age (eq. J0_AB).
# The reference labelled KT25 is the published KT25 closed form (Dawson/asinh, KT25 eqs. 28-29 and
# App. A10), which differs from J0_AB for unequal delays and agrees with it for equal delays. The
# log/J0_AB uniform curve is recorded separately as the 'uniform KTZ limit'.

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

# %% Block 3: synthetic galaxies (clustered sources, beam-convolved kernels sampled at grid centres)
Nr, Nphi = 200, 400
varpi_edges = np.linspace(R_in, R_out, Nr + 1)
phi_edges = np.linspace(0, 2 * np.pi, Nphi + 1)
varpi_c = 0.5 * (varpi_edges[:-1] + varpi_edges[1:])
phi_c = 0.5 * (phi_edges[:-1] + phi_edges[1:])
area_cells = (varpi_c * np.diff(varpi_edges))[:, None] * (2 * np.pi / Nphi)
lam_grid = lambda_rate(varpi_c[:, None], phi_c[None, :])
expected_total = np.sum(lam_grid * area_cells) * t_star
lam_max = np.max(lam_grid)

grid_x = np.arange(-grid_radius, grid_radius + dx / 2, dx)
X, Y = np.meshgrid(grid_x, grid_x)
R_grid = np.hypot(X, Y)

def sample_spiral_positions(n_sample, rng):
    if n_sample <= 0:
        return np.empty(0), np.empty(0)
    v_out, p_out = np.empty(0), np.empty(0)
    n_cand = max(int(3 * n_sample), 1)
    while len(v_out) < n_sample:
        v = np.sqrt(rng.uniform(R_in**2, R_out**2, n_cand))
        p = rng.uniform(0, 2 * np.pi, n_cand)
        acc = rng.uniform(size=n_cand) < lambda_rate(v, p) / lam_max
        v_out = np.concatenate([v_out, v[acc]]); p_out = np.concatenate([p_out, p[acc]])
        n_cand = max(2 * (n_sample - len(v_out)), 1)
    return v_out[:n_sample], p_out[:n_sample]

def sample_clustered_positions(rng):
    n_parent = rng.poisson(cluster_fraction * expected_total / mean_children_per_cluster)
    if n_parent == 0:
        return np.empty(0), np.empty(0)
    pv, pp = sample_spiral_positions(n_parent, rng)
    counts = rng.poisson(mean_children_per_cluster, n_parent)
    n_child = int(counts.sum())
    cx = np.repeat(pv * np.cos(pp), counts) + rng.normal(0, cluster_radius, n_child)
    cy = np.repeat(pv * np.sin(pp), counts) + rng.normal(0, cluster_radius, n_child)
    keep = (np.hypot(cx, cy) >= R_in) & (np.hypot(cx, cy) <= R_out)
    return cx[keep], cy[keep]

def generate_synthetic_galaxy(rng, return_events=False):
    n_poisson = rng.poisson((1.0 - cluster_fraction) * expected_total)
    cx, cy = sample_clustered_positions(rng)
    pv, pp = sample_spiral_positions(n_poisson, rng)
    ex = np.concatenate([pv * np.cos(pp), cx]); ey = np.concatenate([pv * np.sin(pp), cy])
    clustered = np.concatenate([np.zeros(n_poisson, bool), np.ones(len(cx), bool)])
    order = rng.permutation(len(ex)); ex, ey, clustered = ex[order], ey[order], clustered[order]
    ages = rng.uniform(0, t_star, len(ex))
    Z = np.zeros_like(X)
    # Beam-convolved event kernel: per-axis variance s^2 = sigma^2 + 2 kappa s_e + sigma_b^2,
    # evaluated at the fixed grid centres from the continuous event position.
    var = sigma_scalar**2 + 2 * kappa * ages + sigma_beam**2
    for i in range(len(ex)):
        cut = 5.0 * np.sqrt(var[i])
        ix0 = max(int(np.floor((ex[i] - cut - grid_x[0]) / dx)), 0)
        ix1 = min(int(np.ceil((ex[i] + cut - grid_x[0]) / dx)) + 1, len(grid_x))
        iy0 = max(int(np.floor((ey[i] - cut - grid_x[0]) / dx)), 0)
        iy1 = min(int(np.ceil((ey[i] + cut - grid_x[0]) / dx)) + 1, len(grid_x))
        gx = np.exp(-(grid_x[ix0:ix1] - ex[i])**2 / (2 * var[i]))
        gy = np.exp(-(grid_x[iy0:iy1] - ey[i])**2 / (2 * var[i]))
        Z[iy0:iy1, ix0:ix1] += Y_val * np.outer(gy, gx) / (2 * np.pi * var[i])
    if return_events:
        return Z, len(ex), ex, ey, clustered
    return Z, len(ex)

t0 = time.time()
Z, N_events, event_x, event_y, event_is_clustered = generate_synthetic_galaxy(rng, return_events=True)
print(f'Expected events {expected_total:.0f}; realisation 1: {N_events} ({event_is_clustered.sum()} clustered); {time.time()-t0:.1f}s')
METRICS['scalar_setup']['expected_events_per_realisation'] = float(expected_total)

# %% Block 4: deterministic axisymmetric mean and residual
disk_mask = (R_grid >= R_in) & (R_grid <= R_out)
window_margin = np.sqrt(2.0) * dx
analysis_R_in = R_in + window_margin
analysis_R_out = R_out - window_margin
Z_bg = np.where(disk_mask, Y_val * lambda0_0 * t_star * np.exp(-R_grid / h_R), np.nan)

def subtract_axisymmetric_mean(Z):
    return np.where(disk_mask, Z - Z_bg, np.nan)

Z_res = subtract_axisymmetric_mean(Z)

# %% Block 5: two-point statistics from the simulated maps (every unordered pair)
radii_test = np.round(np.linspace(analysis_R_in + 0.5, analysis_R_out - 0.5, 6), 2)
midpoint_annulus_halfwidth = 0.5 * dx
max_pair_separation = R_out
r_bins = np.linspace(0.1, max_pair_separation, 30)
r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
sector_last_bin = np.flatnonzero(r_centers <= sector_validation_rmax)[-1]
block = 64

def measure_full_pair_statistics(Z_res):
    m = np.isfinite(Z_res) & (R_grid >= analysis_R_in) & (R_grid <= analysis_R_out)
    px, py, pv = X[m], Y[m], Z_res[m]
    n = pv.size; nb = len(r_centers); nm = len(radii_test)
    x3p = np.zeros(nb); x3w = np.zeros(nb); x3n = np.zeros(nb, np.int64)
    sp = np.zeros(nb); sw = np.zeros(nb); sn = np.zeros(nb, np.int64)
    x2p = np.zeros((nm, nb)); x2w = np.zeros((nm, nb)); x2n = np.zeros((nm, nb), np.int64)
    idx_all = np.arange(n)[None, :]; v_all = pv[None, :]; p_all = v_all**2
    for s0 in range(0, n, block):
        s1 = min(s0 + block, n)
        upper = idx_all > np.arange(s0, s1)[:, None]
        ddx = px[None, :] - px[s0:s1, None]; ddy = py[None, :] - py[s0:s1, None]
        sep = np.hypot(ddx, ddy)
        sb = np.searchsorted(r_bins, sep, side='right') - 1
        ok = upper & (sb >= 0) & (sb < nb)
        bv = pv[s0:s1, None]
        prod = bv * v_all; powr = 0.5 * (bv**2 + p_all)
        mx = 0.5 * (px[None, :] + px[s0:s1, None]); my = 0.5 * (py[None, :] + py[s0:s1, None])
        mr = np.hypot(mx, my)
        full = ok & (mr >= analysis_R_in) & (mr <= analysis_R_out)
        b = sb[full]
        x3n += np.bincount(b, minlength=nb); x3p += np.bincount(b, prod[full], nb); x3w += np.bincount(b, powr[full], nb)
        cand = full & (mr >= sector_analysis_R_in) & (mr <= sector_analysis_R_out) & (sb <= sector_last_bin)
        th = spiral_phase(mr[cand], np.arctan2(my[cand], mx[cand]))
        ins = np.abs((th + np.pi) % (2 * np.pi) - np.pi) < sector_phase_halfwidth
        b = sb[cand][ins]
        sn += np.bincount(b, minlength=nb); sp += np.bincount(b, prod[cand][ins], nb); sw += np.bincount(b, powr[cand][ins], nb)
        for k, vm in enumerate(radii_test):
            sel = ok & (np.abs(mr - vm) < midpoint_annulus_halfwidth)
            b = sb[sel]
            x2n[k] += np.bincount(b, minlength=nb); x2p[k] += np.bincount(b, prod[sel], nb); x2w[k] += np.bincount(b, powr[sel], nb)
    div = lambda a, w, c: np.divide(a, w, out=np.full(a.shape, np.nan), where=(c > 0) & (w > 0))
    x2 = div(x2p, x2w, x2n)
    return dict(xi2={vm: x2[k] for k, vm in enumerate(radii_test)}, xi3=div(x3p, x3w, x3n),
                xi_sector=div(sp, sw, sn), xi3_pair_count=x3n)

xi2_real, xi3_real, sec_real = [], [], []
for ireal in range(n_realizations):
    t0 = time.time()
    if ireal == 0:
        res = Z_res
    else:
        Zi, Ni = generate_synthetic_galaxy(rng)
        res = subtract_axisymmetric_mean(Zi)
    st = measure_full_pair_statistics(res)
    xi2_real.append(st['xi2']); xi3_real.append(st['xi3']); sec_real.append(st['xi_sector'])
    print(f'realisation {ireal+1}/{n_realizations}: {time.time()-t0:.0f}s', flush=True)

sim_xi2_mean = {vm: np.nanmean([r[vm] for r in xi2_real], 0) for vm in radii_test}
sim_xi2_std = {vm: np.nanstd([r[vm] for r in xi2_real], 0, ddof=1) for vm in radii_test}
sim_xi2_sem = {vm: sim_xi2_std[vm] / np.sqrt(n_realizations) for vm in radii_test}
xi3_stack = np.vstack(xi3_real); sec_stack = np.vstack(sec_real)
xi3_sim_mean, xi3_sim_std = np.nanmean(xi3_stack, 0), np.nanstd(xi3_stack, 0, ddof=1)
sec_sim_mean, sec_sim_std = np.nanmean(sec_stack, 0), np.nanstd(sec_stack, 0, ddof=1)
xi3_sim_sem = xi3_sim_std / np.sqrt(n_realizations); sec_sim_sem = sec_sim_std / np.sqrt(n_realizations)
np.savez(OUT_DIR / 'scalar_simulation_statistics.npz', r_bins=r_bins, radii=radii_test,
         xi2=np.array([[r[vm] for vm in radii_test] for r in xi2_real]), xi3=xi3_stack, sector=sec_stack)

# %% Block 6: KTZ prediction with the same Gaussian observation operator
from functools import lru_cache
# u-coordinate in physical units: p = 2 B_s = sigma^2 + 2 kappa s; the beam adds sigma_b^2 to every kernel
p0_val = sigma_scalar**2 + sigma_beam**2
p_ast_val = sigma_scalar**2 + 2 * kappa * t_star + sigma_beam**2
component_order = ['CP0', 'CPN', 'CM', 'CNP']
component_colors = {'CP0': '#7fc97f', 'CPN': '#ffff99', 'CM': '#fdc086', 'CNP': '#beaed4'}
component_labels = {'CP0': r'$C_{\rm 2p,0}$', 'CPN': r'$C_{\rm 2p,N}$', 'CM': r'$C_{\rm M}$', 'CNP': r'$C_{\rm NP}$'}

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

t0 = time.time()
th_xi2 = {vm: to_arrays([xi2_theory(vm, lo, hi) for lo, hi in zip(r_bins[:-1], r_bins[1:])]) for vm in radii_test}
th_xi2_0 = {vm: xi2_theory(vm, 0.0, 0.0) for vm in radii_test}
assert QUICK or max(np.nanmax(np.abs(th_xi2[vm]['CPN'])) for vm in radii_test) < 1e-12
print(f'xi2 theory {time.time()-t0:.0f}s', flush=True)

vmask = r_centers <= validation_rmax
smask = r_centers <= sector_validation_rmax
t0 = time.time()
th_xi3 = to_arrays([xi3_theory(lo, hi) for lo, hi in zip(r_bins[:-1][vmask], r_bins[1:][vmask])])
th_xi3_0 = xi3_theory(0.0, 0.0)
assert QUICK or np.nanmax(np.abs(th_xi3['CPN'])) < 1e-12
th_sec = to_arrays([xi3_theory(lo, hi, True) for lo, hi in zip(r_bins[:-1][smask], r_bins[1:][smask])])
th_sec_0 = xi3_theory(0.0, 0.0, True)
print(f'xi3/sector theory {time.time()-t0:.0f}s', flush=True)

# %% Block 7: field figure (simulation / subtracted axisymmetric mean / fluctuation used for xi)
def full_mean_field(Xg, Yg):
    """Beam-smoothed ensemble mean mu = mu_bar + mu_mod in the local-WKBJ approximation."""
    vr = np.hypot(Xg, Yg); ph = np.arctan2(Yg, Xg)
    vr_safe = np.clip(vr, R_in, R_out)
    th = spiral_phase(vr_safe, ph); Qb = k_perp(vr_safe)
    mod = np.zeros_like(vr)
    for n, g, a in zip(harmonic_n, harmonic_g, harmonic_alpha):
        Q = n * Qb
        tn = np.exp(-0.5 * (sigma_scalar**2 + sigma_beam**2) * Q**2) * (1 - np.exp(-kappa * t_star * Q**2)) / (kappa * Q**2)
        mod += eta * g * source_harmonic_damping(Q) * (tn / t_star) * np.cos(n * th + a)
    return np.where(disk_mask, Y_val * lambda0_0 * t_star * np.exp(-vr / h_R) * (1.0 + mod), np.nan)

mu_full = full_mean_field(X, Y)
Z_plot = np.where(disk_mask, Z, np.nan)
stoch = Z_plot - mu_full  # Preserve the published stochastic-field diagnostics below.
dX_ax = Z_res  # Block 4: the fluctuation passed to the pair estimator.
ana = np.isfinite(dX_ax) & (R_grid >= analysis_R_in) & (R_grid <= analysis_R_out)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
vmin, vmax = np.nanpercentile(Z_plot, [1, 99])
for ax, field, lab, title in [(axes[0], Z_plot, r'$\Sigma_X$', 'Simulation'),
                             (axes[1], Z_bg, r'$\bar\mu$', 'Subtracted axisymmetric mean')]:
    im = ax.pcolormesh(X, Y, field, cmap='plasma', shading='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=lab); ax.set_title(title)
lim = np.nanpercentile(np.abs(dX_ax[ana]), 99.5)
im = axes[2].pcolormesh(X, Y, dX_ax, cmap='RdBu_r', shading='auto', vmin=-lim, vmax=lim)
plt.colorbar(im, ax=axes[2], label=r'$\delta\Sigma_{X,{\rm ax}}=\Sigma_X-\bar\mu$')
axes[2].set_title(r'Fluctuation used for $\xi$ (simulation $-\;\bar\mu$)')
for rr in (analysis_R_in, analysis_R_out):
    axes[2].add_patch(plt.Circle((0, 0), rr, fill=False, ls='--', lw=0.8, color='0.25'))
for ax in axes:
    ax.set_xlim(-R_out, R_out); ax.set_ylim(-R_out, R_out); ax.set_aspect('equal'); ax.set_xlabel(r'$x\;({\rm kpc})$')
axes[0].set_ylabel(r'$y\;({\rm kpc})$')
plt.tight_layout(); fig.savefig(figure_path('scalar_fields'), dpi=160, bbox_inches='tight'); plt.close(fig)
METRICS['scalar_fields'] = dict(residual_mean_over_window=float(np.nanmean(stoch[(R_grid >= analysis_R_in) & (R_grid <= analysis_R_out)])),
                                residual_rms=float(np.nanstd(stoch[(R_grid >= analysis_R_in) & (R_grid <= analysis_R_out)])),
                                mean_field_rms=float(np.nanstd(mu_full - Z_bg)))

# %% Block 8: xi2, xi3 and arm-sector comparisons; metrics
def stack_plot(ax, r, comp, labels):
    cum = np.zeros_like(r)
    for k in component_order:
        ax.fill_between(r, cum, cum + comp[k], color=component_colors[k], alpha=0.42, lw=0, label=labels[k], zorder=1)
        cum = cum + comp[k]

def summarize(sim, th, unc):
    sim, th, unc = map(np.asarray, (sim, th, unc))
    ok = np.isfinite(sim) & np.isfinite(th) & np.isfinite(unc)
    d = sim[ok] - th[ok]
    return dict(n=int(ok.sum()), rmse=float(np.sqrt(np.mean(d**2))), bias=float(np.mean(d)),
                pearson=float(np.corrcoef(sim[ok], th[ok])[0, 1]), max_abs=float(np.max(np.abs(d))),
                within_1sigma=int(np.sum(np.abs(d) <= unc[ok])))

ann_labels = dict(component_labels); ann_labels['CPN'] = r'$C_{\rm 2p,N}=0$'
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True); axes = axes.flatten()
for i, (ax, vm) in enumerate(zip(axes, radii_test)):
    r = np.concatenate([[0.0], r_centers])
    comp = {k: np.concatenate([[th_xi2_0[vm][k]], th_xi2[vm][k]]) for k in component_order}
    tot = sum(comp[k] for k in component_order)
    nonp = np.concatenate([[1.0], th_xi2[vm]['noCNP']])
    sm = np.concatenate([[1.0], sim_xi2_mean[vm]]); sd = np.concatenate([[0.0], sim_xi2_std[vm]]); se = np.concatenate([[0.0], sim_xi2_sem[vm]])
    stack_plot(ax, r, comp, ann_labels)
    ax.plot(r, sm, 'o-', color='k', ms=4, label='Simulation mean', zorder=4)
    ax.fill_between(r, sm - sd, sm + sd, color='k', alpha=0.16, lw=0, label=r'Single-realisation $1\sigma$', zorder=3)
    ax.errorbar(r, sm, yerr=se, fmt='none', ecolor='k', elinewidth=0.9, capsize=2, label=r'Mean $1\sigma$ uncertainty', zorder=5)
    ax.plot(r, tot, '-', color='#d62728', lw=1.6, label='Full KTZ', zorder=6)
    ax.plot(r, nonp, ls='--', color='#6a3d9a', lw=1.6, label=r'KTZ without $C_{\rm NP}$', zorder=6)
    ax.axhline(0, color='grey', lw=0.5); ax.set_title(fr'$\varpi_m = {vm}$ kpc'); ax.set_xlabel('r (kpc)'); ax.set_xlim(0, validation_rmax)
    if i == 0:
        ax.legend(loc='lower left', fontsize=7.5, ncol=2)
axes[0].set_ylabel(r'$\xi_{2,{\rm pp}}(\varpi_m,r)$'); axes[3].set_ylabel(r'$\xi_{2,{\rm pp}}(\varpi_m,r)$')
plt.tight_layout(); fig.savefig(figure_path('scalar_xi2'), dpi=160, bbox_inches='tight'); plt.close(fig)

xm = r_centers <= validation_rmax
sim2 = np.concatenate([sim_xi2_mean[vm][xm] for vm in radii_test])
unc2 = np.concatenate([sim_xi2_sem[vm][xm] for vm in radii_test])
th2 = np.concatenate([sum(th_xi2[vm][k] for k in component_order)[xm] for vm in radii_test])
th2n = np.concatenate([th_xi2[vm]['noCNP'][xm] for vm in radii_test])
METRICS['xi2'] = summarize(sim2, th2, unc2); METRICS['xi2_noCNP'] = summarize(sim2, th2n, unc2)

def single_panel(stem, rr, comp0, comp, nonp, sm, sd, se, rmax, ylabel, title=None, labels=component_labels):
    r = np.concatenate([[0.0], rr])
    c = {k: np.concatenate([[comp0[k]], comp[k]]) for k in component_order}
    fig, ax = plt.subplots(figsize=(7, 5))
    stack_plot(ax, r, c, labels)
    sm = np.concatenate([[1.0], sm]); sd = np.concatenate([[0.0], sd]); se = np.concatenate([[0.0], se])
    ax.plot(r, sm, 'o-', color='k', ms=4, label='Simulation mean', zorder=4)
    ax.fill_between(r, sm - sd, sm + sd, color='k', alpha=0.14, lw=0, label=r'Single-realisation $1\sigma$', zorder=3)
    ax.errorbar(r, sm, yerr=se, fmt='none', ecolor='k', elinewidth=0.9, capsize=2, label=r'Mean $1\sigma$ uncertainty', zorder=5)
    ax.plot(r, sum(c[k] for k in component_order), '-', color='#d62728', lw=1.6, label='Full KTZ', zorder=6)
    ax.plot(r, np.concatenate([[1.0], nonp]), ls='--', color='#6a3d9a', lw=1.6, label=r'KTZ without $C_{\rm NP}$', zorder=6)
    ax.axhline(0, color='grey', lw=0.5); ax.set_xlabel('r (kpc)'); ax.set_ylabel(ylabel); ax.set_xlim(0, rmax)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    plt.tight_layout(); fig.savefig(figure_path(stem), dpi=160, bbox_inches='tight'); plt.close(fig)

single_panel('scalar_xi3', r_centers[xm], th_xi3_0, th_xi3, th_xi3['noCNP'], xi3_sim_mean[xm], xi3_sim_std[xm], xi3_sim_sem[xm],
             validation_rmax, r'$\xi_{3,{\rm pp}}(r)$', labels=ann_labels)
single_panel('scalar_sector', r_centers[smask], th_sec_0, th_sec, th_sec['noCNP'], sec_sim_mean[smask], sec_sim_std[smask], sec_sim_sem[smask],
             sector_validation_rmax, r'$\xi_{\rm arm,pp}(r)$',
             title=rf'Arm-phase sector: $|\Theta_m|<30^\circ$, ${sector_analysis_R_in:.1f}<\varpi_m/{{\rm kpc}}<{sector_analysis_R_out:.1f}$')
METRICS['xi3'] = summarize(xi3_sim_mean[xm], sum(th_xi3[k] for k in component_order), xi3_sim_sem[xm])
METRICS['xi3_noCNP'] = summarize(xi3_sim_mean[xm], th_xi3['noCNP'], xi3_sim_sem[xm])
METRICS['sector'] = summarize(sec_sim_mean[smask], sum(th_sec[k] for k in component_order), sec_sim_sem[smask])
METRICS['sector_noCNP'] = summarize(sec_sim_mean[smask], th_sec['noCNP'], sec_sim_sem[smask])
METRICS['sector']['max_abs_CPN'] = float(np.nanmax(np.abs(th_sec['CPN'])))
METRICS['component_shares_r0'] = {
    'xi3': {k: float(th_xi3_0[k]) for k in component_order},
    'sector': {k: float(th_sec_0[k]) for k in component_order}}
METRICS['max_abs_CNP_contribution'] = {'xi2': float(max(np.nanmax(np.abs(th_xi2[vm]['CNP'][xm])) for vm in radii_test)),
                                       'xi3': float(np.nanmax(np.abs(th_xi3['CNP'])))}
np.savez(OUT_DIR / 'scalar_theory.npz', r_centers=r_centers, radii=radii_test,
         **{f'xi2_{vm}_{k}': th_xi2[vm][k] for vm in radii_test for k in component_order + ['noCNP']},
         **{f'xi3_{k}': th_xi3[k] for k in component_order + ['noCNP']}, **{f'sector_{k}': th_sec[k] for k in component_order + ['noCNP']})
for k in ['xi2', 'xi2_noCNP', 'xi3', 'xi3_noCNP', 'sector', 'sector_noCNP']:
    print(k, METRICS[k])
json.dump(METRICS, open(OUT_DIR / 'validation_metrics.json', 'w'), indent=1)
print(f'scalar part done in {time.time()-T_START:.0f}s', flush=True)

# %% [markdown]
# # Zero-lag elemental cross-correlation (same clustered-source process, Gaussian beam FWHM = 2 dx_cross)

# %% Block 9: KT25 yields, delays and stochastic SLUG yield moments
import astropy.units as u
from matplotlib.colors import TwoSlopeNorm
KT25_REPO = ROOT / 'data' / 'kt25'
for pth in (str(ROOT), str(KT25_REPO)):
    if pth not in sys.path:
        sys.path.insert(0, pth)
from slugpy_shim import read_cluster
from asplund09_abd import solar_abd
from seitenzahl13_yld import s13_yld

kt25_channels = ['snii', 'agb', 'snia']
kt25_r_process = ['Ru', 'Rh', 'Rd', 'Ag', 'Cd', 'In', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'La', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb']
kt25_prod = {row['sym']: {} for row in solar_abd}
slug_agb = read_cluster(str(KT25_REPO / 'slug' / 'yields_agb'))
slug_snii = read_cluster(str(KT25_REPO / 'slug' / 'yields_snii'))
kt25_time = slug_agb.time * u.yr
tmin, tmax, a_dtd = 40 * u.Myr, 13.7 * u.Gyr, -1.1
norm = 1.3e-3 * (1 + a_dtd) / (tmax * (tmax / tmin)**a_dtd - tmin)
nIa = norm * (kt25_time * (kt25_time / tmin)**a_dtd - tmin) / (1 + a_dtd); nIa[kt25_time < tmin] = 0
for el in kt25_prod:
    cy = {nm: np.zeros(kt25_time.size) for nm in ['tot', 'snii', 'agb', 'snia']}
    for idx in np.where(slug_snii.isotope_name == el)[0]:
        cy['snii'] += slug_snii.yld[:, idx]; cy['tot'] += slug_snii.yld[:, idx]
    for idx in np.where(slug_agb.isotope_name == el)[0]:
        cy['agb'] += slug_agb.yld[:, idx]; cy['tot'] += slug_agb.yld[:, idx]
    for idx in np.where(np.asarray(s13_yld['sym']) == el)[0]:
        ia = s13_yld['ylds']['N100'][idx] * nIa; cy['snia'] += ia.value if hasattr(ia, 'value') else ia; cy['tot'] += ia.value if hasattr(ia, 'value') else ia
    for nm, val in cy.items():
        kt25_prod[el]['yld_' + nm] = val
for el in kt25_prod:
    if kt25_prod[el]['yld_tot'][-1] <= 0:
        continue
    for ch in kt25_channels:
        y = kt25_prod[el]['yld_' + ch]
        kt25_prod[el]['t50_' + ch] = float(kt25_time[np.argmax(y > 0.5 * y[-1])].to(u.Gyr).value)
kt25_elements = [row['sym'] for row in solar_abd if 6 <= row['Z'] <= 60 and row['sym'] not in kt25_r_process and kt25_prod[row['sym']]['yld_tot'][-1] > 0]
kt25_mean_yield = np.array([[kt25_prod[e]['yld_' + c][-1] for c in kt25_channels] for e in kt25_elements])
kt25_delay = np.array([[kt25_prod[e]['t50_' + c] for c in kt25_channels] for e in kt25_elements])
slug_moments = np.load(KT25_REPO / 'slug_stochastic' / 'kt25_slug_yield_moments.npz', allow_pickle=False)
assert list(slug_moments['elements']) == kt25_elements, 'element ordering differs from the stochastic SLUG moments'
assert list(slug_moments['channels']) == kt25_channels
kt25_imf_cov_per_mass = slug_moments['imf_covariance_per_mass']
kt25_event_mass_mean = float(slug_moments['event_mass_mean'])
kt25_event_mass_second = float(slug_moments['event_mass_second'])
kt25_sigma_w2 = float(slug_moments['sigma_w2'])
kt25_second_yield_normalized = (kt25_event_mass_second * np.einsum('ia,jb->iajb', kt25_mean_yield, kt25_mean_yield)
                                + kt25_event_mass_mean * kt25_imf_cov_per_mass) / kt25_event_mass_second
_kms = (1 * u.km / u.s).to(u.kpc / u.Gyr).value
kt25_lcorr = 1.0
kt25_kappa = (150 * u.pc * 7 * u.km / u.s / 3).to(u.kpc**2 / u.Gyr).value
kt25_sigma_v = 2.0 * _kms
kt25_t_corr = kt25_lcorr**2 / kt25_kappa
kt25_Gamma = (3e-6 / u.Myr / u.pc**2).to(1 / u.Gyr / u.kpc**2).value
kt25_sigma_inj = np.array([0.060, 0.001, 0.060])
print(f'KT25: {len(kt25_elements)} elements, t_corr={kt25_t_corr:.3f} Gyr, kappa={kt25_kappa:.4f}, sigma_v={kt25_sigma_v:.4f}')

# %% Block 10: KT25 baseline and full-window KTZ prediction
def _drift_kernel(num, den, beta):
    """Age integral of eq. (J0_AB): int dS / [S U(S)] with U = 1 + beta/S over S in [den, num],
    where S is the summed per-axis variance of the two kernels and beta = sigma_v^2 dt^2 for the
    2D Gaussian drift of eq. (velocity_distribution). Equals ln[(num+beta)/(den+beta)]."""
    return np.log((num + beta) / (den + beta))

def _kt25_kernel_arcsinh(a, b, c):
    # published KT25 closed form (Krumholz & Ting 2025, eqs. 28-29, App. A10): Dawson/asinh drift average.
    # Used ONLY for the published KT25 reference; the KTZ prediction uses _drift_kernel (eq. J0_AB).
    return np.log(a) + 2 * b / np.sqrt(1 + b**2) * np.arcsinh(b) - 2 * c / np.sqrt(1 + c**2) * np.arcsinh(c)

def kt25_covariance_matrix(sigma_inj, channel_yields=None, channel_second_moment=None, kernel='J0'):
    if channel_second_moment is None:
        channel_second_moment = np.einsum('ia,jb->iajb', channel_yields, channel_yields)
    ne_ = len(kt25_elements); cov = np.zeros((ne_, ne_))
    for i in range(ne_):
        for j in range(i, ne_):
            val = 0.0
            for a in range(3):
                la2 = kt25_kappa * kt25_delay[i, a]
                for b in range(3):
                    ys = channel_second_moment[i, a, j, b]
                    if ys == 0:
                        continue
                    lb2 = kt25_kappa * kt25_delay[j, b]
                    s2 = sigma_inj[a]**2 + sigma_inj[b]**2
                    dd = abs(kt25_delay[i, a] - kt25_delay[j, b])
                    num = 4 * kt25_lcorr**2 - 2 * la2 - 2 * lb2 + s2
                    den = 2 * abs(la2 - lb2) + s2
                    if kernel == 'J0':
                        val += ys * _drift_kernel(num, den, (kt25_sigma_v * dd)**2)
                    else:
                        val += ys * _kt25_kernel_arcsinh(num / den, kt25_sigma_v * dd / np.sqrt(num), kt25_sigma_v * dd / np.sqrt(den))
            cov[i, j] = cov[j, i] = val
    return cov

def normalize_cross_covariance(c):
    v = np.diag(c); d = np.sqrt(np.outer(v, v))
    return np.divide(c, d, out=np.full_like(c, np.nan), where=d > 0)

# Published KT25 baseline (uniform sources, mean yields, no beam) with the published Dawson/asinh drift average
Xi_KT25 = normalize_cross_covariance(kt25_covariance_matrix(kt25_sigma_inj, channel_yields=kt25_mean_yield, kernel='arcsinh'))
# Same uniform baseline with the direct 2D drift average of eq. (J0_AB): the 'uniform KTZ limit', not KT25
Xi_uniform_KTZ_limit = normalize_cross_covariance(kt25_covariance_matrix(kt25_sigma_inj, channel_yields=kt25_mean_yield, kernel='J0'))
print('equal-delay check: J0 and published kernels agree when dt=0:',
      np.allclose(_drift_kernel(3.0, 0.5, 0.0), _kt25_kernel_arcsinh(3.0 / 0.5, 0.0, 0.0)))

dx_cross = 0.05
fwhm_beam_cross = 2.0 * dx_cross            # 0.10 kpc
sigma_beam_cross = fwhm_beam_cross * FWHM_TO_SIGMA
cross_sigma_obs = np.sqrt(kt25_sigma_inj**2 + sigma_beam_cross**2)   # beam added to every channel kernel
cross_margin = np.sqrt(2.0) * dx_cross
cross_R_in_a, cross_R_out_a = R_in + cross_margin, R_out - cross_margin
cross_rho2 = np.linspace(cross_R_in_a**2, cross_R_out_a**2, 31)
cross_varpi_nodes = np.sqrt(0.5 * (cross_rho2[:-1] + cross_rho2[1:]))
cross_w = np.exp(-cross_varpi_nodes / h_R)
cross_pref0 = 8 * np.pi * kt25_kappa * kt25_Gamma / (1 + kt25_sigma_w2)
ne = len(kt25_elements)
k_cross = np.geomspace(1e-4, 80.0, 1000)
# Thomas-process residual source spectrum of distinct parent events (Sigma_src = Gamma t_corr)
P_source = cluster_fraction * mean_children_per_cluster / (kt25_Gamma * kt25_t_corr) * np.exp(-(cluster_radius * k_cross)**2)

def _transfer_k(delay, sigma, k):
    if delay >= kt25_t_corr:
        return np.zeros_like(k)
    dur = kt25_t_corr - delay
    return np.exp(-0.5 * (sigma * k)**2) * np.exp(-0.5 * (kt25_sigma_v * delay * k)**2) * (-np.expm1(-kt25_kappa * dur * k**2) / (kt25_kappa * k**2))

def _cnp_matrix(sigma_inj):
    T = np.zeros((ne, len(k_cross)))
    for e in range(ne):
        for c in range(3):
            if kt25_mean_yield[e, c] != 0:
                T[e] += kt25_mean_yield[e, c] * _transfer_k(kt25_delay[e, c], sigma_inj[c], k_cross)
    sw = k_cross * P_source / (2 * np.pi)
    cov = np.zeros((ne, ne))
    for i in range(ne):
        for j in range(i, ne):
            cov[i, j] = cov[j, i] = cross_pref0 * simpson(sw * T[i] * T[j], x=k_cross)
    return cov

def ktz_window_components(sigma_inj):
    cp0 = kt25_covariance_matrix(sigma_inj, channel_second_moment=kt25_second_yield_normalized)
    cnp = _cnp_matrix(sigma_inj)
    comps = {k: np.zeros_like(cp0) for k in component_order}
    for v, w in zip(cross_varpi_nodes, cross_w):
        local_cm = np.zeros_like(cp0)
        for n, g, a in zip(harmonic_n, harmonic_g, harmonic_alpha):
            Q = n * k_perp(v); amp = eta * g * source_harmonic_damping(Q)
            mean_vec = np.array([sum(kt25_mean_yield[e, c] * _transfer_k(kt25_delay[e, c], sigma_inj[c], np.array([Q]))[0]
                                     for c in range(3) if kt25_mean_yield[e, c] != 0) for e in range(ne)])
            local_cm += 0.5 * cross_pref0 * w * np.outer(amp * mean_vec, amp * mean_vec)
        comps['CP0'] += w * cp0; comps['CNP'] += w * cnp; comps['CM'] += w * local_cm
    for k in comps:
        comps[k] /= cross_w.sum()
    return comps

cross_comps = ktz_window_components(cross_sigma_obs)
Xi_KTZ_obs = normalize_cross_covariance(sum(cross_comps.values()))
Xi_KTZ_obs_noCNP = normalize_cross_covariance(sum(v for k, v in cross_comps.items() if k != 'CNP'))
assert np.count_nonzero(cross_comps['CPN']) == 0

# %% Block 11: multi-element Monte Carlo realisations
cross_elements = ['C', 'N', 'O', 'Mg', 'Si', 'S', 'Ca', 'V', 'Mn', 'Fe', 'Ni', 'Sr', 'Y', 'Zr', 'Ba', 'Ce']
cross_idx = np.array([kt25_elements.index(e) for e in cross_elements])
n_cross_real = 1 if QUICK else 12
cross_age_bins = 40
cross_source_R_in = max(0.30, R_in - 6.0); cross_source_R_out = R_out + 6.0
cgx = np.arange(-cross_source_R_out, cross_source_R_out + dx_cross / 2, dx_cross)
NX = len(cgx)
cX, cY = np.meshgrid(cgx, cgx); cR = np.hypot(cX, cY)
c_disk = (cR >= R_in) & (cR <= R_out)
c_ana = (cR >= cross_R_in_a) & (cR <= cross_R_out_a)
NFFT = 1024
kfx = 2 * np.pi * np.fft.fftfreq(NFFT, d=dx_cross); kfy = 2 * np.pi * np.fft.rfftfreq(NFFT, d=dx_cross)
K2 = kfx[:, None]**2 + kfy[None, :]**2
narrow_limit = 2.0 * dx_cross   # kernels with s < 2 dx are evaluated exactly at the grid centres

def _radial_int(a, b):
    prim = lambda r: -h_R * (r + h_R) * np.exp(-r / h_R)
    return prim(b) - prim(a)

cross_expected = 2 * np.pi * kt25_Gamma * kt25_t_corr * _radial_int(cross_source_R_in, cross_source_R_out)
_rt = np.linspace(cross_source_R_in, cross_source_R_out, 20000)
_rc = np.array([_radial_int(cross_source_R_in, r) for r in _rt]); _rc /= _rc[-1]
_cmax = np.max(1.0 + eta * h_func(Theta_check))

def sample_parent(n, rng):
    rr, pp = np.empty(0), np.empty(0)
    while len(rr) < n:
        m_ = max(2 * (n - len(rr)), 256)
        cr = np.interp(rng.uniform(size=m_), _rc, _rt); cp = rng.uniform(0, 2 * np.pi, m_)
        acc = rng.uniform(size=m_) < (1 + eta * h_func(spiral_phase(cr, cp))) / _cmax
        rr = np.concatenate([rr, cr[acc]]); pp = np.concatenate([pp, cp[acc]])
    return rr[:n], pp[:n]

def sample_cross_clustered(rng):
    npar = rng.poisson(cluster_fraction * cross_expected / mean_children_per_cluster)
    pr, pp = sample_parent(npar, rng)
    cnt = rng.poisson(mean_children_per_cluster, npar); nc = int(cnt.sum())
    x = np.repeat(pr * np.cos(pp), cnt) + rng.normal(0, cluster_radius, nc)
    y = np.repeat(pr * np.sin(pp), cnt) + rng.normal(0, cluster_radius, nc)
    keep = (np.hypot(x, y) >= cross_source_R_in) & (np.hypot(x, y) <= cross_source_R_out)
    return x[keep], y[keep]

def sample_mass(n, rng):
    a, b = float(slug_moments['cmf_min']), float(slug_moments['cmf_max'])
    return 1.0 / (1.0 / a - rng.uniform(size=n) * (1.0 / a - 1.0 / b))

cmean = kt25_mean_yield[cross_idx]; cdelay = kt25_delay[cross_idx]
flat = np.array([e * 3 + c for e in cross_idx for c in range(3)])
cK = kt25_imf_cov_per_mass.reshape(ne * 3, ne * 3)[np.ix_(flat, flat)]; cK = 0.5 * (cK + cK.T)
ev, evec = np.linalg.eigh(cK); cK_fac = evec * np.sqrt(np.clip(ev, 0, None))

def sample_yields(mass, rng):
    res = rng.normal(size=(len(mass), len(flat))) @ cK_fac.T
    return (mass[:, None] * cmean.reshape(-1)[None, :] + np.sqrt(mass)[:, None] * res).reshape(len(mass), len(cross_elements), 3)

def deposit_exact(field, x, y, w, var):
    """Evaluate each continuous-position Gaussian of per-axis variance var at the grid centres."""
    half = int(np.ceil(5 * np.sqrt(var.max()) / dx_cross)) + 1
    off = np.arange(-half, half + 1)
    ix0 = np.rint((x - cgx[0]) / dx_cross).astype(int); iy0 = np.rint((y - cgx[0]) / dx_cross).astype(int)
    ix = ix0[:, None] + off[None, :]; iy = iy0[:, None] + off[None, :]
    gx = np.exp(-(cgx[np.clip(ix, 0, NX - 1)] - x[:, None])**2 / (2 * var[:, None])) * ((ix >= 0) & (ix < NX))
    gy = np.exp(-(cgx[np.clip(iy, 0, NX - 1)] - y[:, None])**2 / (2 * var[:, None])) * ((iy >= 0) & (iy < NX))
    vals = (w / (2 * np.pi * var))[:, None, None] * gy[:, :, None] * gx[:, None, :]
    lin = np.clip(iy, 0, NX - 1)[:, :, None] * NX + np.clip(ix, 0, NX - 1)[:, None, :]
    field += np.bincount(lin.ravel(), vals.ravel(), NX * NX).reshape(NX, NX)

def ngp(x, y, w):
    ix = np.rint((x - cgx[0]) / dx_cross).astype(int); iy = np.rint((y - cgx[0]) / dx_cross).astype(int)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NX)
    out = np.zeros((NFFT, NFFT))
    out[:NX, :NX] = np.bincount(iy[ok] * NX + ix[ok], w[ok], NX * NX).reshape(NX, NX)
    return out

# Common width bins (in s^2, uniform in log) per channel for the broad branch
s2_max = np.max(cross_sigma_obs**2) + 2 * kt25_kappa * kt25_t_corr
broad_edges = np.geomspace(narrow_limit**2, s2_max * 1.0001, cross_age_bins + 1)
broad_mid = np.sqrt(broad_edges[:-1] * broad_edges[1:])
assert NFFT - NX > 5 * np.sqrt(s2_max) / dx_cross   # periodic wrap never reaches the stored grid
cross_bg = np.zeros((len(cross_elements),) + cR.shape)
for e in range(len(cross_elements)):
    resp = sum(cmean[e, c] * (kt25_t_corr - cdelay[e, c]) for c in range(3) if cdelay[e, c] < kt25_t_corr)
    cross_bg[e] = kt25_Gamma * np.exp(-np.maximum(cR, cross_source_R_in) / h_R) * kt25_event_mass_mean * resp

def cross_realization(rng):
    n_p = rng.poisson((1 - cluster_fraction) * cross_expected)
    clx, cly = sample_cross_clustered(rng)
    pr, pp = sample_parent(n_p, rng)
    ex = np.concatenate([pr * np.cos(pp), clx]); ey = np.concatenate([pr * np.sin(pp), cly])
    perm = rng.permutation(len(ex)); ex, ey = ex[perm], ey[perm]; n = len(ex)
    age = rng.uniform(0, kt25_t_corr, n)
    vx = rng.normal(0, kt25_sigma_v, n); vy = rng.normal(0, kt25_sigma_v, n)
    yields = sample_yields(sample_mass(n, rng), rng)
    fields = np.zeros((len(cross_elements), NX, NX)); n_narrow = 0
    for e in range(len(cross_elements)):
        F = np.zeros((NFFT, NFFT // 2 + 1), complex)
        for c in range(3):
            d = cdelay[e, c]
            if d >= kt25_t_corr:
                continue
            tau = age - d; ret = tau > 0
            if not np.any(ret):
                continue
            x = ex[ret] + vx[ret] * d; y = ey[ret] + vy[ret] * d; w = yields[ret, e, c]
            var = cross_sigma_obs[c]**2 + 2 * kt25_kappa * tau[ret]
            nar = var < narrow_limit**2
            if np.any(nar):
                deposit_exact(fields[e], x[nar], y[nar], w[nar], var[nar]); n_narrow += nar.sum()
            b = np.clip(np.searchsorted(broad_edges, var[~nar], side='right') - 1, 0, cross_age_bins - 1)
            xb, yb, wb = x[~nar], y[~nar], w[~nar]
            for k in np.unique(b):
                s = b == k
                F += np.fft.rfft2(ngp(xb[s], yb[s], wb[s])) * np.exp(-0.5 * K2 * broad_mid[k]) / dx_cross**2
        fields[e] += np.fft.irfft2(F, s=(NFFT, NFFT))[:NX, :NX]
    resid = fields - cross_bg
    vals = resid[:, c_ana]
    return dict(cov=vals @ vals.T / vals.shape[1], n=n, n_narrow=int(n_narrow),
                display=dict(fields=fields[[2, 1]].astype(np.float32), resid=resid[[2, 1]].astype(np.float32)))

cache = OUT_DIR / 'cross_covariance_stack.npz'
crng = default_rng(20260625)
covs = []; disp = None
t0 = time.time()
for i in range(n_cross_real):
    rr_ = cross_realization(crng); covs.append(rr_['cov'])
    if disp is None:
        disp = rr_['display']
    print(f'cross realisation {i+1}/{n_cross_real}: {rr_["n"]:,} events, {rr_["n_narrow"]:,} exact narrow returns, {time.time()-t0:.0f}s', flush=True)
    np.savez_compressed(cache, covariance_stack=np.array(covs))
cov_stack = np.array(covs)
sim_cov = cov_stack.mean(0); Xi_sim = normalize_cross_covariance(sim_cov)
brng = default_rng(20260626)
boot = np.array([normalize_cross_covariance(cov_stack[brng.integers(0, n_cross_real, n_cross_real)].mean(0)) for _ in range(1000)])
Xi_sim_err = boot.std(0, ddof=1) if n_cross_real > 1 else np.full_like(Xi_sim, np.nan)

# %% Block 12: matrices and scatter figures; metrics
th_sub = Xi_KTZ_obs[np.ix_(cross_idx, cross_idx)]
th_sub_n = Xi_KTZ_obs_noCNP[np.ix_(cross_idx, cross_idx)]
kt_sub = Xi_KT25[np.ix_(cross_idx, cross_idx)]
diff = th_sub - kt_sub; lim = np.nanmax(np.abs(diff))
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), constrained_layout=True)
for ax, (M, title, cmap, norm_) in zip(axes, [
        (kt_sub, r'Published KT25 (uniform): $\Xi_{AB}$', 'viridis', None),
        (th_sub, r'Full KTZ (beam-matched): $\Xi_{AB}$', 'viridis', None),
        (diff, r'$\Delta\Xi_{AB}={\rm KTZ}-{\rm KT25}$', 'RdBu_r', TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))]):
    im = ax.imshow(M, origin='lower', cmap=cmap, vmin=None if norm_ else 0, vmax=None if norm_ else 1, norm=norm_)
    ax.set_xticks(range(16)); ax.set_yticks(range(16)); ax.set_xticklabels(cross_elements, rotation=90); ax.set_yticklabels(cross_elements)
    ax.set_title(title, fontsize=11); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
fig.savefig(figure_path('cross_matrices'), dpi=160, bbox_inches='tight'); plt.close(fig)

iu = np.triu_indices(16, 1)
s_, se_, t_, tn_ = Xi_sim[iu], Xi_sim_err[iu], th_sub[iu], th_sub_n[iu]
assert len(s_) == 120
chan = {'snii': 'CCSN', 'agb': 'AGB', 'snia': 'TNSN'}
main = np.array([chan[kt25_channels[i]] for i in np.argmax(np.abs(cmean), axis=1)])
grp = np.where(main[iu[0]] == main[iu[1]], main[iu[0]], 'Mixed source')
cols = {'CCSN': '#4c78a8', 'AGB': '#59a14f', 'TNSN': '#e15759', 'Mixed source': '#b07aa1'}
fig, ax = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
for lab in ['CCSN', 'AGB', 'TNSN', 'Mixed source']:
    s = grp == lab
    if not s.any():
        continue
    ax.errorbar(s_[s], t_[s], xerr=se_[s], fmt='none', ecolor=cols[lab], elinewidth=0.8, alpha=0.25, zorder=1)
    ax.scatter(s_[s], t_[s], s=25, color=cols[lab], alpha=0.8, edgecolors='none', label=f'{lab} ({s.sum()})', zorder=2)
lo_, hi_ = min(0, s_.min(), t_.min(), tn_.min()), max(1, s_.max(), t_.max())
ax.plot([lo_, hi_], [lo_, hi_], 'k--', lw=1.2, label='1:1')
ax.set_xlim(lo_ - 0.03, hi_ + 0.03); ax.set_ylim(lo_ - 0.03, hi_ + 0.03); ax.set_aspect('equal', adjustable='box')
ax.set_xlabel(r'Simulation $\Xi^{\rm sim}_{AB,\Omega}$'); ax.set_ylabel(r'KTZ prediction $\Xi^{\rm KTZ}_{AB,\Omega}$')
ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc='lower right')
fig.savefig(figure_path('cross_scatter'), dpi=160, bbox_inches='tight'); plt.close(fig)

dd = s_ - t_
METRICS['cross_setup'] = dict(dx_kpc=dx_cross, fwhm_beam_kpc=fwhm_beam_cross, sigma_beam_kpc=float(sigma_beam_cross),
                              q_AGB=float(dx_cross / np.sqrt(kt25_sigma_inj[1]**2 + sigma_beam_cross**2)),
                              q_CCSN=float(dx_cross / np.sqrt(kt25_sigma_inj[0]**2 + sigma_beam_cross**2)),
                              expected_parent_events=float(cross_expected), n_realisations=n_cross_real,
                              narrow_limit_kpc=narrow_limit, NFFT=NFFT)
METRICS['cross'] = dict(rmse=float(np.sqrt(np.mean(dd**2))), bias=float(dd.mean()), pearson=float(np.corrcoef(s_, t_)[0, 1]),
                        max_abs=float(np.abs(dd).max()), within_2sigma=int(np.sum(np.abs(dd) <= 2 * se_)))
ddn = s_ - tn_
METRICS['cross_noCNP'] = dict(rmse=float(np.sqrt(np.mean(ddn**2))), bias=float(ddn.mean()), max_abs=float(np.abs(ddn).max()))
METRICS['ktz_minus_kt25'] = dict(min=float(diff[iu].min()), max=float(diff[iu].max()),
                                 baseline='published KT25 Dawson/asinh closed form (KT25 eqs. 28-29, App. A10)')
_uk = Xi_uniform_KTZ_limit[np.ix_(cross_idx, cross_idx)]
METRICS['ktz_minus_uniform_ktz_limit'] = dict(min=float((th_sub - _uk)[iu].min()), max=float((th_sub - _uk)[iu].max()))
METRICS['uniform_ktz_limit_minus_kt25'] = dict(min=float((_uk - kt_sub)[iu].min()), max=float((_uk - kt_sub)[iu].max()),
                                               max_abs=float(np.max(np.abs((_uk - kt_sub)[iu]))))
# Diagnostic: the arcsinh closed form used by the earlier notebooks, which differs from eq. (J0_AB) when dt != 0
_cp0_as = kt25_covariance_matrix(cross_sigma_obs, channel_second_moment=kt25_second_yield_normalized, kernel='arcsinh')
_cp0_j0 = kt25_covariance_matrix(cross_sigma_obs, channel_second_moment=kt25_second_yield_normalized)
_tot_as = sum(cross_comps.values()) - cross_comps['CP0'] + _cp0_as * (cross_comps['CP0'][0, 0] / _cp0_j0[0, 0])
_th_as = normalize_cross_covariance(_tot_as)[np.ix_(cross_idx, cross_idx)][iu]
METRICS['cross_arcsinh_kernel_diagnostic'] = dict(rmse=float(np.sqrt(np.mean((s_ - _th_as)**2))), bias=float(np.mean(s_ - _th_as)),
    max_abs=float(np.max(np.abs(s_ - _th_as))),
    kt25_baseline_change_max=float(np.max(np.abs((Xi_uniform_KTZ_limit - Xi_KT25)[np.ix_(cross_idx, cross_idx)][iu]))))
np.savez(OUT_DIR / 'cross_results.npz', Xi_sim=Xi_sim, Xi_sim_err=Xi_sim_err, Xi_KTZ=th_sub, Xi_KTZ_noCNP=th_sub_n, Xi_KT25=kt_sub, Xi_uniform_KTZ_limit=_uk, elements=np.array(cross_elements))
print('cross', METRICS['cross']); print('cross_noCNP', METRICS['cross_noCNP']); print('KTZ-KT25', METRICS['ktz_minus_kt25'])
METRICS['runtime_s'] = time.time() - T_START
json.dump(METRICS, open(OUT_DIR / 'validation_metrics.json', 'w'), indent=1)
print(json.dumps(METRICS, indent=1))
