# %% [markdown]
# # Unified clustered-source validation of the KTZ statistic (Paper III, Section 3)
#
# One clustered-source Monte Carlo experiment tests the complete residual second moment, with
# stochastic covariance $C_{\rm conn}=C_{\rm U}+C_{\rm T}+C_{\rm D}$ and the complete deterministic mean.
# Half of the enrichment events are drawn in Thomas-process complexes whose parents follow the
# spiral-modulated intensity $\lambda(\boldsymbol x)$.
#
# Simulation and prediction use the same operators of Sec. 2.3.4 (2026-11-03 unified treatment):
# response $\boldsymbol\Psi$ (Gaussian beam ${\rm FWHM}_b=2\Delta x$, sampled at the valid cell centres),
# annular projection $\mathbf P$ with the annular mean estimated from every realisation on the valid cells,
# sums over the unordered valid-cell pairs ${\cal P}_b$ of each separation bin, and normalisation last:
# $\xi_b=\sum_{{\cal P}_b}S_{ij}/[\tfrac12\sum_{{\cal P}_b}(S_{ii}+S_{jj})]$ with
# $\mathbf S=\mathbf P\boldsymbol\Psi C_{\rm conn}\boldsymbol\Psi^{\sf T}\mathbf P^{\sf T}+\bs m\bs m^{\sf T}$, $\bs m=\mathbf P\boldsymbol\Psi\mu$.
# The prediction uses legal source-position (Gram) integrals over the finite source support, evaluated
# exactly in polar harmonics (validation/ktz_unified_ops.py, validation/ktz_unified_scalar.py).
# The phase-restricted window and the fixed-midpoint-radius check (appendix) are the same estimator with
# a different pair window.  The earlier known-profile outputs in validation/outputs are kept for audit.
#
# Multi-element same-event term: direct two-dimensional Gaussian drift average (eq. J0_AB); the
# reference labelled KT25 is the published Dawson/asinh closed form.  The zero-lag statistic now uses the
# same annular projection of every element map and of the transported mean (validation/ktz_unified_cross.py).

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN_TAG = os.environ.get('KTZ_RUN_TAG', 'rework_20261104')
FIG_DIR = ROOT / 'validation' / f'figs_{RUN_TAG}'; FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / 'validation' / f'outputs_{RUN_TAG}'; OUT_DIR.mkdir(parents=True, exist_ok=True)
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

# %% Block 4: valid cells, annular projection P and pair windows P_b (shared by simulation and prediction)
from validation import ktz_unified_ops as uop
from validation.ktz_unified_scalar import ScalarSetup, build_prediction

disk_mask = (R_grid >= R_in) & (R_grid <= R_out)
window_margin = np.sqrt(2.0) * dx
analysis_R_in = R_in + window_margin
analysis_R_out = R_out - window_margin
valid = disk_mask & (R_grid >= analysis_R_in) & (R_grid <= analysis_R_out)
cell_iy, cell_ix = np.nonzero(valid)
cell_x, cell_y = X[valid], Y[valid]
n_cells = cell_x.size
# Annular projection: equal-width annuli of about 3.4 cells across the analysis window
# (the observational maps use 0.2- and 0.4-kpc annuli on 60- and 120-pc cells, i.e. 3.3 cells).
n_annuli = 17
annulus_edges = np.linspace(analysis_R_in, analysis_R_out, n_annuli + 1)
annulus_edges[-1] += 1e-9
cell_ann = uop.annulus_labels(np.hypot(cell_x, cell_y), annulus_edges)
# Separation bins and pair windows.  Window 0: every unordered pair of valid cells (the final xi of
# Sec. 2.3.4); window 1: arm-phase sector (midpoint phase, midpoint radius and r <= 1.5 kpc); windows
# 2..7: one-cell-wide fixed midpoint-radius windows (radial check of the appendix).
radii_test = np.round(np.linspace(analysis_R_in + 0.5, analysis_R_out - 0.5, 6), 2)
midpoint_annulus_halfwidth = 0.5 * dx
r_bins = np.linspace(0.1, R_out, 30)
r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
sector_last_bin = np.flatnonzero(r_centers <= sector_validation_rmax)[-1]
window_par = np.array([sector_analysis_R_in, sector_analysis_R_out, sector_phase_halfwidth, sector_last_bin,
                       m_arms, m_arms / np.tan(pitch_rad), R_in, Theta0], float)
n_windows = 2 + radii_test.size
W_FULL, W_SECTOR = 0, 1
NTHREADS = int(os.environ.get('KTZ_THREADS', '2'))
METRICS['unified_operators'] = dict(
    n_valid_cells=int(n_cells), n_annuli=n_annuli, annulus_width_kpc=float(np.diff(annulus_edges[:2])[0]),
    response='Gaussian beam FWHM_b = 2 dx, sampled at valid cell centres (no cell average)',
    full_window='both endpoints valid; no midpoint cut', threads=NTHREADS,
    sector_window=dict(R_m=[sector_analysis_R_in, sector_analysis_R_out], phase_halfwidth_deg=30.0, r_max_kpc=sector_validation_rmax),
    radial_check_radii=radii_test.tolist(), radial_check_halfwidth_kpc=midpoint_annulus_halfwidth)
print(METRICS['unified_operators'], flush=True)

def map_values(Zmap):
    return Zmap[valid]

def projected_pair_statistic(Zmap):
    """Data estimator of Sec. 2.3.4: P on the valid cells, then pair sums over every window, then ratio."""
    r = uop.project(map_values(Zmap), cell_ann, n_annuli)
    num, den, cnt = uop.run_rows(uop.pair_products, n_cells, NTHREADS, cell_x, cell_y, r[None], (r**2)[None],
                                 r_bins, window_par, radii_test, midpoint_annulus_halfwidth, n_windows)
    return num[0], den[0], cnt

# %% Block 5: simulated maps (seed 42, regenerated in the original order) and their projected statistic
fields_cache = OUT_DIR / 'scalar_fields_raw.npz'
if fields_cache.exists():
    _fc = np.load(fields_cache)
    Z_stack = _fc['Z']; N_events_stack = _fc['N_events']
    assert Z_stack.shape[0] == n_realizations
    print(f'loaded {n_realizations} archived raw fields from {fields_cache.name}', flush=True)
else:
    Z_stack = [Z]; N_events_stack = [N_events]
    for ireal in range(1, n_realizations):
        Zi, Ni = generate_synthetic_galaxy(rng)
        Z_stack.append(Zi); N_events_stack.append(Ni)
        print(f'generated realisation {ireal+1}/{n_realizations}', flush=True)
    Z_stack = np.array(Z_stack); N_events_stack = np.array(N_events_stack)
    _tmp = OUT_DIR / 'scalar_fields_raw.tmp.npz'
    np.savez_compressed(_tmp, Z=Z_stack, N_events=N_events_stack, seed=42, grid_x=grid_x)
    os.replace(_tmp, fields_cache)

# Audit: the regenerated fields reproduce the archived statistic of the earlier known-profile prescription
legacy_file = ROOT / 'validation' / 'outputs' / 'scalar_simulation_statistics.npz'
def legacy_full_window(Zmap):
    """Earlier prescription (known axisymmetric profile, midpoint cut) for the field audit only."""
    Z_bg = Y_val * lambda0_0 * t_star * np.exp(-R_grid / h_R)
    r = (Zmap - Z_bg)[valid]
    wp = window_par.copy(); wp[0], wp[1] = analysis_R_in, analysis_R_out; wp[2] = 4.0; wp[3] = len(r_centers)
    num, den, cnt = uop.run_rows(uop.pair_products, n_cells, NTHREADS, cell_x, cell_y, r[None], (r**2)[None],
                                 r_bins, wp, np.zeros(0), 0.0, 2)
    return num[0, 1] / den[0, 1]   # window 1 here = midpoint inside the analysis window
if legacy_file.exists() and not QUICK:
    _old = np.load(legacy_file)['xi3']
    _aud = [float(np.nanmax(np.abs(legacy_full_window(Z_stack[k]) - _old[k]))) for k in (0, n_realizations - 1)]
    METRICS['field_audit_max_abs_vs_archived_xi3'] = dict(realisation_1=_aud[0], realisation_last=_aud[1])
    print('field audit vs archived known-profile xi3:', _aud, flush=True)
    assert max(_aud) < 1e-10, 'regenerated fields differ from the archived realisations'

sim_num, sim_den, sim_cnt = [], [], []
for ireal in range(n_realizations):
    t0 = time.time()
    nm_, dn_, ct_ = projected_pair_statistic(Z_stack[ireal])
    sim_num.append(nm_); sim_den.append(dn_); sim_cnt = ct_
    print(f'projected statistic {ireal+1}/{n_realizations}: {time.time()-t0:.0f}s', flush=True)
sim_num = np.array(sim_num); sim_den = np.array(sim_den)
with np.errstate(invalid='ignore', divide='ignore'):
    sim_xi = sim_num / sim_den                      # (realisation, window, bin)
sim_mean = np.nanmean(sim_xi, 0); sim_std = np.nanstd(sim_xi, 0, ddof=1); sim_sem = sim_std / np.sqrt(n_realizations)
np.savez(OUT_DIR / 'scalar_simulation_statistics_unified.npz', r_bins=r_bins, radii=radii_test,
         windows=np.array(['full', 'sector'] + [f'fixed_radius_{v}' for v in radii_test]),
         num=sim_num, den=sim_den, counts=sim_cnt, xi=sim_xi, annulus_edges=annulus_edges)

# %% Block 6: KTZ prediction with the same Psi, P and P_b (legal source-position integrals)
setup = ScalarSetup(dict(R_in=R_in, R_out=R_out, kappa=kappa, sigma=sigma_scalar, t_star=t_star, sigma_b=sigma_beam,
                         cell_var=0.0, m_arms=m_arms, pitch_deg=pitch_angle, Theta0=Theta0, eta=eta,
                         harm_n=list(harmonic_n), harm_g=list(harmonic_g), harm_alpha=list(harmonic_alpha),
                         lambda00=lambda0_0, h_R=h_R, f_cl=cluster_fraction, sigma_cl=cluster_radius,
                         N_ch=mean_children_per_cluster, dx=dx, grid0=grid_x[0], ngrid=grid_x.size,
                         rho_max=analysis_R_out + 0.05))
PRED_NODES = dict(n_age=int(os.environ.get('KTZ_NAGE', 16)), n_age_D=int(os.environ.get('KTZ_NAGED', 16)))
pred_cache = OUT_DIR / f"scalar_prediction_unified_a{PRED_NODES['n_age']}_d{PRED_NODES['n_age_D']}.npz"
if pred_cache.exists():
    _pc = np.load(pred_cache)
    pred = dict(comp={k: _pc[k] for k in ('CU', 'CT', 'CD', 'CDmod', 'SM')}, den=_pc['den'], den_dmod=_pc['den_dmod'],
                pdiag=_pc['pdiag'], m=_pc['m'], mu=_pc['mu'], mu_ax=_pc['mu_ax'], cnt=_pc['cnt'])
    print(f'loaded prediction {pred_cache.name}', flush=True)
else:
    pred = build_prediction(setup, cell_x, cell_y, cell_ix.astype(np.int64), cell_iy.astype(np.int64), cell_ann, n_annuli,
                            r_bins, window_par, radii_test, midpoint_annulus_halfwidth, n_windows,
                            nthreads=NTHREADS, do_dmod=True, **PRED_NODES)
    _tmp = OUT_DIR / 'scalar_prediction.tmp.npz'
    np.savez(_tmp, **pred['comp'], den=pred['den'], den_dmod=pred['den_dmod'], pdiag=pred['pdiag'], m=pred['m'],
             mu=pred['mu'], mu_ax=pred['mu_ax'], cnt=pred['cnt'], **{f'nodes_{k}': v for k, v in PRED_NODES.items()})
    os.replace(_tmp, pred_cache)
    METRICS['prediction_runtime_s'] = pred['runtime']
assert np.array_equal(pred['cnt'], sim_cnt), 'prediction and data use different pair sets'
component_order = ['CU', 'CT', 'SM', 'CD']
component_colors = {'CU': '#7fc97f', 'CT': '#ffff99', 'SM': '#fdc086', 'CD': '#beaed4'}
component_labels = {'CU': r'$C_{\rm U}$', 'CT': r'$C_{\rm T}$', 'SM': r'$\mathbf{S}_{\rm M}$', 'CD': r'$C_{\rm D}$'}
with np.errstate(invalid='ignore', divide='ignore'):
    th = {k: pred['comp'][k] / pred['den'] for k in component_order}
    th_total = sum(th[k] for k in component_order)
# prediction without C_D: same pair sums with the C_D numerator and endpoint terms removed
pd_ = pred['pdiag']
_vals = np.stack([pd_[2]])
_, den_D, _ = uop.run_rows(uop.pair_products, n_cells, NTHREADS, cell_x, cell_y, np.zeros((1, n_cells)), _vals,
                           r_bins, window_par, radii_test, midpoint_annulus_halfwidth, n_windows)
with np.errstate(invalid='ignore', divide='ignore'):
    th_noD = (pred['comp']['CU'] + pred['comp']['CT'] + pred['comp']['SM']) / (pred['den'] - den_D[0])
    th_dmod = (pred['comp']['CU'] + pred['comp']['CT'] + pred['comp']['SM'] + pred['comp']['CDmod']) / pred['den_dmod']
# zero-lag anchors: component shares of the window-averaged endpoint second moment
_d0 = {'CU': pd_[0], 'CT': pd_[1], 'SM': pred['m']**2, 'CD': pd_[2]}
METRICS['component_shares_r0_full'] = {k: float(_d0[k].sum() / sum(v.sum() for v in _d0.values())) for k in component_order}

# %% Block 7: field figure (simulated field, annular mean estimated from this map, projected residual)
Z_plot = np.where(disk_mask, Z_stack[0], np.nan)
ann_mean_map = np.full(Z_plot.shape, np.nan)
_v = map_values(Z_stack[0]); _am = np.bincount(cell_ann, _v, n_annuli) / np.bincount(cell_ann, minlength=n_annuli)
ann_mean_map[valid] = _am[cell_ann]
res_map = np.full(Z_plot.shape, np.nan); res_map[valid] = _v - _am[cell_ann]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
vmin, vmax = np.nanpercentile(Z_plot, [1, 99])
for ax, field, lab, title in [(axes[0], Z_plot, r'$\Sigma_X$', 'Simulation'),
                             (axes[1], ann_mean_map, r'annular mean', 'Annular mean of this map')]:
    im = ax.pcolormesh(X, Y, field, cmap='plasma', shading='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=lab); ax.set_title(title)
lim = np.nanpercentile(np.abs(res_map[valid]), 99.5)
im = axes[2].pcolormesh(X, Y, res_map, cmap='RdBu_r', shading='auto', vmin=-lim, vmax=lim)
plt.colorbar(im, ax=axes[2], label=r'$(\mathbf{P}\Sigma_X)_i$')   # Sigma_X is already beam-convolved and centre-sampled
axes[2].set_title(r'Projected residual used for $\xi$')
for ax in axes:
    for rr in (analysis_R_in, analysis_R_out):
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, ls='--', lw=0.8, color='0.25'))
    ax.set_xlim(-R_out, R_out); ax.set_ylim(-R_out, R_out); ax.set_aspect('equal'); ax.set_xlabel(r'$x\;({\rm kpc})$')
axes[0].set_ylabel(r'$y\;({\rm kpc})$')
plt.tight_layout(); fig.savefig(figure_path('unified_fields'), dpi=160, bbox_inches='tight'); plt.close(fig)
np.savez_compressed(OUT_DIR / 'fig_unified_fields.npz', X=X, Y=Y, Z=Z_plot, annular_mean=ann_mean_map, residual=res_map,
                    analysis_R=np.array([analysis_R_in, analysis_R_out]))
METRICS['scalar_fields_unified'] = dict(
    projected_residual_rms=float(np.nanstd(res_map[valid])),
    projected_mean_rms=float(np.std(pred['m'])),
    projected_mean_axisymmetric_residual_rms=float(np.std(uop.project(pred['mu_ax'], cell_ann, n_annuli))))

# %% Block 8: comparisons (full window, arm sector, fixed-radius check); metrics
def summarize(sim, th_, unc):
    sim, th_, unc = map(np.asarray, (sim, th_, unc))
    ok = np.isfinite(sim) & np.isfinite(th_) & np.isfinite(unc)
    d = sim[ok] - th_[ok]
    return dict(n=int(ok.sum()), rmse=float(np.sqrt(np.mean(d**2))), bias=float(np.mean(d)),
                pearson=float(np.corrcoef(sim[ok], th_[ok])[0, 1]), max_abs=float(np.max(np.abs(d))),
                within_1sigma=int(np.sum(np.abs(d) <= unc[ok])))

def stack_plot(ax, r, comp):
    cum = np.zeros_like(r); cumn = np.zeros_like(r)
    for k in component_order:
        v = comp[k]
        pos = np.clip(v, 0, None); neg = np.clip(v, None, 0)
        ax.fill_between(r, cum, cum + pos, color=component_colors[k], alpha=0.42, lw=0, label=component_labels[k], zorder=1)
        ax.fill_between(r, cumn, cumn + neg, color=component_colors[k], alpha=0.42, lw=0, zorder=1)
        cum = cum + pos; cumn = cumn + neg

def comparison_panel(ax, w, rmax, title=None, legend=True, r0=True):
    sel = r_centers <= rmax
    r = r_centers[sel]
    comp = {k: th[k][w][sel] for k in component_order}
    if r0:
        r = np.concatenate([[0.0], r])
        comp = {k: np.concatenate([[METRICS['component_shares_r0_full'][k] if w == W_FULL else np.nan], comp[k]]) for k in comp}
    stack_plot(ax, r, comp)
    pad = (lambda a, v: np.concatenate([[v], a])) if r0 else (lambda a, v: a)
    sm, sd, se = pad(sim_mean[w][sel], 1.0), pad(sim_std[w][sel], 0.0), pad(sim_sem[w][sel], 0.0)
    ax.plot(r, sm, 'o-', color='k', ms=4, label='Simulation mean', zorder=4)
    ax.fill_between(r, sm - sd, sm + sd, color='k', alpha=0.14, lw=0, label=r'Single-realisation $1\sigma$', zorder=3)
    ax.errorbar(r, sm, yerr=se, fmt='none', ecolor='k', elinewidth=0.9, capsize=2, label=r'Mean $1\sigma$ uncertainty', zorder=5)
    ax.plot(r, pad(th_total[w][sel], 1.0), '-', color='#d62728', lw=1.6, label='Full KTZ', zorder=6)
    ax.plot(r, pad(th_noD[w][sel], 1.0), ls='--', color='#6a3d9a', lw=1.6, label=r'KTZ without $C_{\rm D}$', zorder=6)
    ax.axhline(0, color='grey', lw=0.5); ax.set_xlabel('r (kpc)'); ax.set_xlim(0, rmax); ax.grid(alpha=0.25)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(fontsize=7.5)

xm = r_centers <= validation_rmax
smask = r_centers <= sector_validation_rmax
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
comparison_panel(axes[0], W_FULL, validation_rmax, title='Full window')
axes[0].set_ylabel(r'$\xi_b$')
comparison_panel(axes[1], W_SECTOR, sector_validation_rmax, legend=False, r0=False,
                 title=rf'Arm-phase sector: $|\Theta_m|<30^\circ$, ${sector_analysis_R_in:.1f}<\varpi_m/{{\rm kpc}}<{sector_analysis_R_out:.1f}$')
plt.tight_layout(); fig.savefig(figure_path('unified_xi'), dpi=160, bbox_inches='tight'); plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True); axes = axes.flatten()
for i, (ax, vm) in enumerate(zip(axes, radii_test)):
    comparison_panel(ax, 2 + i, validation_rmax, title=fr'$|\varpi_m-{vm}\,{{\rm kpc}}|<\Delta x/2$', legend=(i == 0), r0=False)
axes[0].set_ylabel(r'$\xi_b$'); axes[3].set_ylabel(r'$\xi_b$')
plt.tight_layout(); fig.savefig(figure_path('unified_radial_check'), dpi=160, bbox_inches='tight'); plt.close(fig)

METRICS['xi_full'] = summarize(sim_mean[W_FULL][xm], th_total[W_FULL][xm], sim_sem[W_FULL][xm])
METRICS['xi_full_noCD'] = summarize(sim_mean[W_FULL][xm], th_noD[W_FULL][xm], sim_sem[W_FULL][xm])
METRICS['sector'] = summarize(sim_mean[W_SECTOR][smask], th_total[W_SECTOR][smask], sim_sem[W_SECTOR][smask])
METRICS['sector_noCD'] = summarize(sim_mean[W_SECTOR][smask], th_noD[W_SECTOR][smask], sim_sem[W_SECTOR][smask])
METRICS['sector_modulated_CD_diagnostic'] = summarize(sim_mean[W_SECTOR][smask], th_dmod[W_SECTOR][smask], sim_sem[W_SECTOR][smask])
METRICS['sector']['max_abs_CT'] = float(np.nanmax(np.abs(th['CT'][W_SECTOR][smask])))
METRICS['xi_full']['max_abs_CT'] = float(np.nanmax(np.abs(th['CT'][W_FULL][xm])))
_rs = lambda f: np.concatenate([f[2 + i][xm] for i in range(radii_test.size)])
METRICS['radial_check'] = summarize(_rs(sim_mean), _rs(th_total), _rs(sim_sem))
METRICS['radial_check_noCD'] = summarize(_rs(sim_mean), _rs(th_noD), _rs(sim_sem))
METRICS['radial_check_modulated_CD_diagnostic'] = summarize(_rs(sim_mean), _rs(th_dmod), _rs(sim_sem))
METRICS['xi_full_modulated_CD_diagnostic'] = summarize(sim_mean[W_FULL][xm], th_dmod[W_FULL][xm], sim_sem[W_FULL][xm])
METRICS['radial_check_per_radius_modulated_CD_diagnostic'] = {str(v): summarize(sim_mean[2 + i][xm], th_dmod[2 + i][xm], sim_sem[2 + i][xm])
                                                              for i, v in enumerate(radii_test)}
METRICS['radial_check_per_radius'] = {str(v): summarize(sim_mean[2 + i][xm], th_total[2 + i][xm], sim_sem[2 + i][xm])
                                      for i, v in enumerate(radii_test)}
METRICS['max_abs_CD_contribution'] = dict(full=float(np.nanmax(np.abs(th['CD'][W_FULL][xm]))),
                                          sector=float(np.nanmax(np.abs(th['CD'][W_SECTOR][smask]))))
np.savez(OUT_DIR / 'scalar_theory_unified.npz', r_centers=r_centers, radii=radii_test, total=th_total, noCD=th_noD,
         CDmod_total=th_dmod, **{k: th[k] for k in component_order}, den=pred['den'], counts=pred['cnt'])
for k in ['xi_full', 'xi_full_noCD', 'sector', 'sector_noCD', 'sector_modulated_CD_diagnostic', 'radial_check', 'radial_check_noCD']:
    print(k, METRICS[k], flush=True)
json.dump(METRICS, open(OUT_DIR / 'validation_metrics_unified.json', 'w'), indent=1)
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
legacy_cross_keys = ['CP0', 'CPN', 'CM', 'CNP']

def source_harmonic_damping(Qn):
    return (1.0 - cluster_fraction) + cluster_fraction * np.exp(-0.5 * (Qn * cluster_radius)**2)

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
    comps = {k: np.zeros_like(cp0) for k in legacy_cross_keys}
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

# Earlier local window prediction (known axisymmetric profile, no annular projection): kept for audit only
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
    resid = fields - cross_bg                      # earlier known-profile residual (audit only)
    vals_old = resid[:, c_ana]
    vals = uop.project(fields[:, cross_valid], cross_ann, cross_n_annuli)   # P Psi Sigma_A on the valid cells
    return dict(cov=vals @ vals.T / vals.shape[1], cov_legacy=vals_old @ vals_old.T / vals_old.shape[1],
                vals=vals.astype(np.float32), n=n, n_narrow=int(n_narrow),
                display=dict(fields=fields[[2, 1]].astype(np.float32)))

# %% Block 11b: unified operators for the multi-element maps (valid cells, annuli, projected theory)
from validation import ktz_unified_cross as ucx


cross_valid = c_ana
cvx, cvy = cX[cross_valid], cY[cross_valid]; cvr = np.hypot(cvx, cvy); cvp = np.arctan2(cvy, cvx)
n_cross_cells = cvx.size
cross_n_annuli = int(round((cross_R_out_a - cross_R_in_a) / (10.0 / 3.0 * dx_cross)))
cross_ann_edges = np.linspace(cross_R_in_a, cross_R_out_a, cross_n_annuli + 1); cross_ann_edges[-1] += 1e-9
cross_ann = uop.annulus_labels(cvr, cross_ann_edges)
cross_n_a = np.bincount(cross_ann, minlength=cross_n_annuli).astype(float)
METRICS['cross_unified_operators'] = dict(n_valid_cells=int(n_cross_cells), n_annuli=cross_n_annuli,
                                          annulus_width_kpc=float(np.diff(cross_ann_edges[:2])[0]),
                                          response='Gaussian beam FWHM_b = 2 dx_cross, sampled at valid cell centres')
cross_theory_cache = OUT_DIR / 'cross_theory_unified.npz'
ce = cross_idx; nce = len(cross_elements)

def channel_pair_params(i, j, sigma_inj, second):
    ys, nu, de, be = [], [], [], []
    for a in range(3):
        la2 = kt25_kappa * kt25_delay[i, a]
        for b in range(3):
            y_ = second[i, a, j, b]
            if y_ == 0:
                continue
            lb2 = kt25_kappa * kt25_delay[j, b]
            s2 = sigma_inj[a]**2 + sigma_inj[b]**2
            dd = abs(kt25_delay[i, a] - kt25_delay[j, b])
            ys.append(y_); nu.append(4 * kt25_lcorr**2 - 2 * la2 - 2 * lb2 + s2); de.append(2 * abs(la2 - lb2) + s2)
            be.append((kt25_sigma_v * dd)**2)
    return np.array(ys), np.array(nu), np.array(de), np.array(be)

if cross_theory_cache.exists():
    _ct = np.load(cross_theory_cache)
    cross_U, cross_D, cross_M = _ct['U'], _ct['D'], _ct['M']
    METRICS['cross_theory_diagnostics'] = json.loads(str(_ct['diag']))
    print('loaded', cross_theory_cache.name, flush=True)
else:
    t0 = time.time()
    ann2d = np.full(cR.shape, -1, int); ann2d[cross_valid] = cross_ann
    varpi_a = np.bincount(cross_ann, cvr, cross_n_annuli) / cross_n_a
    Hw, _npairs = ucx.annulus_offset_histograms(cross_valid, ann2d, cross_n_annuli, dx_cross, np.exp(-varpi_a / h_R) / cross_n_a)
    assert _npairs == np.sum(cross_n_a**2), 'annulus pair counts'
    nz = np.flatnonzero(Hw); r_n = np.sqrt(nz) * dx_cross; Hw_nz = Hw[nz]
    lam_sum = np.exp(-cvr / h_R).sum()
    print(f'annulus offset histograms: {len(nz)} distinct separations ({time.time()-t0:.0f}s)', flush=True)
    # same-parent term: local zero-lag kernel at each cell minus the annular-mean correction
    cross_U = np.zeros((nce, nce)); U_diag = np.zeros((nce, nce))
    for p_, i in enumerate(ce):
        for q_, j in enumerate(ce):
            if q_ < p_:
                continue
            prm = channel_pair_params(i, j, cross_sigma_obs, kt25_second_yield_normalized)
            k0 = ucx.same_parent_kernel_r(np.array([0.0]), *prm)[0]
            ring = np.sum(Hw_nz * ucx.same_parent_kernel_r(r_n, *prm))
            U_diag[p_, q_] = U_diag[q_, p_] = lam_sum * k0
            cross_U[p_, q_] = cross_U[q_, p_] = (lam_sum * k0 - ring) / n_cross_cells
    print(f'same-parent projected term ({time.time()-t0:.0f}s)', flush=True)
    # distinct-parent term: local isotropic kernel C_D(r) (Hankel transform of the Thomas spectrum)
    kq = np.linspace(0.0, 12.0, 6001)
    Psrc = cluster_fraction * mean_children_per_cluster / (kt25_Gamma * kt25_t_corr) * np.exp(-(cluster_radius * kq)**2)
    Tq = np.zeros((nce, kq.size))
    for p_, e in enumerate(ce):
        for c in range(3):
            if kt25_mean_yield[e, c] != 0:
                Tq[p_] += kt25_mean_yield[e, c] * np.nan_to_num(_transfer_k(kt25_delay[e, c], cross_sigma_obs[c], np.maximum(kq, 1e-12)))
    r_tab = np.arange(0.0, r_n.max() + 0.05, 0.01)
    J = ucx.j0(np.outer(r_tab, kq)) * (kq * Psrc / (2 * np.pi))[None, :]
    wq = np.full(kq.size, kq[1] - kq[0]); wq[[0, -1]] *= 0.5
    cross_D = np.zeros((nce, nce)); D0 = np.zeros((nce, nce))
    for p_ in range(nce):
        for q_ in range(p_, nce):
            kd = cross_pref0 * (J * (Tq[p_] * Tq[q_] * wq)[None, :]).sum(1)
            ring = np.sum(Hw_nz * uop.cubic_uniform(r_tab, kd, r_n))
            D0[p_, q_] = D0[q_, p_] = kd[0]
            cross_D[p_, q_] = cross_D[q_, p_] = (lam_sum * kd[0] - ring) / n_cross_cells
    del J
    print(f'distinct-parent projected term ({time.time()-t0:.0f}s)', flush=True)
    # complete mean: exact smoothing of the one-point rate on the padded support, drift and delays
    nodes_c = np.concatenate([uop.radial_nodes(cross_source_R_in, 3.0, 0.004, 6)[0], uop.radial_nodes(3.0, cross_source_R_out, 0.02, 6)[0]])
    wts_c = np.concatenate([uop.radial_nodes(cross_source_R_in, 3.0, 0.004, 6)[1], uop.radial_nodes(3.0, cross_source_R_out, 0.02, 6)[1]])
    lam0n = np.exp(-nodes_c / h_R)
    Phin = (m_arms / np.tan(pitch_rad)) * np.log(nodes_c / R_in)
    lamh = {0: lam0n.astype(complex)}
    for n_, g, a in zip(harmonic_n, harmonic_g, harmonic_alpha):
        lamh[int(n_ * m_arms)] = eta * g * lam0n * np.exp(1j * (n_ * (Phin + Theta0) + a))
    lamh_cl = {k: uop.smoothing_matrix(k, cluster_radius**2, nodes_c, nodes_c, wts_c) @ c for k, c in lamh.items()}
    lamh1 = {k: (1 - cluster_fraction) * lamh[k] + cluster_fraction * lamh_cl[k] for k in lamh}
    print(f'one-point rate on {nodes_c.size} padded-support nodes ({time.time()-t0:.0f}s)', flush=True)
    rho_out_c = np.arange(cross_R_in_a - 0.05, cross_R_out_a + 0.05, 0.005)
    Vlist = []
    for e in ce:
        for c in range(3):
            if kt25_mean_yield[e, c] != 0 and kt25_delay[e, c] < kt25_t_corr:
                v0 = cross_sigma_obs[c]**2 + (kt25_sigma_v * kt25_delay[e, c])**2
                Vlist += [v0, v0 + 2 * kt25_kappa * (kt25_t_corr - kt25_delay[e, c])]
    lnV_nodes = np.linspace(np.log(min(Vlist)) - 0.3, np.log(max(Vlist)) + 0.3, int(os.environ.get('KTZ_CROSS_NV', 80)))
    mtab = ucx.transported_mean_tables(lamh1, nodes_c, wts_c, rho_out_c, np.exp(lnV_nodes))
    print(f'mean smoothing tables for {lnV_nodes.size} variances ({time.time()-t0:.0f}s)', flush=True)
    xg, xw = np.polynomial.legendre.leggauss(int(os.environ.get('KTZ_CROSS_NTAU', 24)))
    mu_hat = np.zeros((nce, n_cross_cells))
    for p_, e in enumerate(ce):
        acc = {k: np.zeros(rho_out_c.size, complex) for k in mtab}
        for c in range(3):
            if kt25_mean_yield[e, c] == 0 or kt25_delay[e, c] >= kt25_t_corr:
                continue
            v0 = cross_sigma_obs[c]**2 + (kt25_sigma_v * kt25_delay[e, c])**2
            v1 = v0 + 2 * kt25_kappa * (kt25_t_corr - kt25_delay[e, c])
            lo, hi = np.log(v0), np.log(v1)
            for x_, w_ in zip(xg, xw):
                lv = 0.5 * (hi - lo) * x_ + 0.5 * (hi + lo); V_ = np.exp(lv)
                dtau = 0.5 * (hi - lo) * w_ * V_ / (2 * kt25_kappa)
                tb = ucx.interp_lnV(mtab, lnV_nodes, lv)
                for k in acc:
                    acc[k] += kt25_mean_yield[e, c] * dtau * tb[k]
        mu_hat[p_] = uop.eval_harmonics(acc, rho_out_c, cvr, cvp)
    m_hat = uop.project(mu_hat, cross_ann, cross_n_annuli)
    cross_M = cross_pref0 * (m_hat @ m_hat.T) / n_cross_cells
    # size of the discrete-ring residual of the linear harmonic term (complete-annulus cancellation)
    th_c = spiral_phase(cvr, cvp)
    lin = max(abs(np.sum(np.exp(-cvr / h_R) * np.cos(n_ * th_c + a))) / lam_sum for n_, a in zip(harmonic_n, harmonic_alpha))
    diag_info = dict(n_distinct_separations=int(len(nz)), annular_correction_fraction_U=float(np.max(1 - cross_U * n_cross_cells / np.where(U_diag != 0, U_diag, np.nan))),
                     max_abs_linear_harmonic_ring_sum=float(lin), mean_radius_nodes=int(nodes_c.size),
                     n_lnV=int(lnV_nodes.size), runtime_s=time.time() - t0,
                     D0_vs_geomspace_cnp_max_rel=float(np.max(np.abs(D0 / _cnp_matrix(cross_sigma_obs)[np.ix_(ce, ce)] - 1))))
    METRICS['cross_theory_diagnostics'] = diag_info
    np.savez(cross_theory_cache, U=cross_U, D=cross_D, M=cross_M, mu_hat=mu_hat.astype(np.float32), diag=json.dumps(diag_info))
    print('cross theory', diag_info, flush=True)

# Same-event diagonal source integral (finite padded support, radial gradient and curvature, drift), the
# annular-mean correction with the local kernels, and the local distinct-event term.  2026-11-04 rework: the age
# integral is Gauss-Legendre in ln(S + sigma_v^2 d^2) (ucx.same_parent_diag_source); the earlier linear-age,
# 24-node evaluation (ucx.same_parent_diag_exact) was not converged and is kept only as an audit variant.
# Cache names carry the implementation version and the integration/table configuration.
DIAG_CFG = dict(version=ucx.DIAG_SOURCE_VERSION, n_t=int(os.environ.get('KTZ_CROSS_NT', 128)),
                n_lnw=int(os.environ.get('KTZ_CROSS_NW', 192)), lnw_range=[4e-4, 60.0], rho_step=0.005, panels=[0.004, 0.02])
_tag = f"v2_nt{DIAG_CFG['n_t']}_nw{DIAG_CFG['n_lnw']}"
tab_cache = OUT_DIR / f"cross_diag_tables_nw{DIAG_CFG['n_lnw']}_rho0.005_p0.004-0.02.npz"
diag_cache = OUT_DIR / f'cross_same_parent_diag_source_{_tag}.npz'
if diag_cache.exists() and json.loads(str(np.load(diag_cache)['config'])) == DIAG_CFG:
    _dc = np.load(diag_cache); U_diag_exact = _dc['U_diag_source']; U_diag_local = _dc['U_diag_local']; U_diag_old24 = _dc['U_diag_old_linear24']
    print('loaded', diag_cache.name, flush=True)
else:
    t0 = time.time()
    lnw_nodes = np.linspace(np.log(DIAG_CFG['lnw_range'][0]), np.log(DIAG_CFG['lnw_range'][1]), DIAG_CFG['n_lnw'])
    if tab_cache.exists() and np.allclose(np.load(tab_cache)['lnw'], lnw_nodes):
        sumtab = np.load(tab_cache)['sums']
    else:
        rho_tab = np.arange(cross_R_in_a - 0.05, cross_R_out_a + 0.05, DIAG_CFG['rho_step'])
        _tabs, sumtab, _ = ucx.axisymmetric_rate_tables(cross_source_R_in, cross_source_R_out, h_R, cluster_fraction, cluster_radius,
                                                        rho_tab, lnw_nodes, cvr, *DIAG_CFG['panels'])
        _tmp = OUT_DIR / (tab_cache.stem + '.tmp.npz')
        np.savez(_tmp, tabs=_tabs, sums=sumtab, lnw=lnw_nodes, rho_tab=rho_tab, version=ucx.DIAG_SOURCE_VERSION); os.replace(_tmp, tab_cache)
    U_diag_exact = np.zeros((nce, nce)); U_diag_local = np.zeros((nce, nce)); U_diag_old24 = np.zeros((nce, nce))
    lam_sum_loc = np.exp(-cvr / h_R).sum()
    for p_, i in enumerate(ce):
        for q_, j in enumerate(ce):
            if q_ < p_:
                continue
            tot = old = 0.0
            for a in range(3):
                for b in range(3):
                    y_ = kt25_second_yield_normalized[i, a, j, b]
                    if y_ == 0:
                        continue
                    args = (cross_sigma_obs[a]**2, cross_sigma_obs[b]**2, kt25_delay[i, a], kt25_delay[j, b])
                    tot += ucx.same_parent_diag_source(y_, *args, kt25_kappa, kt25_sigma_v, kt25_t_corr, lnw_nodes, sumtab, n_t=DIAG_CFG['n_t'])
                    old += ucx.same_parent_diag_exact(None, None, None, None, y_, *args[:2], args[2], args[3], None, kt25_kappa,
                                                      kt25_sigma_v, kt25_t_corr, lnw_nodes, sumtab=sumtab, n_s=24)
            U_diag_exact[p_, q_] = U_diag_exact[q_, p_] = tot; U_diag_old24[p_, q_] = U_diag_old24[q_, p_] = old
            k0 = ucx.same_parent_kernel_r(np.array([0.0]), *channel_pair_params(i, j, cross_sigma_obs, kt25_second_yield_normalized))[0]
            U_diag_local[p_, q_] = U_diag_local[q_, p_] = lam_sum_loc * k0
    _tmp = OUT_DIR / (diag_cache.stem + '.tmp.npz')
    np.savez(_tmp, U_diag_source=U_diag_exact, U_diag_local=U_diag_local, U_diag_old_linear24=U_diag_old24, lnw_nodes=lnw_nodes,
             sumtab=sumtab, config=json.dumps(DIAG_CFG)); os.replace(_tmp, diag_cache)
    print(f'same-event diagonal source integral ({time.time()-t0:.0f}s); max rel diff vs local {np.max(np.abs(U_diag_exact / U_diag_local - 1)):.4f}', flush=True)
METRICS['cross_diag_config'] = DIAG_CFG
U_ring = U_diag_local - cross_U * n_cross_cells          # local annular-mean correction recovered from the cache
cross_U_exactdiag = (U_diag_exact - U_ring) / n_cross_cells
METRICS['cross_theory_diagnostics']['same_parent_diag_source_vs_local_max_rel'] = float(np.max(np.abs(U_diag_exact / U_diag_local - 1)))
METRICS['cross_theory_diagnostics']['same_parent_diag_source_vs_old_linear24_max_rel'] = float(np.max(np.abs(U_diag_old24 / U_diag_exact - 1)))
Xi_KTZ_unified_old24 = normalize_cross_covariance((U_diag_old24 - U_ring) / n_cross_cells + cross_D + cross_M)
Xi_KTZ_unified_localdiag = normalize_cross_covariance(cross_U + cross_D + cross_M)
Xi_KTZ_unified = normalize_cross_covariance(cross_U_exactdiag + cross_D + cross_M)
Xi_KTZ_unified_noD = normalize_cross_covariance(cross_U_exactdiag + cross_M)


# %% Block 12: multi-element Monte Carlo realisations (seed 20260625, original order; resumable)
cache = OUT_DIR / 'cross_covariance_stack_unified.npz'
crng = default_rng(20260625)
covs, covs_old, done = [], [], 0
if cache.exists():
    _cc = np.load(cache, allow_pickle=True)
    covs, covs_old = list(_cc['covariance_stack']), list(_cc['covariance_stack_known_profile'])
    done = len(covs); crng.bit_generator.state = json.loads(str(_cc['rng_state']))
    print(f'resuming cross realisations after {done}', flush=True)
t0 = time.time()
for i in range(done, n_cross_real):
    rr_ = cross_realization(crng); covs.append(rr_['cov']); covs_old.append(rr_['cov_legacy'])
    np.save(OUT_DIR / f'cross_projected_values_r{i+1:02d}.npy', rr_['vals'])
    print(f'cross realisation {i+1}/{n_cross_real}: {rr_["n"]:,} events, {rr_["n_narrow"]:,} exact narrow returns, {time.time()-t0:.0f}s', flush=True)
    _tmp = OUT_DIR / 'cross_covariance_stack_unified.tmp.npz'
    np.savez_compressed(_tmp, covariance_stack=np.array(covs), covariance_stack_known_profile=np.array(covs_old),
                        rng_state=json.dumps(crng.bit_generator.state), elements=np.array(cross_elements))
    os.replace(_tmp, cache)
cov_stack = np.array(covs)
_arch = ROOT / 'validation' / 'outputs' / 'cross_covariance_stack.npz'
if _arch.exists() and not QUICK:
    _a = np.load(_arch)['covariance_stack']
    METRICS['cross_field_audit_max_rel_vs_archived'] = float(np.max(np.abs(np.array(covs_old) - _a)) / np.max(np.abs(_a)))
    print('cross field audit vs archived known-profile covariances:', METRICS['cross_field_audit_max_rel_vs_archived'], flush=True)
sim_cov = cov_stack.mean(0); Xi_sim = normalize_cross_covariance(sim_cov)
brng = default_rng(20260626)
boot = np.array([normalize_cross_covariance(cov_stack[brng.integers(0, n_cross_real, n_cross_real)].mean(0)) for _ in range(1000)])
Xi_sim_err = boot.std(0, ddof=1) if n_cross_real > 1 else np.full_like(Xi_sim, np.nan)

# %% Block 13: matrices and scatter figures; metrics
th_sub = Xi_KTZ_unified
th_sub_n = Xi_KTZ_unified_noD
th_sub_legacy = Xi_KTZ_obs[np.ix_(cross_idx, cross_idx)]
kt_sub = Xi_KT25[np.ix_(cross_idx, cross_idx)]
diff = th_sub - kt_sub; lim = np.nanmax(np.abs(diff))
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), constrained_layout=True)
for ax, (M, title, cmap, norm_) in zip(axes, [
        (kt_sub, r'Published KT25 (uniform): $\Xi_{AB}$', 'viridis', None),
        (th_sub, r'Full KTZ (projected): $\Xi_{AB,\Omega}$', 'viridis', None),
        (diff, r'$\Delta\Xi_{AB}={\rm KTZ}-{\rm KT25}$', 'RdBu_r', TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))]):
    im = ax.imshow(M, origin='lower', cmap=cmap, vmin=None if norm_ else 0, vmax=None if norm_ else 1, norm=norm_)
    ax.set_xticks(range(16)); ax.set_yticks(range(16)); ax.set_xticklabels(cross_elements, rotation=90); ax.set_yticklabels(cross_elements)
    ax.set_title(title, fontsize=11); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
fig.savefig(figure_path('unified_cross_matrices'), dpi=160, bbox_inches='tight'); plt.close(fig)

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
fig.savefig(figure_path('unified_cross_scatter'), dpi=160, bbox_inches='tight'); plt.close(fig)

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
# Audit of the earlier prescription on the same regenerated fields: known axisymmetric profile, no projection,
# local window prediction (reproduces the archived 'before' metrics), and its arcsinh-kernel diagnostic.
Xi_sim_old = normalize_cross_covariance(np.array(covs_old).mean(0))
_so = Xi_sim_old[iu]; _dl = _so - th_sub_legacy[iu]
METRICS['cross_known_profile_legacy'] = dict(rmse=float(np.sqrt(np.mean(_dl**2))), bias=float(_dl.mean()), max_abs=float(np.abs(_dl).max()),
                                             note='earlier known-profile prescription, before; not the unified statistic')
_cp0_as = kt25_covariance_matrix(cross_sigma_obs, channel_second_moment=kt25_second_yield_normalized, kernel='arcsinh')
_cp0_j0 = kt25_covariance_matrix(cross_sigma_obs, channel_second_moment=kt25_second_yield_normalized)
_tot_as = sum(cross_comps.values()) - cross_comps['CP0'] + _cp0_as * (cross_comps['CP0'][0, 0] / _cp0_j0[0, 0])
_th_as = normalize_cross_covariance(_tot_as)[np.ix_(cross_idx, cross_idx)][iu]
METRICS['cross_arcsinh_kernel_diagnostic_legacy'] = dict(rmse=float(np.sqrt(np.mean((_so - _th_as)**2))),
    kt25_baseline_change_max=float(np.max(np.abs((Xi_uniform_KTZ_limit - Xi_KT25)[np.ix_(cross_idx, cross_idx)][iu]))))
_o24 = s_ - Xi_KTZ_unified_old24[iu]
METRICS['cross_old_linear24_variant'] = dict(rmse=float(np.sqrt(np.mean(_o24**2))), bias=float(_o24.mean()), max_abs=float(np.abs(_o24).max()),
                                             max_abs_change=float(np.max(np.abs(Xi_KTZ_unified_old24 - th_sub))), note='11-03 delivered (unconverged) age integral')
# element-group and variance diagnostics (re-evaluated after the age-integral fix)
_tot = cross_U_exactdiag + cross_D + cross_M
_auto = np.diag(sim_cov) / np.diag(_tot)
METRICS['cross_auto_variance_ratio_sim_over_theory_rel_median'] = {e: float(v) for e, v in zip(cross_elements, _auto / np.median(_auto))}
_agbish = np.isin(np.array(cross_elements), ['N', 'Sr', 'Y', 'Zr', 'Ba', 'Ce'])
_g = np.array([_agbish[i] or _agbish[j] for i, j in zip(*iu)])
METRICS['cross_groups'] = {k: dict(n=int(sel.sum()), rmse=float(np.sqrt(np.mean(dd[sel]**2))), bias=float(dd[sel].mean()))
                           for k, sel in (('without_N_or_s_process', ~_g), ('with_N_or_s_process', _g))}
_ld = s_ - Xi_KTZ_unified_localdiag[iu]
METRICS['cross_localdiag_variant'] = dict(rmse=float(np.sqrt(np.mean(_ld**2))), bias=float(_ld.mean()), max_abs=float(np.abs(_ld).max()), within_2sigma=int(np.sum(np.abs(_ld) <= 2 * se_)))
METRICS['cross_legacy_local_prediction_vs_projected_sim'] = dict(rmse=float(np.sqrt(np.mean((s_ - th_sub_legacy[iu])**2))), note='earlier known-profile local prediction compared with the projected simulation; not a valid comparison, recorded for audit')
np.savez(OUT_DIR / 'cross_results_unified.npz', Xi_KTZ_old_linear24=Xi_KTZ_unified_old24, Xi_sim_known_profile=Xi_sim_old, Xi_sim=Xi_sim, Xi_sim_err=Xi_sim_err, Xi_KTZ=th_sub, Xi_KTZ_noCNP=th_sub_n, Xi_KT25=kt_sub, Xi_uniform_KTZ_limit=_uk, Xi_KTZ_legacy_local=th_sub_legacy, elements=np.array(cross_elements))
print('cross', METRICS['cross']); print('cross_noCNP', METRICS['cross_noCNP']); print('KTZ-KT25', METRICS['ktz_minus_kt25'])
METRICS['runtime_s'] = time.time() - T_START
json.dump(METRICS, open(OUT_DIR / 'validation_metrics_unified.json', 'w'), indent=1)
print(json.dumps(METRICS, indent=1))
