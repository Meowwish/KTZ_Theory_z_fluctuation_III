# Single-element entry point exported from KTZ_validation.ipynb.

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
RUN_TAG = os.environ.get('KTZ_RUN_TAG', 'single_element')
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
json.dump(METRICS, open(OUT_DIR / 'validation_metrics_single_element.json', 'w'), indent=1)
print(f'scalar part done in {time.time()-T_START:.0f}s', flush=True)
