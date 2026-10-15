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
cluster_fraction = float(os.getenv('KTZ_FCL','0.5'))
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

component_order = ['CP0', 'CPN', 'CM', 'CNP']
component_labels = {'CP0': 'a', 'CPN': 'b', 'CM': 'c', 'CNP': 'd'}
def source_harmonic_damping(Qn):
    return (1.0 - cluster_fraction) + cluster_fraction * np.exp(-0.5 * (Qn * cluster_radius)**2)
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
def _kt25_kernel(a, b, c):
    return np.log(a) + 2 * b / np.sqrt(1 + b**2) * np.arcsinh(b) - 2 * c / np.sqrt(1 + c**2) * np.arcsinh(c)

def kt25_covariance_matrix(sigma_inj, channel_yields=None, channel_second_moment=None):
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
                    val += ys * _kt25_kernel(num / den, kt25_sigma_v * dd / np.sqrt(num), kt25_sigma_v * dd / np.sqrt(den))
            cov[i, j] = cov[j, i] = val
    return cov

def normalize_cross_covariance(c):
    v = np.diag(c); d = np.sqrt(np.outer(v, v))
    return np.divide(c, d, out=np.full_like(c, np.nan), where=d > 0)

Xi_KT25 = normalize_cross_covariance(kt25_covariance_matrix(kt25_sigma_inj, channel_yields=kt25_mean_yield))

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
        (kt_sub, r'Spatially uniform KT25: $\Xi_{AB}$', 'viridis', None),
        (th_sub, r'Full KTZ (beam-matched): $\Xi_{AB}$', 'viridis', None),
        (diff, r'$\Delta\Xi_{AB}={\rm KTZ}-{\rm KT25}$', 'RdBu_r', TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))]):
    im = ax.imshow(M, origin='lower', cmap=cmap, vmin=None if norm_ else 0, vmax=None if norm_ else 1, norm=norm_)
    ax.set_xticks(range(16)); ax.set_yticks(range(16)); ax.set_xticklabels(cross_elements, rotation=90); ax.set_yticklabels(cross_elements)
    ax.set_title(title, fontsize=11); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
fig.savefig(FIG_DIR/'chk_cross_matrices.png', dpi=160, bbox_inches='tight'); plt.close(fig)

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
ax.scatter(s_, tn_, s=9, facecolors='none', edgecolors='0.45', lw=0.6, label=r'KTZ without $C_{AB,{\rm NP}}$', zorder=1.5)
lo_, hi_ = min(0, s_.min(), t_.min(), tn_.min()), max(1, s_.max(), t_.max())
ax.plot([lo_, hi_], [lo_, hi_], 'k--', lw=1.2, label='1:1')
ax.set_xlim(lo_ - 0.03, hi_ + 0.03); ax.set_ylim(lo_ - 0.03, hi_ + 0.03); ax.set_aspect('equal', adjustable='box')
ax.set_xlabel(r'Simulation $\langle\Xi_{AB}\rangle$'); ax.set_ylabel(r'KTZ prediction $\Xi_{AB}$')
ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc='lower right')
fig.savefig(FIG_DIR/'chk_cross_scatter.png', dpi=160, bbox_inches='tight'); plt.close(fig)

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
METRICS['ktz_minus_kt25'] = dict(min=float(diff[iu].min()), max=float(diff[iu].max()))
np.savez(OUT_DIR / 'cross_results.npz', Xi_sim=Xi_sim, Xi_sim_err=Xi_sim_err, Xi_KTZ=th_sub, Xi_KTZ_noCNP=th_sub_n, Xi_KT25=kt_sub, elements=np.array(cross_elements))
print('cross', METRICS['cross']); print('cross_noCNP', METRICS['cross_noCNP']); print('KTZ-KT25', METRICS['ktz_minus_kt25'])
METRICS['runtime_s'] = time.time() - T_START
json.dump(METRICS, open(OUT_DIR / 'validation_metrics.json', 'w'), indent=1)
print(json.dumps(METRICS, indent=1))
