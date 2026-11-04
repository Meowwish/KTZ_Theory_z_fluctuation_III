"""Projected zero-lag elemental statistic Xi_AB for the multi-element experiment (Sec. 3.4).

Data:   r_A = P Psi Sigma_A on the valid cells (annular mean of each element map subtracted on the same
        annuli), Xi_AB = sum_i r_Ai r_Bi / [sum_i r_Ai^2 sum_i r_Bi^2]^{1/2}; covariances are averaged over
        realisations before normalisation.
Theory: sum_i S^{AB}_ii with S = P Psi C Psi^T P^T + m m^T, m = P Psi mu, i.e.
            sum_i (P C P^T)_ii = sum_i C_ii - sum_a N_a^{-1} sum_{j,k in a} C_jk ,
        C_ii from the source-integrated same-parent term and the local distinct-parent kernel;
        the annular-mean correction uses the same local isotropic kernels at separation r with the exact
        discrete pair-offset counts of each annulus and the annulus rate lambda_0(varpi_a); the complete mean
        mu_A is the exactly smoothed transported source rate (drift, delays, finite padded support), projected
        on the valid cells before its single outer product.  Units: U0 = Gamma <M^2>/(8 pi kappa)."""
import numpy as np
from scipy.special import exp1, j0
from . import ktz_unified_ops as op


def annulus_offset_histograms(valid2d, ann2d, nann, dxc, weights):
    """sum_a weights[a] * hist_a[n], hist_a[n] = number of ordered cell pairs (j,k) in annulus a whose lattice
    offset (di, dj) has di^2 + dj^2 = n (exact discrete counts from the FFT autocorrelation of the indicator)."""
    ny, nx = valid2d.shape
    N = 1 << int(np.ceil(np.log2(2 * max(nx, ny))))
    ii = np.arange(N); ii = np.where(ii < N // 2, ii, ii - N)
    n2 = (ii[:, None]**2 + ii[None, :]**2).ravel()
    nmax = int(n2.max()) + 1
    Hw = np.zeros(nmax); total = 0
    for a in range(nann):
        I = np.zeros((N, N)); I[:ny, :nx] = (ann2d == a) & valid2d
        F = np.fft.rfft2(I)
        ac = np.rint(np.fft.irfft2(F * np.conj(F), s=(N, N))).ravel()
        total += ac.sum()
        Hw += weights[a] * np.bincount(n2, ac, nmax)
    return Hw, total


def same_parent_kernel_r(r, ys, num, den, beta):
    """sum_ab Y_ab [E1(r^2/2(den+beta)) - E1(r^2/2(num+beta))]  (U0 units); r = 0 gives the log kernel."""
    r = np.asarray(r, float)
    out = np.zeros_like(r)
    pos = r > 0
    for y_, nu, de, be in zip(ys, num, den, beta):
        t1, t0 = nu + be, de + be
        v = np.full(r.shape, np.log(t1 / t0))
        v[pos] = exp1(r[pos]**2 / (2 * t1)) - exp1(r[pos]**2 / (2 * t0))
        out += y_ * v
    return out


def distinct_parent_kernel_r(r, k, P_source, TA, TB, pref0):
    """pref0 * int dk k P_source(k) T_A(k) T_B(k) J0(k r) / (2 pi)   (U0 units, local isotropic C_D)."""
    integrand = (k * P_source * TA * TB / (2 * np.pi))[None, :] * j0(np.outer(r, k))
    return pref0 * np.trapezoid(integrand, k, axis=1)


def transported_mean_tables(lam_hat1, nodes, wts, rho_out, V_nodes):
    """Exact Gaussian smoothing of the one-point rate for a grid of per-axis variances V (log grid):
    returns dict k -> array (nV, nrho)."""
    out = {k: np.zeros((V_nodes.size, rho_out.size), complex) for k in lam_hat1}
    for j, V in enumerate(V_nodes):
        for k, c in lam_hat1.items():
            out[k][j] = op.smoothing_matrix(k, V, rho_out, nodes, wts) @ c
    return out


def interp_lnV(tables, lnV_nodes, lnV):
    """Cubic Lagrange interpolation of the smoothing tables in ln V (uniform ln V grid)."""
    return {k: _cub_axis0(lnV_nodes, t, lnV) for k, t in tables.items()}


def _cub_axis0(grid, vals, x):
    h = grid[1] - grid[0]
    t = (x - grid[0]) / h
    i = int(np.clip(np.floor(t), 1, grid.size - 3)); u = t - i
    c = (-u * (u - 1) * (u - 2) / 6, (u + 1) * (u - 1) * (u - 2) / 2, -(u + 1) * u * (u - 2) / 2, (u + 1) * u * (u - 1) / 6)
    return c[0] * vals[i - 1] + c[1] * vals[i] + c[2] * vals[i + 1] + c[3] * vals[i + 2]


def same_parent_diag_exact(cell_r, lam1_0, nodes, wts, ys, Va0, Vb0, Da, Db, dDa, kappa, sigma_v, t_corr, lnw_nodes,
                           sumtab=None, n_s=24):
    """Exact Gram diagonal sum_i C_AB,ii (U0 units) of the same-parent term for one channel pair, with the
    axisymmetric one-point rate lam1_0 on its finite support.  For a common parent velocity v ~ N(0, sigma_v^2 I),
    returns displaced by v Delta_a and v Delta_b, kernel variances Va = Va0 + 2 kappa (s - Delta_a) etc.:
        int dy lambda(y) <G_Va(x - y - v Da) G_Vb(x - y - v Db)>_v = G_{S + sigma_v^2 d^2}(0) (lambda * G_w')(x),
        S = Va + Vb, d = Db - Da, m = (Vb Da + Va Db)/S, w' = Va Vb / S + m^2 / (1/sigma_v^2 + d^2/S).
    sumtab[j] = sum_i (lam1_0 * G_{w_j})(varpi_i) on the ln w grid (computed once and reused)."""
    s0 = max(Da, Db)
    if s0 >= t_corr:
        return 0.0
    x, w = np.polynomial.legendre.leggauss(n_s)
    s = 0.5 * (t_corr - s0) * x + 0.5 * (t_corr + s0); ds = 0.5 * (t_corr - s0) * w
    Va = Va0 + 2 * kappa * (s - Da); Vb = Vb0 + 2 * kappa * (s - Db); S = Va + Vb
    d = Db - Da; m = (Vb * Da + Va * Db) / S
    wp = Va * Vb / S + (m**2 / (1.0 / sigma_v**2 + d**2 / S) if sigma_v > 0 else 0.0)
    G0 = 1.0 / (2 * np.pi * (S + sigma_v**2 * d**2))
    lw = np.log(wp)
    vals = np.array([_cub_axis0(lnw_nodes, sumtab, l) for l in lw])
    return float(np.sum(ds * G0 * vals) * 8 * np.pi * kappa * ys)


# ------------------------------------------------------------------------------------------------------------
# 2026-11-04 rework.  same_parent_diag_exact above integrates linearly in age with n_s = 24 Gauss-Legendre
# nodes by default; for the youngest, narrowest kernels (1/S varies by ~10^3 over the age range) this is not
# converged (24 nodes underestimate the uniform-rate AGB+beam logarithm by 3.9 per cent).  It is kept unchanged
# for audit.  The function below integrates in t = ln(S + sigma_v^2 d^2), where the integrand
# G_{S+beta}(0) ds = dt / (8 pi kappa) is flat and only the smooth spatial factor varies.
# ------------------------------------------------------------------------------------------------------------
DIAG_SOURCE_VERSION = 'same_parent_diag_source/v2-lnS-2026-11-04'


def same_parent_diag_source(ys, Va0, Vb0, Da, Db, kappa, sigma_v, t_corr, lnw_nodes, sumtab, n_t=64):
    """Same-event diagonal source integral sum_i C_AB,ii (U0 units) for one channel pair:
        8 pi kappa Y int_{s0}^{t*} ds G_{S+beta}(0) Sigma(w'(s)),   beta = sigma_v^2 d^2,
    with S = Va + Vb, d = Db - Da, m = (Vb Da + Va Db)/S, w' = Va Vb/S + m^2/(1/sigma_v^2 + d^2/S) and
    Sigma(w) = sum_i (lambda_1 * G_w)(varpi_i) tabulated on the uniform ln w grid (cubic interpolation).
    Since dS/ds = 4 kappa, 8 pi kappa G_{S+beta}(0) ds = dt with t = ln(S + beta); Gauss-Legendre in t."""
    s0 = max(Da, Db)
    if s0 >= t_corr:
        return 0.0
    d = Db - Da; beta = sigma_v**2 * d**2
    S0 = Va0 + Vb0 + 2 * kappa * (2 * s0 - Da - Db); S1 = Va0 + Vb0 + 2 * kappa * (2 * t_corr - Da - Db)
    t0, t1 = np.log(S0 + beta), np.log(S1 + beta)
    x, w = np.polynomial.legendre.leggauss(n_t)
    t = 0.5 * (t1 - t0) * x + 0.5 * (t1 + t0); dt = 0.5 * (t1 - t0) * w
    S = np.exp(t) - beta
    s = (S - Va0 - Vb0 + 2 * kappa * (Da + Db)) / (4 * kappa)
    Va = Va0 + 2 * kappa * (s - Da); Vb = Vb0 + 2 * kappa * (s - Db)
    m = (Vb * Da + Va * Db) / S
    wp = Va * Vb / S + (m**2 / (1.0 / sigma_v**2 + d**2 / S) if sigma_v > 0 else 0.0)
    lw = np.log(wp)
    h = lnw_nodes[1] - lnw_nodes[0]
    assert lw.min() > lnw_nodes[0] + h and lw.max() < lnw_nodes[-1] - 2 * h, 'w outside the tabulated range'
    vals = np.array([_cub_axis0(lnw_nodes, sumtab, l) for l in lw])
    return float(ys * np.sum(dt * vals))


def same_parent_offdiag_source(ys, Va0, Vb0, Da, Db, kappa, sigma_v, t_corr, r, cstar_fun, n_t=48):
    """Same-event source integral C_AB(x, x') (U0 units) for one channel pair at separation r, for checks of the
    local annular-mean correction.  Integrating y and the common velocity v:
        C = 8 pi kappa Y int ds G_{S+beta}(r) Lambda_{w''}(c),  c = (Vb x + Va x')/S - mu_v m,
        mu_v = -sigma_v^2 d (x - x')/(S + beta),  w'' = Va Vb/S + m^2 sigma_v^2 S/(S + beta).
    cstar_fun(S, Va, Vb, m, muv_factor, w2) must return Lambda_{w''}(c) for each node (arrays over pairs)."""
    s0 = max(Da, Db)
    if s0 >= t_corr:
        return 0.0
    d = Db - Da; beta = sigma_v**2 * d**2
    S0 = Va0 + Vb0 + 2 * kappa * (2 * s0 - Da - Db); S1 = Va0 + Vb0 + 2 * kappa * (2 * t_corr - Da - Db)
    t0, t1 = np.log(S0 + beta), np.log(S1 + beta)
    x, w = np.polynomial.legendre.leggauss(n_t)
    out = 0.0
    for tt, ww in zip(0.5 * (t1 - t0) * x + 0.5 * (t1 + t0), 0.5 * (t1 - t0) * w):
        Sb = np.exp(tt); S = Sb - beta
        s = (S - Va0 - Vb0 + 2 * kappa * (Da + Db)) / (4 * kappa)
        Va = Va0 + 2 * kappa * (s - Da); Vb = Vb0 + 2 * kappa * (s - Db)
        m = (Vb * Da + Va * Db) / S
        w2 = Va * Vb / S + m**2 * sigma_v**2 * S / Sb
        g = np.exp(-r**2 / (2 * Sb))                 # 2 pi (S+beta) G_{S+beta}(r); ds-measure gives dt/(8 pi kappa)
        out = out + ww * g * cstar_fun(S, Va, Vb, m, -sigma_v**2 * d / Sb, w2)
    return ys * out


def axisymmetric_rate_tables(R_src_in, R_src_out, h_R, f_cl, sigma_cl, rho_tab, lnw_nodes, cell_r,
                             panel_inner=0.004, panel_outer=0.02, order=6):
    """Radial tables Lambda_w(rho) = (lambda_1 * G_w)(rho) of the axisymmetric one-point rate (children displaced by
    sigma_cl and discarded outside the padded support) on rho_tab for every ln w node, and their cell sums
    Sigma(w) = sum_i Lambda_w(varpi_i).  lambda_hat = exp(-varpi/h_R) in units of Gamma."""
    nA = op.radial_nodes(R_src_in, 3.0, panel_inner, order); nB = op.radial_nodes(3.0, R_src_out, panel_outer, order)
    nodes = np.concatenate([nA[0], nB[0]]); wts = np.concatenate([nA[1], nB[1]])
    lam0 = np.exp(-nodes / h_R).astype(complex)
    lam1 = (1 - f_cl) * lam0 + f_cl * (op.smoothing_matrix(0, sigma_cl**2, nodes, nodes, wts) @ lam0)
    tabs = np.zeros((lnw_nodes.size, rho_tab.size)); sums = np.zeros(lnw_nodes.size)
    for j, lw in enumerate(lnw_nodes):
        tabs[j] = (op.smoothing_matrix(0, np.exp(lw), rho_tab, nodes, wts) @ lam1).real
        sums[j] = np.sum(op.cubic_uniform(rho_tab, tabs[j], cell_r))
    return tabs, sums, nodes.size


def _smooth_apply(k, w, rho_out, nodes, wts, c, band=10.0):
    """Memory-light (row-by-row) version of smoothing_matrix(k, w, rho_out, nodes, wts) @ c."""
    from scipy.special import ive
    out = np.zeros(rho_out.size, complex)
    half = band * np.sqrt(w)
    for o, ro in enumerate(rho_out):
        sel = np.abs(nodes - ro) < half
        if np.any(sel):
            rq = nodes[sel]
            out[o] = np.sum(wts[sel] * rq / w * np.exp(-(ro - rq)**2 / (2 * w)) * ive(abs(k), ro * rq / w) * c[sel])
    return out


def axisymmetric_rate_sums_light(R_src_in, R_src_out, h_R, f_cl, sigma_cl, rho_tab, lnw_nodes, cell_r,
                                 panel_inner, panel_outer, order=6):
    """As axisymmetric_rate_tables but without storing dense matrices (for refinement checks); returns the sums."""
    nA = op.radial_nodes(R_src_in, 3.0, panel_inner, order); nB = op.radial_nodes(3.0, R_src_out, panel_outer, order)
    nodes = np.concatenate([nA[0], nB[0]]); wts = np.concatenate([nA[1], nB[1]])
    lam0 = np.exp(-nodes / h_R).astype(complex)
    lam1 = (1 - f_cl) * lam0 + f_cl * _smooth_apply(0, sigma_cl**2, nodes, nodes, wts, lam0)
    sums = np.zeros(lnw_nodes.size)
    for j, lw in enumerate(lnw_nodes):
        tab = _smooth_apply(0, np.exp(lw), rho_tab, nodes, wts, lam1).real
        sums[j] = np.sum(op.cubic_uniform(rho_tab, tab, cell_r))
    return sums, nodes.size
