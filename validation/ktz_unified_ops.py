"""Unified finite-map operators for the KTZ single-element statistic xi (Paper III, Sec. 2.3.4).

The same operators are used for the controlled simulations and for their prediction:

    continuous field  ->  response Psi (beam, optional cell response, sampling at valid cells)
                      ->  annular projection P on the actual valid cells
                      ->  sums over the unordered valid-cell pairs P_b of each separation bin
                      ->  normalisation  xi_b = sum_{P_b} S_ij / [ (1/2) sum_{P_b} (S_ii + S_jj) ]

with S = P Psi C_conn Psi^T P^T + m m^T and m = P Psi mu.  The data estimator replaces S_ij by the
product of the projected map values.

Prediction (legal field statistics, no pair-midpoint rate substitution):
  * same-event term: Gram integral over source positions with the non-negative one-point rate,
        C_SE(x,x') = int ds G_{2V_s}(x-x') (lambda_1 * G_{V_s/2})(x_m)     (Gaussian product identity);
  * distinct-event term: Gram integral over parent positions c (isotropic Gaussian pair excess),
        C_D(x,x')  = K_D int ds ds' G_{a+b}(x-x') (lambda_p * G_{ab/(a+b)})(c*),  c* = (b x + a x')/(a+b);
  * complete mean: mu(x) = int ds (lambda_1 * G_{V_s})(x), transported, responded, projected, one outer product.
The source rates are given on their own finite support as polar harmonics; their Gaussian smoothing is
evaluated exactly by radial Bessel quadrature (no plane-wave or WKBJ approximation, no continuation).
"""
import os
import numpy as np
from scipy.special import ive
import numba as nb

# ----------------------------------------------------------------------------------------------
# Polar-harmonic representation of a source rate and its exact Gaussian smoothing
# f(rho, phi) = sum_k Re[ c_k(rho) exp(-i k phi) ]   (k = 0 entry real)
# ----------------------------------------------------------------------------------------------

def radial_nodes(r0, r1, panel=0.025, order=6):
    """Gauss-Legendre nodes/weights on [r0, r1] in panels (integrands are smooth inside the support)."""
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(r0, r1, int(np.ceil((r1 - r0) / panel)) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    nodes = (0.5 * (b - a) * x[None, :] + 0.5 * (b + a)).ravel()
    wts = (0.5 * (b - a) * w[None, :]).ravel()
    return nodes, wts


def smoothing_matrix(k, w, rho_out, nodes, wts, band=10.0):
    """M[o, q] such that (f*G_w)_k(rho_o) = sum_q M[o,q] c_k(node_q) for the harmonic exp(-i k phi).
    G_w is the 2D isotropic Gaussian with per-axis variance w."""
    rho_out = np.asarray(rho_out, float)
    M = np.zeros((rho_out.size, nodes.size))
    half = band * np.sqrt(w)
    for o, ro in enumerate(rho_out):
        sel = np.abs(nodes - ro) < half
        if not np.any(sel):
            continue
        rq = nodes[sel]
        z = ro * rq / w
        M[o, sel] = wts[sel] * rq / w * np.exp(-(ro - rq) ** 2 / (2 * w)) * ive(abs(k), z)
    return M


def smooth_harmonics(coeffs, w, rho_out, nodes, wts):
    """Apply exact 2D Gaussian smoothing (per-axis variance w) to a harmonic field given on nodes."""
    return {k: smoothing_matrix(k, w, rho_out, nodes, wts) @ c for k, c in coeffs.items()}


def eval_harmonics(table, rho_grid, rho, phi):
    """Evaluate sum_k Re[c_k(rho) e^{-ik phi}] from tabulated c_k on a uniform rho_grid (cubic)."""
    out = np.zeros(np.shape(rho))
    for k, c in table.items():
        cr = cubic_uniform(rho_grid, c, rho)
        out = out + (cr.real if k == 0 else (cr * np.exp(-1j * k * phi)).real)
    return out


def cubic_uniform(grid, vals, x):
    """4-point Lagrange (cubic) interpolation on a uniform grid; vals may be complex."""
    h = grid[1] - grid[0]
    t = (np.asarray(x, float) - grid[0]) / h
    i = np.clip(np.floor(t).astype(int), 1, grid.size - 3)
    u = t - i
    v0, v1, v2, v3 = vals[i - 1], vals[i], vals[i + 1], vals[i + 2]
    return (-u * (u - 1) * (u - 2) / 6 * v0 + (u + 1) * (u - 1) * (u - 2) / 2 * v1
            - (u + 1) * u * (u - 2) / 2 * v2 + (u + 1) * u * (u - 1) / 6 * v3)


# ----------------------------------------------------------------------------------------------
# Annular projection P on the valid cells
# ----------------------------------------------------------------------------------------------

def annulus_labels(rc, edges):
    a = np.searchsorted(edges, rc, side='right') - 1
    assert np.all((a >= 0) & (a < len(edges) - 1)), 'valid cell outside the projection annuli'
    return a.astype(np.int64)


def project(f, ann, nann):
    """(P f)_i = f_i - mean_{k in a(i)} f_k over the valid cells (works on the last axis)."""
    f = np.asarray(f, float)
    cnt = np.bincount(ann, minlength=nann).astype(float)
    if f.ndim == 1:
        mean = np.bincount(ann, f, nann) / cnt
        return f - mean[ann]
    out = np.empty_like(f)
    for idx in np.ndindex(f.shape[:-1]):
        mean = np.bincount(ann, f[idx], nann) / cnt
        out[idx] = f[idx] - mean[ann]
    return out


# ----------------------------------------------------------------------------------------------
# Pair windows: bit 0 full window (both endpoints valid); bit 1 arm-phase sector; bits 2.. fixed
# midpoint radius.  Every window is an explicit selection of unordered valid-cell pairs.
# ----------------------------------------------------------------------------------------------

@nb.njit(cache=True, inline='always')
def _wrap(t):
    return (t + np.pi) % (2 * np.pi) - np.pi


@nb.njit(cache=True, inline='always')
def window_mask(mx, my, b, wpar, radii, halfw):
    """wpar = [sector_Rin, sector_Rout, sector_halfphase, sector_lastbin, m_arms, phase_coeff, R_in_phase, Theta0]."""
    mask = 1
    rm = np.sqrt(mx * mx + my * my)
    if rm > wpar[0] and rm < wpar[1] and b <= wpar[3]:
        th = wpar[5] * np.log(rm / wpar[6]) - wpar[4] * np.arctan2(my, mx) + wpar[7]
        if abs(_wrap(th)) < wpar[2]:
            mask |= 2
    for k in range(radii.size):
        if abs(rm - radii[k]) < halfw:
            mask |= (4 << k)
    return mask


@nb.njit(cache=True, nogil=True)
def pair_products(px, py, vals, sq, edges, wpar, radii, halfw, nwin, i0, i1):
    """For rows i in [i0,i1) and all j>i: per window and bin, sum v_i v_j, sum (sq_i+sq_j)/2 and counts.
    vals, sq: (nv, n).  Returns num (nv,nwin,nb), den (nv,nwin,nb), cnt (nwin,nb)."""
    nv, n = vals.shape
    nbins = edges.size - 1
    num = np.zeros((nv, nwin, nbins)); den = np.zeros((nv, nwin, nbins)); cnt = np.zeros((nwin, nbins))
    e0, e1 = edges[0], edges[-1]
    for i in range(i0, i1):
        xi, yi = px[i], py[i]
        for j in range(i + 1, n):
            dx = px[j] - xi; dy = py[j] - yi
            r = np.sqrt(dx * dx + dy * dy)
            if r < e0 or r >= e1:
                continue
            b = np.searchsorted(edges, r, side='right') - 1
            msk = window_mask(0.5 * (px[j] + xi), 0.5 * (py[j] + yi), b, wpar, radii, halfw)
            for w in range(nwin):
                if (msk >> w) & 1:
                    cnt[w, b] += 1
                    for v in range(nv):
                        num[v, w, b] += vals[v, i] * vals[v, j]
                        den[v, w, b] += 0.5 * (sq[v, i] + sq[v, j])
    return num, den, cnt


@nb.njit(cache=True, nogil=True)
def pair_projection_terms(px, py, ann, Rbar, Bbar, edges, wpar, radii, halfw, nwin, i0, i1):
    """sum over window pairs of [-Rbar_c[i,a(j)] - Rbar_c[j,a(i)] + Bbar_c[a(i),a(j)]] for each component c."""
    nc = Rbar.shape[0]; n = px.size; nbins = edges.size - 1
    out = np.zeros((nc, nwin, nbins))
    e0, e1 = edges[0], edges[-1]
    for i in range(i0, i1):
        xi, yi, ai = px[i], py[i], ann[i]
        for j in range(i + 1, n):
            dx = px[j] - xi; dy = py[j] - yi
            r = np.sqrt(dx * dx + dy * dy)
            if r < e0 or r >= e1:
                continue
            b = np.searchsorted(edges, r, side='right') - 1
            msk = window_mask(0.5 * (px[j] + xi), 0.5 * (py[j] + yi), b, wpar, radii, halfw)
            aj = ann[j]
            for c in range(nc):
                t = -Rbar[c, i, aj] - Rbar[c, j, ai] + Bbar[c, ai, aj]
                for w in range(nwin):
                    if (msk >> w) & 1:
                        out[c, w, b] += t
    return out


# ----------------------------------------------------------------------------------------------
# Covariance pass: field covariances of all pairs within r_cut (Gram forms), row-annulus sums,
# diagonal and window sums.  Components: 0 = C_U, 1 = C_T, 2 = C_D (isotropic), 3 = C_D (modulated
# parent rate; diagnostic only).
# ----------------------------------------------------------------------------------------------

@nb.njit(cache=True, inline='always')
def _cubic(tab, x0, h, x):
    t = (x - x0) / h
    i = int(np.floor(t))
    if i < 1:
        i = 1
    if i > tab.size - 3:
        i = tab.size - 3
    u = t - i
    return (-u * (u - 1) * (u - 2) / 6 * tab[i - 1] + (u + 1) * (u - 1) * (u - 2) / 2 * tab[i]
            - (u + 1) * u * (u - 2) / 2 * tab[i + 1] + (u + 1) * u * (u - 1) / 6 * tab[i + 2])


@nb.njit(cache=True, nogil=True)
def covariance_pass(px, py, ixh, iyh, ann, nann, rcut,
                    se_w, se_V, LamU, LamT,
                    d_w, d_a, KD, D0tab, Dmtab_re, Dmtab_im, dm_k, rho0, hrho, do_dmod,
                    edges, wpar, radii, halfw, nwin, i0, i1):
    """Rows i in [i0, i1); pairs j >= i within rcut (C is symmetric).  Returns
       R (4, n, nann): this block's contributions to the row-annulus sums sum_{j in a} C_ij (summed over blocks),
       diag (4, n), win (4, nwin, nb): window sums over unordered pairs (i<j) of C_ij."""
    n = px.size; nbins = edges.size - 1; ns = se_w.size; nd = d_w.size; nk = dm_k.size
    R = np.zeros((4, n, nann)); diag = np.zeros((4, n)); win = np.zeros((4, nwin, nbins))
    rc2 = rcut * rcut
    for i in range(i0, i1):
        xi, yi = px[i], py[i]
        for j in range(i, n):
            dx = px[j] - xi; dy = py[j] - yi
            r2 = dx * dx + dy * dy
            if r2 > rc2:
                continue
            mxh = ixh[i] + ixh[j]; myh = iyh[i] + iyh[j]
            cu = 0.0; ct = 0.0
            for s in range(ns):
                g = se_w[s] * np.exp(-r2 / (4.0 * se_V[s])) / (4.0 * np.pi * se_V[s])
                cu += g * LamU[s, myh, mxh]
                ct += g * LamT[s, myh, mxh]
            cd = 0.0; cdm = 0.0
            for p in range(nd):
                for q in range(nd):
                    a = d_a[p]; b = d_a[q]; ab = a + b
                    g = d_w[p] * d_w[q] * np.exp(-r2 / (2.0 * ab)) / (2.0 * np.pi * ab)
                    cx = (b * xi + a * px[j]) / ab; cy = (b * yi + a * py[j]) / ab
                    rr = np.sqrt(cx * cx + cy * cy)
                    t = p * nd + q
                    v0 = _cubic(D0tab[t], rho0, hrho, rr)
                    cd += g * v0
                    if do_dmod:
                        ph = np.arctan2(cy, cx); vm = v0
                        for kk in range(nk):
                            re = _cubic(Dmtab_re[t, kk], rho0, hrho, rr); im = _cubic(Dmtab_im[t, kk], rho0, hrho, rr)
                            ang = dm_k[kk] * ph
                            vm += re * np.cos(ang) + im * np.sin(ang)    # Re[(re + i im) e^{-ik phi}]
                        cdm += g * vm
            vals = (cu, ct, KD * cd, KD * cdm)
            aj = ann[j]; ai = ann[i]
            for c in range(4):
                R[c, i, aj] += vals[c]
                if j != i:
                    R[c, j, ai] += vals[c]
            if j == i:
                for c in range(4):
                    diag[c, i] = vals[c]
            elif j > i:
                r = np.sqrt(r2)
                if r >= edges[0] and r < edges[-1]:
                    bb = np.searchsorted(edges, r, side='right') - 1
                    msk = window_mask(0.5 * (px[j] + xi), 0.5 * (py[j] + yi), bb, wpar, radii, halfw)
                    for w in range(nwin):
                        if (msk >> w) & 1:
                            for c in range(4):
                                win[c, w, bb] += vals[c]
    return R, diag, win


def run_rows(func, n, nthreads, *args):
    """Split rows over threads (numba nogil) and sum the per-thread outputs."""
    from concurrent.futures import ThreadPoolExecutor
    cuts = np.linspace(0, n, nthreads * 8 + 1).astype(int)
    jobs = list(zip(cuts[:-1], cuts[1:]))
    # interleave the row blocks so that the triangular (j > i) work is balanced
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        futs = [ex.submit(func, *args, int(a), int(b)) for a, b in jobs]
        res = [f.result() for f in futs]
    if isinstance(res[0], tuple):
        return tuple(sum(r[k] for r in res) for k in range(len(res[0])))
    return sum(res)
