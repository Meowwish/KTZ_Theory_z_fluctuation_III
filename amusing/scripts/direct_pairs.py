"""Independent brute-force finite-window pair calculation of every KTZ term (2026-10-16).

For one parameter point it forms, on the actual valid-cell mask, the cell covariances of each term with kernels
evaluated at the EXACT pair separations (no kernel-radius grid, no slope or (ell, eps, ell_src) interpolation):
  C_2p,0: rate(mid) K(r)            C_2p,N: eta rate(mid) g(mid) K(r)      C_NP: betaA rate(mid)^2 K_NP(r)
  C_M   : beta (f0 + eta f1)_i (f0 + eta f1)_j  with f0 = P rate, f1 = P(rate g)
with rate(mid) = exp(s (|x_mid| - r_ref)) and g(mid) the template contrast at the pair midpoint (bilinear on the
half-cell grid, same definition as build_window_response).  Pair separations are integer pixel lags, so each kernel
is tabulated exactly on every squared lag n = di^2 + dj^2 (r = dx sqrt(n)); the CNP age integral uses q_cnp-point
Gauss-Legendre (default 64, i.e. converged).  The annular projection P C P^T is applied exactly:
numerator_b = sum_{(i,j) in b} [C_ij - Cbar_{i,a(j)} - Cbar_{a(i),j} + Cbar_{a(i),a(j)}],
denominator_b = sum_{(i,j) in b} (d_i + d_j)/2, d_i = (P C P^T)_ii, both per ordered pair with r > 0.
Independent of response_library.py: no hat-function kernel-radius basis, no histogramming of kernels, different
loop structure (dense block matrices + one-hot annulus matrix).
"""
import numpy as np
from scipy.special import exp1
from scipy.ndimage import map_coordinates


def kt18_kernel(r, ell, eps):
    out = np.empty_like(r, dtype=float); z = r <= 0
    out[z] = -np.log(eps)
    rr = r[~z]; out[~z] = 0.5*(exp1(rr**2/(4*ell**2))-exp1(rr**2/(4*(eps*ell)**2)))
    return out


def cnp_kernel(r, ell, eps, length, q):
    xg, wg = np.polynomial.legendre.leggauss(q); a = 0.5*(xg+1); w = 0.5*wg
    wf = eps**2+(1-eps**2)*a
    den = (length**2+ell**2*(wf[:, None]+wf[None, :])).ravel()
    pw = (w[:, None]*w[None, :]*length**2/den.reshape(q, q)).ravel()
    rr = np.asarray(r, float).ravel(); out = np.zeros(rr.size)
    for c0 in range(0, len(den), 256):
        out += np.exp(-rr[:, None]**2/(2*den[None, c0:c0+256])) @ pw[c0:c0+256]
    return out.reshape(np.shape(r))


def direct_terms(pm, r_ref, edges, annulus_width, ell_obs, eps_obs, slope, eta, beta, betaA, ell_src, q_cnp=64, block=512,
                 components=("p0", "pN", "np", "m")):
    valid = pm["valid_mask"].astype(bool); X, Y = pm["x_grid_kpc"], pm["y_grid_kpc"]; dx = float(pm["pixel_size_kpc"])
    iy, ix = np.nonzero(valid); x, y = X[valid], Y[valid]; n = len(x)
    rad = np.hypot(X, Y); lab_g = np.floor(rad/annulus_width).astype(int)
    labs, ann = np.unique(lab_g[valid], return_inverse=True); na = len(labs); counts = np.bincount(ann, minlength=na).astype(float)
    onehot = np.zeros((n, na)); onehot[np.arange(n), ann] = 1.0
    mod = pm["arm_source_modulation"]
    hy, hx = np.meshgrid(np.arange(2*valid.shape[0]-1)/2, np.arange(2*valid.shape[1]-1)/2, indexing="ij")
    half = map_coordinates(mod, [hy, hx], order=1, mode="nearest")
    nb = len(edges)-1; edges = np.asarray(edges)
    comps = [c for c in components if c != "m"]
    nmax = int((np.ptp(iy))**2+(np.ptp(ix))**2)
    rtab = dx*np.sqrt(np.arange(nmax+1))
    ktab = {}
    if "p0" in comps or "pN" in comps: ktab["p"] = kt18_kernel(rtab, ell_obs, eps_obs)
    if "np" in comps: ktab["np"] = cnp_kernel(rtab, ell_obs, eps_obs, ell_src, q_cnp)
    rowmean = {c: np.zeros((n, na)) for c in comps}; numC = {c: np.zeros(nb) for c in comps}
    pa = np.zeros((n, nb)); ann_pair = np.zeros((na, na, nb)); cnt = np.zeros(nb)
    t1sum = {c: np.zeros(nb) for c in comps}
    for s0 in range(0, n, block):
        s1 = min(n, s0+block)
        lag = (iy[s0:s1, None]-iy[None, :])**2+(ix[s0:s1, None]-ix[None, :])**2
        r = np.hypot(x[s0:s1, None]-x[None, :], y[s0:s1, None]-y[None, :])
        b = np.searchsorted(edges, r, side="right")-1; inb = (r > 0) & (b >= 0) & (b < nb)
        mr = np.hypot(0.5*(x[s0:s1, None]+x[None, :]), 0.5*(y[s0:s1, None]+y[None, :]))
        rate = np.exp(slope*(mr-r_ref))
        C = {}
        if "p0" in comps: C["p0"] = rate*ktab["p"][lag]
        if "pN" in comps:
            gm = half[iy[s0:s1, None]+iy[None, :], ix[s0:s1, None]+ix[None, :]]
            C["pN"] = eta*rate*gm*ktab["p"][lag]
        if "np" in comps: C["np"] = betaA*rate**2*ktab["np"][lag]
        bb = np.where(inb, b, nb)                                   # out-of-range pairs -> overflow bin
        for c in comps:
            rowmean[c][s0:s1] = (C[c] @ onehot)/counts[None, :]
            numC[c] += np.bincount(bb.ravel(), weights=C[c].ravel(), minlength=nb+1)[:nb]
        rows = np.arange(s1-s0)[:, None]
        pa[s0:s1] = np.bincount((rows*(nb+1)+bb).ravel(), minlength=(s1-s0)*(nb+1)).reshape(s1-s0, nb+1)[:, :nb]
        idx = ((ann[s0:s1, None]*na+ann[None, :])*(nb+1)+bb).ravel()
        ann_pair += np.bincount(idx, minlength=na*na*(nb+1)).reshape(na, na, nb+1)[:, :, :nb]
        cnt += np.bincount(bb.ravel(), minlength=nb+1)[:nb]
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for c in comps:
            Rm = rowmean[c]; Am = (onehot.T @ Rm)/counts[:, None]                    # Cbar[i,a], Cbar[a,a']
            # sum over ordered pairs of Cbar_{i,a(j)}: need per-row partner counts per annulus -> recompute blockwise
            t2 = np.zeros(nb)
            for s0 in range(0, n, block):
                s1 = min(n, s0+block)
                r = np.hypot(x[s0:s1, None]-x[None, :], y[s0:s1, None]-y[None, :])
                b = np.searchsorted(edges, r, side="right")-1; inb = (r > 0) & (b >= 0) & (b < nb)
                bb = np.where(inb, b, nb)
                w = Rm[s0:s1][:, ann]                                               # Cbar_{i, a(j)}
                t2 += np.bincount(bb.ravel(), weights=w.ravel(), minlength=nb+1)[:nb]
            t4 = np.einsum("ac,acb->b", Am, ann_pair)
            num = numC[c]-2*t2+t4
            e0 = np.exp(slope*(np.hypot(x, y)-r_ref))
            if c == "p0": Cii = e0*ktab["p"][0]
            elif c == "pN": Cii = eta*e0*mod[valid]*ktab["p"][0]
            else: Cii = betaA*e0**2*ktab["np"][0]
            dg = Cii-2*Rm[np.arange(n), ann]+Am[ann, ann]
            den = pa.T @ dg
            out[c] = (num/np.where(cnt > 0, cnt, np.nan), den/np.where(cnt > 0, cnt, np.nan))
        if "m" in components:
            rate_c = np.exp(slope*(rad-r_ref))
            def proj(f):
                o = np.where(valid, f, 0.0).astype(float)
                for L in labs:
                    s = valid & (lab_g == L); o[s] -= o[s].mean()
                return o[valid]
            mu = np.sqrt(beta)*(proj(rate_c)+eta*proj(rate_c*mod))
            num = np.zeros(nb); den = np.zeros(nb)
            for s0 in range(0, n, block):
                s1 = min(n, s0+block)
                r = np.hypot(x[s0:s1, None]-x[None, :], y[s0:s1, None]-y[None, :])
                b = np.searchsorted(edges, r, side="right")-1; inb = (r > 0) & (b >= 0) & (b < nb)
                bb = np.where(inb, b, nb).ravel()
                num += np.bincount(bb, weights=(mu[s0:s1, None]*mu[None, :]).ravel(), minlength=nb+1)[:nb]
                den += np.bincount(bb, weights=(0.5*(mu[s0:s1, None]**2+mu[None, :]**2)).ravel(), minlength=nb+1)[:nb]
            out["m"] = (num/np.where(cnt > 0, cnt, np.nan), den/np.where(cnt > 0, cnt, np.nan))
    out["counts"] = cnt
    return out
