"""Prediction of the unified finite-map statistic xi for the controlled single-element experiment.

Inputs are the controlled source process of Sec. 3 on its own finite support (3 <= varpi/kpc <= 15):
    lambda(x)   = lambda_00 exp(-varpi/h_R) [1 + eta h0(Theta)]             (parent and Poisson events)
    lambda_1(x) = (1-f_cl) lambda(x) + f_cl 1_A(x) (lambda * G_{sigma_cl^2})(x)  (one-point rate of all events,
                  children displaced by Gaussian offsets and discarded outside the source annulus A)
    C_D         = isotropic Gaussian pair excess of the Thomas model (l_src = sqrt2 sigma_cl), written as a Gram
                  integral over the parent position with the axisymmetric parent rate lambda_0 1_A (the model's
                  isotropic approximation; the modulated-parent form is a separate diagnostic).
The response Psi is the Gaussian beam followed by sampling at the valid cell centres, exactly as in the
simulated maps; optionally an effective square-cell variance dx^2/12 can be added (observational choice)."""
import time
import numpy as np
from . import ktz_unified_ops as op


class ScalarSetup:
    def __init__(self, p):
        self.__dict__.update(p)
        self.Vb = self.sigma**2 + self.sigma_b**2 + self.cell_var          # youngest kernel variance (per axis)
        self.V1 = self.Vb + 2 * self.kappa * self.t_star                   # oldest kernel variance
        self.phase_coeff = self.m_arms / np.tan(np.deg2rad(self.pitch_deg))

    # --- age quadrature: s in [0, t*], V = Vb + 2 kappa s, Gauss-Legendre in ln V ---
    def age_nodes(self, n):
        x, w = np.polynomial.legendre.leggauss(n)
        lo, hi = np.log(self.Vb), np.log(self.V1)
        t = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        V = np.exp(t)
        ds = 0.5 * (hi - lo) * w * V / (2 * self.kappa)
        return V, ds

    # --- source-rate harmonics on the radial nodes of the support ---
    def rate_harmonics(self, nodes):
        lam0 = self.lambda00 * np.exp(-nodes / self.h_R)
        Phi = self.phase_coeff * np.log(nodes / self.R_in)
        c = {0: lam0.astype(complex)}
        for n_, g, a in zip(self.harm_n, self.harm_g, self.harm_alpha):
            k = int(n_ * self.m_arms)
            c[k] = c.get(k, 0) + self.eta * g * lam0 * np.exp(1j * (n_ * (Phi + self.Theta0) + a))
        return c


def build_prediction(S, px, py, ixh, iyh, ann, nann, edges, wpar, radii, halfw, nwin,
                     n_age=16, n_age_D=16, panel=0.025, order=6, hrho=0.01, rcut=7.0, nthreads=2,
                     do_dmod=True, log=print, rate_model='exact'):
    """Return dict with component window sums (num_c), denominator, and auxiliary arrays.
    rate_model='exact' uses lambda_1 with child truncation; 'nominal' uses the earlier local child damping."""
    t0 = time.time()
    nodes, wts = op.radial_nodes(S.R_in, S.R_out, panel, order)
    rho_out = np.arange(0.0, S.rho_max + 4 * hrho, hrho)
    lam = S.rate_harmonics(nodes)
    # one-point rate lambda_1 on the support
    if rate_model == 'exact':
        lam_cl = {k: op.smoothing_matrix(k, S.sigma_cl**2, nodes, nodes, wts) @ c for k, c in lam.items()}
        lam1 = {k: (1 - S.f_cl) * lam[k] + S.f_cl * lam_cl[k] for k in lam}
    else:   # nominal: harmonics damped with the local arm-normal wavenumber (Eq. gn_child_smoothing)
        Qb = np.sqrt((S.phase_coeff / nodes)**2 + (S.m_arms / nodes)**2)
        lam1 = {0: lam[0].copy()}
        for n_ in S.harm_n:
            k = int(n_ * S.m_arms)
            d = (1 - S.f_cl) + S.f_cl * np.exp(-0.5 * (n_ * Qb * S.sigma_cl)**2)
            lam1[k] = lam[k] * d
    log(f'  rates on {nodes.size} radial nodes, harmonics {sorted(lam1)}  ({time.time()-t0:.1f}s)')
    # same-event: Lambda_s = lambda_1 * G_{V_s/2} on the half-cell lattice of pair midpoints
    V, ds = S.age_nodes(n_age)
    lat = S.grid0 + 0.5 * S.dx * np.arange(2 * S.ngrid - 1)
    LX, LY = np.meshgrid(lat, lat)
    LR, LP = np.hypot(LX, LY), np.arctan2(LY, LX)
    inside = LR <= S.rho_max
    LamU = np.zeros((n_age,) + LX.shape); LamT = np.zeros_like(LamU)
    for s in range(n_age):
        tab = op.smooth_harmonics(lam1, V[s] / 2, rho_out, nodes, wts)
        LamU[s][inside] = op.eval_harmonics({0: tab[0]}, rho_out, LR[inside], LP[inside])
        LamT[s][inside] = op.eval_harmonics({k: v for k, v in tab.items() if k}, rho_out, LR[inside], LP[inside])
    log(f'  same-event midpoint tables ({n_age} ages)  ({time.time()-t0:.1f}s)')
    # complete mean at the valid cells and its projection
    Vm, dsm = S.age_nodes(n_age)
    rc, pc = np.hypot(px, py), np.arctan2(py, px)
    mu = np.zeros(px.size); mu_ax = np.zeros(px.size)
    for s in range(n_age):
        tab = op.smooth_harmonics(lam1, Vm[s], rho_out, nodes, wts)
        mu += dsm[s] * op.eval_harmonics(tab, rho_out, rc, pc)
        mu_ax += dsm[s] * op.eval_harmonics({0: tab[0]}, rho_out, rc, pc)
    m = op.project(mu, ann, nann)
    log(f'  complete mean and projection  ({time.time()-t0:.1f}s)')
    # distinct-event (isotropic, axisymmetric parent rate) and modulated-parent diagnostic tables
    Vd, dsd = S.age_nodes(n_age_D)
    a_d = Vd + S.sigma_cl**2
    KD = S.f_cl * S.N_ch / S.t_star
    par0 = {0: lam[0]}
    par_mod = {k: v for k, v in lam.items() if k}
    kset = np.array(sorted(par_mod), dtype=np.float64)
    D0 = np.zeros((n_age_D * n_age_D, rho_out.size))
    Dre = np.zeros((n_age_D * n_age_D, max(len(kset), 1), rho_out.size)); Dim = np.zeros_like(Dre)
    for p in range(n_age_D):
        for q in range(p, n_age_D):
            wv = a_d[p] * a_d[q] / (a_d[p] + a_d[q])
            t0v = op.smoothing_matrix(0, wv, rho_out, nodes, wts) @ par0[0]
            D0[p * n_age_D + q] = D0[q * n_age_D + p] = t0v.real
            if do_dmod:
                for kk, k in enumerate(kset):
                    tk = op.smoothing_matrix(int(k), wv, rho_out, nodes, wts) @ par_mod[int(k)]
                    Dre[p * n_age_D + q, kk] = Dre[q * n_age_D + p, kk] = tk.real
                    Dim[p * n_age_D + q, kk] = Dim[q * n_age_D + p, kk] = tk.imag
    log(f'  distinct-event parent tables ({n_age_D}^2 age pairs, dmod={do_dmod})  ({time.time()-t0:.1f}s)')
    # covariance pass
    n = px.size
    R, diag, win = op.run_rows(op.covariance_pass, n, nthreads,
                               px, py, ixh, iyh, ann, nann, rcut,
                               ds, V, LamU, LamT,
                               dsd, a_d, KD, D0, Dre, Dim, kset, rho_out[0], hrho, do_dmod,
                               edges, wpar, radii, halfw, nwin)
    log(f'  covariance pass over pairs within {rcut} kpc  ({time.time()-t0:.1f}s)')
    cnt_a = np.bincount(ann, minlength=nann).astype(float)
    Rbar = R / cnt_a[None, None, :]
    Bbar = np.stack([np.stack([np.bincount(ann, Rbar[c][:, a2], nann) for a2 in range(nann)], 1) / cnt_a[:, None]
                     for c in range(4)])
    proj = op.run_rows(op.pair_projection_terms, n, nthreads, px, py, ann, Rbar, Bbar, edges, wpar, radii, halfw, nwin)
    # projected diagonal per component
    pdiag = np.stack([diag[c] - 2 * Rbar[c, np.arange(n), ann] + Bbar[c, ann, ann] for c in range(4)])
    # mean products and denominators: S_ii for the model (C_D isotropic) and for the modulated diagnostic
    d_main = pdiag[0] + pdiag[1] + pdiag[2] + m**2
    d_dmod = pdiag[0] + pdiag[1] + pdiag[3] + m**2
    vals = np.stack([m, np.zeros(n), np.zeros(n)])
    sq = np.stack([m**2, d_main, d_dmod])
    num_m, den, cnt = op.run_rows(op.pair_products, n, nthreads, px, py, vals, sq, edges, wpar, radii, halfw, nwin)
    log(f'  projection terms, mean products and denominators  ({time.time()-t0:.1f}s)')
    comp = {'CU': win[0] + proj[0], 'CT': win[1] + proj[1], 'CD': win[2] + proj[2], 'CDmod': win[3] + proj[3],
            'SM': num_m[0]}
    # zero-lag (r = 0) normalised components: sum_i S^c_ii / sum_i S_ii over the window cells is not a pair
    # statistic; the reported r = 0 anchor is unity by normalisation.
    return dict(comp=comp, den=den[1], den_dmod=den[2], cnt=cnt, m=m, mu=mu, mu_ax=mu_ax, pdiag=pdiag,
                runtime=time.time() - t0, nodes=nodes.size)
