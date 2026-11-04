"""2026-11-04 rework: targeted check of the local annular-mean correction of the multi-element statistic.

usage: python -m validation.rework_ring_check OUTDIR
For representative element pairs (short-delay CCSN, AGB, long-delay TNSN mix, mixed-delay cross pairs) and
three annuli (inner, middle, outer), compares the annular sum sum_{j,k in a} C_AB(x_j, x_k) of the same-event
term evaluated (i) with the local isotropic kernel at the annulus rate and exact discrete offset counts, as in
the production prediction, and (ii) by the source-position integral over the axisymmetric one-point rate with
the common-velocity drift average (ktz_unified_cross.same_parent_offdiag_source, written out in numba).
The ratio is then interpolated in radius to all annuli to estimate the effect on Xi for the pairs among
{N, O, Fe, Ba}.  This is a check of the approximation, not a replacement of the production prediction."""
import os, sys, json, time
import numpy as np
import numba as nb
from pathlib import Path
from validation import ktz_unified_ops as op
from validation import ktz_unified_cross as ucx

OUT = Path(sys.argv[1])
ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / 'KTZ_validation.py').read_text()
scope = dict(ROOT=ROOT, np=np, sys=sys)
exec(compile(src.split('# %% Block 9:', 1)[1].split('\n', 1)[1].split('# %% Block 10:', 1)[0], 'Block9', 'exec'), scope)
EL = ['C', 'N', 'O', 'Mg', 'Si', 'S', 'Ca', 'V', 'Mn', 'Fe', 'Ni', 'Sr', 'Y', 'Zr', 'Ba', 'Ce']
idx = {e: scope['kt25_elements'].index(e) for e in EL}
kappa, sv, tc = scope['kt25_kappa'], scope['kt25_sigma_v'], scope['kt25_t_corr']
delay, Y2 = scope['kt25_delay'], scope['kt25_second_yield_normalized']
sig_b = 0.1 / (2 * np.sqrt(2 * np.log(2))); V0 = scope['kt25_sigma_inj']**2 + sig_b**2
dxc = 0.05; m_ = np.sqrt(2) * dxc; h_R = 4.8
cgx = np.arange(-21.0, 21.0 + dxc / 2, dxc); X, Yg = np.meshgrid(cgx, cgx); R = np.hypot(X, Yg)
valid = (R >= 3.0 + m_) & (R <= 15.0 - m_)
na = int(round((15.0 - 2 * m_ - 3.0) / (10 / 3 * dxc))); aed = np.linspace(3.0 + m_, 15.0 - m_, na + 1); aed[-1] += 1e-9
ann2d = np.full(R.shape, -1); ann2d[valid] = op.annulus_labels(R[valid], aed)
cx, cy, ca = X[valid], Yg[valid], ann2d[valid]
z = np.load(OUT / 'cross_diag_tables_nw192_rho0.005_p0.004-0.02.npz')
tabs, lnw, rho_tab = z['tabs'], z['lnw'], z['rho_tab']

@nb.njit(parallel=True, cache=True)
def ring_sum(x, y, Sb, cA, cB, kv, wt, tabn, rho0, hr):
    n = x.size; nt = Sb.size; acc = np.zeros(n)
    for j in nb.prange(n):
        s = 0.0
        for k in range(n):
            dx = x[j] - x[k]; dy = y[j] - y[k]; r2 = dx * dx + dy * dy
            for t in range(nt):
                px = cA[t] * x[j] + cB[t] * x[k] + kv[t] * dx; py = cA[t] * y[j] + cB[t] * y[k] + kv[t] * dy
                rr = np.sqrt(px * px + py * py); u = (rr - rho0) / hr; i = int(np.floor(u))
                if i < 1: i = 1
                if i > tabn.shape[1] - 3: i = tabn.shape[1] - 3
                f = u - i
                v = (-f * (f - 1) * (f - 2) / 6 * tabn[t, i - 1] + (f + 1) * (f - 1) * (f - 2) / 2 * tabn[t, i]
                     - (f + 1) * f * (f - 2) / 2 * tabn[t, i + 1] + (f + 1) * f * (f - 1) / 6 * tabn[t, i + 2])
                s += wt[t] * np.exp(-r2 / (2 * Sb[t])) * v
        acc[j] = s
    return acc.sum()

def exact_ring(A, B, a_idx, n_t=32):
    sel = ca == a_idx; x, y = cx[sel], cy[sel]
    Sb_l, cA_l, cB_l, kv_l, wt_l, tb_l = [], [], [], [], [], []
    i, j = idx[A], idx[B]
    xg, wg = np.polynomial.legendre.leggauss(n_t)
    for a in range(3):
        for b in range(3):
            yv = Y2[i, a, j, b]; Da, Db = delay[i, a], delay[j, b]; s0 = max(Da, Db)
            if yv == 0 or s0 >= tc:
                continue
            d = Db - Da; beta = sv**2 * d**2
            S0 = V0[a] + V0[b] + 2 * kappa * (2 * s0 - Da - Db); S1 = V0[a] + V0[b] + 2 * kappa * (2 * tc - Da - Db)
            t0, t1 = np.log(S0 + beta), np.log(S1 + beta)
            for tt, ww in zip(0.5 * (t1 - t0) * xg + 0.5 * (t1 + t0), 0.5 * (t1 - t0) * wg):
                Sb = np.exp(tt); S = Sb - beta; s = (S - V0[a] - V0[b] + 2 * kappa * (Da + Db)) / (4 * kappa)
                Va = V0[a] + 2 * kappa * (s - Da); Vb = V0[b] + 2 * kappa * (s - Db); mm = (Vb * Da + Va * Db) / S
                w2 = Va * Vb / S + mm**2 * sv**2 * S / Sb
                Sb_l.append(Sb); cA_l.append(Vb / S); cB_l.append(Va / S); kv_l.append(sv**2 * d * mm / Sb); wt_l.append(yv * ww)
                tb_l.append(ucx._cub_axis0(lnw, tabs, np.log(w2)))
    return ring_sum(x, y, np.array(Sb_l), np.array(cA_l), np.array(cB_l), np.array(kv_l), np.array(wt_l), np.array(tb_l),
                    rho_tab[0], rho_tab[1] - rho_tab[0]), int(sel.sum())

def channel_params(A, B):
    i, j = idx[A], idx[B]; ys, nu, de, be = [], [], [], []
    for a in range(3):
        for b in range(3):
            yv = Y2[i, a, j, b]
            if yv == 0:
                continue
            la2, lb2 = kappa * delay[i, a], kappa * delay[j, b]
            ys.append(yv); nu.append(4 - 2 * la2 - 2 * lb2 + V0[a] + V0[b]); de.append(2 * abs(la2 - lb2) + V0[a] + V0[b])
            be.append((sv * abs(delay[i, a] - delay[j, b]))**2)
    return [np.array(v) for v in (ys, nu, de, be)]

# local ring sum per annulus (production approximation): lambda_hat0(varpi_a) sum_n hist_a[n] k_AB(r_n)
def local_ring(A, B, a_idx):
    w = np.zeros(na); w[a_idx] = 1.0
    H, _ = ucx.annulus_offset_histograms(valid, ann2d, na, dxc, w)
    nz = np.flatnonzero(H); rn = np.sqrt(nz) * dxc
    va = R[valid][ca == a_idx].mean()
    return np.exp(-va / h_R) * np.sum(H[nz] * ucx.same_parent_kernel_r(rn, *channel_params(A, B)))

t0 = time.time()
pairs = [('O', 'O'), ('Ba', 'Ba'), ('Fe', 'Fe'), ('N', 'N'), ('O', 'Ba'), ('N', 'Fe'), ('O', 'Fe'), ('Fe', 'Ba'), ('N', 'O'), ('N', 'Ba')]
annuli = [0, na // 2, na - 1]
res = {}
for A, B in pairs:
    for a in annuli:
        ex, n_a = exact_ring(A, B, a)
        lo = local_ring(A, B, a)
        res[f'{A}-{B}@{a}'] = dict(annulus=a, n_cells=n_a, exact=ex, local=lo, ratio=ex / lo)
        print(A, B, a, n_a, f'exact/local = {ex/lo:.5f}  ({time.time()-t0:.0f}s)', flush=True)
# age-node convergence of the exact ring sum for two cases
for (A, B, a) in [('Ba', 'Ba', 0), ('O', 'Ba', na - 1)]:
    e32, _ = exact_ring(A, B, a, 32); e64, _ = exact_ring(A, B, a, 64)
    res[f'nt_convergence_{A}-{B}@{a}'] = dict(rel_change_32_to_64=e64 / e32 - 1)
json.dump(res, open(OUT / 'rework_ring_check.json', 'w'), indent=1)
print('done', time.time() - t0, flush=True)
