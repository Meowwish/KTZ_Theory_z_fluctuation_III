"""Brute-force check of the optimised single-element prediction (run: python -m validation.test_unified_prediction OUT.json).

On a narrow band of valid cells the dense matrices C_U, C_T, C_D and the projected mean are built directly in
Python from the Gram-form identities (explicit harmonic evaluation at every pair midpoint and c*), then P C P^T,
S = sum + m m^T, the window pair sums and xi_b are formed explicitly and compared with build_prediction."""
import sys, json
import numpy as np
from validation import ktz_unified_ops as op
from validation.ktz_unified_scalar import ScalarSetup, build_prediction

dx = 0.2; FW = 1 / (2 * np.sqrt(2 * np.log(2)))
gx = np.arange(-16.0, 16.0 + dx / 2, dx); X, Y = np.meshgrid(gx, gx); RG = np.hypot(X, Y)
rin, rout = 3.0 + np.sqrt(2) * dx, 3.0 + np.sqrt(2) * dx + 0.7
valid = (RG >= rin) & (RG <= rout)
iy, ix = np.nonzero(valid); px, py = X[valid], Y[valid]; n = px.size
aed = np.linspace(rin, rout, 2); aed[-1] += 1e-9; ann = op.annulus_labels(np.hypot(px, py), aed); nann = 1
edges = np.linspace(0.1, 15.0, 30); rc = 0.5 * (edges[:-1] + edges[1:])
radii = np.array([3.6]); halfw = 0.1
wpar = np.array([3.3, 3.9, np.deg2rad(30.0), np.flatnonzero(rc <= 1.5)[-1], 3.0, 3.0 / np.tan(np.deg2rad(20.0)), 3.0, 0.0])
nwin = 2 + radii.size
S = ScalarSetup(dict(R_in=3.0, R_out=15.0, kappa=0.3579, sigma=0.06, t_star=0.5, sigma_b=2 * dx * FW, cell_var=0.0,
                     m_arms=3, pitch_deg=20.0, Theta0=0.0, eta=0.40, harm_n=[1, 2, 3], harm_g=[1.0, 0.4, 0.2],
                     harm_alpha=[0, 0, 0], lambda00=500.0, h_R=4.8, f_cl=0.5, sigma_cl=0.35, N_ch=20.0,
                     dx=dx, grid0=-16.0, ngrid=gx.size, rho_max=14.77))
NA, ND = 5, 3
res = build_prediction(S, px, py, ix.astype(np.int64), iy.astype(np.int64), ann, nann, edges, wpar, radii, halfw, nwin,
                       n_age=NA, n_age_D=ND, nthreads=2, do_dmod=True, log=lambda *a: None)

# ---- direct dense construction
nodes, wts = op.radial_nodes(3.0, 15.0, 0.025, 6)
lam = S.rate_harmonics(nodes)
lam_cl = {k: op.smoothing_matrix(k, 0.35**2, nodes, nodes, wts) @ c for k, c in lam.items()}
lam1 = {k: 0.5 * lam[k] + 0.5 * lam_cl[k] for k in lam}
rho_out = np.arange(0.0, 14.77 + 0.04, 0.01)
XM, YM = 0.5 * (px[:, None] + px[None, :]), 0.5 * (py[:, None] + py[None, :])
R2 = (px[:, None] - px[None, :])**2 + (py[:, None] - py[None, :])**2
RM, PM = np.hypot(XM, YM).ravel(), np.arctan2(YM, XM).ravel()
V, ds = S.age_nodes(NA)
CU = np.zeros((n, n)); CT = np.zeros((n, n)); mu = np.zeros(n)
for v, d in zip(V, ds):
    tab = op.smooth_harmonics(lam1, v / 2, rho_out, nodes, wts)
    g = d * np.exp(-R2 / (4 * v)) / (4 * np.pi * v)
    CU += g * op.eval_harmonics({0: tab[0]}, rho_out, RM, PM).reshape(n, n)
    CT += g * op.eval_harmonics({k: t for k, t in tab.items() if k}, rho_out, RM, PM).reshape(n, n)
    tabm = op.smooth_harmonics(lam1, v, rho_out, nodes, wts)
    mu += d * op.eval_harmonics(tabm, rho_out, np.hypot(px, py), np.arctan2(py, px))
Vd, dsd = S.age_nodes(ND); a_d = Vd + 0.35**2; KD = 0.5 * 20.0 / 0.5
CD = np.zeros((n, n)); CDm = np.zeros((n, n))
for p in range(ND):
    for q in range(ND):
        A, B = a_d[p], a_d[q]; w = A * B / (A + B)
        cx = (B * px[:, None] + A * px[None, :]) / (A + B); cy = (B * py[:, None] + A * py[None, :]) / (A + B)
        cr, cp = np.hypot(cx, cy).ravel(), np.arctan2(cy, cx).ravel()
        t0 = {0: op.smoothing_matrix(0, w, rho_out, nodes, wts) @ lam[0]}
        tm = op.smooth_harmonics(lam, w, rho_out, nodes, wts)
        g = KD * dsd[p] * dsd[q] * np.exp(-R2 / (2 * (A + B))) / (2 * np.pi * (A + B))
        CD += g * op.eval_harmonics(t0, rho_out, cr, cp).reshape(n, n)
        CDm += g * op.eval_harmonics(tm, rho_out, cr, cp).reshape(n, n)
Pm = np.eye(n) - np.ones((n, n)) / n
m = Pm @ mu
comp_d = {'CU': Pm @ CU @ Pm.T, 'CT': Pm @ CT @ Pm.T, 'CD': Pm @ CD @ Pm.T, 'CDmod': Pm @ CDm @ Pm.T, 'SM': np.outer(m, m)}
Stot = comp_d['CU'] + comp_d['CT'] + comp_d['CD'] + comp_d['SM']

def window_sums(M):
    out = np.zeros((nwin, len(edges) - 1))
    for i in range(n):
        for j in range(i + 1, n):
            r = np.sqrt(R2[i, j])
            if r < edges[0] or r >= edges[-1]:
                continue
            b = int(np.searchsorted(edges, r, side='right') - 1)
            msk = op.window_mask(XM[i, j], YM[i, j], b, wpar, radii, halfw)
            for w in range(nwin):
                if (msk >> w) & 1:
                    out[w, b] += M[i, j]
    return out

D = 0.5 * (np.diag(Stot)[:, None] + np.diag(Stot)[None, :])
den_d = window_sums(D)
rep = {}
for k in comp_d:
    num_d = window_sums(comp_d[k])
    scale = np.max(np.abs(den_d))
    rep[f'pairsum_{k}_max_abs_over_den'] = float(np.max(np.abs(num_d - res['comp'][k])) / scale)
rep['den_rel'] = float(np.max(np.abs(den_d - res['den'])) / np.max(np.abs(den_d)))
with np.errstate(invalid='ignore', divide='ignore'):
    xi_d = sum(window_sums(comp_d[k]) for k in ('CU', 'CT', 'CD', 'SM')) / den_d
    xi_o = sum(res['comp'][k] for k in ('CU', 'CT', 'CD', 'SM')) / res['den']
rep['xi_max_abs_diff'] = float(np.nanmax(np.abs(xi_d - xi_o)))
rep['psd_total_min_eig_rel'] = float(np.linalg.eigvalsh(CU + CT + CD).min() / np.linalg.eigvalsh(CU + CT + CD).max())
rep['psd_CD_min_eig_rel'] = float(np.linalg.eigvalsh(CD).min() / np.linalg.eigvalsh(CD).max())
rep['psd_CDmod_min_eig_rel'] = float(np.linalg.eigvalsh(CDm).min() / np.linalg.eigvalsh(CDm).max())
rep['psd_projected_total_min_eig_rel'] = float(np.linalg.eigvalsh(Stot).min() / np.linalg.eigvalsh(Stot).max())
rep['n_cells'] = int(n)
rep['passed'] = bool(rep['xi_max_abs_diff'] < 1e-9 and rep['den_rel'] < 1e-9 and rep['psd_total_min_eig_rel'] > -1e-10
                     and rep['psd_CD_min_eig_rel'] > -1e-10 and rep['psd_projected_total_min_eig_rel'] > -1e-10)
print(json.dumps(rep, indent=1))
if len(sys.argv) > 1:
    json.dump(rep, open(sys.argv[1], 'w'), indent=1)
