"""Independent checks of the unified operators (run: python -m validation.test_unified_ops OUT.json).

Each check compares the optimised implementation with a direct, separately coded evaluation:
projector, projected covariance, pair sums/windows, exact harmonic smoothing, Gram same-event,
distinct-event and mean integrals (brute-force polar source quadrature), PSD, cell response,
thread invariance and the Monte Carlo moment relation E[(P z)(P z)^T] = P C P^T + (P mu)(P mu)^T."""
import sys, json, time
import numpy as np
from validation import ktz_unified_ops as op
from validation.ktz_unified_scalar import ScalarSetup

rng = np.random.default_rng(1103)
REPORT = {}

def rec(name, value, tol, kind='abs'):
    ok = bool(value <= tol)
    REPORT[name] = dict(value=float(value), tol=tol, kind=kind, passed=ok)
    print(f'{"PASS" if ok else "FAIL"}  {name}: {value:.3e} (tol {tol:g})', flush=True)

# ------------------------------------------------------------------ geometry for small tests
def small_geometry(n_side=31, dx=0.2, rin=1.0, rout=3.0):
    g = (np.arange(n_side) - n_side // 2) * dx
    X, Y = np.meshgrid(g, g); R = np.hypot(X, Y)
    v = (R >= rin) & (R <= rout)
    # punch a hole to test masks
    v &= ~((X > 1.2) & (X < 1.9) & (Y > -0.4) & (Y < 0.3))
    iy, ix = np.nonzero(v)
    return X[v], Y[v], ix, iy

px, py, ix, iy = small_geometry()
n = px.size
edges_a = np.linspace(1.0, 3.0 + 1e-9, 4); nann = 3
ann = op.annulus_labels(np.hypot(px, py), edges_a)
edges = np.linspace(0.1, 4.0, 9)
radii = np.array([1.5, 2.5]); halfw = 0.1
wpar = np.array([1.2, 2.8, np.deg2rad(40.0), 4, 3.0, 3.0 / np.tan(np.deg2rad(20.0)), 1.0, 0.3])
nwin = 2 + radii.size

# T1 projector
f = rng.normal(size=n)
Pf = f.copy()
for a in range(nann):
    s = ann == a
    Pf[s] = f[s] - f[s].mean()
rec('projector_vs_loop', np.max(np.abs(op.project(f, ann, nann) - Pf)), 1e-13)
Pm = np.eye(n)
for i in range(n):
    s = ann == ann[i]
    Pm[i, s] -= 1.0 / s.sum()
rec('projector_matrix_idempotent', np.max(np.abs(Pm @ Pm - Pm)), 1e-12)

# brute-force window membership and pair sums
def brute_windows(i, j):
    mx, my = 0.5 * (px[i] + px[j]), 0.5 * (py[i] + py[j]); r = np.hypot(px[j] - px[i], py[j] - py[i])
    if r < edges[0] or r >= edges[-1]:
        return None, []
    b = int(np.searchsorted(edges, r, side='right') - 1)
    ws = [0]
    rm = np.hypot(mx, my)
    th = wpar[5] * np.log(rm / wpar[6]) - wpar[4] * np.arctan2(my, mx) + wpar[7]
    if wpar[0] < rm < wpar[1] and b <= wpar[3] and abs((th + np.pi) % (2 * np.pi) - np.pi) < wpar[2]:
        ws.append(1)
    for k, R0 in enumerate(radii):
        if abs(rm - R0) < halfw:
            ws.append(2 + k)
    return b, ws

def brute_pair_sums(M):
    """window sums over unordered pairs of the matrix elements M_ij and of (M_ii + M_jj)/2"""
    num = np.zeros((nwin, len(edges) - 1)); den = np.zeros_like(num); cnt = np.zeros_like(num)
    for i in range(n):
        for j in range(i + 1, n):
            b, ws = brute_windows(i, j)
            for w in ws:
                num[w, b] += M[i, j]; den[w, b] += 0.5 * (M[i, i] + M[j, j]); cnt[w, b] += 1
    return num, den, cnt

# T2/T3: pair products of a vector vs brute force, endpoint/ordering invariance
v = rng.normal(size=n)
num, den, cnt = op.pair_products(px, py, v[None], (v**2)[None], edges, wpar, radii, halfw, nwin, 0, n)
bn, bd, bc = brute_pair_sums(np.outer(v, v))
rec('pair_products_num_vs_brute', np.max(np.abs(num[0] - bn)), 1e-11)
rec('pair_products_den_vs_brute', np.max(np.abs(den[0] - bd)), 1e-11)
rec('pair_counts_vs_brute', np.max(np.abs(cnt - bc)), 0)
perm = rng.permutation(n)
num_p, den_p, cnt_p = op.pair_products(px[perm], py[perm], v[perm][None], (v[perm]**2)[None], edges, wpar, radii, halfw, nwin, 0, n)
rec('pair_sums_order_invariance', max(np.max(np.abs(num_p - num)), np.max(np.abs(cnt_p - cnt))), 1e-11)
REPORT['empty_bins_have_zero_counts'] = dict(n_empty=int((cnt[0] == 0).sum()), passed=bool(np.all(num[0][0][cnt[0] == 0] == 0)))

# T2b: projected covariance pair sums via Rbar/Bbar decomposition vs explicit P C P^T
A = rng.normal(size=(n, n)); C = A @ A.T / n
Cp = Pm @ C @ Pm.T
cnt_a = np.bincount(ann, minlength=nann).astype(float)
Rm = np.stack([np.bincount(ann, C[i], nann) for i in range(n)])          # R[i,a] = sum_{k in a} C_ik
Rbar = (Rm / cnt_a[None, :])[None]
Bbar = (np.stack([np.bincount(ann, Rbar[0][:, a2], nann) for a2 in range(nann)], 1) / cnt_a[:, None])[None]
proj = op.pair_projection_terms(px, py, ann, Rbar, Bbar, edges, wpar, radii, halfw, nwin, 0, n)
bnC, _, _ = brute_pair_sums(C)
bnP, bdP, _ = brute_pair_sums(Cp)
rec('projected_cov_pair_sums_decomposition', np.max(np.abs(bnC + proj[0] - bnP)), 1e-11)
pdiag = np.diag(C) - 2 * Rbar[0][np.arange(n), ann] + Bbar[0][ann, ann]
rec('projected_cov_diagonal', np.max(np.abs(pdiag - np.diag(Cp))), 1e-12)

# T11: Monte Carlo moment relation with a fixed mean, projected once
mu = rng.normal(size=n) * 0.7
L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
z = mu[None, :] + rng.normal(size=(200000, n)) @ L.T
pz = op.project(z, ann, nann)
emp = pz.T @ pz / z.shape[0]
mproj = op.project(mu, ann, nann)
theory = Cp + np.outer(mproj, mproj)
rec('mc_moment_relation_rel', np.max(np.abs(emp - theory)) / np.max(np.abs(theory)), 0.02)
# mean product enters once: S_M = (P mu)(P mu)^T is rank one
rec('mean_product_rank_one', np.linalg.svd(np.outer(mproj, mproj), compute_uv=False)[1], 1e-12)

# ------------------------------------------------------------------ source-integral checks (scalar experiment)
dx = 0.2; FW = 1 / (2 * np.sqrt(2 * np.log(2)))
S = ScalarSetup(dict(R_in=3.0, R_out=15.0, kappa=0.3579, sigma=0.06, t_star=0.5, sigma_b=2 * dx * FW, cell_var=0.0,
                     m_arms=3, pitch_deg=20.0, Theta0=0.0, eta=0.40, harm_n=[1, 2, 3], harm_g=[1.0, 0.4, 0.2],
                     harm_alpha=[0, 0, 0], lambda00=500.0, h_R=4.8, f_cl=0.5, sigma_cl=0.35, N_ch=20.0,
                     dx=dx, grid0=-16.0, ngrid=161, rho_max=14.77))
nodes, wts = op.radial_nodes(3.0, 15.0, 0.025, 6)
lam = S.rate_harmonics(nodes)

def lam_direct(rho, phi):
    th = S.phase_coeff * np.log(rho / 3.0) - 3 * phi
    h0 = np.cos(th) + 0.4 * np.cos(2 * th) + 0.2 * np.cos(3 * th)
    return np.where((rho >= 3) & (rho <= 15), 500 * np.exp(-rho / 4.8) * (1 + 0.4 * h0), 0.0)

# brute polar source quadrature (independent of the Bessel route): rho' GL panels (finer), phi' uniform
qr, qw = op.radial_nodes(3.0, 15.0, 0.02, 8)
NPHI = 3072
qp = 2 * np.pi * np.arange(NPHI) / NPHI
QR, QP = np.meshgrid(qr, qp, indexing='ij')
QX, QY = QR * np.cos(QP), QR * np.sin(QP)
QW = (qw * qr)[:, None] * (2 * np.pi / NPHI)
LAM = lam_direct(QR, QP)

def G(V, dx_, dy_):
    return np.exp(-(dx_**2 + dy_**2) / (2 * V)) / (2 * np.pi * V)

# T4 exact harmonic smoothing vs brute quadrature
pts = [(3.4, 0.2), (7.1, 2.0), (14.6, -1.1), (2.5, 0.4), (10.0, 3.0)]
err = 0.0; scale = 0.0
for w in (0.0162, 0.1, 0.39):
    rho_out = np.arange(0.0, 15.5, 0.01)
    tab = op.smooth_harmonics(lam, w, rho_out, nodes, wts)
    for rr, ph in pts:
        a = op.eval_harmonics(tab, rho_out, np.array([rr]), np.array([ph]))[0]
        x0, y0 = rr * np.cos(ph), rr * np.sin(ph)
        b = np.sum(QW * LAM * G(w, QX - x0, QY - y0))
        err = max(err, abs(a - b)); scale = max(scale, abs(b))
rec('harmonic_smoothing_vs_brute_rel', err / scale, 2e-6)

# T5 same-event Gram (midpoint identity) vs direct double-kernel source integral, f_cl = 0 rate
Vn, dsn = S.age_nodes(16)
pairs = [((4.0, 0.3), (4.4, 0.1)), ((8.0, 1.0), (9.5, 1.6)), ((3.3, 2.0), (3.6, -1.5)), ((12.0, -0.5), (13.0, -0.2))]
rho_out = np.arange(0.0, 15.5, 0.01)
err = 0.0; scale = 0.0
for (r1, p1), (r2, p2) in pairs:
    x1, y1, x2, y2 = r1 * np.cos(p1), r1 * np.sin(p1), r2 * np.cos(p2), r2 * np.sin(p2)
    xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2); rr2 = (x1 - x2)**2 + (y1 - y2)**2
    a = b = 0.0
    for V, d in zip(Vn, dsn):
        tab = op.smooth_harmonics(lam, V / 2, rho_out, nodes, wts)
        a += d * np.exp(-rr2 / (4 * V)) / (4 * np.pi * V) * op.eval_harmonics(tab, rho_out, np.array([np.hypot(xm, ym)]), np.array([np.arctan2(ym, xm)]))[0]
        b += d * np.sum(QW * LAM * G(V, QX - x1, QY - y1) * G(V, QX - x2, QY - y2))
    err = max(err, abs(a - b)); scale = max(scale, abs(b))
rec('same_event_gram_vs_direct_rel', err / scale, 2e-6)

# age-quadrature convergence of the same-event diagonal (16 vs 48 nodes, direct age integral)
x1, y1 = 5.0, 0.0
def se_diag(nage):
    V_, d_ = S.age_nodes(nage)
    return sum(d * np.sum(QW * LAM * G(V, QX - x1, QY - y1)**2) for V, d in zip(V_, d_))
rec('same_event_age_quadrature_16_vs_48_rel', abs(se_diag(16) / se_diag(48) - 1), 5e-4)

# T6 distinct-event Gram (c* identity) vs direct parent-position integral, axisymmetric parent rate
LAM0 = np.where((QR >= 3) & (QR <= 15), 500 * np.exp(-QR / 4.8), 0.0)
Vd, dsd = S.age_nodes(6)
ad = Vd + 0.35**2
rho_out = np.arange(0.0, 15.5, 0.01)
err = 0.0; scale = 0.0
for (r1, p1), (r2, p2) in pairs[:3]:
    x1, y1, x2, y2 = r1 * np.cos(p1), r1 * np.sin(p1), r2 * np.cos(p2), r2 * np.sin(p2)
    rr2 = (x1 - x2)**2 + (y1 - y2)**2
    a = b = 0.0
    for p in range(6):
        for q in range(6):
            A_, B_ = ad[p], ad[q]
            w = A_ * B_ / (A_ + B_)
            cx, cy = (B_ * x1 + A_ * x2) / (A_ + B_), (B_ * y1 + A_ * y2) / (A_ + B_)
            t0 = (op.smoothing_matrix(0, w, rho_out, nodes, wts) @ lam[0]).real
            a += dsd[p] * dsd[q] * np.exp(-rr2 / (2 * (A_ + B_))) / (2 * np.pi * (A_ + B_)) * op.cubic_uniform(rho_out, t0, np.array([np.hypot(cx, cy)]))[0]
            b += dsd[p] * dsd[q] * np.sum(QW * LAM0 * G(A_, QX - x1, QY - y1) * G(B_, QX - x2, QY - y2))
    err = max(err, abs(a - b)); scale = max(scale, abs(b))
rec('distinct_event_gram_vs_direct_rel', err / scale, 2e-6)

# child one-point stage: (lambda * G_cl) on the nodes vs brute force
tabc = {k: op.smoothing_matrix(k, 0.35**2, np.array([4.2, 9.9]), nodes, wts) @ c for k, c in lam.items()}
b = np.sum(QW * LAM * G(0.35**2, QX - 4.2 * np.cos(0.7), QY - 4.2 * np.sin(0.7)))
tab_direct = sum((c[0] if k == 0 else c[0] * np.exp(-1j * k * 0.7)).real for k, c in tabc.items())
rec('child_smoothing_vs_brute_rel', abs(tab_direct - b) / abs(b), 2e-6)

# T8 PSD of the Gram-form field covariances on a dense cell subset (diag + offdiag from the same identities)
gx = np.arange(-1.0, 1.01, 0.2) + 8.0; gy = np.arange(-1.0, 1.01, 0.2)
PX, PY = [a.ravel() for a in np.meshgrid(gx, gy)]
Vn, dsn = S.age_nodes(12)
rho_out = np.arange(0.0, 15.5, 0.01)
tabs = [op.smooth_harmonics(lam, V / 2, rho_out, nodes, wts) for V in Vn]
MX, MY = 0.5 * (PX[:, None] + PX[None, :]), 0.5 * (PY[:, None] + PY[None, :])
R2 = (PX[:, None] - PX[None, :])**2 + (PY[:, None] - PY[None, :])**2
Cse = np.zeros_like(R2); Cu = np.zeros_like(R2)
for V, d, tab in zip(Vn, dsn, tabs):
    lamm = op.eval_harmonics(tab, rho_out, np.hypot(MX, MY).ravel(), np.arctan2(MY, MX).ravel()).reshape(R2.shape)
    lam0m = op.eval_harmonics({0: tab[0]}, rho_out, np.hypot(MX, MY).ravel(), np.arctan2(MY, MX).ravel()).reshape(R2.shape)
    g = d * np.exp(-R2 / (4 * V)) / (4 * np.pi * V)
    Cse += g * lamm; Cu += g * lam0m
ev = np.linalg.eigvalsh(Cse); evu = np.linalg.eigvalsh(Cu); evt = np.linalg.eigvalsh(Cse - Cu)
rec('psd_same_event_min_eig_rel', max(0.0, -ev.min() / ev.max()), 1e-10)
rec('psd_unmodulated_min_eig_rel', max(0.0, -evu.min() / evu.max()), 1e-10)
REPORT['C_T_alone_min_eig_rel'] = dict(value=float(evt.min() / ev.max()), note='C_T is a signed correction; PSD not required')
print('info  C_T alone min eigenvalue / max(C_SE):', evt.min() / ev.max())

# T9 cell response: effective Gaussian (V + dx^2/12) vs exact square-cell average of a Gaussian kernel
def cell_avg_gauss(V, d, dxc, nq=64):
    u = (np.arange(nq) + 0.5) / nq * dxc - dxc / 2
    UX, UY = np.meshgrid(u, u)
    return np.mean(G(V, d[0] + UX, d[1] + UY))
worst = 0.0
for V in (0.0324, 0.1, 0.39):
    for d in ((0.0, 0.0), (0.3, 0.1), (0.8, -0.4)):
        ex = cell_avg_gauss(V, d, 0.2); ef = G(V + 0.2**2 / 12, *d)
        worst = max(worst, abs(ex - ef) / G(V + 0.2**2 / 12, 0, 0))
rec('cell_response_effective_gaussian_rel_to_peak', worst, 2e-2)

# T10 thread invariance of the optimised passes
num1, den1, cnt1 = op.run_rows(op.pair_products, n, 1, px, py, v[None], (v**2)[None], edges, wpar, radii, halfw, nwin)
num2, den2, cnt2 = op.run_rows(op.pair_products, n, 2, px, py, v[None], (v**2)[None], edges, wpar, radii, halfw, nwin)
rec('serial_vs_parallel_pair_products', max(np.max(np.abs(num1 - num2)), np.max(np.abs(cnt1 - cnt2))), 1e-11)

out = sys.argv[1] if len(sys.argv) > 1 else None
if out:
    json.dump(REPORT, open(out, 'w'), indent=1)
print('all passed:', all(v.get('passed', True) for v in REPORT.values() if isinstance(v, dict)))
