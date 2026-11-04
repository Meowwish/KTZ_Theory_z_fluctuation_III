"""2026-11-04 rework: convergence of the multi-element same-event diagonal source integral.

usage: python -m validation.rework_cross_diag OUTDIR
Writes OUTDIR/cross_diag_source_<config>.npz (tables, cell sums, U matrices) and OUTDIR/rework_cross_diag.json.
1. analytic limit: uniform rate (Sigma = const) for every channel pair used -> exact logarithm ln[(S1+b)/(S0+b)];
2. age convergence (n_t = 16..256) of the full diagonal for all element/channel pairs with the real tables;
3. spatial-table convergence: ln w grid 96 vs 192 nodes; radial panels/rho step halved at a subset of w."""
import os, sys, json, time
import numpy as np
from pathlib import Path
from validation import ktz_unified_ops as op
from validation import ktz_unified_cross as ucx

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / 'KTZ_validation.py').read_text()
scope = dict(ROOT=ROOT, np=np, sys=sys)
blk = src.split('# %% Block 9:', 1)[1].split('\n', 1)[1].split('# %% Block 10:', 1)[0]
exec(compile(blk, 'KTZ_validation.py:Block9', 'exec'), scope)
el = ['C', 'N', 'O', 'Mg', 'Si', 'S', 'Ca', 'V', 'Mn', 'Fe', 'Ni', 'Sr', 'Y', 'Zr', 'Ba', 'Ce']
idx = [scope['kt25_elements'].index(e) for e in el]
kappa, sv, tc = scope['kt25_kappa'], scope['kt25_sigma_v'], scope['kt25_t_corr']
delay, Y2 = scope['kt25_delay'], scope['kt25_second_yield_normalized']
sig_b = 0.1 / (2 * np.sqrt(2 * np.log(2)))
V0 = scope['kt25_sigma_inj']**2 + sig_b**2
# geometry of the multi-element valid cells (as in KTZ_validation.py Block 11/11b)
dxc = 0.05; m_ = np.sqrt(2) * dxc
cgx = np.arange(-21.0, 21.0 + dxc / 2, dxc); X, Yg = np.meshgrid(cgx, cgx); R = np.hypot(X, Yg)
cell_r = R[(R >= 3.0 + m_) & (R <= 15.0 - m_)]
rep = dict(version=ucx.DIAG_SOURCE_VERSION, n_cells=int(cell_r.size))
pairs = [(p, q) for p in range(16) for q in range(p, 16)]
chan = [(p, q, a, b) for p, q in pairs for a in range(3) for b in range(3) if Y2[idx[p], a, idx[q], b] != 0]
rep['n_channel_pairs'] = len(chan)
log = lambda *a: print(*a, flush=True)

def U_matrix(lnw, sums, n_t, fun='new'):
    U = np.zeros((16, 16))
    for p, q, a, b in chan:
        i, j = idx[p], idx[q]
        if fun == 'new':
            v = ucx.same_parent_diag_source(Y2[i, a, j, b], V0[a], V0[b], delay[i, a], delay[j, b], kappa, sv, tc, lnw, sums, n_t=n_t)
        else:
            v = ucx.same_parent_diag_exact(None, None, None, None, Y2[i, a, j, b], V0[a], V0[b], delay[i, a], delay[j, b], None,
                                           kappa, sv, tc, lnw, sumtab=sums, n_s=n_t)
        U[p, q] += v
    return U + np.triu(U, 1).T

# 1. analytic uniform-rate limit for every channel pair
lnw_c = np.linspace(np.log(4e-4), np.log(60.0), 96); ones = np.ones(96)
worst_new, worst_old = {}, {}
for n_t in (16, 32, 64, 128):
    e = 0.0
    for p, q, a, b in chan:
        i, j = idx[p], idx[q]
        s0 = max(delay[i, a], delay[j, b])
        if s0 >= tc:
            continue
        beta = (sv * (delay[j, b] - delay[i, a]))**2
        S0 = V0[a] + V0[b] + 2 * kappa * (2 * s0 - delay[i, a] - delay[j, b]); S1 = V0[a] + V0[b] + 2 * kappa * (2 * tc - delay[i, a] - delay[j, b])
        exact = Y2[i, a, j, b] * np.log((S1 + beta) / (S0 + beta))
        v = ucx.same_parent_diag_source(Y2[i, a, j, b], V0[a], V0[b], delay[i, a], delay[j, b], kappa, sv, tc, lnw_c, ones, n_t=n_t)
        e = max(e, abs(v / exact - 1))
    worst_new[n_t] = e
for n_s in (24, 48, 96, 192):
    e = 0.0
    for p, q, a, b in chan:
        i, j = idx[p], idx[q]
        s0 = max(delay[i, a], delay[j, b])
        if s0 >= tc:
            continue
        beta = (sv * (delay[j, b] - delay[i, a]))**2
        S0 = V0[a] + V0[b] + 2 * kappa * (2 * s0 - delay[i, a] - delay[j, b]); S1 = V0[a] + V0[b] + 2 * kappa * (2 * tc - delay[i, a] - delay[j, b])
        exact = Y2[i, a, j, b] * np.log((S1 + beta) / (S0 + beta))
        v = ucx.same_parent_diag_exact(None, None, None, None, Y2[i, a, j, b], V0[a], V0[b], delay[i, a], delay[j, b], None,
                                       kappa, sv, tc, lnw_c, sumtab=ones, n_s=n_s)
        e = max(e, abs(v / exact - 1))
    worst_old[n_s] = e
rep['analytic_uniform_limit_max_rel_err'] = dict(new_lnS={str(k): v for k, v in worst_new.items()},
                                                 old_linear_s={str(k): v for k, v in worst_old.items()})
log('analytic', rep['analytic_uniform_limit_max_rel_err'])

# 3a. spatial tables: baseline (96 ln w nodes, rho step 0.005) and refined ln w grid (192 nodes)
rho_tab = np.arange(3.07 - 0.05, 14.93 + 0.05, 0.005)
t0 = time.time()
cfg = {}
for nw in (96, 192):
    f = OUT / f'cross_diag_tables_nw{nw}_rho0.005_p0.004-0.02.npz'
    lnw = np.linspace(np.log(4e-4), np.log(60.0), nw)
    if f.exists():
        z = np.load(f); tabs, sums = z['tabs'], z['sums']
    else:
        tabs, sums, nn = ucx.axisymmetric_rate_tables(0.30, 21.0, 4.8, 0.5, 0.35, rho_tab, lnw, cell_r)
        tmp = OUT / (f.stem + '.tmp.npz'); np.savez(tmp, tabs=tabs, sums=sums, lnw=lnw, rho_tab=rho_tab, version=ucx.DIAG_SOURCE_VERSION); os.replace(tmp, f)
    cfg[nw] = (lnw, sums, tabs)
    log(f'tables nw={nw} ready ({time.time()-t0:.0f}s)')
# 3b. radial refinement at a subset of w: panels and rho step halved
sub = np.arange(0, 96, 4)
lnw_s = cfg[96][0][sub]
f = OUT / 'cross_diag_tables_subset_rho0.0025_p0.002-0.01.npz'
if f.exists():
    z = np.load(f); sums_f = z['sums']
else:
    rho_f = np.arange(3.07 - 0.05, 14.93 + 0.05, 0.0025)
    sums_f, _ = ucx.axisymmetric_rate_sums_light(0.30, 21.0, 4.8, 0.5, 0.35, rho_f, lnw_s, cell_r, panel_inner=0.002, panel_outer=0.01)
    tmp = OUT / (f.stem + '.tmp.npz'); np.savez(tmp, sums=sums_f, lnw=lnw_s); os.replace(tmp, f)
rep['radial_refinement_max_rel_sum_change'] = float(np.max(np.abs(sums_f / cfg[96][1][sub] - 1)))
log('radial refinement', rep['radial_refinement_max_rel_sum_change'], f'({time.time()-t0:.0f}s)')

# 2. age convergence with the real tables, and ln w grid convergence
Us = {}
for n_t in (16, 32, 64, 128, 256):
    Us[n_t] = U_matrix(cfg[192][0], cfg[192][1], n_t)
U96 = U_matrix(cfg[96][0], cfg[96][1], 128)
rep['age_convergence_max_rel_U_vs_256'] = {str(n): float(np.max(np.abs(Us[n] / Us[256] - 1))) for n in (16, 32, 64, 128)}
rep['lnw_grid_96_vs_192_max_rel_U'] = float(np.max(np.abs(U96 / Us[128] - 1)))
Uold24 = U_matrix(cfg[96][0], cfg[96][1], 24, fun='old')
rep['old_linear24_vs_new_max_rel_U'] = float(np.max(np.abs(Uold24 / Us[128] - 1)))
log(json.dumps(rep, indent=1))
np.savez(OUT / 'cross_diag_source_v2_nt128_nw192.npz', U_diag_source=Us[128], U_diag_source_nt256=Us[256], U_nw96=U96,
         U_old_linear24=Uold24, lnw=cfg[192][0], sums=cfg[192][1], version=ucx.DIAG_SOURCE_VERSION,
         config=json.dumps(dict(n_t=128, n_lnw=192, lnw_range=[4e-4, 60.0], rho_step=0.005, panels=[0.004, 0.02])))
json.dump(rep, open(OUT / 'rework_cross_diag.json', 'w'), indent=1)
