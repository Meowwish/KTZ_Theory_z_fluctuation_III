"""Estimate the effect on Xi of the local annular-mean correction (2026-11-04 rework).
usage: python -m validation.rework_ring_impact OUTDIR
Uses the exact/local ratios of rework_ring_check.json at the inner, middle and outer annuli, interpolated
linearly in annulus radius, to rescale the per-annulus local correction for the pairs among {N, O, Fe, Ba};
reports the resulting change of Xi_AB relative to the production prediction (which uses the local correction)."""
import sys, json
import numpy as np
from pathlib import Path
from validation import ktz_unified_ops as op
from validation import ktz_unified_cross as ucx

OUT = Path(sys.argv[1]); ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / 'KTZ_validation.py').read_text()
scope = dict(ROOT=ROOT, np=np, sys=sys)
exec(compile(src.split('# %% Block 9:', 1)[1].split('\n', 1)[1].split('# %% Block 10:', 1)[0], 'Block9', 'exec'), scope)
EL = ['C', 'N', 'O', 'Mg', 'Si', 'S', 'Ca', 'V', 'Mn', 'Fe', 'Ni', 'Sr', 'Y', 'Zr', 'Ba', 'Ce']
idx = {e: scope['kt25_elements'].index(e) for e in EL}
kappa, sv = scope['kt25_kappa'], scope['kt25_sigma_v']; delay, Y2 = scope['kt25_delay'], scope['kt25_second_yield_normalized']
V0 = scope['kt25_sigma_inj']**2 + (0.1 / (2 * np.sqrt(2 * np.log(2))))**2
dxc = 0.05; m_ = np.sqrt(2) * dxc; h_R = 4.8
cgx = np.arange(-21.0, 21.0 + dxc / 2, dxc); X, Yg = np.meshgrid(cgx, cgx); R = np.hypot(X, Yg)
valid = (R >= 3.0 + m_) & (R <= 15.0 - m_)
na = int(round((15.0 - 2 * m_ - 3.0) / (10 / 3 * dxc))); aed = np.linspace(3.0 + m_, 15.0 - m_, na + 1); aed[-1] += 1e-9
ann = op.annulus_labels(R[valid], aed); ann2d = np.full(R.shape, -1); ann2d[valid] = ann
va = np.bincount(ann, R[valid], na) / np.bincount(ann, minlength=na)
chk = json.load(open(OUT / 'rework_ring_check.json'))

def params(A, B):
    i, j = idx[A], idx[B]; o = [[], [], [], []]
    for a in range(3):
        for b in range(3):
            y = Y2[i, a, j, b]
            if y == 0:
                continue
            la2, lb2 = kappa * delay[i, a], kappa * delay[j, b]
            o[0].append(y); o[1].append(4 - 2 * la2 - 2 * lb2 + V0[a] + V0[b]); o[2].append(2 * abs(la2 - lb2) + V0[a] + V0[b])
            o[3].append((sv * abs(delay[i, a] - delay[j, b]))**2)
    return [np.array(v) for v in o]

pairs = [('O', 'O'), ('Ba', 'Ba'), ('Fe', 'Fe'), ('N', 'N'), ('O', 'Ba'), ('N', 'Fe'), ('O', 'Fe'), ('Fe', 'Ba'), ('N', 'O'), ('N', 'Ba')]
# per-annulus local ring sums (same construction as production)
ring_loc = {p: np.zeros(na) for p in pairs}
prm = {p: params(*p) for p in pairs}
Nfft = 1 << int(np.ceil(np.log2(2 * R.shape[0])))
_ii = np.arange(Nfft); _ii = np.where(_ii < Nfft // 2, _ii, _ii - Nfft); _n2 = (_ii[:, None]**2 + _ii[None, :]**2).ravel()
for a in range(na):
    # exact offset counts of annulus a with itself (one FFT autocorrelation per annulus)
    I = np.zeros((Nfft, Nfft)); I[:R.shape[0], :R.shape[1]] = (ann2d == a)
    F = np.fft.rfft2(I); ac = np.rint(np.fft.irfft2(F * np.conj(F), s=(Nfft, Nfft))).ravel()
    H = np.bincount(_n2, ac, int(_n2.max()) + 1) * np.exp(-va[a] / h_R) / np.sum(ann == a)   # production weight lambda_hat0(a)/N_a
    nz = np.flatnonzero(H); rn = np.sqrt(nz) * dxc
    for p in pairs:
        ring_loc[p][a] = np.sum(H[nz] * ucx.same_parent_kernel_r(rn, *prm[p]))
    if a % 10 == 0:
        print('annulus', a, flush=True)
ncell = int(valid.sum())
dU = {}
for A, B in pairs:
    rr = [(chk[f'{A}-{B}@{a}']['annulus'], chk[f'{A}-{B}@{a}']['ratio']) for a in (0, na // 2, na - 1)]
    ratio = np.interp(va, [va[a] for a, _ in rr], [r for _, r in rr])
    dU[(A, B)] = -np.sum(ring_loc[(A, B)] * (ratio - 1)) / ncell        # extra subtraction from the exact ring
diag = sorted(OUT.glob('cross_same_parent_diag_source_*.npz'))[0]
z = np.load(diag); th = np.load(OUT / 'cross_theory_unified.npz')
U = (z['U_diag_source'] - (z['U_diag_local'] - th['U'] * ncell)) / ncell
T = U + th['D'] + th['M']
sub = ['N', 'O', 'Fe', 'Ba']; ii = [EL.index(e) for e in sub]
T2 = T.copy()
for (A, B), d in dU.items():
    p, q = EL.index(A), EL.index(B); T2[p, q] += d; T2[q, p] = T2[p, q]
nrm = lambda M: M / np.sqrt(np.outer(np.diag(M), np.diag(M)))
X1, X2 = nrm(T)[np.ix_(ii, ii)], nrm(T2)[np.ix_(ii, ii)]
iu = np.triu_indices(4, 1)
out = dict(pairs_checked=[f'{a}-{b}' for a, b in pairs], ring_fraction_of_diag={f'{a}-{b}': float(ring_loc[(a, b)].sum() / (z['U_diag_source'][EL.index(a), EL.index(b)])) for a, b in pairs},
           rel_change_of_U={f'{a}-{b}': float(dU[(a, b)] / U[EL.index(a), EL.index(b)]) for a, b in pairs},
           max_abs_dXi_among_N_O_Fe_Ba=float(np.max(np.abs((X2 - X1)[iu]))), dXi={f'{sub[i]}-{sub[j]}': float((X2 - X1)[i, j]) for i, j in zip(*iu)})
print(json.dumps(out, indent=1))
json.dump(out, open(OUT / 'rework_ring_impact.json', 'w'), indent=1)
