"""2026-11-04 rework: check of the projected transported mean of the multi-element statistic.
usage: python -m validation.rework_mean_check PROJ_DIR OUTDIR
PROJ_DIR holds cross_projected_values_rNN.npy (projected residual cell values, 16 elements x cells, from the
11-03 run of the same fields).  Compares the simulated ensemble mean of each element with the predicted projected
mean m_hat = P mu_hat (shape correlation and least-squares amplitude, relative to the median over elements), and
splits the simulated zero-lag variance into mean and fluctuation parts relative to the prediction."""
import sys, json, glob
import numpy as np
from pathlib import Path
PROJ, OUT = Path(sys.argv[1]), Path(sys.argv[2])
fs = sorted(glob.glob(str(PROJ / 'cross_projected_values_r*.npy')))
th = np.load(OUT / 'cross_theory_unified.npz'); res = np.load(OUT / 'cross_results_unified.npz')
EL = [str(e) for e in res['elements']]
mu = th['mu_hat'].astype(float)
# projection: subtract annulus means (same labels as production)
src = (Path(__file__).resolve().parents[1] / 'KTZ_validation.py').read_text()
from validation import ktz_unified_ops as op
dxc = 0.05; m_ = np.sqrt(2) * dxc
cgx = np.arange(-21.0, 21.0 + dxc / 2, dxc); X, Yg = np.meshgrid(cgx, cgx); R = np.hypot(X, Yg)
valid = (R >= 3.0 + m_) & (R <= 15.0 - m_)
na = int(round((15.0 - 2 * m_ - 3.0) / (10 / 3 * dxc))); aed = np.linspace(3.0 + m_, 15.0 - m_, na + 1); aed[-1] += 1e-9
ann = op.annulus_labels(R[valid], aed)
mh = op.project(mu, ann, na)
s = np.zeros_like(mh); ss = np.zeros(len(EL))
for f in fs:
    z = np.load(f).astype(float); s += z; ss += np.einsum('ij,ij->i', z, z)
n = len(fs); s /= n; ss /= n
corr = np.array([np.corrcoef(s[i], mh[i])[0, 1] for i in range(len(EL))])
amp = np.einsum('ij,ij->i', s, mh) / np.einsum('ij,ij->i', mh, mh)
pref = th['M'][np.arange(16), np.arange(16)] / (np.einsum('ij,ij->i', mh, mh) / mh.shape[1])   # theory units per (mu_hat units)^2
T = th['U'] + th['D'] + th['M']
tot_sim = ss / mh.shape[1]; mean_sim = np.einsum('ij,ij->i', s, s) / mh.shape[1]
# common scale sim -> theory from the median total-variance ratio
mu2 = (n * mean_sim - tot_sim) / (n - 1)          # unbiased squared ensemble mean
out = dict(n_realisations=n, elements=EL,
           mean_shape_correlation=dict(zip(EL, corr.tolist())),
           mean_amplitude_rel_median=dict(zip(EL, (amp / np.median(amp)).tolist())),
           max_abs_mean_amplitude_rel_median_minus1=float(np.max(np.abs(amp / np.median(amp) - 1))),
           mean_share_of_zero_lag_theory=dict(zip(EL, (np.diag(th['M']) / np.diag(T)).tolist())),
           fluct_var_ratio_sim_over_theory_common_scale_rel_median=dict(zip(EL, (lambda r: r / np.median(r))((tot_sim - mu2) / np.diag(th['U'] + th['D'])).tolist())),
           mean_var_ratio_sim_over_theory_rel_median=dict(zip(EL, (lambda r: r / np.median(r))(mu2 / np.diag(th['M'])).tolist())),
           note='sim mean variance = (n|s_bar|^2 - total)/(n-1) (unbiased); fluctuation = total - mean; ratios relative to the median over elements because Xi is scale-free per element')
print(json.dumps(out, indent=1))
json.dump(out, open(OUT / 'rework_mean_check.json', 'w'), indent=1)
