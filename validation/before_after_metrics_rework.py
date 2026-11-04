"""Before/after table for the 2026-11-04 rework: archived known-profile metrics, the 11-03 unified delivery
(24-node linear-age diagonal, unconverged) and the 11-04 rework (converged ln-variance diagonal).
usage: python validation/before_after_metrics_rework.py METRICS_1103.json OUT.md OUT.json"""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parent
old = json.load(open(R / 'outputs' / 'validation_metrics.json'))
d03 = json.load(open(sys.argv[1]))
new = json.load(open(R / 'outputs_rework_20261104' / 'validation_metrics_unified.json'))
ring = json.load(open(R / 'outputs_rework_20261104' / 'rework_ring_impact.json'))
mean = json.load(open(R / 'outputs_rework_20261104' / 'rework_mean_check.json'))
g = lambda d, *k: (lambda v: v)(__import__('functools').reduce(lambda a, b: a.get(b) if isinstance(a, dict) else None, k, d))
rows = [
    ('full window: RMSE', g(old, 'xi3', 'rmse'), g(d03, 'xi_full', 'rmse'), g(new, 'xi_full', 'rmse')),
    ('full window: mean residual', g(old, 'xi3', 'bias'), g(d03, 'xi_full', 'bias'), g(new, 'xi_full', 'bias')),
    ('full window: max |residual|', g(old, 'xi3', 'max_abs'), g(d03, 'xi_full', 'max_abs'), g(new, 'xi_full', 'max_abs')),
    ('sector: RMSE', g(old, 'sector', 'rmse'), g(d03, 'sector', 'rmse'), g(new, 'sector', 'rmse')),
    ('fixed radius: RMSE', g(old, 'xi2', 'rmse'), g(d03, 'radial_check', 'rmse'), g(new, 'radial_check', 'rmse')),
    ('Xi_AB: RMSE', g(old, 'cross', 'rmse'), g(d03, 'cross', 'rmse'), g(new, 'cross', 'rmse')),
    ('Xi_AB: mean residual', g(old, 'cross', 'bias'), g(d03, 'cross', 'bias'), g(new, 'cross', 'bias')),
    ('Xi_AB: Pearson', g(old, 'cross', 'pearson'), g(d03, 'cross', 'pearson'), g(new, 'cross', 'pearson')),
    ('Xi_AB: max |residual|', g(old, 'cross', 'max_abs'), g(d03, 'cross', 'max_abs'), g(new, 'cross', 'max_abs')),
    ('Xi_AB: pairs within 2 sigma (of 120)', g(old, 'cross', 'within_2sigma'), g(d03, 'cross', 'within_2sigma'), g(new, 'cross', 'within_2sigma')),
    ('Xi_AB without C_D: RMSE', g(old, 'cross_noCNP', 'rmse'), g(d03, 'cross_noCNP', 'rmse'), g(new, 'cross_noCNP', 'rmse')),
    ('Xi_AB without C_D: max |residual|', g(old, 'cross_noCNP', 'max_abs'), g(d03, 'cross_noCNP', 'max_abs'), g(new, 'cross_noCNP', 'max_abs')),
    ('Xi_AB local-diagonal variant: RMSE', None, g(d03, 'cross_localdiag_variant', 'rmse'), g(new, 'cross_localdiag_variant', 'rmse')),
    ('Xi_AB 11-03 24-node diagonal recomputed: RMSE', None, g(d03, 'cross', 'rmse'), g(new, 'cross_old_linear24_variant', 'rmse')),
    ('Xi_AB 24-node vs converged: max |change|', None, None, g(new, 'cross_old_linear24_variant', 'max_abs_change')),
    ('Xi_AB 45 pairs without N/s-process: RMSE', None, None, g(new, 'cross_groups', 'without_N_or_s_process', 'rmse')),
    ('Xi_AB 45 pairs without N/s-process: mean residual', None, None, g(new, 'cross_groups', 'without_N_or_s_process', 'bias')),
    ('Xi_AB 75 pairs with N/s-process: RMSE', None, None, g(new, 'cross_groups', 'with_N_or_s_process', 'rmse')),
    ('Xi_AB 75 pairs with N/s-process: mean residual', None, None, g(new, 'cross_groups', 'with_N_or_s_process', 'bias')),
    ('ring check: max |dXi| among N,O,Fe,Ba', None, None, ring['max_abs_dXi_among_N_O_Fe_Ba']),
    ('projected mean: max |relative amplitude - 1|', None, None, mean['max_abs_mean_amplitude_rel_median_minus1']),
    ('KTZ - KT25: min', g(old, 'ktz_minus_kt25', 'min'), g(d03, 'ktz_minus_kt25', 'min'), g(new, 'ktz_minus_kt25', 'min')),
    ('KTZ - KT25: max', g(old, 'ktz_minus_kt25', 'max'), g(d03, 'ktz_minus_kt25', 'max'), g(new, 'ktz_minus_kt25', 'max')),
]
fmt = lambda v: '—' if v is None else (f'{v:d}' if isinstance(v, int) else f'{v:.4g}')
lines = ['| quantity | before (known profile) | 11-03 unified (24-node diagonal) | 11-04 rework |', '|---|---|---|---|']
lines += [f'| {a} | {fmt(b)} | {fmt(c)} | {fmt(d)} |' for a, b, c, d in rows]
open(sys.argv[2], 'w').write('\n'.join(lines) + '\n')
json.dump([dict(quantity=a, before=b, delivered_1103=c, rework_1104=d) for a, b, c, d in rows], open(sys.argv[3], 'w'), indent=1)
print('\n'.join(lines))
