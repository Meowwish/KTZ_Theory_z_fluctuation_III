"""Before/after table: archived known-profile metrics (validation/outputs/validation_metrics.json) versus the
unified finite-map statistic (validation/outputs_unified_20261103/validation_metrics_unified.json).
usage: python validation/before_after_metrics.py OUT.md OUT.json"""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parent
old = json.load(open(R / 'outputs' / 'validation_metrics.json'))
new = json.load(open(R / 'outputs_unified_20261103' / 'validation_metrics_unified.json'))
f = lambda d, k: d.get(k) if isinstance(d, dict) else None
rows = [
    ('full window: RMSE', old['xi3']['rmse'], new['xi_full']['rmse']),
    ('full window: mean residual', old['xi3']['bias'], new['xi_full']['bias']),
    ('full window: Pearson', old['xi3']['pearson'], new['xi_full']['pearson']),
    ('full window: max |residual|', old['xi3']['max_abs'], new['xi_full']['max_abs']),
    ('full window: bins within 1 sigma (of 10)', old['xi3']['within_1sigma'], new['xi_full']['within_1sigma']),
    ('full window without C_D: RMSE', old['xi3_noCNP']['rmse'], new['xi_full_noCD']['rmse']),
    ('full window without C_D: max |residual|', old['xi3_noCNP']['max_abs'], new['xi_full_noCD']['max_abs']),
    ('r=0 share of C_D', old['component_shares_r0']['xi3']['CNP'], new['component_shares_r0_full']['CD']),
    ('r=0 share of mean term (C_M before, S_M after)', old['component_shares_r0']['xi3']['CM'], new['component_shares_r0_full']['SM']),
    ('r=0 share of C_U', old['component_shares_r0']['xi3']['CP0'], new['component_shares_r0_full']['CU']),
    ('sector: RMSE', old['sector']['rmse'], new['sector']['rmse']),
    ('sector: mean residual', old['sector']['bias'], new['sector']['bias']),
    ('sector: max |residual|', old['sector']['max_abs'], new['sector']['max_abs']),
    ('sector: max |C_T|', old['sector']['max_abs_CPN'], new['sector']['max_abs_CT']),
    ('sector without C_D: RMSE', old['sector_noCNP']['rmse'], new['sector_noCD']['rmse']),
    ('sector, phase-modulated C_D diagnostic: RMSE', 0.005, new['sector_modulated_CD_diagnostic']['rmse']),
    ('sector, phase-modulated C_D diagnostic: max', 0.008, new['sector_modulated_CD_diagnostic']['max_abs']),
    ('fixed radius (xi2 before, appendix check after): RMSE', old['xi2']['rmse'], new['radial_check']['rmse']),
    ('fixed radius: mean residual', old['xi2']['bias'], new['radial_check']['bias']),
    ('fixed radius: Pearson', old['xi2']['pearson'], new['radial_check']['pearson']),
    ('fixed radius: max |residual|', old['xi2']['max_abs'], new['radial_check']['max_abs']),
    ('fixed radius: bins within 1 sigma (of 60)', old['xi2']['within_1sigma'], new['radial_check']['within_1sigma']),
    ('fixed radius without C_D: RMSE', old['xi2_noCNP']['rmse'], new['radial_check_noCD']['rmse']),
    ('Xi_AB: RMSE', old['cross']['rmse'], new['cross']['rmse']),
    ('Xi_AB: mean residual', old['cross']['bias'], new['cross']['bias']),
    ('Xi_AB: Pearson', old['cross']['pearson'], new['cross']['pearson']),
    ('Xi_AB: max |residual|', old['cross']['max_abs'], new['cross']['max_abs']),
    ('Xi_AB: pairs within 2 sigma (of 120)', old['cross']['within_2sigma'], new['cross']['within_2sigma']),
    ('Xi_AB without C_D: RMSE', old['cross_noCNP']['rmse'], new['cross_noCNP']['rmse']),
    ('Xi_AB local-diagonal variant: RMSE', None, new['cross_localdiag_variant']['rmse']),
    ('Xi_AB earlier prescription recomputed on regenerated fields: RMSE', old['cross']['rmse'], new['cross_known_profile_legacy']['rmse']),
    ('KTZ - KT25: min', old['ktz_minus_kt25']['min'], new['ktz_minus_kt25']['min']),
    ('KTZ - KT25: max', old['ktz_minus_kt25']['max'], new['ktz_minus_kt25']['max']),
]
fmt = lambda v: '—' if v is None else (f'{v:d}' if isinstance(v, int) else f'{v:.4g}')
lines = ['| quantity | before (known profile, local prediction) | after (unified xi) |', '|---|---|---|']
lines += [f'| {a} | {fmt(b)} | {fmt(c)} |' for a, b, c in rows]
open(sys.argv[1], 'w').write('\n'.join(lines) + '\n')
json.dump([dict(quantity=a, before=b, after=c) for a, b, c in rows], open(sys.argv[2], 'w'), indent=1)
print('\n'.join(lines))
