"""Compare convergence variants of the single-element prediction with the production prediction.
usage: python -m validation.compare_convergence OUTDIR  (OUTDIR = validation/outputs_unified_20261103)"""
import sys, json, glob, os
import numpy as np

od = sys.argv[1]
prod = np.load(os.path.join(od, 'scalar_prediction_unified_a16_d16.npz'))
edges = np.linspace(0.1, 15.0, 30); rc = 0.5 * (edges[:-1] + edges[1:])
masks = {0: rc <= 5.0, 1: rc <= 1.5}
for w in range(2, 8):
    masks[w] = rc <= 5.0

def xi(z, key_den='den', comps=('CU', 'CT', 'CD', 'SM')):
    with np.errstate(invalid='ignore', divide='ignore'):
        return sum(z[k] for k in comps) / z[key_den]

ref = xi(prod); ref_m = xi(prod, 'den_dmod', ('CU', 'CT', 'CDmod', 'SM'))
out = {}
for f in sorted(glob.glob(os.path.join(od, 'convergence', 'pred_*.npz'))):
    z = np.load(f); v = xi(z)
    d = max(float(np.nanmax(np.abs(v[w][masks[w]] - ref[w][masks[w]]))) for w in masks)
    rec = dict(max_abs_dxi_all_windows=d)
    if np.any(z['CDmod']):
        vm = xi(z, 'den_dmod', ('CU', 'CT', 'CDmod', 'SM'))
        rec['max_abs_dxi_modulated_CD'] = max(float(np.nanmax(np.abs(vm[w][masks[w]] - ref_m[w][masks[w]]))) for w in masks)
    for k in ('CU', 'CT', 'CD', 'SM'):
        rec[f'max_abs_d_{k}_over_den'] = float(np.nanmax(np.abs((z[k] - prod[k]) / prod['den'])[0][masks[0]]))
    out[os.path.basename(f)] = rec
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(od, 'convergence', 'convergence_summary.json'), 'w'), indent=1)
