"""Redraw Figs. 2 (xi2), 3 (xi3, arm sector) from archived arrays, without simulation (label update, 2026-10-28).

Run from the repository root:
    python validation/replot_scalar_components.py --outdir /tmp/ktz-figures --dump /tmp/scalar-figure-fingerprints.json

Executes only Block 1 (parameters), Block 2 (geometry), the analysis-window definitions of Block 4, the bin
definitions of Block 5 and the deterministic KTZ prediction of Block 6 (no events are generated, no pair
statistics are measured).  The recomputed theory arrays are compared bin by bin with
validation/outputs/scalar_theory.npz; the simulation means, scatters and SEMs are taken from
validation/outputs/scalar_simulation_statistics.npz.  The plotting code of Block 8 is executed verbatim
(METRICS are recomputed in memory and compared with validation/outputs/validation_metrics.json, not written).
"""
import argparse, hashlib, json, os, sys, tempfile
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--script', default='KTZ_validation.py')
ap.add_argument('--outdir', default='figs')
ap.add_argument('--dump')
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
script = Path(args.script) if Path(args.script).is_absolute() else ROOT / args.script
src = script.read_text()
tmp = tempfile.mkdtemp(prefix='ktz_replot_')
os.environ['KTZ_VALIDATION_ROOT'] = tmp          # Block 1 creates figs/ and outputs/ here, never in the repository
os.environ['KTZ_QUICK'] = '0'
ns = {}

def run(a, b, tag):
    i = src.index(a); j = src.index(b, i)
    exec(compile(src[i:j], f'{script.name} [{tag}]', 'exec'), ns)

run('# %% Block 1:', '# %% Block 3:', 'parameters, geometry')
run('window_margin = ', 'Z_bg = ', 'analysis window')
run('radii_test = ', 'block = 64', 'bins')
run('# %% Block 6:', '# %% Block 7:', 'deterministic KTZ prediction')

out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
ns['figure_path'] = lambda stem: out / f'validation_{stem}.png'
A = dict(np.load(ROOT / 'validation/outputs/scalar_theory.npz'))
S = dict(np.load(ROOT / 'validation/outputs/scalar_simulation_statistics.npz'))
assert np.array_equal(S['r_bins'], ns['r_bins']) and np.array_equal(S['radii'], ns['radii_test']) and np.array_equal(A['r_centers'], ns['r_centers'])
order = ns['component_order'] + ['noCNP']
check = {}
for vm in ns['radii_test']:
    for k in order:
        check[f'xi2_{vm}_{k}'] = float(np.nanmax(np.abs(ns['th_xi2'][vm][k] - A[f'xi2_{vm}_{k}'])))
for k in order:
    check[f'xi3_{k}'] = float(np.nanmax(np.abs(ns['th_xi3'][k] - A[f'xi3_{k}'])))
    check[f'sector_{k}'] = float(np.nanmax(np.abs(ns['th_sec'][k] - A[f'sector_{k}'])))
max_theory_diff = max(check.values())
assert max_theory_diff < 1e-12, f'recomputed theory differs from archive by {max_theory_diff}'
# archived simulation statistics (identical reductions to Block 5)
xi2_real = S['xi2']; radii = ns['radii_test']; n_real = xi2_real.shape[0]
ns['sim_xi2_mean'] = {vm: np.nanmean(xi2_real[:, i], 0) for i, vm in enumerate(radii)}
ns['sim_xi2_std'] = {vm: np.nanstd(xi2_real[:, i], 0, ddof=1) for i, vm in enumerate(radii)}
ns['sim_xi2_sem'] = {vm: ns['sim_xi2_std'][vm] / np.sqrt(n_real) for vm in radii}
xi3_stack, sec_stack = S['xi3'], S['sector']
ns['xi3_sim_mean'], ns['xi3_sim_std'] = np.nanmean(xi3_stack, 0), np.nanstd(xi3_stack, 0, ddof=1)
ns['sec_sim_mean'], ns['sec_sim_std'] = np.nanmean(sec_stack, 0), np.nanstd(sec_stack, 0, ddof=1)
ns['xi3_sim_sem'] = ns['xi3_sim_std'] / np.sqrt(n_real); ns['sec_sim_sem'] = ns['sec_sim_std'] / np.sqrt(n_real)
ns['METRICS'] = {}
# Block 8 plotting and metric code, verbatim up to (not including) the archive writes
i = src.index('# %% Block 8:'); j = src.index("np.savez(OUT_DIR / 'scalar_theory.npz'", i)
exec(compile(src[i:j], f'{script.name} [Block 8 plotting]', 'exec'), ns)
arch = json.load(open(ROOT / 'validation/outputs/validation_metrics.json'))
metric_diff = {k: {m: abs(ns['METRICS'][k][m] - arch[k][m]) for m in ns['METRICS'][k] if isinstance(ns['METRICS'][k][m], (int, float)) and m in arch.get(k, {})}
               for k in ['xi2', 'xi2_noCNP', 'xi3', 'xi3_noCNP', 'sector', 'sector_noCNP']}
h = lambda x: hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
record = dict(inputs={k: h(v) for k, v in {**{'theory_' + a: b for a, b in A.items()}, **{'sim_' + a: b for a, b in S.items()}}.items()},
              max_abs_theory_recompute_minus_archive=max_theory_diff,
              metric_abs_diff_vs_archive=metric_diff,
              r0_values={'xi3': {k: float(ns['th_xi3_0'][k]) for k in ns['component_order']},
                         'sector': {k: float(ns['th_sec_0'][k]) for k in ns['component_order']},
                         'xi2': {str(vm): {k: float(ns['th_xi2_0'][vm][k]) for k in ns['component_order']} for vm in radii}},
              labels=dict(ns['component_labels']), figures=sorted(str(p.name) for p in out.glob('validation_scalar_*.png')))
if args.dump:
    Path(args.dump).write_text(json.dumps(record, indent=1))
print('max |theory - archive| =', max_theory_diff)
print('metrics max |diff| =', max(v for d in metric_diff.values() for v in d.values()))
