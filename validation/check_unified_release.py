"""Verify the 2026-11-04 public validation release without running a fit."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'validation/release_20261104'
for line in (RELEASE / 'SHA256SUMS').read_text().splitlines():
    digest, name = line.split('  ', 1)
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

out = ROOT / 'validation/outputs_rework_20261104'
s = np.load(out / 'scalar_simulation_statistics_unified.npz')
t = np.load(out / 'scalar_theory_unified.npz')
mask = t['r_centers'] <= 5
scalar = float(np.sqrt(np.mean((s['xi'].mean(0)[0, mask]-t['total'][0, mask])**2)))
c = np.load(out / 'cross_results_unified.npz')
iu = np.triu_indices(len(c['elements']), 1)
cross = float(np.sqrt(np.mean((c['Xi_sim'][iu]-c['Xi_KTZ'][iu])**2)))
assert abs(scalar-0.004563522058184) < 1e-12
assert abs(cross-0.014384640502545) < 1e-12
config = json.loads(str(np.load(out / 'cross_same_parent_diag_source_v2_nt128_nw192.npz')['config']))
assert config['version'] == 'same_parent_diag_source/v2-lnS-2026-11-04'
assert config['n_t'] == 128 and config['n_lnw'] == 192
assert (ROOT / 'amusing/source_integral/si_plot_fig78.py').read_text().count(r'\xi_b') == 2
print(json.dumps({'hashes': 'passed', 'scalar_rmse': scalar,
                  'cross_rmse': cross, 'diagonal_config': config}, indent=2))
