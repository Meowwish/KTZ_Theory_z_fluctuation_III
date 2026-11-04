"""Analytic-limit tests of the multi-element same-event diagonal integral (2026-11-04 rework).
run: python -m validation.test_cross_diag OUT.json
(a) uniform rate, every channel pair actually used: Sigma = const -> Y ln[(S1+beta)/(S0+beta)];
(b) non-constant spatial factor with an analytic answer: equal delays, Sigma(w) = w ->
    Y [ (S1-S0)/4 + Delta^2 sigma_v^2 ln(S1/S0) ];
both for the new ln-variance integration and, for comparison, the 11-03 linear-age 24-node rule."""
import sys, json
import numpy as np
from pathlib import Path
from validation import ktz_unified_cross as ucx

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / 'KTZ_validation.py').read_text()
scope = dict(ROOT=ROOT, np=np, sys=sys)
exec(compile(src.split('# %% Block 9:', 1)[1].split('\n', 1)[1].split('# %% Block 10:', 1)[0], 'Block9', 'exec'), scope)
EL = ['C', 'N', 'O', 'Mg', 'Si', 'S', 'Ca', 'V', 'Mn', 'Fe', 'Ni', 'Sr', 'Y', 'Zr', 'Ba', 'Ce']
idx = [scope['kt25_elements'].index(e) for e in EL]
k, sv, tc = scope['kt25_kappa'], scope['kt25_sigma_v'], scope['kt25_t_corr']
D, Y2 = scope['kt25_delay'], scope['kt25_second_yield_normalized']
V0 = scope['kt25_sigma_inj']**2 + (0.1 / (2 * np.sqrt(2 * np.log(2))))**2
lnw = np.linspace(np.log(4e-4), np.log(60.0), 192)
rep = {}
ea = {'new_nt64': 0.0, 'new_nt128': 0.0, 'old_ns24': 0.0}
for p in range(16):
    for q in range(p, 16):
        i, j = idx[p], idx[q]
        for a in range(3):
            for b in range(3):
                y = Y2[i, a, j, b]; s0 = max(D[i, a], D[j, b])
                if y == 0 or s0 >= tc:
                    continue
                be = (sv * (D[j, b] - D[i, a]))**2
                S0 = V0[a] + V0[b] + 2 * k * (2 * s0 - D[i, a] - D[j, b]); S1 = V0[a] + V0[b] + 2 * k * (2 * tc - D[i, a] - D[j, b])
                ex = y * np.log((S1 + be) / (S0 + be))
                args = (V0[a], V0[b], D[i, a], D[j, b], k, sv, tc, lnw, np.ones(lnw.size))
                ea['new_nt64'] = max(ea['new_nt64'], abs(ucx.same_parent_diag_source(y, *args, n_t=64) / ex - 1))
                ea['new_nt128'] = max(ea['new_nt128'], abs(ucx.same_parent_diag_source(y, *args, n_t=128) / ex - 1))
                ea['old_ns24'] = max(ea['old_ns24'], abs(ucx.same_parent_diag_exact(None, None, None, None, y, V0[a], V0[b], D[i, a], D[j, b], None,
                                                                                   k, sv, tc, lnw, sumtab=np.ones(lnw.size), n_s=24) / ex - 1))
rep['uniform_rate_all_channel_pairs_max_rel_err'] = ea
# (b) Sigma(w) = w, equal delays (both channels of the same element and delay), narrowest AGB+beam kernel and CCSN
eb = {}
for name, c, dl in (('AGB+beam', 1, D[idx[EL.index('Ba')], 1]), ('CCSN+beam', 0, D[idx[EL.index('O')], 0]), ('TNSN+beam', 2, D[idx[EL.index('Fe')], 2])):
    if dl >= tc:
        continue
    S0 = 2 * V0[c]; S1 = 2 * V0[c] + 4 * k * (tc - dl)
    ex = (S1 - S0) / 4 + dl**2 * sv**2 * np.log(S1 / S0)
    args = (V0[c], V0[c], dl, dl, k, sv, tc, lnw, np.exp(lnw))
    eb[name] = dict(new_nt128=abs(ucx.same_parent_diag_source(1.0, *args, n_t=128) / ex - 1),
                    old_ns24=abs(ucx.same_parent_diag_exact(None, None, None, None, 1.0, V0[c], V0[c], dl, dl, None, k, sv, tc, lnw,
                                                            sumtab=np.exp(lnw), n_s=24) / ex - 1))
rep['linear_spatial_factor_equal_delay_rel_err'] = eb
rep['passed'] = bool(ea['new_nt128'] < 1e-10 and all(v['new_nt128'] < 1e-5 for v in eb.values()))
print(json.dumps(rep, indent=1))
if len(sys.argv) > 1:
    json.dump(rep, open(sys.argv[1], 'w'), indent=1)
