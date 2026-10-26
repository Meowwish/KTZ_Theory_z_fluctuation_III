"""Read-only verification of the 26 October source-integral release.

Usage: python amusing/source_integral/check_release.py [--manuscript PATH]
No response tables, model fits, or raw data are required.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT/'amusing/release_20261026'
sys.path.insert(0, str(ROOT/'ktzobs'))
from scripts.physical_prior import inputs_from_metadata, retained_prior_mass

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript', type=Path)
    args = parser.parse_args()
    evs = json.loads((REL/'metadata/evidences.json').read_text())
    for key, ev in evs.items():
        d = REL/'outputs'/key
        case = json.loads((d/f'case_{key}.json').read_text())[key]
        pcs = np.array(case['pieces'])
        assert pcs[0, 0] == .08 and pcs[-1, 1] == 3.2
        assert np.array_equal(pcs[:-1, 1], pcs[1:, 0])
        f = np.log(pcs[:, 1]/pcs[:, 0])/np.log(40)
        morph = json.loads((ROOT/case['morphology']).read_text())
        mass_inputs = inputs_from_metadata(morph, (.08, 3.2))
        g = case['galaxy'].lower()
        for model in ('L0', 'L1', 'L1n', 'L2'):
            raw = []
            for a, b in pcs:
                stem = f"{g}_paper_nested_{case['tagprefix']}_ell{a:g}-{b:g}"
                item = json.loads((d/(stem+'.json')).read_text())[model]
                assert item['source_completion'] == 'C1'
                assert item['eps_prior'] == 'manuscript' and item['support'] == 'intrinsic'
                with np.load(d/(stem+'.npz'), allow_pickle=True) as z:
                    assert z[model+'_samples'].shape == (4000, 8)
                    assert np.isfinite(z[model+'_pred_p50']).all()
                raw.append(item['logZ'])
            box = float(np.logaddexp.reduce(np.log(f)+raw))
            F = retained_prior_mass(mass_inputs, model)
            assert abs(box-np.log(F)-ev[model]['logZ']) < 1e-10
            np.testing.assert_allclose(np.exp(np.log(f)+raw-box), ev[model]['weights'], atol=1e-12)
        print(key, 'full prior and posterior records PASS')
    table = json.loads((ROOT/'amusing/metadata/table3_numbers.json').read_text())
    assert table == json.loads((REL/'metadata/table3_numbers.json').read_text())
    for name, key, eb in [('S_final','S','S_EB2'), ('D_final','D16','D16_EB2'), ('N_final','N1084D16','N1084D16_EB2')]:
        for model in ('L0','L1','L1n','L2'):
            np.testing.assert_allclose(table[name][model], [evs[key][model]['chi2_min'], evs[key][model]['dlnZ'], evs[eb][model]['dlnZ']], atol=1e-12)
    for gal, key in [('NGC2466','S'), ('NGC1084','N1084D16')]:
        d = REL/'figure_inputs'/gal
        ev = evs[key]
        pcs = json.loads((d/f'case_{key}.json').read_text())[key]['pieces']
        dump = json.loads((REL/'figures'/f'{gal}_dump.json').read_text())
        for model in ('L0','L1','L1n','L2'):
            a,b = pcs[int(np.argmax(ev[model]['weights']))]
            assert dump['dominant_intervals'][model]['interval'] == [a,b]
            with np.load(d/f'{gal.lower()}_paper_nested_v3_ell{a:g}-{b:g}.npz') as z:
                digest = hashlib.sha256(np.ascontiguousarray(z[model+'_pred_p50'], dtype=float).tobytes()).hexdigest()[:16]
            assert digest == dump['inputs_used'][model+'_pred_p50']
    state = json.loads((REL/'metadata/manuscript_state.json').read_text())
    for name, digest in state['figures'].items():
        assert hashlib.sha256((ROOT/'figs'/name).read_bytes()).hexdigest() == digest
    if args.manuscript:
        assert hashlib.sha256(args.manuscript.read_bytes()).hexdigest() == state['manuscript_sha256']
        text = args.manuscript.read_text().split('\\label{tab:single_pointing_evidence}',1)[1].split('\\end{tabular}',1)[0]
        for model in ('L0','L1','L1n','L2'):
            row = next(x for x in text.splitlines() if x.startswith(model+' &'))
            actual = [float(x.replace('$','')) for x in row.rstrip('\\').split('&')[2:]]
            expected = [float(f'{x:.1f}') for name in ('S_final','D_final','N_final') for x in table[name][model]]
            assert actual == expected, (model, actual, expected)
    manifest = REL/'SHA256SUMS.txt'
    if manifest.exists():
        for row in manifest.read_text().splitlines():
            digest, name = row.split(None,1)
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    print('Table 3, per-model figure curves, manuscript state and manifest PASS')

if __name__ == '__main__':
    main()
