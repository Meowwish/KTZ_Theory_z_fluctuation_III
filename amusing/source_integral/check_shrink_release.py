"""Check the fixed-null shrinkage sensitivity archive without response tables.

Recombine all 40 nested-sampling runs, including physical-prior normalization,
and verify the six fixed covariance matrices. This does not rerun a likelihood
or certify the generative error model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ktzobs'))
from scripts.physical_prior import inputs_from_metadata, retained_prior_mass


def check(release):
    for line in (release / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = release / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Checksum mismatch: {name}')
    summary = json.loads((release / 'summary.json').read_text())
    seen, evidence = set(), []
    for field in ('S', 'D16', 'N1084D16'):
        galaxy = 'ngc1084' if field.startswith('N1084') else 'ngc2466'
        for shrink in (.2, .5):
            variant = field + f'_sh{round(shrink*100):03d}'
            inp = release / 'inputs' / variant
            with np.load(inp / (galaxy + '_fit_covariance.npz'), allow_pickle=False) as data:
                raw, cov = data['raw_covariance'], data['covariance']
            expect = (1-shrink)*raw + shrink*np.diag(np.diag(raw)) + 1e-4*np.eye(len(raw))
            np.testing.assert_allclose(cov, expect, rtol=1e-13, atol=1e-16)
            assert np.linalg.eigvalsh(cov).min() > 0
            meta = json.loads((inp / (galaxy + '_morphology.json')).read_text())
            prior_inputs = inputs_from_metadata(meta, (.08, 3.2))
            ep = next((release / 'outputs').rglob('evidence_' + variant + '.json'))
            combined = json.loads(ep.read_text())[variant]
            for model in ('L0', 'L2'):
                jobs = [json.loads(p.read_text()) for p in (release / 'jobs' / variant).glob('*__' + model + 'q.json')]
                jobs.sort(key=lambda j: j['cfg']['ell_min'])
                intervals = [(j['cfg']['ell_min'], j['cfg']['ell_max']) for j in jobs]
                assert intervals[0][0] == .08 and intervals[-1][1] == 3.2
                assert all(a[1] == b[0] for a,b in zip(intervals, intervals[1:]))
                lz, le, chi = [], [], []
                for job in jobs:
                    assert job['seed'] == 2 and job['nlive'] == 400 and job['eps_prior'] == 'manuscript'
                    path = release / 'runs' / (Path(job['out']).name + '.npz')
                    with np.load(path, allow_pickle=False) as run:
                        assert json.loads(str(run['job'])) == job
                        assert run['samples'].shape[1] == (4 if model == 'L0' else 8)
                        lz.append(float(run['logz'][-1])); le.append(float(run['logzerr'][-1]))
                        chi.append(float(-2*run['logl'].max()))
                    seen.add(path.name)
                mass = np.array([np.log(b/a)/np.log(3.2/.08) for a,b in intervals])
                box = float(np.logaddexp.reduce(np.log(mass)+lz))
                weights = np.exp(np.log(mass)+lz-box)
                retained = retained_prior_mass(prior_inputs, model)
                lnz = float(box-np.log(retained))
                result = combined[model]
                np.testing.assert_allclose(lz, result['logZ_pieces'], rtol=0, atol=1e-10)
                np.testing.assert_allclose(weights, result['weights'], rtol=0, atol=1e-10)
                np.testing.assert_allclose([lnz, retained, min(chi), np.sqrt(np.sum((weights*le)**2))],
                    [result['logZ'], result['F'], result['chi2_min'], result['logzerr']], rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(lnz, summary[field][f's{shrink:.2f}'][model]['lnZ'], rtol=0, atol=1e-10)
                evidence.append(dict(field=field, shrinkage=shrink, model=model, logZ=lnz))
    assert len(seen) == 40
    return dict(pass_checks=True, runs_recombined=40, covariance_matrices=6, evidence=evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=ROOT/'amusing/release_20261118_shrink')
    args = parser.parse_args()
    print(json.dumps(check(args.release.resolve()), indent=2))
