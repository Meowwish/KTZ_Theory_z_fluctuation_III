"""Read-only checks of the released conditional common-covariance products.

Requires NumPy only. This does not run fits, generate fields, or establish a
calibrated likelihood. Optional physical-prior reintegration requires SciPy.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT/'amusing/release_20261030_pps'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--physical-prior',action='store_true',
                    help='Also reintegrate global prior masses using SciPy')
    args=ap.parse_args()
    count=0
    for line in (RELEASE/'SHA256SUMS.txt').read_text().splitlines():
        digest,name=line.split(None,1)
        path=ROOT/name
        assert path.resolve().is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
        count+=1
    summary=json.loads((RELEASE/'summary.json').read_text())
    index=json.loads((RELEASE/'index.json').read_text())
    if args.physical_prior:
        import sys
        sys.path.insert(0,str(ROOT/'ktzobs'))
        from scripts.physical_prior import inputs_from_metadata,retained_prior_mass
    margins={}
    for field,entry in index['fields'].items():
        for suffix,digest in entry['input_hashes'].items():
            path=ROOT/entry['null_input']/f"{entry['galaxy']}_{suffix}"
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,str(path)
        margins[field]={}
        for ref in ('L2gen','L0gen'):
            cov=np.load(RELEASE/f'covariances/{field}_{ref}.npz',allow_pickle=False)
            assert all(np.isfinite(cov[k]).all() for k in cov.files)
            raw=cov['conditional'];regular=.65*raw+.35*np.diag(np.diag(raw))+1e-4*np.eye(len(raw))
            np.testing.assert_array_equal(raw,cov['raw_covariance'])
            np.testing.assert_allclose(cov['pooled'],raw+cov['between'],rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(cov['noise_block'],raw-cov['cond_no_noise'],rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(cov['cross_block'],raw-cov['cond_crossfree'],rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(regular,cov['covariance'],rtol=1e-12,atol=1e-14)
            np.linalg.cholesky(regular)
            assert np.linalg.eigvalsh(raw).min()>-1e-12
            assert cov['fitted_bins'].sum()==(82 if field=='N1084D16' else 63)
            p=json.loads((RELEASE/f'points/{field}_{ref}.json').read_text())
            w=np.array([q['w'] for q in p['points']])
            np.testing.assert_allclose(w.sum(),1,atol=1e-12)
            np.testing.assert_allclose(w,cov['weights'],atol=1e-12)
            np.testing.assert_allclose(w@cov['mu_points'],cov['mu_pooled'],rtol=1e-12,atol=1e-14)
            if ref=='L0gen':
                assert all(q['beta']==q['eta']==q['betaA']==0 for q in p['points'])
            evidence=json.loads((RELEASE/f'evidence/{field}_{ref}.json').read_text())
            pieces=json.loads((RELEASE/f'pieces/{field}_{ref}.json').read_text())
            bounds=evidence['_meta']['pieces']
            f=np.log([b/a for a,b in bounds])/np.log(bounds[-1][1]/bounds[0][0])
            for model in ('L0','L2'):
                rawz=np.array([part[model]['logZ'] for part in pieces])
                box=float(np.logaddexp.reduce(np.log(f)+rawz))
                e=evidence[model]
                assert abs(box-e['logZ_box'])<1e-10
                assert abs(box-np.log(e['F'])-e['logZ'])<1e-10
                np.testing.assert_allclose(np.exp(np.log(f)+rawz-box),e['weights'],atol=1e-10)
                s=summary['fields'][field][ref][model]
                for key in ('logZ','F','logzerr','chi2_min'):
                    assert abs(e[key]-s[key])<1e-10,(field,ref,model,key)
                if args.physical_prior:
                    morphology=json.loads((ROOT/entry['morphology']).read_text())
                    mass=retained_prior_mass(inputs_from_metadata(morphology,[bounds[0][0],bounds[-1][1]]),model)
                    assert abs(mass-e['F'])<1e-10
            delta=evidence['L2']['logZ']-evidence['L0']['logZ']
            assert abs(delta-summary['fields'][field][ref]['dlnZ_L2_L0'])<1e-10
            margins[field][ref]=delta
    print(json.dumps(dict(manifest_files=count,evidence_margins=margins,
                          physical_prior_reintegrated=args.physical_prior,
                          status='PASS; artifact consistency, not generative calibration'),indent=2))


if __name__=='__main__':
    main()
