"""Fast independent tests; no response libraries, maps or sampler reruns needed.

Run: OPENBLAS_NUM_THREADS=1 python amusing/tests/physical_prior_test.py
"""
import copy
import json
import sys
import types
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ktzobs'))
from scripts.ktz_exact_window import ExactWindowKTZ
from scripts.ktz_exact_window_evidence import NestedModel
from scripts.physical_prior import inputs_from_model, inputs_from_metadata, retained_prior_mass, condition_summary, condition_split_summary


def assert_summary_close(actual, expected, path='$'):
    """Keep structure and discrete values exact; allow round-off in floats."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            assert_summary_close(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (value, reference) in enumerate(zip(actual, expected)):
            assert_summary_close(value, reference, f'{path}[{i}]')
    elif isinstance(expected, (float, np.floating)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=path)
    else:
        assert actual == expected, path


report=json.loads((ROOT/'amusing/metadata/prior_normalization_20261018.json').read_text())
checks=0
# Round-off must pass, while numerical or structural changes must still fail.
assert_summary_close({'weights': [0.5 + 3.6e-15]}, {'weights': [0.5]})
checks+=1
for changed in ({'weights': [0.5 + 1e-9]}, {'weights': [0.5], 'extra': 1}):
    try:
        assert_summary_close(changed, {'weights': [0.5]})
    except AssertionError:
        checks+=1
    else:
        raise AssertionError('Summary comparison accepted a material change')
for label,case in report['cases'].items():
    inp=case['prior_inputs']; result=case['summary']
    for name in ('L0','L1','L1n','L2'):
        row=result[name]; f=retained_prior_mass(inp,name)
        assert abs(np.log(f/row['retained_prior_mass']))<2e-6
        assert abs(sum(row['prior_masses_physical'])-1)<1e-12
        z=np.logaddexp.reduce(np.log(row['prior_masses_physical'])+row['logZ_pieces'])
        assert abs(z-row['logZ'])<1e-12
        assert abs(row['logZ']-(row['logZ_box']-np.log(row['retained_prior_mass'])))<1e-12
        assert_summary_close(condition_summary(row,row['retained_prior_mass']),row)
        checks+=5
    assert abs(result['L1n']['retained_prior_mass']/result['L0']['retained_prior_mass']-5/13)<1e-14
    assert abs(result['L2']['retained_prior_mass']/result['L1']['retained_prior_mass']-5/13)<1e-14
    assert_summary_close(condition_split_summary(result,inp),result)
    checks+=3

# Independent direct count of the actual NestedModel transform and bounds_ok,
# rather than the lower-dimensional quadrature's analytic beta integration.
labels=[k for k in report['cases'] if ('scal_mockcov_fid0_v2_pixbeam' in k and k.endswith('v3.json'))
        or ('NGC1084/ktz/d16_mockcov_v2_pixbeam' in k and k.endswith('v3.json'))]
for label in labels:
    inp=report['cases'][label]['prior_inputs']
    m=types.SimpleNamespace(eta_prior_kind='flat',ell_grid=np.array(inp['ell_bounds']),eps_grid=np.array([.015,.9]),
         src_grid=np.array([.12,2.5]),src_lo=.12,src_hi=2.5,eta_positive_limit=1.2,
         slope_mu=inp['slope_mu'],slope_sd=inp['slope_sd'],slope_bound=inp['slope_bound'],
         sigma_q_max=3.,area_radius=np.array(inp['area_radius']),area_weight=np.array(inp['area_weight']),
         mass_factor=inp['mass_factor'],gamma_area=inp['gamma_area'],metadata={'r_ref_kpc':inp['r_ref']},
         y=np.zeros(1),precision_root_joint=np.eye(1))
    m.derived=types.MethodType(ExactWindowKTZ.derived,m)
    assert_summary_close(inputs_from_model(m),inp)
    for name in ('L0','L1','L1n','L2'):
        nm=NestedModel(m,name+'q',eps_prior='manuscript')
        u=qmc.Sobol(len(nm.active),scramble=True,seed=1810).random_base2(19)
        x=nm.fill(u); valid=nm.bounds_ok(x); empirical=float(valid.mean())
        analytic=retained_prior_mass(inp,name)
        assert abs(empirical-analytic)<7e-4,(label,name,empirical,analytic)
        nm.predict_joint=lambda x:(np.full((len(x),1),2.),None)
        point=u[np.flatnonzero(valid)[0]]
        assert nm.loglike(point)==-2.,'Prior normalisation must not be inserted into the likelihood'
        print(label.split('/')[0],name,'quadrature',round(analytic,9),'direct Sobol',round(empirical,9))
        checks+=2
with tempfile.TemporaryDirectory(prefix='ktz_prior_cli_') as tmp:
    folder=Path(tmp)/'outputs/NGCX/ktz/test'; folder.mkdir(parents=True)
    morph=dict(analysis_rmin_kpc=.3,analysis_rmax_kpc=3.9,r_ref_kpc=2.,
               event_rate_log_slope_kpc_inv=-.78,event_rate_log_slope_error_kpc_inv=.2,
               halpha_sfr_msun_per_yr=.11)
    (folder/'ngcx_morphology.json').write_text(json.dumps(morph))
    pieces=[(.08,.5),(.5,3.2)]; source_bytes={}
    raw={name:dict(logZ_pieces=[],weights=[]) for name in ('L0','L1','L1n','L2')}
    for i,(lo,hi) in enumerate(pieces):
        data={name:dict(logZ=-8+j+i,chi2_min=6-i,n_params=4,n_data=8)
              for j,name in enumerate(raw)}
        stem=folder/f'ngcx_paper_nested_v3_ell{lo:g}-{hi:g}'
        # Append extensions explicitly because the intrinsic bounds contain dots.
        jp=Path(str(stem)+'.json'); jp.write_text(json.dumps(data));source_bytes[jp]=jp.read_bytes()
        np.savez(str(stem)+'.npz',**{name+'_derived':np.ones((4,9)) for name in raw})
        for name in raw:raw[name]['logZ_pieces'].append(data[name]['logZ'])
    f=np.array([np.log(b/a)/np.log(40) for a,b in pieces])
    for name in raw:
        lz=np.array(raw[name]['logZ_pieces']); z=float(np.logaddexp.reduce(np.log(f)+lz))
        raw[name].update(logZ=z,weights=np.exp(np.log(f)+lz-z).tolist())
    raw['_pieces']=pieces
    inputs=inputs_from_metadata(morph,(.08,3.2)); expected=condition_split_summary(raw,inputs)
    command=[sys.executable,str(ROOT/'amusing/scripts/combine_split_evidence.py'),
             'NGCX','test','.08','.5','.5','3.2','v3','--physical-prior']
    subprocess.run(command,cwd=tmp,check=True,capture_output=True,text=True)
    output=folder/'ngcx_paper_nested_split_v3_physical.json'; first=json.loads(output.read_text())
    for name in ('L0','L1','L1n','L2'):
        assert abs(first[name]['logZ']-expected[name]['logZ'])<1e-12
        checks+=1
    # The same combiner must also accept separately normalised piece inputs.
    for lo,hi in pieces:
        stem=folder/f'ngcx_paper_nested_v3_ell{lo:g}-{hi:g}'
        data=json.loads(Path(str(stem)+'.json').read_text())
        pi=inputs_from_metadata(morph,(lo,hi))
        data={name:condition_summary(row,retained_prior_mass(pi,name)) for name,row in data.items()}
        Path(str(stem)+'_physical.json').write_text(json.dumps(data))
        np.savez(str(stem)+'_physical.npz',**{name+'_derived':np.ones((4,9)) for name in data})
    subprocess.run(command,cwd=tmp,check=True,capture_output=True,text=True)
    second=json.loads(output.read_text())
    for name in ('L0','L1','L1n','L2'):
        assert abs(second[name]['logZ']-first[name]['logZ'])<1e-12
        assert_summary_close(second[name]['weights'],first[name]['weights'])
        checks+=2
    assert all(p.read_bytes()==b for p,b in source_bytes.items());checks+=1
print(f'PASS: {checks} assertions; evidence-only correction, split weights, idempotence, direct counts, unmodified likelihood and both CLI input conventions')
