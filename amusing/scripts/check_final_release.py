"""Check current paper products without running a sampler or response builder.

python amusing/scripts/check_final_release.py [--manuscript current.tex]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REL=ROOT/'amusing/release_20261022'
sys.path.insert(0,str(ROOT/'ktzobs'))
from scripts.physical_prior import inputs_from_metadata,retained_prior_mass
from scripts.ktz_exact_window_evidence import eps_from_u,u_from_eps

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def check(manuscript=None):
    report={}
    manifest=REL/'SHA256SUMS'
    if manifest.exists():
        rows=manifest.read_text().splitlines()
        for row in rows:
            digest,name=row.split(maxsplit=1)
            assert sha(ROOT/name.strip())==digest,name
        report['manifest_files']=len(rows)
    results=read(REL/'metadata/results_null_fa.json')|read(REL/'metadata/results_eb_fa.json')
    cases=read(REL/'provenance/cases_null.json')|read(REL/'provenance/cases_eb.json')
    assert results.keys()==cases.keys()
    archives=0
    for name,c in cases.items():
        d=REL/'outputs'/c['galaxy']/'ktz'/c['variant'];g=c['galaxy'].lower();pieces=c['pieces']
        assert pieces[0][0]==.08 and pieces[-1][1]==3.2
        assert all(a[1]==b[0] for a,b in zip(pieces[:-1],pieces[1:]))
        f=np.log([b/a for a,b in pieces])/np.log(40)
        assert abs(f.sum()-1)<1e-14
        J=[read(d/f'{g}_paper_nested_{c["tagprefix"]}_ell{a:g}-{b:g}.json') for a,b in pieces]
        for model in ('L0','L1','L1n','L2'):
            r=results[name][model];lz=np.array([j[model]['logZ'] for j in J])
            box=float(np.logaddexp.reduce(np.log(f)+lz))
            assert abs(box-r['logZ_box'])<1e-12
            assert abs(box-np.log(r['F'])-r['logZ'])<1e-12
            assert np.max(abs(np.exp(np.log(f)+lz-box)-r['weights']))<1e-12
            assert abs(r['dlnZ']-(r['logZ']-results[name]['L0']['logZ']))<1e-12
            assert r['chi2_min']==min(j[model]['chi2_min'] for j in J)
        for (lo,hi),j in zip(pieces,J):
            z=np.load(d/f'{g}_paper_nested_{c["tagprefix"]}_ell{lo:g}-{hi:g}.npz',allow_pickle=False)
            for model in ('L0','L1','L1n','L2'):
                assert j[model]['eps_prior']=='manuscript' and j[model]['support']=='intrinsic'
                x=z[model+'_samples'];der=z[model+'_derived'];assert der.shape==(4000,9)
                assert np.isfinite(der).all()
                assert np.all((der[:,0]>=lo-1e-12)&(der[:,0]<=hi+1e-12))
                assert np.all((der[:,1]>=.015-1e-12)&(der[:,1]<=.9+1e-12))
                assert np.all((der[:,6]>=.015-1e-12)&(der[:,6]<=.25+1e-12))
                assert np.allclose(der[:,0],np.exp(x[:,0]),atol=1e-14)
                if model in ('L1','L2'):
                    assert np.all((der[:,4]>=.05-1e-12)&(der[:,4]<=1.5+1e-12))
                    assert np.all((der[:,5]>=.05-1e-12)&(der[:,5]<=3+1e-12))
                if model in ('L1n','L2'):assert np.all((der[:,7]>=.01-1e-12)&(der[:,7]<=1e3+1e-10))
                archives+=1
    report.update(cases=len(cases),posterior_archives=archives)
    table=read(ROOT/'amusing/metadata/table3_numbers.json')
    for tag,n,e in [('S_final','S_null','S_EB'),('D_final','D16_null','D16_EB'),('N_final','N1084_D16_null','N1084_D16_EB')]:
        for model in ('L0','L1','L1n','L2'):
            assert table[tag][model]==[results[n][model]['chi2_min'],results[n][model]['dlnZ'],results[e][model]['dlnZ']]
    report['current_table3']=True
    stage=REL/'stage_inputs'
    mapping=read(stage/'case_inputs.json')
    assert not mapping['missing']
    for c in mapping['cases'].values():
        for suffix,name in c['inputs'].items():
            assert sha(stage/name)==c['input_sha256'][suffix]
    for fn in ('baseline_null.csv','baseline_eb.csv'):
        for row in csv.DictReader((REL/'prior_sensitivity'/fn).open()):
            for suffix,column in (('morphology.json','morphology_sha256'),('fit_covariance.npz','fit_covariance_sha256')):
                assert sha(stage/'inputs'/row['data_dir']/f"{row['galaxy']}_{suffix}")==row[column]
    pairs={}
    for label,key,galaxy in [('NGC2466_S','S_pixbeam_fa|v3','NGC2466'),('NGC2466_D16','D16_pixbeam_fa|v3','NGC2466'),('NGC1084_D16','N1084D16_pixbeam_fa|v3','NGC1084')]:
        c=mapping['cases'][key]
        assert sha(REL/f'metadata/morphology/{label}.json')==c['input_sha256']['morphology.json']
        pm=np.load(stage/c['inputs']['pixel_map.npz'],allow_pickle=False)
        n=int(pm['valid_mask'].sum());pairs[label]=n*(n-1)//2
        if label!='NGC2466_D16':
            for suffix,name in c['inputs'].items():
                assert sha(REL/'figure_inputs'/galaxy/f'{galaxy.lower()}_{suffix}')==sha(stage/name)
    report.update(stage_cases=len(mapping['cases']),unordered_pairs=pairs)
    for label,c in [('NGC2466_S','S_null'),('NGC2466_D16','D16_null'),('NGC1084_D16','N1084_D16_null')]:
        morph=read(REL/f'metadata/morphology/{label}.json')
        factor=morph['pixel_kpc']**2*np.cos(np.deg2rad(morph['inclination_deg']))/(.2*morph['distance_mpc']*1000*np.pi/180/3600)**2
        assert abs(factor/morph['halpha_flux_area_factor']-1)<1e-12
        assert abs(morph['halpha_sfr_msun_per_yr']/morph['halpha_sfr_unscaled_pre20261021_msun_per_yr']/factor-1)<1e-10
        p=inputs_from_metadata(morph,(.08,3.2))
        for model in ('L0','L1','L1n','L2'):
            assert abs(retained_prior_mass(p,model)/results[c][model]['F']-1)<1e-10
    neff={}
    for name in ('S','D16','N1084D16'):
        a=np.load(REL/f'covariance/{name}_ppe_fa.npz',allow_pickle=False)
        raw=a['mock']+a['measurement'];cov=.65*raw+.35*np.diag(np.diag(raw))+.01**2*np.eye(len(raw))
        assert np.max(abs(cov-a['covariance']))<1e-15
        assert np.linalg.eigvalsh(cov)[0]>0
        key={'S':'S_ppe_fa|v3','D16':'D16_ppe_fa|v3','N1084D16':'N1084D16_ppe_fa|v3'}[name]
        current=np.load(stage/mapping['cases'][key]['inputs']['fit_covariance.npz'],allow_pickle=False)
        assert np.array_equal(cov,current['covariance'])
        rho=cov/np.sqrt(np.outer(cov.diagonal(),cov.diagonal()))
        neff[name]=float(len(cov)**2/np.sum(rho**2))
    report['EB_neff_full']=neff
    u=np.linspace(0,1,100001)
    assert np.max(abs(u_from_eps(eps_from_u(u))-u))<1e-14
    state=read(REL/'metadata/manuscript_state.json')
    for name,digest in state['figures'].items():assert sha(ROOT/'figs'/name)==digest
    for name,digest in state['numerical_validation'].items():assert sha(ROOT/name)==digest
    if manuscript:
        assert sha(manuscript)==state['manuscript_sha256']
        tab=manuscript.read_text().split(r'\label{tab:single_pointing_evidence}')[1].split(r'\end{tabular}')[0]
        for model in ('L0','L1','L1n','L2'):
            row=next(x for x in tab.splitlines() if x.startswith(model+' &'))
            actual=[float(x.strip().strip('$').strip('+')) for x in row.rstrip('\\').split('&')[2:]]
            expected=[round(x,1) for tag in ('S_final','D_final','N_final') for x in table[tag][model]]
            assert actual==expected
        report['manuscript_hash_and_table']=True
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manuscript',type=Path)
    a=ap.parse_args();print(json.dumps(check(a.manuscript),indent=2));print('PASS: no fit or response calculation performed')
