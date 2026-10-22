"""Build response caches from prepared stage files; does not run a fit.

This is an expensive geometry operation. Prints the plan unless --execute is set.
"""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ktzobs'));sys.path.insert(0,str(ROOT/'amusing/scripts'))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path);ap.add_argument('galaxy',choices=['NGC2466','NGC1084'])
    ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    d=a.directory;g=a.galaxy.lower()
    print('Build 7-node base window and 65-node Hermite stochastic responses in',d)
    if not a.execute:sys.exit(0)
    from scripts import ktz_exact_window as kew
    from response_library import build_library,chebyshev_lobatto_nodes
    pm=dict(np.load(d/f'{g}_pixel_map.npz',allow_pickle=False));m=json.loads((d/f'{g}_morphology.json').read_text())
    if not m.get('halpha_flux_area_factor',0)>0:raise ValueError('Use corrected flux-area morphology')
    diag=np.genfromtxt(d/f'{g}_diagnostics.csv',delimiter=',',names=True)
    dr=diag['r_kpc'][1]-diag['r_kpc'][0];edges=np.r_[diag['r_kpc'][0]-.5*dr,diag['r_kpc']+.5*dr]
    aw=.4 if float(pm['pixel_size_kpc'])>=.1 else .2
    labels=np.zeros(pm['valid_mask'].shape,int)
    base=d/f'{g}_window_response_all.npz';lib=d/f'{g}_stochastic_response_H65.npz'
    if not base.exists():
        x=kew.build_window_response(pm,labels,m['r_ref_kpc'],selection='all',rate_log_slope_grid=np.linspace(-.9,.9,7),edges=edges,annulus_width=aw)
        np.savez_compressed(str(base)+'.partial.npz',**x);os.replace(str(base)+'.partial.npz',base)
    if not lib.exists():
        import hashlib
        x=build_library(pm,labels,m['r_ref_kpc'],chebyshev_lobatto_nodes(65,.9),edges,aw,kr_mode='hermite')
        x.update(annulus_width=aw,edges=edges,pixel_map_md5=hashlib.md5((d/f'{g}_pixel_map.npz').read_bytes()).hexdigest(),r_ref=m['r_ref_kpc'],bound=.9,nodes=65,subset='None',kr_mode='hermite')
        np.savez(str(lib)+'.partial.npz',**x);os.replace(str(lib)+'.partial.npz',lib)
    print(base,lib)
