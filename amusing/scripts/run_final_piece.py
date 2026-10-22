"""Explicit, guarded entrypoint for one current-manuscript fit interval.

Prints the command by default. Only --execute starts a potentially long fit.
Prepare the stage maps and H65 response library first; see release_20261022/README.md.
"""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CASES={
 'S':('NGC2466','scal_mockcov_fid0_v2_pixbeam_unif65e_fa',1,.45,4.),
 'D16':('NGC2466','d16_mockcov_fid0_v2_pixbeam_unif65e_fa',9,.45,4.),
 'S_r5':('NGC2466','scal_mockcov_fid0_v2_pixbeam_unif65e_fa',1,.45,4.),
 'N1084D16':('NGC1084','d16_mockcov_v2_pixbeam_unif65e_fa',-87,.12,2.5),
 'N1084t05':('NGC1084','d16_mockcov_t05_v2_pixbeam_unif65e_fa',-87,.12,2.5),
}

def command(case,lo,hi,variant=None):
    galaxy,default,phase,rmin,src=CASES[case]
    if not .08<=lo<hi<=3.2:
        raise ValueError('Intrinsic ell_* interval must lie in [0.08, 3.2] kpc')
    tag=f'_v3{"r5" if case=="S_r5" else ""}_ell{lo:g}-{hi:g}'
    cmd=[sys.executable,str(ROOT/'scripts/paper_single_pointing.py'),
         '--name',galaxy,'--variant',variant or default,'--m1',str(phase),
         '--rmin-fit',str(rmin),'--src-max',str(src),'--ell-min',str(lo),'--ell-max',str(hi),
         '--tag',tag,'--cm-nodes','65','--unified-response','--support','intrinsic',
         '--eps-prior','manuscript','--evidence-prior','physical','--nlive','400','--seed','2',
         '--figname',f'refit_{case}_ell{lo:g}-{hi:g}']
    if case=='S_r5':cmd+=['--rmax-fit','5']
    return cmd

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',required=True,choices=CASES)
    ap.add_argument('--ell-min',type=float,required=True)
    ap.add_argument('--ell-max',type=float,required=True)
    ap.add_argument('--variant',help='Prepared null or EB stage directory name under amusing/outputs/GALAXY/ktz/')
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args();cmd=command(a.case,a.ell_min,a.ell_max,a.variant)
    print(shlex.join(cmd),flush=True)
    if a.execute:
        g,v,*_=CASES[a.case];v=a.variant or v
        d=ROOT/'outputs'/g/'ktz'/v
        morph=json.loads((d/f'{g.lower()}_morphology.json').read_text())
        if not morph.get('halpha_flux_area_factor',0)>0:
            raise SystemExit('Refusing legacy unscaled H-alpha inputs: run apply_sfr_area_fix.py or the corrected stage first.')
        expected=math.sqrt(max(morph['target_fwhm_kpc'],morph['psf_kpc'])**2
                           +8*math.log(2)*morph['pixel_kpc']**2/12)
        if not math.isclose(morph.get('effective_fwhm_kpc',0),expected,rel_tol=2e-5):
            raise SystemExit('Use beam+cell effective metadata: sigma_eff^2=sigma_beam^2+pixel^2/12, once only.')
        if not (d/f'{g.lower()}_stochastic_response_H65.npz').exists():
            raise SystemExit('Build the H65 Hermite response library first; see release documentation.')
        subprocess.run(cmd,check=True)
