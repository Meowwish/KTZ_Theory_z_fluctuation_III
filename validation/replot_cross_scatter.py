"""Redraw Fig. 5 from archived arrays and original yields, without simulation.

Run from the repository root:
    python validation/replot_cross_scatter.py --outdir /tmp/ktz-figures
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('--script',default='KTZ_validation.py')
ap.add_argument('--outdir',default='figs')
ap.add_argument('--dump')
args=ap.parse_args()
ROOT=Path(__file__).resolve().parents[1]
script=Path(args.script)
if not script.is_absolute():
    script=ROOT/script
src=script.read_text()
ns=dict(ROOT=ROOT,np=np,sys=sys,plt=plt)
# Only read the deterministic KT25 yields; never enter any simulation block.
start=src.index('import astropy.units as u',src.index('# %% Block 9:'))
end=src.index('kt25_delay =',start)
exec(compile(src[start:end],args.script+' [yield inputs only]','exec'),ns)
a=dict(np.load(ROOT/'validation/outputs/cross_results.npz'))
idx=[ns['kt25_elements'].index(e) for e in a['elements']]
ns.update(Xi_sim=a['Xi_sim'],Xi_sim_err=a['Xi_sim_err'],th_sub=a['Xi_KTZ'],th_sub_n=a['Xi_KTZ_noCNP'],cmean=ns['kt25_mean_yield'][idx])
out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
ns['figure_path']=lambda name:out/f'validation_{name}.png'
start=src.index('iu = np.triu_indices(16, 1)',src.index('# %% Block 12:'))
end=src.index('\ndd = s_ - t_',start)
exec(compile(src[start:end],args.script+' [scatter plotting only]','exec'),ns)
def h(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
ax=ns['ax']
record={'inputs':{k:h(v) for k,v in a.items()},'groups':ns['grp'].tolist(), 'lines':[h(l.get_xydata()) for l in ax.lines], 'xlabel':ax.get_xlabel(),'ylabel':ax.get_ylabel(),'limits':[list(ax.get_xlim()),list(ax.get_ylim())]}
if args.dump:
    Path(args.dump).write_text(json.dumps(record,indent=2))
print(args.outdir,record['xlabel'],record['ylabel'])
