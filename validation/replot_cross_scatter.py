"""Redraw Fig. 5 (and optionally Fig. 4) from archived arrays, without simulation.

Run from the repository root:
    python validation/replot_cross_scatter.py --matrices --outdir /tmp/ktz-figures
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
ap.add_argument('--matrices', action='store_true', help='Also redraw the Fig. 4 matrices')
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
ns.update(Xi_sim=a['Xi_sim'],Xi_sim_err=a['Xi_sim_err'],th_sub=a['Xi_KTZ'],th_sub_n=a['Xi_KTZ_noCNP'],kt_sub=a['Xi_KT25'],cross_elements=a['elements'].tolist(),cmean=ns['kt25_mean_yield'][idx])
out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
ns['figure_path']=lambda name:out/f'validation_{name}.png'
def h(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
matrix_panels=[]
if args.matrices:
    start=src.index('diff = th_sub - kt_sub;',src.index('# %% Block 12:'))
    end=src.index('\niu = np.triu_indices(16, 1)',start)
    exec(compile(src[start:end],args.script+' [matrix plotting only]','exec'),ns)
    matrix_panels=[{'title':ax.get_title(),'data':h(ax.images[0].get_array()),'clim':list(ax.images[0].get_clim())} for ax in ns['axes']]
start=src.index('iu = np.triu_indices(16, 1)',src.index('# %% Block 12:'))
end=src.index('\ndd = s_ - t_',start)
exec(compile(src[start:end],args.script+' [scatter plotting only]','exec'),ns)
ax=ns['ax']
record={'inputs':{k:h(v) for k,v in a.items()},'groups':ns['grp'].tolist(), 'lines':[h(l.get_xydata()) for l in ax.lines], 'xlabel':ax.get_xlabel(),'ylabel':ax.get_ylabel(),'limits':[list(ax.get_xlim()),list(ax.get_ylim())]}
record['matrix_panels']=matrix_panels
record['scatter']=[{'label':c.get_label(),'offsets':h(c.get_offsets()),'facecolors':h(c.get_facecolors()),'sizes':h(c.get_sizes())} for c in ax.collections if hasattr(c,'get_sizes')]
record['errorbars']=[{'segments':h(np.asarray(c.get_segments())),'colors':h(c.get_colors())} for c in ax.collections if hasattr(c,'get_segments')]
record['legend']=ax.get_legend_handles_labels()[1]
if args.dump:
    Path(args.dump).write_text(json.dumps(record,indent=2))
print(args.outdir,record['xlabel'],record['ylabel'])
