"""Combine split-prior nested runs: ln Z = ln(sum_i f_i Z_i), f_i = prior mass of each ln(ell*) interval.
usage: combine_split_evidence.py GALAXY VARIANT lo1 hi1 [lo2 hi2 ...]"""
import json, sys, numpy as np
g, v = sys.argv[1], sys.argv[2]; lg = g.lower(); rest = sys.argv[3:]; tp = rest[-1] if len(rest) % 2 == 1 else ""; tp, outtag = (tp.split(":")+[""])[:2] if ":" in tp else (tp, ""); edges = list(map(float, rest[:len(rest)-(len(rest)%2)])); pieces = [(edges[i], edges[i+1]) for i in range(0, len(edges), 2)]
pre = f"{lg}_paper_nested_{tp}_ell" if tp else f"{lg}_paper_nested_ell"
NAMES=["ell_star","epsilon","beta_src","eta","kappa","t_star","sigma","A_src","ell_src"]
J = [json.load(open(f'outputs/{g}/ktz/{v}/{pre}{lo:g}-{hi:g}.json')) for lo, hi in pieces]
Z = [np.load(f'outputs/{g}/ktz/{v}/{pre}{lo:g}-{hi:g}.npz', allow_pickle=True) for lo, hi in pieces]
L = np.log(pieces[-1][1]/pieces[0][0]); f = [np.log(hi/lo)/L for lo, hi in pieces]
out = {}
for k in [k for k in J[0] if not k.startswith('_')]:
    lz = np.array([j[k]['logZ'] for j in J]); w = np.log(f)+lz; z = np.logaddexp.reduce(w); pw = np.exp(w-z)
    # posterior-weighted parameter summary across pieces: resample derived samples with piece weights
    der = np.concatenate([Z[i][f'{k}_derived'][np.random.default_rng(0).choice(len(Z[i][f'{k}_derived']), max(1,int(round(4000*pw[i]))))] for i in range(len(pieces))])
    pars = {p: [float(x) for x in np.percentile(der[:, NAMES.index(p)], [16,50,84])] for p in NAMES}
    chi2 = min(j[k]['chi2_min'] for j in J); best = int(np.argmin([j[k]['chi2_min'] for j in J]))
    out[k] = dict(logZ=float(z), logZ_pieces=[float(x) for x in lz], weights=[float(x) for x in pw], chi2_min=float(chi2), chi2_pieces=[j[k]['chi2_min'] for j in J], best_piece=best, params=pars, n_params=J[0][k]['n_params'], n_data=J[0][k]['n_data'])
z0 = out['L0']['logZ']
print(f"== {g} {v}: pieces {pieces}, prior masses {np.round(f,2)}")
for k in out:
    o = out[k]; print(f"  {k:3s} lnZ={o['logZ']:7.1f} dlnZ={o['logZ']-z0:+6.1f} chi2_min={o['chi2_min']:6.1f} pieces lnZ={np.round(o['logZ_pieces'],1)} chi2={np.round(o['chi2_pieces'],1)} w={np.round(o['weights'],2)} | ell*={np.round(o['params']['ell_star'],3)} eta={np.round(o['params']['eta'],3)} ell_src={np.round(o['params']['ell_src'],2)} A={np.round(o['params']['A_src'],2)}")
out['_pieces'] = pieces
json.dump(out, open(f'outputs/{g}/ktz/{v}/{lg}_paper_nested_split' + (f'_{tp}{outtag}' if tp else '') + '.json', 'w'), indent=1)
