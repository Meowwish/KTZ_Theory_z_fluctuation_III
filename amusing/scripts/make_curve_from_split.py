"""curves.npz (r, comp_median, ktz_median) for --mock-curve from split-prior runs: takes the model's dominant piece.
usage: make_curve_from_split.py GALAXY VARIANT TAGPREFIX MODEL OUT"""
import sys, json, numpy as np
g, v, tp, model, dst = sys.argv[1:6]; lg = g.lower()
S = json.load(open(f'outputs/{g}/ktz/{v}/{lg}_paper_nested_split_{tp}.json')) if tp else json.load(open(f'outputs/{g}/ktz/{v}/{lg}_paper_nested_split.json'))
pieces = S['_pieces']; i = int(np.argmax(S[model]['weights'])); lo, hi = pieces[i]
z = np.load(f'outputs/{g}/ktz/{v}/{lg}_paper_nested_{tp}_ell{lo:g}-{hi:g}.npz' if tp else f'outputs/{g}/ktz/{v}/{lg}_paper_nested_ell{lo:g}-{hi:g}.npz', allow_pickle=True)
np.savez(dst, r=z["r"], comp_median=z[f"{model}_comp_p50"], ktz_median=z[f"{model}_pred_p50"]); print("wrote", dst, "from", model, "piece", (lo, hi))
