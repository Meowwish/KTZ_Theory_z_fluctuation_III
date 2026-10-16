#!/usr/bin/env python3
"""Stochastic-term window-response library on arbitrary rate-log-slope nodes.

Algebraically identical to ktz_exact_window.build_window_response(selection="all") for the three
stochastic components (0: C_2p,0 [rate], 1: C_2p,N [rate*g_mid], 2: C_NP [rate^2]) but evaluates all
requested slopes in one geometry pass (sparse histogram x factor matrix) and never stores the full
(npix, n_ann, n_kernel_radius) row-mean array, so many slope nodes fit in memory.

Usage: response_library.py PIXEL_MAP MORPHOLOGY_JSON OUT.npz --nodes N [--subset i0:i1] [--slopes a,b,...]
Output: response (n_slope, 3, 2, nb, nk), kernel_radius, rate_log_slope_grid, ordered_pair_counts, provenance.
"""
import argparse, json, os, sys, time, hashlib
import numpy as np
from scipy import sparse
from scipy.ndimage import map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))


def chebyshev_lobatto_nodes(n, bound):
    k = np.arange(n)
    return np.sort(-bound*np.cos(np.pi*k/(n-1)))


def annular_project(field, valid, labels):
    out = np.zeros_like(field, dtype=float)
    for L in np.unique(labels[valid]):
        m = valid & (labels == L)
        out[m] = field[m]-field[m].mean()
    return out


def build_library(pm, labels, r_ref, slopes, edges, annulus_width, block=16, log=print, kr_mode="legacy"):
    valid = pm["valid_mask"].astype(bool); X, Y = pm["x_grid_kpc"], pm["y_grid_kpc"]
    modulation = pm["arm_source_modulation"]; pgrid = np.asarray(labels)
    iy, ix = np.nonzero(valid); x, y = X[valid], Y[valid]; pnt = pgrid[valid]
    radius = np.hypot(X, Y)
    ann_grid = np.floor(radius/annulus_width).astype(int)
    lab_vals, ann = np.unique(ann_grid[valid], return_inverse=True)
    npix, na, nb = len(x), len(lab_vals), len(edges)-1
    ann_counts = np.bincount(ann, minlength=na).astype(float)
    dmax = np.hypot(np.ptp(x), np.ptp(y))
    if kr_mode == "legacy":   # build_window_response grid, hat (linear) basis
        kr = np.concatenate((np.arange(0.0, 0.600001, 0.01), np.arange(0.66, dmax+0.12, 0.06)))
    else:                     # 2026-10-16: cubic-Hermite basis on 10 pc (r <= 0.6 kpc) / 20 pc steps
        kr = np.round(np.concatenate((np.arange(0.0, 0.600001, 0.01), np.arange(0.62, dmax+0.06, 0.02))), 10)
    nk = len(kr); herm = kr_mode != "legacy"; nks = 2*nk if herm else nk    # kernel slots: values (+ derivatives)
    hk = np.diff(kr)
    hy, hx = np.meshgrid(np.arange(2*valid.shape[0]-1)/2, np.arange(2*valid.shape[1]-1)/2, indexing="ij")
    half_mod = map_coordinates(modulation, [hy, hx], order=1, mode="nearest")
    slopes = np.asarray(slopes, float); ns = len(slopes); nc = 3*ns

    def geom(a, b):
        d = np.hypot(x[a:b, None]-x[None, :], y[a:b, None]-y[None, :])
        mr = np.hypot(0.5*(x[a:b, None]+x[None, :]), 0.5*(y[a:b, None]+y[None, :]))
        gm = half_mod[iy[a:b, None]+iy[None, :], ix[a:b, None]+ix[None, :]]
        lo = np.clip(np.searchsorted(kr, d, side="right")-1, 0, nk-2)
        fr = (d-kr[lo])/(kr[lo+1]-kr[lo])
        ob = np.searchsorted(edges, d, side="right")-1
        sel = (d > 0) & (ob >= 0) & (ob < nb)
        return mr, gm, lo, fr, ob, sel

    def basis(lo, fr):
        """(slot index offsets, weights) of the kernel basis for pairs at fraction fr of interval lo."""
        if not herm:
            return [(lo, 1-fr), (lo+1, fr)]
        t = fr; t2 = t*t; t3 = t2*t; h = hk[lo]
        return [(lo, 2*t3-3*t2+1), (lo+1, -2*t3+3*t2), (nk+lo, h*(t3-2*t2+t)), (nk+lo+1, h*(t3-t2))]

    t0 = time.time()
    partner = np.zeros((npix, na, nb))
    for a in range(0, npix, 64):
        b = min(a+64, npix); *_, ob, sel = geom(a, b)
        idx = (np.arange(b-a)[:, None]*na + ann[None, :])*nb + ob
        partner[a:b] = np.bincount(idx[sel], minlength=(b-a)*na*nb).reshape(b-a, na, nb)
    ordered = partner.sum(axis=(0, 1))
    pflat = np.ascontiguousarray(partner.reshape(npix*na, nb).T)      # (nb, npix*na)
    ew = partner.sum(axis=1).T                                          # (nb, npix)
    app = np.asarray([partner[ann == a].sum(axis=0) for a in range(na)])  # (na, na, nb)
    del partner
    log(f"pair geometry {time.time()-t0:.0f}s npix={npix} na={na} nb={nb} nk={nk} slopes={ns}")

    raw = np.zeros((nb*nks, nc)); pr = np.zeros((nb, nks*nc)); t4 = np.zeros((nb, nks*nc))
    d1 = np.zeros((nb, nks*nc)); d2 = np.zeros((nb, nks*nc))
    ew_ann = np.stack([ew[:, ann == a].sum(axis=1) for a in range(na)], axis=1)   # (nb, na)
    # annulus-mean terms accumulated row by row (no (na, na, nks, nc) array):
    #   sum_{a,a'} app[a,a',b] Cbar[a,a'] = sum_i sum_a' app[a(i),a',b]/N_a(i) Cbar[i,a'];  sum_a ew_ann[b,a] Cbar[a,a] likewise
    for a in range(0, npix, block):
        b = min(a+block, npix); m = b-a
        mr, gm, lo, fr, ob, sel = geom(a, b)
        u = (mr-r_ref).ravel()
        rate = np.exp(np.outer(u, slopes))                              # (P, ns)
        g = gm.ravel()[:, None]
        F = np.concatenate([rate, rate*g, rate*rate], axis=1)           # (P, nc)
        rowann = (np.arange(m)[:, None]*na + ann[None, :]).ravel()
        lo_r, fr_r = lo.ravel(), fr.ravel(); P = len(u); cols = np.arange(P)
        bw = basis(lo_r, fr_r)
        A = sparse.csr_matrix((np.concatenate([w for _, w in bw]),
                               (np.concatenate([rowann*nks+o for o, _ in bw]), np.concatenate([cols]*len(bw)))),
                              shape=(m*na*nks, P))
        H = (A @ F).reshape(m, na, nks, nc)/ann_counts[None, :, None, None]   # row means
        s = sel.ravel(); ob_r = ob.ravel()[s]; cs = cols[s]
        bws = basis(lo_r[s], fr_r[s])
        B = sparse.csr_matrix((np.concatenate([w for _, w in bws]),
                               (np.concatenate([ob_r*nks+o for o, _ in bws]), np.concatenate([cs]*len(bws)))),
                              shape=(nb*nks, P))
        raw += B @ F
        pr += pflat[:, a*na:b*na] @ H.reshape(m*na, nks*nc)
        wt = app[ann[a:b]]/ann_counts[ann[a:b], None, None]            # (m, na, nb)
        t4 += wt.reshape(m*na, nb).T @ H.reshape(m*na, nks*nc)
        own = H[np.arange(m), ann[a:b]].reshape(m, nks*nc)              # (m, nks*nc)
        d1 += ew[:, a:b] @ own
        d2 += (ew_ann[:, ann[a:b]]/ann_counts[ann[a:b]][None, :]) @ own
        if (a//block) % 100 == 0:
            log(f"  rows {b}/{npix} ({time.time()-t0:.0f}s)")
    numer = raw.reshape(nb, nks, nc) - 2*pr.reshape(nb, nks, nc) + t4.reshape(nb, nks, nc)
    denom = -2*d1.reshape(nb, nks, nc) + d2.reshape(nb, nks, nc)
    rate_pix = np.exp(np.outer(radius[valid]-r_ref, slopes))
    fd = np.concatenate([rate_pix, rate_pix*modulation[valid][:, None], rate_pix**2], axis=1)
    denom[:, 0, :] += ew @ fd
    resp = np.stack([numer, denom], axis=0)/ordered[None, :, None, None]      # (2, nb, nk, nc)
    resp = resp.reshape(2, nb, nks, 3, ns).transpose(4, 3, 0, 1, 2)           # (ns, 3, 2, nb, nks)
    log(f"done {time.time()-t0:.0f}s")
    return dict(kernel_radius=kr, rate_log_slope_grid=slopes, response=resp, ordered_pair_counts=ordered,
                basis=np.array("hermite" if herm else "hat"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pixel_map"); p.add_argument("morphology"); p.add_argument("out")
    p.add_argument("--nodes", type=int, default=65); p.add_argument("--bound", type=float, default=0.9)
    p.add_argument("--subset", default=None, help="i0:i1 slice of the node list")
    p.add_argument("--slopes", default=None, help="explicit comma-separated slopes (overrides nodes)")
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--diagnostics", required=True)
    p.add_argument("--kr", default="hermite", choices=("hermite", "legacy"))
    a = p.parse_args()
    pm = dict(np.load(a.pixel_map)); morph = json.load(open(a.morphology))
    labels = np.zeros(pm["valid_mask"].shape, int)   # single pointing (as amusing_ktz_stage.run_fit)
    # same edge / annulus derivation as paper_single_pointing.py (lines 45-46)
    import glob
    diag = np.genfromtxt(a.diagnostics, delimiter=",", names=True)
    dr = diag["r_kpc"][1]-diag["r_kpc"][0]
    edges = np.concatenate(([diag["r_kpc"][0]-0.5*dr], diag["r_kpc"]+0.5*dr))
    aw = 0.4 if float(pm["pixel_size_kpc"]) >= 0.1 else 0.2
    if a.slopes:
        slopes = np.array([float(v) for v in a.slopes.split(",")])
    else:
        slopes = chebyshev_lobatto_nodes(a.nodes, a.bound)
        if a.subset:
            i0, i1 = map(int, a.subset.split(":")); slopes = slopes[i0:i1]
    lib = build_library(pm, labels, morph["r_ref_kpc"], slopes, edges, aw, block=a.block, kr_mode=a.kr,
                        log=lambda m: print(f"[{os.path.basename(a.out)}] {m}", flush=True))
    lib.update(annulus_width=aw, edges=edges, pixel_map_md5=hashlib.md5(open(a.pixel_map, "rb").read()).hexdigest(),
               r_ref=morph["r_ref_kpc"], bound=a.bound, nodes=a.nodes, subset=str(a.subset), kr_mode=a.kr)
    tmp = a.out + ".partial.npz"; np.savez(tmp, **lib); os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
