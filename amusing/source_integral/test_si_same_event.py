"""Algebra regression: sensitivity-map same-event integrals == brute-force dense Gram covariance,
projected with the actual annulus labels and summed over actual ordered pairs (same source nodes)."""
import sys, time, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from si_geometry import Geometry, load_variant, bin_edges
from si_same_event import SourceGrid, SameEventMaps, midpoint_maps, g1

BASE = Path(__file__).resolve().parents[1]/"release_20261022/stage_inputs/inputs"


def crop(pm, sl):
    out = dict(pm)
    for k in ("valid_mask", "x_grid_kpc", "y_grid_kpc", "arm_source_modulation", "template_support"):
        out[k] = pm[k][sl]
    return out


def brute(geom, grid, V, h):
    vy, vx = np.nonzero(geom.valid)
    x, y = geom.X1[vx], geom.Y1[vy]
    yy, xx = np.meshgrid(grid.ys, grid.xs, indexing="ij")
    G = np.exp(-((x[:, None]-xx.ravel()[None])**2+(y[:, None]-yy.ravel()[None])**2)/(2*V))/(2*np.pi*V)
    C = (G*h.ravel()[None]*grid.step**2) @ G.T
    lab = geom.ann[vy, vx]
    P = np.eye(len(x))
    for a in range(geom.na):
        s = lab == a; P[np.ix_(s, s)] -= 1.0/s.sum()
    C = P @ C @ P.T
    d = np.hypot(x[:, None]-x[None], y[:, None]-y[None])
    bb = np.searchsorted(geom.edges, d, side="right")-1
    ok = (d > 0) & (bb >= 0) & (bb < geom.nb)
    num = np.bincount(bb[ok], weights=C[ok], minlength=geom.nb)
    diag = np.diag(C); end = 0.5*(diag[:, None]+diag[None])
    den = np.bincount(bb[ok], weights=end[ok], minlength=geom.nb)
    cnt = np.bincount(bb[ok], minlength=geom.nb)
    return num, den, cnt


if __name__ == "__main__":
    pm, morph, diag = load_variant(BASE/"N1084D16_pixbeam_fa", "ngc1084")
    edges = bin_edges(diag)
    sub = crop(pm, (slice(40, 62), slice(70, 96)))            # 22 x 26 cells incl. holes
    geom = Geometry(sub, edges[:12])                            # first 11 bins (< 0.69 kpc)
    print("cells", int(geom.valid.sum()), "annuli", geom.na, "bins", geom.nb)
    sb2 = (morph["effective_fwhm_kpc"]/2.354820045)**2
    rng = np.random.default_rng(1)
    worst = 0.0
    for V in (sb2, sb2+0.01, sb2+0.2):
        ext = 6*np.sqrt(V)
        grid = SourceGrid(geom.X1[0]-ext, geom.X1[-1]+ext, geom.Y1[0]-ext, geom.Y1[-1]+ext, 0.02)
        R = grid.radius(); h = np.exp(-0.78*(R-2.0))*(1+0.5*np.cos(3*np.arctan2(grid.ys[:, None], grid.xs[None, :])))
        t0 = time.time(); Y = midpoint_maps(geom, [V]); S = SameEventMaps(geom, grid, V, Y[:, 0], h[None]); r = S.contract()[0]
        t1 = time.time(); n, d, c = brute(geom, grid, V, h); t2 = time.time()
        assert np.array_equal(c, geom.counts)
        en = np.max(np.abs(r[0]-n))/np.max(np.abs(n)); ed = np.max(np.abs(r[1]-d))/np.max(np.abs(d))
        worst = max(worst, en, ed)
        print(f"V={V:.4f} maps {t1-t0:.1f}s brute {t2-t1:.1f}s  rel.err num {en:.2e} den {ed:.2e}")
    # pair_sums of explicit fields vs brute force
    f = rng.normal(size=geom.valid.shape); Pf = geom.project(f)
    vy, vx = np.nonzero(geom.valid); v = Pf[vy, vx]
    C = np.outer(v, v); x, y = geom.X1[vx], geom.Y1[vy]
    d = np.hypot(x[:, None]-x[None], y[:, None]-y[None]); bb = np.searchsorted(geom.edges, d, side="right")-1; ok = (d > 0) & (bb >= 0) & (bb < geom.nb)
    num = np.bincount(bb[ok], weights=C[ok], minlength=geom.nb); den = np.bincount(bb[ok], weights=(0.5*(v[:, None]**2+v[None]**2))[ok], minlength=geom.nb)
    a, b_ = geom.pair_sums(Pf)
    e2 = max(np.max(np.abs(a-num))/np.max(np.abs(num)), np.max(np.abs(b_-den))/np.max(np.abs(den)))
    print(f"pair_sums field check rel.err {e2:.2e}")
    # kernel distances use integer lags x pixel (as PairStatistic); the stored float32 lattice differs by <3e-7 kpc,
    # which changes G_2V(d) at the 1e-8 level only (bin assignment is identical: no separation lies within 1e-5 kpc of an edge)
    assert worst < 1e-6 and e2 < 1e-10
    print("PASS")
