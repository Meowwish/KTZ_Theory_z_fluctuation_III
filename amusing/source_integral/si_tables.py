#!/usr/bin/env python3
"""Build the source-integral response tables for one variant (2026-10-26).

Stages (each resumable, atomic writes, one output file per work unit):
  geom   : window geometry cache (pair counts per bin/annulus)
  se V_j : same-event sensitivity contractions for response variance node V_j, all slope nodes,
           pieces [R, R g]; also the transported mean fields (G_V * h) at the cells
  mean   : cross pair sums of the projected transported mean fields over (piece, V_j) x (piece', V_j')
  cnp    : see si_cnp_build.py

Source completion C1 (user decision 2026-10-26): R(y) = exp(slope (|y| - r_ref)) on the whole plane
(no truncation at the central observational hole or the field edge); arm modulation g = bilinear
interpolation of the released template where measured, 0 elsewhere.  The beam/effective-cell
variance enters once, in V = intrinsic variance + sigma_eff^2.
"""
from __future__ import annotations
import argparse, json, os, sys, time, hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
sys.path.insert(0, str(Path(__file__).resolve().parent))
from si_geometry import Geometry, load_variant, bin_edges
from si_same_event import SourceGrid, SameEventMaps, midpoint_maps, g1

ELL_MIN, ELL_MAX, EPS_MIN = 0.08, 3.2, 0.015


def cl_nodes(n, lo, hi):
    return 0.5*(lo+hi)-0.5*(hi-lo)*np.cos(np.pi*np.arange(n)/(n-1))


def atomic_savez(path, **arrays):
    tmp = str(path)+f".{os.getpid()}.partial.npz"; np.savez(tmp, **arrays); os.replace(tmp, path)


class Config:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir, self.galaxy, self.out = Path(cfg["dir"]), cfg["galaxy"], Path(cfg["out"])
        self.pm, self.morph, self.diag = load_variant(self.dir, self.galaxy)
        self.edges = bin_edges(self.diag)
        self.sb2 = (self.morph["effective_fwhm_kpc"]/2.354820045)**2
        self.r_ref = self.morph["r_ref_kpc"]
        mu, sd = self.morph["event_rate_log_slope_kpc_inv"], self.morph["event_rate_log_slope_error_kpc_inv"]
        self.s_lo, self.s_hi = max(-0.9, mu-cfg.get("slope_nsigma", 8)*sd), min(0.9, mu+cfg.get("slope_nsigma", 8)*sd)
        self.s_nodes = cl_nodes(cfg["n_slope"], self.s_lo, self.s_hi) if "s_nodes_explicit" not in cfg else np.asarray(cfg["s_nodes_explicit"], float)
        self.lnV_lo = np.log(self.sb2+(ELL_MIN*EPS_MIN)**2)-1e-4
        self.lnV_hi = np.log(self.sb2+ELL_MAX**2)+1e-4
        self.V_nodes = np.exp(cl_nodes(cfg["n_V"], self.lnV_lo, self.lnV_hi))
        self.pixel = float(self.pm["pixel_size_kpc"])
        self.fine_div = cfg.get("fine_div", 4); self.coarse_frac = cfg.get("coarse_frac", 0.5)
        self.out.mkdir(parents=True, exist_ok=True)

    def provenance(self):
        files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(self.dir.iterdir()) if f.name.startswith(self.galaxy.lower()+"_") and f.suffix in (".npz", ".json", ".csv") and "response" not in f.name}
        return dict(cfg=self.cfg, inputs=files, s_nodes=self.s_nodes.tolist(), V_nodes=self.V_nodes.tolist(), sb2=self.sb2, r_ref=self.r_ref)

    def geometry(self, annulus_counts=True):
        return Geometry(self.pm, self.edges, annulus_counts=annulus_counts, cache=self.out/'geometry_cache.npz')

    def g_on(self, grid):
        mod = np.asarray(self.pm["arm_source_modulation"], float)
        X1, Y1 = self.pm["x_grid_kpc"][0].astype(float), self.pm["y_grid_kpc"][:, 0].astype(float)
        iy = (grid.ys-Y1[0])/(Y1[1]-Y1[0]); ix = (grid.xs-X1[0])/(X1[1]-X1[0])
        IY, IX = np.meshgrid(iy, ix, indexing="ij")
        return map_coordinates(mod, [IY, IX], order=1, mode="constant", cval=0.0)

    def grids(self, geom, V):
        sp = np.sqrt(V/2)
        # fine grid over the template support box (g piece): pixel/fine_div, never coarser than 0.5 sigma_p
        sup = np.asarray(self.pm["arm_source_modulation"]) != 0
        rows, cols = np.nonzero(sup)
        X1, Y1 = geom.X1, geom.Y1
        # composite Gauss-Legendre on the template pixel cells (edges at pixel centres, where bilinear g has kinks);
        # sub-cells per pixel so that each sub-cell is <= 1.5 sigma_p (4-point rule; refinement checked in check_se_grids.py)
        nsub = max(1, int(np.ceil(self.pixel/(1.5*sp))))
        ex = np.arange(cols.min()-1, cols.max()+1+1e-9, 1.0/nsub); ey = np.arange(rows.min()-1, rows.max()+1+1e-9, 1.0/nsub)
        fx = X1[0]+ex*(X1[-1]-X1[0])/(len(X1)-1); fy = Y1[0]+ey*(Y1[-1]-Y1[0])/(len(Y1)-1)
        fine = SourceGrid.gauss_cells(fx, fy, self.cfg.get("gl_points", 4))
        # coarse extended grid (R piece, C1 completion over the whole plane)
        smax = max(abs(self.s_lo), abs(self.s_hi))
        ext = np.sqrt(40*V)+smax*V+0.5
        step_c = min(self.coarse_frac*sp, 0.5)
        coarse = SourceGrid(X1[0]-ext, X1[-1]+ext, Y1[0]-ext, Y1[-1]+ext, step_c)
        return fine, coarse


def rates(cfg, grid):
    R = grid.radius()
    return np.exp(cfg.s_nodes[:, None, None]*(R[None]-cfg.r_ref))          # (ns, ny, nx)


def smooth_to_cells(geom, grid, V, H):
    """(n, ny, nx) source maps -> (n, H, W) cell fields of G_V * h (grid quadrature weights)."""
    Gy = g1(geom.Y1[:, None]-grid.ys[None, :], V)*grid.wy[None]; Gx = g1(geom.X1[:, None]-grid.xs[None, :], V)*grid.wx[None]
    return np.einsum("ij,njk,lk->nil", Gy, H, Gx, optimize=True)


def stage_se(cfg, geom, j, log=print):
    out = cfg.out/f"se_V{j:03d}.npz"
    if out.exists():
        log(f"se V{j} exists"); return
    V = cfg.V_nodes[j]; t0 = time.time()
    Y = midpoint_maps(geom, [V])[:, 0]
    fine, coarse = cfg.grids(geom, V)
    res = {}
    for name, grid in (("R", coarse), ("g", fine)):
        Hs = rates(cfg, grid)
        if name == "g":
            Hs = Hs*cfg.g_on(grid)[None]
        S = SameEventMaps(geom, grid, V, Y, Hs)
        res[name] = S.contract()                                             # (ns, 2, nb)
        res[name+"_mean"] = smooth_to_cells(geom, grid, V, Hs)              # (ns, H, W)
        res[name+"_grid"] = np.array([grid.step, len(grid.ys), len(grid.xs), grid.xs[0], grid.ys[0]])
        del S
    atomic_savez(out, se=np.stack([res["R"], res["g"]], axis=1), mean=np.stack([res["R_mean"], res["g_mean"]], axis=1),
                 V=V, grid_R=res["R_grid"], grid_g=res["g_grid"], pid=os.getpid(), seconds=time.time()-t0)
    log(f"se V{j} ({V:.4g}) done {time.time()-t0:.0f}s grids {res['R_grid'][1:3]} {res['g_grid'][1:3]}")


def stage_mean(cfg, geom, i, log=print):
    """cross pair sums for slope node i: MX (2, 2nV, 2nV, nb) [num/den] over fields (piece, V_j) projected."""
    out = cfg.out/f"mean_s{i:03d}.npz"
    if out.exists():
        log(f"mean s{i} exists"); return
    t0 = time.time(); nV = len(cfg.V_nodes)
    F = np.stack([np.load(cfg.out/f"se_V{j:03d}.npz", mmap_mode="r")["mean"][i] for j in range(nV)], axis=1)   # (2, nV, H, W)
    F = geom.project(F.reshape(2*nV, geom.H, geom.W))
    H, W = geom.H, geom.W
    FF = np.fft.rfft2(np.where(geom.valid, F, 0.0), s=(2*H-1, 2*W-1))
    sel = geom.lagbin >= 0; lb = geom.lagbin[sel]; Ly, Lx = geom.ly[sel] % (2*H-1), geom.lx[sel] % (2*W-1)
    n = 2*nV; num = np.zeros((n, n, geom.nb)); den = np.zeros((n, n, geom.nb))
    Fv = F[:, geom.valid]; ev = geom.e[:, geom.valid]
    for a in range(n):
        corr = np.fft.irfft2(np.conj(FF[a])[None]*FF[a:], s=(2*H-1, 2*W-1))[:, Ly, Lx]
        for k in range(corr.shape[0]):
            v = np.bincount(lb, weights=corr[k], minlength=geom.nb); num[a, a+k] = v; num[a+k, a] = v
        d = (Fv[a][None]*Fv[a:]) @ ev.T
        den[a, a:] = d; den[a:, a] = d
    atomic_savez(out, num=num, den=den, pid=os.getpid(), seconds=time.time()-t0)
    log(f"mean s{i} done {time.time()-t0:.0f}s")


def main():
    p = argparse.ArgumentParser(); p.add_argument("config"); p.add_argument("stage"); p.add_argument("units", nargs="*", type=int)
    a = p.parse_args(); cfgd = json.load(open(a.config)); cfg = Config(cfgd)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')} pid {os.getpid()}] {m}", flush=True)
    gpath = cfg.out/"geometry_provenance.json"
    if not gpath.exists():
        tmp = str(gpath)+f".{os.getpid()}.partial"; json.dump(cfg.provenance(), open(tmp, "w"), indent=1); os.replace(tmp, gpath)
    geom = cfg.geometry(); log(f"geometry: {int(geom.valid.sum())} cells, {geom.na} annuli, {geom.nb} bins")
    if a.stage == "se":
        for j in (a.units or range(len(cfg.V_nodes))):
            stage_se(cfg, geom, j, log)
    elif a.stage == "mean":
        for i in (a.units or range(len(cfg.s_nodes))):
            stage_mean(cfg, geom, i, log)


if __name__ == "__main__":
    main()
