"""Exact finite-window geometry for the source-integral production evaluator (2026-10-26).

Everything here depends only on the actual valid-cell mask, the annulus labels used by the
production annular projection P, and the separation bins: no source model enters.

Conventions are those of the production code (ktz_exact_window / response_library / direct_pairs):
* cells: valid_mask of the released pixel map, positions on the stored tensor lattice
  (x along axis 1, y along axis 0; float64 copies of the stored float32 coordinates);
* P: subtraction of the mean over valid cells of each annulus floor(R/aw), aw = 0.4 kpc for
  0.12-kpc cells and 0.2 kpc otherwise (paper_single_pointing.py convention);
* bins: edges from the diagnostics file, ordered pairs (k, l), k != l, with
  edges[b] <= |x_k - x_l| < edges[b+1];
* per bin: numerator = sum over ordered pairs of (P C P^T)_kl, denominator = sum over ordered
  pairs of [(PCP^T)_kk + (PCP^T)_ll]/2 = sum_k e_b(k) (PCP^T)_kk; both later divided by the
  ordered-pair count (as response_library.py).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve


def load_variant(directory, galaxy):
    d = Path(directory); g = galaxy.lower()
    pm = dict(np.load(d/f"{g}_pixel_map.npz"))
    morph = json.loads((d/f"{g}_morphology.json").read_text())
    diag = np.genfromtxt(d/f"{g}_diagnostics.csv", delimiter=",", names=True)
    return pm, morph, diag


def bin_edges(diag):
    r = diag["r_kpc"]; dr = r[1]-r[0]
    return np.concatenate(([r[0]-0.5*dr], r+0.5*dr))


class Geometry:
    def __init__(self, pm, edges, annulus_width=None, annulus_counts=True, cache=None):
        self.valid = np.asarray(pm["valid_mask"]).astype(bool)
        self.H, self.W = self.valid.shape
        X, Y = np.asarray(pm["x_grid_kpc"], float), np.asarray(pm["y_grid_kpc"], float)
        assert np.all(X == X[0:1, :]) and np.all(Y == Y[:, 0:1]), "cells must lie on a tensor lattice"
        self.X1, self.Y1 = X[0].copy(), Y[:, 0].copy()
        self.pixel = float(pm["pixel_size_kpc"])
        self.aw = annulus_width if annulus_width is not None else (0.4 if self.pixel >= 0.1 else 0.2)
        self.edges = np.asarray(edges, float); self.nb = len(self.edges)-1
        R = np.hypot(X, Y)
        lab = np.floor(R/self.aw).astype(int)
        self.label_values, inv = np.unique(lab[self.valid], return_inverse=True)
        self.na = len(self.label_values)
        self.ann = np.full(self.valid.shape, -1, int); self.ann[self.valid] = inv
        self.n_a = np.bincount(inv, minlength=self.na).astype(float)
        self.u = np.stack([(self.ann == a).astype(float) for a in range(self.na)])      # (na, H, W)
        self.m = self.valid.astype(float)
        # lag grid (same separations as PairStatistic: integer lags times the pixel size)
        self.ly, self.lx = np.meshgrid(np.arange(1-self.H, self.H), np.arange(1-self.W, self.W), indexing="ij")
        self.lag = np.hypot(self.lx*self.pixel, self.ly*self.pixel)
        lb = np.searchsorted(self.edges, self.lag, side="right")-1
        self.lagbin = np.where((self.lag > 0) & (lb >= 0) & (lb < self.nb), lb, -1)
        if cache is not None and Path(cache).exists():
            z = np.load(cache)
            self.e, self.counts = z["e"], z["counts"]
            if annulus_counts:
                self.c, self.Nab, self.Eab = z["c"], z["Nab"], z["Eab"]
            else:
                self.c = None
            assert np.array_equal(z["edges"], self.edges) and np.array_equal(z["valid"], self.valid)
        else:
            self._counts(annulus_counts)
            if cache is not None and annulus_counts:
                import os; tmp = str(cache)+f".{os.getpid()}.partial.npz"
                np.savez(tmp, e=self.e, counts=self.counts, c=self.c, Nab=self.Nab, Eab=self.Eab, edges=self.edges, valid=self.valid)
                import os; os.replace(tmp, cache)

    # ---- pair counts -------------------------------------------------------------------
    def ring(self, b):
        return (self.lagbin == b).astype(float)

    def conv_ring(self, field, b):
        """sum over lags in bin b of field shifted: (K_b * field)(x) on the cell lattice."""
        out = fftconvolve(field, self.ring(b), mode="full")
        return out[self.H-1:2*self.H-1, self.W-1:2*self.W-1]

    def _counts(self, annulus_counts=True):
        nb, na = self.nb, self.na
        self.e = np.zeros((nb, self.H, self.W))          # e_b(k): partners of k in bin b
        self.c = np.zeros((nb, na, self.H, self.W)) if annulus_counts else None   # c_{b,a}(l)
        for b in range(nb):
            Kb = self.ring(b)
            self.e[b] = np.rint(fftconvolve(self.m, Kb, mode="full")[self.H-1:2*self.H-1, self.W-1:2*self.W-1])*self.m
            for a in (range(na) if annulus_counts else ()):
                self.c[b, a] = np.rint(fftconvolve(self.u[a], Kb, mode="full")[self.H-1:2*self.H-1, self.W-1:2*self.W-1])*self.m
        self.counts = self.e.sum(axis=(1, 2))                                   # ordered pairs per bin
        if not annulus_counts:
            return
        self.Nab = np.einsum("bayx,cyx->bac", self.c, self.u)                   # N_b(a, a')
        self.Eab = np.einsum("byx,ayx->ba", self.e, self.u)                      # E_{b,a}
        assert np.allclose(self.Nab.sum(axis=(1, 2)), self.counts)

    # ---- projection and pair sums of explicit cell fields -------------------------------
    def project(self, f):
        """P f for a cell field (H, W) or a stack (..., H, W)."""
        f = np.where(self.valid, f, 0.0)
        means = np.einsum("...yx,ayx->...a", f, self.u)/self.n_a
        return f - np.einsum("...a,ayx->...yx", means, self.u)

    def lagbin_sum(self, corr):
        """bin a full lag-correlation array (2H-1, 2W-1) into separation bins."""
        sel = self.lagbin >= 0
        return np.bincount(self.lagbin[sel], weights=corr[sel], minlength=self.nb)

    def pair_sums(self, f, g=None):
        """ordered-pair numerator sum_{(k,l) in b} f_k g_l and endpoint denominator sum_k e_b(k) f_k g_k
        for cell fields (already projected if wanted).  g defaults to f.  Exact (FFT correlation)."""
        g = f if g is None else g
        fm, gm = np.where(self.valid, f, 0.0), np.where(self.valid, g, 0.0)
        corr = fftconvolve(gm, fm[::-1, ::-1], mode="full")      # corr[lag] = sum_k f_k g_{k+lag}
        num = self.lagbin_sum(corr)
        den = np.einsum("byx,yx->b", self.e, fm*gm)
        return num, den
