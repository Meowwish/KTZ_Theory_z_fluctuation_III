"""Polar-harmonic C_NP == brute-force Cartesian source integral (dense, FFT for E), on a cropped real mask."""
import sys, time
import numpy as np
from pathlib import Path
from scipy.signal import fftconvolve
sys.path.insert(0, str(Path(__file__).resolve().parent))
from si_geometry import Geometry, load_variant, bin_edges
from si_cnp_polar import PolarWindow, RadialKernels
from test_si_same_event import BASE, crop


def brute_cnp(geom, V, V2, L, slope, r_ref, step=0.02, ext=None):
    vy, vx = np.nonzero(geom.valid); x, y = geom.X1[vx], geom.Y1[vy]
    ext = ext or 6*np.sqrt(max(V, V2))+4*L
    xs = np.arange(x.min()-ext, x.max()+ext+step/2, step); ys = np.arange(y.min()-ext, y.max()+ext+step/2, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij"); Rr = np.exp(slope*(np.hypot(xx, yy)-r_ref))
    def T(Vv):
        return np.exp(-((x[:, None]-xx.ravel()[None])**2+(y[:, None]-yy.ravel()[None])**2)/(2*Vv))/(2*np.pi*Vv)*(Rr.ravel()*step**2)[None]
    A, B = T(V), T(V2)
    kx = np.arange(-int(5*L/step)-1, int(5*L/step)+2)*step
    E = np.exp(-(kx[:, None]**2+kx[None]**2)/(2*L*L))
    EB = np.stack([fftconvolve(bb.reshape(yy.shape), E, mode="same").ravel() for bb in B])
    C = A @ EB.T; C = 0.5*(C+C.T)
    lab = geom.ann[vy, vx]; P = np.eye(len(x))
    for a in range(geom.na):
        s = lab == a; P[np.ix_(s, s)] -= 1.0/s.sum()
    C = P @ C @ P.T
    d = np.hypot(x[:, None]-x[None], y[:, None]-y[None]); bb = np.searchsorted(geom.edges, d, side="right")-1
    ok = (d > 0) & (bb >= 0) & (bb < geom.nb)
    num = np.bincount(bb[ok], weights=C[ok], minlength=geom.nb)
    dg = np.diag(C); den = np.bincount(bb[ok], weights=(0.5*(dg[:, None]+dg[None]))[ok], minlength=geom.nb)
    return num, den


def polar_cnp(geom, pw, V, V2, L, slope, r_ref, mmax, rho_step=0.01, nq=400, ext=None):
    ext = ext or 7*np.sqrt(max(V, V2))+6*L
    rho = np.arange(0, pw.nodes.max()+ext, rho_step); rw = np.full(len(rho), rho_step); rw[0] *= 0.5
    rw = rw*rho
    qmax = min(8/L, 8/np.sqrt(min(V, V2)))
    xg, wg = np.polynomial.legendre.leggauss(nq); q = 0.5*qmax*(xg+1); qw = 0.5*qmax*wg
    rk = RadialKernels(pw.nodes, rho, rw, q, qw)
    rate = np.exp(slope*(rho-r_ref)); Eh = 2*np.pi*L*L*np.exp(-0.5*L*L*q*q)
    num = np.zeros(geom.nb); den = np.zeros(geom.nb)
    for m in range(mmax+1):
        a1, a2 = rk.a(V, m, rate), rk.a(V2, m, rate)
        Mm = 2*np.pi*(a1*(Eh*q*qw)[None]) @ a2.T; Mm = 0.5*(Mm+Mm.T)
        Hh, Dh = pw.harmonics(m)
        c = 1.0 if m == 0 else 2.0
        num += c*np.einsum("ij,bij->b", Mm, Hh); den += c*np.einsum("ij,bij->b", Mm, Dh)
    return num, den


if __name__ == "__main__":
    pm, morph, diag = load_variant(BASE/"N1084D16_pixbeam_fa", "ngc1084")
    edges = bin_edges(diag)
    sub = crop(pm, (slice(55, 75), slice(80, 100)))   # crop near the centre: radii ~0.3-1.4 kpc
    geom = Geometry(sub, edges[:10])
    sb2 = (morph["effective_fwhm_kpc"]/2.354820045)**2
    print("cells", int(geom.valid.sum()), "na", geom.na)
    pw = PolarWindow(geom, 0.02)
    worst = 0
    for (V, V2, L, s, mm) in ((sb2, sb2, 0.12, -0.78, 40), (sb2+0.02, sb2+0.2, 0.3, -0.5, 40), (sb2+0.2, sb2+0.2, 0.6, 0.6, 30)):
        t0 = time.time(); nb_, db_ = brute_cnp(geom, V, V2, L, s, morph["r_ref_kpc"]); t1 = time.time()
        npl, dpl = polar_cnp(geom, pw, V, V2, L, s, morph["r_ref_kpc"], mm); t2 = time.time()
        en = np.max(np.abs(npl-nb_))/np.max(np.abs(nb_)); ed = np.max(np.abs(dpl-db_))/np.max(np.abs(db_))
        worst = max(worst, en, ed)
        print(f"V={V:.4f} V'={V2:.4f} L={L} s={s}: brute {t1-t0:.0f}s polar {t2-t1:.0f}s  rel.err num {en:.2e} den {ed:.2e}")
    print("worst", worst)
