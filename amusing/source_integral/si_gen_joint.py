#!/usr/bin/env python3
"""si_gen_joint.py CONFIG.json POINTS.json OUTDIR --draws K --seed S [--npilot 40] [--points a:b]
Joint generative propagation of the COMPLETE fitted model (KTZ core + random dipole nuisance) with one measurement-noise
realisation, through projection, ordered-pair numerators and endpoint denominators (review round 3, 2026-10-26).

For every parameter point i of POINTS.json (list of dicts: physical parameters as si_generate.Generator + 'w' weight +
'piece' label + 'idx' sample index) and every draw k:
    z = amp_i * ( m + e_P + e_NP + c f_dip ) + n,      c ~ N(0, q^2),  q = A_dip,
    m      : deterministic transported mean (FIXED for the point, never randomised),
    e_P, e_NP : legal same-event / distinct-event field draws (si_generate.Generator.draw_stochastic),
    f_dip  : production detrended m=1 template; the coefficient is a zero-mean random variable whose rms is q, as in
             the likelihood (q^2 * dipole pair sums, no mean, no cross term with m),
    n      : one of the stored line-flux Monte Carlo noise maps (measurement layer, added once, uniform random choice).
amp_i normalises the COMPLETE signal: amp_i^2 * < mean over valid cells of [P(m + e + c f_dip)]^2 > = sig2, with
sig2 = Var_valid(residual) - mean(z_err^2) (the production signal-variance definition); the expectation is a Monte Carlo
average over npilot independent pilot draws (own RNG stream), whose standard error gives the MC uncertainty of amp_i.
This global, cell-based normalisation is NOT the production per-bin s_b/n_b layer (s_b, n_b come from data endpoint
denominators and noise-map pair sums per bin); the generated statistic includes the noise directly and is not passed
through s_b/n_b.  It is a sensitivity construction, not an exact joint propagation of line-flux noise and selection.
Stored per draw (numerator, denominator per bin, ordered-pair sums): 'full' = m+e+d+n, 'full_nn' = m+e+d (no noise),
'nomean' = e+d+n; per point 'mean_only' = amp*m (so that cross terms can be isolated by bilinearity).
One atomic output file per point (OUTDIR/point_XXX.npz): resumable; existing files are skipped."""
import sys, json, time, os, hashlib
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import si_generate as sg


def full_amplitude(G, q, rng, npilot):
    pm = G.cfg.pm; v = G.geom.valid
    sig2 = max(float(np.var(np.asarray(pm["residual_dex"], float)[v])-np.nanmean(np.asarray(pm["metallicity_error_dex"], float)[v]**2)), 1e-8)
    acc = []
    for _ in range(npilot):
        e, enp = G.draw_stochastic(); c = q*rng.standard_normal()
        acc.append(np.mean(G.geom.project(e+enp+G.mean+c*G.fdip)[v]**2))
    acc = np.asarray(acc); mu = acc.mean(); se = acc.std(ddof=1)/np.sqrt(len(acc))
    amp = float(np.sqrt(sig2/mu)); amp_rel_se = float(0.5*se/mu)          # d amp/amp = -1/2 d mu/mu
    return amp, amp_rel_se, sig2


def main():
    a = sys.argv; cfgd = json.load(open(a[1])); pts = json.load(open(a[2])); pts = pts['points'] if isinstance(pts, dict) else pts; outdir = Path(a[3]); outdir.mkdir(parents=True, exist_ok=True)
    K = int(a[a.index("--draws")+1]); seed = int(a[a.index("--seed")+1]); npilot = int(a[a.index("--npilot")+1]) if "--npilot" in a else 40
    lo, hi = (map(int, a[a.index("--points")+1].split(":")) if "--points" in a else (0, len(pts)))
    code_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest(); gen_sha = hashlib.sha256(open(sg.__file__, "rb").read()).hexdigest()
    for i in range(lo, min(hi, len(pts))):
        out = outdir/f"point_{i:03d}.npz"
        if out.exists():
            continue
        t0 = time.time(); par = dict(pts[i]); q = float(par.get("A_dip", 0.0))
        rng = np.random.default_rng([seed, i, 0]); rng_pilot = np.random.default_rng([seed, i, 1])
        G = sg.Generator(cfgd, par, rng=rng)
        main_rng, G.rng = G.rng, rng_pilot          # pilot draws use their own stream
        amp, amp_rel_se, sig2 = full_amplitude(G, q, rng_pilot, npilot)
        G.rng = main_rng
        noise = G.noise; nn = 0 if noise is None else len(noise)
        rec = {k: np.empty((K, 2, G.geom.nb)) for k in ("full", "full_nn", "nomean")}; cs = np.empty(K); ni = np.empty(K, int)
        M = amp*G.mean
        for k in range(K):
            e, enp = G.draw_stochastic(); c = q*rng.standard_normal(); d = amp*c*G.fdip
            j = int(rng.integers(nn)) if nn else -1; n = np.nan_to_num(noise[j]) if nn else 0.0
            s = amp*(e+enp)
            rec["full"][k] = G.statistic(s+M+d+n); rec["full_nn"][k] = G.statistic(s+M+d); rec["nomean"][k] = G.statistic(s+d+n)
            cs[k] = c; ni[k] = j
        mo = np.array(G.statistic(M))
        tmp = str(out)+f".{os.getpid()}.partial.npz"
        np.savez(tmp, **rec, mean_only=mo, dipole_coef=cs, noise_index=ni, counts=G.geom.counts, amp=amp, amp_rel_se=amp_rel_se, signal_var=sig2,
                 par=json.dumps(par), seed=np.array([seed, i]), npilot=npilot, n_age=len(G.V), seconds=time.time()-t0,
                 code_sha256=code_sha, si_generate_sha256=gen_sha, config=json.dumps(cfgd))
        os.replace(tmp, out)
        print(f"point {i} piece {par.get('piece')} w {par.get('w', 1):.4f}: amp {amp:.4g} (MC rel se {amp_rel_se:.3%}) {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
