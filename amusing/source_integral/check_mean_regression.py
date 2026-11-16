#!/usr/bin/env python3
"""Compare transported-mean and observed template slopes for NGC2466 S.

Reuses the 40 posterior indices and pilot-normalised amplitudes of the released
regression test. No fits or stochastic fields are generated. The padded-grid
Fourier transfer integrates a uniform source age exactly; source interpolation
and the spatial grid are checked by doubling their resolution at three points.
This is a conditional amplitude diagnostic, not a joint predictive calibration.

Usage from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
    amusing/source_integral/check_mean_regression.py --output RESULT.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.fft import fftfreq, irfft2, next_fast_len, rfft2, rfftfreq
from scipy.ndimage import map_coordinates
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "amusing/release_20261026"


def project(field, valid, ann):
    av = ann[valid]
    counts = np.bincount(av)
    means = np.bincount(av, weights=field[valid]) / np.maximum(counts, 1)
    result = np.zeros_like(field)
    result[valid] = field[valid] - means[av]
    return result


def evaluate(refine=4, limit=None):
    folder = REL / "figure_inputs/NGC2466"
    pm_path = folder / "ngc2466_pixel_map.npz"
    morph_path = folder / "ngc2466_morphology.json"
    reg_path = REL / "checks/regsig/S_L2_postpred.json"
    post_path = REL / "outputs/S/ngc2466_paper_nested_v3_ell0.08-0.22.npz"
    pm = np.load(pm_path)
    morph = json.loads(morph_path.read_text())
    reg = json.loads(reg_path.read_text())
    samples = np.load(post_path)["L2_samples"]
    indices = np.random.default_rng(reg["seed"]).choice(
        len(samples), size=reg["posterior_predictive"]["K"], replace=False)
    amps = np.array(reg["posterior_predictive"]["amps"])
    if limit is not None:
        indices, amps = indices[:limit], amps[:limit]
    X, Y = pm["x_grid_kpc"], pm["y_grid_kpc"]
    valid = pm["valid_mask"].astype(bool)
    support = valid & pm["template_support"].astype(bool)
    g = pm["arm_source_modulation"].astype(float)
    phase = np.arctan2(Y, X)
    ann = np.floor(np.hypot(X, Y) / .4).astype(int)
    contrast = g[support] - g[support].mean()
    design = np.c_[contrast, np.cos(phase[support]), np.sin(phase[support])]
    pinv = np.linalg.pinv(design)
    observed = float((pinv @ pm["residual_dex"][support])[0])

    dx = float(pm["pixel_size_kpc"])
    step, pad = dx / refine, 30 * refine  # 3.6-kpc spatial padding
    height, width = g.shape
    nh = next_fast_len((height - 1) * refine + 2 * pad + 1)
    nw = next_fast_len((width - 1) * refine + 2 * pad + 1)
    fy = Y[0, 0] + (np.arange(nh) - pad) * step
    fx = X[0, 0] + (np.arange(nw) - pad) * step
    iy, ix = np.meshgrid((fy - Y[0, 0]) / dx,
                         (fx - X[0, 0]) / dx, indexing="ij")
    gs = map_coordinates(g, [iy, ix], order=1, mode="constant", cval=0)
    radius = np.hypot(fy[:, None], fx[None, :])
    k2 = ((2 * np.pi * fftfreq(nh, step))[:, None] ** 2
          + (2 * np.pi * rfftfreq(nw, step))[None, :] ** 2)
    beam_variance = (morph["effective_fwhm_kpc"] / 2.354820045) ** 2
    dipole = project(np.cos(phase - np.deg2rad(reg["par"]["m1"])), valid, ann)
    points = []
    for index, amp in zip(indices, amps):
        x = samples[index]
        ell, eps = np.exp(x[0]), expit(x[1])
        beta, eta = np.exp(x[2]), np.exp(x[3])
        source = np.exp(x[4] * (radius - morph["r_ref_kpc"])) * (1 + eta * gs)
        a = .5 * ell ** 2 * (1 - eps ** 2) * k2
        age_transfer = np.ones_like(a)
        np.divide(-np.expm1(-a), a, out=age_transfer, where=a != 0)
        transfer = np.exp(-.5 * (beam_variance + ell ** 2 * eps ** 2) * k2)
        smooth = irfft2(rfft2(source) * transfer * age_transfer, s=source.shape)
        mean = np.sqrt(beta) * smooth[
            pad:pad + height * refine:refine, pad:pad + width * refine:refine]
        model = amp * project(mean, valid, ann)
        slope = float((pinv @ model[support])[0])
        dip_variance = amp ** 2 * x[7] ** 2 * np.mean(dipole[valid] ** 2)
        correction = np.sqrt(reg["signal_var"] / (reg["signal_var"] + dip_variance))
        points.append(dict(
            posterior_index=int(index), predicted_slope=slope,
            mean_variance_fraction=float(np.mean(model[valid] ** 2)
                                         / np.var(pm["residual_dex"][valid])),
            predicted_slope_including_dipole_in_amplitude_normalisation=float(slope * correction)))
    quantiles = lambda key: np.percentile([p[key] for p in points], [16, 50, 84]).tolist()
    return dict(refinement=refine, actual_observed_slope=observed,
                archived_observed_slope=reg["slope_obs"],
                predicted_slope_quantiles_16_50_84=quantiles("predicted_slope"),
                mean_variance_fraction_quantiles_16_50_84=quantiles("mean_variance_fraction"),
                points=points,
                inputs_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (pm_path, morph_path, reg_path, post_path)})


def generator_check(point):
    """Independent comparison with the production mean branch at one point."""
    sys.path.insert(0, str(ROOT / "amusing/source_integral"))
    sys.path.insert(0, str(ROOT / "ktzobs"))
    import si_generate as sg
    from scripts import ktz_exact_window as kew

    samples = np.load(REL / "outputs/S/ngc2466_paper_nested_v3_ell0.08-0.22.npz")["L2_samples"]
    x = samples[point["posterior_index"]]
    # ell_src and betaA do not enter the mean; avoid a large unused allocation.
    par = dict(ell=float(np.exp(x[0])), eps=float(expit(x[1])),
               beta=float(np.exp(x[2])), eta=float(np.exp(x[3])), slope=float(x[4]),
               m1=1., A_dip=float(x[7]), ell_src=.12, betaA=0.)
    with tempfile.TemporaryDirectory(prefix="ktz-mean-check-") as scratch:
        cfg = dict(dir=str(REL / "figure_inputs/NGC2466"), galaxy="NGC2466",
                   out=scratch, n_slope=3, n_V=4)
        generator = sg.Generator(cfg, par, n_age=16)
    pm = generator.cfg.pm
    valid = pm["valid_mask"].astype(bool)
    support = valid & pm["template_support"].astype(bool)
    phase = np.arctan2(pm["y_grid_kpc"], pm["x_grid_kpc"])
    radius = np.hypot(pm["x_grid_kpc"], pm["y_grid_kpc"])
    g = pm["arm_source_modulation"][support]
    design = np.c_[g - g.mean(), np.cos(phase[support]), np.sin(phase[support])]
    reg = json.loads((REL / "checks/regsig/S_L2_postpred.json").read_text())
    mean = kew.detrend(reg["posterior_predictive"]["amps"][0] * generator.mean,
                      valid, radius, width=generator.geom.aw)
    slope = float(np.linalg.lstsq(design, mean[support], rcond=None)[0][0])
    return dict(posterior_index=point["posterior_index"],
                production_Generator_mean_slope=slope,
                relative_difference=abs(slope / point["predicted_slope"] - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = dict(scope="NGC2466 S, 40 dominant-interval posterior means under the stored signal-amplitude recipe; not a refit or joint predictive calibration. Stored amplitudes have finite pilot-MC error.",
                  S=evaluate(), refinement=evaluate(8, limit=3))
    result["max_relative_refinement_change"] = max(
        abs(f["predicted_slope"] / c["predicted_slope"] - 1)
        for f, c in zip(result["refinement"]["points"], result["S"]["points"]))
    result["generator_check"] = generator_check(result["S"]["points"][0])
    assert abs(result["S"]["actual_observed_slope"] - result["S"]["archived_observed_slope"]) < 1e-8
    assert result["max_relative_refinement_change"] < .0005
    assert result["generator_check"]["relative_difference"] < .0005
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(json.dumps({k: v for k, v in result.items() if k not in ("S", "refinement")}, indent=2))
    print("slope quantiles:", result["S"]["predicted_slope_quantiles_16_50_84"])


if __name__ == "__main__":
    main()
