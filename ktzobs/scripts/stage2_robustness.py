"""Shared helpers for the stage-2 robustness checks (appendix of the observation paper).

All checks re-use the stage-2 products of NGC<n>_PHANGS_exact_window_joint.ipynb (pointing labels, window
responses, split statistics) and the final stage-2 settings: flat eta prior, ell_src >= 0.12 kpc,
sigma_q ~ U[0, 3], 64 walkers, chains started from the stage-2 run-A posterior.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from scripts import ktz_exact_window as kew

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "implications"
ELL_SRC_LOWER_KPC = 0.12
ETA_PRIOR = "flat"
SIGMA_Q_MAX = 3.0


def stage_dirs(galaxy):
    return OUT / f"{galaxy}_halpha_map_systematic", OUT / f"{galaxy}_exact_window_joint"


def load_products(galaxy):
    """Stage-1 and stage-2 inputs of one galaxy."""
    g = galaxy.lower(); s1, s2 = stage_dirs(galaxy)
    P = dict(galaxy=galaxy, prefix=g, stage1=s1, stage2=s2)
    P["morphology"] = json.loads((s1 / f"{g}_morphology_summary.json").read_text())
    P["diagnostics"] = np.genfromtxt(s1 / f"{g}_residual_covariance_diagnostics.csv", delimiter=",", names=True)
    P["fit_covariance"] = np.load(s1 / f"{g}_fit_covariance.npz")["covariance"]
    with np.load(s1 / f"{g}_muse_scal_7p_ktz_cnp_full_pair_emulator.npz") as c:
        P["grids"] = {k: c[k] for k in ("ell_grid", "epsilon_grid", "source_length_grid")}
    P["pixel_map"] = dict(np.load(s1 / f"{g}_muse_scal_metallicity_residual_pixel_map.npz"))
    P["labels"] = dict(np.load(s2 / f"{g}_pointing_labels_faceon.npz"))
    P["response"] = {k: dict(np.load(s2 / f"{g}_window_response_{k}.npz")) for k in ("all", "same", "cross")}
    P["split"] = dict(np.load(s2 / f"{g}_split_data.npz"))
    P["joint_covariance"] = np.load(s2 / f"{g}_joint_covariance.npz")["covariance"]
    P["summary"] = json.loads((s2 / f"{g}_exact_window_summary.json").read_text())
    with np.load(s2 / f"{g}_chain_A.npz") as c:
        P["chain_A"] = c["samples"]
    return P


def make_model(P, split=None, covariance=None, diagnostics=None, sigma_q_max=SIGMA_Q_MAX, use_D=True):
    """SplitKTZ with the stage-2 settings; split / covariance / diagnostics can be replaced."""
    return kew.SplitKTZ(P["morphology"], P["diagnostics"] if diagnostics is None else diagnostics, P["fit_covariance"],
                        P["response"]["all"], P["response"]["same"], P["response"]["cross"],
                        P["split"] if split is None else split, P["joint_covariance"] if covariance is None else covariance,
                        P["grids"], use_D=use_D, sigma_q_max=sigma_q_max, src_length_bounds=(ELL_SRC_LOWER_KPC, None), eta_prior=ETA_PRIOR)


def start_positions(model, samples, rng, n_walkers=64, jitter=0.0):
    """Walker starts drawn from an existing posterior (stage-2 run A), filtered on finite probability."""
    cand = samples[rng.choice(len(samples), 4000, replace=False)].copy()
    if jitter > 0:
        cand[:, :7] += rng.normal(0, jitter, cand[:, :7].shape); cand[:, 7] = np.abs(cand[:, 7] + rng.normal(0, jitter, len(cand)))
    cand = cand[np.isfinite(model.log_probability(cand))]
    while len(cand) < 4 * n_walkers:
        jit = cand[rng.choice(len(cand), 400)] + rng.normal(0, 0.05, (400, cand.shape[1])); jit[:, 7] = np.abs(jit[:, 7])
        cand = np.vstack([cand, jit[np.isfinite(model.log_probability(jit))]])
    return cand[rng.choice(len(cand), n_walkers, replace=False)]


def run_fit(model, samples, n_steps, seed, tag="fit", log=print):
    rng = np.random.default_rng(seed); start = start_positions(model, samples, rng); t0 = time.time()
    res = kew.run_ensemble(model, start, n_steps, seed=int(rng.integers(2**31)), thin=5, burn=n_steps // 4,
                           progress=lambda k: log(f"[{tag}] step {k:,}/{n_steps:,} ({time.time() - t0:.0f} s)") if k % 10000 == 0 else None)
    summary, curves = kew.summarise_joint(model, res["samples"])
    summary["rhat_max"] = float(np.nanmax(res["rhat"])); summary["acceptance"] = float(res["acceptance"].mean())
    summary["n_steps"] = n_steps; summary["time_s"] = time.time() - t0
    log(f"[{tag}] done in {time.time() - t0:.0f} s; R-hat max {summary['rhat_max']:.3f}; ell* {summary['params']['ell_star'][1]:.3f}; "
        f"sigma_q {summary['params']['sigma_q'][1]:.2f} [{summary['params']['sigma_q'][0]:.2f}, {summary['params']['sigma_q'][2]:.2f}]; chi2_joint {summary['chi2_joint_median']:.1f}")
    return summary, curves, res["samples"]


def replace_field(pixel_map, field):
    pm = dict(pixel_map); pm["residual_dex"] = np.asarray(field, float); return pm


def ensure_positive_definite(C, nb):
    """Halve the xi_all-D cross block until the joint covariance is positive definite; returns (C, scale)."""
    C = np.array(C, float); scale = 1.0
    while True:
        try:
            np.linalg.cholesky(C); return C, scale
        except np.linalg.LinAlgError:
            scale *= 0.5; C[:nb, nb:] *= 0.5; C[nb:, :nb] *= 0.5
