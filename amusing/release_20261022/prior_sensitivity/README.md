# Scalar supplement to the prior-sensitivity test (2026-10-21)

No fit was rerun. The test design is unchanged. Overleaf, GitHub, the manuscript tables, figures and model settings were not touched. Source run files are in `TITANIC/prior_sens_20261021_094030/runs{,_eb}/` (variants) and `TITANIC/fluxarea_fix_20261021_074830/macfit/runs_fa`, `runs_eb_fa` and reused `~/macrun/macfit/runs_e` (baseline).

## 1. Machine-readable scalars

| File | Content |
|---|---|
| `variants_null.{csv,json}` | the 60 null variant runs (L1, L2 × 10 pieces × 3 variants) |
| `variants_eb.{csv,json}` | the 60 EB variant runs |
| `baseline_null.{csv,json}` | the 40 baseline null runs: L1/L2 from `runs_fa`; L0/L1n are the reused runs named in the post-processing jobs |
| `baseline_eb.{csv,json}` | the 40 baseline EB runs (all four models) |
| `extract_run_scalars.py` | the extraction script, reading job JSON + run NPZ; no sample arrays exported |
| `recompute_evidence.py` | independent piece combination and F_M conditioning from the CSVs only, using `code/physical_prior.py` (production) and `code/physical_prior_cuts.py` (cuts as a parameter; identical to production at the production cuts) and the morphology files actually used (`morphology/`) |
| `recomputed.json`, `recompute_evidence.out` | its output; it reproduces `results/prior_sens_{null,eb}.json` and, for the baseline, `results_{null,eb}_fa.json` |
| `delta_summary.py`, `delta_summary.{json,out}` | item 2 below |

Fields per run in the CSV/JSON:
- run file, its SHA256, `reused` flag, variant, model, galaxy, data directory;
- transport bounds `ell_min`/`ell_max`, `m1`, `rmin_fit`, `src_max`, `nlive`, `seed`, `eps_prior` (as stored in the NPZ and in the job), `cuts`;
- the SFR of the morphology actually used;
- final `logz`/`logzerr`, `max_logl`, `chi2_min = -2 max_logl`, `niter`, total `ncall`, `n_samples`;
- the input `fit_covariance.npz` path and SHA256, the morphology SHA256, and the dynesty/numpy versions.

Checks:
- All 200 runs have `eps_prior = manuscript`, seed 2 and nlive 400. The variants use the same seed as the baseline.
- Each variant uses the same covariance SHA as its baseline: null 07423a8b / 46449a87 / a64ee019; EB 36052c2d / 99fca60d / d5254ba5.

## 2. Change relative to baseline versus span between settings

| Covariance | Model | max \|Δ(ΔlnZ)\| vs baseline | max span over all 4 settings |
|---|---|---|---|
| null | L1 | **0.91** (D16, ×2) | **1.45** (NGC 1084) |
| null | L2 | 0.86 (D16, wide cuts) | 0.92 (S) |
| null | L1n | 0 (not refitted) | 0 |
| EB | L1 | 1.03 (NGC 1084, ×2) | 1.98 (NGC 1084) |
| EB | L2 | 1.03 (D16, wide cuts) | 1.17 (S) |
| EB | L1n | 0 (not refitted) | 0 |

Corrections to my earlier summary:
- The "at most 1.45" was the **span between settings**. The largest change relative to the baseline under the null covariance is **0.91**.
- The earlier EB ranges (e.g. NGC 1084 3.7–5.7) refer to **L1 and L2 only**. L1n is unchanged by construction.
- The ordering is not strictly unchanged:
  - Null, NGC 2466 D16: L2−L1n goes from −0.64 (baseline) to +0.22 (wide cuts).
  - Null, NGC 2466 S: L2−L1 is only 0.22–0.88.
  - EB: NGC 1084 and NGC 2466 D16 change order in several pairs.
- The appropriate statement is therefore stability of the **main preferences and near-degeneracies** within the tested perturbations, not data-driven ordering in general.

Pairwise differences for every setting are in `delta_summary.json`.

## 3. Suggested Appendix A paragraph

`SUGGESTED_APPENDIX_A_PARAGRAPH.tex`. It reports only the three perturbations performed and states the scope:
- three maps;
- L1/L2 refitted, L0/L1n unchanged;
- EB covariance reused, not rebuilt;
- a single seed;
- NGC 1084 EB extensions not distinguished.

It makes no claim of general prior independence. Every number in it is traceable to `delta_summary.json`:

| Claim in the paragraph | Values |
|---|---|
| null change vs reference ≤ 0.92 | 0.913 (rounded up) |
| span across reference + 3 perturbations < 1.5 | 1.451 |
| S: L2−L1 | 0.22–0.88 |
| S: L1/L2 above L1n | 3.19–4.55 (quoted 3.1–4.6) |
| D16/NGC 1084: \|L2−L1n\| | ≤ 0.64 |
| D16/NGC 1084: L1n/L2 above L1 | ≥ 5.35 |
| EB change vs reference ≤ 1.04 | 1.034 (rounded up) |
| NGC 1084 EB pairwise \|diff\| | ≤ 1.37 |
| D16 EB max margin | 1.53 |
| per-run logzerr | 0.12–0.24 |
