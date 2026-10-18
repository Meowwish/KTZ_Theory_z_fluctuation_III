# AMUSING++ applications: NGC 1084 and NGC 2466

The two observational applications are implemented as ordinary Python modules and shell drivers, not as Jupyter notebooks.

* `scripts/amusing_firstlook.py` and `scripts/amusing_ktz_stage.py` prepare the AMUSING++ maps and KTZ inputs.
* `scripts/paper_single_pointing.py` performs the single-pointing fits used in the manuscript.
* `scripts/mean_field_significance.py` evaluates the phase-resolved mean-field significance.
* `scripts/run_v3_ngc1084.sh`, `scripts/run_v3_ngc2466.sh`, and the `run_v4_effbeam*.sh` drivers reproduce the fitted variants and effective-beam results.
* `../ktzobs/scripts/` contains the finite-window response, source-template, evidence, and robustness modules called by the AMUSING++ drivers.
* `metadata/table3_numbers.json` records Table 3. The `*_final` entries include the 2026-10-18 conditional physical-prior normalisation; `*_final_box_prior` preserves the preceding coordinate-box evidence values. Posterior parameters, curves and chi-squared values are unchanged.
* `scripts/ktz_unified_response.py`, `scripts/response_library.py` and `scripts/cnp_fast.py` implement the converged stochastic responses used for the final Table 3 (65 Chebyshev--Lobatto rate-slope nodes, cubic-Hermite separation basis, per-sample kernels); `paper_single_pointing.py --unified-response --support intrinsic --eps-prior manuscript` selects them. `scripts/direct_pairs.py` evaluates the brute-force pair sums used to verify them.
* The finite-map mean-field term is the projected full ensemble mean, `P W mu` with `mu = mu_bar + mu_mod` (`ktz_exact_window.exact_mean_field_response`), entered once as a rank-one term.
* `tests/` contains the convergence, prior-support and epsilon-prior regression tests.

## Conditional physical-prior evidences

The documented independent densities define a base prior. The sigma, kappa, age and source-amplitude bounds condition that prior jointly. If `F_M` is the retained base-prior probability, the reported evidence is `Z_box / F_M`. The likelihood remains `-chi2/2` on the legal domain; this normalisation is not an epsilon weight or an extra likelihood term. The conditioned marginal priors need not retain the base density's individual forms.

`../ktzobs/scripts/physical_prior.py` integrates the beta and beta-times-amplitude constraints analytically and the remaining dimensions by quadrature. `metadata/prior_normalization_20261018.json` records the scalar prior inputs, source hashes, all 16 corrected fit summaries, uncorrected evidences and unchanged posterior piece weights. No large response arrays are needed to reproduce the correction.

Run `OPENBLAS_NUM_THREADS=1 python amusing/tests/physical_prior_test.py` from the repository root for the independent direct-prior-draw and split-weight regression checks.

For an archived combined split result, run `python amusing/scripts/normalize_physical_evidence.py SUMMARY.json MORPHOLOGY.json OUTPUT.json`. Use a new output filename; the input is preserved. New `paper_single_pointing.py --eps-prior manuscript` runs report conditional physical evidences by default and use an `_physical` output suffix; `--evidence-prior box` retains the old convention for audit comparisons. `combine_split_evidence.py ... --physical-prior` uses existing `_physical` pieces when present, otherwise archived box pieces, and writes a separately suffixed combined result. It recombines box evidences first and then applies the global normalisation. Equivalently the conditional piece prior weights are `f_i F_Mi / F_M`, not the unconditioned log-interval masses `f_i`.

The public repository contains the analysis code and compact manuscript products. Large intermediate response arrays, posterior samples, and logs are intentionally omitted.

The single-pointing figures show curves and posterior bands conditional on each model's highest-posterior-mass transport-scale interval. They are illustrative conditional summaries, not globally marginalised posterior bands. Evidences and tabulated posterior parameter summaries combine all computational intervals. The physical-prior normalisation changes neither the within-model posterior nor these figure files.
