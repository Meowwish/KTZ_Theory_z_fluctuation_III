# AMUSING++ applications: NGC 1084 and NGC 2466

The two observational applications are implemented as ordinary Python modules and shell drivers, not as Jupyter notebooks.

* `scripts/amusing_firstlook.py` and `scripts/amusing_ktz_stage.py` prepare the AMUSING++ maps and KTZ inputs.
* `scripts/paper_single_pointing.py` performs the single-pointing fits used in the manuscript.
* `scripts/mean_field_significance.py` evaluates the phase-resolved mean-field significance.
* `scripts/run_v3_ngc1084.sh`, `scripts/run_v3_ngc2466.sh`, and the `run_v4_effbeam*.sh` drivers reproduce the fitted variants and effective-beam results.
* `../ktzobs/scripts/` contains the finite-window response, source-template, evidence, and robustness modules called by the AMUSING++ drivers.
* `metadata/table3_numbers.json` records the values reported in Table 3; the `*_final` entries (2026-10-16) are the published ones.
* `scripts/ktz_unified_response.py`, `scripts/response_library.py` and `scripts/cnp_fast.py` implement the converged stochastic responses used for the final Table 3 (65 Chebyshev--Lobatto rate-slope nodes, cubic-Hermite separation basis, per-sample kernels); `paper_single_pointing.py --unified-response --support intrinsic --eps-prior manuscript` selects them. `scripts/direct_pairs.py` evaluates the brute-force pair sums used to verify them.
* The finite-map mean-field term is the projected full ensemble mean, `P W mu` with `mu = mu_bar + mu_mod` (`ktz_exact_window.exact_mean_field_response`), entered once as a rank-one term.
* `tests/` contains the convergence, prior-support and epsilon-prior regression tests.

The public repository contains the analysis code and compact manuscript products. Large intermediate response arrays, posterior samples, and logs are intentionally omitted.
