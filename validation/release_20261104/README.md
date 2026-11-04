# Unified finite-map validation: 4 November 2026

The single-element final statistic is xi_b: response, per-map annular projection,
valid unordered pairs, then normalisation. Fixed-midpoint-radius and arm-sector
checks use the same estimator with explicitly restricted pair windows. The
multi-element Xi_AB uses projected zero-lag second moments, averaged before
normalising. This release fixes the unconverged 24-node linear-age evaluation
of its same-event diagonal, using converged log-variance quadrature instead.

Current numerical products are in `../outputs_rework_20261104/`; the previous
known-profile outputs and the 3 November metrics remain historical comparisons.
The 3 November cross-element RMSE of 0.0192 is superseded, not a final result.
The current RMSE is 0.0143846405; the single-element full-window RMSE is
0.0045635221. The remaining element-dependent residual has no established origin.
The representative ring checks do not bound every approximation or element pair.

Production, tests, and dependencies are documented in `../README.md`.
Run `python validation/check_unified_release.py` from the repository root to
verify the released checksums and reproduce both RMSEs from the arrays without
generating a field or fitting an observational model. Scientific diagnostics and
convergence JSON files are provided; internal review reports, machine logs and
full manuscript sources are not included.

The observational plot producer changes only its two xi axis labels; curves,
bands, selected intervals, Table 3 and posterior arrays remain unchanged.
The observational source-integral likelihood and its documented null/EB and
common-covariance sensitivity limitations are not changed or newly calibrated.
The complete manuscript and paper figures are maintained in the existing private
TITANIC repository. `manuscript_state.json` records the matching private source
hash. No raw observational cubes, MAUVE data, credentials, or private manuscript
text are added here.

`SHA256SUMS` uses repository-relative paths and excludes itself. It covers this
release's code, figures, numerical products and metadata. Regenerating numerical
archives may change their byte-level hashes even if their arrays agree; retain
the committed archive when checking the release.
