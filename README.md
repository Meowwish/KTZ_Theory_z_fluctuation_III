# KTZ Paper III: validation and observational applications

The current observational release is [26 October 2026](amusing/release_20261026/README.md).
It uses source-position-dependent covariance integrals and transported template
responses, with recomputed null/EB fits and Figures 7/8. The primary covariance
is the unchanged fixed null reference; the rebuilt stationary EB covariance is
an empirical sensitivity check. Model comparisons remain conditional on those
working likelihoods and the stated source-completion assumption.

- Numerical validation: `KTZ_validation.ipynb` and `KTZ_validation.py`;
  [execution and Figures 1/5 redraw instructions](validation/README.md).
  This experiment is unchanged.
- Observational production code: [source-integral entrypoints](amusing/source_integral/),
  ordinary Python scripts rather than notebooks.
- Current evidence table: `amusing/metadata/table3_numbers.json`.
- Current per-interval posteriors, prepared-input index, convergence records and
  figure inputs: `amusing/release_20261026/`.
- Figures 7/8: `figs/ngc2466_single_pointing.pdf` and
  `figs/ngc1084_single_pointing.pdf`; redraw from the released arrays without
  fitting a model.

The full manuscript remains in the existing private manuscript repository.
Its hash is recorded in `amusing/release_20261026/metadata/manuscript_state.json`.
Older releases and production workers are retained for provenance and do not
produce the current source-integral results. No raw cubes or MAUVE products are
added by this release.
