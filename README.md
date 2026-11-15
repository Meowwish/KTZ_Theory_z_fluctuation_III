# KTZ Paper III: validation and observational applications

## Single-element entry — 15 November 2026

TITANIC now focuses on single-element spatial correlations. `KTZ_validation.ipynb`
and the matching `KTZ_validation.py` contain the single-element experiment only,
with independent caches under `validation/outputs_single_element/` and metrics in
`validation_metrics_single_element.json`. The physical calculation and numerical
results are unchanged. See [the current validation instructions](validation/README.md).

The cross-element working manuscript, figures and independent validation entry
are maintained at <https://github.com/Meowwish/Cross-correlation>. Earlier combined
outputs and helper snapshots in this repository are retained for provenance.
This split does not change observational production code or posterior products.

## Observational release and earlier validation products

The current observational release is [26 October 2026](amusing/release_20261026/README.md).
It uses source-position-dependent covariance integrals and transported template
responses, with recomputed null/EB fits and Figures 7/8. The primary covariance
is the unchanged fixed null reference; the rebuilt stationary EB covariance is
an empirical sensitivity check. Model comparisons remain conditional on those
working likelihoods and the stated source-completion assumption.

- Numerical validation: `KTZ_validation.ipynb` and `KTZ_validation.py`;
  [execution instructions](validation/README.md).
  The 4 November combined synthetic products remain archived in
  `validation/outputs_rework_20261104/`; the current single-element copy is in
  `validation/outputs_single_element/`.
- Observational production code: [source-integral entrypoints](amusing/source_integral/),
  ordinary Python scripts rather than notebooks.
- Current evidence table: `amusing/metadata/table3_numbers.json`.
- Current per-interval posteriors, prepared-input index, convergence records and
  figure inputs: `amusing/release_20261026/`.
- Conditional common-covariance sensitivity: [30 October supplement](amusing/release_20261030_pps/README.md),
  with generation/assembly code, binned matrices and L0/L2 refit summaries.
  This does not replace Table 3 or certify generative calibration.
- Observational figures: `figs/ngc2466_single_pointing.pdf` and
  `figs/ngc1084_single_pointing.pdf`; redraw from the released arrays without
  fitting a model.

The full manuscript remains in the existing private manuscript repository.
The base-release hash is recorded in `amusing/release_20261026/metadata/manuscript_state.json`;
the text follow-up hash is in `amusing/release_20261030_pps/manuscript_state.json`.
Older releases and production workers are retained for provenance and do not
produce the current source-integral results. No raw cubes or MAUVE products are
added by this release.

The current manuscript fingerprint and the 4 November validation checksums are in
`validation/release_20261104/`. The complete manuscript is not public.

The 15 November split is indexed in `validation/release_20261115/`, including
payload checksums and the corresponding manuscript fingerprint. The earlier
release index is retained as a historical record.
