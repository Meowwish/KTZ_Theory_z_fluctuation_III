# Readability synchronization, 22 October 2026

Current Overleaf manuscript SHA-256:
`4a4dfb14d664a0e548222b955b1d355d9cc7bdb67f9d1162afda9fd152b70392`.
The manuscript source and PDF remain on Overleaf; this repository retains its
existing scope of scientific code, derived products and version records.

## Text and notation

The text distinguishes disc inclination from spiral pitch angle, residual
second moments from connected covariances, and a parent complex from the
star-formation events that it generates. Short guides explain averaging
subscripts, the summed-modulation label, the covariance objects and the three
template/abundance supports. Dense validation, observational and prior paragraphs
are split by purpose. The midpoint-subscript/arm-number convention is retained.

The update also includes the preceding citation and validation-scope
clarifications. It describes NGC 1084's L1n/L2 evidence contrast as a weak
preference, discloses the existing negative-discrete-power clipping in proxy-field
generation, and calls the regression significance a moment-based
Gaussian-equivalent local score. These are descriptions of existing calculations,
not changes to them.

## Plotting

- Figure 5 labels the full-window coefficients with `sim` and `KTZ` superscripts
  and an `Omega` subscript. It no longer puts averaging brackets around an
  already normalized coefficient. Script and notebook labels match.
- Figures 7/8 replace `sum (L2)` by `L2 total`, and legend fontsize increases
  from 6.8 to 7.2. The caption retains the nuisance/noise and component-median
  qualifications; the complete median is not generally a sum of component medians.
- `validation/replot_cross_scatter.py` redraws Figure 5 from saved arrays and the
  original yield tables without generating a new realization. The existing
  `amusing/scripts/plot_from_archive_fa.py` redraws Figures 7/8 without a model,
  response library or sampler.

Before/after fingerprints of all plotted scientific arrays agree. Table 3,
posterior archives, covariance matrices, validation statistics, masks, priors,
operators and likelihoods are unchanged. No simulation or fit was rerun. The
updated manuscript remains 31 pages, with no compilation errors, undefined
references/citations or missing figures. See `VERIFICATION.md` for check scope.
