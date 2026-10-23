# Manuscript version record

## Current manuscript binding, 24 October 2026

Manuscript SHA-256:
`8d3a4e8d34ce4826fd9b5e600a59d0d14465609fa0730293a35f569fbf08695d`.

The text now previews the two illustrative results in the Introduction and the
four-term decomposition at the start of the Model section. Font conventions,
the unmodulated kernel in Table 1, the conditional relation between the mean
filters, and the L1n naming rule are explicit. Repeated explanations of the
smooth-template approximation and covariance limits now refer to their full
definitions. Necessary scope statements remain with the results.

Technical cell-response checks, covariance standard deviations and effective-bin
counts have moved to labelled parts of Appendix A without changing their values.
Figures 7/8 start with the interpretation of the displayed model, and the
application synthesis distinguishes the morphology and covariance-shape tests.
The C_NP sensitivity sentence links to the Thomas construction and midpoint test.
No unverified filtering percentages were added.

All 124 displayed equation blocks, Table 3 values, numerical products, figures,
validation code and bibliography are unchanged. The only changed symbol-table
entry adds the existing dimensionless kernel; only Figures 7/8 captions change.
The source compiled successfully in the built-in editor. No fit or simulation
was run. This version record does not assert Overleaf synchronization.

## Historical manuscript binding and plot cleanup, 23 October 2026

Manuscript SHA-256:
`4dcb2e23e5d84a80e773ea3cef86f8b56a33210d9049e5d50f06b847081fb53d`.
Bibliography SHA-256:
`aa539d5a9b0747100e62bd75802b4f509924c02685eee6e1a3af9ec717d037b7`.

The independent-review corrections fix the introduction to the variables and
covariance amplitude, round the NGC 2466 S L2 association-length upper uncertainty to
0.6 kpc from the archived posterior quantiles, and use the general template-mode
notation nu_k = k Delta x instead of the spiral-specific nu_n. No new numerical
diagnostic or sensitivity claim is added. The local source compiled successfully
with the desktop editor's compiler.

The subsequent display-only cleanup removes `(published)` from the third
Figure 4 panel title and the grey no-C_AB,NP comparison points and legend entry
from Figure 5. Its caption no longer describes the removed points. Both figures
were redrawn from archived arrays with Matplotlib 3.10.9; control redraws match
the preceding figures pixel for pixel, and retained matrix values, colour scales,
scatter coordinates, error bars and source-channel assignments are unchanged.
The script and notebook plotting cells match. No fit or simulation was rerun,
and no scientific array or quoted validation metric was edited.

GitHub publication is tracked by repository commits. This record does not
assert that Overleaf is synchronized. The 22 October record below is retained
as historical provenance, including its source hash and compilation statement.

## Historical readability synchronization, 22 October 2026

Overleaf manuscript SHA-256 recorded on that date:
`4a4dfb14d664a0e548222b955b1d355d9cc7bdb67f9d1162afda9fd152b70392`.
The manuscript source and PDF remain on Overleaf; this repository retains its
existing scope of scientific code, derived products and version records.

### Text and notation

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

### Plotting

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
