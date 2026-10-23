# KTZ Paper III — unified Monte Carlo validation (2026-10-14/15)

Generates every number and figure of Section 3 of the manuscript.

The 23 October plot-only cleanup removes `(published)` from the third Figure 4
panel title and removes the grey no-`C_AB,NP` comparison points and legend entry
from Figure 5. The script and notebook plotting cells agree; archived numerical
results and quoted comparison metrics are unchanged. The simulation was not rerun.

* `KTZ_validation.ipynb` — executed notebook (clean kernel, 0 errors, runtime 994 s on 2 cores).
  It is built from `KTZ_validation.py` (`# %%` cells) with
  `python3 validation/py2nb.py KTZ_validation.py KTZ_validation.ipynb`.
* One clustered-source experiment (Thomas process, `f_cl = 0.5`, `sigma_cl = 0.35 kpc`, `N_ch = 20`) is used for both the scalar and 16-element cross-correlation tests; the theory includes all four terms.
* The observation operator is a Gaussian beam with `FWHM = 2 dx` (0.40 kpc scalar, 0.10 kpc cross), followed by sampling at grid centres. No square-pixel operator is used in the numerical validation.
* `validation/outputs/validation_metrics.json` contains all metrics quoted in the paper.
* Figure 1 now shows the simulated field, the subtracted axisymmetric mean, and
  the fluctuation used for the two-point statistics. Its seed-42 first field is
  archived in `validation/outputs/fig1_realisation_fields.npz`. Redraw only this
  figure, without generating events or running statistics, with
  `python validation/replot_scalar_fields.py --output /tmp/figure1.png`.
  Matplotlib 3.10.9 reproduces the published PNG pixel for pixel. Block 7 in
  both notebook and script uses the same plot; its original stochastic-field
  `scalar_fields` diagnostics are retained unchanged, rather than redefined
  for the newly displayed fluctuation.
* `validation/sector_cnp_phase_diagnostic.py` is the arm-sector check with midpoint-modulated `C_NP`.
* `validation/diagnostics/` records the focused checks used to diagnose the earlier cross-correlation bias.

Redraw Figures 4 and 5 from `validation/outputs/cross_results.npz`:

```bash
python validation/replot_cross_scatter.py --matrices --outdir /tmp/ktz-figures --dump /tmp/cross-figure-fingerprints.json
```

This reads the original deterministic yield tables to preserve the source-channel
groups and executes only the matrix/scatter plotting blocks of `KTZ_validation.py`.
Omit `--matrices` to redraw Figure 5 alone. It does
not generate events, recompute covariance matrices or run a fit. The axes denote
the full-window coefficients `Xi_AB,Omega` with `sim` and `KTZ` superscripts;
covariance matrices are averaged before normalization. NumPy, Matplotlib and
Astropy are required; Matplotlib 3.10.9 was used for the published redraw.

The KTZ same-event prediction uses the direct two-dimensional Gaussian drift average and its logarithmic age integral. The reference labelled KT25 uses the published Dawson/inverse-hyperbolic-sine form. The uniform KTZ-limit curve is stored separately.

The KT25 yield tables and stochastic SLUG moments are read from `data/kt25/`. Run from the repository root with:

```bash
jupyter nbconvert --to notebook --execute --inplace KTZ_validation.ipynb
```

`CNP_validation.ipynb` is retained as a historical, separate backup; Section 3 is reproduced by the unified notebook.
