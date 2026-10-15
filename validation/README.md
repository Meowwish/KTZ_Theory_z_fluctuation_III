# KTZ Paper III — unified Monte Carlo validation (2026-10-14/15)

Generates every number and figure of Section 3 of the manuscript.

* `KTZ_validation.ipynb` — executed notebook (clean kernel, 0 errors, runtime 994 s on 2 cores).
  It is built from `KTZ_validation.py` (`# %%` cells) with
  `python3 validation/py2nb.py KTZ_validation.py KTZ_validation.ipynb`.
* One clustered-source experiment (Thomas process, `f_cl = 0.5`, `sigma_cl = 0.35 kpc`, `N_ch = 20`) is used for both the scalar and 16-element cross-correlation tests; the theory includes all four terms.
* The observation operator is a Gaussian beam with `FWHM = 2 dx` (0.40 kpc scalar, 0.10 kpc cross), followed by sampling at grid centres. No square-pixel operator is used in the numerical validation.
* `validation/outputs/validation_metrics.json` contains all metrics quoted in the paper.
* `validation/sector_cnp_phase_diagnostic.py` is the arm-sector check with midpoint-modulated `C_NP`.
* `validation/diagnostics/` records the focused checks used to diagnose the earlier cross-correlation bias.

The KTZ same-event prediction uses the direct two-dimensional Gaussian drift average and its logarithmic age integral. The reference labelled KT25 uses the published Dawson/inverse-hyperbolic-sine form. The uniform KTZ-limit curve is stored separately.

The KT25 yield tables and stochastic SLUG moments are read from `data/kt25/`. Run from the repository root with:

```bash
jupyter nbconvert --to notebook --execute --inplace KTZ_validation.ipynb
```

`CNP_validation.ipynb` is retained as a historical, separate backup; Section 3 is reproduced by the unified notebook.
