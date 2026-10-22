# Current observational release: 22 October 2026

This is the production version corresponding to the manuscript hash in
`metadata/manuscript_state.json`. The release itself does not rerun a fit or
change any accepted scientific result. The two reference fields are selected
illustrations, not a population sample. The numerical validation is still
reproduced by the root `KTZ_validation.ipynb`.

The [readability update](READABILITY_UPDATE.md) binds these unchanged scientific
products to the latest Overleaf text and updates the plotting labels in Figures
5, 7 and 8. It introduces no new fit, covariance, posterior or evidence result.

## Current products, not historical defaults

- `../metadata/table3_numbers.json`: current Table 3, full-precision values with
  columns chi-square (null), Delta ln Z (null), Delta ln Z (EB). Display at one decimal.
- `metadata/results_null_fa.json`, `results_eb_fa.json`, `results_extra_fa.jsonl`:
  final conditioned physical evidences and corresponding model summaries.
- `outputs/`: per-interval prediction percentiles, component medians, 4,000
  resampled posterior points/model/interval, and coordinate-box evidence records.
  **Unsuffixed `*_split_v3*.json` in this archive retain box-prior log evidences.**
  Their posterior summaries remain valid, but the final evidence is in `metadata/`:
  combine box pieces with their log-interval masses, then divide by global F_M.
- `metadata/morphology/`: corrected per-map SFR, slope and beam inputs for the
  three Table 3 maps. SFRs are 2.41668/1.78099/1.02100 solar masses/year for
  NGC 2466 S/D16 and NGC 1084 D16. They are analysed-footprint normalizations;
  D16 uses the original unattenuated H-alpha prescription, not total-galaxy SFR.
- `figure_inputs/`: exact archived inputs used by the pure plotting driver.
- `stage_inputs/`: exact final pixel maps, diagnostics, covariance and morphology
  for ten cases, deduplicated across eight stage directories. `case_inputs.json`
  maps every case/file to its original SHA-256, beam and SFR metadata. These are
  derived AMUSING++ products, not raw line cubes. The same S mask is reused for
  the r<=5-kpc separation cut; the three EB cases reuse their null pixel maps.
- `covariance/`: rebuilt EB matrices with field/measurement components retained.
- `significance/`: final local H-alpha regression results, not population p-values.
- `prior_sensitivity/`: scalar records of 120 variant and 80 baseline runs,
  their morphology inputs and the scalar-only recomputation code. EB covariance
  is held fixed in these tests; one seed was used.
- `convergence/`: original direct-pair reports and generating code for four
  reference maps. These use reference-run posterior points, inter-node points
  and selected domain edges, not an exhaustive test of final posterior samples.
  Near-zero terms use absolute errors. NGC 1084 EB and S were not separately
  direct-pair tested; see the manuscript's stated limitations.
- `provenance/`: pre-release table and original case-layout records. Absolute
  historical paths in those records identify the production run; they are not
  portable reproduction commands.

Historical `run_v3*`, `run_v4*`, the 18 October normalization record and the
pre-release table are retained as audit comparisons, not as the current entrypoint.

## Quick checks and exact figure redraw (no fitting)

From the repository root, with NumPy and SciPy installed:

```bash
OPENBLAS_NUM_THREADS=1 python amusing/scripts/check_final_release.py
OPENBLAS_NUM_THREADS=1 python amusing/tests/physical_prior_test.py
```

With Matplotlib 3.10.9, redraw Figures 7/8 directly from released arrays:

```bash
python amusing/scripts/plot_from_archive_fa.py --script amusing/scripts/paper_single_pointing.py --name NGC2466 --outdir /tmp/ktz-figures
python amusing/scripts/plot_from_archive_fa.py --script amusing/scripts/paper_single_pointing.py --name NGC1084 --outdir /tmp/ktz-figures
```

This executes only the load/figure block. It does not build a model, likelihood,
response library or sampler. Plotted L0 uses its own dominant interval; the other
curves/components share L2's dominant interval. These are conditional summaries;
Table 3 and full posterior summaries combine all intervals. The plotted complete
median curve need not equal the sum of component medians. An excluded-point
legend entry appears only if an excluded point is visible in the plotted y-range.
PDF metadata can differ across redraws; compare the pixel rendering and array
fingerprints rather than expecting a byte-identical PDF timestamp.
The 22 October check reproduced every scientific curve, error-bar, image array
and text fingerprint. A NumPy 2.5.4 redraw differed from the archived renderer
only in a few H-alpha contour raster pixels; published PDFs remain the exact
accepted files used on Overleaf. Figure 1 redraw was pixel-identical.

## Production code and a new full fit

`../scripts/amusing_ktz_stage.py` now includes the accepted sky-cell/native-spaxel
area correction in the normal stage entrypoint. `apply_sfr_area_fix.py` applies
the same correction to old stored stage maps after reproducing the old SFR.
`muse_cube_linemaps.py` supplies the upstream line-map/2-Angstrom absorption code.
Raw ESO FITS and large response libraries are not redistributed here.

`../production/` contains the actual lean fitting, checkpoint runner,
post-processing, evidence conditioning and evaluator-compilation chain. Only
repository-relative imports were adapted for publication; scientific expressions
and sampling settings were not changed. These workers require local job paths
and prepared response arrays, not the historical session paths in provenance.
The simpler explicit current entrypoint is `../scripts/run_final_piece.py`.

Prepare a **work copy** of the four stage inputs (pixel map, morphology,
diagnostics, fit covariance) under `amusing/outputs/GALAXY/ktz/VARIANT/`, for example:

```bash
python amusing/scripts/materialize_stage_inputs.py --case 'S_pixbeam_fa|v3' --directory amusing/outputs/NGC2466/ktz/scal_mockcov_fid0_v2_pixbeam_unif65e_fa --execute
```

This copies only; non-identical existing files are refused. Omit `--execute` to
print a plan. Use the mapped EB case rather than a null covariance for an EB fit.
The morphology used by a final fit must include both the flux-area correction
and the beam+cell variance `sigma_eff^2 = sigma_beam^2 + pixel^2/12`, once only.
The released per-map final morphologies provide the reference values.
In particular, use the final D16 morphology's 0.455321-kpc effective FWHM, not
the 0.447956-kpc parent beam stored as provenance.

Build missing response arrays (expensive; not performed for this release):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python amusing/scripts/prepare_final_responses.py WORK_DIRECTORY NGC2466 --execute
```

The guarded fit entrypoint prints its command unless `--execute` is appended:

```bash
python amusing/scripts/run_final_piece.py --case S --ell-min 0.08 --ell-max 0.22
```

It explicitly selects `--cm-nodes 65 --unified-response --support intrinsic
--eps-prior manuscript --evidence-prior physical --nlive 400 --seed 2`, the common
fixed-phase dipole, and the map's separation cut. It refuses uncorrected SFR or
beam+cell inputs. A refit should use a new variant directory and preserve the
released products. For NGC 2466 use intervals [0.08,0.22], [0.22,0.35],
[0.35,0.5], [0.5,3.2]; for NGC 1084 use [0.08,0.5], [0.5,3.2]. The 0.5 boundary
is computational, not an upper physical prior. Combine with
`combine_split_evidence.py ... --physical-prior`; never average conditional
piece evidences with unconditioned interval weights.

For an EB refit, prepare the appropriate EB stage covariance before sampling.
`assemble_eb_cov.py` combines the new field component with measurement covariance
once. NGC 2466 uses stochastic-only input curves; NGC 1084 uses the disclosed
complete-curve proxy. Neither is a fully generative covariance treatment.

Production dependencies include NumPy, SciPy, Matplotlib, Astropy and dynesty;
the checkpoint batch runner additionally uses GNU `timeout`. Plot-only redraw
does not need Astropy or dynesty. Pin versions for bit-level work; the archived
covariance environment was Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, Astropy 8.0.2.

This release contains AMUSING++ derived products only. It adds or changes no
MAUVE files, no credentials and no proprietary raw data. The manuscript itself
remains on Overleaf; its hash links the published products to the exact text.
