# Source-integral observational release: 26 October 2026

This release uses source-position-dependent covariance integrals in the
observational likelihood. The complete same-event covariance is a Gram integral
with non-negative source weights. The Gaussian distinct-event term retains both
source-position rate factors. The transported deterministic mean is projected
before its single outer product is formed.

The source rate is continued radially over the unobserved plane. The H-alpha
contrast is zero where unmeasured; the source rate is not zero there. This is a
source-completion assumption. The observed masks, bins, effective beams,
sub-beam cuts, dipole phases, intrinsic priors and physical conditioning are
unchanged. The fixed null covariance remains the primary working likelihood.
The stationary EB proxy is rebuilt from the new L2 curves and is a separate
empirical check. These comparisons are conditional on the covariance and source
assumptions, not a full generative calibration of the fitted field.

## Products

- `metadata/table3_numbers.json` and `../metadata/table3_numbers.json` contain
  chi-square under null, Delta ln Z under null, and Delta ln Z under EB.
  The displayed table uses one decimal place.
- `outputs/` contains all ten null/EB/subset/template cases, including every
  computational transport interval, prediction percentiles and equal-weight
  posterior samples. Final physical-prior evidences are in `evidence_*.json`;
  individual interval summaries retain coordinate-box evidences.
- `metadata/input_index.json` maps the inputs. Existing public pixel maps and
  null covariance files are reused from the 22 October archive. New EB matrices,
  diagnostics and morphologies are in `inputs/`.
- `figure_inputs/` and `figures/` contain the inputs and numerical fingerprints
  for Figures 7/8. Each model uses its own highest-posterior-mass interval for its
  representative conditional median curve. The complete median need not equal
  the sum of the component medians. Evidence and parameter summaries combine
  all transport intervals.
- `checks/` contains source-integral convergence, evaluator, template-response,
  EB reconstruction and local template-regression records. The local regression
  score is conditional on its null fields and does not jointly propagate shared
  H-alpha noise and cell selection.
- `checks/tmplcmp/STATUS.json` flags the non-finite archived NGC1084 S L1 and
  0.5-kpc-template L1 diagnostic summaries. They are not usable quantitative
  tests. The manuscript's mean-filter comparison quotes medians only from the
  specified finite NGC2466 S/D16 tests, not sample maxima or a global bound.
- `metadata/manuscript_state.json` records the private manuscript's SHA-256 and
  the hashes of its updated figures. The full manuscript is not published here.

The 22 October products are retained for provenance. Their midpoint-rate fits,
prior-sensitivity runs and modulation refit are not source-integral results.
The numerical-validation notebook and its experiment are unchanged.

## Read-only checks and figure redraw

From the repository root, with NumPy, SciPy and Matplotlib installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python amusing/source_integral/check_release.py
python amusing/source_integral/si_plot_fig78.py --name NGC2466 --inputs amusing/release_20261026/figure_inputs --evidence amusing/release_20261026/figure_inputs/NGC2466/evidence_S.json --variant S --outdir /tmp/ktz-si-figures
python amusing/source_integral/si_plot_fig78.py --name NGC1084 --inputs amusing/release_20261026/figure_inputs --evidence amusing/release_20261026/figure_inputs/NGC1084/evidence_N1084D16.json --variant N1084D16 --outdir /tmp/ktz-si-figures
```

No fitting is needed for these commands. PDF timestamps can differ; compare
array fingerprints and the rendered figure.

## Production entrypoints

`../source_integral/` contains the source integrals, tabulation, fitting,
post-processing and plotting code. Publication changes to these modules are
import paths and the test-fixture path. The mathematical expressions and
sampler settings are unchanged. NumPy, SciPy and dynesty 3.1.0 are required for
the numerical pipeline; Matplotlib is required for plotting.

Prepare an independent work copy without overwriting differing files:

```bash
python amusing/source_integral/prepare_work.py /absolute/path/to/ktz-work
python amusing/source_integral/prepare_work.py /absolute/path/to/ktz-work --execute
```

Response tables are large and are not included. Build them before fitting;
this is an expensive calculation. For each generated table config, run
`si_tables.py CONFIG se J` for every V-node index and `si_tables.py CONFIG mean I`
for every slope index. Run `si_cnp_build.py CONFIG M0 M1 --s I:I+1` over the
configured slopes and disjoint harmonic ranges, then
`si_assemble.py CONFIG CNP_DIRECTORY TABLES.npz`. It checks harmonic coverage.
`si_fasttables.py INPUT_CONFIG TABLES.npz FAST.npz --n-ell 32 --n-eps2 24`
builds the contracted evaluator. Its input config needs `dir` and `galaxy`.
Keep BLAS/OpenMP single-threaded when running parallel jobs.

Generate all-interval jobs with
`make_jobs.py INPUTS TABLES RUNS JOBDIR S S_r5 D16 N1084D16 N1084t05 N1084S S_EB2 S_r5_EB2 D16_EB2 N1084D16_EB2`.
Run `si_fit.py JOB.json`, then use `post_variant.py` to post-process each case
and combine all intervals with their logarithmic prior masses and global
physical-prior normalization. The 0.5-kpc split is not an upper prior bound.
Checkpoints and final result files are written atomically. Use separate output
and cache directories for independent jobs.

Raw cubes, private data and the full manuscript are not included. This release
does not change or add MAUVE products.
