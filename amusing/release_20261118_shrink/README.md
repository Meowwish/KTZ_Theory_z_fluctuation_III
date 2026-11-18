# Fixed-null covariance shrinkage sensitivity

These are additional L0/L2 fits at diagonal shrinkages 0.20 and 0.50, not
replacements for the primary 0.35 fits in `release_20261026`. The covariance is

```
Sigma(s) = (1-s) Sigma_raw + s diag(Sigma_raw) + 0.01^2 I.
```

Within each field and shrinkage, L0 and L2 use the same fixed matrix. The
observed statistic, masks, effective response, sub-beam cut, fixed dipole phase,
source integrals, all priors and physical cuts are unchanged. All intrinsic
transport intervals partition 0.08--3.2 kpc. The fits use 400 live points, seed 2,
the manuscript epsilon prior and the existing source-integral evaluator.
Interval coordinate-box evidences are combined with logarithmic prior masses
before conditioning on the global physical-prior domain.

## Results

L2 minus L0 log-evidence, with the sampler's propagated standard error:

| Field | s=0.20 | s=0.35 reference | s=0.50 |
|---|---:|---:|---:|
| NGC2466 S | 10.10 +/- 0.20 | 11.50 +/- 0.20 | 13.53 +/- 0.21 |
| NGC2466 D16 | 18.74 +/- 0.23 | 22.09 +/- 0.22 | 27.95 +/- 0.22 |
| NGC1084 D16 | 15.00 +/- 0.25 | 15.86 +/- 0.25 | 17.29 +/- 0.25 |

The NGC1084 0.35 reference here is the same-machine archived run. The published
cloud run remains 15.22 in the primary evidence table; it is not replaced by
15.86. The two values differ by 0.64. Two realizations do not determine an
empirical sampler-scatter estimate. The sensitivity fits change the magnitude,
most strongly for NGC2466 D16, but not the L2/L0 ordering at the tested values.
This varies regularization within the fixed-null covariance family.

## Files and verification

- `summary.json` and `summary.csv`: full-range evidence comparisons, parameter
  quantiles and boundary masses. Boundary masses use the outer 5% of the stated
  prior coordinate, not a threshold of 0.95 kpc. For example the high-ell flag is
  `(ln(ell)-ln(0.08))/ln(3.2/0.08)>0.95`, or ell > 2.66 kpc.
- `runs/`: all 40 final nested-sampling arrays. Samples are stored in unit-cube
  coordinates; `NestedModel.fill` in `ktzobs/scripts/ktz_exact_window_evidence.py`
  gives the physical coordinates. Path-valued metadata are portable; scientific
  arrays are unchanged. No checkpoints or executable pickle objects are used.
- `jobs/`: corresponding portable configurations.
- `outputs/`: per-interval predictions/posteriors and globally conditioned
  evidence for the six sensitivity cases and three archived 0.35 references.
- `inputs/`: the six matrices, observed binned statistics, morphology inputs
  and covariance provenance. No pixel maps or raw line data are added here.
- `validation.json`: independent recombination and matrix checks.
- `provenance.json`: original input/run checksums and publication baseline.
- `SHA256SUMS.txt`: checksums of this curated payload.

From the repository root, with NumPy and SciPy installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python amusing/source_integral/check_shrink_release.py
```

This recomputes all 40 run evidences, interval weights, prior normalization and
six covariance transformations without loading the response tables or fitting.

## Repeating the fits with existing production tables

First prepare the production inputs as described in `release_20261026/README.md`.
The accepted source-integral tables must already be in `SOURCE_WORK/tables`,
and its input aliases must be in `SOURCE_WORK/inputs`. Tables are not included
in this archive. Use a separate destination directory:

```bash
python amusing/source_integral/materialize_shrink_jobs.py /absolute/SOURCE_WORK /absolute/NEW_SHRINK_WORK
python amusing/source_integral/materialize_shrink_jobs.py /absolute/SOURCE_WORK /absolute/NEW_SHRINK_WORK --execute
```

The default is a dry-run. Execution only materializes inputs and jobs and
refuses differing pre-existing files; it starts no fits. Run each generated
job with `python amusing/source_integral/si_fit.py JOB.json`. Keep BLAS/OpenMP
single-threaded and choose concurrency from available memory; NGC1084 jobs can
use about 1.5 GB each. Post-process each variant with the existing code:

```bash
SI_MODELS=L0,L2 python amusing/source_integral/post_variant.py /absolute/NEW_SHRINK_WORK/jobs/S_sh020 /absolute/NEW_SHRINK_WORK/runs /absolute/NEW_SHRINK_WORK/outputs S_sh020 S_sh020_null
```

Repeat for the other five variants. `si_condition_subset.py` applies the same
global physical-prior normalization and interval weights checked above.
No dipole correction or reinterpretation is applied in this release.

The full manuscript, internal reports, host/session logs, credentials, response
tables, raw/private data and MAUVE products are not part of this payload.
