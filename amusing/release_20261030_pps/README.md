# Conditional common-covariance sensitivity supplement: 30 October 2026

This supplement archives the two common-covariance comparisons discussed in the
manuscript. It does not replace the [26 October production release](../release_20261026/README.md),
Table 3, or Figures 7/8. The fixed null covariance is still the primary working
likelihood; the stationary EB covariance is a separate sensitivity test.

For each field, one covariance is generated at points from its null-fit L2
posterior (`L2gen`); another uses its null-fit L0 posterior (`L0gen`). Each
covariance is then held fixed in both L0 and L2 refits. The evidence differences
below compare models within one column, not absolute evidences across columns.

| Field | Fixed null | L2-generated common covariance | L0-generated common covariance |
| --- | ---: | ---: | ---: |
| NGC 2466 S | 11.5 | 2.8 | 1.3 |
| NGC 2466 D16 | 22.1 | -0.9 | 2.9 |
| NGC 1084 D16 | 15.2 | 2.0 | 3.9 |

Entries are Delta ln Z(L2-L0), with the original global physical-prior
conditioning and all intrinsic transport intervals combined. The computations
use the same source-completion assumption, masks, beams, cuts, priors and fixed
dipole phases as the production release. This is a conditional sensitivity
construction, not an independently calibrated replacement likelihood.

## Products and definitions

- `index.json` identifies existing public inputs, their hashes and numerical
  settings. No new pixel maps or native line-flux products are included here.
- `points/` contains the actual posterior points and weights used for generation.
  Intervals with posterior mass below 0.005 were omitted, their mass was recorded
  in `excluded_mass`, and retained weights were renormalised. These points are
  not a substitute for the complete posterior.
- `configs/` contains the source-integral configurations with repository-relative
  input paths. The preparation helper rewrites paths for an independent workdir.
- `covariances/` contains dimensionless binned matrices and means, not generated
  maps. `conditional` is the weighted average of the within-point sample
  covariances of the measured ratio N/D. `between` is the weighted spread of
  the conditional means; `pooled = conditional + between`. Only
  `covariance = 0.65*conditional + 0.35*diag(conditional) + 0.01^2*I`
  is used for fitting. The parameter-between block is not added to the likelihood.
  `raw_covariance` duplicates `conditional`. `r_kpc` and `fitted_bins` identify
  bin centres and the production separation cut.
- `noise_block` is `conditional - cond_no_noise`. `cross_block` is
  `conditional - cond_crossfree`, where the latter forms a ratio after removing
  mean--fluctuation cross-products from both numerator and denominator.
  These difference matrices need not be positive semidefinite or additive
  independent noise components. They are deletion diagnostics.
- `split_half_a/b` are within-point covariances from the two halves of the draws,
  posterior-weighted in the same way. They are diagnostics, not a certification
  that finite-Monte-Carlo errors in evidence have converged.
- `evidence/` contains combined evidences; `pieces/` contains the interval
  summaries and archived run hashes needed to check the combination.
  `summary.json` also supplies null/EB parameter comparisons. Parameter triples
  are the 16th, 50th and 84th percentiles combined over all intervals.
  L0's inactive source parameters are stored placeholders, not constraints.
  `mean_share_first_fitted_bin` is the conditional component share in the first fitted
  bin of the dominant interval, not a unique global explained-variance fraction.
- `regression_noise_mean.json` records the fixed-template, stored-noise-map
  slope diagnostic: the mean is about 6--8% of the observed NGC 2466 slope and
  is not subtracted from the reported regression score. This is not a joint
  template-noise or selection-bias calibration.
- `manuscript_state.json` records the matching private manuscript hash. The
  manuscript itself is not published in this repository. `SHA256SUMS.txt`
  covers this supplement and its new code entrypoints, with paths relative to
  the repository root; it does not revise historical release manifests.

## Generation and scope

`si_gen_joint.py` propagates a fixed transported mean, legal stochastic fields,
a zero-mean random dipole and one stored measurement-noise map through the same
projection and pair numerator/denominator. L0 generation sets the mean,
template modulation and distinct-event excess to zero. Pilot draws normalise
the complete projected signal to the observed signal variance; generated ratios
are not passed through the production per-bin signal/noise fractions again.
Measurement noise is included once inside these draws; no extra Sigma_MC is
added by `si_ppcov.py`.

Each point has 50 main draws and 100 independent pilot draws, with 16 age nodes.
Generation seeds are 303 (L2gen) and 404 (L0gen); point selection uses seed 31.
The 50 stored noise maps are reused. Incomplete noise-map support, fixed-template
conditioning, the ratio-of-moments approximation, and lack of joint line-flux
selection/template-noise propagation remain limitations. Sampler errors in the
evidence summaries do not include these effects or finite covariance-simulation
uncertainty. No evidence-error convergence certification is supplied.

## Read-only checks

From the repository root, using NumPy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python amusing/source_integral/check_pps_release.py
```

This checks hashes, covariance assembly and split-evidence bookkeeping without
running a fit or drawing a field. With SciPy installed, add `--physical-prior`
to also reintegrate the global physical-prior masses from the released metadata.

## Reproduction entrypoints

Preparation starts no numerical jobs and refuses to overwrite differing files:

```bash
python amusing/source_integral/prepare_pps_work.py /absolute/path/to/pps-work
python amusing/source_integral/prepare_pps_work.py /absolute/path/to/pps-work --execute
```

It prepares three base inputs, six fit inputs using the archived matrices,
three table/generation configs and 40 L0/L2 interval job files. Response tables
and fast tables are not distributed. Build them using the production release's
instructions and the generated `configs/*.json`, with filenames matching each
job. Use `--n-ell 32 --n-eps2 24` for the fast tables. Keep BLAS/OpenMP
single-threaded, use independent output paths and control concurrency to suit
available memory. Table building, generation and refitting are expensive.

To regenerate, for example, the S L2-generated covariance in a separate output:

```bash
python amusing/source_integral/si_gen_joint.py /absolute/path/to/pps-work/configs/S.json amusing/release_20261030_pps/points/S_L2gen.json /absolute/path/to/pps-work/gen/S_L2gen --draws 50 --seed 303 --npilot 100
python amusing/source_integral/si_ppcov.py /absolute/path/to/pps-work/inputs/S_pixbeam_fa amusing/release_20261030_pps/points/S_L2gen.json /absolute/path/to/pps-work/gen/S_L2gen /absolute/path/to/pps-work/reassembled/S_L2gen --shrink 0.35 --floor 0.01
```

Use the corresponding released point files for other cases and seed 404 for
L0gen. The archived points avoid resampling differences; `si_pp_points.py`
provides the original selection algorithm if selecting points anew from
production posterior files. The generator skips existing point files on resume;
do not reuse a generation directory with changed inputs or seeds. The covariance
assembler refuses to overwrite an existing covariance.

Each prepared job can be run with `si_fit.py JOB.json`. After all runs of a
variant exist, post-process and combine intervals, for example:

```bash
SI_MODELS=L0,L2 python amusing/source_integral/post_subset_variant.py /absolute/path/to/pps-work/jobs /absolute/path/to/pps-work/runs /absolute/path/to/pps-work/outputs S_L2gen scal_L2gen
```

The subset helpers retain the production post-processing and physical-prior
normalisation, without changing the four-model production entrypoints. Fitting
requires NumPy, SciPy and dynesty 3.1.0. No full run payloads, generated/noise
maps, raw cubes, private review material or MAUVE data are added by this supplement.
