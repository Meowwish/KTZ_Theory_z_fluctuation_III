# Release verification, 22 October 2026

No sampler, simulation or large response calculation was run for this release.
It publishes the previously accepted flux-area-corrected results and makes the
current inputs, products and explicit reproduction entrypoints available together.

- Ten null/EB cases and 128 posterior model/piece archives were checked. Box
  evidences recombine with log-interval prior masses, then the global retained
  physical-prior mass gives the final evidence. All current Table 3 values agree;
  the manuscript retains one decimal. The 0.5-kpc boundary is computational.
- All ten final stage-input mappings pass SHA-256 checks. The six baseline
  morphology/covariance hashes match the independent prior-sensitivity records.
  Current D16 metadata use the final effective beam, not the parent beam.
- Valid-cell counts are 4,806 (NGC 2466 S), 13,968 (D16) and 10,034
  (NGC 1084 D16). Their all-pair unordered counts are 11,546,415, 97,545,528
  and 50,335,561. These support the corrected approximate pair counts in the text.
- The synthetic WCS/flux-area check passes, including rotated TAN WCS,
  deprojection and smoothing/resampling checks. Recovered integrated-flux ratios
  range from 0.99992 to 1.00044 in the recorded test configurations.
- All three EB matrices equal the released final stage matrices. Independent
  component assembly reproduces shrinkage plus the diagonal floor without adding
  measurement covariance twice. They are positive definite. Full-bin effective
  counts are 8.91351 (S), 6.73765 (D16) and 5.27225 (NGC 1084).
- The physical-prior regression test passes 400 assertions, including independent
  direct prior draws, split weights and idempotence. Epsilon inverse-CDF checks
  also pass. No likelihood or posterior was changed for publication.
- Figures 7/8 redraw without a likelihood or sampler. Every scientific line,
  band, error-bar, image-array and text fingerprint matches the archived render.
  Only contour-geometry fingerprints differ with the local NumPy 2.5.4 build:
  9 and 18 raster pixels differ, respectively, in the 1,470 x 495 PNGs. The exact
  accepted PDF files, not these redraws, are published and match Overleaf hashes.
  Figure 1 redraw is pixel-identical. All six numerical-validation PNGs are
  pixel-identical to the prior GitHub versions; five were byte-synchronized to
  Overleaf to remove encoding/metadata differences only. Validation code is unchanged.
- Python syntax and all five explicit current fit-command configurations pass.
  Input-copy tests pass checksum, repeat-copy idempotence and overwrite refusal.
  These checks do not claim that a fresh end-to-end fit was executed.
- Overleaf source was read back in full and exactly matches the manuscript SHA
  in `metadata/manuscript_state.json`. Only four local wording replacements in
  two paragraphs were made in Reviewing mode: the map-specific pair counts and
  the actual scope of the original direct-pair tests. The Abstract,
  Acknowledgements, numerical results and other scientific content are unchanged.
- The updated project compiles independently in the local document compiler.
  Actual Overleaf compilation is 31 pages with zero errors, undefined
  references/citations or missing figures, and no overfull boxes. Existing font,
  underfull-box and PDF-destination warnings remain; no claim is made that every
  raw-log warning has been removed. The changed scientific pages were rendered
  and inspected for clipping, caption/figure continuity and table layout.

## Limits preserved from the manuscript

The original direct-pair reports cover reference-run posterior points, inter-node
points and selected domain edges for four reference maps, not an exhaustive
test of final flux-area-corrected posteriors. NGC 1084 EB and S were not separately
direct-pair tested. Their actual reported maxima support the stated reference
test accuracy. No new precision claim is introduced here.

The EB covariance is a disclosed sensitivity proxy, not a fully generative
covariance analysis. The prior-sensitivity experiment uses one seed and holds
its EB matrices fixed. These limitations remain explicit.

No MAUVE data, raw FITS, credentials, raw sampler checkpoints or proprietary
data are added or changed. The baseline download, updated source and compile log are retained locally;
the manuscript itself is not newly published in this repository.
