# Release verification

## Final reader edits, 24 October 2026

The saved source compiled successfully in the built-in desktop editor after
adding a short caption for reference metadata alongside the multi-paragraph
Figure 7 caption. The free-parameter table moved intact into Section 4.3, after
the evidence table; both table numbers remain unchanged. The NGC 2466 opening
result text was split without changing its words, and one adjacent redundant
map-variance versus bin-share sentence was removed.

All 126 displayed mathematical environments (124 equation/align blocks plus
2 eqnarray blocks), all 5 table environments and all 6 tabular blocks are
byte-identical in content. Seven of eight figure environments are unchanged;
only Figure 7's caption was reorganised. All eight figure files, all four
numerical-validation files and the bibliography retain their previous hashes.
All 179 labels and 50 citation keys remain; internal references resolve.
The Abstract and Acknowledgements are unchanged. Outside the moved table,
Figure 7 caption and removed duplicate sentence, the source is identical after
whitespace normalisation. No scientific calculation or Overleaf operation is
part of this update.

## Historical readability synthesis, 24 October 2026

The final source compiled successfully in the built-in desktop editor. All 124
displayed equation blocks are byte-identical to the preceding manuscript, as
are all tabular blocks except the Table 1 symbol entry. Table 3 remains at one
decimal. All original labels remain; two labels locate the relocated Appendix A
checks. Internal references resolve and the 50 citation keys are unchanged.

The cell-response applicability criteria, exact-cell comparison and covariance
diagnostics were moved without changing their numerical text. The newly stated
mean-filter relation is restricted to zero delay, equal injection width and
common diffusion coefficient/time horizon; it follows by changing the source-age
variable. Independent quadratures agree to within 6e-17 in the tested examples.
This is a definition check, not a new fit or physical validation.

All 8 figure files and 4 numerical-validation file hashes are unchanged. The
reported scientific results, priors, covariances, masks and figures are retained.
No unverified template-filtering percentage is introduced. No Overleaf update,
project cleanup, simulation or fit is part of this revision.

## Historical manuscript binding and plot-only cleanup, 23 October 2026

The current `metadata/manuscript_state.json` records the checked manuscript
revision and its bibliography, figures and numerical-validation code hashes.
Three verified text corrections fix a misplaced introduction, the NGC 2466 S
L2 association-length upper uncertainty (+0.6 kpc from archived quantiles), and
the general template-mode notation. No new diagnostic or scientific claim was
added. The subsequent Figure 4/5 cleanup changes only the third matrix-panel
title and removes the no-C_AB,NP scatter comparison and its caption clause.

Both figures were redrawn from `cross_results.npz`, without entering simulation
or covariance-calculation blocks. Matplotlib 3.10.9 control redraws match the
preceding images pixel for pixel. Fingerprints of retained matrix values,
colour scales, scatter coordinates, error bars, channel groups, axis limits and
the 1:1 line agree before and after; the no-CNP numerical results and all quoted
validation metrics are retained. Python and notebook plotting cells agree.
The final saved manuscript compiled successfully in the desktop editor.
These checks do not claim a new Overleaf compilation or synchronization.

## Historical readability and plotting-label synchronization, 22 October 2026

The following records the 22 October Overleaf revision, whose source hash was
`4a4dfb14d664a0e548222b955b1d355d9cc7bdb67f9d1162afda9fd152b70392`.
The corresponding Figure 5 axes use full-window `Xi_AB,Omega` notation, and
Figures 7/8 use `L2 total` instead of `sum (L2)`, with slightly larger legends.
Python and notebook plotting-source labels agree. Plot-only redraws preserve
every scientific curve, errorbar, band, map and contour array; the archived
numerical products and all Table 3 values are unchanged.

The manuscript's 133 display-equation environments and Table 3 data were
compared byte-for-byte before and after the readability revision. Its 158 labels,
333 references and 50 citation keys resolve. The downloaded Overleaf project
matches local source, bibliography and all three replaced figure files. The
31-page document compiles independently and on Overleaf without errors,
undefined references/citations, missing figures or overfull boxes; two inherited
font-warning messages remain. Text changes retain Reviewing history.

The repository publishes a version hash and a change summary, not the manuscript
source or PDF. No raw or proprietary data are added. The scientific release
checks below are rerun against the updated manifest without running fits.

## Original scientific-release checks

The following records the earlier flux-area release. Its statements about
unchanged plotting code and the exact prior artwork refer to that release,
before the label-only synchronization described above.

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
- Overleaf source was read back in full and exactly matched the manuscript SHA
  recorded for that historical release. Only four local wording replacements in
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
