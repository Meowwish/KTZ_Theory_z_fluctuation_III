# KTZ Paper III: validation and observational applications

The current manuscript products are indexed in
[`amusing/release_20261022/README.md`](amusing/release_20261022/README.md).
That release incorporates the integrated H-alpha flux-area correction, the full
intrinsic transport support and the normalized manuscript/physical priors.
It supersedes the older observational summaries, not the numerical-validation
experiment. No fit was rerun to make the release.

- Numerical validation: `KTZ_validation.ipynb` and `KTZ_validation.py`;
  [execution and Figure 1 redraw instructions](validation/README.md).
- Observations: ordinary Python scripts, not notebooks;
  [current workflow](amusing/README.md).
- Current evidence table: `amusing/metadata/table3_numbers.json`.
- Figures 7/8: `figs/ngc2466_single_pointing.pdf` and
  `figs/ngc1084_single_pointing.pdf`; they can be redrawn from the released arrays
  without running a model or sampler.

Historical files remain for provenance and must not be mistaken for current
manuscript inputs. In particular, the old `run_v3*` and `run_v4*` scripts are not
the current release entrypoint. This update does not add or modify MAUVE data.
