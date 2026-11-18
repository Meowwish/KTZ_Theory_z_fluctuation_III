# Finite-map component labels: 18 November 2026

The active single-element script and matching notebook label the finite-map
components as bold S_U, S_T, S_M and S_D. The observational figure producer
`amusing/source_integral/si_plot_fig78.py` uses the same labels. Continuous-field
quantities retain C. No physical expression, cached prediction, fitted curve,
posterior or primary evidence has changed.

`manuscript_state.json` links the manuscript fingerprint, active producers and
four replacement figures. `figure_checks.json` records the independently
verified numerical fingerprints and pixel comparisons. `SHA256SUMS.txt` records
these files and the matching active sources and figures from the repository root.

The cached validation replay used Python 3.13, Matplotlib 3.11.2 and DejaVu Sans.
The old validation PNGs were reproduced byte for byte before the label changes;
the resulting pixel differences are confined to their legends. Existing caches
are in `validation/outputs_single_element/`; run the active validation entry from
the repository root with `KTZ_RUN_TAG=single_element` to reuse them.

For observational redraw commands, see
`amusing/release_20261026/README.md`. The producer now selects DejaVu Sans
explicitly; the delivered PDFs used Matplotlib 3.11.2 and contourpy 1.4.0.
Against the preceding PDFs, the unpatched replay has identical page drawing
coordinates except for contour close-path markers (2 and 9 operators), affecting
2 and 14 pixels at 150 dpi in the map panels. All curve, input and interval
fingerprints match. Within the replay environment, relabeling changes no plotted
array and changes pixels only in the three right-panel legend labels.

Earlier release manifests are historical fingerprints of their respective Git
versions; this manifest records the current producers. The full manuscript,
raw/private data and internal review reports are not included here.
