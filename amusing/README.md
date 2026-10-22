# AMUSING++ applications: NGC 1084 and NGC 2466

The two observational applications are implemented as ordinary Python modules and shell drivers, not as Jupyter notebooks.

**Current manuscript release:** [release_20261022/README.md](release_20261022/README.md).
Use its final products and explicit reproduction commands, not the historical driver defaults.

* `scripts/amusing_firstlook.py` and `scripts/amusing_ktz_stage.py` prepare the AMUSING++ maps and KTZ inputs.
* `scripts/run_final_piece.py` is the guarded current fit entrypoint; `scripts/paper_single_pointing.py` is its low-level worker, with legacy modes retained for audit. `scripts/prepare_final_responses.py` builds missing response caches without running a sampler.
* `scripts/mean_field_significance.py` evaluates the phase-resolved mean-field significance.
* `scripts/run_v3_ngc1084.sh`, `scripts/run_v3_ngc2466.sh`, and `run_v4_effbeam*.sh` are historical drivers, not current manuscript reproduction commands.
* `../ktzobs/scripts/` contains the finite-window response, source-template, evidence, and robustness modules called by the AMUSING++ drivers.
* `metadata/table3_numbers.json` now records the actual current Table 3: complete intrinsic support, corrected manuscript epsilon prior, normalized joint physical prior and the 2026-10-21 flux-area repair. The superseded table is preserved explicitly under `release_20261022/provenance/`.
* `scripts/ktz_unified_response.py`, `scripts/response_library.py` and `scripts/cnp_fast.py` implement the converged stochastic responses used for the final Table 3 (65 Chebyshev--Lobatto rate-slope nodes, cubic-Hermite separation basis, per-sample kernels); `paper_single_pointing.py --unified-response --support intrinsic --eps-prior manuscript` selects them. `scripts/direct_pairs.py` evaluates the brute-force pair sums used to verify them.
* The finite-map mean-field term is the projected full ensemble mean, `P W mu` with `mu = mu_bar + mu_mod` (`ktz_exact_window.exact_mean_field_response`), entered once as a rank-one term.
* `tests/` contains convergence, prior-support, epsilon-prior and synthetic flux-area regression tests. `scripts/check_final_release.py` checks final evidence recombination, posterior bounds, area factors, covariance assembly and manuscript products without fitting.
* `production/` publishes the lean production workers, with repository-relative imports; `scripts/plot_from_archive_fa.py` redraws the actual current Figures 7/8 from `release_20261022/figure_inputs/` without loading a likelihood or sampler.

## Conditional physical-prior evidences

The documented independent densities define a base prior. The sigma, kappa, age and source-amplitude bounds condition that prior jointly. If `F_M` is the retained base-prior probability, the reported evidence is `Z_box / F_M`. The likelihood remains `-chi2/2` on the legal domain; this normalisation is not an epsilon weight or an extra likelihood term. The conditioned marginal priors need not retain the base density's individual forms.

`../ktzobs/scripts/physical_prior.py` integrates the beta and beta-times-amplitude constraints analytically and the remaining dimensions by quadrature. `release_20261022/metadata/` records current evidences, physical-prior masses and corrected per-map inputs. `metadata/prior_normalization_20261018.json` is the historical pre-flux-area regression fixture, not a source for current Table 3 numbers. No large response arrays are needed to reproduce evidence conditioning from the released piece summaries.

Run `OPENBLAS_NUM_THREADS=1 python amusing/tests/physical_prior_test.py` from the repository root for the independent direct-prior-draw and split-weight regression checks.

For an archived combined split result, run `python amusing/scripts/normalize_physical_evidence.py SUMMARY.json MORPHOLOGY.json OUTPUT.json`. Use a new output filename; the input is preserved. New `paper_single_pointing.py --eps-prior manuscript` runs report conditional physical evidences by default and use an `_physical` output suffix; `--evidence-prior box` retains the old convention for audit comparisons. `combine_split_evidence.py ... --physical-prior` uses existing `_physical` pieces when present, otherwise archived box pieces, and writes a separately suffixed combined result. It recombines box evidences first and then applies the global normalisation. Equivalently the conditional piece prior weights are `f_i F_Mi / F_M`, not the unconditioned log-interval masses `f_i`.

The public repository contains analysis code, current manuscript products and resampled posterior archives for the final cases. Large intermediate response arrays, raw sampler runs, raw FITS and most logs are omitted; the released figures and evidence combination do not require them.

The single-pointing figures show L0 conditional on its own highest-posterior-mass transport-scale interval. The L1, L1n and L2 curves and the L2 decomposition share L2's highest-posterior-mass interval: [0.08, 0.22] kpc for NGC 2466 S and [0.08, 0.5] kpc for NGC 1084 D16. This shared interval need not dominate the posterior of L1 or L1n. The shaded bands and transport scales in the figure legends are conditional summaries; the evidences and tabulated posterior parameter summaries combine all computational intervals. The physical-prior normalisation changes neither the within-model posterior nor these figure files.

In L1n the identifiable clustering amplitude is the product `beta_src * A_src`; the implementation fixes `beta_src = 1` as a convention. Its amplitude cut is therefore a cut on that product. In L2, `beta_src` is sampled and the same numerical cut applies to the derived `A_src`. The retained source-amplitude prior mass is 5/13 in both models.
