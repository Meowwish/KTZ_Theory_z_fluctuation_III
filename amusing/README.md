# AMUSING++ applications: NGC 1084 and NGC 2466

The two observational applications are implemented as ordinary Python modules and shell drivers, not as Jupyter notebooks.

* `scripts/amusing_firstlook.py` and `scripts/amusing_ktz_stage.py` prepare the AMUSING++ maps and KTZ inputs.
* `scripts/paper_single_pointing.py` performs the single-pointing fits used in the manuscript.
* `scripts/mean_field_significance.py` evaluates the phase-resolved mean-field significance.
* `scripts/run_v3_ngc1084.sh`, `scripts/run_v3_ngc2466.sh`, and the `run_v4_effbeam*.sh` drivers reproduce the fitted variants and effective-beam results.
* `../ktzobs/scripts/` contains the finite-window response, source-template, evidence, and robustness modules called by the AMUSING++ drivers.
* `metadata/table3_numbers.json` records the one-decimal values reported in Table 3.

The public repository contains the analysis code and compact manuscript products. Large intermediate response arrays, posterior samples, and logs are intentionally omitted.
