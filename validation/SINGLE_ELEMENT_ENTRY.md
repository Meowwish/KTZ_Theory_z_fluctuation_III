# Single-element validation entry

`KTZ_validation.ipynb` now contains only the single-element experiment. `KTZ_validation.py` is its matching executable export. The physical parameters, seed 42 and computation cells are unchanged; the default output tag is `single_element` and the metrics file is `validation_metrics_single_element.json`.

The cross-element manuscript, independent Notebook setup, shared code snapshot and cached matrices are maintained at <https://github.com/Meowwish/Cross-correlation>. Historical validation outputs and diagnostic scripts remain for provenance; they are not part of the active single-element entry point.

The split is structural. It does not rerun observational fits, change the masks or priors, or regenerate manuscript figures from different inputs.
