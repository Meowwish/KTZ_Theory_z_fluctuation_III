# Single-element validation entry: 15 November 2026

The active producer is `KTZ_validation.py`, with its matching Notebook. It now
contains only the single-element experiment; its default output tag is
`single_element`, and its metrics filename is `validation_metrics_single_element.json`.

The physical parameters, original seed 42 and calculation cells are unchanged.
The ten-field archive, prediction table and single-element summary arrays are
copied from the existing 4 November combined release. Replaying the independent
entry reproduces those arrays exactly and gives full-window RMSE
0.004563522058183998. No observational fit is rerun or changed by this split.

The cross-element entry, working manuscript, figures and cached matrices are
maintained at <https://github.com/Meowwish/Cross-correlation>. Earlier combined
products in this repository are retained as historical snapshots, rather than
deleted. Cross-element helper snapshots that read the former combined producer
are historical; use the migrated helpers in that repository for current execution.

`SHA256SUMS` records this payload. `manuscript_state.json` records the corresponding
manuscript fingerprint, not its source. This release adds no manuscript text,
raw observational products, proprietary data, machine logs or internal reviews.
