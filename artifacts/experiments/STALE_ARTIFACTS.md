# Stale development-era experiment artifacts

The existing `baselines/` and `model_selection/` artifacts were generated using historical
train/validation partitions after target-aware EDA. They are retained for provenance but are not
independent or confirmatory and must not supply headline metrics, figures, application performance
claims, behavioral generalization claims, or model selection.

Likewise, existing files under `artifacts/analysis/`, `reports/baselines.md`,
`reports/model_selection.md`, `reports/behavioral_analysis.md`, `reports/error_analysis.md`, and
their generated tables and figures are stale until regenerated from complete nested outer OOF
predictions.

Canonical replacements will be written under `artifacts/experiments/nested_model_selection/` by
an explicitly executed official nested-CV run. No such official run was performed as part of the
protocol implementation.

`data/processed/choices13k_splits.csv` and `artifacts/manifests/split_summary.json` are historical
development-partition records. They are not inputs to the nested protocol.

These legacy artifacts predate centralized run provenance. Their originating Git commit must not
be inferred from the current checkout, and they will not receive retroactively generated
provenance manifests. Only a newly executed eligible run can replace their stale status.
