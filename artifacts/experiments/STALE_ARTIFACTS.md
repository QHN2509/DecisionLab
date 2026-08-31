# Stale development-era experiment artifacts

The existing `baselines/` and `model_selection/` artifacts were generated using historical
train/validation partitions after target-aware EDA. They are retained for provenance but are not
independent or confirmatory and must not supply headline metrics, figures, application performance
claims, behavioral generalization claims, or model selection.

The old `reports/baselines.md` and `reports/model_selection.md` files remain historical reports for
those legacy directories. They are not the official nested-CV reports.

The following historical comparison tables are also stale/legacy and are labeled within each file:

- `reports/tables/baseline_comparison.md`
- `reports/tables/baseline_comparison.csv`
- `reports/tables/model_comparison.md`
- `reports/tables/model_comparison.csv`

Current official comparisons are `reports/tables/nested_baselines.csv` and
`reports/tables/nested_model_comparison.csv`.

Official replacements generated from clean commit `1499328a7b7ba1e7ec8ad450a7e345735f560e0a`
are stored under `artifacts/experiments/nested_baselines/` and
`artifacts/experiments/nested_model_selection/`. Regenerated behavioral and error artifacts under
`artifacts/analysis/` consume the complete nested outer-OOF predictions and carry their own
provenance manifests.

`data/processed/choices13k_splits.csv` and `artifacts/manifests/split_summary.json` are historical
development-partition records. They are not inputs to the nested protocol.

The legacy `baselines/` and `model_selection/` artifacts predate centralized run provenance. Their
originating Git commit must not be inferred from the current checkout, and they have not received
retroactively generated provenance manifests.
