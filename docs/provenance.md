# Experiment provenance

EDA, baseline, nested model-selection, behavioral-analysis, error-analysis, and application-metadata
runs automatically create centralized provenance manifests. Provenance is captured before any
generated file is changed and records:

- the exact DecisionLab Git commit and pre-run dirty/clean state;
- the pinned choices13k source revision and verified data-file checksums;
- individual and combined hashes for every configuration file, explicit run arguments, and a
  hash over the complete configuration record;
- runtime Python/platform details plus individual and combined environment hashes covering
  `pyproject.toml` and `requirements-dev.lock`;
- the nested-fold specification identifier and persisted fold-artifact hashes;
- hashes of the experiment entry module and all transitively imported DecisionLab modules;
- hashes of every declared generated output, including predictions, metrics, models, reports,
  feature tables, and fold-specific pipelines where applicable.

Behavioral and error analyses additionally hash their upstream model provenance, OOF predictions,
feature contract, engineered table, and (where used) fold pipelines. They refuse to run if the
regenerated model provenance is absent, so legacy model artifacts cannot silently become current
analysis inputs.

The provenance file cannot contain its own stable hash, so this exclusion is explicit in the
manifest.

## Dirty repositories

Official runners refuse a dirty repository by default. `--allow-dirty` exists for development
smoke runs only: it emits a runtime warning and records
`run_eligibility = non_official_dirty_worktree`. Such a run must not replace official results.

The tracked manifests are written to:

- `artifacts/manifests/eda_provenance.json`
- `artifacts/experiments/nested_baselines/provenance.json`
- `artifacts/experiments/nested_model_selection/provenance.json`
- `artifacts/analysis/behavioral/provenance.json`
- `artifacts/analysis/errors/provenance.json`
- `artifacts/application/provenance.json`

## Feature lineage

The standalone `decisionlab-build-features` command can emit
`artifacts/manifests/feature_build_provenance.json`, but DecisionLab does not claim that an
official standalone feature-build manifest is currently tracked. Canonical feature lineage is
instead recorded by the official nested model-experiment provenance. That manifest hashes the
validated source data, complete configuration, transitive feature-engineering implementation,
engineered feature table, feature-name contract, folds, and model outputs. Downstream behavioral
and application provenance verifies those upstream hashes. Existing feature outputs are not
retroactively attributed to a standalone manifest they never had.

## Legacy artifacts

Artifacts generated before this provenance system remain legacy/stale. Their originating commit
must not be inferred from the current checkout, and no new provenance manifest is created for them
retroactively. They become current only after regeneration by an eligible run.
