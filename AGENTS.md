# DecisionLab Agent Guide

## Objective

DecisionLab predicts and analyzes aggregate human choice under risk and uncertainty using the choices13k dataset. Prioritize rigorous, interpretable, reproducible ML over application or web complexity. Begin with simple baselines before adding complex models.

## Target

`bRate` is the aggregate rate at which participants selected Gamble B for a decision problem. It is a continuous target in `[0, 1]`, computed by averaging participant-level selection rates across repeated trials. It is not an individual binary choice, utility estimate, or causal outcome. Treat each decision problem as the primary evaluation unit; use `n`-weighted metrics only as clearly labeled sensitivity analyses.

## Leakage Rules

- Split before fitting imputers, encoders, scalers, feature selection, calibration, or models.
- Group all rows for the same underlying problem across partitions. Use both `problem` and an exact structural fingerprint; feedback variants and duplicates must not cross splits.
- Never use the locked test set for EDA, feature design, tuning, model selection, calibration, or subgroup definition.
- Do not use target-derived features unless they are constructed strictly out of fold.
- Treat `n` as measurement metadata, not a default predictor.
- Never use `bRate_std` as a predictor; it is computed from the same participant responses as the target.
- Keep participant-visible features separate from oracle/design features, especially probabilities hidden under ambiguity.
- Join problem JSON using its documented row-index key and validate alignment; do not assume it matches `problem`.
- Persist and test split assignments. Report any dependence or leakage risk that cannot be eliminated.

## Coding Conventions

- Support Python 3.12 and use a `src/decisionlab` package layout.
- Use type hints, focused functions, docstrings for public APIs, and deterministic behavior where possible.
- Keep preprocessing and modeling in tested modules. Notebooks are thin analysis clients, not authoritative pipelines.
- Store configuration separately from code; avoid hard-coded paths, seeds, thresholds, and reported results.
- Keep raw data immutable and out of Git. Preserve source revision and checksums.
- Add tests for data contracts, joins, feature calculations, split isolation, and evaluation logic.
- Do not implement application or web features until the research pipeline is approved and stable.

## Repository Structure

```text
configs/                 experiment and data configuration
data/{raw,interim,processed}/
docs/                    protocol, data dictionary, estimand, limitations
src/decisionlab/         data, features, splitting, models, evaluation, analysis
notebooks/               numbered exploratory and reporting notebooks
scripts/                 thin command entry points
tests/                   tests mirroring package responsibilities
reports/{figures,tables}/
artifacts/manifests/     run metadata, hashes, seeds, and versions
```

## Canonical Checks

Preserve these repository-level commands:

```bash
python -m pytest
ruff check .
ruff format --check .
```

Do not claim these checks pass unless they were executed successfully in the current worktree.

## Experiment Reproducibility

- Pin dependencies and record Python/package versions, random seeds, source-data checksum, configuration, split ID, code revision, and run timestamp.
- Use grouped cross-validation on development data and evaluate the locked test set only after model selection.
- Compare models on identical folds and report uncertainty using problem-grouped resampling where appropriate.
- Use unweighted MAE as the primary validation metric. Treat participant-count-weighted metrics as sensitivity analyses, not replacements for problem-level evaluation.
- Generate tables and figures from saved run artifacts. Never hand-enter or invent metrics.
- Every reported metric must be traceable to an actually executed experiment, including the dataset hash, split, configuration, and output artifact.
- Clearly distinguish primary unweighted problem-level metrics from weighted or stress-test results.

## Research Claims and Documentation

- Describe feature effects, subgroup differences, SHAP values, and model behavior as associations unless a dedicated analysis establishes a defensible causal estimand and identification strategy.
- Do not equate disagreement with expected value to irrationality, especially under ambiguity or feedback.
- State the population, prediction setting, split design, uncertainty, limitations, and whether features were participant-visible or oracle.
- Update the README, relevant files in `docs/`, configuration examples, and tests whenever data behavior, target semantics, interfaces, commands, metrics, or experimental behavior changes.
