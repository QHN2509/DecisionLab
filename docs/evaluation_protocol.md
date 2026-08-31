# Evaluation protocol

DecisionLab uses deterministic **5 × 5 nested grouped cross-validation** over the complete
choices13k development dataset. Earlier target-aware EDA means no existing partition is an
untouched confirmatory holdout. Historical train/validation/test labels are development-era
partitions only and are not used by the canonical evaluation.

## Structural grouping

Rows are grouped by a target-free SHA-256 fingerprint of both gamble distributions and their
structural descriptors. `Feedback`, `Block`, `n`, `bRate`, and `bRate_std` are excluded from the
fingerprint. Feedback variants and exact duplicate structures therefore cannot cross a fold
boundary.

`decisionlab-create-folds` writes:

- `data/processed/nested_outer_folds.csv`: one outer test-fold assignment per row;
- `data/processed/nested_inner_folds.csv`: one inner validation-fold assignment for every row in
  each applicable outer training set;
- `artifacts/manifests/nested_cv_summary.json`: hashes, fold counts, and leakage audit results.

## Outer CV

Each structural problem appears in exactly one of five outer test folds. A full pipeline is fit
on the other four folds and predicts that outer test fold once. Concatenating the five held-out
parts produces one outer out-of-fold prediction per development row. Headline performance comes
only from this complete OOF table.

The estimate is leakage-controlled with respect to the frozen tuning and family-selection
procedure. It is not independent of the earlier EDA and must not be described as confirmatory.

## Inner CV and selection

For each outer fold, five grouped inner folds are formed using only its outer training rows. Every
candidate pipeline—including learned preprocessing—is freshly fit on an inner training fold.
Hyperparameters are selected from pooled inner OOF equal-problem MAE. Family selection also uses
inner OOF results only, applying the configured complexity tolerance and deterministic tie rules.
The outer test targets are never available to these decisions.

After selection, the chosen pipeline is refit on all outer training rows and applied once to the
outer test rows. A separate full-development-data fit may be produced for the application after
evaluation; that production fit has no in-sample performance claim.

## Metrics

The primary metric first averages absolute loss within each structural problem, then averages
those problem losses equally. A paired-condition problem therefore has the same total weight as
a singleton problem.

Condition-row metrics and participant-count-weighted metrics are secondary. Uncertainty uses
structural-problem resampling of the complete outer OOF predictions and is conditional on those
predictions.

## Analysis use

Generalization-oriented error, normative, and behavioral prediction analyses must use the outer
OOF prediction artifact. Full-data model sensitivity is descriptive and must be labeled as such.

## Reproducibility

Seeds and fold counts are frozen in `configs/evaluation.json` and
`configs/model_selection.json`. Persisted assignment hashes are recorded in experiment metadata.
Tests verify full row coverage, structural-group isolation, outer-test exclusion from inner CV,
and deterministic assignment.
