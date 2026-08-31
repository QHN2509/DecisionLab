# Model comparison and selection

> **Stale development-era artifact.** This report predates nested grouped CV and must not be used
> for headline performance or model selection. Regenerate from complete outer OOF predictions.

This report was generated from the grouped training/validation experiment. The locked test partition was neither predicted nor evaluated.

## Comparison

| Model | Selected | Validation MAE | RMSE | R² | n-weighted MAE | n-weighted RMSE | CV MAE ± SD | Complexity |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| `random_forest` | yes | 0.0809 | 0.1012 | 0.7910 | 0.0807 | 0.1010 | 0.0825 ± 0.0015 | 2 |
| `gradient_boosting` | no | 0.0831 | 0.1047 | 0.7762 | 0.0828 | 0.1045 | 0.0843 ± 0.0017 | 3 |

Unweighted MAE is primary. Participant-count-weighted metrics are sensitivity analyses only.

## Selection

**Selected model: `random_forest`.** The best setting was first chosen within each family using mean five-fold grouped training MAE. On validation, models within 0.005 MAE of the best were treated as practically tied; the lower-complexity model then won, followed by lower CV MAE and CV variability as tie-breakers.

The selected model has validation MAE 0.0809 and grouped-CV MAE 0.0825 ± 0.0015. This balances held-out performance, fold stability, and complexity rather than automatically preferring the most flexible candidate.

## Interpretability and complexity

- **random_forest:** moderate: impurity importances and partial dependence are available Complexity rank: 2.
- **gradient_boosting:** moderate-low: sequential trees support global importance but interact nonlinearly Complexity rank: 3.

XGBoost was not added: the configuration records that its large optional dependency and overlap with sklearn gradient boosting were not justified for this modest comparison. This can be revisited if these ensembles leave a material performance gap.

Feature importance values describe predictive use by the fitted ensemble, not causal effects. The feature set includes design-oracle probabilities under ambiguity, so performance does not represent a strictly participant-visible deployment setting.

## Saved artifacts

- `artifacts/experiments/model_selection/selected_pipeline.joblib`
- `artifacts/experiments/model_selection/selected_model.joblib`
- `artifacts/experiments/model_selection/preprocessing_pipeline.joblib`
- `artifacts/experiments/model_selection/metrics.json`
- `artifacts/experiments/model_selection/tuning_results.csv`
- `artifacts/experiments/model_selection/feature_names.json`
- `artifacts/experiments/model_selection/experiment_config.json`
