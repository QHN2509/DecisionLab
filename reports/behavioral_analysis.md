# Behavioral analysis of selected-model predictions

> **Stale development-era artifact.** This analysis is not based on complete nested outer OOF
> predictions and must not be presented as generalization evidence.

This report interprets the selected Random Forest on the grouped validation split. It describes predictive associations and model sensitivity, not causal effects or participant-level psychological mechanisms. The locked test split was not used.

## Main findings

- The three largest grouped permutation signals were expected value (0.0856), payoff and risk (0.0432), probability structure (0.0345). Values are increases in held-out MAE after jointly permuting each domain; larger values indicate greater predictive reliance.
- The five largest individual permutation signals were `expected_value_difference_b_minus_a_oracle` (0.0646), `loss_probability_difference_b_minus_a_oracle` (0.0256), `maximum_payoff_difference_b_minus_a` (0.0183), `minimum_payoff_difference_b_minus_a` (0.0122), `expected_value_difference_oracle_x_feedback` (0.0067). Correlated and engineered features can divide or duplicate importance, so these should not be read as isolated effects.
- Coherently switching feedback on while updating its EV interaction changed the mean prediction by +0.0063. Switching ambiguity on with its interaction changed it by +0.0408. These are model-sensitivity contrasts over synthetic feature settings, not treatment effects.
- Validation-bin plots show how predictions and observations co-vary with expected-value, probability, payoff, and risk differences. They are observational predictive relationships and retain the oracle limitation under ambiguity.
- Across the lowest-to-highest validation quantile bins, mean prediction changed from 0.289 to 0.716 for B-minus-A expected value and from 0.683 to 0.329 for relative loss probability. Thus, higher B expected value was associated with more predicted B choice, while higher relative B loss probability was associated with less.
- The corresponding endpoint changes were 0.575 to 0.458 for relative payoff dispersion and 0.578 to 0.464 for relative best-payoff probability. Both curves are non-monotone, so endpoint contrasts should not be interpreted as constant slopes.
- Near-equal-EV problems had MAE 0.0886, versus 0.0794 when B had lower EV and 0.0786 when B had higher EV. This is an error concentration, not evidence about causal difficulty.

## Normative benchmark versus observed behavior

Expected-value maximization is used here only as a simple normative benchmark. ‘Strongly favors’ means `|EV_B − EV_A| ≥ 5` payoff units; aggregate humans favor B at `bRate ≥ 0.60` and A at `bRate ≤ 0.40`. These labels describe benchmark deviations, not irrational decisions.

- EV strongly favored A while aggregate humans favored B in 16 validation rows.
- EV strongly favored B while aggregate humans favored A in 25 validation rows.
- 349 rows were highly divided, defined as `|bRate − 0.5| ≤ 0.05`.
- The ML model successfully predicted 8 strong deviations, meaning it predicted the observed side of 0.5 with absolute error at most 0.08. It failed on 18, meaning it predicted the opposite side or had absolute error at least 0.15. Intermediate cases satisfy neither label.

### Selected examples

| Category | Problem | Gamble A | Gamble B | ΔEV B−A | Observed | ML prediction | Error |
|---|---:|---|---|---:|---:|---:|---:|
| strong ev a humans favor b | 7105 | p=1→-9; p=0→-9 | p=0.75→-42; p=0.0078125→42.5; p=0.0390625→43.5; p=0.078125→44.5; p=0.078125→45.5; p=0.0390625→46.5; p=0.0078125→47.5 | -11.25 | 0.600 | 0.518 | 0.082 |
| strong ev a humans favor b | 8914 | p=0.4→32; p=0.6→0 | p=0.8→-16; p=0.2→76 | -10.40 | 0.613 | 0.447 | 0.166 |
| strong ev b humans favor a | 7232 | p=0.1→57; p=0.9→27 | p=0.2→-30; p=0.8→62 | +13.60 | 0.373 | 0.391 | 0.018 |
| strong ev b humans favor a | 4000 | p=0.6→27; p=0.4→19 | p=0.4→-20; p=0.3→72.5; p=0.3→73.5 | +12.00 | 0.358 | 0.449 | 0.091 |
| highly divided | 8384 | p=1→12; p=0→12 | p=0.01→-1; p=0.495→12.5; p=0.495→13.5 | +0.86 | 0.500 | 0.551 | 0.051 |
| highly divided | 5829 | p=1→6; p=0→6 | p=0.5→-50; p=0.25→83; p=0.125→81; p=0.0625→77; p=0.03125→69; p=0.015625→53; p=0.0078125→21; p=0.0078125→-43 | +7.50 | 0.500 | 0.566 | 0.066 |
| ml successfully predicts deviation | 11340 | p=1→-2; p=0→-2 | p=0.4→-40; p=0.01875→10.5; p=0.09375→11.5; p=0.1875→12.5; p=0.1875→13.5; p=0.09375→14.5; p=0.01875→15.5 | -6.20 | 0.662 | 0.656 | 0.006 |
| ml successfully predicts deviation | 1785 | p=1→-10; p=0→-10 | p=0.8→-27; p=0.0015625→19.5; p=0.0109375→20.5; p=0.0328125→21.5; p=0.0546875→22.5; p=0.0546875→23.5; p=0.0328125→24.5; p=0.0109375→25.5; p=0.0015625→26.5 | -7.00 | 0.680 | 0.667 | 0.013 |
| ml fails to predict deviation | 5987 | p=0.5→25; p=0.5→3 | p=0.8→6; p=0.1→77; p=0.1→79 | +6.40 | 0.338 | 0.665 | 0.328 |
| ml fails to predict deviation | 1518 | p=0.4→35; p=0.6→-38 | p=0.25→-42; p=0.375→15; p=0.1875→13; p=0.09375→9; p=0.046875→1; p=0.046875→-15 | +6.55 | 0.350 | 0.636 | 0.286 |

[Normative comparison](figures/normative_vs_observed.png) and [case examples](figures/normative_case_examples.png) visualize these results. The complete case and example tables are saved as machine-readable CSV files.

## Error analysis

| Slice | Rows | MAE | RMSE | Mean bias |
|---|---:|---:|---:|---:|
| feedback: no feedback | 348 | 0.0823 | 0.1031 | +0.0025 |
| feedback: feedback | 1,846 | 0.0806 | 0.1009 | -0.0038 |
| ambiguity: known probabilities | 1,785 | 0.0803 | 0.1003 | -0.0023 |
| ambiguity: ambiguous b | 409 | 0.0837 | 0.1049 | -0.0052 |
| expected value regime: b lower ev | 839 | 0.0794 | 0.0995 | +0.0009 |
| expected value regime: near equal ev | 427 | 0.0886 | 0.1109 | -0.0041 |
| expected value regime: b higher ev | 928 | 0.0786 | 0.0981 | -0.0056 |
| participant count: n at or below 16 | 1,479 | 0.0814 | 0.1018 | -0.0014 |
| participant count: n above 16 | 715 | 0.0799 | 0.1000 | -0.0056 |

Participant-count groups use the training-set median (`n = 16`), not a validation-optimized cutoff. Expected-value regimes use the prespecified ±1 payoff-unit near-tie band.

## Figures

- [Permutation importance](figures/behavioral_permutation_importance.png)
- [Predictive relationships](figures/behavioral_prediction_relationships.png)
- [Condition sensitivity](figures/behavioral_condition_sensitivity.png)
- [Error slices](figures/behavioral_error_slices.png)

## Interpretation limits

- Permutation importance measures loss of predictive accuracy, not causal importance. Correlated features and derived interactions can share signal.
- The condition-sensitivity chart is PDP-like. It updates binary interactions and one-hot sets coherently, but the resulting settings are still synthetic and may be sparsely represented in the data.
- Validation-bin relationships combine model behavior with the observed feature distribution; they do not isolate a variable while holding every confounder fixed.
- Ten features use design-oracle probabilities under ambiguity. Accordingly, ambiguity-related interpretation does not describe a strictly participant-visible prediction setting.
- SHAP was not run: Permutation importance and behaviorally coherent sensitivity analyses answer the prespecified questions without adding the optional SHAP dependency.
- Error differences across slices may reflect sample composition and target measurement noise. They are diagnostics, not evidence that a condition causes higher error.

## Reproducibility

All plotted values are stored under `artifacts/analysis/behavioral/`. The analysis verifies the selected-model artifact hash, exact feature order, grouped validation assignments, and zero test predictions before producing outputs.
