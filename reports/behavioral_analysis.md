# Behavioral analysis of selected-model predictions

This report interprets complete nested outer out-of-fold predictions. It describes predictive associations and model sensitivity, not causal effects or participant-level psychological mechanisms. No confirmatory holdout is claimed.

## Main findings

- Grouped coherent domain perturbations ranked as complete problem block (0.1575; 0.1555–0.1594), gamble structure (0.1572; 0.1553–0.1591), information and experience conditions (0.0090; 0.0084–0.0095). Parenthesized values are equal-problem outer OOF MAE increases and 95% structural-group bootstrap intervals. These overlapping domains describe model reliance, not isolated feature effects.
- Coherent primitive/dependency-family perturbations ranked as `gamble_b_distribution_and_lottery` (0.1847; 0.1825–0.1870), `gamble_a_distribution` (0.1662; 0.1638–0.1684), `ambiguity_condition` (0.0083; 0.0078–0.0088), `feedback_condition_block` (0.0008; 0.0006–0.0009), `correlation_condition` (-0.0000; -0.0000–-0.0000). Every dependent engineered feature was rebuilt through the production feature pipeline; these are family-level reliance estimates, not individual-feature effects.
- Coherently switching feedback on while updating its EV interaction changed the mean prediction by +0.0055. Switching ambiguity on with its interaction changed it by +0.0427. These are model-sensitivity contrasts over synthetic feature settings, not treatment effects.
- Validation-bin plots show how predictions and observations co-vary with expected-value, probability, payoff, and risk differences. They are observational predictive relationships and retain the oracle limitation under ambiguity.
- Across the lowest-to-highest validation quantile bins, mean prediction changed from 0.303 to 0.724 for B-minus-A expected value and from 0.683 to 0.343 for relative loss probability. Thus, higher B expected value was associated with more predicted B choice, while higher relative B loss probability was associated with less.
- The corresponding endpoint changes were 0.562 to 0.460 for relative payoff dispersion and 0.590 to 0.455 for relative best-payoff probability. Both curves are non-monotone, so endpoint contrasts should not be interpreted as constant slopes.
- Near-equal-EV problems had MAE 0.0892, versus 0.0784 when B had lower EV and 0.0772 when B had higher EV. This is an error concentration, not evidence about causal difficulty.

## Normative benchmark versus observed behavior

Expected-value maximization is used here only as a simple normative benchmark. ‘Strongly favors’ means `|EV_B − EV_A| ≥ 5` payoff units; aggregate humans favor B at `bRate ≥ 0.60` and A at `bRate ≤ 0.40`. These labels describe benchmark deviations, not irrational decisions.

- EV strongly favored A while aggregate humans favored B in 156 outer OOF rows.
- EV strongly favored B while aggregate humans favored A in 133 outer OOF rows.
- 2287 rows were highly divided, defined as `|bRate − 0.5| ≤ 0.05`.
- The ML model successfully predicted 74 strong deviations, meaning it predicted the observed side of 0.5 with absolute error at most 0.08. It failed on 134, meaning it predicted the opposite side or had absolute error at least 0.15. Intermediate cases satisfy neither label.

### Selected examples

| Category | Problem | Gamble A | Gamble B | ΔEV B−A | Observed | ML prediction | Error |
|---|---:|---|---|---:|---:|---:|---:|
| strong ev a humans favor b | 10989 | p=0.95→-4; p=0.05→-22 | p=0.8→-27; p=0.00625→15.5; p=0.03125→16.5; p=0.0625→17.5; p=0.0625→18.5; p=0.03125→19.5; p=0.00625→20.5 | -13.10 | 0.725 | 0.492 | 0.233 |
| strong ev a humans favor b | 12534 | p=0.2→-1; p=0.8→-5 | p=0.99→-17; p=0.0003125→22.5; p=0.0015625→23.5; p=0.003125→24.5; p=0.003125→25.5; p=0.0015625→26.5; p=0.0003125→27.5 | -12.38 | 0.688 | 0.631 | 0.057 |
| strong ev b humans favor a | 7232 | p=0.1→57; p=0.9→27 | p=0.2→-30; p=0.8→62 | +13.60 | 0.373 | 0.407 | 0.033 |
| strong ev b humans favor a | 614 | p=0.95→9; p=0.05→-46 | p=0.75→-7; p=0.25→94 | +12.00 | 0.360 | 0.490 | 0.130 |
| highly divided | 625 | p=1→16; p=0→16 | p=0.8→4; p=0.025→38.5; p=0.075→39.5; p=0.075→40.5; p=0.025→41.5 | -4.80 | 0.500 | 0.544 | 0.044 |
| highly divided | 3744 | p=1→-5; p=0→-5 | p=0.9→-14; p=0.1→13 | -6.30 | 0.500 | 0.449 | 0.051 |
| ml successfully predicts deviation | 8873 | p=1→-4; p=0→-4 | p=0.9→-20; p=0.1→84 | -5.60 | 0.675 | 0.669 | 0.006 |
| ml successfully predicts deviation | 11667 | p=1→-1; p=0→-1 | p=0.99→-8; p=0.01→68 | -6.24 | 0.680 | 0.686 | 0.006 |
| ml fails to predict deviation | 6716 | p=0.25→73; p=0.75→-32 | p=0.75→-39; p=0.125→59; p=0.0625→61; p=0.03125→65; p=0.03125→73 | -8.00 | 0.859 | 0.421 | 0.437 |
| ml fails to predict deviation | 4945 | p=1→0; p=0→0 | p=0.99→-10; p=0.005→51; p=0.0025→49; p=0.00125→45; p=0.000625→37; p=0.000625→21 | -9.43 | 0.875 | 0.505 | 0.370 |

[Normative comparison](figures/normative_vs_observed.png) and [case examples](figures/normative_case_examples.png) visualize these results. The complete case and example tables are saved as machine-readable CSV files.

## Error analysis

| Slice | Rows | Groups | Problem-group MAE | Condition-row MAE |
|---|---:|---:|---:|---:|
| feedback: no feedback | 2,380 | 2,380 | 0.0855 | 0.0855 |
| feedback: feedback | 12,188 | 12,188 | 0.0795 | 0.0795 |
| ambiguity: known probabilities | 11,759 | 10,493 | 0.0788 | 0.0793 |
| ambiguity: ambiguous b | 2,809 | 2,513 | 0.0849 | 0.0858 |
| expected value regime: b lower ev | 5,844 | 5,222 | 0.0784 | 0.0788 |
| expected value regime: near equal ev | 2,816 | 2,532 | 0.0892 | 0.0899 |
| expected value regime: b higher ev | 5,908 | 5,252 | 0.0772 | 0.0778 |
| participant count: n at or below 16 | 9,881 | 8,830 | 0.0817 | 0.0822 |
| participant count: n above 16 | 4,687 | 4,639 | 0.0770 | 0.0769 |

Participant-count groups use the complete development-data median (`n = 16`), not a validation-optimized cutoff. Expected-value regimes use the prespecified ±1 payoff-unit near-tie band.

## Figures

- [Permutation importance](figures/behavioral_permutation_importance.png)
- [Predictive relationships](figures/behavioral_prediction_relationships.png)
- [Condition sensitivity](figures/behavioral_condition_sensitivity.png)
- [Error slices](figures/behavioral_error_slices.png)

## Interpretation limits

- Permutation importance measures loss of predictive accuracy, not causal importance. Families overlap and must not be added or interpreted as mutually exclusive effects.
- Donor structural groups are drawn only from the same outer fold. Complete groups move together, engineered dependencies are recomputed, and uncertainty resamples whole structural groups.
- The feedback block perturbation preserves paired rows. Its estimate may be driven mainly by singleton groups because paired feedback/no-feedback blocks contain the same condition pattern.
- The condition-sensitivity chart is PDP-like. It updates feedback and ambiguity interactions and the correlation one-hot family coherently, but the resulting settings are still synthetic and may be sparsely represented in the data. Lottery shape is excluded because changing its encoding without rebuilding Gamble B's outcome distribution would be internally inconsistent.
- Validation-bin relationships combine model behavior with the observed feature distribution; they do not isolate a variable while holding every confounder fixed.
- Ten features use design-oracle probabilities under ambiguity. Accordingly, ambiguity-related interpretation does not describe a strictly participant-visible prediction setting.
- SHAP was not run: Permutation importance and behaviorally coherent sensitivity analyses answer the prespecified questions without adding the optional SHAP dependency.
- Error differences across slices may reflect sample composition and target measurement noise. They are diagnostics, not evidence that a condition causes higher error.

## Reproducibility

All plotted values are stored under `artifacts/analysis/behavioral/`. The analysis verifies every outer-fold pipeline hash, exact feature order, complete OOF coverage, and structural-group isolation before producing outputs.
