# Systematic model error analysis

> **Stale development-era artifact.** This analysis is not based on complete nested outer OOF
> predictions and must not be presented as generalization evidence.

This report analyzes the selected Random Forest on the grouped validation split. The locked test split was not inspected. Differences are descriptive predictive patterns, not causal effects or explanations of participant behavior.

## Summary

Overall validation MAE was **0.0809**, RMSE was **0.1012**, and mean bias was **-0.0028** across 2,194 rows.
The absolute EV threshold for an extreme case was **8.550**, calculated as the configured 90% quantile using training predictors only.

The five regimes with the largest descriptive MAE were: expected value regime/near equal ev (0.0886), lottery shape b/left skewed (0.0844), ambiguity/ambiguous b (0.0837), feedback/no feedback (0.0823), aggregate division/leaning (0.0822). Overlapping bootstrap intervals mean small differences should not be treated as clear population differences.

Regimes above the overall error on both MAE and RMSE were: feedback/no feedback, ambiguity/ambiguous b, expected value regime/near equal ev, aggregate division/leaning, participant count/n at or below 16, lottery shape b/single outcome, lottery shape b/left skewed. These overlapping slices are descriptive diagnostics, not independent effects.

## Largest absolute errors

| Problem | Conditions | Gamble A | Gamble B | Actual | Predicted | Abs. error | EV benchmark | ΔEV B−A |
|---:|---|---|---|---:|---:|---:|:---:|---:|
| 1014 | feedback, known | p=1→0; p=0→0 | p=0.05→-20; p=0.95→1 | 0.860 | 0.397 | 0.463 | A | -0.05 |
| 5987 | feedback, known | p=0.5→25; p=0.5→3 | p=0.8→6; p=0.1→77; p=0.1→79 | 0.338 | 0.665 | 0.328 | B | +6.40 |
| 6868 | feedback, known | p=0.1→25; p=0.9→7 | p=0.4→7; p=0.6→8 | 0.579 | 0.268 | 0.311 | A | -1.20 |
| 3535 | feedback, known | p=1→23; p=0→23 | p=0.8→14; p=0.2→76 | 0.289 | 0.583 | 0.294 | B | +3.40 |
| 3256 | feedback, ambiguous | p=0.95→26; p=0.05→-9 | p=0.8→3; p=0.00625→71.5; p=0.03125→72.5; p=0.0625→73.5; p=0.0625→74.5; p=0.03125→75.5; p=0.00625→76.5 | 0.350 | 0.642 | 0.292 | A | -7.05 |
| 3521 | feedback, known | p=0.4→53; p=0.6→-49 | p=0.1→-41; p=0.45→1; p=0.225→-1; p=0.225→-5 | 0.224 | 0.513 | 0.290 | B | +3.20 |
| 10983 | feedback, known | p=1→30; p=0→30 | p=0.03125→27.5; p=0.15625→28.5; p=0.3125→29.5; p=0.3125→30.5; p=0.15625→31.5; p=0.03125→32.5 | 0.200 | 0.488 | 0.288 | tie | +0.00 |
| 2403 | no feedback, known | p=1→30; p=0→30 | p=0.6→-12; p=0.4→107 | 0.663 | 0.375 | 0.287 | B | +5.60 |
| 1518 | no feedback, known | p=0.4→35; p=0.6→-38 | p=0.25→-42; p=0.375→15; p=0.1875→13; p=0.09375→9; p=0.046875→1; p=0.046875→-15 | 0.350 | 0.636 | 0.286 | B | +6.55 |
| 6255 | feedback, ambiguous | p=0.75→33; p=0.25→1 | p=0.05→-12; p=0.95→30 | 0.093 | 0.371 | 0.278 | B | +2.90 |

## Behavioral regimes

Intervals below resample rows sharing a structural fingerprint as one cluster within each behavioral slice.

| Dimension | Regime | Rows | MAE | 95% MAE interval | RMSE | Bias |
|---|---|---:|---:|---:|---:|---:|
| feedback | no feedback | 348 | 0.0823 | [0.0756, 0.0886] | 0.1031 | +0.0025 |
| feedback | feedback | 1,846 | 0.0806 | [0.0778, 0.0832] | 0.1009 | -0.0038 |
| ambiguity | known probabilities | 1,785 | 0.0803 | [0.0776, 0.0830] | 0.1003 | -0.0023 |
| ambiguity | ambiguous b | 409 | 0.0837 | [0.0773, 0.0903] | 0.1049 | -0.0052 |
| expected value regime | extreme a advantage | 100 | 0.0745 | [0.0640, 0.0849] | 0.0906 | +0.0076 |
| expected value regime | moderate a advantage | 739 | 0.0801 | [0.0757, 0.0846] | 0.1006 | +0.0000 |
| expected value regime | near equal ev | 427 | 0.0886 | [0.0821, 0.0949] | 0.1109 | -0.0041 |
| expected value regime | moderate b advantage | 804 | 0.0796 | [0.0756, 0.0836] | 0.0992 | -0.0035 |
| expected value regime | extreme b advantage | 124 | 0.0722 | [0.0623, 0.0821] | 0.0903 | -0.0191 |
| aggregate division | approximately 50 50 | 349 | 0.0774 | [0.0719, 0.0832] | 0.0949 | -0.0057 |
| aggregate division | leaning | 1,375 | 0.0822 | [0.0789, 0.0854] | 0.1028 | -0.0008 |
| aggregate division | strong consensus | 470 | 0.0797 | [0.0742, 0.0853] | 0.1012 | -0.0066 |
| participant count | n at or below 16 | 1,479 | 0.0814 | [0.0784, 0.0845] | 0.1018 | -0.0014 |
| participant count | n above 16 | 715 | 0.0799 | [0.0754, 0.0843] | 0.1000 | -0.0056 |
| lottery shape b | single outcome | 1,172 | 0.0816 | [0.0783, 0.0851] | 0.1019 | -0.0006 |
| lottery shape b | symmetric | 615 | 0.0787 | [0.0739, 0.0836] | 0.0994 | -0.0034 |
| lottery shape b | right skewed | 222 | 0.0800 | [0.0723, 0.0883] | 0.1009 | -0.0150 |
| lottery shape b | left skewed | 185 | 0.0844 | [0.0763, 0.0931] | 0.1034 | -0.0004 |

## Requested special cases

- Ambiguity: 409 rows; feedback: 1,846 rows.
- Extreme EV difference: 224 rows using the training-derived threshold above.
- Approximately 50/50 human choice: 349 rows with `|bRate − 0.5| ≤ 0.05`.
- Participant count had Pearson correlation -0.018 and Spearman correlation -0.019 with absolute error. These row-level associations are descriptive and the available n range is narrow.

## Representative failures and possible modeling considerations

The considerations below are generated from observed case properties. They are plausible limitations to investigate, not causal explanations.

### Largest Overall: problem 1014

- Conditions: feedback; known probabilities; n = 20.
- Gamble A: p=1→0; p=0→0 (EV = 0.000).
- Gamble B: p=0.05→-20; p=0.95→1 (oracle EV = -0.050).
- Observed bRate = **0.860**; predicted bRate = **0.397**; absolute error = **0.463**.
- Simple EV benchmark: **A** (B − A = -0.050).
- Possible modeling considerations: Feedback condition: a binary indicator cannot represent realized experience histories; Expected values are close, so prediction depends on other represented structure; Aggregate choice is on the opposite side of the simple EV benchmark.

### Ambiguity: problem 3256

- Conditions: feedback; ambiguous B; n = 16.
- Gamble A: p=0.95→26; p=0.05→-9 (EV = 24.250).
- Gamble B: p=0.8→3; p=0.00625→71.5; p=0.03125→72.5; p=0.0625→73.5; p=0.0625→74.5; p=0.03125→75.5; p=0.00625→76.5 (oracle EV = 17.200).
- Observed bRate = **0.350**; predicted bRate = **0.642**; absolute error = **0.292**.
- Simple EV benchmark: **A** (B − A = -7.050).
- Possible modeling considerations: Ambiguous B: the model uses design-oracle probabilities that participants did not see; Feedback condition: a binary indicator cannot represent realized experience histories; The aggregate target is based on a relatively small participant count.

### Feedback: problem 1014

- Conditions: feedback; known probabilities; n = 20.
- Gamble A: p=1→0; p=0→0 (EV = 0.000).
- Gamble B: p=0.05→-20; p=0.95→1 (oracle EV = -0.050).
- Observed bRate = **0.860**; predicted bRate = **0.397**; absolute error = **0.463**.
- Simple EV benchmark: **A** (B − A = -0.050).
- Possible modeling considerations: Feedback condition: a binary indicator cannot represent realized experience histories; Expected values are close, so prediction depends on other represented structure; Aggregate choice is on the opposite side of the simple EV benchmark.

### Extreme Expected Value: problem 4653

- Conditions: feedback; ambiguous B; n = 17.
- Gamble A: p=1→10; p=0→10 (EV = 10.000).
- Gamble B: p=0.99→-5; p=0.01→17 (oracle EV = -4.780).
- Observed bRate = **0.141**; predicted bRate = **0.399**; absolute error = **0.258**.
- Simple EV benchmark: **A** (B − A = -14.780).
- Possible modeling considerations: Ambiguous B: the model uses design-oracle probabilities that participants did not see; Feedback condition: a binary indicator cannot represent realized experience histories; Expected-value difference lies in the training-distribution tail.

### Approximately 50 50: problem 972

- Conditions: feedback; known probabilities; n = 17.
- Gamble A: p=0.1→46; p=0.9→0 (EV = 4.600).
- Gamble B: p=0.6→-24; p=0.4→34 (oracle EV = -0.800).
- Observed bRate = **0.506**; predicted bRate = **0.231**; absolute error = **0.275**.
- Simple EV benchmark: **A** (B − A = -5.400).
- Possible modeling considerations: Feedback condition: a binary indicator cannot represent realized experience histories; Aggregate choice is on the opposite side of the simple EV benchmark; Aggregate choices are nearly evenly divided.


## Interpretation limits

- This is post-selection analysis on validation data; it is exploratory and does not replace evaluation on the still-locked test split.
- Regimes overlap, so their error differences are not independent effects. The bootstrap intervals quantify group-resampling variation, not causal uncertainty.
- The extreme-EV threshold uses raw payoff units and is therefore scale-dependent.
- Some exact participant-count levels have few rows; their estimates and intervals are correspondingly unstable.
- bRate is an aggregate estimate. Its sampling variability can contribute to observed prediction error, especially for smaller n, without identifying why a specific group chose as it did.

## Figures and artifacts

- [Residual diagnostics](figures/model_error_diagnostics.png)
- [Error by behavioral regime](figures/model_error_by_regime.png)
- [Error by participant count](figures/model_error_by_participant_count.png)
- [Largest failures](figures/representative_model_failures.png)

Complete machine-readable tables are stored in `artifacts/analysis/errors/`. The report should be read as validation error analysis after model selection, not as untouched confirmatory test evidence.
