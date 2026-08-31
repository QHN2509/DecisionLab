# Systematic model error analysis

This report analyzes complete nested outer out-of-fold predictions. No partition is described as a confirmatory holdout. Differences are descriptive predictive patterns, not causal effects or explanations of participant behavior.

## Summary

Primary equal-structural-group outer OOF MAE was **0.0800** across 13,006 groups. Secondary condition-row MAE was **0.0805** across 14,568 rows.
The absolute EV threshold for an extreme case was **8.500**, calculated as the configured 90% quantile using development predictors.

The five regimes with the largest descriptive MAE were: expected value regime/near equal ev (0.0892), lottery shape b/left skewed (0.0870), feedback/no feedback (0.0855), ambiguity/ambiguous b (0.0849), lottery shape b/right skewed (0.0834). Overlapping bootstrap intervals mean small differences should not be treated as clear population differences.

Regimes above the overall equal-problem error on both MAE and RMSE were: feedback/no feedback, ambiguity/ambiguous b, expected value regime/near equal ev, aggregate division/leaning, aggregate division/strong consensus, participant count/n at or below 16, lottery shape b/right skewed, lottery shape b/left skewed. These overlapping slices are descriptive diagnostics, not independent effects.

## Largest absolute errors

| Problem | Conditions | Gamble A | Gamble B | Actual | Predicted | Abs. error | EV benchmark | ΔEV B−A |
|---:|---|---|---|---:|---:|---:|:---:|---:|
| 1014 | feedback, known | p=1→0; p=0→0 | p=0.05→-20; p=0.95→1 | 0.860 | 0.360 | 0.500 | A | -0.05 |
| 6716 | no feedback, ambiguous | p=0.25→73; p=0.75→-32 | p=0.75→-39; p=0.125→59; p=0.0625→61; p=0.03125→65; p=0.03125→73 | 0.859 | 0.421 | 0.437 | A | -8.00 |
| 238 | no feedback, ambiguous | p=1→-6; p=0→-6 | p=0.01→-15; p=0.99→-3 | 0.263 | 0.697 | 0.435 | B | +2.88 |
| 2346 | no feedback, ambiguous | p=1→2; p=0→2 | p=0.5→5; p=0.25→3; p=0.125→-1; p=0.125→-9 | 0.175 | 0.573 | 0.398 | tie | +0.00 |
| 11363 | feedback, known | p=0.1→-7; p=0.9→-9 | p=0.05→-31; p=0.475→-5; p=0.2375→-7; p=0.2375→-11 | 0.200 | 0.592 | 0.392 | B | +0.60 |
| 3777 | feedback, known | p=0.75→18; p=0.25→3 | p=0.1→7; p=0.1125→15.5; p=0.3375→16.5; p=0.3375→17.5; p=0.1125→18.5 | 0.263 | 0.652 | 0.390 | B | +1.75 |
| 11371 | feedback, known | p=1→21; p=0→21 | p=0.9→11; p=0.1→65 | 0.693 | 0.309 | 0.384 | A | -4.60 |
| 8601 | feedback, known | p=1→1; p=0→1 | p=0.8→0; p=0.2→26 | 0.362 | 0.740 | 0.377 | B | +4.20 |
| 4132 | no feedback, known | p=0.1→23; p=0.9→-11 | p=0.95→-7; p=0.025→1; p=0.025→3 | 0.163 | 0.537 | 0.375 | B | +1.05 |
| 4945 | feedback, ambiguous | p=1→0; p=0→0 | p=0.99→-10; p=0.005→51; p=0.0025→49; p=0.00125→45; p=0.000625→37; p=0.000625→21 | 0.875 | 0.505 | 0.370 | A | -9.43 |

## Behavioral regimes

Intervals below resample structural groups and give every sampled group equal total weight within each behavioral slice.

| Dimension | Regime | Rows | Groups | Problem-group MAE | 95% MAE interval | Condition-row MAE |
|---|---|---:|---:|---:|---:|---:|
| feedback | no feedback | 2,380 | 2,380 | 0.0855 | [0.0828, 0.0881] | 0.0855 |
| feedback | feedback | 12,188 | 12,188 | 0.0795 | [0.0784, 0.0806] | 0.0795 |
| ambiguity | known probabilities | 11,759 | 10,493 | 0.0788 | [0.0777, 0.0800] | 0.0793 |
| ambiguity | ambiguous b | 2,809 | 2,513 | 0.0849 | [0.0822, 0.0874] | 0.0858 |
| expected value regime | extreme a advantage | 717 | 642 | 0.0725 | [0.0682, 0.0769] | 0.0731 |
| expected value regime | moderate a advantage | 5,127 | 4,580 | 0.0792 | [0.0776, 0.0809] | 0.0796 |
| expected value regime | near equal ev | 2,816 | 2,532 | 0.0892 | [0.0866, 0.0915] | 0.0899 |
| expected value regime | moderate b advantage | 5,127 | 4,558 | 0.0782 | [0.0766, 0.0798] | 0.0788 |
| expected value regime | extreme b advantage | 781 | 694 | 0.0703 | [0.0664, 0.0744] | 0.0712 |
| aggregate division | approximately 50 50 | 2,287 | 2,227 | 0.0751 | [0.0729, 0.0774] | 0.0747 |
| aggregate division | leaning | 9,148 | 8,476 | 0.0813 | [0.0800, 0.0825] | 0.0814 |
| aggregate division | strong consensus | 3,133 | 2,951 | 0.0829 | [0.0804, 0.0854] | 0.0822 |
| participant count | n at or below 16 | 9,881 | 8,830 | 0.0817 | [0.0805, 0.0830] | 0.0822 |
| participant count | n above 16 | 4,687 | 4,639 | 0.0770 | [0.0753, 0.0787] | 0.0769 |
| lottery shape b | single outcome | 7,818 | 6,973 | 0.0788 | [0.0775, 0.0802] | 0.0792 |
| lottery shape b | symmetric | 4,010 | 3,593 | 0.0789 | [0.0770, 0.0808] | 0.0794 |
| lottery shape b | right skewed | 1,539 | 1,373 | 0.0834 | [0.0800, 0.0869] | 0.0851 |
| lottery shape b | left skewed | 1,201 | 1,067 | 0.0870 | [0.0831, 0.0910] | 0.0872 |

## Requested special cases

- Ambiguity: 2,809 rows; feedback: 12,188 rows.
- Extreme EV difference: 1,498 rows using the training-derived threshold above.
- Approximately 50/50 human choice: 2,287 rows with `|bRate − 0.5| ≤ 0.05`.
- Participant count had Pearson correlation -0.045 and Spearman correlation -0.041 with absolute error. These row-level associations are descriptive and the available n range is narrow.

## Representative failures and possible modeling considerations

The considerations below are generated from observed case properties. They are plausible limitations to investigate, not causal explanations.

### Largest Overall: problem 1014

- Conditions: feedback; known probabilities; n = 20.
- Gamble A: p=1→0; p=0→0 (EV = 0.000).
- Gamble B: p=0.05→-20; p=0.95→1 (oracle EV = -0.050).
- Observed bRate = **0.860**; predicted bRate = **0.360**; absolute error = **0.500**.
- Simple EV benchmark: **A** (B − A = -0.050).
- Possible modeling considerations: Feedback condition: a binary indicator cannot represent realized experience histories; Expected values are close, so prediction depends on other represented structure; Aggregate choice is on the opposite side of the simple EV benchmark.

### Ambiguity: problem 6716

- Conditions: no feedback; ambiguous B; n = 17.
- Gamble A: p=0.25→73; p=0.75→-32 (EV = -5.750).
- Gamble B: p=0.75→-39; p=0.125→59; p=0.0625→61; p=0.03125→65; p=0.03125→73 (oracle EV = -13.750).
- Observed bRate = **0.859**; predicted bRate = **0.421**; absolute error = **0.437**.
- Simple EV benchmark: **A** (B − A = -8.000).
- Possible modeling considerations: Ambiguous B: the model uses design-oracle probabilities that participants did not see; Aggregate choice is on the opposite side of the simple EV benchmark.

### Feedback: problem 1014

- Conditions: feedback; known probabilities; n = 20.
- Gamble A: p=1→0; p=0→0 (EV = 0.000).
- Gamble B: p=0.05→-20; p=0.95→1 (oracle EV = -0.050).
- Observed bRate = **0.860**; predicted bRate = **0.360**; absolute error = **0.500**.
- Simple EV benchmark: **A** (B − A = -0.050).
- Possible modeling considerations: Feedback condition: a binary indicator cannot represent realized experience histories; Expected values are close, so prediction depends on other represented structure; Aggregate choice is on the opposite side of the simple EV benchmark.

### Extreme Expected Value: problem 4945

- Conditions: feedback; ambiguous B; n = 16.
- Gamble A: p=1→0; p=0→0 (EV = 0.000).
- Gamble B: p=0.99→-10; p=0.005→51; p=0.0025→49; p=0.00125→45; p=0.000625→37; p=0.000625→21 (oracle EV = -9.430).
- Observed bRate = **0.875**; predicted bRate = **0.505**; absolute error = **0.370**.
- Simple EV benchmark: **A** (B − A = -9.430).
- Possible modeling considerations: Ambiguous B: the model uses design-oracle probabilities that participants did not see; Feedback condition: a binary indicator cannot represent realized experience histories; Expected-value difference lies in the training-distribution tail; Aggregate choice is on the opposite side of the simple EV benchmark; The aggregate target is based on a relatively small participant count.

### Approximately 50 50: problem 5777

- Conditions: no feedback; ambiguous B; n = 16.
- Gamble A: p=0.95→12; p=0.05→-8 (EV = 11.000).
- Gamble B: p=0.01→-10; p=0.00773437→15.5; p=0.0541406→16.5; p=0.162422→17.5; p=0.270703→18.5; p=0.270703→19.5; p=0.162422→20.5; p=0.0541406→21.5; p=0.00773437→22.5 (oracle EV = 18.710).
- Observed bRate = **0.463**; predicted bRate = **0.750**; absolute error = **0.288**.
- Simple EV benchmark: **B** (B − A = +7.710).
- Possible modeling considerations: Ambiguous B: the model uses design-oracle probabilities that participants did not see; Aggregate choice is on the opposite side of the simple EV benchmark; Aggregate choices are nearly evenly divided; The aggregate target is based on a relatively small participant count.


## Interpretation limits

- This post-selection analysis uses complete nested outer out-of-fold predictions. It provides exploratory generalization diagnostics, not independent confirmatory evidence.
- Regimes overlap, so their error differences are not independent effects. The bootstrap intervals quantify group-resampling variation, not causal uncertainty.
- The extreme-EV threshold uses raw payoff units and is therefore scale-dependent.
- Some exact participant-count levels have few rows; their estimates and intervals are correspondingly unstable.
- bRate is an aggregate estimate. Its sampling variability can contribute to observed prediction error, especially for smaller n, without identifying why a specific group chose as it did.

## Figures and artifacts

- [Residual diagnostics](figures/model_error_diagnostics.png)
- [Error by behavioral regime](figures/model_error_by_regime.png)
- [Error by participant count](figures/model_error_by_participant_count.png)
- [Largest failures](figures/representative_model_failures.png)

Complete machine-readable tables are stored in `artifacts/analysis/errors/`. The report uses outer OOF errors from the nested selection procedure and is not untouched confirmatory evidence.
