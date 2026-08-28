# Evaluation protocol

This protocol was generated before model training from the checksum-validated choices13k data at commit `821ae7e88386b508ebb46fae76fac63cb62ec876`. It locks grouping, partitioning, metrics, weighting, and test-use rules.

## Problem, feedback, and Block investigation

The CSV has 14,568 rows and 13,006 unique `Problem` IDs. 11,444 problems occur once and 1,562 occur twice. Every repeated problem is one no-feedback row plus one feedback row with identical gamble structure.

Observed `Feedback`/`Block` counts:

| Feedback | Block | Rows |
|---|---:|---:|
| False | 1 | 2,380 |
| True | 2 | 3,006 |
| True | 3 | 3,039 |
| True | 4 | 3,077 |
| True | 5 | 3,066 |

No-feedback observations are always in `Block=1`; feedback observations are always in blocks 2–5. Therefore, `Block` is not an underlying-problem ID and is a deterministic encoding of feedback status in this release. `Feedback` is a condition attached to a problem, not the split group.

## Grouping variable

The split group is `structural_fingerprint`: SHA-256 over both full gamble distributions plus `Ha`, `pHa`, `La`, `Hb`, `pHb`, `Lb`, `LotShapeB`, `LotNumB`, `Amb`, and `Corr`. It deliberately excludes `Feedback`, `Block`, `n`, `bRate`, and `bRate_std`.

At the pinned source revision, 13,006 fingerprints map one-to-one to the same number of `Problem` IDs: there are 0 Problem IDs with multiple fingerprints and 0 fingerprints shared by different Problem IDs. Thus, grouping by `Problem` is sufficient for the current release, while the fingerprint is the more defensive implementation because it would also merge exact structures assigned different IDs.

## Why ordinary row splitting leaks

A row-level split treats feedback variants as independent. A learner can then see the exact gamble structure under one feedback condition during training and be evaluated on the same structure under another condition. That makes prediction artificially easy even when `Feedback` is present as a feature.

The deterministic row-split demonstration placed 706 of 1,562 repeated structural groups across multiple partitions. Singleton problems cannot create this exact paired-row leak, but grouping remains required because the dataset contains paired problems and future revisions could contain duplicate structures under different IDs.

## Locked grouped partitions

Groups are assigned by stable SHA-256 hashing with seed `decisionlab-structural-groups-v1` and thresholds 70%/15%/15%. The hash uses no target values, and adding unrelated rows does not reshuffle existing groups.

| Split | Rows | Structural groups | Feedback rows | No-feedback rows |
|---|---:|---:|---:|---:|
| train | 10,237 | 9,138 | 8,554 | 1,683 |
| validation | 2,194 | 1,953 | 1,846 | 348 |
| test | 2,137 | 1,915 | 1,788 | 349 |

Audit result: **PASS** with 0 structural-group overlaps and 0 Problem-ID overlaps.

The validation split is used for model and hyperparameter selection. The test split must remain untouched until a final pipeline is selected. Any later cross-validation must also group by structural fingerprint and remain inside the development data.

The earlier EDA inspected the full dataset before these assignments existed. Consequently, hypotheses recorded by that EDA can be treated as prespecified for future modeling, but the final test set is not untouched with respect to broad exploratory knowledge. New test-set patterns must not be presented as confirmatory.

## Metrics for bRate

For rows `i=1,…,N`, let `yᵢ` be observed `bRate` and `ŷᵢ` the prediction.
Predictions must be finite and in `[0,1]`; evaluation never clips them silently.

### Primary unweighted metric

- **MAE:** `mean(|ŷᵢ − yᵢ|)`. This is the locked model-selection metric. It is in choice-rate units, treats every problem-condition row equally, and is less dominated by a few noisy large errors than squared loss.

### Secondary unweighted metrics

- **RMSE:** `sqrt(mean((ŷᵢ − yᵢ)²))`; emphasizes large misses and remains in bRate units.
- **R²:** `1 − Σ(ŷᵢ − yᵢ)² / Σ(yᵢ − mean(y))²`; contextualizes performance against the evaluation-set mean and may be negative.
- **Mean bias:** `mean(ŷᵢ − yᵢ)`; positive values indicate systematic overprediction.
- **Calibration diagnostics:** binned observed-versus-predicted plots plus calibration intercept and slope. These are diagnostics, not tuning targets on the test set.

### Participant-count-weighted sensitivity metrics

Repeat MAE, RMSE, R², and mean bias using normalized weights `wᵢ = nᵢ / Σnᵢ`. Larger-`n` rows estimate aggregate rates more precisely, so these metrics are informative sensitivity analyses. They answer a different question—performance weighted by participant-row contributions—and must not replace the primary problem-level metric.

Do not interpret `5n` repeated responses as independent Bernoulli trials: each participant contributed five responses to a problem. For that reason, binomial log loss or likelihood weighted by `5n` is not a primary metric without a model that accounts for within-participant clustering.

## Comparison and uncertainty rules

- Compare candidate pipelines on identical validation rows and the same primary MAE.
- Report all prespecified metrics; do not select whichever metric favors a model.
- Estimate uncertainty and pairwise model differences by resampling structural groups, keeping all feedback variants together within each bootstrap replicate.
- Report overall metrics first. Feedback, ambiguity, correlation, and lottery-shape subgroup metrics are prespecified diagnostics and require row/group counts.
- Every reported value must be generated from saved predictions and the locked split assignments. No model results are produced by this protocol stage.
