# Behavioral feature reference

This document is generated from `FEATURE_DEFINITIONS` by `decisionlab-build-features`. Engineered predictors are separate from the raw predictor table. `row_index` and `problem` appear in output files only as join/grouping keys and must not be model inputs. No feature uses `bRate`, `bRate_std`, `n`, or `Block`.

## Feature contracts

### `expected_value_a`

- **Formula:** pHa × Ha + (1 − pHa) × La
- **Interpretation:** Probability-weighted mean payoff of Gamble A.
- **Assumptions:** A probabilities and payoffs are participant-visible.
- **Potential limitations:** Expected value does not represent risk preferences or nonlinear utility.
- **Availability:** participant-visible
- **Leakage audit:** PASS — Uses only pre-choice Gamble A parameters; no outcome or target fields.

### `expected_value_b_oracle`

- **Formula:** pHb × Hb + (1 − pHb) × Lb
- **Interpretation:** Probability-weighted mean payoff of the complete Gamble B distribution.
- **Assumptions:** Hb is the mean of the embedded B sublottery, as defined upstream.
- **Potential limitations:** B probabilities are hidden when Amb=True, so this is then an oracle feature.
- **Availability:** oracle under ambiguity
- **Leakage audit:** PASS — Pre-choice design data only, but unavailable to participants under ambiguity.

### `expected_value_difference_b_minus_a_oracle`

- **Formula:** expected_value_b_oracle − expected_value_a
- **Interpretation:** Objective expected-value advantage of B over A.
- **Assumptions:** Both gamble means are comparable in the payoff units of the task.
- **Potential limitations:** Inherits the oracle status of expected_value_b_oracle under ambiguity.
- **Availability:** oracle under ambiguity
- **Leakage audit:** PASS — Derived only from audited expected values; unavailable under ambiguous B.

### `payoff_range_a`

- **Formula:** max(A payoffs) − min(A payoffs)
- **Interpretation:** Outcome spread of Gamble A without probability weighting.
- **Assumptions:** Displayed A outcomes define the relevant payoff support.
- **Potential limitations:** Range ignores outcome probabilities and distribution shape.
- **Availability:** participant-visible
- **Leakage audit:** PASS — Uses only the displayed payoff support; no response information.

### `payoff_range_b`

- **Formula:** max(B payoffs) − min(B payoffs)
- **Interpretation:** Outcome spread of Gamble B without probability weighting.
- **Assumptions:** B payoff values remain visible when their probabilities are ambiguous.
- **Potential limitations:** Range ignores probabilities and may be dominated by rare outcomes.
- **Availability:** participant-visible
- **Leakage audit:** PASS — Uses only the displayed payoff support; no hidden probability or response data.

### `payoff_range_difference_b_minus_a`

- **Formula:** payoff_range_b − payoff_range_a
- **Interpretation:** How much wider B's payoff support is than A's.
- **Assumptions:** Ranges are comparable because both gambles use the same payoff units.
- **Potential limitations:** A difference in ranges is not a complete measure of relative risk.
- **Availability:** participant-visible
- **Leakage audit:** PASS — Difference of two participant-visible, non-target-derived features.

### `maximum_payoff_difference_b_minus_a`

- **Formula:** max(B payoffs) − max(A payoffs)
- **Interpretation:** B's best possible payoff relative to A's best possible payoff.
- **Assumptions:** Participants attend to the displayed extreme outcomes.
- **Potential limitations:** Ignores the probabilities of reaching those maxima.
- **Availability:** participant-visible
- **Leakage audit:** PASS — Uses displayed payoff extrema only; no target or post-choice fields.

### `minimum_payoff_difference_b_minus_a`

- **Formula:** min(B payoffs) − min(A payoffs)
- **Interpretation:** B's worst possible payoff relative to A's worst possible payoff.
- **Assumptions:** Participants attend to the displayed extreme outcomes.
- **Potential limitations:** Ignores the probabilities of reaching those minima.
- **Availability:** participant-visible
- **Leakage audit:** PASS — Uses displayed payoff extrema only; no target or post-choice fields.

### `payoff_std_a`

- **Formula:** sqrt(Σᵢ p(Aᵢ) × (Aᵢ − expected_value_a)²)
- **Interpretation:** Probability-weighted payoff dispersion of Gamble A.
- **Assumptions:** The population standard deviation of the stated A distribution is relevant.
- **Potential limitations:** Dispersion treats upside and downside variation symmetrically.
- **Availability:** participant-visible
- **Leakage audit:** PASS — Uses only participant-visible A probabilities and payoffs.

### `payoff_std_b_oracle`

- **Formula:** sqrt(Σᵢ p(Bᵢ) × (Bᵢ − expected_value_b_oracle)²)
- **Interpretation:** Probability-weighted payoff dispersion of Gamble B.
- **Assumptions:** The full marginal B distribution in the JSON is the relevant distribution.
- **Potential limitations:** Probabilities are not participant-visible when Amb=True.
- **Availability:** oracle under ambiguity
- **Leakage audit:** PASS — Pre-choice design data only, but hidden B probabilities make it oracle under ambiguity.

### `payoff_std_difference_b_minus_a_oracle`

- **Formula:** payoff_std_b_oracle − payoff_std_a
- **Interpretation:** Difference in objective payoff dispersion between B and A.
- **Assumptions:** Both standard deviations use the same payoff units.
- **Potential limitations:** Not a complete risk measure and oracle under ambiguity.
- **Availability:** oracle under ambiguity
- **Leakage audit:** PASS — Inherits only the documented hidden-probability risk from its B component.

### `best_payoff_probability_a`

- **Formula:** Σᵢ p(Aᵢ) × 1[Aᵢ = max(A payoffs)]
- **Interpretation:** Probability that A yields its best displayed payoff.
- **Assumptions:** Probabilities for tied maximum outcomes are summed.
- **Potential limitations:** A best-outcome probability does not describe the remaining distribution.
- **Availability:** participant-visible
- **Leakage audit:** PASS — Uses only stated A outcomes and probabilities.

### `best_payoff_probability_b_oracle`

- **Formula:** Σᵢ p(Bᵢ) × 1[Bᵢ = max(B payoffs)]
- **Interpretation:** Probability that B yields its best displayed payoff.
- **Assumptions:** Probabilities for tied maximum outcomes are summed.
- **Potential limitations:** The probability is hidden when Amb=True.
- **Availability:** oracle under ambiguity
- **Leakage audit:** PASS — Uses no response data, but uses hidden design probabilities under ambiguity.

### `best_payoff_probability_difference_b_minus_a_oracle`

- **Formula:** best_payoff_probability_b_oracle − best_payoff_probability_a
- **Interpretation:** Difference in the chance of obtaining each gamble's best payoff.
- **Assumptions:** Best-outcome events are comparable summaries across gambles.
- **Potential limitations:** The two best payoffs can differ greatly in magnitude; oracle under ambiguity.
- **Availability:** oracle under ambiguity
- **Leakage audit:** PASS — Inherits hidden-probability risk only from the B component.

### `loss_probability_a`

- **Formula:** Σᵢ p(Aᵢ) × 1[Aᵢ < 0]
- **Interpretation:** Objective probability that Gamble A produces a negative payoff.
- **Assumptions:** Zero is the behaviorally meaningful gain/loss reference point.
- **Potential limitations:** Does not distinguish small from severe losses.
- **Availability:** participant-visible
- **Leakage audit:** PASS — Uses only stated A outcomes and probabilities.

### `loss_probability_b_oracle`

- **Formula:** Σᵢ p(Bᵢ) × 1[Bᵢ < 0]
- **Interpretation:** Objective probability that Gamble B produces a negative payoff.
- **Assumptions:** Zero is the behaviorally meaningful gain/loss reference point.
- **Potential limitations:** Hidden probabilities make this unavailable when Amb=True.
- **Availability:** oracle under ambiguity
- **Leakage audit:** PASS — Uses no response data, but uses hidden design probabilities under ambiguity.

### `loss_probability_difference_b_minus_a_oracle`

- **Formula:** loss_probability_b_oracle − loss_probability_a
- **Interpretation:** B's objective loss probability relative to A's.
- **Assumptions:** A zero-payoff reference point is appropriate for both gambles.
- **Potential limitations:** Does not encode loss magnitude and is oracle under ambiguity.
- **Availability:** oracle under ambiguity
- **Leakage audit:** PASS — Inherits hidden-probability risk only from the B component.

### `ambiguity_indicator`

- **Formula:** 1[Amb = True]
- **Interpretation:** Whether Gamble B probabilities were hidden from participants.
- **Assumptions:** The upstream Amb field correctly describes information availability.
- **Potential limitations:** Does not quantify the degree of ambiguity beyond this binary design.
- **Availability:** experimental condition
- **Leakage audit:** PASS — Known before choice; safe if ambiguity is known in the prediction setting.

### `feedback_indicator`

- **Formula:** 1[Feedback = True]
- **Interpretation:** Whether participants received obtained and forgone outcome feedback.
- **Assumptions:** Feedback status is known for the prediction task.
- **Potential limitations:** Feedback is entangled with block/order and observed experience histories are unavailable.
- **Availability:** experimental condition
- **Leakage audit:** PASS — Known before the modeled condition, but can proxy for block/order; paired rows must be grouped.

### `lottery_shape_b_undefined`

- **Formula:** 1[LotShapeB = 0]
- **Interpretation:** Indicator that the B sublottery shape is undefined because it has one outcome.
- **Assumptions:** Upstream category 0 has its documented meaning.
- **Potential limitations:** Redundant with LotNumB=1 and must not be treated as ordered.
- **Availability:** categorical encoding
- **Leakage audit:** PASS — Pre-choice structural category; no target information.

### `lottery_shape_b_symmetric`

- **Formula:** 1[LotShapeB = 1]
- **Interpretation:** Indicator for a symmetric B sublottery.
- **Assumptions:** Upstream shape labels describe the generated sublottery.
- **Potential limitations:** Shape alone does not identify spread or tail magnitude.
- **Availability:** categorical encoding
- **Leakage audit:** PASS — Pre-choice structural category; no target information.

### `lottery_shape_b_right_skewed`

- **Formula:** 1[LotShapeB = 2]
- **Interpretation:** Indicator for a right-skewed B sublottery.
- **Assumptions:** Upstream shape labels describe the generated sublottery.
- **Potential limitations:** Shape alone does not identify spread or tail magnitude.
- **Availability:** categorical encoding
- **Leakage audit:** PASS — Pre-choice structural category; no target information.

### `lottery_shape_b_left_skewed`

- **Formula:** 1[LotShapeB = 3]
- **Interpretation:** Indicator for a left-skewed B sublottery.
- **Assumptions:** Upstream shape labels describe the generated sublottery.
- **Potential limitations:** Shape alone does not identify spread or tail magnitude.
- **Availability:** categorical encoding
- **Leakage audit:** PASS — Pre-choice structural category; no target information.

### `correlation_negative`

- **Formula:** 1[Corr = −1]
- **Interpretation:** Indicator for negatively correlated gamble payoffs.
- **Assumptions:** Upstream correlation category is known for the condition.
- **Potential limitations:** Category does not provide correlation magnitude.
- **Availability:** categorical encoding
- **Leakage audit:** PASS — Pre-choice structural category; no target information.

### `correlation_zero`

- **Formula:** 1[Corr = 0]
- **Interpretation:** Indicator for uncorrelated gamble payoffs.
- **Assumptions:** Upstream correlation category is known for the condition.
- **Potential limitations:** Category does not establish empirical independence in aggregate responses.
- **Availability:** categorical encoding
- **Leakage audit:** PASS — Pre-choice structural category; no target information.

### `correlation_positive`

- **Formula:** 1[Corr = 1]
- **Interpretation:** Indicator for positively correlated gamble payoffs.
- **Assumptions:** Upstream correlation category is known for the condition.
- **Potential limitations:** Category does not provide correlation magnitude.
- **Availability:** categorical encoding
- **Leakage audit:** PASS — Pre-choice structural category; no target information.

### `expected_value_difference_oracle_x_feedback`

- **Formula:** expected_value_difference_b_minus_a_oracle × feedback_indicator
- **Interpretation:** Allows EV sensitivity to differ descriptively when outcome feedback is available.
- **Assumptions:** A feedback-specific EV slope is behaviorally interpretable.
- **Potential limitations:** Requires both main effects later; oracle under ambiguity and not causal.
- **Availability:** oracle under ambiguity
- **Leakage audit:** PASS — No target data, but inherits oracle status and feedback's block/order caveat.

### `expected_value_difference_oracle_x_ambiguity`

- **Formula:** expected_value_difference_b_minus_a_oracle × ambiguity_indicator
- **Interpretation:** Marks the oracle EV slope specifically for ambiguous Gamble B problems.
- **Assumptions:** Attenuated oracle-EV sensitivity under ambiguity is behaviorally testable.
- **Potential limitations:** The interacted EV was not participant-visible and cannot represent explicit EV calculation.
- **Availability:** oracle under ambiguity
- **Leakage audit:** PASS — No target data, but intentionally uses latent design information in ambiguous rows.

## Deliberately excluded transformations

- `pHb - pHa` is not included because these probabilities refer to different events: entering B's sublottery versus obtaining A's high outcome.
- Ratios of expected values or payoffs are not included because zero and negative denominators make their behavioral interpretation unstable.
- Polynomial, logarithmic, rank, quantile, and target-encoded transformations are not included without a prespecified behavioral rationale.
- `bRate_std`, `n`, and `Block` are excluded from predictors. The first is target-derived, the second is measurement metadata, and the third deterministically encodes feedback status in this dataset.

## Participant-visible versus oracle features

Names ending in `_oracle`, and interactions built from them, use objective Gamble B probabilities. They may be used for an explicitly labeled design/oracle analysis. When `Amb=True`, they must not be described as information participants could calculate from the choice display. Future participant-visible models must omit or mask these features for ambiguous rows rather than silently treating them as observed.

Payoff-support features such as ranges and extrema do not use probabilities and remain participant-visible under the documented ambiguity manipulation.
