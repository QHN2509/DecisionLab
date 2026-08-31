# DecisionLab

DecisionLab is a reproducible machine-learning study of aggregate human choice
under risk and uncertainty. Given the structure of two gambles, the project
predicts `bRate`: the proportion of choices assigned to Gamble B for a
problem-condition row in choices13k.

The project asks three practical questions:

1. How accurately can decision structure predict aggregate choice rates?
2. Where does observed behavior depart from a simple expected-value benchmark?
3. How are expected value, risk, ambiguity, feedback, and lottery structure
   associated with model predictions and errors?

This is an aggregate prediction task. DecisionLab does **not** predict an
individual person's choice, infer individual utility, or estimate causal
effects.

## Why the evaluation is group-based

choices13k contains 14,568 condition rows representing 13,006 decision
problems. Of these problems, 1,562 appear twice: once without feedback and once
with feedback, using the same gamble structure.

A random row split can therefore train on one condition and evaluate on the
same underlying problem under another condition. In the repository's
deterministic leakage demonstration, ordinary row splitting placed **706**
repeated structures across partitions.

DecisionLab assigns a structural fingerprint to each problem using the complete
gamble distributions and design variables. All rows sharing either a problem ID
or exact structure remain in the same fold. Five outer folds provide complete
out-of-fold predictions; five inner grouped folds within each outer training set
perform tuning and model-family selection.

The primary metric is equal-structural-problem-group MAE: loss is averaged within
each structural group and then equally across groups. Condition-row and
participant-count-weighted metrics are secondary because neither defines the
primary prediction unit. Earlier train/validation/test labels are historical
development-era partitions, not an untouched confirmatory holdout.

## Data and target

`bRate` is the mean participant-level rate of selecting Gamble B. Each
participant completed five trials for a problem, so the target is continuous in
`[0,1]`; it is not an individual binary response. Participant counts range from
15 to 33 per row.

Raw files are downloaded from a commit-pinned public repository, verified by
SHA-256, made read-only, and never modified in place. The JSON file is joined by
its documented zero-based CSV row key rather than by `Problem` ID.

## Features and leakage controls

The model uses 28 behaviorally interpretable features derived from validated
design variables:

- expected values and the B-minus-A expected-value difference;
- payoff ranges, extrema, and probability-weighted dispersion;
- best-payoff and loss probabilities;
- ambiguity and feedback indicators;
- lottery-shape and correlation encodings;
- two prespecified EV interactions with ambiguity and feedback.

Feature construction deliberately excludes `bRate`, `bRate_std`, participant
count `n`, `Block`, and identifiers. Schema validation eliminates the need for
imputation; scaling where required and all model fitting use training data only.
Ten features use objective Gamble B probabilities that participants could not
observe under ambiguity; results using them are therefore labeled
**design/oracle predictions**.

Mathematical definitions and per-feature leakage assessments are in
[docs/features.md](docs/features.md).

## Model results

Official results were regenerated from clean DecisionLab commit `1499328a` using five outer and
five inner structural-group folds. The inner-selected procedure—Random Forest in every outer
fold—achieved equal-problem outer-OOF MAE **0.0800** (95% structural-group bootstrap interval
0.0789–0.0810), RMSE **0.1011**, and R² **0.7923**. Secondary condition-row MAE was 0.0805 and
participant-count-weighted MAE was 0.0801. These are development-data generalization estimates,
not results from an untouched confirmatory holdout.

The strongest simple baseline was the shallow decision tree (equal-problem MAE 0.0950), followed
by ridge regression (0.1167), the constant mean (0.1836), and the deliberately hard expected-value
rule (0.3614). The latter predicts only 0, 0.5, or 1 and is a behavioral benchmark rather than a
calibrated aggregate-choice model. Machine-readable comparisons are in
[`reports/tables/nested_baselines.csv`](reports/tables/nested_baselines.csv) and
[`reports/tables/nested_model_comparison.csv`](reports/tables/nested_model_comparison.csv).

New feature, EDA, experiment, and downstream analysis runs automatically record Git state,
dataset and complete configuration hashes, environment lock hashes, transitive pipeline-source
hashes, fold identifiers, input lineage where applicable, and output hashes. A dirty worktree is
refused by default; an explicit override is marked non-official. Behavioral and error analyses
also require regenerated model provenance, so legacy artifacts cannot silently become current. See
[docs/provenance.md](docs/provenance.md).

The pooled outer-OOF equal-problem metric is the headline; condition-row and
participant-count-weighted metrics are secondary. Random Forest outperformed Gradient Boosting on
the same outer folds (equal-problem MAE 0.0800 versus 0.0830) and was selected independently in
each outer fold using inner OOF results only.

### What the expected-value baseline teaches

The hard benchmark predicts B with probability 1 when `EV(B) > EV(A)`, A with
probability 1 when `EV(A) > EV(B)`, and 0.5 for a tie. It tests how a deterministic
expected-value rule compares with aggregate observed rates; disagreement is a
deviation from this simple benchmark, not evidence of irrationality.

## Behavioral interpretation

Behavioral, normative, and error analyses were regenerated from complete outer OOF predictions.
Behavioral reliance is assessed with grouped, dependency-preserving perturbations: complete
structural groups move only within their outer fold, and dependent engineered features are rebuilt
through production feature engineering. The resulting family rankings describe model reliance and
sensitivity—not isolated effects, causal effects, or participant-level mechanisms. See
[`reports/behavioral_analysis.md`](reports/behavioral_analysis.md) and
[`reports/error_analysis.md`](reports/error_analysis.md).

## Interactive research dashboard

Using the regenerated official nested-CV artifacts, the Streamlit application lets a user
construct a supported risky-choice problem and displays:

- both gamble distributions and their expected values;
- the simple expected-value benchmark;
- predicted aggregate Gamble B choice rate;
- global model-driver domains;
- coherent feedback and ambiguity what-if comparisons;
- warnings when engineered values fall outside the training range.

The application loads the official production pipeline under
`artifacts/experiments/nested_model_selection/` and verifies its model and behavioral provenance
before prediction. Its displayed error is the outer-OOF dataset estimate, not an uncertainty
interval for the constructed scenario. Feature formulas remain shared with the research pipeline
rather than duplicated in the UI.

## Limitations

- Broad target-aware EDA preceded the nested protocol. Outer OOF performance is
  leakage-controlled for the frozen selection procedure but is not an independent
  confirmatory result unaffected by prior exploration.
- Oracle probability features limit the interpretation of ambiguous scenarios:
  they represent analyst-known design information, not participant-visible
  information.
- Feedback is represented as a condition indicator; realized experience
  histories are unavailable.
- Predictions describe aggregate rates for choices13k-style problems and do not
  transfer automatically to individuals, new populations, or high-stakes
  decisions.
- Permutation importance, partial dependence, subgroup errors, and what-if
  comparisons are predictive associations or model sensitivities—not causal
  estimates.
- The application reports dataset-level validation error, not a calibrated
  uncertainty interval for a constructed scenario.

## Quick start: launch the tracked demo

Python 3.12 or newer is required. The tracked production model and provenance-covered application
metadata are sufficient to launch Streamlit; downloading choices13k is not required for this path.

```bash
git clone https://github.com/QHN2509/DecisionLab.git
cd DecisionLab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-build-isolation --no-deps -e .

streamlit run app.py
```

This quick path consumes the checked-in official artifacts. It does not regenerate or make a new
official experimental claim.

## Local reproduction in one worktree

The following path downloads and validates the data, rebuilds features and folds, and reruns the
experiments. It is convenient for checking the project locally. Once an earlier stage changes the
worktree, `--allow-dirty` keeps later commands executable while marking their manifests
`non_official_dirty_worktree`; it does not weaken or bypass the provenance label.

```bash
decisionlab-fetch-data
decisionlab-validate-data
decisionlab-eda
decisionlab-build-features --allow-dirty
decisionlab-create-folds
decisionlab-run-baselines --allow-dirty
decisionlab-select-model --allow-dirty
```

Downstream behavioral, error, and application-metadata runs require eligible official model
artifacts. In a one-worktree check, retain the checked-in official model artifacts and run those
analyses before overwriting them, or use the official staged workflow below. The quick Streamlit
path remains available without any local data.

## Official provenance-controlled regeneration

Official runners start only from a clean commit. Use a dedicated reproduction branch, run one stage,
review and commit its generated outputs, then begin the next stage from that new clean commit:

```bash
git switch -c reproduce/official

decisionlab-fetch-data
decisionlab-validate-data

decisionlab-eda
# review, then: git add -A && git commit -m "Reproduce EDA"

decisionlab-build-features
# review, then: git add -A && git commit -m "Reproduce features"

decisionlab-create-folds
# review, then: git add -A && git commit -m "Reproduce grouped folds"

decisionlab-run-baselines
# review, then: git add -A && git commit -m "Reproduce nested baselines"

decisionlab-select-model
# review, then: git add -A && git commit -m "Reproduce nested model selection"

decisionlab-behavioral-analysis
# review, then: git add -A && git commit -m "Reproduce behavioral analysis"

decisionlab-error-analysis
# review, then: git add -A && git commit -m "Reproduce error analysis"

decisionlab-build-app-metadata
# review, then: git add -A && git commit -m "Reproduce application metadata"

python -m pytest
ruff check .
ruff format --check .
python -m pip check

streamlit run app.py
```

Do not collapse the official sequence into an uninterrupted dirty-worktree script. The clean-state
check is part of the provenance contract. Separate clean worktrees may be used instead of the
review-and-commit checkpoints. See [docs/application.md](docs/application.md) for the application
contract and [docs/evaluation_protocol.md](docs/evaluation_protocol.md) for the split and metric
rationale.

## Repository layout

```text
configs/             reproducible data, split, experiment, and analysis settings
data/                immutable raw data and regenerated processed tables
src/decisionlab/     validation, features, splitting, models, analysis, and app services
tests/               data contracts, formulas, leakage, serialization, and prediction tests
artifacts/           machine-readable provenance, predictions, metrics, and models
reports/             generated research reports, figures, and comparison tables
docs/                data, feature, evaluation, and application contracts
```

## Citation

DecisionLab uses choices13k from:

Peterson, J. C., Bourgin, D. D., Agrawal, M., Reichman, D., & Griffiths,
T. L. (2021). Using large-scale experiments and machine learning to discover
theories of human decision-making. *Science, 372*(6547), 1209–1214.
https://doi.org/10.1126/science.abe2629

Bourgin, D. D., Peterson, J. C., Reichman, D., Russell, S. J., & Griffiths,
T. L. (2019). Cognitive model priors for predicting human decisions.
*Proceedings of the 36th International Conference on Machine Learning*,
PMLR 97, 5133–5141. https://proceedings.mlr.press/v97/peterson19a.html

Dataset repository: https://github.com/jcpeterson/choices13k, pinned here to
commit `821ae7e88386b508ebb46fae76fac63cb62ec876`.
