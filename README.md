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

Official nested-CV results are pending. Existing experiment metrics, model-selection
reports, behavioral reports, error reports, tables, figures, and serialized models
were generated from development-era partitions and are explicitly stale. They are
retained only for provenance and are not headline evidence.

New feature, EDA, experiment, and downstream analysis runs automatically record Git state,
dataset and complete configuration hashes, environment lock hashes, transitive pipeline-source
hashes, fold identifiers, input lineage where applicable, and output hashes. A dirty worktree is
refused by default; an explicit override is marked non-official. Behavioral and error analyses
also require regenerated model provenance, so legacy artifacts cannot silently become current. See
[docs/provenance.md](docs/provenance.md).

The replacement experiment will report pooled outer OOF equal-problem MAE as its
headline, with condition-row and participant-count-weighted metrics labeled as
secondary. It will compare the complete inner-selection procedure and independently
tuned candidate families on identical outer folds.

### What the expected-value baseline teaches

The hard benchmark predicts B with probability 1 when `EV(B) > EV(A)`, A with
probability 1 when `EV(A) > EV(B)`, and 0.5 for a tie. It tests how a deterministic
expected-value rule compares with aggregate observed rates; disagreement is a
deviation from this simple benchmark, not evidence of irrationality.

## Behavioral interpretation

Generalization-oriented behavioral, normative, and error analyses will be
regenerated from complete outer OOF predictions. Model sensitivity computed from
a full-data production fit will be labeled descriptive rather than held-out evidence.
No existing numeric interpretation is currently treated as official.

## Interactive research dashboard

After official nested-CV artifacts are regenerated and wired into deployment, the
Streamlit application lets a user construct a supported risky-choice problem and displays:

- both gamble distributions and their expected values;
- the simple expected-value benchmark;
- predicted aggregate Gamble B choice rate;
- global model-driver domains;
- coherent feedback and ambiguity what-if comparisons;
- warnings when engineered values fall outside the training range.

The checked-in application still references the stale development-era serialized model and must
not currently be presented with official performance claims. Feature formulas remain shared with
the research pipeline rather than duplicated in the UI.

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

## Reproduce the project

Python 3.12 or newer and internet access are required for the initial install
and dataset download. Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-build-isolation --no-deps -e .

decisionlab-fetch-data
decisionlab-validate-data --output artifacts/manifests/data_validation_summary.json
decisionlab-eda
decisionlab-build-features
decisionlab-create-folds
decisionlab-run-baselines
decisionlab-select-model
decisionlab-behavioral-analysis
decisionlab-error-analysis

python -m pytest
ruff check .
ruff format --check .
python -m pip check

streamlit run app.py
```

The analysis commands regenerate local data, experiment artifacts, reports, and
the selected pipeline. Production prediction tests intentionally run after
those artifacts exist. See [docs/application.md](docs/application.md) for the
application contract and [docs/evaluation_protocol.md](docs/evaluation_protocol.md)
for the complete split and metric rationale.

For an **official** baseline or model-selection run, the repository must be clean immediately
before `decisionlab-run-baselines` or `decisionlab-select-model` starts. The commands above are a
development workflow, not an instruction to run both official experiments uninterrupted: review
and commit intended generated changes between stages, or run each experiment from a separate clean
checkout of the intended commit. `--allow-dirty` is only for smoke testing and marks the result
non-official.

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

Dataset repository: https://github.com/jcpeterson/choices13k, pinned here to
commit `821ae7e88386b508ebb46fae76fac63cb62ec876`.
