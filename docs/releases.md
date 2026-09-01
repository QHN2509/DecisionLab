# DecisionLab releases

## v0.2.1 — current portfolio release

v0.2.1 supersedes v0.2.0 as the current portfolio-ready implementation. It includes:

- equal-structural-problem weighting as the primary metric;
- nested grouped cross-validation with fold-local preprocessing and outer OOF evaluation;
- executable feature-leakage protections and complete experiment provenance;
- group- and dependency-preserving behavioral permutation analysis;
- a fresh-clone Streamlit bundle with shared structural scenario validation;
- reproducible staged and non-official quick-start workflows;
- complete choices13k citations and tracked standalone EDA provenance.

The predictive results were not rerun merely for the release-version update. Generated artifacts
retain the package version and exact Git revision recorded when their underlying experiment ran.
Those historical provenance fields must not be rewritten to `0.2.1` after the fact.

## v0.2.0 — superseded release

v0.2.0 was published before its release metadata was committed. Its Git tag remains unchanged as
an immutable record of that published state, but the tagged package still identifies itself as
version 0.1.0. Use v0.2.1 for the current portfolio implementation.

## v0.1.0 — historical pre-methodology-fix checkpoint

v0.1.0 identifies the initial reproducible checkpoint before the methodology, provenance,
interpretation, and application-release fixes summarized above. It remains available only as
project history. Do not use v0.1.0 as the current DecisionLab implementation and do not cite its
results as the current official evaluation.
