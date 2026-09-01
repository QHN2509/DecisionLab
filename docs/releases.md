# DecisionLab releases

## v0.2.0 — current portfolio release

v0.2.0 is the current portfolio-ready implementation. It includes:

- equal-structural-problem weighting as the primary metric;
- nested grouped cross-validation with fold-local preprocessing and outer OOF evaluation;
- executable feature-leakage protections and complete experiment provenance;
- group- and dependency-preserving behavioral permutation analysis;
- a fresh-clone Streamlit bundle with shared structural scenario validation;
- reproducible staged and non-official quick-start workflows;
- complete choices13k citations and tracked standalone EDA provenance.

The predictive results were not rerun merely for the release-version update. Generated artifacts
retain the package version and exact Git revision recorded when their underlying experiment ran.
Those historical provenance fields must not be rewritten to `0.2.0` after the fact.

## v0.1.0 — historical pre-fix checkpoint

v0.1.0 identifies the initial reproducible checkpoint before the methodology, provenance,
interpretation, and application-release fixes summarized above. It remains available only as
project history. Do not use v0.1.0 as the current DecisionLab implementation and do not cite its
results as the current official evaluation.
