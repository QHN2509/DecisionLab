# Interactive application

The DecisionLab Streamlit application is a research dashboard for the selected
Random Forest model. Run it from the repository root with:

```bash
streamlit run app.py
```

The application loads the tracked official nested model-selection production pipeline and a small
tracked application-metadata artifact containing the training-feature ranges. It verifies the model,
application metadata, and behavioral statistics against their provenance manifests. A fresh clone
therefore does not need the ignored processed feature table or raw dataset merely to launch the app.
The service uses the exact saved feature order and calls
`engineer_scenario_features` from the production behavioral-feature module. The
UI does not implement feature formulas.

## Supported scenario

Gamble A is a binary lottery with a high payoff and probability plus a low
payoff. Gamble B has a low branch and a choices13k-style sublottery described by
its mean, branch probability, shape, and number of outcomes. Symmetric
sublotteries use centered binomial probabilities. Right- and left-skewed
sublotteries use the mirrored geometric construction represented in choices13k.
The single-outcome shape has no within-sublottery dispersion.

Input defaults and primitive ranges are declared in `configs/app.json`. The app
also checks every engineered value against its training range and labels any
scenario that requires feature-level extrapolation.

The shared prediction service rejects non-finite or non-numeric payoffs,
probabilities outside `[0, 1]`, inconsistent lottery-shape/outcome-count pairs,
invalid correlation categories, and non-boolean feedback or ambiguity values.
It also enforces the choices13k ordering contract that Gamble B's sublottery mean is at least its
low-branch payoff.
This validation applies even when the service is called outside Streamlit.

The application metadata is stored at `artifacts/application/metadata.json`, with lineage in
`artifacts/application/provenance.json`. `decisionlab-build-app-metadata` regenerates it from the
provenance-verified official engineered feature table without fitting or changing the model.

## Interpretation contract

- The displayed prediction is an aggregate `bRate`, not a personalized choice
  prediction or recommendation.
- The expected-value result is a simple benchmark, not a definition of
  rational behavior.
- Important drivers are grouped, dependency-preserving outer-OOF permutation-reliance domains
  from the behavioral analysis. They are global predictive reliance measures, not
  causal effects or local SHAP values.
- Feedback and ambiguity comparisons are coherent model-sensitivity checks.
  They are not treatment-effect estimates.
- Under ambiguity, the selected model still uses analyst-known design
  probabilities. The app labels this as an oracle/design prediction because
  participants did not observe those probabilities.
- The displayed performance metric is the corrected equal-problem outer-OOF estimate. It is not a
  calibrated prediction interval for a constructed scenario, and no confirmatory holdout is
  claimed.
