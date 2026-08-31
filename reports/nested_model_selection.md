# Nested grouped-CV model selection

All choices13k rows are development data. This report uses complete outer out-of-fold predictions from a 5 × 5 nested structural-group CV procedure; it does not claim an untouched confirmatory holdout.

## Headline

The inner-selected procedure achieved equal-problem outer OOF MAE **0.0800** (95% structural-group bootstrap interval 0.0789–0.0810).

Condition-row and participant-count-weighted metrics in the machine-readable table are secondary.

## Isolation

Hyperparameters and model family were selected only from inner OOF predictions within each outer training portion. Every outer test group was predicted once by a pipeline fit without that group.

The production model was selected and fit separately on all development rows and has no direct in-sample performance claim.
