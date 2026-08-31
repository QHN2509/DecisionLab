> **STALE / LEGACY:** This development-partition comparison predates the official nested grouped-CV evaluation. It is preserved only for historical comparison and must not be cited as current official performance. See `nested_baselines.csv` for the official outer-OOF results.

| Baseline | MAE | RMSE | R² | n-weighted MAE | n-weighted RMSE |
|---|---:|---:|---:|---:|---:|
| `constant_training_mean` | 0.1834 | 0.2214 | -0.0000 | 0.1808 | 0.2195 |
| `expected_value_hard_rule_oracle` | 0.3599 | 0.4044 | -2.3371 | 0.3615 | 0.4056 |
| `ridge_engineered_oracle` | 0.1159 | 0.1451 | 0.5706 | 0.1153 | 0.1447 |
| `shallow_decision_tree_engineered_oracle` | 0.0952 | 0.1202 | 0.7052 | 0.0951 | 0.1200 |
