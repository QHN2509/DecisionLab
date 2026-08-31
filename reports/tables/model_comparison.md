> **STALE / LEGACY:** This development-partition comparison predates the official nested grouped-CV evaluation. It is preserved only for historical comparison and must not be cited as current official performance. See `nested_model_comparison.csv` for the official outer-OOF results.

| Model | Selected | Validation MAE | RMSE | R² | n-weighted MAE | n-weighted RMSE | CV MAE ± SD | Complexity |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| `random_forest` | yes | 0.0809 | 0.1012 | 0.7910 | 0.0807 | 0.1010 | 0.0825 ± 0.0015 | 2 |
| `gradient_boosting` | no | 0.0831 | 0.1047 | 0.7762 | 0.0828 | 0.1045 | 0.0843 ± 0.0017 | 3 |
