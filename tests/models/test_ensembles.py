from __future__ import annotations

import joblib
import numpy as np
import pytest

from decisionlab.models.ensembles import build_candidate_pipeline


@pytest.mark.parametrize("name", ["random_forest", "gradient_boosting"])
def test_candidate_pipeline_is_deterministic_and_bounded(name: str) -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    target = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if name == "random_forest":
        parameters = {
            "criterion": "squared_error",
            "n_estimators": 10,
            "n_jobs": 1,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        }
    else:
        parameters = {
            "loss": "squared_error",
            "n_estimators": 10,
            "learning_rate": 0.05,
            "max_depth": 2,
            "min_samples_leaf": 1,
        }

    first = build_candidate_pipeline(name, parameters, random_seed=17).fit(features, target)
    second = build_candidate_pipeline(name, parameters, random_seed=17).fit(features, target)
    first_predictions = first.predict(features)

    assert np.array_equal(first_predictions, second.predict(features))
    assert np.all((first_predictions >= 0.0) & (first_predictions <= 1.0))
    assert tuple(first.named_steps) == ("preprocess", "model")


def test_candidate_pipeline_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown candidate"):
        build_candidate_pipeline("xgboost", {}, random_seed=17)


def test_fitted_pipeline_round_trip_preserves_exact_predictions(tmp_path) -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    target = np.asarray([0.0, 0.3, 0.7, 1.0])
    pipeline = build_candidate_pipeline(
        "random_forest",
        {
            "n_estimators": 12,
            "n_jobs": 1,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        },
        random_seed=17,
    ).fit(features, target)
    path = tmp_path / "pipeline.joblib"

    joblib.dump(pipeline, path)
    restored = joblib.load(path)

    assert tuple(restored.named_steps) == ("preprocess", "model")
    assert np.array_equal(pipeline.predict(features), restored.predict(features))
