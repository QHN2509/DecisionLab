"""Tree-ensemble candidates for grouped DecisionLab model selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


class ClippedRegressor(RegressorMixin, BaseEstimator):
    """Fit a regressor and explicitly clip its predictions to known target bounds."""

    def __init__(
        self,
        estimator: RegressorMixin,
        *,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
    ) -> None:
        self.estimator = estimator
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def fit(self, features: np.ndarray, target: np.ndarray) -> ClippedRegressor:
        """Fit a cloned underlying estimator."""
        if self.lower_bound >= self.upper_bound:
            raise ValueError("Prediction lower bound must be below upper bound")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(features, target)
        return self

    def predict_raw(self, features: np.ndarray) -> np.ndarray:
        """Return predictions before the declared target-bound postprocessing."""
        return self.estimator_.predict(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return explicitly bounded predictions."""
        return np.clip(self.predict_raw(features), self.lower_bound, self.upper_bound)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Expose fitted tree importance values for reporting."""
        return self.estimator_.feature_importances_


def make_model_pipeline(model: RegressorMixin) -> Pipeline:
    """Create a serializable identity-preprocessing pipeline for numeric features."""
    return Pipeline(
        [
            (
                "preprocess",
                FunctionTransformer(validate=False, feature_names_out="one-to-one"),
            ),
            ("model", ClippedRegressor(model)),
        ]
    )


def build_candidate_pipeline(
    name: str, parameters: dict[str, Any], *, random_seed: int
) -> Pipeline:
    """Build one deterministic candidate from an explicit parameter dictionary."""
    if name == "random_forest":
        estimator = RandomForestRegressor(random_state=random_seed, **parameters)
    elif name == "gradient_boosting":
        estimator = GradientBoostingRegressor(random_state=random_seed, **parameters)
    else:
        raise ValueError(f"Unknown candidate model: {name}")
    return make_model_pipeline(estimator)
