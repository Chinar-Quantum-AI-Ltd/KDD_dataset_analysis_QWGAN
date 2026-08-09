"""Random Forest classifier wrapper for FR-5.

Provides a balanced RandomForestClassifier adhering to ``BaseClassifier``.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .base import BaseClassifier


class RFClassifier(BaseClassifier):
    """Random Forest Classifier for tabular network flow data."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(name="random_forest", random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> RFClassifier:
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

        self.model.fit(X_arr, y_arr)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict(X_arr)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict_proba(X_arr)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> RFClassifier:
        obj = joblib.load(path)
        if not isinstance(obj, RFClassifier):
            raise TypeError(f"Loaded object is not RFClassifier: {type(obj)}")
        return obj
