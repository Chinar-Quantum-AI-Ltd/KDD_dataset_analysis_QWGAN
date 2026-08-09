"""XGBoost classifier wrapper for FR-5.

Provides XGBoost classifier adhering to ``BaseClassifier``.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .base import BaseClassifier


class XGBClassifierWrapper(BaseClassifier):
    """XGBoost Classifier wrapper for tabular network flow data."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(name="xgboost", random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.label_encoder = LabelEncoder()
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="logloss",
        )

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> XGBClassifierWrapper:
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

        y_encoded = self.label_encoder.fit_transform(y_arr)
        self.classes_ = self.label_encoder.classes_

        self.model.fit(X_arr, y_encoded)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        preds_int = self.model.predict(X_arr)
        return self.label_encoder.inverse_transform(preds_int)

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
    def load(cls, path: str | Path) -> XGBClassifierWrapper:
        obj = joblib.load(path)
        if not isinstance(obj, XGBClassifierWrapper):
            raise TypeError(f"Loaded object is not XGBClassifierWrapper: {type(obj)}")
        return obj
