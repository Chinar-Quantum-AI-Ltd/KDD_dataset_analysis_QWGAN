"""Base classifier interface for FR-5 evaluation panel.

Every classifier in the panel (Random Forest, XGBoost, PyTorch FC-DNN,
Quantum-Kernel SVM) inherits from ``BaseClassifier`` and provides a uniform
fit/predict/predict_proba/save/load API.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseClassifier(ABC):
    """Abstract Base Classifier contract for FR-5."""

    def __init__(self, name: str, random_state: int = 42) -> None:
        self.name = name
        self.random_state = random_state
        self.is_fitted = False
        self.classes_: np.ndarray | None = None

    @abstractmethod
    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> BaseClassifier:
        """Fit the model on training feature matrix X and labels y."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class labels for feature matrix X."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (N, n_classes) for feature matrix X."""
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Serialize model weights and metadata to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseClassifier:
        """Deserialize model from disk."""
        pass
