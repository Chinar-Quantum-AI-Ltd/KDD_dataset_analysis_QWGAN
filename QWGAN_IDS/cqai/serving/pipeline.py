"""FR-7 Pure Classical Live Serving Pipeline.

Enforces pure classical live scoring path with zero live quantum calls.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from cqai.classifiers import BaseClassifier


class ClassicalServingPipeline:
    """Offline-prepared pure classical streaming inference pipeline."""

    def __init__(
        self,
        classifier: BaseClassifier,
        transformer: Any | None = None,
    ) -> None:
        if classifier is None:
            raise ValueError("Serving pipeline requires a registered classical classifier model.")
        self.classifier = classifier
        self.transformer = transformer

    def predict_flow(self, flow_input: pd.DataFrame | np.ndarray) -> dict[str, Any]:
        """Score a network flow or batch of network flows.

        Parameters
        ----------
        flow_input : pd.DataFrame | np.ndarray
            Raw network flow features.

        Returns
        -------
        dict[str, Any]
            Inference result containing predicted_class, threat_probabilities, and flow_count.
        """
        if self.transformer is not None:
            features = self.transformer.transform(flow_input)
        else:
            features = np.asarray(flow_input)

        if isinstance(features, pd.DataFrame):
            features = features.values

        predictions = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)

        return {
            "predicted_class": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "flow_count": len(predictions),
            "is_quantum_free": True,
        }
