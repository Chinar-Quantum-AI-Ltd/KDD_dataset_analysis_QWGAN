"""FR-7 pure-classical, fail-closed live scoring pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from cqai.classifiers.base import BaseClassifier
else:
    BaseClassifier = Any


_PROHIBITED_IDENTITIES = (
    "quantum",
    "qnode",
    "pennylane",
    "qwgan",
    "wgan",
    "generator",
)


class ClassicalServingPipeline:
    """Registered transform plus registered classical classifier only.

    Artifact loading, fitting, augmentation, generation, and quantum execution
    are intentionally absent from this class and must happen offline.
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        transformer: Any,
        *,
        max_batch_size: int = 1024,
    ) -> None:
        if classifier is None:
            raise ValueError("A registered classical classifier is mandatory")
        if transformer is None or not callable(getattr(transformer, "transform", None)):
            raise ValueError("A registered transform bundle with transform() is mandatory")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        identity = " ".join(
            (
                classifier.__class__.__module__,
                classifier.__class__.__name__,
                str(getattr(classifier, "name", "")),
            )
        ).lower()
        rejected = [token for token in _PROHIBITED_IDENTITIES if token in identity]
        if rejected:
            raise TypeError(
                "Live scoring requires a classical classifier; rejected identity "
                f"tokens: {rejected}"
            )
        if not callable(getattr(classifier, "predict_proba", None)):
            raise TypeError("Registered classical classifier must implement predict_proba()")
        if getattr(classifier, "classes_", None) is None:
            raise ValueError("Registered classifier has no fitted classes_ metadata")

        self.classifier = classifier
        self.transformer = transformer
        self.max_batch_size = int(max_batch_size)

    def _validate_request(self, flow_input: pd.DataFrame) -> None:
        if not isinstance(flow_input, pd.DataFrame):
            raise TypeError("Live flow input must be a pandas DataFrame")
        if flow_input.empty:
            raise ValueError("Live flow batch must contain at least one row")
        if len(flow_input) > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(flow_input)} exceeds maximum {self.max_batch_size}"
            )
        numeric = flow_input.select_dtypes(include=[np.number])
        if numeric.shape[1] and not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("Live flow contains NaN or infinite numeric values")
        # NSL-KDD raw counters, flags, byte counts, and rates are non-negative.
        # This rejects the audit's -1e9 sentinel before it can be scored benign.
        if numeric.shape[1] and bool((numeric < 0).any().any()):
            raise ValueError("Live flow contains negative raw-feature sentinel values")

    def predict_flow(self, flow_input: pd.DataFrame) -> dict[str, Any]:
        """Measureable path: validate -> transform -> one classical traversal."""

        self._validate_request(flow_input)
        transformed = self.transformer.transform(flow_input)
        features = (
            transformed.to_numpy(dtype=np.float64, copy=False)
            if isinstance(transformed, pd.DataFrame)
            else np.asarray(transformed)
        )
        if features.ndim != 2 or features.shape[0] != len(flow_input):
            raise ValueError(
                "Transform output shape does not match request: "
                f"input_rows={len(flow_input)}, output_shape={features.shape}"
            )
        if not np.issubdtype(features.dtype, np.number):
            raise ValueError("Transform output must be numeric")
        if not np.isfinite(features).all():
            raise ValueError("Transform output contains NaN or infinite values")

        # PERF-01: one forest traversal.  predict() would traverse every tree a
        # second time; labels are exactly recoverable from the probabilities.
        probabilities = np.asarray(self.classifier.predict_proba(features))
        classes = np.asarray(self.classifier.classes_)
        if probabilities.ndim != 2 or probabilities.shape != (len(features), len(classes)):
            raise ValueError(
                "Classifier probability shape does not match rows/classes: "
                f"{probabilities.shape}"
            )
        if not np.isfinite(probabilities).all():
            raise ValueError("Classifier returned non-finite probabilities")
        predictions = classes[np.argmax(probabilities, axis=1)]

        return {
            "predicted_class": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "flow_count": int(len(predictions)),
            "is_quantum_free": True,
        }
