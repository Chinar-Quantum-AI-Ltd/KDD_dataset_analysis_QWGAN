"""Evaluation Harness & Manifest Generator for FR-5 / FR-8.

Provides ``EvaluationHarness`` to evaluate trained classifiers on held-out test
partitions, enforcing zero test augmentation and generating FR-8 compliant manifests.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..classifiers.base import BaseClassifier
from .metrics import evaluate_classifier_metrics


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EvaluationHarness:
    """Evaluation Harness for FR-5 / FR-8."""

    def __init__(self, run_id: str, output_dir: str | Path) -> None:
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        classifier: BaseClassifier,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
        arm_name: str = "A_real_only",
        seed: int = 42,
        dataset_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a classifier on held-out test data and persist the FR-8 manifest.

        Parameters
        ----------
        classifier :
            Trained classifier instance inheriting from ``BaseClassifier``.
        X_test :
            Held-out real test feature matrix.
        y_test :
            Held-out real test ground truth labels.
        arm_name :
            Ablation arm identifier (e.g. A_real_only, B_smote, C_cwgan, D_qwgan).
        seed :
            Random seed used for this evaluation run.
        dataset_meta :
            Metadata dictionary (dataset_id, partition_id, hashes).

        Returns
        -------
        Dictionary containing evaluation results and manifest path.
        """
        start_time = datetime.now(timezone.utc)

        # 1. Predictions
        preds = classifier.predict(X_test)
        probs: np.ndarray | None = None
        try:
            probs = classifier.predict_proba(X_test)
        except Exception:
            pass

        # 2. Metrics calculation
        classes = classifier.classes_.tolist() if classifier.classes_ is not None else None
        metrics_res = evaluate_classifier_metrics(y_test, preds, y_prob=probs, classes=classes)

        end_time = datetime.now(timezone.utc)

        # 3. FR-8 Manifest
        y_test_bytes = (
            y_test.to_numpy().tobytes() if isinstance(y_test, pd.Series) else np.asarray(y_test).tobytes()
        )
        test_hash = _sha256_bytes(y_test_bytes)

        manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "arm_name": arm_name,
            "classifier_name": classifier.name,
            "seed": seed,
            "start_time_utc": start_time.isoformat(),
            "end_time_utc": end_time.isoformat(),
            "test_sample_count": len(X_test),
            "test_labels_sha256": test_hash,
            "dataset_metadata": dataset_meta or {},
            "metrics": metrics_res,
        }

        # Save manifest
        manifest_path = self.output_dir / f"eval_{classifier.name}_{arm_name}_seed{seed}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        return {"manifest_path": str(manifest_path), "metrics": metrics_res, "manifest": manifest}
