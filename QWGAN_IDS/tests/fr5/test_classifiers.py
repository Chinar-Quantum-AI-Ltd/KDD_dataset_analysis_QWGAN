from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cqai.classifiers import (
    PyTorchDNNClassifier,
    QuantumKernelSVM,
    RFClassifier,
    XGBClassifierWrapper,
)
from cqai.lineage import runtime_versions, sha256_file


class ClassifierPanelTests(unittest.TestCase):
    """Test common interface, prediction shapes, probabilities, and serialization for all 4 FR-5 classifiers."""

    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.n_samples = 60
        self.n_features = 10
        self.X_train = rng.normal(size=(self.n_samples, self.n_features))
        self.y_train = np.array(["normal"] * 40 + ["r2l"] * 15 + ["dos"] * 5)

        self.X_test = rng.normal(size=(20, self.n_features))
        self.y_test = np.array(["normal"] * 12 + ["r2l"] * 6 + ["dos"] * 2)

    def _test_classifier_contract(self, clf) -> None:
        clf.fit(self.X_train, self.y_train)
        self.assertTrue(clf.is_fitted)

        preds = clf.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

        probs = clf.predict_proba(self.X_test)
        self.assertEqual(probs.shape, (len(self.X_test), len(clf.classes_)))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

        # Test Save & Load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.joblib"
            clf.save(save_path)
            self.assertTrue(save_path.exists())

            # Loading an executable joblib artifact requires trusted identity
            # and fitting-version provenance before deserialization.
            loaded = type(clf).load(
                save_path,
                expected_sha256=sha256_file(save_path),
                fitting_versions=runtime_versions(),
            )
            self.assertTrue(loaded.is_fitted)
            loaded_preds = loaded.predict(self.X_test)
            np.testing.assert_array_equal(preds, loaded_preds)

    def test_random_forest(self) -> None:
        rf = RFClassifier(n_estimators=10, random_state=42)
        self._test_classifier_contract(rf)

    def test_xgboost(self) -> None:
        xgb = XGBClassifierWrapper(n_estimators=10, max_depth=3, random_state=42)
        self._test_classifier_contract(xgb)

    def test_pytorch_dnn(self) -> None:
        dnn = PyTorchDNNClassifier(hidden_dim=32, epochs=5, batch_size=16, random_state=42)
        self._test_classifier_contract(dnn)

    def test_quantum_kernel_svm(self) -> None:
        qsvm = QuantumKernelSVM(n_qubits=10, max_samples=30, C=1.0, random_state=42)
        self._test_classifier_contract(qsvm)
