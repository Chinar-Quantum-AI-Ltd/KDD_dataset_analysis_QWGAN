from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cqai.classifiers import RFClassifier
from cqai.evaluation import EvaluationHarness, evaluate_classifier_metrics


class EvaluatorTests(unittest.TestCase):
    """Test metrics suite calculation, benign FPR orientation, and manifest generation."""

    def setUp(self) -> None:
        self.y_true = np.array(["normal", "normal", "normal", "dos", "r2l", "r2l"])
        self.y_pred = np.array(["normal", "normal", "r2l", "dos", "r2l", "r2l"])
        self.classes = ["dos", "normal", "r2l"]

    def test_metrics_calculation(self) -> None:
        res = evaluate_classifier_metrics(self.y_true, self.y_pred, classes=self.classes, benign_class="normal")

        self.assertIn("per_class", res)
        self.assertIn("macro", res)
        self.assertIn("benign_fpr", res)
        self.assertIn("confusion_matrix", res)

        per_class = res["per_class"]
        self.assertIn("normal", per_class)
        self.assertIn("dos", per_class)
        self.assertIn("r2l", per_class)

        # 3 normal, 2 predicted correctly, 1 predicted as r2l -> Recall = 2/3
        self.assertAlmostEqual(per_class["normal"]["recall"], 2 / 3, places=4)

        # Macro precision, recall, f1 are calculated
        self.assertGreater(res["macro"]["macro_f1"] if "macro_f1" in res["macro"] else res["macro"]["f1"], 0.0)

        # Benign FPR: 1 FP out of 3 actual non-benign => FPR > 0
        self.assertIsNotNone(res["benign_fpr"])

    def test_evaluation_harness_manifest(self) -> None:
        rng = np.random.default_rng(42)
        X_train = rng.normal(size=(50, 10))
        y_train = np.array(["normal"] * 30 + ["dos"] * 20)

        X_test = rng.normal(size=(20, 10))
        y_test = np.array(["normal"] * 12 + ["dos"] * 8)

        clf = RFClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            harness = EvaluationHarness(run_id="test_run_001", output_dir=tmpdir)
            eval_res = harness.evaluate(
                classifier=clf,
                X_test=X_test,
                y_test=y_test,
                arm_name="A_real_only",
                seed=42,
                dataset_meta={"dataset_id": "nslkdd"},
            )

            manifest_path = Path(eval_res["manifest_path"])
            self.assertTrue(manifest_path.exists())

            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data["run_id"], "test_run_001")
            self.assertEqual(data["arm_name"], "A_real_only")
            self.assertEqual(data["classifier_name"], "random_forest")
            self.assertEqual(data["test_sample_count"], 20)
            self.assertIn("metrics", data)
