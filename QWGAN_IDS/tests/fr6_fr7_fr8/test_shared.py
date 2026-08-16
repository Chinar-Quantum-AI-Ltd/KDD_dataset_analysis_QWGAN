"""Unit and integration tests for FR-6, FR-7, and FR-8."""
from __future__ import annotations

import tempfile
import unittest
import numpy as np
import pandas as pd

from cqai.ablation import AblationRunner, build_ablation_arm, compute_paired_ttest
from cqai.classifiers import RFClassifier
from cqai.lineage import build_lineage_manifest
from cqai.serving import ClassicalServingPipeline, benchmark_serving_latency


class _IdentityTransform:
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.to_numpy(dtype=float)


class TestFR6Ablation(unittest.TestCase):
    def test_build_ablation_arm_shapes(self) -> None:
        rng = np.random.default_rng(42)
        X_train = rng.normal(size=(50, 10))
        y_train = np.array(["normal"] * 35 + ["r2l"] * 15)

        # Arm A: Real only
        Xa, ya = build_ablation_arm("A", X_train, y_train, target_class="r2l")
        self.assertEqual(len(Xa), 50)
        self.assertEqual(len(ya), 50)

        # Arm B: SMOTE
        Xb, yb = build_ablation_arm("B", X_train, y_train, target_class="r2l", target_ratio=0.50, seed=42)
        self.assertGreaterEqual(len(Xb), 50)
        self.assertGreaterEqual((yb == "r2l").sum(), 15)

    def test_paired_ttest_math(self) -> None:
        d = [0.90, 0.92, 0.91]
        a = [0.80, 0.81, 0.79]
        t_stat, p_val = compute_paired_ttest(d, a)
        self.assertGreater(t_stat, 0.0)
        self.assertLess(p_val, 0.05)

    def test_ablation_runner(self) -> None:
        rng = np.random.default_rng(42)
        X_tr = rng.normal(size=(40, 10))
        y_tr = np.array(["normal"] * 30 + ["r2l"] * 10)
        X_te = rng.normal(size=(10, 10))
        y_te = np.array(["normal"] * 7 + ["r2l"] * 3)

        runner = AblationRunner(
            X_tr,
            y_tr,
            X_te,
            y_te,
            target_class="r2l",
            seeds=[42, 123],
            classifiers=[RFClassifier(n_estimators=5, random_state=42)],
        )
        report = runner.run_ablation()
        self.assertIn("summary_by_classifier", report)
        self.assertIn("random_forest", report["summary_by_classifier"])


class TestFR7Serving(unittest.TestCase):
    def test_serving_pipeline_predict(self) -> None:
        rng = np.random.default_rng(42)
        X_tr = rng.normal(size=(30, 8))
        y_tr = np.array(["normal"] * 20 + ["r2l"] * 10)
        X_te = pd.DataFrame(np.abs(rng.normal(size=(5, 8))))

        clf = RFClassifier(n_estimators=5, random_state=42)
        clf.fit(X_tr, y_tr)

        pipeline = ClassicalServingPipeline(
            classifier=clf, transformer=_IdentityTransform()
        )
        res = pipeline.predict_flow(X_te)
        self.assertTrue(res["is_quantum_free"])
        self.assertEqual(res["flow_count"], 5)

    def test_latency_benchmark_sla(self) -> None:
        rng = np.random.default_rng(42)
        X_tr = rng.normal(size=(30, 8))
        y_tr = np.array(["normal"] * 20 + ["r2l"] * 10)
        X_te = pd.DataFrame(np.abs(rng.normal(size=(2, 8))))

        clf = RFClassifier(n_estimators=5, random_state=42)
        clf.fit(X_tr, y_tr)

        pipeline = ClassicalServingPipeline(
            classifier=clf, transformer=_IdentityTransform()
        )
        bench = benchmark_serving_latency(
            pipeline, X_te, n_iterations=100, max_p99_ms=50.0
        )
        self.assertTrue(bench["sla_passed"])
        self.assertLessEqual(bench["p99_ms"], 50.0)


class TestFR8Lineage(unittest.TestCase):
    def test_build_lineage_manifest(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("dummy artifact content")
            tmp_path = tmp.name

        manifest = build_lineage_manifest(
            run_id="run_123",
            run_type="fr6_ablation",
            dataset_name="NSL-KDD",
            metrics={"f1": 0.95},
            artifact_paths=[tmp_path],
        )

        self.assertEqual(manifest["run_id"], "run_123")
        self.assertIn("git_commit_sha", manifest)
        self.assertIn(tmp_path, manifest["artifact_hashes"])
