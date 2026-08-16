"""Regression tests for the August 2026 TombakNet audit findings."""
from __future__ import annotations

import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from cqai.lineage.artifacts import (
    ArtifactCompatibilityError,
    ArtifactIntegrityError,
    dump_joblib_artifact,
    runtime_versions,
    sha256_file,
    verified_joblib_load,
)
from cqai.serving import ClassicalServingPipeline, benchmark_batch_latency
from cqai.serving.latency import latency_percentiles
from src.transform_bundle import TransformBundle


class RecordingTransform:
    def __init__(self, *, invalid_output: bool = False) -> None:
        self.calls = 0
        self.invalid_output = invalid_output

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        result = frame[["duration", "src_bytes"]].to_numpy(dtype=float)
        if self.invalid_output:
            result[0, 0] = np.nan
        return result


class RecordingClassifier:
    name = "random_forest"
    classes_ = np.array(["normal", "attack"])

    def __init__(self) -> None:
        self.predict_proba_calls = 0
        self.predict_calls = 0
        self.fit_calls = 0

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.predict_proba_calls += 1
        attack = np.clip(features[:, 0] / 100.0, 0.0, 1.0)
        return np.column_stack((1.0 - attack, attack))

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.predict_calls += 1
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def fit(self, *args, **kwargs):
        self.fit_calls += 1
        raise AssertionError("fit() must never run in live scoring")


def flows(count: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "duration": np.arange(count, dtype=float),
            "src_bytes": np.arange(count, dtype=float) + 1.0,
        }
    )


class ArtifactBoundaryTests(unittest.TestCase):
    def test_valid_envelope_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.joblib"
            dump_joblib_artifact({"weights": [1, 2, 3]}, path, kind="test")
            loaded = verified_joblib_load(
                path,
                expected_sha256=sha256_file(path),
                expected_type=dict,
                expected_kind="test",
                fitting_versions=runtime_versions(),
            )
            self.assertEqual(loaded, {"weights": [1, 2, 3]})

    def test_hash_mismatch_stops_before_deserializer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.joblib"
            path.write_bytes(b"tampered pickle payload")
            with patch("cqai.lineage.artifacts.joblib.load") as deserializer:
                with self.assertRaises(ArtifactIntegrityError):
                    verified_joblib_load(path, expected_sha256="0" * 64)
                deserializer.assert_not_called()

    def test_missing_or_malformed_hash_stops_before_deserializer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.joblib"
            path.write_bytes(b"payload")
            with patch("cqai.lineage.artifacts.joblib.load") as deserializer:
                with self.assertRaises(ArtifactIntegrityError):
                    verified_joblib_load(path, expected_sha256="unknown")
                deserializer.assert_not_called()

    def test_version_mismatch_stops_before_deserializer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.joblib"
            path.write_bytes(b"payload")
            with patch("cqai.lineage.artifacts.joblib.load") as deserializer:
                with self.assertRaises(ArtifactCompatibilityError):
                    verified_joblib_load(
                        path,
                        expected_sha256=sha256_file(path),
                        fitting_versions={"scikit_learn": "0.0.invalid"},
                        require_envelope=False,
                    )
                deserializer.assert_not_called()

    def test_missing_external_version_metadata_stops_before_deserializer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.joblib"
            dump_joblib_artifact({"weights": []}, path, kind="test")
            with patch("cqai.lineage.artifacts.joblib.load") as deserializer:
                with self.assertRaises(ArtifactCompatibilityError):
                    verified_joblib_load(
                        path,
                        expected_sha256=sha256_file(path),
                        expected_kind="test",
                    )
                deserializer.assert_not_called()

    def test_feature_name_array_does_not_need_pickle(self) -> None:
        project = Path(__file__).resolve().parents[2]
        names = np.load(
            project / "data/selected_feature_names.npy", allow_pickle=False
        )
        self.assertTrue(np.issubdtype(names.dtype, np.str_))


class ServingBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transform = RecordingTransform()
        self.classifier = RecordingClassifier()
        self.pipeline = ClassicalServingPipeline(
            self.classifier, self.transform, max_batch_size=16
        )

    def test_transform_and_one_classifier_traversal_only(self) -> None:
        result = self.pipeline.predict_flow(flows(8))
        self.assertEqual(result["flow_count"], 8)
        self.assertEqual(self.transform.calls, 1)
        self.assertEqual(self.classifier.predict_proba_calls, 1)
        self.assertEqual(self.classifier.predict_calls, 0)
        self.assertEqual(self.classifier.fit_calls, 0)

    def test_probability_argmax_is_equivalent_to_predict(self) -> None:
        matrix = self.transform.transform(flows(8))
        expected = self.classifier.predict(matrix)
        self.classifier.predict_calls = 0
        self.classifier.predict_proba_calls = 0
        actual = self.pipeline.predict_flow(flows(8))["predicted_class"]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.classifier.predict_calls, 0)
        self.assertEqual(self.classifier.predict_proba_calls, 1)

    def test_transformer_is_mandatory(self) -> None:
        with self.assertRaises(ValueError):
            ClassicalServingPipeline(self.classifier, None)

    def test_nan_inf_negative_and_oversized_requests_fail_closed(self) -> None:
        invalid = flows(2)
        invalid.loc[0, "duration"] = np.nan
        with self.assertRaises(ValueError):
            self.pipeline.predict_flow(invalid)
        invalid = flows(2)
        invalid.loc[0, "duration"] = np.inf
        with self.assertRaises(ValueError):
            self.pipeline.predict_flow(invalid)
        invalid = flows(2)
        invalid.loc[0, "duration"] = -1e9
        with self.assertRaises(ValueError):
            self.pipeline.predict_flow(invalid)
        with self.assertRaises(ValueError):
            self.pipeline.predict_flow(flows(17))

    def test_invalid_transform_output_fails_closed(self) -> None:
        pipeline = ClassicalServingPipeline(
            self.classifier, RecordingTransform(invalid_output=True)
        )
        with self.assertRaises(ValueError):
            pipeline.predict_flow(flows(2))

    def test_quantum_and_generator_modules_are_unreachable(self) -> None:
        forbidden = {
            name: None
            for name in (
                "pennylane",
                "cqai.qwgan.generator",
                "src.classical_wgan_gp",
                "src.augmentation.classical_wgan_gp",
            )
        }
        with patch.dict(sys.modules, forbidden):
            result = self.pipeline.predict_flow(flows(2))
        self.assertTrue(result["is_quantum_free"])

    def test_quantum_or_generator_classifier_identity_is_rejected(self) -> None:
        for name in ("quantum_kernel_svm", "qwgan", "classical_wgan_generator"):
            candidate = RecordingClassifier()
            candidate.name = name
            with self.assertRaises(TypeError):
                ClassicalServingPipeline(candidate, self.transform)

    def test_unknown_categories_are_rejected_by_transform_bundle(self) -> None:
        class Encoder:
            categories_ = [
                np.array(["tcp"]),
                np.array(["http"]),
                np.array(["SF"]),
            ]

        bundle = TransformBundle(
            encoder=Encoder(), scaler=object(), pca=None,
            feature_names=[], selected_feature_names=[]
        )
        bundle.validate_schema = lambda frame: {"passed": True}
        raw = pd.DataFrame(
            {
                "protocol_type": ["tcp"],
                "service": ["attacker_service"],
                "flag": ["SF"],
                "duration": [0.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "out-of-distribution"):
            bundle._validate_raw_input(raw)


class LatencyTests(unittest.TestCase):
    def test_percentile_calculation(self) -> None:
        stats = latency_percentiles(range(1, 101))
        self.assertAlmostEqual(stats["p50_ms"], 50.5)
        self.assertAlmostEqual(stats["p95_ms"], 95.05)
        self.assertAlmostEqual(stats["p99_ms"], 99.01)

    def test_sla_uses_request_latency_not_amortized_latency(self) -> None:
        pipeline = ClassicalServingPipeline(
            RecordingClassifier(), RecordingTransform(), max_batch_size=16
        )
        clock = []
        for _ in range(100):
            clock.extend((0, 60_000_000))
        with patch("cqai.serving.latency.time.perf_counter_ns", side_effect=clock):
            report = benchmark_batch_latency(
                pipeline,
                flows(32),
                batch_sizes=(16,),
                warmup_runs=0,
                measured_runs=100,
                max_p99_ms=50.0,
                raise_on_sla=False,
            )
        row = report["summary"][0]
        self.assertEqual(row["p99_ms"], 60.0)
        self.assertEqual(row["mean_amortized_ms_per_flow"], 3.75)
        self.assertFalse(row["sla_pass"])
        self.assertFalse(report["overall"]["sla_pass"])


class RepositoryAuditPolicyTests(unittest.TestCase):
    def test_only_central_helper_calls_joblib_load(self) -> None:
        project = Path(__file__).resolve().parents[2]
        offenders = []
        for path in project.rglob("*.py"):
            if "tests" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if "joblib.load(" in text and path.name != "artifacts.py":
                offenders.append(str(path.relative_to(project)))
        self.assertEqual(offenders, [])

    def test_qml_chunking_assertion_uses_physics_tolerance(self) -> None:
        project = Path(__file__).resolve().parents[2]
        text = (project / "tests/fr4/test_generate.py").read_text(encoding="utf-8")
        method = text.split("def test_chunking_does_not_change_the_result", 1)[1]
        method = method.split("def test_the_same_seed", 1)[0]
        self.assertIn("assert_allclose", method)
        self.assertIn("atol=1e-12", method)

    def test_xgboost_default_is_single_threaded(self) -> None:
        project = Path(__file__).resolve().parents[2]
        text = (project / "cqai/classifiers/xgboost_model.py").read_text(encoding="utf-8")
        self.assertIn("n_jobs: int = 1", text)
        self.assertNotIn("n_jobs: int = -1", text)

    def test_runtime_requirements_are_exactly_pinned(self) -> None:
        project = Path(__file__).resolve().parents[2]
        lines = (project / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
        packages = [line for line in lines if line and not line.startswith("#")]
        self.assertTrue(packages)
        self.assertTrue(all("==" in line for line in packages))

    def test_no_tracked_python_bytecode(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            ["git", "ls-files"], cwd=repository, capture_output=True, text=True, check=True
        )
        tracked = [
            line for line in result.stdout.splitlines()
            if "__pycache__" in line or line.endswith(".pyc")
        ]
        self.assertEqual(tracked, [])


if __name__ == "__main__":
    unittest.main()
