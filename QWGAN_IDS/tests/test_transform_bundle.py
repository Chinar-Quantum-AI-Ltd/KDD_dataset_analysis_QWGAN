import os
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd

from src.transform_bundle import TransformBundle

ARTIFACT_PATH = "artifacts/transform_bundle.joblib"
ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data"


def _load_bundle():
    if os.path.exists(ARTIFACT_PATH):
        return joblib.load(ARTIFACT_PATH)
    return TransformBundle.load_from_artifacts(ARTIFACTS_DIR, DATA_DIR)


class TestTransformBundle(unittest.TestCase):
    @unittest.skipIf(not os.path.exists("artifacts/encoder.pkl"), "FR-1 artifacts missing")
    def test_schema_validation_roundtrip(self):
        bundle = _load_bundle()
        sample_path = os.path.join(DATA_DIR, "kdd_clean.csv")
        if not os.path.exists(sample_path):
            self.skipTest("kdd_clean.csv not present; skipping schema validation")
        df = pd.read_csv(sample_path)
        checks = bundle.validate_schema(df.head(5))
        self.assertTrue(isinstance(checks, dict) and checks.get("passed", True))

        df2 = df.head(5).copy()
        df2.drop(columns=[bundle.feature_order[0]], inplace=True)
        with self.assertRaises(Exception):
            bundle.validate_schema(df2)

    @unittest.skipIf(not os.path.exists("artifacts/encoder.pkl"), "FR-1 artifacts missing")
    def test_deterministic_transform(self):
        bundle = _load_bundle()
        sample_path = os.path.join(DATA_DIR, "kdd_clean.csv")
        if not os.path.exists(sample_path):
            self.skipTest("kdd_clean.csv not present; skipping deterministic transform test")
        df = pd.read_csv(sample_path).head(10)
        X1 = bundle.transform(df)
        X2 = bundle.transform(df)
        pd.testing.assert_frame_equal(X1, X2)

    @unittest.skipIf(not os.path.exists("artifacts/pca.pkl"), "PCA artifact missing")
    def test_latent_dimension_and_angle_range(self):
        bundle = _load_bundle()
        sample_path = os.path.join(DATA_DIR, "kdd_clean.csv")
        if not os.path.exists(sample_path):
            self.skipTest("kdd_clean.csv not present; skipping latent/angle tests")
        df = pd.read_csv(sample_path).head(20)
        latent = bundle.transform_to_latent(df)
        self.assertEqual(latent.shape[1], bundle.latent_dim)
        angles = bundle.transform_to_angles(df)
        self.assertEqual(angles.shape, (latent.shape[0], latent.shape[1]))
        self.assertTrue(np.isfinite(angles).all())
        self.assertGreaterEqual(float(np.min(angles)), -1e-6)
        self.assertLessEqual(float(np.max(angles)), np.pi + 1e-6)

    @unittest.skipIf(not os.path.exists("artifacts/encoder.pkl"), "FR-1 artifacts missing")
    def test_serialization_roundtrip(self):
        bundle = _load_bundle()
        sample_path = os.path.join(DATA_DIR, "kdd_clean.csv")
        if not os.path.exists(sample_path):
            self.skipTest("kdd_clean.csv not present; skipping serialization roundtrip")
        df = pd.read_csv(sample_path).head(5)
        out1 = bundle.transform(df)
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = os.path.join(tmp_dir, "bundle.joblib")
            joblib.dump(bundle, dest)
            loaded = joblib.load(dest)
            out2 = loaded.transform(df)
            pd.testing.assert_frame_equal(out1, out2)

    @unittest.skipIf(not os.path.exists(os.path.join(DATA_DIR, "feature_matrix.pkl")), "feature_matrix.pkl missing")
    def test_train_serve_consistency_small(self):
        bundle = _load_bundle()
        feature_matrix = pd.read_pickle(os.path.join(DATA_DIR, "feature_matrix.pkl"))
        sample_path = os.path.join(DATA_DIR, "kdd_clean.csv")
        if not os.path.exists(sample_path):
            self.skipTest("kdd_clean.csv not present; skipping consistency test")
        df = pd.read_csv(sample_path)
        transformed = bundle.transform(df)
        common_cols = [c for c in bundle.feature_order if c in feature_matrix.columns]
        self.assertTrue(bool(common_cols), "No common columns between bundle and saved feature_matrix")
        n = min(20, feature_matrix.shape[0], transformed.shape[0])
        fm = feature_matrix.loc[: n - 1, common_cols].reset_index(drop=True)
        tr = transformed.loc[: n - 1, common_cols].reset_index(drop=True)
        np.testing.assert_allclose(fm.values.astype(float), tr.values.astype(float), rtol=1e-6, atol=1e-6)
