import os
import tempfile
import joblib
import numpy as np
import pandas as pd
import pytest

from QWGAN_IDS.src.transform_bundle import TransformBundle

ARTIFACT_PATH = "artifacts/transform_bundle.joblib"
ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data"


def _load_bundle():
    # Prefer an already-saved bundle; otherwise construct from artifacts
    if os.path.exists(ARTIFACT_PATH):
        return joblib.load(ARTIFACT_PATH)
    # Construct from artifacts (will raise if missing)
    return TransformBundle.load_from_artifacts(ARTIFACTS_DIR, DATA_DIR)


@pytest.mark.skipif(not os.path.exists("artifacts/encoder.pkl"), reason="FR-1 artifacts missing")
def test_schema_validation_roundtrip():
    bundle = _load_bundle()
    # load a small reference of cleaned data if available
    sample_path = os.path.join(DATA_DIR, "kdd_clean.csv")
    if not os.path.exists(sample_path):
        pytest.skip("kdd_clean.csv not present; skipping schema validation")
    df = pd.read_csv(sample_path)
    # should validate
    checks = bundle.validate_schema(df.head(5))
    assert isinstance(checks, dict) and checks.get("passed", True)  # verify returns checks dict or True

    # missing column should raise
    df2 = df.head(5).copy()
    df2.drop(columns=[bundle.feature_order[0]], inplace=True)
    with pytest.raises(Exception):
        bundle.validate_schema(df2)


@pytest.mark.skipif(not os.path.exists("artifacts/encoder.pkl"), reason="FR-1 artifacts missing")
def test_deterministic_transform():
    bundle = _load_bundle()
    sample_path = os.path.join(DATA_DIR, "kdd_clean.csv")
    if not os.path.exists(sample_path):
        pytest.skip("kdd_clean.csv not present; skipping deterministic transform test")
    df = pd.read_csv(sample_path).head(10)
    X1 = bundle.transform(df)
    X2 = bundle.transform(df)
    # numeric equality
    pd.testing.assert_frame_equal(X1, X2)


@pytest.mark.skipif(not os.path.exists("artifacts/pca.pkl"), reason="PCA artifact missing")
def test_latent_dimension_and_angle_range():
    bundle = _load_bundle()
    sample_path = os.path.join(DATA_DIR, "kdd_clean.csv")
    if not os.path.exists(sample_path):
        pytest.skip("kdd_clean.csv not present; skipping latent/angle tests")
    df = pd.read_csv(sample_path).head(20)
    latent = bundle.transform_to_latent(df)
    assert latent.shape[1] == bundle.latent_dim
    angles = bundle.transform_to_angles(df)
    assert angles.shape == (latent.shape[0], latent.shape[1])
    assert np.isfinite(angles).all()
    assert float(np.min(angles)) >= -1e-6
    assert float(np.max(angles)) <= np.pi + 1e-6


@pytest.mark.skipif(not os.path.exists("artifacts/encoder.pkl"), reason="FR-1 artifacts missing")
def test_serialization_roundtrip(tmp_path):
    bundle = _load_bundle()
    sample_path = os.path.join(DATA_DIR, "kdd_clean.csv")
    if not os.path.exists(sample_path):
        pytest.skip("kdd_clean.csv not present; skipping serialization roundtrip")
    df = pd.read_csv(sample_path).head(5)
    out1 = bundle.transform(df)
    dest = tmp_path / "bundle.joblib"
    joblib.dump(bundle, dest)
    loaded = joblib.load(dest)
    out2 = loaded.transform(df)
    pd.testing.assert_frame_equal(out1, out2)


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA_DIR, "feature_matrix.pkl")), reason="feature_matrix.pkl missing")
def test_train_serve_consistency_small():
    """Compare original feature_matrix.pkl (FR-1 output) to bundle.transform applied to the raw clean CSV.
    This validates that the saved encoder/scaler are applied identically in the bundle.
    """
    bundle = _load_bundle()
    feature_matrix = pd.read_pickle(os.path.join(DATA_DIR, "feature_matrix.pkl"))
    sample_path = os.path.join(DATA_DIR, "kdd_clean.csv")
    if not os.path.exists(sample_path):
        pytest.skip("kdd_clean.csv not present; skipping consistency test")
    df = pd.read_csv(sample_path)
    # Transform using bundle
    transformed = bundle.transform(df)
    # feature_matrix may have been saved for the same rows — compare columns intersection
    common_cols = [c for c in bundle.feature_order if c in feature_matrix.columns]
    assert common_cols, "No common columns between bundle and saved feature_matrix"
    # Compare first N rows
    n = min(20, feature_matrix.shape[0], transformed.shape[0])
    # Align columns
    fm = feature_matrix.loc[: n - 1, common_cols].reset_index(drop=True)
    tr = transformed.loc[: n - 1, common_cols].reset_index(drop=True)
    # Numerical tolerance for scaler rounding
    np.testing.assert_allclose(fm.values.astype(float), tr.values.astype(float), rtol=1e-6, atol=1e-6)
