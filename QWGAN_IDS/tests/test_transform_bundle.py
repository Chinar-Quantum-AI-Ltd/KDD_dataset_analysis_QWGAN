import os
import tempfile
import joblib
from QWGAN_IDS.src.transform_bundle import save_transform_bundle, load_transform_bundle


def test_transform_bundle_roundtrip():
    bundle = {
        'encoder': None,
        'scaler': None,
        'feature_selector': None,
        'pca': None,
        'categorical_columns': [],
        'numerical_columns': [],
        'feature_metadata': {},
        'angle_scaling': {},
    }
    td = tempfile.mkdtemp()
    path = os.path.join(td, 'bundle.joblib')
    save_transform_bundle(path, bundle)
    loaded = load_transform_bundle(path)
    assert set(loaded.keys()) == set(bundle.keys())
