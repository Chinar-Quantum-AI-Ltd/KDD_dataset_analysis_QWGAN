from QWGAN_IDS.src.qwgan_adapter import QWGANAdapter, FakeQWGANAdapter
import pytest
import numpy as np


def test_qwgan_adapter_raises_when_no_checkpoint():
    with pytest.raises(RuntimeError) as exc:
        QWGANAdapter()
    assert 'PennyLane' in str(exc.value) or 'checkpoint' in str(exc.value)


def test_fake_qwgan_adapter_generate_shape():
    n_features = 8
    fake = FakeQWGANAdapter(n_features=n_features)
    samples = fake.generate(10, seed=42)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (10, n_features)
