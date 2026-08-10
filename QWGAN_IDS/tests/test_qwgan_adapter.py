import unittest
import numpy as np
from src.qwgan_adapter import QWGANAdapter, FakeQWGANAdapter


class TestQWGANAdapter(unittest.TestCase):
    def test_qwgan_adapter_raises_when_no_checkpoint(self):
        with self.assertRaises(RuntimeError) as exc:
            QWGANAdapter()
        self.assertTrue('PennyLane' in str(exc.exception) or 'checkpoint' in str(exc.exception))

    def test_fake_qwgan_adapter_generate_shape(self):
        n_features = 8
        fake = FakeQWGANAdapter(n_features=n_features)
        samples = fake.generate(10, seed=42)
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.shape, (10, n_features))
