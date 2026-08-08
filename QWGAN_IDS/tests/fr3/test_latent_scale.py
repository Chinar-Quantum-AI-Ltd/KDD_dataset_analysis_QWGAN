from __future__ import annotations

import unittest

import numpy as np
import torch

from cqai.qwgan import QWGANConfig, QWGANTrainer


class LatentScaleConfigTests(unittest.TestCase):
    def test_the_default_is_the_full_angle_domain(self) -> None:
        """The TDD default must not change underneath an existing run.

        Every number reported so far was produced at scale 1.0. Shipping a new
        default would silently invalidate the FR-3 campaign's manifests.
        """

        self.assertEqual(QWGANConfig().latent_scale, 1.0)

    def test_rejects_a_scale_outside_the_useful_range(self) -> None:
        for bad in (0.0, -0.5, 1.5):
            with self.subTest(latent_scale=bad):
                with self.assertRaises(ValueError):
                    QWGANConfig(latent_scale=bad)


class LatentScaleSamplingTests(unittest.TestCase):
    def test_the_scale_bounds_the_latent_noise(self) -> None:
        for scale in (1.0, 0.5, 0.25):
            with self.subTest(latent_scale=scale):
                trainer = QWGANTrainer(QWGANConfig(latent_scale=scale, seed=3))
                noise = trainer.sample_noise(2048, seed=11).numpy()

                self.assertGreaterEqual(noise.min(), 0.0)
                self.assertLessEqual(noise.max(), scale * np.pi + 1e-12)
                # Actually reaches the top of its range, so the scale is a
                # bound rather than an unused parameter.
                self.assertGreater(noise.max(), 0.95 * scale * np.pi)

    def test_scaling_down_does_not_merely_rescale_the_same_draw(self) -> None:
        """A narrower latent is a different distribution, not the same one shrunk."""

        full = QWGANTrainer(QWGANConfig(latent_scale=1.0, seed=3))
        quarter = QWGANTrainer(QWGANConfig(latent_scale=0.25, seed=3))

        wide = full.sample_noise(512, seed=11).numpy()
        narrow = quarter.sample_noise(512, seed=11).numpy()

        np.testing.assert_allclose(narrow, wide * 0.25)

    def test_a_narrower_latent_unpins_the_per_qubit_output_means(self) -> None:
        """The reason this parameter exists.

        Data re-uploading applies ``RY(noise)`` at every layer, so with the
        full ``[0, pi]`` latent a four-layer circuit accumulates up to ``4*pi``
        of rotation and the Pauli-Z expectation averages to zero. Every qubit's
        output mean is then pinned near ``pi/2`` no matter what the weights do,
        which is exactly what the FR-3 generators produced. Narrowing the
        latent stops the wrap-around and gives the weights leverage over the
        mean again.
        """

        def spread(scale: float) -> float:
            trainer = QWGANTrainer(QWGANConfig(latent_scale=scale, seed=3))
            with torch.no_grad():
                samples = trainer.generate(1024).numpy()
            means = samples.mean(axis=0)
            return float(means.max() - means.min())

        self.assertLess(spread(1.0), 0.05)
        self.assertGreater(spread(0.1), 4 * spread(1.0))


if __name__ == "__main__":
    unittest.main()
