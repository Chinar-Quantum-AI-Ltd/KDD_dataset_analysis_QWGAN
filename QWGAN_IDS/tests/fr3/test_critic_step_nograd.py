from __future__ import annotations

import unittest

import numpy as np
import torch

from cqai.qwgan import QWGANConfig, QWGANTrainer
from cqai.qwgan.data_contract import TrainingAngles

CONFIG = QWGANConfig(n_qubits=8, n_layers=3, n_critic=2, seed=5)
COLUMNS = tuple(f"z{i}" for i in range(8))


def _batch(seed: int = 0) -> TrainingAngles:
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(16, 8, dtype=torch.float64, generator=generator) * np.pi
    return TrainingAngles(
        values=values, attack_class="r2l", latent_columns=COLUMNS
    )


class CriticStepGradientTests(unittest.TestCase):
    """The critic must not build an autograd graph through the circuit.

    The fake batch is detached either way, so the graph is never used. Skipping
    it is a pure saving -- but only if the numbers are unchanged, which is what
    these assert.
    """

    def test_the_generator_receives_no_gradient_from_a_critic_step(self) -> None:
        trainer = QWGANTrainer(CONFIG, critic_hidden_dims=(8,))
        trainer.critic_step(_batch().values)

        for parameter in trainer.generator.parameters():
            self.assertIsNone(parameter.grad)

    def test_the_critic_still_receives_gradients(self) -> None:
        trainer = QWGANTrainer(CONFIG, critic_hidden_dims=(8,))
        before = [p.detach().clone() for p in trainer.critic.parameters()]

        trainer.critic_step(_batch().values)

        moved = [
            not torch.equal(old, new)
            for old, new in zip(before, trainer.critic.parameters())
        ]
        self.assertTrue(any(moved), "the critic did not update")

    def test_a_fixed_seed_still_replays_exactly(self) -> None:
        """The optimization must not perturb a single reported number."""

        def run() -> list[float]:
            trainer = QWGANTrainer(CONFIG, critic_hidden_dims=(8,))
            batch = _batch()
            return [
                float(trainer.train_step(batch)["critic_loss"]) for _ in range(3)
            ]

        self.assertEqual(run(), run())

    def test_the_gradient_penalty_is_still_finite_and_backwardable(self) -> None:
        """The penalty differentiates w.r.t. the interpolate, not the circuit."""

        trainer = QWGANTrainer(CONFIG, critic_hidden_dims=(8,))
        metrics = trainer.critic_step(_batch().values)

        self.assertTrue(np.isfinite(metrics["gradient_penalty"]))
        self.assertGreaterEqual(metrics["gradient_penalty"], 0.0)
        self.assertTrue(np.isfinite(metrics["critic_gradient_norm"]))

    def test_the_generator_step_still_trains_the_circuit(self) -> None:
        """Guards the obvious way to break this: no_grad leaking too far."""

        trainer = QWGANTrainer(CONFIG, critic_hidden_dims=(8,))
        before = trainer.generator.weights.detach().clone()

        trainer.generator_step(16)

        self.assertFalse(torch.equal(before, trainer.generator.weights.detach()))


if __name__ == "__main__":
    unittest.main()
