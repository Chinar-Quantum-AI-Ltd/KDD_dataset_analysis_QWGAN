from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cqai.qwgan import QWGANConfig, QWGANTrainer, TrainingAngles


class CheckpointTests(unittest.TestCase):
    def test_model_optimizers_config_and_metadata_round_trip(self) -> None:
        config = QWGANConfig(
            n_qubits=8,
            n_layers=3,
            n_critic=1,
            seed=23,
        )
        batch = TrainingAngles.from_array(
            np.linspace(0.1, 3.0, 16).reshape(2, 8),
            config=config,
            partition="train",
            attack_class="u2r",
            latent_columns=tuple(f"z{i}" for i in range(8)),
        )
        trainer = QWGANTrainer(config, critic_hidden_dims=(8,))
        trainer.train_step(batch)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            trainer.save_checkpoint(path, metadata={"run_id": "tiny-seed23"})
            restored, metadata = QWGANTrainer.load_checkpoint(path)

        self.assertEqual(restored.config, trainer.config)
        self.assertEqual(restored.global_step, trainer.global_step)
        self.assertEqual(metadata, {"run_id": "tiny-seed23"})
        for expected, actual in zip(
            trainer.generator.parameters(), restored.generator.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(
            trainer.critic.parameters(), restored.critic.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(restored.generator_optimizer.state)
        self.assertTrue(restored.critic_optimizer.state)

        uninterrupted = trainer.train_step(batch)
        resumed = restored.train_step(batch)
        for key in set(uninterrupted) - {"wall_time_seconds"}:
            self.assertEqual(uninterrupted[key], resumed[key], key)


if __name__ == "__main__":
    unittest.main()
