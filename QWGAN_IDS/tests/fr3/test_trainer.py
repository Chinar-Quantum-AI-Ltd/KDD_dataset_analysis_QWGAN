from __future__ import annotations

import unittest

import numpy as np
import torch

from cqai.qwgan import QWGANConfig, QWGANTrainer, TrainingAngles


def parameters_changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, module.parameters(), strict=True)
    )


class QWGANTrainerTests(unittest.TestCase):
    def test_critic_and_generator_steps_update_parameters_independently(self) -> None:
        config = QWGANConfig(
            n_qubits=8,
            n_layers=3,
            backend="default.qubit",
            diff_method="backprop",
            n_critic=1,
            seed=17,
        )
        batch = TrainingAngles.from_array(
            np.linspace(0.0, np.pi, 32).reshape(4, 8),
            config=config,
            partition="train",
            attack_class="u2r",
            latent_columns=tuple(f"z{i}" for i in range(8)),
        )
        trainer = QWGANTrainer(config, critic_hidden_dims=(16, 8))

        generator_before = [p.detach().clone() for p in trainer.generator.parameters()]
        critic_before = [p.detach().clone() for p in trainer.critic.parameters()]
        critic_metrics = trainer.critic_step(batch.values)

        self.assertFalse(parameters_changed(generator_before, trainer.generator))
        self.assertTrue(parameters_changed(critic_before, trainer.critic))
        self.assertTrue(np.isfinite(critic_metrics["critic_loss"]))
        self.assertGreaterEqual(critic_metrics["gradient_penalty"], 0.0)

        generator_before = [p.detach().clone() for p in trainer.generator.parameters()]
        critic_before = [p.detach().clone() for p in trainer.critic.parameters()]
        generator_metrics = trainer.generator_step(batch.values.shape[0])

        self.assertTrue(parameters_changed(generator_before, trainer.generator))
        self.assertFalse(parameters_changed(critic_before, trainer.critic))
        self.assertTrue(np.isfinite(generator_metrics["generator_loss"]))
        self.assertTrue(np.isfinite(generator_metrics["generator_gradient_variance"]))

    def test_alternating_step_reports_required_diagnostics(self) -> None:
        config = QWGANConfig(
            n_qubits=8,
            n_layers=3,
            backend="default.qubit",
            diff_method="backprop",
            n_critic=1,
            seed=19,
        )
        batch = TrainingAngles.from_array(
            np.linspace(0.1, 3.0, 16).reshape(2, 8),
            config=config,
            partition="train",
            attack_class="r2l",
            latent_columns=tuple(f"z{i}" for i in range(8)),
        )
        trainer = QWGANTrainer(config, critic_hidden_dims=(8,))

        diagnostics = trainer.train_step(batch)

        expected = {
            "critic_loss",
            "generator_loss",
            "wasserstein_estimate",
            "critic_gradient_norm",
            "generator_gradient_variance",
            "gradient_penalty",
            "circuit_depth",
            "circuit_gate_count",
            "wall_time_seconds",
            "device",
            "estimated_cost_usd",
            "global_step",
            "attack_class",
        }
        self.assertEqual(set(diagnostics), expected)
        for key in expected - {"device", "attack_class"}:
            self.assertTrue(np.isfinite(diagnostics[key]), key)
        self.assertEqual(diagnostics["device"], "default.qubit")
        self.assertEqual(diagnostics["attack_class"], "r2l")
        self.assertGreater(diagnostics["circuit_depth"], 0)
        self.assertGreater(diagnostics["circuit_gate_count"], 0)
        self.assertEqual(diagnostics["estimated_cost_usd"], 0.0)
        self.assertEqual(diagnostics["global_step"], 1)

    def test_fixed_seed_replays_deterministically_on_cpu_backend(self) -> None:
        config = QWGANConfig(
            n_qubits=8,
            n_layers=3,
            backend="default.qubit",
            diff_method="backprop",
            n_critic=1,
            seed=29,
        )
        batch = TrainingAngles.from_array(
            np.linspace(0.2, 2.9, 16).reshape(2, 8),
            config=config,
            partition="train",
            attack_class="u2r",
            latent_columns=tuple(f"z{i}" for i in range(8)),
        )

        first = QWGANTrainer(config, critic_hidden_dims=(8,))
        second = QWGANTrainer(config, critic_hidden_dims=(8,))
        first_metrics = first.train_step(batch)
        second_metrics = second.train_step(batch)

        ignored = {"wall_time_seconds"}
        for key in set(first_metrics) - ignored:
            self.assertEqual(first_metrics[key], second_metrics[key], key)
        for expected, actual in zip(
            first.generator.parameters(), second.generator.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(
            first.critic.parameters(), second.critic.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
