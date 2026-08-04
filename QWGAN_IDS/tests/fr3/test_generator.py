from __future__ import annotations

import unittest

import torch

from cqai.qwgan import DecodedQuantumGenerator, QWGANConfig, QuantumGenerator


class QuantumGeneratorTests(unittest.TestCase):
    def test_batch_output_is_bounded_and_differentiable(self) -> None:
        config = QWGANConfig(
            n_qubits=8,
            n_layers=3,
            backend="default.qubit",
            diff_method="backprop",
            seed=7,
        )
        generator = QuantumGenerator(config)
        noise = torch.rand(2, 8, dtype=torch.float64, requires_grad=True)

        generated = generator(noise)
        generated.sum().backward()

        self.assertEqual(generated.shape, (2, 8))
        self.assertEqual(generated.dtype, torch.float64)
        self.assertTrue(torch.isfinite(generated).all())
        self.assertGreaterEqual(float(generated.detach().min()), -1.0)
        self.assertLessEqual(float(generated.detach().max()), 1.0)
        self.assertIsNotNone(noise.grad)
        self.assertGreater(float(noise.grad.abs().sum()), 0.0)
        self.assertIsNotNone(generator.weights.grad)
        self.assertGreater(float(generator.weights.grad.abs().sum()), 0.0)

    def test_gradients_flow_through_explicit_angle_domain_decoder(self) -> None:
        config = QWGANConfig(
            n_qubits=8,
            n_layers=3,
            backend="default.qubit",
            diff_method="backprop",
            seed=11,
        )
        quantum = QuantumGenerator(config)
        decoder = torch.nn.Linear(8, 12, dtype=torch.float64)
        generator = DecodedQuantumGenerator(quantum, decoder)

        decoded = generator(torch.rand(2, 8, dtype=torch.float64))
        decoded.square().mean().backward()

        self.assertEqual(decoded.shape, (2, 12))
        self.assertGreater(float(quantum.weights.grad.abs().sum()), 0.0)
        self.assertGreater(float(decoder.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
