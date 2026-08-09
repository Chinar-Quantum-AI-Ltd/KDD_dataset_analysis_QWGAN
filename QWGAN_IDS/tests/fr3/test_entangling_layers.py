from __future__ import annotations

import unittest

import numpy as np
import torch

from cqai.qwgan import QWGANConfig, QuantumGenerator


class EntanglingLayerConfigTests(unittest.TestCase):
    def test_the_default_entangles_every_layer(self) -> None:
        """The TDD's structure stays the default.

        Every reported run entangled at every layer. A new default would change
        what the existing manifests mean without anyone choosing it.
        """

        self.assertIsNone(QWGANConfig().entangling_layers)

    def test_rejects_a_layer_index_the_circuit_does_not_have(self) -> None:
        with self.assertRaises(ValueError):
            QWGANConfig(n_layers=4, entangling_layers=(4,))
        with self.assertRaises(ValueError):
            QWGANConfig(n_layers=4, entangling_layers=(-1,))

    def test_rejects_a_duplicated_layer(self) -> None:
        with self.assertRaises(ValueError):
            QWGANConfig(n_layers=4, entangling_layers=(1, 1))

    def test_an_empty_tuple_is_allowed_and_means_no_entanglement(self) -> None:
        """Explicitly separable, because it is a real experimental arm.

        It is also the arm that forfeits the quantum claim, which is why it has
        to be spelled out rather than reachable by accident.
        """

        config = QWGANConfig(n_layers=4, entangling_layers=())
        self.assertEqual(config.entangling_layers, ())


class EntanglingLayerCircuitTests(unittest.TestCase):
    def _cnot_count(self, config: QWGANConfig) -> int:
        generator = QuantumGenerator(config)
        drawing = str(
            __import__("pennylane").draw(generator._circuit)(
                torch.zeros(1, config.n_qubits, dtype=torch.float64),
                generator.weights,
            )
        )
        return drawing.count("╭●") + drawing.count("─●")

    def test_fewer_entangling_layers_means_fewer_two_qubit_gates(self) -> None:
        full = QuantumGenerator(QWGANConfig(n_qubits=8, n_layers=4))
        last = QuantumGenerator(
            QWGANConfig(n_qubits=8, n_layers=4, entangling_layers=(3,))
        )
        none = QuantumGenerator(
            QWGANConfig(n_qubits=8, n_layers=4, entangling_layers=())
        )

        self.assertGreater(
            full.circuit_resources()["circuit_gate_count"],
            last.circuit_resources()["circuit_gate_count"],
        )
        self.assertGreater(
            last.circuit_resources()["circuit_gate_count"],
            none.circuit_resources()["circuit_gate_count"],
        )

    def test_output_stays_bounded_and_differentiable_without_entanglement(self) -> None:
        generator = QuantumGenerator(
            QWGANConfig(n_qubits=8, n_layers=3, entangling_layers=())
        )
        noise = torch.rand(16, 8, dtype=torch.float64, requires_grad=False) * np.pi

        output = generator(noise)
        self.assertEqual(output.shape, (16, 8))
        self.assertTrue(bool((output.abs() <= 1.0).all()))

        output.sum().backward()
        self.assertIsNotNone(generator.weights.grad)
        self.assertTrue(bool((generator.weights.grad.abs() > 0).any()))

    def test_entangling_every_layer_concentrates_the_local_expectations(self) -> None:
        """The measurement this parameter exists for.

        A CNOT ring at every layer drives the single-qubit reduced states toward
        maximally mixed, so the local Pauli-Z expectations concentrate at zero
        and the generator cannot produce the spread the real data has. Removing
        the ring recovers it -- at the cost of a separable circuit, which is why
        the default is unchanged.
        """

        def spread(entangling: tuple[int, ...] | None) -> float:
            config = QWGANConfig(
                n_qubits=8, n_layers=4, seed=13, entangling_layers=entangling
            )
            generator = QuantumGenerator(config)
            with torch.no_grad():
                # Weights comparable to a trained model's magnitude.
                generator.weights.mul_(50.0)
                noise = torch.rand(
                    512, 8, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(7),
                ) * (0.25 * np.pi)
                return float(generator(noise).std(dim=0).mean())

        self.assertGreater(spread(()), 4 * spread(None))
        self.assertGreater(spread((3,)), spread(None))


if __name__ == "__main__":
    unittest.main()
