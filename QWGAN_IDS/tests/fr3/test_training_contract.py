from __future__ import annotations

import unittest

import numpy as np

from cqai.qwgan import QWGANConfig, TrainingAngles


class TrainingAnglesTests(unittest.TestCase):
    def test_accepts_only_finite_train_partition_angles_in_declared_order(self) -> None:
        config = QWGANConfig(n_qubits=8, n_layers=3)
        valid = np.linspace(0.0, np.pi, 32).reshape(4, 8)
        batch = TrainingAngles.from_array(
            valid,
            config=config,
            partition="train",
            attack_class="u2r",
            latent_columns=tuple(f"z{i}" for i in range(8)),
        )

        self.assertEqual(batch.values.shape, (4, 8))
        self.assertEqual(batch.attack_class, "u2r")

        invalid_cases = (
            {"values": valid, "partition": "test"},
            {"values": np.full((4, 8), np.nan), "partition": "train"},
            {"values": np.full((4, 8), np.pi + 0.1), "partition": "train"},
        )
        for case in invalid_cases:
            with self.subTest(case=case["partition"]):
                with self.assertRaises(ValueError):
                    TrainingAngles.from_array(
                        case["values"],
                        config=config,
                        partition=case["partition"],
                        attack_class="u2r",
                        latent_columns=tuple(f"z{i}" for i in range(8)),
                    )


if __name__ == "__main__":
    unittest.main()
