from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cqai.data import ContractSpec, build_train_contract, load_train_contract
from cqai.qwgan import QWGANConfig

from ..fixtures import write_nslkdd_fixture

SPEC = ContractSpec(n_qubits=8, top_k=12, val_fraction=0.25, split_seed=3)


class TrainContractBuilderTests(unittest.TestCase):
    def test_emits_angle_partitions_and_a_hashed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)

            contract = build_train_contract(source, root / "contract", spec=SPEC)

            self.assertEqual(
                contract.latent_columns, tuple(f"z{i}" for i in range(8))
            )
            self.assertEqual(contract.angle_range, (0.0, float(np.pi)))

            train = contract.angles("train")
            validation = contract.angles("val")
            self.assertEqual(train.shape[1], 8)
            self.assertEqual(validation.shape[1], 8)
            self.assertGreater(train.shape[0], validation.shape[0])
            for partition in (train, validation):
                self.assertTrue(np.isfinite(partition).all())
                self.assertGreaterEqual(partition.min(), 0.0)
                self.assertLessEqual(partition.max(), float(np.pi))

            # Every row is assigned to exactly one partition; nothing is lost.
            self.assertEqual(
                train.shape[0] + validation.shape[0],
                contract.metadata["cleaned_rows"],
            )

            manifest = json.loads(
                (contract.root / "contract.json").read_text(encoding="utf-8")
            )
            for key in (
                "contract_id",
                "contract_version",
                "schema_version",
                "created_utc",
                "source_file",
                "source_sha256",
                "spec",
                "latent_columns",
                "angle_range",
                "partitions",
                "attack_families",
                "artifact_sha256",
            ):
                self.assertIn(key, manifest)
            self.assertEqual(manifest["angle_range"], [0.0, float(np.pi)])
            self.assertNotIn("test", manifest["partitions"])

            # Every persisted artefact is hashed, and the hashes are real.
            self.assertTrue(manifest["artifact_sha256"])
            for name, digest in manifest["artifact_sha256"].items():
                self.assertTrue((contract.root / name).exists(), name)
                self.assertEqual(len(digest), 64, name)

    def test_transforms_are_fitted_on_the_train_partition_only(self) -> None:
        """Perturbing validation rows must not move a single train angle.

        This is the leakage check that the checked-in ``data/angles.npy``
        fails: it was produced by fitting on merged train+test data.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)

            baseline = build_train_contract(source, root / "a", spec=SPEC)
            baseline_train = baseline.angles("train").copy()
            validation_index = set(baseline.row_index("val").tolist())

            # Rewrite the same file with the validation rows blown far out of
            # range. If any transform saw them, train angles would shift.
            lines = source.read_text(encoding="utf-8").splitlines()
            for position in validation_index:
                fields = lines[position].split(",")
                fields[4] = "999999999"  # src_bytes
                fields[5] = "888888888"  # dst_bytes
                lines[position] = ",".join(fields)
            perturbed = root / "perturbed.txt"
            perturbed.write_text("\n".join(lines) + "\n", encoding="utf-8")

            shifted = build_train_contract(perturbed, root / "b", spec=SPEC)

            np.testing.assert_allclose(
                shifted.angles("train"), baseline_train, atol=1e-12
            )
            self.assertFalse(
                np.allclose(shifted.angles("val"), baseline.angles("val"))
            )

    def test_is_reproducible_and_reloadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)

            first = build_train_contract(source, root / "a", spec=SPEC)
            second = build_train_contract(source, root / "b", spec=SPEC)
            np.testing.assert_array_equal(
                first.angles("train"), second.angles("train")
            )
            self.assertEqual(first.source_sha256, second.source_sha256)

            reloaded = load_train_contract(first.root)
            self.assertEqual(reloaded.latent_columns, first.latent_columns)
            self.assertEqual(reloaded.source_sha256, first.source_sha256)
            np.testing.assert_array_equal(
                reloaded.angles("train"), first.angles("train")
            )


class TrainContractConsumptionTests(unittest.TestCase):
    def test_yields_per_class_training_angles_and_rejects_other_partitions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)
            contract = build_train_contract(source, root / "contract", spec=SPEC)
            config = QWGANConfig(n_qubits=8, n_layers=3)

            self.assertEqual(
                set(contract.attack_classes), {"normal", "dos", "probe", "r2l", "u2r"}
            )

            batch = contract.training_angles("u2r", config=config)
            self.assertEqual(batch.attack_class, "u2r")
            self.assertEqual(batch.values.shape[1], 8)
            self.assertEqual(
                int(batch.values.shape[0]),
                contract.class_counts("train")["u2r"],
            )
            self.assertGreaterEqual(float(batch.values.min()), 0.0)
            self.assertLessEqual(float(batch.values.max()), float(np.pi))

            # A class the contract does not carry must fail loudly.
            with self.assertRaises(KeyError):
                contract.training_angles("heartbleed", config=config)

            # The qubit count must match the contract's latent width.
            with self.assertRaises(ValueError):
                contract.training_angles(
                    "u2r", config=QWGANConfig(n_qubits=10, n_layers=3)
                )

    def test_validation_angles_are_available_but_never_used_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)
            contract = build_train_contract(source, root / "contract", spec=SPEC)

            held_out = contract.validation_angles("u2r")
            self.assertGreater(held_out.shape[0], 0)
            self.assertEqual(held_out.shape[1], 8)

            train_rows = {tuple(row) for row in contract.training_angles(
                "u2r", config=QWGANConfig(n_qubits=8, n_layers=3)
            ).values.numpy()}
            for row in held_out:
                self.assertNotIn(tuple(row), train_rows)


if __name__ == "__main__":
    unittest.main()
