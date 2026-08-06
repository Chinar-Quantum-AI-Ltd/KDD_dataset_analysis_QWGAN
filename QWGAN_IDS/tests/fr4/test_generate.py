from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cqai.data import ContractSpec, build_train_contract
from cqai.qwgan import QWGANConfig, TrainingPlan, run_training
from cqai.synthesis import SyntheticBatch, generate_samples

from ..fixtures import write_nslkdd_fixture

SPEC = ContractSpec(n_qubits=8, top_k=12, val_fraction=0.25, split_seed=3)
CONFIG = QWGANConfig(n_qubits=8, n_layers=3, n_critic=1, seed=5)


def _trained(root: Path):
    """A tiny real FR-3 run, so generation is exercised against real artefacts."""

    source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)
    contract = build_train_contract(source, root / "contract", spec=SPEC)
    result = run_training(
        contract,
        config=CONFIG,
        plan=TrainingPlan(
            attack_classes=("u2r",),
            seeds=(13,),
            epochs=1,
            batch_size=4,
            critic_hidden_dims=(8,),
        ),
        output_dir=root / "runs",
    )
    checkpoint = sorted(result.root.glob("checkpoints/u2r/seed13/epoch*.pt"))[-1]
    return contract, checkpoint


class GenerateTests(unittest.TestCase):
    def test_produces_exactly_the_requested_count_in_both_domains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, checkpoint = _trained(root)

            batch = generate_samples(
                checkpoint, contract=contract, count=37, seed=11
            )

            self.assertIsInstance(batch, SyntheticBatch)
            self.assertEqual(batch.angles.shape, (37, SPEC.n_qubits))
            self.assertEqual(len(batch.decoded), 37)
            self.assertEqual(
                list(batch.decoded.columns), list(contract.manifest["selected_features"])
            )
            self.assertEqual(batch.attack_class, "u2r")
            self.assertEqual(batch.seed, 11)

    def test_angles_stay_inside_the_declared_domain(self) -> None:
        """The generator's output must land where the decoder expects it.

        Angles outside ``[0, pi]`` would be decoded by an inverse MinMax that
        was never fitted on them, producing silent nonsense downstream.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, checkpoint = _trained(root)

            batch = generate_samples(
                checkpoint, contract=contract, count=64, seed=11
            )

            low, high = contract.angle_range
            self.assertTrue(np.isfinite(batch.angles).all())
            self.assertGreaterEqual(float(batch.angles.min()), low)
            self.assertLessEqual(float(batch.angles.max()), high)

    def test_chunking_does_not_change_the_result(self) -> None:
        """Chunking exists for memory, not for behaviour.

        Broadcasting 21 550 samples through ``default.qubit`` in one call
        allocates roughly 350 MB of statevector, so the real run chunks. If
        chunk size changed the samples, every reported number would depend on
        an implementation detail.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, checkpoint = _trained(root)

            one_call = generate_samples(
                checkpoint, contract=contract, count=50, seed=11, chunk_size=1024
            )
            chunked = generate_samples(
                checkpoint, contract=contract, count=50, seed=11, chunk_size=7
            )

            np.testing.assert_array_equal(one_call.angles, chunked.angles)

    def test_the_same_seed_replays_and_a_different_seed_diverges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, checkpoint = _trained(root)

            first = generate_samples(checkpoint, contract=contract, count=32, seed=1)
            again = generate_samples(checkpoint, contract=contract, count=32, seed=1)
            other = generate_samples(checkpoint, contract=contract, count=32, seed=2)

            np.testing.assert_array_equal(first.angles, again.angles)
            self.assertFalse(np.array_equal(first.angles, other.angles))

    def test_synthesis_seed_is_independent_of_where_training_stopped(self) -> None:
        """Two checkpoints of the same run must be seedable identically.

        If synthesis inherited the trainer's restored RNG state, the samples
        would depend on the epoch the checkpoint happened to be taken at, and a
        run could not be replayed from a different checkpoint.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)
            contract = build_train_contract(source, root / "contract", spec=SPEC)
            result = run_training(
                contract,
                config=CONFIG,
                plan=TrainingPlan(
                    attack_classes=("u2r",),
                    seeds=(13,),
                    epochs=2,
                    batch_size=4,
                    critic_hidden_dims=(8,),
                ),
                output_dir=root / "runs",
            )
            checkpoints = sorted(result.root.glob("checkpoints/u2r/seed13/epoch*.pt"))
            self.assertEqual(len(checkpoints), 2)

            noise_first = generate_samples(
                checkpoints[0], contract=contract, count=16, seed=99
            ).noise
            noise_second = generate_samples(
                checkpoints[1], contract=contract, count=16, seed=99
            ).noise

            np.testing.assert_array_equal(noise_first, noise_second)

    def test_refuses_a_checkpoint_built_against_a_different_contract(self) -> None:
        """Decoding with the wrong transforms is silent, not loud.

        The inverse PCA and scalers would happily run and return plausible
        numbers, so the mismatch has to be caught by hash rather than noticed
        later in a metric.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, checkpoint = _trained(root)

            other_source = write_nslkdd_fixture(
                root / "other.txt", rows_per_label=24, seed=99
            )
            other = build_train_contract(other_source, root / "other", spec=SPEC)

            with self.assertRaises(ValueError) as caught:
                generate_samples(checkpoint, contract=other, count=8, seed=1)
            self.assertIn("contract", str(caught.exception).lower())

    def test_refuses_a_non_positive_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, checkpoint = _trained(root)
            with self.assertRaises(ValueError):
                generate_samples(checkpoint, contract=contract, count=0, seed=1)


if __name__ == "__main__":
    unittest.main()
