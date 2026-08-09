from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cqai.data import ContractSpec, build_train_contract
from cqai.qwgan import QWGANConfig, TrainingPlan, run_training

from ..fixtures import write_nslkdd_fixture

SPEC = ContractSpec(n_qubits=8, top_k=12, val_fraction=0.25, split_seed=3)
CONFIG = QWGANConfig(n_qubits=8, n_layers=3, n_critic=1, seed=5)


def _contract(root: Path):
    source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)
    return build_train_contract(source, root / "contract", spec=SPEC)


def _plan(**overrides) -> TrainingPlan:
    arguments = {
        "attack_classes": ("u2r",),
        "seeds": (1,),
        "epochs": 40,
        "batch_size": 4,
        "critic_hidden_dims": (8,),
    }
    arguments.update(overrides)
    return TrainingPlan(**arguments)


class EarlyStopConfigTests(unittest.TestCase):
    def test_it_is_off_by_default(self) -> None:
        """A schedule change must be asked for, not inherited.

        Every reported run so far ran its full schedule; switching the default
        would make old and new runs incomparable without anyone choosing it.
        """

        self.assertIsNone(TrainingPlan(attack_classes=("u2r",)).early_stop_patience)

    def test_rejects_a_nonsensical_patience_or_interval(self) -> None:
        for bad in ({"early_stop_patience": 0}, {"early_stop_check_every": 0}):
            with self.subTest(**bad):
                with self.assertRaises(ValueError):
                    TrainingPlan(attack_classes=("u2r",), **bad)


class EarlyStopBehaviourTests(unittest.TestCase):
    def test_a_full_schedule_records_that_it_was_not_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_training(
                _contract(root),
                config=CONFIG,
                plan=_plan(epochs=4),
                output_dir=root / "runs",
            )

            entry = result.manifest["results"][0]
            self.assertEqual(entry["epochs_completed"], 4)
            self.assertEqual(entry["stop_reason"], "schedule_complete")
            self.assertFalse(entry["early_stopped"])

    def test_a_flat_run_stops_before_the_schedule_ends(self) -> None:
        """The point of the feature: stop paying for epochs that do nothing."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_training(
                _contract(root),
                config=CONFIG,
                plan=_plan(
                    epochs=60,
                    early_stop_patience=2,
                    early_stop_check_every=1,
                    early_stop_min_epochs=3,
                ),
                output_dir=root / "runs",
            )

            entry = result.manifest["results"][0]
            self.assertLess(entry["epochs_completed"], 60)
            self.assertTrue(entry["early_stopped"])
            self.assertEqual(entry["stop_reason"], "wasserstein_plateau")

    def test_it_never_stops_before_the_minimum(self) -> None:
        """Early WGAN dynamics are noisy; a dip in epoch two means nothing."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_training(
                _contract(root),
                config=CONFIG,
                plan=_plan(
                    epochs=20,
                    early_stop_patience=1,
                    early_stop_check_every=1,
                    early_stop_min_epochs=12,
                ),
                output_dir=root / "runs",
            )

            self.assertGreaterEqual(
                result.manifest["results"][0]["epochs_completed"], 12
            )

    def test_checkpoints_and_diagnostics_stop_with_the_run(self) -> None:
        """A stopped run must not leave artefacts claiming epochs it never ran."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_training(
                _contract(root),
                config=CONFIG,
                plan=_plan(
                    epochs=60,
                    checkpoint_every=1,
                    early_stop_patience=2,
                    early_stop_check_every=1,
                    early_stop_min_epochs=3,
                ),
                output_dir=root / "runs",
            )

            completed = result.manifest["results"][0]["epochs_completed"]
            checkpoints = sorted(result.root.glob("checkpoints/u2r/seed1/epoch*.pt"))
            self.assertEqual(len(checkpoints), completed)

            epochs = {
                json.loads(line)["epoch"]
                for line in (result.root / "diagnostics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            }
            self.assertEqual(max(epochs), completed)

    def test_the_validation_partition_is_not_used_to_decide(self) -> None:
        """Early stopping must not consume the FR-4 gate's real reference.

        Stopping on held-out performance is legitimate model selection, but
        these are the same rows the fidelity gate scores against. Selecting on
        them would make the gate's verdict a measure of its own selection.
        The signal is therefore the training-side Wasserstein estimate.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _contract(root)
            result = run_training(
                contract,
                config=CONFIG,
                plan=_plan(
                    epochs=30,
                    early_stop_patience=2,
                    early_stop_check_every=1,
                    early_stop_min_epochs=3,
                ),
                output_dir=root / "runs",
            )

            self.assertEqual(result.manifest["partition"], "train")
            self.assertEqual(
                result.manifest["results"][0]["early_stop_signal"],
                "train_wasserstein_estimate",
            )


if __name__ == "__main__":
    unittest.main()
