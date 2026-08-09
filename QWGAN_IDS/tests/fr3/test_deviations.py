from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from cqai.qwgan.cli import TDD_DEFAULTS, load_experiment

BASE = {
    "version": "1.0.0",
    "contract": {"n_qubits": 10},
    "model": {"n_qubits": 10, "n_layers": 4},
    "training": {
        "attack_classes": ["u2r"],
        "seeds": [13, 42, 1337],
        "epochs": 1,
        "batch_size": 64,
    },
}


def _write(directory: str, document: dict) -> Path:
    path = Path(directory) / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


class DeviationTests(unittest.TestCase):
    """`AGENTS.md`: deviations must be recorded and reported, never silent.

    The enforcement lives in the loader rather than only in a test, so a config
    cannot quietly drift away from the TDD in a run that nobody reviewed.
    """

    def test_a_config_matching_the_tdd_needs_no_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, config, _ = load_experiment(_write(directory, BASE))
            self.assertEqual(config.learning_rate, TDD_DEFAULTS["learning_rate"])

    def test_an_undeclared_deviation_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = yaml.safe_load(yaml.safe_dump(BASE))
            document["model"]["learning_rate"] = 1e-3

            with self.assertRaises(ValueError) as caught:
                load_experiment(_write(directory, document))

            message = str(caught.exception)
            self.assertIn("learning_rate", message)
            self.assertIn("deviations", message)

    def test_a_declared_deviation_loads_and_keeps_its_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = yaml.safe_load(yaml.safe_dump(BASE))
            document["model"]["learning_rate"] = 1e-3
            document["deviations"] = {
                "learning_rate": "1e-4 leaves the circuit at its initialization"
            }

            _, config, _ = load_experiment(_write(directory, document))
            self.assertEqual(config.learning_rate, 1e-3)

    def test_an_empty_reason_is_not_a_declaration(self) -> None:
        """A checkbox is not a justification."""

        with tempfile.TemporaryDirectory() as directory:
            document = yaml.safe_load(yaml.safe_dump(BASE))
            document["model"]["learning_rate"] = 1e-3
            document["deviations"] = {"learning_rate": "   "}

            with self.assertRaises(ValueError):
                load_experiment(_write(directory, document))

    def test_declaring_something_that_did_not_deviate_is_refused(self) -> None:
        """Stale declarations rot into permission slips for future changes."""

        with tempfile.TemporaryDirectory() as directory:
            document = yaml.safe_load(yaml.safe_dump(BASE))
            document["deviations"] = {"learning_rate": "not actually changed"}

            with self.assertRaises(ValueError) as caught:
                load_experiment(_write(directory, document))
            self.assertIn("does not deviate", str(caught.exception))

    def test_every_hard_tdd_value_is_covered(self) -> None:
        """The guarded set must include the values a claim rests on."""

        for name in (
            "n_layers",
            "lambda_gp",
            "n_critic",
            "learning_rate",
            "beta1",
            "beta2",
            "latent_scale",
        ):
            self.assertIn(name, TDD_DEFAULTS)


if __name__ == "__main__":
    unittest.main()
