from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cqai.qwgan.cli import build_parser, load_experiment, main

from ..fixtures import write_nslkdd_fixture

EXPERIMENT = {
    "contract": {
        "n_qubits": 8,
        "top_k": 12,
        "val_fraction": 0.25,
        "split_seed": 3,
    },
    "model": {
        "n_qubits": 8,
        "n_layers": 3,
        "backend": "default.qubit",
        "diff_method": "backprop",
        "n_critic": 1,
        "lambda_gp": 10.0,
        "learning_rate": 1e-4,
    },
    "training": {
        "attack_classes": ["u2r"],
        "seeds": [11, 22, 33],
        "epochs": 1,
        "batch_size": 4,
        "critic_hidden_dims": [8],
    },
    # This fixture is shrunk for test speed, which is itself a deviation the
    # loader now requires to be named. Declaring it here keeps the tests honest
    # about the fact that they are not running the TDD configuration.
    "deviations": {
        "n_layers": "3 instead of 4 to keep the fast CPU suite fast",
        "n_critic": "1 instead of 5 to keep the fast CPU suite fast",
    },
}


class ExperimentConfigTests(unittest.TestCase):
    def test_versioned_config_resolves_to_spec_config_and_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            path.write_text(yaml.safe_dump(EXPERIMENT), encoding="utf-8")

            spec, config, plan = load_experiment(path)

            self.assertEqual(spec.n_qubits, 8)
            self.assertEqual(config.n_layers, 3)
            self.assertEqual(config.n_critic, 1)
            self.assertEqual(plan.attack_classes, ("u2r",))
            self.assertEqual(plan.seeds, (11, 22, 33))
            self.assertEqual(plan.critic_hidden_dims, (8,))

    def test_rejects_a_qubit_count_the_contract_cannot_serve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            broken = json.loads(json.dumps(EXPERIMENT))
            broken["model"]["n_qubits"] = 10  # contract still emits 8 latents
            path.write_text(yaml.safe_dump(broken), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_experiment(path)


#: A finished run exits 0 when every class is stable across seeds and 2 when it
#: is not. Tests that are about something else (contract reuse, run collision)
#: accept either, because a tiny 1-epoch fixture has no business being stable —
#: what they assert is that the run completed rather than crashed.
COMPLETED = (0, 2)


class CliTests(unittest.TestCase):
    def test_every_shipped_config_is_valid(self) -> None:
        """Every versioned FR-3 config must load and obey the TDD.

        This is deliberately a directory sweep rather than a named file: a new
        experiment config is exactly the kind of thing that gets added without
        a matching test, and a config that silently drops a seed or a WGAN-GP
        default would produce results nobody could defend. The glob is prefixed
        because each requirement's config has its own schema and its own sweep;
        FR-4's lives in ``tests/fr4/test_cli.py``.
        """

        configs = sorted(
            (Path(__file__).resolve().parents[2] / "configs").glob("fr3_*.yaml")
        )
        self.assertTrue(configs, "no versioned FR-3 experiment configs found")

        for shipped in configs:
            with self.subTest(config=shipped.name):
                spec, config, plan = load_experiment(shipped)
                self.assertEqual(spec.n_qubits, config.n_qubits)
                self.assertEqual(
                    len(plan.seeds), 3, "FR-6 requires exactly three seeds"
                )
                # Deviations from the TDD are no longer asserted away here:
                # `load_experiment` refuses any config that changes a TDD-fixed
                # value without declaring it with a reason, so reaching this
                # line already means every difference is documented. Pinning
                # the values again would forbid documented changes outright,
                # which is a different rule from forbidding silent ones.
                self.assertEqual(config.lambda_gp, 10.0)
                self.assertEqual(config.n_critic, 5)
                self.assertGreaterEqual(plan.batch_size, 64)
                self.assertLessEqual(plan.batch_size, 256)
                self.assertGreaterEqual(config.n_qubits, 8)
                self.assertLessEqual(config.n_qubits, 12)
                self.assertGreaterEqual(config.n_layers, 3)
                self.assertLessEqual(config.n_layers, 6)

    def test_end_to_end_run_builds_a_contract_and_trains_three_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)
            experiment = root / "experiment.yaml"
            experiment.write_text(yaml.safe_dump(EXPERIMENT), encoding="utf-8")

            exit_code = main(
                [
                    "--experiment",
                    str(experiment),
                    "--train-file",
                    str(source),
                    "--contract-dir",
                    str(root / "contract"),
                    "--output-dir",
                    str(root / "runs"),
                    "--run-id",
                    "test-run",
                ]
            )

            run_dir = root / "runs" / "test-run"
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["seeds"], [11, 22, 33])
            self.assertEqual(len(manifest["results"]), 3)
            self.assertEqual(manifest["partition"], "train")
            self.assertTrue((root / "contract" / "contract.json").is_file())

            # Every run carries its own cross-seed verdict, and the exit code
            # is that verdict rather than "the process did not throw".
            stability = json.loads(
                (run_dir / "stability.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stability["parent_run_id"], "test-run")
            self.assertEqual(exit_code, 0 if stability["stable"] else 2)
            self.assertTrue((run_dir / "stability.md").is_file())

    def test_an_unstable_run_does_not_exit_zero(self) -> None:
        """A campaign that cannot support its own claim must not look successful.

        One seed can never be reported as stable, so this is deterministic
        without having to induce a real plateau.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)
            single_seed = json.loads(json.dumps(EXPERIMENT))
            single_seed["training"]["seeds"] = [11]
            experiment = root / "experiment.yaml"
            experiment.write_text(yaml.safe_dump(single_seed), encoding="utf-8")

            exit_code = main(
                [
                    "--experiment", str(experiment),
                    "--train-file", str(source),
                    "--contract-dir", str(root / "contract"),
                    "--output-dir", str(root / "runs"),
                    "--run-id", "one-seed",
                ]
            )

            self.assertEqual(exit_code, 2)
            stability = json.loads(
                (root / "runs" / "one-seed" / "stability.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(stability["stable"])
            self.assertIn(
                "insufficient_seeds", stability["classes"]["u2r"]["reasons"]
            )

    def test_reuses_an_existing_contract_instead_of_rebuilding_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)
            experiment = root / "experiment.yaml"
            experiment.write_text(yaml.safe_dump(EXPERIMENT), encoding="utf-8")

            common = [
                "--experiment", str(experiment),
                "--train-file", str(source),
                "--contract-dir", str(root / "contract"),
                "--output-dir", str(root / "runs"),
            ]
            self.assertIn(main([*common, "--run-id", "first"]), COMPLETED)
            built_at = (root / "contract" / "contract.json").stat().st_mtime_ns

            self.assertIn(main([*common, "--run-id", "second"]), COMPLETED)
            self.assertEqual(
                (root / "contract" / "contract.json").stat().st_mtime_ns, built_at
            )

    def test_refuses_to_overwrite_an_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)
            experiment = root / "experiment.yaml"
            experiment.write_text(yaml.safe_dump(EXPERIMENT), encoding="utf-8")

            args = [
                "--experiment", str(experiment),
                "--train-file", str(source),
                "--contract-dir", str(root / "contract"),
                "--output-dir", str(root / "runs"),
                "--run-id", "collide",
            ]
            self.assertIn(main(args), COMPLETED)
            with self.assertRaises(FileExistsError):
                main(args)

    def test_parser_requires_an_experiment_file(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])


if __name__ == "__main__":
    unittest.main()
